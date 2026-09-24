//! Sensor payloads for a Copper stereo-visual-inertial graph: the imagery and inertial types a
//! stereo driver produces and a VIO component consumes, plus the pose it emits.
//!
//! A driver names these types without depending on any estimator, and an estimator consumes them
//! without depending on any driver; this crate is the only thing the two share.
//!
//! # Built on the upstream payloads
//!
//! * Each eye of a [`StereoPair`] is a [`CuImage<Vec<u8>>`](cu_sensor_payloads::CuImage): the
//!   pixels sit behind a pool-backed [`CuHandle`], so a copperlist slot holds a refcount, not a
//!   frame, and a source can recycle its buffers through a `CuHostMemoryPool`.
//! * Inertial samples are upstream [`ImuPayload`]s, batched in [`ImuBatch`] because an IMU runs
//!   an order of magnitude faster than the camera.
//! * The estimated pose is a [`cu_spatial_payloads::Pose<f64>`]. Its optional map snapshot is
//!   a pool-backed [`CuHandle`] too, for the same reason as the eyes.
//!
//! # Time lives on the envelope
//!
//! No payload here carries a message timestamp: the time of validity is the `tov` of the `CuMsg`
//! that carries it. The one exception is per-sample time inside a batch, because the samples in
//! one [`ImuBatch`] were not captured at the same instant; the enclosing message then carries
//! [`ImuBatch::tov`], a `Tov::Range` over the samples. This follows upstream's
//! `PeerRangeSnapshot`.
//!
//! # Hand-written `Decode`
//!
//! Types that hold a [`CuImage`], a [`Pose`] or a [`CuHandle`] implement `Decode<()>` by hand:
//! those only implement `Decode<()>`, so the derive's generic `impl<C> Decode<C>` cannot be
//! written over them. This follows `cu_anynet::StereoPair`. Every other trait is derived.

#![deny(missing_docs)]

use bincode::de::Decoder;
use bincode::enc::Encoder;
use bincode::error::{DecodeError, EncodeError};
use bincode::{Decode, Encode};
use cu_sensor_payloads::{CuImage, CuImageBufferFormat};
use cu_spatial_payloads::Pose;
use cu29::clock::{CuTime, CuTimeRange, Tov};
use cu29::pool::{CuHandle, HandleContent, HandleContentAware};
use cu29::prelude::{CuError, CuResult, Reflect};
// `derive(Reflect)` expands to `bevy_reflect::` paths when a consumer turns on cu29's `reflect`
// feature (any app that links `cu29-export`, for one), and to nothing that needs it otherwise.
#[allow(unused_imports)]
use cu29::prelude::bevy_reflect;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use serde::de;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub use cu_sensor_payloads::ImuPayload;

/// FourCC of the only pixel format a [`StereoPair`] eye may use: 8-bit grayscale.
pub const GRAY8: [u8; 4] = *b"GRAY";

/// The geometry a consumer needs to interpret a rectified pair.
///
/// RECTIFIED intrinsics, the common K of the virtual rectified camera, not either eye's raw
/// factory values. Pairing raw intrinsics with rectified pixels yields a plausible reconstruction
/// at the wrong scale. The image size is not repeated here; it is each eye's
/// [`CuImageBufferFormat`].
#[derive(Clone, Copy, Debug, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect)]
pub struct RectifiedStereo {
    /// Rectified focal length x, pixels.
    pub fx: f64,
    /// Rectified focal length y, pixels.
    pub fy: f64,
    /// Rectified principal point x, pixels.
    pub cx: f64,
    /// Rectified principal point y, pixels.
    pub cy: f64,
    /// Distance between the two rectified optical centres. What makes the estimate metric rather
    /// than up-to-scale, so it rides with the pixels rather than being configured separately.
    pub baseline: Length,
}

impl Default for RectifiedStereo {
    fn default() -> Self {
        Self {
            fx: 0.0,
            fy: 0.0,
            cx: 0.0,
            cy: 0.0,
            baseline: Length::new::<meter>(0.0),
        }
    }
}

impl RectifiedStereo {
    /// `fx * baseline` in pixel-metres, the disparity-to-depth constant. Derived so it cannot
    /// disagree with the two values it comes from.
    pub fn bf(&self) -> f64 {
        self.fx * self.baseline.get::<meter>()
    }
}

/// One rectified GRAY8 stereo pair.
///
/// Both eyes are rectified: rows are epipolar lines, which is what the stereo matcher's row
/// search assumes. The calibration rides on every pair rather than being latched, so a consumer
/// that joins late or replays from the middle of a log never holds pixels it cannot interpret.
/// The frame counter is each eye's [`CuImage::seq`]; a gap in it is a frame the consumer never
/// saw, while [`StereoPair::dropped`] counts frames the source itself discarded.
///
/// `Default` exists only because `CuMsgPayload` requires it: upstream `CuHandle::default()`
/// panics, so build a pair with [`StereoPair::new`] and never call `default()`.
#[derive(Default, Debug, Clone, Encode, Serialize, Deserialize, Reflect)]
#[reflect(from_reflect = false, no_field_bounds)]
pub struct StereoPair {
    /// Rectified left eye.
    pub left: CuImage<Vec<u8>>,
    /// Rectified right eye, same format as `left`.
    pub right: CuImage<Vec<u8>>,
    /// Rectified geometry of the pair.
    pub calib: RectifiedStereo,
    /// Frames the source produced but never handed to the graph, cumulative.
    pub dropped: u64,
}

impl Decode<()> for StereoPair {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            left: Decode::decode(decoder)?,
            right: Decode::decode(decoder)?,
            calib: Decode::decode(decoder)?,
            dropped: Decode::decode(decoder)?,
        })
    }
}

/// The buffer format of one tightly packed GRAY8 eye.
pub fn gray8_format(width: u32, height: u32) -> CuImageBufferFormat {
    CuImageBufferFormat {
        width,
        height,
        stride: width,
        pixel_format: GRAY8,
    }
}

impl StereoPair {
    /// Builds a pair from two GRAY8 eye buffers, checking them with [`StereoPair::validate`].
    pub fn new(
        width: u32,
        height: u32,
        left: CuHandle<Vec<u8>>,
        right: CuHandle<Vec<u8>>,
        calib: RectifiedStereo,
    ) -> CuResult<Self> {
        let format = gray8_format(width, height);
        let pair = Self {
            left: CuImage {
                seq: 0,
                format,
                buffer_handle: left,
            },
            right: CuImage {
                seq: 0,
                format,
                buffer_handle: right,
            },
            calib,
            dropped: 0,
        };
        pair.validate()?;
        Ok(pair)
    }

    /// Checks that both eyes are tightly packed GRAY8 of one size, and that each buffer holds
    /// the bytes its format declares.
    ///
    /// Worth running at every boundary: a stride mistake produces a sheared image that the
    /// stereo matcher consumes without complaint.
    pub fn validate(&self) -> CuResult<()> {
        let (l, r) = (self.left.format, self.right.format);
        if l.width != r.width || l.height != r.height {
            return Err(CuError::from(format!(
                "stereo pair: left is {}x{} but right is {}x{}",
                l.width, l.height, r.width, r.height
            )));
        }
        if l.width == 0 || l.height == 0 {
            return Err(CuError::from("stereo pair: empty image"));
        }
        for (eye, img) in [("left", &self.left), ("right", &self.right)] {
            let f = img.format;
            if f.pixel_format != GRAY8 {
                return Err(CuError::from(format!(
                    "stereo pair: {eye} eye is {:?}, expected GRAY",
                    f.pixel_format
                )));
            }
            if f.stride != f.width {
                return Err(CuError::from(format!(
                    "stereo pair: {eye} eye stride {} is not its width {}",
                    f.stride, f.width
                )));
            }
            let have = img.buffer_handle.with_inner(|b| b.len());
            let want = f.required_bytes();
            if have < want {
                return Err(CuError::from(format!(
                    "stereo pair: {eye} eye holds {have} bytes, its format needs {want}"
                )));
            }
        }
        Ok(())
    }

    /// Runs `f` over both eyes' pixel bytes, each exactly `width * height` long.
    ///
    /// Call [`StereoPair::validate`] first; this checks only what it needs to slice safely.
    pub fn with_eyes<R>(&self, f: impl FnOnce(&[u8], &[u8]) -> R) -> CuResult<R> {
        self.left.with_plane_bytes(0, |left, _| {
            self.right.with_plane_bytes(0, |right, _| f(left, right))
        })?
    }
}

// Codegen gate for a non-default `handle_content` logging mode on a source emitting this type;
// the inherent forwards below are what propagate the policy to both eyes.
impl HandleContentAware for StereoPair {}

impl StereoPair {
    /// Whether the unified log should write the eyes' bytes: true if either eye's handle says so.
    pub fn payload_should_log(&self) -> bool {
        self.left.payload_should_log() || self.right.payload_should_log()
    }

    /// Stamps a source's configured logging policy on both eyes.
    pub fn apply_handle_content_policy(&self, mode: HandleContent) {
        self.left.apply_handle_content_policy(mode);
        self.right.apply_handle_content_policy(mode);
    }

    /// Marks both eyes as read, for the `TouchedOnly` logging policy.
    pub fn mark_touched(&self) {
        self.left.mark_touched();
        self.right.mark_touched();
    }
}

/// One inertial sample with its own time of validity, for use inside an [`ImuBatch`].
///
/// A single `CuMsg<ImuPayload>` carries its time on the envelope; this type exists only because
/// the samples of one batch were captured at different instants.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub struct ImuSample {
    /// Capture time, on the same clock as the images' `tov`.
    pub tov: CuTime,
    /// The measurement, in the IMU's own axes.
    pub imu: ImuPayload,
}

/// Bounded batch of timestamped IMU samples, oldest first.
///
/// An IMU at 200-400 Hz against a camera at 10-30 Hz delivers several samples per graph cycle;
/// batching keeps them on one edge at the camera's rate. The message carrying a batch should use
/// [`ImuBatch::tov`] as its `tov`.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
pub struct ImuBatch<const N: usize> {
    len: usize,
    samples: [ImuSample; N],
    /// The source's cumulative count of samples it lost before they reached the graph. A driver
    /// never resets it; a consumer diffs consecutive values to find the holes.
    pub dropped: u64,
    /// True when the samples are already rotated into the camera frame. A consumer that also
    /// applies a camera-to-IMU extrinsic would then rotate them twice.
    pub camera_aligned: bool,
}

impl<const N: usize> Default for ImuBatch<N> {
    fn default() -> Self {
        Self {
            len: 0,
            samples: [ImuSample::default(); N],
            dropped: 0,
            camera_aligned: false,
        }
    }
}

/// Raised by [`ImuBatch::push`] on a full batch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImuBatchFull {
    /// The batch capacity.
    pub capacity: usize,
}

impl core::fmt::Display for ImuBatchFull {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "IMU batch capacity {} is full", self.capacity)
    }
}

impl core::error::Error for ImuBatchFull {}

impl<const N: usize> ImuBatch<N> {
    /// An empty batch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of samples held.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the batch holds no sample.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the batch is at capacity.
    pub fn is_full(&self) -> bool {
        self.len >= N
    }

    /// The samples held, in push order.
    pub fn samples(&self) -> &[ImuSample] {
        &self.samples[..self.len]
    }

    /// Appends a sample, refusing rather than overwriting when full.
    pub fn push(&mut self, sample: ImuSample) -> Result<(), ImuBatchFull> {
        if self.is_full() {
            return Err(ImuBatchFull { capacity: N });
        }
        self.samples[self.len] = sample;
        self.len += 1;
        Ok(())
    }

    /// Empties the batch, keeping `dropped` and `camera_aligned`.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// The earliest and latest sample times, or `None` for an empty batch.
    pub fn tov_range(&self) -> Option<CuTimeRange> {
        let (first, rest) = self.samples().split_first()?;
        let (start, end) = rest.iter().fold((first.tov, first.tov), |(lo, hi), s| {
            (lo.min(s.tov), hi.max(s.tov))
        });
        Some(CuTimeRange { start, end })
    }

    /// The time of validity for the message carrying this batch: `Tov::Range` over the samples,
    /// or `Tov::None` when empty.
    pub fn tov(&self) -> Tov {
        self.tov_range().map_or(Tov::None, Tov::Range)
    }
}

impl<const N: usize> Encode for ImuBatch<N> {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        Encode::encode(&self.len, encoder)?;
        for sample in self.samples() {
            Encode::encode(sample, encoder)?;
        }
        Encode::encode(&self.dropped, encoder)?;
        Encode::encode(&self.camera_aligned, encoder)
    }
}

impl<const N: usize> Decode<()> for ImuBatch<N> {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let len = <usize as Decode<()>>::decode(decoder)?;
        if len > N {
            return Err(DecodeError::ArrayLengthMismatch {
                required: N,
                found: len,
            });
        }
        let mut batch = Self::default();
        for slot in &mut batch.samples[..len] {
            *slot = Decode::decode(decoder)?;
        }
        batch.len = len;
        batch.dropped = Decode::decode(decoder)?;
        batch.camera_aligned = Decode::decode(decoder)?;
        Ok(batch)
    }
}

impl<const N: usize> Serialize for ImuBatch<N> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("ImuBatch", 3)?;
        state.serialize_field("samples", self.samples())?;
        state.serialize_field("dropped", &self.dropped)?;
        state.serialize_field("camera_aligned", &self.camera_aligned)?;
        state.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for ImuBatch<N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct Wire {
            samples: Vec<ImuSample>,
            dropped: u64,
            camera_aligned: bool,
        }
        let wire = Wire::deserialize(deserializer)?;
        let mut batch = Self {
            dropped: wire.dropped,
            camera_aligned: wire.camera_aligned,
            ..Self::default()
        };
        for sample in wire.samples {
            batch
                .push(sample)
                .map_err(|e| de::Error::custom(format!("{e}")))?;
        }
        Ok(batch)
    }
}

/// Tracker bookkeeping that accompanies each pose.
///
/// There is no tracking-state field: a frame the tracker could not place carries NO payload, and
/// that absence is the one way "untracked" is said. A consumer must skip such frames rather than
/// hold the previous pose, which would read as a stationary camera.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub struct VioStatus {
    /// Whether a keyframe was inserted. Per-frame cost is bimodal on this flag, so a mean frame
    /// time across both is meaningless.
    pub keyframe: bool,
    /// Which tracker WORLD this pose belongs to. Incremented whenever the tracker's frame of
    /// reference changes: an explicit reset, or a re-bootstrap after a sustained tracking loss.
    /// A consumer anchoring tracker poses into another world must discard poses from a stale
    /// epoch and re-anchor on the first pose of a new one.
    pub reset_epoch: u64,
    /// Live (not culled) map points in the tracker's map.
    pub landmarks: u32,
}

/// One point of the tracker's map, as carried by [`VioPose::map_points`].
///
/// A struct rather than `[f32; 4]` with the index cast to a float: an `f32` holds integers
/// exactly only up to 2^24, past which two points would share an index. A `u32` is exact to 2^32.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Encode, Decode, Serialize, Deserialize, Reflect,
)]
pub struct Landmark {
    /// Position in the tracker's world frame (the frame of [`VioPose::cam_in_world`]), metres.
    pub position: [f32; 3],
    /// The point's index in the tracker's map, stable within one [`VioStatus::reset_epoch`]: a
    /// point keeps it while its position is refined. For correlating one point across snapshots,
    /// not for accumulating them; see [`VioPose::map_points`].
    pub index: u32,
}

/// One estimated pose.
///
/// **Camera-in-world**, already inverted from the tracker's world-to-camera convention. The
/// inversion happens once, at the type boundary: publishing the un-inverted pose looks like a
/// plausible trajectory travelled in reverse.
///
/// `Default` is safe to call (no map points), unlike [`StereoPair`]'s.
#[derive(Default, Debug, Clone, Encode, Serialize, Deserialize, Reflect)]
#[reflect(from_reflect = false, no_field_bounds)]
pub struct VioPose {
    /// Camera pose in the tracker's world frame, metres.
    pub cam_in_world: Pose<f64>,
    /// Tracker bookkeeping for this pose.
    pub status: VioStatus,
    /// A snapshot of map points, or `None`. A viewer payload, not part of the estimate.
    ///
    /// Each snapshot is the COMPLETE current view the producer chose to send: a consumer
    /// REPLACES what it displays with it, rather than merging it into earlier ones, so points
    /// the tracker culled or fused disappear and a new [`VioStatus::reset_epoch`] needs no special
    /// case. When it is present and which points it holds is the producer's policy, documented
    /// on the producing task. Not to be confused with [`VioStatus::landmarks`], the count of
    /// every live point, which a bounded snapshot can fall short of.
    ///
    /// Behind a [`CuHandle`], so a producer can recycle the buffers through a
    /// `CuHostMemoryPool` and a copperlist slot holds a refcount, not the snapshot.
    #[reflect(ignore)]
    pub map_points: Option<CuHandle<Vec<Landmark>>>,
}

impl Decode<()> for VioPose {
    fn decode<D: Decoder<Context = ()>>(decoder: &mut D) -> Result<Self, DecodeError> {
        Ok(Self {
            cam_in_world: Decode::decode(decoder)?,
            status: Decode::decode(decoder)?,
            map_points: Decode::decode(decoder)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::config::standard;
    use cu29::units::si::acceleration::meter_per_second_squared;
    use cu29::units::si::angular_velocity::radian_per_second;

    fn encode<T: Encode>(value: &T) -> Vec<u8> {
        bincode::encode_to_vec(value, standard()).expect("encode")
    }

    fn decode<T: Decode<()>>(bytes: &[u8]) -> T {
        let (value, used): (T, usize) =
            bincode::decode_from_slice(bytes, standard()).expect("decode");
        assert_eq!(used, bytes.len(), "trailing bytes");
        value
    }

    fn calib() -> RectifiedStereo {
        RectifiedStereo {
            fx: 1.0,
            fy: 2.0,
            cx: 3.0,
            cy: 4.0,
            baseline: Length::new::<meter>(5.0),
        }
    }

    fn pair() -> StereoPair {
        let mut p = StereoPair::new(
            4,
            2,
            CuHandle::new_detached(vec![1, 2, 3, 4, 5, 6, 7, 8]),
            CuHandle::new_detached(vec![9, 10, 11, 12, 13, 14, 15, 255]),
            calib(),
        )
        .expect("valid pair");
        p.left.seq = 7;
        p.right.seq = 7;
        p.dropped = 3;
        p
    }

    fn imu(i: f32) -> ImuPayload {
        ImuPayload::from_raw([i, i + 1.0, i + 2.0], [i + 3.0, i + 4.0, i + 5.0], i + 6.0)
    }

    // --- golden bytes ---------------------------------------------------------------------------
    //
    // A round trip CANNOT catch a transposition inside a run of same-width fields: a swapped
    // `Encode` and `Decode` agree with each other while both disagree with every log already
    // written. So each wire layout is pinned to literal bytes, with a distinct value per field.

    #[test]
    fn test_rectified_stereo_golden_bytes() {
        let bytes = encode(&calib());
        #[rustfmt::skip]
        let golden: &[u8] = &[
            0, 0, 0, 0, 0, 0, 0xf0, 0x3f, // fx = 1.0
            0, 0, 0, 0, 0, 0, 0x00, 0x40, // fy = 2.0
            0, 0, 0, 0, 0, 0, 0x08, 0x40, // cx = 3.0
            0, 0, 0, 0, 0, 0, 0x10, 0x40, // cy = 4.0
            0, 0, 0, 0, 0, 0, 0x14, 0x40, // baseline = 5.0 m
        ];
        assert_eq!(bytes, golden);
        assert_eq!(decode::<RectifiedStereo>(&bytes), calib());
    }

    #[test]
    fn test_stereo_pair_golden_bytes() {
        let bytes = encode(&pair());
        #[rustfmt::skip]
        let golden: &[u8] = &[
            // left: seq, format {width, height, stride, fourcc}, then the Vec<u8> buffer
            7, 4, 2, 4, b'G', b'R', b'A', b'Y', 8, 1, 2, 3, 4, 5, 6, 7, 8,
            // right
            7, 4, 2, 4, b'G', b'R', b'A', b'Y', 8, 9, 10, 11, 12, 13, 14, 15, 255,
            // calib
            0, 0, 0, 0, 0, 0, 0xf0, 0x3f,
            0, 0, 0, 0, 0, 0, 0x00, 0x40,
            0, 0, 0, 0, 0, 0, 0x08, 0x40,
            0, 0, 0, 0, 0, 0, 0x10, 0x40,
            0, 0, 0, 0, 0, 0, 0x14, 0x40,
            // dropped
            3,
        ];
        assert_eq!(bytes, golden);

        let back: StereoPair = decode(&bytes);
        assert_eq!(back.left.seq, 7);
        assert_eq!(back.calib, calib());
        assert_eq!(back.dropped, 3);
        back.validate().expect("decoded pair is valid");
        back.with_eyes(|l, r| {
            assert_eq!(l, &[1, 2, 3, 4, 5, 6, 7, 8]);
            assert_eq!(r, &[9, 10, 11, 12, 13, 14, 15, 255]);
        })
        .expect("eyes");
    }

    #[test]
    fn test_imu_batch_golden_bytes() {
        let mut batch = ImuBatch::<4>::new();
        batch
            .push(ImuSample {
                tov: CuTime::from_nanos(10),
                imu: imu(1.0),
            })
            .expect("fits");
        batch.dropped = 5;
        batch.camera_aligned = true;
        let bytes = encode(&batch);
        #[rustfmt::skip]
        let golden: &[u8] = &[
            1,                      // len
            10,                     // tov, ns
            0x00, 0x00, 0x80, 0x3f, // accel_x = 1.0 m/s^2
            0x00, 0x00, 0x00, 0x40, // accel_y = 2.0
            0x00, 0x00, 0x40, 0x40, // accel_z = 3.0
            0x00, 0x00, 0x80, 0x40, // gyro_x = 4.0 rad/s
            0x00, 0x00, 0xa0, 0x40, // gyro_y = 5.0
            0x00, 0x00, 0xc0, 0x40, // gyro_z = 6.0
            0x33, 0x13, 0x8c, 0x43, // temperature = 7.0 degC, stored as 280.15 K
            5,                      // dropped
            1,                      // camera_aligned
        ];
        assert_eq!(bytes, golden);
        assert_eq!(decode::<ImuBatch<4>>(&bytes), batch);
    }

    #[test]
    fn test_vio_pose_golden_bytes() {
        let pose = VioPose {
            cam_in_world: Pose::from_matrix([
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 3.0],
                [0.0, 0.0, 1.0, 4.0],
                [0.0, 0.0, 0.0, 1.0],
            ]),
            status: VioStatus {
                keyframe: false,
                reset_epoch: 6,
                landmarks: 9,
            },
            map_points: None,
        };
        let bytes = encode(&pose);
        const ONE: [u8; 8] = [0, 0, 0, 0, 0, 0, 0xf0, 0x3f];
        const ZERO: [u8; 8] = [0; 8];
        let row = |a: [u8; 8], b: [u8; 8], c: [u8; 8], d: [u8; 8]| [a, b, c, d].concat();
        let two = 2.0f64.to_le_bytes();
        let three = 3.0f64.to_le_bytes();
        let four = 4.0f64.to_le_bytes();
        let golden = [
            row(ONE, ZERO, ZERO, two),
            row(ZERO, ONE, ZERO, three),
            row(ZERO, ZERO, ONE, four),
            row(ZERO, ZERO, ZERO, ONE),
            // keyframe = false, reset_epoch, landmarks
            vec![0, 6, 9],
            // no map_points snapshot: the Option tag alone
            vec![0],
        ]
        .concat();
        assert_eq!(bytes, golden);

        let back: VioPose = decode(&bytes);
        assert_eq!(back.status, pose.status);
        assert_eq!(back.cam_in_world.to_matrix(), pose.cam_in_world.to_matrix());
        assert!(back.map_points.is_none());
    }

    #[test]
    fn test_vio_pose_with_map_points_golden_bytes() {
        let points = vec![
            Landmark {
                position: [1.0, 2.0, 3.0],
                index: 7,
            },
            Landmark {
                position: [-1.0, 0.5, 4.0],
                index: 300,
            },
        ];
        let pose = VioPose {
            cam_in_world: Pose::default(),
            status: VioStatus {
                keyframe: true,
                reset_epoch: 2,
                landmarks: 5,
            },
            map_points: Some(CuHandle::new_detached(points.clone())),
        };
        let bytes = encode(&pose);
        let head = encode(&pose.cam_in_world);
        #[rustfmt::skip]
        let tail: &[u8] = &[
            1, 2, 5,                // keyframe = true, reset_epoch, landmarks
            1,                      // Some: a snapshot follows
            2,                      // two points
            0x00, 0x00, 0x80, 0x3f, // x = 1.0
            0x00, 0x00, 0x00, 0x40, // y = 2.0
            0x00, 0x00, 0x40, 0x40, // z = 3.0
            7,                      // index, varint
            0x00, 0x00, 0x80, 0xbf, // x = -1.0
            0x00, 0x00, 0x00, 0x3f, // y = 0.5
            0x00, 0x00, 0x80, 0x40, // z = 4.0
            251, 0x2c, 0x01,        // index = 300, varint u16 form
        ];
        assert_eq!(bytes, [head.as_slice(), tail].concat());

        let back: VioPose = decode(&bytes);
        assert_eq!(back.status, pose.status);
        let got = back
            .map_points
            .expect("the snapshot survives the round trip")
            .with_inner(|v| v.to_vec());
        assert_eq!(got, points);
    }

    #[test]
    fn test_landmark_index_is_exact_past_f32_precision() {
        // 2^24 + 1 is the first integer an f32 cannot hold: the reason the index is a u32.
        let lm = Landmark {
            position: [0.0; 3],
            index: (1 << 24) + 1,
        };
        assert_eq!(decode::<Landmark>(&encode(&lm)), lm);
    }

    #[test]
    fn test_vio_pose_default_carries_no_map_points() {
        assert!(VioPose::default().map_points.is_none());
    }

    // --- behaviour ------------------------------------------------------------------------------

    #[test]
    fn test_bf_is_fx_times_baseline() {
        let c = RectifiedStereo {
            fx: 457.418,
            baseline: Length::new::<meter>(0.110_078),
            ..Default::default()
        };
        assert!((c.bf() - 50.352).abs() < 1e-3, "bf = {}", c.bf());
    }

    #[test]
    fn test_validate_rejects_mismatched_eyes() {
        let mut p = pair();
        p.right.format = gray8_format(2, 4);
        assert!(p.validate().is_err(), "4x2 left against 2x4 right");
    }

    #[test]
    fn test_validate_rejects_a_short_buffer() {
        let err = StereoPair::new(
            4,
            2,
            CuHandle::new_detached(vec![0; 8]),
            CuHandle::new_detached(vec![0; 7]),
            calib(),
        );
        assert!(err.is_err(), "7 bytes cannot hold a 4x2 eye");
    }

    #[test]
    fn test_validate_rejects_a_colour_eye() {
        let mut p = pair();
        p.left.format.pixel_format = *b"RGB3";
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_clone_shares_the_eye_buffers() {
        let p = pair();
        let c = p.clone();
        assert_eq!(
            p.left.buffer_handle.storage_id(),
            c.left.buffer_handle.storage_id(),
            "a clone is a refcount, not a frame copy"
        );
    }

    #[test]
    fn test_imu_batch_refuses_overflow_and_reports_its_range() {
        let mut batch = ImuBatch::<2>::new();
        assert_eq!(batch.tov(), Tov::None);
        for ns in [30, 10] {
            batch
                .push(ImuSample {
                    tov: CuTime::from_nanos(ns),
                    imu: ImuPayload::default(),
                })
                .expect("fits");
        }
        assert_eq!(
            batch.push(ImuSample::default()),
            Err(ImuBatchFull { capacity: 2 })
        );
        let range = batch.tov_range().expect("non-empty");
        assert_eq!(range.start, CuTime::from_nanos(10));
        assert_eq!(range.end, CuTime::from_nanos(30));
        assert!(matches!(batch.tov(), Tov::Range(_)));
    }

    #[test]
    fn test_imu_batch_decode_refuses_more_than_capacity() {
        let mut big = ImuBatch::<3>::new();
        for _ in 0..3 {
            big.push(ImuSample::default()).expect("fits");
        }
        let bytes = encode(&big);
        let res: Result<(ImuBatch<2>, usize), _> = bincode::decode_from_slice(&bytes, standard());
        assert!(res.is_err());
    }

    #[test]
    fn test_imu_units_survive_the_batch() {
        let mut batch = ImuBatch::<1>::new();
        batch
            .push(ImuSample {
                tov: CuTime::from_nanos(1),
                imu: imu(1.0),
            })
            .expect("fits");
        let back: ImuBatch<1> = decode(&encode(&batch));
        let s = back.samples()[0].imu;
        assert_eq!(s.accel_x.get::<meter_per_second_squared>(), 1.0);
        assert_eq!(s.gyro_z.get::<radian_per_second>(), 6.0);
    }

    /// `CuMsgPayload` is a blanket impl, so a missing bound shows up only where a graph names
    /// the type. Name each one here instead.
    #[test]
    fn test_types_satisfy_the_payload_contract() {
        fn assert_payload<T: cu29::prelude::CuMsgPayload>() {}
        assert_payload::<StereoPair>();
        assert_payload::<ImuBatch<32>>();
        assert_payload::<ImuPayload>();
        assert_payload::<VioPose>();
    }

    /// The pool bound: a `CuHostMemoryPool<Vec<Landmark>>` exists only for an `ElementType`.
    #[test]
    fn test_landmark_is_a_pool_element() {
        fn assert_element<T: cu29::pool::ElementType>() {}
        assert_element::<Landmark>();
    }
}
