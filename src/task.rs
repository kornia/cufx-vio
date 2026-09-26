//! `vio`: the tracker, driven from a copper task.
//!
//! A thin shell over [`Tracker`]. All the orchestration over read-only kornia-slam already
//! lives there; this file owns only the payload-to-image conversion and the
//! world-to-camera inversion, and must add no estimation logic of its own: anything it
//! computed would be a second implementation to keep in step with any offline replay of
//! [`Tracker`].

use std::sync::Arc;

use crate::config::{InertialRon, OnTrackingLoss};
use crate::imu_channel::ImuQueue;
use crate::pose::cam_in_world;
use crate::reset::ResetEpoch;
use crate::stats::{Durations, TrackerTiming};
use crate::task_error::{Eye, TaskError};
use crate::track::{TrackStatus, Tracker, TrackerConfig};
use cu_stereo_payloads::{Landmark, RectifiedStereo, StereoPair, VioPose, VioStatus};
use cu29::pool::{CuHandle, CuHostMemoryPool, CuPool};
use cu29::prelude::*;
use kornia_3d::camera::PinholeCamera;
use kornia_image::{Image, ImageSize};

/// Most map points one [`VioPose::map_points`] snapshot carries.
///
/// The live map passes this within minutes; the snapshot keeps the NEWEST points (see
/// [`Tracker::newest_live_map_points`]). At 16 bytes a point a full snapshot is 128 KiB, which
/// is also what each pool buffer reserves, so a snapshot never reallocates.
pub const MAX_LANDMARKS: usize = 8192;

/// Default number of landmark snapshot buffers; the `landmark_buffers` key overrides it.
///
/// A snapshot stays checked out of the pool while anything holds its handle: the background
/// wrapper's committed output, each copperlist slot that received it until that slot is
/// reused, and any consumer that keeps a clone. Snapshots are taken on keyframes only, so few
/// are alive at once; 16 is 2 MiB and covers a graph of 8 copperlists with a consumer that
/// forwards the handle into its own. An exhausted pool drops the snapshot, never the pose.
pub const DEFAULT_LANDMARK_BUFFERS: usize = 16;

/// Largest `landmark_buffers` accepted: 256 buffers is 32 MiB.
///
/// The pool allocates every buffer up front, at 128 KiB each, so an unbounded key turns a
/// typo into a startup allocation of gigabytes. 256 is 32 times the default, well past any
/// copperlist count a graph needs.
pub const MAX_LANDMARK_BUFFERS: usize = 256;

/// Pool id reported in copper's pool statistics.
const LANDMARK_POOL_ID: &str = "cu_kornia_vio.landmarks";

/// Fills a pooled snapshot from `points`, or returns `None` when every buffer is checked out.
///
/// The buffer is cleared and refilled in place: the pool hands back the same `Vec` with its
/// capacity intact, so a steady state allocates nothing.
fn snapshot_map_points(
    pool: &CuHostMemoryPool<Vec<Landmark>>,
    points: impl Iterator<Item = (usize, [f64; 3])>,
) -> Option<CuHandle<Vec<Landmark>>> {
    let handle = pool.acquire()?;
    handle.with_inner_mut(|inner| {
        let buf: &mut Vec<Landmark> = inner.as_mut();
        buf.clear();
        // An index past u32::MAX is four billion points into one map; skipped rather than
        // wrapped onto an older point's index.
        buf.extend(points.filter_map(|(idx, [x, y, z])| {
            Some(Landmark {
                position: [x as f32, y as f32, z as f32],
                index: u32::try_from(idx).ok()?,
            })
        }));
    });
    Some(handle)
}

/// Resource bindings of [`StereoVio`], both provided by a [`VioBus`](crate::VioBus).
#[doc(hidden)]
#[allow(missing_docs)]
pub mod vio_resources {
    use super::{ImuQueue, ResetEpoch};
    use cu29::resources;

    resources!({
        imu => Shared<ImuQueue>,
        epoch => Shared<ResetEpoch>,
    });
}

/// Tracks a stereo stream into a pose per frame.
///
/// ONE input, deliberately. `cu29`'s `CuAsyncTask`, which is what `background: true` wraps a
/// task in, is implemented only for `T: CuTask<Input<'i> = CuMsg<I>>`: a single message, not a
/// pack. So a second input edge is what stands between this task and being backgrounded, and
/// backgrounding is the whole point on a live graph: keyframe insertion costs 0.6-1.0 s against
/// a 66 ms frame interval, and while this runs inline EVERY other task in the graph waits behind
/// it. Measured on a Jetson Orin with an OAK-D at 640x400, an inline run published at 1.2 Hz from
/// a camera producing 15.
///
/// The inertial samples do not ride a second input, and are not bundled into the frame payload
/// either: `CuAsyncTask` refuses the arriving input while the previous solve is `Running` and
/// again while `Waiting`, so with a solve that overruns the 66 ms frame interval on keyframes
/// and at its tail, roughly one frame in four is dropped, and a payload-borne batch would be
/// dropped with them. Samples
/// arrive through the bus's [`ImuQueue`] instead, pushed by an inline
/// [`ImuFeed`](crate::ImuFeed); see [`crate::imu_channel`] for what that costs in replay
/// fidelity. The queue is armed only when this task has an `inertial` block: without one,
/// `ImuFeed`'s push is a single atomic load and the inertial path cannot start.
///
/// Reset requests arrive the same way, through the bus's [`ResetEpoch`].
///
/// # Landmarks
///
/// A pose from a frame that inserted a keyframe carries [`VioPose::map_points`]: the live points
/// among the newest [`MAX_LANDMARKS`] map slots, world frame, each with its map index. Every
/// other pose carries `None`. Keyframes are the cadence because that is when the map changes:
/// new points are created, old ones culled or fused and local bundle adjustment moves the rest,
/// while a plain tracking frame only reads the map.
///
/// Each snapshot is the complete current view, so a consumer replaces what it displays with it:
/// culled points drop out, and after a [`VioStatus::reset_epoch`] change the next snapshot
/// describes the new world whether the map was rebuilt or kept. Until that snapshot arrives the
/// displayed one belongs to the old epoch; a consumer that must not show it clears on the epoch
/// change. Snapshot buffers come from a `CuHostMemoryPool` this task owns, sized by the
/// `landmark_buffers` key (default [`DEFAULT_LANDMARK_BUFFERS`]); when it is exhausted the pose
/// is published without its snapshot, and the stop line counts how often.
#[derive(Reflect)]
#[reflect(from_reflect = false)]
pub struct StereoVio {
    /// Built on the FIRST frame, not at construction: the geometry rides on the payload, so
    /// the task never has to be told what the source already knows.
    #[reflect(ignore)]
    tracker: Option<Tracker>,
    /// The first frame's calibration, which the tracker was built from. Every later frame must
    /// carry the same one, so a source whose calibration changes mid-stream is a visible error
    /// rather than a silently stale camera model.
    #[reflect(ignore)]
    calib: RectifiedStereo,
    /// The first frame's eye size; every later frame must match it.
    eye_size: (u32, u32),
    tracked: u64,
    untracked: u64,
    /// The reset epoch this task has APPLIED; lags the requested epoch until the next frame.
    /// Private edge detector only: the PUBLISHED counter is the tracker's world generation.
    reset_epoch: u64,
    /// Per-graph policy, from the RON config; see [`OnTrackingLoss`].
    #[reflect(ignore)]
    on_tracking_loss: OnTrackingLoss,
    /// Keyframe budget from the RON config; `None` is unbounded.
    max_keyframes: Option<usize>,
    /// Tracking-cost knobs from the RON config; `None` keeps the tracker's own default.
    /// `estimate_pose` is 96 ms of a 100 ms tracking step on a Jetson Orin, so these are the
    /// only levers on the frame rate that do not touch kornia-slam.
    orb_keypoints: Option<usize>,
    search_radius_px: Option<f64>,
    max_covisible_keyframes: Option<usize>,
    /// LM iteration ceiling per PnP refit round; kornia-slam's default is 50.
    pnp_lm_iterations: Option<usize>,
    /// The `inertial` RON block, or `None`, the default. Parsed in `new` rather than on the
    /// first frame so a malformed block fails at graph construction, not 40 minutes in.
    #[reflect(ignore)]
    inertial: Option<InertialRon>,
    /// One-shot latch for the `push_imu` rejection log: the failure is a wiring mistake that
    /// repeats every frame, and a per-frame error line would bury the graph's own output.
    imu_push_warned: bool,
    #[reflect(ignore)]
    timing: TrackerTiming,
    /// Per-eye image buffers, reused across frames. See `eye`.
    #[reflect(ignore)]
    scratch_left: Vec<u8>,
    #[reflect(ignore)]
    scratch_right: Vec<u8>,
    #[reflect(ignore)]
    imu: Arc<ImuQueue>,
    #[reflect(ignore)]
    epoch: Arc<ResetEpoch>,
    /// Buffers for [`VioPose::map_points`]; see the type docs.
    #[reflect(ignore)]
    landmark_pool: Arc<CuHostMemoryPool<Vec<Landmark>>>,
    /// Keyframe poses published with a snapshot, and without one because the pool was empty.
    landmark_snapshots: u64,
    landmark_pool_exhausted: u64,
}

/// A no-op, following `cu_anynet::AnyNetStereo`, whose `Freezable` is empty because nothing it
/// holds can be snapshotted per job.
///
/// The state that matters here is the kornia-slam map, which has no serialised form, and the
/// background wrapper snapshots the task after EVERY solve, so even a hand-rolled encoding would
/// cost a full map copy per frame. What that gives up: a resim reproduces this task only from
/// the start of a log, never from a mid-log keyframe.
impl Freezable for StereoVio {}

/// Copies one eye out of a payload into an image the tracker can consume.
///
/// `Image` owns its buffer while the payload's eye sits behind a shared `CuHandle`, so one copy
/// per eye happens here: 256 kB at 640x400, twice per frame, each past glibc's 128 kB mmap
/// threshold, hence the reused scratch buffer.
fn eye(
    bytes: &[u8],
    (width, height): (u32, u32),
    seq: u64,
    which: Eye,
    scratch: &mut Vec<u8>,
) -> Result<Image<u8, 1>, TaskError> {
    let (w, h) = (width as usize, height as usize);
    let want = w * h;
    if bytes.len() != want {
        return Err(TaskError::FrameShape {
            seq,
            eye: which,
            got: bytes.len(),
            want,
            width,
            height,
        });
    }
    // REUSED buffer, not `to_vec()`: the allocation is 256 kB and lands past glibc's mmap
    // threshold, so a fresh one per eye per frame is two mmap/munmap pairs and ~126 page
    // faults. `Image::new` takes the Vec by value, so the caller MUST hand it back with
    // `into_vec()` after the tracker is done — see `process`. Taking it without returning it
    // would allocate every frame anyway, just less obviously.
    scratch.clear();
    scratch.extend_from_slice(bytes);
    Ok(Image::new(
        ImageSize {
            width: w,
            height: h,
        },
        std::mem::take(scratch),
    )?)
}

impl CuTask for StereoVio {
    type Input<'m> = input_msg!(StereoPair);
    type Output<'m> = output_msg!(VioPose);
    type Resources<'r> = vio_resources::Resources;

    fn new(config: Option<&ComponentConfig>, resources: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        crate::config::deny_unknown_keys(
            config,
            "cu_kornia_vio::StereoVio",
            crate::config::CONFIG_KEYS,
        )?;
        let inertial = crate::config::optional_inertial(config)?;
        let vio_resources::Resources { imu, epoch } = resources;
        let landmark_buffers =
            crate::config::optional_positive_usize(config, crate::config::LANDMARK_BUFFERS)?
                .unwrap_or(DEFAULT_LANDMARK_BUFFERS);
        if landmark_buffers > MAX_LANDMARK_BUFFERS {
            return Err(format!(
                "`{}` must be at most {MAX_LANDMARK_BUFFERS} (128 KiB each, allocated up front), \
                 got {landmark_buffers}",
                crate::config::LANDMARK_BUFFERS
            )
            .into());
        }
        if inertial.is_some() {
            // Declares the consumer. Until this runs the producer's push is a single atomic
            // load and no lock: the right state for every graph without this block.
            imu.arm();
        }
        Ok(Self {
            tracker: None,
            calib: RectifiedStereo::default(),
            eye_size: (0, 0),
            on_tracking_loss: crate::config::on_tracking_loss(config)?,
            max_keyframes: crate::config::optional_positive_usize(
                config,
                crate::config::MAX_KEYFRAMES,
            )?,
            orb_keypoints: crate::config::optional_positive_usize(
                config,
                crate::config::ORB_KEYPOINTS,
            )?,
            search_radius_px: crate::config::optional_positive_f64(
                config,
                crate::config::SEARCH_RADIUS_PX,
            )?,
            max_covisible_keyframes: crate::config::optional_positive_usize(
                config,
                crate::config::MAX_COVISIBLE_KEYFRAMES,
            )?,
            pnp_lm_iterations: crate::config::optional_positive_usize(
                config,
                crate::config::PNP_LM_ITERATIONS,
            )?,
            inertial,
            imu_push_warned: false,
            tracked: 0,
            untracked: 0,
            // Whatever was requested before this task existed is already applied: a fresh
            // tracker is the state a reset would produce.
            reset_epoch: epoch.requested(),
            timing: TrackerTiming::default(),
            scratch_left: Vec::new(),
            scratch_right: Vec::new(),
            imu,
            epoch,
            // Built last, after every fallible config read above, so a refused config neither
            // allocates the buffers nor registers the pool in copper's global statistics.
            // Each buffer is filled to full length up front, so the pool reports its real
            // footprint and a snapshot's `clear` + `extend` never grows it.
            landmark_pool: CuHostMemoryPool::new(LANDMARK_POOL_ID, landmark_buffers, || {
                vec![Landmark::default(); MAX_LANDMARKS]
            })?,
            landmark_snapshots: 0,
            landmark_pool_exhausted: 0,
        })
    }

    fn process<'i, 'o>(
        &mut self,
        _ctx: &CuContext,
        input: &Self::Input<'i>,
        output: &mut Self::Output<'o>,
    ) -> CuResult<()> {
        // Mandatory under `background: true`: the worker's output starts from a default
        // `CuMsg`, so a pose would otherwise carry no time of validity at all.
        output.tov = input.tov;
        // Cleared up front so that every early return, including the error paths below, leaves
        // an empty slot rather than a stale pose from an earlier cycle.
        output.clear_payload();
        let Some(frame) = input.payload() else {
            // The clip is finished, or an upstream step was skipped. Emit nothing; a
            // defaulted pose would read as the camera teleporting to the origin.
            return Ok(());
        };
        let seq = frame.left.seq;
        let stamp = match input.tov {
            Tov::Time(t) => t,
            // A range stamp on a stereo pair means the eyes were not captured together; the
            // start is the reference the tracker's IMU windows are built against.
            Tov::Range(r) => r.start,
            Tov::None => return Err(TaskError::Untimed { seq }.into_cu_error("vio")),
        };
        frame.validate()?;

        let tracker = match &mut self.tracker {
            Some(t) => {
                if frame.calib != self.calib {
                    return Err(TaskError::CalibrationChanged { seq }.into_cu_error("vio"));
                }
                t
            }
            none => {
                let c = frame.calib;
                self.calib = c;
                self.eye_size = (frame.left.format.width, frame.left.format.height);
                // Distortion is zero because these pixels are RECTIFIED — the rectifier has
                // already absorbed it. Passing the raw coefficients here would undistort a
                // second time, which produces a plausible reconstruction at the wrong scale.
                let camera = PinholeCamera {
                    fx: c.fx,
                    fy: c.fy,
                    cx: c.cx,
                    cy: c.cy,
                    k1: 0.0,
                    k2: 0.0,
                    p1: 0.0,
                    p2: 0.0,
                };
                let mut config = TrackerConfig::new(c.baseline);
                // See `OnTrackingLoss` for why a long-running graph drops the map.
                config.reset_map_on_loss =
                    matches!(self.on_tracking_loss, OnTrackingLoss::ResetMap);
                config.max_keyframes = self.max_keyframes;
                if let Some(n) = self.orb_keypoints {
                    config.orb.n_keypoints = n;
                }
                if let Some(r) = self.search_radius_px {
                    // Both the initial search and the local-map refinement: leaving the wider
                    // refinement pass at its default would cap any saving from the first.
                    config.map_projection.projection.search_radius = r as f32;
                    config.map_projection.local_projection.search_radius = r as f32;
                }
                if let Some(n) = self.max_covisible_keyframes {
                    config.max_covisible_keyframes = n;
                }
                if let Some(n) = self.pnp_lm_iterations {
                    // Both PnP sites: the initial solve and the local-map refinement share
                    // this config, and capping only one would halve the saving.
                    config.map_projection.pnp.lm_max_iterations = n;
                }
                // Absent from every graph this repo ships. Present, it is the integrator
                // asserting a measured camera-to-IMU extrinsic; `to_config` rejects one that
                // is not even a rigid rotation, but nothing can check it is the RIGHT one.
                config.inertial = self
                    .inertial
                    .as_ref()
                    .map(|ron| {
                        ron.to_config()
                            .map_err(|e| TaskError::from(e).into_cu_error("vio"))
                    })
                    .transpose()?;
                none.insert(Tracker::new(camera, config))
            }
        };
        let requested = self.epoch.requested();
        if requested != self.reset_epoch {
            // Fresh SLAM: new map, new tracker world anchored at THIS frame. Map-point indices
            // restart with the new map; every pose from here on carries the new epoch, and a
            // consumer re-anchors on the first one it sees (keyed by that pose's own stamp).
            tracker.reset();
            self.reset_epoch = requested;
        }
        // Inertial ingest, before the solve: a keyframe can close inside `process_stereo`, and
        // the samples covering the interval it ends have to be buffered by then. Draining here
        // (rather than after) is also what makes the read content-deterministic: the window the
        // tracker asks for is bounded by two keyframe stamps, not by when this ran.
        if self.inertial.is_some() {
            // Straight from the queue's deque into the tracker's ring, under the queue's
            // lock: one copy per sample, no scratch buffer to keep clear. An empty drain is
            // skipped so a drop report with no samples waits for the batch it belongs to,
            // exactly as `ImuBuffer::push` would attach it.
            let pushed = self
                .imu
                .drain_with(|samples, dropped, aligned| {
                    (samples.len() > 0).then(|| tracker.push_raw_imu(samples, dropped, aligned))
                })
                .flatten();
            if let Some(Err(e)) = pushed
                && !self.imu_push_warned
            {
                self.imu_push_warned = true;
                // Not a graph error: the tracker keeps producing visual poses, and killing a
                // live graph over the inertial term would be a worse failure than
                // running without it. Logged once; the stop line carries the counts.
                let error = e.to_string();
                error!("vio: IMU samples rejected by the tracker: {}", error);
            }
        }

        let size = self.eye_size;
        let (scratch_left, scratch_right) = (&mut self.scratch_left, &mut self.scratch_right);
        let (left, right) = frame
            .with_eyes(|l, r| {
                Ok::<_, TaskError>((
                    eye(l, size, seq, Eye::Left, scratch_left)?,
                    eye(r, size, seq, Eye::Right, scratch_right)?,
                ))
            })?
            .map_err(|e| e.into_cu_error("vio"))?;

        // Times process_stereo ONLY — not the payload-to-image copy above, which is this
        // representation's cost rather than the tracker's, and not the graph plumbing.
        // Wall clock by design, not the copper clock: the figure is the real cost on the machine
        // that ran it, which is what a rate target is sized from, and it only feeds the stop
        // line, never the output, so a resim under a mocked clock still reproduces the poses.
        let t0 = std::time::Instant::now();
        let tracked = tracker.process_stereo(&left, &right, stamp);
        let dt = t0.elapsed();
        // RECLAIM the two 256 kB buffers before the `?`, so an error path does not quietly
        // drop them and reintroduce the per-frame allocation on the next good frame.
        self.scratch_left = left.into_vec();
        self.scratch_right = right.into_vec();
        let tracked = tracked.map_err(|e| TaskError::from(e).into_cu_error("vio"))?;

        match tracked {
            TrackStatus::Tracked(pose) => {
                self.tracked += 1;
                let is_kf = pose.keyframe;
                self.timing.push(dt, is_kf);
                // cam_in_world, NOT pose_world_to_cam. The tracker's convention is
                // world-to-camera; publishing it unchanged is the frustum-flies-backwards
                // bug, which looks like a plausible trajectory travelled in reverse.
                let cam_in_world = cam_in_world(&pose.pose_world_to_cam);
                // Live points only, and cached by the tracker: a per-frame scan of every point
                // ever created would be an unbounded per-frame cost for one status field.
                let landmarks = tracker.live_map_points();
                let map_points = if is_kf {
                    let snapshot = snapshot_map_points(
                        &self.landmark_pool,
                        tracker.newest_live_map_points(MAX_LANDMARKS),
                    );
                    if snapshot.is_some() {
                        self.landmark_snapshots += 1;
                    } else {
                        if self.landmark_pool_exhausted == 0 {
                            // Once: a pool too small for the graph repeats every keyframe.
                            warning!(
                                "vio: landmark pool exhausted, publishing keyframe poses without \
                                 a snapshot; raise `landmark_buffers`"
                            );
                        }
                        self.landmark_pool_exhausted += 1;
                    }
                    snapshot
                } else {
                    None
                };
                let payload = VioPose {
                    cam_in_world,
                    status: VioStatus {
                        keyframe: is_kf,
                        // The tracker owns this counter: `Tracker::reset` and the internal loss
                        // re-bootstrap both bump it.
                        reset_epoch: tracker.world_generation(),
                        landmarks: u32::try_from(landmarks).unwrap_or(u32::MAX),
                    },
                    map_points,
                };
                output.set_payload(payload);
            }
            TrackStatus::Untracked => {
                self.untracked += 1;
                // An untracked frame still cost time; recorded as non-keyframe so the budget
                // reflects every frame the graph actually processed.
                self.timing.push(dt, false);
            }
        }
        Ok(())
    }

    fn stop(&mut self, _ctx: &CuContext) -> CuResult<()> {
        info!(
            "vio: {} tracked, {} untracked",
            self.tracked, self.untracked
        );
        info!(
            "vio: {} landmark snapshots, {} keyframes without one (pool exhausted)",
            self.landmark_snapshots, self.landmark_pool_exhausted
        );
        // One structured line per population, each figure its own field, in milliseconds. NaN
        // marks an empty population, which is not the same thing as a zero-cost one.
        let t = &self.timing;
        let ms = |d: &Durations, p: f64| d.pct(p).unwrap_or(f64::NAN);
        info!(
            "vio timing, no keyframe: n={} p50_ms={} p99_ms={} max_ms={}",
            t.tracking.len(),
            ms(&t.tracking, 50.0),
            ms(&t.tracking, 99.0),
            ms(&t.tracking, 100.0)
        );
        // The deadline is set by THIS population, not the one above: a rate target must clear the
        // keyframe p99.
        info!(
            "vio timing, keyframe inserted: n={} p50_ms={} p99_ms={} max_ms={}",
            t.keyframe.len(),
            ms(&t.keyframe, 50.0),
            ms(&t.keyframe, 99.0),
            ms(&t.keyframe, 100.0)
        );
        info!(
            "vio: {} of {} frames inserted a keyframe",
            t.keyframe_frames(),
            t.total_frames()
        );
        // Only when non-zero. The gyro prior falls back to the constant-velocity model in
        // silence, so a prior that never once applied is indistinguishable from a working one.
        if let Some(n) = self.tracker.as_ref().map(|t| t.gyro_prior_unavailable())
            && n > 0
        {
            info!("vio: gyro prior unavailable on {} frames", n);
        }
        // Refusals are the number to watch: a refused interval leaves the keyframe pair
        // visual-only and no other trace anywhere, so a rate here is the only way to see an IMU
        // stream that has quietly stopped covering its intervals.
        if let Some(stats) = self.tracker.as_ref().and_then(|t| t.inertial_stats()) {
            info!(
                "vio inertial: {} factors ({} retained, pinning {} raw samples), refused {} \
                 (dropped {}, gap {}, sparse {}, empty {}, bad-interval {}, zero-dt {}); \
                 samples {} accepted / {} out-of-order / {} evicted, {} drop reports; init {}/{} \
                 accepted, initialized={} refine_1={} refine_2={}",
                stats.factors_added,
                stats.retained_factors,
                stats.retained_samples,
                stats.refused_total(),
                stats.refused_dropped,
                stats.refused_gap,
                stats.refused_too_few,
                stats.refused_no_samples,
                stats.refused_bad_interval,
                stats.refused_zero_dt,
                stats.buffer.accepted,
                stats.buffer.out_of_order,
                stats.buffer.evicted,
                stats.buffer.drop_reports,
                stats.init_accepted,
                stats.init_attempts,
                stats.initialized,
                stats.first_refinement_done,
                stats.second_refinement_done,
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::clock::{CuTime, CuTimeRange};
    use cu29::pool::CuHandle;
    use cu29::units::si::f64::Length;
    use cu29::units::si::length::meter;

    const W: u32 = 320;
    const H: u32 = 240;

    fn vio() -> (StereoVio, Arc<ImuQueue>, Arc<ResetEpoch>) {
        let (imu, epoch) = (Arc::new(ImuQueue::new()), Arc::new(ResetEpoch::new()));
        let task = StereoVio::new(
            None,
            vio_resources::Resources {
                imu: Arc::clone(&imu),
                epoch: Arc::clone(&epoch),
            },
        )
        .expect("an empty config is valid");
        (task, imu, epoch)
    }

    /// A random-dot plane at a constant 8 px disparity: enough texture for ORB and the matcher.
    fn frame(tov: Tov) -> CuMsg<StereoPair> {
        let texture = |x: u32, y: u32| {
            let h = (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663))
                .wrapping_mul(2_654_435_761);
            (h >> 24) as u8
        };
        let eye = |shift: u32| {
            (0..H)
                .flat_map(|y| (0..W).map(move |x| texture(x + shift, y)))
                .collect::<Vec<u8>>()
        };
        pair_msg(eye(8), eye(0), tov)
    }

    /// A `W`x`H` pair on the test calibration (fx = fy = 300, 0.1 m baseline), stamped `tov`.
    fn pair_msg(left: Vec<u8>, right: Vec<u8>, tov: Tov) -> CuMsg<StereoPair> {
        let calib = RectifiedStereo {
            fx: 300.0,
            fy: 300.0,
            cx: 160.0,
            cy: 120.0,
            baseline: Length::new::<meter>(0.1),
        };
        let pair = StereoPair::new(
            W,
            H,
            CuHandle::new_detached(left),
            CuHandle::new_detached(right),
            calib,
        )
        .expect("valid pair");
        let mut msg = CuMsg::new(Some(pair));
        msg.tov = tov;
        msg
    }

    #[test]
    fn test_output_tov_is_the_input_tov() {
        let (mut task, _, _) = vio();
        let ctx = CuContext::new_with_clock();
        for tov in [
            Tov::Time(CuTime::from_nanos(1_000_000)),
            Tov::Range(CuTimeRange {
                start: CuTime::from_nanos(2_000_000),
                end: CuTime::from_nanos(2_000_500),
            }),
        ] {
            let mut out = CuMsg::<VioPose>::default();
            task.process(&ctx, &frame(tov), &mut out)
                .expect("a valid frame");
            assert_eq!(
                out.tov, tov,
                "the pose must carry the frame's time of validity"
            );
        }
    }

    #[test]
    fn test_an_untimed_frame_is_refused_but_still_propagates_its_tov() {
        let (mut task, _, _) = vio();
        let ctx = CuContext::new_with_clock();
        let mut out = CuMsg::new(Some(VioPose::default()));
        out.tov = Tov::Time(CuTime::from_nanos(5));
        let res = task.process(&ctx, &frame(Tov::None), &mut out);
        assert!(res.is_err(), "the tracker needs a capture time per frame");
        assert_eq!(
            out.tov,
            Tov::None,
            "no stale tov may survive a refused frame"
        );
        assert!(
            out.payload().is_none(),
            "no stale pose may survive a refused frame"
        );
    }

    #[test]
    fn test_an_empty_input_clears_the_output() {
        let (mut task, _, _) = vio();
        let ctx = CuContext::new_with_clock();
        let mut input = CuMsg::<StereoPair>::new(None);
        input.tov = Tov::Time(CuTime::from_nanos(7));
        let mut out = CuMsg::new(Some(VioPose::default()));
        task.process(&ctx, &input, &mut out)
            .expect("no payload is not an error");
        assert!(out.payload().is_none());
        assert_eq!(out.tov, input.tov);
    }

    #[test]
    fn test_a_reset_request_on_the_bus_is_applied_at_the_next_frame() {
        let (mut task, _, epoch) = vio();
        let ctx = CuContext::new_with_clock();
        let mut out = CuMsg::<VioPose>::default();
        task.process(&ctx, &frame(Tov::Time(CuTime::from_nanos(1))), &mut out)
            .expect("frame");
        epoch.request();
        assert_eq!(task.reset_epoch, 0, "applied only at a frame boundary");
        task.process(&ctx, &frame(Tov::Time(CuTime::from_nanos(2))), &mut out)
            .expect("frame");
        assert_eq!(task.reset_epoch, 1);
    }

    #[test]
    fn test_a_mid_stream_calibration_change_is_refused() {
        let (mut task, _, _) = vio();
        let ctx = CuContext::new_with_clock();
        let mut out = CuMsg::<VioPose>::default();
        task.process(&ctx, &frame(Tov::Time(CuTime::from_nanos(1))), &mut out)
            .expect("frame");
        let mut changed = frame(Tov::Time(CuTime::from_nanos(2)));
        if let Some(pair) = changed.payload_mut() {
            pair.calib.baseline = Length::new::<meter>(0.12);
        }
        let res = task.process(&ctx, &changed, &mut out);
        assert!(
            res.is_err(),
            "the tracker keeps the first frame's camera model"
        );
        task.process(&ctx, &frame(Tov::Time(CuTime::from_nanos(3))), &mut out)
            .expect("the original calibration is still accepted");
    }

    /// A frame the tracker can bootstrap on, unlike [`frame`]'s one-pixel noise: 2x2-pixel
    /// cells so ORB finds corners at every pyramid level, and three depth bands so stereo
    /// triangulates a 3-D structure. The same scene as `examples/stereo_vio.rs`, which tracks.
    fn trackable_frame(seq: u32, tov: Tov) -> CuMsg<StereoPair> {
        const BANDS: [u32; 3] = [16, 24, 32];
        let texture = |x: u32, y: u32| {
            let h = (x / 2).wrapping_mul(73_856_093) ^ (y / 2).wrapping_mul(19_349_663);
            (h.wrapping_mul(2_654_435_761) >> 24) as u8
        };
        let (mut left, mut right) = (Vec::new(), Vec::new());
        for y in 0..H {
            let d = BANDS[((y / (H / 3)) as usize).min(2)];
            let shift = seq.wrapping_mul(d / 8);
            for x in 0..W {
                left.push(texture(x.wrapping_add(shift), y));
                right.push(texture(x.wrapping_add(d).wrapping_add(shift), y));
            }
        }
        pair_msg(left, right, tov)
    }

    #[test]
    fn test_keyframe_poses_and_only_they_carry_a_map_points_snapshot() {
        let (mut task, _, _) = vio();
        let ctx = CuContext::new_with_clock();
        let (mut saw_snapshot, mut saw_plain_pose) = (false, false);
        for seq in 0..6u32 {
            let mut out = CuMsg::<VioPose>::default();
            let tov = Tov::Time(CuTime::from_nanos(1_000_000 + u64::from(seq) * 66_000_000));
            task.process(&ctx, &trackable_frame(seq, tov), &mut out)
                .expect("frame");
            let Some(pose) = out.payload() else { continue };
            assert_eq!(
                pose.map_points.is_some(),
                pose.status.keyframe,
                "the snapshot cadence is the keyframe cadence"
            );
            let Some(handle) = &pose.map_points else {
                saw_plain_pose = true;
                continue;
            };
            saw_snapshot = true;
            let points = handle.with_inner(|v| v.to_vec());
            assert!(
                !points.is_empty(),
                "a keyframe of a textured plane has points"
            );
            // Small map, so the newest-`MAX_LANDMARKS` window is the whole map: the snapshot
            // must hold exactly the live points the status counts.
            assert_eq!(points.len(), pose.status.landmarks as usize);
            assert!(
                points.windows(2).all(|w| w[0].index < w[1].index),
                "indices are unique and ascending"
            );
            assert!(
                points
                    .iter()
                    .all(|p| p.position.iter().all(|c| c.is_finite())),
                "world positions are finite"
            );
        }
        assert!(saw_snapshot, "the bootstrap frame inserts a keyframe");
        assert!(
            saw_plain_pose,
            "a tracked non-keyframe pose must be exercised too, or the cadence is untested"
        );
        assert_eq!(task.landmark_pool_exhausted, 0);
    }

    #[test]
    fn test_an_exhausted_landmark_pool_yields_no_snapshot_until_a_buffer_returns() {
        let pool = CuHostMemoryPool::new("cu_kornia_vio.test_landmarks", 1, || {
            vec![Landmark::default(); 4]
        })
        .expect("pool");
        let held = snapshot_map_points(&pool, [(3, [1.0, 2.0, 3.0])].into_iter())
            .expect("one free buffer");
        assert!(
            snapshot_map_points(&pool, std::iter::empty()).is_none(),
            "the only buffer is still checked out"
        );
        drop(held);
        let again = snapshot_map_points(&pool, [(9, [4.0, 5.0, 6.0])].into_iter())
            .expect("the buffer came back");
        assert_eq!(
            again.with_inner(|v| v.to_vec()),
            vec![Landmark {
                position: [4.0, 5.0, 6.0],
                index: 9,
            }],
            "a recycled buffer holds only the new snapshot, not the previous fill"
        );
    }

    fn vio_with_landmark_buffers(n: usize) -> CuResult<StereoVio> {
        let mut cfg = ComponentConfig::new();
        cfg.set(crate::config::LANDMARK_BUFFERS, n as u32);
        StereoVio::new(
            Some(&cfg),
            vio_resources::Resources {
                imu: Arc::new(ImuQueue::new()),
                epoch: Arc::new(ResetEpoch::new()),
            },
        )
    }

    #[test]
    fn test_zero_landmark_buffers_is_refused() {
        assert!(
            vio_with_landmark_buffers(0).is_err(),
            "a pool of zero buffers would never snapshot"
        );
    }

    #[test]
    fn test_landmark_buffers_past_the_cap_is_refused() {
        assert!(
            vio_with_landmark_buffers(MAX_LANDMARK_BUFFERS).is_ok(),
            "the cap itself is accepted"
        );
        assert!(
            vio_with_landmark_buffers(MAX_LANDMARK_BUFFERS + 1).is_err(),
            "every buffer is allocated up front, so the count is bounded"
        );
    }

    #[test]
    fn test_only_an_inertial_config_arms_the_imu_queue() {
        let (_, imu, _) = vio();
        assert!(
            !imu.is_armed(),
            "a stereo-only task must leave the producer inert"
        );
    }
}
