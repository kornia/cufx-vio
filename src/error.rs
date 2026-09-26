//! Typed errors for the cu-kornia-vio tracking core.
//!
//! thiserror only; `anyhow` is not used anywhere in this crate.

use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;
use kornia_image::ImageError;

/// Errors returned by [`crate::track::Tracker`].
///
/// These are *hard* failures: a malformed input, a mis-specified camera, or a
/// kornia error. A frame that simply could not be tracked is **not** an error —
/// [`crate::track::Tracker::process_stereo`] returns
/// [`crate::track::TrackStatus::Untracked`] for that, and
/// the dead-reckoned pose is still available via
/// [`crate::track::Tracker::last_pose`].
#[derive(Debug, thiserror::Error)]
pub enum VioError {
    /// The left and right images have different dimensions. `compute_stereo_matches`
    /// would silently produce garbage (it row-indexes the right image by the left
    /// image's row count), so this is rejected up front.
    #[error("stereo pair size mismatch: left {left_w}x{left_h}, right {right_w}x{right_h}")]
    StereoSizeMismatch {
        /// Left image width.
        left_w: usize,
        /// Left image height.
        left_h: usize,
        /// Right image width.
        right_w: usize,
        /// Right image height.
        right_h: usize,
    },

    /// The camera handed to the tracker carries non-zero distortion.
    ///
    /// `kornia_slam::stereo::unproject_stereo` back-projects **raw** keypoint
    /// coordinates and the keyframe-growth pass builds `K^-1` by hand from
    /// `fx/fy/cx/cy` only — both silently assume a rectified, zero-distortion
    /// camera. Feeding a distorted model makes stereo back-projection disagree
    /// with every other stage, and nothing upstream checks it. So we do.
    #[error(
        "camera must be rectified (zero distortion), got k1={k1} k2={k2} p1={p1} p2={p2}; \
         build it with kornia_3d::stereo::StereoRectifier::rectified_camera()"
    )]
    DistortedCamera {
        /// Radial distortion coefficient k1.
        k1: f64,
        /// Radial distortion coefficient k2.
        k2: f64,
        /// Tangential distortion coefficient p1.
        p1: f64,
        /// Tangential distortion coefficient p2.
        p2: f64,
    },

    /// A non-positive or non-finite stereo baseline was configured. `bf` and the
    /// near/far split `close_depth_threshold = 35 * baseline` both derive from it, so a wrong
    /// baseline corrupts metric scale and the densification gate simultaneously.
    #[error("stereo baseline must be finite and > 0, got {} m", .baseline.get::<meter>())]
    InvalidBaseline {
        /// The offending baseline.
        baseline: Length,
    },

    /// Per-keypoint arrays handed to [`crate::track::Tracker::process_frame`] are
    /// not index-parallel with `features.keypoints_xy`.
    #[error(
        "frame arrays not parallel: {n_keypoints} keypoints but \
         u_right={n_u_right} depth={n_depth} colors={n_colors} \
         (descriptors={n_descriptors}, octaves={n_octaves}, orientations={n_orientations})"
    )]
    FrameArraysNotParallel {
        /// Number of keypoints.
        n_keypoints: usize,
        /// Length of `u_right`.
        n_u_right: usize,
        /// Length of `depth`.
        n_depth: usize,
        /// Length of `keypoint_colors`.
        n_colors: usize,
        /// Length of `features.descriptors`.
        n_descriptors: usize,
        /// Length of `features.octaves`.
        n_octaves: usize,
        /// Length of `features.orientations`.
        n_orientations: usize,
    },

    /// A per-keypoint array handed to
    /// [`crate::track::Tracker::process_frame`] carries a non-finite value.
    ///
    /// The sentinel for "no stereo match" is `-1.0`, which is finite, so this is
    /// always a defect upstream and never a legal encoding. It is worth its own
    /// check because `Frame::stereo_depth` only tests `z > 0.0`: NaN fails that
    /// and is dropped safely, but `+inf` **passes**. `unproject_stereo` then
    /// produces an infinite world point, and the NaNs it turns into propagate
    /// through `run_local_ba` and poison every pose in the active window with
    /// no way back. Rejecting the frame is far cheaper than diagnosing that.
    #[error("frame array `{field}` holds a non-finite value at index {index}: {value}")]
    NonFiniteFrameValue {
        /// Which per-keypoint array: `keypoints_xy`, `u_right` or `depth`.
        field: &'static str,
        /// Index of the offending element.
        index: usize,
        /// The offending value.
        value: f64,
    },

    /// IMU samples were handed to a tracker whose inertial path is not configured.
    ///
    /// Loud rather than silent because the two ways to get here are a graph that forgot the
    /// `inertial` config block and a graph that pushes samples nobody reads — and the second
    /// one looks exactly like a working inertial front end from the outside. The tracker
    /// keeps running visual-only either way; it is the caller that must stop pushing.
    #[error(
        "inertial path is not configured (TrackerConfig::inertial is None); \
         supply the camera-to-IMU extrinsic and the IMU noise densities"
    )]
    InertialDisabled,

    /// The producer says the samples are already rotated into the camera frame.
    ///
    /// `imu_t_bc` would then apply that rotation a SECOND time. The result is a gravity vector
    /// tilted by the extrinsic, which `ImuInitializer` absorbs into the accelerometer bias and
    /// the world rotation — a plausible map, tipped over, with every health field green.
    #[error(
        "IMU samples are flagged as already rotated into the camera frame, but imu_t_bc is \
         configured to do that rotation; publish raw chip axes or clear imu_t_bc"
    )]
    InertialSamplesAligned,

    /// The configured `imu_t_bc` rotation is not a rotation.
    ///
    /// Checked because the usual way to get one wrong is a row/column-major mix-up, and a
    /// TRANSPOSED rotation is still a perfectly valid rotation that nothing downstream can
    /// flag. This catches only the cruder mistake — a typo, a scale factor, a reflection — but
    /// that one it catches at construction instead of after a flight.
    #[error(
        "imu_t_bc rotation is not a rigid rotation: max|R^T R - I| = {orthonormality_error:e}, \
         det = {determinant} (want 0 and +1)"
    )]
    InertialExtrinsicNotRigid {
        /// Largest absolute entry of `R^T R - I`.
        orthonormality_error: f64,
        /// Determinant of `R`. A value near -1 is a reflection, i.e. a mirrored frame.
        determinant: f64,
    },

    /// An IMU noise density, or the sample rate, is not a usable positive number.
    ///
    /// A zero or negative noise density divides by zero inside the covariance propagation and
    /// produces an infinite information matrix — the IMU term then outweighs every visual
    /// constraint in the window and BA converges onto the inertial dead reckoning alone.
    #[error("inertial parameter `{field}` must be finite and > 0, got {value}")]
    InertialParamInvalid {
        /// Which parameter.
        field: &'static str,
        /// The offending value.
        value: f64,
    },

    /// ORB detection or a pyramid resize failed inside kornia-imgproc.
    #[error("image operation failed: {0}")]
    Image(#[from] ImageError),
}
