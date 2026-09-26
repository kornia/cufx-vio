//! Stereo visual-odometry tracking core.
//!
//! This module is the **composition root** over `kornia-slam`'s public API.
//! `kornia-slam` ships the algorithms (stereo matching, map-projection PnP,
//! bundle adjustment, the map itself) but its orchestration lives in
//! `examples/orb_slam/src/pipeline.rs`, a `publish = false` binary crate with no
//! `lib.rs` — unreachable as a dependency. `kornia-slam` is read-only for us, so
//! the *logic* of that pipeline is re-implemented here for a live stereo camera.
//! Nothing is vendored.
//!
//! # Pose convention
//!
//! Every pose in this module — [`TrackedPose::pose_world_to_cam`], the poses
//! stored on keyframes, everything `kornia_slam` returns — is
//! **world-to-camera** (`kornia_3d::pose::Pose3d`): `p_cam = R * p_world + t`.
//!
//! A viewer or a `TransformStamped` almost always wants the **inverse**:
//! camera-in-world, i.e. where the camera *is*. Use
//! [`TrackedPose::cam_in_world`]. Publishing `pose_world_to_cam` directly is the
//! classic "frustum flies backwards" bug.
//!
//! The world frame is fixed at the first successfully bootstrapped frame's pose,
//! which is identity. Scale is **metric**, taken from the stereo baseline — no
//! Sim(3) gauge freedom, which is why the synthetic test asserts absolute
//! trajectory error without any alignment step.
//!
//! # Input contract
//!
//! [`Tracker::process_stereo`] takes a **rectified, row-aligned** pair and a
//! zero-distortion [`PinholeCamera`] describing it — build both with
//! `kornia_3d::stereo::StereoRectifier`. `kornia_slam::stereo::unproject_stereo`
//! back-projects raw keypoint coordinates and the keyframe-growth pass builds
//! `K^-1` by hand from `fx/fy/cx/cy`; both silently assume zero distortion and
//! neither checks. [`Tracker`] checks, and returns
//! [`VioError::DistortedCamera`].
//!
//! # Frame indices
//!
//! `kornia_slam` keys keyframes by `Frame::idx` (`Map::get_keyframe` is a linear
//! scan over it) and `KeyframePolicy::should_insert` measures the keyframe gap in
//! those indices. The tracker therefore owns the counter: it stamps every frame
//! with its own **received-frame count**, ignoring any `idx` on a caller-supplied
//! [`Frame`]. Dropped camera frames consequently do *not* inflate the gap and
//! force a keyframe, which is the behaviour a live variable-rate stream wants;
//! the upstream example, reading a dataset, keys off source indices instead.
//!
//! # The inertial path, and why it is off
//!
//! [`TrackerConfig::inertial`] is `None` unless the caller supplies a full
//! camera-to-IMU extrinsic and the four IMU noise densities, and with it `None`
//! this module is byte-for-byte the stereo-only tracker it has always been: no
//! buffer is allocated, `Map::add_imu_factor` is never called, local BA stays on
//! `run_local_ba`, and [`Tracker::push_imu`] refuses.
//!
//! It is off by default because an OAK-D answers `imu_to_camera_extrinsics` with
//! "IMU calibration data is not available on device yet", so without a measured
//! extrinsic there is no value that would be anything but a guess. The
//! static gravity read pins roll and pitch (chip +y is up, camera +y is down);
//! yaw about gravity is unresolved and needs a motion-correlated hand-eye solve.
//! A guessed extrinsic is not a degraded answer: `ImuInitializer` solves for
//! gravity in the frame the extrinsic names, and `apply_initialization` then
//! ROTATES THE WHOLE MAP onto it, so a wrong extrinsic produces a self-consistent
//! trajectory that is tipped over, with every health field reading green.
//!
//! # What is deliberately absent
//!
//! No loop closure. No transport, no camera I/O, no threads — this is a pure
//! function of the frames and samples you feed it, so it is unit-testable.

use std::collections::HashSet;

use cu29::prelude::*;
use cu29::units::si::f64::{Length, Time};
use cu29::units::si::frequency::hertz;
use cu29::units::si::length::meter;
use cu29::units::si::time::second;
use kornia_3d::camera::PinholeCamera;
use kornia_3d::pose::{Pose3d, TriangulationConfig, triangulate_matched_points};
use kornia_algebra::{Mat3F64, Vec2F64, Vec3F64};
use kornia_image::{Image, ImageSize, InterpolationMode};
use kornia_imgproc::features::{
    OrbDetector, OrbMatchConfig, hamming_distance, match_orb_descriptors,
};
use kornia_imgproc::resize::resize_fast_mono;
use kornia_sensors::imu::{GRAVITY_MAGNITUDE, ImuBias, PreintegratedImu};
use kornia_slam::Frame;
use kornia_slam::estimation::imu_init::ImuInitResult;
use kornia_slam::estimation::map_projection::{MapProjectionConfig, MapProjectionRejectReason};
use kornia_slam::estimation::pnp::{self, PnpConfig};
use kornia_slam::estimation::two_view::TwoViewInitConfig;
use kornia_slam::estimation::{ImuInitializer, MapProjectionEstimator};
use kornia_slam::map::{Keyframe, Map, ORB_N_LEVELS, ORB_SCALE_FACTOR, TriangulatedPoint};
use kornia_slam::stereo::{StereoMatchConfig, compute_stereo_matches, unproject_stereo};
use kornia_slam::system::{
    KeyframePolicy, SystemMode, SystemState, TrackingLossRecoveryPolicy, TrackingStatus,
};
use kornia_slam::tracking::{LocalMapSelectionConfig, select_local_map_points};

use crate::error::VioError;
use crate::imu::{
    ImuBuffer, ImuWindowError, InertialConfig, InertialStats, RawImuSample, WindowExcitation,
    ts_sec, window_excitation,
};
use cu_stereo_payloads::ImuSample;

/// Close-depth threshold in baselines: the near/far split is `35 * baseline`
/// metres (the stereo close/far split of Campos et al., ORB-SLAM3, IEEE T-RO
/// 2021). Points nearer than that are back-projected straight from stereo at
/// every keyframe; farther ones are left to multi-view triangulation.
pub const CLOSE_DEPTH_BASELINES: f64 = 35.0;

/// Minimum stereo-depth keypoints required to bootstrap a metric map from a
/// single frame (single-frame stereo initialization, as in Campos et al.,
/// ORB-SLAM3, IEEE T-RO 2021). This is the *only*
/// bootstrap gate: no parallax check, no two-view estimation, no post-BA health
/// check — metric scale comes from the baseline, not from motion.
pub const DEFAULT_MIN_BOOTSTRAP_STEREO_POINTS: usize = 50;

/// Relocalization-on-loss: match the lost frame's descriptors against candidate keyframes'
/// map-point associations and PnP back onto the EXISTING map, instead of surrendering the
/// map to a re-bootstrap. Thresholds are provisional — mirrored from the tracking-path
/// gates where analogous; revisit against a measured loss corpus. When that corpus exists,
/// candidate retrieval should move to `kornia_slam::place_recognition` (BoW, handles the
/// post-drift/kidnap cases distance ordering cannot) and the matcher to
/// `match_orb_descriptors` (ratio + orientation gates); the estimator itself is
/// upstream-shaped (`kornia_slam::estimation`) once measured.
#[derive(Debug, Clone)]
pub struct RelocConfig {
    /// Candidate keyframes per attempt, ordered by camera-centre distance to the last known
    /// pose (the overwhelmingly common loss is a blip near the frontier).
    pub max_candidates: usize,
    /// Hamming gate for descriptor pairing (the projection matcher's lost-mode gate).
    pub max_hamming: u32,
    /// Minimum 2D-3D pairs before PnP is attempted for a candidate.
    pub min_matches: usize,
    /// Minimum PnP inliers to accept a relocalization — the acceptance gate applied here,
    /// after `solve_pnp` returns (`PnpConfig::min_inliers` is not read by that solver).
    pub min_inliers: usize,
    /// Prior-reprojection gate handed to PnP. The prior is the CANDIDATE KEYFRAME's own
    /// pose, which can be far from the truth — wide by design; the tight final threshold
    /// still does the accepting.
    pub prior_reproj_threshold_px: f64,
    /// Keep attempting relocalization this long before the loss falls through to the
    /// re-bootstrap (or the map drop, with [`TrackerConfig::reset_map_on_loss`]).
    pub patience: Time,
}

impl Default for RelocConfig {
    fn default() -> Self {
        Self {
            max_candidates: 12,
            max_hamming: 60,
            min_matches: 20,
            min_inliers: 30,
            prior_reproj_threshold_px: 200.0,
            patience: Time::new::<second>(3.0),
        }
    }
}

/// How many recent keyframes the growth and fuse passes consider as
/// "covisible" neighbours (recency is the covisibility proxy; there is no
/// covisibility graph in the frontend yet).
pub const DEFAULT_MAX_COVISIBLE_KEYFRAMES: usize = 10;

/// Sample every Nth keyframe for the `map growth` line; a line per keyframe is ~1 MB/h.
const KF_GROWTH_SAMPLE: usize = 25;

/// Minimum epipolar-consistent matches before the growth pass will triangulate.
const MIN_GROWTH_MATCHES: usize = 15;
/// 1-DOF chi-square gate at 95% for point-to-epipolar-line distance, scaled per
/// octave (the epipolar check of Campos et al., ORB-SLAM3, IEEE T-RO 2021).
const EPIPOLAR_CHI2: f64 = 3.84;
/// Fuse pass: projection search radius, in pixels.
const FUSE_SEARCH_RADIUS_PX: f32 = 7.0;
/// Fuse pass: descriptor distance ceiling.
const FUSE_MAX_HAMMING: u32 = 50;
/// How many trailing keyframes `Map::run_local_ba` leaves free (its private
/// `MAX_ACTIVE_KFS`). Mirrored here only to know when at least one pose is
/// *fixed*, i.e. when the BA gauge is anchored — see the call site. If upstream
/// changes its constant, BA here simply starts one keyframe early or late; it
/// can never make the solve unanchored, because the guard is `>`.
const BA_ACTIVE_KEYFRAMES: usize = 3;

/// Tuning for [`Tracker`].
///
/// Construct with [`TrackerConfig::new`], which needs the metric stereo
/// baseline: `bf = fx * baseline` sets metric depth, and `35 * baseline` sets
/// the near/far densification split, so a wrong baseline corrupts scale and the
/// split at once. There is no `Default` on purpose — a guessed baseline is a
/// silent metric error.
///
/// Not `Debug`/`Clone`: `kornia_imgproc::features::OrbDetector` implements
/// neither, and wrapping it would hide which detector is actually in use.
pub struct TrackerConfig {
    /// Metric stereo baseline. Take it from
    /// `StereoRectifier::baseline()`, *not* from depthai's
    /// `getBaselineDistance()` (which defaults to design values in centimetres).
    pub baseline: Length,
    /// ORB detector used for both eyes.
    ///
    /// `downscale` / `n_scales` are pinned to `kornia_slam::map`'s
    /// `ORB_SCALE_FACTOR` / `ORB_N_LEVELS` by [`TrackerConfig::new`]: the stereo
    /// coordinate mapping, the scale-invariance gates and the epipolar sigma all
    /// read those constants independently, and a mismatch is silent.
    pub orb: OrbDetector,
    /// Keyframe insertion heuristics.
    pub keyframe_policy: KeyframePolicy,
    /// Map-projection tracker thresholds (projection search, PnP).
    pub map_projection: MapProjectionConfig,
    /// Supplies the descriptor matcher + triangulation thresholds used by the
    /// keyframe growth pass. (Two-view *initialization* is never run: stereo
    /// bootstraps from one frame.)
    pub two_view_init: TwoViewInitConfig,
    /// Grace period before a sustained tracking failure resets to bootstrap.
    pub tracking_loss_recovery: TrackingLossRecoveryPolicy,
    /// Run local bundle adjustment after each keyframe insertion.
    pub enable_local_ba: bool,
    /// Close-depth threshold. `None` disables per-keyframe stereo
    /// densification. [`TrackerConfig::new`] sets `Some(35 * baseline)`.
    pub stereo_close_depth: Option<Length>,
    /// Bootstrap gate; see [`DEFAULT_MIN_BOOTSTRAP_STEREO_POINTS`].
    pub min_bootstrap_stereo_points: usize,
    /// Relocalize against the existing map while lost, before any re-bootstrap. `None`
    /// disables (pre-relocalization behaviour).
    pub relocalization: Option<RelocConfig>,
    /// On a sustained tracking loss, drop the WHOLE map (full [`reset`](Tracker::reset)) instead
    /// of re-bootstrapping into it. The append-to-map path leaves `current_keyframe_idx` at
    /// `None`, so the next local-map build returns every non-culled point and per-frame
    /// tracking cost jumps with total map size — measured on a Jetson Orin with an OAK-D at
    /// 640x400: 3 frames processed in 15 s after a loss, a spiral that never re-establishes.
    /// Dropping the map bounds cost and memory; a consumer re-anchors via
    /// [`world_generation`](Tracker::world_generation). Default false: accumulating across
    /// losses is the upstream behaviour, and what offline evaluation expects.
    pub reset_map_on_loss: bool,
    /// Drop the map once it holds this many keyframes. `None` is unbounded.
    ///
    /// kornia-slam never removes a keyframe — `Map::cull` only marks map points, and
    /// `keyframes`/`map_points` are `Vec`s that only push — so insertion cost rises without
    /// limit. Measured on a Jetson Orin with an OAK-D at 640x400: 62 ms at 25 keyframes, 260 ms
    /// at 137, 427 ms at 156, then a silent stall (one thread at 99.9 %, 757 MB RSS, zero poses
    /// after 2.5 h).
    /// `reset_map_on_loss` bounds this ONLY when tracking fails; tracking cleanly in good light
    /// never trips it, so the better the conditions the sooner VIO dies.
    pub max_keyframes: Option<usize>,
    /// Neighbour window for grow/fuse; see [`DEFAULT_MAX_COVISIBLE_KEYFRAMES`].
    pub max_covisible_keyframes: usize,
    /// Inertial front end. `None` — what [`TrackerConfig::new`] sets — is not a degraded mode
    /// but the ONLY mode until the camera-to-IMU extrinsic is measured; see the module docs.
    ///
    /// `Option` rather than a `bool` beside a defaulted `InertialConfig` on purpose: there is
    /// no `Default for InertialConfig`, and its only constructor demands the extrinsic and the
    /// noise densities, so "inertial on with a guessed extrinsic" is not a state this type can
    /// represent.
    pub inertial: Option<InertialConfig>,
}

impl TrackerConfig {
    /// Builds a config for a rectified pair with the given metric baseline.
    pub fn new(baseline: Length) -> Self {
        // Upstream's `PipelineConfig::default` tightens two triangulation
        // thresholds relative to kornia-3d's defaults; both feed the growth pass.
        let mut two_view_init = TwoViewInitConfig::default();
        two_view_init.triangulation_config.max_midpoint_gap = 0.25;
        two_view_init.triangulation_config.max_reprojection_error = 3.0;

        Self {
            baseline,
            orb: OrbDetector {
                n_keypoints: 1000,
                downscale: ORB_SCALE_FACTOR as f32,
                n_scales: ORB_N_LEVELS,
                ..OrbDetector::default()
            },
            keyframe_policy: KeyframePolicy::default(),
            // Upstream's `MapProjectionConfig::default()`, unmodified.
            //
            // Lowering `pnp.outlier_rounds` from 4 to 2 does not show a reproducible saving: one
            // A/B read 13.6 ms (est_ms median 83.8 -> 70.2, 500 frames each), but four repeats of
            // the rounds=2 config gave medians of 70.2, 90.7, 91.1 and 88.5 — a ~20 ms run-to-run
            // spread that is larger than the effect, and which puts the rounds=4 baseline (83.8)
            // BELOW the rounds=2 cluster. Matched settle windows minutes apart were not enough:
            // something drifts between runs (map state, scene, or the board's thermal state) that
            // dominates.
            //
            // Fewer refit rounds is still plausibly cheaper, but it costs outlier
            // re-classification on a tracker whose reliability is already the weak link, and an
            // unproven saving does not buy that.
            //
            // To settle it properly, `outlier_rounds` needs to be runtime-configurable so the two
            // can be A/B'd many times within one session without a rebuild, or the measurement
            // needs to control for map size rather than only for settle time.
            map_projection: MapProjectionConfig::default(),
            two_view_init,
            tracking_loss_recovery: TrackingLossRecoveryPolicy::default(),
            enable_local_ba: true,
            stereo_close_depth: Some(baseline * CLOSE_DEPTH_BASELINES),
            min_bootstrap_stereo_points: DEFAULT_MIN_BOOTSTRAP_STEREO_POINTS,
            relocalization: Some(RelocConfig::default()),
            reset_map_on_loss: false,
            max_keyframes: None,
            max_covisible_keyframes: DEFAULT_MAX_COVISIBLE_KEYFRAMES,
            inertial: None,
        }
    }
}

/// The outcome of one frame: a measured pose, or none.
///
/// Untracked is a controlled outcome of a healthy tracker, not a failure, so it is a variant here
/// rather than an error or an `Option` inside the `Result`: the `Err` side of
/// [`Tracker::process_stereo`] is reserved for malformed input and kornia errors.
#[derive(Debug, Clone)]
pub enum TrackStatus {
    /// The pose was measured: the frame was tracked, or a keyframe was accepted.
    Tracked(TrackedPose),
    /// The frame could not be placed. The tracker still coasts an extrapolated pose under the
    /// constant-velocity model, available from [`Tracker::last_pose`], but it is unverified and
    /// must not be published as a measurement.
    Untracked,
}

/// A MEASURED pose for one input frame.
///
/// Only a measured frame produces one: a frame the tracker could not place is
/// [`TrackStatus::Untracked`], so there is no "untracked" state to check here as well.
#[derive(Debug, Clone)]
pub struct TrackedPose {
    /// Capture time of the frame this pose belongs to, exactly as handed to
    /// [`Tracker::process_stereo`].
    pub stamp: CuTime,
    /// **World-to-camera** pose. See the module docs before publishing it.
    pub pose_world_to_cam: Pose3d,
    /// Whether a keyframe was inserted for this frame. Per-frame cost is bimodal on it.
    pub keyframe: bool,
}

impl TrackedPose {
    /// Camera-in-world: the transform a viewer / `TransformStamped` wants.
    pub fn cam_in_world(&self) -> Pose3d {
        self.pose_world_to_cam.inverse()
    }
}

/// Snapshot of tracker health, for the periodic log. Diagnostics only — none of
/// these is an acceptance criterion (an odometry front end can publish 4x the true
/// translation at a perfect RMSE and inlier fraction).
#[derive(Debug, Clone, Copy)]
pub struct TrackerStats {
    /// Frames handed to the tracker so far.
    pub frames: u64,
    /// Keypoints detected in the most recent left image.
    pub keypoints: usize,
    /// Left keypoints with a valid stereo depth in the most recent frame.
    pub stereo_matches: usize,
    /// PnP inliers on the most recent frame (0 when tracking failed).
    pub inliers: usize,
    /// Non-culled map points. Culling is logical, so this can shrink while
    /// `Map::num_map_points()` only grows.
    pub map_points: usize,
    /// Keyframes in the map. Never culled, so this only grows — including
    /// across tracking losses, which do *not* clear the map on the stereo path.
    pub keyframes: usize,
    /// Status of the most recent frame; `None` before the first frame.
    pub status: Option<TrackingStatus>,
    /// Current pipeline mode.
    pub mode: SystemMode,
    /// Consecutive frames that failed to track (0 while healthy).
    pub consecutive_failures: u32,
}

impl Default for TrackerStats {
    fn default() -> Self {
        Self {
            frames: 0,
            keypoints: 0,
            stereo_matches: 0,
            inliers: 0,
            map_points: 0,
            keyframes: 0,
            status: None,
            mode: SystemMode::Bootstrap,
            consecutive_failures: 0,
        }
    }
}

/// The tracker's inertial half: the sample ring, the initializer, and the estimates
/// `kornia_slam`'s inertial APIs keep OUTSIDE the map.
///
/// `Map` stores per-keyframe velocity and bias, but the *current* linearization bias, the
/// gravity direction and the staged initialization's progress live nowhere in the library — the pipeline
/// owns them, and this tracker does not use the pipeline. This struct is that ownership.
struct InertialState {
    cfg: InertialConfig,
    buffer: ImuBuffer,
    initializer: ImuInitializer,
    /// Linearization point for the NEXT edge. Refreshed from the newest keyframe after every
    /// local BA: leaving it at the bootstrap value makes every edge linearize at a stale bias,
    /// which drives `Map`'s 0.02 repropagation threshold to fire on every factor, every solve.
    bias: ImuBias,
    /// Gravity in the world frame. `(0, 0, -9.81)` until `apply_initialization` gravity-aligns
    /// the map, after which it is `(0, +9.81, 0)` — OpenCV Y-down, matching
    /// `ViBaParams::default().gravity`. Handing `run_local_inertial_ba` the pre-init value
    /// would rotate the inertial residual 90 degrees against the map it is constraining.
    gravity_world: Vec3F64,
    /// `Frame::idx` of the keyframe the initialization window starts at. The window never
    /// shrinks, which is why the retry below is throttled.
    start_kf_idx: Option<usize>,
    /// Timestamp of that keyframe: the start of the initialization window, and the clock the
    /// two refinement stages are scheduled on.
    window_start_sec: Option<f64>,
    last_init_attempt_sec: Option<f64>,
    /// When [`Tracker::log_init_window`] last spoke, and the verdict it reported. Its OWN clock,
    /// because `last_init_attempt_sec` is not assigned until `ready` first answers true and so
    /// throttles nothing in the case that actually runs long: a parked robot.
    last_window_log: Option<(f64, bool)>,
    /// Set when `enable_inertial_ba` is off and `try_initialize` has returned an accepted
    /// result for the current window that was then NOT applied. The dry run exists to answer
    /// one question — would this extrinsic pass the initializer's gates, and what gravity and
    /// bias would it produce — and one acceptance answers it. Without this latch the initial-solve
    /// branch would keep paying a 200-iteration LM over a never-shrinking window every
    /// `init_retry` for the rest of the session, because `state.imu_initialized` (the
    /// thing that normally ends that loop) is only ever set by `apply_initialization`.
    dry_run_accepted: bool,
    /// Counters, plus the staged initialization's progress (`first_refinement_done` / `second_refinement_done` mark the
    /// first and second refinement, live here and nowhere else, and are read by the ladder and
    /// published as-is).
    stats: InertialStats,
}

impl InertialState {
    fn new(cfg: InertialConfig) -> Self {
        Self {
            buffer: ImuBuffer::with_capacity(cfg.buffer_capacity),
            initializer: ImuInitializer::new(cfg.init.clone()),
            bias: cfg.initial_bias,
            // Mirrors the pipeline's pre-initialization value. It is never read before
            // `apply_initialization` overwrites it, but a zero vector here would be a silent
            // "no gravity" if that invariant ever slipped.
            gravity_world: Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE),
            start_kf_idx: None,
            window_start_sec: None,
            last_init_attempt_sec: None,
            last_window_log: None,
            dry_run_accepted: false,
            stats: InertialStats::default(),
            cfg,
        }
    }

    /// Back to the pre-bootstrap state, keeping the config and the lifetime counters.
    ///
    /// `keep_bias` carries the estimated IMU bias across the reset. Gyro and accelerometer
    /// bias are properties of the SENSOR, not of the map, so a map that was dropped for being
    /// too large says nothing about them — and re-estimating from
    /// [`InertialConfig::initial_bias`] makes the next initial solve land wide and the first
    /// refinement that corrects it rotate the whole world. Measured on a Jetson Orin with an
    /// OAK-D at 640x400: gravity_y 8.94 at the initial solve then 9.81 at the first refinement,
    /// i.e. one large world rotation per map cycle.
    ///
    /// `gravity_world` is NOT carried: it is expressed in the world frame, and the bootstrap
    /// that follows a reset defines a new one.
    fn reset(&mut self, keep_bias: bool) {
        self.buffer.clear();
        if !keep_bias {
            self.bias = self.cfg.initial_bias;
        }
        self.gravity_world = Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE);
        self.start_kf_idx = None;
        self.window_start_sec = None;
        self.last_init_attempt_sec = None;
        // Cleared, so the first window after a reset is described rather than throttled against
        // the window it replaced.
        self.last_window_log = None;
        self.dry_run_accepted = false;
        self.stats.first_refinement_done = false;
        self.stats.second_refinement_done = false;
    }
}

/// Whether a throttled line should speak, given when it last did and what it then said.
///
/// A flipped `verdict` ALWAYS speaks: the transition is the one line worth keeping, and a plain
/// time throttle is exactly what would drop it. Otherwise once per `retry_sec`. Split out so the
/// throttle is testable without a map behind it.
///
/// Two callers. `log_init_window` passes `ready`, where the transition out of "never ready" is the
/// only line anyone waits for. The tracking-loss line passes `map_established`, where losing an
/// ESTABLISHED map and failing to bootstrap one are different faults a time throttle would blend
/// into one stream.
fn should_log_throttled(
    last: Option<(f64, bool)>,
    now_sec: f64,
    verdict: bool,
    retry_sec: f64,
) -> bool {
    match last {
        None => true,
        Some((_, was)) if was != verdict => true,
        Some((then, _)) => now_sec - then >= retry_sec,
    }
}

/// The stages of the staged inertial initialization: the initial solve, then the refinements
/// scheduled at 5 s and 15 s into the window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InitStage {
    Initial,
    Refine1,
    Refine2,
}

impl InitStage {
    /// The stage's label in the structured log.
    fn as_str(self) -> &'static str {
        match self {
            Self::Initial => "initial",
            Self::Refine1 => "refine_1",
            Self::Refine2 => "refine_2",
        }
    }
}

impl std::fmt::Display for InitStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How often the tracking-loss line may repeat while nothing about the fault changes.
///
/// Not cosmetic. Measured on a Jetson Orin with an OAK-D at 640x400 over a 7 h run with a healthy
/// camera: 100,444 losses at 3.88/s, ~26 % of all frames. Unthrottled, that volume is enough to
/// fill an embedded disk. 5 s is ~1 line per 19 losses; `suppressed` carries the rest, so
/// throttling never hides the volume.
const LOSS_LOG_RETRY_SEC: f64 = 5.0;

/// Replace a world-to-camera pose's rotation while leaving the camera CENTRE where it is.
///
/// `Pose3d` is world-to-camera (`X_cam = R X_world + t`), so the centre is `C = -R^T t` and
/// writing `rotation` alone moves it by ~theta * |t| — metres once the robot is any distance
/// from the map origin, and worst exactly while turning, which is when a rotation prior is for.
/// Extracted so a test can exercise THIS, rather than re-deriving the same two lines beside it.
fn rotate_about_centre(pose: Pose3d, rotation: Mat3F64) -> Pose3d {
    let centre = pose.inverse().translation;
    Pose3d::from_rt(rotation, -(rotation * centre))
}

/// Stereo visual-odometry tracker.
///
/// Feed it rectified stereo pairs with [`Tracker::process_stereo`]; it
/// bootstraps a metric map from the first usable frame and then tracks against
/// it, inserting keyframes and running local BA as needed. See the module docs
/// for the pose convention and the input contract.
pub struct Tracker {
    camera: PinholeCamera,
    config: TrackerConfig,
    stereo_config: StereoMatchConfig,
    estimator: MapProjectionEstimator,
    map: Map,
    state: SystemState,
    /// Stamp of the previous frame, for the gyro prior's integration interval.
    last_frame_stamp_ns: Option<u64>,
    /// Frames the gyro prior could not be built for. Counted because the fallback — the
    /// constant-velocity model — is silent and plausible: a prior that never once applied
    /// looks exactly like one that applied perfectly.
    gyro_prior_unavailable: u64,
    /// Set when an insertion took the map past `max_keyframes`; acted on next frame.
    pending_map_reset: bool,
    /// Monotonic count of frames received; stamped onto `Frame::idx`.
    next_frame_idx: usize,
    /// First timestamp seen; all internal times are seconds relative to it, so
    /// f64 keeps full nanosecond resolution regardless of the caller's epoch.
    epoch_ns: Option<u64>,
    last_keyframe_timestamp_sec: Option<f64>,
    /// The same stamp in the caller's nanosecond clock. Kept alongside the seconds value rather
    /// than reconstructed from it: the IMU interval is queried in ns and integrated in seconds,
    /// and round-tripping through the f64 seconds would move the interval boundary by hundreds
    /// of nanoseconds — enough to include or exclude a sample at the edge, which is exactly the
    /// kind of difference `from_measurements` absorbs without a word.
    last_keyframe_stamp_ns: Option<u64>,
    /// `None` unless [`TrackerConfig::inertial`] was supplied; see the module docs.
    inertial: Option<InertialState>,
    stats: TrackerStats,
    /// When the tracking-loss line last spoke, and what `map_established` was then. Drives
    /// [`should_log_throttled`].
    last_loss_log: Option<(f64, bool)>,
    /// Loss events swallowed by that throttle since it last spoke. Reported on the next line
    /// that does, so the throttle cannot turn 19 losses into the appearance of one.
    losses_suppressed: u64,
    /// Bumped every time the pose chain's frame of reference stops being trustworthy: a
    /// requested [`reset`](Self::reset), or an internal loss re-bootstrap (which re-anchors at
    /// a dead-reckoned coast, not a measurement). Consumers that map tracker poses into
    /// another frame must re-derive their transform when this changes.
    world_generation: u64,
    /// Live (not culled) map points, and the `(keyframe count, world generation)` they were
    /// counted at; see the refresh in `process_frame`.
    live_map_points: usize,
    live_map_points_key: (usize, u64),
}

impl Tracker {
    /// Creates a tracker for a **rectified, zero-distortion** camera.
    ///
    /// The camera and baseline are not validated here (the signature has no
    /// `Result`); they are checked on every `process_*` call, which returns
    /// [`VioError::DistortedCamera`] / [`VioError::InvalidBaseline`]. A bad
    /// configuration is also logged at `error` level right here so it shows up
    /// at startup rather than only in a per-frame error.
    pub fn new(camera: PinholeCamera, config: TrackerConfig) -> Self {
        let baseline_m = config.baseline.get::<meter>();
        let close_depth_m = config.stereo_close_depth.map(|d| d.get::<meter>());
        let stereo_config = StereoMatchConfig::new(
            baseline_m as f32,
            camera.fx as f32,
            ORB_SCALE_FACTOR as f32,
            ORB_N_LEVELS,
        );

        // Startup log mirrors upstream's: a wrong baseline is invisible in the
        // output but corrupts metric scale and the near/far split at once.
        info!(
            "cu-kornia-vio stereo tracker configured: fx={} fy={} cx={} cy={} baseline_m={} bf={} close_depth_m={}",
            camera.fx, camera.fy, camera.cx, camera.cy, baseline_m, stereo_config.bf, close_depth_m
        );
        if let Err(e) = validate_camera(&camera, baseline_m) {
            error!(
                "cu-kornia-vio tracker configuration is invalid: {}",
                e.to_string()
            );
        }

        let inertial = config.inertial.clone().map(InertialState::new);
        if let Some(inert) = &inertial {
            // Loud at startup, because from here on the only externally visible difference is
            // that the world gets rotated once, silently, when initialization succeeds.
            warning!(
                "cu-kornia-vio INERTIAL path enabled: poses are only as good as the configured \
                 imu_t_bc, which nothing downstream can validate (gyro_noise={} accel_noise={} \
                 gyro_bias_noise={} accel_bias_noise={} rate_hz={} inertial_ba={})",
                inert.cfg.calib.gyro_noise,
                inert.cfg.calib.accel_noise,
                inert.cfg.calib.gyro_bias_noise,
                inert.cfg.calib.accel_bias_noise,
                inert.cfg.gates.nominal_rate.get::<hertz>(),
                inert.cfg.enable_inertial_ba
            );
        }

        Self {
            camera,
            stereo_config,
            estimator: MapProjectionEstimator::new(config.map_projection.clone()),
            config,
            map: Map::new(),
            state: SystemState::new(),
            last_frame_stamp_ns: None,
            gyro_prior_unavailable: 0,
            pending_map_reset: false,
            next_frame_idx: 0,
            epoch_ns: None,
            last_keyframe_timestamp_sec: None,
            last_keyframe_stamp_ns: None,
            inertial,
            stats: TrackerStats::default(),
            last_loss_log: None,
            losses_suppressed: 0,
            world_generation: 0,
            live_map_points: 0,
            live_map_points_key: (0, 0),
        }
    }

    /// Feeds one **rectified** stereo pair.
    ///
    /// Returns [`TrackStatus::Tracked`] when the pose is *measured* (tracked, or a
    /// keyframe was accepted) and [`TrackStatus::Untracked`] when the frame could not be
    /// tracked. On `Untracked` the tracker still advances an extrapolated pose under
    /// the constant-velocity model — available from [`Tracker::last_pose`] — but
    /// it is unverified and must not be published as a measurement.
    ///
    /// The frame on which a tracking loss re-bootstraps is `Untracked` too, even
    /// though it *does* create a keyframe: the new map is anchored at the
    /// dead-reckoned pose, so publishing it would be publishing a coast as a
    /// measurement.
    ///
    /// `stamp` is the pair's capture time, on the same clock as the IMU samples. It drives the
    /// tracking-loss grace period and the inertial windows, and is returned unmodified on
    /// [`TrackedPose::stamp`].
    pub fn process_stereo(
        &mut self,
        left: &Image<u8, 1>,
        right: &Image<u8, 1>,
        stamp: CuTime,
    ) -> Result<TrackStatus, VioError> {
        validate_camera(&self.camera, self.config.baseline.get::<meter>())?;
        if left.width() != right.width() || left.height() != right.height() {
            return Err(VioError::StereoSizeMismatch {
                left_w: left.width(),
                left_h: left.height(),
                right_w: right.width(),
                right_h: right.height(),
            });
        }

        let t0 = std::time::Instant::now();
        let features = self.config.orb.detect_and_extract_u8(left)?;
        let right_features = self.config.orb.detect_and_extract_u8(right)?;
        let orb_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Both pyramids come from the same builder, so the SAD correlation
        // compares like with like. NOTE: this is a *single-resize* pyramid
        // (every level resamples the full-res image), which is what
        // `kornia_slam::stereo` assumes when it maps coordinates by
        // `u * 1.2^-octave`. `OrbDetector` builds a *successive*-downscale
        // pyramid internally, whose true scale at level o drifts from 1.2^-o.
        // Do not swap in a successive pyramid here without re-deriving that
        // mapping — and the mapping lives in read-only kornia-slam.
        let left_pyramid = build_u8_pyramid(left)?;
        let right_pyramid = build_u8_pyramid(right)?;
        let pyr_ms = t0.elapsed().as_secs_f64() * 1e3 - orb_ms;

        let stereo = compute_stereo_matches(
            &left_pyramid,
            &right_pyramid,
            &features,
            &right_features,
            &self.stereo_config,
        );
        let stereo_ms = t0.elapsed().as_secs_f64() * 1e3 - orb_ms - pyr_ms;

        let keypoint_colors = sample_keypoint_colors(left, &features.keypoints_xy);
        let image_size = ImageSize {
            width: left.width(),
            height: left.height(),
        };

        let frame = Frame {
            idx: 0, // replaced by `process_frame`
            features,
            pose_world_to_cam: Pose3d::IDENTITY, // replaced by the pipeline
            image_size,
            keypoint_colors,
            u_right: stereo.u_right,
            depth: stereo.depth,
            keypoints_undist: Vec::new(), // filled by `ensure_undistorted`
        };

        let result = self.process_frame(frame, stamp);
        // Per-stage split at debug level: the frontend (ORB + pyramids + stereo) is fixed
        // cost per frame, while `track_ms` (colour sampling + Frame build + process_frame,
        // dominated by process_frame) scales with the map — which one saturates a live loop
        // decides whether the fix is a feature budget or a map bound. The four fields sum
        // to the whole call.
        let track_ms = t0.elapsed().as_secs_f64() * 1e3 - orb_ms - pyr_ms - stereo_ms;
        debug!(
            "process_stereo stage split: orb_ms={} pyr_ms={} stereo_ms={} track_ms={}",
            orb_ms, pyr_ms, stereo_ms, track_ms
        );
        result
    }

    /// Feeds a frame whose features and per-keypoint stereo depths were built
    /// elsewhere — the seam the synthetic tests drive.
    ///
    /// `frame.idx` and `frame.pose_world_to_cam` are **ignored and overwritten**
    /// (see the module docs on frame indices). Every per-keypoint array must be
    /// index-parallel with `features.keypoints_xy`, with `-1.0` sentinels in
    /// `u_right` / `depth` for keypoints without a stereo match; a fully empty
    /// `depth` means "monocular", which this stereo-only tracker can never
    /// bootstrap from and reports as [`TrackingStatus::Skipped`].
    pub fn process_frame(
        &mut self,
        mut frame: Frame,
        stamp: CuTime,
    ) -> Result<TrackStatus, VioError> {
        validate_camera(&self.camera, self.config.baseline.get::<meter>())?;
        validate_frame_arrays(&frame)?;
        let stamp_ns = stamp.as_nanos();

        self.apply_pending_map_reset();

        frame.idx = self.next_frame_idx;
        self.next_frame_idx += 1;

        // Relative seconds: an absolute nanosecond epoch (~1.7e18) would leave
        // f64 with ~400 ns of resolution. Only differences matter here.
        let epoch = *self.epoch_ns.get_or_insert(stamp_ns);
        let ts_sec = stamp_ns.saturating_sub(epoch) as f64 * 1e-9;

        // Fill the per-frame undistortion cache once; tracking, BA gathering,
        // growth and fuse all read from it. Must happen before anything else.
        frame.ensure_undistorted(&self.camera);

        self.stats.frames += 1;
        self.stats.keypoints = frame.features.keypoints_xy.len();
        self.stats.stereo_matches = frame.depth.iter().filter(|&&d| d > 0.0).count();
        self.stats.inliers = 0;

        let status = match self.state.mode {
            SystemMode::Bootstrap => self.bootstrap_stereo(frame, ts_sec, stamp_ns),
            SystemMode::Tracking => self.tracking_step(frame, ts_sec, stamp_ns),
            SystemMode::ImuInit => {
                // Unreachable in BOTH configurations, and deliberately so. `SlamPipeline` parks
                // in `ImuInit` and stops tracking until initialization succeeds; this tracker
                // keeps tracking visually throughout and runs the initializer as a side effect
                // of keyframe insertion, so the mode is never entered even with the inertial
                // path on. Recover rather than panic on a live stream.
                error!("cu-kornia-vio reached SystemMode::ImuInit, which neither path ever sets");
                self.state.mode = SystemMode::Bootstrap;
                self.bootstrap_stereo(frame, ts_sec, stamp_ns)
            }
        };

        // EVERY path, not just `tracking_step`: a frame handled by `bootstrap_stereo` that left
        // this stale would make the first tracking frame afterwards integrate the gyro across
        // the whole bootstrap gap — seconds of rotation applied as if it were one frame.
        self.last_frame_stamp_ns = Some(stamp_ns);

        self.stats.status = Some(status);
        self.stats.mode = self.state.mode;
        // `map_points` and `keyframes` are deliberately NOT filled in here.
        // `num_active_map_points()` is an O(every map point ever created) scan
        // and this runs on every frame, for two numbers the health log reads
        // once every 30 — an unbounded per-frame cost for a log field. They are
        // computed on demand in [`Tracker::stats`] instead.
        //
        // The live-point count a consumer publishes per frame is cached instead. Every point this
        // tracker creates or culls does so inside a keyframe insertion (bootstrap included), and
        // every map drop bumps the world generation, so the scan runs only when one of those two
        // moved: once per keyframe, not once per frame.
        let key = (self.map.keyframes().len(), self.world_generation);
        if key != self.live_map_points_key {
            self.live_map_points = self.map.num_active_map_points();
            self.live_map_points_key = key;
        }

        match status {
            TrackingStatus::Skipped => {
                self.stats.consecutive_failures = self.stats.consecutive_failures.saturating_add(1);
                Ok(TrackStatus::Untracked)
            }
            TrackingStatus::Tracked | TrackingStatus::KeyframeAccepted => {
                self.stats.consecutive_failures = 0;
                Ok(TrackStatus::Tracked(TrackedPose {
                    stamp,
                    pose_world_to_cam: self.state.pose_world_to_cam,
                    keyframe: status == TrackingStatus::KeyframeAccepted,
                }))
            }
        }
    }

    /// Health snapshot; see [`TrackerStats`].
    ///
    /// `map_points` / `keyframes` are counted here rather than on every frame:
    /// both are linear scans over the whole map, so paying them per frame would
    /// make frame time grow with trajectory length for the sake of a log line.
    /// Call this at the log cadence, not per frame.
    pub fn stats(&self) -> TrackerStats {
        TrackerStats {
            map_points: self.map.num_active_map_points(),
            keyframes: self.map.keyframes().len(),
            ..self.stats
        }
    }

    /// Rotation the gyro says the CAMERA turned through over `[t0_ns, t1_ns]`.
    ///
    /// Returns `R_c0_c1`: `R_world_to_cam` at t1 is `delta.transpose() * R_world_to_cam(t0)`.
    /// Derivation, because the transposes are where this goes silently wrong —
    /// `dR_wb/dt = R_wb [w]x` gives `R_wb1 = R_wb0 dR_b`, and with `X_body = R_BC X_cam`
    /// (`imu_t_bc`), `dR_c = R_BC^T dR_b R_BC`.
    ///
    /// Integrated first-order per sample and re-orthonormalised at the end: at 200 Hz one step
    /// is ~1e-4 rad, far below where the small-angle error matters over a 66 ms frame interval.
    fn gyro_camera_delta(&mut self, t0_ns: u64, t1_ns: u64) -> Option<Mat3F64> {
        let inert = self.inertial.as_ref()?;
        let bias = inert.bias.gyro;
        let r_bc = inert.cfg.imu_t_bc.rotation;
        // Straight off the borrowed window, holding the previous sample: no per-frame copy of
        // the interval.
        let integrated = inert
            .buffer
            .window(t0_ns, t1_ns, &inert.cfg.gates)
            .ok()
            .and_then(|mut window| {
                let mut prev = *window.next()?;
                let mut d = Mat3F64::IDENTITY;
                let mut pairs = 0usize;
                for &s in window {
                    pairs += 1;
                    let dt = (s.stamp_ns.saturating_sub(prev.stamp_ns)) as f64 * 1e-9;
                    if dt > 0.0 {
                        let w = Vec3F64::new(prev.gyro[0], prev.gyro[1], prev.gyro[2]) - bias;
                        let k = Mat3F64::from_cols_array(&[
                            0.0, w.z, -w.y, //
                            -w.z, 0.0, w.x, //
                            w.y, -w.x, 0.0,
                        ]);
                        d += d * k * dt;
                    }
                    prev = s;
                }
                (pairs >= 1).then_some(d)
            });
        let Some(d) = integrated else {
            self.gyro_prior_unavailable += 1;
            return None;
        };
        // Gram-Schmidt: the first-order steps drift off SO(3), and `estimate_pose` is handed
        // this as a rotation.
        let c = d.to_cols_array();
        let x = Vec3F64::new(c[0], c[1], c[2]).normalize();
        let y0 = Vec3F64::new(c[3], c[4], c[5]);
        let y = (y0 - x * x.dot(y0)).normalize();
        let z = x.cross(y);
        let d = Mat3F64::from_cols_array(&[x.x, x.y, x.z, y.x, y.y, y.z, z.x, z.y, z.z]);

        Some(r_bc.transpose() * d * r_bc)
    }

    /// Arm the map bound if this insertion took the map to [`TrackerConfig::max_keyframes`].
    ///
    /// Split out from the insertion path so the predicate is reachable from a test: every
    /// conjunct here is a way to silently disable the bound or to fire it every frame.
    fn arm_map_bound(&mut self, inserted: bool) {
        if let Some(cap) = self.config.max_keyframes
            && inserted
            && self.map.keyframes().len() >= cap
        {
            self.pending_map_reset = true;
        }
    }

    /// Drop the map if the last insertion tripped [`TrackerConfig::max_keyframes`].
    ///
    /// DEFERRED from the keyframe that tripped it: resetting there would wipe the state the
    /// caller is still reading that frame's pose out of. Bumps `world_generation` like any
    /// other reset, so a consumer re-anchors on `reset_epoch`.
    fn apply_pending_map_reset(&mut self) {
        if !std::mem::take(&mut self.pending_map_reset) {
            return;
        }
        warning!(
            "map bound reached: dropping the map (TrackerConfig::max_keyframes) keyframes={} cap={}",
            self.map.keyframes().len(),
            self.config.max_keyframes
        );
        // Keeps the IMU bias: the map hit a size cap, the sensor did not change.
        self.reset_keeping_calibration();
    }

    /// Frames the gyro rotation prior could not be built for. See the field.
    pub fn gyro_prior_unavailable(&self) -> u64 {
        self.gyro_prior_unavailable
    }

    /// The tracker's current world-to-camera pose, measured or extrapolated.
    ///
    /// After a frame that returned [`TrackStatus::Untracked`] this is the constant-velocity
    /// prediction, not a measurement.
    pub fn last_pose(&self) -> Pose3d {
        self.state.pose_world_to_cam
    }

    /// Read-only access to the map (points, keyframes) for viewers and tests.
    pub fn map(&self) -> &Map {
        &self.map
    }

    /// Current pipeline mode.
    pub fn mode(&self) -> SystemMode {
        self.state.mode
    }

    /// The current world generation; see the field doc. Monotone, starts at 0.
    ///
    /// A successful visual-inertial initialization bumps this too — but only with
    /// `enable_inertial_ba` on: `apply_initialization` scales and rotates the whole map to put
    /// gravity on world +y, so poses before and after it are in different frames. With the
    /// flag off the initialization is a dry run and this counter does not move.
    pub fn world_generation(&self) -> u64 {
        self.world_generation
    }

    /// Live (not culled) map points, as of the last frame. Cheap: counted once per keyframe
    /// insertion or map drop, not per call. [`Tracker::stats`] recounts on demand.
    pub fn live_map_points(&self) -> usize {
        self.live_map_points
    }

    /// The live (not culled) points among the newest `cap` map-point slots, oldest first, each
    /// as `(map-point index, world-frame position in metres)`.
    ///
    /// Newest, not all: kornia-slam's map is append-only, so the tail is the frontier being built
    /// now, and walking only the tail keeps the cost at O(`cap`) however long the map has grown.
    /// The result is a complete view of that window, meant to REPLACE the previous one: a point
    /// culled or fused since is simply absent, and after a [`Tracker::world_generation`] change
    /// (map dropped, or kept in a moved world) the window describes the new world. The index is
    /// the point's position in [`Map::map_points`], stable while the generation holds.
    pub fn newest_live_map_points(
        &self,
        cap: usize,
    ) -> impl Iterator<Item = (usize, [f64; 3])> + '_ {
        let mps = self.map.map_points();
        let first = mps.len().saturating_sub(cap);
        mps[first..]
            .iter()
            .enumerate()
            .filter(|(_, mp)| !mp.culled)
            .map(move |(i, mp)| {
                let p = mp.position;
                (first + i, [p.x, p.y, p.z])
            })
    }

    /// Hands raw IMU samples to the tracker, to be preintegrated into the edge between the next
    /// two keyframes.
    ///
    /// `dropped_cumulative` is the producer's CUMULATIVE dropped-sample count (never reset, so
    /// only its increase is a new hole) and `aligned` says whether the samples were already
    /// rotated into the camera frame. Returns how many samples were accepted; samples whose
    /// stamp does not advance are rejected and counted rather than silently sorted later.
    ///
    /// Each sample's `tov` must be on the SAME clock as the `stamp` handed to
    /// [`Tracker::process_stereo`] — for an OAK-D stereo pair and IMU they are, by construction.
    ///
    /// # Errors
    ///
    /// [`VioError::InertialDisabled`] when no extrinsic is configured, and
    /// [`VioError::InertialSamplesAligned`] when the samples are already in the camera frame.
    /// Both are wiring mistakes that otherwise present as a plausible, wrong map: the first
    /// means nobody is reading what the caller is pushing, the second means `imu_t_bc` would
    /// rotate the samples a second time.
    pub fn push_imu(
        &mut self,
        samples: impl IntoIterator<Item = ImuSample>,
        dropped_cumulative: u64,
        aligned: bool,
    ) -> Result<usize, VioError> {
        self.push_raw_imu(
            samples.into_iter().map(RawImuSample::from),
            dropped_cumulative,
            aligned,
        )
    }

    /// [`Tracker::push_imu`] for samples already unpacked, which is what the bus queue holds.
    pub(crate) fn push_raw_imu(
        &mut self,
        samples: impl IntoIterator<Item = RawImuSample>,
        dropped_cumulative: u64,
        aligned: bool,
    ) -> Result<usize, VioError> {
        let Some(inert) = self.inertial.as_mut() else {
            return Err(VioError::InertialDisabled);
        };
        if aligned {
            return Err(VioError::InertialSamplesAligned);
        }
        Ok(inert.buffer.push(samples, dropped_cumulative))
    }

    /// Inertial counters, or `None` when the inertial path is off.
    ///
    /// The refusal counters are the externally visible number: a refused interval leaves the
    /// keyframe pair visual-only and no other trace anywhere, so a rising refusal rate is the
    /// only symptom of an IMU stream that has stopped covering its intervals.
    pub fn inertial_stats(&self) -> Option<InertialStats> {
        self.inertial.as_ref().map(|i| {
            let factors = self.map.imu_factors();
            InertialStats {
                buffer: i.buffer.stats(),
                initialized: self.state.imu_initialized,
                // Walks every factor, which is fine for a stop line and would not be for a
                // per-frame call: `Map::imu_factors` is push-only (see `crate::imu`).
                retained_factors: factors.len(),
                retained_samples: factors.iter().map(|f| f.raw_samples.len()).sum(),
                ..i.stats
            }
        })
    }

    /// Drops the map and all state, returning to bootstrap at identity.
    ///
    /// Note this is *stronger* than what a tracking loss does by default: there a loss
    /// calls `SystemState::reset` and re-bootstraps into the **same** map at the preserved
    /// pose, so keyframes and old map points accumulate — unless
    /// [`TrackerConfig::reset_map_on_loss`] routes losses here too. This is the only way to
    /// actually bound memory.
    pub fn reset(&mut self) {
        self.reset_inner(false);
    }

    /// Reset for a dropped MAP — the size bound or a tracking loss — keeping the IMU
    /// calibration, because nothing about the sensor changed. [`Tracker::reset`] is the only
    /// path that starts the bias over, and it means "different robot or re-calibration", not
    /// "the map went away". See [`InertialState::reset`].
    fn reset_keeping_calibration(&mut self) {
        self.reset_inner(true);
    }

    fn reset_inner(&mut self, keep_calibration: bool) {
        self.map = Map::new();
        self.state = SystemState::new();
        self.last_keyframe_timestamp_sec = None;
        self.last_keyframe_stamp_ns = None;
        // The buffered samples span the OLD map's last keyframe, and `start_kf_idx` names a
        // keyframe that no longer exists — `ImuInitializer` would silently skip every factor
        // whose endpoints it cannot find rather than report an empty problem.
        if let Some(inert) = self.inertial.as_mut() {
            inert.reset(keep_calibration);
        }
        self.stats = TrackerStats::default();
        // Cleared, so the first loss after a reset is described rather than throttled against a
        // loss of the map it replaced — the same rule `last_window_log` follows. The suppressed
        // count goes with it: it counts losses of a map that no longer exists.
        self.last_loss_log = None;
        self.losses_suppressed = 0;
        // The map this armed against is gone. Left set, a requested reset landing while the
        // bound was armed would drop the fresh map on the next frame and bump the generation
        // a SECOND time, for no reason a consumer could see.
        self.pending_map_reset = false;
        self.last_frame_stamp_ns = None;
        self.world_generation += 1;
        // `next_frame_idx` and `epoch_ns` deliberately keep counting: frame
        // indices must stay unique for `Map::get_keyframe`, and restarting the
        // clock would corrupt the loss grace period.
    }

    // ── Bootstrap ────────────────────────────────────────────────────────────

    /// Single-frame metric initialization from stereo depth (as in Campos et al.,
    /// ORB-SLAM3, IEEE T-RO 2021).
    ///
    /// The map is built at the *current* odometry pose, not at identity, so a
    /// re-bootstrap after a tracking loss continues the same trajectory. The
    /// returned pose is the input pose unchanged: bootstrap never moves the
    /// camera.
    fn bootstrap_stereo(
        &mut self,
        mut curr_frame: Frame,
        timestamp_sec: f64,
        stamp_ns: u64,
    ) -> TrackingStatus {
        curr_frame.pose_world_to_cam = self.state.pose_world_to_cam;

        let cam_points = unproject_stereo(&curr_frame, &self.camera);
        if cam_points.len() < self.config.min_bootstrap_stereo_points {
            debug!(
                "bootstrap_stereo: too few stereo points, frame dropped: frame={} stereo_points={} needed={}",
                curr_frame.idx,
                cam_points.len(),
                self.config.min_bootstrap_stereo_points
            );
            return TrackingStatus::Skipped;
        }

        // Captured before `from_frame` consumes the frame.
        let pose_inv = curr_frame.pose_world_to_cam.inverse();
        let mut keyframe = Keyframe::from_frame(curr_frame);
        let curr_idx = keyframe.frame.idx;

        let mut points: Vec<TriangulatedPoint> = Vec::with_capacity(cam_points.len());
        for (desc_idx, p_cam) in &cam_points {
            let p_world = pose_inv.transform_point(p_cam);
            let descriptor = keyframe.frame.features.descriptors[*desc_idx];
            let color = keypoint_color(&keyframe.frame, *desc_idx);
            points.push((p_world, descriptor, color, *desc_idx, *desc_idx));
        }

        // `prev_kf = None`: only `curr_kf` gets associations and observations.
        // The return value is `points.len()` unconditionally — an attempt count,
        // never a success signal.
        let first_new_mp_idx = self.map.num_map_points();
        let attempted = self
            .map
            .add_triangulated_points(None, &mut keyframe, &points);
        self.map.upsert_keyframe(keyframe);
        // See `refresh_new_map_point_geometry`: the geometry pass inside
        // `add_triangulated_points` ran while `keyframe` was still local, so it
        // was a no-op and every point above still has `max_distance = 0`.
        self.refresh_new_map_point_geometry(first_new_mp_idx);

        info!(
            "bootstrap_stereo: metric map created from one stereo frame: frame={} points={}",
            curr_idx, attempted
        );

        self.state.current_keyframe_idx = Some(curr_idx);
        self.state.last_keyframe_idx = Some(curr_idx);
        self.state.velocity = None;
        self.state.mode = SystemMode::Tracking;
        self.last_keyframe_timestamp_sec = Some(timestamp_sec);
        self.last_keyframe_stamp_ns = Some(stamp_ns);
        self.arm_inertial_window(curr_idx, timestamp_sec, stamp_ns);

        TrackingStatus::KeyframeAccepted
    }

    /// Anchors the inertial initialization window on the keyframe the map was just built from.
    ///
    /// Every gate in `ImuInitializer` is measured from `start_kf_idx` forward, and a stale one
    /// would keep counting keyframes and IMU seconds from a map that no longer exists — the
    /// readiness test would pass while `try_initialize` silently skipped every factor whose
    /// endpoints it could not find. No-op with the inertial path off.
    fn arm_inertial_window(&mut self, kf_idx: usize, timestamp_sec: f64, stamp_ns: u64) {
        let Some(inert) = self.inertial.as_mut() else {
            return;
        };
        inert.start_kf_idx = Some(kf_idx);
        inert.window_start_sec = Some(timestamp_sec);
        inert.last_init_attempt_sec = None;
        // A new window is a new question — for the dry run, and for the line that describes it,
        // which would otherwise stay throttled against the window this one replaces. A loss with
        // `reset_map_on_loss` off (the default) re-arms here without any `InertialState::reset`.
        inert.dry_run_accepted = false;
        inert.last_window_log = None;
        inert.stats.first_refinement_done = false;
        inert.stats.second_refinement_done = false;
        // Samples older than the anchor can never bound an interval again.
        inert.buffer.prune_before(stamp_ns);
    }

    /// Attempts to relocalize a lost frame against the EXISTING map: 2D-3D descriptor
    /// matching against candidate keyframes' map-point associations, then PnP with the
    /// candidate keyframe's own pose as the prior. Read-only; on success the caller
    /// re-enters normal tracking from the returned pose — same world, same map-point
    /// indices, no generation bump.
    ///
    /// Cost: ~(associated kf descriptors x frame descriptors) Hamming ops per candidate,
    /// a few ms each — and it runs only on lost frames, which are otherwise wasted.
    fn try_relocalize(&self, frame: &Frame) -> Option<(Pose3d, usize, usize)> {
        let cfg = self.config.relocalization.as_ref()?;
        let kfs = self.map.keyframes();
        let descs = &frame.features.descriptors;
        if kfs.is_empty() || descs.is_empty() {
            return None;
        }
        // Candidates: nearest camera centres to the last known pose, best first.
        let here = self.state.pose_world_to_cam.inverse().translation;
        let mut order: Vec<(usize, f64)> = kfs
            .iter()
            .enumerate()
            .map(|(i, kf)| {
                let c = kf.frame.pose_world_to_cam.inverse().translation;
                let d = c - here;
                (i, d.dot(d))
            })
            .collect();
        order.sort_by(|a, b| a.1.total_cmp(&b.1));
        order.truncate(cfg.max_candidates);

        let mps = self.map.map_points();
        for (kf_i, _) in order {
            let kf = &kfs[kf_i];
            // Pair each of the keyframe's map-point descriptors with its best frame
            // keypoint, then keep one map point per keypoint (best Hamming wins).
            let mut pairs: Vec<(usize, usize, u32)> = Vec::new();
            for (d_idx, slot) in kf.map_point_by_desc_idx.iter().enumerate() {
                let Some(mp_idx) = slot else { continue };
                let Some(mp) = mps.get(*mp_idx) else { continue };
                if mp.culled {
                    continue;
                }
                let kd = &kf.frame.features.descriptors[d_idx];
                let (mut best, mut best_kp) = (u32::MAX, usize::MAX);
                for (kp_idx, fd) in descs.iter().enumerate() {
                    let d = hamming_distance(kd, fd);
                    if d < best {
                        best = d;
                        best_kp = kp_idx;
                    }
                }
                if best <= cfg.max_hamming {
                    pairs.push((best_kp, *mp_idx, best));
                }
            }
            pairs.sort_by_key(|&(kp, _, d)| (kp, d));
            pairs.dedup_by_key(|&mut (kp, _, _)| kp);
            if pairs.len() < cfg.min_matches {
                continue;
            }

            let mut points_world = Vec::with_capacity(pairs.len());
            let mut points_image = Vec::with_capacity(pairs.len());
            for &(kp, mp_idx, _) in &pairs {
                // The same coordinates (and distortion fallback) the tracking-path PnP uses.
                let Some(px) = frame.undistorted_xy(kp, &self.camera) else {
                    continue;
                };
                points_world.push(mps[mp_idx].position);
                points_image.push(kornia_algebra::Vec2F32::new(px[0], px[1]));
            }
            let pnp_config = PnpConfig {
                prior_reproj_threshold_px: cfg.prior_reproj_threshold_px,
                ..self.config.map_projection.pnp.clone()
            };
            if let Some((pose, inliers)) = pnp::solve_pnp(
                &points_world,
                &points_image,
                &self.camera,
                &kf.frame.pose_world_to_cam,
                &pnp_config,
            ) && inliers >= cfg.min_inliers
            {
                return Some((pose, inliers, kf.frame.idx));
            }
        }
        None
    }

    // ── Tracking ─────────────────────────────────────────────────────────────

    fn tracking_step(&mut self, frame: Frame, timestamp_sec: f64, stamp_ns: u64) -> TrackingStatus {
        let image_size = frame.image_size;
        // Bound before any path that can move `frame` into `bootstrap_stereo`.
        let frame_idx = frame.idx;
        let pose_before = self.state.pose_world_to_cam;

        // Constant-velocity motion model. `velocity` is stored below as
        // `Pose3d::between(&pose_before, &estimate.pose)` and applied here as
        // `velocity.compose(&pose_before)`; the two halves are a matched pair —
        // do not re-derive one from an assumed convention.
        let mut candidate_pose = self
            .state
            .velocity
            .map(|v| v.compose(&pose_before))
            .unwrap_or(pose_before);

        // GYRO PRIOR on the rotation only. The constant-velocity model above predicts the next
        // ROTATION from the last one, which is wrong the moment angular rate changes — exactly
        // when the platform turns, and exactly when tracking was being lost. The gyro
        // measured that turn at 200 Hz. Translation stays on constant velocity: predicting it
        // needs double-integrated acceleration plus gravity, which is a different error budget.
        let gyro_delta = self
            .last_frame_stamp_ns
            .and_then(|t0| self.gyro_camera_delta(t0, stamp_ns));
        if let Some(d) = gyro_delta {
            candidate_pose =
                rotate_about_centre(candidate_pose, d.transpose() * pose_before.rotation);
        }

        // Widen the search in proportion to how long we have been failing: a
        // pose compounded over seconds of loss carries far more uncertainty
        // than a single-frame prediction.
        let currently_lost_for = self
            .state
            .lost_since_sec
            .map_or(0.0, |t0| timestamp_sec - t0);
        let search_scale = self.estimator.config().search_scale_for(currently_lost_for);

        let t_est = std::time::Instant::now();
        let result = self.estimator.estimate_pose(
            &frame,
            &candidate_pose,
            &pose_before,
            &self.map,
            &self.camera,
            self.state.current_keyframe_idx,
            search_scale,
            // No KLT/flow front-end here: no pre-seeded correspondences.
            None,
        );
        let est_ms = t_est.elapsed().as_secs_f64() * 1e3;
        let t_rest = std::time::Instant::now();

        let (mut status, matches, tracked_inliers) = match result {
            Ok(estimate) => {
                // GYRO vs VISION, over the same interval. The transposes in
                // `gyro_camera_delta` cannot be checked by inspection — a wrong one yields a
                // rotation of the right MAGNITUDE pointing the wrong way, which still tracks
                // in a slow scene. This prints the disagreement: near 0 deg means the
                // convention holds, and roughly twice the frame's own rotation means it does
                // not.
                if let Some(d) = gyro_delta {
                    // `between(a, b)` is the T with `T.compose(a) == b`, so its rotation is
                    // `R_cw1 * R_cw0^T` = `d^T`. The residual is therefore `d * visual`, which
                    // is identity when they agree. Comparing `d^T * visual` instead computes
                    // `d^T d^T` and reads as TWICE the true rotation — small when parked and
                    // ~9 deg while turning.
                    let visual = Pose3d::between(&pose_before, &estimate.pose).rotation;
                    let resid = d * visual;
                    let cos = ((resid.x_axis.x + resid.y_axis.y + resid.z_axis.z) - 1.0) * 0.5;
                    let gyro_vs_visual_deg = cos.clamp(-1.0, 1.0).acos().to_degrees();
                    debug!(
                        "gyro prior residual: gyro_vs_visual_deg={}",
                        gyro_vs_visual_deg
                    );
                }
                self.state.pose_world_to_cam = estimate.pose;
                self.state.velocity = Some(Pose3d::between(&pose_before, &estimate.pose));
                // `Estimate::matches` is the FULL correspondence set, not the
                // inlier set — `solve_pnp` returns `(Pose3d, usize)` and throws
                // the mask away. Everything downstream of here writes those
                // correspondences into the map permanently (keyframe
                // associations, `register_observation`, and the `n_found`
                // counter that `cull()` keys off), and `run_local_ba` uses
                // `BaParams::default()`, whose kernel is `Identity` with
                // `robust_scale_sq = INFINITY` — plain L2, no Huber. So a
                // correspondence that PnP itself rejected would enter BA at
                // full weight and, because its `n_found` rises in lockstep with
                // `n_visible`, would keep a perfect found-ratio and never be
                // culled. Re-apply PnP's own final gate here, before any of
                // those writes, so a rejected correspondence never reaches
                // the map's bookkeeping.
                let matches = self.reject_pnp_outliers(&frame, &estimate.pose, &estimate.matches);
                debug!(
                    "track: ok frame={} matches={} kept={} inliers={}",
                    frame.idx,
                    estimate.matches.len(),
                    matches.len(),
                    estimate.inliers
                );
                (TrackingStatus::Tracked, matches, estimate.inliers)
            }
            Err(reason) => {
                // Carry the *predicted* pose forward instead of freezing at
                // `pose_before`: freezing desyncs position from velocity and
                // compounds one bad frame into a full loss. `velocity` is left
                // untouched, so this is a constant-velocity coast.
                self.state.pose_world_to_cam = candidate_pose;
                debug!(
                    "track: rejected frame={} reason={}",
                    frame.idx,
                    reject_reason_str(reason)
                );
                (TrackingStatus::Skipped, Vec::new(), 0)
            }
        };
        self.stats.inliers = tracked_inliers;

        if status == TrackingStatus::Tracked {
            // Visibility bookkeeping over the local map only; a full-map scan
            // here would grow with trajectory length.
            let current_kf = self
                .state
                .current_keyframe_idx
                .and_then(|ki| self.map.get_keyframe(ki));
            // `default()` is behaviour-preserving: 10/10/15/4 are the exact literals the
            // `Map` method this replaced hardcoded, including its `< 4` fallback.
            let local_indices = select_local_map_points(
                &self.map,
                &matches,
                current_kf,
                LocalMapSelectionConfig::default(),
            );
            // Evaluated at `candidate_pose`, NOT at the estimated pose: visibility
            // is counted before refinement, matching the ORB-SLAM3 reference
            // implementation. It biases `n_visible`, hence `cull()`'s found_ratio.
            let visible = self.map.map_points_in_frustum(
                &local_indices,
                &self.camera,
                &candidate_pose,
                image_size,
            );
            self.map.update_observation_counts(&visible, &matches);

            let bookkeep_ms = t_rest.elapsed().as_secs_f64() * 1e3;
            let t_kf = std::time::Instant::now();
            let inserted = self.try_insert_keyframe(
                &frame,
                timestamp_sec,
                stamp_ns,
                tracked_inliers,
                &matches,
            );
            if inserted {
                status = TrackingStatus::KeyframeAccepted;
            }
            let kf_ms = t_kf.elapsed().as_secs_f64() * 1e3;
            // Splits `track_ms` from the `process_stereo stage split` line one level further.
            // `estimate_pose` is projection matching + PnP; `bookkeep` is the local-map
            // visibility accounting that feeds culling; `kf` is insertion (grow/fuse/BA).
            debug!(
                "tracking_step split: est_ms={} bookkeep_ms={} kf_ms={}",
                est_ms, bookkeep_ms, kf_ms
            );

            // MAP GROWTH, at info so it survives a production log filter that drops debug.
            // Nothing culls keyframes, so insertion cost rises with `kfs` until the graph stalls
            // silently: measured on a Jetson Orin with an OAK-D at 640x400, 757 MB RSS and zero
            // poses after 2.5 h of clean tracking. This is the curve that sets a keyframe bound.
            //
            // SAMPLED ONLY, with no "slow insertion" escape: past the knee EVERY insertion is
            // slow, so such an escape degenerates to a line per keyframe on exactly the
            // unbounded run it is meant to diagnose.
            if inserted && self.map.keyframes().len().is_multiple_of(KF_GROWTH_SAMPLE) {
                info!(
                    "map growth: kfs={} map_points={} kf_ms={}",
                    self.map.keyframes().len(),
                    self.map.num_map_points(),
                    kf_ms
                );
            }

            self.arm_map_bound(inserted);
        }

        if status == TrackingStatus::Skipped {
            let lost_since = *self.state.lost_since_sec.get_or_insert(timestamp_sec);
            let recently_lost_for = timestamp_sec - lost_since;
            // `imu_confident` is always false: the inertial path does not feed loss recovery.
            let grace_period_sec = self.config.tracking_loss_recovery.grace_period_sec(false);
            let map_established = self.map.keyframes().len()
                > self.config.tracking_loss_recovery.min_keyframes_for_grace;

            // Relocalization first: a PnP pose against the EXISTING map is a measurement —
            // strictly better than the widening projection search (which needs the coast
            // prior to be roughly right) and than any re-bootstrap (which moves the world).
            if map_established && let Some((pose, inliers, kf_idx)) = self.try_relocalize(&frame) {
                warning!(
                    "relocalized against the existing map: frame={} inliers={} kf={} lost_for_sec={}",
                    frame_idx,
                    inliers,
                    kf_idx,
                    recently_lost_for
                );
                self.state.pose_world_to_cam = pose;
                // The jump invalidates the constant-velocity model; rebuild from the
                // next tracked pair.
                self.state.velocity = None;
                self.state.current_keyframe_idx = Some(kf_idx);
                self.state.lost_since_sec = None;
                self.state.last_frame_timestamp_sec = timestamp_sec;
                // Measured (PnP inliers), so it may leave as Tracked — unlike the
                // re-bootstrap coast below, which must not.
                return TrackingStatus::Tracked;
            }
            // While relocalization is enabled, give it its patience window before
            // surrendering the map — the grace period alone was tuned for blind coasting.
            let give_up_sec = self
                .config
                .relocalization
                .as_ref()
                .map_or(grace_period_sec, |r| {
                    grace_period_sec.max(r.patience.get::<second>())
                });

            if !map_established || recently_lost_for >= give_up_sec {
                if should_log_throttled(
                    self.last_loss_log,
                    timestamp_sec,
                    map_established,
                    LOSS_LOG_RETRY_SEC,
                ) {
                    // `stats()` and not the `self.stats` fields: it is the bundle, and its doc
                    // asks to be called at the log cadence rather than per frame — which is
                    // exactly where the throttle has put this. It also folds in `map_points`,
                    // the other whole-map scan, which a hand-picked field list left out.
                    let s = self.stats();
                    warning!(
                        "tracking lost: resetting to bootstrap: frame={} lost_for_sec={} \
                         map_established={} map_cleared={} keypoints={} stereo_matches={} \
                         inliers={} consecutive_failures={} keyframes={} map_points={} \
                         suppressed={}",
                        frame.idx,
                        recently_lost_for,
                        map_established,
                        self.config.reset_map_on_loss,
                        s.keypoints,
                        s.stereo_matches,
                        s.inliers,
                        s.consecutive_failures,
                        s.keyframes,
                        s.map_points,
                        self.losses_suppressed
                    );
                    self.last_loss_log = Some((timestamp_sec, map_established));
                    self.losses_suppressed = 0;
                } else {
                    self.losses_suppressed += 1;
                }
                if self.config.reset_map_on_loss {
                    // Full reset (map dropped): bounds the per-frame cost the append-to-map
                    // arm below lets grow without limit; see the config field doc for the
                    // measurements. Bumps world_generation.
                    //
                    // KEEPS THE IMU BIAS, for the same reason the `max_keyframes` drop does: a
                    // map thrown away says nothing about the sensor. This is the reset a live
                    // graph takes most often, so it is the one that most needs the bias.
                    self.reset_keeping_calibration();
                } else {
                    // `SystemState::reset` clears mode/keyframe pointers but keeps the map
                    // and `pose_world_to_cam`: the re-bootstrap APPENDS at the preserved
                    // pose — a coast, not a measurement, hence the generation bump — and
                    // with `current_keyframe_idx == None` the next local-map build returns
                    // every non-culled point.
                    self.state.reset();
                    self.world_generation += 1;
                }
                let recovered = self.bootstrap_stereo(frame, timestamp_sec, stamp_ns);
                // The pose the new map was just anchored at is `candidate_pose`
                // — pure constant-velocity extrapolation over every frame since
                // tracking failed, which at 30 fps and a 0.5 s grace is up to
                // ~15 compoundings with no correction. It is a coast, not a
                // measurement, so it MUST NOT leave here as `KeyframeAccepted`:
                // `process_frame` maps that to `TrackStatus::Tracked`, which the
                // task publishes and which also resets
                // `consecutive_failures` to 0 — a fabricated pose, published as
                // measured, with every health field reading green. Report
                // `Skipped` regardless of whether the map was rebuilt; the next
                // frame that actually tracks against the new map is the first
                // measured pose again.
                if recovered == TrackingStatus::KeyframeAccepted {
                    warning!(
                        "re-bootstrapped at the extrapolated pose on frame {}; reporting Skipped \
                         because that pose is dead-reckoned, not measured",
                        frame_idx
                    );
                }
                // Early return, so `last_frame_timestamp_sec` is not updated on
                // this frame — harmless for stereo (only the IMU window reads
                // it), but do not "fix" it silently if IMU is ever added.
                return TrackingStatus::Skipped;
            }
        } else {
            self.state.lost_since_sec = None;
        }
        self.state.last_frame_timestamp_sec = timestamp_sec;
        status
    }

    /// Keeps only the correspondences that reproject within PnP's own final
    /// inlier threshold at the estimated pose.
    ///
    /// The threshold is `PnpConfig::final_reproj_threshold_px` **unscaled**:
    /// `search_scale` widens the projection search and the PnP *prior* gate
    /// during a loss, but upstream leaves the final gate alone precisely
    /// because it is the one that decides what is true, so widening it here
    /// would let the map absorb garbage exactly when tracking is weakest.
    ///
    /// A correspondence whose map point or undistorted keypoint cannot be read
    /// is dropped: it could not have contributed to the solve either.
    fn reject_pnp_outliers(
        &self,
        frame: &Frame,
        pose: &Pose3d,
        matches: &[(usize, usize)],
    ) -> Vec<(usize, usize)> {
        let th = self.estimator.config().pnp.final_reproj_threshold_px;
        let th_sq = th * th;
        matches
            .iter()
            .copied()
            .filter(|&(mp_idx, curr_idx)| {
                let Some(mp) = self.map.map_points().get(mp_idx) else {
                    return false;
                };
                if mp.culled {
                    return false;
                }
                let Some(kp) = frame.undistorted_xy(curr_idx, &self.camera) else {
                    return false;
                };
                self.camera
                    .reprojection_error_sq_world(pose, &mp.position, kp[0] as f64, kp[1] as f64)
                    .is_some_and(|err_sq| err_sq <= th_sq)
            })
            .collect()
    }

    /// Canonical order: associate -> densify close stereo -> grow(xN) -> upsert
    /// -> fuse -> local BA -> pose sync-back -> cull.
    fn try_insert_keyframe(
        &mut self,
        frame: &Frame,
        timestamp_sec: f64,
        stamp_ns: u64,
        tracked_inliers: usize,
        matches: &[(usize, usize)],
    ) -> bool {
        // Bound BEFORE the block below overwrites both: the IMU edge spans from the PREVIOUS
        // keyframe, and both halves of that interval are about to be replaced in place.
        let prev_edge = self
            .state
            .last_keyframe_idx
            .zip(self.last_keyframe_stamp_ns);

        let n_ref_map_points = self
            .state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki))
            .map(|kf| kf.num_associated_points())
            .unwrap_or(0);

        if !self.config.keyframe_policy.should_insert(
            frame.idx,
            self.state.last_keyframe_idx,
            tracked_inliers,
            n_ref_map_points,
        ) {
            return false;
        }

        // The reference keyframe must exist before we can triangulate against it.
        if self
            .state
            .current_keyframe_idx
            .and_then(|ki| self.map.get_keyframe(ki))
            .is_none()
        {
            return false;
        }

        let mut curr_kf = Keyframe::from_frame(Frame {
            idx: frame.idx,
            features: frame.features.clone(),
            pose_world_to_cam: self.state.pose_world_to_cam,
            image_size: frame.image_size,
            keypoint_colors: frame.keypoint_colors.clone(),
            u_right: frame.u_right.clone(),
            depth: frame.depth.clone(),
            // Carry the undistortion cache over — dropping it would silently
            // re-undistort per consumer for the rest of this keyframe's life.
            keypoints_undist: frame.keypoints_undist.clone(),
        });

        for &(mp_idx, curr_idx) in matches {
            curr_kf.associate_map_point(curr_idx, mp_idx);
            self.map.register_observation(mp_idx, &curr_kf, curr_idx);
        }

        // Every map point created below is created while `curr_kf` is still a
        // local value, so `update_map_point_geometry`'s `get_keyframe(ref_kf_idx)`
        // lookup inside `add_triangulated_points` misses and returns early —
        // see the `refresh_new_map_point_geometry` call after `upsert_keyframe`.
        let first_new_mp_idx = self.map.num_map_points();

        // Stereo densification. Must run after association (so the
        // "already tracked" filter is meaningful) and before grow (so grow's
        // "unassociated only" filter sees these as taken).
        if let Some(close_depth_threshold) =
            self.config.stereo_close_depth.map(|d| d.get::<meter>())
            && curr_kf.frame.is_stereo()
        {
            let n_close = self.add_close_stereo_points(&mut curr_kf, close_depth_threshold);
            debug!(
                "keyframe: stereo densification frame={} close_points={}",
                frame.idx, n_close
            );
        }

        // Recency as a covisibility proxy, newest first. `curr_kf` is not in the
        // map yet, so it cannot appear here.
        let neighbor_kf_indices: Vec<usize> = self
            .map
            .keyframes()
            .iter()
            .rev()
            .take(self.config.max_covisible_keyframes)
            .map(|kf| kf.frame.idx)
            .collect();

        let match_config = self.config.two_view_init.match_config;
        let triangulation_config = self.config.two_view_init.triangulation_config.clone();

        let mut total_grown = 0usize;
        for &nb_kf_idx in &neighbor_kf_indices {
            total_grown += self.grow_map_points_from_keyframe_pair(
                nb_kf_idx,
                &mut curr_kf,
                match_config,
                &triangulation_config,
            );
        }

        // The keyframe enters the map only here.
        self.map.upsert_keyframe(curr_kf);
        self.last_keyframe_timestamp_sec = Some(timestamp_sec);
        self.last_keyframe_stamp_ns = Some(stamp_ns);
        self.state.current_keyframe_idx = Some(frame.idx);
        self.state.last_keyframe_idx = Some(frame.idx);

        // Inertial edge + initialization ladder. Placed here, immediately after the upsert and
        // before fuse, to match `SlamPipeline`'s factor ordering: both endpoints of the edge
        // are now in the map, which is the invariant `run_local_inertial_ba` and
        // `ImuInitializer` both silently `continue` past when it does not hold. `None` — the
        // configuration this repo ships — means every line below behaves exactly as before.
        let inertial_ba = self.on_keyframe_inertial(prev_edge, frame.idx, stamp_ns);

        // Now that the reference keyframe is reachable, give the new points the
        // scale geometry they could not get at creation time.
        self.refresh_new_map_point_geometry(first_new_mp_idx);

        // Fuse before local BA, deliberately: BA then sees the extra
        // reprojection constraints.
        let n_fused = self.fuse_into_neighbors(frame.idx, &neighbor_kf_indices);

        // Local BA only once the gauge is anchored. `run_local_ba` frees the
        // last `BA_ACTIVE_KEYFRAMES` poses and fixes the rest, so at 2 or 3
        // keyframes it fixes NOTHING — `bundle_adjust_schur` only requires
        // `n_free_poses > 0`, so the whole map, keyframe 0 included, is free to
        // translate and rotate, held near the initialization by LM damping
        // alone. On noisy input that rigidly shifts the world frame away from
        // the bootstrap frame, and keyframe 0 is then fixed wherever it landed:
        // every later pose inherits the offset. A Sim(3)-aligned ATE cannot see
        // it, and neither can the noise-free synthetic test (zero residual =>
        // zero gradient) — but the unaligned tape-measure test is measuring
        // exactly that offset. Skipping BA for the first two insertions costs
        // very little; upstream's own bootstrap BA (`run_initial_ba`) pins the
        // older keyframe for the same reason.
        let ran_ba =
            self.config.enable_local_ba && self.map.keyframes().len() > BA_ACTIVE_KEYFRAMES;
        if ran_ba {
            match inertial_ba {
                // 15 DOF per keyframe (pose, velocity, bias) instead of 6. Mirrors
                // kornia-slam's `local_mapping::solve_snapshot`, which branches on exactly this
                // flag.
                // Called on the LIVE map rather than through `Map::local_ba_snapshot()`:
                // that path deep-clones the map INCLUDING every factor's retained raw
                // samples once per keyframe (~53 MB and ~6k allocations after an hour) and
                // merges factors back in O(F^2). It exists to move the solve off-thread,
                // and this BA is already synchronous.
                Some((imu_t_bc, gravity_world)) => {
                    self.map
                        .run_local_inertial_ba(&self.camera, Some(imu_t_bc), gravity_world);
                }
                None => self.map.run_local_ba(&self.camera),
            }
        }

        // The keyframe just upserted is the newest: `Frame::idx` only grows and `upsert_keyframe`
        // appends a new index. O(1) here, where `Map::get_keyframe(frame.idx)` is a linear scan
        // over every keyframe since the last reset.
        if let Some(newest_kf) = self.map.keyframes().last() {
            debug_assert_eq!(newest_kf.frame.idx, frame.idx);
            // Pose sync-back: BA may have moved the newest keyframe.
            if ran_ba {
                self.state.pose_world_to_cam = newest_kf.frame.pose_world_to_cam;
            }
            // After BA, not before: the inertial solve writes velocity and bias onto the last
            // three keyframes, and the NEXT edge must be linearized at that refreshed bias.
            // Without this the next edge is linearized at the bootstrap bias forever, which
            // drives `Map`'s 0.02 repropagation threshold to fire on every factor of every solve
            // — a purely numerical cost that also makes the first-order bias correction the
            // dominant residual, which the optimizer cannot tell apart from signal. Runs on the
            // early (pre-BA) keyframes too, where it is a copy of what the edge already used.
            if let Some(inert) = self.inertial.as_mut() {
                inert.bias = newest_kf.imu_bias;
                self.state.velocity_world = newest_kf.velocity_world;
            }
        }

        self.map.cull();

        debug!(
            "keyframe inserted: frame={} grown={} fused={} neighbors={} map_points={} keyframes={}",
            frame.idx,
            total_grown,
            n_fused,
            neighbor_kf_indices.len(),
            self.map.num_active_map_points(),
            self.map.keyframes().len()
        );
        true
    }

    // ── Inertial ─────────────────────────────────────────────────────────────

    /// Everything the inertial path does on a keyframe insertion, in the order
    /// `SlamPipeline` does it: build the edge, prune, then run the initialization ladder.
    ///
    /// Returns `Some((imu_t_bc, gravity_world))` when local BA should take the inertial path,
    /// and `None` — including for every call with the inertial path off — when it should not.
    ///
    /// `enable_inertial_ba` gates BOTH ways the inertial path can move a published pose: the
    /// 15-DOF local BA selected by the return value here, and the initialization commit
    /// (`apply_initialization` scales and rotates the WHOLE map and bumps the world
    /// generation). With it off, the ladder still runs
    /// `try_initialize` and counts the result, but never applies it: see
    /// [`Self::accept_inertial_init`].
    ///
    /// The state is `take`n for the duration because the initializer wants `&Map` and
    /// `&mut SystemState` at the same time, which it cannot have while a field of `self` is
    /// borrowed. The move is a handful of scalars plus the ring's `VecDeque` handle; nothing is
    /// copied.
    fn on_keyframe_inertial(
        &mut self,
        prev_edge: Option<(usize, u64)>,
        curr_kf_idx: usize,
        stamp_ns: u64,
    ) -> Option<(Pose3d, Vec3F64)> {
        let mut inert = self.inertial.take()?;
        // `epoch_ns` is set by `process_frame` before this can run; the fallback only keeps the
        // arithmetic sane if that ever stops being true.
        let epoch = self.epoch_ns.unwrap_or(stamp_ns);

        if let Some((prev_kf_idx, prev_stamp_ns)) = prev_edge {
            self.add_imu_edge(
                &mut inert,
                prev_kf_idx,
                prev_stamp_ns,
                curr_kf_idx,
                stamp_ns,
                epoch,
            );
        }
        // Unconditional — refusal included. `t0` for the next interval is exactly this stamp,
        // so nothing earlier can ever be asked for again, and this (not the capacity bound) is
        // what holds the buffer at one keyframe interval in steady state. Without it, a run of
        // refusals would grow the ring until the capacity bound started evicting, at which
        // point the refusals become self-sustaining.
        inert.buffer.prune_before(stamp_ns);

        self.run_inertial_init(&mut inert, ts_sec(stamp_ns, epoch));

        let handles = (self.state.imu_initialized && inert.cfg.enable_inertial_ba)
            .then_some((inert.cfg.imu_t_bc, inert.gravity_world));
        self.inertial = Some(inert);
        handles
    }

    /// Preintegrates `[prev_stamp_ns, stamp_ns]` and adds it to the map, or refuses and says why.
    ///
    /// A refusal costs one visual-only keyframe pair. Every refusal here is a case where
    /// `PreintegratedImu::from_measurements` would have returned a well-formed delta anyway:
    /// see the gates in [`crate::imu`]. That asymmetry — a missing factor is recoverable, a
    /// wrong one poisons a bias shared across the whole window — is why the gates are strict.
    fn add_imu_edge(
        &mut self,
        inert: &mut InertialState,
        prev_kf_idx: usize,
        prev_stamp_ns: u64,
        curr_kf_idx: usize,
        stamp_ns: u64,
        epoch: u64,
    ) {
        // Converted once, straight out of the ring, into the one owned copy `add_imu_factor`
        // keeps. Seconds relative to the tracker's frame epoch — the SAME conversion
        // `process_frame` applies to frame stamps. `from_measurements` filters the samples by
        // `[t0, t1]`, so a second epoch here would drop every one of them and return the
        // zero-delta case.
        let raw: Vec<_> = match inert
            .buffer
            .window(prev_stamp_ns, stamp_ns, &inert.cfg.gates)
        {
            Ok(samples) => samples.map(|s| s.to_measurement(epoch)).collect(),
            Err(error) => {
                match error {
                    ImuWindowError::DroppedSamples { .. } => inert.stats.refused_dropped += 1,
                    ImuWindowError::Gap { .. } => inert.stats.refused_gap += 1,
                    ImuWindowError::TooFewSamples { .. } => inert.stats.refused_too_few += 1,
                    ImuWindowError::NoSamples { .. } => inert.stats.refused_no_samples += 1,
                    ImuWindowError::NonPositiveInterval { .. } => {
                        inert.stats.refused_bad_interval += 1
                    }
                }
                let span_ms = (stamp_ns.saturating_sub(prev_stamp_ns)) as f64 * 1e-6;
                warning!(
                    "inertial: IMU edge refused; this keyframe pair stays visual-only: \
                     prev_kf={} kf={} span_ms={} error={}",
                    prev_kf_idx,
                    curr_kf_idx,
                    span_ms,
                    error.to_string()
                );
                return;
            }
        };

        let t0 = ts_sec(prev_stamp_ns, epoch);
        let t1 = ts_sec(stamp_ns, epoch);
        let preint = PreintegratedImu::from_measurements(inert.bias, inert.cfg.calib, &raw, t0, t1);

        // Negated on purpose: a NaN `dt` must be refused too, which `dt <= 0.0` would let through.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let non_positive_dt = !(preint.dt > 0.0);
        if non_positive_dt {
            // Should be unreachable behind the gates above. Checked anyway because
            // `add_imu_factor` is a bare `push` with no validation and `vi_ba_schur` answers a
            // singular covariance with a 1e6-diagonal information matrix and a printed warning
            // — a hugely-weighted zero-motion constraint, not an error.
            inert.stats.refused_zero_dt += 1;
            warning!(
                "inertial: preintegration returned dt = 0 despite passing coverage; edge dropped: \
                 prev_kf={} kf={} samples={}",
                prev_kf_idx,
                curr_kf_idx,
                raw.len()
            );
            return;
        }

        // Both indices are `Frame::idx`, NOT positions in `Map::keyframes` — `ImuFactor` is
        // keyed by frame index everywhere it is read. An index naming no keyframe is not an
        // error anywhere downstream; the factor simply vanishes from the problem.
        //
        // `raw` is handed over as well, and kept for the life of the factor: it is what
        // `Map`'s repropagation path re-integrates once the bias has drifted past 0.02 from
        // this edge's linearization point. Passing an empty Vec compiles and leaves the edge
        // permanently stuck at a stale linearization.
        self.map
            .add_imu_factor(prev_kf_idx, curr_kf_idx, preint, raw, t0, t1);
        inert.stats.factors_added += 1;
    }

    /// The staged visual-inertial initialization of Campos et al., ORB-SLAM3, IEEE T-RO 2021:
    /// an initial solve once the window is ready, then a first refinement at 5 s and a second
    /// at 15 s.
    ///
    /// All three stages are needed. kornia-slam's own EuRoC V101 numbers put the gravity error
    /// at 23 degrees after the initial solve, 1.13 after the first refinement and 0.78 after the
    /// second — and gravity error is not a diagnostic here, it is the axis the whole map gets
    /// rotated onto.
    ///
    /// With `enable_inertial_ba` off, `state.imu_initialized` never becomes true (only
    /// `apply_initialization` sets it), so only the initial-solve branch is reachable; it runs
    /// once to a dry acceptance and then stops.
    fn run_inertial_init(&mut self, inert: &mut InertialState, now_sec: f64) {
        let Some(start_idx) = inert.start_kf_idx else {
            return;
        };

        if !self.state.imu_initialized {
            // Checked before the throttle and before `ready`, so that a latched dry run costs
            // nothing per keyframe — not even the keyframe/factor scan `ready` does.
            if inert.dry_run_accepted {
                return;
            }
            // Throttled because `try_initialize` runs a 200-iteration LM over a window that
            // never shrinks, so retrying on every keyframe is a cost that grows with the
            // session for a solve that just failed its own gates.
            if inert
                .last_init_attempt_sec
                .is_some_and(|t| now_sec - t < inert.cfg.init_retry.get::<second>())
            {
                return;
            }
            let ready = inert.initializer.ready(&self.map, Some(start_idx));
            // Gates the raw-sample scan inside the line, not just the print. Measured on a
            // Jetson Orin with an OAK-D at 640x400, parked: per keyframe this was 1,920 lines and
            // ~900 kB in 90 s, i.e. ~78 MB/day, each one re-scanning a window of 9,353 samples
            // and growing.
            if should_log_throttled(
                inert.last_window_log,
                now_sec,
                ready,
                inert.cfg.init_retry.get::<second>(),
            ) {
                inert.last_window_log = Some((now_sec, ready));
                self.log_init_window(inert, start_idx, ready);
            }
            if !ready {
                return;
            }
            inert.last_init_attempt_sec = Some(now_sec);
            inert.stats.init_attempts += 1;

            // prior_a = 1e5, the STEREO value: this tracker cannot bootstrap monocularly (see
            // `bootstrap_stereo`), so the 1e10 accelerometer-bias prior the mono path uses to
            // stop scale and bias trading off against each other does not apply. Scale is
            // pinned to 1 by the stereo baseline and the factor zeroes its column.
            let result = inert.initializer.try_initialize(
                &self.map,
                Some(inert.cfg.imu_t_bc),
                inert.bias,
                start_idx,
                1e2,
                1e5,
                false,
            );
            match result {
                Some(init) => {
                    self.accept_inertial_init(inert, init, start_idx, now_sec, InitStage::Initial)
                }
                None => info!(
                    "inertial: initial solve rejected (solve failed, or |accel bias| over 1 m/s^2): start_kf={}",
                    start_idx
                ),
            }
            return;
        }

        let Some(window_start) = inert.window_start_sec else {
            return;
        };
        let init_elapsed = now_sec - window_start;
        // Past 50 s the window is far too large for a single joint solve to be worth attempting;
        // this mirrors the pipeline's own ceiling.
        if init_elapsed >= 50.0 {
            return;
        }
        let (prior_g, stage) = if !inert.stats.first_refinement_done && init_elapsed > 5.0 {
            (1.0, InitStage::Refine1)
        } else if inert.stats.first_refinement_done
            && !inert.stats.second_refinement_done
            && init_elapsed > 15.0
        {
            (0.0, InitStage::Refine2)
        } else {
            return;
        };

        inert.stats.init_attempts += 1;
        // `already_initialized = true` changes the seeding, not just the priors: velocities come
        // from the keyframes instead of finite differences, and Rwg seeds at the fixed
        // gI -> (0, +G, 0) rotation rather than identity.
        let result = inert.initializer.try_initialize(
            &self.map,
            Some(inert.cfg.imu_t_bc),
            inert.bias,
            start_idx,
            prior_g,
            1e5,
            true,
        );
        match result {
            Some(init) => self.accept_inertial_init(inert, init, start_idx, now_sec, stage),
            None => info!("inertial: refinement rejected: stage={}", stage.as_str()),
        }
        // Marked done either way: the stage is scheduled once, as in the ORB-SLAM3 reference implementation, and
        // a rejected refinement that retried every keyframe would be the unthrottled solve the
        // initial-solve path is careful to avoid.
        if stage == InitStage::Refine1 {
            inert.stats.first_refinement_done = true;
        } else {
            inert.stats.second_refinement_done = true;
        }
    }

    /// What the initialization window looks like, at every `ready` evaluation and on BOTH of its
    /// answers. Observation only; nothing here gates anything.
    ///
    /// New measurement, not a restatement of the gate: no deployed gate (kornia-slam 9d3a927
    /// `imu_init.rs:93-140` — >=10 kfs, >=1 s dt, >=0.05 m of VISUAL displacement, which VO drift
    /// satisfies on a parked robot) can see whether the IMU moved, and `ready` answers a bare
    /// bool its caller drops. `info!` because a production log filter typically drops debug,
    /// and this line is the only record of why the initializer is still waiting.
    ///
    /// Throttled by the caller on [`InertialState::last_window_log`]. Each call is one line AND
    /// one pass over the window's raw samples, so the caller gates both; do not add further
    /// per-sample cost here.
    fn log_init_window(&self, inert: &InertialState, start_idx: usize, ready: bool) {
        // Scoped exactly as `ImuInitializer::ready` scopes its own — keyframes by `frame.idx`,
        // factors by `curr_kf_idx` — so the line explains THAT verdict and not a differently
        // scoped one.
        let kfs: Vec<&Keyframe> = self
            .map
            .keyframes()
            .iter()
            .filter(|kf| kf.frame.idx >= start_idx)
            .collect();
        let factors = || {
            self.map
                .imu_factors()
                .iter()
                .filter(|f| f.curr_kf_idx >= start_idx)
        };
        let imu_dt_s: f64 = factors().map(|f| f.preintegrated.dt).sum();
        let disp_m = match (kfs.first(), kfs.last()) {
            (Some(first), Some(last)) => (last.frame.pose_world_to_cam.inverse().translation
                - first.frame.pose_world_to_cam.inverse().translation)
                .length(),
            _ => 0.0,
        };
        let exc = self.init_window_excitation(start_idx);
        // Which of `ready`'s two time thresholds applied: 1 s stereo, 2 s mono.
        let stereo = kfs.first().is_some_and(|kf| kf.frame.is_stereo());
        info!(
            "inertial: init window: ready={} start_kf={} kfs={} imu_dt_s={} disp_m={} \
             rot_exc_rad={} accel_var={} imu_samples={} stereo={} init_retry_sec={}",
            ready,
            start_idx,
            kfs.len(),
            imu_dt_s,
            disp_m,
            exc.rot_rad,
            exc.accel_var,
            exc.samples,
            stereo,
            inert.cfg.init_retry.get::<second>()
        );
    }

    /// Excitation over exactly the factors `ImuInitializer::ready` counts for its summed-dt gate:
    /// `curr_kf_idx >= start_idx`. Split out of the log line so the scoping is testable — a
    /// window that swept in pre-bootstrap factors would read excited for motion that predates it.
    fn init_window_excitation(&self, start_idx: usize) -> WindowExcitation {
        window_excitation(
            self.map
                .imu_factors()
                .iter()
                .filter(|f| f.curr_kf_idx >= start_idx)
                .map(|f| f.raw_samples.as_slice()),
        )
    }

    /// The one place an accepted initialization is allowed to touch the map, and the gate on it.
    ///
    /// `enable_inertial_ba` off means the result is counted and logged but NOT applied. This is
    /// the whole content of the flag's "cannot move a published pose" promise, and it is not
    /// redundant with the local-BA selection in `on_keyframe_inertial`: `apply_initialization`
    /// calls `Map::scale_world` and `Map::rotate_world` on every keyframe and map point, then
    /// the world generation goes up, which `task.rs` publishes as `VioStatus::reset_epoch` and
    /// which every consumer answers by wiping its trail and re-deriving its anchor. Without this
    /// gate, a yaw-unverified extrinsic with the flag off would produce that wipe at the initial
    /// solve (~2-5 s in) and at both refinements (>5 s, >15 s) — three frame snaps per run, on
    /// a guess.
    ///
    /// What the dry run keeps: `init_attempts`, `init_accepted` (so `init N/M accepted,
    /// initialized=false` in the stop line reads as "the gates pass, nothing was applied"),
    /// and a `warn!` carrying the scale, gravity and bias the solve produced — the numbers a
    /// caller compares against the static gravity read to decide whether the yaw candidate
    /// is the right one. With the wrong candidate the gyro/visual inconsistency lands in the
    /// bias estimate, so a large `gyro_bias` here is the symptom to look for.
    fn accept_inertial_init(
        &mut self,
        inert: &mut InertialState,
        init: ImuInitResult,
        start_idx: usize,
        now_sec: f64,
        stage: InitStage,
    ) {
        if inert.cfg.enable_inertial_ba {
            self.apply_inertial_init(inert, init, start_idx, now_sec, stage);
            return;
        }
        inert.stats.init_accepted += 1;
        inert.dry_run_accepted = true;
        warning!(
            "inertial: initialization ACCEPTED but NOT applied (enable_inertial_ba is off); \
             the map, the world generation and the bias are untouched: stage={} scale={} \
             gravity_x={} gravity_y={} gravity_z={} gyro_bias={} accel_bias={}",
            stage.as_str(),
            init.scale,
            init.gravity_world.x,
            init.gravity_world.y,
            init.gravity_world.z,
            init.bias.gyro.length(),
            init.bias.accel.length()
        );
    }

    /// Commits an accepted initialization to the map and bumps the world generation.
    ///
    /// Only reachable through [`Self::accept_inertial_init`] with `enable_inertial_ba` on.
    fn apply_inertial_init(
        &mut self,
        inert: &mut InertialState,
        init: ImuInitResult,
        start_idx: usize,
        now_sec: f64,
        stage: InitStage,
    ) {
        let (scale, gravity, bias) = (init.scale, init.gravity_world, init.bias);
        inert.initializer.apply_initialization(
            &mut self.map,
            &mut self.state,
            &mut inert.bias,
            &mut inert.gravity_world,
            init,
            start_idx,
        );
        // `apply_initialization` calls `Map::scale_world` then `Map::rotate_world`: every
        // keyframe pose and every map point moves, so that gravity lands on world +y. Poses
        // published before this instant and after it are in DIFFERENT frames, and the only
        // thing that says so on the wire is this counter — the same contract a requested reset
        // and a loss re-bootstrap already use. Consumers that map tracker poses into another
        // frame re-derive their anchor when it changes.
        self.world_generation += 1;
        self.state.imu_init_timestamp_sec = Some(now_sec);
        inert.stats.init_accepted += 1;
        warning!(
            "inertial: initialization applied; the map has been rotated onto gravity: stage={} \
             scale={} gravity_x={} gravity_y={} gravity_z={} gyro_bias={} accel_bias={} \
             world_generation={}",
            stage.as_str(),
            scale,
            gravity.x,
            gravity.y,
            gravity.z,
            bias.gyro.length(),
            bias.accel.length(),
            self.world_generation
        );
    }

    /// Recomputes the scale geometry of every map point from `first_idx` on.
    ///
    /// `Map::add_triangulated_points` already calls `update_map_point_geometry`
    /// for each point it creates — but every one of our creation sites runs
    /// while the reference keyframe is still a *local* `Keyframe`, not yet in
    /// the map, so that call hits `let Some(ref_kf) = self.get_keyframe(..)
    /// else { return; }` and does nothing. The points are then left with
    /// `min_distance = max_distance = 0` and a zero mean viewing direction,
    /// which makes `match_by_projection` skip its whole scale-invariance block
    /// (`if mp.max_distance > 0.0`): the distance gate and the predicted-octave
    /// window are silently off for exactly the freshly created points.
    ///
    /// `run_local_ba` would repair it, but only if it gets past all four of its
    /// early returns — and not at all when `enable_local_ba` is false. So do it
    /// here, unconditionally, as soon as the keyframe is reachable.
    fn refresh_new_map_point_geometry(&mut self, first_idx: usize) {
        for mp_idx in first_idx..self.map.num_map_points() {
            self.map
                .update_map_point_geometry(mp_idx, ORB_SCALE_FACTOR, ORB_N_LEVELS);
        }
    }

    /// Back-projects `curr_kf`'s unassociated *close* (`z <= close_depth_threshold`)
    /// stereo keypoints into new metric map points. Far points are left to multi-view
    /// triangulation in the growth pass.
    fn add_close_stereo_points(
        &mut self,
        curr_kf: &mut Keyframe,
        close_depth_threshold: f64,
    ) -> usize {
        let cam_points = unproject_stereo(&curr_kf.frame, &self.camera);
        if cam_points.is_empty() {
            return 0;
        }
        let pose_inv = curr_kf.frame.pose_world_to_cam.inverse();

        let mut points: Vec<TriangulatedPoint> = Vec::new();
        for (desc_idx, p_cam) in &cam_points {
            if p_cam.z > close_depth_threshold {
                continue;
            }
            if curr_kf.map_point(*desc_idx).is_some() {
                continue;
            }
            let p_world = pose_inv.transform_point(p_cam);
            let descriptor = curr_kf.frame.features.descriptors[*desc_idx];
            let color = keypoint_color(&curr_kf.frame, *desc_idx);
            points.push((p_world, descriptor, color, *desc_idx, *desc_idx));
        }

        self.map.add_triangulated_points(None, curr_kf, &points)
    }

    /// Triangulates new map points from the unassociated features shared by
    /// `prev_kf_idx` and `curr_kf`, gated by the pose-derived epipolar geometry
    /// (the triangulation search of Campos et al., ORB-SLAM3, IEEE T-RO 2021).
    fn grow_map_points_from_keyframe_pair(
        &mut self,
        prev_kf_idx: usize,
        curr_kf: &mut Keyframe,
        match_config: OrbMatchConfig,
        triangulation_config: &TriangulationConfig,
    ) -> usize {
        // Read-only phase; the shared borrow of `self.map` ends with this block.
        let points: Vec<TriangulatedPoint> = {
            let Some(prev_kf) = self.map.get_keyframe(prev_kf_idx) else {
                return 0;
            };

            // Matching the full descriptor arrays and filtering afterwards
            // discards almost everything once the keyframes are mature: the best
            // matches always land on already-tracked features.
            let prev_unassoc: Vec<usize> = (0..prev_kf.frame.features.descriptors.len())
                .filter(|&i| prev_kf.map_point(i).is_none())
                .collect();
            let curr_unassoc: Vec<usize> = (0..curr_kf.frame.features.descriptors.len())
                .filter(|&i| curr_kf.map_point(i).is_none())
                .collect();
            if prev_unassoc.is_empty() || curr_unassoc.is_empty() {
                return 0;
            }

            let rel = Pose3d::between(
                &prev_kf.frame.pose_world_to_cam,
                &curr_kf.frame.pose_world_to_cam,
            );
            if rel.translation.length() <= 1e-8 {
                // No baseline: epipolar geometry degenerates and triangulation
                // would reject everything anyway.
                return 0;
            }
            let t = rel.translation;
            let t_skew = Mat3F64::from_cols(
                Vec3F64::new(0.0, t.z, -t.y),
                Vec3F64::new(-t.z, 0.0, t.x),
                Vec3F64::new(t.y, -t.x, 0.0),
            );
            let camera = &self.camera;
            // K^-1 built by hand: no distortion terms, valid only because the
            // camera is rectified (checked in `validate_camera`).
            let k_inv = Mat3F64::from_cols(
                Vec3F64::new(1.0 / camera.fx, 0.0, 0.0),
                Vec3F64::new(0.0, 1.0 / camera.fy, 0.0),
                Vec3F64::new(-camera.cx / camera.fx, -camera.cy / camera.fy, 1.0),
            );
            let f_mat = k_inv.transpose() * (t_skew * rel.rotation) * k_inv;

            // Near the epipole every keypoint is close to every epipolar line,
            // so the chi-square gate is uninformative there: wrong matches
            // survive and triangulate to depth-garbage that still reprojects
            // well in both views.
            let prev_center_world = prev_kf.frame.pose_world_to_cam.inverse().translation;
            let epipole_cam = curr_kf
                .frame
                .pose_world_to_cam
                .transform_point(&prev_center_world);
            let epipole_px = (epipole_cam.z.abs() > 1e-9).then(|| {
                Vec2F64::new(
                    camera.fx * epipole_cam.x / epipole_cam.z + camera.cx,
                    camera.fy * epipole_cam.y / epipole_cam.z + camera.cy,
                )
            });

            let prev_orients: Vec<f32> = prev_unassoc
                .iter()
                .map(|&i| prev_kf.frame.features.orientations[i])
                .collect();
            let prev_descs: Vec<[u8; 32]> = prev_unassoc
                .iter()
                .map(|&i| prev_kf.frame.features.descriptors[i])
                .collect();
            let curr_orients: Vec<f32> = curr_unassoc
                .iter()
                .map(|&i| curr_kf.frame.features.orientations[i])
                .collect();
            let curr_descs: Vec<[u8; 32]> = curr_unassoc
                .iter()
                .map(|&i| curr_kf.frame.features.descriptors[i])
                .collect();

            let sub_matches = match_orb_descriptors(
                &prev_orients,
                &prev_descs,
                &curr_orients,
                &curr_descs,
                match_config,
            );

            let mut pair_indices: Vec<(usize, usize)> = Vec::new();
            let mut matched_prev: Vec<Vec2F64> = Vec::new();
            let mut matched_curr: Vec<Vec2F64> = Vec::new();
            // `match_orb_descriptors` is ONE-DIRECTIONAL: it loops over side 1
            // (prev), takes the best j on side 2 (curr) and ratio-tests on the
            // prev side only. There is no cross-check and no uniqueness on the
            // curr side, so several `prev_idx` can name the same `curr_idx` —
            // and on a repeated pattern (tiled floor, brick, window mullions)
            // they all lie on the shared epipolar line by construction, so the
            // chi-square gate below keeps every one of them. The `curr_kf
            // .map_point(curr_idx).is_some()` guard further down runs *before*
            // `add_triangulated_points`, so it cannot see duplicates created in
            // the same batch: two distinct 3D points at different depths would
            // be built from one measurement, the second association silently
            // overwriting the first. The loser keeps `observation_kf_indices =
            // [curr_kf]` while `curr_kf` no longer references it — desyncing
            // the two directions that `fuse_into_neighbors` uses as its
            // "already observes" test — and enters BA with a single
            // observation, i.e. 2 residuals for 3 DoF.
            let mut claimed_curr: HashSet<usize> = HashSet::new();
            for (prev_sub, curr_sub) in sub_matches {
                let (Some(&prev_idx), Some(&curr_idx)) =
                    (prev_unassoc.get(prev_sub), curr_unassoc.get(curr_sub))
                else {
                    continue;
                };
                if !claimed_curr.insert(curr_idx) {
                    continue;
                }
                let (Some(pu), Some(qu)) = (
                    prev_kf.frame.undistorted_xy(prev_idx, camera),
                    curr_kf.frame.undistorted_xy(curr_idx, camera),
                ) else {
                    continue;
                };
                let p = Vec2F64::new(pu[0] as f64, pu[1] as f64);
                let q = Vec2F64::new(qu[0] as f64, qu[1] as f64);

                let octave = curr_kf
                    .frame
                    .features
                    .octaves
                    .get(curr_idx)
                    .copied()
                    .unwrap_or(0);

                if let Some(e) = epipole_px {
                    let dx = q.x - e.x;
                    let dy = q.y - e.y;
                    if dx * dx + dy * dy < 100.0 * ORB_SCALE_FACTOR.powi(octave as i32) {
                        continue;
                    }
                }

                let l = f_mat * Vec3F64::new(p.x, p.y, 1.0);
                let line_norm_sq = l.x * l.x + l.y * l.y;
                if line_norm_sq <= 1e-12 {
                    continue;
                }
                let d = l.x * q.x + l.y * q.y + l.z;
                let sigma_sq = ORB_SCALE_FACTOR.powi(2 * octave as i32);
                if d * d > EPIPOLAR_CHI2 * sigma_sq * line_norm_sq {
                    continue;
                }

                pair_indices.push((prev_idx, curr_idx));
                matched_prev.push(p);
                matched_curr.push(q);
            }
            if pair_indices.len() < MIN_GROWTH_MATCHES {
                return 0;
            }

            let triangulated = match triangulate_matched_points(
                &matched_prev,
                &matched_curr,
                &prev_kf.frame.pose_world_to_cam,
                &curr_kf.frame.pose_world_to_cam,
                camera,
                triangulation_config,
            ) {
                Ok(pts) => pts,
                Err(e) => {
                    debug!("grow: triangulation failed: {}", e.to_string());
                    return 0;
                }
            };

            let mut points = Vec::new();
            for tp in &triangulated {
                let Some(&(prev_idx, curr_idx)) = pair_indices.get(tp.pair_index) else {
                    continue;
                };
                if curr_kf.map_point(curr_idx).is_some() {
                    continue;
                }
                let color = keypoint_color(&curr_kf.frame, curr_idx);
                points.push((
                    tp.position,
                    curr_kf.frame.features.descriptors[curr_idx],
                    color,
                    prev_idx,
                    curr_idx,
                ));
            }
            points
        };

        // Write phase. `curr_kf` is registered as the first observer inside
        // `add_triangulated_points`.
        let first_mp_idx = self.map.num_map_points();
        let added = self.map.add_triangulated_points(None, curr_kf, &points);

        // Register the neighbour as a second observer: without it every new
        // point has a single observation, which biases the scale/normal geometry
        // and makes `cull()` over-aggressive.
        for (i, &(_, _, _, prev_desc_idx, _)) in points.iter().take(added).enumerate() {
            let mp_idx = first_mp_idx + i;
            self.map
                .register_observation_at(mp_idx, prev_kf_idx, prev_desc_idx);
            if let Some(prev_live) = self.map.get_keyframe_mut(prev_kf_idx) {
                prev_live.associate_map_point(prev_desc_idx, mp_idx);
            }
        }

        added
    }

    /// Forward-only subset of the neighbour fusion of Campos et al., ORB-SLAM3,
    /// IEEE T-RO 2021: project each map
    /// point observed by the current keyframe into every neighbour that does not
    /// yet observe it, and register the observation when it lands on a matching
    /// unassociated keypoint. No duplicate merging, no second-hop expansion.
    ///
    /// Cost is `O(neighbors x curr_map_points x nb_keypoints)` with no spatial
    /// index — profile this first if the live loop misses frame rate.
    fn fuse_into_neighbors(&mut self, curr_kf_idx: usize, neighbor_kf_indices: &[usize]) -> usize {
        let curr_mp_indices: Vec<usize> = match self.map.get_keyframe(curr_kf_idx) {
            Some(kf) => kf.map_point_by_desc_idx.iter().flatten().copied().collect(),
            None => return 0,
        };
        if curr_mp_indices.is_empty() {
            return 0;
        }

        let r2 = FUSE_SEARCH_RADIUS_PX * FUSE_SEARCH_RADIUS_PX;
        let mut n_fused = 0usize;

        for &nb_kf_idx in neighbor_kf_indices {
            if nb_kf_idx == curr_kf_idx {
                continue;
            }

            // (kp_idx_in_nb_kf, mp_idx, hamming), collected under a shared
            // borrow and resolved below so one keypoint can't be double-claimed.
            let mut proposals: Vec<(usize, usize, u32)> = Vec::new();
            {
                let Some(nb_kf) = self.map.get_keyframe(nb_kf_idx) else {
                    continue;
                };

                for &mp_idx in &curr_mp_indices {
                    let mp = match self.map.map_points().get(mp_idx) {
                        Some(mp) if !mp.culled => mp,
                        _ => continue,
                    };
                    if mp.observation_kf_indices.contains(&nb_kf_idx) {
                        continue;
                    }

                    let p_cam = nb_kf.frame.pose_world_to_cam.transform_point(&mp.position);
                    if p_cam.z <= 0.0 {
                        continue;
                    }
                    let Ok(pixel) =
                        self.camera
                            .project_to_image(&p_cam, 0.0, nb_kf.frame.image_size)
                    else {
                        continue;
                    };
                    let u = pixel.x as f32;
                    let v = pixel.y as f32;

                    let mut best_dist = u32::MAX;
                    let mut best_kp = usize::MAX;
                    for kp_idx in 0..nb_kf.frame.features.keypoints_xy.len() {
                        if nb_kf.map_point(kp_idx).is_some() {
                            continue;
                        }
                        let Some(kp) = nb_kf.frame.undistorted_xy(kp_idx, &self.camera) else {
                            continue;
                        };
                        let dx = kp[0] - u;
                        let dy = kp[1] - v;
                        if dx * dx + dy * dy > r2 {
                            continue;
                        }
                        let dist = hamming_distance(
                            &mp.descriptor,
                            &nb_kf.frame.features.descriptors[kp_idx],
                        );
                        if dist < best_dist {
                            best_dist = dist;
                            best_kp = kp_idx;
                        }
                    }

                    if best_dist <= FUSE_MAX_HAMMING && best_kp != usize::MAX {
                        proposals.push((best_kp, mp_idx, best_dist));
                    }
                }
            }

            proposals.sort_by_key(|&(_, _, dist)| dist);
            let mut taken_kp: HashSet<usize> = HashSet::new();
            for (kp_idx, mp_idx, _) in proposals {
                if taken_kp.contains(&kp_idx) {
                    continue;
                }
                let already = self
                    .map
                    .get_keyframe(nb_kf_idx)
                    .and_then(|kf| kf.map_point(kp_idx))
                    .is_some();
                if already {
                    continue;
                }
                self.map.register_observation_at(mp_idx, nb_kf_idx, kp_idx);
                if let Some(nb_live) = self.map.get_keyframe_mut(nb_kf_idx) {
                    nb_live.associate_map_point(kp_idx, mp_idx);
                }
                taken_kp.insert(kp_idx);
                n_fused += 1;
            }
        }

        n_fused
    }
}

// ── Free helpers ─────────────────────────────────────────────────────────────

/// A static label for a map-projection rejection, for the structured log (the upstream enum
/// implements neither `Display` nor `Serialize`).
fn reject_reason_str(reason: MapProjectionRejectReason) -> &'static str {
    match reason {
        MapProjectionRejectReason::LowProjectionMatches => "low_projection_matches",
        MapProjectionRejectReason::PnpFailed => "pnp_failed",
        MapProjectionRejectReason::LowPnpInliers => "low_pnp_inliers",
        MapProjectionRejectReason::LowReferenceMatches => "low_reference_matches",
        MapProjectionRejectReason::LowReferenceCorrespondences => "low_reference_correspondences",
    }
}

fn validate_camera(camera: &PinholeCamera, baseline_m: f64) -> Result<(), VioError> {
    if camera.k1 != 0.0 || camera.k2 != 0.0 || camera.p1 != 0.0 || camera.p2 != 0.0 {
        return Err(VioError::DistortedCamera {
            k1: camera.k1,
            k2: camera.k2,
            p1: camera.p1,
            p2: camera.p2,
        });
    }
    if !baseline_m.is_finite() || baseline_m <= 0.0 {
        return Err(VioError::InvalidBaseline {
            baseline: Length::new::<meter>(baseline_m),
        });
    }
    Ok(())
}

fn validate_frame_arrays(frame: &Frame) -> Result<(), VioError> {
    let n = frame.features.keypoints_xy.len();
    let stereo_ok = (frame.u_right.is_empty() && frame.depth.is_empty())
        || (frame.u_right.len() == n && frame.depth.len() == n);
    if !stereo_ok
        || frame.keypoint_colors.len() != n
        || frame.features.descriptors.len() != n
        || frame.features.octaves.len() != n
        || frame.features.orientations.len() != n
    {
        return Err(VioError::FrameArraysNotParallel {
            n_keypoints: n,
            n_u_right: frame.u_right.len(),
            n_depth: frame.depth.len(),
            n_colors: frame.keypoint_colors.len(),
            n_descriptors: frame.features.descriptors.len(),
            n_octaves: frame.features.octaves.len(),
            n_orientations: frame.features.orientations.len(),
        });
    }

    // Lengths are not enough: `process_frame` is public and is the seam the
    // tests and offline replays drive, so a caller that did its own stereo
    // can hand us `+inf`. See `VioError::NonFiniteFrameValue` for why that one
    // in particular is unrecoverable rather than merely wrong.
    let non_finite =
        |field: &'static str, index: usize, value: f64| VioError::NonFiniteFrameValue {
            field,
            index,
            value,
        };
    for (i, kp) in frame.features.keypoints_xy.iter().enumerate() {
        if !kp[0].is_finite() || !kp[1].is_finite() {
            let bad = if kp[0].is_finite() { kp[1] } else { kp[0] };
            return Err(non_finite("keypoints_xy", i, bad as f64));
        }
    }
    for (i, &u) in frame.u_right.iter().enumerate() {
        if !u.is_finite() {
            return Err(non_finite("u_right", i, u as f64));
        }
    }
    for (i, &d) in frame.depth.iter().enumerate() {
        if !d.is_finite() {
            return Err(non_finite("depth", i, d as f64));
        }
    }
    Ok(())
}

fn keypoint_color(frame: &Frame, desc_idx: usize) -> [u8; 3] {
    frame
        .keypoint_colors
        .get(desc_idx)
        .copied()
        .unwrap_or([128; 3])
}

/// Level `level`'s dimension, from the level-0 dimension.
///
/// Byte-for-byte the formula `OrbDetector` uses (`pyramid_size_at_level` in
/// kornia-imgproc's ORB extractor): f64 arithmetic and `round_ties_even`. It is
/// duplicated rather than approximated because the two pyramids MUST come out
/// the same size — `kornia_slam::stereo` takes a keypoint detected by
/// `OrbDetector` at octave `o`, scales its coordinate by `1.2^-o`, and indexes
/// *our* level `o` with it. A resolution whose scaled dimension lands on `.5`
/// resolves differently under `f32::round` (half away from zero) than under
/// `f64::round_ties_even`, and the two pyramids would silently disagree by a
/// pixel at that octave. 640x400 and 752x480 happen not to hit it; that is luck,
/// not a property, so do not "simplify" this back to `as f32` + `.round()`.
fn pyramid_dim_at_level(base: usize, level: usize) -> usize {
    // `as f32 as f64` is NOT redundant. `OrbDetector::downscale` is an f32 field, which
    // `TrackerConfig::new` sets to `ORB_SCALE_FACTOR as f32`, and `pyramid_size_at_level`
    // then widens it back with `(downscale as f64)`. That round trip gives
    // 1.2000000476837158, not the f64 1.2 that `ORB_SCALE_FACTOR` is — a ~4e-8 relative
    // difference, which is irrelevant everywhere except within one ULP of a `.5` rounding
    // boundary, where it is the whole answer. Reproduce the round trip rather than the
    // intent.
    let inv_scale = (ORB_SCALE_FACTOR as f32 as f64).powi(-(level as i32));
    ((base as f64 * inv_scale).round_ties_even() as usize).max(1)
}

/// Builds the octave pyramid `kornia_slam::stereo` reads: level `o` is the
/// **full-resolution** image resampled once by `1 / 1.2^o`.
///
/// This is not `OrbDetector`'s internal pyramid, and the difference is *pixel
/// content*, not geometry. Both size every level from level 0 (see
/// [`pyramid_dim_at_level`]) — `OrbDetector` does so deliberately, its own doc
/// noting that chaining `prev / scale` accumulates rounding drift — so the
/// `u * 1.2^-o` coordinate mapping is equally valid against either. What
/// differs is that `OrbDetector` chains its bilinear reductions level-to-level
/// (each level resampled from the one above), while this resamples every level
/// straight from full resolution: our upper levels are therefore more aliased
/// than the ones the descriptors were extracted from, which degrades the SAD
/// parabola fit at octave >= 1 rather than shifting any coordinate.
///
/// Two consequences worth knowing before changing this. It is not free — 7
/// full-image resizes per eye per frame, plus a `clone()` for level 0,
/// rebuilding a pyramid `OrbDetector` just built and discarded. And chaining
/// the reductions here (matching `OrbDetector` exactly, which is strictly more
/// faithful) is a behaviour change to stereo depth, so it needs the
/// `tests/stereo_depth.rs` numbers re-measured, not just a green test run.
fn build_u8_pyramid(img: &Image<u8, 1>) -> Result<Vec<Image<u8, 1>>, VioError> {
    let mut pyramid = Vec::with_capacity(ORB_N_LEVELS);
    pyramid.push(img.clone());
    for level in 1..ORB_N_LEVELS {
        let width = pyramid_dim_at_level(img.width(), level);
        let height = pyramid_dim_at_level(img.height(), level);
        let mut dst = Image::from_size_val(ImageSize { width, height }, 0u8)?;
        resize_fast_mono(img, &mut dst, InterpolationMode::Bilinear)?;
        pyramid.push(dst);
    }
    Ok(pyramid)
}

/// Samples one grey value per keypoint from the left image, truncating then
/// clamping the coordinates (the upstream kornia-slam example does exactly this).
fn sample_keypoint_colors(gray: &Image<u8, 1>, keypoints_xy: &[[f32; 2]]) -> Vec<[u8; 3]> {
    let (width, height) = (gray.width(), gray.height());
    if width == 0 || height == 0 {
        return vec![[128; 3]; keypoints_xy.len()];
    }
    let data = gray.as_slice();
    keypoints_xy
        .iter()
        .map(|kp| {
            // Negative f32 saturates to 0 on an `as usize` cast.
            let x = (kp[0] as usize).min(width - 1);
            let y = (kp[1] as usize).min(height - 1);
            let g = data.get(y * width + x).copied().unwrap_or(128);
            [g, g, g]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::units::si::f64::Frequency;

    fn rectified_camera() -> PinholeCamera {
        PinholeCamera {
            fx: 400.0,
            fy: 400.0,
            cx: 320.0,
            cy: 200.0,
            k1: 0.0,
            k2: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    #[test]
    fn test_rejects_a_distorted_camera() {
        let camera = PinholeCamera {
            k1: -0.12,
            ..rectified_camera()
        };
        let err = validate_camera(&camera, 0.075).unwrap_err();
        assert!(matches!(err, VioError::DistortedCamera { .. }), "{err:?}");
    }

    #[test]
    fn test_rejects_a_non_positive_baseline() {
        let err = validate_camera(&rectified_camera(), 0.0).unwrap_err();
        assert!(matches!(err, VioError::InvalidBaseline { .. }), "{err:?}");
    }

    #[test]
    fn test_close_depth_is_thirty_five_baselines() {
        let cfg = TrackerConfig::new(Length::new::<meter>(0.075));
        let close = cfg
            .stereo_close_depth
            .expect("densification enabled")
            .get::<meter>();
        assert!((close - 2.625).abs() < 1e-12, "got {close}");
    }

    #[test]
    fn test_orb_pyramid_geometry_matches_kornia_slam_scale_tables() {
        let img = Image::from_size_val(
            ImageSize {
                width: 640,
                height: 400,
            },
            7u8,
        )
        .unwrap();
        let pyramid = build_u8_pyramid(&img).unwrap();
        assert_eq!(pyramid.len(), ORB_N_LEVELS);
        assert_eq!(pyramid[0].width(), 640);
        for (level, img) in pyramid.iter().enumerate() {
            let expect = (640.0 / 1.2f64.powi(level as i32)).round() as usize;
            assert_eq!(img.width(), expect.max(1), "level {level}");
        }
    }

    /// Our pyramid must be dimension-identical to `OrbDetector`'s at every
    /// level, for every resolution we ship.
    ///
    /// `kornia_slam::stereo` scales a keypoint that `OrbDetector` found at
    /// octave `o` by `1.2^-o` and indexes OUR level `o` with it. If the two
    /// pyramids disagree by even one pixel at some level, that indexing is
    /// wrong for every keypoint at that octave and the failure is a quiet
    /// disparity bias, not a panic. Checked against the exact upstream formula
    /// (f64 + `round_ties_even`), not against `1.2^-o` rounded some other way —
    /// that is the difference this test exists to catch.
    #[test]
    fn test_pyramid_levels_match_the_orb_extractor_sizes_exactly() {
        // Verbatim from kornia-imgproc `features/orb/extractor.rs`:
        // `pyramid_size_at_level`.
        fn kornia_size(base: usize, downscale: f32, level: usize) -> usize {
            let inv_scale = (downscale as f64).powi(-(level as i32));
            (base as f64 * inv_scale).round_ties_even() as usize
        }

        // 640x400 = the OAK stereo mode; 752x480 = EuRoC; the rest are odd
        // sizes chosen to exercise the rounding boundary.
        for (w0, h0) in [
            (640usize, 400usize),
            (752, 480),
            (1280, 720),
            (641, 401),
            (255, 173),
        ] {
            let img = Image::from_size_val(
                ImageSize {
                    width: w0,
                    height: h0,
                },
                7u8,
            )
            .unwrap();
            let pyr = build_u8_pyramid(&img).unwrap();
            assert_eq!(pyr.len(), ORB_N_LEVELS);
            for (level, img_l) in pyr.iter().enumerate() {
                assert_eq!(
                    (img_l.width(), img_l.height()),
                    (
                        kornia_size(w0, ORB_SCALE_FACTOR as f32, level).max(1),
                        kornia_size(h0, ORB_SCALE_FACTOR as f32, level).max(1)
                    ),
                    "{w0}x{h0} level {level}: our pyramid and OrbDetector's disagree, so \
                     kornia_slam::stereo's `u * 1.2^-o` mapping indexes the wrong pixels"
                );
            }
        }
    }

    #[test]
    fn test_colors_are_clamped_not_wrapped() {
        let img = Image::from_size_val(
            ImageSize {
                width: 4,
                height: 4,
            },
            9u8,
        )
        .unwrap();
        // Out of range on both sides; must clamp rather than index out of bounds.
        let colors = sample_keypoint_colors(&img, &[[-5.0, -5.0], [99.0, 99.0], [1.0, 1.0]]);
        assert_eq!(colors, vec![[9; 3]; 3]);
    }

    // ── Inertial ─────────────────────────────────────────────────────────────
    //
    // These drive `on_keyframe_inertial` directly with a hand-built map rather than through
    // synthetic imagery: what is under test is the ingest contract — which intervals are
    // refused, and what bounds reach `Map::add_imu_factor` — and none of that depends on how
    // the keyframes were produced. `add_imu_factor` itself does not validate that its indices
    // name live keyframes (an unknown index makes the factor silently vanish from the solve
    // later), so an empty map is a faithful stand-in here and a hazard worth stating.

    /// A real epoch, not 0: the whole reason samples are buffered in `u64` nanoseconds is that
    /// this value in `f64` seconds leaves ~400 ns of resolution against a 5 ms sample period.
    const EPOCH_NS: u64 = 1_757_000_000_000_000_000;
    const IMU_PERIOD_NS: u64 = 5_000_000;

    fn inertial_config() -> InertialConfig {
        // 90 degrees about z. Asymmetric under transpose, so a convention slip in anything
        // that consumes it would change the answer rather than cancel.
        let rotation = Mat3F64::from_cols(
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(-1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        InertialConfig::new(
            Pose3d::from_rt(rotation, Vec3F64::new(0.01, -0.02, 0.03)),
            // EuRoC ADIS16448 densities. Fine for a test, wrong for any other IMU — which is
            // why the real ones have to come from config with no default.
            kornia_sensors::imu::ImuCalib {
                gyro_noise: 1.6968e-4,
                accel_noise: 2.0e-3,
                gyro_bias_noise: 1.9393e-5,
                accel_bias_noise: 3.0e-3,
            },
            Frequency::new::<hertz>(200.0),
        )
        .expect("a proper rotation and positive densities")
    }

    fn tracker_with_inertial() -> Tracker {
        // `InertialConfig::new` defaults the flag to true (the RON reader is what defaults it
        // to false), so this is the fully-armed tracker.
        tracker_with_inertial_ba(true)
    }

    fn tracker_with_inertial_ba(enable_inertial_ba: bool) -> Tracker {
        let mut config = TrackerConfig::new(Length::new::<meter>(0.075));
        let mut inertial = inertial_config();
        inertial.enable_inertial_ba = enable_inertial_ba;
        config.inertial = Some(inertial);
        let mut tracker = Tracker::new(rectified_camera(), config);
        // Normally set by the first frame; set here because these tests never feed one.
        tracker.epoch_ns = Some(EPOCH_NS);
        tracker
    }

    /// The landmark snapshot window: the newest `cap` SLOTS, culled ones skipped rather than
    /// backfilled from older slots, indices exactly as the map holds them.
    #[test]
    fn test_newest_live_map_points_windows_the_tail_and_skips_culled() {
        let mut tracker = Tracker::new(
            rectified_camera(),
            TrackerConfig::new(Length::new::<meter>(0.075)),
        );
        for i in 0..10 {
            let p = Vec3F64::new(i as f64, -(i as f64), 0.5 * i as f64);
            tracker
                .map
                .push_map_point(kornia_slam::map::MapPoint::new(p, [0; 32], 0, [0; 3], 0));
        }
        tracker.map.map_points_mut()[2].mark_culled();
        tracker.map.map_points_mut()[7].mark_culled();

        let got: Vec<_> = tracker.newest_live_map_points(5).collect();
        let want: Vec<_> = [5usize, 6, 8, 9]
            .into_iter()
            .map(|i| (i, [i as f64, -(i as f64), 0.5 * i as f64]))
            .collect();
        assert_eq!(got, want);

        // A cap past the map's size is the whole live map.
        assert_eq!(tracker.newest_live_map_points(100).count(), 8);
        assert_eq!(tracker.newest_live_map_points(0).count(), 0);
    }

    /// Nothing in kornia-slam removes a keyframe, so an unbounded map is the shipped behaviour
    /// and `max_keyframes` is the only thing standing between clean tracking and the silent
    /// stall measured on a Jetson Orin with an OAK-D at 640x400 (757 MB RSS, zero poses after
    /// 2.5 h).
    #[test]
    fn test_the_map_bound_drops_the_map_and_bumps_the_generation() {
        let mut tracker = tracker_with_inertial();
        tracker.config.max_keyframes = Some(4);
        for i in 0..6 {
            tracker
                .map
                .upsert_keyframe(bare_keyframe(i, Vec3F64::new(0.01 * i as f64, 0.0, 0.0)));
        }
        let gen_before = tracker.world_generation();

        // Not armed: the bound must never fire on its own, or every consumer sees spurious
        // reset epochs.
        tracker.apply_pending_map_reset();
        assert_eq!(tracker.map.keyframes().len(), 6, "unarmed reset fired");
        assert_eq!(tracker.world_generation(), gen_before);

        tracker.pending_map_reset = true;
        tracker.apply_pending_map_reset();
        assert!(tracker.map.keyframes().is_empty(), "map survived the bound");
        assert_eq!(
            tracker.world_generation(),
            gen_before + 1,
            "a dropped map MUST bump the generation, or a consumer composes two worlds"
        );
        // One-shot: the flag is consumed, so the next frame does not reset again.
        assert!(!tracker.pending_map_reset);
    }

    /// The arming predicate itself, which nothing else reaches: `tracking_step` is private and
    /// no test drives a frame through it, so every conjunct here is a way to disable the bound
    /// or fire it every frame with a green suite.
    #[test]
    fn test_the_bound_arms_only_on_an_insertion_that_reaches_the_cap() {
        let cases = [
            // (cap, keyframes, inserted, should_arm)
            (Some(4), 4, true, true),
            (Some(4), 5, true, true),
            (Some(4), 3, true, false),
            // Not an insertion: the map did not grow, so nothing can have crossed.
            (Some(4), 9, false, false),
            // Opt-in. `None` is every graph without the key, and must never arm.
            (None, 50, true, false),
        ];
        for (cap, kfs, inserted, want) in cases {
            let mut tracker = tracker_with_inertial();
            tracker.config.max_keyframes = cap;
            for i in 0..kfs {
                tracker
                    .map
                    .upsert_keyframe(bare_keyframe(i, Vec3F64::new(0.01 * i as f64, 0.0, 0.0)));
            }
            tracker.arm_map_bound(inserted);
            assert_eq!(
                tracker.pending_map_reset, want,
                "cap={cap:?} keyframes={kfs} inserted={inserted}"
            );
        }
    }

    /// The TRACKING-LOSS reset must keep the bias too — it is the drop a live graph takes most.
    ///
    /// A test that only exercised the `max_keyframes` bound could not see a loss path that ran
    /// `reset()` and re-estimated the bias from zero on nearly every real drop, so this drives
    /// the loss path's own entry point.
    #[test]
    fn test_a_tracking_loss_map_drop_keeps_the_imu_bias() {
        let measured = ImuBias {
            gyro: Vec3F64::new(0.0026, -0.0011, 0.0009),
            accel: Vec3F64::new(0.01, -0.02, 0.03),
        };
        let mut tracker = tracker_with_inertial();
        tracker.config.reset_map_on_loss = true;
        tracker.inertial.as_mut().unwrap().bias = measured;

        // The loss path the live graph takes, reached the way `tracking_step` reaches it.
        tracker.reset_keeping_calibration();
        assert_eq!(
            tracker.inertial.as_ref().unwrap().bias.gyro,
            measured.gyro,
            "a tracking-loss drop re-estimated the bias from zero"
        );

        // And the requested reset remains the one path that DOES start it over.
        tracker.inertial.as_mut().unwrap().bias = measured;
        let initial = tracker.inertial.as_ref().unwrap().cfg.initial_bias;
        tracker.reset();
        assert_eq!(
            tracker.inertial.as_ref().unwrap().bias.gyro,
            initial.gyro,
            "a requested reset must still start the bias over"
        );
    }

    /// A gyro prior must rotate the camera IN PLACE, not swing it about the world origin.
    ///
    /// `Pose3d` is world-to-camera, so the centre is `-R^T t`. Writing `rotation` alone moves
    /// the predicted centre by ~theta * |t|, which is metres once the robot is any distance
    /// out, and worst exactly while turning. The rotation residual
    /// that validated the prior compares rotations only, so it could not see this.
    #[test]
    fn test_the_gyro_prior_rotates_about_the_camera_centre_not_the_world_origin() {
        use kornia_3d::pose::Pose3d;
        // Camera 5 m from the map origin, looking along +x.
        let centre = Vec3F64::new(5.0, 0.0, 0.0);
        let before = Pose3d::from_rt(Mat3F64::IDENTITY, -(Mat3F64::IDENTITY * centre));
        assert!(
            (before.inverse().translation - centre).length() < 1e-9,
            "fixture wrong: centre must round-trip"
        );

        // A 0.03 rad yaw, the size one 66 ms frame at 0.45 rad/s produces.
        let th: f64 = 0.03;
        let (c, sn) = (th.cos(), th.sin());
        let d = Mat3F64::from_cols_array(&[c, sn, 0.0, -sn, c, 0.0, 0.0, 0.0, 1.0]);

        // THE PRODUCTION FUNCTION, not a copy of its body beside it: re-deriving the two lines
        // here would pass whether or not `tracking_step` actually calls them.
        let candidate = rotate_about_centre(before, d.transpose() * before.rotation);

        let moved = (candidate.inverse().translation - centre).length();
        assert!(
            moved < 1e-9,
            "the gyro prior teleported the camera centre by {moved:.4} m"
        );

        // And the naive version really does move it — otherwise this test proves nothing.
        let mut naive = before;
        naive.rotation = d.transpose() * before.rotation;
        let naive_moved = (naive.inverse().translation - centre).length();
        assert!(
            naive_moved > 0.1,
            "writing rotation alone should move the centre ~theta*|t| = 0.15 m; got {naive_moved:.4}"
        );
    }

    /// A map-size reset keeps the IMU bias; a requested reset does not.
    ///
    /// Bias is a property of the sensor. Dropping it when the MAP hit a size cap makes the next
    /// initial solve land wide and the first refinement that corrects it rotate the whole world
    /// — measured as gravity_y 8.94 then 9.81, one large world rotation per map cycle.
    #[test]
    fn test_a_map_bound_reset_keeps_the_imu_bias_and_a_requested_reset_does_not() {
        let measured = ImuBias {
            gyro: Vec3F64::new(0.0026, -0.0011, 0.0009),
            accel: Vec3F64::new(0.01, -0.02, 0.03),
        };

        let mut tracker = tracker_with_inertial();
        tracker.inertial.as_mut().unwrap().bias = measured;
        tracker.reset_keeping_calibration();
        let kept = tracker.inertial.as_ref().unwrap().bias;
        assert_eq!(
            kept.gyro, measured.gyro,
            "map-bound reset lost the gyro bias"
        );
        assert_eq!(
            kept.accel, measured.accel,
            "map-bound reset lost the accel bias"
        );

        let mut tracker = tracker_with_inertial();
        tracker.inertial.as_mut().unwrap().bias = measured;
        let initial = tracker.inertial.as_ref().unwrap().cfg.initial_bias;
        tracker.reset();
        assert_eq!(
            tracker.inertial.as_ref().unwrap().bias.gyro,
            initial.gyro,
            "a requested reset must start the bias over"
        );

        // Neither carries `gravity_world`: the bootstrap after a reset defines a new world.
        // MOVED OFF THE DEFAULT FIRST — asserting it equals the default without ever changing
        // it passes whether or not the reset touches it, which is no test at all.
        let mut tracker = tracker_with_inertial();
        tracker.inertial.as_mut().unwrap().gravity_world = Vec3F64::new(0.2, 8.9, -4.0);
        tracker.reset_keeping_calibration();
        assert!(
            (tracker.inertial.as_ref().unwrap().gravity_world.z + GRAVITY_MAGNITUDE).abs() < 1e-9,
            "gravity_world must not survive a reset — it is world-frame"
        );
    }

    /// A requested reset while the bound is armed must not drop the FRESH map on the next
    /// frame and bump the generation a second time for no visible reason.
    #[test]
    fn test_a_reset_disarms_the_pending_bound() {
        let mut tracker = tracker_with_inertial();
        tracker.config.max_keyframes = Some(2);
        for i in 0..3 {
            tracker
                .map
                .upsert_keyframe(bare_keyframe(i, Vec3F64::new(0.01 * i as f64, 0.0, 0.0)));
        }
        tracker.arm_map_bound(true);
        assert!(tracker.pending_map_reset);

        tracker.reset();
        assert!(!tracker.pending_map_reset, "reset left the bound armed");
        let gen_after_reset = tracker.world_generation();
        tracker.apply_pending_map_reset();
        assert_eq!(
            tracker.world_generation(),
            gen_after_reset,
            "a disarmed bound must not bump the generation again"
        );
    }

    /// A featureless keyframe at `idx` with its camera centre at `centre`. `add_imu_factor`,
    /// `ready` and `apply_initialization` all read `frame.idx` and `pose_world_to_cam` and
    /// nothing else, so this is enough to make them do their real work.
    fn bare_keyframe(idx: usize, centre: Vec3F64) -> Keyframe {
        Keyframe::from_frame(Frame {
            idx,
            features: kornia_imgproc::features::OrbFeatures {
                keypoints_xy: Vec::new(),
                orientations: Vec::new(),
                descriptors: Vec::new(),
                octaves: Vec::new(),
            },
            pose_world_to_cam: Pose3d::from_rt(Mat3F64::IDENTITY, -centre),
            image_size: ImageSize {
                width: 640,
                height: 400,
            },
            keypoint_colors: Vec::new(),
            u_right: Vec::new(),
            depth: Vec::new(),
            keypoints_undist: Vec::new(),
        })
    }

    /// What `try_initialize` hands back when its gates pass. Gravity deliberately OFF world
    /// +y and scale deliberately not 1, so that `apply_initialization` — if it runs — cannot
    /// be a no-op on the map: `rotate_world` gets a non-identity rotation and `scale_world` a
    /// non-unit factor.
    fn accepted_init() -> ImuInitResult {
        ImuInitResult {
            scale: 1.7,
            gravity_world: Vec3F64::new(0.3, 9.7, -0.8),
            velocities_world: vec![Vec3F64::new(0.1, 0.0, 0.0)],
            bias: ImuBias {
                gyro: Vec3F64::new(0.01, -0.02, 0.03),
                accel: Vec3F64::new(0.1, 0.2, -0.1),
            },
        }
    }

    fn pose_bits(p: &Pose3d) -> ([u64; 9], [u64; 3]) {
        (
            p.rotation.to_cols_array().map(f64::to_bits),
            p.translation.to_array().map(f64::to_bits),
        )
    }

    fn imu_samples(from_ns: u64, n: usize) -> Vec<RawImuSample> {
        (0..n)
            .map(|i| RawImuSample {
                stamp_ns: from_ns + i as u64 * IMU_PERIOD_NS,
                gyro: [0.0, 0.0, 0.05],
                accel: [0.0, 9.81, 0.0],
            })
            .collect()
    }

    #[test]
    fn test_the_inertial_path_is_off_unless_an_extrinsic_is_configured() {
        assert!(
            TrackerConfig::new(Length::new::<meter>(0.075))
                .inertial
                .is_none(),
            "the default config must never carry a guessed extrinsic"
        );
        let mut tracker = Tracker::new(
            rectified_camera(),
            TrackerConfig::new(Length::new::<meter>(0.075)),
        );
        assert!(tracker.inertial_stats().is_none());

        // Pushing samples at a tracker that cannot read them is a wiring mistake, and it looks
        // exactly like a working inertial front end from the outside. So it is an error.
        let err = tracker
            .push_raw_imu(imu_samples(EPOCH_NS, 10), 0, false)
            .unwrap_err();
        assert!(matches!(err, VioError::InertialDisabled), "{err:?}");
    }

    #[test]
    fn test_a_disabled_tracker_adds_no_factor_and_keeps_local_ba_visual() {
        let mut tracker = Tracker::new(
            rectified_camera(),
            TrackerConfig::new(Length::new::<meter>(0.075)),
        );
        tracker.epoch_ns = Some(EPOCH_NS);
        let handles = tracker.on_keyframe_inertial(
            Some((0, EPOCH_NS + 1_000_000_000)),
            1,
            EPOCH_NS + 1_500_000_000,
        );
        // `None` is what selects `run_local_ba` over `run_local_inertial_ba` at the call site.
        assert!(handles.is_none());
        assert!(tracker.map().imu_factors().is_empty());
    }

    #[test]
    fn test_already_aligned_samples_are_refused_rather_than_rotated_twice() {
        let mut tracker = tracker_with_inertial();
        let err = tracker
            .push_raw_imu(imu_samples(EPOCH_NS, 10), 0, true)
            .unwrap_err();
        assert!(matches!(err, VioError::InertialSamplesAligned), "{err:?}");
        // And nothing was buffered, so a later interval cannot half-succeed on them.
        assert_eq!(tracker.inertial_stats().unwrap().buffer.accepted, 0);
    }

    /// The public edge takes the unit-typed payload sample; its `tov` and rad/s, m/s^2 values
    /// must land in the ring unchanged.
    #[test]
    fn test_push_imu_accepts_payload_samples() {
        use cu_stereo_payloads::ImuPayload;

        let mut tracker = tracker_with_inertial();
        let payload = (0..3u64).map(|i| ImuSample {
            tov: CuTime::from_nanos(EPOCH_NS + i * IMU_PERIOD_NS),
            imu: ImuPayload::from_raw([0.0, 0.0, 9.81], [0.1, 0.0, 0.0], 25.0),
        });
        assert_eq!(tracker.push_imu(payload, 0, false).expect("inertial on"), 3);
        let raw = RawImuSample::from(ImuSample {
            tov: CuTime::from_nanos(EPOCH_NS),
            imu: ImuPayload::from_raw([0.0, 0.0, 9.81], [0.1, 0.0, 0.0], 25.0),
        });
        assert_eq!(raw.stamp_ns, EPOCH_NS);
        assert!((raw.accel[2] - f64::from(9.81f32)).abs() < 1e-12, "{raw:?}");
        assert!((raw.gyro[0] - f64::from(0.1f32)).abs() < 1e-12, "{raw:?}");
    }

    #[test]
    fn test_an_imu_edge_spans_exactly_the_two_keyframe_stamps() {
        let mut tracker = tracker_with_inertial();
        let t0_ns = EPOCH_NS + 1_000_000_000;
        let t1_ns = EPOCH_NS + 1_600_000_000;
        // 1.0 s through 1.995 s, so the interval is fully covered at both ends.
        tracker
            .push_raw_imu(imu_samples(t0_ns, 200), 0, false)
            .expect("inertial path is on");

        let handles = tracker.on_keyframe_inertial(Some((7, t0_ns)), 11, t1_ns);
        // Not initialized yet (no keyframes, so `ready` is false), so BA stays visual.
        assert!(handles.is_none());

        let factors = tracker.map().imu_factors();
        assert_eq!(factors.len(), 1);
        let f = &factors[0];
        // Both indices are `Frame::idx`, not positions in the keyframe vector.
        assert_eq!((f.prev_kf_idx, f.curr_kf_idx), (7, 11));
        // Seconds since the TRACKER's epoch, the same rebasing `process_frame` applies to
        // frame stamps — a second epoch here would make `from_measurements` filter every
        // sample away and return a zero delta with no error.
        assert!((f.t0 - 1.0).abs() < 1e-12, "t0 = {}", f.t0);
        assert!((f.t1 - 1.6).abs() < 1e-12, "t1 = {}", f.t1);
        // The integration covers the whole interval, not just the span of the samples.
        assert!(
            (f.preintegrated.dt - 0.6).abs() < 1e-9,
            "dt = {}",
            f.preintegrated.dt
        );
        // Samples at 1.000, 1.005 ... 1.600 inclusive.
        assert_eq!(f.raw_samples.len(), 121);
        assert!(
            f.raw_samples.first().unwrap().timestamp >= f.t0
                && f.raw_samples.last().unwrap().timestamp <= f.t1
        );
        // Retained on the factor, which is what `Map`'s repropagation path re-integrates once
        // the bias has drifted past its linearization point. An empty Vec would compile.
        assert!(!f.raw_samples.is_empty());

        let stats = tracker.inertial_stats().unwrap();
        assert_eq!(stats.factors_added, 1);
        // What the map is pinning, as the stop line reports it: the one factor and its 121
        // retained raw samples.
        assert_eq!((stats.retained_factors, stats.retained_samples), (1, 121));
        // Pruned to the keyframe stamp: 1.600 s through 1.995 s inclusive.
        assert_eq!(stats.buffer.buffered, 80);
    }

    /// The excitation statistics must be scoped to the window `ImuInitializer::ready` is scoped
    /// to — factors with `curr_kf_idx >= start_idx`. A pre-window factor swept in would credit
    /// the window with motion that happened before it opened, and the whole point of measuring
    /// this is to decide later whether the window itself was excited.
    ///
    /// The per-statistic maths is pinned in `imu.rs`; what this pins is the filter.
    #[test]
    fn test_init_window_excitation_covers_exactly_the_factors_ready_counts() {
        let mut tracker = tracker_with_inertial();
        let stamp = |i: u64| EPOCH_NS + 1_000_000_000 + i * 500_000_000;
        // 3.0 s of 200 Hz samples from the first stamp: covers both intervals with margin.
        tracker
            .push_raw_imu(imu_samples(stamp(0), 600), 0, false)
            .unwrap();
        // Two edges: 3 -> 4 lands before a window opened at 5, and 7 -> 8 lands inside it.
        tracker.on_keyframe_inertial(Some((3, stamp(0))), 4, stamp(1));
        tracker.on_keyframe_inertial(Some((7, stamp(1))), 8, stamp(2));
        assert_eq!(tracker.map().imu_factors().len(), 2);

        // 0.5 s of the fixture's constant 0.05 rad/s about z, over 101 samples.
        let inside = tracker.init_window_excitation(5);
        assert_eq!(inside.samples, 101);
        assert!((inside.rot_rad - 0.05 * 0.5).abs() < 1e-9, "{inside:?}");
        // Both factors in scope: the angle doubles. This is what an unscoped (or inverted)
        // filter would return for the call above, so the two numbers must differ.
        let both = tracker.init_window_excitation(0);
        // 202, not 201: abutting factors each retain the sample on their shared boundary, so a
        // boundary sample is counted twice. Harmless for both statistics at 200 Hz, stated here
        // because the count is otherwise off by one against the sample stream.
        assert_eq!(both.samples, 202);
        assert!((both.rot_rad - 2.0 * 0.05 * 0.5).abs() < 1e-9, "{both:?}");
        // The fixture IMU is parked: one constant 9.81 magnitude, so the translational
        // statistic is exactly zero whichever scoping applies.
        assert_eq!((inside.accel_var, both.accel_var), (0.0, 0.0));
    }

    #[test]
    fn test_an_interval_with_reported_drops_is_refused_not_shortened() {
        let mut tracker = tracker_with_inertial();
        let t0_ns = EPOCH_NS + 1_000_000_000;
        let t1_ns = EPOCH_NS + 1_600_000_000;
        // Two batches at an unbroken 200 Hz cadence, so there is NO gap to detect: the only
        // evidence that samples went missing is the source's cumulative counter. Without the
        // check, `from_measurements` returns a full-0.6 s delta and nothing anywhere says the
        // stream was holed.
        tracker
            .push_raw_imu(imu_samples(t0_ns, 100), 0, false)
            .unwrap();
        tracker
            .push_raw_imu(imu_samples(t0_ns + 100 * IMU_PERIOD_NS, 100), 12, false)
            .unwrap();

        let handles = tracker.on_keyframe_inertial(Some((7, t0_ns)), 11, t1_ns);
        assert!(handles.is_none());
        assert!(
            tracker.map().imu_factors().is_empty(),
            "a holed interval must produce NO factor, not a short one"
        );
        let stats = tracker.inertial_stats().unwrap();
        assert_eq!(stats.refused_dropped, 1);
        assert_eq!(stats.factors_added, 0);
        // Still pruned: the refusal must not let the buffer grow without bound.
        assert_eq!(stats.buffer.buffered, 80);
    }

    #[test]
    fn test_an_under_covered_interval_is_refused() {
        let mut tracker = tracker_with_inertial();
        let t0_ns = EPOCH_NS + 1_000_000_000;
        let t1_ns = EPOCH_NS + 1_600_000_000;
        // Every other sample: 100 Hz. Each gap is 10 ms, inside the 15 ms limit, so only the
        // coverage count can catch it.
        let thinned: Vec<RawImuSample> = imu_samples(t0_ns, 200).into_iter().step_by(2).collect();
        tracker.push_raw_imu(thinned, 0, false).unwrap();

        tracker.on_keyframe_inertial(Some((7, t0_ns)), 11, t1_ns);
        assert!(tracker.map().imu_factors().is_empty());
        assert_eq!(tracker.inertial_stats().unwrap().refused_too_few, 1);
    }

    #[test]
    fn test_the_very_first_keyframe_has_no_edge_and_that_is_not_a_refusal() {
        let mut tracker = tracker_with_inertial();
        tracker
            .push_raw_imu(imu_samples(EPOCH_NS, 200), 0, false)
            .unwrap();
        // `prev_edge` is `None` on the bootstrap keyframe: there is no previous keyframe to
        // span from.
        tracker.on_keyframe_inertial(None, 0, EPOCH_NS + 500_000_000);
        let stats = tracker.inertial_stats().unwrap();
        assert_eq!(stats.factors_added, 0);
        assert_eq!(stats.refused_total(), 0);
    }

    /// The flag's promise, as the config comments state it: with `enable_inertial_ba` off the
    /// inertial path cannot move a published pose. An accepted initialization is the other
    /// way it could (`apply_initialization` rotates and scales the whole map and bumps the
    /// world generation, which goes on the wire as `reset_epoch`), so the gate has to hold
    /// regardless of what the solve returns.
    #[test]
    fn test_with_inertial_ba_off_an_accepted_initialization_is_counted_but_never_applied() {
        let mut tracker = tracker_with_inertial_ba(false);
        let centre = Vec3F64::new(0.5, -0.25, 2.0);
        tracker.map.upsert_keyframe(bare_keyframe(3, centre));
        tracker.state.pose_world_to_cam = tracker.map.keyframes()[0].frame.pose_world_to_cam;
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        let pose_before = pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam);
        let state_pose_before = pose_bits(&tracker.state.pose_world_to_cam);

        let mut inert = tracker.inertial.take().unwrap();
        tracker.accept_inertial_init(&mut inert, accepted_init(), 3, 2.5, InitStage::Initial);
        tracker.inertial = Some(inert);

        // Byte-identical, not approximately equal: a rotation by the ~5 degrees between the
        // synthetic gravity and +y would pass a loose tolerance.
        assert_eq!(
            pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam),
            pose_before,
            "the keyframe pose moved"
        );
        assert_eq!(
            pose_bits(&tracker.state.pose_world_to_cam),
            state_pose_before
        );
        assert_eq!(
            tracker.world_generation(),
            0,
            "reset_epoch would have gone on the wire"
        );
        assert!(!tracker.state.imu_initialized);
        assert!(tracker.state.imu_init_timestamp_sec.is_none());
        let inert = tracker.inertial.as_ref().unwrap();
        // Neither the linearization bias nor the gravity handed to inertial BA moved either.
        assert_eq!(
            (inert.bias.gyro, inert.bias.accel),
            (inert.cfg.initial_bias.gyro, inert.cfg.initial_bias.accel)
        );
        assert_eq!(
            inert.gravity_world,
            Vec3F64::new(0.0, 0.0, -GRAVITY_MAGNITUDE)
        );
        // ...but the acceptance is observable, which is the point of running dry at all.
        assert_eq!(inert.stats.init_accepted, 1);
        assert!(inert.dry_run_accepted);
        // And local BA stays on the visual path.
        let handles = tracker.on_keyframe_inertial(None, 4, EPOCH_NS + 3_000_000_000);
        assert!(handles.is_none());
    }

    /// Positive control for the test above: the same call with the flag ON does move
    /// everything the negative test asserts is still. Without this, that test would also pass
    /// against an `apply_initialization` that happened to be a no-op on a one-keyframe map.
    #[test]
    fn test_with_inertial_ba_on_an_accepted_initialization_rotates_the_map() {
        let mut tracker = tracker_with_inertial_ba(true);
        let centre = Vec3F64::new(0.5, -0.25, 2.0);
        tracker.map.upsert_keyframe(bare_keyframe(3, centre));
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        let pose_before = pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam);

        let mut inert = tracker.inertial.take().unwrap();
        tracker.accept_inertial_init(&mut inert, accepted_init(), 3, 2.5, InitStage::Initial);
        tracker.inertial = Some(inert);

        assert_ne!(
            pose_bits(&tracker.map.keyframes()[0].frame.pose_world_to_cam),
            pose_before,
            "the synthetic init must be one that moves a pose, or the dry-run test is vacuous"
        );
        assert_eq!(tracker.world_generation(), 1);
        assert!(tracker.state.imu_initialized);
        assert_eq!(tracker.state.imu_init_timestamp_sec, Some(2.5));
        let inert = tracker.inertial.as_ref().unwrap();
        let want = accepted_init().bias;
        assert_eq!((inert.bias.gyro, inert.bias.accel), (want.gyro, want.accel));
        assert_eq!(
            inert.gravity_world,
            Vec3F64::new(0.0, GRAVITY_MAGNITUDE, 0.0)
        );
        assert_eq!(inert.stats.init_accepted, 1);
        assert!(!inert.dry_run_accepted);
    }

    /// A dry acceptance ends the initial-solve loop. Otherwise, with `state.imu_initialized` never
    /// set, the ladder would re-solve every `init_retry` for the rest of the session over
    /// a window that never shrinks. The window here is built to make `ready` TRUE, so the
    /// early return is what is being measured — a false `ready` would hide a missing latch.
    #[test]
    fn test_a_dry_acceptance_stops_the_initializer_from_re_solving() {
        let mut tracker = tracker_with_inertial_ba(false);
        // Ten keyframes 0.25 s apart, 0.03 m apart: 2.25 s of IMU time over nine edges and
        // 0.27 m of displacement, past every `ready` gate (10 kfs / 1 s stereo, 2 s mono /
        // 0.05 m). The frames carry no stereo data, so the mono 2 s applies.
        const N: usize = 10;
        const STEP_NS: u64 = 250_000_000;
        let stamp = |i: usize| EPOCH_NS + 1_000_000_000 + i as u64 * STEP_NS;
        for i in 0..N {
            tracker
                .map
                .upsert_keyframe(bare_keyframe(i, Vec3F64::new(0.03 * i as f64, 0.0, 0.0)));
        }
        // Samples covering the whole span at 200 Hz, plus a margin past the last stamp.
        let n_samples = (N as u64 * STEP_NS / IMU_PERIOD_NS) as usize + 10;
        tracker
            .push_raw_imu(imu_samples(stamp(0), n_samples), 0, false)
            .unwrap();
        tracker.arm_inertial_window(0, 1.0, stamp(0));
        // Simulate the acceptance that already happened for this window.
        tracker.inertial.as_mut().unwrap().dry_run_accepted = true;

        for i in 1..N {
            let handles = tracker.on_keyframe_inertial(Some((i - 1, stamp(i - 1))), i, stamp(i));
            assert!(handles.is_none());
        }

        let inert = tracker.inertial.as_ref().unwrap();
        assert_eq!(inert.stats.factors_added, N as u64 - 1);
        assert!(
            inert.initializer.ready(&tracker.map, Some(0)),
            "the window must be ready, or the latch is not what stopped the solve"
        );
        assert_eq!(
            inert.stats.init_attempts, 0,
            "the ladder re-solved after a dry acceptance"
        );
        // The latch belongs to the window: re-arming (a re-bootstrap) clears it.
        tracker.arm_inertial_window(0, 1.0, stamp(0));
        assert!(!tracker.inertial.as_ref().unwrap().dry_run_accepted);
    }

    #[test]
    fn test_a_reset_drops_the_buffer_and_the_initialization_window() {
        let mut tracker = tracker_with_inertial();
        tracker
            .push_raw_imu(imu_samples(EPOCH_NS, 200), 0, false)
            .unwrap();
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        tracker.inertial.as_mut().unwrap().last_window_log = Some((0.0, false));
        assert!(tracker.inertial.as_ref().unwrap().start_kf_idx.is_some());

        tracker.reset();
        let inert = tracker.inertial.as_ref().unwrap();
        // `start_kf_idx` named a keyframe in the dropped map. `ImuInitializer::ready` would
        // keep counting against it and `try_initialize` would silently skip every factor whose
        // endpoints it could not find — a solve over an empty problem, reported as a rejection.
        assert!(inert.start_kf_idx.is_none());
        assert!(inert.last_window_log.is_none());
        assert_eq!(inert.buffer.stats().buffered, 0);
        assert!(tracker.last_keyframe_stamp_ns.is_none());
    }

    #[test]
    fn test_pose_convention_is_world_to_cam() {
        // The rotation MUST be non-identity. With `R = I` the correct inverse
        // translation `-R^T t` and the classic wrong one `-t` are numerically
        // identical, so an identity-rotation version of this test passes
        // against exactly the bug it is named after.
        //
        // 90 deg yaw about y, column-major: a camera whose centre is at world
        // (2, 0, 3), so both the rotation and the translation are load-bearing.
        let r_wc = Mat3F64::from_cols(
            Vec3F64::new(0.0, 0.0, -1.0),
            Vec3F64::new(0.0, 1.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
        );
        let centre = Vec3F64::new(2.0, 0.0, 3.0);
        let world_to_cam = Pose3d::new(r_wc, -(r_wc * centre));
        let tracked = TrackedPose {
            stamp: CuTime::from_nanos(0),
            pose_world_to_cam: world_to_cam,
            keyframe: false,
        };
        let back = tracked.cam_in_world();
        assert!(
            (back.translation - centre).length() < 1e-12,
            "cam_in_world gave {:?}, want {centre:?}",
            back.translation
        );
        // Negative control: negating the translation without applying R^T is
        // the bug, and it must give a different answer here — otherwise the
        // assertion above is satisfied for the wrong reason.
        assert!(
            (-world_to_cam.translation - centre).length() > 1.0,
            "pick a pose where -t and -R^T t actually differ"
        );
        // And the rotation half round-trips: R_cw * R_wc = I.
        let round_trip = back.rotation * world_to_cam.rotation;
        assert!(
            (round_trip - Mat3F64::IDENTITY)
                .to_cols_array()
                .iter()
                .all(|v| v.abs() < 1e-12)
        );
    }

    /// On a parked robot the excitation gate holds `ready=false` forever, which is CORRECT, so
    /// "never ready" is the normal state and the throttle has to survive it without ever
    /// swallowing the transition out of it.
    #[test]
    fn test_the_init_window_line_is_throttled_but_never_loses_the_transition() {
        const RETRY: f64 = 5.0;
        let due = |last: Option<(f64, bool)>, now_sec: f64, ready: bool| {
            should_log_throttled(last, now_sec, ready, RETRY)
        };

        // First sighting always speaks.
        assert!(due(None, 100.0, false));

        // Same verdict inside the interval stays quiet; at the interval it speaks again.
        assert!(!due(Some((100.0, false)), 100.5, false));
        assert!(!due(Some((100.0, false)), 104.9, false));
        assert!(due(Some((100.0, false)), 105.0, false));

        // A FLIPPED verdict speaks immediately, however recently we spoke. This is the case a
        // plain time throttle would swallow, and it is the only line anyone actually waits for.
        assert!(due(Some((100.0, false)), 100.01, true));
        assert!(due(Some((100.0, true)), 100.01, false));
    }

    /// The truth table above is the whole of `should_log_throttled`, so the loss caller needs no
    /// second copy of it — only that its interval is a usable one.
    #[test]
    fn test_the_loss_throttle_interval_is_positive() {
        const { assert!(LOSS_LOG_RETRY_SEC > 0.0) };
    }

    /// A requested reset drops the map, so a loss logged against the OLD map must not silence the
    /// first loss of the new one — and the suppressed tally must not carry across, or it would
    /// report losses of a map that no longer exists.
    #[test]
    fn test_a_reset_clears_the_loss_throttle_and_its_tally() {
        let mut tracker = tracker_with_inertial();
        tracker.last_loss_log = Some((10.0, true));
        tracker.losses_suppressed = 17;

        tracker.reset();

        assert!(tracker.last_loss_log.is_none());
        assert_eq!(tracker.losses_suppressed, 0);
    }

    /// A re-bootstrap re-arms the window WITHOUT an `InertialState::reset` — a tracking loss with
    /// `reset_map_on_loss` off, the default, takes exactly that path. The throttle state has to go
    /// with the window, or the first description of the new one is silenced by its predecessor.
    #[test]
    fn test_re_arming_the_window_clears_the_line_throttle() {
        let mut tracker = tracker_with_inertial();
        tracker.arm_inertial_window(3, 0.0, EPOCH_NS);
        tracker.inertial.as_mut().unwrap().last_window_log = Some((10.0, false));

        tracker.arm_inertial_window(7, 1.0, EPOCH_NS);
        assert!(tracker.inertial.as_ref().unwrap().last_window_log.is_none());
    }
}
