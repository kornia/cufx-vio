//! Inertial ingest for [`crate::track::Tracker`]: the sample buffer, the coverage gates it
//! refuses on, and the configuration the inertial path cannot run without.
//!
//! # Why this exists as a buffer and not as a per-frame delta
//!
//! `kornia_slam::map::Map::add_imu_factor` wants an edge spanning
//! `[previous_keyframe, this_keyframe]` — 0.6-1.0 s measured on a Jetson Orin with an OAK-D at
//! 640x400, ~120-200 samples at a measured 199.7 Hz — and it keeps the raw samples so the factor
//! can be re-integrated when the bias estimate moves. Neither a per-frame preintegration nor a
//! single driver batch can answer that query: a 256-sample batch is 1.28 s at 200 Hz, which a
//! p99 keyframe stall exceeds. So samples are accumulated here, stamp-indexed, and queried by
//! interval.
//!
//! # Everything here fails silently upstream, which is why it is all a `Result`
//!
//! `kornia_sensors::imu::PreintegratedImu::from_measurements` is defensive in the worst way: it
//! filters to `[t0, t1]`, sorts, and zero-order-holds each surviving sample across the gap to
//! the next, then extrapolates the last one out to `t1`. A window that is 10 % populated
//! therefore returns a fully-formed delta at the wrong magnitude and the full `dt`, and an
//! EMPTY window returns `ΔR = I, Δv = 0, Δp = 0, dt = 0` with an all-zero covariance — which
//! `vi_ba_schur`'s information matrix turns into a 1e6 diagonal fallback, i.e. a maximally
//! confident "these two keyframes did not move" constraint. Nothing in that chain returns an
//! error or logs. `ImuBuffer::window` is the place that does.
//!
//! # Known limit: `Map::imu_factors` is push-only
//!
//! Nothing here bounds the factors once they are in the map. `Map::add_imu_factor` is a bare
//! `push`, each factor retains its raw samples for repropagation (~8.7 kB per keyframe edge at
//! 200 Hz and 1.7 keyframes/s, so ~53 MB/hour), and the only thing that ever frees them is
//! `Map::clear_active()` — i.e. a full map reset. With `reset_map_on_loss` enabled that reset
//! happens on every tracking loss, which is then the only bound on keyframe growth too; without
//! it, an IMU-enabled session grows without limit. There is no API in kornia-slam to
//! drop or marginalize a single factor, and the unmerged keyframe-culling work upstream becomes
//! a no-op the moment factors chain consecutive keyframes (its pinned set is then every
//! keyframe). Enabling the inertial path for a long run needs that answered upstream first.
//! The sample ring below is bounded; the factors it feeds are not.

use std::collections::VecDeque;

use cu29::prelude::{CuDuration, CuTime};
use cu29::units::si::acceleration::meter_per_second_squared;
use cu29::units::si::angular_velocity::radian_per_second;
use cu29::units::si::f32::{Acceleration, AngularVelocity};
use cu29::units::si::f64::{Frequency, Ratio, Time};
use cu29::units::si::frequency::hertz;
use cu29::units::si::ratio::ratio;
use cu29::units::si::time::second;
use kornia_3d::pose::Pose3d;
use kornia_algebra::{Mat3F64, Vec3F64};
use kornia_sensors::imu::{ImuBias, ImuCalib, ImuMeasurement};
use kornia_slam::estimation::ImuInitConfig;

use crate::error::VioError;

/// One raw IMU sample, in the units and the clock the driver publishes.
///
/// Deliberately NOT [`kornia_sensors::imu::ImuMeasurement`], which carries `f64` seconds: the
/// tracker rebases every stamp against the epoch of its own first frame (an absolute ~1.7e18 ns
/// epoch leaves an `f64` about 400 ns of resolution), and that epoch is not known until the
/// first frame arrives — which can be after the first samples. Converting at the ingest
/// boundary would fix a second epoch here and hand `from_measurements` a `[t0, t1]` window
/// silently offset from the frames it is supposed to span, whereupon its own filter drops every
/// sample and the edge becomes the zero-delta case above.
///
/// Crate-private: the public edge takes the unit-typed [`cu_stereo_payloads::ImuSample`] and
/// converts once, on ingest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RawImuSample {
    /// Capture stamp in nanoseconds, on the SAME timeline as the image stamps handed to
    /// [`crate::track::Tracker::process_stereo`]. On the OAK-D path both come from depthai's
    /// host `steady_clock` stamp shifted by one process-wide epoch offset, so no alignment step
    /// is needed — but that is a property of the source, not something checked here.
    pub(crate) stamp_ns: u64,
    /// Angular velocity, rad/s, in the IMU's own axes.
    pub(crate) gyro: [f64; 3],
    /// Linear acceleration, m/s², in the IMU's own axes.
    pub(crate) accel: [f64; 3],
}

impl RawImuSample {
    /// Converts to the type `kornia_sensors` integrates, rebasing the stamp onto `epoch_ns`.
    ///
    /// `epoch_ns` must be the tracker's frame epoch, not a fresh one; see the type doc.
    pub(crate) fn to_measurement(self, epoch_ns: u64) -> ImuMeasurement {
        ImuMeasurement {
            timestamp: ts_sec(self.stamp_ns, epoch_ns),
            gyro: Vec3F64::new(self.gyro[0], self.gyro[1], self.gyro[2]),
            accel: Vec3F64::new(self.accel[0], self.accel[1], self.accel[2]),
        }
    }
}

impl From<cu_stereo_payloads::ImuSample> for RawImuSample {
    /// Unpacks the unit-typed payload into the tracker's plain-`f64` sample: the stamp is the
    /// sample's own `tov` (the same clock as the images'), the units are rad/s and m/s^2. A
    /// `From` so a producer on the IMU-rate path can hand a batch straight through
    /// [`ImuQueue::push`](crate::ImuQueue::push) with no intermediate `Vec`.
    fn from(s: cu_stereo_payloads::ImuSample) -> Self {
        let a = |q: Acceleration| f64::from(q.get::<meter_per_second_squared>());
        let w = |q: AngularVelocity| f64::from(q.get::<radian_per_second>());
        Self {
            stamp_ns: s.tov.as_nanos(),
            gyro: [w(s.imu.gyro_x), w(s.imu.gyro_y), w(s.imu.gyro_z)],
            accel: [a(s.imu.accel_x), a(s.imu.accel_y), a(s.imu.accel_z)],
        }
    }
}

/// Samples held by a ring — `ImuBuffer` and [`crate::ImuQueue`] alike.
///
/// 2048 = 10.2 s at 200 Hz, ~115 kB. Sized against the LONGEST interval that can be asked for,
/// not the frame interval: a keyframe edge spans back to the previous keyframe, keyframes can be
/// seconds apart when the scene barely changes, and the channel has to hold what arrives while
/// the tracker is mid-solve and not draining. One constant for both rings so they cannot drift
/// apart in size with nothing noticing.
pub const DEFAULT_BUFFER_CAPACITY: usize = 2048;

/// Appends to a capacity-bounded ring, evicting the oldest sample and counting the eviction.
///
/// Shared by [`ImuBuffer::push`] and [`ImuQueue::push`](crate::ImuQueue::push), which are the same ring
/// discipline on two sides of a lock.
pub(crate) fn push_bounded(
    ring: &mut VecDeque<RawImuSample>,
    sample: RawImuSample,
    capacity: usize,
    evicted: &mut u64,
) {
    ring.push_back(sample);
    while ring.len() > capacity {
        ring.pop_front();
        *evicted += 1;
    }
}

/// Nanoseconds since `epoch_ns`, in seconds.
///
/// The one conversion in this crate, shared by frames and samples so that the two can never
/// drift onto different clocks. `saturating_sub` matches `Tracker::process_frame`: a sample
/// stamped before the first frame collapses to 0 rather than wrapping to 5.8e11 s.
pub(crate) fn ts_sec(stamp_ns: u64, epoch_ns: u64) -> f64 {
    stamp_ns.saturating_sub(epoch_ns) as f64 * 1e-9
}

/// Coverage thresholds `ImuBuffer::window` enforces before an interval may be integrated.
///
/// All four exist because the corresponding failure is invisible downstream; see the module
/// docs. They are algorithm thresholds, not physical constants of the platform — except
/// `nominal_rate`, which is a property of the device and therefore has no default.
#[derive(Debug, Clone, Copy)]
pub struct ImuGates {
    /// The sample rate the device is configured for. Only used as the denominator of
    /// the coverage count, so setting it too LOW silently disables that gate; there is no
    /// default for exactly that reason.
    pub nominal_rate: Frequency,
    /// Largest tolerated gap between consecutive samples inside the interval — and between
    /// `t0` and the first sample, and between the last sample and `t1`. A gap wider than this
    /// is held at a constant angular velocity and acceleration by `from_measurements`, which is
    /// a plausible-looking delta at the wrong magnitude rather than a detectable error.
    pub max_sample_gap: Time,
    /// Minimum fraction of `nominal_rate * (t1 - t0)` samples that must be present. Catches
    /// uniform decimation, which the per-gap check above misses when the holes are small.
    pub min_coverage_ratio: Ratio,
    /// Absolute floor on the sample count, independent of the interval length. Two samples
    /// integrate to almost nothing but still produce a positive `dt`.
    pub min_samples: usize,
}

impl ImuGates {
    /// Gates for a device running at `nominal_rate`.
    ///
    /// `max_sample_gap` defaults to three nominal periods (15 ms at 200 Hz) — the same
    /// order as the driver's own report-hole granularity, and well under the 66 ms frame
    /// interval, so a single missed report is tolerated and a run of them is not.
    /// `min_coverage_ratio` 0.9 leaves room for the interval boundaries landing mid-period
    /// without leaving room for a real dropout.
    pub fn new(nominal_rate: Frequency) -> Self {
        Self {
            nominal_rate,
            max_sample_gap: Time::new::<second>(3.0 / nominal_rate.get::<hertz>()),
            min_coverage_ratio: Ratio::new::<ratio>(0.9),
            min_samples: 4,
        }
    }
}

/// Why an interval was refused. Every variant is a case that would otherwise reach
/// `PreintegratedImu::from_measurements` and come back as a confident wrong answer.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum ImuWindowError {
    /// `t1 <= t0`. Two keyframes at the same stamp, or a stamp that went backwards.
    #[error("non-positive IMU interval: t0={t0} t1={t1}")]
    NonPositiveInterval {
        /// Start of the requested interval.
        t0: CuTime,
        /// End of the requested interval.
        t1: CuTime,
    },

    /// The source reported dropped samples inside this interval.
    ///
    /// A short integration is a WRONG delta, not a missing one, and nothing downstream can tell
    /// the difference — so the edge is refused outright. A missing IMU factor costs one visual-
    /// only keyframe pair; a wrong one poisons the bias estimate, which is shared across the
    /// whole window by the random-walk residuals.
    #[error(
        "IMU source reported dropped samples at {at}, inside [{t0}, {t1}]: \
         a decimated window integrates to a full-dt delta at the wrong magnitude"
    )]
    DroppedSamples {
        /// Stamp at which the drop was observed.
        at: CuTime,
        /// Start of the requested interval.
        t0: CuTime,
        /// End of the requested interval.
        t1: CuTime,
    },

    /// No buffered sample falls in `[t0, t1]` at all.
    #[error("no buffered IMU samples in [{t0}, {t1}] ({buffered} buffered)")]
    NoSamples {
        /// Start of the requested interval.
        t0: CuTime,
        /// End of the requested interval.
        t1: CuTime,
        /// How many samples the buffer holds in total.
        buffered: usize,
    },

    /// A gap wider than [`ImuGates::max_sample_gap`], at either boundary or inside.
    #[error("IMU gap of {gap} at {at} exceeds the {max_gap} limit")]
    Gap {
        /// Width of the offending gap.
        gap: CuDuration,
        /// Where it starts.
        at: CuTime,
        /// The configured limit.
        max_gap: CuDuration,
    },

    /// Enough samples to look integrable, too few to be one.
    #[error(
        "only {got} IMU samples over {span}, need {want} \
         ({:.1} Hz nominal x {} coverage)",
        .rate.get::<hertz>(),
        .coverage.get::<ratio>()
    )]
    TooFewSamples {
        /// Samples actually present in the interval.
        got: usize,
        /// Samples required.
        want: usize,
        /// Length of the interval.
        span: CuDuration,
        /// The nominal rate the requirement was derived from.
        rate: Frequency,
        /// The configured coverage ratio.
        coverage: Ratio,
    },
}

/// Counters for everything the buffer silently absorbed or loudly refused.
///
/// A rising `out_of_order` or `evicted` is the only observable signal of an upstream fault:
/// neither is reported by the driver's own `dropped` counter, and `from_measurements` sorts
/// out-of-order input rather than complaining about it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ImuBufferStats {
    /// Samples currently held.
    pub buffered: usize,
    /// Samples accepted over the buffer's lifetime.
    pub accepted: u64,
    /// Samples rejected at ingest because their stamp did not advance. Order is *assumed*
    /// everywhere in the driver chain and *verified* nowhere, so it is verified here.
    pub out_of_order: u64,
    /// Samples pushed out by the capacity bound before anything read them.
    pub evicted: u64,
    /// Times the source's cumulative drop counter increased.
    pub drop_reports: u64,
}

/// A stamp-indexed ring of raw IMU samples, queried by interval.
///
/// Capacity-bounded on purpose: `kornia_slam`'s own `pending_imu` is pruned only when a
/// keyframe is inserted, so a sustained tracking loss grows it at 200 Hz x 56 B = 11 kB/s with
/// no ceiling. Here an overrun evicts the oldest sample and increments
/// [`ImuBufferStats::evicted`]; the interval that needed it is then refused by the leading-gap
/// check rather than integrated from a truncated window.
#[derive(Debug)]
pub(crate) struct ImuBuffer {
    samples: VecDeque<RawImuSample>,
    capacity: usize,
    /// Stamps at which the source's cumulative drop counter was seen to increase. A window
    /// containing one of these is refused. Pruned alongside `samples`, so a drop during
    /// start-up ages out instead of poisoning every later interval.
    drop_marks: VecDeque<u64>,
    /// Last cumulative value seen. `None` until the first push: the first observation
    /// establishes the baseline rather than reporting the whole backlog as a fresh drop.
    last_dropped_cumulative: Option<u64>,
    /// A drop was reported by a batch that carried no samples, so it has no stamp yet; it is
    /// attached to the next sample pushed.
    pending_drop_mark: bool,
    stats: ImuBufferStats,
}

impl ImuBuffer {
    /// A buffer holding at most `capacity` samples; see [`DEFAULT_BUFFER_CAPACITY`] for how to
    /// size it.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity.min(4096)),
            capacity: capacity.max(1),
            drop_marks: VecDeque::new(),
            last_dropped_cumulative: None,
            pending_drop_mark: false,
            stats: ImuBufferStats::default(),
        }
    }

    /// Appends a batch, taking the source's CUMULATIVE drop count with it.
    ///
    /// Cumulative, not per-batch: the source batch's `dropped` counter is never reset, so a
    /// consumer that reads it as a per-batch figure re-reports every historical drop on every
    /// batch. Only the increase is a new hole.
    ///
    /// Returns how many samples were accepted. Samples whose stamp does not strictly advance
    /// are dropped and counted: the whole driver path preserves order but nothing asserts it,
    /// and `from_measurements` would silently sort them into a plausible answer.
    ///
    /// Takes an iterator, not a slice, so the samples can flow from the producer's payload (or
    /// the channel's deque) straight into this ring with no intermediate `Vec` — each one is
    /// copied exactly once, here.
    pub fn push(
        &mut self,
        batch: impl IntoIterator<Item = RawImuSample>,
        dropped_cumulative: u64,
    ) -> usize {
        let mut batch = batch.into_iter().peekable();
        let grew = match self.last_dropped_cumulative {
            Some(prev) => dropped_cumulative > prev,
            // First observation: adopt the baseline. Reporting it as a drop would refuse the
            // first keyframe interval for holes that predate the tracker.
            None => false,
        };
        self.last_dropped_cumulative = Some(dropped_cumulative);
        if grew {
            self.stats.drop_reports += 1;
            match batch.peek() {
                // The hole sits between the previously buffered sample and this batch, so the
                // batch's own first stamp is the earliest interval boundary it can invalidate.
                Some(first) => self.drop_marks.push_back(first.stamp_ns),
                None => self.pending_drop_mark = true,
            }
        }

        let mut accepted = 0;
        for sample in batch {
            if let Some(last) = self.samples.back()
                && sample.stamp_ns <= last.stamp_ns
            {
                self.stats.out_of_order += 1;
                continue;
            }
            if self.pending_drop_mark {
                self.drop_marks.push_back(sample.stamp_ns);
                self.pending_drop_mark = false;
            }
            push_bounded(
                &mut self.samples,
                sample,
                self.capacity,
                &mut self.stats.evicted,
            );
            accepted += 1;
        }
        self.stats.accepted += accepted as u64;
        // A mark only matters while some interval could still straddle it, and the earliest
        // interval start the ring can serve is its own oldest sample. Without this, a burst of
        // drops during start-up would accumulate marks forever.
        if let Some(oldest) = self.samples.front().map(|s| s.stamp_ns) {
            self.forget_marks_before(oldest);
        }
        accepted
    }

    /// Forgets drop marks stamped strictly before `stamp_ns` — the ones no interval the ring can
    /// still serve could straddle.
    fn forget_marks_before(&mut self, stamp_ns: u64) {
        self.drop_marks.retain(|m| *m >= stamp_ns);
    }

    /// Drops everything stamped strictly before `stamp_ns`.
    ///
    /// Called with the keyframe stamp after every insertion, successful or refused: the next
    /// interval starts at exactly this stamp, so anything earlier can never be asked for again.
    /// This — not the capacity bound — is what keeps the buffer at one keyframe interval in
    /// steady state.
    pub fn prune_before(&mut self, stamp_ns: u64) {
        while self.samples.front().is_some_and(|s| s.stamp_ns < stamp_ns) {
            self.samples.pop_front();
        }
        self.forget_marks_before(stamp_ns);
    }

    /// Empties the buffer and every drop mark, keeping the lifetime counters.
    pub fn clear(&mut self) {
        self.samples.clear();
        self.drop_marks.clear();
        self.pending_drop_mark = false;
    }

    /// Counters, plus the current occupancy.
    pub fn stats(&self) -> ImuBufferStats {
        ImuBufferStats {
            buffered: self.samples.len(),
            ..self.stats
        }
    }

    /// The samples covering `[t0_ns, t1_ns]`, or the reason the interval must not be integrated.
    ///
    /// Checks run cheapest-and-most-damning first, so the reported reason is the root cause
    /// rather than a downstream symptom: a reported drop explains the gap it caused, and there
    /// is no point measuring coverage of an interval the buffer never held.
    ///
    /// Returns a borrowed iterator over the ring rather than a `Vec`: the only owned copy the
    /// caller needs is the `ImuMeasurement` vector `Map::add_imu_factor` takes, so the samples
    /// are converted once, straight out of the ring. The window's bounds are two binary searches
    /// — the ring is stamp-sorted by construction, which is exactly the invariant [`Self::push`]
    /// pays for by refusing every stamp that does not strictly advance.
    pub fn window(
        &self,
        t0_ns: u64,
        t1_ns: u64,
        gates: &ImuGates,
    ) -> Result<std::collections::vec_deque::Iter<'_, RawImuSample>, ImuWindowError> {
        if t1_ns <= t0_ns {
            return Err(ImuWindowError::NonPositiveInterval {
                t0: CuTime::from_nanos(t0_ns),
                t1: CuTime::from_nanos(t1_ns),
            });
        }
        if let Some(&at_ns) = self
            .drop_marks
            .iter()
            .find(|m| **m >= t0_ns && **m <= t1_ns)
        {
            return Err(ImuWindowError::DroppedSamples {
                at: CuTime::from_nanos(at_ns),
                t0: CuTime::from_nanos(t0_ns),
                t1: CuTime::from_nanos(t1_ns),
            });
        }

        let lo = self.samples.partition_point(|s| s.stamp_ns < t0_ns);
        let hi = self.samples.partition_point(|s| s.stamp_ns <= t1_ns);
        if lo >= hi {
            return Err(ImuWindowError::NoSamples {
                t0: CuTime::from_nanos(t0_ns),
                t1: CuTime::from_nanos(t1_ns),
                buffered: self.samples.len(),
            });
        }
        let (first, last) = (&self.samples[lo], &self.samples[hi - 1]);
        let in_range = self.samples.range(lo..hi);

        // Nanoseconds, so the gate is exact integer arithmetic rather than a float comparison
        // against a stamp difference.
        let max_gap_ns = (gates.max_sample_gap.get::<second>() * 1e9).max(0.0) as u64;

        // Leading edge. This is also the eviction detector: if the capacity bound threw away
        // the start of the interval, the surviving first sample is far from `t0`.
        let lead = first.stamp_ns - t0_ns;
        if lead > max_gap_ns {
            return Err(ImuWindowError::Gap {
                gap: CuDuration::from_nanos(lead),
                at: CuTime::from_nanos(t0_ns),
                max_gap: CuDuration::from_nanos(max_gap_ns),
            });
        }
        // Trailing edge. `from_measurements` extrapolates the LAST sample all the way out to
        // `t1`, so an interval that ends in a hole is the most damaging shape of all: the
        // extrapolated stretch carries full weight.
        let tail = t1_ns - last.stamp_ns;
        if tail > max_gap_ns {
            return Err(ImuWindowError::Gap {
                gap: CuDuration::from_nanos(tail),
                at: CuTime::from_nanos(last.stamp_ns),
                max_gap: CuDuration::from_nanos(max_gap_ns),
            });
        }
        for (a, b) in in_range.clone().zip(in_range.clone().skip(1)) {
            let gap = b.stamp_ns - a.stamp_ns;
            if gap > max_gap_ns {
                return Err(ImuWindowError::Gap {
                    gap: CuDuration::from_nanos(gap),
                    at: CuTime::from_nanos(a.stamp_ns),
                    max_gap: CuDuration::from_nanos(max_gap_ns),
                });
            }
        }

        let span_sec = (t1_ns - t0_ns) as f64 * 1e-9;
        let coverage = gates.min_coverage_ratio.get::<ratio>();
        let want = ((gates.nominal_rate.get::<hertz>() * span_sec * coverage).ceil() as usize)
            .max(gates.min_samples);
        if in_range.len() < want {
            return Err(ImuWindowError::TooFewSamples {
                got: in_range.len(),
                want,
                span: CuDuration::from_nanos(t1_ns - t0_ns),
                rate: gates.nominal_rate,
                coverage: gates.min_coverage_ratio,
            });
        }

        Ok(in_range)
    }
}

/// Everything the inertial path needs and cannot invent.
///
/// There is no `Default`, and [`crate::track::TrackerConfig`] holds this as an `Option` that is
/// `None` unless the caller supplies every field: `imu_t_bc` and the four
/// [`ImuCalib`] densities are physical facts of a particular mounting and a particular chip, and a
/// guessed value for either is silently wrong rather than visibly broken.
///
/// * A wrong `imu_t_bc` rotation makes `ImuInitializer` solve for gravity in the wrong frame,
///   and `apply_initialization` then ROTATES THE WHOLE MAP to align that wrong gravity with
///   world +y. The trajectory stays self-consistent and tips over.
/// * Wrong `calib` densities set the IMU information matrix directly, i.e. the visual/inertial
///   weighting. `kornia_slam::pipeline` hard-codes EuRoC ADIS16448 values; on a BNO086 those
///   are not a starting point, they are a different sensor.
#[derive(Debug, Clone)]
pub struct InertialConfig {
    /// Camera-to-IMU extrinsic, in `kornia_slam`'s convention: `X_body = T_BC * X_cam`, where
    /// "cam" is the frame the tracker's poses live in — the RECTIFIED left camera.
    ///
    /// If it is ever derived from a raw (unrectified) camera-to-IMU extrinsic, the rectifying
    /// rotation has to be folded in the way `kornia-slam-app`'s EuRoC source does
    /// (`T_B,rect = T_BS * R_rect^T`). If instead it is measured against the tracker's own
    /// published pose, that correction is ALREADY in it and applying it again rectifies twice.
    pub imu_t_bc: Pose3d,
    /// Continuous-time noise densities, datasheet convention. See the type doc.
    pub calib: ImuCalib,
    /// Linearization point for the first edges, before the initializer estimates one.
    /// Zero is a legitimate value here (a stationary gyro bias magnitude of 0.0031 rad/s was
    /// measured on a BNO086), unlike the extrinsic and the densities.
    pub initial_bias: ImuBias,
    /// Coverage gates; see [`ImuGates`].
    pub gates: ImuGates,
    /// Sample-ring capacity; see `ImuBuffer::with_capacity`.
    pub buffer_capacity: usize,
    /// Readiness thresholds for `ImuInitializer`. Defaults to `kornia_slam::pipeline`'s
    /// (10 keyframes / 1.0 s of integrated IMU time / 0.05 m of displacement), which are the
    /// readiness gate for the first inertial-only initialization in Campos et al., ORB-SLAM3,
    /// IEEE T-RO 2021.
    pub init: ImuInitConfig,
    /// Minimum wall time between initialization attempts. `try_initialize` runs a 200-iteration
    /// LM over the whole window on every call and the window never shrinks, so an ungated retry
    /// on every keyframe is a growing per-keyframe cost for a solve that just failed.
    pub init_retry: Time,
    /// Whether to switch local BA to `run_local_inertial_ba` once initialization succeeds.
    /// Turning this off keeps the factors and the initializer (so gravity, velocities and bias
    /// are estimated and observable) while leaving the optimizer on the visual-only path — the
    /// setting to run first on hardware, because it cannot move a pose.
    pub enable_inertial_ba: bool,
}

impl InertialConfig {
    /// Builds a config from the three things that must be measured: the extrinsic, the noise
    /// densities, and the device's sample rate.
    ///
    /// Everything else is an algorithm threshold and gets `kornia_slam::pipeline`'s value.
    ///
    /// This is the ONLY constructor, and it is fallible, so that the checks below cannot be
    /// skipped by building the struct literally: every one of them guards a mistake whose
    /// symptom is a plausible map rather than an error. It cannot check the extrinsic is the
    /// RIGHT rotation — nothing can, short of the hand-eye calibration that produces it — only
    /// that it is a rotation at all.
    pub fn new(
        imu_t_bc: Pose3d,
        calib: ImuCalib,
        nominal_rate: Frequency,
    ) -> Result<Self, VioError> {
        validate_rotation(&imu_t_bc.rotation)?;
        for (field, value) in [
            ("imu_t_bc.translation.x", imu_t_bc.translation.x),
            ("imu_t_bc.translation.y", imu_t_bc.translation.y),
            ("imu_t_bc.translation.z", imu_t_bc.translation.z),
        ] {
            if !value.is_finite() {
                return Err(VioError::InertialParamInvalid { field, value });
            }
        }
        for (field, value) in [
            ("gyro_noise", calib.gyro_noise),
            ("accel_noise", calib.accel_noise),
            ("gyro_bias_noise", calib.gyro_bias_noise),
            ("accel_bias_noise", calib.accel_bias_noise),
            ("nominal_rate_hz", nominal_rate.get::<hertz>()),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(VioError::InertialParamInvalid { field, value });
            }
        }
        Ok(Self {
            imu_t_bc,
            calib,
            initial_bias: ImuBias::default(),
            gates: ImuGates::new(nominal_rate),
            buffer_capacity: DEFAULT_BUFFER_CAPACITY,
            init: ImuInitConfig {
                min_keyframes: 10,
                min_time_sec: 1.0,
                min_motion: 0.05,
            },
            init_retry: Time::new::<second>(5.0),
            enable_inertial_ba: true,
        })
    }
}

/// Rejects a matrix that is not a proper rotation.
///
/// Tolerance 1e-6 on both tests: a hand-entered rotation written to six decimals lands around
/// 1e-6, and anything looser stops distinguishing a typo from rounding. The determinant test is
/// the one that matters — a reflection (det = -1) mirrors the body frame, and a mirrored
/// gravity direction is what `apply_initialization` would then rotate the whole map onto.
fn validate_rotation(r: &Mat3F64) -> Result<(), VioError> {
    let residual = r.transpose() * *r - Mat3F64::IDENTITY;
    let orthonormality_error = residual
        .to_cols_array()
        .iter()
        .fold(0.0f64, |acc, v| acc.max(v.abs()));
    let determinant = r.determinant();
    if !orthonormality_error.is_finite()
        || orthonormality_error > 1e-6
        || (determinant - 1.0).abs() > 1e-6
    {
        return Err(VioError::InertialExtrinsicNotRigid {
            orthonormality_error,
            determinant,
        });
    }
    Ok(())
}

/// What the inertial path has done and refused. Diagnostics only.
///
/// The refusal counters are the point: an interval that is refused costs one visual-only
/// keyframe pair and leaves no trace anywhere else, so a rising refusal rate is the only
/// observable symptom of an IMU stream that has quietly stopped covering its intervals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InertialStats {
    /// Sample-buffer counters.
    pub buffer: ImuBufferStats,
    /// IMU factors handed to `Map::add_imu_factor`.
    pub factors_added: u64,
    /// Intervals refused because the source reported dropped samples.
    pub refused_dropped: u64,
    /// Intervals refused for a gap at a boundary or inside.
    pub refused_gap: u64,
    /// Intervals refused for insufficient sample coverage.
    pub refused_too_few: u64,
    /// Intervals refused because the buffer held nothing in range.
    pub refused_no_samples: u64,
    /// Intervals refused because `t1 <= t0`.
    pub refused_bad_interval: u64,
    /// Intervals where every gate passed but the integration still came back with `dt == 0`.
    /// Should be unreachable behind the gates; counted because `add_imu_factor` does not check
    /// it and `vi_ba_schur` answers a singular covariance with a 1e6-diagonal information
    /// matrix rather than an error.
    pub refused_zero_dt: u64,
    /// Calls to `ImuInitializer::try_initialize`.
    pub init_attempts: u64,
    /// Accepted initializations, including the two later refinement passes (Campos et al.,
    /// ORB-SLAM3, IEEE T-RO 2021).
    pub init_accepted: u64,
    /// Whether visual-inertial initialization has succeeded.
    pub initialized: bool,
    /// Whether the first inertial refinement pass (after ~5 s of initialized tracking) has run.
    pub first_refinement_done: bool,
    /// Whether the second inertial refinement pass (after ~15 s of initialized tracking) has run.
    pub second_refinement_done: bool,
    /// IMU factors the map currently holds. `Map::imu_factors` is push-only (see the module
    /// docs), so on a graph without `reset_map_on_loss` this only ever grows; it is here so the
    /// growth is visible in the stop line rather than only in RSS.
    pub retained_factors: usize,
    /// Raw samples retained across those factors, 56 B each — the memory the factors pin.
    pub retained_samples: usize,
}

impl InertialStats {
    /// Every refused interval, whatever the reason. One sum, so a seventh refusal counter cannot
    /// be added without this total seeing it.
    pub fn refused_total(&self) -> u64 {
        self.refused_dropped
            + self.refused_gap
            + self.refused_too_few
            + self.refused_no_samples
            + self.refused_bad_interval
            + self.refused_zero_dt
    }
}

/// How much the IMU itself was excited over an initialization window. OBSERVATION ONLY — nothing
/// here gates anything yet.
///
/// Measured over 47 live runs on a Jetson Orin with an OAK-D at 640x400: 461 of 474 initializations
/// were rejected for "implausible accel bias", 168 byte-identical at `ba = (-0.204, -0.087,
/// -7.640)` — 78% of gravity in the bias, which is what an unexcited window does: nothing
/// separates gravity from bias, so the solve puts the specific force in the bias.
/// `ImuInitializer::ready` cannot see that — its displacement gate reads the VISUAL pose.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct WindowExcitation {
    /// Integrated gyro angle, radians: `sum ||omega|| * dt` over consecutive raw samples.
    /// Unsigned, so a rotation and its reversal add rather than cancel — this is how much the
    /// gyro saw, not net attitude change.
    pub rot_rad: f64,
    /// Population variance of accelerometer MAGNITUDE about its window mean, (m/s^2)^2.
    ///
    /// Magnitude, not per-axis: a parked IMU reads a near-constant ~9.81 whatever its attitude,
    /// and pure rotation moves gravity between axes without changing its length. So this is ~0
    /// for a parked AND for a purely-rotating IMU; only linear acceleration puts signal in it.
    pub accel_var: f64,
    /// Raw samples both statistics were computed over.
    pub samples: usize,
}

/// Computes [`WindowExcitation`] over the raw samples of an initialization window, one slice per
/// `ImuFactor` (`ImuFactor::raw_samples`, retained for the life of the factor).
///
/// Streaming (Welford) rather than collecting the magnitudes: a 47 s window pins ~118k samples,
/// and the caller runs this inside the tracker once per KEYFRAME until the window first goes
/// ready (`Tracker::log_init_window` says why it is not throttled), so it must allocate nothing.
pub(crate) fn window_excitation<'a, I>(factor_samples: I) -> WindowExcitation
where
    I: IntoIterator<Item = &'a [ImuMeasurement]>,
{
    let mut rot_rad = 0.0f64;
    let mut samples = 0usize;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;

    for slice in factor_samples {
        // dt comes from the sample stamps, not the nominal rate: the integral has to be right
        // for a stream that skipped, and a non-positive step (out-of-order) contributes nothing
        // rather than subtracting angle.
        for pair in slice.windows(2) {
            let dt = pair[1].timestamp - pair[0].timestamp;
            if dt > 0.0 {
                rot_rad += pair[0].gyro.length() * dt;
            }
        }
        for m in slice {
            samples += 1;
            let mag = m.accel.length();
            let delta = mag - mean;
            mean += delta / samples as f64;
            m2 += delta * (mag - mean);
        }
    }

    WindowExcitation {
        rot_rad,
        accel_var: if samples < 2 {
            0.0
        } else {
            m2 / samples as f64
        },
        samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 200 Hz, exactly.
    const PERIOD_NS: u64 = 5_000_000;

    fn samples(from_ns: u64, n: usize) -> Vec<RawImuSample> {
        (0..n)
            .map(|i| RawImuSample {
                stamp_ns: from_ns + i as u64 * PERIOD_NS,
                gyro: [0.0, 0.0, 0.01],
                accel: [0.0, 9.81, 0.0],
            })
            .collect()
    }

    fn gates() -> ImuGates {
        ImuGates::new(Frequency::new::<hertz>(200.0))
    }

    #[test]
    fn test_a_covered_interval_returns_every_sample_in_it() {
        let mut buf = ImuBuffer::with_capacity(1024);
        buf.push(samples(1_000_000_000, 400), 0);
        // [1.0 s, 1.5 s]: samples at 1.000, 1.005, ... 1.500 inclusive => 101.
        let got: Vec<RawImuSample> = buf
            .window(1_000_000_000, 1_500_000_000, &gates())
            .expect("covered")
            .copied()
            .collect();
        assert_eq!(got.len(), 101);
        assert_eq!(got.first().unwrap().stamp_ns, 1_000_000_000);
        assert_eq!(got.last().unwrap().stamp_ns, 1_500_000_000);
    }

    #[test]
    fn test_a_reported_drop_inside_the_interval_refuses_it() {
        let mut buf = ImuBuffer::with_capacity(1024);
        buf.push(samples(1_000_000_000, 100), 0);
        // Same cadence, so there is no gap to find: the ONLY evidence is the counter.
        buf.push(samples(1_000_000_000 + 100 * PERIOD_NS, 100), 7);
        let err = buf
            .window(1_000_000_000, 1_900_000_000, &gates())
            .unwrap_err();
        assert!(
            matches!(err, ImuWindowError::DroppedSamples { .. }),
            "{err:?}"
        );
        assert_eq!(buf.stats().drop_reports, 1);
    }

    #[test]
    fn test_the_first_drop_count_is_a_baseline_not_a_drop() {
        let mut buf = ImuBuffer::with_capacity(1024);
        // A tracker started mid-run sees a large cumulative count on its very first batch.
        buf.push(samples(1_000_000_000, 200), 4242);
        assert_eq!(buf.stats().drop_reports, 0);
        buf.window(1_000_000_000, 1_500_000_000, &gates())
            .expect("historical drops must not refuse the first interval");
    }

    #[test]
    fn test_a_drop_ages_out_with_the_samples_it_invalidated() {
        let mut buf = ImuBuffer::with_capacity(1024);
        buf.push(samples(1_000_000_000, 100), 0);
        buf.push(samples(1_000_000_000 + 100 * PERIOD_NS, 200), 1);
        // The interval that straddled the hole is gone; the next one must not inherit it.
        buf.prune_before(1_600_000_000);
        buf.window(1_600_000_000, 2_000_000_000, &gates())
            .expect("a pruned drop mark must not refuse later intervals");
    }

    #[test]
    fn test_a_hole_in_the_middle_is_refused_even_with_full_coverage_either_side() {
        let mut buf = ImuBuffer::with_capacity(1024);
        buf.push(samples(1_000_000_000, 100), 0);
        // 100 ms of nothing, then the stream resumes. No drop is reported: this is the
        // device-queue overwrite case, which nothing in the driver counts.
        buf.push(samples(1_600_000_000, 100), 0);
        let err = buf
            .window(1_000_000_000, 2_000_000_000, &gates())
            .unwrap_err();
        match err {
            ImuWindowError::Gap { gap, .. } => assert_eq!(gap.as_nanos(), 105_000_000),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_an_interval_ending_in_a_hole_is_refused() {
        let mut buf = ImuBuffer::with_capacity(1024);
        buf.push(samples(1_000_000_000, 100), 0);
        // Asking out to 2.0 s when the samples stop at 1.495 s: `from_measurements` would
        // extrapolate the last sample across the missing half second at full weight.
        let err = buf
            .window(1_000_000_000, 2_000_000_000, &gates())
            .unwrap_err();
        assert!(matches!(err, ImuWindowError::Gap { .. }), "{err:?}");
    }

    #[test]
    fn test_an_interval_whose_start_was_evicted_is_refused() {
        let mut buf = ImuBuffer::with_capacity(64);
        // 400 samples span 1.000-2.995 s; only the last 64 (2.680-2.995 s) survive.
        buf.push(samples(1_000_000_000, 400), 0);
        assert_eq!(buf.stats().evicted, 336);
        let err = buf
            .window(2_500_000_000, 2_900_000_000, &gates())
            .unwrap_err();
        // Reported as a leading gap, which is exactly what an eviction is from the
        // integrator's point of view.
        assert!(matches!(err, ImuWindowError::Gap { .. }), "{err:?}");
    }

    #[test]
    fn test_uniform_decimation_is_refused_by_coverage_not_by_gap() {
        let mut buf = ImuBuffer::with_capacity(1024);
        // Every other sample, i.e. 100 Hz: each gap is 10 ms, under the 15 ms limit.
        let thinned: Vec<RawImuSample> =
            samples(1_000_000_000, 200).into_iter().step_by(2).collect();
        buf.push(thinned, 0);
        let err = buf
            .window(1_000_000_000, 1_500_000_000, &gates())
            .unwrap_err();
        match err {
            ImuWindowError::TooFewSamples { got, want, .. } => {
                assert_eq!(got, 51);
                assert_eq!(want, 90);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn test_out_of_order_samples_are_rejected_at_ingest() {
        let mut buf = ImuBuffer::with_capacity(1024);
        buf.push(samples(1_000_000_000, 10), 0);
        // A repeat of the same stamps: `from_measurements` would sort and `dt > 0.0`-filter
        // these into a plausible answer.
        buf.push(samples(1_000_000_000, 10), 0);
        assert_eq!(buf.stats().out_of_order, 10);
        assert_eq!(buf.stats().buffered, 10);
    }

    #[test]
    fn test_an_empty_or_inverted_interval_is_refused() {
        let mut buf = ImuBuffer::with_capacity(1024);
        buf.push(samples(1_000_000_000, 200), 0);
        assert!(matches!(
            buf.window(1_500_000_000, 1_500_000_000, &gates())
                .unwrap_err(),
            ImuWindowError::NonPositiveInterval { .. }
        ));
        assert!(matches!(
            buf.window(1_500_000_000, 1_400_000_000, &gates())
                .unwrap_err(),
            ImuWindowError::NonPositiveInterval { .. }
        ));
    }

    #[test]
    fn test_an_interval_the_buffer_never_saw_is_refused_as_no_samples() {
        let mut buf = ImuBuffer::with_capacity(1024);
        buf.push(samples(1_000_000_000, 100), 0);
        let err = buf
            .window(5_000_000_000, 5_500_000_000, &gates())
            .unwrap_err();
        assert!(matches!(err, ImuWindowError::NoSamples { .. }), "{err:?}");
    }

    #[test]
    fn test_stamps_are_rebased_on_the_tracker_epoch_not_on_the_buffer() {
        // The epoch is ~1.7e18 ns; converting absolute stamps to f64 seconds would leave
        // ~400 ns of resolution, which is a tenth of the 5 ms sample period.
        let epoch = 1_757_000_000_000_000_000u64;
        let s = RawImuSample {
            stamp_ns: epoch + 5_000_000,
            gyro: [1.0, 2.0, 3.0],
            accel: [4.0, 5.0, 6.0],
        };
        let m = s.to_measurement(epoch);
        assert!((m.timestamp - 0.005).abs() < 1e-12, "{}", m.timestamp);
        assert_eq!(m.gyro.x, 1.0);
        assert_eq!(m.accel.z, 6.0);
        // A sample from before the first frame collapses to 0 rather than wrapping.
        let before = RawImuSample {
            stamp_ns: epoch - 1,
            ..s
        };
        assert_eq!(before.to_measurement(epoch).timestamp, 0.0);
    }

    // ── Window excitation ────────────────────────────────────────────────────
    //
    // Hand-built windows with KNOWN motion, because the point of the two statistics is that
    // they separate cases `ImuInitializer::ready` cannot: parked, rotating-in-place, and
    // translating. Each test therefore carries the negative control for the other statistic —
    // a rotating window MUST leave `accel_var` at zero, and a shaking one MUST leave
    // `rot_rad` at zero — since a per-axis (rather than per-magnitude) accel variance, or a
    // signed gyro integral, passes the parked case and fails exactly those.

    /// 200 Hz over 2.000 s inclusive of both ends: 401 samples, 400 steps.
    const EXC_N: usize = 401;
    const EXC_DT: f64 = 0.005;
    const EXC_SPAN_S: f64 = 2.0;
    /// Measured on a BNO086: accepted initializations agreed on |bg| ~= 0.0028 rad/s, so this is
    /// what a PARKED gyro reads and what `rot_rad` must stay down at.
    const PARKED_GYRO_RAD_S: f64 = 0.0028;

    fn exc_window(
        gyro: impl Fn(f64) -> Vec3F64,
        accel: impl Fn(f64) -> Vec3F64,
    ) -> Vec<ImuMeasurement> {
        (0..EXC_N)
            .map(|i| {
                let t = i as f64 * EXC_DT;
                ImuMeasurement {
                    timestamp: t,
                    gyro: gyro(t),
                    accel: accel(t),
                }
            })
            .collect()
    }

    #[test]
    fn test_a_parked_window_excites_neither_statistic() {
        let w = exc_window(
            |_| Vec3F64::new(PARKED_GYRO_RAD_S, 0.0, 0.0),
            |_| Vec3F64::new(0.0, 9.81, 0.0),
        );
        let exc = window_excitation([w.as_slice()]);

        assert_eq!(exc.samples, EXC_N);
        // Bias only: 0.0028 rad/s * 2.000 s. This is the floor the rotating case has to clear.
        assert!(
            (exc.rot_rad - PARKED_GYRO_RAD_S * EXC_SPAN_S).abs() < 1e-9,
            "{exc:?}"
        );
        // Every magnitude identical, so the variance is exactly zero, not merely small.
        assert_eq!(exc.accel_var, 0.0, "{exc:?}");
    }

    #[test]
    fn test_rotation_in_place_shows_in_rot_rad_and_stays_out_of_accel_var() {
        // 0.5 rad/s about z for 2 s: gravity sweeps between the x and y axes at constant
        // magnitude, which is exactly what a robot pivoting on the spot feeds the accelerometer.
        const OMEGA: f64 = 0.5;
        let w = exc_window(
            |_| Vec3F64::new(0.0, 0.0, OMEGA),
            |t| {
                let th = OMEGA * t;
                Vec3F64::new(9.81 * th.sin(), 9.81 * th.cos(), 0.0)
            },
        );
        let exc = window_excitation([w.as_slice()]);

        assert!((exc.rot_rad - OMEGA * EXC_SPAN_S).abs() < 1e-9, "{exc:?}");
        // Two orders of magnitude above the parked window — the separation the statistic exists
        // for, not just "greater than zero".
        assert!(
            exc.rot_rad > 100.0 * PARKED_GYRO_RAD_S * EXC_SPAN_S,
            "{exc:?}"
        );
        // The negative control. A variance taken per accelerometer AXIS would read ~48 here
        // (gravity swinging through 1 rad on two axes) and would call a pivoting-in-place robot
        // translationally excited, which is the failure this whole measurement is chasing.
        assert!(exc.accel_var < 1e-20, "{exc:?}");
    }

    #[test]
    fn test_linear_acceleration_is_what_shows_in_accel_var() {
        // 1 m/s^2 amplitude along the gravity axis at 1 Hz: two whole periods inside the window,
        // and no rotation at all.
        let w = exc_window(
            |_| Vec3F64::ZERO,
            |t| Vec3F64::new(0.0, 9.81 + (std::f64::consts::TAU * t).sin(), 0.0),
        );
        let exc = window_excitation([w.as_slice()]);

        assert_eq!(exc.rot_rad, 0.0, "{exc:?}");
        // Population variance over 401 samples of which 400 span two whole periods:
        // (1/401) * 400 * mean(sin^2) = 200/401. Sample variance (n-1) would be 0.5 — a
        // difference this tolerance is deliberately tight enough to see.
        assert!(
            (exc.accel_var - 200.0 / 401.0).abs() < 1e-9,
            "{exc:?}, want {}",
            200.0 / 401.0
        );
    }

    #[test]
    fn test_excitation_sums_across_factors_and_survives_a_degenerate_one() {
        // The real caller hands one slice per `ImuFactor`, so the angle has to accumulate across
        // slices; a per-slice max or a first-slice-only reduction passes every test above.
        let a = exc_window(
            |_| Vec3F64::new(0.0, 0.0, 0.5),
            |_| Vec3F64::new(0.0, 9.81, 0.0),
        );
        let b = exc_window(
            |_| Vec3F64::new(0.0, 0.0, 0.5),
            |_| Vec3F64::new(0.0, 9.81, 0.0),
        );
        let exc = window_excitation([a.as_slice(), b.as_slice()]);
        assert!(
            (exc.rot_rad - 2.0 * 0.5 * EXC_SPAN_S).abs() < 1e-9,
            "{exc:?}"
        );
        assert_eq!(exc.samples, 2 * EXC_N);

        // A one-sample factor has no interval to integrate over and no variance to take. The
        // first is what `windows(2)` handles; the second is the n < 2 guard.
        let one = window_excitation([&a[..1]]);
        assert_eq!((one.rot_rad, one.accel_var, one.samples), (0.0, 0.0, 1));
        let none = window_excitation(std::iter::empty::<&[ImuMeasurement]>());
        assert_eq!(none, WindowExcitation::default());
    }
}
