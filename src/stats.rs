//! Latency accounting for the tracker.
//!
//! # Why percentiles, and why split by keyframe
//!
//! Per-frame tracker cost is **bimodal**: a frame that inserts a keyframe runs fuse, local BA and
//! cull on top of tracking, and the design's own numbers put keyframe insertion at 1.2-3.3 per
//! second. A mean across both populations describes neither, and sizing a rate target or an async
//! drop budget off it is how a pipeline that "meets 93 ms on average" misses its deadline on every
//! keyframe.
//!
//! So every measurement here is reported as p50 / p99 / max, **split by whether a keyframe was
//! inserted**, and the two populations are never averaged together.
//!
//! # These are wall-clock samples, not a benchmark
//!
//! Collected in the live graph, in release, on the machine that ran it. That is the point — the
//! numbers exist to set `rate_target_hz` and the ring depths, and a microbenchmark of the tracker
//! in isolation would not include decode, rectify, the copperlist plumbing or the logger's
//! synchronous per-iteration `log(cl)`.

use std::time::Duration;

/// Cap on retained samples: ~3 hours at 10 Hz, 800 kB.
const MAX_SAMPLES: usize = 100_000;

/// A stream of durations, kept in full so percentiles are exact — up to [`MAX_SAMPLES`].
///
/// Retaining every sample is affordable for an offline clip (584 frames is 584 f64s) and avoids
/// a streaming estimator's error on exactly the tail we care about. A running graph can feed the
/// same struct for days at 10 Hz, so it is a ring: past the cap the oldest sample is dropped and
/// the percentiles describe the most recent window rather than the whole run. For a long-running
/// system that is the more useful statistic anyway — but it is not a lifetime figure, and `stop`
/// lines from long runs should be read that way.
pub struct Durations {
    ms: std::collections::VecDeque<f64>,
}

impl Default for Durations {
    /// Reserves the full [`MAX_SAMPLES`] ring up front: [`Durations::push`] runs inside the
    /// task's `process`, and a growing ring would reallocate on that path. 800 kB per
    /// population, paid once at construction.
    fn default() -> Self {
        Self {
            ms: std::collections::VecDeque::with_capacity(MAX_SAMPLES),
        }
    }
}

impl Durations {
    /// Records one sample, evicting the oldest past [`MAX_SAMPLES`].
    pub fn push(&mut self, d: Duration) {
        if self.ms.len() == MAX_SAMPLES {
            self.ms.pop_front();
        }
        self.ms.push_back(d.as_secs_f64() * 1e3);
    }

    /// How many samples.
    pub fn len(&self) -> usize {
        self.ms.len()
    }

    /// Nearest-rank percentile in milliseconds, or `None` if empty.
    ///
    /// Nearest-rank rather than interpolated: with a few hundred samples the interpolation is
    /// noise, and a percentile that is an actual observed sample is easier to reason about
    /// against a deadline.
    pub fn pct(&self, p: f64) -> Option<f64> {
        if self.ms.is_empty() {
            return None;
        }
        let mut v: Vec<f64> = self.ms.iter().copied().collect();
        // `total_cmp` orders every finite non-negative duration exactly as `partial_cmp` would,
        // and cannot panic.
        v.sort_by(f64::total_cmp);
        let rank = ((p / 100.0) * v.len() as f64).ceil() as usize;
        Some(v[rank.clamp(1, v.len()) - 1])
    }
}

/// Tracker cost, split by the branch that makes it bimodal.
#[derive(Default)]
pub struct TrackerTiming {
    /// Frames that did NOT insert a keyframe.
    pub tracking: Durations,
    /// Frames that DID — tracking plus fuse, local BA and cull.
    pub keyframe: Durations,
    /// LIFETIME counts, kept apart from the rings above.
    ///
    /// The two rings cap INDEPENDENTLY, and keyframes are ~13 % of frames, so `tracking` fills
    /// long before `keyframe` does. Deriving the ratio from `len()` then drifts from 13 % toward
    /// 50 % over a long run — wrong on exactly the runs the cap exists for, and it is the number
    /// the keyframe-cost argument rests on.
    tracking_n: u64,
    keyframe_n: u64,
}

impl TrackerTiming {
    /// Records one frame against the right population.
    pub fn push(&mut self, d: Duration, keyframe_inserted: bool) {
        if keyframe_inserted {
            self.keyframe.push(d);
            self.keyframe_n += 1;
        } else {
            self.tracking.push(d);
            self.tracking_n += 1;
        }
    }

    /// Frames that inserted a keyframe, over the whole run (not capped like the rings).
    pub fn keyframe_frames(&self) -> u64 {
        self.keyframe_n
    }

    /// Every frame recorded, over the whole run.
    pub fn total_frames(&self) -> u64 {
        self.tracking_n + self.keyframe_n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The keyframe ratio must survive the rings capping at different times.
    #[test]
    fn test_keyframe_ratio_is_not_distorted_by_the_ring_caps() {
        let mut t = TrackerTiming::default();
        // 13 % keyframes, run long enough that the tracking ring caps and the keyframe one
        // does not — the regime where deriving the ratio from `len()` drifts toward 50 %.
        // Long enough that the BUGGY form is unambiguously wrong: once `tracking` caps and
        // `keyframe` keeps filling, len()-derived drifts toward 50 %. 20k past the cap left
        // the two forms within rounding of each other, which is no discrimination at all.
        for i in 0..(MAX_SAMPLES * 8) {
            t.push(Duration::from_millis(1), i % 8 == 0);
        }
        let pct = 100.0 * t.keyframe_frames() as f64 / t.total_frames() as f64;
        assert!(
            (pct - 12.5).abs() < 0.5,
            "ratio drifted with the ring caps: reported {pct} %, truth 12.5 %"
        );
        assert!(
            t.tracking.len() < t.total_frames() as usize,
            "the tracking ring must have capped for this test to discriminate"
        );
    }

    /// The ring must evict the OLDEST sample, and actually cap.
    ///
    /// Written because neither was asserted: a ring that evicted the newest would report the
    /// first three hours of a week-long run forever, and one that never capped is the unbounded
    /// Vec this replaced. Both look identical in a stop line.
    #[test]
    fn test_duration_ring_caps_and_drops_the_oldest() {
        let mut d = Durations::default();
        for i in 0..MAX_SAMPLES + 500 {
            d.push(Duration::from_millis(i as u64));
        }
        assert_eq!(d.len(), MAX_SAMPLES, "the ring did not cap");
        // The first 500 are gone, so the smallest surviving sample is 500 ms. If eviction took
        // the NEWEST instead, the minimum would still be 0.
        assert!(
            d.pct(0.0).expect("non-empty") >= 500.0,
            "eviction dropped the newest sample, not the oldest"
        );
        assert!(
            d.pct(100.0).expect("non-empty") >= (MAX_SAMPLES + 499) as f64,
            "the most recent sample was evicted"
        );
    }

    fn ms(v: f64) -> Duration {
        Duration::from_secs_f64(v / 1e3)
    }

    #[test]
    fn test_nearest_rank_percentiles() {
        let mut d = Durations::default();
        for v in [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0] {
            d.push(ms(v));
        }
        assert_eq!(d.pct(50.0).map(|v| v.round()), Some(50.0));
        assert_eq!(d.pct(99.0).map(|v| v.round()), Some(100.0));
        assert_eq!(d.pct(100.0).map(|v| v.round()), Some(100.0));
    }

    #[test]
    fn test_empty_reports_no_samples_rather_than_zero() {
        let d = Durations::default();
        assert_eq!(d.len(), 0);
        assert_eq!(
            d.pct(50.0),
            None,
            "an empty stream has no p50, not a p50 of 0"
        );
    }

    /// The property the whole module exists for: the two populations must not be merged, because
    /// the mean of a bimodal distribution sits where no frame actually lands.
    #[test]
    fn test_keyframe_and_tracking_stay_separate() {
        let mut t = TrackerTiming::default();
        for _ in 0..90 {
            t.push(ms(10.0), false);
        }
        for _ in 0..10 {
            t.push(ms(100.0), true);
        }
        assert_eq!(t.tracking.pct(99.0).map(|v| v.round()), Some(10.0));
        assert_eq!(t.keyframe.pct(99.0).map(|v| v.round()), Some(100.0));
        // The merged mean would be 19 ms — a figure no frame ever took.
        assert_eq!((t.keyframe_frames(), t.total_frames()), (10, 100));
    }
}
