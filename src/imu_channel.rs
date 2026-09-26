//! The queue that gets IMU samples into a BACKGROUNDED tracker.
//!
//! # Why this is not a second graph edge, and not a field on the frame payload
//!
//! `StereoVio` has exactly one input because `cu29`'s `CuAsyncTask`, which is what
//! `background: true` wraps a task in, is implemented only for
//! `T: CuTask<Input<'i> = CuMsg<I>>`. A second edge would un-background the solve, and the solve
//! costs p50 53 ms / p99 91 ms, and p50 93 ms on a keyframe, against a 66 ms frame interval
//! (measured on a Jetson Orin with an OAK-D at 640x400): inline, it holds every other task in the
//! graph behind it.
//!
//! Bundling the samples into the `StereoPair` payload keeps one input, and costs only ~2 % of the
//! payload, but puts the inertial stream on the ONE edge that is designed to drop: `CuAsyncTask`
//! discards the arriving input while the previous solve is `Running`, and again while `Waiting`
//! for the length of that solve, so roughly one frame in four is refused, taking its samples
//! with them, with the producer's drop counter reading 0 because the SOURCE dropped nothing.
//! `from_measurements` then turns the decimated remainder into a full-`dt` delta at the wrong
//! magnitude. The inertial stream must not ride the dropping edge.
//!
//! So the samples travel beside the copperlist, through an [`ImuQueue`] shared as a resource of
//! the [`VioBus`](crate::VioBus) bundle: [`ImuFeed`](crate::ImuFeed), an inline sink that sees
//! every [`ImuBatch`](cu_stereo_payloads::ImuBatch) the source emits, pushes into it, and
//! `StereoVio` drains it on the frame it is given.
//!
//! # What this costs
//!
//! The IMU batches are still a logged graph edge (the one into `ImuFeed`), but the drain is not:
//! which samples a given solve saw depends on when the background worker ran. The preintegrated
//! delta is a pure function of the logged samples plus two keyframe stamps, so an offline replay
//! of the tracker can recompute it, but a resim of the graph does not reproduce the inertial term bit for bit.
//!
//! # Why the producer, not the queue, is what has to be inert
//!
//! A graph can carry an IMU edge whether or not its `vio` node has an `inertial` block. Without a
//! consumer nothing ever drains, so an unconditional push fills the ring to capacity within ~10 s
//! and from then on takes the mutex and does a push_back + pop_front per sample for the life of
//! the process, with `evicted` climbing meaninglessly. So the queue is ARMED by the consumer
//! (`StereoVio::new`, when it has an `inertial` block) and [`ImuQueue::push`] returns before the
//! lock until then: the same off-by-default discipline `TrackerConfig::inertial = None` applies
//! to the tracker, applied to the seam. It also makes `evicted` mean what its doc says: samples
//! lost because the consumer was slow, not because there was none.

use std::collections::VecDeque;
use std::collections::vec_deque::Drain;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::imu::RawImuSample;
use cu_stereo_payloads::ImuSample;

/// Samples held between drains. The tracker's own ring is sized the same way, and for the same
/// interval; see the constant's doc.
pub(crate) use crate::imu::DEFAULT_BUFFER_CAPACITY as CAPACITY;

#[derive(Debug, Default)]
struct Pending {
    samples: VecDeque<RawImuSample>,
    /// The producer's own cumulative drop count, as last reported.
    source_dropped: u64,
    /// Samples this queue itself evicted because nobody drained it in time. Folded into the
    /// number [`ImuQueue::drain_with`] hands over, because from the integrator's point of view a
    /// sample lost here and a sample lost in the device queue punch the identical hole.
    evicted: u64,
    /// Set by a producer that reports the samples are already rotated into the camera frame.
    /// Sticky until [`ImuQueue::clear`], because a single aligned batch is enough to make the
    /// extrinsic wrong for the interval containing it.
    aligned: bool,
}

/// A bounded queue of IMU samples between an inline producer and a backgrounded consumer.
///
/// One instance per [`VioBus`](crate::VioBus): two estimators in one process each get their
/// own, which a process-global could not give them.
#[derive(Debug, Default)]
pub struct ImuQueue {
    pending: Mutex<Pending>,
    /// Whether a consumer exists. `false`, the state of every graph without an `inertial` block,
    /// makes [`ImuQueue::push`] a single relaxed load and no lock. See the module doc.
    armed: AtomicBool,
}

impl ImuQueue {
    /// An empty, unarmed queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares that a consumer will drain this queue. Called by the tracker task when it is
    /// configured for the inertial path; never called otherwise, so [`ImuQueue::push`] stays
    /// free.
    pub fn arm(&self) {
        self.armed.store(true, Ordering::Relaxed);
    }

    /// Whether a consumer has armed the queue.
    pub fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Relaxed)
    }

    /// Hands a batch of samples to whichever tracker drains next.
    ///
    /// `dropped_cumulative` is the producer's CUMULATIVE drop counter (a driver never resets
    /// it), and `aligned` says the samples have already been rotated into the camera frame, in
    /// which case a configured `imu_t_bc` would apply the same rotation a second time. The
    /// tracker refuses the inertial path in that case rather than producing a tilted gravity
    /// vector that reads exactly like an accelerometer bias.
    ///
    /// Takes an iterator so the producer can map its payload type straight into the ring under
    /// the lock, with no intermediate `Vec` on the IMU-rate path.
    ///
    /// Call this from an INLINE task. Calling it from the backgrounded tracker's own upstream
    /// would defeat the entire point of the module.
    ///
    /// The lock this takes is shared with `ImuQueue::drain_with` on the backgrounded side, so
    /// an inline caller can wait on it. The wait is bounded by what the consumer does under it:
    /// one copy of the pending samples into the tracker's ring: at most [`CAPACITY`] samples,
    /// in practice what arrived since the previous solve (~120 at 200 Hz and a 600 ms solve),
    /// with no solve and no I/O inside. The consumer never holds it across a solve.
    ///
    /// A poisoned lock is ignored: the alternative is a panic in a graph task over a buffer the
    /// tracker's coverage gates already guard, since they refuse an interval whose samples went
    /// missing for any reason.
    pub fn push(
        &self,
        samples: impl IntoIterator<Item = ImuSample>,
        dropped_cumulative: u64,
        aligned: bool,
    ) {
        if !self.is_armed() {
            return;
        }
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        let pending = &mut *pending;
        pending.source_dropped = dropped_cumulative;
        pending.aligned |= aligned;
        for sample in samples {
            crate::imu::push_bounded(
                &mut pending.samples,
                sample.into(),
                CAPACITY,
                &mut pending.evicted,
            );
        }
    }

    /// Runs `f` under the lock over everything the queue holds, emptying it.
    ///
    /// `f` receives the draining iterator, then `(dropped_cumulative, aligned)`: the drop figure
    /// is the producer's count plus this queue's own evictions, which is the number
    /// [`crate::track::Tracker::push_imu`] wants, since both are holes in the same stream.
    /// Handing the consumer the iterator rather than filling a `Vec` for it means each sample is
    /// copied once, from this deque into the tracker's ring; the lock is held for that
    /// ~120-element copy, well under the producer's tick.
    ///
    /// `None` on a poisoned lock; the samples are then simply not delivered, and the coverage
    /// gates refuse the interval.
    pub(crate) fn drain_with<R>(
        &self,
        f: impl FnOnce(Drain<'_, RawImuSample>, u64, bool) -> R,
    ) -> Option<R> {
        let Ok(mut pending) = self.pending.lock() else {
            return None;
        };
        let pending = &mut *pending;
        let dropped = pending.source_dropped + pending.evicted;
        let aligned = pending.aligned;
        Some(f(pending.samples.drain(..), dropped, aligned))
    }

    /// Empties the queue and forgets the alignment flag, for a consumer that wants a clean slate
    /// rather than a backlog spanning a reset. The drop counters are kept: they are cumulative.
    pub fn clear(&self) {
        let Ok(mut pending) = self.pending.lock() else {
            return;
        };
        pending.samples.clear();
        pending.aligned = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu_stereo_payloads::ImuPayload;
    use cu29::clock::CuTime;

    fn batch(from_ns: u64, n: u64) -> impl Iterator<Item = ImuSample> {
        (0..n).map(move |i| ImuSample {
            tov: CuTime::from_nanos(from_ns + i),
            imu: ImuPayload::from_raw([0.0; 3], [0.0; 3], 25.0),
        })
    }

    fn drain_all(queue: &ImuQueue) -> (Vec<RawImuSample>, u64, bool) {
        queue
            .drain_with(|s, d, a| (s.collect(), d, a))
            .expect("lock not poisoned")
    }

    #[test]
    fn test_an_unarmed_queue_does_not_buffer() {
        let queue = ImuQueue::new();
        queue.push(batch(0, 3), 1, false);
        let (out, dropped, _) = drain_all(&queue);
        assert!(out.is_empty(), "an unarmed queue must not buffer");
        assert_eq!(
            dropped, 0,
            "an unarmed push must not even record the drop count"
        );
    }

    #[test]
    fn test_push_then_drain_round_trips_samples_and_drops() {
        let queue = ImuQueue::new();
        queue.arm();
        queue.push(batch(1_000, 3), 5, false);
        let (out, dropped, aligned) = drain_all(&queue);
        assert_eq!(
            out.iter().map(|s| s.stamp_ns).collect::<Vec<_>>(),
            [1_000, 1_001, 1_002]
        );
        assert_eq!(dropped, 5);
        assert!(!aligned);

        // Drained means drained: a second call must not re-deliver.
        let (out, dropped, _) = drain_all(&queue);
        assert!(out.is_empty());
        assert_eq!(dropped, 5, "the cumulative count survives the drain");
    }

    #[test]
    fn test_overflow_is_counted_as_a_drop_and_alignment_is_sticky() {
        let queue = ImuQueue::new();
        queue.arm();
        queue.push(batch(2_000, CAPACITY as u64 + 10), 5, true);
        queue.push(batch(1_000_000, 0), 5, false);
        let (out, dropped, aligned) = drain_all(&queue);
        assert_eq!(out.len(), CAPACITY);
        assert_eq!(
            out[0].stamp_ns, 2_010,
            "the OLDEST samples are the ones evicted"
        );
        assert_eq!(dropped, 15, "5 reported by the source + 10 evicted here");
        assert!(aligned, "a later unaligned batch must not clear the flag");

        queue.clear();
        let (_, dropped, aligned) = drain_all(&queue);
        assert!(!aligned, "clear forgets the alignment flag");
        assert_eq!(dropped, 15, "clear keeps the cumulative drop count");
    }

    #[test]
    fn test_two_queues_are_independent() {
        let (a, b) = (ImuQueue::new(), ImuQueue::new());
        a.arm();
        b.arm();
        a.push(batch(0, 2), 0, false);
        assert_eq!(drain_all(&a).0.len(), 2);
        assert!(
            drain_all(&b).0.is_empty(),
            "a push to one bus reached the other"
        );
    }
}
