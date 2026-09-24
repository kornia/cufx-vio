//! Requested SLAM restarts.
//!
//! A monotone EPOCH rather than a boolean latch. The task applies a pending reset at the next
//! frame boundary and publishes the tracker's world generation on every pose, so a consumer can
//! tell exactly which tracker world a pose belongs to. A flag cannot say WHICH frame the new
//! world anchored on, and re-anchoring on a guess offsets the whole map by whatever the camera
//! moved in between.
//!
//! The counter is a resource of the [`VioBus`](crate::VioBus) bundle, so the request reaches
//! the task through the graph's own resource wiring: any task bound to the same `reset_epoch`
//! resource can call [`ResetEpoch::request`], and two estimators on two buses reset
//! independently.

use std::sync::atomic::{AtomicU64, Ordering};

/// A monotone count of requested resets, shared between whoever requests them and the tracker
/// task that applies them.
#[derive(Debug, Default)]
pub struct ResetEpoch {
    requested: AtomicU64,
}

impl ResetEpoch {
    /// A counter with no reset requested.
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests a SLAM reset. The task applies it at the next frame boundary; consumers see the
    /// change as a bump of `VioStatus::reset_epoch`.
    pub fn request(&self) {
        self.requested.fetch_add(1, Ordering::Relaxed);
    }

    /// The epoch the task should be running under.
    pub fn requested(&self) -> u64 {
        self.requested.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_each_request_bumps_the_epoch_once() {
        let epoch = ResetEpoch::new();
        assert_eq!(epoch.requested(), 0);
        epoch.request();
        epoch.request();
        assert_eq!(epoch.requested(), 2);
    }
}
