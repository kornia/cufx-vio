//! `ImuFeed`: the inline sink that moves IMU batches from the graph onto a [`VioBus`].
//!
//! [`VioBus`]: crate::VioBus

use std::sync::Arc;

use cu29::prelude::*;
use cufx_sensor_payloads::ImuBatch;

use crate::imu_channel::ImuQueue;

/// Resource bindings of [`ImuFeed`].
#[doc(hidden)]
#[allow(missing_docs)]
pub mod feed_resources {
    use super::ImuQueue;
    use cu29::resources;

    resources!({
        imu => Shared<ImuQueue>,
    });
}

/// Pushes every [`ImuBatch`] it receives into the bus's [`ImuQueue`].
///
/// A sink rather than a second input on `StereoVio`: the tracker must keep exactly one input to
/// be backgrounded, and a backgrounded task refuses most of its inputs (see
/// [`crate::imu_channel`]). This task runs inline, so it sees every batch, and the batches stay
/// a logged graph edge. Until a `StereoVio` with an `inertial` block arms the queue, a push is a
/// single atomic load.
///
/// `N` is the source's batch capacity; the RON names it, as in `cufx_vio::ImuFeed<32>`.
#[derive(Reflect)]
#[reflect(no_field_bounds, from_reflect = false)]
pub struct ImuFeed<const N: usize> {
    #[reflect(ignore)]
    imu: Arc<ImuQueue>,
}

// Holds no state of its own: the queue is a bus resource, drained by the tracker.
impl<const N: usize> Freezable for ImuFeed<N> {}

impl<const N: usize> CuSinkTask for ImuFeed<N> {
    type Resources<'r> = feed_resources::Resources;
    type Input<'m> = input_msg!(ImuBatch<N>);

    fn new(config: Option<&ComponentConfig>, resources: Self::Resources<'_>) -> CuResult<Self>
    where
        Self: Sized,
    {
        crate::config::deny_unknown_keys(config, "cufx_vio::ImuFeed", &[])?;
        Ok(Self { imu: resources.imu })
    }

    fn process<'i>(&mut self, _ctx: &CuContext, input: &Self::Input<'i>) -> CuResult<()> {
        let Some(batch) = input.payload() else {
            return Ok(());
        };
        self.imu.push(
            batch.samples().iter().copied(),
            batch.dropped,
            batch.camera_aligned,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cu29::clock::CuTime;
    use cufx_sensor_payloads::{ImuPayload, ImuSample};

    fn feed() -> (ImuFeed<4>, Arc<ImuQueue>) {
        let imu = Arc::new(ImuQueue::new());
        let task = ImuFeed::<4>::new(
            None,
            feed_resources::Resources {
                imu: Arc::clone(&imu),
            },
        )
        .expect("no config to reject");
        (task, imu)
    }

    fn batch(stamps: &[u64]) -> ImuBatch<4> {
        let mut batch = ImuBatch::<4>::new();
        for &ns in stamps {
            batch
                .push(ImuSample {
                    tov: CuTime::from_nanos(ns),
                    imu: ImuPayload::from_raw([0.0, 0.0, 9.81], [0.1, 0.0, 0.0], 25.0),
                })
                .expect("fits");
        }
        batch
    }

    #[test]
    fn test_imu_feed_pushes_every_sample_into_the_bus() {
        let (mut task, imu) = feed();
        imu.arm();
        let mut b = batch(&[10, 20, 30]);
        b.dropped = 2;
        b.camera_aligned = true;
        let mut msg = CuMsg::new(Some(b));
        msg.tov = b.tov();
        let ctx = CuContext::new_with_clock();
        task.process(&ctx, &msg).expect("push");

        let (stamps, dropped, aligned) = imu
            .drain_with(|s, d, a| (s.map(|s| s.stamp_ns).collect::<Vec<_>>(), d, a))
            .expect("lock");
        assert_eq!(
            stamps,
            [10, 20, 30],
            "per-sample tov survives into the queue"
        );
        assert_eq!(dropped, 2);
        assert!(aligned);
    }

    #[test]
    fn test_imu_feed_ignores_an_empty_message() {
        let (mut task, imu) = feed();
        imu.arm();
        let msg = CuMsg::<ImuBatch<4>>::new(None);
        let ctx = CuContext::new_with_clock();
        task.process(&ctx, &msg)
            .expect("no payload is not an error");
        assert_eq!(imu.drain_with(|s, _, _| s.count()), Some(0));
    }

    #[test]
    fn test_imu_feed_refuses_config_keys() {
        let mut cfg = ComponentConfig::new();
        cfg.set("capacity", 4u32);
        let res = ImuFeed::<4>::new(
            Some(&cfg),
            feed_resources::Resources {
                imu: Arc::new(ImuQueue::new()),
            },
        );
        assert!(res.is_err(), "ImuFeed reads no key, so any key is a typo");
    }

    /// The message carrying a batch is stamped with the sample RANGE, not one instant.
    #[test]
    fn test_batch_tov_is_the_range_of_its_samples() {
        let b = batch(&[30, 10, 20]);
        match b.tov() {
            Tov::Range(r) => {
                assert_eq!(r.start, CuTime::from_nanos(10));
                assert_eq!(r.end, CuTime::from_nanos(30));
            }
            other => panic!("expected a range, got {other:?}"),
        }
        assert_eq!(ImuBatch::<4>::new().tov(), Tov::None);
    }
}
