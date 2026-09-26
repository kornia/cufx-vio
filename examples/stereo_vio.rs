//! A headless stereo-VIO graph on synthetic input.
//!
//! `SyntheticStereo` renders a textured scene of three depth bands seen by a stereo camera
//! translating sideways, into pooled GRAY8 eyes; `SyntheticImu` emits matching 200 Hz batches;
//! `StereoVio` tracks in the background; `PoseSink` logs each pose. The graph is
//! `examples/stereo_vio.ron`.
//!
//! ```text
//! cargo run --example stereo_vio -- [ITERATIONS] [LOG_PATH]
//! ```
//!
//! The unified log lands at `LOG_PATH` (default: `stereo_vio.copper` in the system temp dir), with
//! task logging on, so each copperlist holds the frame and the pose computed from it.

use std::path::PathBuf;
use std::sync::Arc;

use cu_stereo_payloads::{
    ImuBatch, ImuPayload, ImuSample, RectifiedStereo, StereoPair, VioPose, gray8_format,
};
use cu29::prelude::*;
use cu29::units::si::f64::Length;
use cu29::units::si::length::meter;

#[copper_runtime(config = "examples/stereo_vio.ron")]
struct App {}

const DEFAULT_ITERATIONS: u64 = 60;
/// 15 fps.
const FRAME_PERIOD: std::time::Duration = std::time::Duration::from_nanos(66_666_667);
const SLAB_SIZE: Option<usize> = Some(64 * 1024 * 1024);

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// Enough handles for the copperlists in flight plus the one the background tracker holds.
const POOL_SLOTS: usize = 16;
const POOL_ID: &str = "stereo-vio-example";
/// Disparity of each horizontal band, top to bottom, in pixels: three depths, so the scene is
/// not a single plane.
const BAND_DISPARITY: [u32; 3] = [16, 24, 32];
/// The camera moves one eighth of the baseline per frame, so a band shifts `disparity / 8`
/// whole pixels per frame and the rendering needs no interpolation.
const SHIFT_DIVISOR: u32 = 8;
/// IMU sample period, 200 Hz.
const IMU_PERIOD_NS: u64 = 5_000_000;
const IMU_BATCH: usize = 32;

fn main() {
    if let Err(error) = run() {
        eprintln!("stereo_vio failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> CuResult<()> {
    let mut args = std::env::args().skip(1);
    let iterations = match args.next() {
        Some(n) => n
            .parse::<u64>()
            .map_err(|e| CuError::new_with_cause("ITERATIONS must be an integer", e))?,
        None => DEFAULT_ITERATIONS,
    };
    let log_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("stereo_vio.copper"));

    let mut app = App::builder()
        .with_log_path(&log_path, SLAB_SIZE)?
        .build()?
        .start()?;
    // `run_one_iteration` is not rate limited (only `run_until_shutdown` honours
    // `rate_target_hz`), so pace the loop at the synthetic camera's frame rate here.
    for _ in 0..iterations {
        let started = std::time::Instant::now();
        app.run_one_iteration()?;
        std::thread::sleep(FRAME_PERIOD.saturating_sub(started.elapsed()));
    }
    app.stop()?;
    println!(
        "stereo_vio: {iterations} iterations, log at {}",
        log_path.display()
    );
    Ok(())
}

/// A deterministic, blocky random texture: 2x2-pixel cells, so ORB finds corners at every
/// pyramid level.
fn texture(x: u32, y: u32) -> u8 {
    let h = (x / 2).wrapping_mul(73_856_093) ^ (y / 2).wrapping_mul(19_349_663);
    (h.wrapping_mul(2_654_435_761) >> 24) as u8
}

/// Stereo source: renders frame `seq` of the translating camera into two pooled eyes.
#[derive(Reflect)]
#[reflect(from_reflect = false)]
pub struct SyntheticStereo {
    #[reflect(ignore)]
    pool: Arc<CuHostMemoryPool<Vec<u8>>>,
    seq: u64,
}

impl Freezable for SyntheticStereo {
    fn freeze<E: cu29::bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), cu29::bincode::error::EncodeError> {
        cu29::bincode::Encode::encode(&self.seq, encoder)
    }

    fn thaw<D: cu29::bincode::de::Decoder>(
        &mut self,
        decoder: &mut D,
    ) -> Result<(), cu29::bincode::error::DecodeError> {
        self.seq = cu29::bincode::Decode::decode(decoder)?;
        Ok(())
    }
}

impl CuSrcTask for SyntheticStereo {
    type Resources<'r> = ();
    type Output<'m> = output_msg!(StereoPair);

    fn new(_config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self> {
        let bytes = gray8_format(WIDTH, HEIGHT).required_bytes();
        let pool = CuHostMemoryPool::new(POOL_ID, POOL_SLOTS, || vec![0; bytes])?;
        Ok(Self { pool, seq: 0 })
    }

    fn process<'o>(&mut self, ctx: &CuContext, output: &mut Self::Output<'o>) -> CuResult<()> {
        let left = self
            .pool
            .acquire()
            .ok_or_else(|| CuError::from("stereo pool exhausted (left eye)"))?;
        let right = self
            .pool
            .acquire()
            .ok_or_else(|| CuError::from("stereo pool exhausted (right eye)"))?;
        let seq = u32::try_from(self.seq).unwrap_or(u32::MAX);
        let band_height = HEIGHT / BAND_DISPARITY.len() as u32;
        left.with_inner_mut(|l| {
            right.with_inner_mut(|r| {
                for y in 0..HEIGHT {
                    let band = ((y / band_height) as usize).min(BAND_DISPARITY.len() - 1);
                    let d = BAND_DISPARITY[band];
                    let shift = seq.wrapping_mul(d / SHIFT_DIVISOR);
                    for x in 0..WIDTH {
                        let i = (y * WIDTH + x) as usize;
                        // A point seen at column u by the left eye is at u - d in the right one.
                        l[i] = texture(x.wrapping_add(shift), y);
                        r[i] = texture(x.wrapping_add(d).wrapping_add(shift), y);
                    }
                }
            })
        });
        let calib = RectifiedStereo {
            fx: 300.0,
            fy: 300.0,
            cx: f64::from(WIDTH) / 2.0,
            cy: f64::from(HEIGHT) / 2.0,
            baseline: Length::new::<meter>(0.1),
        };
        let mut pair = StereoPair::new(WIDTH, HEIGHT, left, right, calib)?;
        pair.left.seq = self.seq;
        pair.right.seq = self.seq;
        self.seq += 1;
        output.set_payload(pair);
        // No sensor clock to read: the render time is the capture time.
        output.tov = Tov::Time(ctx.now());
        Ok(())
    }
}

/// IMU source: a stationary 200 Hz stream, batched per graph cycle, stamped with the batch's
/// sample range.
#[derive(Reflect)]
pub struct SyntheticImu {
    /// Time of the next sample to emit; `None` until the first cycle.
    next: Option<CuTime>,
}

impl Freezable for SyntheticImu {
    fn freeze<E: cu29::bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), cu29::bincode::error::EncodeError> {
        cu29::bincode::Encode::encode(&self.next, encoder)
    }

    fn thaw<D: cu29::bincode::de::Decoder>(
        &mut self,
        decoder: &mut D,
    ) -> Result<(), cu29::bincode::error::DecodeError> {
        self.next = cu29::bincode::Decode::decode(decoder)?;
        Ok(())
    }
}

impl CuSrcTask for SyntheticImu {
    type Resources<'r> = ();
    type Output<'m> = output_msg!(ImuBatch<IMU_BATCH>);

    fn new(_config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self> {
        Ok(Self { next: None })
    }

    fn process<'o>(&mut self, ctx: &CuContext, output: &mut Self::Output<'o>) -> CuResult<()> {
        let now = ctx.now();
        let mut t = self.next.unwrap_or(now);
        let mut batch = ImuBatch::<IMU_BATCH>::new();
        while t <= now && !batch.is_full() {
            let sample = ImuSample {
                tov: t,
                // At rest, camera-style axes: gravity reads along -y.
                imu: ImuPayload::from_raw([0.0, -9.81, 0.0], [0.0, 0.0, 0.0], 25.0),
            };
            batch
                .push(sample)
                .map_err(|e| CuError::new_with_cause("imu batch", e))?;
            t += CuDuration::from_nanos(IMU_PERIOD_NS);
        }
        self.next = Some(t);
        output.tov = batch.tov();
        output.set_payload(batch);
        Ok(())
    }
}

/// Logs each pose the tracker publishes.
#[derive(Reflect)]
pub struct PoseSink {
    poses: u64,
}

impl Freezable for PoseSink {}

impl CuSinkTask for PoseSink {
    type Resources<'r> = ();
    type Input<'m> = input_msg!(VioPose);

    fn new(_config: Option<&ComponentConfig>, _resources: Self::Resources<'_>) -> CuResult<Self> {
        Ok(Self { poses: 0 })
    }

    fn process<'i>(&mut self, _ctx: &CuContext, input: &Self::Input<'i>) -> CuResult<()> {
        let Some(pose) = input.payload() else {
            return Ok(());
        };
        self.poses += 1;
        let m = pose.cam_in_world.to_matrix();
        let (x, y, z) = (m[0][3], m[1][3], m[2][3]);
        // The capture time as a value, not a formatted string: the log macros take it directly,
        // and a per-tick `format!` would allocate on every pose. `StereoVio` stamps a pose with
        // its frame's `tov`, so a range start is the frame's reference time too.
        let tov_ns = match input.tov {
            Tov::Time(t) => t.as_nanos(),
            Tov::Range(r) => r.start.as_nanos(),
            Tov::None => 0,
        };
        let epoch = pose.status.reset_epoch;
        let landmarks = pose.status.landmarks;
        info!(
            "pose tov_ns={} x={} y={} z={} epoch={} landmarks={}",
            tov_ns, x, y, z, epoch, landmarks
        );
        Ok(())
    }

    fn stop(&mut self, _ctx: &CuContext) -> CuResult<()> {
        info!("poses: {} tracked poses received", self.poses);
        Ok(())
    }
}
