# cufx-vio

[Copper](https://github.com/copper-project/copper-rs) tasks wrapping
[kornia-slam](https://github.com/kornia/kornia-slam) stereo visual-inertial odometry.

A stereo source feeds `StereoVio`, which runs in the background and publishes one camera pose per
solved frame. IMU samples reach it through a shared resource bundle, so the tracker keeps a single
input edge and can run with `background: true`.

## Crates

- **`cufx-vio`**: the tasks (`StereoVio`, `ImuFeed`) and the `VioBus` resource bundle.
- **`cufx-sensor-payloads`**: the wire types alone. It depends on cu29, the upstream copper
  payload crates, bincode and serde and nothing else, so a stereo driver can name the types it
  produces without depending on `cufx-vio` and, through it, on kornia-slam. That is the whole
  reason it is a separate crate rather than a module.

Inside `cufx-vio` there are two layers:

- **`track`**: the composition root over kornia-slam. kornia-slam ships stereo matching, the
  map, map-projection PnP and bundle adjustment; this is the orchestration over them, written for
  a live camera. It uses no copper runtime or graph types (no `CuMsg`, no tasks), only the cu29
  unit types, `CuTime` and the logging macros, so the estimation is testable without a graph.
- **`task`**: the `CuTask` shell. It owns only the payload-to-image conversion and the
  world-to-camera inversion and adds no estimation logic of its own. Anything it computed would be
  a second implementation to keep in step with `track`.

## Payload types

All from `cufx-sensor-payloads`. Time rides on the message `tov`, never in the payload.

| Type | Carries | `tov` |
|---|---|---|
| `StereoPair` | two `CuImage` eyes (`CuHandle`-backed buffers) plus `RectifiedStereo`: intrinsics and baseline of the rectified pair | capture time of the pair |
| `ImuBatch<N>` | up to `N` `ImuSample`s (an upstream `cu_sensor_payloads::ImuPayload` plus its own capture time), oldest first, a cumulative `dropped` counter and a `camera_aligned` flag | `ImuBatch::tov()`, a `Tov::Range` over the samples |
| `VioPose` | a `cu_spatial_payloads::Pose<f64>` (camera in world) plus `VioStatus`: keyframe flag, reset epoch, landmark count; and `landmarks`, an optional map snapshot, `Option<CuHandle<Vec<Landmark>>>`. An untracked frame carries no payload | the `tov` of the frame it was computed from |
| `Landmark` | one map point: world-frame `position: [f32; 3]` in metres and its map `index: u32` | element of a `VioPose` snapshot, no time of its own |

`Landmark::index` is a `u32`, not a float in a fourth column: a consumer accumulates the map by
upserting on it, and an `f32` is exact only to 2^24, past which two points would silently merge.

The eye geometry rides on `StereoPair`, so `StereoVio` is never told what the source already
knows: the tracker is built on the first frame, and every later frame must match its size.

## Using it

Neither crate is on crates.io yet (kornia-slam is consumed from git), so depend on the repo. Name
both crates with the identical spec string; see [the spec-string rule](#the-spec-string-rule).

```toml
[dependencies]
cufx-vio = { git = "https://github.com/kornia/cufx-vio", branch = "main" }
cufx-sensor-payloads = { git = "https://github.com/kornia/cufx-vio", branch = "main" }
cu29 = { git = "https://github.com/copper-project/copper-rs", rev = "fe2061dc10539868334f6ded55a9e75feb0f2b62" }
```

A stereo driver that only produces frames needs `cufx-sensor-payloads` alone. It must fill a
`StereoPair` with rectified GRAY8 eyes and stamp the message `tov` with the capture time; an
untimed frame (`Tov::None`) is refused.

## Wiring

```ron
resources: [ ( id: "bus", provider: "cufx_vio::VioBus" ) ],
tasks: [
    ( id: "imu_feed", type: "cufx_vio::ImuFeed<32>", resources: { "imu": "bus.imu" } ),
    ( id: "vio", type: "cufx_vio::StereoVio", background: true,
      config: { "on_tracking_loss": "reset_map", "max_keyframes": 120 },
      resources: { "imu": "bus.imu", "epoch": "bus.reset_epoch" } ),
],
cnx: [
    ( src: "stereo", dst: "vio", msg: "cufx_sensor_payloads::StereoPair" ),
    ( src: "imu", dst: "imu_feed", msg: "cufx_sensor_payloads::ImuBatch<32>" ),
    ( src: "vio", dst: "poses", msg: "cufx_sensor_payloads::VioPose" ),
],
```

- **`VioBus`** is a resource bundle with two members: `imu`, a bounded queue of IMU samples, and
  `reset_epoch`, a counter a caller bumps with `ResetEpoch::request` to drop the map on the next
  frame. Each bus instance is independent, so two estimators in one graph need two buses.
- **`ImuFeed<N>`** is an inline sink that pushes each `ImuBatch<N>` into the bus queue. Its `N`
  must match the source's batch capacity. The queue is armed only when `StereoVio` has an
  `inertial` config block; without one the push is a single atomic load.
- **`StereoVio`** takes a `StereoPair` and emits a `VioPose`, draining the bus queue once per
  frame. Unknown config keys are refused at startup rather than ignored. The recognised keys are
  `on_tracking_loss` (`keep_map`, the default, or `reset_map` for a long-running graph),
  `max_keyframes`, `orb_keypoints`, `search_radius_px`, `max_covisible_keyframes`,
  `pnp_lm_iterations`, `inertial` and `landmark_buffers`; `cufx_vio::config::CONFIG_KEYS` lists
  them. Its `Freezable` is a no-op (the kornia-slam map has no
  serialised form), so a resim of a `background: true` graph reproduces it only from the start of
  a log; see Known limitations.

### Landmarks

`StereoVio` attaches a map snapshot to the pose of every frame that inserted a keyframe, and to
no other: a keyframe is when points are created, culled and moved by local bundle adjustment,
while a plain tracking frame only reads the map. The snapshot holds the live points among the
newest `MAX_LANDMARKS` (8192) map slots, so its cost does not grow with the map. A consumer
builds the whole map by upserting on `Landmark::index` and clears what it holds when
`VioStatus::reset_epoch` changes, since a new map restarts the indices, and rebuilds from the
next snapshot of the new epoch.

The buffers come from a `CuHostMemoryPool` owned by the task, `landmark_buffers` of them
(default 16, 128 KiB each). A snapshot stays checked out while anything holds its handle; when
none is free the pose is published without one, and the stop line counts how often.

Why the IMU does not ride a second input: `CuAsyncTask`, which `background: true` wraps a task
in, accepts a single input message. And while a solve is running the background task refuses the
arriving input, so an IMU batch bundled into the frame payload would be dropped along with most
frames. Measured on a Jetson Orin with an OAK-D at 640x400, a 596 ms p50 solve against a 66 ms
frame interval refuses roughly nine frames in ten.

`examples/stereo_vio.ron` is a complete graph on synthetic input:

```sh
cargo run --release --example stereo_vio -- [ITERATIONS] [LOG_PATH]
cargo run --release --example stereo_vio_logreader -- <LOG_PATH> extract-copperlists
```

The second command dumps each copperlist as JSON: the `stereo` frame and the `vio` pose, each with
its `tov`. Because the tracker runs in the background, a pose lands one or more copperlists after
the frame it was computed from, and its `tov` is that frame's `tov`.

## Inertial path

Off by default: without an `inertial` block `StereoVio` runs stereo-only and the IMU queue is
never armed. To enable it, add the block to the `vio` config. Every field is required and none
has a default, because a guessed extrinsic or noise density yields a plausible but wrong map:

```ron
config: {
    "inertial": (
        t_bc_rotation: [/* R_BC, row-major, rectified-left-camera axes into IMU axes */],
        t_bc_translation: [/* lever arm in metres, IMU origin relative to that camera */],
        gyro_noise: 0.0,        // rad/s/sqrt(Hz), must be > 0
        accel_noise: 0.0,       // m/s^2/sqrt(Hz), must be > 0
        gyro_bias_noise: 0.0,   // rad/s^2/sqrt(Hz), must be > 0
        accel_bias_noise: 0.0,  // m/s^3/sqrt(Hz), must be > 0
        rate_hz: 200.0,
    ),
},
```

The rustdoc of `cufx_vio::config` explains each field and why no default is defensible.

## Documentation and development

```sh
cargo doc --workspace --no-deps --open   # API docs; every public item is documented
cargo test --workspace                   # unit tests, including golden wire-format fixtures
cargo clippy --workspace --all-targets -- -D warnings
```

## The spec-string rule

There is **no `[patch]` table**. Cargo keys a git source on the exact spec STRING, not on the
resolved commit, so two spellings of one dependency are two packages at the same sha, and the
type error names the same path twice. It is the most expensive mistake available here.

It applies in three directions:

- **copper**: every copper-rs crate (`cu29`, `cu-sensor-payloads`, `cu-spatial-payloads`) is
  pinned by the same `rev =`. A consumer must spell that identical rev, or it gets a second
  `CuMsg` and a second `CuImage`.
- **kornia**: every consumer must spell `branch = "main"` identically for the four kornia-rs
  crates, and `branch = "develop"` for `kornia-slam` AND `kornia-sensors` (the latter is a
  workspace member of the kornia-slam repo, not of kornia-rs).
- **this repo's own two crates**: a consumer naming both `cufx-vio` and `cufx-sensor-payloads`
  must give them the identical URL and the identical `branch = "main"`. `cufx-vio` reaches the
  payload crate by path, so it inherits the git source it was itself pulled from; pinning the
  payload line with `rev =` against a `branch =` on the other splits `StereoPair` in two. The
  instinct to pin a payload crate is exactly the trap here.

`Cargo.lock` is tracked so that a clean clone builds: it holds a kornia-rs / kornia-slam pair
known to compile together. A consumer's own lock still wins.

## Known limitations

- **Background only.** `StereoVio` is designed for `background: true`. A keyframe insertion costs
  0.6-1.0 s, so inline it stalls every other task in the graph for that long.
- **No map persistence or resim.** The kornia-slam map has no serialised form, so `StereoVio`'s
  `Freezable` is a no-op. A resim reproduces the task only from the start of a log, never from a
  mid-log keyframe, and a map cannot be saved or reloaded across runs.
- **IMU replay is approximate.** Samples cross into the background worker through the bus queue,
  not a logged input of the tracker, so which samples a given solve saw depends on when the worker
  ran. The `ImuFeed` edge is logged, but a replay is not bit-identical on the inertial path.
- **The map is unbounded unless told otherwise.** kornia-slam never culls keyframes, and with
  neither `max_keyframes` nor `on_tracking_loss: "reset_map"` set the map grows until per-frame
  cost stalls the tracker. Set `max_keyframes` on any long-running graph; the map is then dropped
  and rebuilt when it reaches the bound.

## License

Apache-2.0; see [LICENSE](LICENSE). [NOTICE](NOTICE) credits ORB-SLAM3 (Campos et al., IEEE
T-RO 2021) as the algorithmic reference for the tracking pipeline design.
