//! Copper tasks wrapping [kornia-slam](https://github.com/kornia/kornia-slam) stereo VIO.
//!
//! Deliberately separable layers:
//!
//! * [`track`]: the composition root over kornia-slam. kornia-slam ships stereo matching, the
//!   map, map-projection PnP and bundle adjustment, but its orchestration lives in an
//!   unreachable example binary; this is that orchestration, written for a live camera. It uses
//!   no copper runtime or graph types (no `CuMsg`, no tasks), only the cu29 unit types, `CuTime`
//!   and the logging macros, so it runs without a graph.
//! * [`task`]: the `CuTask` shell that drives it from a copper graph. It owns only the
//!   payload-to-image conversion and the world-to-camera inversion, and must add no estimation
//!   logic of its own: anything it computed would be a second implementation to keep in step
//!   with any offline replay of [`track`].
//! * [`imu`]: the inertial ingest: the stamp-indexed sample ring, the coverage gates every
//!   interval must pass before it may be preintegrated, and the configuration the inertial
//!   path refuses to run without. Likewise free of copper runtime and graph types.
//! * [`bus`], [`feed`], [`imu_channel`], [`reset`]: what reaches a BACKGROUNDED tracker without
//!   a second input edge. A [`VioBus`] resource bundle holds an [`ImuQueue`] that the inline
//!   [`ImuFeed`] sink fills from the graph's IMU edge, and a [`ResetEpoch`] any bound task can
//!   bump. One bus per estimator, so two estimators in one process stay independent.
//!
//! That split is the reason this crate can be reviewed as a copper component rather than as a
//! slice of an application: the estimation is testable without a graph, and the graph binding
//! is small enough to read in one sitting.

#![deny(missing_docs)]

pub mod bus;
pub mod config;
pub mod error;
pub mod feed;
pub mod imu;
pub mod imu_channel;
pub mod pose;
pub mod reset;
mod stats;
pub mod task;
pub mod task_error;
pub mod track;

pub use bus::VioBus;
pub use config::OnTrackingLoss;
pub use error::VioError;
pub use feed::ImuFeed;
pub use imu::{ImuGates, ImuWindowError, InertialConfig, InertialStats};
pub use imu_channel::ImuQueue;
pub use reset::ResetEpoch;
pub use task::{DEFAULT_LANDMARK_BUFFERS, MAX_LANDMARK_BUFFERS, MAX_LANDMARKS, StereoVio};
pub use task_error::{Eye, TaskError};
pub use track::{TrackStatus, TrackedPose, Tracker, TrackerConfig, TrackerStats};
