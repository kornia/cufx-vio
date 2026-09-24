//! Reads the unified log that the `stereo_vio` example writes.
//!
//! ```text
//! cargo run --example stereo_vio_logreader -- <LOG_PATH> extract-copperlists
//! ```
//!
//! Each copperlist prints the `stereo` output (the frame, with its `tov`) and the `vio` output
//! (the pose, with its own `tov`). The tracker runs in the background, so a pose usually lands in
//! a later copperlist than the frame it was computed from; its `tov` is still that frame's `tov`.

use cu29::prelude::*;
use cu29_export::run_cli;

gen_cumsgs!("examples/stereo_vio.ron");

fn main() {
    if let Err(error) = run_cli::<CuMsgs>() {
        eprintln!("stereo_vio_logreader failed: {error}");
        std::process::exit(1);
    }
}
