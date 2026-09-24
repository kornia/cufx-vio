//! Errors the copper task raises, and the single boundary conversion to cu29's error type.
//!
//! Separate from [`VioError`](crate::error::VioError), which belongs to the tracker (no copper graph types):
//! these are the failures of DRIVING the tracker from a graph, such as a payload whose image
//! size changed mid-stream or a message with no time of validity.

use cu29::prelude::CuError;

/// One eye of a stereo pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eye {
    /// The left, reference eye.
    Left,
    /// The right eye.
    Right,
}

impl std::fmt::Display for Eye {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Left => "left",
            Self::Right => "right",
        })
    }
}

/// A failure while driving the tracker from a copper graph.
#[derive(Debug, thiserror::Error)]
pub enum TaskError {
    /// The tracker rejected a frame.
    #[error("tracker: {0}")]
    Vio(#[from] crate::error::VioError),

    /// An eye's size disagrees with the geometry the tracker was built for. Checked rather than
    /// trusted: a stride mistake produces a sheared image the matcher still consumes.
    #[error("frame {seq}: {eye} eye has {got} bytes, but {width}x{height} needs {want}")]
    FrameShape {
        /// Frame sequence number.
        seq: u64,
        /// Which eye.
        eye: Eye,
        /// Bytes actually present.
        got: usize,
        /// Bytes required.
        want: usize,
        /// Expected width.
        width: u32,
        /// Expected height.
        height: u32,
    },

    /// A frame's calibration differs from the first frame's, which the tracker was built from.
    /// The camera model is fixed for the tracker's lifetime, so continuing would track new
    /// pixels against the old intrinsics and baseline.
    #[error("frame {seq}: calibration differs from the first frame's; the tracker keeps its first")]
    CalibrationChanged {
        /// Frame sequence number.
        seq: u64,
    },

    /// The stereo message carried `Tov::None`. The tracker needs a capture time per frame, and
    /// only the source knows it.
    #[error("frame {seq} has no time of validity; the source must stamp `tov`")]
    Untimed {
        /// Frame sequence number.
        seq: u64,
    },

    /// Building an image from a payload buffer failed.
    #[error("image: {0}")]
    Image(#[from] kornia_image::ImageError),
}

impl TaskError {
    /// The single boundary conversion to cu29's error type.
    pub fn into_cu_error(self, context: &str) -> CuError {
        CuError::new_with_cause(context, self)
    }
}
