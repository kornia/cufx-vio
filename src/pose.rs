//! Pose-convention conversions.

use cu_spatial_payloads::Pose;
use kornia_3d::pose::Pose3d;

/// Converts the tracker's pose into the camera-in-world transform a consumer wants.
///
/// **The tracker's convention is world-to-camera; a transform wants camera-in-world.**
/// Publishing the un-inverted pose is the classic frustum-flies-backwards bug, and it does not
/// look like a bug: it looks like the camera driving the path in reverse.
pub fn cam_in_world(pose_world_to_cam: &Pose3d) -> Pose<f64> {
    let inv = pose_world_to_cam.inverse();
    // `Mat3F64` is column-major; `Pose::from_matrix` takes `mat[row][column]`.
    let cols = [
        inv.rotation.x_axis().to_array(),
        inv.rotation.y_axis().to_array(),
        inv.rotation.z_axis().to_array(),
    ];
    let t = inv.translation.to_array();
    let row = |r: usize| [cols[0][r], cols[1][r], cols[2][r], t[r]];
    Pose::from_matrix([row(0), row(1), row(2), [0.0, 0.0, 0.0, 1.0]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_algebra::{Mat3F64, Vec3F64};

    /// A camera at (1, 2, 3) yawed +90 deg about world Z. The inverse must put the translation
    /// back at the camera centre and transpose the rotation, row by row.
    #[test]
    fn test_cam_in_world_inverts_and_keeps_row_major_order() {
        // world-to-camera rotation: yaw -90 deg about Z (columns listed).
        let r_wc = Mat3F64::from_cols(
            Vec3F64::new(0.0, -1.0, 0.0),
            Vec3F64::new(1.0, 0.0, 0.0),
            Vec3F64::new(0.0, 0.0, 1.0),
        );
        let centre = Vec3F64::new(1.0, 2.0, 3.0);
        let t_wc = -(r_wc * centre);
        let m = cam_in_world(&Pose3d::new(r_wc, t_wc)).to_matrix();
        let eps = 1e-12;
        let want = [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        for (r, (got, want)) in m.iter().zip(want).enumerate() {
            for c in 0..4 {
                assert!((got[c] - want[c]).abs() < eps, "[{r}][{c}]: {m:?}");
            }
        }
    }
}
