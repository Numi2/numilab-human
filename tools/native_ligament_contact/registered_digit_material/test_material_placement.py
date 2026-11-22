import unittest
import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation
from fit_material_gradients import field_system, unilateral_z
from condition_connected_pose import metric


class MaterialPlacementTests(unittest.TestCase):
    def test_affine_rotation_matches_gradient_and_dual_load(self):
        x = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        t = np.array([[0, 1, 2, 3]])
        rot = Rotation.from_rotvec([0.4, -0.2, 0.1]).as_matrix()
        for protection in (True, False):
            k, f, w = field_system(
                x,
                t,
                np.array([0]),
                np.zeros(4, int),
                rot[None],
                np.array([2.0]),
                protection,
            )
            disp = x @ rot.T - x + np.array([0.6, 0.1, -0.45])
            np.testing.assert_allclose(k @ disp, f, atol=1e-14)
            np.testing.assert_allclose(w.sum(), 1 / 3, atol=1e-14)
            self.assertGreater(metric(x, x + disp, t)[0], 1.0)
        reflected = x.copy()
        reflected[:, 2] *= -1.0
        self.assertLess(metric(x, reflected, t)[0], 0.0)
        stretched = x.copy()
        stretched[:, 0] *= 1.6
        self.assertLess(metric(x, stretched, t)[0], 1.0)

    def test_plane_acts_only_on_penetrating_outer_coordinates(self):
        free = np.array([0, 1, 2])
        f = splu(eye(3, format="csc"))
        y, r = unilateral_z(
            f, free, np.array([-0.04, 0.1, -0.2, 0.3]), np.array([0, 1, 3])
        )
        np.testing.assert_equal(y, [0.0, 0.1, -0.2, 0.3])
        self.assertEqual(r["active"], 1)
        with self.assertRaisesRegex(ValueError, "attachment"):
            unilateral_z(f, free, np.array([0.0, 0.1, -0.2, -0.1]), np.array([0, 1, 3]))


if __name__ == "__main__":
    unittest.main()
