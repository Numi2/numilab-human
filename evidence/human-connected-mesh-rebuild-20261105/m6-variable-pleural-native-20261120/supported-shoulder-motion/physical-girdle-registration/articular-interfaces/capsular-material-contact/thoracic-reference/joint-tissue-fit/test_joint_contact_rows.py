import unittest
import numpy as np
from fit_contact_and_cells import project_axes


class ContactAxes(unittest.TestCase):
    def test_selected_plane_was_a_material_separator(self):
        bone = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        tissue = np.array([[[0.2, 0.1, 1.0], [0.3, 0.2, 0.5], [0.2, 0.3, 1.0]]])
        crossed = tissue.copy()
        crossed[0, 1, 2] = -0.1
        normal, valid = project_axes(crossed, bone, tissue, bone)
        self.assertTrue(valid.all())
        p = (tissue @ normal[:, :, None]).min()
        q = (bone @ normal[:, :, None]).max()
        self.assertGreater(p - q, 0)
        self.assertLessEqual((crossed @ normal[:, :, None]).min() - q, 0)

    def test_existing_shared_corner_keeps_material_identity(self):
        a = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
        b = np.array([[[0.0, 0.0, 0.0], [-1.0, 0.0, 0.1], [0.0, -1.0, 0.1]]])
        n, ok = project_axes(a, b, a, b)
        self.assertTrue(ok.all())
        self.assertGreaterEqual(
            float((a @ n[:, :, None]).min() - (b @ n[:, :, None]).max()), -1e-13
        )


if __name__ == "__main__":
    unittest.main()
