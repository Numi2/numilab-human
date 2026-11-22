import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from solve_extensor_sites import bind_dofs, step_rows


class ExtensorRegistrationTest(unittest.TestCase):
    def test_serialized_rotations_preserve_each_donors_named_sites(self):
        # A single site permits three rotations. Two sites leave only the
        # rotation about their line. Three noncollinear sites fix the frame.
        base = Rotation.from_rotvec([0.17, -0.51, 0.39])
        rows = [(i, 0.02, -0.04, 0.11, *base.as_quat(), 1.2) for i in range(4)]
        sites = np.array(
            [
                [-0.2, 0.05, 0.01],
                [0.04, 0.05, 0.03],
                [0.06, 0.115, 0.07],
                [0.01, 0.08, 0.025],
                [0.09, 0.08, 0.025],
                [0.09, 0.045, 0.025],
            ]
        )
        owner = np.array([0, 1, 1, 2, 2, 2])
        spec = bind_dofs(rows, sites, owner)
        self.assertEqual([(j, len(basis)) for j, _, basis in spec], [(0, 3), (1, 1)])
        result = step_rows(rows, spec, np.array([0.04, -0.09, 0.025, 0.17]))
        for j, (old, new) in enumerate(zip(rows, result)):
            self.assertEqual(old[0], new[0])
            if j >= 2:
                self.assertEqual(old, new)
                continue
            self.assertNotEqual(old[4:8], new[4:8])

            def local(q):
                return (
                    Rotation.from_quat(q[4:8]).apply(sites[owner == j]) * q[8] + q[1:4]
                )

            np.testing.assert_allclose(local(old), local(new), rtol=0, atol=5e-8)


if __name__ == "__main__":
    unittest.main()
