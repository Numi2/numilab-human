import unittest
import numpy as np

from posterior_local_relaxation import coloring, relax


class PosteriorRelaxationTests(unittest.TestCase):
    def test_conflicting_vertices_are_separated(self):
        tets = np.array([[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7]])
        free = np.array([1, 2, 4, 5, 7])
        groups = coloring(tets, free)
        self.assertEqual(
            sorted(i for group in groups for i in group[0]), list(range(len(free)))
        )
        for members, cells, corners, owners in groups:
            np.testing.assert_array_equal(tets[cells, corners], free[members][owners])
            self.assertEqual(len(cells), len(set(cells)))

    def test_independent_material_cells_release_strain(self):
        unit = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        source = np.vstack([unit, unit + [4.0, 0.0, 0.0]])
        tet = np.arange(8).reshape(2, 4)
        free = np.array([3, 7])
        y = source.copy()
        y[3] += [0.06, -0.02, 0.12]
        y[7] += [-0.03, 0.04, -0.08]
        y = y.astype("f4").astype(float)
        gradients = np.tile(np.vstack([-np.ones(3), np.eye(3)])[None], (2, 1, 1))
        volume = np.ones(2) / 6
        mu = np.array([5.0, 20.0])
        bulk = np.array([50.0, 30.0])

        def F(position):
            p = position[tet]
            return (p[:, 1:] - p[:, :1]).transpose(0, 2, 1)

        def energy(position):
            T = F(position)
            logj = np.log(np.linalg.det(T))
            strain = (
                0.5 * mu * (np.sum(T * T, axis=(1, 2)) - 3 - 2 * logj)
                + 0.5 * bulk * logj**2
            )
            return float(volume @ strain)

        def gradient(position):
            T = F(position)
            n = np.linalg.inv(T).transpose(0, 2, 1)
            P = (
                mu[:, None, None] * T
                + (bulk * np.log(np.linalg.det(T)) - mu)[:, None, None] * n
            )
            return volume[:, None] * P[:, :, 2]

        def valid(position):
            s = np.linalg.svd(F(position), compute_uv=False)
            return s.min() >= 2 / 3 and s.max() <= 1.5

        result, history = relax(
            y,
            tet,
            free,
            gradients,
            volume,
            mu,
            bulk,
            np.zeros(2),
            energy,
            gradient,
            valid,
        )
        np.testing.assert_array_equal(result[[0, 1, 2, 4, 5, 6]], y[[0, 1, 2, 4, 5, 6]])
        np.testing.assert_allclose(result, source, atol=2e-6, rtol=0)
        self.assertLess(energy(result), energy(y) / 1e8)
        measured = [energy(y)] + [r["energy_J"] for r in history]
        self.assertTrue(np.all(np.diff(measured) <= 1e-12))
        self.assertTrue(np.array_equal(result, result.astype("f4")))


if __name__ == "__main__":
    unittest.main()
