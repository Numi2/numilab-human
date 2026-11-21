import unittest

import numpy as np

from integrate_reference import cofactors
from restore_emitted import restore
from transfer_connected_volume import geometry


class EmittedTissueTests(unittest.TestCase):
    def test_determinant_gradient(self):
        points = np.array(
            [
                [
                    [0.02, 0.01, -0.01],
                    [0.09, 0.02, 0.0],
                    [0.0, 0.06, 0.03],
                    [0.01, 0.0, 0.08],
                ]
            ]
        )
        direction = np.random.default_rng(3542).normal(size=points.shape)
        gradient, _ = cofactors(points)
        _, upper = cofactors(points + 1e-7 * direction)
        _, lower = cofactors(points - 1e-7 * direction)
        np.testing.assert_allclose(
            (gradient * direction).sum(axis=(1, 2)),
            (upper - lower) / 2e-7,
            rtol=1e-9,
            atol=1e-11,
        )
        np.testing.assert_allclose(gradient.sum(axis=1), 0.0, atol=1e-15)

    def test_emitted_repair_keeps_disjoint_material(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        x = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.505, 0.5, 0.5],
                [0.5, 0.505, 0.5],
                [0.5, 0.5, 0.50002],
                [0.4, 0.4, 0.4],
                [0.41, 0.4, 0.4],
                [0.4, 0.41, 0.4],
                [0.4, 0.4, 0.41],
            ],
            dtype="f4",
        )
        y = x.copy()
        y[3, 2] = np.nextafter(y[0, 2], np.float32(0))
        t = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertLess(geometry(y, t)[0][0], 0)
        with TemporaryDirectory() as tmp:
            recovered = restore(x.astype(float), y, t, [], Path(tmp) / "cells.npy", 10)
        np.testing.assert_array_equal(recovered, recovered.astype("f4"))
        np.testing.assert_array_equal(recovered[4:], x[4:])
        volume, quality = geometry(recovered, t)
        self.assertTrue(np.all(volume > 1e-18))
        self.assertTrue(np.all(quality > 1e-5))


if __name__ == "__main__":
    unittest.main()
