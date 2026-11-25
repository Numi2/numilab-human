import unittest

import numpy as np

from fit_cavity_carrier_masses import fit_carrier_masses


class CarrierRouting(unittest.TestCase):
    def test_locked_receiver_changes_child_masses_in_both_frames(self):
        centers = np.array(
            [[[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[3, 1, 0], [3, 2, 0], [3, 3, 0]]],
            dtype=float,
        )
        child = fit_carrier_masses(
            [2, 2],
            [[1], [0, 2]],
            [1.5, 1, 1.5],
            [1, 1, 1],
            (0.1, 3),
            [1, 2, 1],
            centers,
            centers,
        )
        np.testing.assert_allclose(child, [1, 2, 1])

    def test_independently_rounded_cell_totals_do_not_starve_carriers(self):
        # Native per-carrier closure is stricter than the whole-cavity mass check.
        supplied = np.array([1e-12, 2.0, 1.0])
        cell_mass = supplied * (1.0 - 3.34e-13)
        centers = np.zeros((2, 3, 3))
        child = fit_carrier_masses(
            supplied,
            [[0], [1], [2]],
            cell_mass,
            supplied,
            (0.5, 1.5),
            cell_mass,
            centers,
            centers,
        )
        np.testing.assert_array_equal(child, supplied)

    def test_moving_frame_cannot_be_dropped_as_redundant(self):
        source = np.array([[[0, 0, 0], [1, 0, 0]], [[0, 0, 0], [2, 0, 0]]], dtype=float)
        child = source.copy()
        child[1, 1, 0] = 1
        self.assertIsNone(
            fit_carrier_masses(
                [1, 1],
                [[0, 1], [0, 1]],
                [1, 1],
                [1, 1],
                (0.001, 3),
                [1, 1],
                source,
                child,
            )
        )

    def test_unreachable_weight_keeps_the_source_cells(self):
        centers = np.zeros((2, 2, 3))
        self.assertIsNone(
            fit_carrier_masses(
                [2, 1],
                [[0], [0]],
                [1.5, 1.5],
                [1, 1],
                (0.5, 3),
                [1, 2],
                centers,
                centers,
            )
        )


if __name__ == "__main__":
    unittest.main()
