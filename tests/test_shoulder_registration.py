import math
import unittest

import numpy as np
from numilab_human.shoulder_registration import (
    fit_articular_sphere,
    preserve_socket_clearance,
    solve_elbow_reach,
)
from scipy.spatial.transform import Rotation


class ShoulderRegistrationTests(unittest.TestCase):
    def test_articular_center_after_translation_with_uneven_areas(self):
        rng = np.random.default_rng(1729)
        directions = rng.normal(size=(150, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        center = np.array([0.03, -0.02, 0.2])
        radius = 0.021
        points = center + radius * directions
        fitted, condition = fit_articular_sphere(
            points, rng.uniform(0.1, 4.0, len(points)), center + 0.002, 0.019
        )
        np.testing.assert_allclose(fitted, [*center, radius], atol=1e-10, rtol=0)
        self.assertLess(condition, 10)

    def test_oblique_segments_close_within_joint_range(self):
        arm = np.array([0.273, 0.012, 0.014])
        forearm = np.array([0.227, -0.006, 0.033])
        axis = np.array([0.6, 0.8, 0.0])
        angle = 0.86
        distance = np.linalg.norm(
            arm + Rotation.from_rotvec(axis * angle).apply(forearm)
        )
        bounds = (0.1 + 4 * math.pi, 1.8 + 4 * math.pi)
        roots, _, reached = solve_elbow_reach(arm, forearm, axis, bounds, distance)
        self.assertTrue(reached)
        self.assertTrue(all(bounds[0] <= q <= bounds[1] for q in roots))
        self.assertTrue(any(abs(q - angle - 4 * math.pi) < 1e-12 for q in roots))
        for q in roots:
            self.assertAlmostEqual(
                np.linalg.norm(arm + Rotation.from_rotvec(axis * q).apply(forearm)),
                distance,
            )

    def test_saturated_reach(self):
        arm = [0.30, 0, 0]
        forearm = [0.24, 0, 0]
        angles, (_, maximum), reached = solve_elbow_reach(
            arm, forearm, [0, 0, 1], (0.0, 2.0), 0.60
        )
        self.assertFalse(reached)
        self.assertEqual(angles, [0.0])
        self.assertAlmostEqual(maximum, 0.54)

    def test_constant_reach(self):
        for target, expected in [(0.5, True), (0.6, False)]:
            roots, limits, reached = solve_elbow_reach(
                [0.3, 0, 0], [0, 0, 0.4], [0, 0, 1], (0.1, 0.9), target
            )
            self.assertEqual(reached, expected)
            self.assertEqual(roots, [0.1])
            np.testing.assert_allclose(limits, [0.5, 0.5])

    def test_socket_gap_and_minimum_translation(self):
        socket = np.array([[0.003, 0, -0.022], [-0.003, 0, -0.022]])
        head = socket + [0, 0, 0.002]
        normals = np.array([[0, 0, 1], [0, 0, 1]])
        delta, reference, gain = preserve_socket_clearance(
            socket, head, normals, np.zeros(3), 1.0, 1.0, 1.05
        )
        np.testing.assert_allclose(delta, [0, 0, 0.001], rtol=0, atol=1e-12)
        np.testing.assert_allclose(reference, 0.002, rtol=0, atol=1e-12)
        np.testing.assert_allclose(gain, 0, rtol=0, atol=1e-12)

    def test_enlarged_head_in_closed_socket_cannot_be_translated_clear(self):
        normals = np.array([[1, 0, 0], [-1, 0, 0]])
        with self.assertRaisesRegex(ValueError, "single rigid translation"):
            preserve_socket_clearance(
                -0.022 * normals, -0.020 * normals, normals, np.zeros(3), 1.0, 1.0, 1.05
            )
