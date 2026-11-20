"""Verify vectorized plane selection and finite support of fixed bones."""
import unittest

import numpy as np

import rebuild  # installs the native authoring tool paths
from human_skin_contact_planes import finite_triangle_support_planes
from skull_contact import SkullContact, finite_outward_planes
from solid_skull import SolidSkull


class FinitePlaneTest(unittest.TestCase):
    def test_stationary_parity_with_scalar_owner(self):
        rng = np.random.default_rng(83)
        a, b = 0.01 * rng.normal(size=(2, 128, 3, 3))
        a += [0.2, -1.1, 0.5]
        b += [0.2, -1.1, 0.5]
        gaps, velocities, normals = finite_outward_planes(a, b, np.zeros_like(a), 0.)
        np.testing.assert_array_equal(velocities, np.zeros_like(velocities))
        expected = []
        for one, two in zip(a, b, strict=True):
            outward = np.cross(two[1] - two[0], two[2] - two[0])
            outward /= np.linalg.norm(outward)
            selected = next(option for option in finite_triangle_support_planes(one, two, outward)
                            if option["normal"] @ outward >= 0.15)
            expected.append(selected["gap_mm"] * 1.e-3)
        np.testing.assert_allclose(gaps.min(axis=1), expected, atol=1.e-16, rtol=1.e-13)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.)

    def test_same_bank_survives_disjoint_trial_and_replay(self):
        rebuild.install_for_float32()
        points = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.]], dtype=np.float32)
        bank = np.zeros(1, dtype=rebuild.TRIANGLE_ROW)
        bank["vertices"][0] = [[.015, .015, -.01], [.015, .015, .01], [.035, .025, .002]]
        contact = SkullContact(np.array([[0, 1, 2]]), [42], ["deep"], bank)
        first = contact.measure(points)
        self.assertEqual(first.counts["deep"], 1)
        translated = points + np.array([.3, 0., 0.], dtype=np.float32)
        self.assertEqual(contact.measure(translated).counts["deep"], 0)
        last = contact.measure(points)
        np.testing.assert_array_equal(first.face_rows, last.face_rows)
        np.testing.assert_array_equal(first.triangle_rows, last.triangle_rows)
        self.assertEqual(first.counts, last.counts)

    def test_new_containment_and_boundary_are_material_regressions(self):
        rebuild.install_for_float32()
        p = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
        bank = np.zeros(4, dtype=rebuild.TRIANGLE_ROW)
        bank["vertices"] = p[[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]]
        old = np.array([[.2, .2, .7], [.2, .2, .0], [.2, .2, .2]], dtype=np.float32)
        solid = SolidSkull(bank, np.arange(3), old)
        self.assertEqual(solid.newly_unsafe(old, np.arange(3)), [])
        trial = np.array([[.25, .25, .5], [.2, .2, .2], [.25, .25, .25]], dtype=np.float32)
        self.assertEqual(solid.newly_unsafe(trial, np.arange(3)), [0, 1])

    def test_affine_constraints_cover_every_corner(self):
        rng = np.random.default_rng(832)
        a, b = 0.015 * rng.normal(size=(2, 96, 3, 3))
        a += [0.2, -1.1, 0.5]
        b += [0.2, -1.1, 0.5]
        motion = rng.normal(size=a.shape)
        motion /= np.linalg.norm(motion, axis=-1, keepdims=True)
        motion[::2, 0] = 0.
        delta = 0.002 * rng.uniform(-1., 1., a.shape[:2])
        gaps, velocities, normals = finite_outward_planes(a, b, motion, .002)
        moved = a + delta[:, :, None] * motion
        expected = (np.einsum("pd,pvd->pv", normals, moved)
                    - np.einsum("pd,pvd->pv", normals, b).max(axis=1)[:, None])
        np.testing.assert_allclose(gaps + velocities * delta, expected, atol=1.e-15, rtol=1.e-13)


if __name__ == "__main__":
    unittest.main()
