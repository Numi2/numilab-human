"""Verify the material/bed Newton derivatives and restored seed forces."""
import unittest

import numpy as np

from predict import ip_contact, read_model, match_ipc_seed
from mattress import Mattress, projected_area


class SeedPredictionTests(unittest.TestCase):
    def check_directional_tangent(self, lateral):
        model, x, *_ = read_model(lateral)
        rng = np.random.default_rng(118)
        direction = rng.normal(size=model.n)
        direction[model.q] *= .1
        direction /= np.linalg.norm(direction)
        e, gradient, _, _, hessian = model.state(x, .95, hessian=True)
        step = 6.e-9
        plus = model.state(x + step * direction, .95)
        minus = model.state(x - step * direction, .95)
        slope = (plus[0] - minus[0]) / (2. * step)
        numeric = (plus[1] - minus[1]) / (2. * step)
        exact = hessian @ direction
        self.assertLess(abs(slope - gradient @ direction), 1.e-5)
        self.assertLess(np.linalg.norm(numeric - exact) / np.linalg.norm(exact), 2.e-6)
        self.assertLess(np.linalg.norm((hessian - hessian.T) @ direction), 1.e-6)
        self.assertTrue(np.isfinite(e))

    def test_normal_directional_tangent(self):
        self.check_directional_tangent(None)

    def test_lateral_directional_tangent(self):
        self.check_directional_tangent(.03)

    def test_warm_mapping_restores_normal_force_at_native_width(self):
        model, x, *_ = read_model()
        x[model.q[0]] -= 2. * model.width
        gap = model.positions(x)[1]
        penalty = 2.e7
        desired = -penalty * np.minimum(0., gap)
        result = match_ipc_seed(model, x, penalty)
        final_gap = model.positions(result)[1]
        _, force, _ = ip_contact(final_gap, model.width, model.mass, model.dt)
        self.assertTrue(np.all(final_gap > 0.))
        np.testing.assert_allclose(force, desired, rtol=1.e-10, atol=1.e-8)

    def test_projected_area_counts_only_the_downward_face(self):
        world = np.array(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
        area = projected_area(world, np.array([[0, 1, 2, 3]]), 3)
        np.testing.assert_allclose(area, np.full(3, 1./6.))

    def test_compliant_bed_balance_energy_and_tangent(self):
        bed = Mattress(np.array((.006, .013, .002)), 1.e4, .12)
        mass = np.array((.01, .02, .03))
        during = np.array((-.04, -.013, .001))
        width, dt = 1.e-5, 1./64000.
        potential, normal, stiffness = bed(during, width, mass, dt)
        gaps, active = bed.clearance_for_load(during, width, mass, dt)
        np.testing.assert_allclose(gaps - normal / bed.k, during, atol=1.e-9, rtol=0.)
        np.testing.assert_array_equal(active, [True, True, False])
        self.assertTrue(np.all(gaps > 0.))
        self.assertEqual(potential[-1], 0.)
        step = 1.e-5
        high_e, high_normal, _ = bed(during + step, width, mass, dt)
        low_e, low_normal, _ = bed(during - step, width, mass, dt)
        np.testing.assert_allclose(-(high_e - low_e) / (2. * step), normal, atol=.001, rtol=5.e-5)
        np.testing.assert_allclose(-(high_normal - low_normal) / (2. * step), stiffness, atol=.04, rtol=5.e-5)


if __name__ == "__main__":
    unittest.main()
