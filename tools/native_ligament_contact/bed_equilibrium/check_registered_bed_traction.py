"""Finite element virtual work, first moments, clipping and infeasible footprint checks."""
import numpy as np
from registered_bed_traction import footprint, pressure_minimax, loads_on_surface, visible_samples
x = np.array([[0., 0., -1.], [2., 0., 1.], [0., 2., 1.]])
tri = np.array([[0, 1, 2]])
ids, areas, bary, pos, nodes = sample = footprint(x, tri, np.arange(3))
np.testing.assert_allclose(areas.sum(), .5)
np.testing.assert_allclose((areas @ pos[:, :2]) / areas.sum(), [1/3, 1/3])
np.testing.assert_allclose(bary.sum(1), 1.)
assert not footprint(x + [0., 0., 2.], tri, np.arange(3))[0].size
assert pressure_minimax(pos[:, :2], areas, np.array([5.,5.]), 12.) is None
rows, loads = loads_on_surface(sample, np.array([[1/3, 1/3, 0.]]), np.array([12.]))
assert rows[0]['moment_feasible']
np.testing.assert_allclose(rows[0]['minimum_peak_pressure_Pa'], 24.)
ind, weights, forces = loads[0]
nodal = np.bincount(ind.ravel(), (weights*forces[:, None]).ravel(), minlength=3)
np.testing.assert_allclose(nodal.sum(), 12.)
np.testing.assert_allclose(nodal @ x[:, :2], [4.,4.])
velocity = np.array([.31, -.46, .8])
np.testing.assert_allclose(nodal @ velocity, forces @ (weights @ velocity))
reverse = footprint(x, tri[:, ::-1], np.arange(3))
np.testing.assert_allclose(reverse[1].sum(), .5)
print('clipping, winding, force, two moments, discrete virtual work, open bed and unreachable center passed')


def stacked_hits(points, xyz, faces):
    return np.array([(i, xyz[face[0], 2], j)
                     for i in range(len(points)) for j, face in enumerate(faces)])


flat = np.array([[0., 0., -1.], [2., 0., -1.], [0., 2., -1.]])
faces = np.arange(6).reshape(2, 3)
for separation, visible in [(0., 0), (-1.e-10, 1)]:
    points = np.vstack((flat, flat + [0., 0., separation]))
    raw = footprint(points, faces, np.arange(6))
    selected = visible_samples(raw, points, faces, stacked_hits)
    np.testing.assert_array_equal(selected[0], np.full(3, visible))
    np.testing.assert_allclose(selected[1].sum(), 2.)
print('coincident and narrowly separated bed faces counted only once')
