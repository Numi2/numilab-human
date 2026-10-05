import unittest
import numpy as np

from numilab_human.resting_respiratory_conforming_field import conform_surface, kuhn_basis, resolve_short_edges


class ConformingFieldTests(unittest.TestCase):
    def test_reciprocal_faces_get_exact_same_partition(self):
        v = np.array([[.003, -.012, -.033, 0, 0, 1],
                      [.061, .005, -.021, 0, 0, 1],
                      [.018, -.055, .041, 0, 0, 1]], np.float32)
        a, af, _, _ = conform_surface(v, [[0, 1, 2]])
        b, bf, _, _ = conform_surface(v, [[0, 2, 1]])
        def keys(vertices, faces):
            return {tuple(sorted(tuple(float(x) for x in p) for p in vertices[f, :3])) for f in faces}
        self.assertEqual(keys(a, af), keys(b, bf))
        self.assertGreater(len(af), 1)
        # Every child has one affine value throughout its interior, up to
        # the explicit Float32 serialization rounding of the source geometry.
        for face in af:
            p = a[face, :3].astype(float)
            values, _ = kuhn_basis(p)
            center, _ = kuhn_basis(p.mean(axis=0, keepdims=True))
            self.assertLess(abs(float(center[0] - values.mean())), 2e-7)

    def test_common_map_stays_monotonic_along_superior_axis(self):
        p = np.array([[x, y, z] for x in (-.02, .03, .08, .12)
                      for y in np.linspace(-.1, .04, 31) for z in (-.09, -.02, .02, .09)])
        value, gradient = kuhn_basis(p)
        self.assertGreaterEqual(value.min(), -1e-14)
        self.assertLessEqual(value.max(), 1 + 1e-14)
        self.assertLessEqual(gradient[:, 1].max(), 0)
        # For positive diaphragm excursion, the longitudinal derivative of
        # y-D*b is >=1; the shared affine cells cannot invert in Y.
        self.assertGreaterEqual((1 - .03 * gradient[:, 1]).min(), 1)

    def test_closed_tetrahedron_preserves_volume_and_edges(self):
        from numilab_human.resting_anatomy_interface_patch import signed_volume, topology_report
        p = np.array([[.003, -.012, -.033], [.061, .005, -.021],
                      [.018, -.055, .041], [.011, .016, .035]], np.float32)
        f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        v = np.column_stack((p, np.tile([0, 0, 1], (4, 1))))
        out, faces, _, _ = conform_surface(v, f)
        topo = topology_report(faces)
        self.assertEqual(topo['boundary_edge_count'], 0)
        self.assertEqual(topo['nonmanifold_edge_count'], 0)
        self.assertEqual(topo['orientation_error_edge_count'], 0)
        self.assertLess(abs(signed_volume(out[:, :3].astype(float), faces) - signed_volume(p.astype(float), f)), 1e-12)

    def test_explicit_numerical_resolution_keeps_closed_topology_and_bounds_motion(self):
        from numilab_human.resting_anatomy_interface_patch import topology_report
        p = np.array([[.00300017, -.01200029, -.03300041], [.06100023, .00500037, -.02100019],
                      [.01800043, -.05500031, .04100013], [.01100029, .01600011, .03500047]], np.float32)
        f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        v = np.column_stack((p, np.tile([0, 0, 1], (4, 1))))
        _, faces, _, details = conform_surface(v, f, coordinate_resolution_m=1e-6)
        topo = topology_report(faces)
        self.assertEqual(topo['boundary_edge_count'], 0)
        self.assertEqual(topo['nonmanifold_edge_count'], 0)
        self.assertEqual(topo['orientation_error_edge_count'], 0)
        self.assertLess(details['maximum_float32_rounding_distance_m'], .88e-6)
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            conform_surface(v, f, coordinate_resolution_m=.001)

    def test_short_edge_resolution_preserves_reciprocal_closed_surfaces(self):
        from numilab_human.resting_anatomy_interface_patch import topology_report, signed_volume
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [6e-8, 0, 0]], np.float32)
        faces = np.array([[0, 2, 4], [4, 2, 1], [0, 4, 3], [4, 1, 3], [1, 2, 3], [2, 0, 3]])
        v = np.column_stack((p, np.tile([0, 0, 1], (len(p), 1)))).astype(np.float32)
        prepared = {1: (v, faces, {i: [i] for i in range(6)}, {}),
                    2: (v.copy(), faces[:, ::-1], {i: [i] for i in range(6)}, {})}
        report = resolve_short_edges(prepared, 1.25e-7)
        self.assertEqual(report['changed_unique_points'], 1)
        self.assertLessEqual(report['maximum_displacement_m'], 1.25e-7)
        for sid in (1, 2):
            vertices, f, mapping, _ = prepared[sid]
            self.assertEqual(len(f), 4)
            self.assertEqual(mapping[0], [])
            self.assertEqual(mapping[2], [])
            topo = topology_report(f)
            for key in ('boundary_edge_count', 'nonmanifold_edge_count', 'orientation_error_edge_count'):
                self.assertEqual(topo[key], 0)
            self.assertAlmostEqual(abs(signed_volume(vertices[:, :3].astype(float), f)), 1/6)
        np.testing.assert_array_equal(prepared[1][0][:, :3], prepared[2][0][:, :3])
        self.assertEqual({tuple(sorted(f)) for f in prepared[1][1]}, {tuple(sorted(f)) for f in prepared[2][1]})

    def test_short_edge_resolution_rejects_transitive_unbounded_motion(self):
        p = np.array([[i*1e-7, 0, 0] for i in range(5)] + [[0, 1, 0]], np.float32)
        v = np.column_stack((p, np.tile([0, 0, 1], (len(p), 1)))).astype(np.float32)
        f = np.array([[i, i+1, 5] for i in range(4)])
        with self.assertRaisesRegex(ValueError, 'transitive'):
            resolve_short_edges({1: (v, f, {i: [i] for i in range(4)}, {})}, 1.25e-7)
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            resolve_short_edges({}, 1e-6)


if __name__ == '__main__':
    unittest.main()
