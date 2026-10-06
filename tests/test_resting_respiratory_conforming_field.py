import unittest
import numpy as np

from numilab_human.resting_respiratory_conforming_field import (
    conform_surface, derive_basal_effective_area, kuhn_basis,
    propagate_exact_source_coordinate_updates, resolve_short_edges,
)


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

    def test_exact_source_coordinate_updates_propagate_to_all_shared_copies(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        normals = np.tile([0, 0, 1], (3, 1)).astype(np.float32)
        source_before = {
            1: {'vertices6': np.column_stack((points, normals)), 'faces': np.array([[0, 1, 2]])},
            2: {'vertices6': np.column_stack((points[[0]], normals[[0]])), 'faces': np.empty((0, 3), np.int32)},
        }
        source_after = {sid: {'vertices6': row['vertices6'].copy(), 'faces': row['faces'].copy()}
                        for sid, row in source_before.items()}
        source_after[1]['vertices6'][0, 1] = np.float32(2e-7)
        source_after[2]['vertices6'][0, 1] = np.float32(2e-7)
        target = {9: {'vertices6': np.column_stack((points, normals)),
                      'faces': np.array([[0, 1, 2]], np.int32)}}
        updated, report = propagate_exact_source_coordinate_updates(
            source_before, source_after, target, (9,), maximum_target_displacement_m=3e-7)
        self.assertEqual(updated[9]['faces'].tolist(), target[9]['faces'].tolist())
        np.testing.assert_array_equal(updated[9]['vertices6'][0, :3], source_after[1]['vertices6'][0, :3])
        np.testing.assert_array_equal(updated[9]['vertices6'][0, 3:6], np.array([0, 0, 1], np.float32))
        self.assertEqual(report['changed_target_vertex_occurrences'], 1)
        self.assertTrue(report['target_faces_preserved'])
        self.assertTrue(report['requires_existing_mesh_compiler'])

    def test_exact_source_coordinate_updates_reject_shared_class_split(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        normals = np.tile([0, 0, 1], (3, 1)).astype(np.float32)
        row = {'vertices6': np.column_stack((points, normals)), 'faces': np.array([[0, 1, 2]])}
        source_before = {1: row, 2: {'vertices6': row['vertices6'][:1].copy(), 'faces': np.empty((0, 3), np.int32)}}
        source_after = {sid: {'vertices6': value['vertices6'].copy(), 'faces': value['faces'].copy()}
                        for sid, value in source_before.items()}
        source_after[2]['vertices6'][0, 2] = np.float32(1e-7)
        with self.assertRaisesRegex(ValueError, 'shared source coordinate class was split'):
            propagate_exact_source_coordinate_updates(source_before, source_after, {}, ())

    def test_exact_source_coordinate_updates_reject_class_merge_and_motion_over_bound(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        normals = np.tile([0, 0, 1], (3, 1)).astype(np.float32)
        source_before = {1: {'vertices6': np.column_stack((points, normals)),
                             'faces': np.array([[0, 1, 2]])}}
        source_after = {1: {'vertices6': source_before[1]['vertices6'].copy(),
                            'faces': source_before[1]['faces'].copy()}}
        source_after[1]['vertices6'][1, :3] = source_after[1]['vertices6'][0, :3]
        with self.assertRaisesRegex(ValueError, 'merged distinct coordinate classes'):
            propagate_exact_source_coordinate_updates(source_before, source_after, {}, ())
        source_after[1]['vertices6'] = source_before[1]['vertices6'].copy()
        source_after[1]['vertices6'][0, 1] = np.float32(2e-6)
        with self.assertRaisesRegex(ValueError, 'exceeds its declared displacement bound'):
            propagate_exact_source_coordinate_updates(
                source_before, source_after, {9: source_before[1]}, (9,), maximum_target_displacement_m=1e-6)

    def test_basal_effective_area_helper_matches_closed_tetrahedron_derivative(self):
        from numilab_human.resting_anatomy_interface_patch import signed_volume
        p = np.array([[.055, -.085, .001], [.075, -.045, .001],
                      [.035, -.045, .021], [.055, -.045, -.019]], np.float32)
        f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])[:, ::-1]
        row = {'vertices6': np.column_stack((p, np.tile([0, 0, 1], (4, 1)))),
               'faces': f}
        report = derive_basal_effective_area({305: row}, (305,), displacement_m=.01)
        points = p.astype(np.float64)
        weight, _ = kuhn_basis(points)
        moved = points.copy()
        moved[:, 1] -= .01 * weight
        expected = (signed_volume(moved, f) - signed_volume(points, f)) / .01
        self.assertAlmostEqual(report['effective_area_m2'], expected, places=14)
        self.assertGreater(report['effective_area_m2'], 0)
        self.assertEqual(len(report['per_lobe']), 1)

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
