import unittest
import numpy as np

from numilab_human.resting_respiratory_mesh_quality import (
    collapse_registered_interior_vertex,
    improve_csg_cut_fragment_faces,
    improve_registered_interior_slivers,
    improve_registered_shared_vertex_stars,
    improve_registered_surface_sliver_faces,
    improve_sliver_faces,
    open_registered_shared_vertex_star,
)


def prepared_quad():
    p = np.array([[.008, .003, .001], [.012, .003, .001],
                  [.010, .00300001, .001], [.010, .002, .001]], np.float32)
    vertices = np.column_stack((p, np.tile([0, 0, 1], (4, 1)))).astype(np.float32)
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    return vertices, faces, {0: [0], 1: [1]}, {}


class RespiratoryMeshQualityTests(unittest.TestCase):
    def test_registered_sliver_pass_selects_nested_owner_edge(self):
        from numilab_human.resting_anatomy_interface_patch import topology_report

        scale = np.float32(50e-6)
        points = np.asarray([[0, 0, 0], [1, 0, 0], [1.5, 1e-5, 0],
                             [0, 1, 0], [-1, 0, 0], [0, 0, -2]], dtype=np.float32) * scale
        faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
                            [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]], dtype=np.int64)
        vertices = np.column_stack((points, np.tile([0, 0, 1], (len(points), 1)))).astype(np.float32)
        center = vertices[:, :3].mean(axis=0)
        for index, face in enumerate(faces):
            tri = vertices[face, :3].astype(float)
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if np.dot(normal, tri.mean(axis=0) - center) < 0:
                faces[index, [1, 2]] = faces[index, [2, 1]]
        tetra_points = np.asarray([points[0], points[3], points[4], [0, 0, float(scale)]], dtype=np.float32)
        tetra_faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
        tetra_vertices = np.column_stack((tetra_points, np.tile([0, 0, 1], (4, 1)))).astype(np.float32)
        trial = {
            305: (tetra_vertices, tetra_faces, {i: [i] for i in range(len(tetra_faces))}, {}),
            308: (vertices.copy(), faces.copy(), {i: [i] for i in range(len(faces))}, {}),
            310: (vertices.copy(), faces.copy(), {i: [i] for i in range(len(faces))}, {}),
        }
        report = improve_registered_interior_slivers(
            trial, minimum_altitude_m=1e-3,
            maximum_source_displacement_m=100e-6,
            maximum_volume_change_m3=1e-9,
            volume_bounded_surface_ids=(308, 310),
            volume_bounded_surface_groups=((308, 310),),
            same_winding_surface_pairs={(308, 310)},
            maximum_operations=1)
        self.assertEqual(report['accepted_operation_count'], 1)
        self.assertEqual(report['operations'][0]['changed_surface_ids'], [308, 310])
        self.assertEqual(report['operations'][0]['unchanged_seam_owners'], [305])
        for sid in trial:
            topology = topology_report(trial[sid][1])
            self.assertEqual(topology['boundary_edge_count'], 0)
            self.assertEqual(topology['nonmanifold_edge_count'], 0)
            self.assertEqual(topology['orientation_error_edge_count'], 0)

    def test_directed_interior_collapse_preserves_existing_reciprocal_seam(self):
        from numilab_human.resting_anatomy_interface_patch import topology_report

        scale = np.float32(50e-6)
        points = np.asarray([
            [0, 0, 0], [1, 0, 0], [1.5, 1e-5, 0],
            [0, 1, 0], [-1, 0, 0], [0, 0, -2],
        ], dtype=np.float32) * scale
        octa_faces = np.asarray([
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
            [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4],
        ], dtype=np.int64)

        def outward(vertices, faces):
            result = faces.copy()
            center = vertices[:, :3].mean(axis=0)
            for index, face in enumerate(result):
                tri = vertices[face, :3].astype(float)
                normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                if float(np.dot(normal, tri.mean(axis=0) - center)) < 0:
                    result[index, [1, 2]] = result[index, [2, 1]]
            return result

        def with_normals(xyz, faces):
            vertices = np.column_stack((xyz, np.zeros((len(xyz), 3), dtype=np.float32)))
            return vertices.astype(np.float32), outward(vertices, faces)

        octa_vertices, octa_faces = with_normals(points, octa_faces)
        # The third owner is a declared coincident exterior representation,
        # so it preserves the same winding as the lung shell.
        reverse_vertices = octa_vertices.copy()
        reverse_faces = octa_faces.copy()
        # A separate closed shell shares a face at the retained seam point, but
        # has no vertex at the interior point that is being collapsed.
        tetra_xyz = np.asarray([
            points[0, :3], points[3, :3], points[4, :3],
            [0, 0, float(scale)],
        ], dtype=np.float32)
        tetra_faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
        tetra_vertices, tetra_faces = with_normals(tetra_xyz, tetra_faces)

        prepared = {
            305: (tetra_vertices, tetra_faces, {i: [i] for i in range(len(tetra_faces))}, {}),
            308: (octa_vertices.copy(), octa_faces.copy(), {i: [i] for i in range(len(octa_faces))}, {}),
            310: (reverse_vertices, reverse_faces, {i: [i] for i in range(len(reverse_faces))}, {}),
        }
        wrong_interface = {
            sid: (row[0].copy(), row[1].copy(), {k: list(v) for k, v in row[2].items()}, dict(row[3]))
            for sid, row in prepared.items()
        }
        with self.assertRaisesRegex(ValueError, 'not opposite'):
            collapse_registered_interior_vertex(
                wrong_interface, keep_point=points[0], drop_point=points[1],
                maximum_source_displacement_m=100e-6, minimum_altitude_m=1e-3,
                volume_bounded_surface_ids=(308, 310),
                volume_bounded_surface_groups=((308, 310),), maximum_volume_change_m3=1e-9)
        for sid in prepared:
            np.testing.assert_array_equal(wrong_interface[sid][0], prepared[sid][0])
            np.testing.assert_array_equal(wrong_interface[sid][1], prepared[sid][1])
        before_topology = {sid: topology_report(row[1]) for sid, row in prepared.items()}
        before_face_count = {sid: len(row[1]) for sid, row in prepared.items()}
        before_vertex_count = {sid: len(row[0]) for sid, row in prepared.items()}
        report = collapse_registered_interior_vertex(
            prepared, keep_point=points[0], drop_point=points[1],
            maximum_source_displacement_m=100e-6,
            minimum_altitude_m=1e-3,
            volume_bounded_surface_ids=(308, 310),
            volume_bounded_surface_groups=((308, 310),),
            maximum_volume_change_m3=1e-9,
            same_winding_surface_pairs={(308, 310)},
        )

        self.assertEqual(report['changed_surface_ids'], [308, 310])
        self.assertEqual(report['unchanged_seam_owners'], [305])
        self.assertEqual(report['drop_point_owners'], [308, 310])
        self.assertEqual(report['keep_point_owners'], [305, 308, 310])
        self.assertEqual(set(report['cumulative_volume_group_change_m3']), {'308+310'})
        expected_motion = float(np.linalg.norm(points[0, :3] - points[1, :3]))
        self.assertAlmostEqual(report['cumulative_source_displacement_m'], expected_motion, places=10)
        for sid, row in prepared.items():
            after = topology_report(row[1])
            self.assertEqual(after['boundary_edge_count'], 0)
            self.assertEqual(after['nonmanifold_edge_count'], 0)
            self.assertEqual(after['orientation_error_edge_count'], 0)
            if sid in (308, 310):
                self.assertEqual(after['edge_count'] - before_topology[sid]['edge_count'], -3)
                self.assertEqual(len(row[1]), before_face_count[sid] - 2)
                self.assertEqual(len(row[0]), before_vertex_count[sid] - 1)
            else:
                self.assertEqual(after['edge_count'], before_topology[sid]['edge_count'])
                self.assertEqual(len(row[1]), before_face_count[sid])
        # The closed shell that lacked the dropped interior point is byte-stable.
        np.testing.assert_array_equal(prepared[305][0], tetra_vertices)
        np.testing.assert_array_equal(prepared[305][1], tetra_faces)
        # Reciprocal owners lose the same two faces and retain their declared
        # coincident-representation winding.
        self.assertEqual(len(prepared[308][1]), len(octa_faces) - 2)
        self.assertEqual(len(prepared[310][1]), len(reverse_faces) - 2)
        self.assertEqual(len(report['seam_checks']), 3)
        self.assertTrue(all(item['unchanged_owner_patch_preserved']
                            for item in report['seam_checks'] if item['other_owner'] == 305))

        equal_owner = {
            308: (octa_vertices.copy(), octa_faces.copy(),
                  {i: [i] for i in range(len(octa_faces))}, {}),
            310: (reverse_vertices.copy(), reverse_faces.copy(),
                  {i: [i] for i in range(len(reverse_faces))}, {}),
        }
        before = {sid: (row[0].copy(), row[1].copy()) for sid, row in equal_owner.items()}
        with self.assertRaisesRegex(ValueError, 'strict subset'):
            collapse_registered_interior_vertex(
                equal_owner, keep_point=points[0], drop_point=points[1],
                maximum_source_displacement_m=100e-6, minimum_altitude_m=1e-3,
                same_winding_surface_pairs={(308, 310)})
        for sid in equal_owner:
            np.testing.assert_array_equal(equal_owner[sid][0], before[sid][0])
            np.testing.assert_array_equal(equal_owner[sid][1], before[sid][1])
        equal_report = collapse_registered_interior_vertex(
            equal_owner, keep_point=points[0], drop_point=points[1],
            maximum_source_displacement_m=100e-6, minimum_altitude_m=1e-3,
            same_winding_surface_pairs={(308, 310)}, allow_equal_owner_sets=True)
        self.assertEqual(equal_report['owner_set_relation'], 'equal')
        self.assertEqual(equal_report['changed_surface_ids'], [308, 310])
        self.assertEqual(equal_report['unchanged_seam_owners'], [])
        self.assertTrue(all(check['shared_triangles_before'] -
                            check['shared_triangles_after'] == 2
                            for check in equal_report['seam_checks']))
        for sid in equal_owner:
            topology = topology_report(equal_owner[sid][1])
            self.assertEqual(topology['boundary_edge_count'], 0)
            self.assertEqual(topology['nonmanifold_edge_count'], 0)
            self.assertEqual(topology['orientation_error_edge_count'], 0)

        equal_owner_batch = {
            308: (octa_vertices.copy(), octa_faces.copy(),
                  {i: [i] for i in range(len(octa_faces))}, {}),
            310: (reverse_vertices.copy(), reverse_faces.copy(),
                  {i: [i] for i in range(len(reverse_faces))}, {}),
        }
        batch_report = improve_registered_interior_slivers(
            equal_owner_batch, minimum_altitude_m=1e-3,
            maximum_source_displacement_m=100e-6,
            same_winding_surface_pairs={(308, 310)}, allow_equal_owner_sets=True,
            maximum_operations=1)
        self.assertEqual(batch_report['accepted_operation_count'], 1)
        self.assertEqual(batch_report['operations'][0]['owner_set_relation'], 'equal')
        self.assertEqual(batch_report['operations'][0]['changed_surface_ids'], [308, 310])

    def test_retained_native_collinearity_regression(self):
        # Actual source faces 402/403 of lobe305 and submitted GPU vertices at
        # accepted step30335 (60.67000288167037 s). Receipt SHA-256:
        # cc9934cd1f8b377f8faa95686ff245708f85abab8f7d05d1652b15e90b4facdf.
        p = np.array([[.027375534176826477, -.06637446582317352, -.025332223623991013],
                      [.02867928147315979, -.06564869731664658, -.024853337556123734],
                      [.028802327811717987, -.06557120382785797, -.02482273057103157],
                      [.028925376012921333, -.06549370288848877, -.024792121723294258]], np.float32)
        world = np.array([[.002383217215538025, -.5238156318664551, .11519312113523483],
                          [.0018961094319820404, -.5246641635894775, .1164914071559906],
                          [.0018652938306331635, -.5247358083724976, .11661400645971298],
                          [.0018344782292842865, -.5248074531555176, .11673660576343536]], np.float32)
        f = np.array([[1, 0, 3], [3, 2, 1]])
        original = world[f[1]].astype(float)
        np.testing.assert_array_equal(np.cross(original[1]-original[0], original[2]-original[0]), 0)
        v = np.column_stack((p, np.tile([0, 0, 1], (4, 1)))).astype(np.float32)
        prepared = {305: (v, f, {402: [0], 403: [1]}, {}),
                    311: (v.copy(), f[:, ::-1].copy(), {0: [0], 1: [1]}, {})}
        report = improve_sliver_faces(prepared)
        self.assertEqual(report['flip_count'], 1)
        for sid in prepared:
            x = world[prepared[sid][1]]
            area = np.cross(x[:, 1]-x[:, 0], x[:, 2]-x[:, 0])
            self.assertTrue(np.all(np.any(area != 0, axis=1)))
            np.testing.assert_array_equal(prepared[sid][0][:, :3], p)

    def test_shared_star_opening_moves_all_exact_owners_without_changing_topology(self):
        from numilab_human.resting_anatomy_interface_patch import topology_report

        xyz = np.asarray([[0, 0, 0], [50e-6, 0, 0], [25e-6, .5e-6, 0],
                          [0, 50e-6, 0], [-50e-6, 0, 0], [0, 0, -50e-6]], dtype=np.float32)
        faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
                            [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]], dtype=np.int64)
        vertices = np.column_stack((xyz, np.tile([0, 0, 1], (len(xyz), 1)))).astype(np.float32)
        center = vertices[:, :3].mean(axis=0)
        for index, face in enumerate(faces):
            tri = vertices[face, :3].astype(float)
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if np.dot(normal, tri.mean(axis=0) - center) < 0:
                faces[index, [1, 2]] = faces[index, [2, 1]]

        def make_prepared():
            return {sid: (vertices.copy(), faces.copy(),
                          {i: [i] for i in range(len(faces))}, {})
                    for sid in (308, 310)}

        ancestry = {tuple(map(float, point)): {tuple(map(float, point))} for point in xyz}
        prepared = make_prepared()
        before_faces = {sid: row[1].copy() for sid, row in prepared.items()}
        report = open_registered_shared_vertex_star(
            prepared, xyz[2], [0, 1, 0], 2e-6,
            maximum_source_displacement_m=100e-6,
            minimum_altitude_m=1.2e-6,
            maximum_volume_change_m3=1e-9,
            volume_bounded_surface_ids=(308, 310),
            source_ancestry_by_point=ancestry)
        self.assertEqual(report['owner_surface_ids'], [308, 310])
        self.assertLess(report['combined_local_sliver_objective_after'],
                        report['combined_local_sliver_objective_before'])
        self.assertLessEqual(report['cumulative_source_displacement_m'], 100e-6)
        self.assertEqual(report['reciprocal_seam_checks'][0]['shared_local_triangle_count'], 4)
        for sid, row in prepared.items():
            np.testing.assert_array_equal(row[1], before_faces[sid])
            self.assertEqual(topology_report(row[1])['boundary_edge_count'], 0)
        self.assertEqual(ancestry[report['new_point']], {tuple(map(float, xyz[2]))})

        batch = make_prepared()
        batch_report = improve_registered_shared_vertex_stars(
            batch, minimum_altitude_m=1.2e-6, target_altitude_m=2.4e-6,
            maximum_source_displacement_m=100e-6,
            volume_bounded_surface_ids=(308, 310),
            maximum_operations=1)
        self.assertEqual(batch_report['accepted_operation_count'], 1)
        self.assertLess(batch_report['operations'][0]['combined_local_sliver_objective_after'],
                        batch_report['operations'][0]['combined_local_sliver_objective_before'])

    def test_shared_star_search_can_trade_one_incident_altitude_for_a_lower_star_score(self):
        xyz = np.asarray([[0, 0, 0], [50e-6, 0, 0], [25e-6, .5e-6, 0],
                          [0, 50e-6, 0], [-50e-6, 0, 0], [0, 0, -50e-6]], dtype=np.float32)
        faces = np.asarray([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],
                            [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]], dtype=np.int64)
        vertices = np.column_stack((xyz, np.tile([0, 0, 1], (len(xyz), 1)))).astype(np.float32)
        center = vertices[:, :3].mean(axis=0)
        for index, face in enumerate(faces):
            tri = vertices[face, :3].astype(float)
            normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if np.dot(normal, tri.mean(axis=0) - center) < 0:
                faces[index, [1, 2]] = faces[index, [2, 1]]
        prepared = {sid: (vertices.copy(), faces.copy(),
                          {i: [i] for i in range(len(faces))}, {})
                    for sid in (308, 310)}
        report = improve_registered_shared_vertex_stars(
            prepared, minimum_altitude_m=1.2e-6, target_altitude_m=2.4e-6,
            minimum_result_altitude_m=1.25e-7,
            allow_local_minimum_regression=True,
            search_all_face_vertices=True,
            maximum_source_displacement_m=100e-6,
            volume_bounded_surface_ids=(308, 310), maximum_operations=1)
        self.assertTrue(report['allow_local_minimum_regression'])
        self.assertTrue(report['search_all_face_vertices'])
        self.assertEqual(report['accepted_operation_count'], 1)
        operation = report['operations'][0]
        self.assertLess(operation['combined_local_sliver_objective_after'],
                        operation['combined_local_sliver_objective_before'])
        self.assertGreaterEqual(operation['combined_local_minimum_altitude_after_m'], 1.25e-7)


    def test_reciprocal_flip_preserves_vertices_boundary_and_lineage(self):
        from numilab_human.resting_anatomy_interface_patch import topology_report
        from numilab_human.resting_anatomy_conforming_refinement import remap_ids
        v, f, mapping, detail = prepared_quad()
        before = topology_report(f)
        prepared = {305: (v, f, mapping, detail),
                    311: (v.copy(), f[:, ::-1].copy(), {0: [0], 1: [1]}, {})}
        report = improve_sliver_faces(prepared)
        self.assertEqual(report['flip_count'], 1)
        self.assertEqual(report['changes'][0]['shared_surface_owners'], [305, 311])
        for sid in prepared:
            vertices, faces, parents, _ = prepared[sid]
            np.testing.assert_array_equal(vertices[:, :3], v[:, :3])
            self.assertEqual(parents, {0: [0, 1], 1: [0, 1]})
            self.assertEqual(remap_ids([0, 1], parents), [0, 1])
            self.assertEqual(set(topology_report(faces)['boundary_edges']), set(before['boundary_edges']))
            self.assertGreater(report['remaining'][str(sid)]['minimum_altitude_m'], 1e-4)
        normals = []
        for sid in prepared:
            vertices, faces, _, _ = prepared[sid]
            normals.append({tuple(sorted(map(int, row))): np.cross(
                vertices[row[1], :3]-vertices[row[0], :3],
                vertices[row[2], :3]-vertices[row[0], :3]) for row in faces})
        self.assertEqual(normals[0].keys(), normals[1].keys())
        for key in normals[0]:
            self.assertLess(float(np.dot(normals[0][key], normals[1][key])), 0)

    def test_interface_boundary_cannot_be_flipped(self):
        v, f, mapping, detail = prepared_quad()
        prepared = {305: (v, f.copy(), mapping, detail),
                    311: (v[:3].copy(), np.array([[0, 2, 1]]), {0: [0]}, {})}
        report = improve_sliver_faces(prepared)
        self.assertEqual(report['flip_count'], 0)
        self.assertGreater(report['skipped_reasons']['anatomical interface ownership boundary'], 0)
        np.testing.assert_array_equal(prepared[305][1], f)

    def test_cannot_cross_a_respiratory_field_cell(self):
        v, f, mapping, detail = prepared_quad()
        v[:, 0] += .006
        prepared = {305: (v, f.copy(), mapping, detail)}
        report = improve_sliver_faces(prepared)
        self.assertEqual(report['flip_count'], 0)
        self.assertEqual(report['skipped_reasons']['different affine field cells'], 1)
        np.testing.assert_array_equal(prepared[305][1], f)

    def test_registered_surface_path_crosses_only_numerical_cell_boundary(self):
        from numilab_human.resting_anatomy_interface_patch import topology_report
        v, f, mapping, detail = prepared_quad()
        v[:, 0] += .006
        same_cell_prepared = {305: (v.copy(), f.copy(), mapping.copy(), detail.copy())}
        same_cell = improve_sliver_faces(same_cell_prepared)
        self.assertEqual(same_cell['flip_count'], 0)

        prepared = {305: (v.copy(), f.copy(), mapping.copy(), detail.copy()),
                    311: (v.copy(), f[:, ::-1].copy(), {0: [0], 1: [1]}, {})}
        before = {sid: row[0][:, :3].copy() for sid, row in prepared.items()}
        report = improve_registered_surface_sliver_faces(prepared)
        self.assertEqual(report['flip_count'], 1)
        self.assertFalse(report['same_affine_field_cell_required'])
        self.assertTrue(report['strict_altitude_improvement_required'])
        self.assertEqual(report['minimum_altitude_target_m'], 1.2e-6)
        self.assertEqual(report['maximum_nonplanarity_m'], 1e-5)
        self.assertEqual(report['changes'][0]['shared_surface_owners'], [305, 311])
        self.assertLessEqual(report['changes'][0]['maximum_patch_nonplanarity_m'], 1e-5)
        self.assertGreaterEqual(report['changes'][0]['new_minimum_altitude_m'], 1.2e-6)
        for sid, (vertices, faces, parents, _) in prepared.items():
            np.testing.assert_array_equal(vertices[:, :3], before[sid])
            self.assertEqual(parents, {0: [0, 1], 1: [0, 1]})
            self.assertEqual(topology_report(faces)['boundary_edge_count'], 4)

    def test_registered_surface_path_can_record_improvement_below_target(self):
        v, f, mapping, detail = prepared_quad()
        v[:, :3] *= np.float32(4e-4)
        default_prepared = {305: (v.copy(), f.copy(), mapping.copy(), detail.copy()),
                            311: (v.copy(), f[:, ::-1].copy(), {0: [0], 1: [1]}, {})}
        default = improve_registered_surface_sliver_faces(default_prepared)
        self.assertEqual(default['flip_count'], 0)

        prepared = {305: (v.copy(), f.copy(), mapping.copy(), detail.copy()),
                    311: (v.copy(), f[:, ::-1].copy(), {0: [0], 1: [1]}, {})}
        report = improve_registered_surface_sliver_faces(
            prepared, minimum_result_altitude_m=1.25e-7)
        self.assertEqual(report['flip_count'], 1)
        self.assertEqual(report['minimum_result_altitude_m'], 1.25e-7)
        result = report['changes'][0]['new_minimum_altitude_m']
        self.assertGreaterEqual(result, 1.25e-7)
        self.assertLess(result, 1.2e-6)

    def test_bounds_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            improve_sliver_faces({}, minimum_altitude_m=.001)

    def _csg_fixture(self, *, source_owner=(0, 0), shift_x=0.0, original_is_source=False):
        v, f, mapping, detail = prepared_quad()
        v[:, 0] += np.float32(shift_x)
        prepared = {305: (v.copy(), f.copy(), mapping, detail),
                    311: (v.copy(), f[:, ::-1].copy(), {0: [0], 1: [1]}, {})}
        owner = np.asarray(source_owner, dtype=np.int64)
        owner_by_face = {305: owner.copy(), 311: owner.copy()}
        if original_is_source:
            original = {sid: (row[0][:, :3].copy(), row[1].copy()) for sid, row in prepared.items()}
        else:
            empty = (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64))
            original = {305: empty, 311: empty}
        return prepared, owner_by_face, original, v.copy()

    def test_csg_fragment_path_is_all_owner_eligible_preserves_multilineage_and_moves_no_vertices(self):
        from numilab_human.resting_anatomy_interface_patch import topology_report
        prepared, owner_by_face, original, before_vertices = self._csg_fixture(shift_x=.01)
        report = improve_csg_cut_fragment_faces(prepared, owner_by_face, original)
        self.assertEqual(report['flip_count'], 1)
        self.assertFalse(report['same_affine_field_cell_required'])
        self.assertTrue(report['all_exact_original_source_faces_excluded'])
        self.assertTrue(report['multi_parent_lineage_preserved'])
        self.assertEqual(report['multi_parent_face_count_by_surface'], {'305': 2, '311': 2})
        for sid, (vertices, faces, parents, _) in prepared.items():
            np.testing.assert_array_equal(vertices[:, :3], before_vertices[:, :3])
            self.assertEqual(parents, {0: [0, 1], 1: [0, 1]})
            self.assertEqual(topology_report(faces)['boundary_edge_count'], 4)

    def test_csg_fragment_path_rejects_any_owner_with_missing_or_mismatched_cut_provenance(self):
        prepared, owner_by_face, original, before_vertices = self._csg_fixture()
        owner_by_face[311] = np.asarray([0, 1], dtype=np.int64)
        report = improve_csg_cut_fragment_faces(prepared, owner_by_face, original)
        self.assertEqual(report['flip_count'], 0)
        self.assertGreater(
            report['skipped_reasons']['outside inferred CSG cut patch or reciprocal provenance differs'], 0
        )
        for vertices, faces, _, _ in prepared.values():
            np.testing.assert_array_equal(vertices[:, :3], before_vertices[:, :3])

    def test_csg_fragment_path_excludes_exact_original_source_faces(self):
        prepared, owner_by_face, original, before_vertices = self._csg_fixture(original_is_source=True)
        report = improve_csg_cut_fragment_faces(prepared, owner_by_face, original)
        self.assertEqual(report['flip_count'], 0)
        self.assertEqual(report['exact_original_source_face_count_by_surface'], {'305': 2, '311': 2})
        self.assertTrue(report['all_exact_original_source_faces_excluded'])
        for vertices, faces, _, _ in prepared.values():
            np.testing.assert_array_equal(vertices[:, :3], before_vertices[:, :3])


if __name__ == '__main__':
    unittest.main()
