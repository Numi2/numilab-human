import unittest
import numpy as np

from numilab_human.resting_respiratory_mesh_quality import improve_sliver_faces


def prepared_quad():
    p = np.array([[.008, .003, .001], [.012, .003, .001],
                  [.010, .00300001, .001], [.010, .002, .001]], np.float32)
    vertices = np.column_stack((p, np.tile([0, 0, 1], (4, 1)))).astype(np.float32)
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    return vertices, faces, {0: [0], 1: [1]}, {}


class RespiratoryMeshQualityTests(unittest.TestCase):
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

    def test_bounds_are_explicit(self):
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            improve_sliver_faces({}, minimum_altitude_m=.001)


if __name__ == '__main__':
    unittest.main()
