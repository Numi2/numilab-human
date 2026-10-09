from __future__ import annotations

import unittest

from numilab_human import cli, model


def fixture_fields():
    vertices_m = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
    ]
    normals = [[0.0, 0.0, 1.0] for _ in vertices_m]
    global_vertices = [[point[0], point[1], point[2]] for point in vertices_m]
    vertex_weights = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    faces = [[0, 1, 2], [1, 0, 3]]
    matrix = [
        [0.001, 0.0, 0.0, 0.0],
        [0.0, 0.001, 0.0, 0.0],
        [0.0, 0.0, 0.001, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    routes = [
        {"body": "a", "world_m": [0.0, 0.0, 0.0], "kind": "site", "core_body_index": 0},
        {"body": "b", "world_m": [1.0, 0.0, 0.0], "kind": "site", "core_body_index": 1},
    ]
    return vertices_m, normals, global_vertices, vertex_weights, faces, matrix, routes


class ConformingSurfaceRefinementTests(unittest.TestCase):
    def test_bisects_both_incident_faces_and_preserves_records_and_winding(self):
        fields = fixture_fields()
        vertices, normals, global_vertices, weights, faces, matrix, routes = fields
        original = (
            [row[:] for row in vertices],
            [row[:] for row in normals],
            [row[:] for row in global_vertices],
            [row[:] for row in weights],
            [row[:] for row in faces],
        )
        result = model._bodyparts_refine_conforming_route_edge(
            64, (0, 1), vertices, normals, global_vertices, weights,
            faces, matrix, ["a", "b"], routes,
        )
        new_vertices, new_normals, new_global, new_weights, new_faces, operation = result
        self.assertEqual(new_vertices[:-1], original[0])
        self.assertEqual(new_normals[:-1], original[1])
        self.assertEqual(new_global[:-1], original[2])
        self.assertEqual(new_weights[:-1], original[3])
        self.assertEqual(new_faces[:2], [[0, 4, 2], [1, 4, 3]])
        self.assertEqual(new_faces[2:], [[4, 1, 2], [4, 0, 3]])
        self.assertEqual(len(new_vertices), 5)
        self.assertEqual(len(new_faces), 4)
        self.assertEqual(operation["incident_source_face_rows"], [0, 1])
        self.assertEqual(operation["incident_edge_directions"], [[0, 1], [1, 0]])
        self.assertEqual(
            [(row["retained_output_face_row"], row["appended_output_face_row"])
             for row in operation["parent_face_children"]],
            [(0, 2), (1, 3)],
        )
        # Each emitted child has positive area and retains the source parent's winding.
        for parent_row, source_face in enumerate(original[4]):
            source_normal = cross(new_vertices, source_face)
            for child in (new_faces[parent_row], new_faces[parent_row + 2]):
                child_normal = cross(new_vertices, child)
                self.assertGreater(dot(source_normal, child_normal), 0.0)

    def test_midpoint_uses_existing_route_weight_owner_and_records_context(self):
        fields = fixture_fields()
        vertices, normals, global_vertices, weights, faces, matrix, routes = fields
        result = model._bodyparts_refine_conforming_route_edge(
            64, (0, 1), vertices, normals, global_vertices, weights,
            faces, matrix, ["a", "b"], routes,
        )
        # Independent statement of the pre-existing per-body nearest-node
        # inverse-square rule, used as the expected midpoint owner.
        query = [0.5, 0.0, 0.0]
        squared_by_body = []
        for body in ("a", "b"):
            squared_by_body.append(min(
                sum((query[axis] - point["world_m"][axis]) ** 2 for axis in range(3))
                for point in routes if point["body"] == body
            ))
        nearest = sorted((squared, index) for index, squared in enumerate(squared_by_body))[:4]
        raw = [1.0 / (squared + 9.0e-6) for squared, _ in nearest]
        total = sum(raw)
        expected = [0.0, 0.0]
        for value, (_, index) in zip(raw, nearest, strict=True):
            expected[index] = value / total
        self.assertEqual(result[3][-1], expected)
        self.assertEqual(result[0][-1], [0.5, 0.0, 0.0])
        self.assertEqual(result[2][-1], [0.5, 0.0, 0.0])
        self.assertEqual(result[5]["global_source_mm_to_myosim_world_m"], matrix)
        self.assertEqual(result[5]["route_body_binding_order"], ["a", "b"])
        self.assertEqual(result[5]["authored_route_points"], routes)

    def test_missing_edge_is_rejected(self):
        vertices, normals, global_vertices, weights, faces, matrix, routes = fixture_fields()
        with self.assertRaisesRegex(model.ImportError, "exactly two indexed"):
            model._bodyparts_refine_conforming_route_edge(
                64, (1, 2), vertices, normals, global_vertices, weights,
                faces, matrix, ["a", "b"], routes,
            )

    def test_nonmanifold_edge_is_rejected(self):
        vertices, normals, global_vertices, weights, faces, matrix, routes = fixture_fields()
        vertices.append([0.0, 0.0, 1.0])
        normals.append([0.0, 0.0, 1.0])
        global_vertices.append([0.0, 0.0, 1.0])
        weights.append([1.0, 0.0])
        faces.append([0, 1, 4])
        with self.assertRaisesRegex(model.ImportError, "exactly two indexed"):
            model._bodyparts_refine_conforming_route_edge(
                64, (0, 1), vertices, normals, global_vertices, weights,
                faces, matrix, ["a", "b"], routes,
            )

    def test_coordinate_duplicate_edge_is_rejected(self):
        vertices, normals, global_vertices, weights, faces, matrix, routes = fixture_fields()
        vertices.extend(([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]))
        normals.extend([[0.0, 0.0, 1.0] for _ in range(3)])
        global_vertices.extend([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        weights.extend([[1.0, 0.0] for _ in range(3)])
        faces.append([4, 5, 6])
        with self.assertRaisesRegex(model.ImportError, "coordinate-welded"):
            model._bodyparts_refine_conforming_route_edge(
                64, (0, 1), vertices, normals, global_vertices, weights,
                faces, matrix, ["a", "b"], routes,
            )

    def test_rejects_duplicate_bindings_and_nonfinite_route_points(self):
        fields = fixture_fields()
        vertices, normals, global_vertices, weights, faces, matrix, routes = fields
        with self.assertRaisesRegex(model.ImportError, "unique named route bindings"):
            model._bodyparts_refine_conforming_route_edge(
                64, (0, 1), vertices, normals, global_vertices, weights,
                faces, matrix, ["a", "a"], routes,
            )
        bad_routes = [dict(routes[0]), dict(routes[1])]
        bad_routes[1]["world_m"] = [float("nan"), 0.0, 0.0]
        with self.assertRaisesRegex(model.ImportError, "non-finite route point"):
            model._bodyparts_refine_conforming_route_edge(
                64, (0, 1), vertices, normals, global_vertices, weights,
                faces, matrix, ["a", "b"], bad_routes,
            )

    def test_rejects_nonfinite_or_nonaffine_transform(self):
        fields = fixture_fields()
        vertices, normals, global_vertices, weights, faces, matrix, routes = fields
        bad_matrix = [row[:] for row in matrix]
        bad_matrix[0][0] = float("inf")
        with self.assertRaisesRegex(model.ImportError, "finite affine 4x4"):
            model._bodyparts_refine_conforming_route_edge(
                64, (0, 1), vertices, normals, global_vertices, weights,
                faces, bad_matrix, ["a", "b"], routes,
            )
        bad_matrix = [row[:] for row in matrix]
        bad_matrix[3][0] = 0.001
        with self.assertRaisesRegex(model.ImportError, "finite affine 4x4"):
            model._bodyparts_refine_conforming_route_edge(
                64, (0, 1), vertices, normals, global_vertices, weights,
                faces, bad_matrix, ["a", "b"], routes,
            )

    def test_rejects_noninteger_source_face_indices(self):
        vertices, normals, global_vertices, weights, faces, matrix, routes = fixture_fields()
        faces[1][2] = 3.0
        with self.assertRaisesRegex(model.ImportError, "integer source faces"):
            model._bodyparts_refine_conforming_route_edge(
                64, (0, 1), vertices, normals, global_vertices, weights,
                faces, matrix, ["a", "b"], routes,
            )

    def test_cli_edge_argument_is_explicit_and_validated(self):
        self.assertEqual(cli._parse_conforming_edge_refinement("64:2597:3054"), (64, 2597, 3054))
        for value in ("64:1", "0:1:2", "64:2:2", "x:1:2"):
            with self.subTest(value=value), self.assertRaises(Exception):
                cli._parse_conforming_edge_refinement(value)


def cross(vertices, face):
    p0, p1, p2 = (vertices[index] for index in face)
    u = [p1[axis] - p0[axis] for axis in range(3)]
    v = [p2[axis] - p0[axis] for axis in range(3)]
    return [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


if __name__ == "__main__":
    unittest.main()

