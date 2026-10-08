import itertools
import unittest

import numpy as np

from numilab_human.resting_anatomy_interface_patch import normals, signed_volume
from numilab_human.resting_lung_edge_repair import _condition_precision_lobes
from numilab_human.surface_precision_retriangulation import validate_closed_oriented_surface


SELECTION_ALTITUDE_M = 128e-9


def _oriented_faces(vertices, triangles):
    """Orient a small convex fixture outward using its interior centroid."""
    center = np.asarray(vertices, dtype=np.float64).mean(axis=0)
    output = []
    for triangle in triangles:
        face = np.asarray(triangle, dtype=np.int64)
        points = np.asarray(vertices, dtype=np.float64)[face]
        face_center = points.mean(axis=0)
        if np.dot(np.cross(points[1] - points[0], points[2] - points[0]), face_center - center) < 0:
            face = face[[0, 2, 1]]
        output.append(face)
    return np.asarray(output, dtype=np.int64)


def _row(stable_id, xyz, faces):
    xyz = np.asarray(xyz, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    return {
        "body_index": int(stable_id),
        "vertices6": np.column_stack((xyz, normals(xyz.astype(np.float64), faces))).astype(np.float32),
        "faces": faces,
    }


def _tetra_row(stable_id, xyz):
    faces = _oriented_faces(xyz, itertools.combinations(range(4), 3))
    return _row(stable_id, xyz, faces)


def _regular_tetra_rows():
    rows = {}
    for stable_id in (307, 308, 309):
        offset = np.asarray([stable_id * 100e-6, -stable_id * 80e-6, stable_id * 60e-6])
        xyz = offset + np.asarray([
            [0.0, 0.0, 0.0],
            [10e-6, 0.0, 0.0],
            [5e-6, 8.66e-6, 0.0],
            [5e-6, 2.887e-6, 8.165e-6],
        ])
        rows[stable_id] = _tetra_row(stable_id, xyz)
    return rows


def _prism_rows():
    # A convex quadrilateral interface with one deliberately low-altitude
    # diagonal AC (33 nm); BD gives two broad triangles. Both rows use the same
    # exact Float32 interface positions and opposite outward orientation.
    base = np.asarray([
        [0.0, 0.0, 0.0],
        [10e-6, 0.0, 0.0],
        [15e-6, 0.05e-6, 0.0],
        [0.0, 10e-6, 0.0],
    ])
    rows = {}
    for stable_id, height in ((305, 10e-6), (306, -10e-6)):
        xyz = np.vstack((base, base + np.asarray([0.0, 0.0, height])))
        triangles = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)]
        for edge_start in range(4):
            edge_end = (edge_start + 1) % 4
            triangles.extend([
                (edge_start, edge_end, edge_end + 4),
                (edge_start, edge_end + 4, edge_start + 4),
            ])
        faces = _oriented_faces(xyz, triangles)
        rows[stable_id] = _row(stable_id, xyz, faces)
    rows.update(_regular_tetra_rows())
    return rows


def _shared_face_map(rows, stable_ids=(305, 306)):
    """Map exact position-bit triangles to their lobe owner and oriented normal."""
    mapped = {}
    for stable_id in stable_ids:
        row = rows[stable_id]
        xyz = row["vertices6"][:, :3]
        for face_index, face in enumerate(row["faces"]):
            key = tuple(sorted(np.ascontiguousarray(xyz[vi], dtype="<f4").tobytes() for vi in face))
            points = xyz[face].astype(np.float64)
            oriented_normal = np.cross(points[1] - points[0], points[2] - points[0])
            mapped.setdefault(key, []).append((stable_id, face_index, oriented_normal))
    return {key: owners for key, owners in mapped.items() if len(owners) > 1}


def _shared_skinny_triangle_rows():
    # Exactly one shared low-altitude triangle; every other face in either
    # tetrahedron is broad. A 2-to-2 patch flip would cross into an unshared
    # face, so the planner must leave it untouched and report the protection.
    a = np.asarray([0.0, 0.0, 0.0])
    b = np.asarray([10e-6, 0.0, 0.0])
    c = np.asarray([15e-6, 0.05e-6, 0.0])
    rows = {
        305: _tetra_row(305, [a, b, c, [5e-6, 4e-6, 10e-6]]),
        306: _tetra_row(306, [a, b, c, [5e-6, 4e-6, -10e-6]]),
    }
    rows.update(_regular_tetra_rows())
    return rows


class RestingLungPrecisionConditioningTests(unittest.TestCase):
    def _assert_closed_topology_unchanged(self, source, conditioned):
        for stable_id in range(305, 310):
            before = validate_closed_oriented_surface(
                source[stable_id]["vertices6"][:, :3], source[stable_id]["faces"]
            )
            after = validate_closed_oriented_surface(
                conditioned[stable_id]["vertices6"][:, :3], conditioned[stable_id]["faces"]
            )
            self.assertEqual(before, after, f"topology changed for lobe {stable_id}")
            self.assertTrue(after["closed_oriented"])
            self.assertEqual(after["zero_area_face_count"], 0)
            self.assertEqual(len(source[stable_id]["faces"]), len(conditioned[stable_id]["faces"]))
            self.assertAlmostEqual(
                signed_volume(conditioned[stable_id]["vertices6"][:, :3].astype(np.float64),
                              conditioned[stable_id]["faces"]),
                signed_volume(source[stable_id]["vertices6"][:, :3].astype(np.float64),
                              source[stable_id]["faces"]),
                delta=1e-12,
            )

    def _assert_positions_unchanged(self, source, conditioned):
        for stable_id in range(305, 310):
            before = np.ascontiguousarray(source[stable_id]["vertices6"][:, :3], dtype="<f4").tobytes()
            after = np.ascontiguousarray(conditioned[stable_id]["vertices6"][:, :3], dtype="<f4").tobytes()
            self.assertEqual(before, after, f"source positions changed for lobe {stable_id}")

    def test_shared_patch_flips_as_a_pair_with_source_lineage_preserved(self):
        source = _prism_rows()
        source_snapshots = {
            stable_id: (row["vertices6"].copy(), row["faces"].copy())
            for stable_id, row in source.items()
        }
        for stable_id in range(305, 310):
            self.assertTrue(validate_closed_oriented_surface(
                source[stable_id]["vertices6"][:, :3], source[stable_id]["faces"]
            )["closed_oriented"])
        source_shared = _shared_face_map(source)
        self.assertEqual(len(source_shared), 2)
        source_shared_indices = {
            stable_id: sorted(owner[1] for owners in source_shared.values() for owner in owners
                              if owner[0] == stable_id)
            for stable_id in (305, 306)
        }
        self.assertTrue(all(len(indices) == 2 for indices in source_shared_indices.values()))
        self.assertTrue(all(
            np.dot(owners[0][2], owners[1][2]) < 0.0 for owners in source_shared.values()
        ))

        conditioned, report = _condition_precision_lobes(source, minimum_altitude_m=1e-6)

        paired = [operation for operation in report["operations"] if operation["paired_interlobar_flip"]]
        self.assertTrue(paired, "the skinny shared interface should be repaired as one paired operation")
        operation = next(op for op in paired if {flip["stable_id"] for flip in op["flips"]} == {305, 306})
        self.assertEqual(len(operation["flips"]), 2)
        for flip in operation["flips"]:
            stable_id = flip["stable_id"]
            self.assertFalse(flip["vertices_modified"])
            self.assertEqual(flip["input_source_face_ancestry"], source_shared_indices[stable_id])
            for lineage in flip["output_face_lineage"]:
                self.assertEqual(lineage["source_parent_face_indices"], source_shared_indices[stable_id])
        self._assert_positions_unchanged(source, conditioned)
        self._assert_closed_topology_unchanged(source, conditioned)

        conditioned_shared = _shared_face_map(conditioned)
        xyz = source[305]["vertices6"][:, :3]
        expected_retriangulated = {
            tuple(sorted(np.ascontiguousarray(xyz[vi], dtype="<f4").tobytes() for vi in triangle))
            for triangle in ((0, 1, 3), (1, 2, 3))
        }
        self.assertEqual(set(conditioned_shared), expected_retriangulated)
        self.assertNotEqual(set(conditioned_shared), set(source_shared))
        self.assertEqual(len(conditioned_shared), 2)
        self.assertTrue(all(
            np.dot(owners[0][2], owners[1][2]) < 0.0 for owners in conditioned_shared.values()
        ))
        self.assertTrue(report["shared_lobe_face_ownership_counts_preserved"])
        for stable_id, (vertices6, faces) in source_snapshots.items():
            np.testing.assert_array_equal(source[stable_id]["vertices6"], vertices6)
            np.testing.assert_array_equal(source[stable_id]["faces"], faces)

    def test_partially_shared_skinny_patch_is_protected(self):
        source = _shared_skinny_triangle_rows()
        source_shared = _shared_face_map(source)
        self.assertEqual(len(source_shared), 1)
        shared_key = next(iter(source_shared))
        self.assertTrue(all(np.dot(owners[0][2], owners[1][2]) < 0.0 for owners in source_shared.values()))
        source_faces = {stable_id: row["faces"].copy() for stable_id, row in source.items()}

        conditioned, report = _condition_precision_lobes(source, minimum_altitude_m=1e-6)

        self.assertEqual(report["operations"], [])
        protected = [attempt for attempt in report["blocked_attempts"]
                     if attempt["stable_id"] in (305, 306)]
        self.assertEqual({attempt["stable_id"] for attempt in protected}, {305, 306})
        self.assertTrue(all(
            "partly shared or ambiguous patch" in attempt["rejections"]
            for attempt in protected
        ))
        self.assertEqual(set(_shared_face_map(conditioned)), set(_shared_face_map(source)))
        self.assertIn(shared_key, _shared_face_map(conditioned))
        for stable_id in range(305, 310):
            np.testing.assert_array_equal(conditioned[stable_id]["faces"], source_faces[stable_id])
        self._assert_positions_unchanged(source, conditioned)
        self._assert_closed_topology_unchanged(source, conditioned)


if __name__ == "__main__":
    unittest.main()
