import itertools
import unittest

import numpy as np

from numilab_human.resting_anatomy_interface_patch import normals, signed_volume
from numilab_human.resting_lung_edge_repair import (
    _condition_precision_lobes, _collapse_short_precision_slivers,
    _relocate_subthreshold_vertices,
    _reconstruct_diaphragm_interface_registration,
    _refresh_diaphragm_interface_registration,
)
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


def _diaphragm_shared_skinny_rows():
    # A narrow exact face belongs to one lobe and one separate closed
    # diaphragm owner. The lobe-only flip planner must protect this interface.
    a = np.asarray([0.0, 0.0, 0.0])
    b = np.asarray([10e-6, 0.0, 0.0])
    c = np.asarray([15e-6, 0.05e-6, 0.0])
    rows = {
        305: _tetra_row(305, [a, b, c, [5e-6, 4e-6, 10e-6]]),
        311: _tetra_row(311, [a, b, c, [5e-6, 4e-6, -10e-6]]),
    }
    rows[306] = _tetra_row(306, [
        [0.05, 0.0, 0.0], [0.06, 0.0, 0.0], [0.055, 0.008, 0.0], [0.055, 0.003, 0.008],
    ])
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

    def test_diaphragm_shared_patch_is_protected_from_lobe_only_flips(self):
        source = _diaphragm_shared_skinny_rows()
        source_faces = {sid: row["faces"].copy() for sid, row in source.items()}
        source_shared = _shared_face_map(source, stable_ids=(305, 311))
        self.assertEqual(len(source_shared), 1)

        conditioned, report = _condition_precision_lobes(source, minimum_altitude_m=1e-6)

        self.assertEqual(report["operations"], [])
        self.assertEqual(report["protected_non_lobe_reciprocal_owner_ids"], [311])
        blocked = [attempt for attempt in report["blocked_attempts"] if attempt["stable_id"] == 305]
        self.assertTrue(blocked)
        self.assertTrue(any(any("shared" in reason for reason in attempt["rejections"])
                            for attempt in blocked))
        self.assertEqual(set(_shared_face_map(conditioned, stable_ids=(305, 311))),
                         set(source_shared))
        for sid in source_faces:
            np.testing.assert_array_equal(conditioned[sid]["faces"], source_faces[sid])

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


    def test_tangent_relocation_propagates_reciprocal_vertices_and_keeps_topology(self):
        source = _prism_rows()
        snapshots = {
            sid: (row["vertices6"].copy(), row["faces"].copy())
            for sid, row in source.items()
        }
        source_shared = _shared_face_map(source, stable_ids=(305, 306))
        self.assertEqual(len(source_shared), 2)

        relocated, report = _relocate_subthreshold_vertices(
            source, selection_altitude_m=SELECTION_ALTITUDE_M,
            altitude_floor_m=512e-9, requested_altitude_m=540e-9,
            max_vertex_displacement_m=1e-6, max_abs_volume_delta_m3=1e-12,
        )

        self.assertTrue(report["operations"])
        self.assertTrue(report["exact_reciprocal_face_owner_indices_preserved"])
        self.assertEqual(report["blocked_attempts"], [])
        self.assertTrue(all(not faces for faces in report["remaining_subfloor_faces"].values()))
        moved = next(op for op in report["operations"]
                     if set(op["affected_owner_ids"]) == {305, 306})
        self.assertGreaterEqual(moved["seed_altitude_after_m"], 512e-9)
        for delta in moved["max_displacement_from_relocation_input_by_owner_m"].values():
            self.assertLessEqual(delta, 1e-6)
        self.assertLessEqual(
            abs(report["per_owner"]["305"]["signed_volume_delta_m3"]), 1e-12)
        self.assertLessEqual(
            abs(report["per_owner"]["306"]["signed_volume_delta_m3"]), 1e-12)

        candidate_shared = _shared_face_map(relocated, stable_ids=(305, 306))
        self.assertEqual(len(candidate_shared), len(source_shared))
        self.assertTrue(all(np.dot(owners[0][2], owners[1][2]) < 0.0
                            for owners in candidate_shared.values()))
        candidate_point = np.asarray(moved["candidate_coordinate_f32_m"], dtype=np.float32)
        self.assertTrue(any(
            np.array_equal(relocated[sid]["vertices6"][vi, :3], candidate_point)
            for sid in (305, 306)
            for vi in range(len(relocated[sid]["vertices6"]))
        ))
        self._assert_closed_topology_unchanged(source, relocated)
        for sid, (vertices6, faces) in snapshots.items():
            np.testing.assert_array_equal(source[sid]["vertices6"], vertices6)
            np.testing.assert_array_equal(source[sid]["faces"], faces)


    def test_pinned_source_reconstructs_stale_diaphragm_ranges_without_geometry_change(self):
        from pathlib import Path
        import hashlib
        import json

        payload = Path(
            "/Users/n/numi-human-resting-evidence-20261005/airway-sibling-overlap-partition-001/"
            "resting-thorax.nhanatomy")
        receipt_path = Path(
            "/Users/n/numi-human-resting-evidence-20261005/skin-lowerlimb-anchor-rebind-004/"
            "resting-anatomy-receipt-recook-004/resting-anatomy-receipt.json")
        if not payload.is_file() or not receipt_path.is_file():
            self.skipTest("pinned Mini source geometry is not installed")
        self.assertEqual(
            hashlib.sha256(payload.read_bytes()).hexdigest(),
            "c6adae6522a2f7e5a8bd661035d464d686c843e96ab04ffc3ccd9aa1371569c1")
        self.assertEqual(
            hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "64c3b182d4ebc13bed468b72871b269887f8aecc2e7dc43622eb90d20bb0545e")

        from numilab_human.resting_anatomy_interface_patch import parse_payload, topology_report
        _, rows = parse_payload(payload)
        receipt = json.loads(receipt_path.read_text())
        original_positions = rows[311]["vertices6"].copy()
        original_faces = rows[311]["faces"].copy()
        original_topology = topology_report(original_faces)
        order, report = _reconstruct_diaphragm_interface_registration(receipt, rows)
        self.assertEqual(report["old_registration_status"],
                         "failed_exact_source_geometry_validation")
        self.assertEqual(report["reconstructed_patch_faces_by_lobe"],
                         {"305": 19668, "306": 21549, "307": 5512, "308": 486})
        self.assertEqual(
            [row["source_geometry_reconstructed_patch_topology"]["euler_characteristic"]
             for row in receipt["provenance"]["diaphragm_lung_interface"]["interface_rows"]],
            [1, 1, 1, 1])
        np.testing.assert_array_equal(rows[311]["vertices6"], original_positions)
        np.testing.assert_array_equal(rows[311]["faces"], original_faces[order])
        self.assertEqual(topology_report(rows[311]["faces"]), original_topology)
        refresh = _refresh_diaphragm_interface_registration(receipt, rows)
        self.assertEqual(refresh["registered_reciprocal_face_count"], 47215)
        for entry in receipt["provenance"]["diaphragm_lung_interface"]["interface_rows"]:
            start = int(entry["diaphragm_patch_face_start"])
            count = int(entry["diaphragm_patch_face_count"])
            self.assertGreater(count, 0)
            self.assertLessEqual(start + count, len(rows[311]["faces"]))

    def test_sliver_collapse_protects_registered_diaphragm_owner(self):
        source = _diaphragm_shared_skinny_rows()
        diaphragm_vertices = source[311]["vertices6"].copy()
        diaphragm_faces = source[311]["faces"].copy()
        source_shared = _shared_face_map(source, stable_ids=(305, 311))

        candidate, report, _ = _collapse_short_precision_slivers(
            source, altitude_limit_m=SELECTION_ALTITUDE_M,
            max_edge_length_m=20e-6, max_endpoint_displacement_m=10e-6,
            max_abs_volume_delta_m3=1e-12, max_operations=16,
            protected_owner_ids=(311,),
        )

        self.assertEqual(report["protected_reciprocal_owner_ids"], [311])
        self.assertTrue(any(311 in attempt.get("peer_owners", [])
                            and "protected reciprocal owner" in attempt["reason"]
                            for attempt in report["blocked_attempts"]))
        np.testing.assert_array_equal(candidate[311]["vertices6"], diaphragm_vertices)
        np.testing.assert_array_equal(candidate[311]["faces"], diaphragm_faces)
        self.assertEqual(set(_shared_face_map(candidate, stable_ids=(305, 311))),
                         set(source_shared))


if __name__ == "__main__":
    unittest.main()
