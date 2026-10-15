"""Typed, exact provenance checks for the explicit rib owner-frame visual bake."""
from __future__ import annotations

import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from numilab_human import model
from numilab_human.upper_limb_pose_audit import _compiled_member_geometry_check

MEMBER_ROWS = {
    48: ("FJ3228", "803a7134c1e69f640b87f9c6c4a846c869379446e771a1a7c9d18914c331a603"),
    49: ("FJ3229", "631b4f63625fa6cc89ad83961b2156abe5878d1baf4a8cb2b2acbaa06bc97f58"),
}
BODY_ARCHIVE_SHA = "31" * 32
MYOSIM_ARCHIVE_SHA = "42" * 32
REGISTRATION_SHA = "53" * 32
TRANSFORM = [
    [0.0, -0.001, 0.0, 0.01],
    [0.001, 0.0, 0.0, 0.02],
    [0.0, 0.0, 0.001, 0.03],
    [0.0, 0.0, 0.0, 1.0],
]
VERTICES_MM = [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]]
TRIANGLES = [[0, 1, 2]]
NORMALS = [(0.0, 0.0, 1.0)] * 3


def parser_fixture(root: Path) -> tuple[Path, dict]:
    expected_vertices = model._bodyparts_bake_vertices_to_core_body(
        VERTICES_MM, NORMALS, TRANSFORM, "rib fixture",
    )
    bake_rows = []
    anchors = []
    record_bytes = []
    vertex_bytes = []
    index_bytes = []
    for stable_id in range(1, 50):
        member_id, member_sha = MEMBER_ROWS.get(stable_id, (f"FJ-fixture-{stable_id}", f"{stable_id:064x}"))
        first_vertex = (stable_id - 1) * 3
        first_index = (stable_id - 1) * 3
        baked = stable_id in MEMBER_ROWS
        attrs = expected_vertices if baked else [
            (x * 0.001, y * 0.001, z * 0.001, *NORMALS[index])
            for index, (x, y, z) in enumerate(VERTICES_MM)
        ]
        attribute_bytes = b"".join(struct.pack("<6f", *vertex) for vertex in attrs)
        row_bake = None
        if baked:
            row_bake = {
                "stable_id": stable_id,
                "semantic": 51004,
                "member_id": member_id,
                "member_sha256": member_sha,
                "bodyparts_archive_sha256": BODY_ARCHIVE_SHA,
                "core_body_index": 20,
                "myosim_body": "torso",
                "source_record_index": 8,
                "source_transform_sha256": model._bodyparts_transform_matrix_sha256(TRANSFORM),
                "source_obj_mm_to_core_inertial_body_m": TRANSFORM,
                "declaration_file": "bake.json",
                "declaration_sha256": "64" * 32,
                "source_vertex_count": 3,
                "source_triangle_count": 1,
                "stored_vertex_attributes_f32_sha256": hashlib.sha256(attribute_bytes).hexdigest(),
                "output_record_pose_f32": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                "rounding": "test fixture",
            }
            bake_rows.append(row_bake)
        anchors.append({
            "member_id": member_id,
            "member_sha256": member_sha,
            "archive_sha256": BODY_ARCHIVE_SHA,
            "core_body_index": 20,
            "myosim_body": "torso",
            "source_record_index": 8,
            "vertex_count": 3,
            "triangle_count": 1,
            **({"body_local_bake": row_bake} if row_bake else {}),
        })
        pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
        record_bytes.append(struct.pack(
            "<6I8fI", 20, first_vertex, 3, first_index, 3, stable_id, *pose, 8,
        ))
        vertex_bytes.append(attribute_bytes)
        index_bytes.append(struct.pack("<3I", first_vertex, first_vertex + 1, first_vertex + 2))
    manifest = {
        "schema": "numi.human.bodyparts3d-myosim-major-bone-visual-payload.v1",
        "payload": {},
        "source": {
            "registration": {"file": "registration.json", "sha256": REGISTRATION_SHA},
            "myosim_source_archive_sha256": MYOSIM_ARCHIVE_SHA,
            "anchors": anchors,
        },
        "body_local_bakes": {
            "schema": model._BODYPARTS_BODY_LOCAL_BAKE_SCHEMA,
            "status": "explicit_visual_owner_frame_bake_not_physics_or_native_admission",
            "declaration_file": "bake.json",
            "declaration_sha256": "64" * 32,
            "registration_sha256": REGISTRATION_SHA,
            "myosim_source_archive_sha256": MYOSIM_ARCHIVE_SHA,
            "rows": bake_rows,
            "native_admission": False,
            "physics_or_mass_change": False,
        },
    }
    header = struct.pack(
        "<8s5I32s", b"NHBONES1", 3, 49, 147, 147,
        int(REGISTRATION_SHA[:8], 16), bytes.fromhex(MYOSIM_ARCHIVE_SHA),
    )
    payload = b"".join([header, *record_bytes, *vertex_bytes, *index_bytes])
    payload_path = root / "bodyparts3d-myosim-major-bones.nhbones"
    payload_path.write_bytes(payload)
    manifest["payload"] = {
        "file": payload_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "registration_fingerprint32": REGISTRATION_SHA[:8],
    }
    (root / "bodyparts3d-myosim-major-bones.manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return root, manifest


class BodyLocalBakeTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_registered_bake_is_exact_and_nonfinite_source_values_reject(self):
        baked = model._bodyparts_bake_vertices_to_core_body(
            VERTICES_MM, NORMALS, TRANSFORM, "rib fixture",
        )
        self.assertEqual(len(baked), 3)
        self.assertAlmostEqual(baked[0][0], 0.01, places=8)
        self.assertAlmostEqual(baked[0][1], 0.02, places=8)
        self.assertAlmostEqual(baked[0][2], 0.03, places=8)
        self.assertEqual(len({struct.pack("<6f", *row) for row in baked}), 3)
        with self.assertRaisesRegex(model.ImportError, "non-finite FP32"):
            model._bodyparts_bake_vertices_to_core_body(
                [[float("nan"), 0.0, 0.0]], [(0.0, 0.0, 1.0)], TRANSFORM, "bad",
            )

    def test_reader_retains_and_authenticates_baked_fp32_attributes(self):
        artifact, _ = parser_fixture(self.root)
        surfaces, descriptor, _ = model._numi_human_bone_envelope_surfaces(
            artifact, MYOSIM_ARCHIVE_SHA,
        )
        row48 = next(surface for surface in surfaces[20] if surface["stable_id"] == 48)
        self.assertEqual(row48["record_pose_f32"], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        self.assertEqual(len(row48["stored_vertex_attributes_f32"]), 3)
        self.assertEqual(row48["body_local_bake"]["member_id"], "FJ3228")
        self.assertEqual(descriptor["payload_abi"], 3)

        payload_path = artifact / "bodyparts3d-myosim-major-bones.nhbones"
        raw = bytearray(payload_path.read_bytes())
        header_size = struct.calcsize("<8s5I32s")
        record_size = struct.calcsize("<6I8fI")
        vertex_offset = header_size + 49 * record_size
        first_baked_vertex = vertex_offset + (48 - 1) * 3 * struct.calcsize("<6f")
        original = struct.unpack_from("<f", raw, first_baked_vertex)[0]
        struct.pack_into("<f", raw, first_baked_vertex, np.nextafter(np.float32(original), np.float32(np.inf)))
        payload_path.write_bytes(raw)
        manifest_path = artifact / "bodyparts3d-myosim-major-bones.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["payload"]["sha256"] = hashlib.sha256(raw).hexdigest()
        manifest["payload"]["bytes"] = len(raw)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(model.ImportError, "stored FP32 attributes drifted"):
            model._numi_human_bone_envelope_surfaces(artifact, MYOSIM_ARCHIVE_SHA)

    def test_composed_partition_reframe_requires_source_rederivation_and_face_ancestry(self):
        anchor = {
            "source": {
                "member_id": "FJ3228", "member_sha256": MEMBER_ROWS[48][1],
                "archive_sha256": BODY_ARCHIVE_SHA,
            },
            "target": {"name": "torso", "core_body_index": 20},
            "registration": {"source_obj_mm_to_core_inertial_body_m": TRANSFORM},
        }
        attributes = np.asarray([
            (0.01, 0.02, 0.03, 0.0, 0.0, 1.0),
            (0.11, 0.02, 0.03, 0.0, 0.0, 1.0),
            (0.01, 0.12, 0.03, 0.0, 0.0, 1.0),
            (0.11, 0.12, 0.03, 0.0, 0.0, 1.0),
        ], dtype=np.float32).astype(float).tolist()
        attribute_bytes = b"".join(struct.pack("<6f", *row) for row in attributes)
        partition_faces = [(0, 1, 2), (1, 3, 2)]
        face_bytes = b"".join(struct.pack("<3I", *triangle) for triangle in partition_faces)
        metadata = {
            "stable_id": 48,
            "semantic": 51004,
            "member_id": "FJ3228",
            "member_sha256": MEMBER_ROWS[48][1],
            "core_body_index": 20,
            "myosim_body": "torso",
            "source_triangle_count": 1,
            "compiled_triangle_count": 2,
            "reframe_mode": "source003_composed_partition_row_to_body20_local",
            "partition_row_face_indices_sha256": hashlib.sha256(face_bytes).hexdigest(),
            "stored_vertex_attributes_f32_sha256": hashlib.sha256(attribute_bytes).hexdigest(),
        }
        surface = {
            "stable_id": 48,
            "member_id": "FJ3228",
            "vertices": [list(row[:3]) for row in attributes],
            "triangles": partition_faces,
            "record_pose_f32": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "stored_vertex_attributes_f32": attributes,
            "body_local_bake": metadata,
            "body_local_bake_source_validation_sha256": "ab" * 32,
        }
        result = _compiled_member_geometry_check(anchor, VERTICES_MM, TRIANGLES, surface, np)
        self.assertTrue(result["passed"])
        self.assertFalse(result["source_topology_matches"])
        self.assertEqual(result["compiled_vertex_count"], 4)
        self.assertEqual(result["compiled_triangle_count"], 2)
        self.assertTrue(result["partition_face_ancestry_matches"])
        self.assertTrue(result["source003_partition_rederived"])

        stale_proof = copy.deepcopy(surface)
        stale_proof["body_local_bake_source_validation_sha256"] = "not-a-hash"
        self.assertFalse(_compiled_member_geometry_check(
            anchor, VERTICES_MM, TRIANGLES, stale_proof, np,
        )["passed"])

        altered_face = copy.deepcopy(surface)
        altered_face["triangles"] = [(0, 2, 1)]
        result = _compiled_member_geometry_check(anchor, VERTICES_MM, TRIANGLES, altered_face, np)
        self.assertFalse(result["passed"])
        self.assertFalse(result["partition_face_ancestry_matches"])

        altered_position = copy.deepcopy(surface)
        altered_position["vertices"][0][0] += 1.0e-6
        self.assertFalse(_compiled_member_geometry_check(
            anchor, VERTICES_MM, TRIANGLES, altered_position, np,
        )["passed"])

    def test_compiled_bake_geometry_uses_exact_stored_bytes_and_identity_pose(self):
        baked = model._bodyparts_bake_vertices_to_core_body(
            VERTICES_MM, NORMALS, TRANSFORM, "rib fixture",
        )
        anchor = {
            "source": {
                "member_id": "FJ3228", "member_sha256": MEMBER_ROWS[48][1],
                "archive_sha256": BODY_ARCHIVE_SHA,
            },
            "target": {"name": "torso", "core_body_index": 20},
            "registration": {"source_obj_mm_to_core_inertial_body_m": TRANSFORM},
        }
        metadata = {
            "stable_id": 48, "semantic": 51004, "member_id": "FJ3228",
            "member_sha256": MEMBER_ROWS[48][1],
            "bodyparts_archive_sha256": BODY_ARCHIVE_SHA,
            "core_body_index": 20, "myosim_body": "torso",
            "source_record_index": 8,
            "source_obj_mm_to_core_inertial_body_m": TRANSFORM,
            "source_transform_sha256": model._bodyparts_transform_matrix_sha256(TRANSFORM),
            "output_record_pose_f32": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        }
        surface = {
            "stable_id": 48, "member_id": "FJ3228",
            "vertices": [list(row[:3]) for row in baked],
            "triangles": [(0, 1, 2)],
            "record_pose_f32": metadata["output_record_pose_f32"],
            "stored_vertex_attributes_f32": baked,
            "body_local_bake": metadata,
        }
        result = _compiled_member_geometry_check(anchor, VERTICES_MM, TRIANGLES, surface, np)
        self.assertTrue(result["passed"])
        self.assertTrue(result["stored_fp32_attributes_exact"])
        self.assertTrue(result["identity_record_pose"])
        damaged = copy.deepcopy(surface)
        altered = list(damaged["stored_vertex_attributes_f32"][0])
        altered[0] = float(np.nextafter(np.float32(altered[0]), np.float32(np.inf)))
        damaged["stored_vertex_attributes_f32"][0] = tuple(altered)
        damaged["vertices"][0][0] = altered[0]
        result = _compiled_member_geometry_check(anchor, VERTICES_MM, TRIANGLES, damaged, np)
        self.assertFalse(result["passed"])
        self.assertFalse(result["stored_fp32_attributes_exact"])
        wrong_pose = copy.deepcopy(surface)
        wrong_pose["record_pose_f32"][0] = 1.0e-6
        result = _compiled_member_geometry_check(anchor, VERTICES_MM, TRIANGLES, wrong_pose, np)
        self.assertFalse(result["passed"])
        self.assertFalse(result["identity_record_pose"])


if __name__ == "__main__":
    unittest.main()
