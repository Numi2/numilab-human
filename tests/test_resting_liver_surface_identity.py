import unittest

import numpy as np

from numilab_human.resting_liver_surface_identity import (
    LIVER_PATCH_IDS,
    LIVER_PATCH_INTERPRETATION,
    LIVER_PATCH_ROLE,
    liver_surface_identity_metadata,
    validate_liver_patch_coverage,
    validate_liver_surface_identity_metadata,
)


def _cube_fixture():
    vertices = np.asarray(
        [
            (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
            (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
        ],
        dtype=np.int64,
    )
    owners = np.asarray([14, 14, 15, 15, 16, 16, 17, 17, 18, 19, 20, 21], dtype=np.int64)
    patch_rows = {}
    source_map = {}
    for stable_id in LIVER_PATCH_IDS:
        face_ids = np.flatnonzero(owners == stable_id)
        local_vertices = []
        local_faces = []
        index_by_source = {}
        for face_id in face_ids:
            local_face = []
            for source_vertex in faces[face_id]:
                source_vertex = int(source_vertex)
                if source_vertex not in index_by_source:
                    index_by_source[source_vertex] = len(local_vertices)
                    local_vertices.append(vertices[source_vertex])
                local_face.append(index_by_source[source_vertex])
            local_faces.append(local_face)
        xyz = np.asarray(local_vertices, dtype=np.float32)
        normals = np.zeros_like(xyz)
        patch_rows[stable_id] = {
            "vertices6": np.column_stack((xyz, normals)).astype(np.float32),
            "faces": np.asarray(local_faces, dtype=np.int64),
        }
        source_map[str(stable_id)] = {
            "body_index": 20,
            "layer": 1,
            "source_owner_metadata": {
                "source_atlas": "Z-Anatomy",
                "geometry_role": LIVER_PATCH_ROLE,
            },
            "repair": {
                "interpretation": LIVER_PATCH_INTERPRETATION,
                "patch_face_count": int(len(face_ids)),
            },
        }
    return vertices, faces, owners, patch_rows, source_map


class RestingLiverSurfaceIdentityTests(unittest.TestCase):
    def test_exact_representation_receipt_contract(self):
        metadata = liver_surface_identity_metadata()
        validate_liver_surface_identity_metadata(metadata)
        bad_alias = dict(metadata, retired_source_aliases={"22": 20})
        with self.assertRaises(ValueError):
            validate_liver_surface_identity_metadata(bad_alias)
        extra_claim = dict(metadata, segment_volumes_defined=True)
        with self.assertRaises(ValueError):
            validate_liver_surface_identity_metadata(extra_claim)
        extra_key = dict(metadata, measured_subject=True)
        with self.assertRaises(ValueError):
            validate_liver_surface_identity_metadata(extra_key)

    def test_aggregate_shell_is_covered_once_by_eight_display_patches(self):
        vertices, faces, owners, rows, source_map = _cube_fixture()
        report = validate_liver_patch_coverage(
            vertices=vertices,
            faces=faces,
            stable_id_per_face=owners,
            patch_rows=rows,
            source_id_map=source_map,
        )
        self.assertEqual(report["aggregate_source_face_count"], 12)
        self.assertTrue(report["all_aggregate_faces_assigned_exactly_once"])
        self.assertFalse(report["segment_volumes_defined"])
        self.assertEqual(set(report["compiled_source_face_count_by_stable_id"]), set(map(str, LIVER_PATCH_IDS)))

    def test_missing_face_and_reversed_patch_are_rejected(self):
        vertices, faces, owners, rows, source_map = _cube_fixture()
        rows[14]["faces"] = rows[14]["faces"][:-1]
        with self.assertRaises(ValueError):
            validate_liver_patch_coverage(
                vertices=vertices, faces=faces, stable_id_per_face=owners,
                patch_rows=rows, source_id_map=source_map,
            )

        vertices, faces, owners, rows, source_map = _cube_fixture()
        rows[14]["faces"][0] = rows[14]["faces"][0][::-1]
        with self.assertRaises(ValueError):
            validate_liver_patch_coverage(
                vertices=vertices, faces=faces, stable_id_per_face=owners,
                patch_rows=rows, source_id_map=source_map,
            )

    def test_extra_or_wrongly_owned_surface_is_rejected(self):
        vertices, faces, owners, rows, source_map = _cube_fixture()
        rows[22] = rows.pop(21)
        with self.assertRaises(ValueError):
            validate_liver_patch_coverage(
                vertices=vertices, faces=faces, stable_id_per_face=owners,
                patch_rows=rows, source_id_map=source_map,
            )

        vertices, faces, owners, rows, source_map = _cube_fixture()
        source_map["16"]["body_index"] = 7
        with self.assertRaises(ValueError):
            validate_liver_patch_coverage(
                vertices=vertices, faces=faces, stable_id_per_face=owners,
                patch_rows=rows, source_id_map=source_map,
            )


if __name__ == "__main__":
    unittest.main()
