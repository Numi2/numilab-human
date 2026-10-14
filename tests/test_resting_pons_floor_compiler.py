from __future__ import annotations

import hashlib

import numpy as np
import pytest

from numilab_human.resting_anatomy_interface_patch import encode_payload
from numilab_human.resting_pleura_proxy import _parse_payload
from numilab_human.resting_pons_floor_compiler import (
    build_row_replacements,
    compile_payload,
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _row(stable_id: int, vertices: np.ndarray, faces: np.ndarray) -> dict:
    return {
        "body_index": 23 if stable_id in (323, 345) else 0,
        "layer": 1,
        "flags": 0,
        "vertices6": np.asarray(vertices, dtype="<f4"),
        "faces": np.asarray(faces, dtype="<i8"),
    }


def _fixture():
    rows = {}
    for stable_id in range(1, 526):
        vertices = np.asarray(
            [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1]],
            dtype="<f4",
        )
        rows[stable_id] = _row(stable_id, vertices, np.asarray([[0, 1, 2]], dtype="<i8"))

    arrays = {}
    metadata = {}
    for stable_id, name, source_vertices, source_faces, candidate_vertices, candidate_faces in (
        (345, "pons", 5154, 10304, 8002, 16000),
        (323, "fourth_ventricle", 3567, 6940, 9506, 12676),
    ):
        prefix = np.resize(rows[stable_id]["vertices6"], (source_vertices, 6)).astype("<f4")
        candidate = np.resize(prefix, (candidate_vertices, 6)).astype("<f4")
        faces = np.resize(np.asarray([[0, 1, 2]], dtype="<i8"), (candidate_faces, 3))
        arrays[f"{name}_vertices6_f32"] = candidate
        arrays[f"{name}_faces_i64"] = faces
        metadata[stable_id] = {
            "source_vertices": source_vertices,
            "source_faces": source_faces,
            "candidate_vertices": candidate_vertices,
            "candidate_faces": candidate_faces,
            "vertices_sha256": _sha(np.ascontiguousarray(candidate, dtype="<f4").tobytes()),
            "faces_sha256": _sha(np.ascontiguousarray(faces, dtype="<i8").tobytes()),
        }
        rows[stable_id] = _row(
            stable_id, prefix, np.resize(np.asarray([[0, 1, 2]], dtype="<i8"), (source_faces, 3))
        )

    arrays.update({
        "pons_face_source_parent_i64": np.zeros(16000, dtype="<i8"),
        "pons_face_child_order_i64": np.zeros(16000, dtype="<i8"),
        "fourth_ventricle_face_source_parent_i64": np.zeros(12676, dtype="<i8"),
        "fourth_ventricle_face_child_order_i64": np.zeros(12676, dtype="<i8"),
        "fourth_ventricle_face_source_surface_stable_id_i64": np.concatenate((
            np.full(9362, 323, dtype="<i8"), np.full(3314, 345, dtype="<i8")
        )),
        "pons_new_vertex_parent_face_i64": np.zeros(2848, dtype="<i8"),
        "pons_new_vertex_parent_weights_f64": np.full((2848, 3), 1 / 3, dtype="<f8"),
        "pons_new_vertex_source_exact_xyz_f64": np.zeros((2848, 3), dtype="<f8"),
        "fourth_ventricle_new_vertex_parent_face_i64": np.zeros(2848, dtype="<i8"),
        "fourth_ventricle_new_vertex_parent_weights_f64": np.full((2848, 3), 1 / 3, dtype="<f8"),
        "fourth_ventricle_new_vertex_source_exact_xyz_f64": np.zeros((2848, 3), dtype="<f8"),
        "fourth_ventricle_cap_face_source_pons_face_i64": np.zeros(3314, dtype="<i8"),
        "fourth_ventricle_cap_face_source_pons_child_order_i64": np.zeros(3314, dtype="<i8"),
        "fourth_ventricle_cap_face_source_pons_output_face_i64": np.zeros(3314, dtype="<i8"),
        "fourth_ventricle_cap_face_source_ventricle_face_parent_i64": np.zeros(3314, dtype="<i8"),
        "fourth_ventricle_cap_face_pons_barycentric_parent_weights_f64": np.full(
            (3314, 3, 3), 1 / 3, dtype="<f8"
        ),
        "fourth_ventricle_cap_face_pons_source_exact_xyz_f64": np.zeros(
            (3314, 3, 3), dtype="<f8"
        ),
    })
    report = {
        "anatomy": {
            "owner_body_index": 23,
            "typed_rows": [[51010, 323], [51010, 345]],
        },
        "pons_candidate": metadata[345],
        "fourth_ventricle_candidate": metadata[323],
        "reciprocal_floor": {"cap_face_count": 3314},
    }
    return rows, arrays, report


def test_compiler_replaces_only_the_two_pinned_rows_and_preserves_abi5_header():
    rows, arrays, report = _fixture()
    raw = encode_payload(
        (b"NHANAT1\0", 5, 525, 0, 0, 0x12345678, b"s" * 32),
        rows,
    )
    _header, _records, decoded_rows = _parse_payload(raw)
    replacements = build_row_replacements(decoded_rows, arrays, report)

    output, proof = compile_payload(raw, replacements)

    assert proof["replaced_stable_ids"] == [323, 345]
    assert proof["unmodified_rows_content_exact"] == 523
    assert proof["registration_and_source_header_unchanged"] is True
    assert len(output) > len(raw)


def test_compiler_rejects_candidate_that_changes_original_source_prefix():
    rows, arrays, report = _fixture()
    arrays["pons_vertices6_f32"][0, 0] = np.float32(0.25)

    with pytest.raises(ValueError, match="candidate vertex digest disagrees"):
        build_row_replacements(rows, arrays, report)
