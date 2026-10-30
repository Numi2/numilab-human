#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROLE_FILES = {
    "O": ("O-fixed-boundary-star-batch-proposal.npz", "candidate-O-csr-canonical.npz"),
    "I": ("I-fixed-boundary-star-batch-proposal.npz", "candidate-I-saved-op-replay.npz"),
    "D": ("D-fixed-boundary-star-batch-proposal.npz", "candidate-D-saved-op-replay-source-mapped.npz"),
}
TYPED_FIELDS = (
    "face_source_layer",
    "face_source_body",
    "face_source_bone",
    "face_source_face_id",
    "face_source_original_id",
    "face_source_identity0",
    "face_source_identity1",
)
FACE_CSR = {
    "O": {
        "face_ancestry_offsets": (
            "face_ancestry_source_face_rows",
            "face_ancestry_source_original_ids",
            "face_ancestry_parent_face_rows",
        ),
        "microedge_parent_input_face_offsets": ("microedge_parent_input_face_rows",),
    },
    "I": {
        "face_parent_offsets": ("face_parent_rows",),
        "edge_flip_parent_offsets": ("edge_flip_parent_face_rows",),
        "microedge_parent_input_face_offsets": ("microedge_parent_input_face_rows",),
    },
    "D": {
        "candidate_face_ancestry_offsets": ("candidate_face_ancestry_rows",),
        "union_face_ancestry_offsets": (
            "union_face_ancestry_rows",
            "union_face_ancestry_run_original_ids",
            "union_face_ancestry_face_ids",
        ),
        "operational_output_face_parent_offsets": ("operational_output_face_parent_rows",),
        "microedge_parent_input_face_offsets": ("microedge_parent_input_face_rows",),
    },
}
POINT_CSR = {
    "O": {
        "microedge_parent_input_point_offsets": ("microedge_parent_input_point_rows",),
        "reference_vertex_parent_face_offsets": (
            "reference_vertex_parent_face_rows",
            "reference_vertex_parent_face_barycentric",
        ),
    },
    "I": {
        "microedge_parent_input_point_offsets": ("microedge_parent_input_point_rows",),
        "reference_original_input_vertex_offsets": ("reference_original_input_vertex_rows",),
        "reference_vertex_parent_face_offsets": (
            "reference_vertex_parent_face_rows",
            "reference_vertex_parent_face_barycentric",
        ),
    },
    "D": {
        "microedge_parent_input_point_offsets": ("microedge_parent_input_point_rows",),
        "output_vertex_input_parent_offsets": ("output_vertex_input_parent_rows",),
    },
}
SYNTHETIC_BASE = {"O": 700_000_000, "I": 702_000_000, "D": 704_000_000}
EXPECTED_SOURCE_SHA = {
    "O": "578cee97a3bbecb73c0e1a6d7c26f45ac6cc111a552996f3910866a4a3652a9b",
    "I": "9721102247157eeb42c59633fda59d63fd5ba48ad11ba4e63987834136711e31",
    "D": "15afa8eecdaeff456cda25ef26890da60d60dd06e0412ed2035c77a8cbeb460f",
}
EXPECTED_RAW_SHA = {
    "O": "e1f406aa14e0cdf8333c56fd0fa34df88f6db3a31c6ddf2d5907a4dd1cfa222d",
    "I": "f2a985696cd0af72e2c9bfd0b76f4282e0af9b0ccc5b726dcdcaa8fedc6e6472",
    "D": "2c46bd93599ead65e1ba55188a7196503e660a5bffd8e739035afa038f5ac786",
}
SEMANTICS = (
    "Each derived face carries the complete deduplicated seven-field typed identity union "
    "of its retired connected DP patch. This is patch-level source support, not a per-triangle "
    "overlap or a single-face ancestry claim."
)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in z.files}

def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def tuple_union(arrays: dict[str, np.ndarray], rows: np.ndarray) -> list[tuple[int, ...]]:
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("patch must retire at least one source face")
    n = len(arrays[TYPED_FIELDS[0]])
    if np.any(rows < 0) or np.any(rows >= n) or len(np.unique(rows)) != len(rows):
        raise ValueError("patch retired-face rows are duplicate or out of range")
    values = [arrays[key][rows].astype(np.int64, copy=False) for key in TYPED_FIELDS]
    return sorted({tuple(int(values[col][i]) for col in range(len(TYPED_FIELDS))) for i in range(len(rows))})

def union_sha(rows: list[tuple[int, ...]]) -> str:
    payload = canonical_json([list(x) for x in rows]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def remap_csr(
    offsets: np.ndarray,
    values_by_lane: dict[str, np.ndarray],
    owner_rows: np.ndarray,
    owner_count: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.shape != (owner_count + 1,) or offsets[0] != 0:
        raise ValueError("CSR offsets have wrong owner domain")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("CSR offsets are not monotone")
    for key, data in values_by_lane.items():
        if np.asarray(data).shape[0] != int(offsets[-1]):
            raise ValueError(f"CSR lane {key} length differs from final offset")
    out_offsets = [0]
    out_chunks: dict[str, list[np.ndarray]] = {key: [] for key in values_by_lane}
    for row in owner_rows:
        if row < 0:
            start = end = 0
        else:
            if row >= owner_count:
                raise ValueError("CSR owner row outside input domain")
            start, end = int(offsets[row]), int(offsets[row + 1])
        for key, data in values_by_lane.items():
            out_chunks[key].append(np.asarray(data)[start:end])
        out_offsets.append(out_offsets[-1] + end - start)
    result = {}
    for key, chunks in out_chunks.items():
        data = np.asarray(values_by_lane[key])
        if chunks and out_offsets[-1]:
            result[key] = np.concatenate(chunks, axis=0).astype(data.dtype, copy=False)
        else:
            result[key] = np.empty((0,) + data.shape[1:], dtype=data.dtype)
    return np.asarray(out_offsets, dtype=np.int64), result

def remap_csr_with_patch_unions(
    offsets: np.ndarray,
    values_by_lane: dict[str, np.ndarray],
    output_to_input: np.ndarray,
    input_count: int,
    patch_rows: dict[int, dict[str, Any]],
    require_nonempty_new: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.shape != (input_count + 1,) or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("face CSR offsets invalid for patch-union reconstruction")
    for key, value in values_by_lane.items():
        if len(value) != int(offsets[-1]):
            raise ValueError(f"face CSR lane {key} length mismatch")
    out_offsets = [0]
    chunks = {key: [] for key in values_by_lane}
    for out_row, source_row in enumerate(output_to_input):
        if source_row >= 0:
            a, b = int(offsets[source_row]), int(offsets[source_row + 1])
            for key, value in values_by_lane.items():
                chunks[key].append(np.asarray(value)[a:b])
            out_offsets.append(out_offsets[-1] + b - a)
            continue
        patch = patch_rows.get(out_row)
        if patch is None:
            raise ValueError(f"derived face {out_row} has no patch record for CSR")
        records = set()
        for parent_row in patch["retired"]:
            a, b = int(offsets[parent_row]), int(offsets[parent_row + 1])
            for i in range(a, b):
                record = []
                for value in values_by_lane.values():
                    item = np.asarray(value)[i]
                    record.append(tuple(item.reshape(-1).tolist()) if np.ndim(item) else int(item))
                records.add(tuple(record))
        ordered = sorted(records)
        if require_nonempty_new and not ordered:
            raise ValueError(f"derived face {out_row} has no parent tuples in this face CSR")
        for lane_index, (key, value) in enumerate(values_by_lane.items()):
            vals = [rec[lane_index] for rec in ordered]
            if not vals:
                chunks[key].append(np.empty((0,) + np.asarray(value).shape[1:], dtype=np.asarray(value).dtype))
            elif np.asarray(value).ndim == 1:
                chunks[key].append(np.asarray(vals, dtype=np.asarray(value).dtype))
            else:
                chunks[key].append(np.asarray(vals, dtype=np.asarray(value).dtype).reshape((-1,) + np.asarray(value).shape[1:]))
        out_offsets.append(out_offsets[-1] + len(ordered))
    out_values = {}
    for key, lane_chunks in chunks.items():
        src = np.asarray(values_by_lane[key])
        if lane_chunks and out_offsets[-1]:
            out_values[key] = np.concatenate(lane_chunks, axis=0).astype(src.dtype, copy=False)
        else:
            out_values[key] = np.empty((0,) + src.shape[1:], dtype=src.dtype)
    return np.asarray(out_offsets, dtype=np.int64), out_values

def build_current_input_face_ancestry(
    source: dict[str, np.ndarray],
    output_to_input: np.ndarray,
    patch_rows: dict[int, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original_ids = np.asarray(source["face_source_original_id"], dtype=np.int64)
    offsets = [0]
    source_rows, source_original_ids, parent_rows = [], [], []
    for output_row, input_row in enumerate(output_to_input):
        rows = [int(input_row)] if input_row >= 0 else list(map(int, patch_rows[output_row]["retired"]))
        tuples = sorted({(row, int(original_ids[row]), row) for row in rows})
        if not tuples:
            raise ValueError(f"face {output_row} has empty current-input ancestry")
        for sr, oid, pr in tuples:
            source_rows.append(sr)
            source_original_ids.append(oid)
            parent_rows.append(pr)
        offsets.append(len(source_rows))
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(source_rows, dtype=np.int64),
        np.asarray(source_original_ids, dtype=np.int64),
        np.asarray(parent_rows, dtype=np.int64),
    )

def compact_rows(raw_faces: np.ndarray, raw_points: np.ndarray, raw_to_input: np.ndarray):
    raw_faces = np.asarray(raw_faces, dtype=np.int64)
    raw_points = np.asarray(raw_points)
    raw_to_input = np.asarray(raw_to_input, dtype=np.int64)
    if raw_points.ndim != 2 or raw_points.shape[1] != 3 or len(raw_points) != len(raw_to_input):
        raise ValueError("raw points and point map domains differ")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
        raise ValueError("raw face array must be triangles")
    if np.any(raw_faces < 0) or np.any(raw_faces >= len(raw_points)):
        raise ValueError("raw face point index out of range")
    if np.any((raw_faces[:, 0] == raw_faces[:, 1]) |
              (raw_faces[:, 1] == raw_faces[:, 2]) |
              (raw_faces[:, 0] == raw_faces[:, 2])):
        raise ValueError("raw face repeats a point row")
    used = np.unique(raw_faces.reshape(-1))
    raw_to_compact = np.full(len(raw_points), -1, dtype=np.int64)
    raw_to_compact[used] = np.arange(len(used), dtype=np.int64)
    points = raw_points[used].copy()
    faces = raw_to_compact[raw_faces]
    if np.any(faces < 0):
        raise ValueError("compaction produced an unmapped face point")
    input_to_compact = np.full(int(raw_to_input.max()) + 1, -1, dtype=np.int64)
    input_to_compact[raw_to_input[used]] = np.arange(len(used), dtype=np.int64)
    return points, faces, used, raw_to_compact, input_to_compact

def _fill_new_faces(values: np.ndarray, output_to_input: np.ndarray) -> np.ndarray:
    src = np.asarray(values)
    if src.ndim == 0:
        raise ValueError("face field unexpectedly scalar")
    if np.issubdtype(src.dtype, np.integer):
        dtype = np.result_type(src.dtype, np.int64)
        out = np.full((len(output_to_input),) + src.shape[1:], -1, dtype=dtype)
    elif np.issubdtype(src.dtype, np.floating):
        out = np.full((len(output_to_input),) + src.shape[1:], np.nan, dtype=src.dtype)
    elif src.dtype.kind in ("U", "S"):
        out = np.full((len(output_to_input),) + src.shape[1:], "", dtype=src.dtype)
    elif src.dtype.kind == "b":
        out = np.zeros((len(output_to_input),) + src.shape[1:], dtype=src.dtype)
    else:
        raise ValueError(f"unsupported face field dtype {src.dtype}")
    retained = np.flatnonzero(output_to_input >= 0)
    out[retained] = src[output_to_input[retained]]
    return out

def _patch_record(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    patches = data.get("patches")
    if not isinstance(patches, list) or len(patches) != 1:
        raise ValueError(f"expected one patch record in {path}")
    patch = patches[0]
    if data.get("patch_id") != patch.get("patch_id"):
        raise ValueError(f"top-level and nested patch IDs differ in {path}")
    patch = dict(patch)
    patch["source_lineage"] = data.get("source_lineage", patch.get("source_lineage", {}))
    patch["role"] = data.get("role")
    patch["operation_json_sha256"] = sha256(path)
    patch["operation_json_path"] = str(path)
    return patch

def verify_map_and_operations(
    role: str,
    src: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    raw_path: Path,
    batch: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    n = len(src["points_world_m"])
    f = len(src["faces"])
    raw_points = raw["points_world_m"]
    raw_faces = raw["faces"].astype(np.int64, copy=False)
    po = raw["point_output_to_input_rows"].astype(np.int64, copy=False)
    pi = raw["point_input_to_output_row"].astype(np.int64, copy=False)
    fo = raw["face_output_to_input_rows"].astype(np.int64, copy=False)
    retired = raw["retired_input_face_rows"].astype(np.int64, copy=False)
    new_input_faces = raw["new_faces_input_point_rows"].astype(np.int64, copy=False)
    patch_ids_per_new = raw["new_output_face_patch_ids"].astype(np.int64, copy=False)
    op_ids = raw["candidate_patch_operation_ids"].astype(str)
    if raw.get("source_candidate_sha256") is None or str(raw["source_candidate_sha256"].reshape(-1)[0]) != str(src.get("_sha256", "")):
        raise ValueError(f"{role} raw proposal source pin does not match source file")
    if len(po) != len(raw_points) or len(pi) != n or len(fo) != len(raw_faces):
        raise ValueError(f"{role} point/face map dimensions do not match")
    if np.any(po < 0) or np.any(po >= n) or len(np.unique(po)) != len(po):
        raise ValueError(f"{role} raw output-to-input point map is not injective")
    expected_pi = np.full(n, -1, dtype=np.int64)
    expected_pi[po] = np.arange(len(po), dtype=np.int64)
    if not np.array_equal(pi, expected_pi):
        raise ValueError(f"{role} point maps are not exact inverses")
    if raw_points.dtype != src["points_world_m"].dtype:
        raise ValueError(f"{role} point dtype changed: {src['points_world_m'].dtype} -> {raw_points.dtype}")
    if raw_points.tobytes() != src["points_world_m"][po].tobytes():
        raise ValueError(f"{role} raw output coordinates differ from exact source point bytes")
    if np.any(fo < -1) or np.any(fo >= f):
        raise ValueError(f"{role} output face map contains invalid source face rows")
    retained_input = fo[fo >= 0]
    if len(np.unique(retained_input)) != len(retained_input):
        raise ValueError(f"{role} output face map duplicates a source face")
    if len(np.unique(retired)) != len(retired) or np.any(retired < 0) or np.any(retired >= f):
        raise ValueError(f"{role} retired face rows are invalid")
    if set(map(int, retained_input)) != set(range(f)) - set(map(int, retired)):
        raise ValueError(f"{role} retained/retired source face sets do not partition input")
    new_rows = np.flatnonzero(fo == -1)
    if len(new_rows) != len(new_input_faces) or len(new_rows) != len(patch_ids_per_new):
        raise ValueError(f"{role} new face rows and patch maps have different lengths")
    if np.any(new_input_faces < 0) or np.any(new_input_faces >= n):
        raise ValueError(f"{role} new triangle references an invalid input point")
    if np.any(pi[new_input_faces] < 0):
        raise ValueError(f"{role} new triangle references a retired point")
    if not np.array_equal(raw_faces[new_rows], pi[new_input_faces]):
        raise ValueError(f"{role} new output triangles do not map to the operation's input-point triangles")
    removed_points = np.flatnonzero(pi < 0)
    if len(np.unique(removed_points)) != len(removed_points):
        raise ValueError("duplicate removed point rows")
    if len(removed_points):
        retained_input_faces = po[raw_faces[fo >= 0]]
        if np.isin(retained_input_faces, removed_points).any():
            raise ValueError(f"{role} a retained face references a retired point")
    operations = []
    operation_files = {}
    for path in batch.glob(f"{role}-star-*.json"):
        patch = _patch_record(path)
        if patch["patch_id"] in operation_files:
            raise ValueError(f"duplicate operation JSON patch id {patch['patch_id']}")
        operation_files[patch["patch_id"]] = patch
    if len(op_ids) != len(np.unique(patch_ids_per_new)):
        raise ValueError(f"{role} operation ID list does not match patch count")
    if len(op_ids) != len(np.unique(op_ids)):
        raise ValueError(f"{role} operation IDs are not unique")
    expected_ids = set(range(len(op_ids)))
    if set(map(int, np.unique(patch_ids_per_new))) != expected_ids:
        raise ValueError(f"{role} numeric patch IDs are not a dense 0-based domain")
    for pid, op_id in enumerate(op_ids.tolist()):
        if op_id not in operation_files:
            raise ValueError(f"{role} missing operation JSON for {op_id}")
        op = operation_files[op_id]
        if op["role"] != role:
            raise ValueError(f"{role} operation file has wrong role")
        idx = np.flatnonzero(patch_ids_per_new == pid)
        out_rows = new_rows[idx]
        op_rows = np.asarray(op["new_output_face_rows"], dtype=np.int64)
        op_new_faces = np.asarray(op["new_faces_input_point_rows"], dtype=np.int64)
        op_retired = np.asarray(op["retired_input_face_rows"], dtype=np.int64)
        if not np.array_equal(out_rows, op_rows):
            raise ValueError(f"{role} {op_id} output face rows differ from raw map")
        if not np.array_equal(new_input_faces[idx], op_new_faces):
            raise ValueError(f"{role} {op_id} new face input rows differ from raw map")
        if not set(map(int, op_retired)).issubset(set(map(int, retired))):
            raise ValueError(f"{role} {op_id} retired rows are absent from raw map")
        op["numeric_patch_id"] = pid
        op["new_output_face_rows"] = out_rows.tolist()
        op["retired_input_face_rows"] = op_retired.tolist()
        operations.append(op)
    if not set(op_ids.tolist()).issubset(operation_files):
        raise ValueError(f"{role} operation file inventory is incomplete")
    centers = [int(op["center_vertex_input_row"]) for op in operations if op.get("center_vertex_input_row") is not None]
    if centers and set(centers) != set(map(int, removed_points)):
        raise ValueError(f"{role} retired points do not equal the operation center-point set")
    return operations, fo, po

def build_role(
    role: str,
    src_path: Path,
    raw_path: Path,
    batch: Path,
    protected_routes: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    source = load_npz(src_path)
    source["_sha256"] = sha256(src_path)
    raw = load_npz(raw_path)
    operations, face_out_to_in, point_raw_to_in = verify_map_and_operations(role, source, raw, raw_path, batch)
    in_faces = source["faces"].astype(np.int64, copy=False)
    out_faces_raw = raw["faces"].astype(np.int64, copy=False)
    out_points_raw = raw["points_world_m"]
    used_raw = np.unique(out_faces_raw.reshape(-1))
    raw_to_compact = np.full(len(out_points_raw), -1, dtype=np.int64)
    raw_to_compact[used_raw] = np.arange(len(used_raw), dtype=np.int64)
    points = out_points_raw[used_raw].copy()
    faces = raw_to_compact[out_faces_raw]
    if np.any(faces < 0):
        raise ValueError(f"{role} compact face references missing point")
    output_to_input_points = point_raw_to_in[used_raw]
    input_to_output_points = np.full(len(source["points_world_m"]), -1, dtype=np.int64)
    input_to_output_points[output_to_input_points] = np.arange(len(used_raw), dtype=np.int64)
    if len(np.unique(output_to_input_points)) != len(output_to_input_points):
        raise ValueError(f"{role} compaction point map is not injective")
    if points.dtype != source["points_world_m"].dtype or points.tobytes() != source["points_world_m"][output_to_input_points].tobytes():
        raise ValueError(f"{role} compaction changed retained coordinate bytes")
    new_face_rows = np.flatnonzero(face_out_to_in == -1)
    patch_ids = raw["new_output_face_patch_ids"].astype(np.int64, copy=False)
    if len(new_face_rows) != len(patch_ids):
        raise ValueError(f"{role} new-face patch vector has wrong length")
    output_face_to_input = face_out_to_in.copy()
    input_face_to_output = np.full(len(in_faces), -1, dtype=np.int64)
    retained = np.flatnonzero(output_face_to_input >= 0)
    input_face_to_output[output_face_to_input[retained]] = retained

    # Preserve every face-domain source array for retained rows; leave direct ancestry empty
    # for derived faces and record its full patch union separately below.
    out: dict[str, np.ndarray] = {}
    face_csr = FACE_CSR[role]
    point_csr = POINT_CSR[role]
    face_csr_data = {key for lanes in face_csr.values() for key in lanes}
    point_csr_data = {key for lanes in point_csr.values() for key in lanes}
    face_csr_offsets = set(face_csr)
    point_csr_offsets = set(point_csr)
    non_face_special = {
        "points_world_m", "points_world_m_f32", "faces",
        "face_source_identity_schema_by_original_id_json",
    }
    for key, value in source.items():
        if key == "_sha256":
            continue
        if key in non_face_special or key in face_csr_offsets or key in face_csr_data or key in point_csr_offsets or key in point_csr_data:
            continue
        if value.ndim > 0 and value.shape[0] == len(in_faces):
            out[key] = _fill_new_faces(value, output_face_to_input)
        elif value.ndim > 0 and value.shape[0] == len(source["points_world_m"]):
            out[key] = value[output_to_input_points].copy()
        else:
            out[key] = value.copy()

    out["points_world_m"] = points
    out["points_world_m_f32"] = points.astype(np.float32, copy=True)
    out["faces"] = faces.astype(np.int64, copy=False)

    # Rebuild face CSR for exact retained ancestry. Derived rows have empty direct rows;
    # full patch support is in the seven-lane patch-union CSR.
    for offsets_key, lanes in face_csr.items():
        values = {lane: source[lane] for lane in lanes}
        offsets, rebuilt = remap_csr(source[offsets_key], values, output_face_to_input, len(in_faces))
        out[offsets_key] = offsets
        out.update(rebuilt)

    # Rebuild point CSR in compacted output order, retaining original source row domains.
    for offsets_key, lanes in point_csr.items():
        values = {lane: source[lane] for lane in lanes}
        offsets, rebuilt = remap_csr(source[offsets_key], values, output_to_input_points, len(source["points_world_m"]))
        out[offsets_key] = offsets
        out.update(rebuilt)

    # Construct a complete typed union per patch and lay it out in final face-row order.
    source_schema = json.loads(str(source["face_source_identity_schema_by_original_id_json"].reshape(-1)[0]))
    existing_orig = set(map(int, np.unique(source["face_source_original_id"])))
    if not set(map(str, existing_orig)).issubset(source_schema):
        raise ValueError(f"{role} source schema omits a used source original ID")
    schema_ids = {int(k) for k in source_schema if str(k).lstrip("-").isdigit()}
    start_id = max(SYNTHETIC_BASE[role], max(existing_orig | schema_ids, default=0) + 1)
    synthetic_ids = {op["numeric_patch_id"]: start_id + op["numeric_patch_id"] for op in operations}
    if set(synthetic_ids.values()) & (existing_orig | schema_ids):
        raise ValueError(f"{role} synthetic source IDs collide with existing identity registry")
    typed_output = {}
    for key in TYPED_FIELDS:
        typed_output[key] = _fill_new_faces(source[key], output_face_to_input).astype(np.int64, copy=False)
    patch_id_output = np.full(len(faces), -1, dtype=np.int64)
    patch_operation_output = np.full(len(faces), "", dtype="<U32")
    patch_rows: dict[int, dict[str, Any]] = {}
    patch_records = []
    for op in operations:
        pid = int(op["numeric_patch_id"])
        out_rows = np.asarray(op["new_output_face_rows"], dtype=np.int64)
        retired = np.asarray(op["retired_input_face_rows"], dtype=np.int64)
        tuples = tuple_union(source, retired)
        expected = np.asarray(tuples, dtype=np.int64).reshape((-1, len(TYPED_FIELDS)))
        if expected.shape[1] != 7:
            raise ValueError("typed union tuple width differs from seven registered fields")
        for col, key in enumerate(TYPED_FIELDS):
            hinted = op.get("source_lineage", {}).get(key)
            if hinted is not None and set(map(int, np.asarray(hinted).reshape(-1))) != set(map(int, np.unique(expected[:, col]))):
                raise ValueError(f"{role} {op['patch_id']} source_lineage disagrees with retired-face {key} union")
        layers = np.unique(expected[:, 0])
        bodies = np.unique(expected[:, 1])
        if len(layers) != 1 or len(bodies) != 1:
            raise ValueError(f"{role} {op['patch_id']} mixes layer/body in one patch union")
        bones = np.unique(expected[:, 2])
        derived_bone = int(bones[0]) if len(bones) == 1 else -1
        synthetic = synthetic_ids[pid]
        union_hash = union_sha(tuples)
        lineage = op.get("source_lineage", {})
        lineage_hash = hashlib.sha256(canonical_json(lineage).encode()).hexdigest()
        for row in out_rows:
            if row < 0 or row >= len(faces) or face_out_to_in[row] != -1:
                raise ValueError(f"{role} {op['patch_id']} names a non-new output face row")
            if row in patch_rows:
                raise ValueError(f"{role} output face row belongs to more than one patch")
            patch_rows[int(row)] = {"pid": pid, "tuples": tuples, "expected": expected, "retired": retired.tolist()}
            typed_output["face_source_layer"][row] = int(layers[0])
            typed_output["face_source_body"][row] = int(bodies[0])
            typed_output["face_source_bone"][row] = derived_bone
            typed_output["face_source_face_id"][row] = -1
            typed_output["face_source_original_id"][row] = synthetic
            typed_output["face_source_identity0"][row] = int(row)
            typed_output["face_source_identity1"][row] = pid
            patch_id_output[row] = pid
            patch_operation_output[row] = str(op["patch_id"])
        source_schema[str(synthetic)] = {
            "role": "DERIVED_PATCH",
            "derived_operation": "quality_star_fixed_boundary_dp_patch_retriangulation",
            "derived_patch_id": str(op["patch_id"]),
            "derived_patch_ordinal": pid,
            "face_count": int(len(out_rows)),
            "identity_keys": ["face_source_identity0", "face_source_identity1"],
            "identity_face_id_key": "face_source_face_id",
            "parent_support_semantics": SEMANTICS,
            "lineage_method": ("current_candidate_input_face_patch_union_no_exact_overlap" if role in ("I", "D")
                               else "complete_retired_patch_source_ancestry_union_no_per_triangle_overlap"),
            "parent_face_row_domain": ("input_candidate_face_row" if role in ("I", "D")
                                       else "retired_patch_source_ancestry_csr"),
            "parent_union_tuple_count": int(len(tuples)),
            "parent_union_tuple_sha256": union_hash,
            "complete_typed_parent_union": [list(t) for t in tuples],
            "source_lineage": lineage,
            "source_lineage_sha256": lineage_hash,
            "operation_json_path": op["operation_json_path"],
            "operation_json_sha256": op["operation_json_sha256"],
        }
        patch_records.append({
            "patch_id": str(op["patch_id"]),
            "numeric_patch_id": pid,
            "synthetic_original_id": synthetic,
            "new_output_face_rows": out_rows.tolist(),
            "retired_input_face_rows": retired.tolist(),
            "complete_typed_parent_union": [list(t) for t in tuples],
            "typed_union_sha256": union_hash,
            "source_lineage": lineage,
            "source_lineage_sha256": lineage_hash,
            "operation_json_path": op["operation_json_path"],
            "operation_json_sha256": op["operation_json_sha256"],
        })
    if set(patch_rows) != set(map(int, new_face_rows)):
        raise ValueError(f"{role} patch operations do not cover every new face exactly once")
    union_offsets = [0]
    union_lanes = {key: [] for key in TYPED_FIELDS}
    for row in range(len(faces)):
        patch = patch_rows.get(row)
        if patch is None:
            union_offsets.append(union_offsets[-1])
            continue
        tuples = patch["tuples"]
        for tup in tuples:
            for key, value in zip(TYPED_FIELDS, tup):
                union_lanes[key].append(int(value))
        union_offsets.append(union_offsets[-1] + len(tuples))
    for key, values in typed_output.items():
        out[key] = values
    out["face_source_identity_schema_by_original_id_json"] = np.asarray([canonical_json(source_schema)])
    out["quality_star_patch_union_offsets"] = np.asarray(union_offsets, dtype=np.int64)
    for key, values in union_lanes.items():
        out[f"quality_star_patch_union_{key}"] = np.asarray(values, dtype=np.int64)
    out["quality_star_patch_id"] = patch_id_output
    out["quality_star_patch_operation_id"] = patch_operation_output
    out["quality_star_patch_union_semantics"] = np.asarray([SEMANTICS])
    out["quality_star_face_output_to_input_row"] = output_face_to_input
    out["quality_star_point_output_to_input_row"] = output_to_input_points
    # Existing face-owned CSR rows must remain nonempty on every replacement face.
    # Their new slices are deduplicated unions of the aligned parent tuples from all retired faces.
    for offsets_key, lanes in face_csr.items():
        rebuilt_offsets, rebuilt_values = remap_csr_with_patch_unions(
            source[offsets_key],
            {lane: source[lane] for lane in lanes},
            output_face_to_input,
            len(in_faces),
            patch_rows,
            require_nonempty_new=True,
        )
        out[offsets_key] = rebuilt_offsets
        out.update(rebuilt_values)
    # I and D did not carry the standardized face_ancestry CSR in their input assets.
    # Emit a declared current-input-face patch-union domain; this is not exact geometric overlap.
    if role in ("I", "D") and "face_ancestry_offsets" not in source:
        fa = build_current_input_face_ancestry(source, output_face_to_input, patch_rows)
        out["face_ancestry_offsets"], out["face_ancestry_source_face_rows"], out["face_ancestry_source_original_ids"], out["face_ancestry_parent_face_rows"] = fa
        if np.any(np.diff(out["face_ancestry_offsets"]) <= 0):
            raise ValueError(f"{role} standardized face ancestry CSR has an empty row")
    elif "face_ancestry_offsets" in out:
        if np.any(np.diff(out["face_ancestry_offsets"]) <= 0):
            raise ValueError(f"{role} face ancestry CSR has an empty output row")
    # Validate the produced CSR
    if len(out["quality_star_patch_union_offsets"]) != len(faces) + 1:
        raise ValueError(f"{role} patch-union CSR owner domain is not final face domain")
    if int(out["quality_star_patch_union_offsets"][-1]) != len(out["quality_star_patch_union_face_source_layer"]):
        raise ValueError(f"{role} patch-union CSR terminal does not match lanes")
    for patch in patch_records:
        expected = np.asarray(patch["complete_typed_parent_union"], dtype=np.int64)
        for face_row in patch["new_output_face_rows"]:
            a = int(out["quality_star_patch_union_offsets"][face_row])
            b = int(out["quality_star_patch_union_offsets"][face_row + 1])
            got = np.column_stack([out[f"quality_star_patch_union_{k}"][a:b] for k in TYPED_FIELDS])
            if not np.array_equal(got, expected):
                raise ValueError(f"{role} face {face_row} patch-union CSR differs from authenticated patch union")
    # Authenticate current protected routes and rebind them through the integer row map.
    route_rows = []
    for route in protected_routes:
        if route["role"] != role:
            continue
        input_row = int(route["surface_vertex_row"])
        if input_row < 0 or input_row >= len(input_to_output_points):
            raise ValueError(f"{role} protected input point outside role domain")
        output_row = int(input_to_output_points[input_row])
        if output_row < 0:
            raise ValueError(f"{role} protected point was retired or compacted away")
        target = np.asarray(route["target_world_m_f32"], dtype=np.float32)
        actual = points[output_row].astype(np.float32, copy=False)
        if target.shape != (3,) or target.tobytes() != actual.tobytes():
            raise ValueError(f"{role} protected target changed under point map")
        route_rows.append({
            "canonical_protected_row": int(route["canonical_protected_row"]),
            "mapping_kind": str(route["mapping_kind"]),
            "role": role,
            "input_surface_vertex_row": input_row,
            "output_surface_vertex_row": output_row,
            "target_world_m_f32": json.dumps(target.tolist(), separators=(",", ":")),
            "source_surface_sha256": source["_sha256"],
        })
    if role == "D" and "point_preservation_source_asset_index" in source:
        for key in ("point_preservation_source_asset_index", "point_preservation_source_vertex_row"):
            if len(out[key]) != len(points):
                raise ValueError(f"D {key} did not follow compacted output rows")
            if not np.array_equal(out[key], source[key][output_to_input_points]):
                raise ValueError(f"D {key} remap mismatch")
    out_path = output_dir / f"candidate-{role}-quality-star-typed.npz"
    maps_path = output_dir / f"row-maps-{role}.npz"
    np.savez_compressed(out_path, **out)
    np.savez_compressed(
        maps_path,
        input_point_to_output_row=input_to_output_points,
        output_point_to_input_row=output_to_input_points,
        raw_output_point_to_input_row=point_raw_to_in,
        raw_output_point_to_compact_row=raw_to_compact,
        input_face_to_output_row=input_face_to_output,
        output_face_to_input_row=output_face_to_input,
        output_face_to_raw_row=np.arange(len(faces), dtype=np.int64),
        retired_input_face_rows=np.asarray(raw["retired_input_face_rows"], dtype=np.int64),
        new_output_face_patch_ids=patch_id_output,
    )
    patch_json_path = output_dir / f"patch-unions-{role}.json"
    patch_json_path.write_text(json.dumps({
        "schema": "quality-star-typed-patch-union.v1",
        "role": role,
        "source_surface_sha256": source["_sha256"],
        "typed_fields": list(TYPED_FIELDS),
        "semantics": SEMANTICS,
        "patches": patch_records,
    }, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "role": role,
        "input_path": str(src_path),
        "input_sha256": source["_sha256"],
        "raw_proposal_path": str(raw_path),
        "raw_proposal_sha256": sha256(raw_path),
        "output_path": str(out_path),
        "output_sha256": sha256(out_path),
        "row_maps_path": str(maps_path),
        "row_maps_sha256": sha256(maps_path),
        "patch_union_path": str(patch_json_path),
        "patch_union_sha256": sha256(patch_json_path),
        "input_point_count": int(len(source["points_world_m"])),
        "output_point_count": int(len(points)),
        "retired_point_count": int(np.count_nonzero(input_to_output_points < 0)),
        "input_face_count": int(len(in_faces)),
        "output_face_count": int(len(faces)),
        "retired_face_count": int(len(raw["retired_input_face_rows"])),
        "new_face_count": int(len(new_face_rows)),
        "patch_count": int(len(operations)),
        "point_dtype": str(points.dtype),
        "protected_route_rows": route_rows,
        "patch_union_tuple_count": int(union_offsets[-1]),
    }

def write_reindexed_protected_csv(
    source_csv: Path,
    routes: list[dict[str, Any]],
    role_reports: dict[str, dict[str, Any]],
    output_csv: Path,
) -> None:
    with source_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    by_canonical = {int(x["canonical_protected_row"]): x for x in routes}
    if len(by_canonical) != 149 or set(by_canonical) != set(range(149)):
        raise ValueError("protected route receipt is not exactly canonical rows 0..148")
    if len(rows) != 149:
        raise ValueError("historical protected CSV is not 149 rows")
    additions = [
        "quality_star_input_surface_vertex_row",
        "quality_star_output_surface_vertex_row",
        "quality_star_source_surface_sha256",
        "quality_star_candidate_surface_sha256",
    ]
    for key in additions:
        if key not in header:
            header.append(key)
    reports = {r: role_reports[r] for r in role_reports}
    # Build per-role direct row maps from the already materialized map NPZs.
    map_by_role = {}
    for role, report in reports.items():
        with np.load(report["row_maps_path"], allow_pickle=False) as z:
            map_by_role[role] = (np.asarray(z["input_point_to_output_row"]).copy(), report)
    for row in rows:
        canonical = int(row["canonical_protected_row"])
        route = by_canonical[canonical]
        role = str(route["role"])
        input_row = int(route["surface_vertex_row"])
        input_to_output, rep = map_by_role[role]
        output_row = int(input_to_output[input_row])
        if output_row < 0:
            raise ValueError(f"canonical protected row {canonical} has no output surface row")
        row["quality_star_input_surface_vertex_row"] = str(input_row)
        row["quality_star_output_surface_vertex_row"] = str(output_row)
        row["quality_star_source_surface_sha256"] = str(rep["input_sha256"])
        row["quality_star_candidate_surface_sha256"] = str(rep["output_sha256"])
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

def run(args: argparse.Namespace) -> dict[str, Any]:
    base = args.base.resolve()
    batch = args.batch.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to reuse existing output directory: {out}")
    out.mkdir(parents=True)
    routes_path = base / "microedge-collar-014-saved-op-replay" / "receipt.json"
    historical_csv = base / "current-D-proof-015" / "operational-protected-149-targets.csv"
    routes_sha = sha256(routes_path)
    if routes_sha != "a7eaac55c3cacf88aa5d31c2f2844b6da957a3348c98e76951bfc96c32968a01":
        raise ValueError(f"authoritative protected-route receipt hash changed: {routes_sha}")
    receipt = json.loads(routes_path.read_text())
    routes = receipt["protected149"]["surface_routes"]
    if int(receipt["protected149"]["count"]) != 149 or len(routes) != 149:
        raise ValueError("authoritative protected-route receipt count mismatch")
    historical_csv_sha = sha256(historical_csv)
    if historical_csv_sha != "a8a3a855e3cbf31c8c4c6288e719917a10e99449c13a316e22c319c466e1d728":
        raise ValueError("historical protected target CSV hash changed")
    role_reports = {}
    for role, (raw_name, src_name) in ROLE_FILES.items():
        src_path = base / ({"O":"microedge-csr-canonical-011/O","I":"local-microedge-saved-operation-replay-011/I","D":"current-D-proof-015"}[role]) / src_name
        raw_path = batch / raw_name
        if sha256(src_path) != EXPECTED_SOURCE_SHA[role]:
            raise ValueError(f"{role} source candidate SHA does not match frozen pin")
        if sha256(raw_path) != EXPECTED_RAW_SHA[role]:
            raise ValueError(f"{role} raw proposal SHA does not match frozen 035 pin")
        role_reports[role] = build_role(role, src_path, raw_path, batch, routes, out)
    if sha256(batch / "report.json") != "7ba174b74a280126e56357a9d6123379e9668b9fff7d54303cfbd017b3d1cb74":
        raise ValueError("035 batch report SHA changed")
    if sha256(batch / "independent-verification-001/report.json") != "32dc1ef7b093615bc1f65d92de959f7d693882118a9ea1addd1a34b6a455b047":
        raise ValueError("035 independent verification SHA changed")
    role_counts = {role: sum(1 for route in routes if route["role"] == role) for role in ("O", "I", "D")}
    if role_counts != {"O": 45, "I": 45, "D": 59}:
        raise ValueError(f"protected route counts changed: {role_counts}")
    protected_csv = out / "protected149-reindexed.csv"
    write_reindexed_protected_csv(historical_csv, routes, role_reports, protected_csv)
    report = {
        "schema": "quality-star-dp-typed-metadata-bundle.v1",
        "status": "PASS_METADATA_REINDEX_ONLY",
        "geometry_or_physics_admission": False,
        "tetgen_run": False,
        "native_runtime_run": False,
        "source_base": str(base),
        "batch_dir": str(batch),
        "batch_report_sha256": sha256(batch / "report.json"),
        "independent_verification_sha256": sha256(batch / "independent-verification-001/report.json"),
        "protected_route_receipt_path": str(routes_path),
        "protected_route_receipt_sha256": routes_sha,
        "historical_protected_csv_path": str(historical_csv),
        "historical_protected_csv_sha256": historical_csv_sha,
        "reindexed_protected_csv_path": str(protected_csv),
        "reindexed_protected_csv_sha256": sha256(protected_csv),
        "roles": role_reports,
        "limitations": [
            "This step only compacts and carries typed source metadata for frozen 035 geometry proposals.",
            "It does not rerun or replace exact geometry, shell, nesting, volume, mass, or native gates.",
            "Patch union is patch-level support and makes no per-triangle overlap claim.",
            "Mixed-source patch identities are synthetic derived IDs; the complete typed union CSR is retained separately.",
        ],
    }
    report_path = out / "metadata-receipt.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    report["receipt_path"] = str(report_path)
    report["receipt_sha256"] = sha256(report_path)
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, sort_keys=True))

if __name__ == "__main__":
    main()

