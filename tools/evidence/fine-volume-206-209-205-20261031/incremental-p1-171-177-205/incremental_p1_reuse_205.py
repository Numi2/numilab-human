#!/usr/bin/env python3
"""Incremental source-only P1 row reuse for fixed-coordinate child retetrahedralization.

Unchanged rows are reusable only under exact ordered-corner identity through an
explicit child-node-to-base-node map. Every unmatched output cell must have a
complete, separately receipted source-217 row bundle. This tool does not infer
new-cell ancestry, change coordinates, allocate mass, or qualify a physical mesh.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

P1_SCHEMA = "numi.native-domain-217.p1-supported-reference-anchor-csr.v1"
BASE_RECEIPT_SCHEMA = "numi.native-domain-217.p1-supported-reference-anchor.v1"
FRESH_RECEIPT_SCHEMA = "numi.native-domain-217.p1-reference-anchor.changed-row-subset.v1"
INCREMENTAL_RECEIPT_SCHEMA = "numi.native-domain-217.p1-reference-anchor.incremental-reuse.v1"
REGISTRATION_SCHEMA = "numi.current-typed-217.registered-node-map-143-713.v1"
INCIDENCE_SCHEMA = "numi.217.current-child-p1-semantic-incidence-union.v1"

# Each CSR group uses one cell-row offset vector. Several aligned payload arrays
# can share the same offsets; this is how the authenticated 217 archive is laid out.
CSR_GROUPS: dict[str, tuple[str, ...]] = {
    "p1_parent_tet_offsets": ("p1_parent_tet_rows",),
    "p1_primary_parent_tet_offsets": ("p1_primary_parent_tet_rows",),
    "reference_anchor_carrier_parent_tet_offsets": ("reference_anchor_carrier_parent_tet_rows",),
    "p1_semantic_tuple_offsets": ("p1_semantic_tuples",),
    "reference_anchor_tie_offsets": (
        "reference_anchor_tie_parent_tet_rows",
        "reference_anchor_tie_carrier_child_node_rows",
        "reference_anchor_tie_barycentric",
        "reference_anchor_tie_distance_m",
    ),
    "reference_anchor_tie_semantic_tuple_offsets": ("reference_anchor_tie_semantic_tuples",),
    "reference_anchor_near_candidate_offsets": (
        "reference_anchor_near_candidate_parent_tet_rows",
        "reference_anchor_near_candidate_child_node_rows",
        "reference_anchor_near_candidate_approx_distance_m",
        "reference_anchor_near_candidate_exact_distance_gap_m",
    ),
}
CSR_OFFSET_KEYS = set(CSR_GROUPS)
CSR_DATA_KEYS = {k for group in CSR_GROUPS.values() for k in group}
CELL_KEYS = (
    "p1_support_role_material", "p1_support_cell_row", "p1_primary_group_id",
    "reference_anchor_carrier_node_group_id", "p1_primary_group_score_numerator",
    "p1_primary_group_score_denominator", "p1_primary_group_tuple",
    "reference_anchor_near_candidate_projection_separation_m",
    "reference_anchor_exact_pool_resolution_used",
    "reference_anchor_exact_min_squared_distance_ratio",
    "reference_anchor_selected_parent_tet_row",
    "reference_anchor_selected_parent_barycentric",
    "reference_anchor_selected_semantic_tuple",
    "centroid_diagnostic_parent_tet_row", "centroid_diagnostic_barycentric",
    "centroid_diagnostic_registered_world_m", "centroid_diagnostic_residual_norm_m",
    "reference_anchor_carrier_child_node_row",
    "reference_anchor_carrier_query_world_m",
    "reference_anchor_centroid_to_projected_anchor_distance_m",
    "reference_anchor_registered_world_m", "reference_anchor_distance_m",
    "reference_anchor_tie_max_projection_separation_m",
    "reference_anchor_tie_max_full86_abs_difference",
)

LABEL_HEADER = ("child_tet_row", "material_index", "geometric_donor_body",
                "anatomical_region_id", "stable_source_face_id")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _data_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            yield stripped.split()


def read_tetgen_node(path: Path) -> tuple[np.ndarray, dict[int, int]]:
    it = iter(_data_lines(path))
    try:
        header = next(it)
    except StopIteration as exc:
        raise ValueError(f"empty TetGen node file: {path}") from exc
    if len(header) < 4:
        raise ValueError(f"invalid TetGen node header in {path}")
    count, dim, attributes, markers = map(int, header[:4])
    if count < 0 or dim != 3 or attributes < 0 or markers not in (0, 1):
        raise ValueError(f"unsupported TetGen node header in {path}: {header}")
    points = np.empty((count, 3), dtype=np.float64)
    node_id_to_row: dict[int, int] = {}
    for row_index in range(count):
        row = next(it, None)
        if row is None or len(row) < 4 + attributes + markers:
            raise ValueError(f"short TetGen node row in {path}")
        node_id = int(row[0])
        if node_id in node_id_to_row:
            raise ValueError(f"duplicate TetGen node ID {node_id} in {path}")
        node_id_to_row[node_id] = row_index
        points[row_index] = [float(v) for v in row[1:4]]
    if len(node_id_to_row) != count or next(it, None) is not None:
        raise ValueError(f"TetGen node row count/ID coverage differs from header in {path}")
    return points.astype(np.float32), node_id_to_row


def read_tetgen_ele(path: Path, node_id_to_row: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    it = iter(_data_lines(path))
    try:
        header = next(it)
    except StopIteration as exc:
        raise ValueError(f"empty TetGen element file: {path}") from exc
    if len(header) < 3:
        raise ValueError(f"invalid TetGen element header in {path}")
    count, corners, attributes = map(int, header[:3])
    if count < 0 or corners != 4 or attributes < 1:
        raise ValueError(f"TetGen elements require four corners and a material attribute: {header}")
    tets = np.empty((count, 4), dtype=np.int64)
    materials = np.empty(count, dtype=np.int64)
    seen: set[int] = set()
    expected = 1 + corners + attributes
    for row_index in range(count):
        row = next(it, None)
        if row is None or len(row) < expected:
            raise ValueError(f"short TetGen element row in {path}")
        elem_id = int(row[0])
        if elem_id in seen:
            raise ValueError(f"duplicate TetGen element ID {elem_id} in {path}")
        seen.add(elem_id)
        node_ids = [int(v) for v in row[1:5]]
        if any(n not in node_id_to_row for n in node_ids):
            raise ValueError(f"TetGen element {elem_id} references a missing node ID")
        tets[row_index] = [node_id_to_row[n] for n in node_ids]
        materials[row_index] = int(row[5])
    if len(seen) != count or next(it, None) is not None:
        raise ValueError(f"TetGen element row count/ID coverage differs from header in {path}")
    return tets, materials


def exact_f32_equal(a: np.ndarray, b: np.ndarray) -> bool:
    aa = np.ascontiguousarray(np.asarray(a, dtype=np.float32)).view(np.uint32)
    bb = np.ascontiguousarray(np.asarray(b, dtype=np.float32)).view(np.uint32)
    return aa.shape == bb.shape and np.array_equal(aa, bb)


def validate_node_map(base_points: np.ndarray, new_points: np.ndarray,
                      new_to_base: np.ndarray) -> tuple[int, bool]:
    mapping = np.asarray(new_to_base, dtype=np.int64)
    if mapping.shape != (len(new_points),):
        raise ValueError("new-to-base node map must have one entry per candidate point")
    if np.any(mapping < -1) or np.any(mapping >= len(base_points)):
        raise ValueError("new-to-base node map contains an out-of-range base row")
    mapped = mapping[mapping >= 0]
    if len(np.unique(mapped)) != len(mapped):
        raise ValueError("new-to-base node map is not one-to-one")
    if len(mapped):
        if not exact_f32_equal(new_points[mapping >= 0], base_points[mapped]):
            raise ValueError("mapped candidate node coordinates differ from base F32 bytes")
    full_bijection = (len(new_points) == len(base_points) and len(mapped) == len(base_points)
                      and len(np.unique(mapped)) == len(base_points))
    return len(mapped), full_bijection


def build_ordered_tet_row_map(base_tets: np.ndarray, base_materials: np.ndarray,
                              candidate_tets: np.ndarray, candidate_materials: np.ndarray,
                              candidate_node_to_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map exact unchanged cells only; corner permutations are deliberately changed rows."""
    old_tets = np.asarray(base_tets, dtype=np.int64)
    old_materials = np.asarray(base_materials, dtype=np.int64)
    new_tets = np.asarray(candidate_tets, dtype=np.int64)
    new_materials = np.asarray(candidate_materials, dtype=np.int64)
    node_map = np.asarray(candidate_node_to_base, dtype=np.int64)
    if old_tets.ndim != 2 or old_tets.shape[1] != 4 or new_tets.ndim != 2 or new_tets.shape[1] != 4:
        raise ValueError("cell arrays must be Nx4 tetrahedra")
    if old_materials.shape != (len(old_tets),) or new_materials.shape != (len(new_tets),):
        raise ValueError("material array shape differs from tetrahedra")
    if np.any(old_tets < 0) or np.any(new_tets < 0) or np.any(new_tets >= len(node_map)):
        raise ValueError("tetrahedron contains invalid node rows")
    lookup: dict[tuple[int, int, int, int, int], int] = {}
    for row, (mat, corners) in enumerate(zip(old_materials, old_tets)):
        key = (int(mat), *map(int, corners))
        if key in lookup:
            raise ValueError(f"base mesh has duplicate ordered cell identity at rows {lookup[key]} and {row}")
        lookup[key] = row
    new_seen: set[tuple[int, int, int, int, int]] = set()
    new_to_old = np.full(len(new_tets), -1, dtype=np.int64)
    old_to_new = np.full(len(old_tets), -1, dtype=np.int64)
    for row, (mat, corners) in enumerate(zip(new_materials, new_tets)):
        key_nodes = node_map[corners]
        key = (int(mat), *map(int, key_nodes))
        if key in new_seen:
            raise ValueError(f"candidate mesh has duplicate ordered cell identity at row {row}")
        new_seen.add(key)
        if np.all(key_nodes >= 0):
            old = lookup.get(key)
            if old is not None:
                if old_to_new[old] >= 0:
                    raise ValueError(f"base cell {old} maps to multiple candidate cells")
                new_to_old[row] = old
                old_to_new[old] = row
    return new_to_old, old_to_new


def _csv_offsets(path: Path, expected_header: tuple[str, ...] | None = None,
                 expected_ids: set[int] | None = None,
                 require_contiguous: int | None = None,
                 row_key_field: str = "child_tet_row",
                 row_key_offset: int = 0) -> tuple[bytes, list[str], dict[int, int]]:
    offsets: dict[int, int] = {}
    with path.open("rb") as f:
        header_line = f.readline()
        if not header_line or not header_line.endswith(b"\n"):
            raise ValueError(f"CSV header missing/unterminated: {path}")
        header = next(csv.reader([header_line.decode("utf-8").rstrip("\r\n")]))
        if row_key_field not in header:
            raise ValueError(f"CSV lacks row key {row_key_field}: {path}")
        row_key_column = header.index(row_key_field)
        if expected_header is not None and tuple(header) != expected_header:
            raise ValueError(f"CSV header differs from authenticated schema: {path}")
        while True:
            offset = f.tell()
            raw = f.readline()
            if not raw:
                break
            if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
                raise ValueError(f"CSV data rows must occupy one physical line: {path}")
            values = next(csv.reader([raw.decode("utf-8").rstrip("\r\n")]))
            if len(values) != len(header):
                raise ValueError(f"CSV row width differs from header in {path}")
            row_id = int(values[row_key_column]) - row_key_offset
            if row_id in offsets:
                raise ValueError(f"duplicate child_tet_row {row_id} in {path}")
            offsets[row_id] = offset
    if require_contiguous is not None and set(offsets) != set(range(require_contiguous)):
        raise ValueError(f"CSV row IDs are not the full contiguous domain 0..{require_contiguous-1}: {path}")
    if expected_ids is not None and set(offsets) != expected_ids:
        missing = sorted(expected_ids - set(offsets))[:8]
        extra = sorted(set(offsets) - expected_ids)[:8]
        raise ValueError(f"CSV changed-row coverage mismatch in {path}; missing={missing}, extra={extra}")
    return header_line, header, offsets


def _read_csv_record_at(f, offset: int, expected_width: int) -> list[str]:
    f.seek(offset)
    raw = f.readline()
    if not raw:
        raise ValueError("indexed CSV row offset points past end of file")
    row = next(csv.reader([raw.decode("utf-8").rstrip("\r\n")]))
    if len(row) != expected_width:
        raise ValueError("indexed CSV row width changed after indexing")
    return row


def merge_csv(base_path: Path, fresh_path: Path, output_path: Path,
              old_for_new: np.ndarray, fresh_row_index: dict[int, int],
              *, base_count: int,
              expected_header: tuple[str, ...] | None = None) -> dict[str, Any]:
    changed_set = set(map(int, np.flatnonzero(np.asarray(old_for_new) < 0)))
    _, header, base_offsets = _csv_offsets(base_path, expected_header,
                                             require_contiguous=base_count)
    _, fresh_header, fresh_offsets = _csv_offsets(fresh_path, expected_header or tuple(header), changed_set)
    if fresh_header != header:
        raise ValueError(f"fresh CSV header differs from base header: {fresh_path}")
    if set(fresh_offsets) != set(fresh_row_index):
        raise ValueError("fresh CSV and fresh P1 row map domains differ")
    row_column = header.index("child_tet_row")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite incremental output: {output_path}")
    with base_path.open("rb") as base_f, fresh_path.open("rb") as fresh_f, \
            output_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, lineterminator="\n")
        writer.writerow(header)
        for new_row, old_row in enumerate(np.asarray(old_for_new, dtype=np.int64)):
            if old_row >= 0:
                values = _read_csv_record_at(base_f, base_offsets[int(old_row)], len(header))
                if int(values[row_column]) != int(old_row):
                    raise ValueError(f"base CSV index mismatch at row {old_row}")
            else:
                if new_row not in fresh_offsets:
                    raise ValueError(f"changed row {new_row} has no fresh source record")
                values = _read_csv_record_at(fresh_f, fresh_offsets[new_row], len(header))
                if int(values[row_column]) != int(new_row):
                    raise ValueError(f"fresh CSV index mismatch at candidate row {new_row}")
            values[row_column] = str(new_row)
            writer.writerow(values)
    return {"path": str(output_path), "sha256": sha256_file(output_path),
            "row_count": len(old_for_new), "header": header}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: np.array(z[key], copy=True) for key in z.files}


def _csr_lengths(offsets: np.ndarray, row_indices: np.ndarray, n_rows: int, name: str) -> np.ndarray:
    off = np.asarray(offsets, dtype=np.int64)
    rows = np.asarray(row_indices, dtype=np.int64)
    if off.shape != (n_rows + 1,) or int(off[0]) != 0 or int(off[-1]) != len(rows) or np.any(off[1:] < off[:-1]):
        raise ValueError(f"malformed {name} CSR offsets")
    return np.diff(off)


def merge_anchor_archives(base: dict[str, np.ndarray], fresh: dict[str, np.ndarray],
                          old_for_new: np.ndarray,
                          fresh_row_index: dict[int, int]) -> dict[str, np.ndarray]:
    if str(base.get("schema", np.asarray("")).item()) != P1_SCHEMA or str(fresh.get("schema", np.asarray("")).item()) != P1_SCHEMA:
        raise ValueError("base/fresh P1 archives must use the authenticated 217 CSR schema")
    old_rows = np.asarray(base.get("child_tet_row"), dtype=np.int64)
    fresh_rows = np.asarray(fresh.get("child_tet_row"), dtype=np.int64)
    if old_rows.ndim != 1 or not np.array_equal(old_rows, np.arange(len(old_rows))):
        raise ValueError("base P1 CSR child_tet_row is not a full contiguous domain")
    if fresh_rows.ndim != 1 or len(np.unique(fresh_rows)) != len(fresh_rows):
        raise ValueError("fresh P1 CSR child_tet_row is malformed or duplicated")
    if set(map(int, fresh_rows)) != set(fresh_row_index):
        raise ValueError("fresh P1 CSR and receipt row maps differ")
    if len(base) != len(fresh) or set(base) != set(fresh):
        raise ValueError("base and fresh P1 archives have different key sets")
    new_n = len(old_for_new)
    if len(fresh_rows) != int(np.count_nonzero(np.asarray(old_for_new) < 0)):
        raise ValueError("fresh P1 row count does not equal candidate changed-row count")
    csr_owned = CSR_OFFSET_KEYS | CSR_DATA_KEYS
    expected_cell = set(CELL_KEYS) | {"child_tet_row"}
    actual_cell = set(base) - csr_owned - {"schema"}
    if actual_cell != expected_cell:
        raise ValueError(f"P1 cell-row schema differs from enumerated 217 fields: missing={sorted(expected_cell-actual_cell)}, extra={sorted(actual_cell-expected_cell)}")
    for offset_key, data_keys in CSR_GROUPS.items():
        if offset_key not in base or any(k not in base for k in data_keys):
            raise ValueError(f"P1 archive lacks CSR group {offset_key}")
    base_n = len(old_rows)
    if np.any(old_for_new >= base_n):
        raise ValueError("candidate-to-base tet map contains out-of-range base rows")
    fresh_lookup = {int(row): i for i, row in enumerate(fresh_rows)}
    if len(fresh_lookup) != len(fresh_rows) or set(fresh_lookup) != set(fresh_row_index):
        raise ValueError("fresh P1 row lookup is not one-to-one")
    fresh_idx = np.full(new_n, -1, dtype=np.int64)
    for row, idx in fresh_row_index.items():
        if row < 0 or row >= new_n or idx != fresh_lookup.get(int(row)):
            raise ValueError("fresh row map does not agree with P1 child_tet_row")
        fresh_idx[int(row)] = int(idx)
    old_for_new = np.asarray(old_for_new, dtype=np.int64)
    old_mask = old_for_new >= 0
    if np.any(~old_mask & (fresh_idx < 0)) or np.any(old_mask & (fresh_idx >= 0)):
        raise ValueError("candidate rows are not partitioned into retained and fresh domains")
    out: dict[str, np.ndarray] = {"schema": np.asarray(P1_SCHEMA)}
    out["child_tet_row"] = np.arange(new_n, dtype=np.int64)
    for key in CELL_KEYS:
        a, b = np.asarray(base[key]), np.asarray(fresh[key])
        if a.shape[0] != base_n or b.shape[0] != len(fresh_rows) or a.dtype != b.dtype or a.shape[1:] != b.shape[1:]:
            raise ValueError(f"cell array shape/dtype mismatch for {key}")
        merged = np.empty((new_n,) + a.shape[1:], dtype=a.dtype)
        merged[old_mask] = a[old_for_new[old_mask]]
        merged[~old_mask] = b[fresh_idx[~old_mask]]
        out[key] = merged
    for offset_key, data_keys in CSR_GROUPS.items():
        oldoff = np.asarray(base[offset_key], dtype=np.int64)
        newoff = np.asarray(fresh[offset_key], dtype=np.int64)
        if oldoff.shape != (base_n + 1,) or newoff.shape != (len(fresh_rows) + 1,):
            raise ValueError(f"row offset shape mismatch for {offset_key}")
        if int(oldoff[0]) != 0 or int(newoff[0]) != 0 or np.any(oldoff[1:] < oldoff[:-1]) or np.any(newoff[1:] < newoff[:-1]):
            raise ValueError(f"nonmonotone CSR offsets for {offset_key}")
        for data_key in data_keys:
            a, b = np.asarray(base[data_key]), np.asarray(fresh[data_key])
            if int(oldoff[-1]) != len(a) or int(newoff[-1]) != len(b):
                raise ValueError(f"CSR terminal offset mismatch for {data_key}")
            if a.dtype != b.dtype or a.shape[1:] != b.shape[1:]:
                raise ValueError(f"CSR payload dtype/shape mismatch for {data_key}")
        old_lengths = np.diff(oldoff)[np.maximum(old_for_new, 0)]
        fresh_lengths = np.zeros(new_n, dtype=np.int64)
        fresh_rows_idx = np.flatnonzero(~old_mask)
        fresh_lengths[fresh_rows_idx] = np.diff(newoff)[fresh_idx[fresh_rows_idx]]
        lengths = np.where(old_mask, old_lengths, fresh_lengths)
        outoff = np.empty(new_n + 1, dtype=np.int64)
        outoff[0] = 0
        np.cumsum(lengths, out=outoff[1:])
        out[offset_key] = outoff
        total = int(outoff[-1])
        out_rows = np.repeat(np.arange(new_n, dtype=np.int64), lengths)
        relative = np.arange(total, dtype=np.int64) - np.repeat(outoff[:-1], lengths)
        old_dest = old_mask[out_rows]
        if np.any(old_dest):
            src_rows = old_for_new[out_rows[old_dest]]
            src_ix = oldoff[src_rows] + relative[old_dest]
            for data_key in data_keys:
                a = np.asarray(base[data_key])
                merged_data = np.empty((total,) + a.shape[1:], dtype=a.dtype)
                merged_data[:0] = a[:0]
                merged_data[old_dest] = a[src_ix]
                out[data_key] = merged_data
        else:
            for data_key in data_keys:
                a = np.asarray(base[data_key])
                out[data_key] = np.empty((total,) + a.shape[1:], dtype=a.dtype)
        fresh_dest = ~old_dest
        if np.any(fresh_dest):
            src_rows = fresh_idx[out_rows[fresh_dest]]
            src_ix = newoff[src_rows] + relative[fresh_dest]
            for data_key in data_keys:
                out[data_key][fresh_dest] = np.asarray(fresh[data_key])[src_ix]
    return out


def validate_changed_semantics(labels: dict[int, list[str]], projection: dict[int, list[str]],
                               fresh: dict[str, np.ndarray], fresh_row_index: dict[int, int],
                               new_tets: np.ndarray, candidate_materials: np.ndarray) -> dict[str, Any]:
    rows = np.asarray(fresh["child_tet_row"], dtype=np.int64)
    if set(map(int, rows)) != set(labels) or set(labels) != set(projection):
        raise ValueError("changed-row labels, projection and CSR domains do not agree")
    checked = 0
    total_semantic_tuples = 0
    for candidate_row in sorted(labels):
        i = fresh_row_index[candidate_row]
        label = tuple(map(int, labels[candidate_row][1:5]))
        p = projection[candidate_row]
        if int(labels[candidate_row][0]) != candidate_row or int(projection[candidate_row][0]) != candidate_row:
            raise ValueError("changed row identifier mismatch")
        mat, donor, region, face = label
        if mat != int(candidate_materials[candidate_row]):
            raise ValueError(f"changed row {candidate_row} scalar material differs from its candidate tet")
        if int(fresh["p1_support_role_material"][i]) != mat:
            raise ValueError(f"changed row {candidate_row} P1 support material differs from scalar label")
        anchor = tuple(map(int, fresh["reference_anchor_selected_semantic_tuple"][i]))
        if anchor != label:
            raise ValueError(f"changed row {candidate_row} scalar label differs from selected P1 anchor tuple")
        primary = tuple(map(int, fresh["p1_primary_group_tuple"][i]))
        if primary != (mat, donor, region):
            raise ValueError(f"changed row {candidate_row} scalar donor/region differs from max-share P1 group")
        primary_gid = int(fresh["p1_primary_group_id"][i])
        if int(fresh["reference_anchor_carrier_node_group_id"][i]) != primary_gid:
            raise ValueError(f"changed row {candidate_row} carrier node group differs from primary group")
        carrier_node = int(fresh["reference_anchor_carrier_child_node_row"][i])
        if carrier_node not in set(map(int, new_tets[candidate_row])):
            raise ValueError(f"changed row {candidate_row} carrier node is not one of its corners")
        selected_parent = int(fresh["reference_anchor_selected_parent_tet_row"][i])
        def slice_set(offset_key: str, data_key: str) -> set[int]:
            offsets = fresh[offset_key]
            return set(map(int, fresh[data_key][int(offsets[i]):int(offsets[i + 1])]))
        all_parents = slice_set("p1_parent_tet_offsets", "p1_parent_tet_rows")
        primary_parents = slice_set("p1_primary_parent_tet_offsets", "p1_primary_parent_tet_rows")
        carrier_parents = slice_set("reference_anchor_carrier_parent_tet_offsets", "reference_anchor_carrier_parent_tet_rows")
        if not all_parents or not primary_parents or not carrier_parents:
            raise ValueError(f"changed row {candidate_row} has empty P1 parent support")
        if not primary_parents.issubset(all_parents) or not carrier_parents.issubset(primary_parents):
            raise ValueError(f"changed row {candidate_row} P1 parent pools are not nested")
        if selected_parent not in carrier_parents:
            raise ValueError(f"changed row {candidate_row} selected anchor parent is outside carrier pool")
        tuple_offsets = fresh["p1_semantic_tuple_offsets"]
        tuple_rows = fresh["p1_semantic_tuples"][int(tuple_offsets[i]):int(tuple_offsets[i + 1])]
        semantic_union = {tuple(map(int, t)) for t in tuple_rows}
        if not semantic_union or any(t[0] != mat for t in semantic_union):
            raise ValueError(f"changed row {candidate_row} has empty or cross-material P1 semantic union")
        if label not in semantic_union:
            raise ValueError(f"changed row {candidate_row} scalar anchor is absent from complete P1 semantic union")
        if int(p[5]) != selected_parent or int(p[1]) != carrier_node:
            raise ValueError(f"changed row {candidate_row} projection CSV disagrees with selected parent/carrier CSR")
        if tuple(map(int, p[17:21])) != label:
            raise ValueError(f"changed row {candidate_row} projection scalar tuple differs from labels")
        total_semantic_tuples += len(semantic_union)
        checked += 1
    return {"changed_rows_validated": checked,
            "changed_row_semantic_tuple_count": total_semantic_tuples,
            "changed_row_anchor_union_status": "PASS"}


def load_csv_rows_by_id(path: Path, expected_header: tuple[str, ...] | None = None,
                        expected_ids: set[int] | None = None,
                        require_contiguous: int | None = None) -> tuple[list[str], dict[int, list[str]]]:
    _, header, offsets = _csv_offsets(path, expected_header, expected_ids, require_contiguous)
    rows: dict[int, list[str]] = {}
    with path.open("rb") as f:
        for row_id, offset in offsets.items():
            rows[row_id] = _read_csv_record_at(f, offset, len(header))
    return header, rows


def _verify_artifact_pin(receipt: dict[str, Any], key: str, path: Path) -> None:
    item = receipt.get("outputs", {}).get(key)
    if not isinstance(item, dict) or item.get("sha256") != sha256_file(path):
        raise ValueError(f"receipt output pin mismatch for {key}")


def verify_base_receipts(labels_path: Path, source_path: Path, projection_path: Path,
                         anchor_path: Path, base_node_path: Path, base_ele_path: Path,
                         base_export_receipt_path: Path,
                         registration_map_path: Path | None,
                         registration_receipt_path: Path | None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    receipt = json.loads(base_export_receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != BASE_RECEIPT_SCHEMA or receipt.get("status") != "P1_SUPPORTED_REFERENCE_ANCHOR_LABELS_EXPORTED_NOT_PHYSICAL_ADMISSION":
        raise ValueError("base 171 receipt is not the authenticated full P1 source-only export")
    if receipt.get("full_child_domain_export") is not True or receipt.get("physical_admission") is not False:
        raise ValueError("base 171 receipt is not full-domain source-only evidence")
    if receipt.get("mass_transport") != "NOT_RUN" or receipt.get("native_steps") != 0:
        raise ValueError("base 171 receipt unexpectedly claims mass transport or native execution")
    for key, path in (("labels", labels_path), ("parent_row_map", source_path),
                      ("p1_anchor_projection_csv", projection_path), ("p1_anchor_csr", anchor_path)):
        _verify_artifact_pin(receipt, key, path)
    if receipt.get("inputs", {}).get("child_node", {}).get("sha256") != sha256_file(base_node_path):
        raise ValueError("base 171 receipt does not pin its node input")
    if receipt.get("inputs", {}).get("child_elements", {}).get("sha256") != sha256_file(base_ele_path):
        raise ValueError("base 171 receipt does not pin its element input")
    if registration_map_path is None and registration_receipt_path is None:
        return receipt, None
    if registration_map_path is None or registration_receipt_path is None:
        raise ValueError("registered-node-map and its receipt must be provided together")
    reg = json.loads(registration_receipt_path.read_text(encoding="utf-8"))
    if reg.get("schema") != REGISTRATION_SCHEMA or reg.get("status") != "REGISTERED_NODE_MAP_PREPARED_SOURCE_ONLY_NOT_PHYSICAL_ADMISSION":
        raise ValueError("base 177 receipt is not the expected source-only node registration map")
    if reg.get("output", {}).get("sha256") != sha256_file(registration_map_path):
        raise ValueError("base 177 receipt does not pin the registered-node-map CSV")
    union = reg.get("authenticated_p1_semantic_incidence_union", {})
    if union.get("schema") != INCIDENCE_SCHEMA or union.get("status") != "PASS_AUTHENTICATED_COMPLETE_P1_UNION":
        raise ValueError("base 177 receipt lacks a passing full-domain P1 incidence-union check")
    base_rows = int(receipt.get("counts", {}).get("exported_child_rows", -1))
    if (union.get("child_cells") != base_rows
            or union.get("selected_anchor_rows_present_in_union") != base_rows
            or union.get("selected_scalar_anchor_rows_exact") != base_rows
            or union.get("cross_material_union_rows") != 0
            or union.get("empty_cell_unions") != 0
            or union.get("source_scalar_labels_unchanged") is not True
            or union.get("geometry_changed") is not False
            or union.get("physical_admission") is not False
            or union.get("mass_mapping") != "NOT_RUN"
            or union.get("runtime") != 0):
        raise ValueError("base 177 P1 incidence receipt is incomplete or claims unsupported work")
    if union.get("p1_csr_sha256") != sha256_file(anchor_path):
        raise ValueError("base 177 incidence check is not bound to the supplied 171 CSR")
    if reg.get("inputs", {}).get("child_labels", {}).get("sha256") != sha256_file(labels_path):
        raise ValueError("base 177 receipt is not bound to supplied 171 labels")
    if reg.get("inputs", {}).get("tetgen_node", {}).get("sha256") != sha256_file(base_node_path):
        raise ValueError("base 177 receipt is not bound to supplied base child node file")
    if reg.get("inputs", {}).get("tetgen_ele", {}).get("sha256") != sha256_file(base_ele_path):
        raise ValueError("base 177 receipt is not bound to supplied base child element file")
    return receipt, reg


def rebase_registered_node_map(base_map_path: Path, output_path: Path,
                               new_to_base: np.ndarray, node_bijection: bool,
                               new_points: np.ndarray) -> dict[str, Any]:
    if not node_bijection:
        return {"status": "REQUIRES_NEW_NODE_REGISTRATION", "output": None,
                "reason": "candidate node domain is not a full one-to-one exact-F32 reindex of the registered base nodes"}
    mapping = np.asarray(new_to_base, dtype=np.int64)
    base_to_new = np.full(len(mapping), -1, dtype=np.int64)
    base_to_new[mapping] = np.arange(len(mapping), dtype=np.int64)
    with base_map_path.open("rb") as f:
        header_line = f.readline()
    if not header_line or not header_line.endswith(b"\n"):
        raise ValueError("177 registered node map has an invalid header")
    header = next(csv.reader([header_line.decode("utf-8").rstrip("\r\n")]))
    if "joint_node_id" not in header:
        raise ValueError("177 registered node map lacks joint_node_id")
    col = header.index("joint_node_id")
    coord_cols = [header.index(f"current_{axis}_m") for axis in "xyz"]
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite rebased node map: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    offsets_by_node: list[list[int]] = [[] for _ in range(len(mapping))]
    with base_map_path.open("rb") as base_f:
        if base_f.readline() != header_line:
            raise ValueError("177 header changed between validation and row scan")
        for row_no, raw in enumerate(base_f, start=2):
            if not raw.endswith(b"\n"):
                raise ValueError("177 map data row is unterminated")
            row = next(csv.reader([raw.decode("utf-8").rstrip("\r\n")]))
            if len(row) != len(header):
                raise ValueError("177 map row width differs from header")
            old_row = int(row[col]) - 1
            if old_row < 0 or old_row >= len(mapping):
                raise ValueError(f"177 row {row_no} references out-of-range base node {old_row + 1}")
            offsets_by_node[old_row].append(base_f.tell() - len(raw))
    missing_base = [i for i, offsets in enumerate(offsets_by_node) if not offsets]
    if missing_base:
        raise ValueError(f"177 map does not cover every base node; missing rows={missing_base[:12]}")
    output_rows = 0
    with base_map_path.open("rb") as base_f, output_path.open("w", encoding="utf-8", newline="") as out_f:
        base_f.readline()
        writer = csv.writer(out_f, lineterminator="\n")
        writer.writerow(header)
        for new_row, old_row_value in enumerate(mapping):
            old_row = int(old_row_value)
            for offset in offsets_by_node[old_row]:
                base_f.seek(offset)
                raw = base_f.readline()
                row = next(csv.reader([raw.decode("utf-8").rstrip("\r\n")]))
                if int(row[col]) != old_row + 1:
                    raise ValueError(f"177 indexed source row differs from base node {old_row + 1}")
                # Coordinates recorded by the sidecar must still match exact F32
                # candidate node bytes; this prevents reusing registration on moved nodes.
                recorded = np.asarray([float(row[k]) for k in coord_cols], dtype=np.float32)
                if not exact_f32_equal(recorded, new_points[new_row]):
                    raise ValueError(f"registered node coordinate differs at candidate row {new_row}")
                row[col] = str(new_row + 1)
                writer.writerow(row)
                output_rows += 1
    return {"status": "PASS_REINDEXED_BASE_NODE_REGISTRATION", "output": {
        "path": str(output_path), "sha256": sha256_file(output_path),
        "rows": output_rows, "unique_nodes": len(new_to_base)}}



def _load_material_node_group_support(path: Path, expected_material: int) -> dict[str, np.ndarray]:
    """Load and validate one material's authenticated 171 node-group CSR."""
    with np.load(path, allow_pickle=False) as z:
        required = ("node_group_offsets", "node_group_ids", "group_tuples",
                    "node_group_parent_tet_offsets", "node_group_parent_tet_rows")
        if any(key not in z.files for key in required):
            raise ValueError(f"171 material-{expected_material} support lacks node-group CSR fields: {path}")
        arrays = {key: np.asarray(z[key]).copy() for key in required}
    offsets = arrays["node_group_offsets"].astype(np.int64, copy=False)
    group_ids = arrays["node_group_ids"].astype(np.int64, copy=False)
    tuples = arrays["group_tuples"]
    parent_offsets = arrays["node_group_parent_tet_offsets"].astype(np.int64, copy=False)
    parent_rows = arrays["node_group_parent_tet_rows"]
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != len(group_ids):
        raise ValueError(f"171 material-{expected_material} node-group offsets are malformed")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"171 material-{expected_material} node-group offsets are not monotone")
    if tuples.ndim != 2 or tuples.shape[1] != 3:
        raise ValueError(f"171 material-{expected_material} group tuple table must be Nx3")
    if parent_offsets.shape != (len(group_ids) + 1,) or parent_offsets[0] != 0 or parent_offsets[-1] != len(parent_rows):
        raise ValueError(f"171 material-{expected_material} node-group parent CSR is malformed")
    if np.any(parent_offsets[1:] < parent_offsets[:-1]):
        raise ValueError(f"171 material-{expected_material} node-group parent offsets are not monotone")
    if np.any(group_ids < 0) or np.any(group_ids >= len(tuples)):
        raise ValueError(f"171 material-{expected_material} node-group CSR has an invalid group id")
    if len(tuples) == 0 or np.any(tuples[:, 0] != expected_material):
        raise ValueError(f"171 role support contains a group outside material {expected_material}")
    if len(group_ids) == 0 or np.any(offsets[1:] == offsets[:-1]):
        raise ValueError(f"171 material-{expected_material} support has a node with no P1 group")
    if len(parent_rows) == 0 or np.any(parent_offsets[1:] == parent_offsets[:-1]):
        raise ValueError(f"171 material-{expected_material} support has a node-group entry with no parent tets")
    arrays["node_group_offsets"] = offsets
    arrays["node_group_ids"] = group_ids
    arrays["node_group_parent_tet_offsets"] = parent_offsets
    arrays["node_group_parent_tet_rows"] = parent_rows.astype(np.int64, copy=False)
    return arrays


def _read_combined_node_roles(path: Path) -> dict[int, set[tuple[str, int]]]:
    """Read the authenticated combined node-map role crosswalk.

    Interface nodes can have reciprocal rows (one for each source partition);
    each such row must describe the same complete primary/peer role pair.
    """
    expected_roles: dict[int, set[tuple[str, int]]] = {}
    primary_rows: dict[int, set[tuple[str, int]]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"joint_node_id", "source_partition", "source_tetgen_node_id",
                    "merge_peer_partition", "merge_peer_tetgen_node_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("authenticated combined node map lacks source role crosswalk columns")
        for row in reader:
            node = int(row["joint_node_id"]) - 1
            if node < 0:
                raise ValueError("combined node map has a nonpositive joint-node ID")
            roles: set[tuple[str, int]] = set()
            partition, source_id = row["source_partition"], int(row["source_tetgen_node_id"])
            if partition not in ("skin", "fat") or source_id < 1:
                raise ValueError(f"combined node map has invalid primary source identity at node {node + 1}")
            primary = (partition, source_id)
            roles.add(primary)
            peer_partition, peer_id_text = row["merge_peer_partition"], row["merge_peer_tetgen_node_id"]
            if bool(peer_partition) != bool(peer_id_text):
                raise ValueError(f"combined node map has incomplete peer identity at node {node + 1}")
            if peer_partition:
                peer_id = int(peer_id_text)
                if peer_partition not in ("skin", "fat") or peer_id < 1 or (peer_partition, peer_id) in roles:
                    raise ValueError(f"combined node map has invalid peer identity at node {node + 1}")
                roles.add((peer_partition, peer_id))
            existing = expected_roles.setdefault(node, roles)
            if existing != roles:
                raise ValueError(f"combined node-map reciprocal role rows disagree at node {node + 1}")
            primaries = primary_rows.setdefault(node, set())
            if primary in primaries:
                raise ValueError(f"combined node map duplicates a primary source row at node {node + 1}")
            primaries.add(primary)
    if set(expected_roles) != set(range(len(expected_roles))):
        raise ValueError("combined node map joint-node domain is not contiguous from 1")
    for node, roles in expected_roles.items():
        if primary_rows[node] != roles:
            raise ValueError(f"combined node map lacks reciprocal role rows at node {node + 1}")
    return expected_roles


def verify_registered_node_role_support(base_registration_map_path: Path,
                                         rebased_registration_map_path: Path,
                                         base_combined_node_map_path: Path,
                                         new_to_base: np.ndarray,
                                         candidate_node_count: int,
                                         skin_support_path: Path,
                                         fat_support_path: Path) -> dict[str, Any]:
    """Validate 177 records by source partition against each 171 material node CSR.

    The 177 donor/region columns remain registration metadata. They are not
    broadcast to every material bit on an interface node or interpreted as a
    semantic authorization union.
    """
    mapping = np.asarray(new_to_base, dtype=np.int64)
    if mapping.shape != (candidate_node_count,) or np.any(mapping < 0):
        raise ValueError("177 role-support replay requires a complete candidate-to-base node map")
    expected_by_base = _read_combined_node_roles(base_combined_node_map_path)
    if len(mapping) != len(expected_by_base) or len(np.unique(mapping)) != len(mapping):
        raise ValueError("177 role-support replay requires a full base-node bijection")
    supports = {"skin": _load_material_node_group_support(skin_support_path, 15),
                "fat": _load_material_node_group_support(fat_support_path, 16)}
    role_material = {"skin": 15, "fat": 16}
    with base_registration_map_path.open("r", encoding="utf-8", newline="") as f:
        base_reader = csv.DictReader(f)
        if not base_reader.fieldnames or "joint_node_id" not in base_reader.fieldnames:
            raise ValueError("base 177 map lacks joint_node_id")
        base_header = list(base_reader.fieldnames)
        base_record_digests: dict[tuple[int, str], str] = {}
        digest_fields = [field for field in base_header if field != "joint_node_id"]
        for row in base_reader:
            key = (int(row["joint_node_id"]) - 1, row.get("source_partition", ""))
            if key in base_record_digests:
                raise ValueError(f"base 177 repeats source-partition record {key}")
            h = hashlib.sha256()
            for field in digest_fields:
                value = row[field].encode("utf-8")
                h.update(len(value).to_bytes(8, "little"))
                h.update(value)
            base_record_digests[key] = h.hexdigest()
    expected_record_keys = {(node, role) for node, role_ids in expected_by_base.items()
                            for role, _source_id in role_ids}
    if set(base_record_digests) != expected_record_keys:
        missing = sorted(expected_record_keys - set(base_record_digests))[:8]
        extra = sorted(set(base_record_digests) - expected_record_keys)[:8]
        raise ValueError(f"base 177 records differ from authenticated combined node roles; missing={missing}, extra={extra}")

    seen_by_candidate: dict[int, set[tuple[str, int]]] = {}
    seen_material_mask: dict[int, int] = {}
    records = groups_checked = parents_checked = 0
    with rebased_registration_map_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != base_header:
            raise ValueError("rebased 177 header differs from authenticated input header")
        for row in reader:
            records += 1
            candidate_node = int(row["joint_node_id"]) - 1
            if candidate_node < 0 or candidate_node >= candidate_node_count:
                raise ValueError("rebased 177 joint_node_id is outside candidate-node domain")
            base_node = int(mapping[candidate_node])
            role = row.get("source_partition", "")
            try:
                source_id = int(row["source_tetgen_node_id"])
            except (KeyError, ValueError) as exc:
                raise ValueError("rebased 177 record lacks valid role-local source node ID") from exc
            key = (role, source_id)
            if key not in expected_by_base.get(base_node, set()):
                raise ValueError(f"rebased 177 role/source identity disagrees with base combined map at candidate node {candidate_node + 1}: {key}")
            bucket = seen_by_candidate.setdefault(candidate_node, set())
            if key in bucket:
                raise ValueError(f"rebased 177 duplicates role/source record at candidate node {candidate_node + 1}: {key}")
            bucket.add(key)
            h = hashlib.sha256()
            for field in base_header:
                if field == "joint_node_id":
                    continue
                value = row[field].encode("utf-8")
                h.update(len(value).to_bytes(8, "little"))
                h.update(value)
            if h.hexdigest() != base_record_digests[(base_node, role)]:
                raise ValueError(f"177 source-partition metadata changed during node reindex at candidate node {candidate_node + 1}")
            mask = int(row["incident_material_mask"])
            if mask <= 0 or mask & ~((1 << 15) | (1 << 16)):
                raise ValueError(f"177 candidate node {candidate_node + 1} has invalid incident material mask")
            old_mask = seen_material_mask.setdefault(candidate_node, mask)
            if old_mask != mask:
                raise ValueError(f"177 duplicate role records disagree on incident material mask at node {candidate_node + 1}")
            if int(row["geometric_donor_body"]) < 0 or int(row["anatomical_region_id"]) < 0:
                raise ValueError(f"177 candidate node {candidate_node + 1} has negative preserved donor/region metadata")
            material = role_material.get(role)
            if material is None:
                raise ValueError(f"177 candidate node {candidate_node + 1} has unknown source partition {role!r}")
            source_row = source_id - 1
            support = supports[role]
            offsets = support["node_group_offsets"]
            if source_row < 0 or source_row + 1 >= len(offsets):
                raise ValueError(f"177 source node {source_id} exceeds 171 {role} P1 node domain")
            first, last = map(int, offsets[source_row:source_row + 2])
            group_ids = support["node_group_ids"][first:last]
            if len(group_ids) == 0:
                raise ValueError(f"177 source node {source_id} has no authenticated 171 {role} P1 group")
            if np.any(support["group_tuples"][group_ids, 0] != material):
                raise ValueError(f"171 {role} P1 node-group CSR crosses material at source node {source_id}")
            parent_offsets = support["node_group_parent_tet_offsets"]
            for group_position in range(first, last):
                gid = int(support["node_group_ids"][group_position])
                p0, p1 = map(int, parent_offsets[group_position:group_position + 2])
                if p1 <= p0:
                    raise ValueError(f"171 {role} node-group entry {group_position} (group {gid}) has no parent support")
                parents_checked += p1 - p0
            groups_checked += len(group_ids)
    if records != len(expected_record_keys):
        raise ValueError(f"rebased 177 record count {records} differs from authenticated role-row count {len(expected_record_keys)}")
    if set(seen_by_candidate) != set(range(candidate_node_count)):
        missing = sorted(set(range(candidate_node_count)) - set(seen_by_candidate))[:12]
        raise ValueError(f"rebased 177 map does not cover every candidate node; missing={missing}")
    for candidate_node, base_node_value in enumerate(mapping):
        expected_roles = expected_by_base[int(base_node_value)]
        if seen_by_candidate[candidate_node] != expected_roles:
            raise ValueError(f"rebased 177 role rows differ from authenticated combined map at candidate node {candidate_node + 1}")
        expected_mask = sum(1 << role_material[role] for role, _source_id in expected_roles)
        if seen_material_mask[candidate_node] != expected_mask:
            raise ValueError(f"177 incident material mask differs from source-partition roles at candidate node {candidate_node + 1}")
    role_counts = {role: sum(1 for role_ids in expected_by_base.values() for item_role, _ in role_ids
                              if item_role == role) for role in role_material}
    return {"schema": "numi.217.registered-node-role-material-p1-support.v1",
            "status": "PASS_REINDEXED_177_ROLE_ROWS_WITH_MATERIAL_SPECIFIC_171_NODE_GROUPS",
            "registered_nodes": candidate_node_count, "registered_node_records": records,
            "role_source_node_records": role_counts, "p1_node_groups_checked": groups_checked,
            "p1_parent_support_rows_checked": parents_checked,
            "scalar_donor_region_fields": "PRESERVED_177_METADATA_NOT_REINTERPRETED_AS_P1_UNION",
            "physical_admission": False, "mass_mapping": "NOT_RUN", "runtime": 0}

def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def merge_incremental(*, base_node_path: Path, base_ele_path: Path,
                      candidate_node_path: Path, candidate_ele_path: Path,
                      node_map_path: Path,
                      base_labels_path: Path, base_source_path: Path,
                      base_projection_path: Path, base_anchor_path: Path,
                      base_export_receipt_path: Path,
                      fresh_dir: Path, fresh_receipt_path: Path,
                      output_dir: Path,
                      base_registration_map_path: Path | None = None,
                      base_registration_receipt_path: Path | None = None) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to write into nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_receipt, reg_receipt = verify_base_receipts(
        base_labels_path, base_source_path, base_projection_path, base_anchor_path,
        base_node_path, base_ele_path, base_export_receipt_path,
        base_registration_map_path, base_registration_receipt_path)
    base_points, base_node_ids = read_tetgen_node(base_node_path)
    candidate_points, candidate_node_ids = read_tetgen_node(candidate_node_path)
    base_tets, base_materials = read_tetgen_ele(base_ele_path, base_node_ids)
    candidate_tets, candidate_materials = read_tetgen_ele(candidate_ele_path, candidate_node_ids)
    node_map = np.load(node_map_path, allow_pickle=False)
    mapped_nodes, node_bijection = validate_node_map(base_points, candidate_points, node_map)
    old_for_new, old_to_new = build_ordered_tet_row_map(base_tets, base_materials,
                                                        candidate_tets, candidate_materials,
                                                        node_map)
    if len(base_tets) != int(base_receipt.get("counts", {}).get("child_tetrahedra", -1)):
        raise ValueError("base tet row count differs from 171 receipt")
    changed_rows = np.flatnonzero(old_for_new < 0).astype(np.int64)
    retired_rows = np.flatnonzero(old_to_new < 0).astype(np.int64)
    changed_set = set(map(int, changed_rows))
    fresh_receipt = json.loads(fresh_receipt_path.read_text(encoding="utf-8"))
    if fresh_receipt.get("schema") != FRESH_RECEIPT_SCHEMA or fresh_receipt.get("status") != "PASS_SOURCE_ONLY_CHANGED_ROW_SUBSET":
        raise ValueError("fresh changed-row artifact lacks the required authenticated 217 subset status")
    if fresh_receipt.get("physical_admission") is not False or fresh_receipt.get("native_steps") != 0 or fresh_receipt.get("mass_transport") != "NOT_RUN":
        raise ValueError("fresh changed-row receipt makes an unsupported execution/admission claim")
    if fresh_receipt.get("full_child_domain_export") is not False:
        raise ValueError("fresh source-217 artifact must be an explicitly partial changed-row export")
    required_input_pins = ("candidate_child_node", "candidate_child_elements")
    for key, path in (("candidate_child_node", candidate_node_path), ("candidate_child_elements", candidate_ele_path)):
        if fresh_receipt.get("inputs", {}).get(key, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"fresh 217 receipt does not pin candidate input {key}")
    invariant_inputs = (
        "parent_mass", "parent_nodes", "parent_source", "parent_tets", "parent_labels",
        "parent_full86_weights", "current_230_owner", "batched_projection_helper",
        "skin_p1_support", "skin_p1_support_receipt", "fat_p1_support", "fat_p1_support_receipt",
        "combined_node_map",
    )
    for key in invariant_inputs:
        base_pin = base_receipt.get("inputs", {}).get(key, {}).get("sha256")
        fresh_pin = fresh_receipt.get("inputs", {}).get(key, {}).get("sha256")
        if not base_pin or fresh_pin != base_pin:
            raise ValueError(f"fresh 217 changed-row run is not bound to unchanged source owner input {key}")
    if fresh_receipt.get("counts", {}).get("exported_child_rows") != len(changed_rows):
        raise ValueError("fresh 217 changed-row count differs from exact unmatched candidate cell count")
    fresh_paths = {
        "labels": fresh_dir / "child.labels.csv",
        "parent_row_map": fresh_dir / "child.labels.csv.source.csv",
        "p1_anchor_projection_csv": fresh_dir / "p1-anchor-projection.csv",
        "p1_anchor_csr": fresh_dir / "p1-anchor.npz",
    }
    for key, path in fresh_paths.items():
        _verify_artifact_pin(fresh_receipt, key, path)
    fresh_arrays = _load_npz(fresh_paths["p1_anchor_csr"])
    fresh_rows = np.asarray(fresh_arrays.get("child_tet_row"), dtype=np.int64)
    if set(map(int, fresh_rows)) != changed_set:
        raise ValueError("fresh P1 CSR does not cover exactly the changed candidate rows")
    fresh_row_index = {int(row): i for i, row in enumerate(fresh_rows)}
    if len(fresh_row_index) != len(fresh_rows):
        raise ValueError("fresh P1 CSR repeats a candidate row")
    labels_header, fresh_labels = load_csv_rows_by_id(fresh_paths["labels"], LABEL_HEADER, changed_set)
    _, fresh_source = load_csv_rows_by_id(fresh_paths["parent_row_map"], expected_ids=changed_set)
    _, fresh_projection = load_csv_rows_by_id(fresh_paths["p1_anchor_projection_csv"], expected_ids=changed_set)
    # Check source-only labels against exact current candidate material and P1
    # anchor union before any unchanged rows are copied into the new domain.
    changed_validation = validate_changed_semantics(fresh_labels, fresh_projection,
                                                    fresh_arrays, fresh_row_index,
                                                    candidate_tets, candidate_materials)
    base_arrays = _load_npz(base_anchor_path)
    base_count = len(np.asarray(base_arrays.get("child_tet_row"), dtype=np.int64))
    if base_count != len(base_tets):
        raise ValueError("base P1 CSR row count differs from base TetGen element count")
    merged_arrays = merge_anchor_archives(base_arrays, fresh_arrays, old_for_new, fresh_row_index)
    merged_labels = output_dir / "child.labels.csv"
    merged_source = output_dir / "child.labels.csv.source.csv"
    merged_projection = output_dir / "p1-anchor-projection.csv"
    merged_anchor = output_dir / "p1-anchor.npz"
    label_info = merge_csv(base_labels_path, fresh_paths["labels"], merged_labels,
                           old_for_new, fresh_row_index, base_count=base_count,
                           expected_header=LABEL_HEADER)
    source_info = merge_csv(base_source_path, fresh_paths["parent_row_map"], merged_source,
                            old_for_new, fresh_row_index, base_count=base_count)
    projection_info = merge_csv(base_projection_path, fresh_paths["p1_anchor_projection_csv"], merged_projection,
                                old_for_new, fresh_row_index, base_count=base_count)
    np.savez_compressed(merged_anchor, **merged_arrays)
    anchor_info = {"path": str(merged_anchor), "sha256": sha256_file(merged_anchor),
                   "child_rows": len(candidate_tets), "csr_payload_rows": {
                       key: int(len(merged_arrays[data_keys[0]])) for key, data_keys in CSR_GROUPS.items()}}
    output_reg_map = None
    node_registration_reuse: dict[str, Any]
    if reg_receipt is not None and base_registration_map_path is not None:
        output_reg_map = output_dir / "registered-node-map.csv"
        node_registration_reuse = rebase_registered_node_map(
            base_registration_map_path, output_reg_map, node_map, node_bijection, candidate_points)
        if node_registration_reuse.get("status") == "PASS_REINDEXED_BASE_NODE_REGISTRATION":
            input_pins = base_receipt.get("inputs", {})
            def pinned_input(key: str) -> Path:
                item = input_pins.get(key, {})
                path = Path(item.get("path", ""))
                if not path.is_file() or item.get("sha256") != sha256_file(path):
                    raise ValueError(f"base 171 receipt does not provide a valid pinned {key} input")
                return path
            combined_map = pinned_input("combined_node_map")
            skin_support = pinned_input("skin_p1_support")
            fat_support = pinned_input("fat_p1_support")
            node_registration_reuse["role_material_p1_support"] = verify_registered_node_role_support(
                base_registration_map_path, output_reg_map, combined_map, node_map,
                len(candidate_points), skin_support, fat_support)
    else:
        node_registration_reuse = {"status": "NOT_REQUESTED"}
    row_map_out = output_dir / "old-row-map-to-candidate.npz"
    np.savez_compressed(row_map_out, candidate_to_base_tet_row=old_for_new,
                        base_to_candidate_tet_row=old_to_new,
                        candidate_to_base_node_row=np.asarray(node_map, dtype=np.int64))
    summary = {
        "schema": INCREMENTAL_RECEIPT_SCHEMA,
        "status": "INCREMENTAL_SOURCE_LABELS_AND_P1_UNION_MERGED_NOT_PHYSICAL_ADMISSION",
        "physical_admission": False, "native_steps": 0, "mass_transport": "NOT_RUN",
        "geometry_qualification": "NOT_ASSERTED_BY_THIS_SOURCE_METADATA_MERGER",
        "base_171_receipt_sha256": sha256_file(base_export_receipt_path),
        "base_177_receipt_sha256": sha256_file(base_registration_receipt_path) if base_registration_receipt_path else None,
        "fresh_changed_row_receipt_sha256": sha256_file(fresh_receipt_path),
        "candidate_inputs": {"node": {"path": str(candidate_node_path), "sha256": sha256_file(candidate_node_path)},
                             "elements": {"path": str(candidate_ele_path), "sha256": sha256_file(candidate_ele_path)},
                             "node_map": {"path": str(node_map_path), "sha256": sha256_file(node_map_path)}},
        "counts": {"base_cells": len(base_tets), "candidate_cells": len(candidate_tets),
                   "reused_exact_ordered_cells": int(np.count_nonzero(old_for_new >= 0)),
                   "fresh_changed_cells": int(len(changed_rows)),
                   "retired_base_cells": int(len(retired_rows)), "mapped_nodes": int(mapped_nodes),
                   "candidate_nodes": len(candidate_points), "base_nodes": len(base_points)},
        "changed_row_validation": changed_validation,
        "node_registration_reuse": node_registration_reuse,
        "output_domain": {"labels": label_info, "source_map": source_info,
                          "p1_anchor_projection": projection_info, "p1_anchor_csr": anchor_info,
                          "registered_node_map": node_registration_reuse.get("output"),
                          "row_map": {"path": str(row_map_out), "sha256": sha256_file(row_map_out)}},
        "reuse_policy": "exact ordered material+corner tuple only; corner-permuted cells are regenerated; mapped F32 node coordinates must be byte-identical",
        "limitations": [
            "Only exact unchanged cell rows are reused. Every changed candidate cell must have a complete source-217 changed-row output bundle.",
            "The 177 node map is reused only for a complete one-to-one exact-F32 point reindex and the changed-row P1 anchor/union checks pass.",
            "This receipt makes no cell geometry, mass-transfer, native-import, physical, or motion qualification claim.",
        ],
    }
    receipt_path = output_dir / "incremental-reuse-receipt.json"
    _write_json(receipt_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-node", required=True, type=Path)
    parser.add_argument("--base-ele", required=True, type=Path)
    parser.add_argument("--candidate-node", required=True, type=Path)
    parser.add_argument("--candidate-ele", required=True, type=Path)
    parser.add_argument("--candidate-node-to-base-npy", required=True, type=Path)
    parser.add_argument("--base-labels", required=True, type=Path)
    parser.add_argument("--base-source", required=True, type=Path)
    parser.add_argument("--base-projection", required=True, type=Path)
    parser.add_argument("--base-p1-csr", required=True, type=Path)
    parser.add_argument("--base-171-receipt", required=True, type=Path)
    parser.add_argument("--base-177-map", type=Path)
    parser.add_argument("--base-177-receipt", type=Path)
    parser.add_argument("--fresh-dir", required=True, type=Path)
    parser.add_argument("--fresh-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = merge_incremental(
        base_node_path=args.base_node, base_ele_path=args.base_ele,
        candidate_node_path=args.candidate_node, candidate_ele_path=args.candidate_ele,
        node_map_path=args.candidate_node_to_base_npy,
        base_labels_path=args.base_labels, base_source_path=args.base_source,
        base_projection_path=args.base_projection, base_anchor_path=args.base_p1_csr,
        base_export_receipt_path=args.base_171_receipt,
        fresh_dir=args.fresh_dir, fresh_receipt_path=args.fresh_receipt,
        output_dir=args.output_dir, base_registration_map_path=args.base_177_map,
        base_registration_receipt_path=args.base_177_receipt)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
