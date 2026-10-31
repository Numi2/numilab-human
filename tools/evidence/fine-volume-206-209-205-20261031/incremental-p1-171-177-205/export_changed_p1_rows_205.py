#!/usr/bin/env python3
"""Regenerate only selected child tetra rows from authenticated current P1 node support.

This is the bounded changed-row producer paired with incremental_p1_reuse_205.py.
For each requested candidate tet it reconstructs the exact integrated group
score from its four existing, authenticated role-local node support rows, builds
the full parent/semantic union, and invokes the pinned source-217 projection and
exact-tie logic. It never reuses a retired cell's semantic tuple as new-cell
ancestry and never performs a mass allocation or physical admission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

import incremental_p1_reuse_205 as reuse

P1_CSR_SCHEMA = "numi.native-domain-217.p1-supported-reference-anchor-csr.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def import_from_path(path: Path, name: str):
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load pinned source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_requested_rows(path: Path, cell_count: int) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["candidate_child_tet_row"]:
            raise ValueError("changed-row CSV header must be exactly candidate_child_tet_row")
        values = [int(row["candidate_child_tet_row"]) for row in reader]
    rows = np.asarray(values, dtype=np.int64)
    if len(rows) == 0 or len(np.unique(rows)) != len(rows):
        raise ValueError("changed-row request must be nonempty and contain unique rows")
    if np.any(rows < 0) or np.any(rows >= cell_count):
        raise ValueError("changed-row request contains an out-of-range candidate tet row")
    if not np.array_equal(rows, np.sort(rows)):
        raise ValueError("changed-row request must be sorted for deterministic output")
    return rows


def _support_groups_for_node(p1, support: dict[str, Any], node_row: int, material: int,
                             parent: dict[str, Any]) -> dict[int, np.ndarray]:
    if node_row < 0 or node_row + 1 >= len(support["node_group_offsets"]):
        raise ValueError(f"role-local P1 node row out of range: {node_row}")
    lo = int(support["node_group_offsets"][node_row])
    hi = int(support["node_group_offsets"][node_row + 1])
    groups = np.asarray(support["node_group_ids"][lo:hi], dtype=np.int64)
    out: dict[int, np.ndarray] = {}
    for gid in groups:
        group_id = int(gid)
        group = tuple(map(int, support["group_tuples"][group_id]))
        if group[0] != material:
            continue
        parent_rows = p1.p1_node_group_parent_rows(support, node_row, group_id)
        if parent_rows is None or len(parent_rows) == 0:
            raise ValueError(f"node {node_row} group {group_id} has no P1 source parents")
        observed = np.column_stack((parent["materials"][parent_rows], parent["donors"][parent_rows],
                                   parent["regions"][parent_rows]))
        if np.any(observed != np.asarray(group, dtype=observed.dtype)):
            raise ValueError(f"node {node_row} group {group_id} parent semantic tuple mismatch")
        if np.any(parent["materials"][parent_rows] != material):
            raise ValueError(f"node {node_row} source parent pool crosses child material")
        out[group_id] = np.asarray(parent_rows, dtype=np.int32)
    return out


def build_cell_support(p1, support: dict[str, Any], node_source_rows: np.ndarray,
                       child_tet: np.ndarray, material: int,
                       parent: dict[str, Any]) -> dict[str, Any]:
    role_column = 0 if material == 15 else 1 if material == 16 else -1
    if role_column < 0:
        raise ValueError(f"unsupported child material {material}")
    per_node: list[dict[int, np.ndarray]] = []
    for child_node in sorted(map(int, child_tet)):
        role_node = int(node_source_rows[child_node, role_column])
        if role_node < 0:
            raise ValueError(f"child node {child_node} has no authenticated role-local P1 node row")
        node_groups = _support_groups_for_node(p1, support, role_node, material, parent)
        if not node_groups:
            raise ValueError(f"child node {child_node} has no source support for material {material}")
        per_node.append(node_groups)
    group_lists = [sorted(groups) for groups in per_node]
    scale = 1
    for groups in group_lists:
        scale = math.lcm(scale, len(groups))
    scores: dict[int, int] = {}
    for groups in group_lists:
        share = scale // len(groups)
        for gid in groups:
            scores[gid] = scores.get(gid, 0) + share
    denominator = 4 * scale
    if not scores or denominator <= 0:
        raise ValueError("four-node P1 support produced no semantic score")
    maximum = max(scores.values())
    primary_group_id = min(gid for gid, score in scores.items() if score == maximum)
    primary_group = tuple(map(int, support["group_tuples"][primary_group_id]))
    if primary_group[0] != material:
        raise ValueError("deterministic primary P1 group differs from child material")
    parent_by_group: dict[int, np.ndarray] = {}
    for gid in sorted(scores):
        rows = [nodes[gid] for nodes in per_node if gid in nodes]
        group_parents = np.unique(np.concatenate(rows)).astype(np.int32, copy=False)
        if not len(group_parents):
            raise ValueError(f"P1 group {gid} has empty four-node parent union")
        group_tuple = tuple(map(int, support["group_tuples"][gid]))
        observed = np.column_stack((parent["materials"][group_parents], parent["donors"][group_parents],
                                   parent["regions"][group_parents]))
        if np.any(observed != np.asarray(group_tuple, dtype=observed.dtype)):
            raise ValueError(f"P1 group {gid} four-node union differs from its typed source tuple")
        parent_by_group[gid] = group_parents
    all_parent_rows = np.unique(np.concatenate(list(parent_by_group.values()))).astype(np.int32, copy=False)
    primary_rows = parent_by_group[primary_group_id]
    semantic_tuples = np.asarray(sorted({
        tuple(map(int, (parent["materials"][row], parent["donors"][row],
                       parent["regions"][row], parent["faces"][row])))
        for row in all_parent_rows}), dtype=np.uint32).reshape((-1, 4))
    if not len(semantic_tuples) or np.any(semantic_tuples[:, 0] != material):
        raise ValueError("candidate cell P1 semantic union is empty or crosses material")
    # The source owner chooses the smallest child node among vertices whose
    # authenticated role-local node+group pool supports this exact primary group.
    eligible = []
    for child_node in sorted(map(int, child_tet)):
        role_node = int(node_source_rows[child_node, role_column])
        rows = p1.p1_node_group_parent_rows(support, role_node, primary_group_id)
        if rows is not None:
            eligible.append((child_node, role_node, np.asarray(rows, dtype=np.int32)))
    if not eligible:
        raise ValueError("no tet vertex has the selected primary P1 node-group support")
    carrier_node, carrier_role_node, carrier_rows = eligible[0]
    carrier_rows = np.unique(carrier_rows).astype(np.int32, copy=False)
    if not np.isin(carrier_rows, primary_rows).all():
        raise ValueError("carrier node-group parent pool is not a subset of the primary cell P1 parent union")
    return {
        "primary_group_id": primary_group_id,
        "primary_group_tuple": np.asarray(primary_group, dtype=np.int32),
        "score_numerator": int(scores[primary_group_id]),
        "score_denominator": int(denominator),
        "all_parent_rows": all_parent_rows,
        "primary_parent_rows": primary_rows,
        "carrier_parent_rows": carrier_rows,
        "semantic_tuples": semantic_tuples,
        "carrier_child_node_row": carrier_node,
        "carrier_role_local_node_row": carrier_role_node,
        "carrier_query": None,
    }


def _append_csr(offsets: dict[str, list[int]], data: dict[str, list[np.ndarray]],
                offset_key: str, values: dict[str, np.ndarray]) -> None:
    current = offsets[offset_key][-1]
    length = len(next(iter(values.values()))) if values else 0
    for key, array in values.items():
        arr = np.asarray(array)
        data[key].append(arr.copy())
        if len(arr) != length:
            raise ValueError(f"unaligned payloads in {offset_key}: {key}")
    offsets[offset_key].append(current + length)


def _csr_output(offsets: dict[str, list[int]], data: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, vals in offsets.items():
        out[key] = np.asarray(vals, dtype=np.int64)
    for key, pieces in data.items():
        if pieces:
            out[key] = np.concatenate(pieces, axis=0)
        else:
            raise ValueError(f"changed-row exporter produced no payload for {key}")
    return out


def regenerate_changed_rows(*, base_receipt_path: Path, candidate_node_path: Path,
                            candidate_ele_path: Path, requested_rows_path: Path,
                            output_dir: Path, max_distance_m: float = 0.02,
                            ambiguity_tolerance_m: float = 1.0e-8,
                            projection_tolerance_m: float = 1.0e-12,
                            full86_tolerance: float = 1.0e-10) -> dict[str, Any]:
    started = time.perf_counter()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty changed-row output directory {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_receipt = json.loads(base_receipt_path.read_text(encoding="utf-8"))
    if (base_receipt.get("schema") != "numi.native-domain-217.p1-supported-reference-anchor.v1"
            or base_receipt.get("status") != "P1_SUPPORTED_REFERENCE_ANCHOR_LABELS_EXPORTED_NOT_PHYSICAL_ADMISSION"
            or base_receipt.get("full_child_domain_export") is not True
            or base_receipt.get("physical_admission") is not False):
        raise ValueError("changed-row exporter requires the authenticated full-domain 171 source receipt")
    inputs = base_receipt.get("inputs", {})
    native_path = Path(inputs["native_217_label_exporter"]["path"])
    p1_path = Path(inputs["p1_anchor_217_exporter"]["path"])
    for key, path in (("native_217_label_exporter", native_path), ("p1_anchor_217_exporter", p1_path)):
        if sha256_file(path) != inputs[key]["sha256"]:
            raise ValueError(f"pinned 217 source module changed: {key}")
    native = import_from_path(native_path, "incremental_native_217_source007")
    # The authenticated 171 run uses the explicit full-reference profile: this
    # child exceeds legacy 300k/200k parsing limits. Parent-source limits remain
    # the pinned legacy limits inside the owner; this only selects the already
    # qualified child-output capacity profile.
    capacity_profile = native.configure_capacity_profile("full-reference-500k-nodes-2m-cells")
    if (capacity_profile.get("max_nodes_per_child") != 500_000 or
            capacity_profile.get("max_cells_per_child") != 2_000_000 or
            capacity_profile.get("max_nodes_combined") != 500_000 or
            capacity_profile.get("max_cells_combined") != 2_000_000):
        raise ValueError("pinned 217 owner did not apply the expected full-reference child profile")
    p1_path.parent.mkdir(parents=True, exist_ok=True)
    p1 = import_from_path(p1_path, "incremental_p1_anchor_217_source007")
    parent_base = Path(inputs["parent_mass"]["path"])
    parent_full86_path = Path(inputs["parent_full86_weights"]["path"])
    skin_support_path = Path(inputs["skin_p1_support"]["path"])
    skin_support_receipt_path = Path(inputs["skin_p1_support_receipt"]["path"])
    fat_support_path = Path(inputs["fat_p1_support"]["path"])
    fat_support_receipt_path = Path(inputs["fat_p1_support_receipt"]["path"])
    combined_node_map_path = Path(inputs["combined_node_map"]["path"])
    combined_receipt_path = Path(inputs["combined_handoff_receipt"]["path"])
    projection_helper_path = Path(inputs["batched_projection_helper"]["path"])
    projection_helper_sha = inputs["batched_projection_helper"]["sha256"]
    owner, owner_pin = native.load_same_material_230_owner(native)
    owner_input = inputs.get("current_230_owner", {})
    if owner_pin.get("sha256") != owner_input.get("sha256"):
        raise ValueError("loaded 230 owner differs from the exact owner pinned by base 171")
    parent, parent_index, _material_indices, parent_meta = owner.load_parent_current(parent_base)
    full86, full86_meta = owner.load_parent_full86_weights(parent_full86_path, parent, parent_meta)
    if full86_meta.get("sha256") != inputs["parent_full86_weights"]["sha256"]:
        raise ValueError("source330 full86 weights differ from base 171 input pin")
    skin_support = p1.load_p1_support("skin", skin_support_path, skin_support_receipt_path,
                                      parent_meta, parent)
    fat_support = p1.load_p1_support("fat", fat_support_path, fat_support_receipt_path,
                                     parent_meta, parent)
    child = native.read_tetgen_vectorized(candidate_node_path, candidate_ele_path)
    node_source_rows = p1.load_combined_node_role_rows(
        combined_node_map_path, combined_receipt_path, child, skin_support, fat_support)
    if np.any(~np.isin(child["materials"], np.asarray([15, 16]))):
        raise ValueError("candidate child element table contains unsupported material rows")
    selected_rows = load_requested_rows(requested_rows_path, len(child["tets"]))
    project_pairs, project_pin = p1._load_project_pairs(projection_helper_path, projection_helper_sha)
    exact_tetra = getattr(owner, "_closest_tetra_exact", None)
    exact_fraction = getattr(owner, "_fraction", None)
    if not callable(exact_tetra) or not callable(exact_fraction):
        raise ValueError("pinned source330 owner lacks exact dyadic point-to-tet projection")
    exact_cache: dict[tuple[int, tuple[float, float, float]], tuple[Any, Any, Any]] = {}
    def exact_project(parent_row: int, point: np.ndarray):
        q = tuple(map(float, np.asarray(point, dtype=np.float64)))
        key = (int(parent_row), q)
        if key not in exact_cache:
            nodes = parent["tets"][parent_row]
            vertices = tuple(tuple(exact_fraction(float(x)) for x in parent["points"][node]) for node in nodes)
            exact_point = tuple(exact_fraction(x) for x in q)
            exact_cache[key] = exact_tetra(vertices, exact_point)
        return exact_cache[key]
    supports = {"skin": skin_support, "fat": fat_support}
    records: list[dict[str, Any]] = []
    centroid_rows: list[np.ndarray] = []
    centroid_points: list[np.ndarray] = []
    centroid_offsets = [0]
    projection_cache: dict[tuple[int, int, tuple[int, ...]], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    missing_keys: dict[tuple[int, int, tuple[int, ...]], tuple[np.ndarray, np.ndarray]] = {}
    for candidate_row in selected_rows:
        row = int(candidate_row)
        material = int(child["materials"][row])
        support = supports["skin" if material == 15 else "fat"]
        tet_nodes = np.asarray(child["tets"][row], dtype=np.int64)
        support_row = build_cell_support(p1, support, node_source_rows, tet_nodes, material, parent)
        carrier = int(support_row["carrier_child_node_row"])
        query = np.asarray(child["points"][carrier], dtype=np.float64)
        pool = support_row["carrier_parent_rows"]
        key = (carrier, int(support_row["primary_group_id"]), tuple(map(int, pool)))
        if key not in projection_cache:
            missing_keys[key] = (pool, query)
        support_row["carrier_query"] = query
        centroid = np.asarray(child["points"][tet_nodes], dtype=np.float64).mean(axis=0)
        centroid_rows.extend(support_row["all_parent_rows"].tolist())
        centroid_points.append(centroid)
        centroid_offsets.append(len(centroid_rows))
        records.append({"candidate_row": row, "material": material, "tet_nodes": tet_nodes,
                        "centroid": centroid, "support": support_row, "cache_key": key})
    if missing_keys:
        ordered_keys = list(missing_keys)
        pair_parent_rows = np.concatenate([missing_keys[k][0] for k in ordered_keys]).astype(np.int64, copy=False)
        pair_points = np.concatenate([np.repeat(missing_keys[k][1].reshape((1, 3)), len(missing_keys[k][0]), axis=0)
                                      for k in ordered_keys], axis=0)
        projected = project_pairs(pair_points, pair_parent_rows, parent["points"], parent["tets"],
                                  parent_index["inverses"])
        cursor = 0
        for key in ordered_keys:
            count = len(missing_keys[key][0]); end = cursor + count
            projection_cache[key] = (
                np.asarray(projected["projected"][cursor:end], dtype=np.float64).copy(),
                np.asarray(projected["bary"][cursor:end], dtype=np.float64).copy(),
                np.asarray(projected["distance"][cursor:end], dtype=np.float64).copy())
            cursor = end
    centroid_points_a = np.asarray(centroid_points, dtype=np.float64)
    centroid_rows_a = np.asarray(centroid_rows, dtype=np.int64)
    centroid_projection = project_pairs(np.repeat(centroid_points_a, np.diff(centroid_offsets), axis=0),
                                        centroid_rows_a, parent["points"], parent["tets"],
                                        parent_index["inverses"])
    # Assemble the same per-cell and CSR arrays as source 217. The sole domain
    # distinction is p1_support_cell_row=-1: a new tetra has no old cell-row ID;
    # its P1 support was recomputed from the four authenticated current node rows.
    cell: dict[str, list[Any]] = {key: [] for key in reuse.CELL_KEYS}
    csr_offsets = {key: [0] for key in reuse.CSR_GROUPS}
    csr_data = {data_key: [] for keys in reuse.CSR_GROUPS.values() for data_key in keys}
    labels_path = output_dir / "child.labels.csv"
    source_path = output_dir / "child.labels.csv.source.csv"
    projection_path = output_dir / "p1-anchor-projection.csv"
    anchor_path = output_dir / "p1-anchor.npz"
    receipt_path = output_dir / "receipt.json"
    outputs = (labels_path, source_path, projection_path, anchor_path, receipt_path)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite changed-row source outputs")
    label_rows = []
    source_rows = []
    projection_rows = []
    counts = {"rows": len(records), "p1_pool_rows": 0, "p1_primary_parent_rows": 0,
              "p1_carrier_parent_rows": 0, "p1_pool_tuples": 0,
              "reference_anchor_tie_rows": 0, "reference_anchor_tie_cells": 0,
              "reference_anchor_tie_tuples": 0, "near_candidate_rows": 0,
              "near_candidate_cells": 0, "exact_pool_resolution_cells": 0,
              "projection_tolerance_max_m": 0.0, "full86_tolerance_max": 0.0,
              "carrier_projection_cache_unique_keys": len(projection_cache),
              "carrier_projection_cache_hit_requests": len(records) - len(projection_cache)}
    for local, rec in enumerate(records):
        row = rec["candidate_row"]
        material = rec["material"]
        sup = rec["support"]
        key = rec["cache_key"]
        candidate_parents = sup["carrier_parent_rows"]
        candidate_proj = projection_cache[key]
        pool_lo, pool_hi = centroid_offsets[local], centroid_offsets[local + 1]
        full_pool = sup["all_parent_rows"]
        centroid_distance = np.asarray(centroid_projection["distance"][pool_lo:pool_hi], dtype=np.float64)
        cmin = float(centroid_distance.min())
        cidx = min((i for i, value in enumerate(centroid_distance) if value == cmin),
                   key=lambda i: int(full_pool[i]))
        centroid_parent_row = int(full_pool[cidx])
        centroid_bary = np.asarray(centroid_projection["bary"][pool_lo + cidx], dtype=np.float64)
        centroid_registered = np.asarray(centroid_projection["projected"][pool_lo + cidx], dtype=np.float64)
        centroid_residual = rec["centroid"] - centroid_registered
        centroid_residual_norm = float(np.linalg.norm(centroid_residual))
        if abs(centroid_residual_norm - float(centroid_distance[cidx])) > 2.0e-9:
            raise ValueError(f"centroid source witness projection mismatch at candidate row {row}")
        chosen = p1.choose_p1_anchor(
            parent, full86, candidate_parents, candidate_proj[0], candidate_proj[1], candidate_proj[2],
            material, max_distance_m, ambiguity_tolerance_m, projection_tolerance_m, full86_tolerance,
            exact_project, candidate_query_points=np.repeat(sup["carrier_query"].reshape((1, 3)), len(candidate_parents), axis=0),
            candidate_child_node_rows=np.full(len(candidate_parents), sup["carrier_child_node_row"], dtype=np.int64),
            semantic_pool_rows=full_pool)
        selected_parent = int(chosen["parent_row"])
        semantic = tuple(map(int, native.semantic(parent, selected_parent)))
        primary_tuple = tuple(map(int, sup["primary_group_tuple"]))
        if semantic[:3] != primary_tuple:
            raise ValueError(f"candidate row {row} selected anchor is outside its exact max-share group")
        if semantic not in {tuple(map(int, t)) for t in chosen["semantic_union"]}:
            raise ValueError(f"candidate row {row} selected scalar tuple is absent from full P1 union")
        if int(chosen["carrier_child_node_row"]) not in set(map(int, rec["tet_nodes"])):
            raise ValueError(f"candidate row {row} carrier child node is not a tet vertex")
        label_rows.append((row, *semantic))
        carrier_node = int(chosen["carrier_child_node_row"])
        anchor_residual = np.asarray(chosen["carrier_query_point"], dtype=np.float64) - np.asarray(chosen["registered_point"], dtype=np.float64)
        projection_rows.append({
            "child_tet_row": row, "carrier_child_node_row": carrier_node,
            **{f"carrier_query_{axis}_m": format(float(chosen["carrier_query_point"][i]), ".17g")
               for i, axis in enumerate(("x", "y", "z"))},
            "selected_parent_tet_row": selected_parent,
            **{f"selected_parent_barycentric{i}": format(float(chosen["bary"][i]), ".17g") for i in range(4)},
            **{f"registered_world_{axis}_m": format(float(chosen["registered_point"][i]), ".17g")
               for i, axis in enumerate(("x", "y", "z"))},
            **{f"residual_{axis}_m": format(float(anchor_residual[i]), ".17g")
               for i, axis in enumerate(("x", "y", "z"))},
            "anchor_distance_m": format(float(chosen["registration_distance_m"]), ".17g"),
            "material_index": semantic[0], "geometric_donor_body": semantic[1],
            "anatomical_region_id": semantic[2], "stable_source_face_id": semantic[3],
            "primary_group_id": int(sup["primary_group_id"]),
            "carrier_node_group_id": int(sup["primary_group_id"]),
            "primary_score_numerator": int(sup["score_numerator"]),
            "primary_score_denominator": int(sup["score_denominator"]),
            "exact_resolution_used": int(bool(chosen["exact_resolution_used"])),
            "exact_min_squared_distance_ratio": chosen["exact_min_squared_distance_ratio"],
            "tie_parent_rows_json": json.dumps(list(map(int, chosen["tie_rows"])), separators=(",", ":")),
            "tie_semantic_tuples_json": json.dumps(chosen["tie_semantic_union"].astype(int).tolist(), separators=(",", ":")),
        })
        centroid_semantic = tuple(map(int, native.semantic(parent, centroid_parent_row)))
        source_rows.append({
            "child_tet_row": row, "selected_parent_tet_row": centroid_parent_row,
            "label_match_kind": "p1_nodal_supported_current_reference_anchor",
            "matching_parent_tet_count": 1, "semantic_tie_count": 0,
            **{f"centroid_barycentric{i}": format(float(centroid_bary[i]), ".17g") for i in range(4)},
            **{f"centroid_registered_reference_{axis}_m": format(float(centroid_registered[i]), ".17g")
               for i, axis in enumerate(("x", "y", "z"))},
            **{f"centroid_reference_residual_{axis}_m": format(float(centroid_residual[i]), ".17g")
               for i, axis in enumerate(("x", "y", "z"))},
            "centroid_reference_residual_norm_m": format(centroid_residual_norm, ".17g"),
            "reference_registration_distance_m": format(centroid_residual_norm, ".17g"),
            "reference_registration_max_distance_m": "",
            "reference_registration_ambiguity_tolerance_m": format(ambiguity_tolerance_m, ".17g"),
            "material_index": centroid_semantic[0], "geometric_donor_body": centroid_semantic[1],
            "anatomical_region_id": centroid_semantic[2], "stable_source_face_id": centroid_semantic[3],
        })
        cell["p1_support_role_material"].append(material)
        cell["p1_support_cell_row"].append(-1)
        cell["p1_primary_group_id"].append(int(sup["primary_group_id"]))
        cell["reference_anchor_carrier_node_group_id"].append(int(sup["primary_group_id"]))
        cell["p1_primary_group_score_numerator"].append(int(sup["score_numerator"]))
        cell["p1_primary_group_score_denominator"].append(int(sup["score_denominator"]))
        cell["p1_primary_group_tuple"].append(sup["primary_group_tuple"])
        cell["reference_anchor_near_candidate_projection_separation_m"].append(float(chosen["near_candidate_projection_separation_m"]))
        cell["reference_anchor_exact_pool_resolution_used"].append(int(chosen["exact_resolution_used"]))
        ratio = str(chosen["exact_min_squared_distance_ratio"])
        if len(ratio) > 78:
            raise ValueError("exact squared-distance ratio exceeds authenticated P1 storage width")
        cell["reference_anchor_exact_min_squared_distance_ratio"].append(ratio)
        cell["reference_anchor_selected_parent_tet_row"].append(selected_parent)
        cell["reference_anchor_selected_parent_barycentric"].append(chosen["bary"])
        cell["reference_anchor_selected_semantic_tuple"].append(semantic)
        cell["centroid_diagnostic_parent_tet_row"].append(centroid_parent_row)
        cell["centroid_diagnostic_barycentric"].append(centroid_bary)
        cell["centroid_diagnostic_registered_world_m"].append(centroid_registered)
        cell["centroid_diagnostic_residual_norm_m"].append(centroid_residual_norm)
        cell["reference_anchor_carrier_child_node_row"].append(carrier_node)
        cell["reference_anchor_carrier_query_world_m"].append(chosen["carrier_query_point"])
        cell["reference_anchor_centroid_to_projected_anchor_distance_m"].append(float(np.linalg.norm(rec["centroid"] - chosen["registered_point"])))
        cell["reference_anchor_registered_world_m"].append(chosen["registered_point"])
        cell["reference_anchor_distance_m"].append(float(chosen["registration_distance_m"]))
        cell["reference_anchor_tie_max_projection_separation_m"].append(float(chosen["maximum_tie_projection_separation_m"]))
        cell["reference_anchor_tie_max_full86_abs_difference"].append(float(chosen["maximum_tie_full86_weight_difference"]))
        _append_csr(csr_offsets, csr_data, "p1_parent_tet_offsets",
                    {"p1_parent_tet_rows": sup["all_parent_rows"]})
        _append_csr(csr_offsets, csr_data, "p1_primary_parent_tet_offsets",
                    {"p1_primary_parent_tet_rows": sup["primary_parent_rows"]})
        _append_csr(csr_offsets, csr_data, "reference_anchor_carrier_parent_tet_offsets",
                    {"reference_anchor_carrier_parent_tet_rows": sup["carrier_parent_rows"]})
        _append_csr(csr_offsets, csr_data, "p1_semantic_tuple_offsets",
                    {"p1_semantic_tuples": chosen["semantic_union"]})
        _append_csr(csr_offsets, csr_data, "reference_anchor_tie_offsets", {
            "reference_anchor_tie_parent_tet_rows": chosen["tie_rows"],
            "reference_anchor_tie_carrier_child_node_rows": chosen["tie_carrier_child_node_rows"],
            "reference_anchor_tie_barycentric": chosen["tie_barycentric"],
            "reference_anchor_tie_distance_m": chosen["tie_distances_m"],
        })
        _append_csr(csr_offsets, csr_data, "reference_anchor_tie_semantic_tuple_offsets",
                    {"reference_anchor_tie_semantic_tuples": chosen["tie_semantic_union"]})
        _append_csr(csr_offsets, csr_data, "reference_anchor_near_candidate_offsets", {
            "reference_anchor_near_candidate_parent_tet_rows": chosen["near_candidate_rows"],
            "reference_anchor_near_candidate_child_node_rows": chosen["near_candidate_child_node_rows"],
            "reference_anchor_near_candidate_approx_distance_m": chosen["near_candidate_approx_distance_m"],
            "reference_anchor_near_candidate_exact_distance_gap_m": chosen["near_candidate_exact_gap_m"],
        })
        counts["p1_pool_rows"] += len(sup["all_parent_rows"])
        counts["p1_primary_parent_rows"] += len(sup["primary_parent_rows"])
        counts["p1_carrier_parent_rows"] += len(sup["carrier_parent_rows"])
        counts["p1_pool_tuples"] += len(chosen["semantic_union"])
        counts["reference_anchor_tie_rows"] += len(chosen["tie_rows"])
        counts["reference_anchor_tie_cells"] += int(len(chosen["tie_rows"]) > 1)
        counts["reference_anchor_tie_tuples"] += len(chosen["tie_semantic_union"])
        counts["near_candidate_rows"] += len(chosen["near_candidate_rows"])
        counts["near_candidate_cells"] += int(len(chosen["near_candidate_rows"]) > 1)
        counts["exact_pool_resolution_cells"] += int(chosen["exact_resolution_used"])
        counts["projection_tolerance_max_m"] = max(counts["projection_tolerance_max_m"],
            float(chosen["maximum_tie_projection_separation_m"]))
        counts["full86_tolerance_max"] = max(counts["full86_tolerance_max"],
            float(chosen["maximum_tie_full86_weight_difference"]))
    arrays: dict[str, np.ndarray] = {"schema": np.asarray(P1_CSR_SCHEMA),
                                    "child_tet_row": selected_rows.astype(np.int64)}
    dtype_by_key = {
        "p1_support_role_material": np.uint8, "p1_support_cell_row": np.int32,
        "p1_primary_group_id": np.int32, "reference_anchor_carrier_node_group_id": np.int32,
        "p1_primary_group_score_numerator": np.int64, "p1_primary_group_score_denominator": np.int64,
        "p1_primary_group_tuple": np.int32,
        "reference_anchor_near_candidate_projection_separation_m": np.float64,
        "reference_anchor_exact_pool_resolution_used": np.uint8,
        "reference_anchor_exact_min_squared_distance_ratio": "U78",
        "reference_anchor_selected_parent_tet_row": np.int32,
        "reference_anchor_selected_parent_barycentric": np.float64,
        "reference_anchor_selected_semantic_tuple": np.uint32,
        "centroid_diagnostic_parent_tet_row": np.int32,
        "centroid_diagnostic_barycentric": np.float64,
        "centroid_diagnostic_registered_world_m": np.float64,
        "centroid_diagnostic_residual_norm_m": np.float64,
        "reference_anchor_carrier_child_node_row": np.int64,
        "reference_anchor_carrier_query_world_m": np.float64,
        "reference_anchor_centroid_to_projected_anchor_distance_m": np.float64,
        "reference_anchor_registered_world_m": np.float64,
        "reference_anchor_distance_m": np.float64,
        "reference_anchor_tie_max_projection_separation_m": np.float64,
        "reference_anchor_tie_max_full86_abs_difference": np.float64,
    }
    for key, dtype in dtype_by_key.items():
        arrays[key] = np.asarray(cell[key], dtype=dtype)
    for key, values in _csr_output(csr_offsets, csr_data).items():
        arrays[key] = values
    labels_path = output_dir / "child.labels.csv"
    source_path = output_dir / "child.labels.csv.source.csv"
    projection_path = output_dir / "p1-anchor-projection.csv"
    anchor_path = output_dir / "p1-anchor.npz"
    receipt_path = output_dir / "receipt.json"
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(native.LABEL_HEADER); w.writerows(label_rows)
    with source_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=native.SOURCE_HEADER, lineterminator="\n"); w.writeheader(); w.writerows(source_rows)
    with projection_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=p1.ANCHOR_PROJECTION_HEADER, lineterminator="\n"); w.writeheader(); w.writerows(projection_rows)
    np.savez_compressed(anchor_path, **arrays)
    invariant_keys = (
        "parent_mass", "parent_nodes", "parent_source", "parent_tets", "parent_labels",
        "parent_full86_weights", "current_230_owner", "batched_projection_helper",
        "skin_p1_support", "skin_p1_support_receipt", "fat_p1_support", "fat_p1_support_receipt",
    )
    output_paths = {"labels": labels_path, "parent_row_map": source_path,
                    "p1_anchor_projection_csv": projection_path, "p1_anchor_csr": anchor_path}
    receipt = {
        "schema": "numi.native-domain-217.p1-reference-anchor.changed-row-subset.v1",
        "status": "PASS_SOURCE_ONLY_CHANGED_ROW_SUBSET",
        "physical_admission": False, "native_steps": 0, "mass_transport": "NOT_RUN",
        "full_child_domain_export": False,
        "labels_are_direct_geometric_ancestry": False,
        "scalar_label_role": "P1-supported current-reference anchor compatibility only; not whole-cell ancestry or mass allocation",
        "p1_support_cell_row_method": "-1 sentinel for newly generated cells; full P1 cell/semantic/parent support is recomputed from the four authenticated current role-local node groups and source330 parent tuples",
        "registration": {"maximum_distance_m": max_distance_m,
                         "ambiguity_tolerance_m": ambiguity_tolerance_m,
                         "projection_same_point_tolerance_m": projection_tolerance_m,
                         "full86_same_weight_tolerance": full86_tolerance,
                         "global_nearest_claim": False},
        "child_capacity_profile": capacity_profile,
        "counts": {**counts, "child_tetrahedra": int(len(child["tets"])),
                   "exported_child_rows": int(len(selected_rows)),
                   "candidate_row_min": int(selected_rows.min()),
                   "candidate_row_max": int(selected_rows.max()),
                   "candidate_row_domain_sha256": sha256_file(requested_rows_path)},
        "inputs": {
            "candidate_child_node": {"path": str(candidate_node_path), "sha256": sha256_file(candidate_node_path)},
            "candidate_child_elements": {"path": str(candidate_ele_path), "sha256": sha256_file(candidate_ele_path)},
            "candidate_row_domain": {"path": str(requested_rows_path), "sha256": sha256_file(requested_rows_path)},
            **{key: inputs[key] for key in invariant_keys},
            "combined_node_map": inputs["combined_node_map"],
            "combined_handoff_receipt": inputs["combined_handoff_receipt"],
            "native_217_label_exporter": inputs["native_217_label_exporter"],
            "p1_anchor_217_exporter": inputs["p1_anchor_217_exporter"],
            "current_230_owner": {"path": owner_pin["path"], "sha256": owner_pin["sha256"]},
            "batched_projection_helper": project_pin,
        },
        "outputs": {key: {"path": str(path), "sha256": sha256_file(path)}
                    for key, path in output_paths.items()},
        "limitations": [
            "This is an incomplete changed-row source-label/P1-support bundle intended for exact stitching with authenticated unchanged 171 rows.",
            "Changed rows were derived from current node-group P1 support; no retired cell's scalar tuple is copied as new-cell ancestry.",
            "The merger must verify the request row set equals all unmatched candidate cells and must recheck 177 node registration against the complete stitched P1 incidence union.",
            "No mass allocation, native import, physical admission, or motion qualification was performed.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-171-receipt", required=True, type=Path)
    parser.add_argument("--candidate-node", required=True, type=Path)
    parser.add_argument("--candidate-ele", required=True, type=Path)
    parser.add_argument("--candidate-rows-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--maximum-distance-m", type=float, default=0.02)
    parser.add_argument("--ambiguity-tolerance-m", type=float, default=1.0e-8)
    parser.add_argument("--projection-tolerance-m", type=float, default=1.0e-12)
    parser.add_argument("--full86-tolerance", type=float, default=1.0e-10)
    args = parser.parse_args()
    receipt = regenerate_changed_rows(
        base_receipt_path=args.base_171_receipt, candidate_node_path=args.candidate_node,
        candidate_ele_path=args.candidate_ele, requested_rows_path=args.candidate_rows_csv,
        output_dir=args.output_dir, max_distance_m=args.maximum_distance_m,
        ambiguity_tolerance_m=args.ambiguity_tolerance_m,
        projection_tolerance_m=args.projection_tolerance_m,
        full86_tolerance=args.full86_tolerance)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
