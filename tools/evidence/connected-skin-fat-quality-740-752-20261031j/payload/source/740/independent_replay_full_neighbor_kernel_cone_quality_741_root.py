#!/usr/bin/env python3
"""Independent exact replay of the 740 fixed-boundary cavity batch (MILP and existing-vertex kernel cone).

This verifier does not import the 684 producer or its solver. It reconstructs
candidate connectivity and cell ancestry from the pinned input state and batch
journal, then independently checks local exact cavities, full target boundary,
duplicates, point identity, row maps, source metadata preservation, and whole
candidate native quality. It never admits physical state or assigns mass.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
QUALITY_PATH = BASE / "fixed_boundary_volume_flip_198_root.py"
EXPECTED_QUALITY_SHA = "650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"
EXPECTED_DRIVER_SHA = "1c40ede68b504e6e781c71fd4623211b9bef9418d208750ad24f5547930e4a1c"
EXPECTED_SOLVER_SHA = "5bca7a54c4bb1642be5153d6553a2ab009bbdd37186b208cc2c57083050a301d"
TOPOLOGY_PATH = BASE / "remesh_complete_surface_cavities_218_root.py"
EXPECTED_TOPOLOGY_SHA = "81e1128ca6f92693039d9ce53bea62e90b1d99f72bd9b9d997826b7ee2877b9a"
KERNEL_PATH = BASE / "quality-kernel-cone-734-root" / "kernel_cone_734.py"
EXPECTED_KERNEL_SHA = "d9a67e7bba636323196af75ac3eb31e4b2faba6258ec998104788a1ac463e2c5"
MUTABLE_OUTPUT_KEYS = {
    "tetrahedra", "material_indices", "boundary_faces", "desired_boundary_faces",
    "output_cell_to_input_cell", "input_cell_to_output_cell",
    "retained_input_cell_rows", "replacement_tetrahedra",
    "replacement_parent_offsets", "replacement_parent_cell_rows",
    "replacement_tet_offsets", "quality_milp_operation_parent_offsets",
    "quality_milp_operation_parent_cell_rows", "quality_milp_input_candidate_path",
    "quality_milp_input_candidate_sha256", "quality_milp_partial_bad_replacement",
    "quality_milp_operation_count", "quality_milp_operation_seed_rows",
    "quality_fixed_boundary_operation_count", "quality_fixed_boundary_operation_methods",
    "quality_fixed_boundary_operation_seed_rows", "quality_fixed_boundary_input_candidate_path",
    "quality_fixed_boundary_input_candidate_sha256", "quality_fixed_boundary_operation_parent_offsets",
    "quality_fixed_boundary_operation_parent_cell_rows",
    "quality_milp_input_node_to_output_node", "quality_milp_output_node_to_input_node",
    "current_geometry_parent_path", "current_geometry_parent_sha256",
    "canonical_parent_union_operation_order",
    # Immediate-parent point maps are mutable here because this step does not
    # move/add/remove points and must rewrite stale inherited aliases to identity.
    "input_node_to_output_node", "output_node_to_input_node",
    "output_point_to_input_point_row", "source_I_vertex_quotient",
    "source_input_vertex_quotient", "source_input_node_to_output_node",
    "input_point_to_output_point_row", "output_point_parent_offsets",
    "output_point_parent_input_cell_rows", "point_parent_offsets",
    "output_point_parent_cell_rows", "point_parent_rows",
}
PAT = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), np.int64)
QUALITY_GATE = 1.0e-5
VOLUME_GATE = 1.0e-18

def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()

def pin(path):
    return {"path": str(path), "sha256": sha(path)}

def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

M = imp("partial_milp_independent_quality198", QUALITY_PATH)
T = imp("partial_milp_independent_top218", TOPOLOGY_PATH)
KERNEL = imp("partial_milp_independent_kernel_cone734", KERNEL_PATH)
if sha(QUALITY_PATH) != EXPECTED_QUALITY_SHA:
    raise RuntimeError("pinned native-quality/exact owner SHA mismatch")
if sha(TOPOLOGY_PATH) != EXPECTED_TOPOLOGY_SHA:
    raise RuntimeError("pinned 218 topology owner SHA mismatch")
if sha(KERNEL_PATH) != EXPECTED_KERNEL_SHA:
    raise RuntimeError("pinned 734 kernel-cone helper SHA mismatch")

def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def parent_state(raw):
    return raw.get("current_verified_state_20261031", raw)

def target_faces(arrays):
    for key in ("desired_boundary_faces", "proposed_current_desired_boundary_faces",
                "boundary_faces"):
        if key in arrays:
            return np.asarray(arrays[key], np.int64), key
    raise KeyError("input has no current target boundary array")

def face_sign(face):
    a, b, c = map(int, face)
    inv = int(a > b) + int(a > c) + int(b > c)
    return 1 if inv % 2 == 0 else -1

def oriented_chain(faces):
    out = {}
    for f in np.asarray(faces, np.int64):
        key = tuple(sorted(map(int, f)))
        out[key] = out.get(key, 0) + face_sign(f)
        if out[key] == 0:
            del out[key]
    return out

def boundary_from_tets(tets):
    f = np.asarray(tets, np.int64)[:, PAT].reshape(-1, 3)
    keys = np.sort(f, axis=1)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    sk = keys[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(sk[1:] != sk[:-1], axis=1))]
    counts = np.diff(np.r_[starts, len(sk)])
    signs = np.asarray([face_sign(x) for x in f], np.int8)
    out_rows = []
    for st, count in zip(starts, counts):
        group = order[st:st + count]
        if count > 2:
            raise ValueError("tetrahedral face multiplicity exceeds two")
        if count == 2 and int(signs[group].sum()) != 0:
            raise ValueError("internal tetrahedral faces do not cancel orientation")
        if count == 1:
            out_rows.append(int(group[0]))
    return f[np.asarray(out_rows, np.int64)]

def exact_positive(points, tets):
    p = np.asarray(points, np.float32)
    xyz = M.integer_points(p)
    ds = [M.exact_det(xyz, tet) for tet in np.asarray(tets, np.int64)]
    return bool(ds) and min(ds) > 0, ds

def quality(points, tets):
    det, q = M.measurements(points, tets)
    ok = np.asarray(M.passes(det, q), bool)
    ratio = np.minimum(np.minimum(q / QUALITY_GATE,
                                 det / (6.0 * VOLUME_GATE)), 1.0)
    deficit = float(-np.log(np.maximum(ratio, 1.0e-300)).sum())
    return int((~ok).sum()), deficit, det, q

def same_array(a, b):
    return (np.asarray(a).dtype == np.asarray(b).dtype and
            np.asarray(a).shape == np.asarray(b).shape and
            np.asarray(a).tobytes(order="C") == np.asarray(b).tobytes(order="C"))

def oriented_face_key(face):
    return (tuple(sorted(map(int, face))), face_sign(face))

def boundary_row_map(input_boundary, output_boundary):
    lookup = {}
    for row, f in enumerate(np.asarray(input_boundary, np.int64)):
        k = oriented_face_key(f)
        if k in lookup:
            raise ValueError("duplicate oriented input boundary row")
        lookup[k] = row
    out_to_in = np.asarray([lookup.get(oriented_face_key(f), -1)
                            for f in np.asarray(output_boundary, np.int64)], np.int64)
    if len(out_to_in) != len(input_boundary) or np.any(out_to_in < 0) or len(np.unique(out_to_in)) != len(out_to_in):
        raise ValueError("boundary output rows do not biject to input target rows")
    in_to_out = np.full(len(input_boundary), -1, np.int64)
    in_to_out[out_to_in] = np.arange(len(out_to_in), dtype=np.int64)
    return out_to_in, in_to_out

CANONICAL_POINT_IDENTITY_KEYS = (
    "input_node_to_output_node", "output_node_to_input_node",
    "output_point_to_input_point_row", "source_I_vertex_quotient",
    "source_input_vertex_quotient", "source_input_node_to_output_node",
    "input_point_to_output_point_row", "quality_milp_input_node_to_output_node",
    "quality_milp_output_node_to_input_node",
)
CANONICAL_POINT_PARENT_OFFSETS_KEYS = ("output_point_parent_offsets", "point_parent_offsets")
CANONICAL_POINT_PARENT_ROWS_KEYS = (
    "output_point_parent_input_cell_rows", "output_point_parent_cell_rows", "point_parent_rows"
)
CANONICAL_POINT_MUTABLE_KEYS = frozenset(
    CANONICAL_POINT_IDENTITY_KEYS + CANONICAL_POINT_PARENT_OFFSETS_KEYS +
    CANONICAL_POINT_PARENT_ROWS_KEYS
)
_REQUIRED_CANONICAL_POINT_KEYS = {
    "input_node_to_output_node", "output_node_to_input_node",
    "output_point_to_input_point_row", "source_I_vertex_quotient",
    "output_point_parent_offsets", "output_point_parent_input_cell_rows",
}
if not _REQUIRED_CANONICAL_POINT_KEYS <= MUTABLE_OUTPUT_KEYS:
    raise RuntimeError("canonical point-map mutations are not declared in the independent mutable-key contract")

def verify_canonical_point_maps(candidate, npoints):
    identity = np.arange(int(npoints), dtype=np.int64)
    missing = _REQUIRED_CANONICAL_POINT_KEYS - set(candidate)
    if missing:
        raise ValueError("candidate omits required canonical point maps: " + ",".join(sorted(missing)))
    for key in CANONICAL_POINT_IDENTITY_KEYS:
        if key in candidate and not same_array(candidate[key], identity):
            raise ValueError(f"canonical unchanged-point map is not identity: {key}")
    for key in CANONICAL_POINT_PARENT_OFFSETS_KEYS:
        if key in candidate and not same_array(candidate[key], np.zeros(int(npoints)+1, np.int64)):
            raise ValueError(f"canonical unchanged-point parent offsets are not empty: {key}")
    for key in CANONICAL_POINT_PARENT_ROWS_KEYS:
        if key in candidate and np.asarray(candidate[key]).size != 0:
            raise ValueError(f"canonical unchanged-point parent rows are not empty: {key}")

def compare_nonmutable_parent_metadata(parent, candidate):
    for key, a in parent.items():
        if key in MUTABLE_OUTPUT_KEYS:
            continue
        if key not in candidate:
            raise ValueError("candidate dropped parent metadata array: " + key)
        if not same_array(a, candidate[key]):
            raise ValueError("candidate changed unrelated parent metadata array: " + key)

def verify(state_path, journal_path, role, map_path):
    state_path = Path(state_path)
    journal_path = Path(journal_path)
    st_pin = pin(state_path)
    journal_pin = pin(journal_path)
    raw_state = json.loads(state_path.read_text())
    state = parent_state(raw_state)
    journal = json.loads(journal_path.read_text())
    if journal.get("schema") != "h706.partial-positive-fixed-boundary-quality-kernel-cone-batch.v2":
        raise ValueError("unexpected fixed-boundary quality batch schema")
    if journal.get("source_state", {}).get("path") != str(state_path):
        raise ValueError("journal source-state path differs from requested input state")
    if journal.get("source_state", {}).get("sha256") != st_pin["sha256"]:
        raise ValueError("journal source-state SHA differs from requested input state")
    if journal.get("producer", {}).get("sha256") != EXPECTED_DRIVER_SHA:
        raise ValueError("journal does not name the frozen 712 full-neighbor driver")
    if journal.get("solver", {}).get("sha256") != EXPECTED_SOLVER_SHA:
        raise ValueError("journal does not name the frozen 684 solver")
    role_entries = [x for x in journal.get("role_results", []) if x.get("role") == role]
    if len(role_entries) != 1:
        raise ValueError("journal must have exactly one requested role result")
    jr = role_entries[0]
    if jr.get("status") != "PASS_BOUNDED_PARTIAL_FIXED_BOUNDARY_QUALITY_BATCH":
        raise ValueError("journal role does not contain an accepted fixed-boundary batch")
    state_key = role + "_current"
    if state_key not in state:
        raise ValueError("role is absent from input state")
    state_entry = state[state_key]
    src_path = Path(state_entry["path"])
    src_pin = pin(src_path)
    if src_pin["sha256"] != state_entry["sha256"]:
        raise ValueError("input state candidate SHA mismatch")
    if jr.get("input") != src_pin:
        raise ValueError("journal role input pin differs from state")
    if jr.get("state_input_sha256") != st_pin["sha256"]:
        raise ValueError("role journal was produced from a different state")
    if state_entry.get("nonpositive") != 0:
        raise ValueError("input state does not declare zero nonpositive cells")
    out_pin = jr.get("output", {})
    out_path = Path(out_pin.get("path", ""))
    if not out_path.is_file() or sha(out_path) != out_pin.get("sha256"):
        raise ValueError("journal output candidate path/SHA mismatch")
    source = npz(src_path)
    output = npz(out_path)
    p = np.asarray(source["points_world_m"])
    old_t = np.asarray(source["tetrahedra"], np.int64)
    old_m = np.asarray(source["material_indices"])
    new_t = np.asarray(output["tetrahedra"], np.int64)
    new_m = np.asarray(output["material_indices"])
    if p.dtype != np.float32 or not np.isfinite(p).all():
        raise ValueError("input point domain is not finite Float32")
    if output["points_world_m"].tobytes(order="C") != source["points_world_m"].tobytes(order="C"):
        raise ValueError("input/output point bytes differ")
    verify_canonical_point_maps(output, len(p))
    compare_nonmutable_parent_metadata(source, output)
    input_target, target_key = target_faces(source)
    input_boundary = boundary_from_tets(old_t)
    target_chain = oriented_chain(input_target)
    if oriented_chain(input_boundary) != target_chain:
        raise ValueError("input mesh is not on the exact current target boundary")
    if "boundary_faces" in source and oriented_chain(source["boundary_faces"]) != oriented_chain(input_boundary):
        raise ValueError("input stored boundary_faces disagree with actual input tets")
    output_boundary = boundary_from_tets(new_t)
    if oriented_chain(output_boundary) != target_chain:
        raise ValueError("output full oriented boundary differs from current target")
    if "desired_boundary_faces" not in output or not np.array_equal(
            np.asarray(output["desired_boundary_faces"], np.int64), input_target):
        raise ValueError("candidate changed current desired-face row identity/order")
    if "boundary_faces" not in output or oriented_chain(output["boundary_faces"]) != target_chain:
        raise ValueError("output stored boundary_faces disagree with output tetrahedra")
    out_bmap, in_bmap = boundary_row_map(input_target, output_boundary)

    operations = jr.get("accepted_operations", [])
    if not operations:
        raise ValueError("batch journal has no accepted cavity operations")
    all_old_rows = []
    replacement_blocks = []
    replacement_materials = []
    replacement_parent_rows = []
    replacement_parent_offsets = [0]
    replacement_tet_offsets = [0]
    op_parent_rows = []
    op_parent_offsets = [0]
    op_seeds = []
    local_results = []
    old_bad_total, old_deficit, old_det, old_q = quality(p, old_t)
    new_bad_total, new_deficit, new_det, new_q = quality(p, new_t)

    for oi, op in enumerate(operations):
        rows = np.asarray(op["old_cell_rows"], np.int64)
        block = np.asarray(op["new_tetrahedra"], np.int64)
        if rows.ndim != 1 or not len(rows) or len(np.unique(rows)) != len(rows):
            raise ValueError(f"operation {oi} has invalid old parent rows")
        if np.any(rows < 0) or np.any(rows >= len(old_t)):
            raise ValueError(f"operation {oi} parent cell row out of range")
        if set(map(int, rows)) & set(all_old_rows):
            raise ValueError(f"operation {oi} overlaps a prior retired cell")
        if block.ndim != 2 or block.shape[1] != 4 or not len(block):
            raise ValueError(f"operation {oi} has malformed new tetrahedra")
        if np.any(block < 0) or np.any(block >= len(p)):
            raise ValueError(f"operation {oi} new point row out of range")
        old = old_t[rows]
        material = np.asarray(old_m[rows])
        if len(np.unique(material)) != 1 or int(material[0]) != int(op["material_index"]):
            raise ValueError(f"operation {oi} changes or mixes material")
        old_pos, old_exact_det = exact_positive(p, old)
        new_pos, new_exact_det = exact_positive(p, block)
        if not old_pos or not new_pos:
            raise ValueError(f"operation {oi} has exact-nonpositive old/new cavity cells")
        ok, why = M.certify(p, old, block)
        if not ok:
            raise ValueError(f"operation {oi} exact cavity chain/volume certificate failed: {why}")
        before_bad = int((~M.passes(old_det[rows], old_q[rows])).sum())
        after_bad, after_deficit, new_det_local, new_q_local = quality(p, block)
        old_count, before_deficit, _, _ = quality(p, old)
        if int(op["bad_before"]) != before_bad or int(op["bad_before"]) != old_count:
            raise ValueError(f"operation {oi} journal old bad count mismatch")
        if int(op["bad_after"]) != after_bad:
            raise ValueError(f"operation {oi} journal new bad count mismatch")
        if after_bad > before_bad:
            raise ValueError(f"operation {oi} increases native-bad cell count")
        solver = op.get("solver_report", {})
        method = op.get("proposal_method", "milp")
        if method == "milp":
            if solver.get("stage1_badcount_optimum") != after_bad:
                raise ValueError(f"operation {oi} is inconsistent with recorded MILP primary optimum")
        elif method == "existing_vertex_kernel_cone":
            verts = np.unique(old)
            if not np.array_equal(verts, np.asarray(op["old_vertex_rows"], np.int64)):
                raise ValueError(f"operation {oi} kernel-cone vertex map is not canonical")
            local_map = np.full(len(p), -1, np.int64)
            local_map[verts] = np.arange(len(verts), dtype=np.int64)
            old_local = local_map[old]
            local_boundary = T.m.boundary(old)
            lb_local = local_map[local_boundary]
            cone = KERNEL.propose_existing_vertex_kernel_cone(
                p[verts], lb_local, old_local, material, merit_policy="strict_lex")
            if cone.get("status") != "PASS_EXISTING_VERTEX_KERNEL_CONE_BEST":
                raise ValueError(f"operation {oi} kernel-cone proposal does not independently reproduce: {cone.get('status')}")
            expected_block = verts[np.asarray(cone["replacement_tetrahedra"], np.int64)]
            if not np.array_equal(expected_block, block):
                raise ValueError(f"operation {oi} kernel-cone tetrahedra differ from independent helper replay")
            if int(cone["apex_vertex"]) != int(solver.get("apex_vertex_local", -1)):
                raise ValueError(f"operation {oi} kernel-cone apex differs from independent replay")
            if int(verts[int(cone["apex_vertex"])]) != int(solver.get("apex_vertex_global", -1)):
                raise ValueError(f"operation {oi} kernel-cone global apex map mismatch")
            if solver.get("kernel_helper_sha256") != EXPECTED_KERNEL_SHA:
                raise ValueError(f"operation {oi} kernel helper pin mismatch")
            if solver.get("status") != cone.get("status") or solver.get("merit_policy") != "strict_lex":
                raise ValueError(f"operation {oi} kernel-cone status or merit-policy mismatch")
            if not bool(solver.get("exact_chain_matches_input")):
                raise ValueError(f"operation {oi} kernel-cone exact-chain receipt missing")
            if int(solver.get("old_bad", -1)) != int(cone["old_bad"]):
                raise ValueError(f"operation {oi} kernel-cone old bad count mismatch")
            if int(cone["new_bad"]) != after_bad or int(solver.get("new_bad", -1)) != after_bad:
                raise ValueError(f"operation {oi} kernel-cone bad count mismatch")
            if abs(float(solver.get("old_log_deficit", float("nan"))) - float(cone["old_log_deficit"])) > 1e-12:
                raise ValueError(f"operation {oi} kernel-cone old deficit mismatch")
            if abs(float(solver.get("new_log_deficit", float("nan"))) - float(cone["new_log_deficit"])) > 1e-12:
                raise ValueError(f"operation {oi} kernel-cone new deficit mismatch")
        else:
            raise ValueError(f"operation {oi} has unknown proposal method {method!r}")
        if bool(op.get("partial_bad_replacement")) != bool(after_bad > 0):
            raise ValueError(f"operation {oi} partial-bad flag mismatch")
        if len(block) != int(op["new_cell_count"]):
            raise ValueError(f"operation {oi} new cell count mismatch")
        if before_bad - after_bad <= 0 and before_deficit - after_deficit <= max(1.0e-8, 1.0e-6 * max(before_deficit, 1.0)):
            raise ValueError(f"operation {oi} has no strict count or tie-break gain")
        if set(map(int, np.unique(old))) != set(map(int, op["old_vertex_rows"])):
            raise ValueError(f"operation {oi} cavity point ancestry set mismatch")
        all_old_rows.extend(map(int, rows))
        replacement_blocks.append(block)
        replacement_materials.append(np.full(len(block), int(material[0]), old_m.dtype))
        replacement_tet_offsets.append(replacement_tet_offsets[-1] + len(block))
        for _ in block:
            replacement_parent_rows.extend(map(int, rows))
            replacement_parent_offsets.append(len(replacement_parent_rows))
        op_parent_rows.extend(map(int, rows))
        op_parent_offsets.append(len(op_parent_rows))
        op_seeds.append(int(op["seed_cell_row"]))
        local_results.append({
            "operation": oi, "old_cell_rows": rows.tolist(),
            "old_cells": int(len(rows)), "new_cells": int(len(block)),
            "old_native_bad": int(before_bad), "new_native_bad": int(after_bad),
            "old_log_deficit": float(before_deficit),
            "new_log_deficit": float(after_deficit),
            "old_exact_positive": bool(old_pos), "new_exact_positive": bool(new_pos),
            "exact_cavity_certificate": why,
            "sum_old_exact_det6": str(sum(old_exact_det)),
            "sum_new_exact_det6": str(sum(new_exact_det)),
        })
    if len(set(all_old_rows)) != len(all_old_rows):
        raise ValueError("duplicate retired input cell row")
    retired = np.asarray(sorted(all_old_rows), np.int64)
    keep = np.ones(len(old_t), bool)
    keep[retired] = False
    retained = np.flatnonzero(keep)
    replacement = np.vstack(replacement_blocks)
    replacement_mat = np.concatenate(replacement_materials)
    expected_t = np.vstack((old_t[keep], replacement))
    expected_m = np.concatenate((old_m[keep], replacement_mat))
    if not np.array_equal(expected_t, new_t):
        raise ValueError("full output connectivity does not replay from journal")
    if not np.array_equal(expected_m, new_m):
        raise ValueError("full output material array does not replay from journal")
    if not same_array(replacement, output["replacement_tetrahedra"]):
        raise ValueError("replacement_tetrahedra array differs from journal")
    if not np.array_equal(output["retained_input_cell_rows"], retained):
        raise ValueError("retained input cell row list mismatch")
    expected_o2i = np.concatenate((retained, np.full(len(replacement), -1, np.int64)))
    expected_i2o = np.full(len(old_t), -1, np.int64)
    expected_i2o[retained] = np.arange(len(retained), dtype=np.int64)
    if not np.array_equal(output["output_cell_to_input_cell"], expected_o2i):
        raise ValueError("output-to-input cell map mismatch")
    if not np.array_equal(output["input_cell_to_output_cell"], expected_i2o):
        raise ValueError("input-to-output cell map mismatch")
    if not np.array_equal(output["replacement_tet_offsets"], replacement_tet_offsets):
        raise ValueError("replacement tet operation offsets mismatch")
    if not np.array_equal(output["replacement_parent_offsets"], replacement_parent_offsets):
        raise ValueError("per-new-cell parent CSR offsets mismatch")
    if not np.array_equal(output["replacement_parent_cell_rows"], replacement_parent_rows):
        raise ValueError("per-new-cell complete cavity parent CSR mismatch")
    methods = [op.get("proposal_method", "milp") for op in operations]
    milp_indices = [i for i, method in enumerate(methods) if method == "milp"]
    milp_parent_rows = [r for i in milp_indices for r in operations[i]["old_cell_rows"]]
    milp_parent_offsets = [0]
    for i in milp_indices:
        milp_parent_offsets.append(milp_parent_offsets[-1] + len(operations[i]["old_cell_rows"]))
    if int(np.asarray(output["quality_fixed_boundary_operation_count"]).ravel()[0]) != len(operations):
        raise ValueError("fixed-boundary operation count metadata mismatch")
    if list(map(str, output["quality_fixed_boundary_operation_methods"].tolist())) != methods:
        raise ValueError("fixed-boundary operation methods metadata mismatch")
    if not np.array_equal(output["quality_fixed_boundary_operation_seed_rows"], op_seeds):
        raise ValueError("fixed-boundary operation seed metadata mismatch")
    if not np.array_equal(output["quality_fixed_boundary_operation_parent_offsets"], op_parent_offsets):
        raise ValueError("fixed-boundary operation parent offsets mismatch")
    if not np.array_equal(output["quality_fixed_boundary_operation_parent_cell_rows"], op_parent_rows):
        raise ValueError("fixed-boundary operation parent rows mismatch")
    if int(np.asarray(output["quality_milp_operation_count"]).ravel()[0]) != len(milp_indices):
        raise ValueError("MILP-only operation count metadata mismatch")
    if not np.array_equal(output["quality_milp_operation_seed_rows"], [op_seeds[i] for i in milp_indices]):
        raise ValueError("MILP-only operation seed row metadata mismatch")
    if not np.array_equal(output["quality_milp_operation_parent_offsets"], milp_parent_offsets):
        raise ValueError("MILP-only operation parent offsets mismatch")
    if not np.array_equal(output["quality_milp_operation_parent_cell_rows"], milp_parent_rows):
        raise ValueError("MILP-only operation parent rows mismatch")
    if str(np.asarray(output["quality_fixed_boundary_input_candidate_path"]).ravel()[0]) != str(src_path):
        raise ValueError("generic fixed-boundary source path mismatch")
    if str(np.asarray(output["quality_fixed_boundary_input_candidate_sha256"]).ravel()[0]) != src_pin["sha256"]:
        raise ValueError("generic fixed-boundary source SHA mismatch")
    identity = np.arange(len(p), dtype=np.int64)
    if not np.array_equal(output["quality_milp_input_node_to_output_node"], identity):
        raise ValueError("input point map is not identity")
    if not np.array_equal(output["quality_milp_output_node_to_input_node"], identity):
        raise ValueError("output point map is not identity")
    for path_key, sha_key in (("quality_milp_input_candidate_path", "quality_milp_input_candidate_sha256"),
                              ("current_geometry_parent_path", "current_geometry_parent_sha256")):
        if str(np.asarray(output[path_key]).ravel()[0]) != str(src_path):
            raise ValueError(f"{path_key} does not name exact input")
        if str(np.asarray(output[sha_key]).ravel()[0]) != src_pin["sha256"]:
            raise ValueError(f"{sha_key} does not bind exact input")
    if len(np.unique(np.sort(new_t, axis=1), axis=0)) != len(new_t):
        raise ValueError("full output contains duplicate tetrahedra")
    if np.any(new_det <= 0) or not np.isfinite(new_det).all():
        raise ValueError("full output has nonpositive measured determinants")
    actual_bad = int((~M.passes(new_det, new_q)).sum())
    if actual_bad != new_bad_total:
        raise ValueError("whole-candidate native quality count mismatch")
    if jr.get("native_bad_before") != old_bad_total or jr.get("native_bad_after") != new_bad_total:
        raise ValueError("journal whole-state native quality count mismatch")
    if jr.get("input_cells") != len(old_t) or jr.get("output_cells") != len(new_t):
        raise ValueError("journal whole-state cell count mismatch")
    if not bool(jr.get("all_replacement_cells_exact_positive")) or int(jr.get("exact_nonpositive_new_cells", -1)) != 0:
        raise ValueError("journal does not claim its exact replacement sign gate")
    map_path = Path(map_path)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(map_path,
                        output_boundary_face_to_input_boundary_face=out_bmap,
                        input_boundary_face_to_output_boundary_face=in_bmap)
    report = {
        "schema": "h706.independent-fixed-boundary-quality-volume-replay.v1",
        "status": "PASS_INDEPENDENT_FIXED_BOUNDARY_QUALITY_REPLAY",
        "role": role,
        "input_state": st_pin, "batch_journal": journal_pin,
        "input_candidate": src_pin, "output_candidate": pin(out_path),
        "quality_owner": pin(QUALITY_PATH),
        "producer_sha256_checked": EXPECTED_DRIVER_SHA,
        "solver_sha256_checked": EXPECTED_SOLVER_SHA,
        "topology_owner_sha256_checked": EXPECTED_TOPOLOGY_SHA,
        "kernel_helper_sha256_checked": EXPECTED_KERNEL_SHA,
        "proposal_method_counts": {method: methods.count(method) for method in sorted(set(methods))},
        "operations_replayed": len(operations),
        "retired_input_cells": int(len(retired)),
        "replacement_cells": int(len(replacement)),
        "input_cells": int(len(old_t)), "output_cells": int(len(new_t)),
        "input_points": int(len(p)), "output_points": int(len(output["points_world_m"])),
        "point_coordinates_byte_identical": True,
        "canonical_point_input_output_maps_identity": True,
        "canonical_point_parent_csr_empty": True,
        "mutable_key_contract_checked": True,
        "per_operation_exact_replay": local_results,
        "whole_input_native_bad_cells": int(old_bad_total),
        "whole_output_native_bad_cells": int(new_bad_total),
        "whole_output_measured_positive_cells": int(len(new_t)),
        "replacement_exact_positive_cells": int(len(replacement)),
        "global_oriented_target_boundary_faces": int(len(input_target)),
        "global_oriented_target_boundary_exact": True,
        "duplicate_tetrahedra": 0,
        "full_parent_CSR_exact": True,
        "source_metadata_unchanged_outside_declared_output_fields": True,
        "boundary_face_row_map": pin(map_path),
        "boundary_face_order_changed": not np.array_equal(out_bmap, np.arange(len(out_bmap))),
        "boundary_face_identity_metadata_status": "OUTPUT_TO_TARGET_ROW_MAP_EMITTED; METADATA_FINALIZER_PENDING",
        "native_thresholds_unchanged": True,
        "native_admission": False, "physical_admission": False, "native_steps": 0,
        "mass_mapping": "NOT_RUN",
    }
    return report

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--journal", type=Path, required=True)
    ap.add_argument("--role", choices=("Skin", "Fat"), required=True)
    ap.add_argument("--output", type=Path, required=True,
                    help="independent replay receipt JSON path")
    args = ap.parse_args()
    out = Path(args.output)
    row_map_path = out.with_name(out.stem + "-boundary-row-map.npz")
    try:
        receipt = verify(args.state, args.journal, args.role, row_map_path)
        receipt["receipt_path"] = str(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2))
    except Exception as exc:
        failure = {
            "schema": "h706.independent-fixed-boundary-quality-volume-replay.v1",
            "status": "FAIL_INDEPENDENT_FIXED_BOUNDARY_QUALITY_REPLAY",
            "role": args.role, "error": repr(exc),
            "input_state": str(args.state), "batch_journal": str(args.journal),
            "physical_admission": False, "native_steps": 0,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(failure, indent=2) + "\n")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        raise SystemExit(2)

if __name__ == "__main__":
    main()
