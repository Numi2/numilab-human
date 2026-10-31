#!/usr/bin/env python3
"""Balanced exact-boundary quality cavities with partial native-bad replacement.

Uses the pinned 218 boundary/manifold owner, tries the pinned 734 existing-vertex
kernel-cone proposal before the 684 exact MILP fallback, and retains the same
strict native lexicographic quality policy and all 724 exact/global gates.
This derivative preserves the unchanged point domain and canonical maps.
Construction proposals only; no mass assignment or physical admission occurs.
"""
from pathlib import Path
from collections import Counter, defaultdict, deque
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
import inspect
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
TOP_PATH = BASE / "remesh_complete_surface_cavities_218_root.py"
SOLVER_PATH = HERE / "solve_partial_oriented_cavity_milp_684.py"

def imp(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

T = imp("partial_quality_top218", TOP_PATH)
S = imp("partial_quality_solver684", SOLVER_PATH)
M = S.M
KERNEL_PATH = BASE / "quality-kernel-cone-734-root" / "kernel_cone_734.py"
EXPECTED_KERNEL_SHA256 = "d9a67e7bba636323196af75ac3eb31e4b2faba6258ec998104788a1ac463e2c5"
if hashlib.sha256(KERNEL_PATH.read_bytes()).hexdigest() != EXPECTED_KERNEL_SHA256:
    raise RuntimeError("pinned 734 kernel-cone helper SHA mismatch")
KERNEL = imp("quality_kernel_cone_734", KERNEL_PATH)
EXPECTED_QUALITY_OWNER_PATH = Path(T.m.m.__file__).resolve()
EXPECTED_QUALITY_OWNER_SHA256 = "650794951cbcf338de9cde691a8f43377b9d023ccea78b876e87c847880243e0"
if hashlib.sha256(EXPECTED_QUALITY_OWNER_PATH.read_bytes()).hexdigest() != EXPECTED_QUALITY_OWNER_SHA256:
    raise RuntimeError("pinned M.certify owner SHA mismatch")
if Path(inspect.getsourcefile(M.certify)).resolve() != EXPECTED_QUALITY_OWNER_PATH or \
        Path(inspect.getsourcefile(KERNEL.M.certify)).resolve() != EXPECTED_QUALITY_OWNER_PATH:
    raise RuntimeError("kernel and 740 must use the same pinned M.certify owner")
PAT = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), np.int64)
QUALITY_GATE = 1.0e-5
VOLUME_GATE = 1.0e-18

# These are immediate-parent maps for this no-point-motion operation. The
# parent artifact remains pinned, so historical quotient lineage stays there;
# this step's canonical maps are identity and its point-parent CSR is empty.
CANONICAL_POINT_IDENTITY_KEYS = (
    "input_node_to_output_node",
    "output_node_to_input_node",
    "output_point_to_input_point_row",
    "source_I_vertex_quotient",
    "source_input_vertex_quotient",
    "source_input_node_to_output_node",
    "input_point_to_output_point_row",
    "quality_milp_input_node_to_output_node",
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

def unchanged_point_lineage_updates(parent_arrays, npoints):
    """Return immediate-parent point maps for an unchanged point namespace."""
    identity = np.arange(int(npoints), dtype=np.int64)
    updates = {}
    for key in CANONICAL_POINT_IDENTITY_KEYS:
        if key in parent_arrays or key in {
            "input_node_to_output_node", "output_node_to_input_node",
            "output_point_to_input_point_row", "source_I_vertex_quotient",
        }:
            updates[key] = identity.copy()
    for key in CANONICAL_POINT_PARENT_OFFSETS_KEYS:
        if key in parent_arrays or key == "output_point_parent_offsets":
            updates[key] = np.zeros(int(npoints) + 1, dtype=np.int64)
    for key in CANONICAL_POINT_PARENT_ROWS_KEYS:
        if key in parent_arrays or key == "output_point_parent_input_cell_rows":
            updates[key] = np.empty(0, dtype=np.int64)
    return updates

def assert_unchanged_point_lineage(arrays, npoints):
    identity = np.arange(int(npoints), dtype=np.int64)
    required = {
        "input_node_to_output_node", "output_node_to_input_node",
        "output_point_to_input_point_row", "source_I_vertex_quotient",
        "output_point_parent_offsets", "output_point_parent_input_cell_rows",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError("canonical point lineage fields missing: " + ",".join(sorted(missing)))
    for key in CANONICAL_POINT_IDENTITY_KEYS:
        if key in arrays and not np.array_equal(np.asarray(arrays[key]), identity):
            raise ValueError(f"unchanged point map is not identity: {key}")
    for key in CANONICAL_POINT_PARENT_OFFSETS_KEYS:
        if key in arrays and not np.array_equal(np.asarray(arrays[key]), np.zeros(int(npoints)+1, np.int64)):
            raise ValueError(f"unchanged-point parent CSR is not empty: {key}")
    for key in CANONICAL_POINT_PARENT_ROWS_KEYS:
        if key in arrays and np.asarray(arrays[key]).size != 0:
            raise ValueError(f"unchanged-point parent rows are not empty: {key}")

def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()

def pin(path):
    return {"path": str(path), "sha256": sha(path)}

def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def boundary_target(arrays):
    for key in ("desired_boundary_faces", "proposed_current_desired_boundary_faces",
                "proposed_current_desired_boundary_faces_rows", "boundary_faces"):
        if key in arrays:
            return np.asarray(arrays[key], np.int64), key
    raise KeyError("candidate has no recognized desired/current boundary faces")

def quality_score(points, tets):
    det, q = M.measurements(points, tets)
    ok = np.asarray(M.passes(det, q), bool)
    ratio = np.minimum(np.minimum(q / QUALITY_GATE, det / (6.0 * VOLUME_GATE)), 1.0)
    deficit = float(-np.log(np.maximum(ratio, 1.0e-300)).sum())
    return int((~ok).sum()), deficit, det, q

def exact_positive(points, tets):
    xyz = [S.S.ip(x) for x in np.asarray(points, np.float32)]
    values = [S.S.det(*(xyz[int(v)] for v in tet)) for tet in np.asarray(tets, np.int64)]
    return bool(values) and min(values) > 0

class CompactFaceAdjacency:
    def __init__(self, rows): self.rows=rows
    def get(self, row, default=()):
        values=self.rows[int(row)]
        return values[values>=0]

def face_bad_adjacency(tets, materials, bad):
    # Include passing neighbors: fixed-boundary repair often needs an adjacent
    # good tetrahedron to remove an isolated sliver. Compact four-face CSR.
    ncell=len(tets)
    faces=np.sort(np.asarray(tets,np.int64)[:,PAT].reshape(-1,3),axis=1)
    order=np.lexsort((faces[:,2],faces[:,1],faces[:,0]));f=faces[order]
    starts=np.r_[0,1+np.flatnonzero(np.any(f[1:]!=f[:-1],axis=1))]
    counts=np.diff(np.r_[starts,len(f)]);starts=starts[counts==2]
    one=order[starts];two=order[starts+1];a=one//4;b=two//4
    same=np.asarray(materials)[a]==np.asarray(materials)[b]
    one=one[same];two=two[same];a=a[same];b=b[same]
    rows=np.full((ncell,4),-1,np.int32)
    rows[a,one%4]=b;rows[b,two%4]=a
    return CompactFaceAdjacency(rows)

def grow_bad_cluster(seed, adjacency, bad, materials, vertices_by_cell,
                     max_cells, max_vertices, reserved_cells, reserved_vertices,
                     deficit_by_cell):
    if seed in reserved_cells or materials[seed] is None:
        return None
    mat = materials[seed]
    chosen = [int(seed)]
    usedv = set(map(int, vertices_by_cell[seed]))
    if usedv & reserved_vertices or len(usedv) > max_vertices:
        return None
    queued = set()
    frontier = [int(seed)]
    badset = set(map(int, bad))
    while frontier and len(chosen) < max_cells:
        current = frontier.pop(0)
        for nb in adjacency.get(current, ()):
            if nb in queued or nb in chosen:
                continue
            queued.add(nb)
            if nb in reserved_cells or materials[nb] != mat:
                continue
            union = usedv | set(map(int, vertices_by_cell[nb]))
            if len(union) > max_vertices or union & reserved_vertices:
                continue
            chosen.append(nb)
            usedv = union
            frontier.append(nb)
            if len(chosen) >= max_cells:
                break
        frontier.sort(key=lambda r: (-float(deficit_by_cell[r]), r))
    return np.asarray(sorted(chosen), np.int64)

def _archive_conflicting_parent_fields(payload, updates):
    for key in updates:
        if key in payload:
            prior = "quality_milp_parent_" + key
            if prior in payload:
                prior = "quality_milp_parent_round2_" + key
            payload[prior] = payload[key]

def run_role(role, entry, state, outdir, deadline, args):
    role_started = time.monotonic()
    source_path = Path(entry["path"])
    source_sha = sha(source_path)
    if source_sha != entry["sha256"]:
        raise ValueError(f"{role} input SHA mismatch")
    a = load_npz(source_path)
    p = np.asarray(a["points_world_m"])
    t = np.asarray(a["tetrahedra"], np.int64)
    mat = np.asarray(a["material_indices"])
    if p.dtype != np.float32 or p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError(f"{role} requires finite F32 volume points")
    if t.ndim != 2 or t.shape[1] != 4 or mat.shape != (len(t),):
        raise ValueError(f"{role} malformed tetrahedron/material arrays")
    target, target_key = boundary_target(a)
    actual_boundary = T.m.boundary(t)
    if not np.array_equal(T.m.boundary_identity(actual_boundary),
                          T.m.boundary_identity(target)):
        raise ValueError(f"{role} input boundary does not match {target_key}")
    old_bad, old_deficit, old_det, old_q = quality_score(p, t)
    declared_bad = entry.get("native_quality_failures")
    if declared_bad is not None and int(declared_bad) != old_bad:
        raise ValueError(f"{role} pinned native-bad count differs from measured input")
    if np.any(old_det <= 0):
        raise ValueError(f"{role} input contains nonpositive tetrahedra")
    declared_nonpositive = entry.get("nonpositive", a.get("nonpositive", None))
    if declared_nonpositive is not None and int(declared_nonpositive) != 0:
        raise ValueError(f"{role} state declares nonpositive source cells")
    # A construction proposal inherits the source's exact-positive claim.
    # Every replaced old and every generated new cell is checked exactly below.
    if declared_nonpositive is None:
        raise ValueError(f"{role} state must pin nonpositive=0 source evidence")
    bad = np.flatnonzero(~M.passes(old_det, old_q))
    if not len(bad):
        return {"role": role, "status": "NO_NATIVE_BAD_CELLS",
                "input": pin(source_path), "bad_before": 0, "bad_after": 0}
    deficits = -np.log(np.maximum(np.minimum(np.minimum(old_q / QUALITY_GATE,
                                    old_det / (6.0 * VOLUME_GATE)), 1.0), 1.0e-300))
    adj = face_bad_adjacency(t, mat, bad)
    vertices_by_cell = t
    seeds = sorted(map(int, bad), key=lambda r: (-float(deficits[r]), r))
    reserved_cells = set()
    reserved_vertices = set()
    attempted = set()
    ops = []
    rejection = Counter()
    trial_rows = []
    # Cache cone proposals by this immutable role input's exact old-cell cluster.
    kernel_cone_cache = {}
    kernel_cone_stats = Counter()
    boundary_before = T.m.boundary_identity(actual_boundary)
    source_float32_bad = old_bad
    for seed in seeds:
        now = time.monotonic()
        if now >= deadline or len(ops) >= args.maximum_operations or len(trial_rows) >= args.maximum_attempts:
            break
        if seed in reserved_cells:
            continue
        rows = grow_bad_cluster(seed, adj, bad, mat, vertices_by_cell,
                                args.maximum_old_cells, args.maximum_points,
                                reserved_cells, reserved_vertices, deficits)
        if rows is None or not len(rows):
            rejection["protected_or_vertex_limit"] += 1
            continue
        key = tuple(sorted(map(int, rows)))
        if key in attempted:
            if key in kernel_cone_cache:
                kernel_cone_stats["cache_hits"] += 1
            continue
        attempted.add(key)
        old = t[rows]
        verts = np.unique(old)
        if len(verts) > args.maximum_points:
            rejection["vertex_limit"] += 1
            continue
        if len(np.unique(mat[rows])) != 1:
            rejection["mixed_material"] += 1
            continue
        local_boundary = T.m.boundary(old)
        closed, open_edges = T.closed(local_boundary)
        topology = T.manifold_topology(local_boundary)
        if (not closed or topology["bad_vertex_links"] or
                topology["boundary_components"] != 1):
            rejection["nonmanifold_cavity"] += 1
            continue
        if not exact_positive(p, old):
            rejection["old_cavity_not_exact_positive"] += 1
            continue
        local_map = np.full(len(p), -1, np.int64)
        local_map[verts] = np.arange(len(verts), dtype=np.int64)
        lp = p[verts]
        lb = local_map[local_boundary]
        cone = kernel_cone_cache.get(key)
        if cone is None:
            try:
                old_local = local_map[old]
                cone = KERNEL.propose_existing_vertex_kernel_cone(
                    lp, lb, old_local, mat[rows], merit_policy="strict_lex")
            except Exception as exc:
                cone = {"status": "KERNEL_CONE_EXCEPTION", "error": repr(exc),
                        "fallback": "existing cavity MILP/solver"}
            kernel_cone_cache[key] = cone
            kernel_cone_stats["evaluations"] += 1
        else:
            kernel_cone_stats["cache_hits"] += 1
        cone_summary = {k: cone[k] for k in (
            "status", "merit_policy", "apex_vertex", "valid_apex_count",
            "boundary_vertex_count", "rejections", "fallback", "reason", "error",
            "old_bad", "new_bad", "old_log_deficit", "new_log_deficit",
            "log_deficit_gain", "exact_det6_delta_integer",
            "exact_chain_matches_input", "no_duplicate_tetrahedra",
            "manifold_face_incidence") if k in cone}
        local_new = None
        report = None
        candidate_method = "milp"
        if cone.get("status") == "PASS_EXISTING_VERTEX_KERNEL_CONE_BEST":
            cone_local = np.asarray(cone["replacement_tetrahedra"], np.int64)
            cone_global = verts[cone_local]
            if not exact_positive(p, cone_global):
                cone_reject = "new_cell_not_exact_positive"
            else:
                cone_ok, cone_why = M.certify(p, old, cone_global)
                cone_before_bad, cone_before_score = quality_score(p, old)[:2]
                cone_after_bad, cone_after_score = quality_score(p, cone_global)[:2]
                cone_floor = max(1.0e-8, 1.0e-6 * max(cone_before_score, 1.0))
                if not cone_ok:
                    cone_reject = "exact_cavity_certificate_" + str(cone_why)
                elif cone_after_bad > cone_before_bad:
                    cone_reject = "bad_count_increase"
                elif cone_after_bad == cone_before_bad and cone_before_score-cone_after_score <= cone_floor:
                    cone_reject = "no_lexicographic_quality_gain"
                else:
                    cone_reject = None
                    local_new = cone_local
                    candidate_method = "existing_vertex_kernel_cone"
                    report = {
                        "proposal_method": candidate_method,
                        "status": cone["status"],
                        "merit_policy": "strict_lex",
                        "apex_vertex_local": int(cone["apex_vertex"]),
                        "apex_vertex_global": int(verts[int(cone["apex_vertex"])]),
                        "old_bad": int(cone_before_bad), "new_bad": int(cone_after_bad),
                        "old_log_deficit": float(cone_before_score),
                        "new_log_deficit": float(cone_after_score),
                        "log_deficit_gain": float(cone_before_score-cone_after_score),
                        "exact_det6_delta_integer": str(cone["exact_det6_delta_integer"]),
                        "exact_chain_matches_input": True,
                        "kernel_helper_path": str(KERNEL_PATH),
                        "kernel_helper_sha256": EXPECTED_KERNEL_SHA256,
                        "quality_owner_path": str(EXPECTED_QUALITY_OWNER_PATH),
                        "quality_owner_sha256": EXPECTED_QUALITY_OWNER_SHA256,
                        "exact_cavity_certificate": cone_why,
                    }
                    kernel_cone_stats["accepted"] += 1
            if cone_reject is not None:
                rejection["kernel_cone_policy_" + cone_reject] += 1
                kernel_cone_stats["milp_fallbacks"] += 1
        else:
            rejection["kernel_cone_" + str(cone.get("status", "NO_RESULT"))] += 1
            kernel_cone_stats["milp_fallbacks"] += 1
        if local_new is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.05:
                trial_rows.append({
                    "seed_cell_row": int(seed), "old_cell_rows": rows.tolist(),
                    "old_vertex_rows": verts.tolist(), "old_cells": int(len(rows)),
                    "old_bad": int(np.count_nonzero(~M.passes(old_det[rows], old_q[rows]))),
                    "old_deficit": float(deficits[rows].sum()), "open_edges": int(open_edges),
                    "proposal_method": "no_solver_time_after_kernel_prepass",
                    "kernel_cone": cone_summary, "proposal_report": report,
                })
                rejection["time_budget_after_kernel_prepass"] += 1
                break
            solve_seconds = min(args.solve_seconds, max(0.05, remaining))
            report, local_new = S.solve(lp, lb, seconds=solve_seconds,
                                        max_points=args.maximum_points,
                                        material=int(mat[rows[0]]))
            candidate_method = "milp"
        trial = {"seed_cell_row": int(seed), "old_cell_rows": rows.tolist(),
                 "old_vertex_rows": verts.tolist(), "old_cells": int(len(rows)),
                 "old_bad": int(np.count_nonzero(~M.passes(old_det[rows], old_q[rows]))),
                 "old_deficit": float(deficits[rows].sum()), "open_edges": int(open_edges),
                 "proposal_method": candidate_method, "kernel_cone": cone_summary,
                 "proposal_report": report}
        trial_rows.append(trial)
        if local_new is None:
            rejection[str(report.get("status", "MILP_NO_SOLUTION"))] += 1
            continue
        new = verts[np.asarray(local_new, np.int64)]
        if not exact_positive(p, new):
            rejection["new_cell_not_exact_positive"] += 1
            continue
        ok, why = M.certify(p, old, new)
        if not ok:
            rejection["exact_cavity_certificate_" + str(why)] += 1
            continue
        before_bad, before_score = quality_score(p, old)[:2]
        after_bad, after_score = quality_score(p, new)[:2]
        improvement_floor = max(1.0e-8, 1.0e-6 * max(before_score, 1.0))
        if after_bad > before_bad:
            rejection["bad_count_increase"] += 1
            continue
        if after_bad == before_bad and before_score - after_score <= improvement_floor:
            rejection["no_lexicographic_quality_gain"] += 1
            continue
        record = {
            "proposal_method": candidate_method,
            "seed_cell_row": int(seed), "old_cell_rows": rows.tolist(),
            "old_vertex_rows": verts.tolist(), "new_tetrahedra": new.tolist(),
            "material_index": int(mat[rows[0]]), "old_cell_count": int(len(old)),
            "new_cell_count": int(len(new)), "bad_before": int(before_bad),
            "bad_after": int(after_bad), "log_deficit_before": float(before_score),
            "log_deficit_after": float(after_score), "exact_cavity_certificate": why,
            "exact_positive_new_cells": True,
            "partial_bad_replacement": bool(after_bad > 0),
            "solver_report": report,
        }
        ops.append(record)
        reserved_cells.update(map(int, rows))
        reserved_vertices.update(map(int, verts))
    if not ops:
        return {"role": role, "status": "NO_VALID_BOUNDED_PARTIAL_FIXED_BOUNDARY_QUALITY_PROGRESS",
                "input": pin(source_path), "input_cells": int(len(t)),
                "native_bad_before": int(old_bad), "native_bad_after": int(old_bad),
                "attempts": trial_rows, "rejections": dict(rejection),
                "kernel_cone_cache": dict(kernel_cone_stats),
                "kernel_cone_cache_entries": len(kernel_cone_cache),
                "elapsed_s": time.monotonic() - role_started}
    removed = np.asarray(sorted({r for op in ops for r in op["old_cell_rows"]}), np.int64)
    keep = np.ones(len(t), bool)
    keep[removed] = False
    retained = np.flatnonzero(keep)
    new_blocks = [np.asarray(op["new_tetrahedra"], np.int64) for op in ops]
    replacement = np.vstack(new_blocks)
    replacement_material = np.concatenate([
        np.full(len(block), int(op["material_index"]), mat.dtype)
        for op, block in zip(ops, new_blocks)
    ])
    out_t = np.vstack((t[keep], replacement))
    out_mat = np.concatenate((mat[keep], replacement_material))
    if p.tobytes(order="C") != np.asarray(a["points_world_m"]).tobytes(order="C"):
        raise AssertionError("point prefix unexpectedly changed")
    det2, q2 = M.measurements(p, out_t)
    if np.any(det2 <= 0):
        raise ValueError(f"{role} output has nonpositive measured cells")
    if not exact_positive(p, replacement):
        raise ValueError(f"{role} output replacement exact-positive gate failed")
    if len(np.unique(np.sort(out_t, axis=1), axis=0)) != len(out_t):
        raise ValueError(f"{role} output has duplicate tetrahedra")
    out_boundary = T.m.boundary(out_t)
    if not np.array_equal(T.m.boundary_identity(out_boundary), boundary_before):
        raise ValueError(f"{role} global oriented boundary changed")
    if not np.array_equal(T.m.boundary_identity(out_boundary), T.m.boundary_identity(target)):
        raise ValueError(f"{role} output no longer matches current target")
    new_bad, new_deficit, _, _ = quality_score(p, out_t)
    if new_bad > old_bad:
        raise ValueError(f"{role} global native-bad count increased")

    origin = np.concatenate((retained, np.full(len(replacement), -1, np.int64)))
    inverse = np.full(len(t), -1, np.int64)
    inverse[retained] = np.arange(len(retained), dtype=np.int64)
    parent_offsets = [0]
    parent_rows = []
    replacement_tet_offsets = [0]
    op_parent_offsets = [0]
    op_parent_rows = []
    for op, block in zip(ops, new_blocks):
        rows = list(map(int, op["old_cell_rows"]))
        for _ in block:
            parent_rows.extend(rows)
            parent_offsets.append(len(parent_rows))
        op_parent_rows.extend(rows)
        op_parent_offsets.append(len(op_parent_rows))
        replacement_tet_offsets.append(replacement_tet_offsets[-1] + len(block))
    updates = {
        "tetrahedra": out_t,
        "material_indices": out_mat,
        "boundary_faces": out_boundary,
        "desired_boundary_faces": np.asarray(target, np.int64),
        "output_cell_to_input_cell": origin,
        "input_cell_to_output_cell": inverse,
        "retained_input_cell_rows": retained,
        "replacement_tetrahedra": replacement,
        "replacement_parent_offsets": np.asarray(parent_offsets, np.int64),
        "replacement_parent_cell_rows": np.asarray(parent_rows, np.int64),
        "replacement_tet_offsets": np.asarray(replacement_tet_offsets, np.int64),
        "quality_milp_operation_parent_offsets": np.asarray(
            [0] + list(np.cumsum([len(op["old_cell_rows"]) for op in ops if op["proposal_method"] == "milp"])), np.int64),
        "quality_milp_operation_parent_cell_rows": np.asarray(
            [r for op in ops if op["proposal_method"] == "milp" for r in op["old_cell_rows"]], np.int64),
        "quality_fixed_boundary_operation_parent_offsets": np.asarray(op_parent_offsets, np.int64),
        "quality_fixed_boundary_operation_parent_cell_rows": np.asarray(op_parent_rows, np.int64),
        "quality_milp_input_candidate_path": np.asarray([str(source_path)]),
        "quality_milp_input_candidate_sha256": np.asarray([source_sha]),
        "quality_milp_partial_bad_replacement": np.asarray([any(
            op["proposal_method"] == "milp" and op["partial_bad_replacement"] for op in ops)]),
        "quality_milp_operation_count": np.asarray([sum(op["proposal_method"] == "milp" for op in ops)], np.int64),
        "quality_milp_operation_seed_rows": np.asarray([op["seed_cell_row"] for op in ops if op["proposal_method"] == "milp"], np.int64),
        "quality_fixed_boundary_operation_count": np.asarray([len(ops)], np.int64),
        "quality_fixed_boundary_operation_methods": np.asarray([op["proposal_method"] for op in ops]),
        "quality_fixed_boundary_operation_seed_rows": np.asarray([op["seed_cell_row"] for op in ops], np.int64),
        "quality_fixed_boundary_input_candidate_path": np.asarray([str(source_path)]),
        "quality_fixed_boundary_input_candidate_sha256": np.asarray([source_sha]),
        "quality_milp_input_node_to_output_node": np.arange(len(p), dtype=np.int64),
        "quality_milp_output_node_to_input_node": np.arange(len(p), dtype=np.int64),
        "current_geometry_parent_path": np.asarray([str(source_path)]),
        "current_geometry_parent_sha256": np.asarray([source_sha]),
        "canonical_parent_union_operation_order": np.asarray("partial-positive-fixed-boundary-cavity-cluster-union-v1"),
    }
    updates.update(unchanged_point_lineage_updates(a, len(p)))
    payload = dict(a)
    payload.update(updates)
    assert_unchanged_point_lineage(payload, len(p))
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{role.lower()}-partial-quality-milp-volume-proposal.npz"
    np.savez_compressed(output, **payload)
    out_sha = sha(output)
    # Read back the actual emitted file, not the in-memory proposal.
    rb = load_npz(output)
    if sha(output) != out_sha:
        raise ValueError("candidate changed during readback")
    if rb["points_world_m"].tobytes(order="C") != a["points_world_m"].tobytes(order="C"):
        raise ValueError("point coordinates changed in emitted candidate")
    rb_boundary = T.m.boundary(rb["tetrahedra"])
    if not np.array_equal(T.m.boundary_identity(rb_boundary), T.m.boundary_identity(target)):
        raise ValueError("emitted candidate boundary readback failed")
    rb_bad, rb_def, rb_det, rb_q = quality_score(rb["points_world_m"], rb["tetrahedra"])
    if np.any(rb_det <= 0) or rb_bad != new_bad:
        raise ValueError("emitted candidate native quality readback failed")
    result = {
        "role": role, "status": "PASS_BOUNDED_PARTIAL_FIXED_BOUNDARY_QUALITY_BATCH",
        "input": pin(source_path), "output": pin(output),
        "state_input_sha256": state.get("_state_sha256"),
        "input_cells": int(len(t)), "output_cells": int(len(out_t)),
        "input_points": int(len(p)), "output_points": int(len(p)),
        "native_bad_before": int(old_bad), "native_bad_after": int(new_bad),
        "log_deficit_before": float(old_deficit), "log_deficit_after": float(new_deficit),
        "parent_declared_nonpositive": int(declared_nonpositive),
        "exact_nonpositive_new_cells": 0,
        "all_output_measured_positive": True,
        "unchanged_cell_signs_inherited_from_pinned_parent": True,
        "all_replacement_cells_exact_positive": True,
        "target_boundary_key": target_key,
        "oriented_target_boundary_unchanged": True,
        "all_new_cells_same_material_as_cavity": True,
        "complete_replacement_parent_union_csr": True,
        "accepted_operations": ops, "attempts": trial_rows,
        "rejections": dict(rejection), "native_thresholds_unchanged": True,
        "kernel_cone_helper": {"path": str(KERNEL_PATH), "sha256": EXPECTED_KERNEL_SHA256},
        "kernel_cone_cache": dict(kernel_cone_stats),
        "kernel_cone_cache_entries": len(kernel_cone_cache),
        "proposal_method_counts": dict(Counter(op["proposal_method"] for op in ops)),
        "native_admission": False, "physical_admission": False, "native_steps": 0,
        "mass_mapping": "NOT_RUN",
        "surface_context": "INHERITED_UNCHANGED_TARGET; no surface candidate modified",
        "elapsed_s": time.monotonic() - role_started,
    }
    return result

def run_state(state_path, output_path, seconds, args):
    start = time.monotonic()
    state_pin = pin(state_path)
    raw = json.loads(Path(state_path).read_text())
    wrapped = "current_verified_state_20261031" in raw
    state = raw.get("current_verified_state_20261031", raw)
    state["_state_sha256"] = state_pin["sha256"]
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    role_results = []
    updated = dict(state)
    available = []
    for role in ("Skin", "Fat"):
        key = role + "_current"
        if key not in state:
            continue
        declared = state[key].get("native_quality_failures")
        if declared is None or int(declared) < 0:
            raise ValueError(f"{role} state must pin a nonnegative native_quality_failures count for scheduling")
        available.append((role, int(declared)))
    # Larger current debt gets first access to the same bounded solver, with
    # exact input-count validation inside run_role. Equal-count ties are Fat-first.
    available.sort(key=lambda item: (-item[1], 0 if item[0] == "Fat" else 1))
    schedule = []
    for idx, (role, pinned_bad) in enumerate(available):
        now = time.monotonic()
        remaining = max(0.0, start + seconds - now)
        roles_left = len(available) - idx
        slice_seconds = remaining / max(roles_left, 1)
        role_deadline = now + slice_seconds
        schedule.append({"role": role, "pinned_native_bad_before": pinned_bad,
                         "budget_seconds": float(slice_seconds),
                         "budget_policy": "equal_remaining_slice_sorted_by_bad_desc_fat_tiebreak"})
        key = role + "_current"
        if slice_seconds <= 0.01:
            role_results.append({"role": role, "status": "SKIPPED_GLOBAL_TIME_BUDGET"})
            continue
        result = run_role(role, state[key], state, output_path, role_deadline, args)
        role_results.append(result)
        if result.get("status") == "PASS_BOUNDED_PARTIAL_FIXED_BOUNDARY_QUALITY_BATCH":
            updated[key] = {**state[key], **result["output"],
                            "cells": result["output_cells"],
                            "native_quality_failures": result["native_bad_after"],
                            "nonpositive": 0}
    updated["physical_admission"] = False
    updated["native_steps"] = 0
    updated["mass_mapping"] = "NOT_RUN"
    updated.pop("independent_replay", None)
    updated.pop("exact_context", None)
    if wrapped:
        final_state = dict(raw)
        final_state["current_verified_state_20261031"] = updated
    else:
        final_state = updated
    state_out = output_path / "state.json"
    state_out.write_text(json.dumps(final_state, indent=2) + "\n")
    report = {
        "schema": "h706.partial-positive-fixed-boundary-quality-kernel-cone-batch.v2",
        "status": "PASS_WITH_PARTIAL_NATIVE_DEFECTS" if any(
            x.get("status") == "PASS_BOUNDED_PARTIAL_FIXED_BOUNDARY_QUALITY_BATCH" for x in role_results
        ) else "NO_VALID_BOUNDED_PROGRESS",
        "source_state": state_pin, "output_state": pin(state_out),
        "producer": pin(Path(__file__)), "solver": pin(SOLVER_PATH),
        "topology_owner": pin(TOP_PATH), "kernel_cone_helper": pin(KERNEL_PATH),
        "role_results": role_results,
        "role_schedule": schedule,
        "roles_sorted_by_measured_source_bad_descending": True,
        "balanced_per_role_deadlines": True,
        "native_thresholds_unchanged": True,
        "all_cells_positive": True,
        "native_admission": False, "physical_admission": False, "native_steps": 0,
        "mass_mapping": "NOT_RUN",
        "elapsed_s": time.monotonic() - start,
        "time_budget_s": float(seconds),
    }
    report_path = output_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report

def fixture(output_path, args):
    """Exercise the actual state/driver/emission path on a 2-to-1 bad-count case."""
    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / "fixture-input"
    inputs.mkdir()
    p = np.asarray([
        [0, 0, 0],
        [1, 0, 0],
        [0.1932799369096756, 0.11271771043539047, 0],
        [0.27614834904670715, 0.032624464482069016, 7.99610570538789e-05],
        [0.3822214603424072, 0.036393679678440094, -8.297244494315237e-05],
    ], np.float32)
    old = M.orient(p, np.asarray([[0, 1, 2, 3], [0, 2, 1, 4]], np.int64))
    boundary = T.m.boundary(old)
    det, q = M.measurements(p, old)
    before_bad = int((~M.passes(det, q)).sum())
    if before_bad != 2 or not exact_positive(p, old):
        raise AssertionError("fixture no longer has two exact-positive native-bad source cells")
    src = inputs / "skin-fixture.npz"
    np.savez_compressed(src, points_world_m=p, tetrahedra=old,
                        material_indices=np.asarray([16, 16], np.int32),
                        boundary_faces=boundary, desired_boundary_faces=boundary)
    state_path = inputs / "state.json"
    state = {
        "schema": "h706.partial-positive-milp-fixture-state.v1",
        "Skin_current": {"path": str(src), "sha256": sha(src),
                         "cells": 2, "nonpositive": 0,
                         "native_quality_failures": before_bad},
        "physical_admission": False, "native_steps": 0,
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    batch_out = out / "batch"
    report = run_state(state_path, batch_out, args.seconds, args)
    role = report["role_results"][0]
    if role["status"] != "PASS_BOUNDED_PARTIAL_FIXED_BOUNDARY_QUALITY_BATCH":
        raise AssertionError(f"batch driver fixture did not pass: {role['status']}")
    if role["native_bad_before"] != 2 or role["native_bad_after"] != 1:
        raise AssertionError(f"expected strict 2->1 partial reduction: {role['native_bad_before']}->{role['native_bad_after']}")
    if not any(op["partial_bad_replacement"] for op in role["accepted_operations"]):
        raise AssertionError("fixture must retain at least one native-bad replacement tet")
    if role["exact_nonpositive_new_cells"] != 0 or not role["oriented_target_boundary_unchanged"]:
        raise AssertionError("fixture exact/native gates failed")
    summary = {
        "schema": "h706.partial-positive-fixed-boundary-quality-kernel-cone-fixture.v1",
        "status": "PASS_PARTIAL_BADCOUNT_REDUCTION_FIXTURE",
        "old_native_bad_cells": 2, "new_native_bad_cells": 1,
        "remaining_bad_cells": 1, "exact_positive_new_cells": 3,
        "exact_oriented_boundary_preserved": True,
        "exact_volume_conserved": True, "same_material": True,
        "native_thresholds_unchanged": True,
        "physical_admission": False, "native_steps": 0,
        "batch_report": pin(batch_out / "report.json"),
        "batch_candidate": role["output"],
        "producer": pin(Path(__file__)), "solver": pin(SOLVER_PATH),
    }
    (out / "fixture-report.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--solve-seconds", type=float, default=2.0)
    ap.add_argument("--maximum-old-cells", type=int, default=12)
    ap.add_argument("--maximum-points", type=int, default=16)
    ap.add_argument("--maximum-attempts", type=int, default=64)
    ap.add_argument("--maximum-operations", type=int, default=8)
    ap.add_argument("--fixture", action="store_true")
    args = ap.parse_args()
    if args.maximum_old_cells < 1 or args.maximum_old_cells > 12:
        ap.error("--maximum-old-cells must be within 1..12")
    if args.maximum_points < 4 or args.maximum_points > 16:
        ap.error("--maximum-points must be within 4..16")
    if args.seconds <= 0 or args.solve_seconds <= 0:
        ap.error("time limits must be positive")
    if args.fixture:
        result = fixture(args.output, args)
    else:
        if args.state is None:
            ap.error("--state is required unless --fixture is selected")
        result = run_state(args.state, args.output, args.seconds, args)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
