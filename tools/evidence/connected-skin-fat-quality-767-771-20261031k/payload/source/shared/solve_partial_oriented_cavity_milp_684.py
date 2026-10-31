#!/usr/bin/env python3
"""Lexicographic positive-candidate MILP for small fixed-boundary quality cavities."""
from pathlib import Path
from itertools import combinations
from collections import Counter
import importlib.util
import json
import math
import sys
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix, vstack, csr_matrix

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD_SOLVER_PATH = ROOT / "complete-surface-cavity-220-root" / "solve_oriented_cavity_milp_220.py"
_SPEC = importlib.util.spec_from_file_location("partial_milp_base220", OLD_SOLVER_PATH)
S = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = S
_SPEC.loader.exec_module(S)
M = S.QUALITY_OWNER
QUALITY_GATE = 1.0e-5
VOLUME_GATE = 1.0e-18
PAT = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), np.int64)


def _chain_faces(faces):
    out = Counter()
    for face in np.asarray(faces, np.int64):
        key = tuple(sorted(map(int, face)))
        a, b, c = map(int, face)
        inversions = int(a > b) + int(a > c) + int(b > c)
        out[key] += -1 if inversions % 2 else 1
    return {k: v for k, v in out.items() if v}


def _log_deficit(det6, normalized):
    ratio = np.minimum(np.minimum(np.asarray(normalized) / QUALITY_GATE,
                                  np.asarray(det6) / (6.0 * VOLUME_GATE)), 1.0)
    return -np.log(np.maximum(ratio, 1.0e-300))


def _exact_face_crossings(points, tets):
    P = S.P
    owner = P / "deep-boundary-checkpoint-349" / "payload" / "exact-owner" / "numilab_human" / "cardiac_cavity_intersections.py"
    sys.path.insert(0, str(owner.parent.parent))
    exact = __import__("numilab_human.cardiac_cavity_intersections", fromlist=["_audit_pair"])
    faces = np.asarray(tets, np.int64)[:, PAT].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2):
        return {"pass": False, "count": int(np.count_nonzero(counts > 2)), "reason": "nonmanifold_face_multiplicity"}
    unique_faces = faces[first]
    records, _ = exact.float32_surface_records(np.asarray(points, np.float32), unique_faces)
    gate = exact._audit_pair(records, records, same_surface=True)
    return {"pass": int(gate["count"]) == 0, "count": int(gate["count"]), "reason": "PASS" if int(gate["count"]) == 0 else "face_crossings"}


def solve(points, boundary_faces, seconds=10.0, max_points=16, material=15):
    """Minimize native-bad count, then log deficit, over positive exact tet candidates."""
    started = time.monotonic()
    p = np.asarray(points, np.float32)
    b = np.asarray(boundary_faces, np.int64)
    if p.ndim != 2 or p.shape[1] != 3 or p.dtype != np.float32 or not np.isfinite(p).all():
        raise ValueError("finite F32 cavity points required")
    if not 0 < len(p) <= int(max_points):
        raise ValueError("bounded cavity point count required")
    if b.ndim != 2 or b.shape[1] != 3 or np.any(b < 0) or np.any(b >= len(p)):
        raise ValueError("invalid local boundary faces")
    edge_rows = np.sort(np.concatenate((b[:, [0, 1]], b[:, [1, 2]], b[:, [2, 0]])), axis=1)
    _, edge_counts = np.unique(edge_rows, axis=0, return_counts=True)
    if np.any(edge_counts != 2):
        raise ValueError("closed manifold boundary required")
    n = len(p)
    xyz = [S.ip(x) for x in p]
    boundary_det = sum(S.triple(*(xyz[int(v)] for v in f)) for f in b)
    if boundary_det <= 0:
        raise ValueError("positive exact oriented boundary volume required")

    combos = np.asarray(list(combinations(range(n), 4)), np.int64)
    candidates = []
    exact_det6 = []
    for row in combos:
        d = S.det(*(xyz[int(v)] for v in row))
        if d == 0:
            continue
        if d < 0:
            row = row.copy()
            row[0], row[1] = row[1], row[0]
            d = -d
        candidates.append(row)
        exact_det6.append(d)
    if not candidates:
        return {"status": "NO_NONDEGENERATE_POSITIVE_CANDIDATES", "runtime_s": time.monotonic() - started}, None
    candidates = np.asarray(candidates, np.int64)
    exact_det6 = np.asarray(exact_det6, dtype=object)
    dfloat, q = M.measurements(p, candidates)
    positive_float = np.isfinite(dfloat) & (dfloat > 0) & np.isfinite(q)
    candidates = candidates[positive_float]
    exact_det6 = exact_det6[positive_float]
    dfloat = dfloat[positive_float]
    q = q[positive_float]
    if not len(candidates):
        return {"status": "NO_FLOAT32_POSITIVE_CANDIDATES", "runtime_s": time.monotonic() - started}, None
    passed = np.asarray(M.passes(dfloat, q), bool)
    bad_cost = (~passed).astype(np.float64)
    deficit_cost = _log_deficit(dfloat, q).astype(np.float64)

    tris = np.asarray(list(combinations(range(n), 3)), np.int64)
    tc = S.codes(tris, n)
    all_candidate_faces = candidates[:, PAT].reshape(-1, 3)
    fr = np.searchsorted(tc, S.codes(all_candidate_faces, n))
    sg = S.signs(all_candidate_faces)
    A = coo_matrix((sg.astype(np.float64),
                    (fr, np.repeat(np.arange(len(candidates)), 4))),
                   shape=(len(tris), len(candidates))).tocsc()
    br = np.searchsorted(tc, S.codes(b, n))
    if len(np.unique(br)) != len(b):
        raise ValueError("duplicate cavity boundary triangle")
    target = np.zeros(len(tris), np.float64)
    target[br] = S.signs(b)
    face_constraint = LinearConstraint(A, target, target)
    options0 = {"time_limit": max(0.02, float(seconds) * 0.48), "mip_rel_gap": 0.0, "presolve": True}
    r1 = milp(bad_cost, integrality=np.ones(len(candidates), np.uint8),
              bounds=Bounds(0, 1), constraints=face_constraint, options=options0)
    base_report = {
        "candidate_count_total_combinations": int(len(combos)),
        "candidate_count_positive_exact_float32": int(len(candidates)),
        "candidate_count_native_passing": int(passed.sum()),
        "candidate_count_native_bad_allowed": int((~passed).sum()),
        "boundary_faces": int(len(b)),
        "material_index": int(material),
        "native_thresholds_unchanged": True,
        "objective": "lexicographic_bad_count_then_sum_log_deficit",
        "stage1_status": int(r1.status),
        "stage1_message": str(r1.message),
        "stage1_runtime_s": time.monotonic() - started,
        "physical_admission": False,
        "native_steps": 0,
    }
    if r1.x is None or int(r1.status) != 0:
        return {**base_report, "status": "NO_OPTIMAL_MIN_BADCOUNT_WITHIN_BOUND",
                "runtime_s": time.monotonic() - started}, None
    x1 = np.rint(r1.x).astype(np.int8)
    if np.any(np.abs(r1.x - x1) > 1e-7) or not np.array_equal(A @ x1, target):
        return {**base_report, "status": "REJECT_NONINTEGRAL_OR_FACE_BALANCE",
                "runtime_s": time.monotonic() - started}, None
    best_bad = int(np.dot(bad_cost, x1))
    elapsed = time.monotonic() - started
    left = float(seconds) - elapsed
    xbest = x1
    stage2_status = "NOT_RUN_BUDGET"
    stage2_optimal = False
    if left > 0.05:
        bad_constraint = csr_matrix(bad_cost.reshape(1, -1))
        A2 = vstack((A, bad_constraint), format="csc")
        lower = np.r_[target, float(best_bad)]
        upper = np.r_[target, float(best_bad)]
        r2 = milp(deficit_cost, integrality=np.ones(len(candidates), np.uint8),
                  bounds=Bounds(0, 1),
                  constraints=LinearConstraint(A2, lower, upper),
                  options={"time_limit": left, "mip_rel_gap": 0.0, "presolve": True})
        stage2_status = str(r2.message)
        stage2_optimal = int(r2.status) == 0
        if r2.x is not None:
            x2 = np.rint(r2.x).astype(np.int8)
            if np.any(np.abs(r2.x - x2) > 1e-7) or not np.array_equal(A @ x2, target):
                return {**base_report, "status": "REJECT_STAGE2_FACE_BALANCE",
                        "stage2_status": int(r2.status), "runtime_s": time.monotonic() - started}, None
            if int(np.dot(bad_cost, x2)) != best_bad:
                return {**base_report, "status": "REJECT_STAGE2_BADCOUNT_CHANGED",
                        "stage2_status": int(r2.status), "runtime_s": time.monotonic() - started}, None
            if float(deficit_cost @ x2) < float(deficit_cost @ xbest) - 1.0e-12:
                xbest = x2
    selected = candidates[xbest.astype(bool)]
    selected_exact = [int(x) for x in exact_det6[xbest.astype(bool)]]
    if not selected_exact or min(selected_exact) <= 0 or sum(selected_exact) != boundary_det:
        return {**base_report, "status": "REJECT_EXACT_SIGN_OR_BOUNDARY_FLUX",
                "runtime_s": time.monotonic() - started}, None
    if _chain_faces(selected[:, PAT].reshape(-1, 3)) != _chain_faces(b):
        return {**base_report, "status": "REJECT_EXACT_ORIENTED_BOUNDARY",
                "runtime_s": time.monotonic() - started}, None
    faces = selected[:, PAT].reshape(-1, 3)
    _, first, counts = np.unique(np.sort(faces, axis=1), axis=0, return_index=True, return_counts=True)
    if np.any(counts > 2) or not np.array_equal(_chain_faces(faces[first[counts == 1]]), _chain_faces(b)):
        return {**base_report, "status": "REJECT_NONMANIFOLD_OR_WRONG_BOUNDARY",
                "runtime_s": time.monotonic() - started}, None
    crossing = _exact_face_crossings(p, selected)
    if not crossing["pass"]:
        return {**base_report, "status": "REJECT_EXACT_INTERNAL_FACE_CROSSING",
                "crossing": crossing, "runtime_s": time.monotonic() - started}, None
    out_det, out_q = M.measurements(p, selected)
    out_pass = np.asarray(M.passes(out_det, out_q), bool)
    report = {
        **base_report,
        "status": "PASS_PARTIAL_POSITIVE_CAVITY_MILP",
        "stage1_badcount_optimum": best_bad,
        "stage2_status": stage2_status,
        "stage2_optimal": bool(stage2_optimal),
        "stage2_deficit_objective": float(deficit_cost @ xbest),
        "selected_cells": int(len(selected)),
        "new_native_bad_cells": int((~out_pass).sum()),
        "new_native_good_cells": int(out_pass.sum()),
        "exact_positive_cells": int(len(selected)),
        "exact_volume_conserved": True,
        "exact_oriented_boundary_preserved": True,
        "exact_internal_face_crossings": 0,
        "runtime_s": time.monotonic() - started,
    }
    return report, selected

