#!/usr/bin/env python3
"""Local fixed-boundary longest-edge bisection proposal; no admission."""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import importlib.util
import json
import math
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_SPEC = importlib.util.spec_from_file_location("bisection683_owner198", ROOT / "fixed_boundary_volume_flip_198_root.py")
M = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = M
_SPEC.loader.exec_module(M)
Q_GATE = 1.0e-5
V_GATE = 1.0e-18


def _point_key(x):
    return np.asarray(x, dtype=np.float32).tobytes()


def _face_coefficient(face):
    a, b, c = map(int, face)
    key = tuple(sorted((a, b, c)))
    inv = int(a > b) + int(a > c) + int(b > c)
    return key, (-1 if inv % 2 else 1)


def _oriented_face(key, coeff):
    a, b, c = key
    return (a, b, c) if coeff > 0 else (a, c, b)


def _split_oriented_face(face, a, b, mid):
    f = list(map(int, face))
    if a not in f or b not in f:
        return None
    w = next(x for x in f if x not in (a, b))
    # Rotate the oriented cycle to put the directed edge first.
    if (f[0], f[1]) in ((a, b), (b, a)):
        u, v = f[0], f[1]
    elif (f[1], f[2]) in ((a, b), (b, a)):
        u, v = f[1], f[2]
    elif (f[2], f[0]) in ((a, b), (b, a)):
        u, v = f[2], f[0]
    else:
        raise ValueError("edge endpoints do not form a face edge")
    return ((u, mid, w), (mid, v, w))


def _boundary_chain(tets):
    return M.chain(np.asarray(tets, np.int64))


def _expected_split_chain(old_tets, a, b, mid):
    old = _boundary_chain(old_tets)
    expected = Counter(old)
    parents = []
    for key, coeff in sorted(old.items()):
        if a not in key or b not in key:
            continue
        face = _oriented_face(key, coeff)
        children = _split_oriented_face(face, a, b, mid)
        del expected[key]
        for child in children:
            ck, cs = _face_coefficient(child)
            expected[ck] += cs
            if expected[ck] == 0:
                del expected[ck]
        parents.append((face, children))
    return dict(expected), parents


def _log_deficit(det, quality):
    ratio = np.minimum(np.minimum(np.asarray(quality) / Q_GATE,
                                  np.asarray(det) / (6.0 * V_GATE)), 1.0)
    return float(-np.log(np.maximum(ratio, 1e-300)).sum())


def _exact_det_rows(points, tets):
    nodes = np.unique(np.asarray(tets, np.int64).ravel())
    local = {int(v): i for i, v in enumerate(nodes)}
    ip = M.integer_points(np.asarray(points, np.float32)[nodes])
    return [M.exact_det(ip, [local[int(x)] for x in tet]) for tet in np.asarray(tets, np.int64)]


def propose(points, tetrahedra, material_indices, edge, star_rows,
            complete_star_rows, fractions=(0.5, 0.375, 0.625),
            existing_point_keys=None, max_support_deviation_m=None):
    """Evaluate F32 edge splits on an authenticated complete star.

    The caller must obtain both row arrays from the current indexed incidence map.
    No star is inferred from a truncated neighborhood. A returned proposal is local
    evidence only; callers still owe paired-interface volume, source-apex, typed
    ancestry and full-context checks.
    """
    p = np.asarray(points, np.float32)
    t = np.asarray(tetrahedra, np.int64)
    mat = np.asarray(material_indices, np.int32)
    a, b = sorted(map(int, edge))
    rows = np.unique(np.asarray(star_rows, np.int64))
    complete = np.unique(np.asarray(complete_star_rows, np.int64))
    if not np.array_equal(rows, complete):
        return {"status": "REJECT_INCOMPLETE_EDGE_STAR", "edge": [a, b]}
    if p.ndim != 2 or p.shape[1] != 3 or t.ndim != 2 or t.shape[1] != 4 or len(mat) != len(t):
        return {"status": "REJECT_INPUT_SHAPE", "edge": [a, b]}
    if len(rows) == 0 or rows.min() < 0 or rows.max() >= len(t) or min(a, b) < 0 or max(a, b) >= len(p):
        return {"status": "REJECT_ROW_RANGE", "edge": [a, b]}
    if a == b or np.any(np.sum(t[rows] == a, axis=1) != 1) or np.any(np.sum(t[rows] == b, axis=1) != 1):
        return {"status": "REJECT_STAR_EDGE_MEMBERSHIP", "edge": [a, b]}
    if len(np.unique(mat[rows])) != 1:
        return {"status": "REJECT_MIXED_MATERIAL_STAR", "edge": [a, b]}
    old = t[rows]
    old_det, old_q = M.measurements(p, old)
    old_pass = np.asarray(M.passes(old_det, old_q), bool)
    if np.any(old_det <= 0):
        return {"status": "REJECT_NONPOSITIVE_OLD_STAR", "edge": [a, b]}
    old_bad = int((~old_pass).sum())
    if old_bad == 0:
        return {"status": "REJECT_NO_BAD_QUALITY_WITNESS", "edge": [a, b]}
    old_score = _log_deficit(old_det, old_q)
    if existing_point_keys is None:
        existing_point_keys = {_point_key(x) for x in p}
    aa = p[a].astype(np.float64)
    bb = p[b].astype(np.float64)
    direction = bb - aa
    denom = float(np.dot(direction, direction))
    if denom <= 0:
        return {"status": "REJECT_ZERO_LENGTH_EDGE", "edge": [a, b]}
    trials = []
    for frac in fractions:
        f = float(frac)
        if not 0.0 < f < 1.0:
            trials.append({"fraction": f, "status": "REJECT_ENDPOINT_PARAMETER"})
            continue
        x = ((1.0 - f) * aa + f * bb).astype(np.float32)
        if _point_key(x) in existing_point_keys:
            trials.append({"fraction": f, "status": "REJECT_DUPLICATE_F32_POINT"})
            continue
        actual_t = float(np.dot(x.astype(np.float64) - aa, direction) / denom)
        nearest = aa + actual_t * direction
        residual = float(np.linalg.norm(x.astype(np.float64) - nearest))
        if max_support_deviation_m is not None and residual > float(max_support_deviation_m):
            trials.append({"fraction": f, "status": "REJECT_SUPPORT_DEVIATION_BUDGET",
                           "support_residual_m": residual, "projected_parameter": actual_t})
            continue
        mid = len(p)
        pp = np.vstack((p, x.reshape(1, 3)))
        children = np.empty((2 * len(old), 4), np.int64)
        for i, tet in enumerate(old):
            ia = int(np.flatnonzero(tet == a)[0])
            ib = int(np.flatnonzero(tet == b)[0])
            left = tet.copy()
            right = tet.copy()
            left[ib] = mid
            right[ia] = mid
            children[2 * i] = left
            children[2 * i + 1] = right
        new_det, new_q = M.measurements(pp, children)
        if np.any(~np.isfinite(new_det)) or np.any(~np.isfinite(new_q)) or np.any(new_det <= 0):
            trials.append({"fraction": f, "status": "REJECT_NONPOSITIVE_FLOAT32_CHILD"})
            continue
        exact_old = _exact_det_rows(p, old)
        exact_new = _exact_det_rows(pp, children)
        if min(exact_old) <= 0 or min(exact_new) <= 0:
            trials.append({"fraction": f, "status": "REJECT_NONPOSITIVE_EXACT_CHILD"})
            continue
        new_pass = np.asarray(M.passes(new_det, new_q), bool)
        if np.any(old_pass & ~new_pass.reshape(len(old), 2).all(axis=1)):
            trials.append({"fraction": f, "status": "REJECT_OLD_GOOD_CELL_REGRESSION"})
            continue
        new_bad = int((~new_pass).sum())
        new_score = _log_deficit(new_det, new_q)
        gain = old_score - new_score
        if new_bad > old_bad:
            trials.append({"fraction": f, "status": "REJECT_BAD_COUNT_INCREASE",
                           "old_bad": old_bad, "new_bad": new_bad, "deficit_gain": gain})
            continue
        if new_bad == old_bad and gain <= max(1e-8, 1e-6 * old_score):
            trials.append({"fraction": f, "status": "REJECT_TIED_COUNT_WITHOUT_STRICT_DEFICIT_GAIN",
                           "old_bad": old_bad, "new_bad": new_bad, "deficit_gain": gain})
            continue
        old_chain = _boundary_chain(old)
        expected_chain, face_parents = _expected_split_chain(old, a, b, mid)
        new_chain = _boundary_chain(children)
        if new_chain != expected_chain:
            trials.append({"fraction": f, "status": "REJECT_ORIENTED_BOUNDARY_CHAIN"})
            continue
        delta = sum(exact_new) - sum(exact_old)
        candidate_keep = np.ones(len(t), bool)
        candidate_keep[rows] = False
        output_tets = np.vstack((t[candidate_keep], children))
        output_mat = np.r_[mat[candidate_keep], np.full(len(children), int(mat[rows[0]]), np.int32)]
        trial = {
            "fraction": f,
            "status": "PASS_LOCAL_EDGE_BISECTION",
            "edge": [a, b],
            "new_point_world_m_f32": x.tolist(),
            "new_point_hex_f32": [hex(int(v)) for v in x.view(np.uint32)],
            "projected_parameter": actual_t,
            "support_residual_m": residual,
            "old_bad": old_bad,
            "new_bad": new_bad,
            "old_log_deficit": old_score,
            "new_log_deficit": new_score,
            "log_deficit_gain": gain,
            "no_old_good_cell_regression": True,
            "all_new_exact_determinants_positive": True,
            "exact_det6_delta_integer": str(delta),
            "exact_volume_delta_m3": float(Fraction(delta, 6 * (1 << 447))),
            "boundary_parent_children": [
                {"parent_face": list(face), "children": [list(x) for x in ch]}
                for face, ch in face_parents
            ],
            "output_points": pp,
            "output_tetrahedra": output_tets,
            "output_material_indices": output_mat,
            "retired_input_cell_rows": rows.copy(),
            "replacement_parent_cell_rows": np.repeat(rows, 2),
            "replacement_tetrahedra": children,
            "new_point_parent_edge": np.asarray([a, b], np.int64),
            "new_point_parameter": f,
            "new_point_support_residual_m": residual,
            "old_oriented_star_boundary": old_chain,
            "new_oriented_star_boundary": new_chain,
            "promotion_ready": False,
            "remaining_required_gates": [
                "paired-interface exact volume cancellation where applicable",
                "current source-face apex and exact full-context checks",
                "typed source-face/point ancestry binding",
                "global target-boundary and material replay",
                "source330 mass conservation",
            ],
        }
        trials.append(trial)
    passing = [x for x in trials if x.get("status") == "PASS_LOCAL_EDGE_BISECTION"]
    if not passing:
        return {"status": "NO_VALID_LOCAL_EDGE_BISECTION", "edge": [a, b], "trials": trials}
    best = min(passing, key=lambda x: (x["new_bad"], x["new_log_deficit"], x["support_residual_m"], x["fraction"]))
    best["status"] = "PASS_LOCAL_EDGE_BISECTION_BEST"
    best["all_fraction_trials"] = [
        {k: v for k, v in x.items() if k not in ("output_points", "output_tetrahedra", "output_material_indices",
                                                 "replacement_tetrahedra", "old_oriented_star_boundary",
                                                 "new_oriented_star_boundary")}
        for x in trials
    ]
    return best


def fixture():
    # One bad tetrahedron; at f=.625 one child crosses the unchanged native q gate.
    eps = np.float32(0.00141421356)
    p = np.asarray(((0, 0, 0), (1, 0, 0), (0.625, eps, 0), (0.625, 0, eps)), np.float32)
    t = M.orient(p, ((0, 1, 2, 3),))
    mats = np.asarray((16,), np.int32)
    got = propose(p, t, mats, (0, 1), np.asarray((0,), np.int64),
                  np.asarray((0,), np.int64), fractions=(0.5, 0.375, 0.625),
                  max_support_deviation_m=0.0)
    assert got["status"] == "PASS_LOCAL_EDGE_BISECTION_BEST", got
    assert got["fraction"] == 0.625 and got["old_bad"] == got["new_bad"] == 1
    assert got["log_deficit_gain"] > 0 and got["support_residual_m"] == 0.0
    assert got["exact_det6_delta_integer"] == "0"
    assert len(got["boundary_parent_children"]) == 2
    assert all(len(x["children"]) == 2 for x in got["boundary_parent_children"])
    assert got["promotion_ready"] is False
    trials = {x["fraction"]: x["status"] for x in got["all_fraction_trials"]}
    assert trials[0.5] == "REJECT_BAD_COUNT_INCREASE"
    assert trials[0.375] == "REJECT_BAD_COUNT_INCREASE"
    return {
        "status": "PASS_QUALITY_EDGE_BISECTION_LOCAL_FIXTURE",
        "controls": [
            "authenticated_complete_star_input",
            "F32_fraction_candidates_0.5_0.375_0.625",
            "exact_F32_positive_children",
            "native_thresholds_unchanged",
            "no_old_good_cell_regression",
            "strict_tied_count_log_deficit_descent",
            "exact_oriented_boundary_split_map",
            "exact_volume_delta",
            "declared_zero_support_deviation_fixture",
            "promotion_remains_false",
        ],
        "fixture_metrics": {
            "fraction": got["fraction"],
            "old_bad": got["old_bad"],
            "new_bad": got["new_bad"],
            "log_deficit_gain": got["log_deficit_gain"],
            "exact_det6_delta_integer": got["exact_det6_delta_integer"],
            "support_residual_m": got["support_residual_m"],
            "surface_parent_faces_split": len(got["boundary_parent_children"]),
        },
        "physical_admission": False,
        "native_steps": 0,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", action="store_true")
    args = ap.parse_args()
    print(json.dumps(fixture(), indent=2) if args.fixture else json.dumps({"status": "HELPER_ONLY", "physical_admission": False}))



def rank_bad_tet_longest_edges(points, tetrahedra, bad_cell_rows, limit=None):
    """Rank longest edges witnessed by bad cells; caller resolves each full star via 379."""
    p = np.asarray(points, np.float32)
    t = np.asarray(tetrahedra, np.int64)
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    witnesses = {}
    for row in np.unique(np.asarray(bad_cell_rows, np.int64)):
        if row < 0 or row >= len(t):
            raise ValueError("bad-cell row outside current tet domain")
        cell = t[int(row)]
        xyz = p[cell].astype(np.float64)
        lengths = [float(np.linalg.norm(xyz[i] - xyz[j])) for i, j in pairs]
        longest = max(lengths)
        for pair, length in zip(pairs, lengths):
            if length < longest * (1.0 - 1.0e-12):
                continue
            edge = tuple(sorted((int(cell[pair[0]]), int(cell[pair[1]]))))
            rec = witnesses.setdefault(edge, {"edge_length_m": length, "bad_cell_rows": set()})
            rec["bad_cell_rows"].add(int(row))
    ranked = [
        {"edge": list(edge), "edge_length_m": rec["edge_length_m"],
         "longest_bad_cell_rows": sorted(rec["bad_cell_rows"]),
         "longest_bad_cell_count": len(rec["bad_cell_rows"])}
        for edge, rec in witnesses.items()
    ]
    ranked.sort(key=lambda x: (-x["longest_bad_cell_count"], -x["edge_length_m"], x["edge"]))
    return ranked if limit is None else ranked[:max(0, int(limit))]

