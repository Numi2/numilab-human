"""Versioned v14b derivative: weak SAT halfspaces for exact shared-feature contacts.

Only same-domain source/source triangle pairs sharing one exact vertex or edge use
this path. It requires nonzero projection spread and support on both sides, and
verifies that current triangles meet only on that shared feature. All other
rejected pairs retain v14b's frozen-node behavior. Final exact surface audits
remain mandatory.
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np

_BASE_PATH = Path(__file__).with_name("quality_lp_629_v14b_witness.py")
_spec = importlib.util.spec_from_file_location("quality_lp_629_v14b_shared_feature_base", _BASE_PATH)
_BASE = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _BASE
_spec.loader.exec_module(_BASE)


def __getattr__(name):
    return getattr(_BASE, name)


def _shared_feature_axis(points, face_a, face_b):
    """Find an oriented weak SAT separator for an exact shared vertex or edge."""
    p = np.asarray(points)
    if p.dtype != np.float32 or p.ndim != 2 or p.shape[1] != 3:
        return None
    va = np.asarray(face_a, dtype=np.int64).reshape(3)
    vb = np.asarray(face_b, dtype=np.int64).reshape(3)
    if len(set(map(int, va))) != 3 or len(set(map(int, vb))) != 3:
        return None
    shared = sorted(set(map(int, va)).intersection(map(int, vb)))
    if len(shared) not in (1, 2):
        return None
    a = p[va].astype(np.float64)
    b = p[vb].astype(np.float64)
    common_a = [int(np.flatnonzero(va == r)[0]) for r in shared]
    common_b = [int(np.flatnonzero(vb == r)[0]) for r in shared]
    if not np.array_equal(p[va[common_a]], p[vb[common_b]]):
        return None
    exclusive_a = [i for i in range(3) if i not in common_a]
    exclusive_b = [i for i in range(3) if i not in common_b]
    if not exclusive_a or not exclusive_b:
        return None

    scale = max(float(np.linalg.norm(np.ptp(a, axis=0))),
                float(np.linalg.norm(np.ptp(b, axis=0))),
                float(np.max(np.abs(np.r_[a.reshape(-1), b.reshape(-1)]))), 1.e-12)
    tol = 128.0 * np.finfo(np.float64).eps * max(scale, 1.0)
    origin = a[common_a].mean(axis=0)
    ea = np.roll(a, -1, axis=0) - a
    eb = np.roll(b, -1, axis=0) - b
    na = np.cross(ea[0], ea[1])
    nb = np.cross(eb[0], eb[1])
    area_floor = 128.0 * np.finfo(np.float64).eps * max(scale * scale, 1.e-30)
    if np.linalg.norm(na) <= area_floor or np.linalg.norm(nb) <= area_floor:
        return None

    # For a shared vertex, reject contacts whose plane intersection line also
    # enters both triangle cones. Near-parallel cases fail closed.
    if len(shared) == 1:
        denom = float(np.linalg.norm(na) * np.linalg.norm(nb))
        sin_angle = float(np.linalg.norm(np.cross(na, nb)) / denom) if denom else 0.0
        if not np.isfinite(sin_angle) or sin_angle <= 1.e-8:
            return None
        line = np.cross(na, nb)
        line_norm = float(np.linalg.norm(line))
        if line_norm <= 0 or not np.isfinite(line_norm):
            return None
        line /= line_norm
        ua = np.stack([a[i] - origin for i in exclusive_a], axis=1)
        ub = np.stack([b[i] - origin for i in exclusive_b], axis=1)
        for sign in (-1.0, 1.0):
            ray = sign * line
            ca = np.linalg.lstsq(ua, ray, rcond=None)[0]
            cb = np.linalg.lstsq(ub, ray, rcond=None)[0]
            ra = float(np.linalg.norm(ua @ ca - ray))
            rb = float(np.linalg.norm(ub @ cb - ray))
            if (ra <= 1.e-9 and rb <= 1.e-9
                    and np.min(ca) >= -1.e-10 and np.min(cb) >= -1.e-10
                    and float(np.sum(ca)) > 1.e-10 and float(np.sum(cb)) > 1.e-10):
                return None

    axes = [("triangle_normal_a", na), ("triangle_normal_b", nb)]
    axes.extend((f"edge_cross_{i}_{j}", np.cross(x, y))
                for i, x in enumerate(ea) for j, y in enumerate(eb))
    axes.extend((f"normal_a_cross_edge_{i}", np.cross(na, edge))
                for i, edge in enumerate(list(ea) + list(eb)))
    axes.extend((f"normal_b_cross_edge_{i}", np.cross(nb, edge))
                for i, edge in enumerate(list(ea) + list(eb)))

    # A shared-edge bisector gives a useful in-plane separator when standard
    # triangle SAT axes collapse onto either triangle's own plane.
    if len(shared) == 2:
        edge = p[shared[1]].astype(np.float64) - p[shared[0]].astype(np.float64)
        edge_norm = float(np.linalg.norm(edge))
        if edge_norm <= tol:
            return None
        ehat = edge / edge_norm
        origin = p[shared[0]].astype(np.float64)
        ua = a[exclusive_a[0]] - origin
        ub = b[exclusive_b[0]] - origin
        ua = ua - ehat * float(ua @ ehat)
        ub = ub - ehat * float(ub @ ehat)
        nua, nub = float(np.linalg.norm(ua)), float(np.linalg.norm(ub))
        if nua > tol and nub > tol:
            axes.append(("shared_edge_bisector", ua / nua - ub / nub))
            axes.append(("shared_edge_antibisector", ua / nua + ub / nub))
            axes.append(("edge_cross_apex_a", np.cross(ehat, ua)))
            axes.append(("edge_cross_apex_b", np.cross(ehat, ub)))

    best = None
    best_score = -1.0
    best_strict = False
    for axis_name, raw in axes:
        raw = np.asarray(raw, dtype=np.float64)
        norm = float(np.linalg.norm(raw))
        if not np.isfinite(norm) or norm <= area_floor:
            continue
        n0 = raw / norm
        for orientation in (1.0, -1.0):
            n = orientation * n0
            pa = (a - origin) @ n
            pb = (b - origin) @ n
            shared_values = np.r_[pa[common_a], pb[common_b]]
            if float(np.max(np.abs(shared_values), initial=0.0)) > tol:
                continue
            if float(np.max(pa)) > tol or float(np.min(pb)) < -tol:
                continue
            spread_a = float(np.ptp(pa))
            spread_b = float(np.ptp(pb))
            margin_a = float(-np.max(pa[exclusive_a]))
            margin_b = float(np.min(pb[exclusive_b]))
            # Require nonzero spread and at least one strictly supported
            # off-feature point on each side; other vertices may lie in the
            # common plane. The exact final audit remains authoritative there.
            if (spread_a <= tol or spread_b <= tol
                    or float(np.min(pa[exclusive_a])) >= -tol
                    or float(np.max(pb[exclusive_b])) <= tol):
                continue
            depth_a = float(-np.min(pa[exclusive_a]))
            depth_b = float(np.max(pb[exclusive_b]))
            score = min(depth_a, depth_b)
            strict = margin_a > tol and margin_b > tol
            # Prefer a strict separator whenever one exists; only compare
            # support depth among candidates with the same strict/weak class.
            if (best is None or (strict and not best_strict)
                    or (strict == best_strict and score > best_score)):
                best_score = score
                best_strict = strict
                best = {"axis": n.copy(), "axis_name": axis_name,
                        "feature_rows": shared, "feature_size": len(shared),
                        "strict_support": bool(strict),
                        "support_depth_a_m": depth_a, "support_depth_b_m": depth_b,
                        "minimum_off_feature_margin_m": min(margin_a, margin_b),
                        "projection_spread_a_m": spread_a,
                        "projection_spread_b_m": spread_b,
                        "tolerance_m": tol, "origin": origin.copy()}
    return best


def _feature_rows_for_pair(points, face_a, face_b, pair_id):
    result = _shared_feature_axis(points, face_a, face_b)
    if result is None:
        return None
    va = np.asarray(face_a, dtype=np.int64).reshape(3)
    vb = np.asarray(face_b, dtype=np.int64).reshape(3)
    n = result["axis"]
    origin = result["origin"]
    rows = []
    # Keep all nine inequalities; a shared shared-pair contributes the exact
    # zero row and cancels without inventing a positive gap.
    for ai in range(3):
        for bi in range(3):
            rows.append({
                "point_rows": np.asarray((int(va[ai]), int(vb[bi])), dtype=np.int64),
                "vectors": np.stack((n, -n)),
                "origin": origin.copy(), "upper_bound": 0.0,
                "face_pair": [-1, -1], "pair_id": int(pair_id),
                "baseline_gap_m": 0.0, "retained_gap_m": 0.0,
                "axis": n.copy(), "kind": "source_source",
                "shared_feature_rows": result["feature_rows"],
                "shared_feature_size": int(result["feature_size"]),
                "shared_feature_strict_support": bool(result["strict_support"]),
                "shared_feature_axis_name": str(result["axis_name"]),
            })
    detail = {k: v for k, v in result.items() if k not in ("axis", "origin")}
    detail["axis"] = np.asarray(result["axis"], dtype=np.float64).tolist()
    return rows, detail


def prepare_triangle_separation_rows(points, faces_a, faces_b, pair_rows, *,
                                     points_b=None, retained_gap_fraction=.5):
    """v14b positive-gap rows plus weak SAT rows at exact shared features.

    Unshared zero-gap, invalid, ambiguous, intersecting, and nonseparable pairs
    retain v14b frozen-node behavior. Final exact surface audit is unchanged.
    """
    base = _BASE.prepare_triangle_separation_rows(
        points, faces_a, faces_b, pair_rows, points_b=points_b,
        retained_gap_fraction=retained_gap_fraction)
    if points_b is not None:
        return base
    p = np.asarray(points)
    fa = np.asarray(faces_a, dtype=np.int64)
    fb = np.asarray(faces_b, dtype=np.int64)
    pairs = np.asarray(pair_rows, dtype=np.int64).reshape((-1, 2))
    base_rows = list(base["constraints"])
    accepted_ids = {int(r["pair_id"]) for r in base_rows}
    rejected_ids = sorted(set(range(len(pairs))) - accepted_ids)
    if not rejected_ids:
        return base

    base_prep = dict(base["preparation"])
    old_details = {int(x["pair_id"]): dict(x)
                   for x in base_prep.get("rejected_pair_details", [])}
    safe_rows = []
    accepted_shared = []
    unsafe_details = []
    unsafe_frozen = set()
    for pid in rejected_ids:
        ia, ib = map(int, pairs[pid])
        detail0 = old_details.get(pid, {"pair_id": pid, "face_rows": [ia, ib],
                                        "reason": "REJECTED_BY_BASE_OWNER"})
        candidate = None
        if detail0.get("reason") == "NO_POSITIVE_BASELINE_GAP":
            candidate = _feature_rows_for_pair(p, fa[ia], fb[ib], pid)
        if candidate is None:
            unsafe_frozen.update(map(int, fa[ia]))
            unsafe_frozen.update(map(int, fb[ib]))
            unsafe_details.append(detail0)
            continue
        rows, feature_detail = candidate
        for row in rows:
            row["face_pair"] = [ia, ib]
        safe_rows.extend(rows)
        accepted_shared.append({"pair_id": int(pid), "face_rows": [ia, ib], **feature_detail})

    constraints = base_rows + safe_rows
    by = {}
    for ri, row in enumerate(constraints):
        for node in np.unique(np.asarray(row["point_rows"], dtype=np.int64)):
            by.setdefault(int(node), []).append(int(ri))
    prep = base_prep
    prep.update({
        "usable_separating_pairs": int(base_prep.get("usable_separating_pairs", 0) + len(accepted_shared)),
        "rejected_pairs": int(len(unsafe_details)),
        "linear_halfspace_rows": int(len(constraints)),
        "source_source_pair_count": int(base_prep.get("source_source_pair_count", 0) + len(accepted_shared)),
        "rejected_pair_details": unsafe_details,
        "rejected_pair_details_omitted_by_owner_limit": 0,
        "freeze_source_nodes": sorted(unsafe_frozen),
        "shared_feature_contact_pairs": int(len(accepted_shared)),
        "shared_feature_strict_support_pairs": int(sum(bool(x["strict_support"]) for x in accepted_shared)),
        "shared_feature_weak_support_pairs": int(sum(not bool(x["strict_support"]) for x in accepted_shared)),
        "shared_feature_contact_details": accepted_shared,
        "shared_feature_rule": "exact shared point/edge only; nonzero spread and support on both sides; nine relative halfspaces; final exact audit mandatory",
    })
    return {"constraints": constraints, "row_indices_by_point": by, "preparation": prep}
