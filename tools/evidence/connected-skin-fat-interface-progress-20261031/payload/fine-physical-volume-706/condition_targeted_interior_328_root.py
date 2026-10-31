#!/usr/bin/env python3
"""Safeguarded F32 quality conditioning of inferred interior Skin nodes.

Connectivity, complete boundary, registered source coordinates, material rows,
and point identities are immutable. This is a construction proposal only.
"""
from pathlib import Path
import argparse, hashlib, json, importlib.util, time, os, csv
import numpy as np
from scipy.optimize import minimize

H = Path(__file__).resolve().parent
BASE = importlib.util.spec_from_file_location("vol198_condition308", H / "fixed_boundary_volume_flip_198_root.py")
m = importlib.util.module_from_spec(BASE)
BASE.loader.exec_module(m)
TOP_SPEC = importlib.util.spec_from_file_location("boundary209_condition308", H / "transfer_surface_contractions_to_volume_209_root.py")
top = importlib.util.module_from_spec(TOP_SPEC)
TOP_SPEC.loader.exec_module(top)
QMIN = 1.0e-5
VMIN = 1.0e-18

def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def f32_key(p):
    return np.asarray(p, dtype=np.float32).tobytes()



def registered_source_rows_from_map(registration_map, source_points):
    """Return source-row identity mask from explicit registered role records.

    Historical source-node coordinate matches alone do not confer a registered
    surface identity. O/I rows must carry direct current typed-vertex identity;
    volume-interior rows must explicitly declare no surface ancestry.
    """
    n = len(source_points)
    seen = np.zeros(n, dtype=bool)
    registered = np.zeros(n, dtype=bool)
    counts = {"O": 0, "I": 0, "interior_no_surface_ancestry": 0}
    vertex_rows = {"O": set(), "I": set()}
    surface_vertex_rows = {"O": set(), "I": set()}
    with Path(registration_map).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"source_partition", "source_tetgen_node_id", "role",
                    "source_domain", "source_mapping_kind", "source_vertex_row",
                    "source_face_row", "registered_surface_layer",
                    "registered_surface_vertex_row", "current_x_m", "current_y_m",
                    "current_z_m", "parent_registration_kind", "parent_tet_row",
                    "anatomical_region_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("registered node map lacks required identity columns")
        for x in reader:
            if x["source_partition"] != "skin":
                continue
            sid = int(x["source_tetgen_node_id"])
            row = sid - 1
            if row < 0 or row >= n or seen[row]:
                raise ValueError("skin TetGen source node IDs are duplicate or out of range")
            seen[row] = True
            mapped = np.asarray([float(x["current_x_m"]), float(x["current_y_m"]),
                                 float(x["current_z_m"])], dtype=np.float32)
            if not np.array_equal(mapped.view(np.uint32), source_points[row].view(np.uint32)):
                raise ValueError("registration-map coordinates disagree with pinned source node row")
            role = x["role"]
            domain = x["source_domain"]
            kind = x["source_mapping_kind"]
            source_vertex = int(x["source_vertex_row"])
            source_face = int(x["source_face_row"])
            surface_layer = x["registered_surface_layer"]
            surface_vertex = int(x["registered_surface_vertex_row"])
            if role in ("O", "I"):
                if domain != "derived_registered_skin_authoring" or kind != "vertex":
                    raise ValueError("O/I point lacks direct registered-surface vertex identity")
                if source_vertex < 0 or source_face != -1 or surface_layer != role or surface_vertex < 0:
                    raise ValueError("O/I point has incomplete or conflicting registered identity")
                if source_vertex in vertex_rows[role] or surface_vertex in surface_vertex_rows[role]:
                    raise ValueError("registered O/I vertex identity is not unique")
                vertex_rows[role].add(source_vertex)
                surface_vertex_rows[role].add(surface_vertex)
                registered[row] = True
                counts[role] += 1
            elif role == "interior":
                if domain != "derived_volume_interior_no_surface_ancestry" or kind != "none":
                    raise ValueError("interior row is not explicitly marked no-surface-ancestry")
                if source_vertex != -1 or source_face != -1 or surface_layer or surface_vertex != -1:
                    raise ValueError("inferred volume interior carries unexpected surface identity")
                if int(x["parent_tet_row"]) < 0 or int(x["anatomical_region_id"]) < 0:
                    raise ValueError("inferred volume interior lacks current inferred parent registration")
                if x["parent_registration_kind"] != "inferred_current_parent_tet_same_material_donor_region":
                    raise ValueError("inferred volume interior parent-registration method is unexpected")
                counts["interior_no_surface_ancestry"] += 1
            else:
                raise ValueError("unknown Skin source-node role in registration map")
    if not np.all(seen):
        raise ValueError(f"registration map omits {int(np.sum(~seen))} Skin source-node rows")
    return registered, counts

def integer_point(p):
    out = []
    for x in np.asarray(p, dtype=np.float32):
        n, d = float(x).as_integer_ratio()
        out.append(n << (149 - (d.bit_length() - 1)))
    return tuple(out)

def exact_star_determinants(points, tetrahedra):
    ids = np.unique(tetrahedra)
    ip = {int(r): integer_point(points[int(r)]) for r in ids}
    return [m.exact_det([ip[int(v)] for v in tet], [0, 1, 2, 3]) for tet in tetrahedra]

def local_metrics(base_points, local_tets, node, xyz):
    pts = base_points[local_tets].astype(np.float64, copy=True)
    rows, corners = np.where(local_tets == int(node))
    if len(rows) != len(local_tets) or np.any(np.bincount(rows, minlength=len(local_tets)) != 1):
        raise ValueError("eligible node must occur exactly once in every incident tet")
    pts[rows, corners, :] = np.asarray(xyz, dtype=np.float64)
    u = pts[:, 1:, :] - pts[:, :1, :]
    det = np.einsum("ij,ij->i", u[:, 0], np.cross(u[:, 1], u[:, 2]))
    maxedge = np.zeros(len(local_tets), dtype=np.float64)
    for i in range(4):
        for j in range(i + 1, 4):
            maxedge = np.maximum(maxedge, np.linalg.norm(pts[:, i] - pts[:, j], axis=1))
    q = det / np.maximum(maxedge ** 3, np.finfo(np.float64).tiny)
    return det, q

def star_boundary_closed(boundary):
    e = np.concatenate((boundary[:, [0, 1]], boundary[:, [1, 2]], boundary[:, [2, 0]]), axis=0)
    se = np.sort(e, axis=1)
    order = np.lexsort((se[:, 1], se[:, 0]))
    ss = se[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.any(ss[1:] != ss[:-1], axis=1))]
    counts = np.diff(np.r_[starts, len(ss)])
    orient = np.where(e[:, 0] < e[:, 1], 1, -1)
    sums = np.add.reduceat(orient[order], starts)
    return bool(np.all(counts == 2) and np.all(sums == 0))

def validate_input(points, tetrahedra, material, source_points, registered_source_rows=None):
    if points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be exact Float32 [N,3]")
    if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4 or tetrahedra.dtype.kind not in "iu":
        raise ValueError("tetrahedra must be integer [M,4]")
    if len(material) != len(tetrahedra):
        raise ValueError("material row count mismatch")
    if not np.isfinite(points).all() or np.any(tetrahedra < 0) or np.any(tetrahedra >= len(points)):
        raise ValueError("nonfinite coordinates or invalid point row")
    det, q = m.measurements(points, tetrahedra)
    if np.any(det <= 0):
        raise ValueError("input has nonpositive Float32 cells")
    if np.any(np.diff(np.sort(tetrahedra, axis=1), axis=1) == 0):
        raise ValueError("input has repeated vertices in a tetrahedron")
    boundary = top.boundary(tetrahedra)
    source_keys = [f32_key(p) for p in source_points]
    if len(set(source_keys)) != len(source_keys):
        raise ValueError("registered source coordinates are not unique")
    source_set = set(source_keys)
    candidate_keys = [f32_key(p) for p in points]
    candidate_set = set(candidate_keys)
    requested_keys = source_keys if registered_source_rows is None else [
        k for k, registered in zip(source_keys, registered_source_rows) if registered]
    if any(k not in candidate_set for k in requested_keys):
        raise ValueError("candidate omits registered source coordinates")
    boundary_rows = np.unique(boundary)
    fixed = np.zeros(len(points), dtype=bool)
    fixed[boundary_rows] = True
    if registered_source_rows is None:
        fixed |= np.fromiter((k in source_set for k in candidate_keys), dtype=bool, count=len(points))
        registered_count = len(source_points)
        historical_interior_count = 0
    else:
        registered_source_rows = np.asarray(registered_source_rows, dtype=bool)
        if registered_source_rows.shape != (len(source_points),):
            raise ValueError("registered identity mask length differs from source node rows")
        if len(points) < len(source_points) or not np.array_equal(points[:len(source_points)][registered_source_rows].view(np.uint32), source_points[registered_source_rows].view(np.uint32)):
            raise ValueError("candidate point prefix is not the pinned source-node row domain")
        fixed[:len(source_points)] |= registered_source_rows
        registered_count = int(registered_source_rows.sum())
        historical_interior_count = len(source_points) - registered_count
    used = np.zeros(len(points), dtype=bool)
    used[np.unique(tetrahedra)] = True
    eligible = np.flatnonzero(used & ~fixed)
    if len(eligible) == 0:
        raise ValueError("no inferred interior points are eligible")
    return det, q, boundary, fixed, eligible, {
        "historical_source_node_rows": len(source_points),
        "registered_source_point_rows": registered_count,
        "historical_volume_interior_no_surface_ancestry_rows": historical_interior_count,
        "candidate_boundary_point_rows": len(boundary_rows),
        "registered_source_coordinate_hits": registered_count,
        "used_point_rows": int(used.sum()),
        "eligible_inferred_interior_point_rows": len(eligible),
        "unused_point_rows": int((~used).sum()),
    }

def safeguarded_trial(points, star, node, trial_coord, before_pass, old_boundary_identity, old_integer_sum, max_displacement, x0):
    trial_points = points.copy()
    trial_points[int(node)] = np.asarray(trial_coord, dtype=np.float32)
    new_det, new_q = m.measurements(trial_points, star)
    if np.any(new_det <= 0):
        return None, "FLOAT32_NONPOSITIVE"
    exact_new = exact_star_determinants(trial_points, star)
    if any(v <= 0 for v in exact_new):
        return None, "EXACT_FLOAT32_NONPOSITIVE"
    if not np.array_equal(old_boundary_identity, top.boundary_identity(top.boundary(star))):
        return None, "STAR_BOUNDARY_IDENTITY_CHANGED"
    if sum(exact_new) != old_integer_sum:
        return None, "EXACT_STAR_VOLUME_CHANGED"
    new_pass = m.passes(new_det, new_q)
    if np.any(before_pass & ~new_pass):
        return None, "NEW_LOCAL_DEFECT"
    if int(np.sum(~new_pass)) >= int(np.sum(~before_pass)):
        return None, "NO_STRICT_GLOBAL_DEFECT_DESCENT"
    disp = float(np.linalg.norm(np.asarray(trial_points[int(node)], dtype=np.float64) - x0))
    if disp > max_displacement * (1 + 1e-12):
        return None, "DISPLACEMENT_BOUND"
    return (trial_points, new_pass, new_det, new_q, disp), "PASS"

def optimize_one_target(points, tetrahedra, node, maxiter=36, target_rank=0):
    incident_rows = np.flatnonzero(np.any(tetrahedra == int(node), axis=1))
    if not len(incident_rows):
        return None, {"status": "SKIP_UNUSED"}
    star = tetrahedra[incident_rows]
    star_boundary = top.boundary(star)
    if np.any(star_boundary == int(node)):
        return None, {"status": "REJECT_NOT_INTERIOR"}
    if not star_boundary_closed(star_boundary):
        return None, {"status": "REJECT_STAR_BOUNDARY_NOT_CLOSED"}
    before_det, before_q = m.measurements(points, star)
    before_pass = m.passes(before_det, before_q)
    if np.all(before_pass):
        return None, {"status": "SKIP_NO_BAD_INCIDENT_CELL"}
    bad = ~before_pass
    good = before_pass
    qratio0 = before_q / QMIN
    vratio0 = (before_det / 6.0) / VMIN
    bad_rows = np.flatnonzero(bad)
    ratios = np.minimum(qratio0[bad_rows], vratio0[bad_rows])
    ranked = bad_rows[np.lexsort((bad_rows, -ratios))]
    if target_rank >= len(ranked):
        return None, {"status": "SKIP_NO_TARGET"}
    selected = np.zeros(len(star), dtype=bool)
    selected[ranked[target_rank]] = True
    other_bad = bad & ~selected
    t0 = float(min(np.min(qratio0[selected]), np.min(vratio0[selected])))
    if not (0.0 < t0 < 1.0):
        return None, {"status": "SKIP_BAD_SCORE"}
    all_edges = np.concatenate([
        star[:, [0, 1]], star[:, [0, 2]], star[:, [0, 3]],
        star[:, [1, 2]], star[:, [1, 3]], star[:, [2, 3]]
    ])
    edge_lengths = np.linalg.norm(points[all_edges[:, 0]].astype(np.float64) - points[all_edges[:, 1]].astype(np.float64), axis=1)
    if not np.isfinite(edge_lengths).all() or np.min(edge_lengths) <= 0:
        return None, {"status": "REJECT_DEGENERATE_EDGE"}
    max_displacement = 0.20 * float(np.median(np.max(edge_lengths.reshape(6, len(star)), axis=0)))
    scale = float(np.median(np.max(edge_lengths.reshape(6, len(star)), axis=0)))
    scale = max(scale, np.finfo(np.float64).tiny)
    radius = max_displacement / np.sqrt(3.0) / scale
    x0 = points[int(node)].astype(np.float64)
    local = points[star].astype(np.float64, copy=True)
    rr, cc = np.where(star == int(node))
    if len(rr) != len(star) or np.any(np.bincount(rr, minlength=len(star)) != 1):
        return None, {"status": "REJECT_STAR_ROW_CONTRACT"}

    def metrics_z(z):
        pts = local.copy()
        pts[rr, cc, :] = x0 + scale * z[:3]
        u = pts[:, 1:, :] - pts[:, :1, :]
        det = np.einsum("ij,ij->i", u[:, 0], np.cross(u[:, 1], u[:, 2]))
        maxedge = np.zeros(len(star), dtype=np.float64)
        for i in range(4):
            for j in range(i + 1, 4):
                maxedge = np.maximum(maxedge, np.linalg.norm(pts[:, i] - pts[:, j], axis=1))
        q = det / np.maximum(maxedge ** 3, np.finfo(np.float64).tiny)
        return det, q

    def constraints(z):
        det, q = metrics_z(z)
        return np.r_[
            q[selected] / QMIN - z[3],
            (det[selected] / 6.0) / VMIN - z[3],
            det[other_bad] / np.maximum(before_det[other_bad], np.finfo(np.float64).tiny) - 1.0e-6,
            q[good] / QMIN - 1.0,
            (det[good] / 6.0) / VMIN - 1.0,
        ]

    def objective(z):
        return -z[3] + 1.0e-8 * float(np.dot(z[:3], z[:3]))

    start = np.zeros(4, dtype=np.float64)
    start[3] = t0
    result = minimize(
        objective, start, method="SLSQP",
        bounds=[(-radius, radius)] * 3 + [(t0, 1.0)],
        constraints=[{"type": "ineq", "fun": constraints}],
        options={"maxiter": int(maxiter), "ftol": 1.0e-10, "eps": 1.0e-5, "disp": False},
    )
    if not np.isfinite(result.x).all():
        return None, {"status": "REJECT_NONFINITE_OPTIMIZER"}
    target = x0 + scale * result.x[:3]
    before_bad = int(np.sum(~before_pass))
    old_integer_sum = sum(exact_star_determinants(points, star))
    old_boundary_identity = top.boundary_identity(star_boundary)
    original_good = before_pass.copy()
    line_trials = []
    accepted = None
    for k in range(25):
        alpha = 2.0 ** (-k)
        trial_coord = np.asarray(x0 + alpha * (target - x0), dtype=np.float32)
        if np.array_equal(trial_coord.view(np.uint32), points[int(node)].view(np.uint32)):
            line_trials.append({"alpha": alpha, "decision": "NO_F32_CHANGE"})
            continue
        checked, decision = safeguarded_trial(points, star, node, trial_coord, before_pass,
                                              old_boundary_identity, old_integer_sum,
                                              max_displacement, x0)
        if checked is None:
            line_trials.append({"alpha": alpha, "decision": decision})
            continue
        trial_points, new_pass, new_det, new_q, disp = checked
        accepted = (trial_coord.copy(), new_pass, new_det, new_q, alpha, disp)
        line_trials.append({"alpha": alpha, "decision": "ACCEPT"})
        break
        break
    if accepted is None:
        return None, {
            "status": "REJECT_LINE_SEARCH",
            "node": int(node), "incident_cells": len(star), "bad_before": before_bad,
            "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
            "optimizer_iterations": int(result.nit), "optimizer_target_scale": result.x[:3].tolist(),
            "line_search": line_trials, "maximum_displacement_m": max_displacement,
        }
    trial_coord, new_pass, new_det, new_q, alpha, disp = accepted
    return trial_coord, {
        "status": "PASS_NODE_CONDITIONING", "targeted_bad_local_row": int(ranked[target_rank]), "target_rank": int(target_rank),
        "node": int(node), "incident_cell_rows": incident_rows.tolist(),
        "incident_cells": len(star), "bad_before": before_bad, "bad_after": int(np.sum(~new_pass)),
        "minimum_normalized_det_before": float(before_q.min()),
        "minimum_normalized_det_after": float(new_q.min()),
        "minimum_volume_before_m3": float(before_det.min() / 6),
        "minimum_volume_after_m3": float(new_det.min() / 6),
        "optimizer_success": bool(result.success), "optimizer_message": str(result.message),
        "optimizer_iterations": int(result.nit), "optimizer_target_scale": result.x[:3].tolist(),
        "accepted_line_alpha": alpha, "displacement_m": disp, "maximum_displacement_m": max_displacement,
        "line_search": line_trials,
        "topology_unchanged": True, "star_boundary_fixed_exactly": True,
        "source_and_attachment_points_fixed_by_identity_mask": True,
        "registered_identity_rule": "all complete boundary rows and explicit O/I registered-surface vertex identities fixed; used no-surface-ancestry volume interior rows remain eligible",
    }

def optimize_node(points, tetrahedra, node, maxiter=36):
    trials = []
    for rank in range(3):
        target, op = optimize_one_target(points, tetrahedra, node, maxiter, rank)
        if target is not None:
            op["target_selection"] = "up to three nearest-to-passing bad cells; other bad cells must stay positive; all formerly-good cells remain passing"
            return target, op
        trials.append(op)
        if op.get("status") not in ("REJECT_LINE_SEARCH", "SKIP_BAD_SCORE"):
            break
    return None, {"status": "REJECT_TARGETED_CONDITIONING", "trials": trials}

def fixture():
    p = np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1e-6,1e-6,1e-6]], np.float32)
    outer = np.asarray([[0,1,2,3]], np.int64)
    b = top.boundary(outer)
    t = m.orient(p, np.column_stack((np.full(len(b),4,np.int64), b)))
    before = p.copy()
    det, q = m.measurements(p, t)
    initial_pass = m.passes(det,q)
    assert np.all(det > 0) and np.any(~initial_pass) and np.any(initial_pass)
    target, op = optimize_node(p,t,4)
    assert target is not None, op
    out = p.copy(); out[4] = target
    nd,nq = m.measurements(out,t)
    after_pass = m.passes(nd,nq)
    assert np.all(nd > 0) and int(np.sum(~after_pass)) < int(np.sum(~initial_pass))
    assert np.all(after_pass[initial_pass]), "initially good incident cells must stay good"
    assert np.array_equal(out[:4].view(np.uint32), p[:4].view(np.uint32))
    inverted_target = np.asarray([2,2,2], np.float32)
    before_boundary = top.boundary_identity(top.boundary(t))
    before_sum = sum(exact_star_determinants(p,t))
    rejected, why = safeguarded_trial(p,t,4,inverted_target,m.passes(det,q),before_boundary,before_sum,1.0,p[4].astype(np.float64))
    assert rejected is None and why == "FLOAT32_NONPOSITIVE"
    assert np.array_equal(p.view(np.uint32), before.view(np.uint32))
    deg = p.copy(); deg[4] = np.asarray([0.25,0.25,0.0], np.float32)
    try:
        validate_input(deg,t,np.ones(len(t),np.int32),p[:4])
        raise AssertionError("degenerate control did not fail")
    except ValueError as e:
        assert "nonpositive" in str(e)
    assert np.array_equal(p.view(np.uint32), before.view(np.uint32))
    source_points = p.copy()
    moved = p.copy(); moved[4] = np.asarray([.1,.1,.1],np.float32)
    registered = np.array([True,True,True,True,False])
    validate_input(moved,t,np.ones(len(t),np.int32),source_points,registered)
    corrupted = moved.copy(); corrupted[0,0] = np.float32(1e-3)
    try:
        validate_input(corrupted,t,np.ones(len(t),np.int32),source_points,registered)
        raise AssertionError("changed registered point was admitted")
    except ValueError as e:
        assert "registered" in str(e) or "prefix" in str(e)
    return {
        "status": "PASS_INTERIOR_NODE_CONDITIONER_CONTROLS",
        "controls": ["actual_optimizer_improves_native_defect_count",
                     "mixed_good_bad_star_preserves_every_initially_good_cell",
                     "registered_boundary_rows_bitwise_fixed",
                     "inverted_F32_trial_rolled_back_without_mutating_input", "degenerate_baseline_fails_closed", "prior_inferred_interior_motion_reaccepted", "changed_registered_identity_rejected"],
        "physical_admission": False, "native_steps": 0,
        "fixture_old_bad": int(np.sum(~initial_pass)), "fixture_new_bad": int(np.sum(~after_pass)),
        "fixture_initially_good_incident_cells": int(np.sum(initial_pass)),
        "fixture_displacement_m": op["displacement_m"], "optimizer_iterations": op["optimizer_iterations"],
    }

def run(a):
    start = time.monotonic()
    a.output.mkdir(parents=True, exist_ok=False)
    input_sha = sha(a.input)
    if a.input_sha256 and input_sha != a.input_sha256:
        raise ValueError("input SHA-256 pin mismatch")
    parent = json.loads(a.parent_report.read_text()) if a.parent_report else None
    if parent is None or not str(parent.get("status","")).startswith("PASS"):
        raise ValueError("a pinned passing parent geometry report is required")
    if parent.get("physical_admission") is not False or parent.get("native_steps") != 0:
        raise ValueError("parent must remain source-only")
    if not parent.get("oriented_boundary_unchanged_exact", parent.get("oriented_boundary_exactly_unchanged", parent.get("exact_oriented_boundary", False))):
        raise ValueError("parent report lacks exact boundary preservation")
    if not parent.get("exact_F32_volume_unchanged", parent.get("exact_volume", False)):
        raise ValueError("parent report lacks exact Float32 volume preservation")
    with np.load(a.input) as z:
        arrays = {k: z[k] for k in z.files}
    p = arrays["points_world_m"]
    t = arrays["tetrahedra"].astype(np.int64, copy=False)
    material = arrays["material_indices"]
    source_table = np.loadtxt(a.source_node, skiprows=1, usecols=(1,2,3), dtype=np.float64)
    source_points = np.asarray(source_table, dtype=np.float32)
    if not a.registration_map:
        raise ValueError("an explicit registered-node identity map is required")
    registered_rows, identity_counts = registered_source_rows_from_map(a.registration_map, source_points)
    if len(p) < len(source_points) or not np.array_equal(p[:len(source_points)][registered_rows].view(np.uint32), source_points[registered_rows].view(np.uint32)):
        raise ValueError("candidate does not preserve the pinned registered-identity point rows")
    det,q,boundary,fixed,eligible,coverage = validate_input(p,t,material,source_points,registered_rows)
    coverage["registered_identity_counts"] = identity_counts
    used_mask = np.zeros(len(p), dtype=bool); used_mask[np.unique(t.ravel())] = True
    boundary_mask = np.zeros(len(p), dtype=bool); boundary_mask[np.unique(boundary)] = True
    registered_mask = np.zeros(len(p), dtype=bool); registered_mask[:len(registered_rows)] = registered_rows
    historical_interior_mask = np.zeros(len(p), dtype=bool); historical_interior_mask[:len(registered_rows)] = ~registered_rows
    coverage["registered_nonboundary_used_rows"] = int(np.sum(registered_mask & used_mask & ~boundary_mask))
    coverage["registered_nonboundary_unused_rows"] = int(np.sum(registered_mask & ~used_mask & ~boundary_mask))
    coverage["inferred_no_surface_ancestry_source_rows_used"] = int(np.sum(historical_interior_mask & used_mask & ~boundary_mask))
    coverage["inferred_no_surface_ancestry_source_rows_unused"] = int(np.sum(historical_interior_mask & ~used_mask & ~boundary_mask))
    if "boundary_faces" in arrays and not np.array_equal(top.boundary_identity(arrays["boundary_faces"]), top.boundary_identity(boundary)):
        raise ValueError("input boundary_faces differ from actual complete tetrahedral boundary")
    if "desired_boundary_faces" in arrays and not np.array_equal(top.boundary_identity(arrays["desired_boundary_faces"]), top.boundary_identity(boundary)):
        raise ValueError("input desired boundary differs from complete tetrahedral boundary")
    initial_bad=~m.passes(det,q)
    boundary_id=top.boundary_identity(boundary)
    point_snapshot=p.copy()
    flat=t.ravel()
    order=np.argsort(flat,kind="stable")
    counts=np.bincount(flat,minlength=len(p))
    offsets=np.r_[0,np.cumsum(counts)]
    bad_hits=np.bincount(t[initial_bad].ravel(),minlength=len(p))
    eligible_mask=np.zeros(len(p),bool);eligible_mask[eligible]=True
    nodes=np.flatnonzero((bad_hits>0)&eligible_mask)
    nodes=sorted(map(int,nodes),key=lambda n:(-int(bad_hits[n]), n))
    operations=[];rejections={};current_bad=int(initial_bad.sum())
    op_log=a.output/"operations.jsonl"
    checkpoint=a.output/"accepted-coordinate-checkpoint.npz"
    op_stream=op_log.open("w")
    def persist_checkpoint():
        tmp=checkpoint.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f,point_rows=np.asarray([x["node"] for x in operations],np.int64),
                                points_world_m=p[np.asarray([x["node"] for x in operations],np.int64)])
        os.replace(tmp,checkpoint)
    for sweep in range(int(a.sweeps)):
        progress=False
        for node in nodes:
            if time.monotonic()-start >= a.seconds or (a.max_nodes and len(operations)>=a.max_nodes):
                break
            ir=np.unique(order[offsets[node]:offsets[node+1]]//4)
            if not len(ir): continue
            dd,qq=m.measurements(p,t[ir])
            before_local=int(np.sum(~m.passes(dd,qq)))
            if before_local==0: continue
            target,op=optimize_node(p,t[ir],node,maxiter=a.maxiter)
            if target is None:
                reason=op.get("status","REJECT")
                rejections[reason]=rejections.get(reason,0)+1
                continue
            trial=p.copy()
            trial[node]=target
            nd,nq=m.measurements(trial,t[ir])
            after_local=int(np.sum(~m.passes(nd,nq)))
            if after_local>=before_local:
                rejections["FINAL_LOCAL_RECHECK"] = rejections.get("FINAL_LOCAL_RECHECK",0)+1
                continue
            if np.any(nd<=0) or any(v<=0 for v in exact_star_determinants(trial,t[ir])):
                rejections["FINAL_EXACT_POSITIVITY"] = rejections.get("FINAL_EXACT_POSITIVITY",0)+1
                continue
            if not np.array_equal(trial[fixed].view(np.uint32),point_snapshot[fixed].view(np.uint32)):
                rejections["FIXED_IDENTITY_CHANGED"] = rejections.get("FIXED_IDENTITY_CHANGED",0)+1
                continue
            prev_bad=current_bad
            current_bad += after_local-before_local
            p=trial
            op["sweep"]=sweep;op["global_bad_before"]=prev_bad;op["global_bad_after"]=current_bad
            operations.append(op)
            op_stream.write(json.dumps(op,separators=(",",":"))+"\\n");op_stream.flush()
            persist_checkpoint()
            progress=True
        if not progress or time.monotonic()-start>=a.seconds or (a.max_nodes and len(operations)>=a.max_nodes):
            break
    op_stream.close()
    final_det,final_q=m.measurements(p,t)
    final_pass=m.passes(final_det,final_q)
    final_boundary=top.boundary(t)
    if not np.array_equal(top.boundary_identity(final_boundary),boundary_id):
        raise ValueError("complete oriented boundary changed")
    if not np.array_equal(t,arrays["tetrahedra"]) or not np.array_equal(material,arrays["material_indices"]):
        raise ValueError("immutable connectivity/material changed")
    if not np.array_equal(p[fixed].view(np.uint32),point_snapshot[fixed].view(np.uint32)):
        raise ValueError("registered or boundary point changed")
    if np.any(final_det<=0) or np.any(~final_pass[~initial_bad]):
        raise ValueError("nonpositive or new quality defect")
    expected=int(initial_bad.sum())-sum(o["bad_before"]-o["bad_after"] for o in operations)
    if int((~final_pass).sum())!=expected:
        raise ValueError("global quality accounting mismatch")
    output=a.output/"conditioned-volume-proposal.npz"
    arrays["points_world_m"]=p
    np.savez_compressed(output,**arrays)
    report={
        "schema":"h706.safeguarded-registered-anchor-interior-quality-conditioning.v1",
        "status":"PASS_CONDITIONING_DESCENT" if operations else "NO_SAFE_CONDITIONING_MOVE",
        "physical_admission":False,"native_steps":0,"mass_mapping_status":"NOT_RUN",
        "input":str(a.input),"input_sha256":input_sha,"parent_report":str(a.parent_report),"parent_report_sha256":sha(a.parent_report),"source_node":str(a.source_node),"source_node_sha256":sha(a.source_node),
        "output":str(output),"output_sha256":sha(output),"script":str(Path(__file__)),
        "script_sha256":sha(Path(__file__)),"input_npz_keys":list(arrays),
        "topology_unchanged":True,"material_rows_unchanged":True,"point_rows_unchanged":True,
        "all_complete_boundary_faces_and_points_fixed_exactly":True,"all_explicit_registered_surface_identity_points_fixed_exactly":True,
        "registration_map":str(a.registration_map),"registration_map_sha256":sha(a.registration_map),
        "registered_identity_counts":identity_counts,
        "eligible_rule":"used point not in current complete tetrahedral boundary and not carrying explicit O/I registered-surface vertex identity; historical volume-interior rows explicitly marked derived_volume_interior_no_surface_ancestry remain eligible",
        "eligible_coverage":coverage,"only_inferred_interior_points_moved":True,
        "exact_F32_positive_all_cells":True,"no_new_quality_defects":True,"strict_global_defect_descent":bool(operations),
        "cavity_preservation":"same connectivity plus exact positive determinants, fixed embedded one-ring star boundaries and exact star-volume preservation; no new edge or cell contact",
        "native_quality_defects_before":int(initial_bad.sum()),"native_quality_defects_after":int((~final_pass).sum()),
        "nonpositive_F32_cells":int(np.sum(final_det<=0)),"eligible_bad_touching_nodes":len(nodes),
        "accepted_operations":len(operations),"accepted_point_rows":[x["node"] for x in operations],"operations":operations,"rejections":rejections,
        "maximum_accepted_displacement_m":max([x["displacement_m"] for x in operations],default=0.0),
        "maximum_displacement_m_policy":"one fifth of median incident maximum edge per node",
        "safeguard":"targeted SLSQP normalized-determinant and absolute-volume ratios; Float32 backtracking with exact integer-lattice determinant, unchanged formerly-good incident cells, unchanged local exact boundary/volume, and strict local/global defect descent",
        "elapsed_s":time.monotonic()-start,
    }
    (a.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    with open(a.output/"operations.jsonl","w") as f:
        for op in operations:f.write(json.dumps(op,separators=(",",":"))+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="operations"},indent=2))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fixture",action="store_true")
    ap.add_argument("--input",type=Path)
    ap.add_argument("--source-node",type=Path)
    ap.add_argument("--registration-map",type=Path)
    ap.add_argument("--parent-report",type=Path)
    ap.add_argument("--input-sha256",default="")
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seconds",type=float,default=120.0)
    ap.add_argument("--sweeps",type=int,default=2)
    ap.add_argument("--max-nodes",type=int,default=0)
    ap.add_argument("--maxiter",type=int,default=36)
    a=ap.parse_args()
    if a.fixture:
        a.output.mkdir(parents=True,exist_ok=False)
        result=fixture()
        (a.output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result,indent=2))
    else:run(a)
if __name__=="__main__":main()
