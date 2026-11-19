"""Test moving the source muscle at the blocked edge while preserving bed support."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse import coo_matrix, csgraph, diags

here = Path(__file__).resolve().parent
baseline = here.with_name("resting-clearance-and-atrium-007")
geometry = here.with_name("resting-clearance-and-atrium-006")
sys.path[:0] = [str(baseline), str(geometry)]
from fit_edge_dependency_repair import capture_in_memory
from fit_clearance_patches import area, decode, mesh_edges
from fit_monotone_contour_patches import exact_intersection, lattice
from classify_native_crossing_geometry import audit_integer
from measure_crossing_curves import metrics
from numilab_human import cardiac_cavity_intersections as predicates
from register_tissue_bellies import (
    READER, SKIN, blend, clear, exact, exact_skin_pair, install_for_float32,
    quality, source,
)

key = (51005, 121)
target = (14040, 1476)
prior = baseline / "recomputed-support-native"
launch = json.loads((prior / "launch.json").read_text())
tissue_path = Path(launch["argv"][launch["argv"].index("--soft-tissue-payload") + 1])
assert hashlib.sha256(tissue_path.read_bytes()).hexdigest() == launch["pins"][str(tissue_path)]
skin = decode((baseline / "recomputed-support-skin.nhskin").read_bytes())
skin_faces = skin["inds"]
skin_rows = np.unique(skin_faces)
assert set(skin_faces[target[0]]) >= {6889, 7240}
bank = source(tissue_path)[key[1]]
points = bank["points"]
faces = bank["source_faces"]
assert target[1] < len(faces)
witness = prior / "native/accepted-geometry/step-8000.mrvpack"
existing = json.loads((prior / "step-8000-skin-exact.json").read_text())
previous_bank = next(r for r in existing["surfaces"] if (r["semantic"], r["stable_id"]) == key)
assert list(target) in previous_bank["triangle_pairs"]
install_for_float32()

edges, incidence = mesh_edges(faces)
fixed = np.zeros(len(points), dtype=bool)
fixed[edges[incidence != 2].ravel()] = True
fixed[clear._exact_coordinate_tied_source_vertex_ids(points.astype("<f4"))] = True
axis = np.linalg.svd(points - points.mean(0), full_matrices=False)[2][0]
axial = points @ axis
q = np.quantile(axial, [.09, .91])
fixed |= (axial <= q[0]) | (axial >= q[1])
active = np.zeros_like(fixed)
active[faces[target[1]]] = True
for _ in range(5):
    active[edges[active[edges].any(axis=1)].ravel()] = True
mobile = np.flatnonzero(active & ~fixed)
local = np.full(len(points), -1, dtype=int)
local[mobile] = np.arange(len(mobile))
assert len(mobile)
e0, e1 = edges.T
w = 1. / np.maximum(0.15, 1000 * np.linalg.norm(points[e0] - points[e1], axis=1))
adj = coo_matrix((np.r_[w, w], (np.r_[e0, e1], np.r_[e1, e0])),
                 shape=(len(points), len(points))).tocsr()
regularizer = 0.42 * csgraph.laplacian(adj).tocsr()[mobile][:, mobile] + diags(
    np.full(len(mobile), .025))
tri = (points[faces] - points.mean(0)) * 1000
a, b, c = tri.transpose(1, 0, 2)
grads = np.zeros_like(points)
for i, g in enumerate((np.cross(b, c), np.cross(c, a), np.cross(a, b))):
    np.add.at(grads, faces[:, i], g)
grads = grads[mobile].ravel()
grads /= np.linalg.norm(grads)
with capture_in_memory(witness) as (pack, bodies, receipt):
    assert receipt["pack_file_sha256"] == existing["accepted_pack_sha256"]
    found = {k: (p, f) for k, _, p, f in READER["primitives"](pack, bodies)
             if k in (key, SKIN)}
    posed_skin, emitted_faces = found[SKIN]
    assert np.array_equal(emitted_faces, np.searchsorted(skin_rows, skin_faces))
    posed, emitted_bank = found[key]
    assert np.array_equal(emitted_bank, faces)
    predicted, world_jac = blend(bank, bodies)
    assert np.max(np.linalg.norm(posed.astype(float) - predicted, axis=1)) < .00012
surface = clear._exact_surface_records(posed_skin, emitted_faces)
exact_index = exact._prepare_surface_aabb(surface)
float_index = clear._float32_skin_aabb_index(surface)
hits = exact_skin_pair(exact_index, float_index, posed, faces)
assert hits == {tuple(row) for row in previous_bank["triangle_pairs"]}
edge_vertices = {6889, 7240}
edge_faces = np.flatnonzero(np.isin(skin_faces, list(edge_vertices)).sum(axis=1) == 2)
assert len(edge_faces) == 2
fixed_edge = lattice(posed_skin[np.searchsorted(skin_rows, sorted(edge_vertices))])
edge_vector = predicates._sub(fixed_edge[1], fixed_edge[0])
edge_squared = predicates._dot(edge_vector, edge_vector)

def obstruction(surface, pairs):
    witnesses = set()
    for a, b in pairs:
        if a not in edge_faces:
            continue
        left = lattice(posed_skin[emitted_faces[a]])
        right = lattice(surface[faces[b]])
        for point in audit_integer(left, right):
            shift = predicates._sub(point, fixed_edge[0])
            along = predicates._dot(edge_vector, shift)
            if not any(predicates._cross(edge_vector, shift)) and 0 <= along <= edge_squared:
                witnesses.add(b)
    return sorted(witnesses)

def curves(surface, pairs):
    lhs = {i: lattice(posed_skin[emitted_faces[i]]) for i, _ in pairs}
    rhs = {j: lattice(surface[faces[j]]) for _,j in pairs}
    return metrics(lhs, rhs, pairs)

base_cuts = obstruction(posed, hits)
base_metrics = curves(posed, hits)
assert target[1] in base_cuts
used_pairs = [p for p in sorted(hits) if np.any(local[faces[p[1]]] >= 0)]
fixed_before = {"fixed_bed_edge": [6889, 7240], "stable_muscle": list(key),
                "target": list(target), "mobile_source_nodes": len(mobile),
                "matched_pose_sha256": receipt["pack_file_sha256"], "base_pairs": len(hits),
                "initial_fixed_edge_cut_triangles": base_cuts, "base_curves": base_metrics}
# Get stable outward candidate axes per witness without constructing a
# whole-body dense parameter basis.
s = posed_skin[emitted_faces[np.asarray(used_pairs)[:, 0]]].astype(float)
t = posed[faces[np.asarray(used_pairs)[:, 1]]].astype(float)
from fit_joint_surface_registration import witness_axes
reference_axis = witness_axes(s, t)
skin_lower = np.einsum("pvi,pi->pv", s-s[:, :1], reference_axis).min(axis=1)
gap = 1000 * (np.einsum("pvi,pi->pv", t-s[:, :1], reference_axis) - skin_lower[:, None])
region_faces = faces[np.asarray(used_pairs)[:, 1]]
grad = np.einsum("pvij,pi->pvj", world_jac[region_faces], reference_axis)
rows = []
coefs = []
rhs = []
priorities = []
for i, pair in enumerate(used_pairs):
    for corner in range(3):
        j = local[region_faces[i, corner]]
        if j >= 0:
            rows.append(3*j + np.arange(3))
            coefs.append(grad[i, corner])
            rhs.append(gap[i, corner] + .06)
            priorities.append(40. if pair == target else 1.)
rhs = np.array(rhs)
priorities = np.array(priorities)
M = coo_matrix((np.ravel(coefs), (np.repeat(np.arange(len(rhs)), 3), np.ravel(rows))),
               shape=(len(rhs), len(mobile)*3)).tocsr()

def objective(x):
    dx = regularizer @ x.reshape(-1, 3)
    volume = grads @ x
    residual = np.maximum(rhs + M @ x, 0)
    return (.5*(x @ dx.ravel()) + 15.*volume*volume + .5*(residual @ (priorities*residual)),
            dx.ravel() + 30.*volume*grads + M.T @ (priorities*residual))

def protect_face_orientation(rest, transport):
    affected = faces[(local[faces] >= 0).any(axis=1)]
    xyz = rest[affected].astype(float) * 1000
    n = area(xyz)
    n_squared = np.einsum("ij,ij->i", n, n)
    assert np.all(n_squared > 0.)
    cross_axes = np.cross(np.roll(xyz, -1, axis=1) - np.roll(xyz, -2, axis=1),
                          n[:, None]) / n_squared[:, None, None]
    if transport is not None:
        cross_axes = np.einsum("pvi,pvij->pvj", cross_axes, transport[affected])
    j = local[affected]
    keep = j >= 0
    row = np.broadcast_to(np.arange(len(j))[:, None], j.shape)[keep]
    col = (3*j[keep, None] + np.arange(3)).ravel()
    data = cross_axes[keep].ravel()
    return coo_matrix((data, (np.repeat(row, 3), col)),
                      shape=(len(j), 3*len(mobile))).tocsr()

def fixed_edge_separator(radius):
    world_edge = posed_skin[np.searchsorted(skin_rows, sorted(edge_vertices))].astype(float)
    a, b = world_edge
    tangent = b-a
    nearby = posed[faces].astype(float)
    lo, hi = nearby.min(1), nearby.max(1)
    overlap = (np.all(hi+radius>=world_edge.min(axis=0), axis=1) &
               np.all(lo-radius<=world_edge.max(axis=0), axis=1))
    rows = np.unique(faces[overlap])
    projected = np.clip((posed[rows]-a) @ tangent / (tangent @ tangent), 0., 1.)
    nearest = a+projected[:, None]*tangent
    rows = rows[np.linalg.norm(posed[rows]-nearest, axis=1) <= radius*2]
    normal_candidates = area(posed_skin[emitted_faces[edge_faces]].astype(float))
    normal = normal_candidates.sum(axis=0)
    normal -= tangent * (normal @ tangent)/(tangent @ tangent)
    normal /= np.linalg.norm(normal)
    projections = (posed[rows]-a) @ normal
    report = dict(radius_m=radius, rows=len(rows),
                  fixed_outside_rows=int(np.sum((local[rows]<0) & (projections>-.00006))))
    if report["fixed_outside_rows"] or not len(rows):
        return None, report
    selected = local[rows]>=0
    grad = np.einsum("vij,i->vj", world_jac[rows[selected]], normal)
    idx = 3*local[rows[selected],None]+np.arange(3)
    op = coo_matrix((grad.ravel(), (np.repeat(np.arange(len(idx)), 3), idx.ravel())),
                    shape=(len(idx), 3*len(mobile))).tocsr()
    upper = -1000*projections[selected]-.06
    return LinearConstraint(op, -np.inf, upper), report

results = []
for bound, constrained, radius in ((.5, True, .0015), (1.0, True, .0025),
                                    (1.7, True, .0035), (2.5, True, .005)):
    separator, separation_report=fixed_edge_separator(radius)
    if separator is None:
        results.append(separation_report)
        continue
    if constrained:
        from scipy.sparse import vstack
        orientation = vstack([protect_face_orientation(points, None),
                              protect_face_orientation(posed, world_jac)])
        constraints = [LinearConstraint(orientation, -.10, np.inf), separator]
    else:
        constraints = []
    sol = minimize(objective, np.zeros(len(mobile)*3), jac=True,
                   method="SLSQP" if constrained else "L-BFGS-B",
                   bounds=[(-bound, bound)]*(len(mobile)*3), constraints=constraints,
                   options={"maxiter": 100, "ftol": 1.e-9})
    if not sol.success:
        results.append(dict(solver_message=sol.message, bound_mm=bound, **separation_report))
    delta = np.zeros_like(points)
    delta[mobile] = sol.x.reshape(-1, 3) * .001
    for alpha in (1., .85, .7, .55, .4, .25):
        offset = alpha*delta
        local_trial = (points + offset).astype("<f4")
        normal_ratio, volume_delta = quality(points, local_trial.astype(float), faces)
        moved = (posed.astype(float) + np.einsum("vij,vj->vi", world_jac, offset)).astype("<f4")
        n0, n1 = area(posed[faces].astype(float)), area(moved[faces].astype(float))
        posed_ratio = np.min(np.einsum("ij,ij->i", n0, n1)/np.einsum("ij,ij->i", n0, n0))
        target_remains = exact_intersection(lattice(posed_skin[emitted_faces[target[0]]]),
                                             lattice(moved[faces[target[1]]]), False, set())
        row = dict(bound_mm=bound, orientation_constrained=constrained, step_scale=alpha,
                   exact_blocked_pair_remains=target_remains,
                   minimum_source_normal_projection=normal_ratio,
                   minimum_posed_normal_projection=float(posed_ratio),
                   relative_source_volume_change=volume_delta)
        if not target_remains and min(normal_ratio, posed_ratio) >= .85 and volume_delta <= .005:
            after = exact_skin_pair(exact_index, float_index, moved, faces)
            row.update(after_pairs=len(after), new_pairs=len(after-hits), removed_pairs=len(hits-after))
            remaining = obstruction(moved, after)
            after_metrics = curves(moved, after)
            row.update(fixed_edge_cut_triangles=remaining, curves=after_metrics)
            if (not remaining and after_metrics["connected_curve_components"] <=
                    base_metrics["connected_curve_components"] and
                    after_metrics["length_m"] <= base_metrics["length_m"] and
                    len(after) <= len(hits)):
                output = here / f"stable-121-terminal-screen-{bound}-{alpha}-{constrained}.npz"
                np.savez_compressed(output, row_ids=bank["row_ids"], source_float_delta_m=offset)
                row["pending_multiframe_all_tissue_delta"] = output.name
        results.append(row)
        print(row, flush=True)
fixed_before["proposals"] = results
fixed_before["source_sha256"] = launch["pins"][str(tissue_path)]
(here / "fixed-edge-tissue-screen.json").write_text(json.dumps(fixed_before, indent=2) + "\n")
