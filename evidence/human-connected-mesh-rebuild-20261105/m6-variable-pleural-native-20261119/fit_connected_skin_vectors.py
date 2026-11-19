"""Fit coherent three-component skin motion against exact multi-pose witnesses."""
import json
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, diags

from fit_connected_skin_offset import SOURCE, HERE, PREVIOUS, PRIOR, BASE, bindings, capture
from fit_connected_skin_offset import compose, decode, axes, Pose, projected_guard
from fit_connected_skin_offset import positioned_records, audit_source, exact, hash_file


def graph(faces, xyz):
    edges, counts = np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1),
                              axis=0, return_counts=True)
    clear = bindings.audit.clearance
    pinned = np.ones(len(xyz), bool)
    pinned[np.unique(faces)] = False
    pinned[edges[counts != 2].ravel()] = True
    pinned[clear._exact_coordinate_tied_source_vertex_ids(xyz)] = True
    a, b = edges.T
    edge_mm = np.linalg.norm(xyz[a] - xyz[b], axis=1) * 1e3
    weights = 1. / np.maximum(edge_mm, .25)
    adj = coo_matrix((np.r_[weights, weights], (np.r_[a, b], np.r_[b, a])),
                     shape=(len(xyz), len(xyz))).tocsr()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    return np.flatnonzero(~pinned), diags(degree) - adj, degree


def prepare(mesh, faces, xyz, old_to_active):
    cases = [
        (BASE, HERE.parent / "resting-surface-clearance-001/initial.json"),
        (PRIOR / "wider-release-variable-4/native/accepted-geometry/step-2000.mrvpack",
         PRIOR / "wider-release-variable-4-exact.json"),
        (HERE / "cardiac-split-native-8000/native/accepted-geometry/step-8000.mrvpack",
         HERE / "cardiac-split-native-8000-skin-exact.json"),
    ]
    states, triples, rhs, witness_weights = [], [], [], []
    for path, report_path in cases:
        receipt, world, native_faces = capture(path)
        assert np.array_equal(faces, native_faces)
        report = json.loads(report_path.read_text())
        hit, normals, gap = compose(path, report, world, faces)
        owner = faces[hit]
        J = np.einsum("vb,bdc->vdc", mesh["weights"], axes(mesh, receipt)[0])
        direction = np.einsum("pd,pvdk->pvk", normals, J[owner])
        rows = old_to_active[owner]
        keep = rows >= 0
        triples.append((rows[keep], direction[keep]))
        # Cap the influence of nearly perpendicular planes and very large
        # witness pairs. A positive residual still requires further repair.
        limit = np.clip(-gap[keep], -5., 12.)
        rhs.append(limit)
        lengths = np.linalg.norm(direction[keep], axis=1)
        witness_weights.append(np.minimum(1., lengths))
        states.append((path, report, world, J))
        print(dict(step=receipt["accepted_step"], retained_rows=int(keep.sum()),
                   unseen_anchored_faces=int((~keep.any(axis=1)).sum())), flush=True)
    ids, values = map(np.concatenate, zip(*triples))
    rhs = np.concatenate(rhs)
    w = np.concatenate(witness_weights)
    n = int(old_to_active.max()) + 1
    operator = coo_matrix((values.ravel(), (np.repeat(np.arange(len(rhs)), 3),
                            (3*ids[:, None] + np.arange(3)).ravel())), shape=(len(rhs), 3*n)).tocsr()
    return states, operator, rhs, w


def fit(operator, rhs, weight, lap, diagonal):
    # Relative displacements along mesh edges are regularized as vectors.
    # This avoids combining large scalar distances with abruptly different
    # vertex normals at wrist folds and other small-radius transitions.
    def objective(flat):
        v = flat.reshape(-1, 3)
        shape = lap @ v + diagonal[:, None] * v
        remaining = np.maximum(rhs - operator @ flat, 0.)
        # Huber loss limits demands from individual long or nearly tangent faces.
        grad = np.minimum(remaining, 2.5) * weight
        data = grad * (remaining - .5 * np.minimum(remaining, 2.5))
        return .5 * np.sum(v*shape) + data.sum(), shape.ravel() - operator.T @ grad

    result = minimize(objective, np.zeros(operator.shape[1]), jac=True, method="L-BFGS-B",
                      options=dict(maxiter=180, ftol=3.e-8, gtol=1.e-4, maxcor=7))
    if not np.isfinite(result.x).all():
        raise RuntimeError(result.message)
    print(dict(converged=bool(result.success), message=str(result.message),
               iterations=int(result.nit), cost=float(result.fun)), flush=True)
    return result.x.reshape(-1, 3)


def main():
    source = SOURCE.parent.parent / "resting-clearance-and-atrium-002/skin-rebound.nhskin"
    mesh = decode(source.read_bytes())
    xyz, faces = mesh["verts"][:, :3], mesh["inds"]
    region, lap, degree = graph(faces, xyz)
    old_to_active = np.full(len(xyz), -1, dtype=int)
    old_to_active[region] = np.arange(len(region))
    samples, operator, rhs, w = prepare(mesh, faces, xyz, old_to_active)
    displacement = np.zeros_like(xyz, dtype=float)
    displacement[region] = fit(operator, rhs, w, 7. * lap[region][:, region], .08*degree[region])*.001
    active = np.linalg.norm(displacement, axis=1) >= .00003
    displacement[~active] = 0.
    print(dict(max_mm=float(1e3*np.linalg.norm(displacement, axis=1).max()),
               changed=int(active.sum())), flush=True)
    fixed = exact._prepare_surface_aabb(positioned_records(xyz, faces, np.arange(len(faces))))
    new = (xyz.astype(float)+displacement).astype("<f4")
    _, source_self = audit_source(xyz, new, faces, active, fixed, set())
    print(dict(source_new_self_pairs=source_self), flush=True)
    results = []
    for path, report, points, J in samples:
        displaced = (points + np.einsum("vij,vj->vi", J, displacement)).astype("<f4")
        pose = Pose(path, report, displaced, active)
        evaluated, _ = pose.audit(active, check_self=True)
        evaluated["orientation_bad_triangles"] = len(projected_guard(pose, active))
        results.append(evaluated)
        print(evaluated, flush=True)
    np.savez_compressed(HERE/"skin-connected-vectors.npz", displacement_m=displacement)
    (HERE/"skin-connected-vectors.json").write_text(json.dumps(dict(
        source_sha256=hash_file(source), source_new_self_pairs=source_self,
        poses=results, native_release_qualified=False), indent=2)+"\n")


if __name__ == "__main__":
    main()
