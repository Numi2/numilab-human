"""Test a coherent smooth skin envelope on retained native poses.

Witness halfspaces give per-vertex clearance demands, not proof. The offset
regularization solves across connected surface edges, including bed-side
skin. Each sample is checked again using the existing exact predicates.
"""
import argparse
import json
import sys

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import cg

from fit_cardiac_volume_grid import HERE, PREVIOUS, hash_file

sys.path.insert(0, str(PREVIOUS))
from fit_skin_source import BASE, PRIOR, bindings, capture, compose, decode
from rebind_arm_witnesses import axes
from trim_three_poses import Pose, positioned_records, projected_guard, exact
from audit_skin_self_clearance import audit_source


SOURCE = PREVIOUS / "guarded-footprint.nhskin"


def prepare(mesh, normals, faces):
    cases = [
        (BASE, HERE.parent / "resting-surface-clearance-001/initial.json"),
        (PRIOR / "wider-release-variable-4/native/accepted-geometry/step-2000.mrvpack",
         PRIOR / "wider-release-variable-4-exact.json"),
        (PREVIOUS / "native-segment32-control-footprint/native/accepted-geometry/step-8000.mrvpack",
         PREVIOUS / "guarded-footprint-native-exact.json"),
    ]
    samples = []
    for path, report_path in cases:
        receipt, world, native_faces = capture(path)
        if not np.array_equal(faces, native_faces):
            raise ValueError(f"skin topology mismatch at {path}")
        report = json.loads(report_path.read_text())
        skin_ids, direction, gap = compose(path, report, world, faces)
        jacobians = np.einsum("vb,bdc->vdc", mesh["weights"], axes(mesh, receipt)[0])
        outward = np.einsum("vij,vj->vi", jacobians, normals)
        alignment = np.einsum("pvd,pd->pv", outward[faces[skin_ids]], direction)
        usable = alignment > 0.25
        requests = np.divide(-gap, alignment, out=np.zeros_like(gap), where=usable)
        active = requests > 0
        print(json.dumps(dict(
            step=receipt["accepted_step"], requests=int(active.sum()),
            no_outward_vertices=int((~np.any(usable, axis=1)).sum()),
            clearance_mm=np.quantile(requests[active], [0., .5, .9, .95, .99, 1.]).tolist(),
        )), flush=True)
        samples.append((path, report, world, outward, faces[skin_ids][active], requests[active]))
    return samples


def fit(points, edges, pinned, samples, cap):
    # Surface-area weighting is approximated by a local edge-length sum.
    # Screened edge diffusion follows the connected surface, including around
    # elbow folds, rather than smoothing through unrelated nearby triangles.
    n = len(points)
    edge_mm = 1e3 * np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    weights = 1.0 / np.maximum(edge_mm, 0.2)
    a, b = edges.T
    adj = coo_matrix((np.tile(weights, 2), (np.r_[a, b], np.r_[b, a])), shape=(n, n)).tocsr()
    degree = np.asarray(adj.sum(axis=1)).ravel()
    operator = diags(degree) - adj
    rows = np.flatnonzero(~pinned & (degree > 0))
    operator = operator[rows][:, rows].tocsr()
    demand = np.zeros(n)
    for *_, ids, requested in samples:
        np.maximum.at(demand, ids, requested)
    demand = np.minimum(demand[rows], cap)
    # Hinge fit penalizes only unmet clearance. Contiguous demand may raise
    # neighboring skin enough to clear the full triangle instead of its tip.
    smooth = 26. * operator
    bias = 0.20 * np.maximum(degree[rows], 0.05)
    loss = 3.5 * np.maximum(degree[rows], 0.05)
    x = np.zeros_like(demand)
    for stage in range(12):
        active = x < demand - 1.e-5
        diagonal = bias + loss * active
        update, status = cg(smooth + diags(diagonal), loss * active * demand,
                            x0=x, rtol=1.e-6, maxiter=450)
        if status != 0:
            raise RuntimeError(f"skin offset solve did not converge at {stage}: {status}")
        x = .7 * update + .3 * x
    value = np.zeros(n)
    value[rows] = np.maximum(x, 0)
    print(json.dumps(dict(changed=int((value > .03).sum()),
          max_mm=float(value.max()), largest_demand_mm=float(demand.max()))), flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-mm", default=18., type=float)
    args = parser.parse_args()
    mesh = decode(SOURCE.read_bytes())
    xyz, faces = mesh["verts"][:, :3], mesh["inds"]
    clear = bindings.audit.clearance
    referenced = np.unique(faces)
    normals = np.zeros_like(xyz)
    normals[referenced] = clear._area_weighted_vertex_normals(
        xyz[referenced], np.searchsorted(referenced, faces))
    edges, counts = np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1),
                              axis=0, return_counts=True)
    pinned = np.ones(len(xyz), bool)
    pinned[referenced] = False
    pinned[edges[counts != 2].ravel()] = True
    pinned[clear._exact_coordinate_tied_source_vertex_ids(xyz)] = True
    samples = prepare(mesh, normals, faces)
    values = fit(xyz, edges, pinned, samples, args.cap_mm)
    active = values > 0.03
    values[~active] = 0.
    displacement = normals * (values[:, None] * 0.001)
    poses = [Pose(path, report, (world + outward * (values[:, None] * .001)).astype("<f4"), active)
             for path, report, world, outward, _, _ in samples]
    fixed = exact._prepare_surface_aabb(positioned_records(xyz, faces, np.arange(len(faces))))
    cache = set()
    metrics = []
    for scale in (1., .65, .4):
        new = (xyz.astype(float) + scale * displacement).astype("<f4")
        _, count = audit_source(xyz, new, faces, active, fixed, cache)
        results = []
        for pose, (_, _, world, outward, _, _) in zip(poses, samples):
            pose.new = (world + outward * (scale * values[:, None] * .001)).astype("<f4")
            row, _ = pose.audit(active, check_self=True)
            row["orientation_bad_triangles"] = len(projected_guard(pose, active))
            results.append(row)
        result = dict(scale=scale, source_new_self_pairs=count, captures=results)
        metrics.append(result)
        print(json.dumps(result), flush=True)
    np.savez_compressed(HERE / "skin-connected-offset.npz", displacement_m=displacement)
    (HERE / "skin-connected-offset.json").write_text(json.dumps(dict(
        source_sha256=hash_file(SOURCE), pinned=int(pinned.sum()), results=metrics,
        native_release_qualified=False), indent=2) + "\n")


if __name__ == "__main__":
    main()
