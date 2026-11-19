"""Keep source and posed skin triangles oriented during the joint vector fit."""
import argparse
import json

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import coo_matrix, vstack

from fit_connected_skin_vectors import (
    graph, prepare, SOURCE, HERE, Pose, projected_guard, exact,
    audit_source, positioned_records, decode, hash_file,
)


def area_constraints(points, jacobian, faces, index, motion):
    seed = points[faces].astype(float) * 1e3
    posed = seed + np.einsum("fijk,fik->fij", jacobian[faces], motion[faces])
    reference = np.cross(seed[:, 1]-seed[:, 0], seed[:, 2]-seed[:, 0])
    scale = np.linalg.norm(reference, axis=1)
    assert np.all(scale > 0.)
    unit = reference / scale[:, None]
    actual = np.einsum("ij,ij->i", np.cross(posed[:, 1]-posed[:, 0], posed[:, 2]-posed[:, 0]), unit)
    selected = (actual < 0.80*scale) & np.any(index[faces] != -1, axis=1)
    changed = faces[selected]
    vertices = posed[selected]
    gradients = np.cross(np.roll(vertices, -1, axis=1)-np.roll(vertices, -2, axis=1), unit[selected, None])
    transformed = np.einsum("fvi,fvij->fvj", gradients, jacobian[changed])
    # Convert area errors to approximately comparable displacement in mm.
    denominator = np.linalg.norm(gradients, axis=(1, 2)).clip(.001)
    transformed /= denominator[:, None, None]
    current = np.einsum("fvi,fvi->f", transformed, motion[changed])
    rhs = current + (0.68*scale[selected] - actual[selected])/denominator
    n = int(index.max()) + 1
    columns = 3 * index[changed, None] + np.arange(3)
    rows = np.broadcast_to(np.arange(len(changed))[:, None, None], columns.shape)
    valid = columns >= 0
    operator = coo_matrix((transformed[valid], (rows[valid], columns[valid])),
                         shape=(len(rhs), 3*n)).tocsr()
    return operator, rhs


def solve(operator, rhs, weight, shape, bias, start):
    def objective(x):
        local = x.reshape(-1, 3)
        ax = shape @ local + bias[:, None]*local
        residual = np.maximum(rhs - operator @ x, 0.)
        derivative = weight*residual
        return .5*(np.sum(local*ax) + derivative.dot(residual)), ax.ravel()-operator.T@derivative

    opt = minimize(objective, start.ravel(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=210, gtol=3.e-4, ftol=4.e-8, maxcor=7))
    if not np.isfinite(opt.x).all():
        raise RuntimeError(opt.message)
    return opt.x.reshape(-1, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial", default="skin-connected-vectors.npz")
    args = p.parse_args()
    source = SOURCE.parent.parent / "resting-clearance-and-atrium-002/skin-rebound.nhskin"
    mesh = decode(source.read_bytes())
    xyz, faces = mesh["verts"][:, :3], mesh["inds"]
    region, lap, degree = graph(faces, xyz)
    index = np.full(len(xyz), -1, dtype=int)
    index[region] = np.arange(len(region))
    poses, contact, gaps, weights = prepare(mesh, faces, xyz, index)
    baseline = np.load(HERE/args.trial)["displacement_m"]*1000
    x = baseline.copy()
    identity = np.tile(np.eye(3), (len(xyz), 1, 1))
    cases = [(xyz, identity)] + [(points, J) for _, _, points, J in poses]
    history = []
    for iteration in range(5):
        area = [area_constraints(points, J, faces, index, x) for points, J in cases]
        counts = [len(rhs) for _, rhs in area]
        print("orientation active rows", iteration, counts, flush=True)
        if not sum(counts):
            break
        A = vstack([contact, *(op for op, _ in area)], format="csr")
        b = np.concatenate([gaps, *(rhs for _, rhs in area)])
        w = np.r_[weights, np.full(sum(counts), 2000.)]
        x[region] = solve(A, b, w, 10.*lap[region][:, region], .12*degree[region], x[region])
        history.append(counts)
    x *= .001
    active = np.linalg.norm(x, axis=1) >= .00003
    x[~active] = 0.
    new = (xyz.astype(float)+x).astype("<f4")
    fixed = exact._prepare_surface_aabb(positioned_records(xyz, faces, np.arange(len(faces))))
    _, count = audit_source(xyz, new, faces, active, fixed, set())
    print("exact source self-intersections introduced", count, flush=True)
    results = []
    for path, report, points, J in poses:
        displaced = (points+np.einsum("vij,vj->vi", J, x)).astype("<f4")
        pose = Pose(path, report, displaced, active)
        row, _ = pose.audit(active, check_self=True)
        row["orientation_bad_triangles"] = len(projected_guard(pose, active))
        results.append(row)
        print(row, flush=True)
    np.savez_compressed(HERE/"skin-connected-oriented.npz", displacement_m=x)
    (HERE/"skin-connected-oriented.json").write_text(json.dumps(dict(
        source_sha256=hash_file(source), source_new_self_pairs=count,
        active_orientation_rows=history, results=results, qualified=False), indent=2)+"\n")


if __name__ == "__main__":
    main()
