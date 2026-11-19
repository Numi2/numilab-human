"""Resolve source and posed self-crossings without separating the connected surface."""
import json

import numpy as np
from scipy.sparse import vstack

from constrain_skin_vector_orientation import area_constraints, solve
from fit_connected_skin_vectors import (
    graph, prepare, SOURCE, HERE, Pose, projected_guard, exact,
    positioned_records, decode, hash_file,
)
from correct_cardiac_field_trust import Inequalities, best_axis, gap_pairs


def self_intersections(points, faces):
    records = positioned_records(points, faces, np.arange(len(faces)))
    tree = exact._prepare_surface_aabb(records)
    audited = exact._audit_record_pairs(
        ((a, b) for a, b in exact._query_surface_aabb(records, tree) if a[3] < b[3]),
        same_surface=True)
    return np.asarray(audited["triangle_pairs"], dtype=int).reshape(-1, 2)


def compile_pairs(matrix, baseline, current, jacobian, faces, vertices, index, witnesses):
    H = baseline.astype(float)*1000.
    X = current.astype(float)*1000.
    for a, b in witnesses:
        first, second = faces[a], faces[b]
        common = np.intersect1d(first, second)
        if len(common) == 0 and not np.any(index[np.r_[first,second]]>=0):
            continue
        if len(common) == 2:
            # For an edge-adjacent folded pair, keep the shared edge and
            # separate only the two remaining vertices about that winding.
            p = int(np.setdiff1d(first, common)[0])
            q = int(np.setdiff1d(second, common)[0])
            e = X[common[1]] - X[common[0]]
            d = H[q]-H[p]
            length = e @ e
            if length <= 1.e-14:
                continue
            n = d-e*((d @ e)/length)
            if np.linalg.norm(n) <= 1.e-7:
                continue
            axis = n / np.linalg.norm(n)
            pairs = [(p, int(r)) for r in common] + [(int(r), q) for r in common]
        elif len(common)<2:
            pairs = gap_pairs(first, second)
            axis = best_axis(first, second, X, H, pairs)
        else:
            continue

        for p, q in pairs:
            demand = .008 - float((H[q] - H[p]) @ axis)
            terms = []
            for v, sign in ((p, 1.), (q, -1.)):
                it = index[v]
                if it >= 0:
                    terms.extend((3*it+k, sign*c) for k,c in enumerate(axis @ jacobian[v]))
            if terms:
                # Store a lower-bound row as the negative of the upper-bound
                # assembly convention used by the existing witness builder.
                matrix.upper(terms, -demand)


def evaluate(xyz, faces, current, J):
    return (xyz + np.einsum("vij,vj->vi", J, current)*.001).astype("<f4")


def main():
    source = SOURCE.parent.parent / "resting-clearance-and-atrium-002/skin-rebound.nhskin"
    mesh = decode(source.read_bytes())
    xyz, faces = mesh["verts"][:, :3], mesh["inds"]
    region, lap, degree = graph(faces, xyz)
    index = np.full(len(xyz), -1, dtype=int)
    index[region] = np.arange(len(region))
    poses, contact, gaps, weights = prepare(mesh, faces, xyz, index)
    start = np.load(HERE/"skin-connected-oriented.npz")["displacement_m"]*1000.
    x = start.copy()
    identity = np.tile(np.eye(3), (len(xyz), 1, 1))
    states = [(xyz, identity)] + [(points, J) for _, _, points, J in poses]
    matrix = Inequalities()
    history = []
    seen = set()
    for iteration in range(3):
        counts = []
        for sample,(points, J) in enumerate(states):
            moved = evaluate(points, faces, x, J)
            pairs = self_intersections(moved, faces)
            counts.append(len(pairs))
            pending = []
            for a, b in pairs:
                tag = sample, int(a), int(b)
                if tag not in seen:
                    seen.add(tag)
                    pending.append((a,b))
            compile_pairs(matrix, points, moved, J, faces, region, index, pending)
        history.append(counts)
        print("exact self intersections", iteration, counts, "constraints", len(matrix.rhs), flush=True)
        if not any(counts) or not matrix.rhs:
            break
        area = [area_constraints(points, J, faces, index, x) for points, J in states]
        active_area = sum(len(rhs) for _,rhs in area)
        A = vstack([contact, -matrix.matrix(3*len(region)), *(op for op, _ in area)], format="csr")
        b = np.r_[gaps, -np.asarray(matrix.rhs), *(rhs for _,rhs in area)]
        w = np.r_[weights, np.full(len(matrix.rhs), 3000.), np.full(active_area, 2500.)]
        shape = 10.*lap[region][:, region]
        x[region] = solve(A, b, w, shape, .12*degree[region], x[region])
    delta = x*.001
    active = np.linalg.norm(delta, axis=1) >= .00003
    delta[~active] = 0.0
    final_self = len(self_intersections(evaluate(xyz, faces, delta*1000., identity), faces))
    result = []
    for path, report, points, J in poses:
        pose = Pose(path, report, evaluate(points, faces, delta*1000., J), active)
        row, _ = pose.audit(active, check_self=True)
        row["orientation_bad_triangles"] = len(projected_guard(pose, active))
        result.append(row)
        print(row, flush=True)
    np.savez_compressed(HERE/"skin-connected-self-fit.npz", displacement_m=delta)
    (HERE/"skin-connected-self-fit.json").write_text(json.dumps(dict(
        source_sha256=hash_file(source), source_self_witnesses=final_self,
        source_and_pose_history=history, results=result, native_qualified=False),
        indent=2)+"\n")


if __name__=="__main__":
    main()
