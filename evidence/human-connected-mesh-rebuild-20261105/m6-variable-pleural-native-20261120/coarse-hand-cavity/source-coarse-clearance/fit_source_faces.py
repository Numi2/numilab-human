"""Fit connected local source patches using complete triangle separation rows."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from audit_source import TRIANGLE_ROW, digest, install_for_float32, load_skin, owner, records
from human_skin_contact_planes import finite_triangle_support_planes

MILLIMETER = 0.001
CLEARANCE_MM = 0.035
MOTION_BOUND_MM = 1.5
MINIMUM_AREA_RATIO = 0.55
MINIMUM_NORMAL_COSINE = 0.5
MAXIMUM_AREA_RATIO = 2.0


def native_pairs(points, faces, banks, bone):
    skin, _ = owner.float32_surface_records(points, faces)
    tree = owner._prepare_surface_aabb(skin)
    pairs = set()
    for begin, count in banks:
        result = owner._audit_pair_prepared_first(tree, records(bone["vertices"][begin:begin+count]))
        pairs.update((int(i), int(begin + j)) for i, j in result["triangle_pairs"])
    return skin, pairs


def geometry(points, faces):
    coord = points[faces].astype(np.float64)
    u = coord[:, 1] - coord[:, 0]
    v = coord[:, 2] - coord[:, 0]
    return u, v, np.cross(u, v)


class Inequalities:
    def __init__(self):
        self.rows = []
        self.columns = []
        self.coefficients = []
        self.upper = []

    def add(self, coefficients, bound):
        for index, value in coefficients:
            if index >= 0 and value != 0:
                self.rows.append(len(self.upper))
                self.columns.append(index)
                self.coefficients.append(value)
        self.upper.append(bound)

    def matrix(self, width):
        return coo_matrix((self.coefficients, (self.rows, self.columns)),
                          shape=(len(self.upper), width)).tocsr()


def fit(points, faces, bone, contacts, edges, pinned, audit_points):
    ids = np.unique(faces[[i for i, _ in contacts]])
    active = np.zeros(len(points), dtype=bool)
    active[ids] = True
    for _ in range(3):
        active[edges[active[edges].any(axis=1)].ravel()] = True
    active &= ~pinned
    moved = np.flatnonzero(active)
    if len(moved) == 0:
        return None, {"reason": "crossings only on pinned geometry"}
    var = np.full(len(points), -1, dtype=int)
    var[moved] = np.arange(len(moved)) * 3
    dimensions = 3*len(moved)
    rows = Inequalities()
    total = ((points[moved].astype(np.float64) - audit_points[moved]) / MILLIMETER).ravel()
    bounds = [(-MOTION_BOUND_MM - d, MOTION_BOUND_MM - d) for d in total]
    objective = [0.] * dimensions
    absolute_start = dimensions
    for i in range(dimensions):
        absolute = absolute_start + i
        rows.add(((i, 1.), (absolute, -1.)), 0.)
        rows.add(((i, -1.), (absolute, -1.)), 0.)
        objective.append(0.6)
        bounds.append((0., None))

    targeted = np.flatnonzero(active[faces].any(axis=1))
    u, v, normals = geometry(points, faces[targeted])
    source_u, _, source_normals = geometry(audit_points, faces[targeted])
    norms = np.linalg.norm(source_normals, axis=1)
    assert np.all(norms > 0)
    direction = source_normals / norms[:, None]
    tangent = source_u / np.linalg.norm(source_u, axis=1, keepdims=True)
    bitangent = np.cross(direction, tangent)
    for directions, lower, upper in ((direction, .65, 1.60),
                                     (tangent, -.50, .50), (bitangent, -.50, .50)):
        jac1 = np.cross(v, directions)
        jac2 = np.cross(directions, u)
        jac = np.stack((-jac1 - jac2, jac1, jac2), axis=1)
        jac *= (MILLIMETER / norms)[:, None, None]
        projection = np.einsum("ij,ij->i", normals, directions) / norms
        for local, face in enumerate(faces[targeted]):
            signed = [(var[node]+k, float(gradient[k]))
                      for node, gradient in zip(face, jac[local], strict=True)
                      if active[node] for k in range(3)]
            rows.add(signed, upper - float(projection[local]))
            rows.add(((index, -value) for index, value in signed), float(projection[local]) - lower)

    # Bounds on tangential/normal edge deformation propagate a local correction
    # to neighboring supported triangles while keeping the boundary explicit.
    for a, b in edges[active[edges].any(axis=1)]:
        cap = float(0.35 * np.linalg.norm(audit_points[b]-audit_points[a]) / MILLIMETER)
        for axis in range(3):
            left = var[a] + axis if active[a] else -1
            right = var[b] + axis if active[b] else -1
            rows.add(((left, 1.), (right, -1.)), cap)
            rows.add(((left, -1.), (right, 1.)), cap)

    frozen_pairs = 0
    for i, j in sorted(contacts):
        shell_nodes = faces[i]
        if not active[shell_nodes].any():
            frozen_pairs += 1
            continue
        a = points[shell_nodes].astype(np.float64)
        b = bone[j]["vertices"].astype(np.float64)
        bone_out = np.cross(b[1]-b[0], b[2]-b[0])
        bone_out /= np.linalg.norm(bone_out)
        planes = finite_triangle_support_planes(a, b, bone_out)
        plane = next(p for p in planes if float(p["normal"] @ bone_out) >= 0.18)
        slack = len(objective)
        objective.append(40.)
        bounds.append((0., None))
        upper = float(plane["b_projection_mm"].max())
        for node, preprojected in zip(shell_nodes, plane["a_projection_mm"], strict=True):
            coefficients = [(slack, -1.)]
            if active[node]:
                coefficients.extend((var[node]+axis, -float(component))
                                    for axis, component in enumerate(plane["normal"]))
            rows.add(coefficients, float(preprojected) - upper - CLEARANCE_MM)

    res = linprog(np.asarray(objective), A_ub=rows.matrix(len(objective)),
                  b_ub=rows.upper, bounds=bounds, method="highs",
                  options={"time_limit": 24.})
    if not res.success:
        return None, {"reason": str(res.message), "frozen_pairs": frozen_pairs,
                      "active_rows": len(moved)}
    delta = np.zeros(points.shape, dtype=np.float64)
    delta[moved] = res.x[:dimensions].reshape(-1,3) * MILLIMETER
    info = {"active_rows": len(moved), "frozen_pairs": frozen_pairs,
            "LP_constraints": len(rows.upper), "LP_iterations": res.nit,
            "nonzero_clearance_slacks": int(np.count_nonzero(res.x[dimensions*2:] > 1e-6)),
            "maximum_proposal_m": float(np.linalg.norm(delta, axis=1).max())}
    return delta, info


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("outdir", type=Path)
    args = parser.parse_args()
    start = monotonic()
    install_for_float32()
    audit = json.loads(args.baseline.read_text())
    bone_path = args.run / "physical-bone-initial.bin"
    coords_path = args.run / "registered-skin-native.csv"
    assert digest(bone_path) == audit["source"][bone_path.name]
    assert digest(coords_path) == audit["source"][coords_path.name]
    argv = json.loads((args.run / "launch.json").read_text())["argv"]
    src = Path(argv[argv.index("--skin-payload")+1])
    assert digest(src) == audit["source"]["NHSKIN_sha256"]
    _, _, _, _, faces, _, _ = load_skin(src)
    csv = np.loadtxt(coords_path, delimiter=",", skiprows=1, usecols=(0,2,3,4))
    original = csv[:, 1:].astype(np.float32)
    assert np.array_equal(csv[:, 0], np.arange(len(original)))
    bone = np.memmap(bone_path, dtype=TRIANGLE_ROW, mode="r")
    _, starts, counts = np.unique(bone["body"], return_index=True, return_counts=True)
    banks = list(zip(starts, counts, strict=True))
    identity = {(int(row["body"]), int(row["bone"]), int(row["triangle"])): index
                for index, row in enumerate(bone)}
    identified = {(row[0], identity[tuple(row[1:])]) for row in audit["witnesses"]}
    del identity
    source_records, contacts = native_pairs(original, faces, banks, bone)
    assert contacts == identified
    referenced = np.unique(faces)
    pinned = np.ones(len(original), dtype=bool)
    pinned[referenced] = False
    edges, count = np.unique(np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1),
                             return_counts=True, axis=0)
    pinned[edges[count != 2].ravel()] = True
    _, inverse, count = np.unique(original[referenced], axis=0, return_inverse=True, return_counts=True)
    pinned[referenced[count[inverse] > 1]] = True
    pinned |= original[:, 2] <= 0.003

    args.outdir.mkdir()
    history = []
    current = original.copy()
    current_pairs = contacts
    for iteration in range(3):
        if not current_pairs:
            break
        delta, step = fit(current, faces, bone, current_pairs, edges, pinned, original)
        step["iteration"] = iteration
        step["tests"] = []
        history.append(step)
        if delta is None or not np.any(delta):
            break
        improved = False
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = (current.astype(np.float64) + fraction * delta).astype(np.float32)
            changed = np.any(candidate != original, axis=1)
            assert np.array_equal(candidate[pinned], original[pinned])
            changed_faces = np.flatnonzero(changed[faces].any(axis=1))
            _, _, ref = geometry(original, faces[changed_faces])
            _, _, posed = geometry(candidate, faces[changed_faces])
            source_area = np.linalg.norm(ref, axis=1)
            posed_area = np.linalg.norm(posed, axis=1)
            ratio = posed_area / source_area
            cosine = np.einsum("ij,ij->i", ref, posed) / np.maximum(source_area * posed_area, 1e-60)
            test = {"fraction": fraction, "changed_nodes": int(changed.sum()),
                    "area_ratio_range": [float(ratio.min()), float(ratio.max())],
                    "minimum_normal_cosine": float(cosine.min())}
            step["tests"].append(test)
            if (ratio.min() < MINIMUM_AREA_RATIO or ratio.max() > MAXIMUM_AREA_RATIO or
                    cosine.min() < MINIMUM_NORMAL_COSINE):
                continue
            candidate_records, new_pairs = native_pairs(candidate, faces, banks, bone)
            test.update(pairs=len(new_pairs), newly_intersecting_pairs=len(new_pairs-contacts))
            if len(new_pairs) >= len(current_pairs):
                continue
            affected = set(map(int, changed_faces))
            src_self = owner._audit_self_pairs_touching_faces(source_records, affected)
            trial_self = owner._audit_self_pairs_touching_faces(candidate_records, affected)
            new_self = set(map(tuple, trial_self["triangle_pairs"])) - set(map(tuple, src_self["triangle_pairs"]))
            test["new_self_intersection_pairs"] = sorted(new_self)
            if new_self:
                continue
            current = candidate
            current_pairs = new_pairs
            improved = True
            test["retained"] = True
            print(json.dumps(test), flush=True)
            break
        if not improved:
            break

    active = np.flatnonzero(np.any(current != original, axis=1))
    archive = args.outdir / "source-adjustment.npz"
    if active.size:
        np.savez_compressed(archive, source_row=active, before_m=original[active], candidate_m=current[active])
    result = {
        "scope": "Offline source-geometry fit measured at the same native initialization pose; no runtime assembly or tissue mechanics admission",
        "baseline_sha256": digest(args.baseline),
        "before_crossings": len(contacts),
        "after_crossings": len(current_pairs),
        "new_bone_pairs": len(current_pairs-contacts),
        "unresolved_bone_pairs": [[i, int(bone[j]["body"]), int(bone[j]["bone"]), int(bone[j]["triangle"])]
                                  for i,j in sorted(current_pairs)],
        "affected_banks": dict(Counter(f"{int(bone[j]['body'])}/{int(bone[j]['bone'])}" for _, j in current_pairs)),
        "unchanged_pinned_rows": int(pinned.sum()),
        "changed_rows": len(active),
        "proposal_sha256": digest(archive) if active.size else None,
        "rounds": history,
        "wall_seconds": monotonic() - start,
    }
    (args.outdir / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key:value for key,value in result.items() if key not in ("rounds","unresolved_bone_pairs","affected_banks")}))

if __name__ == "__main__":
    main()
