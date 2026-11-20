"""Fit a connected posterior shell using actual contact witnesses and determinant bounds."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from audit import BONE, accepted_pose, digest, fem_boundaries, muscle_geometry, records, rows
from audit_bed_layer_surfaces import audit, install_for_float32, owner, role_surfaces
from check_candidate import targets
from extension import barycentric, determinant, overlap_xy, quality, signed_height_gradients, wedge
from map_mass import regional_context


def index_pairs(shell, faces, target, target_ids):
    a, _ = owner.float32_surface_records(shell, faces)
    triangles = target[target_ids]
    b, _ = owner.float32_surface_records(
        triangles.reshape(-1, 3), np.arange(triangles.size // 3).reshape(-1, 3))
    pairs = owner._audit_pair_prepared_first(owner._prepare_surface_aabb(a), b)["triangle_pairs"]
    return {(int(i), int(target_ids[j])) for i, j in pairs}


def separation_rows(a, b):
    # All requested faces lie below the internal anatomy along the bed normal.
    # Avoid a near-singular projected triangle, then certify exact Float32
    # geometry independently of this linear approximation.
    if any(abs(wedge(t[1, :2] - t[0, :2], t[2, :2] - t[0, :2])) <= 1e-13
           for t in (a, b)):
        return None
    p = overlap_xy(a[:, :2], b[:, :2])
    if not len(p):
        return None
    weights = barycentric(p, a[:, :2])
    gap = barycentric(p, b[:, :2]) @ b[:, 2] - weights @ a[:, 2]
    return weights, gap


class Constraints:
    def __init__(self, nodes):
        self.length = len(nodes)
        self.id = {int(node): i for i, node in enumerate(nodes)}
        self.rhs, self.rows, self.cols, self.values = [], [], [], []

    def add(self, node_ids, coeff, upper, relax=None):
        row = len(self.rhs)
        self.rhs.append(upper)
        for node, value in zip(node_ids, coeff, strict=True):
            if node in self.id and value:
                self.rows.append(row)
                self.cols.append(self.id[node])
                self.values.append(value)
        if relax is not None:
            self.rows.append(row)
            self.cols.append(self.length + relax)
            self.values.append(-1.)


def solve(base, ref, cells, nodes, limits, near, compiler, faces, groups, cuts, preserve):
    # z increments in mm. Preserving x and y makes the cell determinant an
    # affine function, so hard geometric bounds hold through the full update.
    inequality = Constraints(nodes)
    for points in (base, ref):
        local = cells[near]
        d0 = determinant(points, local)
        grad = signed_height_gradients(points, local)
        for tet, g in zip(local, grad / d0[:, None], strict=True):
            inequality.add(tet, -g / 1000., .55)

    # Enforce the typed donor/material volume intervals needed to route the
    # originally measured mass at the existing scene density limits.
    pool = compiler.group_for_parent
    volume = np.bincount(pool, weights=determinant(ref, cells))
    lo, hi = compiler.determinant_bounds()
    grad = signed_height_gradients(ref, cells[near])
    for group in np.unique(pool[near]):
        mask = pool[near] == group
        tets = cells[near][mask].ravel()
        slopes = grad[mask].ravel() / (1000. * volume[group])
        d = volume[group]
        # Preserve density headroom for the later distance-limited mass LP.
        # If the input is already closer to one limit, keep its clearance.
        reserve = 1e-7 * d
        lower, upper = min(lo[group] + reserve, d), max(hi[group] - reserve, d)
        inequality.add(tets, slopes, (upper - d) / d)
        inequality.add(tets, -slopes, (d - lower) / d)

    count = 0
    for group, i, j in sorted(cuts):
        f = faces[i]
        eq = separation_rows(base[f], groups[group][j])
        if eq is None:
            continue
        w, g = eq
        for weight, gap in zip(w, g, strict=True):
            inequality.add(f, weight, (gap - .0001) * 1000., count)
        count += 1
    for group, i, j in sorted(preserve):
        f = faces[i]
        eq = separation_rows(base[f], groups[group][j])
        if eq is None or (min(eq[1]) <= 0. and max(eq[1]) >= 0.):
            for node in f:
                inequality.add([node], [1.], 0.)
                inequality.add([node], [-1.], 0.)
            continue
        w, g = eq
        direction = 1. if min(g) > 0. else -1.
        for weight, gap in zip(w, g, strict=True):
            # Keep half the original clearance if it was already below
            # the registration tolerance. The starting geometry is feasible.
            margin = min(.00015, .5 * abs(gap))
            inequality.add(f, weight * direction, (abs(gap) - margin) * 1000.)
    absolute = count
    for i, node in enumerate(nodes):
        inequality.add([node], [1.], 0., absolute + i)
        inequality.add([node], [-1.], 0., absolute + i)
    total = len(nodes) + count + len(nodes)
    objective = np.r_[np.zeros(len(nodes)), np.ones(count), np.full(len(nodes), .035)]
    matrix = coo_matrix((inequality.values, (inequality.rows, inequality.cols)),
                        shape=(len(inequality.rhs), total)).tocsr()
    bounds = [(-v, v * .35) for v in limits] + [(0., None)] * (count + len(nodes))
    result = linprog(objective, A_ub=matrix, b_ub=inequality.rhs,
                     bounds=bounds, method="highs", options={"time_limit": 10.})
    assert result.success, result.message
    delta = np.zeros_like(base, dtype=float)
    delta[nodes, 2] = result.x[:len(nodes)] / 1000.
    return (base + delta).astype(np.float32), ref + delta, {
        "slack_witness_count": int(np.sum(result.x[len(nodes):len(nodes)+count] > 1e-5)),
        "covered_witness_constraints": count, "maximum_displacement_m": float(np.max(abs(delta))),
    }


def fit(run, candidate, bone_pin, reader, shared, glute, out, *, pelvis_only=False, pelvis_bank=False):
    install_for_float32()
    compiler, _, _, cells = regional_context(run, shared)
    with np.load(candidate) as data:
        base, ref = data["nodes_current"], data["nodes_reference"]
        assert np.array_equal(data["cells"], cells)
    roles = role_surfaces(rows(run / "shell-mass.csv.tets.csv"))
    faces = np.array([face for fs in roles.values() for face in fs])
    by_role = [role for role, fs in roles.items() for _ in fs]
    face_ids = {tuple(sorted(f)): i for i, f in enumerate(faces)}
    bones = np.memmap(bone_pin, dtype=BONE, mode="r")
    scene_nodes, anchors, scene_cells, evidence = records(run / "stdout.log")
    pose, _ = accepted_pose(run / "native/resting-body-motion-partition.csv",
                            json.loads(evidence["resting_initial_anatomical_registration"].partition("=")[2]))
    muscle, muscle_id, tissue_sha, _, starts = muscle_geometry(
        json.loads((run / "launch.json").read_text()), pose, reader)
    fem, fem_id = fem_boundaries(scene_nodes, scene_cells, anchors)
    with glute.open() as f:
        screen = json.load(f)
    assert screen["tissue_sha256"] == tissue_sha and screen["shell_sha256"] == digest(candidate)
    seed_faces = {face_ids[tuple(sorted(v["target"]["nodes"]))] for bank in screen["banks"]
                  for v in bank["skin_fat"]["witnesses"]}
    # Include the existing left pelvis crossings on a deep face adjacent to
    # the fitted support region. Other pelvis triangles are audited too.
    if pelvis_only or pelvis_bank:
        seed_faces.clear()
    seed_faces.add(face_ids[(6094, 6349, 6430)])
    n = len(base) // 3
    active = np.unique(faces[sorted(seed_faces)])
    assert active.min() >= n
    # One triangle ring distributes the correction across the welded column
    # surface, while outer skin and unrelated patches remain exactly fixed.
    neighbor_faces = faces[np.isin(faces, active).any(axis=1)]
    nodes = np.unique(neighbor_faces)
    nodes = nodes[nodes >= n]
    active_faces = np.flatnonzero(np.isin(faces, nodes).any(axis=1))
    active_by_global = {v: i for i, v in enumerate(active_faces)}
    f = faces[active_faces]
    near = np.flatnonzero(np.isin(cells, nodes).any(axis=1))
    limits = np.where(nodes < 2*n, 3., 15.)
    expand = .015001
    box = (base[f].min(axis=(0, 1)) - [0., 0., expand],
           base[f].max(axis=(0, 1)) + [0., 0., expand])
    geometry = {"bone": bones["vertices"], "muscle": muscle, "fem": fem}
    cropped = {k: targets(p, box) for k, p in geometry.items()}
    identities = {
        "bone": lambda i: {k: int(bones[k][i]) for k in ("body", "bone", "triangle")},
        "muscle": muscle_id,
        "fem": fem_id.__getitem__,
    }

    def verify(world):
        return {(group, i, j) for group in geometry
                for i, j in index_pairs(world, f, geometry[group], cropped[group])}

    initial = verify(base)
    aim = set() if (pelvis_only or pelvis_bank) else {("muscle", active_by_global[face_ids[tuple(sorted(v["target"]["nodes"]))]],
            starts[bank["stable_id"]] + v["gluteal_source_face"]) for bank in screen["banks"]
           for v in bank["skin_fat"]["witnesses"]}
    aim |= {hit for hit in initial if hit[0] == "bone" and
            tuple(f[hit[1]]) == (6094, 6349, 6430) and int(bones["body"][hit[2]]) == 128}
    if pelvis_bank:
        aim |= {hit for hit in initial if hit[0] == "bone"
                and (int(bones["body"][hit[2]]), int(bones["bone"][hit[2]])) == (128, 19)}
    assert aim <= initial and len(aim) >= (7 if pelvis_only or pelvis_bank else 257)
    print(json.dumps({"baseline_pairs": len(initial), "target_pairs": len(aim),
                      "target_faces": len({i for _, i, _ in aim})}), flush=True)
    preserve = set()
    stages = []
    best = None
    unmodified = np.any(base != np.array(
        [[v[f"current_{axis}_m"] for axis in "xyz"] for v in rows(run / "shell-mass.csv.nodes.csv")],
        dtype=np.float32), axis=1)
    initial_shell, _ = audit(base, roles, unmodified | np.isin(np.arange(len(base)), nodes))
    for iteration in range(6):
        world, reference, stage = solve(base, ref, cells, nodes, limits, near, compiler, f, geometry, aim, preserve)
        hits = verify(world)
        new = hits - initial
        shell, _ = audit(world, roles, unmodified | np.isin(np.arange(len(base)), nodes))
        q = [determinant(p, cells) for p in (world, reference)]
        change = np.any(world != base, axis=1)
        stage.update({"iteration": iteration, "target_pairs": len(aim & hits),
                      "all_pairs": len(hits), "added_pairs": len(new),
                      "added_by_bank": dict(Counter((str((g, identities[g](j).get("stable_id",
                                         identities[g](j).get("body"))))) for g, i, j in new)),
                      "shell_self_intersections": sum(map(len, shell.values())),
                      "minimum_determinant_m3": float(min(min(q[0]), min(q[1]))),
                      "minimum_shape": float(quality(world, cells, q[0]).min()),
                      "changed_nodes": int(change.sum())})
        print(json.dumps(stage), flush=True)
        stages.append(stage)
        if not new and shell == initial_shell and stage["minimum_determinant_m3"] > 0. and (
                best is None or stage["target_pairs"] < best[0]["target_pairs"]):
            best = (stage, world, reference, hits)
        if not new:
            break
        preserve.update(new)
    output = {"anchor_candidate_sha256": digest(candidate), "baseline_pairs": len(initial),
              "baseline_target_pairs": len(aim), "stages": stages, "best": None}
    if best is not None:
        stage, world, reference, hits = best
        file = out / "posterior-interface-fit.npz"
        with file.open("xb") as fd:
            np.savez_compressed(fd, nodes_current=world, nodes_reference=reference,
                               cells=cells.astype(np.uint32))
        output.update(best={"iteration": stage["iteration"], "candidate_sha256": digest(file),
                            "removed_by_group": dict(Counter(k for k, _, _ in initial - hits)),
                            "remaining_witnesses": [{"kind": g, "face": f[i].tolist(),
                                                     "role": by_role[active_faces[i]],
                                                     "target": identities[g](j)}
                                                    for g, i, j in sorted(aim & hits)]})
    with (out / "posterior-interface-fit.json").open("x") as fd:
        json.dump(output, fd, indent=2, allow_nan=False)
        fd.write("\n")


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for k in ("run", "candidate", "bone_pin", "reader", "shared", "glute", "out"):
        p.add_argument(k, type=Path)
    p.add_argument("--pelvis-only", action="store_true")
    p.add_argument("--pelvis-bank", action="store_true")
    fit(**vars(p.parse_args()))
