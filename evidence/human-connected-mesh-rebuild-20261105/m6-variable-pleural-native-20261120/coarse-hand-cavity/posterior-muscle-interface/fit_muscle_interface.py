"""Fit each registered gluteal volume to the cooked posterior layer boundary."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from audit import BONE, SHELL, accepted_pose, digest, fem_boundaries, muscle_geometry, records, rows
from audit_bed_layer_surfaces import install_for_float32, owner, role_surfaces
from check_candidate import targets
from extension import determinant, quality, signed_height_gradients
from fit_interface import Constraints, separation_rows


MUSCLES = (29, 30)
TRAVEL_M = .0075
MARGIN_M = .0001


def prepare_target(points, use):
    triangles = points[use]
    records, _ = owner.float32_surface_records(
        triangles.reshape(-1, 3), np.arange(triangles.size // 3).reshape(-1, 3))
    return records


def exact_pairs(points, faces, targets_by_owner):
    records, _ = owner.float32_surface_records(points, faces)
    prepared = owner._prepare_surface_aabb(records)
    return {(group, int(face), int(ids[other]))
            for group, (ids, records) in targets_by_owner.items()
            for face, other in owner._audit_pair_prepared_first(prepared, records)["triangle_pairs"]}


def optimize(base, cells, faces, free, near, shell, cuts, surrounding, keep):
    constraints = Constraints(free)
    local = cells[near]
    det = determinant(base, cells)
    gradient = signed_height_gradients(base, local)
    for cell, g, value in zip(local, gradient, det[near], strict=True):
        constraints.add(cell, -g / (1000. * value), .45)
    units = gradient.ravel() / (1000. * det.sum())
    constraints.add(local.ravel(), units, 0.)
    constraints.add(local.ravel(), -units, 0.)
    used = 0
    for i, j in sorted(cuts):
        face = faces[i]
        eq = separation_rows(base[face], shell[j])
        if eq is None:
            continue
        weights, gap = eq
        for coeff, g in zip(weights, gap, strict=True):
            constraints.add(face, -coeff, (-g - MARGIN_M) * 1000., used)
        used += 1
    for group, i, j in sorted(keep):
        face = faces[i]
        eq = separation_rows(base[face], surrounding[group][j])
        if eq is None:
            for node in face:
                constraints.add([node], [1.], 0.)
                constraints.add([node], [-1.], 0.)
            continue
        weights, gap = eq
        if not (gap.min() > 0. or gap.max() < 0.):
            continue
        direction = 1. if gap.min() > 0. else -1.
        for coeff, g in zip(weights, gap, strict=True):
            constraints.add(face, direction * coeff, (.5 * abs(g)) * 1000.)
    for i, node in enumerate(free):
        constraints.add([node], [1.], 0., used + i)
        constraints.add([node], [-1.], 0., used + i)
    total = 2 * len(free) + used
    operator = coo_matrix((constraints.values, (constraints.rows, constraints.cols)),
                          shape=(len(constraints.rhs), total)).tocsr()
    objective = np.r_[np.zeros(len(free)), np.ones(used), np.full(len(free), .015)]
    bound = TRAVEL_M * 1000.
    sol = linprog(objective, A_ub=operator, b_ub=constraints.rhs, method="highs",
                  bounds=[(-bound, bound)] * len(free) + [(0., None)] * (used + len(free)),
                  options={"time_limit": 12.})
    if not sol.success:
        raise RuntimeError(sol.message)
    result = base.copy()
    result[free, 2] += sol.x[:len(free)] / 1000.
    return result.astype(np.float32), used, int(np.count_nonzero(sol.x[len(free):len(free)+used] > 1e-5))


def solve_muscle(path, geometry, roles, out, protected_faces=()):
    with np.load(path) as asset:
        base, cells = asset["nodes"], asset["cells"]
        source, source_faces = asset["source_nodes"], asset["source_faces"]
        vertex_ids = asset["source_vertex_ids"]
    lookup = {tuple(p): i for i, p in enumerate(base)}
    indices = np.array([lookup[tuple(p)] for p in source])
    faces = indices[source_faces]
    pad = [1.e-7, 1.e-7, TRAVEL_M + 1.e-7]
    box = base.min(axis=0) - pad, base.max(axis=0) + pad
    target_records = {k: (ids, prepare_target(v, ids)) for k, v in geometry.items()
                      if (ids := targets(v, box)).size}
    initial = exact_pairs(base, faces, target_records)
    start = {(i, j) for group, i, j in initial if group == "layers"}
    frozen = np.unique(faces[sorted({i for group, i, _ in initial if group == "bones"})])
    neighborhood = np.unique(faces[sorted({i for i, _ in start})])
    for _ in range(2):
        neighborhood = np.unique(cells[np.isin(cells, neighborhood).any(axis=1)])
    protected = np.unique(faces[np.asarray(protected_faces, dtype=int)])
    fixed = np.union1d(frozen, protected)
    free = np.setdiff1d(neighborhood, fixed)
    assert len(free) and start and np.isfinite(base).all()
    near = np.flatnonzero(np.isin(cells, free).any(axis=1))
    control_det = determinant(base, cells)
    control_quality = quality(base, cells, control_det)
    assert control_det.min() > 0.
    cuts, keep = set(start), set()
    rounds = []
    best_key = None
    for i in range(4):
        candidate, constraints, slack = optimize(base, cells, faces, free, near, geometry["layers"], cuts, geometry, keep)
        assert np.array_equal(base[fixed], candidate[fixed])
        det = determinant(candidate.astype(float), cells)
        geo_quality = quality(candidate.astype(float), cells, det)
        after = exact_pairs(candidate, faces, target_records)
        layer_pairs = {(a, b) for group, a, b in after if group == "layers"}
        new_outside = {p for p in after - initial if p[0] != "layers"}
        records, _ = owner.float32_surface_records(candidate, faces)
        moved = np.any(base != candidate, axis=1)
        changed_faces = set(np.flatnonzero(moved[faces].any(axis=1)).tolist())
        self_pairs = owner._audit_self_pairs_touching_faces(records, changed_faces)["triangle_pairs"]
        trial = dict(round=i+1, cells_nonpositive=int((det <= 0.).sum()),
                     cells_low_quality=int((geo_quality < 1e-5).sum()),
                     minimum_determinant_ratio=float(min(det / control_det)),
                     minimum_quality=float(geo_quality.min()),
                     absolute_relative_volume_change=float(abs(det.sum()/control_det.sum()-1.)),
                     moved_nodes=int(moved.sum()), max_move_m=float(np.max(abs(candidate-base))),
                     layer_pairs=dict(Counter(roles[j] for _, j in layer_pairs)),
                     new_outside_pairs=dict(Counter(p[0] for p in new_outside)),
                     total_pairs=dict(Counter(p[0] for p in after)),
                     new_self_pairs=len(self_pairs), cut_rows=constraints, slack_rows=slack)
        rounds.append(trial)
        exact_gates = not trial["cells_nonpositive"] and not trial["cells_low_quality"] and trial["absolute_relative_volume_change"] <= 1.e-6
        key = (not exact_gates, len(new_outside)+len(self_pairs), len(layer_pairs))
        if best_key is None or key < best_key:
            best_key = key
            selected = dict(trial, layer_witnesses=sorted(layer_pairs),
                            new_outside_witnesses=sorted(new_outside), self_witnesses=self_pairs)
            np.savez_compressed(out / path.name, nodes=candidate, cells=cells,
                                source_nodes=candidate[indices], source_faces=source_faces,
                                source_vertex_ids=vertex_ids)
        print(json.dumps(trial), flush=True)
        next_cuts = cuts | layer_pairs
        next_keep = keep | new_outside
        if key == (False, 0, 0) or (next_cuts == cuts and next_keep == keep):
            break
        cuts, keep = next_cuts, next_keep
    return dict(input_sha256=digest(path), baseline_pairs=dict(Counter(p[0] for p in initial)),
                baseline_layers=dict(Counter(roles[j] for _, j in start)),
                original_minimum_quality=float(control_quality.min()), original_volume_m3=float(control_det.sum()/6.),
                bone_frozen_nodes=len(frozen), protected_source_faces=list(protected_faces),
                adjustable_nodes=len(free), rounds=rounds, selected=selected)


def main():
    parser = ArgumentParser(description=__doc__)
    for label in ("prior", "bone_path", "reader", "output"):
        parser.add_argument(label, type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--proposals", type=Path)
    mode.add_argument("--protect-folds", type=Path)
    args = parser.parse_args()
    install_for_float32()
    v = args.prior
    run = v / "native-candidate-128"
    geometry = {}
    with np.load(v / "native-cooked-float32.npz") as asset:
        shell = asset["nodes_current"]
    layers = role_surfaces(rows(run / "shell-mass.csv.tets.csv"))
    roles = {i: key for i, key in enumerate(k for k, fs in layers.items() for _ in fs)}
    geometry["layers"] = shell[np.array([face for fs in layers.values() for face in fs])]
    bones = np.memmap(args.bone_path, mode="r", dtype=BONE)
    geometry["bones"] = bones["vertices"]
    capture = v / "native-candidate-04e" / "stdout.log"
    nodes, anchors, cells, evidence = records(capture)
    shell_nodes = np.unique(cells[SHELL])
    assert np.array_equal(nodes[shell_nodes], shell)
    geometry["other_fem"], _ = fem_boundaries(nodes, cells, anchors)
    registration = json.loads(evidence["resting_initial_anatomical_registration"].partition("=")[2])
    pose, pose_check = accepted_pose(run / "native/resting-body-motion-partition.csv", registration)
    source, _, source_sha, _, start = muscle_geometry(json.loads((run / "launch.json").read_text()), pose, args.reader)
    args.output.mkdir()
    use = np.ones(len(source), dtype=bool)
    offsets = sorted(start.values()) + [len(source)]
    for i in MUSCLES:
        offset = start[i]
        use[offset:offsets[offsets.index(offset)+1]] = False
    geometry["other_muscles"] = source[use]
    if args.proposals is None:
        assessment = {}
        previous = {} if args.protect_folds is None else json.loads(args.protect_folds.read_text())["assessment"]
        for stable_id, other in (MUSCLES, MUSCLES[::-1]):
            with np.load(v / f"gluteal-{other}.npz") as asset:
                geometry["other_gluteus"] = asset["source_nodes"][asset["source_faces"]]
            pairs = previous.get(str(stable_id), {}).get("selected", {}).get("self_witnesses", ())
            protected = sorted({face for pair in pairs for face in pair})
            assessment[str(stable_id)] = solve_muscle(
                v / f"gluteal-{stable_id}.npz", geometry, roles, args.output, protected)
    else:
        from verify_bilateral import check
        assessment = check(v, args.proposals, geometry, roles, args.output)
    result = dict(bone_sha256=digest(args.bone_path), shell_sha256=digest(v / "native-cooked-float32.npz"),
                  posed_source_sha256=source_sha, fem_capture_sha256=digest(capture),
                  pose_check=pose_check, assessment=assessment)
    (args.output / "assessment.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
