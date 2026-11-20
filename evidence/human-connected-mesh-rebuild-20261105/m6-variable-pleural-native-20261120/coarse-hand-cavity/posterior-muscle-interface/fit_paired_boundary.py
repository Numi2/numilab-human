"""Fit the remaining posterior layer witnesses with the verified gluteal geometry."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit import BONE, SHELL, accepted_pose, digest, fem_boundaries, muscle_geometry, records, rows
from audit_bed_layer_surfaces import audit, install_for_float32, role_surfaces
from check_candidate import targets
from extension import determinant, quality
from fit_interface import solve
from fit_muscle_interface import MUSCLES, exact_pairs, prepare_target
from map_mass import regional_context


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("reference", "verified_muscles", "bone", "reader", "shared", "output"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    a.output.mkdir()
    install_for_float32()
    run = a.reference / "native-candidate-128"
    capture = a.reference / "native-candidate-04e" / "stdout.log"
    prep, _, _, cells = regional_context(run, a.shared)
    cook = a.reference / "native-cooked-float32.npz"
    with np.load(cook) as asset:
        base, ref = asset["nodes_current"], asset["nodes_reference"]
        assert np.array_equal(asset["cells"], cells)
    roles = role_surfaces(rows(run / "shell-mass.csv.tets.csv"))
    faces = np.array([face for group in roles.values() for face in group])
    selected = json.loads((a.verified_muscles / "assessment.json").read_text())["assessment"]["selected"]
    expected = set()
    for bank in MUSCLES:
        expected.update((bank, j, i) for i, j in selected[str(bank)]["retained_layer_pairs"])
    active = np.unique(faces[sorted({i for _, i, _ in expected})])
    n = len(base) // 3
    move = np.unique(faces[np.isin(faces, active).any(axis=1)])
    # Keep the outer and bonded-interface coordinates fixed. Only the deep
    # layer participates because all remaining witnesses are on that boundary.
    move = move[move >= 2*n]
    tested = np.flatnonzero(np.isin(faces, move).any(axis=1))
    trial_faces = faces[tested]
    affected = np.flatnonzero(np.isin(cells, move).any(axis=1))
    changed = np.isin(np.arange(len(base)), move)
    source, anchors, objects, debug = records(capture)
    assert np.array_equal(source[np.unique(objects[SHELL])], base)
    pose, _ = accepted_pose(run / "native/resting-body-motion-partition.csv", json.loads(
        debug["resting_initial_anatomical_registration"].partition("=")[2]))
    source_tri, _, _, _, begins = muscle_geometry(json.loads((run / "launch.json").read_text()), pose, a.reader)
    ordered = sorted(begins.values()) + [len(source_tri)]
    for i in MUSCLES:
        path = a.verified_muscles / f"gluteal-{i}.npz"
        assert digest(path) == selected[str(i)]["geometry_sha256"]
        start = begins[i]
        with np.load(path) as asset:
            source_tri[start:ordered[ordered.index(start)+1]] = asset["source_nodes"][asset["source_faces"]]
    fem, _ = fem_boundaries(source, objects, anchors)
    bones = np.memmap(a.bone, mode="r", dtype=BONE)
    scene = dict(bones=bones["vertices"], muscles=source_tri, other_fem=fem)
    pad = np.array([1e-7, 1e-7, .015001])
    box = (base[trial_faces].min(axis=(0, 1)) - pad, base[trial_faces].max(axis=(0, 1)) + pad)
    prepared = {k: (idx, prepare_target(v, idx)) for k, v in scene.items()
                if len(idx := targets(v, box))}
    before = exact_pairs(base, trial_faces, prepared)
    local_faces = {v: i for i, v in enumerate(tested)}
    target = {("muscles", local_faces[i], begins[b] + j) for b, i, j in expected}
    assert target <= before
    before_shell, _ = audit(base, roles, changed)
    preserve, results, best, candidates = set(), [], len(target), set(target)
    for iteration in range(6):
        posed, rest, meta = solve(base, ref, cells, move, np.full(len(move), 15.),
                                   affected, prep, trial_faces, scene, candidates, preserve)
        after = exact_pairs(posed, trial_faces, prepared)
        shell, _ = audit(posed, roles, changed)
        det = determinant(posed.astype(float), cells)
        ref_det = determinant(rest, cells)
        q = min(quality(posed.astype(float), cells, det).min(), quality(rest, cells, ref_det).min())
        added = after - before
        remaining = len(after & target)
        result = dict(iteration=iteration + 1, remaining_target_pairs=remaining,
                      added_pairs=dict(Counter(k for k, _, _ in added)),
                      shell_pairs=sum(map(len, shell.values())),
                      minimum_normalized_determinant=float(q),
                      minimum_determinant=float(min(det.min(), ref_det.min())), **meta)
        results.append(result)
        print(json.dumps(result), flush=True)
        if remaining < best and not added and shell == before_shell and q >= 1e-5:
            np.savez_compressed(a.output / "posterior-interface-fit.npz", nodes_current=posed,
                                nodes_reference=rest, cells=cells.astype(np.uint32))
            best = remaining
        if added <= preserve or remaining == 0:
            break
        preserve |= added
    summary = dict(source_sha256=digest(cook), bones_sha256=digest(a.bone),
                   verified_muscles_sha256=digest(a.verified_muscles / "assessment.json"),
                   initial_target_pairs=len(target), best_target_pairs=best,
                   adjustable_deep_nodes=move.tolist(), rounds=results)
    (a.output / "assessment.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
