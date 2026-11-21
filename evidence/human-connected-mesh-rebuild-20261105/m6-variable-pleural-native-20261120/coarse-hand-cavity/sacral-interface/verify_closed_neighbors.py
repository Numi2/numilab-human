"""Check both containment directions against the actual closed posterior neighbors."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from audit import BONE, accepted_pose, digest, muscle_geometry, rows
from build_posterior_obstacles import MUSCLES, NEIGHBORS
from verify_containment import classify, shell_surface, surface


def solid_from_triangles(triangles):
    points, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    return points, surface(points, inverse.reshape(-1, 3))


def pair_checks(p, target_model, target_nodes, subject_model):
    return {"muscle_in_neighbor": classify(p, target_model),
            "neighbor_in_muscle": classify(target_nodes, subject_model)}


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("reference", "bone", "reader", "candidate", "output"):
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    assert not a.output.exists()
    run = a.reference / "paired-native-128"
    shell = a.reference / "paired-06/posterior-interface-fit.npz"
    tets = [[int(r[f"parent_node{i}"]) for i in range(4)]
            for r in rows(run / "shell-mass.csv.tets.csv")]
    fat, fat_faces, fat_model = shell_surface(shell, tets)
    fat_boundary = fat[np.unique(fat_faces)]
    with (run / "stdout.log").open() as f:
        reg = json.loads(next(s.partition("=")[2] for s in f
                              if s.startswith("resting_initial_anatomical_registration=")))
    pose, _ = accepted_pose(run / "native/resting-body-motion-partition.csv", reg)
    muscle_tri, _, muscle_sha, closed, starts = muscle_geometry(
        json.loads((run / "launch.json").read_text()), pose, a.reader)
    offsets = sorted(starts.values()) + [len(muscle_tri)]
    bones = np.memmap(a.bone, mode="r", dtype=BONE)
    result = dict(witnesses={}, bone_sha256=digest(a.bone), shell_sha256=digest(shell),
                  muscle_source_sha256=muscle_sha, candidate_sha256={})
    for side in (29, 30):
        subject = a.candidate / f"gluteal-{side}.npz"
        with np.load(subject) as m:
            p, faces = m["source_nodes"], m["source_faces"]
        model = surface(p, faces)
        scope = {"bonded_shell": pair_checks(p, fat_model, fat_boundary, model)}
        for body, bank in NEIGHBORS[side]:
            target = bones[(bones["body"] == body) & (bones["bone"] == bank)]["vertices"]
            both, tree = solid_from_triangles(target)
            scope[f"bone-{body}-{bank}"] = pair_checks(p, tree, both, model)
        for bank in MUSCLES[side]:
            assert closed[bank]
            start = starts[bank]
            end = offsets[offsets.index(start)+1]
            both, tree = solid_from_triangles(muscle_tri[start:end])
            scope[f"muscle-{bank}"] = pair_checks(p, tree, both, model)
        result["candidate_sha256"][str(side)] = digest(subject)
        result["witnesses"][str(side)] = scope
        print(side, {key: sum(len(x) for x in checks.values()) for key, checks in scope.items()}, flush=True)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
