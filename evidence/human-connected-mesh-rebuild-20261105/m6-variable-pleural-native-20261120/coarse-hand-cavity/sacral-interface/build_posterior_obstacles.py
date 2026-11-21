"""Combine the registered bone and bonded-shell solids at each gluteal interface."""
from argparse import ArgumentParser
import json
from pathlib import Path

from manifold3d import Manifold, Mesh64, OpType
import numpy as np

from audit import BONE, accepted_pose, closed_surface, digest, muscle_geometry, rows
from fit_muscle_interface import MARGIN_M
from verify_containment import shell_surface


NEIGHBORS = {29: ((1, 1), (128, 18), (131, 2)),
             30: ((1, 1), (128, 19), (145, 3))}
MUSCLES = {29: (13, 17, 43, 53, 63), 30: (14, 18, 44, 54, 64)}


def pad(points, faces):
    assert closed_surface(points, faces)
    solid = Manifold(Mesh64(points.astype(float), faces.astype(np.uint64)))
    assert solid.num_tri() > 0 and solid.volume() > 0
    envelope = solid.minkowski_sum(Manifold.cube((2 * MARGIN_M,) * 3, True))
    assert envelope.num_tri() > 0 and envelope.volume() > solid.volume()
    return envelope


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("bone", "shell", "shell_cells", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--reader", type=Path)
    parser.add_argument("--simplification-m", type=float, default=0.)
    args = parser.parse_args()
    assert 0 <= args.simplification_m < MARGIN_M
    assert (args.native is None) == (args.reader is None)
    args.output.mkdir()
    bones = np.memmap(args.bone, dtype=BONE, mode="r")
    tets = [[int(r[f"parent_node{i}"]) for i in range(4)] for r in rows(args.shell_cells)]
    points, faces, _ = shell_surface(args.shell, tets)
    fat_margin = pad(points, faces)
    pads = {}
    for pair in sorted({key for keys in NEIGHBORS.values() for key in keys}):
        body, bank = pair
        bank_triangles = bones[(bones["body"] == body) & (bones["bone"] == bank)]["vertices"]
        source, inverse = np.unique(bank_triangles.reshape(-1, 3), axis=0, return_inverse=True)
        pads[pair] = pad(source, inverse.reshape(-1, 3))
        print("bone", pair, pads[pair].num_tri(), flush=True)
    muscles = {}
    muscle_sha = ""
    if args.native is not None:
        with (args.native / "stdout.log").open() as f:
            reg = json.loads(next(s.partition("=")[2] for s in f
                                  if s.startswith("resting_initial_anatomical_registration=")))
        pose, _ = accepted_pose(args.native / "native/resting-body-motion-partition.csv", reg)
        geometry, _, muscle_sha, closed, starts = muscle_geometry(
            json.loads((args.native / "launch.json").read_text()), pose, args.reader)
        offsets = sorted(starts.values()) + [len(geometry)]
        for bank in sorted({i for group in MUSCLES.values() for i in group}):
            start = starts[bank]
            stop = offsets[offsets.index(start) + 1]
            assert closed[bank]
            points, inverse = np.unique(geometry[start:stop].reshape(-1, 3),
                                        axis=0, return_inverse=True)
            muscles[bank] = pad(points, inverse.reshape(-1, 3))
            print("muscle", bank, muscles[bank].num_tri(), flush=True)
    bone_id = digest(args.bone)
    shell_id = digest(args.shell)
    for muscle, neighbors in NEIGHBORS.items():
        surrounding = list(MUSCLES[muscle]) if muscles else []
        union = Manifold.batch_boolean([fat_margin, *(pads[p] for p in neighbors),
                                        *(muscles[i] for i in surrounding)], OpType.Add)
        assert union.num_tri() >= fat_margin.num_tri() // 2 and union.volume() > fat_margin.volume()
        if args.simplification_m:
            union = union.simplify(args.simplification_m)
            assert union.num_tri() > 0 and union.volume() > 0
        mesh = union.to_mesh64()
        np.savez_compressed(args.output / f"obstacles-{muscle}.npz",
                            vertices=mesh.vert_properties[:, :3], faces=mesh.tri_verts.astype(np.int64),
                            neighbor_banks=neighbors, expansion_m=MARGIN_M,
                            source_sha256=bone_id, shell_sha256=shell_id,
                            neighboring_muscles=surrounding, muscle_sha256=muscle_sha,
                            simplification_m=args.simplification_m)
        print("union", muscle, union.num_tri(), union.volume(), flush=True)


if __name__ == "__main__":
    main()
