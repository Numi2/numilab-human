"""Prepare a margin around the exact source sacrum for interface authoring."""
from argparse import ArgumentParser
from pathlib import Path

from manifold3d import Manifold, Mesh64
import numpy as np

from audit import BONE, closed_surface, digest
from fit_muscle_interface import MARGIN_M


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("bone", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    bones = np.memmap(args.bone, dtype=BONE, mode="r")
    selected = bones[(bones["body"] == 1) & (bones["bone"] == 1)]
    points, inverse = np.unique(selected["vertices"].reshape(-1, 3),
                                axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    assert closed_surface(points, faces)
    source = Manifold(Mesh64(points.astype(float), faces.astype(np.uint64)))
    padded = source.minkowski_sum(Manifold.cube((2 * MARGIN_M,) * 3, True))
    mesh = padded.to_mesh64()
    assert padded.num_tri() and padded.volume() > source.volume() > 0.
    np.savez_compressed(args.output, vertices=mesh.vert_properties[:, :3],
                        faces=mesh.tri_verts.astype(np.int64),
                        source_body=1, source_bone=1, expansion_m=MARGIN_M,
                        source_sha256=digest(args.bone))
    print(source.num_tri(), padded.num_tri(), source.volume(), padded.volume())


if __name__ == "__main__":
    main()
