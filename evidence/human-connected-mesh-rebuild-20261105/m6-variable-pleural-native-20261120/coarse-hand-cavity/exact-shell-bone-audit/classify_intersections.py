"""Classify exact transverse shell–bone segments separately from contact coincidences."""
from argparse import ArgumentParser
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from audit import TRIANGLE_ROW, emitted, owner


def points_on_plane(first, second):
    normal = owner._cross(owner._sub(first[1], first[0]), owner._sub(first[2], first[0]))
    return tuple(sum(a * b for a, b in zip(normal, owner._sub(point, first[0])))
                 for point in second)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    before, _ = emitted(args.run)
    proof = json.loads(args.audit.read_text())
    bones = np.memmap(args.run / "physical-bone-initial.bin", dtype=TRIANGLE_ROW, mode="r")
    ids = {(int(row["body"]), int(row["bone"]), int(row["triangle"])): index
           for index, row in enumerate(bones)}
    assert len(ids) == len(bones)
    shell = [owner.float32_point_lattice_key(point) for point in before]

    @lru_cache(maxsize=32768)
    def bone(index):
        return tuple(owner.float32_point_lattice_key(point) for point in bones[index]["vertices"])

    labels = {}
    for role, _, nodes, body, stable_id, triangle in proof["witnesses"]:
        s = tuple(shell[index] for index in nodes)
        b = bone(ids[body, stable_id, triangle])
        first = points_on_plane(s, b)
        second = points_on_plane(b, s)
        intersection = owner._triangle_intersection_points_integer(s, b)
        assert intersection
        if len(intersection) == 1:
            contact = "point"
        elif not any(first) and not any(second):
            contact = "coplanar"
        elif min(first) < 0 < max(first) and min(second) < 0 < max(second):
            contact = "transverse_segment"
        else:
            contact = "boundary_segment"
        labels.setdefault(role, Counter())[contact] += 1

    assert sum(sum(values.values()) for values in labels.values()) == proof["crossing_pairs"]
    with args.output.open("x") as stream:
        json.dump(labels, stream, indent=2)
        stream.write("\n")
    print(json.dumps(labels))


if __name__ == "__main__":
    main()
