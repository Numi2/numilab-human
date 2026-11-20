"""Audit the compiled shell against its actual registered physical bone surfaces."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import re
import sys
from time import monotonic

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audit_bed_layer_surfaces import install_for_float32, owner, role_surfaces
from build_hand_trial import digest
from compare_bank_trials import emitted
from fit_hand_cavity import PARENT, table

TRIANGLE_ROW = np.dtype([("body", "<u4"), ("bone", "<u4"), ("triangle", "<u4"),
                         ("vertices", "<f4", (3, 3))])


def records(vertices):
    for index, triangle in enumerate(vertices):
        tri = tuple(owner.float32_point_lattice_key(point) for point in triangle)
        lo = tuple(map(min, zip(*tri)))
        hi = tuple(map(max, zip(*tri)))
        yield (tri, lo, hi, index, ())


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = monotonic()
    install_for_float32()
    with (args.run / "stdout.log").open() as stream:
        inventory = next(line for line in stream
                         if line.startswith("whole_body_skin_bone_triangle_contact_inventory="))
    expected_bodies = int(re.search(r"\beligible_registered_bone_bodies=(\d+)", inventory)[1])
    expected_faces = int(re.search(r"\bregistered_bone_triangles=(\d+)", inventory)[1])
    data_path = args.run / "physical-bone-initial.bin"
    assert data_path.stat().st_size == TRIANGLE_ROW.itemsize * expected_faces
    bone = np.memmap(data_path, dtype=TRIANGLE_ROW, mode="r")
    identities = np.unique(bone[["body", "bone", "triangle"]])
    assert len(identities) == len(bone)
    assert np.isfinite(bone["vertices"]).all()
    bodies, starts, counts = np.unique(bone["body"], return_index=True, return_counts=True)
    assert len(bodies) == expected_bodies and np.array_equal(starts, np.r_[0, counts.cumsum()[:-1]])
    node_path = args.run / "initial-skin-fem.csv"
    points, _ = emitted(args.run)
    tets = PARENT.with_name(PARENT.name + ".tets.csv")
    groups = role_surfaces(table(tets))
    roles = [role for role, faces in groups.items() for _ in faces]
    faces = [face for bank in groups.values() for face in bank]
    shell_records, _ = owner.float32_surface_records(points, faces)
    shell = owner._prepare_surface_aabb(shell_records)
    counts_by_role = Counter()
    witnesses, by_bank = [], []
    for body, begin, count in zip(bodies, starts, counts):
        bank = bone[begin:begin + count]
        measured = owner._audit_pair_prepared_first(shell, records(bank["vertices"]))
        for shell_face, bone_local in measured["triangle_pairs"]:
            source = bank[bone_local]
            role = roles[shell_face]
            counts_by_role[role] += 1
            witnesses.append([role, shell_face, list(faces[shell_face]), int(body),
                              int(source["bone"]), int(source["triangle"])])
        if measured["count"]:
            by_bank.append([int(body), measured["count"], measured["aabb_candidate_pairs"]])
    result = {
        "scope": "Closed-box and exact Float32 triangle predicates against separate surfaces at native startup; point containment and continuous motion are separate checks",
        "source": {p.name: digest(p) for p in (data_path, node_path, tets, args.run / "launch.json")},
        "shell_faces": {role: len(group) for role, group in groups.items()},
        "bone_triangles": expected_faces,
        "bone_bodies": expected_bodies,
        "crossing_pairs": len(witnesses),
        "pairs_by_role": dict(counts_by_role),
        "affected_banks": by_bank,
        "witness_columns": ["role", "shell_face", "shell_local_nodes", "body", "bone", "bone_source_face"],
        "witnesses": witnesses,
        "audit_wall_seconds": monotonic() - start,
    }
    assert len(witnesses) == sum(row[1] for row in by_bank)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: val for key, val in result.items() if key != "witnesses"}))


if __name__ == "__main__":
    main()
