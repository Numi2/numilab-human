"""Compare the registered detailed source and reduced shell in one native pose."""
from argparse import ArgumentParser
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "exact-shell-bone-audit-032"))
sys.path.insert(0, str(Path.home() / "Work/skin-fat-native-integration-209/tools"))
from audit import TRIANGLE_ROW, digest, install_for_float32, owner, records
from prepare_candidate033_d37 import load_skin


def load_registered_points(path, count):
    points = np.empty((count, 3), dtype=np.float32)
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        rows = 0
        for row in reader:
            assert int(row["vertex"]) == rows and rows < count
            points[rows] = tuple(float(row[axis]) for axis in ("wx", "wy", "wz"))
            rows += 1
    assert rows == count and np.isfinite(points).all()
    return points


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    start = monotonic()
    install_for_float32()
    source_proof = ROOT / "exact-shell-bone-audit-032/exact-startup-crossings.json"
    coarse = json.loads(source_proof.read_text())
    state = json.loads((args.run / "execution.json").read_text())
    assert state["exit"] == 0
    launch = args.run / "launch.json"
    argv = json.loads(launch.read_text())["argv"]
    skin_path = Path(argv[argv.index("--skin-payload") + 1])
    source_digest = digest(skin_path)
    _, _, _, _, faces, _, _ = load_skin(skin_path)
    source_coords = args.run / "registered-skin-native.csv"
    pose_file = args.run / "registered-skin-poses.json"
    pose = json.loads(pose_file.read_text())
    points = load_registered_points(source_coords, pose["vertex_count"])
    node_path = args.run / "initial-skin-fem.csv"
    data_path = args.run / "physical-bone-initial.bin"
    for p in (node_path, data_path):
        assert digest(p) == coarse["source"][p.name], "mismatched native scene " + p.name
    bone = np.memmap(data_path, dtype=TRIANGLE_ROW, mode="r")
    bodies, starts, counts = np.unique(bone["body"], return_index=True, return_counts=True)
    assert len(bone) == coarse["bone_triangles"] and len(bodies) == coarse["bone_bodies"]
    assert np.array_equal(starts, np.r_[0, counts.cumsum()[:-1]])
    mesh, _ = owner.float32_surface_records(points, faces)
    shell = owner._prepare_surface_aabb(mesh)
    witnesses = []
    by_bank = Counter()
    for body, begin, count in zip(bodies, starts, counts):
        bank = bone[begin:begin + count]
        measured = owner._audit_pair_prepared_first(shell, records(bank["vertices"]))
        for face, bone_local in measured["triangle_pairs"]:
            source = bank[bone_local]
            key = (int(body), int(source["bone"]))
            witnesses.append([int(face), *key, int(source["triangle"])])
            by_bank[key] += 1
    coarse_counts = Counter((body, bone_id) for role, _, _, body, bone_id, _ in coarse["witnesses"]
                            if role == "outer")
    all_keys = sorted(coarse_counts.keys() | by_bank.keys())
    report = {
        "scope": "Exact Float32 intersections against the same 86 physical bone banks at native startup. Compared pairs refer to their own surface discretizations.",
        "source": {"NHSKIN_sha256": source_digest, **{
            name.name: digest(name) for name in (source_coords, pose_file, node_path, data_path, launch, source_proof)
        }},
        "source_vertex_rows": len(points),
        "source_faces": len(faces),
        "source_pairs": len(witnesses),
        "source_faces_with_pairs": len({row[0] for row in witnesses}),
        "coarse_outer_pairs": coarse["pairs_by_role"]["outer"],
        "coarse_outer_pairs_on_source_clear_banks": sum(
            coarse_counts[key] for key in coarse_counts.keys() - by_bank.keys()
        ),
        "banks": [{"body": key[0], "bone": key[1], "source_pairs": by_bank[key],
                   "coarse_outer_pairs": coarse_counts[key]} for key in all_keys],
        "witness_columns": ["source_surface_face", "body", "bone", "bone_source_face"],
        "witnesses": witnesses,
        "audit_wall_seconds": monotonic() - start,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: val for key, val in report.items() if key not in ("witnesses", "banks")}))


if __name__ == "__main__":
    main()
