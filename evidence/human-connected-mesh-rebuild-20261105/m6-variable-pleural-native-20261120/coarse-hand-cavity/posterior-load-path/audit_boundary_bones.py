"""Verify the whole cooked skin/fat boundary against all registered bone banks."""
from argparse import ArgumentParser
from collections import Counter
import csv
import json
from pathlib import Path
from time import monotonic

import numpy as np

from audit import BONE, digest, rows
from audit_bed_layer_surfaces import install_for_float32, owner, role_surfaces
from audit_bed_motion_shell import terminal_positions
from tissue_payload_owner import rotation_xyzw


def bone_endpoints(data, registration, motion, step):
    before = {r["body_index"]: r for r in registration}
    after = {}
    with motion.open() as f:
        for row in csv.DictReader(f):
            if int(row["accepted_step"]) == step:
                after[int(row["body_index"])] = row
    for body in np.unique(data["body"]):
        i = int(body)
        old, new = before[i], after[i]
        shift = np.array([float(new[f"root_{a}_m"]) + float(new[f"com_relative_root_{a}_m"])
                          for a in "xyz"])
        q = np.array([float(new[f"orientation_{a}"]) for a in "xyzw"])
        R = rotation_xyzw(q) @ rotation_xyzw(old["quaternion_xyzw"]).T
        bank = data[data["body"] == body]
        start = bank["vertices"]
        end = ((start - np.array(old["position_m"])) @ R.T + shift).astype(np.float32)
        assert np.isfinite(end).all()
        yield bank, end


def audit(run, bone_path, step):
    install_for_float32()
    node_file = run / "shell-mass.csv.nodes.csv"
    cell_file = run / "shell-mass.csv.tets.csv"
    log = run / "stdout.log"
    motion = run / "native/resting-body-motion-partition.csv"
    initial = np.array([[r[f"current_{a}_m"] for a in "xyz"]
                        for r in rows(node_file)], dtype=np.float32)
    endpoint = terminal_positions(log, initial)
    with log.open() as f:
        data = next(line for line in f if line.startswith("resting_initial_anatomical_registration="))
    registration = json.loads(data.partition("=")[2])["body_poses"]
    boundaries = role_surfaces(rows(cell_file))
    faces = np.array([face for group in boundaries.values() for face in group])
    roles = [role for role, group in boundaries.items() for _ in group]
    positions = (initial, endpoint)
    prepared = [owner._prepare_surface_aabb(owner.float32_surface_records(x, faces)[0])
                for x in positions]

    result = [[], []]
    bone = np.memmap(bone_path, mode="r", dtype=BONE)
    assert len(np.unique(bone[["body", "bone", "triangle"]])) == len(bone)
    for bank, final_bone in bone_endpoints(bone, registration, motion, step):
        for k, (shell, target) in enumerate(zip(prepared, (bank["vertices"], final_bone))):
            # Preparation returns closed AABBs on the same exact float lattice
            # as the predicate. This filter only removes disjoint boxes.
            low, high = positions[k].min(axis=0), positions[k].max(axis=0)
            included = np.flatnonzero(np.all(target.max(axis=1) >= low, axis=1) &
                                      np.all(target.min(axis=1) <= high, axis=1))
            triangles = target[included]
            if not triangles.size:
                continue
            tri_points = triangles.reshape(-1, 3)
            records, _ = owner.float32_surface_records(tri_points, np.arange(len(tri_points)).reshape(-1, 3))
            pairs = owner._audit_pair_prepared_first(shell, records)["triangle_pairs"]
            for f, triangle in pairs:
                b = bank[included[triangle]]
                result[k].append([roles[f], f, faces[f].tolist(),
                                  int(b["body"]), int(b["bone"]), int(b["triangle"])])

    def endpoint_report(witnesses):
        return {"pairs": len(witnesses),
                "by_role": dict(Counter(r[0] for r in witnesses)),
                "by_bank": dict(Counter(f"{r[3]}/{r[4]}" for r in witnesses)),
                "witnesses": sorted(witnesses)}

    return {
        "scope": "All cooked skin/fat layer boundaries against all registered physical bone triangle banks at two poses. Terminal triangle offsets are reconstructed from rounded startup bone vertices and accepted rigid poses. Tests static surface intersections only.",
        "accepted_endpoint_step": step, "bone_faces": len(bone),
        "shell_faces": {role: len(faces) for role, faces in boundaries.items()},
        "initial": endpoint_report(result[0]),
        "terminal": endpoint_report(result[1]),
        "input_hashes": {p.name: digest(p) for p in (node_file, cell_file, log, motion, bone_path)},
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("run", "bone_path", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    start = monotonic()
    report = audit(args.run, args.bone_path, args.step)
    report["wall_seconds"] = monotonic() - start
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({phase: {k:v for k,v in report[phase].items() if k != "witnesses"}
                      for phase in ("initial", "terminal")}, indent=2))
