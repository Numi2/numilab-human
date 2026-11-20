"""Audit the repaired deep fat against the posed left iliac bone at both native endpoints."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path

import numpy as np

from audit import BONE, digest, rows
from audit_bed_layer_surfaces import install_for_float32, role_surfaces
from audit_bed_motion_shell import terminal_positions
from fit_interface import index_pairs
from tissue_payload_owner import rotation_xyzw


def audit_pelvis(reference, native, bone_dump):
    install_for_float32()
    nodes = rows(reference / "shell-mass.csv.nodes.csv")
    before = np.array([[r[f"current_{a}_m"] for a in "xyz"] for r in nodes], dtype=np.float32)
    now = rows(native / "shell-mass.csv.nodes.csv")
    initial = np.array([[r[f"current_{a}_m"] for a in "xyz"] for r in now], dtype=np.float32)
    end = terminal_positions(native / "stdout.log", initial)
    changed = np.any(before != initial, axis=1)
    n = len(before) // 3
    assert np.all(~changed[:2*n])
    labels = role_surfaces(rows(reference / "shell-mass.csv.tets.csv"))
    deep = np.asarray(labels["deep"])
    faces = deep[changed[deep].any(axis=1)]
    raw = np.memmap(bone_dump, mode="r", dtype=BONE)
    pelvis = raw[(raw["body"] == 128) & (raw["bone"] == 19)]
    start_bone = pelvis["vertices"]
    start_pairs = index_pairs(initial, faces, start_bone, np.arange(len(pelvis)))
    old_pairs = index_pairs(before, faces, start_bone, np.arange(len(pelvis)))
    final_pose = None
    pose_csv = native / "native/resting-body-motion-partition.csv"
    with pose_csv.open() as f:
        for row in csv.DictReader(f):
            if int(row["body_index"]) == 128:
                final_pose = row
    assert final_pose is not None
    with (native / "stdout.log").open() as f:
        event = next(row for row in f if row.startswith("resting_initial_anatomical_registration="))
    poses = json.loads(event.partition("=")[2])["body_poses"]
    first = next(row for row in poses if int(row["body_index"]) == 128)
    final_p = np.array([float(final_pose[f"root_{a}_m"]) +
                        float(final_pose[f"com_relative_root_{a}_m"]) for a in "xyz"])
    final_q = np.array([float(final_pose[f"orientation_{a}"]) for a in "xyzw"])
    motion = rotation_xyzw(final_q) @ rotation_xyzw(first["quaternion_xyzw"]).T
    terminal_bone = ((start_bone - np.asarray(first["position_m"])) @ motion.T + final_p).astype(np.float32)
    terminal_pairs = index_pairs(end, faces, terminal_bone, np.arange(len(pelvis)))
    result = {
        "scope": "Exact repaired deep-fat triangle pairs against left ilium bank 128/19; terminal bone placed from accepted rigid poses using the retained initial bone triangles.",
        "native_launch_sha256": digest(native / "launch.json"),
        "bone_source_sha256": digest(bone_dump),
        "body_motion_sha256": digest(pose_csv), "repair_faces": len(faces),
        "last_accepted_step": int(final_pose["accepted_step"]),
        "baseline_intersections": len(old_pairs),
        "native_start_intersections": len(start_pairs),
        "native_endpoint_intersections": len(terminal_pairs),
        "maximum_iliac_displacement_m": float(np.linalg.norm(
            terminal_bone.astype(float) - start_bone, axis=2).max()),
    }
    assert result["baseline_intersections"] == 28
    assert not start_pairs and not terminal_pairs
    return result


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for name in ("reference", "native", "bone_dump", "out"):
        p.add_argument(name, type=Path)
    params = vars(p.parse_args())
    out = params.pop("out")
    result = audit_pelvis(**params)
    with out.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(result))
