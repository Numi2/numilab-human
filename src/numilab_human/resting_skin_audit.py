"""Inspect the complete source skin against the bed at a native GPU pose.

This is offline evidence analysis, not a CPU physics or skinning runtime.
The pose file is emitted by the existing native Human viewer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .resting_scene import load_rigid, load_skin, source_shell_world, _require


def audit(rigid_path: Path, skin_path: Path, native_poses: Path,
          maximum_penetration_m: float = .001) -> dict:
    _require(np.isfinite(maximum_penetration_m) and maximum_penetration_m >= 0,
             "invalid skin penetration tolerance")
    rigid = load_rigid(rigid_path)
    skin = load_skin(skin_path, rigid)
    snapshot = json.loads(native_poses.read_text())
    _require(snapshot.get("schema") == "numi.human.native-skin-pose-snapshot.v1" and
             snapshot.get("registration_fingerprint32") == int(skin["registration_fingerprint"], 16) and
             snapshot.get("vertex_count") == skin["vertex_count"] and
             snapshot.get("binding_count") == skin["binding_count"] and
             snapshot.get("payload_abi") == 5,
             "native pose snapshot does not identify the source skin")
    owners = set()
    for row in snapshot["bodies"]:
        owner = row["body_index"]
        _require(owner in rigid["core_poses"] and owner not in owners,
                 "native pose snapshot has invalid or duplicate body owners")
        owners.add(owner)
        pose = np.asarray(row["current"]["position_world_m"] +
                          row["current"]["orientation_world_xyzw"], dtype=np.float64)
        _require(pose.shape == (7,) and np.isfinite(pose).all() and
                 abs(np.linalg.norm(pose[3:]) - 1) < 2e-5,
                 "native body pose is invalid")
        # The native inspector emits raw source-rest COM frames beside the
        # accepted frames. Check them before using the accepted transforms.
        rest = np.asarray(row["rest"]["position_world_m"] +
                          row["rest"]["orientation_world_xyzw"], dtype=np.float64)
        expected = rigid["core_poses"][owner]
        quaternion_error = min(np.linalg.norm(rest[3:] - expected[3:]),
                               np.linalg.norm(rest[3:] + expected[3:]))
        _require(np.max(np.abs(rest[:3] - expected[:3])) < 2e-5 and quaternion_error < 2e-5,
                 "native source-rest frame differs from the rigid payload")
        rigid["core_poses"][owner] = pose
    _require(owners == {row[0] for row in skin["bindings"]},
             "native snapshot omits a source skin owner")
    world, bodies, _ = source_shell_world(rigid, skin)
    z = world[:, 2]
    return {
        "rigid_sha256": rigid["sha256"], "skin_sha256": skin["sha256"],
        "native_poses_sha256": hashlib.sha256(native_poses.read_bytes()).hexdigest(),
        "vertex_count": len(world), "plane_z_m": 0,
        "minimum_gap_m": float(z.min()), "minimum_vertex": int(z.argmin()),
        "minimum_body": int(bodies[z.argmin()]),
        "allowed_penetration_m": maximum_penetration_m,
        "passed": bool(z.min() >= -maximum_penetration_m),
        "counts_below_m": {str(t): int((z < -t).sum()) for t in (0, .0001, .001, .005, .01)},
        "regional_minimum_gap_m": {str(b): float(z[bodies == b].min()) for b in sorted(set(bodies))},
        "bounds_m": [world.min(axis=0).tolist(), world.max(axis=0).tolist()],
        "boundary": "All positive ABI5 skin weights and unmodified native accepted poses; "
                    "offline inspection only, no integration or bed clamping. Respiratory "
                    "deformation and intermediate states require the native live audit.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rigid", type=Path, required=True)
    parser.add_argument("--skin", type=Path, required=True)
    parser.add_argument("--native-poses", type=Path, required=True)
    parser.add_argument("--maximum-penetration-m", type=float, default=.001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.rigid, args.skin, args.native_poses, args.maximum_penetration_m)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("passed", "minimum_gap_m", "minimum_body", "vertex_count")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
