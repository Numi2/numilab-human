"""Compile two shoulder registrations into the articulated model and check the source rest oracle."""
from pathlib import Path
import hashlib
import json
import struct
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from compile_source_frames import bind


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix(pose):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    transform[:3, 3] = pose[:3]
    return transform


def compile_joint(job, fits, archive, reference, output):
    launch = json.loads((job / "launch.json").read_text())
    original = Path(launch["argv"][1])
    raw = original.read_bytes()
    if digest(original) != launch["pins"][str(original)] or raw[:8] != b"NHRIGID2":
        raise ValueError("Unexpected articulated source")
    version, engine, count, bodies, joints = struct.unpack_from("<5I", raw, 8)
    if (version, engine, count, bodies, joints) != (1, 5, 103, 157, 156):
        raise ValueError("Unexpected rigid layout")
    ids = np.frombuffer(raw, "<u4", count, len(raw)-32*count)
    oracle_offset = len(raw)-28*count
    prior_poses = np.frombuffer(raw, "<f4", count*7, oracle_offset).reshape(-1, 7)
    tree = json.loads(reference.read_text())["core_tree"]["source_body_records"]
    if list(ids) != [r["core_body_index"] for r in tree]:
        raise ValueError("Mismatch between model reference and runtime bodies")
    release = json.loads((job/"native-initial-bodies.json").read_bytes())["all_bodies"]
    edited = bytearray(raw)
    frames = []
    affected = {}
    checks = []
    for side, index, parent in [("l", 0, 84), ("r", 1, 34)]:
        pick = json.loads((fits/(side+".json")).read_text())
        local = np.array(pick["shoulder_to_arm_parent_mm"], dtype=float)
        local[:3, 3] *= .001
        rotation = Rotation.from_matrix(local[:3, :3])
        if not np.allclose(rotation.as_matrix(), local[:3, :3], atol=1e-10, rtol=0):
            raise ValueError("Invalid joint rotation")
        joint_offset = 224 + 160*bodies + parent*144
        if struct.unpack_from("<3I", raw, joint_offset) != (parent,parent+1,0):
            raise ValueError("Unexpected physical base joint")
        anchor = struct.unpack_from("<3f", raw, joint_offset+80)
        quat = struct.unpack_from("<4f", raw, joint_offset+112)
        struct.pack_into("<3f", edited, joint_offset+80, *(rotation.apply(anchor)+local[:3,3]))
        struct.pack_into("<4f", edited, joint_offset+112, *(rotation*Rotation.from_quat(quat)).as_quat())
        frames.append(dict(side=index,parent_translation_base_m=local[:3,3].tolist(),
                           parent_rotation_xyzw=rotation.as_quat().tolist()))
        p = release[parent]["rest"]
        parent_pose = matrix([*p["position_world_m"],*p["orientation_world_xyzw"]])
        world = parent_pose@local@np.linalg.inv(parent_pose)
        # Determine ownership through the actual compiled tree, including shared
        # ancestors and non-source intermediate bodies.
        for body in ids:
            walk = int(body)
            while walk:
                if walk == parent+1:
                    affected[int(body)] = world
                    break
                off = 224+160*bodies+144*(walk-1)
                ancestor, child = struct.unpack_from("<2I", raw, off)
                if child != walk or ancestor >= child:
                    raise ValueError("Unexpected physical parent chain")
                walk = ancestor
    output.mkdir()
    draft = output / "joint.nhrigid"
    draft.write_bytes(edited)
    cal = output / "frames.json"
    cal.write_text(json.dumps(frames,indent=2)+"\n")
    candidate = output/"source-bound.nhrigid"
    report = output/"source-bind.json"
    bind(original, draft, cal, archive, reference, candidate, report)
    result = candidate.read_bytes()
    poses = np.frombuffer(result,"<f4",count*7,oracle_offset).reshape(-1,7)
    for body,before,after in zip(ids,prior_poses,poses):
        b = int(body)
        if b not in affected and not np.array_equal(before,after):
            raise ValueError(f"Unrelated rest record changed for body {b}")
        expected = affected.get(b,np.eye(4))@matrix(before)
        actual = matrix(after)
        displacement = np.linalg.norm(actual[:3,3]-expected[:3,3])
        angle = Rotation.from_matrix(actual[:3,:3]@expected[:3,:3].T).magnitude()
        if displacement>2e-6 or angle>2e-6:
            raise ValueError(f"Rest oracle disagrees for body {b}: {displacement}, {angle}")
        checks.append((float(displacement),float(angle)))
    receipt = dict(launch=digest(job/"launch.json"),control=digest(original),source=digest(archive),
                   selected={s:digest(fits/(s+".json")) for s in ("l","r")},candidate=digest(candidate),
                   body_count=len(ids),changed_source_bodies=len(affected),changed_joint_rows=[34,84],
                   max_rest_position_error_m=max(x[0] for x in checks),max_rest_rotation_error_rad=max(x[1] for x in checks))
    (output/"result.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    compile_joint(*map(Path,sys.argv[1:]))
