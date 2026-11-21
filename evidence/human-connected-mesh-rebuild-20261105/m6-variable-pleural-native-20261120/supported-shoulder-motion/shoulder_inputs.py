"""Fit the right humeral insertion in the original source patch through supported settling."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import audit_shoulder_prefix as old


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(launch, registration, motion):
    cfg = json.loads(launch.read_text())
    old.audit.ag.RIGID, old.audit.ag.BONES = map(Path, (cfg["argv"][1], cfg["argv"][3]))
    _, raw, _, _, _, _, _, catalog, _, vo, io = old.audit.ag.read_payloads()
    anchors = json.loads(registration.read_text())
    assert hashlib.sha256(raw).hexdigest() == anchors["bone_payload_sha256"]
    meshes = old.load_local_meshes(raw, catalog, vo, io)
    nbody = max(r["body"] for r in catalog.values()) + 1
    observed = {}
    for r in csv.DictReader(motion.open()):
        step = int(r["accepted_step"])
        body = int(r["body_index"])
        if body < nbody:
            pose = observed.setdefault(step, np.full((nbody, 7), np.nan))
            assert not np.isfinite(pose[body]).any()
            pose[body] = old.row_pose(r)
    assert sorted(observed) == list(range(32, 8001, 32))
    for i, entrance in enumerate(anchors["native_pose_inputs"]):
        p = Path(entrance["path"])
        assert digest(p) == entrance["sha256"]
        record = next(json.loads(s.partition("=")[2]) for s in p.open() if s.startswith("registered_ligament_actual_pose_json="))
        assert record["bonePayloadSha256"] == anchors["bone_payload_sha256"]
        observed[-i-1] = np.asarray(record["bodyPoses"])[:nbody]
    assert all(np.isfinite(p).all() for p in observed.values())
    return cfg, catalog, meshes, anchors, observed


def unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def transforms(pose, ref, other):
    a,b=pose[ref],pose[other]
    inv=Rotation.from_quat(a[3:]).inv()
    return inv.apply(b[:3]-a[:3]), inv*Rotation.from_quat(b[3:])
