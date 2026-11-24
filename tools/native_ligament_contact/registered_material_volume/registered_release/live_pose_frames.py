"""Resolve the native rigid state used to move the registered material."""
import json
from pathlib import Path

import numpy as np


def current_donors(case):
    case = Path(case).resolve()
    launch = json.loads((case / "launch.json").read_bytes())
    authoring = Path(launch["environment"]["NUMI_HUMAN_SKIN_REFERENCE_POSE_DUMP"]).resolve()
    live = authoring.parent / "native-initial-bodies.json"
    if live.parent != case or authoring.samefile(live):
        raise ValueError("The native and authoring frames must be separate files in this run")
    snapshot = json.loads(live.read_bytes())
    source = json.loads(authoring.read_bytes())
    for key in ("schema", "registration_fingerprint32", "binding_count", "payload_abi", "vertex_count"):
        if snapshot[key] != source[key]:
            raise ValueError(f"Frame capture and registration disagree on {key}")
    if snapshot["schema"] != "numi.human.native-skin-pose-snapshot.v1":
        raise ValueError("Unknown native pose schema")
    donors = snapshot["bodies"]
    skeleton = snapshot["all_bodies"]
    if [row["body_index"] for row in skeleton] != list(range(len(skeleton))):
        raise ValueError("Incomplete native skeleton")
    keys = [row["body_index"] for row in donors]
    if (len(keys) != len(set(keys)) or len(keys) != snapshot["binding_count"]
            or set(keys) != {row["body_index"] for row in source["bodies"]}):
        raise ValueError("Duplicate or unmatched material donor")
    for row in donors:
        i = row["body_index"]
        pose = row["current"]
        if i < 0 or i >= len(skeleton) or pose != skeleton[i]["current"]:
            raise ValueError("Binding and native state differ")
        position = np.asarray(pose["position_world_m"])
        orientation = np.asarray(pose["orientation_world_xyzw"])
        if (position.shape != (3,) or orientation.shape != (4,)
                or not np.isfinite(np.r_[position, orientation]).all()
                or abs(orientation @ orientation - 1.0) > 3e-5):
            raise ValueError("Invalid rigid transform")
    return live, donors
