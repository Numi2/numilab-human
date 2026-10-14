
import copy
import hashlib
import json
import struct

import numpy as np
import pytest

from numilab_human import model
from numilab_human.skin_source_payload_preflight import (
    _canonical_skin_registration_reemission,
)


def fixture():
    frame = np.eye(4); frame[:3, :3] *= .001
    anchors, bindings = [], []
    for core, name in ((2, "torso"), (3, "radius_l")):
        target = {"core_body_index": core, "name": name, "source_body_id": core + 20,
                  "default_com_position_world_m": [float(core), 0., 0.],
                  "default_inertial_quaternion_world_xyzw": [0., 0., 0., 1.]}
        local = model._bodyparts_local_registration_matrix(
            frame.tolist(), target["default_com_position_world_m"],
            target["default_inertial_quaternion_world_xyzw"])
        tr, quat, scale = model._bodyparts_visual_local_pose(local, "fixture")
        bindings.append(struct.pack("<I8f", core, *tr, *quat, scale))
        anchors.append({"source": {"member_id": "FJ" + str(core), "member_sha256": "ab" * 32},
                        "target": target,
                        "registration": {"source_obj_mm_to_core_inertial_body_m": local}})
    old = {"schema": "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2",
           "source": {"bodyparts": {"id": "reference"}, "myosim": {"source": {"archive_sha256": "cd" * 32}}},
           "coordinate_system": {"global_source_mm_to_myosim_world_m": frame.tolist()},
           "anchors": anchors}
    new = copy.deepcopy(old)
    new["anchors"][1]["registration"]["source_obj_mm_to_core_inertial_body_m"][0][3] += .04
    old_sha = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    new_sha = hashlib.sha256(json.dumps(new, sort_keys=True).encode()).hexdigest()
    vertices = np.zeros((4, 14), dtype="<f4")
    vertices[:, :3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 2]]
    vertices[:, 5] = 1
    vertices[:, 10] = 1
    weights = np.array([[1., 0.], [.5, .5], [0., 1.], [.3, .7]], dtype="<f4")
    raw = (struct.pack("<8s5I32s", b"NHSKIN1\0", 5, 2, 4, 3,
                       int(old_sha[:8], 16), bytes.fromhex("cd" * 32))
           + b"".join(bindings) + vertices.tobytes()
           + np.asarray([0, 1, 2], dtype="<u4").tobytes() + weights.tobytes())
    return raw, old, new, old_sha, new_sha


def test_reemission_changes_only_discriminator_and_preserves_repaired_geometry():
    raw, old, new, old_sha, new_sha = fixture()
    candidate, report = _canonical_skin_registration_reemission(raw, old, new, old_sha, new_sha, {})
    assert candidate[:24] == raw[:24]
    assert candidate[28:] == raw[28:]
    assert struct.unpack_from("<I", candidate, 24)[0] == int(new_sha[:8], 16)
    assert report["canonical_binding_count"] == 2
    assert report["all_geometry_normals_topology_and_full_weights_byte_exact"]


@pytest.mark.parametrize("change", ["global_frame", "target_frame", "source_member", "archive", "duplicate_anchor"])
def test_reemission_rejects_owner_or_source_drift(change):
    raw, old, new, old_sha, new_sha = fixture()
    if change == "global_frame":
        new["coordinate_system"]["global_source_mm_to_myosim_world_m"][0][3] += .001
    elif change == "target_frame":
        new["anchors"][0]["target"]["default_com_position_world_m"][0] += .001
    elif change == "source_member":
        new["anchors"][0]["source"]["member_sha256"] = "ef" * 32
    elif change == "archive":
        new["source"]["myosim"]["source"]["archive_sha256"] = "ef" * 32
    else:
        new["anchors"].append(copy.deepcopy(new["anchors"][0]))
    with pytest.raises(model.ImportError):
        _canonical_skin_registration_reemission(raw, old, new, old_sha, new_sha, {})


def test_noncanonical_binding_cannot_be_promoted_to_current_registration():
    raw, old, new, old_sha, new_sha = fixture()
    damaged = bytearray(raw)
    struct.pack_into("<f", damaged, 64, -1.99)
    with pytest.raises(model.ImportError, match="canonical atlas owner"):
        _canonical_skin_registration_reemission(bytes(damaged), old, new, old_sha, new_sha, {})


def test_previous_fingerprint_is_required():
    raw, old, new, _old_sha, new_sha = fixture()
    with pytest.raises(model.ImportError, match="previous registration"):
        _canonical_skin_registration_reemission(raw, old, new, "11" * 32, new_sha, {})
