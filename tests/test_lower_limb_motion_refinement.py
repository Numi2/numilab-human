"""Registration geometry regressions; these are not loaded mechanics evidence."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from numilab_human.lower_limb_pose_audit import _posed_continuity_gates
from numilab_human.lower_limb_source_registration import (
    _bounded_interface_translation, _transfer_fit_between_default_frames,
)
from numilab_human.upper_limb_registration import _minimum_gap, _interface_patch_metrics


def _interfaces(delta, flexed_gap=0.00627120646313):
    # Neutral and flexed planes model one mesh carried by two different proper
    # body orientations. A default-world correction must rotate with its owner.
    plane = np.array([[x, y, 0.] for x in np.linspace(-.006, .006, 7)
                      for y in np.linspace(-.006, .006, 7)])
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    rest = (plane, plane + [0., 0., .001] + delta, .004, .005)
    source = {"minimum_vertex_gap_m": .00247097662499,
              "interface_patch": {"bidirectional_p90_m": .0082947554376}}
    _, _, gap_gate, patch_gate = _posed_continuity_gates(.004, source, .003)
    flexed = (plane @ rotation.T, plane @ rotation.T + [0., flexed_gap, 0.]
              + rotation @ delta, gap_gate, patch_gate)
    return rest, flexed


def _objective(delta, flexed_gap=0.00627120646313, *, rest_only=False):
    records = _interfaces(delta, flexed_gap)
    if rest_only:
        records = records[:1]
    errors = []
    for first, second, gap_gate, patch_gate in records:
        gap, _, _ = _minimum_gap(first, second, np)
        patch = _interface_patch_metrics(first, second, np)["bidirectional_p90_m"]
        errors.append(max(gap / gap_gate, patch / patch_gate))
    return max(errors), sum(errors), float(np.linalg.norm(delta))


def test_neutral_only_search_misses_detachment_and_pose_search_repairs_it():
    old_delta, old_initial, _ = _bounded_interface_translation(lambda d: _objective(d, rest_only=True), np)
    assert old_initial[0] < 1.
    assert np.array_equal(old_delta, np.zeros(3))
    assert _objective(old_delta)[0] > 1.
    delta, initial, final = _bounded_interface_translation(_objective, np)
    assert initial[0] > 1.
    assert final[0] <= 1.
    assert 0. < np.linalg.norm(delta) <= .0015 + 1e-12
    assert _objective(delta, rest_only=True)[0] <= 1.


def test_valid_source_supported_variation_is_not_moved():
    for gap in (.003, .0049, .0054):
        delta, initial, final = _bounded_interface_translation(lambda d: _objective(d, gap), np)
        assert initial[0] <= 1.
        assert np.array_equal(delta, np.zeros(3))
        assert final == initial


def test_unrepairable_pose_does_not_expand_translation_or_tolerance():
    delta, initial, final = _bounded_interface_translation(lambda d: _objective(d, .008), np)
    assert initial[0] > 1.
    assert final[0] > 1.
    assert np.linalg.norm(delta) <= .0015 + 1e-12
    _, _, allowed, _ = _posed_continuity_gates(.004, {"minimum_vertex_gap_m": .00247097662499,
        "interface_patch": {"bidirectional_p90_m": .0082947554376}}, .003)
    assert allowed == pytest.approx(.00547097662499)


def test_coupled_search_bounds_each_owner_independently():
    def coupled(deltas):
        residuals = [abs(deltas[0, 0] - .001) / .0002,
                     abs(deltas[1, 2] + .001) / .0002]
        return max(residuals), sum(residuals), float(np.linalg.norm(deltas))
    delta, initial, final = _bounded_interface_translation(coupled, np, group_count=2)
    assert initial[0] > 1. and final[0] <= 1.
    assert np.all(np.linalg.norm(delta, axis=1) <= .0015 + 1e-12)


def test_preserved_foot_does_not_add_roundoff_motion_to_its_toes():
    foot = {"default_inertial_quaternion_world_xyzw": [0., -.7071067811865475, 0., .7071067811865475],
            "default_com_position_world_m": [.17, .06, .04]}
    toes = {"default_inertial_quaternion_world_xyzw": [.5, .5, .5, .5],
            "default_com_position_world_m": [.17, -.11, .03]}
    rotation, translation = _transfer_fit_between_default_frames(np.eye(3), np.zeros(3), foot, toes, np)
    assert np.array_equal(rotation, np.eye(3))
    assert np.array_equal(translation, np.zeros(3))


def test_source_pose_audit_rejects_original_patella_error_and_accepts_bounded_repair():
    """Opt-in exact source geometry; unrelated motion failures stay visible."""
    from numilab_human.lower_limb_pose_audit import audit_lower_limb_poses
    from numilab_human.upper_limb_pose_audit import PoseAuditError

    keys = ("SOURCES", "ARTIFACT", "ORIGINAL", "REPAIRED")
    paths = {key: os.environ.get(f"NUMILAB_HUMAN_MOTION_{key}") for key in keys}
    if not all(paths.values()):
        pytest.skip("exact source-pose geometry regression inputs were not supplied")

    def measured(registration):
        try:
            return audit_lower_limb_poses(sources=Path(paths["SOURCES"]),
                artifact=Path(paths["ARTIFACT"]), registration_path=Path(registration))
        except PoseAuditError as error:
            return error.result

    original = json.loads(Path(paths["ORIGINAL"]).read_text())
    repaired = json.loads(Path(paths["REPAIRED"]).read_text())
    assert original["source"] == repaired["source"]
    matrix_key = "source_obj_mm_to_core_inertial_body_m"
    changed = []
    for before, after in zip(original["anchors"], repaired["anchors"], strict=True):
        assert before["source"] == after["source"]
        assert before["target"] == after["target"]
        a, b = np.asarray(before["registration"][matrix_key]), np.asarray(after["registration"][matrix_key])
        if not np.array_equal(a, b):
            changed.append(before["source"]["member_id"])
            assert np.array_equal(a[:3, :3], b[:3, :3])
            assert 0. < np.linalg.norm(a[:3, 3] - b[:3, 3]) <= .0015 + 1e-12
    assert changed == ["FJ3275"]

    before, after = measured(paths["ORIGINAL"]), measured(paths["REPAIRED"])
    def patella(result, pose_name):
        pose = next(p for p in result["poses"] if p["name"] == pose_name)
        return next(i for i in pose["continuity"] if i["name"] == "left_femur_to_patella")
    assert patella(before, "neutral")["passed"]
    assert not patella(before, "bilateral_deep_crouch")["passed"]
    for pose in after["poses"]:
        assert patella(after, pose["name"])["passed"]
        for key in ("posed_maximum_allowed_gap_m", "posed_maximum_allowed_interface_patch_p90_m"):
            assert patella(before, pose["name"])[key] == patella(after, pose["name"])[key]
