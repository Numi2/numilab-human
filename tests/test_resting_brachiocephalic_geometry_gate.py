import numpy as np
import pytest

from numilab_human import resting_anatomy_verify as av
from numilab_human.resting_brachiocephalic_geometry_gate import (
    BRIDGE_KEYS,
    PORT_TARGET_IDS,
    audit_captured_pose_surface_intersections,
)


def _tetra(apex_z, offset_x=0.0):
    points = np.asarray([
        [offset_x, 0, 0], [offset_x + 1, 0, 0],
        [offset_x, 1, 0], [offset_x, 0, apex_z],
    ], dtype="<f4")
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype="<u4")
    return points, faces


def _pack_surface(points, faces):
    return (points, faces, None, np.arange(len(points), dtype=np.uint64))


def _inputs():
    left, left_faces = _tetra(1.0)
    right, right_faces = _tetra(1.0, 10.0)
    targets = {}
    left_svc, svc_faces = _tetra(-1.0)
    right_svc, right_svc_faces = _tetra(-1.0, 10.0)
    svc_positions = np.concatenate((left_svc, right_svc), axis=0)
    svc_faces = np.concatenate((svc_faces, right_svc_faces + 4), axis=0)
    targets[(51011, 10)] = _pack_surface(svc_positions, svc_faces)
    for stable_id in (469, 473):
        targets[(51011, stable_id)] = _pack_surface(*_tetra(-1.0))
    for stable_id in (468, 472):
        targets[(51011, stable_id)] = _pack_surface(*_tetra(-1.0, 10.0))
    targets[(51005, 900)] = _pack_surface(*_tetra(1.0, 100.0))
    positions = {BRIDGE_KEYS[0]: left, BRIDGE_KEYS[1]: right}
    faces = {BRIDGE_KEYS[0]: left_faces, BRIDGE_KEYS[1]: right_faces}
    all_pairs = [[a, b] for a in range(4) for b in range(4)]
    ports = {}
    for stable_id in PORT_TARGET_IDS[525]:
        ports[(BRIDGE_KEYS[0], (51011, stable_id))] = all_pairs
    for stable_id in PORT_TARGET_IDS[526]:
        target_pairs = ([[a, b + 4] for a in range(4) for b in range(4)]
                        if stable_id == 10 else all_pairs)
        ports[(BRIDGE_KEYS[1], (51011, stable_id))] = target_pairs
    return targets, positions, faces, ports


def test_all_six_declared_ports_and_all_decoded_targets_are_checked_but_not_admitted():
    targets, positions, faces, ports = _inputs()
    result = audit_captured_pose_surface_intersections(
        av, targets, positions, faces, ports)

    assert result["surface_intersection_stage_passed"]
    assert result["status"].endswith("CONTINUITY_UNRESOLVED")
    assert len(result["captured_target_keys"]) == 6
    assert [51005, 900] in result["captured_target_keys"]
    assert len(result["verifier_result"]["rows"]) >= 2
    assert result["limits"]["lumen_continuity"] == "not_evaluated"
    assert result["limits"]["calibre_continuity"] == "not_evaluated"
    assert result["limits"]["anatomical_admission"] is False


def test_missing_port_pair_witness_fails_closed():
    targets, positions, faces, ports = _inputs()
    del ports[(BRIDGE_KEYS[0], (51011, 469))]
    with pytest.raises(ValueError, match="missing frozen face-pair witness"):
        audit_captured_pose_surface_intersections(av, targets, positions, faces, ports)


def test_a_port_pair_list_must_match_exact_observed_pair_set():
    targets, positions, faces, ports = _inputs()
    pair_key = (BRIDGE_KEYS[0], (51011, 469))
    ports[pair_key] = ports[pair_key][:-1]

    result = audit_captured_pose_surface_intersections(
        av, targets, positions, faces, ports)

    assert not result["surface_intersection_stage_passed"]
    assert result["status"] == "CAPTURED_POSE_SURFACE_INTERSECTION_STAGE_REJECTED"
    assert any(row["target"] == [51011, 469]
               and row["unexpected_actual_face_pairs"]
               for row in result["port_pair_mismatches"])


def test_input_geometry_is_not_mutated():
    targets, positions, faces, ports = _inputs()
    before = {key: value[0].tobytes() for key, value in targets.items()}
    before.update({key: value.tobytes() for key, value in positions.items()})

    audit_captured_pose_surface_intersections(av, targets, positions, faces, ports)

    after = {key: value[0].tobytes() for key, value in targets.items()}
    after.update({key: value.tobytes() for key, value in positions.items()})
    assert after == before
