"""Fail-closed multi-pose audit for registered upper-limb source bones.

This audit does not alter anatomy. It replays a bounded bilateral pose suite in
the pinned MyoSim model, carries each admitted BodyParts3D mesh with its owning
source body, and proves that shoulder, elbow, wrist, hand, and digit interfaces
remain continuous away from the neutral pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from . import model as human_model
from .myosim_export import export_fullbody
from .upper_limb_registration import (
    REGISTRATION_SCHEMA,
    SCHEMA as UPPER_REGISTRATION_SCHEMA,
    _minimum_gap,
    _rotation_xyzw,
    _upper_names,
)


SCHEMA = "numi.human.bodyparts3d-myosim-upper-limb-multi-pose-audit.v1"
POSE_CONTINUITY_ALLOWANCE_M = 0.001
BILATERAL_GAP_PARITY_MAXIMUM_M = 0.002
DEFAULT_FRAME_RESIDUAL_MAXIMUM_M = 1.0e-9

# Indices are the exact source qpos addresses used by NHRIGID/Metal --pose-q.
# Values are deliberately bounded functional inspections, not range extrema.
POSE_SUITE: tuple[tuple[str, tuple[tuple[int, float], ...]], ...] = (
    ("neutral", ()),
    ("bilateral_shoulder_elevation", ((36, 1.2), (74, 1.2))),
    ("bilateral_elbow_flexion", ((39, 1.4), (77, 1.4))),
    ("bilateral_forearm_pronation", ((40, 1.2), (78, 1.2))),
    ("bilateral_wrist_deviation_flexion", (
        (41, 0.25), (42, 0.6), (79, 0.25), (80, 0.6),
    )),
    ("bilateral_functional_fist", (
        (43, -0.4), (45, -0.5), (46, -0.7),
        (47, 1.0), (49, 1.0), (50, 0.7),
        (51, 1.0), (53, 1.0), (54, 0.7),
        (55, 1.0), (57, 1.0), (58, 0.7),
        (59, 1.0), (61, 1.0), (62, 0.7),
        (81, -0.4), (83, -0.5), (84, -0.7),
        (85, 1.0), (87, 1.0), (88, 0.7),
        (89, 1.0), (91, 1.0), (92, 0.7),
        (93, 1.0), (95, 1.0), (96, 0.7),
        (97, 1.0), (99, 1.0), (100, 0.7),
    )),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _continuity_transitions() -> list[tuple[str, str, str, float]]:
    return list(human_model._NUMI_HUMAN_UPPER_LIMB_CONTINUITY_TRANSITIONS) + [
        (name, first, second, human_model._NUMI_HUMAN_HAND_CONTINUITY_MAXIMUM_GAP_M)
        for name, first, second in human_model._NUMI_HUMAN_HAND_CONTINUITY_TRANSITIONS
    ]


def _project_joint_equalities(model: Any, qpos: Any, mujoco: Any) -> tuple[int, float]:
    count = 0
    maximum_correction = 0.0
    for equality_index in range(model.neq):
        if int(model.eq_type[equality_index]) != int(mujoco.mjtEq.mjEQ_JOINT):
            continue
        dependent_joint = int(model.eq_obj1id[equality_index])
        driver_joint = int(model.eq_obj2id[equality_index])
        dependent_q = int(model.jnt_qposadr[dependent_joint])
        driver = 0.0 if driver_joint < 0 else float(
            qpos[int(model.jnt_qposadr[driver_joint])]
        )
        coefficients = model.eq_data[equality_index, :5]
        projected = sum(
            float(coefficients[degree]) * driver ** degree for degree in range(5)
        )
        maximum_correction = max(maximum_correction, abs(float(qpos[dependent_q]) - projected))
        qpos[dependent_q] = projected
        count += 1
    return count, maximum_correction


def _joint_for_qpos(model: Any, q_index: int) -> int | None:
    for joint_index in range(model.njnt):
        if int(model.jnt_qposadr[joint_index]) == q_index:
            return joint_index
    return None


def _pose_qpos(model: Any, pose: tuple[tuple[int, float], ...], mujoco: Any, np: Any) -> tuple[Any, int, float]:
    qpos = np.asarray(model.qpos0, dtype=float).copy()
    seen: set[int] = set()
    for q_index, value in pose:
        if q_index in seen or not 0 <= q_index < model.nq or not math.isfinite(value):
            raise RuntimeError("upper-limb pose audit contains an invalid source coordinate override")
        seen.add(q_index)
        joint_index = _joint_for_qpos(model, q_index)
        if joint_index is None:
            raise RuntimeError(f"upper-limb pose audit q index {q_index} is not a joint coordinate")
        if bool(model.jnt_limited[joint_index]):
            lower, upper = (float(item) for item in model.jnt_range[joint_index])
            if value < lower - 1.0e-12 or value > upper + 1.0e-12:
                raise RuntimeError(
                    f"upper-limb pose audit q index {q_index} exceeds its source range"
                )
        qpos[q_index] = value
    equality_count, maximum_correction = _project_joint_equalities(model, qpos, mujoco)
    return qpos, equality_count, maximum_correction


def audit_upper_limb_poses(*, sources: Path, registration_path: Path) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - source environment only
        raise RuntimeError(
            "upper-limb pose audit requires the pinned MyoSim/MuJoCo environment"
        ) from error

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise RuntimeError("upper-limb pose audit requires registration candidate v2")
    upper_receipt = registration.get("upper_limb_source_mesh_registration")
    if not isinstance(upper_receipt, dict) or upper_receipt.get("schema") != UPPER_REGISTRATION_SCHEMA:
        raise RuntimeError("upper-limb pose audit requires admitted upper-limb registration v1")

    exported = export_fullbody(sources)
    source_bodies = {int(body["id"]): body for body in exported["bodies"]}
    model = build_model("myofullbody")
    data = mujoco.MjData(model)
    expected_names = _upper_names("r") | _upper_names("l")
    anchors_by_name: dict[str, dict[str, Any]] = {}
    local_vertices: dict[str, tuple[int, Any]] = {}

    for anchor in registration.get("anchors", []):
        name = anchor.get("target", {}).get("name")
        if name not in expected_names:
            continue
        if name in anchors_by_name:
            raise RuntimeError(f"upper-limb pose audit has repeated anatomy for {name}")
        anchors_by_name[name] = anchor
        target = anchor["target"]
        source_body_id = int(target["source_body_id"])
        source_body = source_bodies.get(source_body_id)
        if source_body is None or source_body.get("name") != name:
            raise RuntimeError(f"upper-limb pose audit body ownership drifted for {name}")
        source_model_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, source_body_id
        )
        if source_model_name != name:
            raise RuntimeError(f"upper-limb pose audit source model body drifted for {name}")
        source = anchor["source"]
        _, member, obj = human_model._bodyparts_obj_member(
            sources, source["hierarchy"], source["member_id"]
        )
        raw_vertices, _ = human_model._bodyparts_obj_triangles(obj, member)
        matrix = np.asarray(
            anchor["registration"]["source_obj_mm_to_core_inertial_body_m"], dtype=float
        )
        if matrix.shape != (4, 4) or not bool(np.all(np.isfinite(matrix))):
            raise RuntimeError(f"upper-limb pose audit has an invalid registration for {name}")
        core_vertices = np.asarray(raw_vertices, dtype=float) @ matrix[:3, :3].T + matrix[:3, 3]
        inertial_position = np.asarray(source_body["inertial_position_body_m"], dtype=float)
        inertial_rotation = _rotation_xyzw(
            source_body["inertial_quaternion_body_xyzw"], np
        )
        # _body_frame_to_core is (body - inertial_position) @ rotation.
        # This exact inverse returns admitted anatomy to the source body frame.
        body_vertices = core_vertices @ inertial_rotation.T + inertial_position
        local_vertices[source["member_id"]] = (source_body_id, body_vertices)

    missing = sorted(expected_names - anchors_by_name.keys())
    if missing:
        raise RuntimeError("upper-limb pose audit is missing bodies: " + ", ".join(missing))

    transitions = _continuity_transitions()
    expected_members = {
        member for _, first, second, _ in transitions for member in (first, second)
    }
    if not expected_members.issubset(local_vertices.keys()):
        raise RuntimeError("upper-limb pose audit transition/member coverage drifted")

    pose_receipts = []
    default_frame_maximum_residual = 0.0
    all_continuity = []
    all_parity = []
    equality_count: int | None = None
    equality_maximum_correction = 0.0
    for pose_name, overrides in POSE_SUITE:
        qpos, current_equality_count, correction = _pose_qpos(
            model, overrides, mujoco, np
        )
        if equality_count is None:
            equality_count = current_equality_count
        elif equality_count != current_equality_count:
            raise RuntimeError("upper-limb pose audit equality coverage changed across poses")
        equality_maximum_correction = max(equality_maximum_correction, correction)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        world_vertices = {
            member_id: (
                vertices @ data.xmat[body_id].reshape(3, 3).T + data.xpos[body_id]
            )
            for member_id, (body_id, vertices) in local_vertices.items()
        }

        if pose_name == "neutral":
            for name, anchor in anchors_by_name.items():
                member_id = anchor["source"]["member_id"]
                centroid = np.mean(world_vertices[member_id], axis=0)
                expected_centroid = np.asarray(
                    anchor["registration"]["default_pose_vertex_centroid_world_m"], dtype=float
                )
                default_frame_maximum_residual = max(
                    default_frame_maximum_residual,
                    float(np.linalg.norm(centroid - expected_centroid)),
                )

        continuity = []
        by_name: dict[str, dict[str, Any]] = {}
        for transition_name, first_member, second_member, rest_gate in transitions:
            gap, _, _ = _minimum_gap(
                world_vertices[first_member], world_vertices[second_member], np
            )
            posed_gate = rest_gate + POSE_CONTINUITY_ALLOWANCE_M
            record = {
                "name": transition_name,
                "source_member_ids": [first_member, second_member],
                "minimum_vertex_gap_m": gap,
                "rest_maximum_allowed_gap_m": rest_gate,
                "posed_maximum_allowed_gap_m": posed_gate,
                "passed": gap <= posed_gate + 1.0e-12,
            }
            continuity.append(record)
            by_name[transition_name] = record
            all_continuity.append({"pose": pose_name, **record})

        parity = []
        for right_name in sorted(name for name in by_name if name.startswith("right_")):
            suffix = right_name[len("right_"):]
            left_name = "left_" + suffix
            if left_name not in by_name:
                raise RuntimeError(
                    f"upper-limb pose audit has no bilateral pair for {right_name}"
                )
            difference = abs(
                by_name[right_name]["minimum_vertex_gap_m"]
                - by_name[left_name]["minimum_vertex_gap_m"]
            )
            record = {
                "transition": suffix,
                "absolute_gap_difference_m": difference,
                "maximum_allowed_difference_m": BILATERAL_GAP_PARITY_MAXIMUM_M,
                "passed": difference <= BILATERAL_GAP_PARITY_MAXIMUM_M + 1.0e-12,
            }
            parity.append(record)
            all_parity.append({"pose": pose_name, **record})

        pose_receipts.append({
            "name": pose_name,
            "source_q_overrides": [
                {"q_index": index, "value": value} for index, value in overrides
            ],
            "joint_equality_count": current_equality_count,
            "joint_equality_maximum_correction": correction,
            "continuity": continuity,
            "bilateral_gap_parity": parity,
        })

    failed_continuity = [
        f"{item['pose']}:{item['name']}" for item in all_continuity if not item["passed"]
    ]
    failed_parity = [
        f"{item['pose']}:{item['transition']}" for item in all_parity if not item["passed"]
    ]
    if default_frame_maximum_residual > DEFAULT_FRAME_RESIDUAL_MAXIMUM_M:
        raise RuntimeError(
            "upper-limb pose audit default source/Core frame transform drifted"
        )
    if failed_continuity:
        raise RuntimeError(
            "upper-limb pose audit violates posed continuity: "
            + ", ".join(failed_continuity[:12])
        )
    if failed_parity:
        raise RuntimeError(
            "upper-limb pose audit violates bilateral gap parity: "
            + ", ".join(failed_parity[:12])
        )

    worst_continuity = max(
        all_continuity,
        key=lambda item: item["minimum_vertex_gap_m"] - item["rest_maximum_allowed_gap_m"],
    )
    worst_parity = max(all_parity, key=lambda item: item["absolute_gap_difference_m"])
    return {
        "schema": SCHEMA,
        "status": "passed_source_owned_bilateral_upper_limb_multi_pose_continuity",
        "inputs": {
            "registration": {
                "file": registration_path.name,
                "sha256": _sha256(registration_path),
            },
            "myosim_archive_sha256": registration["source"]["myosim"]["source"][
                "archive_sha256"
            ],
        },
        "source_body_count": len(expected_names),
        "source_member_count": len(local_vertices),
        "pose_count": len(POSE_SUITE),
        "continuity_transition_count_per_pose": len(transitions),
        "continuity_evaluation_count": len(all_continuity),
        "bilateral_parity_evaluation_count": len(all_parity),
        "joint_equality_count": equality_count,
        "joint_equality_maximum_correction": equality_maximum_correction,
        "default_frame_maximum_centroid_residual_m": default_frame_maximum_residual,
        "default_frame_maximum_allowed_residual_m": DEFAULT_FRAME_RESIDUAL_MAXIMUM_M,
        "posed_continuity_allowance_m": POSE_CONTINUITY_ALLOWANCE_M,
        "bilateral_gap_parity_maximum_m": BILATERAL_GAP_PARITY_MAXIMUM_M,
        "worst_continuity": worst_continuity,
        "worst_bilateral_gap_parity": worst_parity,
        "poses": pose_receipts,
        "evidence_boundary": (
            "Rigid BodyParts3D bones were replayed through pinned MyoSim kinematics and exact "
            "polynomial joint equality projection. Passing proves body ownership, default frame "
            "identity, bounded posed surface continuity, and bilateral gap parity for this pose "
            "suite. It is not cartilage/contact, ligament constraint, loaded dynamics, clinical "
            "registration, or a deformable tendon solve."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = audit_upper_limb_poses(
            sources=arguments.sources.resolve(),
            registration_path=arguments.registration.resolve(),
        )
    except (RuntimeError, OSError, ValueError) as error:
        print(f"numilab-human upper-limb pose audit: {error}", file=sys.stderr)
        return 2
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
