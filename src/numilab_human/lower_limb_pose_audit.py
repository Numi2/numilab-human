"""Fail-closed multi-pose audit for registered lower-limb source bones.

The audit carries every admitted BodyParts3D femur, patella, tibia/fibula,
ankle, foot, and complete toe-compound surface with its owning MyoSim body.
It evaluates robust bilateral joint-interface patches after exact dependent
coordinate projection.  It does not alter anatomy or add toe articulation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from . import model as human_model
from .lower_limb_source_registration import (
    SCHEMA as LOWER_REGISTRATION_SCHEMA, _BODY_GATES, _lower_limb_source_frame_checks,
    _femoral_head_articular_metrics,
)
from .myosim_bone_proximity import _compiled_meshes_by_body
from .myosim_export import export_fullbody
from .upper_limb_pose_audit import (
    PoseAuditError, _finish_pose_audit, _pose_qpos,
    _compiled_bone_members, _compiled_member_geometry_check,
    _compiled_bone_geometry_checks,
    _pose_joint_range_context, _projected_joint_range_checks,
    _joint_equality_program_checks,
    PROJECTED_JOINT_RANGE_TOLERANCE,
)
from .joint_constraint_consistency import (
    joint_equality_driver_domain_audit, source_equality_projection_oracle,
)
from .upper_limb_registration import (
    INTERFACE_PATCH_GATE_MULTIPLIER,
    REGISTRATION_SCHEMA,
    _interface_patch_metrics,
    _minimum_gap,
    _rotation_xyzw,
    _surface_split_metrics, _body_frame_to_core,
)


SCHEMA = "numi.human.bodyparts3d-myosim-lower-limb-multi-pose-audit.v1"
POSE_CONTINUITY_ALLOWANCE_M = 0.001
# BodyParts3D's independently segmented load-bearing surfaces are not exact
# sagittal mirrors.  Four millimetres keeps parity far below the admitted
# 12 mm bilateral surface-fit envelope while avoiding a false requirement to
# warp one atlas knee into the other; same-pose source-relative gates remain
# authoritative for each side.
BILATERAL_GAP_PARITY_MAXIMUM_M = 0.004
DEFAULT_FRAME_RESIDUAL_MAXIMUM_M = 1.0e-9
MECHANICS_REFERENCE_INTERFACE_ALLOWANCE_M = 0.003
RIGID_TOE_COMPOUND_REFERENCE_ALLOWANCE_M = 0.0035

# Exact source qpos addresses shared by MyoSim, NHRIGID, NHEQ1, and --pose-q.
# These are bounded inspection states, not range extrema or a gait trajectory.
POSE_SUITE: tuple[tuple[str, tuple[tuple[int, float], ...]], ...] = (
    ("neutral", ()),
    ("bilateral_hip_flexion", ((101, 0.70), (115, 0.70))),
    ("bilateral_knee_flexion", ((106, 0.90), (120, 0.90))),
    ("bilateral_ankle_dorsiflexion", ((109, -0.25), (123, -0.25))),
    ("bilateral_subtalar_rotation", ((110, 0.20), (124, 0.20))),
    ("bilateral_mtp_flexion", ((111, 0.35), (125, 0.35))),
    ("bilateral_deep_crouch", (
        (101, 0.9), (106, 1.4), (109, -0.25), (111, 0.2),
        (115, 0.9), (120, 1.4), (123, -0.25), (125, 0.2),
    )),
    ("bilateral_functional_crouch", (
        (101, 0.50), (106, 0.75), (109, -0.15), (111, 0.20),
        (115, 0.50), (120, 0.75), (123, -0.15), (125, 0.20),
    )),
)

LOWER_BODY_NAMES = {
    f"{family}_{side}"
    for family in ("femur", "tibia", "talus", "calcn", "toes", "patella")
    for side in ("r", "l")
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _continuity_transitions() -> list[tuple[str, str, str, float]]:
    return [
        (
            name, first, second,
            human_model._NUMI_HUMAN_KNEE_CONTINUITY_MAXIMUM_GAP_M,
        )
        for name, first, second in human_model._NUMI_HUMAN_KNEE_CONTINUITY_TRANSITIONS
    ] + [
        (
            name, first, second,
            human_model._NUMI_HUMAN_FOOT_CONTINUITY_MAXIMUM_GAP_M,
        )
        for name, first, second in human_model._NUMI_HUMAN_FOOT_CONTINUITY_TRANSITIONS
    ]


def _posed_continuity_gates(
    rest_gate: float, mechanics_reference: dict[str, Any] | None,
    reference_allowance: float,
) -> tuple[float, float, float, float]:
    """One tolerance basis for the source registrar and independent pose audit."""
    base_gap = rest_gate + POSE_CONTINUITY_ALLOWANCE_M
    base_patch = INTERFACE_PATCH_GATE_MULTIPLIER * base_gap
    gap, patch = base_gap, base_patch
    if mechanics_reference is not None:
        gap = max(gap, mechanics_reference["minimum_vertex_gap_m"] + reference_allowance)
        patch = max(patch, mechanics_reference["interface_patch"]["bidirectional_p90_m"] + reference_allowance)
    return base_gap, base_patch, gap, patch


def _bilateral_interface_parity(by_name: dict[str, Any]) -> list[dict[str, Any]]:
    """Measure the existing bilateral interface bound without changing its basis."""
    parity = []
    for right_name in sorted(name for name in by_name if name.startswith("right_")):
        suffix = right_name[len("right_"):]
        left_name = "left_" + suffix
        if left_name not in by_name:
            raise RuntimeError(
                f"lower-limb pose audit has no bilateral pair for {right_name}"
            )
        difference = abs(
            by_name[right_name]["minimum_vertex_gap_m"]
            - by_name[left_name]["minimum_vertex_gap_m"]
        )
        patch_difference = abs(
            by_name[right_name]["interface_patch"]["bidirectional_p90_m"]
            - by_name[left_name]["interface_patch"]["bidirectional_p90_m"]
        )
        record = {
            "transition": suffix,
            "right_minimum_vertex_gap_m": by_name[right_name]["minimum_vertex_gap_m"],
            "left_minimum_vertex_gap_m": by_name[left_name]["minimum_vertex_gap_m"],
            "right_interface_patch_p90_m": by_name[right_name]["interface_patch"][
                "bidirectional_p90_m"
            ],
            "left_interface_patch_p90_m": by_name[left_name]["interface_patch"][
                "bidirectional_p90_m"
            ],
            "absolute_gap_difference_m": difference,
            "absolute_interface_patch_p90_difference_m": patch_difference,
            "maximum_allowed_difference_m": BILATERAL_GAP_PARITY_MAXIMUM_M,
            "passed": (
                difference <= BILATERAL_GAP_PARITY_MAXIMUM_M + 1.0e-12
                and patch_difference <= BILATERAL_GAP_PARITY_MAXIMUM_M + 1.0e-12
            ),
        }
        parity.append(record)
    return parity


def audit_lower_limb_poses(
    *, sources: Path, registration_path: Path, artifact: Path,
    bone_artifact: Path | None = None,
) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - source environment only
        raise RuntimeError(
            "lower-limb pose audit requires the pinned MyoSim/MuJoCo environment"
        ) from error

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise RuntimeError("lower-limb pose audit requires registration candidate v2")
    lower_receipt = registration.get("lower_limb_source_mesh_registration")
    if (
        not isinstance(lower_receipt, dict)
        or lower_receipt.get("schema") != LOWER_REGISTRATION_SCHEMA
    ):
        raise RuntimeError("lower-limb pose audit requires admitted lower-limb registration v3")
    runtime_reference, runtime_bodies = human_model._bodyparts_runtime_bindings(registration, artifact)
    compiled_members, bone_descriptor = ({}, None)
    if bone_artifact is not None:
        compiled_members, bone_descriptor = _compiled_bone_members(
            bone_artifact, registration_path, registration,
            runtime_reference, runtime_bodies,
        )
    compiled_geometry_by_body = (
        _compiled_bone_geometry_checks(sources, registration, compiled_members, np)
        if bone_descriptor is not None else {}
    )
    common_frame = human_model._bodyparts_source_common_frame(sources, registration, runtime_bodies)
    source_frame_checks = _lower_limb_source_frame_checks(registration, common_frame, runtime_bodies)

    exported = export_fullbody(sources)
    if exported.get("source") != registration["source"]["myosim"]["source"]:
        raise RuntimeError("lower-limb pose audit pinned source export differs from its registered runtime source")
    source_bodies = {int(body["id"]): body for body in exported["bodies"]}
    model = build_model("myofullbody")
    data = mujoco.MjData(model)
    joint_ranges = _pose_joint_range_context(artifact, runtime_reference, model, mujoco)
    equality_programs = _joint_equality_program_checks(artifact, runtime_reference, exported, joint_ranges)
    rigid_program = human_model._myosim_rigid_program_checks(artifact, exported)
    compiled_meshes_by_body = _compiled_meshes_by_body(model, mujoco, np)
    anchors_by_name: dict[str, list[dict[str, Any]]] = {
        name: [] for name in LOWER_BODY_NAMES
    }
    anchors_by_member: dict[str, dict[str, Any]] = {}
    local_vertices: dict[str, tuple[int, Any]] = {}
    registered_local_vertices: dict[str, tuple[int, Any]] = {}
    source_local_vertices: dict[int, Any] = {}

    for anchor in registration.get("anchors", []):
        name = anchor.get("target", {}).get("name")
        if name not in LOWER_BODY_NAMES:
            continue
        target = anchor["target"]
        source_body_id = int(target["source_body_id"])
        source_body = source_bodies.get(source_body_id)
        if source_body is None or source_body.get("name") != name:
            raise RuntimeError(f"lower-limb pose audit body ownership drifted for {name}")
        source_model_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, source_body_id
        )
        if source_model_name != name:
            raise RuntimeError(f"lower-limb pose audit source model body drifted for {name}")
        if source_body_id not in source_local_vertices:
            source_meshes = compiled_meshes_by_body.get(source_body_id, [])
            if not source_meshes:
                raise RuntimeError(
                    f"lower-limb pose audit has no mechanics mesh for {name}"
                )
            source_local_vertices[source_body_id] = np.concatenate([
                np.asarray(mesh["vertices"], dtype=float) for mesh in source_meshes
            ])
        source = anchor["source"]
        member_id = str(source["member_id"])
        if member_id in anchors_by_member:
            raise RuntimeError(f"lower-limb pose audit repeats source member {member_id}")
        _, member, obj = human_model._bodyparts_obj_member(
            sources, source["hierarchy"], member_id
        )
        raw_vertices, raw_triangles = human_model._bodyparts_obj_triangles(obj, member)
        if (
            source.get("member_sha256") != hashlib.sha256(obj).hexdigest()
            or source.get("vertex_count") != len(raw_vertices)
            or source.get("triangle_count") != len(raw_triangles)
        ):
            raise RuntimeError(f"lower-limb pose audit pinned mesh identity drifted for {member_id}")
        matrix = np.asarray(
            anchor["registration"]["source_obj_mm_to_core_inertial_body_m"],
            dtype=float,
        )
        if matrix.shape != (4, 4) or not bool(np.all(np.isfinite(matrix))):
            raise RuntimeError(f"lower-limb pose audit has an invalid registration for {member_id}")
        core_vertices = np.einsum(
            "ki,ji->kj", np.asarray(raw_vertices, dtype=float), matrix[:3, :3]
        ) + matrix[:3, 3]
        inertial_position = np.asarray(
            source_body["inertial_position_body_m"], dtype=float
        )
        inertial_rotation = _rotation_xyzw(
            source_body["inertial_quaternion_body_xyzw"], np
        )
        body_vertices = np.einsum(
            "ki,ji->kj", core_vertices, inertial_rotation
        ) + inertial_position
        registered_local_vertices[member_id] = (source_body_id, body_vertices)
        if bone_descriptor is not None:
            surface = compiled_members[member_id]
            body_vertices = np.einsum(
                "ki,ji->kj", np.asarray(surface["vertices"], dtype=float),
                inertial_rotation,
            ) + inertial_position
        anchors_by_name[name].append(anchor)
        anchors_by_member[member_id] = anchor
        local_vertices[member_id] = (source_body_id, body_vertices)

    missing_bodies = sorted(
        name for name, anchors in anchors_by_name.items() if not anchors
    )
    if missing_bodies:
        raise RuntimeError(
            "lower-limb pose audit is missing bodies: " + ", ".join(missing_bodies)
        )

    source_geometry_checks = [
        record for name, record in compiled_geometry_by_body.items() if name not in LOWER_BODY_NAMES
    ]
    for name in sorted(LOWER_BODY_NAMES):
        anchors = anchors_by_name[name]
        body_id = int(anchors[0]["target"]["source_body_id"])
        frame_checks = [check for check in source_frame_checks if check["myosim_body"] == name]
        record = {
            "myosim_body": name, "source_member_ids": [a["source"]["member_id"] for a in anchors],
            "source_frame_checks": frame_checks, "passed": all(c["passed"] for c in frame_checks),
        }
        if bone_descriptor is not None:
            checks = compiled_geometry_by_body[name]["compiled_bone_geometry_checks"]
            record["source_member_ids"] = compiled_geometry_by_body[name]["source_member_ids"]
            record["compiled_bone_geometry_checks"] = checks
            record["passed"] = record["passed"] and all(c["passed"] for c in checks)
        if name.startswith("toes_"):
            record["source_surface_fit_disposition"] = "inherited_complete_rigid_foot_transform_not_collective_toe_mesh_refit"
        else:
            vertices = np.concatenate([local_vertices[a["source"]["member_id"]][1] for a in anchors])
            training, held_out, training_count, held_out_count = _surface_split_metrics(
                vertices, source_local_vertices[body_id], np.eye(3), np.zeros(3), np,
            )
            gate = _BODY_GATES[name.rsplit("_", 1)[0]]["held_out_p90_m"]
            record.update({
                "training_surface_metrics": training, "held_out_surface_metrics": held_out,
                "training_vertex_count": training_count, "held_out_vertex_count": held_out_count,
                "maximum_held_out_p90_m": gate,
            })
            record["passed"] = record["passed"] and held_out["p90_m"] <= gate
            if name.startswith("femur_"):
                source_body = source_bodies[body_id]
                body_record = {
                    "source_body": source_body,
                    "vertices": _body_frame_to_core(vertices, source_body, np),
                    "source_vertices": _body_frame_to_core(source_local_vertices[body_id], source_body, np),
                }
                try:
                    articular = _femoral_head_articular_metrics(body_record, {
                        "rotation": np.eye(3), "translation": np.zeros(3), "uniform_scale": 1.,
                    }, np)
                except RuntimeError as error:
                    articular = {"passed": False, "error": str(error)}
                record["femoral_head_articular_gate"] = articular
                record["passed"] = record["passed"] and articular["passed"]
        source_geometry_checks.append(record)

    transitions = _continuity_transitions()
    expected_members = {
        member for _, first, second, _ in transitions for member in (first, second)
    }
    if not expected_members.issubset(local_vertices):
        missing_members = sorted(expected_members - local_vertices.keys())
        raise RuntimeError(
            "lower-limb pose audit is missing transition members: "
            + ", ".join(missing_members)
        )

    pose_receipts = []
    default_frame_maximum_residual = 0.0
    default_frame_worst_member = "none"
    all_continuity = []
    all_parity = []
    equality_count: int | None = None
    equality_maximum_correction = 0.0

    # Registration centroids are authored in the source model's literal qpos0
    # frame.  Patellofemoral equality projection intentionally moves each
    # patella by about 52 mm at knee_angle=0, so frame identity must be checked
    # before that runtime projection rather than conflated with the neutral
    # posed-continuity state below.
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    default_world_vertices = {
        member_id: (
            np.einsum(
                "ki,ji->kj", vertices, data.xmat[body_id].reshape(3, 3)
            ) + data.xpos[body_id]
        )
        for member_id, (body_id, vertices) in registered_local_vertices.items()
    }
    for anchor in anchors_by_member.values():
        member_id = anchor["source"]["member_id"]
        centroid = np.mean(default_world_vertices[member_id], axis=0)
        expected_centroid = np.asarray(
            anchor["registration"]["default_pose_vertex_centroid_world_m"],
            dtype=float,
        )
        residual = float(np.linalg.norm(centroid - expected_centroid))
        if residual > default_frame_maximum_residual:
            default_frame_maximum_residual = residual
            default_frame_worst_member = str(member_id)

    for pose_name, overrides in POSE_SUITE:
        qpos, current_equality_count, correction = _pose_qpos(
            model, overrides, mujoco, np
        )
        if equality_count is None:
            equality_count = current_equality_count
        elif equality_count != current_equality_count:
            raise RuntimeError("lower-limb pose audit equality coverage changed across poses")
        equality_maximum_correction = max(equality_maximum_correction, correction)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        world_vertices = {
            member_id: (
                np.einsum(
                    "ki,ji->kj", vertices, data.xmat[body_id].reshape(3, 3)
                ) + data.xpos[body_id]
            )
            for member_id, (body_id, vertices) in local_vertices.items()
        }
        source_world_vertices = {
            body_id: (
                np.einsum(
                    "ki,ji->kj", vertices, data.xmat[body_id].reshape(3, 3)
                ) + data.xpos[body_id]
            )
            for body_id, vertices in source_local_vertices.items()
        }

        continuity = []
        by_name: dict[str, dict[str, Any]] = {}
        for transition_name, first_member, second_member, rest_gate in transitions:
            gap, first_witness, second_witness = _minimum_gap(
                world_vertices[first_member], world_vertices[second_member], np
            )
            patch = _interface_patch_metrics(
                world_vertices[first_member], world_vertices[second_member], np
            )
            first_body_id = local_vertices[first_member][0]
            second_body_id = local_vertices[second_member][0]
            mechanics_reference = None
            if first_body_id != second_body_id:
                source_gap, _, _ = _minimum_gap(
                    source_world_vertices[first_body_id],
                    source_world_vertices[second_body_id], np,
                )
                source_patch = _interface_patch_metrics(
                    source_world_vertices[first_body_id],
                    source_world_vertices[second_body_id], np,
                )
                mechanics_reference = {
                    "first_source_body_id": first_body_id,
                    "second_source_body_id": second_body_id,
                    "minimum_vertex_gap_m": source_gap,
                    "interface_patch": source_patch,
                }
            reference_allowance = (
                RIGID_TOE_COMPOUND_REFERENCE_ALLOWANCE_M
                if "metatarsal_to" in transition_name
                else MECHANICS_REFERENCE_INTERFACE_ALLOWANCE_M
            )
            base_posed_gate, base_posed_patch_gate, posed_gate, posed_patch_gate = _posed_continuity_gates(
                rest_gate, mechanics_reference, reference_allowance,
            )
            record = {
                "name": transition_name,
                "source_member_ids": [first_member, second_member],
                "minimum_vertex_gap_m": gap,
                "minimum_gap_witness_world_m": [first_witness.tolist(), second_witness.tolist()],
                "rest_maximum_allowed_gap_m": rest_gate,
                "base_posed_maximum_allowed_gap_m": base_posed_gate,
                "posed_maximum_allowed_gap_m": posed_gate,
                "interface_patch": patch,
                "mechanics_reference_interface": mechanics_reference,
                "mechanics_reference_interface_allowance_m": reference_allowance,
                "base_posed_maximum_allowed_interface_patch_p90_m": (
                    base_posed_patch_gate
                ),
                "posed_maximum_allowed_interface_patch_p90_m": posed_patch_gate,
                "passed": (
                    gap <= posed_gate + 1.0e-12
                    and patch["bidirectional_p90_m"] <= posed_patch_gate + 1.0e-12
                ),
            }
            continuity.append(record)
            by_name[transition_name] = record
            all_continuity.append({"pose": pose_name, **record})

        parity = _bilateral_interface_parity(by_name)
        all_parity.extend({"pose": pose_name, **record} for record in parity)

        pose_receipts.append({
            "name": pose_name,
            "source_q_overrides": [
                {"q_index": index, "value": value} for index, value in overrides
            ],
            "joint_equality_count": current_equality_count,
            "joint_equality_maximum_correction": correction,
            "source_equality_projection_oracle": source_equality_projection_oracle(
                model, data, mujoco, PROJECTED_JOINT_RANGE_TOLERANCE,
            ),
            "projected_joint_range_checks": _projected_joint_range_checks(qpos, joint_ranges, np),
            "continuity": continuity,
            "bilateral_gap_parity": parity,
        })

    worst_continuity = max(
        all_continuity,
        key=lambda item: (
            item["minimum_vertex_gap_m"] / item["posed_maximum_allowed_gap_m"]
        ),
    )
    worst_interface_patch = max(
        all_continuity,
        key=lambda item: (
            item["interface_patch"]["bidirectional_p90_m"]
            / item["posed_maximum_allowed_interface_patch_p90_m"]
        ),
    )
    worst_parity = max(
        all_parity,
        key=lambda item: max(
            item["absolute_gap_difference_m"],
            item["absolute_interface_patch_p90_difference_m"],
        ),
    )
    result = {
        "schema": SCHEMA,
        "status": "passed_source_owned_bilateral_lower_limb_multi_pose_interface_patches",
        "inputs": {
            "registration": {
                "file": registration_path.name,
                "sha256": _sha256(registration_path),
            },
            "runtime_reference": runtime_reference,
            "evaluated_geometry": (
                "decoded_NHBONES1_source_owned_vertices_and_triangles"
                if bone_descriptor is not None else "registration_source_OBJ_vertices"
            ),
            **({"bone_payload": bone_descriptor} if bone_descriptor is not None else {}),
            "myosim_archive_sha256": registration["source"]["myosim"]["source"][
                "archive_sha256"
            ],
        },
        "source_body_count": len(LOWER_BODY_NAMES),
        "source_member_count": len(local_vertices),
        "source_common_frame": common_frame,
        "source_geometry_checks": source_geometry_checks,
        "pose_count": len(POSE_SUITE),
        "continuity_transition_count_per_pose": len(transitions),
        "continuity_evaluation_count": len(all_continuity),
        "bilateral_parity_evaluation_count": len(all_parity),
        "joint_equality_count": equality_count,
        "joint_equality_program_checks": equality_programs,
        "joint_equality_driver_domain_audit": joint_equality_driver_domain_audit(
            exported, joint_ranges, np, PROJECTED_JOINT_RANGE_TOLERANCE,
        ),
        "rigid_source_program_checks": rigid_program,
        "joint_equality_maximum_correction": equality_maximum_correction,
        "default_frame_maximum_centroid_residual_m": default_frame_maximum_residual,
        "default_frame_worst_member": default_frame_worst_member,
        "default_frame_maximum_allowed_residual_m": DEFAULT_FRAME_RESIDUAL_MAXIMUM_M,
        "posed_continuity_allowance_m": POSE_CONTINUITY_ALLOWANCE_M,
        "mechanics_reference_interface_allowance_m": (
            MECHANICS_REFERENCE_INTERFACE_ALLOWANCE_M
        ),
        "rigid_toe_compound_reference_allowance_m": (
            RIGID_TOE_COMPOUND_REFERENCE_ALLOWANCE_M
        ),
        "interface_patch_gate_multiplier": INTERFACE_PATCH_GATE_MULTIPLIER,
        "bilateral_gap_parity_maximum_m": BILATERAL_GAP_PARITY_MAXIMUM_M,
        "worst_continuity": worst_continuity,
        "worst_interface_patch": worst_interface_patch,
        "worst_bilateral_gap_parity": worst_parity,
        "poses": pose_receipts,
        "evidence_boundary": (
            (
                "The decoded NHBONES1 payload was compared with the pinned source geometry "
                "after the specified FP32 round trip, including topology and NHRIGID "
                "source/Core owners. Pose metrics below use those decoded payload vertices. "
                if bone_descriptor is not None else
                "Pose metrics below use registered source OBJ geometry; no compiled bone "
                "payload was evaluated. "
            ) +
            "Rigid BodyParts3D lower-limb bones were replayed through pinned MyoSim "
            "kinematics and exact polynomial joint-equality projection. Passing proves "
            "body ownership, default-frame identity, freshly measured regional source-surface "
            "fits and femoral-head gates, total orientation/scale bounds against the source-derived "
            "atlas frame, bounded minimum-gap and robust "
            "bidirectional interface-patch continuity relative to the same-pose pinned "
            "mechanics surfaces, post-projection source and consumed native position ranges, and bilateral parity for this "
            "pose suite, with independent MuJoCo constraint-residual checks at every pose. "
            "A separate full-driver-domain diagnostic retains conflicts beyond the sampled poses "
            "without rewriting source ranges or polynomial laws. The complete toe compound retains one MTP body. This is not "
            "cartilage/contact, ligament restraint, loaded dynamics, gait, clinical "
            "registration, or a deformable tendon solve. Range coordinates are projected from "
            "the source model and rounded to FP32. The consumed rigid program, including body "
            "inertia, joint axes/frames, DoF policy, default state and source mappings, is joined "
            "to a fresh lowering of the pinned source model. Normalized direction and antipodal "
            "quaternion representations retain the existing native admission bounds. Native equality program bytes are checked against "
            "the pinned source compiler, including declared compliance parameters. When a bone payload is "
            "supplied, its complete skeleton is checked against registered source geometry; only regional "
            "interfaces are posed. This audit does not "
            "execute those native programs or qualify their loaded response."
        ),
    }

    return _finish_pose_audit(result, "lower-limb")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--bone-artifact", type=Path,
        help="evaluate the decoded native NHBONES1 payload against source geometry and runtime owners",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    failure = None
    try:
        result = audit_lower_limb_poses(
            sources=arguments.sources.resolve(),
            registration_path=arguments.registration.resolve(),
            artifact=arguments.artifact.resolve(),
            bone_artifact=arguments.bone_artifact.resolve() if arguments.bone_artifact else None,
        )
    except PoseAuditError as error:
        result = error.result
        failure = error
    except (RuntimeError, OSError, ValueError) as error:
        print(f"numilab-human lower-limb pose audit: {error}", file=sys.stderr)
        return 2
    result["inputs"]["reproduction_command"] = [
        sys.executable, "-m", "numilab_human.lower_limb_pose_audit",
        "--sources", str(arguments.sources.resolve()),
        "--artifact", str(arguments.artifact.resolve()),
        "--registration", str(arguments.registration.resolve()),
        "--output", str(arguments.output.resolve()),
    ]
    if arguments.bone_artifact is not None:
        result["inputs"]["reproduction_command"].extend([
            "--bone-artifact", str(arguments.bone_artifact.resolve()),
        ])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {arguments.output}")
    if failure is not None:
        print(f"numilab-human lower-limb pose audit: {failure}; diagnostics={arguments.output.resolve()}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
