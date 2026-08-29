"""Preserve qualified anatomy while correcting Rajagopal rigid-foot ownership."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "numi.human.bodyparts3d-myosim-lower-limb-source-mesh-registration.v1"
REGISTRATION_SCHEMA = "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"

RIGID_FOOT_MEMBERS = {
    "FJ3308": "calcn_r", "FJ3377": "calcn_r", "FJ3370": "calcn_r",
    "FJ3373": "calcn_r", "FJ3364": "calcn_r", "FJ3351": "calcn_r",
    "FJ3353": "calcn_r", "FJ3355": "calcn_r", "FJ3357": "calcn_r",
    "FJ3359": "calcn_r",
    "FJ3307": "calcn_l", "FJ3271": "calcn_l", "FJ3264": "calcn_l",
    "FJ3267": "calcn_l", "FJ3258": "calcn_l", "FJ3241": "calcn_l",
    "FJ3244": "calcn_l", "FJ3247": "calcn_l", "FJ3250": "calcn_l",
    "FJ3253": "calcn_l",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _anchors_by_member(candidate: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    if candidate.get("schema") != REGISTRATION_SCHEMA:
        raise ValueError(f"{label} has an unsupported registration schema")
    anchors = candidate.get("anchors")
    if not isinstance(anchors, list):
        raise ValueError(f"{label} has no anchor table")
    result: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        member = anchor.get("source", {}).get("member_id") if isinstance(anchor, dict) else None
        if not isinstance(member, str) or member in result:
            raise ValueError(f"{label} has an invalid member identity")
        result[member] = anchor
    return result


def _rotation_xyzw(quaternion: Any) -> list[list[float]]:
    values = [float(value) for value in quaternion]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("lower-limb registration encountered an invalid quaternion")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError("lower-limb registration encountered a zero quaternion")
    x, y, z, w = (value / norm for value in values)
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _matmul(first: list[list[float]], second: list[list[float]]) -> list[list[float]]:
    return [
        [sum(first[row][inner] * second[inner][column] for inner in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[row][column] * vector[column] for column in range(3)) for row in range(3)]


def _reparent_preserving_world_pose(
    old: dict[str, Any], new_target_anchor: dict[str, Any],
) -> dict[str, Any]:
    """Express one unchanged world-space source mesh in a new rigid frame."""
    result = json.loads(json.dumps(old))
    old_target = old["target"]
    new_target = new_target_anchor["target"]
    old_rotation = _rotation_xyzw(old_target["default_inertial_quaternion_world_xyzw"])
    new_rotation = _rotation_xyzw(new_target["default_inertial_quaternion_world_xyzw"])
    new_world_to_core = _transpose(new_rotation)
    old_matrix = old["registration"]["source_obj_mm_to_core_inertial_body_m"]
    old_linear = [[float(old_matrix[row][column]) for column in range(3)] for row in range(3)]
    old_translation = [float(old_matrix[row][3]) for row in range(3)]
    new_linear = _matmul(_matmul(new_world_to_core, old_rotation), old_linear)
    old_position = [float(value) for value in old_target["default_com_position_world_m"]]
    new_position = [float(value) for value in new_target["default_com_position_world_m"]]
    world_translation = [
        value + old_position[index]
        for index, value in enumerate(_matvec(old_rotation, old_translation))
    ]
    new_translation = _matvec(
        new_world_to_core,
        [world_translation[index] - new_position[index] for index in range(3)],
    )
    result["target"] = json.loads(json.dumps(new_target))
    result["registration"] = {
        "source_obj_mm_to_core_inertial_body_m": [
            [*new_linear[row], new_translation[row]] for row in range(3)
        ] + [[0.0, 0.0, 0.0, 1.0]],
        "default_pose_vertex_centroid_world_m": old["registration"][
            "default_pose_vertex_centroid_world_m"
        ],
        "vertex_centroid_to_source_com_residual_m": None,
        "status": "rigid_foot_reparented_with_exact_default_world_pose_preservation",
        "rigid_foot_reparent": {
            "old_body": old_target["name"],
            "new_body": new_target["name"],
            "default_world_pose_delta_m": 0.0,
            "new_joint_count": 0,
        },
    }
    return result


def propose_lower_limb_registration(
    *, registration_path: Path, rigid_foot_base_path: Path,
) -> dict[str, Any]:
    """Rebind the midfoot as one coherent rigid-foot segment.

    The qualified candidate remains authoritative for every unaffected member.
    The freshly generated base supplies target body/frame records only for the
    exact 20 tarsal/metatarsal meshes whose old toe ownership was incorrect.
    No endpoint, mesh vertex, joint, or force parameter is edited.
    """
    prior = json.loads(registration_path.read_text(encoding="utf-8"))
    base = json.loads(rigid_foot_base_path.read_text(encoding="utf-8"))
    prior_by_member = _anchors_by_member(prior, "qualified registration")
    base_by_member = _anchors_by_member(base, "rigid-foot base registration")
    if set(prior_by_member) != set(base_by_member):
        raise ValueError("lower-limb registration anchor identity set drifted")
    prior_source = prior.get("source")
    if prior_source != base.get("source"):
        raise ValueError("lower-limb registration sources are not identical")

    replacements: dict[str, dict[str, Any]] = {}
    for member, expected_body in sorted(RIGID_FOOT_MEMBERS.items()):
        old = prior_by_member.get(member)
        replacement = base_by_member.get(member)
        if old is None or replacement is None:
            raise ValueError(f"lower-limb registration is missing {member}")
        old_body = old.get("target", {}).get("name")
        new_body = replacement.get("target", {}).get("name")
        if old_body != "toes_" + expected_body[-1] or new_body != expected_body:
            raise ValueError(
                f"lower-limb registration {member} does not change toes to rigid foot"
            )
        identity_fields = (
            "archive", "archive_sha256", "concept_id", "hierarchy", "member",
            "member_id", "member_sha256", "name", "triangle_count", "vertex_count",
        )
        if any(
            old.get("source", {}).get(field) != replacement.get("source", {}).get(field)
            for field in identity_fields
        ):
            raise ValueError(f"lower-limb registration source identity drifted for {member}")
        replacements[member] = _reparent_preserving_world_pose(old, replacement)

    output = json.loads(json.dumps(prior))
    output["anchors"] = [
        replacements.get(anchor["source"]["member_id"], anchor)
        for anchor in output["anchors"]
    ]
    receipt = {
        "schema": SCHEMA,
        "status": "candidate_corrected_rigid_foot_ownership_without_added_articulation",
        "inputs": {
            "qualified_registration": {
                "file": registration_path.name, "sha256": _sha256(registration_path),
            },
            "fresh_rigid_foot_base": {
                "file": rigid_foot_base_path.name, "sha256": _sha256(rigid_foot_base_path),
            },
        },
        "rebound_member_count": len(replacements),
        "rebound_members": [
            {"member_id": member, "old_body": "toes_" + body[-1], "new_body": body}
            for member, body in sorted(RIGID_FOOT_MEMBERS.items())
        ],
        "new_joint_count": 0,
        "endpoint_migration_m": 0.0,
        "promotion_requirement": (
            "Compile exact paired NHBONES1/NHTENDON2 artifacts, preserve every prior "
            "admitted law, and qualify foot continuity plus M4 Pro motion/visual evidence."
        ),
    }
    output["lower_limb_source_mesh_registration"] = receipt
    output["status"] = "provisional_visual_registration_not_admitted_to_collision_or_physics"
    output["evidence_boundary"] = (
        "This candidate corrects rigid-body ownership of exact BodyParts3D midfoot and "
        "metatarsal geometry while preserving the qualified upper-body registration. It "
        "does not move a MyoSim endpoint, add toe articulation, or by itself admit tendon mechanics."
    )
    return output
