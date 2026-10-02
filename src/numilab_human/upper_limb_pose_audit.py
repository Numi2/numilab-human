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
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from . import model as human_model
from .myosim_export import export_fullbody
from .joint_constraint_consistency import (
    joint_equality_driver_domain_audit, source_equality_projection_oracle,
)
from .upper_limb_registration import (
    INTERFACE_PATCH_GATE_MULTIPLIER,
    REGISTRATION_SCHEMA,
    SCHEMA as UPPER_REGISTRATION_SCHEMA,
    _interface_patch_metrics,
    _minimum_gap,
    _rotation_xyzw,
    _upper_names,
)


SCHEMA = "numi.human.bodyparts3d-myosim-upper-limb-multi-pose-audit.v1"
POSE_CONTINUITY_ALLOWANCE_M = 0.001
BILATERAL_GAP_PARITY_MAXIMUM_M = 0.002
DEFAULT_FRAME_RESIDUAL_MAXIMUM_M = 1.0e-9
# Match native --pose-q arithmetic admission, including its FP32 range table.
# This new post-projection gate does not change the 1e-12 input override gate.
PROJECTED_JOINT_RANGE_TOLERANCE = 1.0e-9


class PoseAuditError(RuntimeError):
    """A failed measurement retains the existing pose-audit diagnostic output."""

    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result


def _finish_pose_audit(result: dict[str, Any], region: str) -> dict[str, Any]:
    failures = []
    rigid = result.get("rigid_source_program_checks")
    if rigid is not None and not rigid["passed"]:
        failures.append(
            f"consumed rigid source program mismatch: affected_fields={rigid['failures']} "
            f"declared_identity_matches={rigid['declared_identity_matches']} "
            f"source_metadata_matches={rigid['source_metadata_matches']} "
            f"actual_sha256={rigid['actual_sha256']} expected_source_sha256={rigid['expected_source_sha256']} "
            f"tolerance_basis={rigid['tolerance_basis']}"
        )
    for program in result.get("joint_equality_program_checks", []):
        if not program["passed"]:
            failures.append(
                f"{program['payload_role']}:source equality program mismatch "
                f"affected_equalities={program['affected_equalities']} "
                f"source_byte_mismatch_count={program['source_byte_mismatch_count']} "
                f"allowed_byte_mismatch_count=0 tolerance_basis={program['tolerance_basis']} "
                f"declared_identity_matches={program['declared_identity_matches']} "
                f"actual_sha256={program['actual_sha256']} expected_source_sha256={program['expected_source_sha256']}"
            )
    residual = result["default_frame_maximum_centroid_residual_m"]
    allowed = result["default_frame_maximum_allowed_residual_m"]
    if residual > allowed:
        failures.append(
            f"default source/Core frame: member={result.get('default_frame_worst_member')} "
            f"residual_m={residual:.12g}, allowed_m={allowed:.12g}"
        )
    for item in result.get("source_geometry_checks", []):
        if item["passed"]:
            continue
        details = []
        for check in item["source_frame_checks"]:
            if not check["passed"]:
                details.append(
                    f"{check['source_member_id']}:atlas_rotation_rad={check['atlas_relative_rotation_angle_rad']:.12g},"
                    f"allowed_rad={check['maximum_rotation_angle_rad']:.12g},"
                    f"scale={check['atlas_relative_uniform_scale']:.12g},allowed_scale={check['uniform_scale_bounds']}"
                )
        for check in item.get("compiled_bone_geometry_checks", []):
            if not check["passed"]:
                details.append(
                    f"{check['source_member_id']}:compiled_vertex_residual_m="
                    f"{check['maximum_vertex_residual_m']},allowed_m=0,"
                    f"structure={check.get('source_member_name', check['source_member_id'])},"
                    f"topology_matches={check['source_topology_matches']},"
                    f"vertices={check['compiled_vertex_count']}/{check['source_vertex_count']},"
                    f"coverage={check.get('coverage_status', 'present')},"
                    f"registration_occurrences={check.get('registration_occurrence_count', 1)}/1,"
                    f"source_identity_matches={check.get('registered_source_identity_matches', True)},"
                    f"tolerance_basis={check['tolerance_basis']}"
                )
        surface = item.get("held_out_surface_metrics")
        if surface is not None and surface["p90_m"] > item["maximum_held_out_p90_m"]:
            details.append(f"source_surface_p90_m={surface['p90_m']:.12g},allowed_m={item['maximum_held_out_p90_m']:.12g}")
        articular = item.get("femoral_head_articular_gate")
        if articular is not None and not articular["passed"]:
            details.append(f"femoral_head_articular_gate={articular}")
        failures.append(f"{region} source geometry:{item['myosim_body']} members={item['source_member_ids']}: " + "; ".join(details))
    for pose in result["poses"]:
        for item in pose.get('patellar_anteriority', []):
            if not item['passed']:
                failures.append(
                    f"{pose['name']}:patella_{item['side']} member={item['source_member_id']} "
                    f"minimum_anterior_offset_m={item['minimum_signed_anterior_offset_m']:.12g} "
                    f"posterior_or_plane_vertices={item['vertices_posterior_or_on_knee_anchor_plane']}"
                )
        oracle = pose.get("source_equality_projection_oracle")
        if oracle is not None and not oracle["passed"]:
            failed_rows = [row["name"] for row in oracle["rows"] if not row["passed"]]
            failures.append(
                f"{pose['name']}:source equality projection differs from MuJoCo oracle "
                f"failed_equalities={failed_rows} unique_coverage={oracle['complete_unique_coverage']} "
                f"measured={oracle['measured_joint_equalities']}/{oracle['expected_active_joint_equalities']} "
                f"maximum_residual={oracle['maximum_absolute_residual']:.12g} "
                f"allowed={oracle['maximum_allowed_residual']:.12g} "
                f"tolerance_basis={oracle['tolerance_basis']}"
            )
        for item in pose.get("projected_joint_range_checks", []):
            if not item["passed"]:
                failures.append(
                    f"{pose['name']}:{item['source_joint_name']} q_index={item['q_index']} "
                    f"projected_value={item['projected_value']:.12g} unit={item['unit']} "
                    f"source_range={item['source_range']} source_violation={item['source_range_violation']:.12g}, "
                    f"native_position_limit_enabled={item['native_position_limit_enabled']} "
                    f"native_range={item['native_position_range']} native_violation={item['native_range_violation']:.12g}, "
                    f"allowed_violation={item['maximum_allowed_range_violation']:.12g} "
                    f"tolerance_basis={item['tolerance_basis']}"
                )
        for item in pose["continuity"]:
            if not item["passed"]:
                failures.append(
                    f"{pose['name']}:{item['name']} members={item['source_member_ids']} "
                    f"minimum_gap_m={item['minimum_vertex_gap_m']:.12g}, "
                    f"allowed_gap_m={item['posed_maximum_allowed_gap_m']:.12g}, "
                    f"patch_p90_m={item['interface_patch']['bidirectional_p90_m']:.12g}, "
                    f"allowed_patch_p90_m={item['posed_maximum_allowed_interface_patch_p90_m']:.12g}"
                )
        for item in pose["bilateral_gap_parity"]:
            if not item["passed"]:
                failures.append(
                    f"{pose['name']}:{item['transition']} bilateral parity: "
                    f"gap_difference_m={item['absolute_gap_difference_m']:.12g}, "
                    f"patch_difference_m={item['absolute_interface_patch_p90_difference_m']:.12g}, "
                    f"allowed_m={item['maximum_allowed_difference_m']:.12g}"
                )
    result["failures"] = failures
    if failures:
        result["status"] = result["status"].replace("passed_", "failed_", 1)
        raise PoseAuditError(
            f"{region} pose audit failed: " + "; ".join(failures[:12])
            + f"; registration_sha256={result['inputs']['registration']['sha256']}"
            + f"; rigid_sha256={result['inputs']['runtime_reference']['rigid']['sha256']}"
            + (f"; bone_sha256={result['inputs']['bone_payload']['sha256']}"
               if "bone_payload" in result["inputs"] else ""),
            result,
        )
    return result

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
    ("bilateral_coupled_reach", (
        (36, 0.8), (39, 1.0), (40, 0.7), (41, 0.15), (42, 0.3),
        (74, 0.8), (77, 1.0), (78, 0.7), (79, 0.15), (80, 0.3),
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


def _compiled_bone_members(
    bone_artifact: Path, registration_path: Path,
    registration: dict[str, Any], runtime_reference: dict[str, Any],
    runtime_bodies: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Use the owning NHBONES reader, then join its owners to consumed NHRIGID."""
    by_body, descriptor, payload_path = human_model._numi_human_bone_envelope_surfaces(
        bone_artifact, runtime_reference["source_archive_sha256"],
    )
    identity = (
        f"bone_sha256={descriptor['sha256']}; "
        f"rigid_sha256={runtime_reference['rigid']['sha256']}"
    )
    if descriptor["payload_abi"] != 3:
        raise RuntimeError(
            "compiled bone geometry requires NHBONES1 ABI 3 source owners; " + identity
        )
    manifest = human_model.read_json(
        bone_artifact / "bodyparts3d-myosim-major-bones.manifest.json"
    )
    source = manifest["source"]
    registration_sha = _sha256(registration_path)
    compiled_registration = source.get("registration")
    if (
        not isinstance(compiled_registration, dict)
        or compiled_registration.get("sha256") != registration_sha
        or descriptor["registration_fingerprint32"] != registration_sha[:8]
    ):
        raise RuntimeError(
            "compiled bone geometry registration identity drifted; "
            f"registration_sha256={registration_sha}; " + identity
        )
    compiled_reference = source.get("runtime_reference", {})
    for owner in ("rigid", "manifest"):
        compiled_owner = compiled_reference.get(owner) if isinstance(compiled_reference, dict) else None
        if not isinstance(compiled_owner, dict) or compiled_owner.get("sha256") != runtime_reference[owner]["sha256"]:
            raise RuntimeError(
                f"compiled bone geometry {owner} identity drifted; " + identity
            )
    registered = {
        anchor["source"]["member_id"]: anchor for anchor in registration["anchors"]
    }
    members: dict[str, dict[str, Any]] = {}
    for core_body_index, surfaces in by_body.items():
        for surface in surfaces:
            member_id = surface["member_id"]
            if member_id in members or member_id not in registered:
                raise RuntimeError(
                    f"compiled bone geometry duplicate or unknown member={member_id}; " + identity
                )
            anchor = registered[member_id]
            name = anchor["target"]["name"]
            expected_core, body = runtime_bodies[name]
            provenance = source["anchors"][surface["stable_id"] - 1]
            if (
                core_body_index != expected_core
                or provenance.get("source_record_index") != body["source_record_index"]
                or provenance.get("myosim_body") != name
                or provenance.get("member_sha256") != anchor["source"]["member_sha256"]
            ):
                raise RuntimeError(
                    f"compiled bone geometry member={member_id} ({name}) owner/source mismatch: "
                    f"Core={core_body_index}, source_record={provenance.get('source_record_index')}; "
                    f"NHRIGID requires Core={expected_core}, source_record={body['source_record_index']}; "
                    + identity
                )
            members[member_id] = surface
    if members.keys() != registered.keys():
        raise RuntimeError("compiled bone geometry member coverage drifted; " + identity)
    return members, {**descriptor, "file": str(payload_path.resolve())}


def _compiled_member_geometry_check(
    anchor: dict[str, Any], raw_vertices: list[Any], raw_triangles: list[Any],
    surface: dict[str, Any], np: Any,
) -> dict[str, Any]:
    """Compare source geometry after the format's specified FP32 round trip.

    This is an identity check, not a larger anatomical tolerance. The source
    frame and regional gates still use their original bounds.
    """
    translation, quaternion, scale = human_model._bodyparts_visual_local_pose(
        anchor["registration"]["source_obj_mm_to_core_inertial_body_m"],
        f"compiled bone geometry {anchor['source']['member_id']}",
    )
    pose = np.asarray([*translation, *quaternion, scale], dtype=np.float32).astype(float)
    rotation = human_model._myosim_matrix_from_quaternion_xyzw(pose[3:7].tolist())
    source_meters = np.asarray([
        [coordinate * .001 for coordinate in vertex] for vertex in raw_vertices
    ], dtype=np.float32).astype(float)
    expected = np.asarray([
        human_model._myosim_add(pose[:3].tolist(), human_model._myosim_matrix_vector(
            rotation, [float(pose[7]) * float(value) for value in vertex],
        )) for vertex in source_meters
    ])
    actual = np.asarray(surface["vertices"], dtype=float)
    shape_matches = actual.shape == expected.shape
    finite = bool(np.all(np.isfinite(actual)))
    residual = (
        float(np.max(np.linalg.norm(actual - expected, axis=1)))
        if shape_matches and finite else None
    )
    topology_matches = surface["triangles"] == [tuple(t) for t in raw_triangles]
    return {
        "source_member_id": anchor["source"]["member_id"],
        "myosim_body": anchor["target"]["name"],
        "core_body_index": anchor["target"]["core_body_index"],
        "source_vertex_count": len(raw_vertices),
        "compiled_vertex_count": len(actual),
        "source_triangle_count": len(raw_triangles),
        "compiled_triangle_count": len(surface["triangles"]),
        "maximum_vertex_residual_m": residual,
        "maximum_allowed_vertex_residual_m": 0.0,
        "tolerance_basis": "exact_source_geometry_after_NHBONES1_FP32_round_trip",
        "source_topology_matches": topology_matches,
        "passed": shape_matches and finite and residual == 0.0 and topology_matches,
    }


def _compiled_bone_geometry_checks(
    sources: Path, registration: dict[str, Any],
    members: dict[str, dict[str, Any]], np: Any,
) -> dict[str, dict[str, Any]]:
    """Check the complete owning skeleton contract, including non-limb bones."""
    registered = {anchor["source"]["member_id"]: anchor for anchor in registration["anchors"]}
    occurrences = Counter(anchor["source"]["member_id"] for anchor in registration["anchors"])
    groups: dict[str, dict[str, Any]] = {}
    archive_hashes: dict[Path, str] = {}
    specifications = {spec["member_id"]: spec for spec in human_model._BODYPARTS_MYOSIM_BONE_ANCHORS}
    for member_id in sorted(specifications.keys() | registered.keys() | members.keys()):
        spec = specifications.get(member_id)
        anchor, surface = registered.get(member_id), members.get(member_id)
        name = spec["myosim_body"] if spec else anchor["target"]["name"] if anchor else "unknown"
        raw_vertices, raw_triangles = [], []
        identity_matches = False
        if spec is not None:
            archive, member, obj = human_model._bodyparts_obj_member(sources, spec["hierarchy"], member_id)
            if archive not in archive_hashes:
                archive_hashes[archive] = _sha256(archive)
            raw_vertices, raw_triangles = human_model._bodyparts_obj_triangles(obj, member)
            if anchor is not None:
                source = anchor["source"]
                identity_matches = (
                    source.get("hierarchy") == spec["hierarchy"]
                    and source.get("name") == spec["bodyparts_name"]
                    and source.get("archive_sha256") == archive_hashes[archive]
                    and source.get("member_sha256") == hashlib.sha256(obj).hexdigest()
                    and source.get("vertex_count") == len(raw_vertices)
                    and source.get("triangle_count") == len(raw_triangles)
                    and anchor["target"]["name"] == name
                )
        if spec is not None and anchor is not None and surface is not None:
            check = _compiled_member_geometry_check(anchor, raw_vertices, raw_triangles, surface, np)
        else:
            check = {
                "source_member_id": member_id, "myosim_body": name,
                "source_vertex_count": len(raw_vertices),
                "compiled_vertex_count": len(surface["vertices"]) if surface else 0,
                "source_triangle_count": len(raw_triangles),
                "compiled_triangle_count": len(surface["triangles"]) if surface else 0,
                "maximum_vertex_residual_m": None, "maximum_allowed_vertex_residual_m": 0.0,
                "source_topology_matches": False, "passed": False,
                "coverage_status": "unexpected_source_member" if spec is None else
                    "missing_registration" if anchor is None else "missing_compiled_member",
                "tolerance_basis": "existing_complete_BodyParts3D_visual_skeleton_anchor_contract",
            }
        check["source_member_name"] = spec["bodyparts_name"] if spec else anchor["source"].get("name", "unknown") if anchor else "unknown"
        check["registered_source_identity_matches"] = identity_matches
        if spec is not None and anchor is not None and not identity_matches:
            check["expected_source_identity"] = {
                "hierarchy": spec["hierarchy"], "name": spec["bodyparts_name"], "myosim_body": name,
                "archive_sha256": archive_hashes[archive], "member_sha256": hashlib.sha256(obj).hexdigest(),
                "vertex_count": len(raw_vertices), "triangle_count": len(raw_triangles),
            }
            check["registered_source_identity"] = {**anchor["source"], "myosim_body": anchor["target"]["name"]}
        check["passed"] = check["passed"] and identity_matches
        if occurrences[member_id] > 1:
            check.update(passed=False, coverage_status="duplicate_registration",
                         registration_occurrence_count=occurrences[member_id])
        group = groups.setdefault(name, {
            "myosim_body": name, "source_member_ids": [], "source_frame_checks": [],
            "compiled_bone_geometry_checks": [], "passed": True,
        })
        group["source_member_ids"].append(member_id)
        group["compiled_bone_geometry_checks"].append(check)
        group["passed"] = group["passed"] and check["passed"]
    return groups


def _project_joint_equalities(model: Any, qpos: Any, mujoco: Any) -> tuple[int, float]:
    count = 0
    maximum_correction = 0.0
    for equality_index in range(model.neq):
        if (
            not bool(model.eq_active0[equality_index])
            or int(model.eq_type[equality_index]) != int(mujoco.mjtEq.mjEQ_JOINT)
        ):
            continue
        dependent_joint = int(model.eq_obj1id[equality_index])
        driver_joint = int(model.eq_obj2id[equality_index])
        dependent_q = int(model.jnt_qposadr[dependent_joint])
        driver_q = int(model.jnt_qposadr[driver_joint]) if driver_joint >= 0 else None
        driver = float(qpos[driver_q] - model.qpos0[driver_q]) if driver_q is not None else 0.0
        coefficients = model.eq_data[equality_index, :5]
        projected = float(model.qpos0[dependent_q]) + sum(
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


def _joint_equality_correction_by_unit(
    model: Any, before: Any, after: Any, mujoco: Any,
) -> dict[str, dict[str, Any] | None]:
    """Report the largest prescribed-coordinate projection separately by unit."""
    maxima: dict[str, dict[str, Any] | None] = {"m": None, "rad": None}
    for equality_index in range(model.neq):
        if (
            not bool(model.eq_active0[equality_index])
            or int(model.eq_type[equality_index]) != int(mujoco.mjtEq.mjEQ_JOINT)
        ):
            continue
        dependent_joint = int(model.eq_obj1id[equality_index])
        dependent_type = int(model.jnt_type[dependent_joint])
        if dependent_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
            unit = "m"
        elif dependent_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            unit = "rad"
        else:
            raise RuntimeError(
                "joint-equality correction diagnostics require scalar slide or hinge "
                f"coordinates; joint_id={dependent_joint}, type={dependent_type}"
            )
        q_index = int(model.jnt_qposadr[dependent_joint])
        signed = float(after[q_index] - before[q_index])
        magnitude = abs(signed)
        current = maxima[unit]
        if current is None or magnitude > current["maximum_absolute_correction"]:
            driver_joint = int(model.eq_obj2id[equality_index])
            maxima[unit] = {
                "maximum_absolute_correction": magnitude,
                "signed_correction": signed,
                "dependent_joint": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_JOINT, dependent_joint
                ),
                "driver_joint": (
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, driver_joint)
                    if driver_joint >= 0 else None
                ),
                "qpos_index": q_index,
            }
    return maxima


def _pose_qpos_with_unit_metrics(
    model: Any, pose: tuple[tuple[int, float], ...], mujoco: Any, np: Any,
) -> tuple[Any, int, float, dict[str, dict[str, Any] | None]]:
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
    requested = qpos.copy()
    equality_count, maximum_correction = _project_joint_equalities(model, qpos, mujoco)
    by_unit = _joint_equality_correction_by_unit(model, requested, qpos, mujoco)
    return qpos, equality_count, maximum_correction, by_unit


def _pose_qpos(
    model: Any, pose: tuple[tuple[int, float], ...], mujoco: Any, np: Any,
) -> tuple[Any, int, float]:
    """Backward-compatible projected pose without dimensional diagnostics."""
    qpos, equality_count, maximum_correction, _ = _pose_qpos_with_unit_metrics(
        model, pose, mujoco, np
    )
    return qpos, equality_count, maximum_correction


def _pose_joint_range_context(
    artifact: Path, runtime_reference: dict[str, Any], model: Any, mujoco: Any,
) -> list[dict[str, Any]]:
    """Join source joints to the exact consumed NHRIGID2 range flags/bytes."""
    manifest_path = Path(runtime_reference["manifest"]["file"])
    if _sha256(manifest_path) != runtime_reference["manifest"]["sha256"]:
        raise RuntimeError("pose audit runtime manifest drifted during range binding")
    manifest = human_model.read_json(manifest_path)
    context = human_model._numi_human_fixed_cluster_context(
        artifact, manifest, manifest["payloads"]["rigid"],
    )
    scalar_joints = {i for i in range(model.njnt) if int(model.jnt_type[i]) in (
        int(mujoco.mjtJoint.mjJNT_SLIDE), int(mujoco.mjtJoint.mjJNT_HINGE),
    )}
    joined = []
    seen = set()
    for row in manifest["core_tree"]["source_joint_map"]:
        joint = row["source_joint_id"]
        v = row["core_v_index"]
        if type(joint) is not int or joint not in scalar_joints or joint in seen:
            raise RuntimeError("pose audit source joint map is incomplete or duplicated")
        if type(v) is not int or not 0 <= v < len(context["dof_properties"]):
            raise RuntimeError("pose audit source joint map has an invalid native DoF")
        native = context["dof_properties"][v]
        enforced = row["core_limit_status"] == "enforced"
        native_range = [struct.unpack("<f", struct.pack("<f", x))[0] for x in row["source_range"]]
        if (
            row["source_name"] != mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            or row["source_type"] != int(model.jnt_type[joint])
            or row["core_q_index"] != int(model.jnt_qposadr[joint])
            or row["source_limited"] != bool(model.jnt_limited[joint])
            or list(row["source_range"]) != [float(x) for x in model.jnt_range[joint]]
            or native["q_index"] != row["core_q_index"] or native["v_index"] != v
            or native["joint_index"] != row["core_joint_index"]
            or bool(native["flags"] & human_model._MR_DOF_POSITION_LIMIT) != enforced
            or native["position_range"] != (native_range if enforced else [0.0, 0.0])
        ):
            raise RuntimeError(
                f"pose audit source/native joint range binding drifted: {row['source_name']} "
                f"q_index={row['core_q_index']} v_index={v} source_range={row['source_range']} "
                f"declared_native_enforced={enforced} consumed_flags={native['flags']} "
                f"consumed_native_range={native['position_range']}; "
                f"rigid_sha256={context['rigid_payload_sha256']}"
            )
        seen.add(joint)
        joined.append({**row, "native_dof": native})
    if seen != scalar_joints:
        raise RuntimeError("pose audit source/native scalar joint range coverage drifted")
    return joined


def _joint_equality_program_checks(
    artifact: Path, runtime_reference: dict[str, Any], exported: dict[str, Any],
    joints: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare native motion/constraint programs with the owning source compiler."""
    manifest_path = Path(runtime_reference["manifest"]["file"])
    if _sha256(manifest_path) != runtime_reference["manifest"]["sha256"]:
        raise RuntimeError("pose audit runtime manifest drifted during equality binding")
    manifest = human_model.read_json(manifest_path)
    rows, projection, compliance = human_model._myosim_joint_equality_bundle(
        exported["source"], exported["model"], exported["joint_equalities"],
        {joint["source_joint_id"]: joint for joint in joints},
        manifest["core_tree"]["nq"], manifest["core_tree"]["nv"],
    )
    programs = [("joint_equalities", projection, 96, 1)]
    if "joint_equalities_source_compliance" in manifest["payloads"]:
        if compliance is None:
            raise RuntimeError("pose audit declares native compliance without a source compliance law")
        programs.append(("joint_equalities_source_compliance", compliance, 112, 2))
    checks = []
    for role, expected, record_bytes, abi in programs:
        descriptor = manifest["payloads"].get(role, {})
        filename = descriptor.get("file")
        path = artifact / filename if isinstance(filename, str) else None
        actual = path.read_bytes() if path is not None and path.is_file() else None
        actual_sha = hashlib.sha256(actual).hexdigest() if actual is not None else None
        declared_matches = (
            actual is not None and descriptor.get("bytes") == len(actual)
            and descriptor.get("sha256") == actual_sha and descriptor.get("payload_abi") == abi
        )
        mismatch_count = (sum(a != b for a, b in zip(actual, expected))
                          + abs(len(actual) - len(expected))) if actual is not None else len(expected)
        affected = []
        for index, row in enumerate(rows):
            start = 80 + record_bytes * index
            if actual is not None and actual[start:start + record_bytes] == expected[start:start + record_bytes]:
                continue
            affected.append({
                "record_index": index, "source_equality_id": row["source_equality_id"],
                "name": row["name"], "dependent_name": row["dependent_name"],
                "dependent_core_q": row["dependent_core_q"], "dependent_core_v": row["dependent_core_v"],
                "master_name": row["master_name"], "master_core_q": row["master_core_q"],
            })
        checks.append({
            "payload_role": role, "file": str(path) if path is not None else None,
            "payload_abi": abi, "record_count": len(rows),
            "actual_sha256": actual_sha, "expected_source_sha256": hashlib.sha256(expected).hexdigest(),
            "actual_bytes": len(actual) if actual is not None else 0, "expected_bytes": len(expected),
            "declared_identity_matches": declared_matches,
            "source_header_matches": actual is not None and actual[:80] == expected[:80],
            "source_byte_mismatch_count": mismatch_count, "maximum_allowed_byte_mismatch_count": 0,
            "tolerance_basis": "exact_pinned_source_compiler_NHEQ_FP32_bytes_including_policy_references_and_compliance",
            "affected_equalities": affected,
            "passed": declared_matches and actual == expected,
        })
    return checks


def _projected_joint_range_checks(qpos: Any, joints: list[dict[str, Any]], np: Any) -> list[dict[str, Any]]:
    """Check source-projected q and its FP32 rounding against bound range bytes."""
    checks = []
    for joint in joints:
        native = joint["native_dof"]
        enabled = bool(native["flags"] & human_model._MR_DOF_POSITION_LIMIT)
        if not joint["source_limited"] and not enabled:
            continue
        value = float(qpos[joint["core_q_index"]])
        if not math.isfinite(value):
            raise RuntimeError(f"pose audit projected a nonfinite coordinate: {joint['source_name']}")
        consumed = float(np.float32(value))
        source_range = joint["source_range"] if joint["source_limited"] else None
        native_range = native["position_range"] if enabled else None
        source_violation = max(0.0, source_range[0] - value, value - source_range[1]) if source_range else 0.0
        native_violation = max(0.0, native_range[0] - consumed, consumed - native_range[1]) if native_range else 0.0
        checks.append({
            "source_joint_id": joint["source_joint_id"], "source_joint_name": joint["source_name"],
            "q_index": native["q_index"], "v_index": native["v_index"], "core_joint_index": native["joint_index"],
            "unit": "m" if joint["source_type"] == 2 else "rad",
            "projected_value": value, "projected_fp32_value": consumed,
            "source_range": source_range, "source_range_violation": source_violation,
            "native_position_limit_enabled": enabled, "native_position_range": native_range,
            "native_range_violation": native_violation,
            "compiler_limit_status": joint["core_limit_status"],
            "maximum_allowed_range_violation": PROJECTED_JOINT_RANGE_TOLERANCE,
            "tolerance_basis": "existing_native_pose_q_position_range_arithmetic_admission_1e-9",
            "source_range_passed": source_violation <= PROJECTED_JOINT_RANGE_TOLERANCE,
            "native_position_range_passed": native_violation <= PROJECTED_JOINT_RANGE_TOLERANCE,
            "passed": max(source_violation, native_violation) <= PROJECTED_JOINT_RANGE_TOLERANCE,
        })
    return checks


def audit_upper_limb_poses(
    *, sources: Path, registration_path: Path, artifact: Path,
    bone_artifact: Path | None = None,
) -> dict[str, Any]:
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

    exported = export_fullbody(sources)
    if exported.get("source") != registration["source"]["myosim"]["source"]:
        raise RuntimeError("upper-limb pose audit pinned source export differs from its registered runtime source")
    source_bodies = {int(body["id"]): body for body in exported["bodies"]}
    model = build_model("myofullbody")
    data = mujoco.MjData(model)
    joint_ranges = _pose_joint_range_context(artifact, runtime_reference, model, mujoco)
    equality_programs = _joint_equality_program_checks(artifact, runtime_reference, exported, joint_ranges)
    rigid_program = human_model._myosim_rigid_program_checks(artifact, exported)
    expected_names = _upper_names("r") | _upper_names("l")
    anchors_by_name: dict[str, dict[str, Any]] = {}
    local_vertices: dict[str, tuple[int, Any]] = {}
    registered_local_vertices: dict[str, tuple[int, Any]] = {}

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
        raw_vertices, raw_triangles = human_model._bodyparts_obj_triangles(obj, member)
        if (
            source.get("member_sha256") != hashlib.sha256(obj).hexdigest()
            or source.get("vertex_count") != len(raw_vertices)
            or source.get("triangle_count") != len(raw_triangles)
        ):
            raise RuntimeError(f"upper-limb pose audit pinned mesh identity drifted for {source['member_id']}")
        matrix = np.asarray(
            anchor["registration"]["source_obj_mm_to_core_inertial_body_m"], dtype=float
        )
        if matrix.shape != (4, 4) or not bool(np.all(np.isfinite(matrix))):
            raise RuntimeError(f"upper-limb pose audit has an invalid registration for {name}")
        core_vertices = np.einsum(
            "ki,ji->kj", np.asarray(raw_vertices, dtype=float), matrix[:3, :3]
        ) + matrix[:3, 3]
        inertial_position = np.asarray(source_body["inertial_position_body_m"], dtype=float)
        inertial_rotation = _rotation_xyzw(
            source_body["inertial_quaternion_body_xyzw"], np
        )
        # _body_frame_to_core is (body - inertial_position) @ rotation.
        # This exact inverse returns admitted anatomy to the source body frame.
        body_vertices = np.einsum(
            "ki,ji->kj", core_vertices, inertial_rotation
        ) + inertial_position
        registered_local_vertices[source["member_id"]] = (source_body_id, body_vertices)
        if bone_descriptor is not None:
            surface = compiled_members[source["member_id"]]
            body_vertices = np.einsum(
                "ki,ji->kj", np.asarray(surface["vertices"], dtype=float),
                inertial_rotation,
            ) + inertial_position
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
    source_geometry_checks = list(compiled_geometry_by_body.values())

    pose_receipts = []
    default_frame_maximum_residual = 0.0
    default_frame_worst_member = "none"
    all_continuity = []
    all_parity = []
    equality_count: int | None = None
    equality_maximum_correction = 0.0
    equality_maximum_by_unit: dict[str, dict[str, Any] | None] = {
        "m": None,
        "rad": None,
    }
    for pose_name, overrides in POSE_SUITE:
        qpos, current_equality_count, correction, correction_by_unit = _pose_qpos_with_unit_metrics(
            model, overrides, mujoco, np
        )
        if equality_count is None:
            equality_count = current_equality_count
        elif equality_count != current_equality_count:
            raise RuntimeError("upper-limb pose audit equality coverage changed across poses")
        equality_maximum_correction = max(equality_maximum_correction, correction)
        for unit, record in correction_by_unit.items():
            current = equality_maximum_by_unit[unit]
            if record is not None and (
                current is None
                or record["maximum_absolute_correction"]
                > current["maximum_absolute_correction"]
            ):
                equality_maximum_by_unit[unit] = {"pose": pose_name, **record}
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

        if pose_name == "neutral":
            for name, anchor in anchors_by_name.items():
                member_id = anchor["source"]["member_id"]
                # Retain the strict registration-frame gate independently of
                # the compiled format's FP32 round trip. Motion/interface
                # measurements above use the actual decoded payload vertices.
                body_id, vertices = registered_local_vertices[member_id]
                centroid = np.mean(np.einsum(
                    "ki,ji->kj", vertices, data.xmat[body_id].reshape(3, 3)
                ) + data.xpos[body_id], axis=0)
                expected_centroid = np.asarray(
                    anchor["registration"]["default_pose_vertex_centroid_world_m"], dtype=float
                )
                residual = float(np.linalg.norm(centroid - expected_centroid))
                if residual > default_frame_maximum_residual:
                    default_frame_maximum_residual = residual
                    default_frame_worst_member = str(member_id)

        continuity = []
        by_name: dict[str, dict[str, Any]] = {}
        for transition_name, first_member, second_member, rest_gate in transitions:
            first_vertices = world_vertices[first_member]
            second_vertices = world_vertices[second_member]
            gap, first_witness, second_witness = _minimum_gap(
                first_vertices, second_vertices, np
            )
            patch = _interface_patch_metrics(first_vertices, second_vertices, np)
            posed_gate = rest_gate + POSE_CONTINUITY_ALLOWANCE_M
            posed_patch_gate = INTERFACE_PATCH_GATE_MULTIPLIER * posed_gate
            record = {
                "name": transition_name,
                "source_member_ids": [first_member, second_member],
                "minimum_vertex_gap_m": gap,
                "minimum_gap_witness_world_m": [first_witness.tolist(), second_witness.tolist()],
                "rest_maximum_allowed_gap_m": rest_gate,
                "posed_maximum_allowed_gap_m": posed_gate,
                "interface_patch": patch,
                "posed_maximum_allowed_interface_patch_p90_m": posed_patch_gate,
                "passed": (
                    gap <= posed_gate + 1.0e-12
                    and patch["bidirectional_p90_m"] <= posed_patch_gate + 1.0e-12
                ),
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
            patch_difference = abs(
                by_name[right_name]["interface_patch"]["bidirectional_p90_m"]
                - by_name[left_name]["interface_patch"]["bidirectional_p90_m"]
            )
            record = {
                "transition": suffix,
                "absolute_gap_difference_m": difference,
                "absolute_interface_patch_p90_difference_m": patch_difference,
                "maximum_allowed_difference_m": BILATERAL_GAP_PARITY_MAXIMUM_M,
                "passed": (
                    difference <= BILATERAL_GAP_PARITY_MAXIMUM_M + 1.0e-12
                    and patch_difference <= BILATERAL_GAP_PARITY_MAXIMUM_M + 1.0e-12
                ),
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
            "joint_equality_maximum_correction_by_unit": correction_by_unit,
            "source_equality_projection_oracle": source_equality_projection_oracle(
                model, data, mujoco, PROJECTED_JOINT_RANGE_TOLERANCE,
            ),
            "projected_joint_range_checks": _projected_joint_range_checks(qpos, joint_ranges, np),
            "continuity": continuity,
            "bilateral_gap_parity": parity,
        })

    worst_continuity = max(
        all_continuity,
        key=lambda item: item["minimum_vertex_gap_m"] - item["rest_maximum_allowed_gap_m"],
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
        "status": "passed_source_owned_bilateral_upper_limb_multi_pose_interface_patches",
        "inputs": {
            "registration": {
                "file": registration_path.name,
                "sha256": _sha256(registration_path),
            },
            "runtime_reference": runtime_reference,
            "myosim_archive_sha256": registration["source"]["myosim"]["source"][
                "archive_sha256"
            ],
        },
        "source_body_count": len(expected_names),
        "source_member_count": len(local_vertices),
        "source_geometry_checks": source_geometry_checks,
        "geometry_basis": (
            "consumed_NHBONES1_ABI3_vertices_and_source_owners"
            if bone_descriptor is not None else "registered_source_vertices"
        ),
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
        "joint_equality_maximum_correction_by_unit": equality_maximum_by_unit,
        "joint_equality_maximum_correction_unit_basis": (
            "legacy scalar is a raw maximum across mixed slide (m) and hinge (rad) "
            "coordinates; use the unit-separated values for physical interpretation"
        ),
        "default_frame_maximum_centroid_residual_m": default_frame_maximum_residual,
        "default_frame_worst_member": default_frame_worst_member,
        "default_frame_maximum_allowed_residual_m": DEFAULT_FRAME_RESIDUAL_MAXIMUM_M,
        "posed_continuity_allowance_m": POSE_CONTINUITY_ALLOWANCE_M,
        "interface_patch_gate_multiplier": INTERFACE_PATCH_GATE_MULTIPLIER,
        "bilateral_gap_parity_maximum_m": BILATERAL_GAP_PARITY_MAXIMUM_M,
        "worst_continuity": worst_continuity,
        "worst_interface_patch": worst_interface_patch,
        "worst_bilateral_gap_parity": worst_parity,
        "poses": pose_receipts,
        "evidence_boundary": (
            "Rigid BodyParts3D bones were replayed through pinned MyoSim kinematics and exact "
            "polynomial joint equality projection. Passing proves body ownership, default frame "
            "identity, bounded one-vertex and robust bidirectional interface-patch continuity, "
            "post-projection source and consumed native position ranges, and bilateral parity for this pose "
            "suite, with independent MuJoCo constraint-residual checks at every pose. A separate "
            "full-driver-domain diagnostic retains conflicts beyond the sampled poses without "
            "rewriting source ranges or polynomial laws. It is not cartilage/contact, ligament constraint, loaded dynamics, clinical "
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
    if bone_descriptor is not None:
        result["inputs"]["bone_payload"] = bone_descriptor
        result["evidence_boundary"] += (
            " NHBONES1 ABI 3 geometry was decoded by the owning payload reader, "
            "joined to the exact NHRIGID source owners and registration, and "
            "checked against source vertices/topology after the specified FP32 "
            "round trip before the pose measurements."
        )

    return _finish_pose_audit(result, "upper-limb")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--bone-artifact", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    failure = None
    try:
        result = audit_upper_limb_poses(
            sources=arguments.sources.resolve(),
            registration_path=arguments.registration.resolve(),
            artifact=arguments.artifact.resolve(),
            bone_artifact=arguments.bone_artifact.resolve() if arguments.bone_artifact else None,
        )
    except PoseAuditError as error:
        result = error.result
        failure = error
    except (RuntimeError, OSError, ValueError) as error:
        print(f"numilab-human upper-limb pose audit: {error}", file=sys.stderr)
        return 2
    result["inputs"]["reproduction_command"] = [
        sys.executable, "-m", "numilab_human.upper_limb_pose_audit",
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
        print(f"numilab-human upper-limb pose audit: {failure}; diagnostics={arguments.output.resolve()}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
