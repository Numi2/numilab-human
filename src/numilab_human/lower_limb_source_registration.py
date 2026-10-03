"""Register BodyParts3D lower-limb groups to pinned MyoSim bone meshes.

BodyParts3D remains the emitted anatomy.  Long-bone segments receive one
bounded proper anthropometric similarity correction; short bones remain rigid,
and the collective toe compound inherits the rigid-foot correction instead of
receiving an independent fit.  No joint, route site, force parameter, or mesh
vertex is edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import model as human_model
from .myosim_bone_proximity import _compiled_meshes_by_body
from .myosim_export import export_fullbody
from .upper_limb_registration import (
    _body_frame_to_core,
    _core_to_world,
    _endpoint_surface_distances,
    _fit_candidates,
    _interface_patch_metrics,
    _minimum_gap,
    _robust_joint_centered_articular_sphere,
    _rotation_xyzw,
    _sample,
    _surface_split_metrics,
    _symmetric_metrics,
    _transform_points,
    _world_delta_to_core,
)


SCHEMA = "numi.human.bodyparts3d-myosim-lower-limb-source-mesh-registration.v3"
REGISTRATION_SCHEMA = "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
TENDON_SCHEMAS = {
    "numi.human.tendon-attachment-envelope-payload.v2",
    "numi.human.tendon-attachment-envelope-payload.v3",
}

# Surface-fit gates acknowledge population/atlas shape differences while
# rejecting reflections, anisotropic warps, wild flips, and corrections large
# enough to hide a source mismatch.  Only long bones may receive bounded
# isotropic anthropometric scale; foot and patellar geometry remains rigid.
_BODY_GATES = {
    "femur": {"held_out_p90_m": 0.015, "rotation_rad": 0.35, "translation_m": 0.025,
              "minimum_scale": 0.93, "maximum_scale": 1.07},
    "tibia": {"held_out_p90_m": 0.015, "rotation_rad": 0.25, "translation_m": 0.030,
              "minimum_scale": 0.93, "maximum_scale": 1.07},
    "talus": {"held_out_p90_m": 0.012, "rotation_rad": 0.70, "translation_m": 0.040,
              "minimum_scale": 1.0, "maximum_scale": 1.0},
    "calcn": {"held_out_p90_m": 0.015, "rotation_rad": 0.70, "translation_m": 0.040,
              "minimum_scale": 1.0, "maximum_scale": 1.0},
    "patella": {"held_out_p90_m": 0.012, "rotation_rad": 0.80, "translation_m": 0.080,
                "minimum_scale": 1.0, "maximum_scale": 1.0},
}
_BILATERAL_SYMMETRY_MEAN_MAXIMUM_M = 0.012
_FEMORAL_HEAD_SELECTION_RADIUS_M = 0.040
_FEMORAL_HEAD_CENTER_MAXIMUM_RESIDUAL_M = 0.003
_FEMORAL_HEAD_RADIUS_MAXIMUM_RESIDUAL_M = 0.004
_FEMORAL_HEAD_MECHANICS_CENTER_MAXIMUM_RESIDUAL_M = 0.005
_FEMORAL_HEAD_REFINEMENT_MAXIMUM_TRANSLATION_M = 0.006
_TOE_COMPOUND_ENTHESIS_TRANSLATION_MAXIMUM_M = 0.0065
_TOE_COMPOUND_ENTHESIS_DISTANCE_MAXIMUM_M = 0.020
_INTERFACE_TRANSLATION_REFINEMENT_MAXIMUM_M = 0.0015
_INTERFACE_TRANSLATION_REFINEMENT_STEPS_M = (
    0.0005, 0.00025, 0.000125, 0.0000625, 0.00003125,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_interface_translation(objective: Any, np: Any, *, group_count: int = 1) -> tuple[Any, Any, Any]:
    """Search the existing 1.5 mm registration bound; never enlarge a gate."""
    # Objectives are pure geometry measurements for one fixed active set.
    # Reuse exact repeated trial vectors without quantization or approximating
    # the distance/patch metric, especially for intersecting paired moves.
    measurements = {}
    uncached_objective = objective
    def objective(candidate: Any) -> Any:
        key = candidate.tobytes()
        if key not in measurements:
            measurements[key] = uncached_objective(candidate)
        return measurements[key]

    delta = np.zeros(3 if group_count == 1 else (group_count, 3))
    initial = objective(delta)
    if initial[0] <= 1.0 + 1.0e-12:
        return delta, initial, initial
    for step in _INTERFACE_TRANSLATION_REFINEMENT_STEPS_M:
        while True:
            best = objective(delta)
            best_delta = delta
            axes = list(np.ndindex(delta.shape))

            def trial(changes: tuple[Any, ...]) -> tuple[Any, Any]:
                candidate = delta.copy()
                for axis, sign in changes:
                    candidate[axis] += sign * step
                rows = candidate.reshape(-1, 3)
                norms = np.linalg.norm(rows, axis=1)
                # Project trial directions onto the same ball. Rejecting
                # every outward coordinate step can trap the search on an
                # axis even when a better direction lies on the boundary.
                for index in np.flatnonzero(norms > _INTERFACE_TRANSLATION_REFINEMENT_MAXIMUM_M):
                    rows[index] *= _INTERFACE_TRANSLATION_REFINEMENT_MAXIMUM_M / norms[index]
                return candidate, objective(candidate)

            for axis in axes:
                for sign in (-1.0, 1.0):
                    candidate, measured = trial(((axis, sign),))
                    if measured < best:
                        best, best_delta = measured, candidate
            if bool(np.array_equal(best_delta, delta)):
                # At intersecting interface constraints, either individual
                # owner move can make the worst error larger. A paired move
                # can reduce both without changing either owner's bound.
                for ordinal, first in enumerate(axes):
                    for second in axes[ordinal+1:]:
                        for first_sign in (-1., 1.):
                            for second_sign in (-1., 1.):
                                candidate, measured = trial(((first, first_sign), (second, second_sign)))
                                if measured < best:
                                    best, best_delta = measured, candidate
            if bool(np.array_equal(best_delta, delta)):
                break
            delta = best_delta
            if best[0] <= 1.0 + 1.0e-12:
                break
        if objective(delta)[0] <= 1.0 + 1.0e-12:
            break
    return delta, initial, objective(delta)


def _body_family(name: str) -> str:
    family = name.rsplit("_", 1)[0]
    if family not in _BODY_GATES:
        raise RuntimeError(f"lower-limb source registration has no gate for {name}")
    return family


def _fit_angle(rotation: Any, np: Any) -> float:
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5))
    return math.acos(cosine)


def _source_frame_check(
    anchor: dict[str, Any], common_frame: dict[str, Any], world_rotation: Any,
    matrix: Any = None,
) -> dict[str, Any]:
    """Measure the emitted transform against the independently reconstructed atlas frame."""
    name = anchor["target"]["name"]
    family = name.rsplit("_", 1)[0]
    gate = _BODY_GATES["calcn" if family == "toes" else family]
    local = matrix if matrix is not None else anchor["registration"]["source_obj_mm_to_core_inertial_body_m"]
    human_model._bodyparts_visual_local_pose(local, f"lower-limb {anchor['source']['member_id']} ({name})")
    linear = human_model._matrix_product(world_rotation, [row[:3] for row in local[:3]])
    common = [row[:3] for row in common_frame["global_source_mm_to_myosim_world_m"][:3]]
    scale = human_model._matrix3_determinant(linear) ** (1. / 3.)
    common_scale = human_model._matrix3_determinant(common) ** (1. / 3.)
    rotation = [[value / scale for value in row] for row in linear]
    common_rotation = [[value / common_scale for value in row] for row in common]
    relative = human_model._matrix_product(rotation, human_model._matrix_transpose(common_rotation))
    angle = math.acos(max(-1., min(1., (sum(relative[i][i] for i in range(3)) - 1.) / 2.)))
    ratio = scale / common_scale
    return {
        "source_member_id": anchor["source"]["member_id"], "myosim_body": name,
        "atlas_relative_rotation_angle_rad": angle, "maximum_rotation_angle_rad": gate["rotation_rad"],
        "atlas_relative_uniform_scale": ratio,
        "uniform_scale_bounds": [gate["minimum_scale"], gate["maximum_scale"]],
        "passed": (angle <= gate["rotation_rad"] + 1e-12
                   and gate["minimum_scale"] - 1e-12 <= ratio <= gate["maximum_scale"] + 1e-12),
    }


def _lower_limb_source_frame_checks(
    registration: dict[str, Any], common_frame: dict[str, Any], runtime_bodies: Any,
) -> list[dict[str, Any]]:
    return [
        _source_frame_check(anchor, common_frame, runtime_bodies[anchor["target"]["name"]][1]["rotation_world"])
        for anchor in registration["anchors"]
        if anchor["target"]["name"].rsplit("_", 1)[0] in {*_BODY_GATES, "toes"}
    ]


def _fit_source_frame_checks(record: dict[str, Any], fit: dict[str, Any], common_frame: Any, np: Any) -> list[dict[str, Any]]:
    checks = []
    for anchor in record["anchors"]:
        matrix = np.asarray(anchor["registration"]["source_obj_mm_to_core_inertial_body_m"]).copy()
        scale = float(fit.get("uniform_scale", 1.))
        matrix[:3, :3] = scale * fit["rotation"] @ matrix[:3, :3]
        matrix[:3, 3] = scale * fit["rotation"] @ matrix[:3, 3] + fit["translation"]
        checks.append(_source_frame_check(anchor, common_frame, record["runtime_body"]["rotation_world"], matrix.tolist()))
    return checks


def _fit_passes(name: str, fit: dict[str, Any], np: Any) -> bool:
    gate = _BODY_GATES[_body_family(name)]
    scale = float(fit.get("uniform_scale", 1.0))
    return bool(
        float(np.linalg.det(fit["rotation"])) > 0.999999
        and float(fit["held_out_metrics"]["p90_m"]) <= gate["held_out_p90_m"]
        and _fit_angle(fit["rotation"], np) <= gate["rotation_rad"] + 1.0e-12
        and float(np.linalg.norm(fit["translation"])) <= gate["translation_m"] + 1.0e-12
        and gate["minimum_scale"] - 1.0e-12 <= scale
        and scale <= gate["maximum_scale"] + 1.0e-12
        and fit.get("femoral_head_articular_gate", {"passed": True})["passed"]
        and all(check["passed"] for check in fit.get("source_frame_checks", []))
    )


def _source_relative_transition_metrics(
    *, name: str, first_member: str, second_member: str, rest_gate: float,
    first_vertices: Any, second_vertices: Any, reference_allowance: float,
    np: Any, source_first_vertices: Any = None, source_second_vertices: Any = None,
) -> dict[str, Any]:
    """Measure a candidate interface using the pose audit's source-relative gates."""
    gap, _, _ = _minimum_gap(first_vertices, second_vertices, np)
    patch = _interface_patch_metrics(first_vertices, second_vertices, np)
    source_reference = None
    if (source_first_vertices is None) != (source_second_vertices is None):
        raise RuntimeError("lower-limb interface source reference is incomplete")
    if source_first_vertices is not None:
        source_gap, _, _ = _minimum_gap(source_first_vertices, source_second_vertices, np)
        source_patch = _interface_patch_metrics(
            source_first_vertices, source_second_vertices, np,
        )
        source_reference = {
            "minimum_vertex_gap_m": source_gap,
            "interface_patch": source_patch,
        }
    from .lower_limb_pose_audit import _posed_continuity_gates

    base_gap, base_patch, allowed_gap, allowed_patch = _posed_continuity_gates(
        rest_gate, source_reference, reference_allowance,
    )
    return {
        "name": name,
        "source_member_ids": [first_member, second_member],
        "minimum_vertex_gap_m": gap,
        "base_maximum_allowed_gap_m": base_gap,
        "maximum_allowed_gap_m": allowed_gap,
        "interface_patch": patch,
        "base_maximum_allowed_interface_patch_p90_m": base_patch,
        "maximum_allowed_interface_patch_p90_m": allowed_patch,
        "mechanics_reference_interface": source_reference,
        "mechanics_reference_interface_allowance_m": reference_allowance,
        "passed": (
            gap <= allowed_gap + 1.0e-12
            and patch["bidirectional_p90_m"] <= allowed_patch + 1.0e-12
        ),
    }


def _femoral_head_articular_metrics(
    body_record: dict[str, Any], fit: dict[str, Any], np: Any,
) -> dict[str, Any]:
    """Gate the emitted femoral head against the mechanics hip center."""
    mechanics_center = _body_frame_to_core(
        np.zeros((1, 3)), body_record["source_body"], np
    )[0]
    source_center, source_radius, source_count = (
        _robust_joint_centered_articular_sphere(
            body_record["source_vertices"], mechanics_center,
            _FEMORAL_HEAD_SELECTION_RADIUS_M, np,
        )
    )
    candidate_center, candidate_radius, candidate_count = (
        _robust_joint_centered_articular_sphere(
            _transform_points(body_record["vertices"], fit, np),
            mechanics_center,
            _FEMORAL_HEAD_SELECTION_RADIUS_M,
            np,
        )
    )
    center_residual = float(np.linalg.norm(candidate_center - source_center))
    radius_residual = abs(candidate_radius - source_radius)
    source_axis_residual = float(np.linalg.norm(source_center - mechanics_center))
    candidate_axis_residual = float(
        np.linalg.norm(candidate_center - mechanics_center)
    )
    passed = bool(
        center_residual <= _FEMORAL_HEAD_CENTER_MAXIMUM_RESIDUAL_M
        and radius_residual <= _FEMORAL_HEAD_RADIUS_MAXIMUM_RESIDUAL_M
        and source_axis_residual <= _FEMORAL_HEAD_MECHANICS_CENTER_MAXIMUM_RESIDUAL_M
        and candidate_axis_residual <= _FEMORAL_HEAD_MECHANICS_CENTER_MAXIMUM_RESIDUAL_M
    )
    return {
        "method": "robust_proximal_articular_sphere_against_pinned_rajagopal_mechanics_mesh",
        "source_surface_vertex_count": source_count,
        "candidate_surface_vertex_count": candidate_count,
        "source_radius_m": source_radius,
        "candidate_radius_m": candidate_radius,
        "source_center_core_m": [float(value) for value in source_center],
        "candidate_center_core_m": [float(value) for value in candidate_center],
        "radius_residual_m": radius_residual,
        "maximum_radius_residual_m": _FEMORAL_HEAD_RADIUS_MAXIMUM_RESIDUAL_M,
        "center_residual_m": center_residual,
        "maximum_center_residual_m": _FEMORAL_HEAD_CENTER_MAXIMUM_RESIDUAL_M,
        "source_center_to_mechanics_axis_m": source_axis_residual,
        "candidate_center_to_mechanics_axis_m": candidate_axis_residual,
        "maximum_center_to_mechanics_axis_m": (
            _FEMORAL_HEAD_MECHANICS_CENTER_MAXIMUM_RESIDUAL_M
        ),
        "passed": passed,
    }


def _refine_femoral_head_center(
    body_record: dict[str, Any], fit: dict[str, Any], np: Any,
) -> dict[str, Any]:
    """Center the hip articular shell without changing rotation or scale."""
    refined = dict(fit)
    refined["translation"] = fit["translation"].copy()
    initial = _femoral_head_articular_metrics(body_record, refined, np)
    total_delta = np.zeros(3)
    final = initial
    for _ in range(3):
        if final["passed"]:
            break
        delta = (
            np.asarray(final["source_center_core_m"], dtype=float)
            - np.asarray(final["candidate_center_core_m"], dtype=float)
        )
        proposed_total = total_delta + delta
        if float(np.linalg.norm(proposed_total)) > (
            _FEMORAL_HEAD_REFINEMENT_MAXIMUM_TRANSLATION_M + 1.0e-12
        ):
            break
        total_delta = proposed_total
        refined["translation"] = refined["translation"] + delta
        final = _femoral_head_articular_metrics(body_record, refined, np)
    (
        refined["training_metrics"],
        refined["held_out_metrics"],
        refined["training_vertex_count"],
        refined["held_out_vertex_count"],
    ) = _surface_split_metrics(
        body_record["vertices"],
        body_record["source_vertices"],
        refined["rotation"],
        refined["translation"],
        np,
        float(refined.get("uniform_scale", 1.0)),
    )
    refined["femoral_head_articular_gate"] = final
    refined["femoral_head_center_refinement"] = {
        "method": "bounded_translation_to_pinned_mechanics_articular_center",
        "initial_center_residual_m": initial["center_residual_m"],
        "final_center_residual_m": final["center_residual_m"],
        "translation_delta_core_m": [float(value) for value in total_delta],
        "translation_delta_norm_m": float(np.linalg.norm(total_delta)),
        "maximum_translation_m": _FEMORAL_HEAD_REFINEMENT_MAXIMUM_TRANSLATION_M,
        "rotation_changed": False,
        "uniform_scale_changed": False,
    }
    return refined


def _transfer_fit_between_default_frames(
    rotation: Any,
    translation: Any,
    source_target: dict[str, Any],
    destination_target: dict[str, Any],
    np: Any,
) -> tuple[Any, Any]:
    """Express one default-world rigid correction in another body frame."""
    if np.array_equal(rotation, np.eye(3)) and np.array_equal(translation, np.zeros(3)):
        return rotation.copy(), translation.copy()
    source_rotation = _rotation_xyzw(
        source_target["default_inertial_quaternion_world_xyzw"], np
    )
    destination_rotation = _rotation_xyzw(
        destination_target["default_inertial_quaternion_world_xyzw"], np
    )
    source_position = np.asarray(
        source_target["default_com_position_world_m"], dtype=float
    )
    destination_position = np.asarray(
        destination_target["default_com_position_world_m"], dtype=float
    )
    world_rotation = source_rotation @ rotation @ source_rotation.T
    world_translation = (
        source_rotation @ translation + source_position
        - world_rotation @ source_position
    )
    destination_local_rotation = (
        destination_rotation.T @ world_rotation @ destination_rotation
    )
    destination_local_translation = destination_rotation.T @ (
        world_rotation @ destination_position + world_translation
        - destination_position
    )
    return destination_local_rotation, destination_local_translation


def propose_lower_limb_source_registration(
    *, sources: Path, registration_path: Path, tendon_manifest_path: Path, artifact: Path,
) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - source environment only
        raise RuntimeError(
            "lower-limb source registration requires the pinned MyoSim/MuJoCo environment"
        ) from error

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    tendon_manifest = json.loads(tendon_manifest_path.read_text(encoding="utf-8"))
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise RuntimeError("lower-limb source registration requires registration candidate v2")
    runtime_reference, runtime_bodies = human_model._bodyparts_runtime_bindings(registration, artifact)
    rigid_program = human_model._require_myosim_rigid_program(sources, artifact)
    common_frame = human_model._bodyparts_source_common_frame(sources, registration, runtime_bodies)
    if tendon_manifest.get("schema") not in TENDON_SCHEMAS:
        raise RuntimeError("lower-limb source registration requires NHTENDON2 or NHTENDON3")
    lower_registration = registration.get("lower_limb_source_mesh_registration")
    input_has_lower_registration = (
        isinstance(lower_registration, dict) and lower_registration.get("schema") == SCHEMA
    )
    tendon_endpoints = tendon_manifest.get("endpoints")
    if not isinstance(tendon_endpoints, list) or not all(
        isinstance(endpoint, dict) for endpoint in tendon_endpoints
    ):
        raise RuntimeError("lower-limb source registration requires a complete endpoint table")
    source_hashes = {
        registration.get("source", {}).get("myosim", {}).get("source", {}).get(
            "archive_sha256"
        ),
        tendon_manifest.get("source", {}).get("myosim_archive_sha256"),
    }
    if len(source_hashes) != 1 or None in source_hashes:
        raise RuntimeError("lower-limb registration inputs do not share one MyoSim source")

    exported = export_fullbody(sources)
    if exported.get("source") != registration["source"]["myosim"]["source"]:
        raise RuntimeError("lower-limb registration source export differs from its registered runtime source")
    model = build_model("myofullbody")
    meshes_by_body = _compiled_meshes_by_body(model, mujoco, np)
    source_bodies = {int(body["id"]): body for body in exported["bodies"]}
    anchors_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anchors_by_member: dict[str, dict[str, Any]] = {}
    for anchor in registration.get("anchors", []):
        name = anchor.get("target", {}).get("name")
        member = anchor.get("source", {}).get("member_id")
        if not isinstance(name, str) or not isinstance(member, str) or member in anchors_by_member:
            raise RuntimeError("lower-limb source registration contains an invalid anchor")
        anchors_by_name[name].append(anchor)
        anchors_by_member[member] = anchor

    selected_names = {
        f"{family}_{side}"
        for family in _BODY_GATES
        for side in ("r", "l")
    }
    toe_names = {"toes_r", "toes_l"}
    body_records: dict[str, dict[str, Any]] = {}
    for name in sorted(selected_names | toe_names):
        anchors = anchors_by_name.get(name)
        if not anchors:
            raise RuntimeError(f"lower-limb source registration has no anatomy for {name}")
        target = anchors[0]["target"]
        if any(anchor["target"] != target for anchor in anchors):
            raise RuntimeError(f"lower-limb source registration target drifted within {name}")
        source_body_id = int(target["source_body_id"])
        source_body = source_bodies.get(source_body_id)
        source_meshes = meshes_by_body.get(source_body_id, [])
        if source_body is None or source_body["name"] != name or not source_meshes:
            raise RuntimeError(f"lower-limb source registration has no MyoSim mesh for {name}")
        vertices = []
        triangles = []
        member_vertices: dict[str, Any] = {}
        offset = 0
        for anchor in anchors:
            source = anchor["source"]
            _, member, obj = human_model._bodyparts_obj_member(
                sources, source["hierarchy"], source["member_id"]
            )
            raw_vertices, raw_triangles = human_model._bodyparts_obj_triangles(obj, member)
            matrix = np.asarray(
                anchor["registration"]["source_obj_mm_to_core_inertial_body_m"],
                dtype=float,
            )
            current = np.einsum(
                "ki,ji->kj", np.asarray(raw_vertices, dtype=float), matrix[:3, :3]
            ) + matrix[:3, 3]
            member_vertices[source["member_id"]] = current
            vertices.append(current)
            triangles.append(np.asarray(raw_triangles, dtype=int) + offset)
            offset += len(current)
        source_vertices_body = np.concatenate([
            np.asarray(mesh["vertices"], dtype=float) for mesh in source_meshes
        ])
        source_vertices_core = _body_frame_to_core(
            source_vertices_body, source_body, np
        )
        body_records[name] = {
            "anchors": anchors,
            "target": target,
            "source_body": source_body,
            "runtime_body": runtime_bodies[name][1],
            "vertices": np.concatenate(vertices),
            "triangles": np.concatenate(triangles),
            "member_vertices": member_vertices,
            "source_vertices": source_vertices_core,
            "fit_candidates": [],
        }
        if name in selected_names:
            gate = _BODY_GATES[_body_family(name)]
            if input_has_lower_registration:
                # Revalidate admitted geometry before refitting it. This lets
                # a motion repair preserve unrelated, already valid segments.
                training, held_out, training_count, held_out_count = _surface_split_metrics(
                    body_records[name]["vertices"], source_vertices_core,
                    np.eye(3), np.zeros(3), np,
                )
                current_fit = {
                    "rotation": np.eye(3), "translation": np.zeros(3), "uniform_scale": 1.0,
                    "start": "validated_current_registration", "iterations": 0,
                    "training_metrics": training, "held_out_metrics": held_out,
                    "training_vertex_count": training_count, "held_out_vertex_count": held_out_count,
                }
                if name in {"femur_r", "femur_l"}:
                    current_fit = _refine_femoral_head_center(body_records[name], current_fit, np)
                current_fit["source_frame_checks"] = _fit_source_frame_checks(body_records[name], current_fit, common_frame, np)
                if _fit_passes(name, current_fit, np):
                    body_records[name]["fit_candidates"] = [current_fit]
                    continue
            body_records[name]["fit_candidates"] = _fit_candidates(
                body_records[name]["vertices"],
                source_vertices_core,
                np,
                scale_bounds=(gate["minimum_scale"], gate["maximum_scale"]),
            )
            if name in {"femur_r", "femur_l"}:
                body_records[name]["fit_candidates"] = [
                    _refine_femoral_head_center(body_records[name], fit, np)
                    for fit in body_records[name]["fit_candidates"]
                ]
            for fit in body_records[name]["fit_candidates"]:
                fit["source_frame_checks"] = _fit_source_frame_checks(body_records[name], fit, common_frame, np)

    plane_samples = []
    for family in _BODY_GATES:
        right = body_records[f"{family}_r"]
        left = body_records[f"{family}_l"]
        right_world = _core_to_world(right["source_vertices"], right["target"], np)
        left_world = _core_to_world(left["source_vertices"], left["target"], np)
        plane_samples.append(
            0.5 * (float(np.mean(right_world[:, 0])) + float(np.mean(left_world[:, 0])))
        )
    sagittal_plane_x = float(sum(plane_samples) / len(plane_samples))

    def pair_symmetry(right_name: str, left_name: str, right_fit: Any, left_fit: Any) -> dict[str, Any]:
        right, left = body_records[right_name], body_records[left_name]
        right_world = _core_to_world(
            _sample(_transform_points(right["vertices"], right_fit, np), 240, np), right["target"], np,
        )
        left_world = _core_to_world(
            _sample(_transform_points(left["vertices"], left_fit, np), 240, np), left["target"], np,
        )
        mirrored = right_world.copy()
        mirrored[:, 0] = 2.0 * sagittal_plane_x - mirrored[:, 0]
        return _symmetric_metrics(mirrored, left_world, np)

    chosen: dict[str, dict[str, Any]] = {}
    bilateral_receipts = []
    for family in sorted(_BODY_GATES):
        right_name = f"{family}_r"
        left_name = f"{family}_l"
        right = body_records[right_name]
        left = body_records[left_name]
        best = None
        for right_fit in right["fit_candidates"]:
            if not _fit_passes(right_name, right_fit, np):
                continue
            for left_fit in left["fit_candidates"]:
                if not _fit_passes(left_name, left_fit, np):
                    continue
                symmetry = pair_symmetry(right_name, left_name, right_fit, left_fit)
                if symmetry["mean_m"] > _BILATERAL_SYMMETRY_MEAN_MAXIMUM_M:
                    continue
                objective = (
                    float(right_fit["held_out_metrics"]["mean_m"])
                    + float(left_fit["held_out_metrics"]["mean_m"])
                    + 0.35 * float(symmetry["mean_m"])
                    # Atlas shapes are not identical.  Prefer the smallest
                    # proper correction when source-surface fits are close so
                    # a near-symmetric mesh cannot win by flipping a segment.
                    + 0.002 * (
                        _fit_angle(right_fit["rotation"], np)
                        + _fit_angle(left_fit["rotation"], np)
                    )
                    + 0.05 * (
                        float(np.linalg.norm(right_fit["translation"]))
                        + float(np.linalg.norm(left_fit["translation"]))
                    )
                    + 0.003 * (
                        abs(float(right_fit.get("uniform_scale", 1.0)) - 1.0)
                        + abs(float(left_fit.get("uniform_scale", 1.0)) - 1.0)
                    )
                )
                candidate = (
                    objective, str(right_fit["start"]), str(left_fit["start"]),
                    right_fit, left_fit, symmetry,
                )
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None:
            def candidate_diagnostics(name: str, record: dict[str, Any]) -> str:
                summaries = []
                for fit in record["fit_candidates"]:
                    articular = fit.get("femoral_head_articular_gate", {})
                    summaries.append(
                        f"{fit['start']}:p90={fit['held_out_metrics']['p90_m']:.6f},"
                        f"scale={fit.get('uniform_scale', 1.0):.6f},"
                        f"translation={float(np.linalg.norm(fit['translation'])):.6f},"
                        f"head_center={articular.get('center_residual_m', 0.0):.6f},"
                        f"head_axis={articular.get('candidate_center_to_mechanics_axis_m', 0.0):.6f},"
                        f"passes={_fit_passes(name, fit, np)},"
                        f"failed_atlas_frames={[check for check in fit.get('source_frame_checks', []) if not check['passed']]}"
                    )
                return ";".join(summaries)
            raise RuntimeError(
                f"lower-limb source registration could not pair {right_name}/{left_name} within gates; "
                f"right=[{candidate_diagnostics(right_name, right)}]; "
                f"left=[{candidate_diagnostics(left_name, left)}]"
            )
        _, _, _, right_fit, left_fit, symmetry = best
        chosen[right_name] = right_fit
        chosen[left_name] = left_fit
        bilateral_receipts.append({
            "right_body": right_name,
            "left_body": left_name,
            "right_start": right_fit["start"],
            "left_start": left_fit["start"],
            "mirrored_surface_metrics": symmetry,
        })

    # Preserve the exact BodyParts3D foot-to-toe rest arrangement.  MyoSim's
    # collective toe display mesh is not an enthesis atlas, so it cannot
    # justify an independent toe fit or digit split.
    for side in ("r", "l"):
        foot_name = f"calcn_{side}"
        toe_name = f"toes_{side}"
        toe_rotation, toe_translation = _transfer_fit_between_default_frames(
            chosen[foot_name]["rotation"], chosen[foot_name]["translation"],
            body_records[foot_name]["target"], body_records[toe_name]["target"], np,
        )
        chosen[toe_name] = {
            "rotation": toe_rotation,
            "translation": toe_translation,
            "uniform_scale": 1.0,
            "start": "inherited_rigid_foot_default_world_transform",
            "iterations": 0,
            "training_metrics": {},
            "held_out_metrics": {},
            "training_vertex_count": 0,
            "held_out_vertex_count": 0,
            "inherited_from": foot_name,
        }

        # BodyParts3D and MyoSim represent different atlas subjects.  After
        # the rigid-foot surface fit, move the *complete* toe compound by at
        # most 6.5 mm to keep both authored hallux routes within the runtime's
        # 20 mm reference-calibration bound.  This is one rest-registration
        # correction, not a digit split or an additional articulation.
        endpoint_points = np.asarray([
            endpoint["source_local_point_m"]
            for endpoint in tendon_endpoints
            if endpoint.get("body_index") == int(body_records[toe_name]["target"]["core_body_index"])
            and endpoint.get("endpoint") == "insertion"
            and endpoint.get("muscle") in {f"ehl_{side}", f"fhl_{side}"}
        ], dtype=float)
        if endpoint_points.shape != (2, 3):
            raise RuntimeError(
                f"lower-limb source registration requires two hallux route landmarks for {toe_name}"
            )
        hallux_member = human_model._NUMI_HUMAN_TOE_ENTHESIS_MEMBERS[
            (f"ehl_{side}", 1)
        ][0]
        hallux_anchor = anchors_by_member.get(hallux_member)
        if hallux_anchor is None or hallux_anchor["target"]["name"] != toe_name:
            raise RuntimeError(f"lower-limb source registration hallux mapping drifted for {toe_name}")
        _, hallux_record, hallux_obj = human_model._bodyparts_obj_member(
            sources, hallux_anchor["source"]["hierarchy"], hallux_member
        )
        _, hallux_triangles = human_model._bodyparts_obj_triangles(
            hallux_obj, hallux_record
        )
        hallux_vertices = _transform_points(
            body_records[toe_name]["member_vertices"][hallux_member],
            chosen[toe_name], np,
        )

        def enthesis_objective(delta: Any) -> tuple[tuple[float, float, float], Any]:
            distances = _endpoint_surface_distances(
                endpoint_points, hallux_vertices, np.asarray(hallux_triangles, dtype=int),
                np.eye(3), delta, np,
            )
            return (
                (float(np.max(distances)), float(np.sum(distances * distances)),
                 float(np.sum(distances))),
                distances,
            )

        _, initial_distances = enthesis_objective(np.zeros(3))
        delta = np.zeros(3)
        if not input_has_lower_registration:
            for step in (0.002, 0.001, 0.0005, 0.00025, 0.0001, 0.00005, 0.000025):
                while True:
                    best_objective, _ = enthesis_objective(delta)
                    best_delta = delta
                    # Restrict the refinement to the shared distal/proximal toe
                    # axis.  Lateral/dorsal drift can reduce point distance while
                    # tearing the hallux or lesser-toe joint surfaces apart.
                    for axis in (2,):
                        for sign in (-1.0, 1.0):
                            candidate = delta.copy()
                            candidate[axis] += sign * step
                            if float(np.linalg.norm(candidate)) > (
                                _TOE_COMPOUND_ENTHESIS_TRANSLATION_MAXIMUM_M + 1.0e-12
                            ):
                                continue
                            candidate_objective, _ = enthesis_objective(candidate)
                            if candidate_objective < best_objective:
                                best_objective = candidate_objective
                                best_delta = candidate
                    if bool(np.array_equal(best_delta, delta)):
                        break
                    delta = best_delta
        _, final_distances = enthesis_objective(delta)
        if float(np.max(final_distances)) > _TOE_COMPOUND_ENTHESIS_DISTANCE_MAXIMUM_M:
            raise RuntimeError(
                f"lower-limb source registration cannot preserve hallux route calibration for {toe_name}"
            )
        chosen[toe_name]["translation"] = chosen[toe_name]["translation"] + delta
        chosen[toe_name]["toe_compound_enthesis_refinement"] = {
            "method": (
                "preserved_prior_complete_toe_compound_translation"
                if input_has_lower_registration else
                "bounded_complete_toe_compound_translation"
            ),
            "initial_route_surface_distances_m": [float(value) for value in initial_distances],
            "final_route_surface_distances_m": [float(value) for value in final_distances],
            "translation_delta_core_m": [float(value) for value in delta],
            "translation_delta_norm_m": float(np.linalg.norm(delta)),
            "maximum_translation_m": _TOE_COMPOUND_ENTHESIS_TRANSLATION_MAXIMUM_M,
            "maximum_route_surface_distance_m": _TOE_COMPOUND_ENTHESIS_DISTANCE_MAXIMUM_M,
            "new_joint_count": 0,
        }

    def world_vertices(member_id: str) -> Any:
        anchor = anchors_by_member[member_id]
        name = anchor["target"]["name"]
        record = body_records.get(name)
        if record is None or name not in chosen:
            raise RuntimeError(f"lower-limb continuity references unregistered {name}")
        points = _transform_points(record["member_vertices"][member_id], chosen[name], np)
        return _core_to_world(points, record["target"], np)

    source_default_data = mujoco.MjData(model)
    source_default_data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, source_default_data)
    source_default_frames = {
        name: (
            source_default_data.ximat[record["source_body"]["id"]].reshape(3, 3).copy(),
            source_default_data.xipos[record["source_body"]["id"]].copy(),
        )
        for name, record in body_records.items()
    }

    def source_world_vertices(member_id: str, frames: Any) -> Any:
        name = anchors_by_member[member_id]["target"]["name"]
        rotation, position = frames[name]
        return body_records[name]["source_vertices"] @ rotation.T + position

    transitions = [
        (
            name,
            first,
            second,
            human_model._NUMI_HUMAN_KNEE_CONTINUITY_MAXIMUM_GAP_M,
        )
        for name, first, second in human_model._NUMI_HUMAN_KNEE_CONTINUITY_TRANSITIONS
    ] + [
        (
            name,
            first,
            second,
            human_model._NUMI_HUMAN_FOOT_CONTINUITY_MAXIMUM_GAP_M,
        )
        for name, first, second in human_model._NUMI_HUMAN_FOOT_CONTINUITY_TRANSITIONS
    ]

    def transition_metrics(
        transition: tuple[str, str, str, float],
        moved_names: set[str] | None = None,
        world_delta: Any | None = None,
    ) -> dict[str, Any]:
        name, first_member, second_member, gate = transition
        first_vertices = world_vertices(first_member)
        second_vertices = world_vertices(second_member)
        if moved_names and world_delta is not None:
            if anchors_by_member[first_member]["target"]["name"] in moved_names:
                first_vertices = first_vertices + world_delta
            if anchors_by_member[second_member]["target"]["name"] in moved_names:
                second_vertices = second_vertices + world_delta
        first_owner = anchors_by_member[first_member]["target"]["name"]
        second_owner = anchors_by_member[second_member]["target"]["name"]
        allowance = (
            RIGID_TOE_COMPOUND_REFERENCE_ALLOWANCE_M if "metatarsal_to" in name
            else MECHANICS_REFERENCE_INTERFACE_ALLOWANCE_M
        )
        source_vertices = (None, None)
        if first_owner != second_owner:
            source_vertices = (
                source_world_vertices(first_member, source_default_frames),
                source_world_vertices(second_member, source_default_frames),
            )
        return _source_relative_transition_metrics(
            name=name, first_member=first_member, second_member=second_member,
            rest_gate=gate, first_vertices=first_vertices,
            second_vertices=second_vertices, reference_allowance=allowance, np=np,
            source_first_vertices=source_vertices[0],
            source_second_vertices=source_vertices[1],
        )

    # Carry the exact same registered Core-local surfaces through the audit's
    # source poses. A neutral fit can leave the patella detached in flexion.
    from .lower_limb_pose_audit import (
        POSE_SUITE, MECHANICS_REFERENCE_INTERFACE_ALLOWANCE_M,
        RIGID_TOE_COMPOUND_REFERENCE_ALLOWANCE_M, _posed_continuity_gates,
        _bilateral_interface_parity,
    )
    from .upper_limb_pose_audit import _pose_qpos
    data = mujoco.MjData(model)
    pose_frames = {}
    for pose_name, overrides in POSE_SUITE:
        qpos, _, _ = _pose_qpos(model, overrides, mujoco, np)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        pose_frames[pose_name] = {
            name: (data.ximat[record["source_body"]["id"]].reshape(3, 3).copy(),
                   data.xipos[record["source_body"]["id"]].copy())
            for name, record in body_records.items()
        }

    def refine_group_translation(label: str, moved_names: set[str]) -> dict[str, Any] | None:
        relevant = [
            transition
            for transition in transitions
            if (
                anchors_by_member[transition[1]]["target"]["name"] in moved_names
            ) != (
                anchors_by_member[transition[2]]["target"]["name"] in moved_names
            )
        ]
        if not relevant:
            raise RuntimeError(
                f"lower-limb interface refinement {label} has no boundary transitions"
            )

        posed_relevant = [
            transition for transition in transitions
            if any(anchors_by_member[member]["target"]["name"] in moved_names
                   for member in transition[1:3])
        ]
        posed_cases = []
        for pose_name, frames in pose_frames.items():
            for transition in posed_relevant:
                name, first, second, rest_gate = transition
                points, delta_rotations, source_points = [], [], []
                for member in (first, second):
                    body_name = anchors_by_member[member]["target"]["name"]
                    record = body_records[body_name]
                    rotation, position = frames[body_name]
                    core_points = _transform_points(record["member_vertices"][member], chosen[body_name], np)
                    points.append(core_points @ rotation.T + position)
                    rest_rotation = _rotation_xyzw(record["target"]["default_inertial_quaternion_world_xyzw"], np)
                    delta_rotations.append(rotation @ rest_rotation.T if body_name in moved_names else np.zeros((3, 3)))
                    source_points.append(record["source_vertices"] @ rotation.T + position)
                same_owner = anchors_by_member[first]["target"]["name"] == anchors_by_member[second]["target"]["name"]
                source_reference = None
                if not same_owner:
                    source_gap, _, _ = _minimum_gap(*source_points, np)
                    source_reference = {
                        "minimum_vertex_gap_m": source_gap,
                        "interface_patch": _interface_patch_metrics(*source_points, np),
                    }
                allowance = (RIGID_TOE_COMPOUND_REFERENCE_ALLOWANCE_M if "metatarsal_to" in name
                             else MECHANICS_REFERENCE_INTERFACE_ALLOWANCE_M)
                _, _, gap_gate, patch_gate = _posed_continuity_gates(rest_gate, source_reference, allowance)
                posed_cases.append((pose_name, name, first, second, points, delta_rotations, gap_gate, patch_gate))

        def objective(delta: Any) -> tuple[tuple[float, float, float], list[dict[str, Any]]]:
            measured = [
                transition_metrics(transition, moved_names, delta)
                for transition in relevant
            ]
            for pose_name, name, first, second, points, rotations, gap_gate, patch_gate in posed_cases:
                first_points, second_points = [p + r @ delta for p, r in zip(points, rotations, strict=True)]
                gap, _, _ = _minimum_gap(first_points, second_points, np)
                patch = _interface_patch_metrics(first_points, second_points, np)
                measured.append({
                    "pose": pose_name, "name": name, "source_member_ids": [first, second],
                    "minimum_vertex_gap_m": gap, "maximum_allowed_gap_m": gap_gate,
                    "interface_patch": patch, "maximum_allowed_interface_patch_p90_m": patch_gate,
                    "passed": gap <= gap_gate + 1.0e-12 and patch["bidirectional_p90_m"] <= patch_gate + 1.0e-12,
                })
            normalized = [
                max(
                    item["minimum_vertex_gap_m"] / item["maximum_allowed_gap_m"],
                    item["interface_patch"]["bidirectional_p90_m"]
                    / item["maximum_allowed_interface_patch_p90_m"],
                )
                for item in measured
            ]
            return (
                (max(normalized), sum(normalized), float(np.linalg.norm(delta))),
                measured,
            )

        delta, initial_objective, final_objective = _bounded_interface_translation(lambda d: objective(d)[0], np)
        _, initial_metrics = objective(np.zeros(3))
        if initial_objective[0] <= 1.0 + 1.0e-12:
            return None
        final_objective, final_metrics = objective(delta)
        if final_objective[0] > 1.0 + 1.0e-12:
            return None
        for body_name in moved_names:
            chosen[body_name]["translation"] = (
                chosen[body_name]["translation"]
                + _world_delta_to_core(delta, body_records[body_name]["target"], np)
            )
        return {
            "label": label,
            "body_names": sorted(moved_names),
            "method": "bounded_shared_world_translation_minimizing_robust_interfaces_across_source_poses",
            "pose_count": len(pose_frames),
            "translation_world_m": [float(value) for value in delta],
            "translation_norm_m": float(np.linalg.norm(delta)),
            "maximum_translation_m": _INTERFACE_TRANSLATION_REFINEMENT_MAXIMUM_M,
            "initial_maximum_normalized_interface_error": initial_objective[0],
            "final_maximum_normalized_interface_error": final_objective[0],
            "initial_boundary_metrics": initial_metrics,
            "final_boundary_metrics": final_metrics,
            "new_joint_count": 0,
        }

    interface_translation_refinements = []
    for label, names in (
        ("right_ankle_complete_foot", {"talus_r", "calcn_r", "toes_r"}),
        ("left_ankle_complete_foot", {"talus_l", "calcn_l", "toes_l"}),
        ("right_complete_toe_compound", {"toes_r"}),
        ("left_complete_toe_compound", {"toes_l"}),
        ("right_patella", {"patella_r"}),
        ("left_patella", {"patella_l"}),
    ):
        refinement = refine_group_translation(label, names)
        if refinement is not None:
            interface_translation_refinements.append(refinement)

    # Both sides can pass their source-relative interface gates while the
    # bilateral motion differs. Translate the adjacent knee bone groups together;
    # keep every member rigid and preserve the existing per-owner 1.5 mm bound.
    groups = tuple(names for side in ("r", "l") for names in (
        {f"femur_{side}", f"patella_{side}"},
        {f"tibia_{side}"},
        {f"talus_{side}", f"calcn_{side}", f"toes_{side}"},
    ))
    group_index = {name: index for index, names in enumerate(groups) for name in names}
    cases = {}
    for pose_name, frames in [("source_default", source_default_frames), *pose_frames.items()]:
        for transition in transitions:
            name, first, second, rest_gate = transition
            owners = [anchors_by_member[member]["target"]["name"] for member in (first, second)]
            if not any(owner in group_index for owner in owners):
                continue
            if frames is None and group_index.get(owners[0], -1) == group_index.get(owners[1], -1):
                continue
            points, rotations, source_points = [], [], []
            for member, owner in zip((first, second), owners, strict=True):
                record = body_records[owner]
                rest_rotation = _rotation_xyzw(record["target"]["default_inertial_quaternion_world_xyzw"], np)
                if pose_name == "source_default":
                    points.append(world_vertices(member))
                    rotations.append(np.eye(3))
                else:
                    rotation, position = frames[owner]
                    core_points = _transform_points(record["member_vertices"][member], chosen[owner], np)
                    points.append(core_points @ rotation.T + position)
                    rotations.append(rotation @ rest_rotation.T)
                    source_points.append(record["source_vertices"] @ rotation.T + position)
            if pose_name == "source_default" and owners[0] != owners[1]:
                source_points = [
                    source_world_vertices(member, source_default_frames)
                    for member in (first, second)
                ]
            reference = None
            if owners[0] != owners[1]:
                gap, _, _ = _minimum_gap(*source_points, np)
                reference = {"minimum_vertex_gap_m": gap,
                             "interface_patch": _interface_patch_metrics(*source_points, np)}
            allowance = (RIGID_TOE_COMPOUND_REFERENCE_ALLOWANCE_M if "metatarsal_to" in name
                         else MECHANICS_REFERENCE_INTERFACE_ALLOWANCE_M)
            _, _, gap_gate, patch_gate = _posed_continuity_gates(rest_gate, reference, allowance)
            cases[f"{pose_name}:{name}"] = (pose_name, name, points, rotations,
                [group_index.get(owner, -1) for owner in owners], gap_gate, patch_gate)

    parity_keys = {
        f"{pose}:bilateral:{name[len('right_'):]}"
        for pose, name, *_ in cases.values() if pose != "source_default" and name.startswith("right_")
    }
    head_keys = {f"source_default:femoral_head:{name}" for name in group_index if name.startswith("femur_")}
    previous_shifts = {name: np.zeros(3) for name in group_index}
    for receipt in interface_translation_refinements:
        for name in receipt["body_names"]:
            if name in previous_shifts:
                previous_shifts[name] += receipt["translation_world_m"]

    def bilateral_objective(delta: Any, selected: set[str] | None = None) -> tuple[Any, Any]:
        # The bound applies to total new translation per body in this invocation,
        # including an earlier ankle/toe refinement, rather than per search stage.
        if any(float(np.linalg.norm(previous_shifts[name] + delta[index])) >
               _INTERFACE_TRANSLATION_REFINEMENT_MAXIMUM_M + 1e-12
               for name, index in group_index.items()):
            return (math.inf, math.inf, float(np.linalg.norm(delta))), {}
        wanted = set(cases) | parity_keys | head_keys if selected is None else selected
        needed = set(wanted) & cases.keys()
        for key in wanted & parity_keys:
            pose, _, suffix = key.split(":", 2)
            needed.update(f"{pose}:{side}_{suffix}" for side in ("right", "left"))
        primitive, metrics = {}, {}
        for key in sorted(needed):
            pose, name, points, rotations, indices, gap_gate, patch_gate = cases[key]
            moved = [point + rotation @ delta[index] if index >= 0 else point
                     for point, rotation, index in zip(points, rotations, indices, strict=True)]
            gap, _, _ = _minimum_gap(*moved, np)
            patch = _interface_patch_metrics(*moved, np)
            primitive[key] = {"pose": pose, "name": name, "minimum_vertex_gap_m": gap,
                "interface_patch": patch, "maximum_allowed_gap_m": gap_gate,
                "maximum_allowed_interface_patch_p90_m": patch_gate,
                "normalized_error": max(gap / gap_gate, patch["bidirectional_p90_m"] / patch_gate)}
            if key in wanted:
                metrics[key] = primitive[key]
        for pose in sorted({key.split(":", 1)[0] for key in wanted & parity_keys}):
            by_name = {item["name"]: item for item in primitive.values() if item["pose"] == pose}
            for parity in _bilateral_interface_parity(by_name):
                key = f"{pose}:bilateral:{parity['transition']}"
                if key in wanted:
                    metrics[key] = {"pose": pose, **parity, "normalized_error": max(
                        parity["absolute_gap_difference_m"], parity["absolute_interface_patch_p90_difference_m"]
                    ) / parity["maximum_allowed_difference_m"]}
        for key in sorted(wanted & head_keys):
            name = key.split(":", 2)[2]
            trial = {**chosen[name], "translation": chosen[name]["translation"] +
                     _world_delta_to_core(delta[group_index[name]], body_records[name]["target"], np)}
            head = _femoral_head_articular_metrics(body_records[name], trial, np)
            metrics[key] = {**head, "normalized_error": max(
                head["center_residual_m"] / head["maximum_center_residual_m"],
                head["radius_residual_m"] / head["maximum_radius_residual_m"],
                head["source_center_to_mechanics_axis_m"] / head["maximum_center_to_mechanics_axis_m"],
                head["candidate_center_to_mechanics_axis_m"] / head["maximum_center_to_mechanics_axis_m"],
            )}
        errors = [item["normalized_error"] for item in metrics.values()]
        return (max(errors, default=0.), sum(errors), float(np.linalg.norm(delta))), metrics

    zero = np.zeros((len(groups), 3))
    initial, initial_metrics = bilateral_objective(zero)
    if initial[0] > 1.0 + 1e-12:
        # Search the failed constraints first. Every proposal is remeasured
        # against all interfaces and all source poses before being applied;
        # add any newly failing constraint and search again within the same bound.
        active = {key for key, item in initial_metrics.items() if item["normalized_error"] > 1.0 + 1e-12}
        attempts = 0
        while True:
            delta, _, _ = _bounded_interface_translation(
                lambda d: bilateral_objective(d, active)[0], np, group_count=len(groups))
            final, final_metrics = bilateral_objective(delta)
            attempts += 1
            if final[0] <= 1.0 + 1e-12:
                break
            added = {key for key, item in final_metrics.items()
                     if item["normalized_error"] > 1.0 + 1e-12} - active
            if not added:
                break
            active.update(added)
        admitted = final[0] <= 1.0 + 1e-12
        if admitted:
            for name, index in group_index.items():
                chosen[name]["translation"] += _world_delta_to_core(delta[index], body_records[name]["target"], np)
        interface_translation_refinements.append({
            "label": "bilateral_complete_knee_chain", "body_names": sorted(group_index) if admitted else [],
            "method": "bounded_paired_world_translation_with_full_source_pose_and_bilateral_interface_remeasurement",
            "passed": admitted, "applied": admitted, "pose_count": len(pose_frames),
            "groups": [{"body_names": sorted(names), "proposed_translation_world_m": delta[index].tolist(),
                        "proposed_translation_norm_m": float(np.linalg.norm(delta[index]))}
                       for index, names in enumerate(groups)],
            "maximum_translation_per_body_m": _INTERFACE_TRANSLATION_REFINEMENT_MAXIMUM_M,
            "search_attempt_count": attempts, "active_constraint_count": len(active),
            "initial_maximum_normalized_interface_error": initial[0],
            "final_maximum_normalized_interface_error": final[0],
            "initial_boundary_metrics": initial_metrics, "final_boundary_metrics": final_metrics,
            "new_joint_count": 0,
        })

    refined_fit_names = {
        name
        for refinement in interface_translation_refinements
        for name in refinement["body_names"]
        if name in selected_names
    }
    for name in refined_fit_names:
        fit = chosen[name]
        (
            fit["training_metrics"],
            fit["held_out_metrics"],
            fit["training_vertex_count"],
            fit["held_out_vertex_count"],
        ) = _surface_split_metrics(
            body_records[name]["vertices"],
            body_records[name]["source_vertices"],
            fit["rotation"],
            fit["translation"],
            np,
            float(fit.get("uniform_scale", 1.0)),
        )
        if name.startswith("femur_"):
            fit["femoral_head_articular_gate"] = _femoral_head_articular_metrics(body_records[name], fit, np)
        if not _fit_passes(name, fit, np):
            raise RuntimeError(
                f"lower-limb interface refinement invalidated the source fit for {name}"
            )

    for receipt in bilateral_receipts:
        right, left = receipt["right_body"], receipt["left_body"]
        symmetry = pair_symmetry(right, left, chosen[right], chosen[left])
        if symmetry["mean_m"] > _BILATERAL_SYMMETRY_MEAN_MAXIMUM_M:
            raise RuntimeError(
                f"lower-limb interface refinement invalidated {right}/{left} symmetry: "
                f"mean_m={symmetry['mean_m']:.12g}, allowed_m={_BILATERAL_SYMMETRY_MEAN_MAXIMUM_M:.12g}"
            )
        receipt["mirrored_surface_metrics"] = symmetry

    for name, fit in chosen.items():
        fit["source_frame_checks"] = _fit_source_frame_checks(body_records[name], fit, common_frame, np)
        if not all(check["passed"] for check in fit["source_frame_checks"]):
            raise RuntimeError(f"lower-limb source registration violates atlas orientation/scale for {name}: {fit['source_frame_checks']}")

    for side in ("r", "l"):
        toe_name = f"toes_{side}"
        endpoint_points = np.asarray([
            endpoint["source_local_point_m"]
            for endpoint in tendon_endpoints
            if endpoint.get("body_index")
            == int(body_records[toe_name]["target"]["core_body_index"])
            and endpoint.get("endpoint") == "insertion"
            and endpoint.get("muscle") in {f"ehl_{side}", f"fhl_{side}"}
        ], dtype=float)
        hallux_member = human_model._NUMI_HUMAN_TOE_ENTHESIS_MEMBERS[
            (f"ehl_{side}", 1)
        ][0]
        hallux_anchor = anchors_by_member[hallux_member]
        _, hallux_record, hallux_obj = human_model._bodyparts_obj_member(
            sources, hallux_anchor["source"]["hierarchy"], hallux_member
        )
        _, hallux_triangles = human_model._bodyparts_obj_triangles(
            hallux_obj, hallux_record
        )
        hallux_vertices = _transform_points(
            body_records[toe_name]["member_vertices"][hallux_member],
            chosen[toe_name], np,
        )
        distances = _endpoint_surface_distances(
            endpoint_points, hallux_vertices,
            np.asarray(hallux_triangles, dtype=int), np.eye(3), np.zeros(3), np,
        )
        if float(np.max(distances)) > _TOE_COMPOUND_ENTHESIS_DISTANCE_MAXIMUM_M:
            raise RuntimeError(
                f"lower-limb interface refinement broke hallux route calibration for {toe_name}"
            )
        chosen[toe_name]["toe_compound_enthesis_refinement"][
            "post_interface_refinement_route_surface_distances_m"
        ] = [float(value) for value in distances]

    continuity = []
    for transition in transitions:
        continuity.append(transition_metrics(transition))
    failed = [record for record in continuity if not record["passed"]]
    if failed:
        raise RuntimeError(
            "lower-limb source registration violates continuity: "
            + ", ".join(
                f"{record['name']}"
                f"(minimum={record['minimum_vertex_gap_m']:.6f},"
                f"patch_p90={record['interface_patch']['bidirectional_p90_m']:.6f},"
                f"allowed_patch_p90={record['maximum_allowed_interface_patch_p90_m']:.6f})"
                for record in failed
            )
        )

    output = json.loads(json.dumps(registration))
    output_anchors_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in output["anchors"]:
        output_anchors_by_name[anchor["target"]["name"]].append(anchor)
    body_receipts = []
    for name in sorted(selected_names | toe_names):
        fit = chosen[name]
        inherited = fit.get("inherited_from")
        receipt = {
            "method": (
                "inherited_complete_rigid_foot_default_world_transform"
                if inherited else
                "fresh_source_mesh_validation_of_current_registration"
                if fit["start"] == "validated_current_registration" else
                "bilaterally_selected_pca_seeded_trimmed_symmetric_bounded_similarity_icp_to_compiled_myosim_segment_mesh"
            ),
            "source_body_id": int(body_records[name]["target"]["source_body_id"]),
            "selected_start": fit["start"],
            "iterations": int(fit["iterations"]),
            "proper_rotation_determinant": float(np.linalg.det(fit["rotation"])),
            "rotation_angle_rad": _fit_angle(fit["rotation"], np),
            "uniform_scale": float(fit.get("uniform_scale", 1.0)),
            "uniform_scale_bounds": (
                [1.0, 1.0] if inherited else [
                    _BODY_GATES[_body_family(name)]["minimum_scale"],
                    _BODY_GATES[_body_family(name)]["maximum_scale"],
                ]
            ),
            "rigid_translation_core_m": [float(value) for value in fit["translation"]],
            "training_metrics": fit["training_metrics"],
            "held_out_metrics": fit["held_out_metrics"],
            "training_vertex_count": int(fit["training_vertex_count"]),
            "held_out_vertex_count": int(fit["held_out_vertex_count"]),
            "independent_articulation_count": 0,
            "source_frame_checks": fit["source_frame_checks"],
        }
        if inherited:
            receipt["inherited_from"] = inherited
            receipt["source_mesh_fit_intentionally_omitted"] = True
            receipt["toe_compound_enthesis_refinement"] = fit[
                "toe_compound_enthesis_refinement"
            ]
        if "femoral_head_articular_gate" in fit:
            receipt["femoral_head_articular_gate"] = fit[
                "femoral_head_articular_gate"
            ]
            receipt["femoral_head_center_refinement"] = fit[
                "femoral_head_center_refinement"
            ]
        for anchor in output_anchors_by_name[name]:
            matrix = np.asarray(
                anchor["registration"]["source_obj_mm_to_core_inertial_body_m"],
                dtype=float,
            )
            uniform_scale = float(fit.get("uniform_scale", 1.0))
            matrix[:3, :3] = uniform_scale * fit["rotation"] @ matrix[:3, :3]
            matrix[:3, 3] = (
                uniform_scale * fit["rotation"] @ matrix[:3, 3]
                + fit["translation"]
            )
            anchor["registration"]["source_obj_mm_to_core_inertial_body_m"] = [
                [float(value) for value in row] for row in matrix
            ]
            centroid_mm = np.asarray(anchor["source"]["vertex_centroid_mm"], dtype=float)
            centroid_core = np.einsum("i,ji->j", centroid_mm, matrix[:3, :3]) + matrix[:3, 3]
            centroid_world = _core_to_world(
                np.asarray([centroid_core]), anchor["target"], np
            )[0]
            anchor["registration"]["default_pose_vertex_centroid_world_m"] = [
                float(value) for value in centroid_world
            ]
            anchor["registration"]["status"] = (
                "provisional_lower_limb_source_mesh_bounded_similarity_registration"
            )
            anchor["registration"]["lower_limb_source_mesh_registration"] = {
                key: value for key, value in receipt.items() if key != "source_frame_checks"
            }
        body_receipts.append({"myosim_body": name, **receipt})

    output["lower_limb_source_mesh_registration"] = {
        "schema": SCHEMA,
        "status": "candidate_passed_bilateral_source_mesh_and_default_pose_continuity_gates",
        "inputs": {
            "registration": {"file": registration_path.name, "sha256": _sha256(registration_path)},
            "tendon_manifest": {"file": tendon_manifest_path.name, "sha256": _sha256(tendon_manifest_path)},
            "myosim_archive_sha256": next(iter(source_hashes)),
            "runtime_reference": runtime_reference,
            "source_common_frame": common_frame,
        },
        "rigid_source_program_checks": rigid_program,
        "sagittal_mirror_plane_world_x_m": sagittal_plane_x,
        "direct_source_mesh_fit_body_count": len(selected_names),
        "inherited_toe_body_count": len(toe_names),
        "body_fits": body_receipts,
        "bilateral_pairs": bilateral_receipts,
        "continuity": continuity,
        "maximum_continuity_gap_m": max(record["minimum_vertex_gap_m"] for record in continuity),
        "maximum_interface_patch_p90_m": max(
            record["interface_patch"]["bidirectional_p90_m"] for record in continuity
        ),
        "interface_translation_refinements": interface_translation_refinements,
        "new_joint_count": 0,
        "endpoint_migration_m": 0.0,
        "promotion_requirement": (
            "Recompile exact paired NHBONES1/NHTENDON3 artifacts, preserve all 832 mechanical laws "
            "and all 18 named migrated foot endpoints, explicitly report every distributed/point "
            "disposition change, run multi-pose knee/ankle/MTP parity, and inspect bilateral "
            "four-angle M4 Pro frames."
        ),
    }
    output["status"] = "provisional_visual_registration_not_admitted_to_collision_or_physics"
    output["evidence_boundary"] = (
        "This candidate applies one bounded proper isotropic source-surface correction per lower-limb mechanics segment; only femur and tibia permit anthropometric scale. "
        "The complete toe compound inherits the rigid-foot correction and retains the existing MTP articulation. "
        "It does not move a MyoSim route site, add a joint, calibrate cartilage/contact, or establish clinical registration."
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--tendon-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = propose_lower_limb_source_registration(
            sources=arguments.sources.resolve(),
            registration_path=arguments.registration.resolve(),
            tendon_manifest_path=arguments.tendon_manifest.resolve(),
            artifact=arguments.artifact.resolve(),
        )
    except (RuntimeError, OSError, ValueError) as error:
        print(f"numilab-human lower-limb source registration: {error}", file=sys.stderr)
        return 2
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
