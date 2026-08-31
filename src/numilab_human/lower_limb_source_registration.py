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
    INTERFACE_PATCH_GATE_MULTIPLIER,
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


def _body_family(name: str) -> str:
    family = name.rsplit("_", 1)[0]
    if family not in _BODY_GATES:
        raise RuntimeError(f"lower-limb source registration has no gate for {name}")
    return family


def _fit_angle(rotation: Any, np: Any) -> float:
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) * 0.5))
    return math.acos(cosine)


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
    )


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
    *, sources: Path, registration_path: Path, tendon_manifest_path: Path,
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
    if tendon_manifest.get("schema") not in TENDON_SCHEMAS:
        raise RuntimeError("lower-limb source registration requires NHTENDON2 or NHTENDON3")
    input_has_lower_registration = isinstance(
        registration.get("lower_limb_source_mesh_registration"), dict
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
        if source_body is None or not source_meshes:
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
            "vertices": np.concatenate(vertices),
            "triangles": np.concatenate(triangles),
            "member_vertices": member_vertices,
            "source_vertices": source_vertices_core,
            "fit_candidates": [],
        }
        if name in selected_names:
            gate = _BODY_GATES[_body_family(name)]
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
                right_world = _core_to_world(
                    _sample(_transform_points(right["vertices"], right_fit, np), 240, np),
                    right["target"], np,
                )
                left_world = _core_to_world(
                    _sample(_transform_points(left["vertices"], left_fit, np), 240, np),
                    left["target"], np,
                )
                mirrored = right_world.copy()
                mirrored[:, 0] = 2.0 * sagittal_plane_x - mirrored[:, 0]
                symmetry = _symmetric_metrics(mirrored, left_world, np)
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
                        f"passes={_fit_passes(name, fit, np)}"
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
        gap, _, _ = _minimum_gap(first_vertices, second_vertices, np)
        patch = _interface_patch_metrics(first_vertices, second_vertices, np)
        patch_gate = INTERFACE_PATCH_GATE_MULTIPLIER * gate
        return {
            "name": name,
            "source_member_ids": [first_member, second_member],
            "minimum_vertex_gap_m": gap,
            "maximum_allowed_gap_m": gate,
            "interface_patch": patch,
            "maximum_allowed_interface_patch_p90_m": patch_gate,
            "passed": (
                gap <= gate + 1.0e-12
                and patch["bidirectional_p90_m"] <= patch_gate + 1.0e-12
            ),
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

        def objective(delta: Any) -> tuple[tuple[float, float, float], list[dict[str, Any]]]:
            measured = [
                transition_metrics(transition, moved_names, delta)
                for transition in relevant
            ]
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

        delta = np.zeros(3)
        initial_objective, initial_metrics = objective(delta)
        if initial_objective[0] <= 1.0 + 1.0e-12:
            return None
        for step in _INTERFACE_TRANSLATION_REFINEMENT_STEPS_M:
            while True:
                best_objective, _ = objective(delta)
                best_delta = delta
                for axis in range(3):
                    for sign in (-1.0, 1.0):
                        candidate = delta.copy()
                        candidate[axis] += sign * step
                        if float(np.linalg.norm(candidate)) > (
                            _INTERFACE_TRANSLATION_REFINEMENT_MAXIMUM_M + 1.0e-12
                        ):
                            continue
                        candidate_objective, _ = objective(candidate)
                        if candidate_objective < best_objective:
                            best_objective = candidate_objective
                            best_delta = candidate
                if bool(np.array_equal(best_delta, delta)):
                    break
                delta = best_delta
                if best_objective[0] <= 1.0 + 1.0e-12:
                    break
            reached_objective, _ = objective(delta)
            if reached_objective[0] <= 1.0 + 1.0e-12:
                break
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
            "method": "bounded_shared_world_translation_minimizing_robust_boundary_interface_error",
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
    ):
        refinement = refine_group_translation(label, names)
        if refinement is not None:
            interface_translation_refinements.append(refinement)

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
        if not _fit_passes(name, fit, np):
            raise RuntimeError(
                f"lower-limb interface refinement invalidated the source fit for {name}"
            )

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
        name, first_member, second_member, gate = transition
        first_vertices = world_vertices(first_member)
        second_vertices = world_vertices(second_member)
        gap, _, _ = _minimum_gap(first_vertices, second_vertices, np)
        patch = _interface_patch_metrics(first_vertices, second_vertices, np)
        patch_gate = INTERFACE_PATCH_GATE_MULTIPLIER * gate
        continuity.append({
            "name": name,
            "source_member_ids": [first_member, second_member],
            "minimum_vertex_gap_m": gap,
            "maximum_allowed_gap_m": gate,
            "interface_patch": patch,
            "maximum_allowed_interface_patch_p90_m": patch_gate,
            "passed": (
                gap <= gate + 1.0e-12
                and patch["bidirectional_p90_m"] <= patch_gate + 1.0e-12
            ),
        })
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
            anchor["registration"]["lower_limb_source_mesh_registration"] = receipt
        body_receipts.append({"myosim_body": name, **receipt})

    output["lower_limb_source_mesh_registration"] = {
        "schema": SCHEMA,
        "status": "candidate_passed_bilateral_source_mesh_and_default_pose_continuity_gates",
        "inputs": {
            "registration": {"file": registration_path.name, "sha256": _sha256(registration_path)},
            "tendon_manifest": {"file": tendon_manifest_path.name, "sha256": _sha256(tendon_manifest_path)},
            "myosim_archive_sha256": next(iter(source_hashes)),
        },
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
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--tendon-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = propose_lower_limb_source_registration(
            sources=arguments.sources.resolve(),
            registration_path=arguments.registration.resolve(),
            tendon_manifest_path=arguments.tendon_manifest.resolve(),
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
