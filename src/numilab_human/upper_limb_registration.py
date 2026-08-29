"""Propose a constrained MyoSim-mesh/BodyParts3D upper-limb registration.

The source bone meshes are used only as mechanics-aligned correspondence
authority.  BodyParts3D remains the emitted anatomical geometry, terminal
MyoSim sites are never moved, and the result remains a candidate until the
normal NHBONES1/NHTENDON2 compilers admit it.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import model as human_model
from .myosim_bone_proximity import (
    AUDIT_SCHEMA,
    WORKLIST_SCHEMA,
    _compiled_meshes_by_body,
    _point_triangle_distances_squared,
)
from .myosim_export import export_fullbody


SCHEMA = "numi.human.bodyparts3d-myosim-upper-limb-source-mesh-registration.v1"
REGISTRATION_SCHEMA = "numi.human.bodyparts3d-myosim-bone-registration-candidate.v2"
TENDON_SCHEMA = "numi.human.tendon-attachment-envelope-payload.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rotation_xyzw(quaternion: Any, np: Any) -> Any:
    values = np.asarray(quaternion, dtype=float)
    if values.shape != (4,) or not bool(np.all(np.isfinite(values))):
        raise RuntimeError("upper-limb registration encountered an invalid quaternion")
    norm = float(np.linalg.norm(values))
    if not norm > 0.0:
        raise RuntimeError("upper-limb registration encountered a zero quaternion")
    x, y, z, w = (float(value) / norm for value in values)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _sample(points: Any, maximum: int, np: Any) -> Any:
    if len(points) <= maximum:
        return np.asarray(points, dtype=float).copy()
    indices = np.linspace(0, len(points) - 1, maximum, dtype=int)
    return np.asarray(points, dtype=float)[indices].copy()


def _nearest(source: Any, target: Any, np: Any) -> tuple[Any, Any]:
    if not len(source) or not len(target):
        raise RuntimeError("upper-limb registration cannot match an empty surface")
    indices = []
    distances = []
    for address in range(0, len(source), 128):
        block = source[address : address + 128]
        with np.errstate(all="ignore"):
            squared = np.sum((block[:, None, :] - target[None, :, :]) ** 2, axis=2)
        if not bool(np.all(np.isfinite(squared))):
            raise RuntimeError("upper-limb registration nearest-neighbour metric became non-finite")
        nearest = np.argmin(squared, axis=1)
        indices.append(nearest)
        distances.append(squared[np.arange(len(nearest)), nearest])
    return np.concatenate(indices), np.concatenate(distances)


def _symmetric_metrics(first: Any, second: Any, np: Any) -> dict[str, float]:
    _, forward = _nearest(first, second, np)
    _, reverse = _nearest(second, first, np)
    distances = np.sqrt(np.concatenate((forward, reverse)))
    return {
        "mean_m": float(np.mean(distances)),
        "median_m": float(np.median(distances)),
        "p90_m": float(np.quantile(distances, 0.90)),
        "maximum_m": float(np.max(distances)),
    }


def _proper_rigid_fit(source: Any, target: Any, np: Any) -> tuple[Any, Any]:
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    with np.errstate(all="ignore"):
        u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = vt.T @ u.T
    if float(np.linalg.det(rotation)) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    if not bool(np.all(np.isfinite(rotation))) or not bool(np.all(np.isfinite(translation))):
        raise RuntimeError("upper-limb registration rigid fit became non-finite")
    return rotation, translation


def _icp_candidate(
    moving: Any, target: Any, initial_rotation: Any, initial_translation: Any, np: Any,
) -> dict[str, Any]:
    rotation = initial_rotation.copy()
    translation = initial_translation.copy()
    current = np.einsum("ki,ji->kj", moving, rotation) + translation
    iterations = 0
    for iterations in range(1, 41):
        forward_indices, forward_squared = _nearest(current, target, np)
        reverse_indices, reverse_squared = _nearest(target, current, np)
        distances = np.sqrt(np.concatenate((forward_squared, reverse_squared)))
        trim = float(np.quantile(distances, 0.90))
        forward_mask = np.sqrt(forward_squared) <= trim
        reverse_mask = np.sqrt(reverse_squared) <= trim
        fit_source = np.concatenate((current[forward_mask], current[reverse_indices[reverse_mask]]))
        fit_target = np.concatenate((target[forward_indices[forward_mask]], target[reverse_mask]))
        if len(fit_source) < 12:
            raise RuntimeError("upper-limb registration ICP retained fewer than 12 pairs")
        incremental_rotation, incremental_translation = _proper_rigid_fit(
            fit_source, fit_target, np
        )
        current = np.einsum("ki,ji->kj", current, incremental_rotation) + incremental_translation
        rotation = incremental_rotation @ rotation
        translation = incremental_rotation @ translation + incremental_translation
        cosine = max(-1.0, min(1.0, (float(np.trace(incremental_rotation)) - 1.0) * 0.5))
        if float(np.linalg.norm(incremental_translation)) <= 2.0e-7 and math.acos(cosine) <= 2.0e-6:
            break
    return {
        "rotation": rotation,
        "translation": translation,
        "iterations": iterations,
        "training_metrics": _symmetric_metrics(current, target, np),
    }


def _fit_candidates(moving_vertices: Any, target_vertices: Any, np: Any) -> list[dict[str, Any]]:
    moving_sample = _sample(moving_vertices, 400, np)
    target_sample = _sample(target_vertices, 400, np)
    moving_addresses = np.arange(len(moving_sample))
    target_addresses = np.arange(len(target_sample))
    moving_training = moving_sample[moving_addresses % 5 != 0]
    target_training = target_sample[target_addresses % 5 != 0]
    moving_held_out = moving_sample[moving_addresses % 5 == 0]
    target_held_out = target_sample[target_addresses % 5 == 0]
    if min(
        len(moving_training), len(target_training),
        len(moving_held_out), len(target_held_out),
    ) < 12:
        raise RuntimeError("upper-limb registration cannot reserve a held-out surface split")
    moving_mean = np.mean(moving_training, axis=0)
    target_mean = np.mean(target_training, axis=0)
    _, moving_axes = np.linalg.eigh(np.cov((moving_training - moving_mean).T))
    _, target_axes = np.linalg.eigh(np.cov((target_training - target_mean).T))
    moving_axes = moving_axes[:, ::-1]
    target_axes = target_axes[:, ::-1]
    starts: list[tuple[str, Any, Any]] = [
        ("current_registration", np.eye(3), np.zeros(3)),
        ("centroid_translation", np.eye(3), target_mean - moving_mean),
    ]
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        rotation = target_axes @ np.diag(signs) @ moving_axes.T
        if float(np.linalg.det(rotation)) <= 0.0:
            continue
        starts.append((
            "pca_" + "_".join("positive" if value > 0.0 else "negative" for value in signs),
            rotation,
            target_mean - rotation @ moving_mean,
        ))
    results = []
    for start_name, start_rotation, start_translation in starts:
        result = _icp_candidate(
            moving_training, target_training, start_rotation, start_translation, np
        )
        transformed = np.einsum(
            "ki,ji->kj", moving_sample, result["rotation"]
        ) + result["translation"]
        held_transformed = np.einsum(
            "ki,ji->kj", moving_held_out, result["rotation"]
        ) + result["translation"]
        forward = _nearest(held_transformed, target_sample, np)[1]
        reverse = _nearest(target_held_out, transformed, np)[1]
        held_distances = np.sqrt(np.concatenate((forward, reverse)))
        result.update({
            "start": start_name,
            "training_vertex_count": len(moving_training) + len(target_training),
            "held_out_vertex_count": len(moving_held_out) + len(target_held_out),
            "held_out_metrics": {
                "mean_m": float(np.mean(held_distances)),
                "median_m": float(np.median(held_distances)),
                "p90_m": float(np.quantile(held_distances, 0.90)),
                "maximum_m": float(np.max(held_distances)),
            },
        })
        results.append(result)
    results.sort(key=lambda value: (
        float(value["held_out_metrics"]["mean_m"]),
        float(value["training_metrics"]["mean_m"]),
        str(value["start"]),
    ))
    return results[:4]


def _transform_points(points: Any, fit: dict[str, Any], np: Any) -> Any:
    return np.einsum("ki,ji->kj", points, fit["rotation"]) + fit["translation"]


def _body_frame_to_core(points: Any, body: dict[str, Any], np: Any) -> Any:
    position = np.asarray(body["inertial_position_body_m"], dtype=float)
    rotation = _rotation_xyzw(body["inertial_quaternion_body_xyzw"], np)
    return np.einsum("ki,ij->kj", points - position, rotation)


def _core_to_world(points: Any, target: dict[str, Any], np: Any) -> Any:
    rotation = _rotation_xyzw(target["default_inertial_quaternion_world_xyzw"], np)
    return np.einsum("ki,ji->kj", points, rotation) + np.asarray(
        target["default_com_position_world_m"], dtype=float
    )


def _world_delta_to_core(delta: Any, target: dict[str, Any], np: Any) -> Any:
    rotation = _rotation_xyzw(target["default_inertial_quaternion_world_xyzw"], np)
    return np.einsum("i,ij->j", delta, rotation)


def _minimum_gap(first: Any, second: Any, np: Any) -> tuple[float, Any, Any]:
    indices, squared = _nearest(first, second, np)
    ordinal = int(np.argmin(squared))
    return math.sqrt(float(squared[ordinal])), first[ordinal], second[int(indices[ordinal])]


def _upper_names(side: str) -> set[str]:
    return {
        f"clavicle_{side}", f"scapula_{side}", f"humerus_{side}",
        f"radius_{side}", f"ulna_{side}", f"triquetrum_{side}",
    } | {
        str(specification["myosim_body"])
        for specification in human_model._BODYPARTS_MYOSIM_WRIST_HAND_EXTENSIONS
        if str(specification["myosim_body"]).endswith(f"_{side}")
    }


def _hand_names(side: str) -> set[str]:
    return _upper_names(side) - {
        f"clavicle_{side}", f"scapula_{side}", f"humerus_{side}",
        f"radius_{side}", f"ulna_{side}",
    }


def propose_upper_limb_registration(
    *,
    sources: Path,
    registration_path: Path,
    source_audit_path: Path,
    worklist_path: Path,
    tendon_manifest_path: Path,
) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - source environment only
        raise RuntimeError(
            "upper-limb registration requires the pinned MyoSim/MuJoCo environment"
        ) from error

    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    worklist = json.loads(worklist_path.read_text(encoding="utf-8"))
    tendon_manifest = json.loads(tendon_manifest_path.read_text(encoding="utf-8"))
    if registration.get("schema") != REGISTRATION_SCHEMA:
        raise RuntimeError("upper-limb registration requires a BodyParts3D/MyoSim v2 registration")
    if source_audit.get("schema") != AUDIT_SCHEMA or worklist.get("schema") != WORKLIST_SCHEMA:
        raise RuntimeError("upper-limb registration requires source-bone audit/worklist v1 inputs")
    if tendon_manifest.get("schema") != TENDON_SCHEMA:
        raise RuntimeError("upper-limb registration requires an NHTENDON2 envelope v2 manifest")
    source_hashes = {
        source_audit.get("source", {}).get("archive_sha256"),
        worklist.get("source", {}).get("myosim_archive_sha256"),
        tendon_manifest.get("source", {}).get("myosim_archive_sha256"),
        registration.get("source", {}).get("myosim", {}).get("source", {}).get("archive_sha256"),
    }
    if len(source_hashes) != 1 or None in source_hashes:
        raise RuntimeError("upper-limb registration inputs do not share one pinned MyoSim archive")

    exported = export_fullbody(sources)
    model = build_model("myofullbody")
    meshes_by_body = _compiled_meshes_by_body(model, mujoco, np)
    source_bodies = {int(body["id"]): body for body in exported["bodies"]}
    anchors_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    anchors_by_member: dict[str, dict[str, Any]] = {}
    for anchor in registration.get("anchors", []):
        name = anchor.get("target", {}).get("name")
        member_id = anchor.get("source", {}).get("member_id")
        if not isinstance(name, str) or not isinstance(member_id, str):
            raise RuntimeError("upper-limb registration contains an unnamed anchor")
        anchors_by_name[name].append(anchor)
        anchors_by_member[member_id] = anchor

    all_upper_names = _upper_names("r") | _upper_names("l")
    candidate_names = all_upper_names - {"clavicle_r", "clavicle_l"}
    selected_names = candidate_names - {"scapula_r", "scapula_l"}
    body_records: dict[str, dict[str, Any]] = {}
    for name in sorted(all_upper_names):
        anchors = anchors_by_name.get(name)
        if not anchors:
            raise RuntimeError(f"upper-limb registration has no BodyParts3D anchor for {name}")
        target = anchors[0]["target"]
        source_body_id = int(target["source_body_id"])
        source_body = source_bodies.get(source_body_id)
        source_meshes = meshes_by_body.get(source_body_id, [])
        if source_body is None or not source_meshes:
            raise RuntimeError(f"upper-limb registration has no source mesh for {name}")
        vertices = []
        triangles = []
        offset = 0
        member_vertices: dict[str, Any] = {}
        for anchor in anchors:
            source = anchor["source"]
            _, member, obj = human_model._bodyparts_obj_member(
                sources, source["hierarchy"], source["member_id"]
            )
            raw_vertices, raw_triangles = human_model._bodyparts_obj_triangles(obj, member)
            matrix = np.asarray(
                anchor["registration"]["source_obj_mm_to_core_inertial_body_m"], dtype=float
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
        source_vertices_core = _body_frame_to_core(source_vertices_body, source_body, np)
        record = {
            "anchors": anchors,
            "target": target,
            "source_body": source_body,
            "vertices": np.concatenate(vertices),
            "triangles": np.concatenate(triangles),
            "member_vertices": member_vertices,
            "source_vertices": source_vertices_core,
            "fit_candidates": [],
        }
        if name in candidate_names:
            record["fit_candidates"] = _fit_candidates(
                record["vertices"], source_vertices_core, np
            )
        body_records[name] = record

    audit_endpoints = {
        (int(endpoint["source_actuator_index"]), str(endpoint["endpoint"])): endpoint
        for endpoint in source_audit["endpoints"]
    }
    all_registration_items = [
        item for item in worklist["work_items"]
        if item["disposition"] == "bodyparts_registration_candidate"
        and item["source_body_name"] in candidate_names
    ]
    registration_items = [
        item for item in all_registration_items
        if item["source_body_name"] in selected_names
    ]
    items_by_body: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in all_registration_items:
        items_by_body[str(item["source_body_name"])].append(item)
    tendon_endpoints = {
        (int(endpoint["muscle_index"]), str(endpoint["endpoint"])): endpoint
        for endpoint in tendon_manifest["endpoints"]
    }
    prior_admitted_by_body: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, endpoint in audit_endpoints.items():
        tendon_endpoint = tendon_endpoints.get(key)
        name = str(endpoint["source_body_name"])
        if (
            name in candidate_names
            and isinstance(tendon_endpoint, dict)
            and tendon_endpoint.get("attachment_mode") == "registered_bone_distributed_envelope"
        ):
            prior_admitted_by_body[name].append(endpoint)

    for name in sorted(candidate_names):
        record = body_records[name]
        endpoint_points = []
        for item in items_by_body.get(name, []):
            endpoint = audit_endpoints[
                (int(item["source_actuator_index"]), str(item["endpoint"]))
            ]
            endpoint_points.append(_body_frame_to_core(
                np.asarray([endpoint["source_site_position_body_m"]], dtype=float),
                record["source_body"], np,
            )[0])
        for fit in record["fit_candidates"]:
            transformed = _transform_points(record["vertices"], fit, np)
            triangles = transformed[record["triangles"]]
            distances = [
                math.sqrt(float(np.min(
                    _point_triangle_distances_squared(point, triangles, np)
                )))
                for point in endpoint_points
            ]
            fit["registration_candidate_endpoint_count"] = len(distances)
            fit["maximum_registration_candidate_endpoint_distance_m"] = (
                max(distances) if distances else 0.0
            )
            fit["registration_candidate_endpoint_gate_passed"] = all(
                distance <= 0.012 for distance in distances
            )
            if len(record["anchors"]) != 1:
                raise RuntimeError(
                    f"upper-limb registration expected one exact BodyParts3D member for {name}"
                )
            source_anchor = record["anchors"][0]["source"]
            surface = {
                "body_index": int(record["target"]["core_body_index"]),
                "stable_id": 1,
                "member_id": source_anchor["member_id"],
                "vertices": [[float(value) for value in point] for point in transformed],
                "triangles": [tuple(int(value) for value in triangle) for triangle in record["triangles"]],
            }
            prior_results = []
            for endpoint in prior_admitted_by_body.get(name, []):
                point = _body_frame_to_core(
                    np.asarray([endpoint["source_site_position_body_m"]], dtype=float),
                    record["source_body"], np,
                )[0]
                envelope, reason = human_model._numi_human_tendon_surface_envelope(
                    [float(value) for value in point], surface, 0.012, 0.012, 4.0
                )
                prior_results.append({
                    "source_actuator_index": int(endpoint["source_actuator_index"]),
                    "muscle": str(endpoint["muscle"]),
                    "endpoint": str(endpoint["endpoint"]),
                    "admitted": envelope is not None,
                    "reason": reason,
                })
            fit["prior_admitted_endpoint_count"] = len(prior_results)
            fit["prior_admitted_endpoint_preserved_count"] = sum(
                1 for result in prior_results if result["admitted"]
            )
            fit["prior_admitted_endpoint_gate_passed"] = all(
                result["admitted"] for result in prior_results
            )

    deferred_pairs = []
    for right_name, left_name in (("scapula_r", "scapula_l"),):
        right_fits = body_records[right_name]["fit_candidates"]
        left_fits = body_records[left_name]["fit_candidates"]
        qualifying_right = [
            fit for fit in right_fits
            if fit["registration_candidate_endpoint_gate_passed"]
            and fit["prior_admitted_endpoint_gate_passed"]
        ]
        qualifying_left = [
            fit for fit in left_fits
            if fit["registration_candidate_endpoint_gate_passed"]
            and fit["prior_admitted_endpoint_gate_passed"]
        ]
        if qualifying_right or qualifying_left:
            raise RuntimeError(
                "upper-limb scapular rigid-fit disposition changed; explicit landmark review is required"
            )
        right_items = items_by_body.get(right_name, [])
        left_items = items_by_body.get(left_name, [])
        deferred_pairs.append({
            "right_body": right_name,
            "left_body": left_name,
            "registration_candidate_endpoint_count": len(right_items) + len(left_items),
            "right_best_maximum_endpoint_distance_m": min(
                float(fit["maximum_registration_candidate_endpoint_distance_m"])
                for fit in right_fits
            ),
            "left_best_maximum_endpoint_distance_m": min(
                float(fit["maximum_registration_candidate_endpoint_distance_m"])
                for fit in left_fits
            ),
            "disposition": "deferred_landmark_constrained_scapular_registration",
            "reason": (
                "no proper rigid source-mesh fit preserves every prior envelope and places every "
                "source-bone-adjacent endpoint within 12 mm"
            ),
        })

    plane_samples = []
    for name in sorted(selected_names):
        if not name.endswith("_r"):
            continue
        paired = name[:-2] + "_l"
        if paired not in body_records:
            continue
        right_world = _core_to_world(
            body_records[name]["source_vertices"], body_records[name]["target"], np
        )
        left_world = _core_to_world(
            body_records[paired]["source_vertices"], body_records[paired]["target"], np
        )
        plane_samples.append(0.5 * (float(np.mean(right_world[:, 0])) + float(np.mean(left_world[:, 0]))))
    sagittal_plane_x = float(sum(plane_samples) / len(plane_samples))

    chosen: dict[str, dict[str, Any]] = {}
    bilateral_receipts = []
    for right_name in sorted(name for name in selected_names if name.endswith("_r")):
        left_name = right_name[:-2] + "_l"
        right = body_records[right_name]
        left = body_records[left_name]
        best = None
        for right_fit in right["fit_candidates"]:
            for left_fit in left["fit_candidates"]:
                if not (
                    right_fit["registration_candidate_endpoint_gate_passed"]
                    and left_fit["registration_candidate_endpoint_gate_passed"]
                    and right_fit["prior_admitted_endpoint_gate_passed"]
                    and left_fit["prior_admitted_endpoint_gate_passed"]
                ):
                    continue
                right_sample = _sample(_transform_points(right["vertices"], right_fit, np), 240, np)
                left_sample = _sample(_transform_points(left["vertices"], left_fit, np), 240, np)
                right_world = _core_to_world(right_sample, right["target"], np)
                left_world = _core_to_world(left_sample, left["target"], np)
                mirrored = right_world.copy()
                mirrored[:, 0] = 2.0 * sagittal_plane_x - mirrored[:, 0]
                symmetry = _symmetric_metrics(mirrored, left_world, np)
                objective = (
                    float(right_fit["held_out_metrics"]["mean_m"])
                    + float(left_fit["held_out_metrics"]["mean_m"])
                    + 0.35 * float(symmetry["mean_m"])
                )
                candidate = (objective, right_fit["start"], left_fit["start"], right_fit, left_fit, symmetry)
                if best is None or candidate[:3] < best[:3]:
                    best = candidate
        if best is None:
            right_distances = [
                (fit["start"], fit["maximum_registration_candidate_endpoint_distance_m"])
                for fit in right["fit_candidates"]
            ]
            left_distances = [
                (fit["start"], fit["maximum_registration_candidate_endpoint_distance_m"])
                for fit in left["fit_candidates"]
            ]
            raise RuntimeError(
                f"upper-limb registration could not pair {right_name}/{left_name}; "
                f"right endpoint maxima {right_distances}, left endpoint maxima {left_distances}"
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

    world_deltas = {name: np.zeros(3) for name in all_upper_names}

    def world_vertices(member_id: str) -> Any:
        anchor = anchors_by_member[member_id]
        name = anchor["target"]["name"]
        points = body_records[name]["member_vertices"][member_id]
        transformed = (
            _transform_points(points, chosen[name], np)
            if name in chosen else points
        )
        transformed = transformed + _world_delta_to_core(
            world_deltas[name], body_records[name]["target"], np
        )
        return _core_to_world(transformed, body_records[name]["target"], np)

    # Named source registration can expose real joint-space differences
    # between subjects. Apply only coherent translations: radius alone at the
    # elbow, and the complete hand as one unit at the wrist. No digit receives
    # an independent gap patch.
    for side in ("r", "l"):
        label_prefix = "right" if side == "r" else "left"
        for _ in range(4):
            transition = next(
                record for record in human_model._NUMI_HUMAN_UPPER_LIMB_CONTINUITY_TRANSITIONS
                if record[0] == f"{label_prefix}_humerus_to_radius"
            )
            _, first_member, second_member, gate = transition
            gap, first_point, second_point = _minimum_gap(
                world_vertices(first_member), world_vertices(second_member), np
            )
            if gap <= 0.90 * gate:
                break
            correction = (first_point - second_point) * ((gap - 0.90 * gate) / gap)
            world_deltas[f"radius_{side}"] += correction

        hand_names = _hand_names(side)
        wrist_transitions = [
            record for record in human_model._NUMI_HUMAN_UPPER_LIMB_CONTINUITY_TRANSITIONS
            if record[0] in {
                f"{label_prefix}_radius_to_scaphoid",
                f"{label_prefix}_radius_to_lunate",
                f"{label_prefix}_ulna_to_triquetrum",
            }
        ]
        for _ in range(32):
            measured = []
            for name, first_member, second_member, gate in wrist_transitions:
                gap, first_point, second_point = _minimum_gap(
                    world_vertices(first_member), world_vertices(second_member), np
                )
                measured.append((gap / gate, name, gap, gate, first_point, second_point))
            ratio, _, gap, gate, first_point, second_point = max(measured, key=lambda value: value[:2])
            if ratio <= 1.0:
                break
            correction = (first_point - second_point) * ((gap - 0.92 * gate) / gap)
            for name in hand_names:
                world_deltas[name] += correction

    continuity = []
    for name, first_member, second_member, gate in human_model._NUMI_HUMAN_UPPER_LIMB_CONTINUITY_TRANSITIONS:
        gap, _, _ = _minimum_gap(world_vertices(first_member), world_vertices(second_member), np)
        continuity.append({
            "name": name,
            "source_member_ids": [first_member, second_member],
            "minimum_vertex_gap_m": gap,
            "maximum_allowed_gap_m": gate,
            "passed": gap <= gate + 1.0e-12,
        })
    for name, first_member, second_member in human_model._NUMI_HUMAN_HAND_CONTINUITY_TRANSITIONS:
        gap, _, _ = _minimum_gap(world_vertices(first_member), world_vertices(second_member), np)
        gate = human_model._NUMI_HUMAN_HAND_CONTINUITY_MAXIMUM_GAP_M
        continuity.append({
            "name": name,
            "source_member_ids": [first_member, second_member],
            "minimum_vertex_gap_m": gap,
            "maximum_allowed_gap_m": gate,
            "passed": gap <= gate + 1.0e-12,
        })
    failed_continuity = [record["name"] for record in continuity if not record["passed"]]
    if failed_continuity:
        raise RuntimeError(
            "upper-limb source-mesh registration violates continuity: " + ", ".join(failed_continuity)
        )

    post_regularization_prior_results = []
    for name in sorted(selected_names):
        record = body_records[name]
        transformed = _transform_points(record["vertices"], chosen[name], np)
        transformed += _world_delta_to_core(
            world_deltas[name], record["target"], np
        )
        source_anchor = record["anchors"][0]["source"]
        surface = {
            "body_index": int(record["target"]["core_body_index"]),
            "stable_id": 1,
            "member_id": source_anchor["member_id"],
            "vertices": [[float(value) for value in point] for point in transformed],
            "triangles": [
                tuple(int(value) for value in triangle)
                for triangle in record["triangles"]
            ],
        }
        for endpoint in prior_admitted_by_body.get(name, []):
            point = _body_frame_to_core(
                np.asarray([endpoint["source_site_position_body_m"]], dtype=float),
                record["source_body"], np,
            )[0]
            envelope, reason = human_model._numi_human_tendon_surface_envelope(
                [float(value) for value in point], surface, 0.012, 0.012, 4.0
            )
            post_regularization_prior_results.append({
                "source_actuator_index": int(endpoint["source_actuator_index"]),
                "muscle": str(endpoint["muscle"]),
                "endpoint": str(endpoint["endpoint"]),
                "source_body_name": name,
                "admitted": envelope is not None,
                "reason": reason,
            })
    post_regularization_losses = [
        result for result in post_regularization_prior_results
        if not result["admitted"]
    ]
    if post_regularization_losses:
        lost = ", ".join(
            f"{result['muscle']}:{result['endpoint']}"
            for result in post_regularization_losses[:12]
        )
        raise RuntimeError(
            "upper-limb continuity regularization loses prior admitted envelopes: " + lost
        )

    candidate_metrics = []
    for item in registration_items:
        name = str(item["source_body_name"])
        endpoint = audit_endpoints[(int(item["source_actuator_index"]), str(item["endpoint"]))]
        body = body_records[name]["source_body"]
        point_core = _body_frame_to_core(
            np.asarray([endpoint["source_site_position_body_m"]], dtype=float), body, np
        )[0]
        before_triangles = body_records[name]["vertices"][body_records[name]["triangles"]]
        after_vertices = _transform_points(body_records[name]["vertices"], chosen[name], np)
        after_vertices += _world_delta_to_core(
            world_deltas[name], body_records[name]["target"], np
        )
        after_triangles = after_vertices[body_records[name]["triangles"]]
        before_distance = math.sqrt(float(np.min(
            _point_triangle_distances_squared(point_core, before_triangles, np)
        )))
        after_distance = math.sqrt(float(np.min(
            _point_triangle_distances_squared(point_core, after_triangles, np)
        )))
        candidate_metrics.append({
            "source_actuator_index": int(item["source_actuator_index"]),
            "muscle": item["muscle"],
            "endpoint": item["endpoint"],
            "source_body_name": name,
            "distance_before_m": before_distance,
            "distance_after_m": after_distance,
            "passed_12mm_gate": after_distance <= 0.012,
        })
    failed_endpoints = [
        f"{item['muscle']}:{item['endpoint']}" for item in candidate_metrics
        if not item["passed_12mm_gate"]
    ]
    if failed_endpoints:
        raise RuntimeError(
            "upper-limb source-mesh registration leaves candidate endpoints outside 12 mm: "
            + ", ".join(failed_endpoints[:12])
        )

    output = json.loads(json.dumps(registration))
    output_anchors_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in output["anchors"]:
        output_anchors_by_name[anchor["target"]["name"]].append(anchor)
    body_receipts = []
    for name in sorted(selected_names):
        fit = chosen[name]
        target = body_records[name]["target"]
        local_delta = _world_delta_to_core(world_deltas[name], target, np)
        cosine = max(-1.0, min(1.0, (float(np.trace(fit["rotation"])) - 1.0) * 0.5))
        receipt = {
            "method": "bilaterally_selected_pca_seeded_trimmed_symmetric_rigid_icp_to_compiled_myosim_bone_mesh",
            "source_body_id": int(target["source_body_id"]),
            "selected_start": fit["start"],
            "iterations": int(fit["iterations"]),
            "proper_rotation_determinant": float(np.linalg.det(fit["rotation"])),
            "rotation_angle_rad": math.acos(cosine),
            "rigid_translation_core_m": [float(value) for value in fit["translation"]],
            "continuity_regularization_translation_world_m": [
                float(value) for value in world_deltas[name]
            ],
            "training_metrics": fit["training_metrics"],
            "held_out_metrics": fit["held_out_metrics"],
            "training_vertex_count": int(fit["training_vertex_count"]),
            "held_out_vertex_count": int(fit["held_out_vertex_count"]),
        }
        for anchor in output_anchors_by_name[name]:
            matrix = np.asarray(
                anchor["registration"]["source_obj_mm_to_core_inertial_body_m"], dtype=float
            )
            matrix[:3, :3] = fit["rotation"] @ matrix[:3, :3]
            matrix[:3, 3] = fit["rotation"] @ matrix[:3, 3] + fit["translation"] + local_delta
            anchor["registration"]["source_obj_mm_to_core_inertial_body_m"] = [
                [float(value) for value in row] for row in matrix
            ]
            centroid_mm = np.asarray(anchor["source"]["vertex_centroid_mm"], dtype=float)
            centroid_core = np.einsum(
                "i,ji->j", centroid_mm, matrix[:3, :3]
            ) + matrix[:3, 3]
            centroid_world = _core_to_world(
                np.asarray([centroid_core]), anchor["target"], np
            )[0]
            anchor["registration"]["default_pose_vertex_centroid_world_m"] = [
                float(value) for value in centroid_world
            ]
            anchor["registration"]["status"] = (
                "provisional_upper_limb_source_mesh_rigid_registration"
            )
            anchor["registration"]["upper_limb_source_mesh_registration"] = receipt
        body_receipts.append({"myosim_body": name, **receipt})

    output["upper_limb_source_mesh_registration"] = {
        "schema": SCHEMA,
        "status": "candidate_passed_source_mesh_endpoint_and_default_pose_continuity_gates",
        "inputs": {
            "registration": {"file": registration_path.name, "sha256": _sha256(registration_path)},
            "source_bone_audit": {"file": source_audit_path.name, "sha256": _sha256(source_audit_path)},
            "registration_worklist": {"file": worklist_path.name, "sha256": _sha256(worklist_path)},
            "tendon_manifest": {"file": tendon_manifest_path.name, "sha256": _sha256(tendon_manifest_path)},
            "myosim_archive_sha256": next(iter(source_hashes)),
        },
        "sagittal_mirror_plane_world_x_m": sagittal_plane_x,
        "body_count": len(body_receipts),
        "deferred_body_pairs": deferred_pairs,
        "registration_candidate_endpoint_count": len(candidate_metrics),
        "registration_candidate_endpoint_gate_pass_count": sum(
            1 for item in candidate_metrics if item["passed_12mm_gate"]
        ),
        "prior_admitted_endpoint_count": len(post_regularization_prior_results),
        "prior_admitted_endpoint_preserved_count": sum(
            1 for result in post_regularization_prior_results if result["admitted"]
        ),
        "maximum_candidate_distance_before_m": max(
            item["distance_before_m"] for item in candidate_metrics
        ),
        "maximum_candidate_distance_after_m": max(
            item["distance_after_m"] for item in candidate_metrics
        ),
        "body_fits": body_receipts,
        "bilateral_pairs": bilateral_receipts,
        "continuity": continuity,
        "candidate_endpoints": candidate_metrics,
        "promotion_requirement": (
            "Recompile exact paired NHBONES1/NHTENDON2 artifacts, prove no admitted endpoint loss or endpoint "
            "migration, execute force/replay/rollback gates, and inspect four-angle M4 Pro frames."
        ),
    }
    output["status"] = "provisional_visual_registration_not_admitted_to_collision_or_physics"
    output["evidence_boundary"] = (
        "This candidate rigidly registers exact BodyParts3D upper-limb bones to pinned compiled MyoSim bone "
        "meshes with bilateral and default-pose continuity gates. It does not move MyoSim sites, admit tendon "
        "surface mechanics, create cartilage/TFCC contact, or establish clinical registration."
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--tendon-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = propose_upper_limb_registration(
            sources=arguments.sources.resolve(),
            registration_path=arguments.registration.resolve(),
            source_audit_path=arguments.source_audit.resolve(),
            worklist_path=arguments.worklist.resolve(),
            tendon_manifest_path=arguments.tendon_manifest.resolve(),
        )
    except (RuntimeError, OSError, ValueError) as error:
        print(f"numilab-human upper-limb registration: {error}", file=sys.stderr)
        return 2
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
