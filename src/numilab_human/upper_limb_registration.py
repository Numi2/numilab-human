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
TENDON_SCHEMAS = {
    "numi.human.tendon-attachment-envelope-payload.v2",
    "numi.human.tendon-attachment-envelope-payload.v3",
}
SCAPULAR_ENDPOINT_MAXIMUM_DISTANCE_M = 0.012
SCAPULAR_REFINEMENT_MAXIMUM_TRANSLATION_M = 0.010
SCAPULAR_HELD_OUT_P90_MAXIMUM_M = 0.015
SCAPULAR_REFINEMENT_STEPS_M = (
    0.008, 0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125, 0.0000625,
)
UPPER_LIMB_UNIFORM_SCALE_BOUNDS = {
    "humerus_r": (0.93, 1.07),
    "humerus_l": (0.93, 1.07),
    "radius_r": (0.95, 1.05),
    "radius_l": (0.95, 1.05),
    "ulna_r": (0.95, 1.05),
    "ulna_l": (0.95, 1.05),
}
HUMERAL_HEAD_SELECTION_RADIUS_M = 0.040
HUMERAL_HEAD_CENTER_MAXIMUM_RESIDUAL_M = 0.003
HUMERAL_HEAD_RADIUS_MAXIMUM_RESIDUAL_M = 0.004
HUMERAL_HEAD_MECHANICS_CENTER_MAXIMUM_RESIDUAL_M = 0.005
INTERFACE_PATCH_FRACTION = 0.02
INTERFACE_PATCH_MINIMUM_VERTEX_COUNT = 12
INTERFACE_PATCH_MAXIMUM_VERTEX_COUNT = 128
INTERFACE_PATCH_QUANTILE = 0.90
INTERFACE_PATCH_GATE_MULTIPLIER = 1.25


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


def _proper_similarity_fit(
    source: Any,
    target: Any,
    minimum_scale: float,
    maximum_scale: float,
    np: Any,
) -> tuple[Any, Any, float]:
    """Return one bounded proper isotropic similarity fit.

    Segment-specific uniform scale is the same anthropometric operation used
    when an anatomical geometry atlas is matched to a mechanics subject.  It
    preserves shape and handedness; reflections and anisotropic bone warps are
    deliberately unavailable.
    """
    rotation, _ = _proper_rigid_fit(source, target, np)
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    denominator = float(np.sum(centered_source * centered_source))
    if not denominator > 0.0:
        raise RuntimeError("upper-limb registration similarity fit is degenerate")
    scale = float(np.sum(
        np.einsum("ki,ji->kj", centered_source, rotation) * centered_target
    ) / denominator)
    scale = max(minimum_scale, min(maximum_scale, scale))
    translation = target_mean - scale * rotation @ source_mean
    if (
        not math.isfinite(scale)
        or not bool(np.all(np.isfinite(translation)))
        or not minimum_scale <= scale <= maximum_scale
    ):
        raise RuntimeError("upper-limb registration similarity fit became non-finite")
    return rotation, translation, scale


def _icp_candidate(
    moving: Any,
    target: Any,
    initial_rotation: Any,
    initial_translation: Any,
    scale_bounds: tuple[float, float],
    np: Any,
) -> dict[str, Any]:
    rotation = initial_rotation.copy()
    translation = initial_translation.copy()
    current = np.einsum("ki,ji->kj", moving, rotation) + translation
    scale = 1.0
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
        incremental_rotation, incremental_translation, incremental_scale = (
            _proper_similarity_fit(
                fit_source,
                fit_target,
                scale_bounds[0] / scale,
                scale_bounds[1] / scale,
                np,
            )
        )
        current = incremental_scale * np.einsum(
            "ki,ji->kj", current, incremental_rotation
        ) + incremental_translation
        rotation = incremental_rotation @ rotation
        translation = (
            incremental_scale * incremental_rotation @ translation
            + incremental_translation
        )
        scale *= incremental_scale
        cosine = max(-1.0, min(1.0, (float(np.trace(incremental_rotation)) - 1.0) * 0.5))
        if (
            float(np.linalg.norm(incremental_translation)) <= 2.0e-7
            and math.acos(cosine) <= 2.0e-6
            and abs(incremental_scale - 1.0) <= 2.0e-7
        ):
            break
    return {
        "rotation": rotation,
        "translation": translation,
        "uniform_scale": scale,
        "iterations": iterations,
        "training_metrics": _symmetric_metrics(current, target, np),
    }


def _fit_candidates(
    moving_vertices: Any,
    target_vertices: Any,
    np: Any,
    maximum_results: int = 4,
    scale_bounds: tuple[float, float] = (1.0, 1.0),
) -> list[dict[str, Any]]:
    if maximum_results <= 0:
        raise RuntimeError("registration fit candidate count must be positive")
    if (
        len(scale_bounds) != 2
        or not all(math.isfinite(value) and value > 0.0 for value in scale_bounds)
        or scale_bounds[0] > 1.0
        or scale_bounds[1] < 1.0
        or scale_bounds[0] > scale_bounds[1]
    ):
        raise RuntimeError("registration similarity scale bounds are invalid")
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
            moving_training,
            target_training,
            start_rotation,
            start_translation,
            scale_bounds,
            np,
        )
        transformed = result["uniform_scale"] * np.einsum(
            "ki,ji->kj", moving_sample, result["rotation"]
        ) + result["translation"]
        held_transformed = result["uniform_scale"] * np.einsum(
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
    return results[:maximum_results]


def _surface_split_metrics(
    moving_vertices: Any,
    target_vertices: Any,
    rotation: Any,
    translation: Any,
    np: Any,
    uniform_scale: float = 1.0,
) -> tuple[dict[str, float], dict[str, float], int, int]:
    """Re-evaluate the deterministic training/held-out split after refinement."""
    moving_sample = _sample(moving_vertices, 400, np)
    target_sample = _sample(target_vertices, 400, np)
    moving_addresses = np.arange(len(moving_sample))
    target_addresses = np.arange(len(target_sample))
    moving_training = moving_sample[moving_addresses % 5 != 0]
    target_training = target_sample[target_addresses % 5 != 0]
    moving_held_out = moving_sample[moving_addresses % 5 == 0]
    target_held_out = target_sample[target_addresses % 5 == 0]
    transformed = uniform_scale * np.einsum(
        "ki,ji->kj", moving_sample, rotation
    ) + translation
    training_transformed = (
        uniform_scale * np.einsum(
            "ki,ji->kj", moving_training, rotation
        ) + translation
    )
    held_transformed = (
        uniform_scale * np.einsum(
            "ki,ji->kj", moving_held_out, rotation
        ) + translation
    )
    forward = _nearest(held_transformed, target_sample, np)[1]
    reverse = _nearest(target_held_out, transformed, np)[1]
    held_distances = np.sqrt(np.concatenate((forward, reverse)))
    return (
        _symmetric_metrics(training_transformed, target_training, np),
        {
            "mean_m": float(np.mean(held_distances)),
            "median_m": float(np.median(held_distances)),
            "p90_m": float(np.quantile(held_distances, 0.90)),
            "maximum_m": float(np.max(held_distances)),
        },
        len(moving_training) + len(target_training),
        len(moving_held_out) + len(target_held_out),
    )


def _endpoint_surface_distances(
    endpoint_points: Any,
    moving_vertices: Any,
    triangles: Any,
    rotation: Any,
    translation: Any,
    np: Any,
) -> Any:
    transformed = np.einsum("ki,ji->kj", moving_vertices, rotation) + translation
    surface_triangles = transformed[triangles]
    return np.asarray([
        math.sqrt(float(np.min(
            _point_triangle_distances_squared(point, surface_triangles, np)
        )))
        for point in endpoint_points
    ])


def _refine_scapular_endpoint_translation(
    moving_vertices: Any,
    triangles: Any,
    target_vertices: Any,
    endpoint_points: Any,
    fit: dict[str, Any],
    np: Any,
) -> dict[str, Any]:
    """Bound one proper rigid fit using distributed scapular attachment landmarks.

    MyoSim endpoint sites remain fixed.  Only one coherent translation is added
    to the complete BodyParts3D scapula, and both source-surface fidelity and
    the unchanged endpoint distance gate remain explicit admission criteria.
    """
    if not len(endpoint_points):
        raise RuntimeError("scapular endpoint refinement requires source landmarks")
    refined = dict(fit)
    rotation = np.asarray(fit["rotation"], dtype=float).copy()
    initial_translation = np.asarray(fit["translation"], dtype=float).copy()
    translation = initial_translation.copy()

    def objective(candidate: Any) -> tuple[float, float, float]:
        distances = _endpoint_surface_distances(
            endpoint_points, moving_vertices, triangles,
            rotation, candidate, np,
        )
        return (
            float(np.max(distances)),
            float(np.sum(distances * distances)),
            float(np.sum(distances)),
        )

    initial_objective = objective(translation)
    accepted_steps = 0
    evaluation_count = 1
    for step in SCAPULAR_REFINEMENT_STEPS_M:
        while True:
            best_objective = objective(translation)
            evaluation_count += 1
            best_translation = translation
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    candidate = translation.copy()
                    candidate[axis] += sign * step
                    if (
                        float(np.linalg.norm(candidate - initial_translation))
                        > SCAPULAR_REFINEMENT_MAXIMUM_TRANSLATION_M + 1.0e-12
                    ):
                        continue
                    candidate_objective = objective(candidate)
                    evaluation_count += 1
                    if candidate_objective < best_objective:
                        best_objective = candidate_objective
                        best_translation = candidate
            if bool(np.array_equal(best_translation, translation)):
                break
            translation = best_translation
            accepted_steps += 1

    final_distances = _endpoint_surface_distances(
        endpoint_points, moving_vertices, triangles,
        rotation, translation, np,
    )
    training, held_out, training_count, held_out_count = _surface_split_metrics(
        moving_vertices, target_vertices, rotation, translation, np,
        float(fit.get("uniform_scale", 1.0)),
    )
    delta = translation - initial_translation
    refined.update({
        "translation": translation,
        "training_metrics": training,
        "held_out_metrics": held_out,
        "training_vertex_count": training_count,
        "held_out_vertex_count": held_out_count,
        "endpoint_landmark_refinement": {
            "method": "bounded_coordinate_descent_translation_of_complete_scapula",
            "initial_maximum_endpoint_distance_m": initial_objective[0],
            "final_maximum_endpoint_distance_m": float(np.max(final_distances)),
            "translation_delta_core_m": [float(value) for value in delta],
            "translation_delta_norm_m": float(np.linalg.norm(delta)),
            "maximum_translation_m": SCAPULAR_REFINEMENT_MAXIMUM_TRANSLATION_M,
            "held_out_surface_p90_m": held_out["p90_m"],
            "maximum_held_out_surface_p90_m": SCAPULAR_HELD_OUT_P90_MAXIMUM_M,
            "accepted_coordinate_steps": accepted_steps,
            "objective_evaluation_count": evaluation_count,
            "endpoint_gate_passed": bool(np.all(
                final_distances <= SCAPULAR_ENDPOINT_MAXIMUM_DISTANCE_M
            )),
            "source_surface_gate_passed": (
                held_out["p90_m"] <= SCAPULAR_HELD_OUT_P90_MAXIMUM_M
            ),
        },
    })
    refined["source_fidelity_gate_passed"] = bool(
        refined["endpoint_landmark_refinement"]["source_surface_gate_passed"]
    )
    return refined


def _transform_points(points: Any, fit: dict[str, Any], np: Any) -> Any:
    return float(fit.get("uniform_scale", 1.0)) * np.einsum(
        "ki,ji->kj", points, fit["rotation"]
    ) + fit["translation"]


def _robust_joint_centered_articular_sphere(
    points: Any,
    mechanics_center: Any,
    selection_radius_m: float,
    np: Any,
) -> tuple[Any, float, int]:
    """Fit a proximal ball-joint articular shell near its mechanics axis."""
    selected = points[
        np.linalg.norm(points - mechanics_center, axis=1)
        <= selection_radius_m
    ]
    if len(selected) < 32:
        raise RuntimeError(
            "anatomical registration has insufficient proximal articular surface"
        )
    for _ in range(3):
        design = np.column_stack((2.0 * selected, np.ones(len(selected))))
        target = np.sum(selected * selected, axis=1)
        solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if int(rank) != 4:
            raise RuntimeError("proximal articular sphere fit is degenerate")
        center = solution[:3]
        radius_squared = float(solution[3] + np.dot(center, center))
        if not radius_squared > 0.0:
            raise RuntimeError("proximal articular sphere radius is invalid")
        radius = math.sqrt(radius_squared)
        residuals = np.abs(np.linalg.norm(selected - center, axis=1) - radius)
        selected = selected[
            residuals <= float(np.quantile(residuals, 0.75))
        ]
    if (
        len(selected) < 24
        or not bool(np.all(np.isfinite(center)))
        or not math.isfinite(radius)
    ):
        raise RuntimeError("proximal articular sphere fit is non-finite")
    return center, radius, len(selected)


def _humeral_head_articular_metrics(
    body_record: dict[str, Any], fit: dict[str, Any], np: Any,
) -> dict[str, Any]:
    mechanics_center = _body_frame_to_core(
        np.zeros((1, 3)), body_record["source_body"], np
    )[0]
    source_center, source_radius, source_count = (
        _robust_joint_centered_articular_sphere(
        body_record["source_vertices"], mechanics_center,
        HUMERAL_HEAD_SELECTION_RADIUS_M, np,
    ))
    candidate_center, candidate_radius, candidate_count = (
        _robust_joint_centered_articular_sphere(
        _transform_points(body_record["vertices"], fit, np),
        mechanics_center,
        HUMERAL_HEAD_SELECTION_RADIUS_M,
        np,
    ))
    center_residual = float(np.linalg.norm(candidate_center - source_center))
    radius_residual = abs(candidate_radius - source_radius)
    source_mechanics_residual = float(
        np.linalg.norm(source_center - mechanics_center)
    )
    candidate_mechanics_residual = float(
        np.linalg.norm(candidate_center - mechanics_center)
    )
    passed = bool(
        center_residual <= HUMERAL_HEAD_CENTER_MAXIMUM_RESIDUAL_M
        and radius_residual <= HUMERAL_HEAD_RADIUS_MAXIMUM_RESIDUAL_M
        and source_mechanics_residual
            <= HUMERAL_HEAD_MECHANICS_CENTER_MAXIMUM_RESIDUAL_M
        and candidate_mechanics_residual
            <= HUMERAL_HEAD_MECHANICS_CENTER_MAXIMUM_RESIDUAL_M
    )
    return {
        "method": "robust_proximal_articular_sphere_against_pinned_mobl_derived_mechanics_mesh",
        "source_surface_vertex_count": source_count,
        "candidate_surface_vertex_count": candidate_count,
        "source_radius_m": source_radius,
        "candidate_radius_m": candidate_radius,
        "radius_residual_m": radius_residual,
        "maximum_radius_residual_m": HUMERAL_HEAD_RADIUS_MAXIMUM_RESIDUAL_M,
        "center_residual_m": center_residual,
        "maximum_center_residual_m": HUMERAL_HEAD_CENTER_MAXIMUM_RESIDUAL_M,
        "source_center_to_mechanics_axis_m": source_mechanics_residual,
        "candidate_center_to_mechanics_axis_m": candidate_mechanics_residual,
        "maximum_center_to_mechanics_axis_m": (
            HUMERAL_HEAD_MECHANICS_CENTER_MAXIMUM_RESIDUAL_M
        ),
        "passed": passed,
    }


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


def _core_delta_to_world(delta: Any, target: dict[str, Any], np: Any) -> Any:
    rotation = _rotation_xyzw(target["default_inertial_quaternion_world_xyzw"], np)
    return np.einsum("i,ji->j", delta, rotation)


def _minimum_gap(first: Any, second: Any, np: Any) -> tuple[float, Any, Any]:
    indices, squared = _nearest(first, second, np)
    ordinal = int(np.argmin(squared))
    return math.sqrt(float(squared[ordinal])), first[ordinal], second[int(indices[ordinal])]


def _interface_patch_metrics(first: Any, second: Any, np: Any) -> dict[str, Any]:
    """Measure a small, bidirectional joint-interface patch, not one vertex.

    The lowest two percent of point-to-surface distances on each bone form a
    deterministic proxy for the opposed articular neighborhood.  The 90th
    percentile of each neighborhood prevents an isolated spur or accidental
    crossing from admitting an otherwise displaced bone.  The metric does not
    assert cartilage contact; anatomically real spaces such as the ulnocarpal
    interval retain their transition-specific allowance.
    """
    first_points = np.asarray(first, dtype=float)
    second_points = np.asarray(second, dtype=float)
    if (
        first_points.ndim != 2 or first_points.shape[1:] != (3,)
        or second_points.ndim != 2 or second_points.shape[1:] != (3,)
        or len(first_points) == 0 or len(second_points) == 0
        or not bool(np.all(np.isfinite(first_points)))
        or not bool(np.all(np.isfinite(second_points)))
    ):
        raise RuntimeError("upper-limb interface patch received invalid vertices")

    def one_direction(source: Any, target: Any) -> tuple[float, int]:
        _, squared = _nearest(source, target, np)
        count = max(
            INTERFACE_PATCH_MINIMUM_VERTEX_COUNT,
            min(
                INTERFACE_PATCH_MAXIMUM_VERTEX_COUNT,
                int(round(INTERFACE_PATCH_FRACTION * len(source))),
            ),
        )
        count = min(count, len(source))
        selected = np.partition(squared, count - 1)[:count]
        distance = math.sqrt(float(np.quantile(selected, INTERFACE_PATCH_QUANTILE)))
        return distance, count

    first_to_second, first_count = one_direction(first_points, second_points)
    second_to_first, second_count = one_direction(second_points, first_points)
    return {
        "bidirectional_p90_m": max(first_to_second, second_to_first),
        "first_to_second_p90_m": first_to_second,
        "second_to_first_p90_m": second_to_first,
        "fraction": INTERFACE_PATCH_FRACTION,
        "quantile": INTERFACE_PATCH_QUANTILE,
        "first_vertex_count": first_count,
        "second_vertex_count": second_count,
    }


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
    if tendon_manifest.get("schema") not in TENDON_SCHEMAS:
        raise RuntimeError(
            "upper-limb registration requires an NHTENDON2 or NHTENDON3 manifest"
        )
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
    selected_names = candidate_names
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
                record["vertices"], source_vertices_core, np,
                scale_bounds=UPPER_LIMB_UNIFORM_SCALE_BOUNDS.get(
                    name, (1.0, 1.0)
                ),
            )
            if name in {"humerus_r", "humerus_l"}:
                for fit in record["fit_candidates"]:
                    fit["humeral_head_articular_gate"] = (
                        _humeral_head_articular_metrics(record, fit, np)
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
        record["registration_endpoint_points"] = np.asarray(endpoint_points)
        if name in {"scapula_r", "scapula_l"}:
            record["fit_candidates"] = [
                _refine_scapular_endpoint_translation(
                    record["vertices"], record["triangles"],
                    record["source_vertices"], np.asarray(endpoint_points), fit, np,
                )
                for fit in record["fit_candidates"]
            ]
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

    deferred_pairs: list[dict[str, Any]] = []

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
                    and right_fit.get("source_fidelity_gate_passed", True)
                    and left_fit.get("source_fidelity_gate_passed", True)
                    and right_fit.get(
                        "humeral_head_articular_gate", {"passed": True}
                    )["passed"]
                    and left_fit.get(
                        "humeral_head_articular_gate", {"passed": True}
                    )["passed"]
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

    # The endpoint landmarks and the glenohumeral/clavicular interfaces constrain
    # one complete scapular rigid transform.  Resolve any remaining default-pose
    # interval by translating only the scapula, while retaining the same 10 mm
    # bound relative to the source-surface ICP result.
    for side in ("r", "l"):
        label_prefix = "right" if side == "r" else "left"
        scapula_name = f"scapula_{side}"
        target = body_records[scapula_name]["target"]
        refinement = chosen[scapula_name]["endpoint_landmark_refinement"]
        endpoint_delta_core = np.asarray(
            refinement["translation_delta_core_m"], dtype=float
        )
        shoulder_transitions = [
            record for record in human_model._NUMI_HUMAN_UPPER_LIMB_CONTINUITY_TRANSITIONS
            if record[0] in {
                f"{label_prefix}_clavicle_to_scapula",
                f"{label_prefix}_scapula_to_humerus",
            }
        ]
        for _ in range(32):
            measured = []
            for transition_name, first_member, second_member, gate in shoulder_transitions:
                gap, first_point, second_point = _minimum_gap(
                    world_vertices(first_member), world_vertices(second_member), np
                )
                measured.append((
                    gap / gate, transition_name, gap, gate,
                    first_member, first_point, second_point,
                ))
            ratio, _, gap, gate, first_member, first_point, second_point = max(
                measured, key=lambda value: value[:2]
            )
            if ratio <= 1.0:
                break
            if first_member == body_records[scapula_name]["anchors"][0]["source"]["member_id"]:
                direction = second_point - first_point
            else:
                direction = first_point - second_point
            correction = direction * ((gap - 0.95 * gate) / gap)
            proposed_world_delta = world_deltas[scapula_name] + correction
            proposed_local_delta = _world_delta_to_core(
                proposed_world_delta, target, np
            )
            combined_delta = endpoint_delta_core + proposed_local_delta
            combined_norm = float(np.linalg.norm(combined_delta))
            if combined_norm > SCAPULAR_REFINEMENT_MAXIMUM_TRANSLATION_M:
                combined_delta *= SCAPULAR_REFINEMENT_MAXIMUM_TRANSLATION_M / combined_norm
                proposed_local_delta = combined_delta - endpoint_delta_core
                proposed_world_delta = _core_delta_to_world(
                    proposed_local_delta, target, np
                )
            if bool(np.allclose(
                proposed_world_delta, world_deltas[scapula_name],
                rtol=0.0, atol=1.0e-12,
            )):
                break
            world_deltas[scapula_name] = proposed_world_delta

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
    for name, first_member, second_member in human_model._NUMI_HUMAN_HAND_CONTINUITY_TRANSITIONS:
        first_vertices = world_vertices(first_member)
        second_vertices = world_vertices(second_member)
        gap, _, _ = _minimum_gap(first_vertices, second_vertices, np)
        patch = _interface_patch_metrics(first_vertices, second_vertices, np)
        gate = human_model._NUMI_HUMAN_HAND_CONTINUITY_MAXIMUM_GAP_M
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
        if name in {"scapula_r", "scapula_l"}:
            local_delta = _world_delta_to_core(
                world_deltas[name], record["target"], np
            )
            training, held_out, training_count, held_out_count = _surface_split_metrics(
                record["vertices"], record["source_vertices"],
                chosen[name]["rotation"], chosen[name]["translation"] + local_delta, np,
            )
            refinement = chosen[name]["endpoint_landmark_refinement"]
            combined_delta = np.asarray(
                refinement["translation_delta_core_m"], dtype=float
            ) + local_delta
            final_distances = _endpoint_surface_distances(
                record["registration_endpoint_points"], record["vertices"],
                record["triangles"], chosen[name]["rotation"],
                chosen[name]["translation"] + local_delta, np,
            )
            refinement.update({
                "continuity_translation_world_m": [
                    float(value) for value in world_deltas[name]
                ],
                "final_total_translation_delta_core_m": [
                    float(value) for value in combined_delta
                ],
                "final_total_translation_delta_norm_m": float(
                    np.linalg.norm(combined_delta)
                ),
                "final_maximum_endpoint_distance_m": float(
                    np.max(final_distances)
                ),
                "endpoint_gate_passed": bool(np.all(
                    final_distances <= SCAPULAR_ENDPOINT_MAXIMUM_DISTANCE_M
                )),
                "held_out_surface_p90_m": held_out["p90_m"],
                "source_surface_gate_passed": (
                    held_out["p90_m"] <= SCAPULAR_HELD_OUT_P90_MAXIMUM_M
                ),
            })
            chosen[name].update({
                "training_metrics": training,
                "held_out_metrics": held_out,
                "training_vertex_count": training_count,
                "held_out_vertex_count": held_out_count,
            })
            if (
                refinement["final_total_translation_delta_norm_m"]
                > SCAPULAR_REFINEMENT_MAXIMUM_TRANSLATION_M + 1.0e-12
                or not refinement["endpoint_gate_passed"]
                or not refinement["source_surface_gate_passed"]
            ):
                raise RuntimeError(
                    f"upper-limb scapular source-fidelity gate failed for {name}"
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
            "method": (
                "bilaterally_selected_pca_seeded_trimmed_symmetric_bounded_similarity_icp_to_compiled_myosim_bone_mesh"
            ),
            "source_body_id": int(target["source_body_id"]),
            "selected_start": fit["start"],
            "iterations": int(fit["iterations"]),
            "proper_rotation_determinant": float(np.linalg.det(fit["rotation"])),
            "rotation_angle_rad": math.acos(cosine),
            "uniform_scale": float(fit.get("uniform_scale", 1.0)),
            "uniform_scale_bounds": list(
                UPPER_LIMB_UNIFORM_SCALE_BOUNDS.get(name, (1.0, 1.0))
            ),
            "rigid_translation_core_m": [float(value) for value in fit["translation"]],
            "continuity_regularization_translation_world_m": [
                float(value) for value in world_deltas[name]
            ],
            "training_metrics": fit["training_metrics"],
            "held_out_metrics": fit["held_out_metrics"],
            "training_vertex_count": int(fit["training_vertex_count"]),
            "held_out_vertex_count": int(fit["held_out_vertex_count"]),
        }
        if "endpoint_landmark_refinement" in fit:
            receipt["endpoint_landmark_refinement"] = fit["endpoint_landmark_refinement"]
        if "humeral_head_articular_gate" in fit:
            receipt["humeral_head_articular_gate"] = fit[
                "humeral_head_articular_gate"
            ]
        for anchor in output_anchors_by_name[name]:
            matrix = np.asarray(
                anchor["registration"]["source_obj_mm_to_core_inertial_body_m"], dtype=float
            )
            uniform_scale = float(fit.get("uniform_scale", 1.0))
            matrix[:3, :3] = (
                uniform_scale * fit["rotation"] @ matrix[:3, :3]
            )
            matrix[:3, 3] = (
                uniform_scale * fit["rotation"] @ matrix[:3, 3]
                + fit["translation"] + local_delta
            )
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
                "provisional_upper_limb_source_mesh_bounded_similarity_registration"
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
        "This candidate registers exact BodyParts3D upper-limb bones to pinned compiled MyoSim bone "
        "meshes with bounded per-segment anthropometric uniform scale, bilateral, articular, and default-pose "
        "continuity gates. It does not move MyoSim sites, admit tendon "
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
