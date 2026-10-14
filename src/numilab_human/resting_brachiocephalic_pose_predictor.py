"""Offline brachiocephalic pose projection from captured SVC motion.

This helper reuses actual accepted SVC vertex motion and frozen common-map
barycentric ancestry. It is a source-only predictor, not a native field run or
anatomical admission gate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _xyz_f32(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype("<f4") or array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must be emitted little-endian Float32 XYZ")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite coordinates")
    return np.ascontiguousarray(array)


def _gamma(operations: int, unit_roundoff: float) -> float:
    product = operations * unit_roundoff
    if product >= 1.0:
        raise ValueError("invalid Float32 roundoff model")
    return product / (1.0 - product)


def predict_brachiocephalic_world_positions(
    bridge_undeformed_world_m: Any,
    svc_native_world_m: Any,
    svc_rigid_only_world_m: Any,
    vertex_ancestry: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Predict bridge positions by projecting captured SVC motion.

    Inputs must refer to the same accepted capture pose. svc_native_world_m is
    the exact packed SVC10 surface. svc_rigid_only_world_m is source SVC10
    transformed by the accepted body pose, before cardiac deformation.
    bridge_undeformed_world_m is the source bridge transformed by its accepted
    rigid body pose. Each ancestry row comes from the compiled common map and
    pins the SVC triangle, barycentric weights, and quintic fade.

    Arithmetic is evaluated in explicit Float32 order. A conservative bound
    encloses conversion, captured-displacement subtraction, barycentric
    interpolation, fade, and final position addition roundoff. The returned
    positions remain an offline prediction; native confirmation is separate.
    """
    bridge = _xyz_f32("bridge_undeformed_world_m", bridge_undeformed_world_m)
    svc_native = _xyz_f32("svc_native_world_m", svc_native_world_m)
    svc_rigid = _xyz_f32("svc_rigid_only_world_m", svc_rigid_only_world_m)
    if svc_native.shape != svc_rigid.shape or len(svc_native) == 0:
        raise ValueError("captured and rigid-only SVC surfaces must have identical nonempty shape")
    if len(bridge) == 0 or len(vertex_ancestry) != len(bridge):
        raise ValueError("bridge ancestry count must equal its nonempty vertex count")

    svc_faces = np.empty((len(bridge), 3), dtype=np.int64)
    weights64 = np.empty((len(bridge), 3), dtype=np.float64)
    weights32 = np.empty((len(bridge), 3), dtype=np.float32)
    scales64 = np.empty(len(bridge), dtype=np.float64)
    scales32 = np.empty(len(bridge), dtype=np.float32)
    exact_ids = np.full(len(bridge), -1, dtype=np.int64)
    distal_zero = np.zeros(len(bridge), dtype=np.bool_)

    for i, record in enumerate(vertex_ancestry):
        if not isinstance(record, Mapping):
            raise ValueError(f"vertex ancestry {i} is not a mapping")
        ids = np.asarray(record.get("svc_local_vertices"), dtype=np.int64)
        weights = np.asarray(record.get("barycentric_weights"), dtype=np.float64)
        scale = record.get("coefficient_scale")
        if ids.shape != (3,) or np.any(ids < 0) or np.any(ids >= len(svc_native)):
            raise ValueError(f"vertex ancestry {i} has invalid SVC triangle vertices")
        if weights.shape != (3,) or not np.isfinite(weights).all():
            raise ValueError(f"vertex ancestry {i} has invalid barycentric weights")
        if np.any(weights < 0.0) or np.any(weights > 1.0) or abs(float(weights.sum()) - 1.0) > 2e-12:
            raise ValueError(f"vertex ancestry {i} weights are outside the barycentric simplex")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale):
            raise ValueError(f"vertex ancestry {i} has invalid fade coefficient")
        scale = float(scale)
        if scale < 0.0 or scale > 1.0:
            raise ValueError(f"vertex ancestry {i} fade coefficient is outside [0, 1]")
        if bool(record.get("distal_zero_field_vertex", False)) and scale != 0.0:
            raise ValueError(f"vertex ancestry {i} distal zero-field collar has nonzero fade")

        exact = bool(record.get("exact_source_coordinate_copy", False))
        if exact:
            copied = record.get("copied_svc_local_vertex")
            if isinstance(copied, bool) or not isinstance(copied, int) or not 0 <= copied < len(svc_native):
                raise ValueError(f"vertex ancestry {i} exact-copy index is invalid")
            if scale != 1.0:
                raise ValueError(f"vertex ancestry {i} exact source copy must retain full field")
            exact_ids[i] = copied

        svc_faces[i] = ids
        weights64[i] = weights
        weights32[i] = weights.astype(np.float32)
        scales64[i] = scale
        scales32[i] = np.float32(scale)
        distal_zero[i] = bool(record.get("distal_zero_field_vertex", False))

    # Match the packed inputs' Float32 coordinate domain and explicit operation
    # order. The ancestry/fade is from the already compiled source map.
    captured_delta = np.subtract(svc_native, svc_rigid, dtype=np.float32)
    projected = np.zeros_like(bridge, dtype=np.float32)
    for i in range(len(bridge)):
        if exact_ids[i] >= 0:
            projected[i] = captured_delta[exact_ids[i]]
        else:
            a, b, c = map(int, svc_faces[i])
            w0, w1, w2 = weights32[i]
            term0 = np.multiply(captured_delta[a], w0, dtype=np.float32)
            term1 = np.multiply(captured_delta[b], w1, dtype=np.float32)
            term2 = np.multiply(captured_delta[c], w2, dtype=np.float32)
            partial = np.add(term0, term1, dtype=np.float32)
            projected[i] = np.add(partial, term2, dtype=np.float32)
        projected[i] = np.multiply(projected[i], scales32[i], dtype=np.float32)
    # Multiplying a negative residual by +0 produces -0; native range W and
    # zero-field collars require canonical positive zero displacement.
    projected[distal_zero] = np.float32(0.0)
    predicted = np.add(bridge, projected, dtype=np.float32)
    if not np.isfinite(predicted).all():
        raise ValueError("predicted bridge positions are nonfinite")
    if np.any(projected[distal_zero].view("<u4") != 0):
        raise ValueError("distal zero-field collar acquired a predicted displacement")

    # A forward-error bound against exact real-valued interpolation of the
    # supplied Float32 coordinates and pinned ancestry scalars.
    u = 2.0 ** -24
    g1, g5, g8 = (_gamma(n, u) for n in (1, 5, 8))
    native64 = svc_native.astype(np.float64)
    rigid64 = svc_rigid.astype(np.float64)
    delta64 = native64 - rigid64
    ideal_displacement = np.zeros((len(bridge), 3), dtype=np.float64)
    absolute_sum = np.zeros_like(ideal_displacement)
    delta_subtraction_bound = np.zeros_like(ideal_displacement)
    for i in range(len(bridge)):
        if exact_ids[i] >= 0:
            ids = np.asarray([exact_ids[i]], dtype=np.int64)
            ws = np.asarray([1.0], dtype=np.float64)
        else:
            ids = svc_faces[i]
            ws = weights64[i]
        local_delta = delta64[ids]
        ideal_displacement[i] = scales64[i] * np.einsum("i,ij->j", ws, local_delta)
        absolute_sum[i] = abs(scales64[i]) * np.einsum("i,ij->j", np.abs(ws), np.abs(local_delta))
        subtraction_magnitude = np.abs(native64[ids]) + np.abs(rigid64[ids])
        delta_subtraction_bound[i] = abs(scales64[i]) * np.einsum(
            "i,ij->j", np.abs(ws), g1 * subtraction_magnitude)
    base64 = bridge.astype(np.float64)
    ideal_positions = base64 + ideal_displacement
    arithmetic_bound = (delta_subtraction_bound
                        + g5 * absolute_sum
                        + (2.0 * u) * absolute_sum
                        + g8 * (np.abs(base64) + np.abs(ideal_displacement)))
    observed_error = np.abs(predicted.astype(np.float64) - ideal_positions)
    arithmetic_bound += 8.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.abs(ideal_positions))
    if np.any(observed_error > arithmetic_bound):
        raise ValueError("Float32 predictor error exceeded its declared forward-error bound")

    return {
        "positions_world_m": predicted,
        "captured_svc_displacement_world_m": captured_delta,
        "projected_bridge_displacement_world_m": projected,
        "status": "OFFLINE_CAPTURED_SVC_DISPLACEMENT_PREDICTION_ONLY",
        "native_confirmation": "required",
        "native_field_error_bound": "not_available",
        "float32_bound_scope": "Roundoff of this captured-displacement barycentric interpolation against the ideal interpolation of supplied Float32 capture coordinates and ancestry only; it does not bound native common-map coefficient reconstruction, polynomial evaluation, or GPU kernel arithmetic.",
        "float32_interpolation_roundoff_bound_max_m": float(arithmetic_bound.max(initial=0.0)),
        "float32_interpolation_roundoff_observed_max_m": float(observed_error.max(initial=0.0)),
        "float32_interpolation_roundoff_bound_per_vertex_xyz_m": arithmetic_bound,
    }
