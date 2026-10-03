"""Compare patellar motion in a source-registered femoral anatomical frame.

The output is a directional kinematic diagnostic across two different knee
specimens. It does not select a MyoSim path or qualify contact, loaded tracking,
clinical anatomy, or whole-Human behavior.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path
from typing import Any

from .open_knee import EXPECTED_HASHES as OPEN_KNEE_SOURCE_HASHES
from .patellofemoral_reference_path_comparison import (
    FLEXION_TARGETS_DEG,
    MYOSIM_ARCHIVE_SHA256,
    MYOSIM_REVISION,
    _interpolate_reference_pose,
    _myosim_rows,
    _normalize_quaternion,
    _quaternion_inverse,
    _quaternion_multiply,
    _quaternion_slerp,
    _relative_reference_rows,
    _sha256,
)


SCHEMA = "numi.human.patellofemoral-anatomical-frame-path-audit.v1"
OPEN_KNEE_MANIFEST_SCHEMA = "numi.human.open-knee-oks003-payload.v3"
INTERSECTION_SCHEMA = "numi.human.patellofemoral-pose-intersection-audit.v1"
COMPONENT_NAMES = ("flexion", "anterior", "proximal")
_FRAME_ATOL = 2.0e-5
_ANGLE_EPSILON_DEG = 1.0e-8


def _display_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _finite_vector(value: Any, *, size: int, label: str, np: Any) -> Any:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not bool(np.all(np.isfinite(vector))):
        raise RuntimeError(f"{label} must contain {size} finite values")
    return vector


def _registered_femoral_frame(manifest: dict[str, Any], np: Any) -> dict[str, Any]:
    if (manifest.get("schema") != OPEN_KNEE_MANIFEST_SCHEMA
            or manifest.get("status") != "exact_source_payload_registered_to_live_left_knee_candidate"):
        raise RuntimeError("Open Knee frame manifest is not the admitted left source-registration candidate")
    source_files = manifest.get("source", {}).get("files", {})
    actual_hashes = {name: source_files.get(name, {}).get("sha256")
                     for name in OPEN_KNEE_SOURCE_HASHES}
    if actual_hashes != OPEN_KNEE_SOURCE_HASHES:
        raise RuntimeError("Open Knee frame manifest does not bind the pinned oks003 source files")
    registration = manifest.get("registration", {})
    if registration.get("output_side") != "left":
        raise RuntimeError("anatomical frame comparison requires the left oks003 registration")
    rotation = np.asarray(registration.get("proper_rotation_source_to_femur_body"), dtype=float)
    if rotation.shape != (3, 3) or not bool(np.all(np.isfinite(rotation))):
        raise RuntimeError("Open Knee source-to-femur rotation must be a finite 3 by 3 matrix")
    if (abs(float(np.linalg.det(rotation)) - 1.0) > _FRAME_ATOL
            or not bool(np.allclose(rotation.T @ rotation, np.eye(3), atol=_FRAME_ATOL, rtol=0.0))):
        raise RuntimeError("Open Knee source-to-femur rotation is not a proper rotation")
    source_axis_names = ("Xf_axis", "Yf_axis", "Zf_axis")
    target_axis_names = (
        "target_flexion_axis_femur_body",
        "target_anterior_axis_femur_body",
        "target_proximal_axis_femur_body",
    )
    source_axes = np.column_stack([
        _finite_vector(registration.get("source_axes", {}).get(name), size=3,
                       label=f"Open Knee {name}", np=np)
        for name in source_axis_names
    ])
    target_axes = np.column_stack([
        _finite_vector(registration.get(name), size=3,
                       label=name, np=np)
        for name in target_axis_names
    ])
    if (not bool(np.allclose(source_axes.T @ source_axes, np.eye(3), atol=_FRAME_ATOL, rtol=0.0))
            or not bool(np.allclose(target_axes.T @ target_axes, np.eye(3), atol=_FRAME_ATOL, rtol=0.0))):
        raise RuntimeError("source or target femoral axes are not an orthonormal anatomical frame")
    if not bool(np.allclose(rotation @ source_axes, target_axes, atol=_FRAME_ATOL, rtol=0.0)):
        raise RuntimeError("Open Knee source axes do not map to the registered femoral anatomical axes")
    scale = float(registration.get("uniform_scale", math.nan))
    if not math.isfinite(scale) or not 0.90 <= scale <= 1.10:
        raise RuntimeError("Open Knee frame scale is outside the source registration bound")
    origin = _finite_vector(registration.get("source_origin_mm"), size=3,
                            label="Open Knee FMO", np=np)
    return {
        "rotation": rotation,
        "scale": scale,
        "target_axes": target_axes,
        "registration": registration,
        "source_origin_mm": origin,
    }


def _rotation_vector_xyzw(quaternion: list[float] | tuple[float, ...]) -> list[float]:
    q = _normalize_quaternion(quaternion)
    if q[3] < 0.0:
        q = [-value for value in q]
    vector_norm = math.sqrt(sum(value * value for value in q[:3]))
    if vector_norm <= 1.0e-14:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, float(q[3]))
    return [float(value) * angle / vector_norm for value in q[:3]]


def _anatomical_components(vector: Any, axes: Any, np: Any) -> list[float]:
    components = axes.T @ np.asarray(vector, dtype=float)
    return [float(value) for value in components]


def _myosim_pose_at(rows: list[dict[str, Any]], target_deg: float) -> tuple[dict[str, Any], list[float]]:
    ordered = sorted(rows, key=lambda row: float(row["knee_flexion_deg"]))
    for row in ordered:
        if abs(float(row["knee_flexion_deg"]) - target_deg) <= _ANGLE_EPSILON_DEG:
            return row, [float(row["knee_flexion_deg"])]
    for lower, upper in zip(ordered, ordered[1:]):
        lower_angle = float(lower["knee_flexion_deg"])
        upper_angle = float(upper["knee_flexion_deg"])
        if lower_angle < target_deg < upper_angle:
            if upper_angle - lower_angle > 0.500001:
                raise RuntimeError(
                    f"MyoSim target {target_deg:.6f} degrees crosses a source-invalid path gap"
                )
            amount = (target_deg - lower_angle) / (upper_angle - lower_angle)
            return {
                "knee_flexion_deg": target_deg,
                "patella_inertial_com_relative_femur_inertial_com_m": [
                    (1.0 - amount) * float(a) + amount * float(b)
                    for a, b in zip(
                        lower["patella_inertial_com_relative_femur_inertial_com_m"],
                        upper["patella_inertial_com_relative_femur_inertial_com_m"],
                    )
                ],
                "patella_rotation_relative_femur_xyzw": _quaternion_slerp(
                    lower["patella_rotation_relative_femur_xyzw"],
                    upper["patella_rotation_relative_femur_xyzw"], amount,
                ),
            }, [lower_angle, upper_angle]
    raise RuntimeError(f"MyoSim target {target_deg:.6f} degrees has no local valid source path sample")


def _reference_pose_at(rows: list[dict[str, Any]], target_deg: float) -> tuple[list[float], list[float], list[int]]:
    ordered = sorted(rows, key=lambda row: float(row["knee_flexion_deg"]))
    for row in ordered:
        if abs(float(row["knee_flexion_deg"]) - target_deg) <= _ANGLE_EPSILON_DEG:
            return (
                list(row["patella_position_relative_femur_com_mm"]),
                list(row["patella_rotation_relative_femur_xyzw"]),
                [int(row["step"])],
            )
    for lower, upper in zip(ordered, ordered[1:]):
        low = float(lower["knee_flexion_deg"])
        high = float(upper["knee_flexion_deg"])
        if low < target_deg < high:
            position, rotation = _interpolate_reference_pose([lower, upper], target_deg)
            return position, rotation, [int(lower["step"]), int(upper["step"])]
    raise RuntimeError(f"Open Knee target {target_deg:.6f} degrees is outside the observed path")


def _path_delta_metrics(
    *, target_deg: float, reference_rows: list[dict[str, Any]], myosim_rows: list[dict[str, Any]],
    baseline_reference: dict[str, Any], baseline_myosim: dict[str, Any],
    frame: dict[str, Any], np: Any,
) -> dict[str, Any]:
    reference_position_mm, reference_rotation, reference_steps = _reference_pose_at(
        reference_rows, target_deg
    )
    myosim_pose, myosim_bracket = _myosim_pose_at(myosim_rows, target_deg)
    rotation = frame["rotation"]
    scale = frame["scale"]
    axes = frame["target_axes"]

    def registered_reference_position(position_mm: list[float]) -> Any:
        return scale * (rotation @ (np.asarray(position_mm, dtype=float) / 1000.0))

    reference_position = registered_reference_position(reference_position_mm)
    reference_baseline_position = registered_reference_position(
        baseline_reference["patella_position_relative_femur_com_mm"]
    )
    myosim_position = np.asarray(
        myosim_pose["patella_inertial_com_relative_femur_inertial_com_m"], dtype=float
    )
    myosim_baseline_position = np.asarray(
        baseline_myosim["patella_inertial_com_relative_femur_inertial_com_m"], dtype=float
    )
    reference_delta = _anatomical_components(
        reference_position - reference_baseline_position, axes, np
    )
    myosim_delta = _anatomical_components(
        myosim_position - myosim_baseline_position, axes, np
    )

    reference_delta_q = _quaternion_multiply(
        _quaternion_inverse(baseline_reference["patella_rotation_relative_femur_xyzw"]),
        reference_rotation,
    )
    myosim_delta_q = _quaternion_multiply(
        _quaternion_inverse(baseline_myosim["patella_rotation_relative_femur_xyzw"]),
        myosim_pose["patella_rotation_relative_femur_xyzw"],
    )
    reference_rotation_vector = rotation @ np.asarray(
        _rotation_vector_xyzw(reference_delta_q), dtype=float
    )
    myosim_rotation_vector = np.asarray(_rotation_vector_xyzw(myosim_delta_q), dtype=float)
    reference_rotation_components = [
        math.degrees(value) for value in _anatomical_components(reference_rotation_vector, axes, np)
    ]
    myosim_rotation_components = [
        math.degrees(value) for value in _anatomical_components(myosim_rotation_vector, axes, np)
    ]
    reference_absolute_components = _anatomical_components(reference_position, axes, np)
    myosim_absolute_components = _anatomical_components(myosim_position, axes, np)
    return {
        "flexion_deg": float(target_deg),
        "open_knee_observation_steps": reference_steps,
        "myosim_source_sample_bracket_deg": myosim_bracket,
        "open_knee_patella_com_minus_femur_com_flexion_anterior_proximal_m": reference_absolute_components,
        "myosim_patella_inertial_com_minus_femur_inertial_com_flexion_anterior_proximal_m": myosim_absolute_components,
        "open_knee_com_path_change_from_first_observation_flexion_anterior_proximal_m": reference_delta,
        "myosim_com_path_change_from_qpos0_flexion_anterior_proximal_m": myosim_delta,
        "myosim_minus_open_knee_com_path_change_flexion_anterior_proximal_m": [
            myosim_delta[index] - reference_delta[index] for index in range(3)
        ],
        "open_knee_patella_rotation_change_flexion_anterior_proximal_deg": reference_rotation_components,
        "myosim_patella_rotation_change_flexion_anterior_proximal_deg": myosim_rotation_components,
        "myosim_minus_open_knee_rotation_change_flexion_anterior_proximal_deg": [
            myosim_rotation_components[index] - reference_rotation_components[index]
            for index in range(3)
        ],
    }


def audit_patellofemoral_anatomical_frame_path(
    *, sources: Path, observations_path: Path, archive_audit_path: Path,
    intersection_receipt_path: Path, open_knee_manifest_path: Path,
) -> dict[str, Any]:
    try:
        import mujoco
        import numpy as np
    except ImportError as error:  # pragma: no cover - pinned source environment only
        raise RuntimeError("anatomical path comparison requires pinned MyoSim, NumPy, and MuJoCo") from error

    sources = Path(sources).resolve()
    observations_path = Path(observations_path).resolve()
    archive_audit_path = Path(archive_audit_path).resolve()
    intersection_receipt_path = Path(intersection_receipt_path).resolve()
    open_knee_manifest_path = Path(open_knee_manifest_path).resolve()
    root = Path(__file__).resolve().parents[2]

    archive_audit = json.loads(archive_audit_path.read_text(encoding="utf-8"))
    archive_info = archive_audit.get("archived_reference", {})
    if (archive_info.get("observations_complete") is not True
            or archive_info.get("normal_termination_reported") is not True
            or archive_info.get("error_termination_reported") is not False
            or archive_info.get("local_reproduction") is not False):
        raise RuntimeError("Open Knee archived reference audit is incomplete or changed identity")
    intersection = json.loads(intersection_receipt_path.read_text(encoding="utf-8"))
    if (intersection.get("schema") != INTERSECTION_SCHEMA
            or intersection.get("source", {}).get("source_model", {}).get("revision") != MYOSIM_REVISION
            or intersection.get("qualification", {}).get(
                "source_myo_sim_patella_femur_surfaces_range_valid_intersections_absent"
            ) is not False):
        raise RuntimeError("current source-mesh intersection failure is not bound to this MyoSim revision")
    manifest = json.loads(open_knee_manifest_path.read_text(encoding="utf-8"))
    frame = _registered_femoral_frame(manifest, np)

    observations_bytes = observations_path.read_bytes()
    observations = json.loads(gzip.decompress(observations_bytes))
    reference_rows = _relative_reference_rows(
        observations, archive_audit["inventory"]["rigid_bodies"]
    )
    myosim_identity, myosim_rows = _myosim_rows(sources, np, mujoco)
    if (myosim_identity.get("source", {}).get("revision") != MYOSIM_REVISION
            or myosim_identity.get("source", {}).get("archive_sha256") != MYOSIM_ARCHIVE_SHA256):
        raise RuntimeError("MyoSim source identity changed during anatomical path comparison")
    baseline_reference = reference_rows[0]
    baseline_myosim = myosim_rows[0]
    reference_angles = [float(row["knee_flexion_deg"]) for row in reference_rows]
    targets = (*FLEXION_TARGETS_DEG[:2], max(reference_angles))
    matched = [
        _path_delta_metrics(
            target_deg=target, reference_rows=reference_rows, myosim_rows=myosim_rows,
            baseline_reference=baseline_reference, baseline_myosim=baseline_myosim,
            frame=frame, np=np,
        )
        for target in targets
    ]

    registration = frame["registration"]
    axes = frame["target_axes"]
    open_knee_anterior_components = [
        float(_anatomical_components(
            frame["scale"] * (frame["rotation"] @ (
                np.asarray(row["patella_position_relative_femur_com_mm"], dtype=float) / 1000.0
            )), axes, np
        )[1])
        for row in reference_rows
    ]
    myosim_anterior_components = [
        float(_anatomical_components(
            row["patella_inertial_com_relative_femur_inertial_com_m"], axes, np
        )[1])
        for row in myosim_rows
    ]
    sweep = intersection.get("sweep", {})
    source_collision = sweep.get("source_myo_sim_mesh_collision_gate", {})
    registered_collision = sweep.get("collision_gate", {})
    return {
        "schema": SCHEMA,
        "status": "exploratory_only_no_qualification",
        "question": "Do source-equality MyoSim and archived Open Knee patella paths move in the same directions in an anatomically registered femoral frame?",
        "method": {
            "component_order": list(COMPONENT_NAMES),
            "component_units": {"position": "m", "orientation": "degrees"},
            "open_knee": "Archived FEBio rigid-body centers and rotations are made femur-relative, then transformed by the pinned proper uniform source-to-MyoSim femoral registration.",
            "myosim": "The pinned full-body source model is equality-projected at each 0.5 degree flexion sample; only samples whose projected joint coordinates remain within all declared source ranges are retained.",
            "comparison": "Open Knee inertial COM path change is relative to its first accepted state; MyoSim inertial COM path change is relative to source-equality-projected qpos0. Patella orientation uses femur-relative quaternion change from each model's baseline, expressed as rotation-vector components.",
            "registration_semantics": "Open Knee Xf/Yf/Zf anatomical axes map to the MyoSim flexion/anterior/proximal axes. The transform preserves handedness and uses the source registration's uniform condylar-width scale.",
            "baseline_states": {
                "open_knee": {"step": int(baseline_reference["step"]), "flexion_deg": float(baseline_reference["knee_flexion_deg"]), "continuation_parameter_is_elapsed_time": False},
                "myosim": {"flexion_deg": 0.0, "state": "source-equality-projected qpos0"},
            },
        },
        "registration": {
            "manifest_schema": manifest["schema"],
            "manifest_sha256": _sha256(open_knee_manifest_path),
            "manifest_file": _display_path(open_knee_manifest_path, root),
            "source_to_myo_femur_body_rotation": registration["proper_rotation_source_to_femur_body"],
            "proper_rotation_determinant": float(np.linalg.det(frame["rotation"])),
            "uniform_scale": frame["scale"],
            "source_origin_fmo_mm": frame["source_origin_mm"].tolist(),
            "source_axes_xyz": registration["source_axes"],
            "target_axes_in_myo_femur_body": {
                name: [float(value) for value in axes[:, index]]
                for index, name in enumerate(COMPONENT_NAMES)
            },
            "source_femoral_condylar_width_m": float(registration["source_condylar_width_m"]),
            "myosim_femoral_condylar_width_m": float(registration["target_condylar_width_m"]),
            "target_anterior_alignment": float(registration["target_anterior_alignment"]),
        },
        "sources": {
            "myosim": {
                "revision": myosim_identity["source"]["revision"],
                "archive_sha256": myosim_identity["source"]["archive_sha256"],
                "mujoco_version": myosim_identity["mujoco_version"],
                "source_range_valid_sample_count": myosim_identity["source_range_valid_sample_count"],
                "source_range_invalid_sample_count": myosim_identity["source_range_invalid_sample_count"],
                "source_range_invalid_ranges_deg": myosim_identity["source_range_invalid_ranges"],
                "source_overlays": myosim_identity["source_files"],
            },
            "open_knee": {
                "observation_sha256": _sha256(observations_path),
                "archive_audit_sha256": _sha256(archive_audit_path),
                "accepted_increment_count": int(archive_info["accepted_increment_count"]),
                "negative_jacobian_trial_diagnostics": int(archive_info["negative_jacobian_trial_diagnostics"]),
                "original_binary_identity_supplied": archive_info["manifest"].get("original_binary_identity") is not None,
            },
            "intersection_receipt_sha256": _sha256(intersection_receipt_path),
        },
        "predicate_sources": {
            name: _sha256(Path(__file__).with_name(name))
            for name in (
                "patellofemoral_anatomical_frame_audit.py",
                "patellofemoral_reference_path_comparison.py",
                "open_knee.py",
            )
        },
        "results": {
            "baseline_patella_com_relative_to_femur_com_flexion_anterior_proximal_m": {
                "open_knee_first_observation": _anatomical_components(
                    frame["scale"] * (frame["rotation"] @ (
                        np.asarray(baseline_reference[
                            "patella_position_relative_femur_com_mm"
                        ], dtype=float) / 1000.0
                    )), axes, np
                ),
                "myosim_source_equality_qpos0": _anatomical_components(
                    baseline_myosim[
                        "patella_inertial_com_relative_femur_inertial_com_m"
                    ], axes, np
                ),
            },
            "matched_flexion_path_components": matched,
            "sampled_anterior_com_component_relative_to_femur_com_m": {
                "component_semantics": "signed projection on the registered anterior axis; not surface clearance",
                "open_knee": {
                    "sample_count": len(open_knee_anterior_components),
                    "minimum_m": min(open_knee_anterior_components),
                    "maximum_m": max(open_knee_anterior_components),
                    "positive_at_every_accepted_sample": min(open_knee_anterior_components) > 0.0,
                },
                "myosim_source_range_valid": {
                    "sample_count": len(myosim_anterior_components),
                    "minimum_m": min(myosim_anterior_components),
                    "maximum_m": max(myosim_anterior_components),
                    "positive_at_every_retained_sample": min(myosim_anterior_components) > 0.0,
                },
            },
            "source_myo_sim_collision_gate": {
                key: source_collision.get(key) for key in (
                    "range_valid_sample_count", "range_valid_intersecting_pose_side_count",
                    "range_valid_intersection_pair_count", "range_valid_pose_surface_intersections_absent",
                )
            },
            "registered_surface_collision_gate": {
                key: registered_collision.get(key) for key in (
                    "range_valid_sample_count", "range_valid_intersecting_pose_side_count",
                    "range_valid_intersection_pair_count", "range_valid_pose_surface_intersections_absent",
                )
            },
        },
        "limitations": {
            "different_specimens_and_mechanics": True,
            "translation_uses_each_model_inertial_com": True,
            "quasi_static_open_knee_reference_not_local_solver_replay": True,
            "open_knee_active_quadriceps_load_present": False,
            "source_range_valid_path_is_discontinuous": bool(myosim_identity["source_range_invalid_sample_count"]),
            "source_mesh_intersection_gate_passed": False,
            "registered_surface_intersection_gate_passed": False,
            "continuous_angle_interval_qualified": False,
            "loaded_tracking_qualified": False,
            "clinical_anatomy_qualified": False,
            "source_path_correction_selected": False,
            "whole_human_qualified": False,
        },
        "finding": (
            "The source-defined frame shows whether the gross patellar path moves anteriorly, "
            "proximally, or around the flexion axis, but these cross-specimen components are "
            "diagnostic only. The MyoSim path remains source-range-discontinuous and its "
            "pinned femur/patella meshes still intersect at sampled range-valid poses; no path "
            "correction is selected by this comparison."
        ),
        "reproduction_command": [
            sys.executable, "-m", "numilab_human.patellofemoral_anatomical_frame_audit",
            "--sources", str(sources), "--observations", str(observations_path),
            "--archive-audit", str(archive_audit_path),
            "--intersection-receipt", str(intersection_receipt_path),
            "--open-knee-manifest", str(open_knee_manifest_path), "--output", "<new-receipt-path>",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--archive-audit", type=Path, required=True)
    parser.add_argument("--intersection-receipt", type=Path, required=True)
    parser.add_argument("--open-knee-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_patellofemoral_anatomical_frame_path(
            sources=args.sources,
            observations_path=args.observations,
            archive_audit_path=args.archive_audit,
            intersection_receipt_path=args.intersection_receipt,
            open_knee_manifest_path=args.open_knee_manifest,
        )
        output = args.output.resolve()
        if output.exists():
            raise RuntimeError(f"refusing to overwrite existing anatomical path receipt: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"patellofemoral anatomical frame path audit: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"schema": SCHEMA, "status": result["status"], "output": str(output),
                      "matched_flexion_count": len(result["results"]["matched_flexion_path_components"])}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
