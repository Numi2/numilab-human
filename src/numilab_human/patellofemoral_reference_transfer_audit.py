"""Test an independently observed patellar path against pinned MyoSim surfaces.

This is a cross-specimen rigid-geometry transfer diagnostic. It does not modify
MyoSim coordinates, source equalities, tendon owners, or runtime state, and it
cannot qualify an anatomical or loaded tracking path.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from .lower_limb_pose_audit import _sha256
from .myosim_bone_proximity import _compiled_meshes_by_body
from .patellofemoral_anatomical_frame_audit import (
    _registered_femoral_frame,
    _rotation_vector_xyzw,
)
from .patellofemoral_pose_intersection_audit import (
    SCHEMA as INTERSECTION_SCHEMA,
    _exact_surface_pair_audit,
)
from .patellofemoral_reference_path_comparison import (
    MYOSIM_ARCHIVE_SHA256,
    MYOSIM_REVISION,
    _myosim_rows,
    _normalize_quaternion,
    _quaternion_angle_degrees,
    _quaternion_inverse,
    _quaternion_multiply,
    _relative_reference_rows,
)
from .upper_limb_pose_audit import _pose_qpos_with_unit_metrics
from .upper_limb_registration import _rotation_xyzw


SCHEMA = "numi.human.patellofemoral-reference-transfer-audit.v1"
PLAN_SCHEMA = "numi.human.patellofemoral-reference-transfer-plan.v1"
_OPEN_KNEE_MIN_DEG = 0.1294684279282639
_OPEN_KNEE_MAX_DEG = 89.94454550419701
_ANGLE_EPSILON_DEG = 1.0e-8


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _rotation_vector_quaternion(vector: Any, np: Any) -> list[float]:
    values = np.asarray(vector, dtype=float)
    angle = float(np.linalg.norm(values))
    if angle <= 1.0e-14:
        return [0.0, 0.0, 0.0, 1.0]
    half = 0.5 * angle
    scale = math.sin(half) / angle
    return _normalize_quaternion([
        float(values[0] * scale), float(values[1] * scale),
        float(values[2] * scale), math.cos(half),
    ])


def _relative_pose_from_reference(
    reference_row: dict[str, Any], baseline_reference: dict[str, Any],
    baseline_myosim: dict[str, Any], frame: dict[str, Any], np: Any,
) -> tuple[Any, list[float]]:
    position = np.asarray(reference_row["patella_position_relative_femur_com_mm"], dtype=float)
    baseline_position = np.asarray(
        baseline_reference["patella_position_relative_femur_com_mm"], dtype=float
    )
    delta_position = frame["scale"] * (
        frame["rotation"] @ ((position - baseline_position) / 1000.0)
    )
    target_position = (
        np.asarray(
            baseline_myosim[
                "patella_inertial_com_relative_femur_inertial_com_m"
            ], dtype=float,
        ) + delta_position
    )

    delta_rotation = _quaternion_multiply(
        _quaternion_inverse(baseline_reference["patella_rotation_relative_femur_xyzw"]),
        reference_row["patella_rotation_relative_femur_xyzw"],
    )
    target_rotation_vector = frame["rotation"] @ np.asarray(
        _rotation_vector_xyzw(delta_rotation), dtype=float
    )
    target_delta_rotation = _rotation_vector_quaternion(target_rotation_vector, np)
    target_rotation = _normalize_quaternion(_quaternion_multiply(
        baseline_myosim["patella_rotation_relative_femur_xyzw"],
        target_delta_rotation,
    ))
    return target_position, target_rotation


def _reference_pose_at(rows: list[dict[str, Any]], target_deg: float) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: float(row["knee_flexion_deg"]))
    for row in ordered:
        if abs(float(row["knee_flexion_deg"]) - target_deg) <= _ANGLE_EPSILON_DEG:
            return row
    for lower, upper in zip(ordered, ordered[1:]):
        low = float(lower["knee_flexion_deg"])
        high = float(upper["knee_flexion_deg"])
        if low < target_deg < high:
            amount = (target_deg - low) / (high - low)
            position = [
                (1.0 - amount) * float(a) + amount * float(b)
                for a, b in zip(
                    lower["patella_position_relative_femur_com_mm"],
                    upper["patella_position_relative_femur_com_mm"],
                    strict=True,
                )
            ]
            q0 = _normalize_quaternion(lower["patella_rotation_relative_femur_xyzw"])
            q1 = _normalize_quaternion(upper["patella_rotation_relative_femur_xyzw"])
            dot = sum(q0[index] * q1[index] for index in range(4))
            if dot < 0.0:
                q1 = [-value for value in q1]
                dot = -dot
            dot = max(-1.0, min(1.0, dot))
            if dot > 0.9995:
                rotation = _normalize_quaternion([
                    (1.0 - amount) * q0[index] + amount * q1[index]
                    for index in range(4)
                ])
            else:
                theta = math.acos(dot)
                denominator = math.sin(theta)
                rotation = _normalize_quaternion([
                    math.sin((1.0 - amount) * theta) / denominator * q0[index]
                    + math.sin(amount * theta) / denominator * q1[index]
                    for index in range(4)
                ])
            return {
                "knee_flexion_deg": target_deg,
                "patella_position_relative_femur_com_mm": position,
                "patella_rotation_relative_femur_xyzw": rotation,
                "interpolation_steps": [int(lower["step"]), int(upper["step"])],
            }
    raise RuntimeError(f"Open Knee path does not cover {target_deg:.9f} degrees")


def _joined_world_mesh(
    meshes: list[dict[str, Any]], origin: Any, rotation: Any, np: Any,
) -> tuple[Any, Any]:
    vertices = []
    faces = []
    offset = 0
    for mesh in meshes:
        local_vertices = np.asarray(mesh["vertices"], dtype=np.float64)
        local_faces = np.asarray(mesh["faces"], dtype=np.int64)
        vertices.append(local_vertices @ rotation.T + origin)
        if len(local_faces):
            faces.append(local_faces + offset)
        offset += len(local_vertices)
    if not vertices or not faces:
        raise RuntimeError("pinned patellofemoral source mesh is empty")
    return np.concatenate(vertices), np.concatenate(faces)


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("status") != (
        "registered_before_transfer_observation"
    ):
        raise RuntimeError("reference-transfer plan is missing or was not preregistered")
    if plan.get("inputs", {}).get("myosim_revision") != MYOSIM_REVISION:
        raise RuntimeError("reference-transfer plan MyoSim revision changed")
    if plan.get("inputs", {}).get("myosim_archive_sha256") != MYOSIM_ARCHIVE_SHA256:
        raise RuntimeError("reference-transfer plan MyoSim archive identity changed")


def audit_reference_transfer(
    *, sources: Path, observations_path: Path, archive_audit_path: Path,
    intersection_receipt_path: Path, open_knee_manifest_path: Path, plan_path: Path,
) -> dict[str, Any]:
    """Compare current left source geometry with a fixed independent path transfer."""
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - pinned source environment only
        raise RuntimeError("reference transfer requires pinned MyoSim, NumPy, and MuJoCo") from error

    paths = {
        "sources": Path(sources).resolve(),
        "observations": Path(observations_path).resolve(),
        "archive_audit": Path(archive_audit_path).resolve(),
        "intersection_receipt": Path(intersection_receipt_path).resolve(),
        "open_knee_manifest": Path(open_knee_manifest_path).resolve(),
        "plan": Path(plan_path).resolve(),
    }
    plan = json.loads(paths["plan"].read_text(encoding="utf-8"))
    _validate_plan(plan)
    expected_paths = {
        "observations": "open_knee_observations_sha256",
        "archive_audit": "open_knee_archive_audit_sha256",
        "intersection_receipt": "source_intersection_receipt_sha256",
        "open_knee_manifest": "open_knee_registration_manifest_sha256",
    }
    for key, digest_key in expected_paths.items():
        actual = _sha256(paths[key])
        if actual != plan["inputs"].get(digest_key):
            raise RuntimeError(f"preregistered {key} hash changed: {actual}")

    intersection = json.loads(paths["intersection_receipt"].read_text(encoding="utf-8"))
    archive_audit = json.loads(paths["archive_audit"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["open_knee_manifest"].read_text(encoding="utf-8"))
    if (intersection.get("schema") != INTERSECTION_SCHEMA
            or intersection.get("source", {}).get("source_model", {}).get("revision")
            != MYOSIM_REVISION
            or intersection.get("qualification", {}).get(
                "source_myo_sim_patella_femur_surfaces_range_valid_intersections_absent"
            ) is not False):
        raise RuntimeError("source intersection failure does not match the preregistered MyoSim input")
    if (archive_audit.get("archived_reference", {}).get("observations_complete") is not True
            or archive_audit.get("archived_reference", {}).get("normal_termination_reported") is not True
            or archive_audit.get("archived_reference", {}).get("error_termination_reported") is not False
            or archive_audit.get("archived_reference", {}).get("local_reproduction") is not False):
        raise RuntimeError("archived Open Knee audit no longer matches the preregistered reference")
    frame = _registered_femoral_frame(manifest, np)
    observations_bytes = paths["observations"].read_bytes()
    observations = json.loads(gzip.decompress(observations_bytes))
    reference_rows = _relative_reference_rows(
        observations, archive_audit["inventory"]["rigid_bodies"]
    )
    if (_sha256_bytes(observations_bytes)
            != plan["inputs"]["open_knee_observations_sha256"]):
        raise RuntimeError("Open Knee observation bytes changed after preregistration")

    myosim_identity, myosim_rows = _myosim_rows(paths["sources"], np, mujoco)
    if (myosim_identity.get("source", {}).get("revision") != MYOSIM_REVISION
            or myosim_identity.get("source", {}).get("archive_sha256")
            != MYOSIM_ARCHIVE_SHA256):
        raise RuntimeError("MyoSim source identity changed during reference transfer")
    baseline_reference = reference_rows[0]
    baseline_myosim = myosim_rows[0]

    source_samples = intersection.get("sweep", {}).get("samples", [])
    selected = []
    for sample in source_samples:
        if not sample.get("projected_ranges", {}).get(
            "all_projected_coordinates_within_declared_ranges"
        ):
            continue
        angle_rad = float(sample["knee_angle_qpos_l_rad"])
        angle_deg = math.degrees(angle_rad)
        if _OPEN_KNEE_MIN_DEG - _ANGLE_EPSILON_DEG <= angle_deg <= (
            _OPEN_KNEE_MAX_DEG + _ANGLE_EPSILON_DEG
        ):
            selected.append((sample, angle_rad, angle_deg))
    if not selected:
        raise RuntimeError("no fixed common MyoSim/Open Knee samples satisfy the plan")

    model = build_model("myofullbody")
    data = mujoco.MjData(model)
    source_meshes_by_body = _compiled_meshes_by_body(model, mujoco, np)
    femur_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "femur_l"))
    patella_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "patella_l"))
    knee_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_l"))
    patella_meshes = source_meshes_by_body.get(patella_body, [])
    femur_meshes = source_meshes_by_body.get(femur_body, [])
    if min(femur_body, patella_body, knee_joint) < 0 or not patella_meshes or not femur_meshes:
        raise RuntimeError("pinned MyoSim left femur/patella source owners changed")
    knee_q = int(model.jnt_qposadr[knee_joint])
    patella_body_qpos = {
        name: int(model.jnt_qposadr[int(mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        ))])
        for name in (
            "knee_angle_beta_translation1_l",
            "knee_angle_beta_translation2_l",
            "knee_angle_beta_rotation1_l",
        )
    }
    if knee_q != 120 or any(value < 0 for value in patella_body_qpos.values()):
        raise RuntimeError("pinned left patella generalized coordinates changed")

    left_meshes = _joined_world_mesh
    femur_qpos, _, _, _ = _pose_qpos_with_unit_metrics(
        model, ((knee_q, 0.0),), mujoco, np
    )
    data.qpos[:] = femur_qpos
    mujoco.mj_forward(model, data)
    results = []
    for sample, angle_rad, angle_deg in selected:
        qpos, _, _, _ = _pose_qpos_with_unit_metrics(
            model, ((106, angle_rad), (knee_q, angle_rad)), mujoco, np
        )
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        femur_rotation = np.asarray(data.xmat[femur_body], dtype=float).reshape(3, 3)
        femur_origin = np.asarray(data.xpos[femur_body], dtype=float)
        femur_world_q_wxyz = [float(value) for value in data.xquat[femur_body]]
        femur_world_q = _normalize_quaternion([
            femur_world_q_wxyz[1], femur_world_q_wxyz[2],
            femur_world_q_wxyz[3], femur_world_q_wxyz[0],
        ])

        femur_vertices, femur_faces = left_meshes(
            femur_meshes, femur_origin, femur_rotation, np
        )
        current_patella_rotation = np.asarray(data.xmat[patella_body], dtype=float).reshape(3, 3)
        current_patella_origin = np.asarray(data.xpos[patella_body], dtype=float)
        control_vertices, control_faces = left_meshes(
            patella_meshes, current_patella_origin, current_patella_rotation, np
        )
        control_audit = _exact_surface_pair_audit(
            femur_vertices, femur_faces, control_vertices, control_faces
        )
        expected_control = sample["source_mesh_by_side"]["l"][
            "surface_intersection_triangle_pair_count"
        ]
        if control_audit["surface_intersection_triangle_pair_count"] != expected_control:
            raise RuntimeError(
                f"fixed control differs from prior source audit at {angle_deg:.6f} degrees"
            )

        reference_pose = _reference_pose_at(reference_rows, angle_deg)
        reference_steps = reference_pose.get("interpolation_steps")
        if reference_steps is None:
            reference_steps = [int(reference_pose["step"])]
        target_com_relative, target_rotation_relative = _relative_pose_from_reference(
            reference_pose, baseline_reference, baseline_myosim, frame, np
        )
        target_com_world = np.asarray(data.xipos[femur_body], dtype=float) + (
            femur_rotation @ target_com_relative
        )
        target_rotation_world_xyzw = _normalize_quaternion(_quaternion_multiply(
            femur_world_q, target_rotation_relative
        ))
        target_rotation_world = _rotation_xyzw(target_rotation_world_xyzw, np)
        target_patella_origin = target_com_world - (
            target_rotation_world @ np.asarray(model.body_ipos[patella_body], dtype=float)
        )
        candidate_vertices, candidate_faces = left_meshes(
            patella_meshes, target_patella_origin, target_rotation_world, np
        )
        candidate_audit = _exact_surface_pair_audit(
            femur_vertices, femur_faces, candidate_vertices, candidate_faces
        )

        current_com_relative = femur_rotation.T @ (
            np.asarray(data.xipos[patella_body], dtype=float)
            - np.asarray(data.xipos[femur_body], dtype=float)
        )
        current_patella_q_wxyz = [float(value) for value in data.xquat[patella_body]]
        current_patella_q = _normalize_quaternion([
            current_patella_q_wxyz[1], current_patella_q_wxyz[2],
            current_patella_q_wxyz[3], current_patella_q_wxyz[0],
        ])
        current_relative_q = _normalize_quaternion(_quaternion_multiply(
            _quaternion_inverse(femur_world_q), current_patella_q
        ))
        rotation_delta = _quaternion_multiply(
            _quaternion_inverse(current_relative_q), target_rotation_relative
        )
        results.append({
            "source_sample_name": sample["name"],
            "flexion_deg": angle_deg,
            "open_knee_reference_steps": reference_steps,
            "source_projected_ranges_pass": True,
            "control": {
                "triangle_pair_intersection_count": control_audit[
                    "surface_intersection_triangle_pair_count"
                ],
                "matches_pinned_intersection_receipt": True,
                "intersection_pair_sha256": control_audit[
                    "surface_intersection_pair_sha256"
                ],
            },
            "transferred_geometry": {
                "triangle_pair_intersection_count": candidate_audit[
                    "surface_intersection_triangle_pair_count"
                ],
                "intersection_pair_sha256": candidate_audit[
                    "surface_intersection_pair_sha256"
                ],
                "patella_com_displacement_from_source_path_mm": 1000.0 * float(
                    np.linalg.norm(target_com_relative - current_com_relative)
                ),
                "patella_orientation_change_from_source_path_deg": (
                    _quaternion_angle_degrees(rotation_delta)
                ),
                "target_patella_com_relative_femur_m": [
                    float(value) for value in target_com_relative
                ],
                "target_patella_rotation_relative_femur_xyzw": target_rotation_relative,
                "intersections_absent": candidate_audit[
                    "surface_intersection_triangle_pair_count"
                ] == 0,
            },
            "transfer_steps": reference_steps,
        })

    control_pair_count = sum(
        row["control"]["triangle_pair_intersection_count"] for row in results
    )
    candidate_pair_count = sum(
        row["transferred_geometry"]["triangle_pair_intersection_count"]
        for row in results
    )
    control_state_count = sum(
        row["control"]["triangle_pair_intersection_count"] > 0 for row in results
    )
    candidate_state_count = sum(
        row["transferred_geometry"]["triangle_pair_intersection_count"] > 0
        for row in results
    )
    candidate_clears = candidate_pair_count == 0 and candidate_state_count == 0
    root = Path(__file__).resolve().parents[2]
    source_names = (
        "patellofemoral_reference_transfer_audit.py",
        "patellofemoral_anatomical_frame_audit.py",
        "patellofemoral_pose_intersection_audit.py",
        "patellofemoral_reference_path_comparison.py",
    )
    predicate_sources = {
        name: _sha256(Path(__file__).with_name(name)) for name in source_names
    }

    def display_path(path: Path) -> str:
        resolved = Path(path).resolve()
        return (resolved.relative_to(root).as_posix()
                if resolved.is_relative_to(root) else str(resolved))

    return {
        "schema": SCHEMA,
        "status": "sampled_geometry_clear" if candidate_clears else "failed_sampled_geometry_gate",
        "qualification": {
            "control_reproduced_exact_source_intersection_counts": True,
            "transferred_sampled_source_mesh_intersections_absent": candidate_clears,
            "continuous_angle_interval_qualified": False,
            "candidate_path_represented_by_myosim_three_patellar_qpos": False,
            "tendon_ligament_attachment_motion_tested": False,
            "loaded_contact_or_pressure_qualified": False,
            "clinical_anatomy_qualified": False,
            "source_path_change_selected": False,
            "whole_human_qualified": False,
        },
        "plan": {"path": paths["plan"].relative_to(root).as_posix(),
                 "sha256": _sha256(paths["plan"])},
        "sources": {
            key: {"path": paths[key].relative_to(root).as_posix(), "sha256": _sha256(paths[key])}
            for key in ("observations", "archive_audit", "intersection_receipt", "open_knee_manifest")
        },
        "myosim": {
            "revision": MYOSIM_REVISION,
            "archive_sha256": MYOSIM_ARCHIVE_SHA256,
            "mujoco_version": mujoco.__version__,
            "frame_rotation_source_to_myo_femur": frame["rotation"].tolist(),
            "uniform_scale": frame["scale"],
            "baseline_source_qpos0_patella_com_relative_femur_m": baseline_myosim[
                "patella_inertial_com_relative_femur_inertial_com_m"
            ],
            "baseline_source_qpos0_patella_rotation_relative_femur_xyzw": baseline_myosim[
                "patella_rotation_relative_femur_xyzw"
            ],
        },
        "sampling": {
            "input_source_sample_count": len(source_samples),
            "selected_common_range_valid_sample_count": len(results),
            "angle_deg_min": min(row["flexion_deg"] for row in results),
            "angle_deg_max": max(row["flexion_deg"] for row in results),
            "angles_are_fixed_by_pinned_intersection_receipt": True,
            "open_knee_range_deg": [_OPEN_KNEE_MIN_DEG, _OPEN_KNEE_MAX_DEG],
        },
        "results": {
            "control": {
                "intersecting_triangle_pair_count": control_pair_count,
                "intersecting_state_count": control_state_count,
            },
            "transferred_geometry": {
                "intersecting_triangle_pair_count": candidate_pair_count,
                "intersecting_state_count": candidate_state_count,
                "pair_count_change": candidate_pair_count - control_pair_count,
                "state_count_change": candidate_state_count - control_state_count,
            },
        },
        "per_state": results,
        "predicate_sources": predicate_sources,
        "boundary": (
            "This cross-specimen diagnostic rigidly transfers a passive archived Open Knee "
            "patellar COM/orientation path onto the current MyoSim source patella geometry. "
            "The independent transfer is not represented by MyoSim's three patellar generalized "
            "coordinates; its QAT/PTL/ligament owners were not moved or force-checked. A sampled "
            "surface-clear result is therefore only a geometry candidate, not a source-path "
            "correction, loaded contact result, clinical anatomy finding, or whole-Human claim."
        ),
        "reproduction_command": [
            "numi", "human", "myosim-patellofemoral-reference-transfer-audit",
            "--sources", display_path(paths["sources"]),
            "--observations", display_path(paths["observations"]),
            "--archive-audit", display_path(paths["archive_audit"]),
            "--intersection-receipt", display_path(paths["intersection_receipt"]),
            "--open-knee-manifest", display_path(paths["open_knee_manifest"]),
            "--plan", display_path(paths["plan"]), "--output", "<new-receipt-path>",
            "--python", sys.executable,
        ],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--archive-audit", type=Path, required=True)
    parser.add_argument("--intersection-receipt", type=Path, required=True)
    parser.add_argument("--open-knee-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_reference_transfer(
            sources=args.sources,
            observations_path=args.observations,
            archive_audit_path=args.archive_audit,
            intersection_receipt_path=args.intersection_receipt,
            open_knee_manifest_path=args.open_knee_manifest,
            plan_path=args.plan,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"patellofemoral reference transfer audit: {error}", file=sys.stderr)
        return 2
    print(json.dumps({
        "schema": SCHEMA,
        "status": result["status"],
        "output": str(args.output.resolve()),
        "sample_count": result["sampling"]["selected_common_range_valid_sample_count"],
        "results": result["results"],
    }, sort_keys=True))
    return 0 if result["qualification"][
        "transferred_sampled_source_mesh_intersections_absent"
    ] else 2


if __name__ == "__main__":  # pragma: no cover - exercised through owner CLI
    raise SystemExit(main())
