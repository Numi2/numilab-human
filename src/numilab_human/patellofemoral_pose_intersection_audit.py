"""Exact sampled-pose intersection audit for source and registered knee meshes.

This audit asks whether the pinned MyoSim source geoms and current registered
bony surface candidates cross under source-equality-projected poses. It is a
necessary geometric check only: an intersection-free sample does not establish
continuous motion, volume containment, cartilage contact, clinical
registration, or loaded patellar tracking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from . import model as human_model
from .cardiac_cavity_intersections import (
    _audit_pair,
    _records,
    triangle_intersection_points,
)
from .lower_limb_pose_audit import POSE_SUITE, _sha256
from .myosim_bone_proximity import _compiled_meshes_by_body
from .myosim_export import export_fullbody
from .upper_limb_pose_audit import (
    _compiled_bone_members,
    _joint_equality_program_checks,
    _pose_joint_range_context,
    _pose_qpos_with_unit_metrics,
    _projected_joint_range_checks,
)
from .upper_limb_registration import _rotation_xyzw


SCHEMA = "numi.human.patellofemoral-pose-intersection-audit.v1"
ROOT = Path(__file__).resolve().parents[2]
SWEEP_STEP_RAD = 0.05
SIDE_MEMBERS = {
    "r": {"femur": "femur_r", "patella": "patella_r"},
    "l": {"femur": "femur_l", "patella": "patella_l"},
}


def _display_path(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def _float_integer_lattice(points: Any) -> tuple[list[tuple[int, int, int]], int]:
    """Map finite binary64 coordinates exactly to one power-of-two lattice."""
    import numpy as np

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise RuntimeError("patellofemoral intersection input has invalid coordinates")
    ratios = [float(value).as_integer_ratio() for value in values.reshape(-1)]
    exponent = max((denominator.bit_length() - 1 for _, denominator in ratios), default=0)
    if any(denominator != 1 << (denominator.bit_length() - 1)
           for _, denominator in ratios):
        raise RuntimeError("binary64 coordinate denominator is not a power of two")
    integers = [numerator << (exponent - (denominator.bit_length() - 1))
                for numerator, denominator in ratios]
    return [tuple(integers[index:index + 3])
            for index in range(0, len(integers), 3)], exponent


def _exact_surface_pair_audit(
    first_vertices: Any,
    first_faces: Any,
    second_vertices: Any,
    second_faces: Any,
) -> dict[str, Any]:
    """Count exact binary64 triangle-surface intersections and retain witnesses."""
    import numpy as np

    first_vertices = np.asarray(first_vertices, dtype=np.float64)
    second_vertices = np.asarray(second_vertices, dtype=np.float64)
    first_faces = np.asarray(first_faces, dtype=np.int64)
    second_faces = np.asarray(second_faces, dtype=np.int64)
    if (first_faces.ndim != 2 or first_faces.shape[1] != 3
            or second_faces.ndim != 2 or second_faces.shape[1] != 3
            or not len(first_faces) or not len(second_faces)
            or int(first_faces.min()) < 0 or int(first_faces.max()) >= len(first_vertices)
            or int(second_faces.min()) < 0 or int(second_faces.max()) >= len(second_vertices)):
        raise RuntimeError("patellofemoral intersection input has invalid triangle topology")

    joined, exponent = _float_integer_lattice(
        np.concatenate((first_vertices, second_vertices), axis=0)
    )
    first_integer = joined[:len(first_vertices)]
    second_integer = joined[len(first_vertices):]
    first_records = _records(first_integer, [tuple(map(int, face)) for face in first_faces])
    second_records = _records(second_integer, [tuple(map(int, face)) for face in second_faces])
    result = _audit_pair(first_records, second_records, same_surface=False)
    intersections = []
    for first_index, second_index in result["triangle_pairs"]:
        intersections.extend(triangle_intersection_points(
            first_records[first_index][0], second_records[second_index][0]
        ))
    unique_points = sorted(set(intersections))
    if unique_points:
        scale = float(1 << exponent)
        bounds = {
            "minimum_world_m": [float(min(point[axis] for point in unique_points)) / scale
                                for axis in range(3)],
            "maximum_world_m": [float(max(point[axis] for point in unique_points)) / scale
                                for axis in range(3)],
        }
    else:
        bounds = None
    pairs = result["triangle_pairs"]
    return {
        "aabb_candidate_triangle_pair_count": result["aabb_candidate_pairs"],
        "surface_intersection_triangle_pair_count": result["count"],
        "surface_intersection_pair_sha256": hashlib.sha256(
            json.dumps(pairs, separators=(",", ":")).encode("ascii")
        ).hexdigest(),
        "first_surface_intersection_triangle_pairs": pairs[:16],
        "exact_unique_intersection_point_count": len(unique_points),
        "intersection_world_bounds_m": bounds,
        "coordinate_lattice_denominator": str(1 << exponent),
        "predicate": "exact_binary64_integer_triangle_intersection",
    }


def _range_check_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [item for item in checks if not item["passed"]]
    return {
        "all_projected_coordinates_within_declared_ranges": not failures,
        "failed_joint_count": len(failures),
        "failed_joints": [
            {
                "source_joint_name": item["source_joint_name"],
                "source_range_violation": item["source_range_violation"],
                "native_range_violation": item["native_range_violation"],
            }
            for item in failures
        ],
    }


def _collision_gate(
    samples: list[dict[str, Any]], surface_field: str = "by_side"
) -> dict[str, Any]:
    admitted = [sample for sample in samples
                if sample["projected_ranges"]["all_projected_coordinates_within_declared_ranges"]]
    collisions = [
        {"pose": sample["name"], "side": side,
         "triangle_pair_count": row["surface_intersection_triangle_pair_count"]}
        for sample in admitted
        for side, row in sample[surface_field].items()
        if row["surface_intersection_triangle_pair_count"] > 0
    ]
    return {
        "range_valid_sample_count": len(admitted),
        "range_valid_intersecting_pose_side_count": len(collisions),
        "range_valid_intersection_pair_count": sum(row["triangle_pair_count"]
                                                    for row in collisions),
        "range_valid_pose_surface_intersections_absent": not collisions,
        "first_failures": collisions[:32],
    }


def _sample_angles(lower: float, upper: float, step: float = SWEEP_STEP_RAD) -> list[float]:
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper
            and math.isfinite(step) and step > 0):
        raise RuntimeError("patellofemoral angle-sweep range is invalid")
    samples = []
    index = 0
    while lower + index * step < upper:
        samples.append(float(lower + index * step))
        index += 1
    if not samples or samples[-1] != upper:
        samples.append(float(upper))
    return samples


def audit_patellofemoral_pose_intersections(
    *, sources: Path, registration_path: Path, artifact: Path, bone_artifact: Path,
) -> dict[str, Any]:
    """Audit the existing pose suite and sampled admitted knee flexion range."""
    try:
        import mujoco
        import numpy as np
        from myo_sim.build.compose import build_model
    except ImportError as error:  # pragma: no cover - source environment only
        raise RuntimeError(
            "patellofemoral pose audit requires the pinned MyoSim/MuJoCo environment"
        ) from error

    registration_path = Path(registration_path).resolve()
    artifact = Path(artifact).resolve()
    bone_artifact = Path(bone_artifact).resolve()
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    runtime_reference, runtime_bodies = human_model._bodyparts_runtime_bindings(
        registration, artifact
    )
    members, bone_descriptor = _compiled_bone_members(
        bone_artifact, registration_path, registration, runtime_reference, runtime_bodies
    )
    if bone_descriptor.get("payload_abi") != 3:
        raise RuntimeError("patellofemoral pose audit requires source-owned NHBONES1 ABI 3")

    exported = export_fullbody(Path(sources).resolve())
    if exported.get("source") != registration["source"]["myosim"]["source"]:
        raise RuntimeError("patellofemoral audit source export differs from registration")
    source_bodies = {int(body["id"]): body for body in exported["bodies"]}
    anchor_by_target: dict[str, list[dict[str, Any]]] = {}
    for anchor in registration.get("anchors", []):
        target_name = anchor.get("target", {}).get("name")
        if target_name in {name for pair in SIDE_MEMBERS.values() for name in pair.values()}:
            anchor_by_target.setdefault(str(target_name), []).append(anchor)
    for side, pair in SIDE_MEMBERS.items():
        for kind, target in pair.items():
            if len(anchor_by_target.get(target, [])) != 1:
                raise RuntimeError(
                    f"patellofemoral audit requires one registered {target} source surface"
                )

    model = build_model("myofullbody")
    data = mujoco.MjData(model)
    source_meshes_by_body = _compiled_meshes_by_body(model, mujoco, np)
    joint_ranges = _pose_joint_range_context(artifact, runtime_reference, model, mujoco)
    rigid_program = human_model._myosim_rigid_program_checks(artifact, exported)
    equality_programs = _joint_equality_program_checks(
        artifact, runtime_reference, exported, joint_ranges
    )
    source_programs_match = (
        rigid_program["passed"] and all(item["passed"] for item in equality_programs)
    )
    range_by_q = {int(item["core_q_index"]): item for item in joint_ranges}
    if 106 not in range_by_q or 120 not in range_by_q:
        raise RuntimeError("patellofemoral audit cannot find both source knee-angle ranges")
    right_range = range_by_q[106]["source_range"]
    left_range = range_by_q[120]["source_range"]
    if right_range != left_range:
        raise RuntimeError("bilateral source knee-angle ranges differ")
    if range_by_q[106]["source_name"] != "knee_angle_r" or range_by_q[120]["source_name"] != "knee_angle_l":
        raise RuntimeError("patellofemoral audit knee-angle source identities changed")

    def posed_mesh(target_name: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        anchors = anchor_by_target[target_name]
        anchor = anchors[0]
        body_id = int(anchor["target"]["source_body_id"])
        body_name = str(anchor["target"]["name"])
        source_body = source_bodies.get(body_id)
        if source_body is None or source_body.get("name") != body_name:
            raise RuntimeError(f"patellofemoral source body ownership changed for {body_name}")
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) != body_name:
            raise RuntimeError(f"patellofemoral runtime body identity changed for {body_name}")
        member_id = str(anchor["source"]["member_id"])
        surface = members[member_id]
        rotation = _rotation_xyzw(source_body["inertial_quaternion_body_xyzw"], np)
        body_vertices = np.einsum(
            "ki,ji->kj", np.asarray(surface["vertices"], dtype=np.float64), rotation
        ) + np.asarray(source_body["inertial_position_body_m"], dtype=np.float64)
        world_vertices = np.einsum(
            "ki,ji->kj", body_vertices, data.xmat[body_id].reshape(3, 3)
        ) + data.xpos[body_id]
        faces = np.asarray(surface["triangles"], dtype=np.int64)
        return world_vertices, faces, [{
            "source_member_id": member_id,
            "source_body_name": body_name,
            "vertex_count": len(world_vertices),
            "triangle_count": len(faces),
        }]

    def posed_source_mesh(target_name: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        anchor = anchor_by_target[target_name][0]
        body_id = int(anchor["target"]["source_body_id"])
        body_name = str(anchor["target"]["name"])
        meshes = source_meshes_by_body.get(body_id, [])
        if not meshes or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) != body_name:
            raise RuntimeError(f"pinned MyoSim source has no owner mesh for {body_name}")
        world_vertices = []
        faces = []
        source_records = []
        vertex_offset = 0
        for mesh in meshes:
            local_vertices = np.asarray(mesh["vertices"], dtype=np.float64)
            posed_vertices = np.einsum(
                "ki,ji->kj", local_vertices, data.xmat[body_id].reshape(3, 3)
            ) + data.xpos[body_id]
            mesh_faces = np.asarray(mesh["faces"], dtype=np.int64)
            if len(mesh_faces):
                faces.append(mesh_faces + vertex_offset)
            world_vertices.append(posed_vertices)
            source_records.append({
                "geom_name": mesh["geom_name"],
                "mesh_name": mesh["mesh_name"],
                "vertex_count": len(local_vertices),
                "triangle_count": len(mesh_faces),
            })
            vertex_offset += len(local_vertices)
        if not faces:
            raise RuntimeError(f"pinned MyoSim source mesh for {body_name} has no triangles")
        return np.concatenate(world_vertices), np.concatenate(faces), source_records

    def audit_qpos(name: str, overrides: tuple[tuple[int, float], ...]) -> dict[str, Any]:
        qpos, _, _, _ = _pose_qpos_with_unit_metrics(model, overrides, mujoco, np)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        range_summary = _range_check_summary(_projected_joint_range_checks(qpos, joint_ranges, np))
        by_side = {}
        source_mesh_by_side = {}
        for side, pair in SIDE_MEMBERS.items():
            femur_vertices, femur_faces, femur_sources = posed_mesh(pair["femur"])
            patella_vertices, patella_faces, patella_sources = posed_mesh(pair["patella"])
            by_side[side] = {
                "source_surfaces": {"femur": femur_sources, "patella": patella_sources},
                **_exact_surface_pair_audit(
                    femur_vertices, femur_faces, patella_vertices, patella_faces
                ),
            }
            source_femur_vertices, source_femur_faces, source_femur_records = posed_source_mesh(
                pair["femur"]
            )
            source_patella_vertices, source_patella_faces, source_patella_records = posed_source_mesh(
                pair["patella"]
            )
            source_mesh_by_side[side] = {
                "source_surfaces": {
                    "femur": source_femur_records,
                    "patella": source_patella_records,
                },
                **_exact_surface_pair_audit(
                    source_femur_vertices, source_femur_faces,
                    source_patella_vertices, source_patella_faces,
                ),
            }
        return {
            "name": name,
            "knee_angle_qpos_r_rad": float(qpos[106]),
            "knee_angle_qpos_l_rad": float(qpos[120]),
            "projected_ranges": range_summary,
            "by_side": by_side,
            "source_mesh_by_side": source_mesh_by_side,
        }

    pose_samples = [audit_qpos(name, overrides) for name, overrides in POSE_SUITE]
    sweep_angles = _sample_angles(float(right_range[0]), float(right_range[1]))
    sweep_samples = [
        audit_qpos(f"bilateral_knee_angle_{index:03d}", ((106, angle), (120, angle)))
        for index, angle in enumerate(sweep_angles)
    ]
    gate = _collision_gate(sweep_samples)
    source_mesh_gate = _collision_gate(sweep_samples, "source_mesh_by_side")
    failed_pose_suite = _collision_gate(pose_samples)
    source_mesh_pose_gate = _collision_gate(pose_samples, "source_mesh_by_side")
    registered_surface_intersections_absent = (
        gate["range_valid_pose_surface_intersections_absent"]
        and failed_pose_suite["range_valid_pose_surface_intersections_absent"]
    )
    source_mesh_intersections_absent = (
        source_mesh_gate["range_valid_pose_surface_intersections_absent"]
        and source_mesh_pose_gate["range_valid_pose_surface_intersections_absent"]
    )
    both_surface_gates_pass = (
        registered_surface_intersections_absent and source_mesh_intersections_absent
    )
    qualification = {
        "source_programs_match_pinned_source": source_programs_match,
        "registered_patella_femur_surfaces_range_valid_intersections_absent": (
            registered_surface_intersections_absent
        ),
        "source_myo_sim_patella_femur_surfaces_range_valid_intersections_absent": (
            source_mesh_intersections_absent
        ),
        "range_valid_sampled_pose_surface_intersections_absent": (
            both_surface_gates_pass
        ),
        "sampled_range_collision_gate_passed": (
            both_surface_gates_pass and source_programs_match
        ),
        "continuous_angle_interval_qualified": False,
        "closed_volume_containment_tested": False,
        "cartilage_contact_or_pressure_qualified": False,
        "loaded_tracking_qualified": False,
        "clinical_anatomy_qualified": False,
    }
    return {
        "schema": SCHEMA,
        "status": ("passed_sampled_surface_intersection_gate"
                   if qualification["sampled_range_collision_gate_passed"]
                   else "failed_sampled_surface_intersection_gate"),
        "qualification": qualification,
        "source": {
            "registration": {"path": _display_path(registration_path),
                             "sha256": _sha256(registration_path)},
            "bone_payload": {
                "path": _display_path(Path(bone_descriptor["file"])),
                             "sha256": bone_descriptor["sha256"],
                             "abi": bone_descriptor["payload_abi"]},
            "bone_manifest": {
                "path": _display_path(
                    bone_artifact / "bodyparts3d-myosim-major-bones.manifest.json"
                ),
                "sha256": _sha256(
                    bone_artifact / "bodyparts3d-myosim-major-bones.manifest.json"
                ),
            },
            "rigid_program_sha256": runtime_reference["rigid"]["sha256"],
            "source_model": exported["source"],
        },
        "source_program_checks": {
            "rigid": rigid_program,
            "joint_equalities": equality_programs,
        },
        "predicate_sources": {
            "patellofemoral_pose_intersection_audit.py": _sha256(Path(__file__).resolve()),
            "cardiac_cavity_intersections.py": _sha256(
                Path(__file__).with_name("cardiac_cavity_intersections.py")
            ),
            "upper_limb_pose_audit.py": _sha256(
                Path(__file__).with_name("upper_limb_pose_audit.py")
            ),
        },
        "sweep": {
            "knee_angle_source_range_rad": [float(right_range[0]), float(right_range[1])],
            "sample_step_rad": SWEEP_STEP_RAD,
            "sample_count": len(sweep_samples),
            "sampling_basis": "uniform_source_range_samples_plus_exact_upper_endpoint",
            "collision_gate": gate,
            "source_myo_sim_mesh_collision_gate": source_mesh_gate,
            "samples": sweep_samples,
        },
        "source_pose_suite": {
            "sample_count": len(pose_samples),
            "registered_source_surface_collision_gate": failed_pose_suite,
            "source_myo_sim_mesh_collision_gate": source_mesh_pose_gate,
            "samples": pose_samples,
        },
        "boundary": (
            "Exact triangle-surface intersections are checked on both pinned MyoSim source "
            "mesh geoms and source-owned ABI 3 NHBONES1 geometry after MyoSim/MuJoCo source "
            "equality projection. A crossing at a pose "
            "whose projected joint coordinates pass declared source and native ranges fails "
            "this sampled geometric gate. Source-mesh crossings locate a source pose/geometry "
            "problem that cannot be explained by BodyParts3D registration alone. Passing discrete samples does not prove a continuous "
            "motion envelope, closed-volume containment, cartilage contact, loaded patellar "
            "tracking, subject calibration, clinical anatomy, or whole-Human qualification."
        ),
        "reproduction_command": [
            sys.executable, "-m", "numilab_human.patellofemoral_pose_intersection_audit",
            "--sources", _display_path(Path(sources)),
            "--artifact", _display_path(artifact),
            "--bone-artifact", _display_path(bone_artifact),
            "--registration", _display_path(registration_path),
            "--output", "<new-receipt-path>",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--bone-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_patellofemoral_pose_intersections(
            sources=args.sources,
            registration_path=args.registration,
            artifact=args.artifact,
            bone_artifact=args.bone_artifact,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f"patellofemoral pose intersection audit: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "output": str(args.output.resolve()),
                      "sample_count": result["sweep"]["sample_count"],
                      "collision_gate": result["sweep"]["collision_gate"]},
                     sort_keys=True))
    return 0 if result["qualification"]["sampled_range_collision_gate_passed"] else 2


if __name__ == "__main__":  # pragma: no cover - exercised through owner CLI
    raise SystemExit(main())
