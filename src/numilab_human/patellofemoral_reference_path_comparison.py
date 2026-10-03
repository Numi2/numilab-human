"""Exploratory comparison of MyoSim and archived Open Knee patellar paths.

The two paths belong to different specimens and different mechanical models.
This report is a diagnostic, not calibration or clinical validation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


SCHEMA = "numi.human.patellofemoral-reference-path-comparison.v1"
MYOSIM_REVISION = "33c89c2bde282553dde3f526768eb3bdcfaa7649"
MYOSIM_ARCHIVE_SHA256 = "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975"
PATELLA_REBASE = {
    "checkout/myo_sim/models/leg/assets/myolegs_chain.xml":
        "98e514ce869683e273a18794f6494df0c64dec13eec636b683a165cf30109b29",
    "checkout/myo_sim/models/leg/assets/myolegs_assets.xml":
        "293a7f618e4ae151ff9cab6f277cbb88caa1d7f418104ab85776c208d6ab3b3d",
}
FLEXION_TARGETS_DEG = (30.0, 60.0, 90.0)
MYOSIM_SAMPLE_COUNT = 181
OPEN_KNEE_BODIES = {"patella": "1", "tibia": "3", "femur": "4"}
OPEN_KNEE_EXPECTED_BODY_NAMES = {"1": "PTB", "3": "TBB", "4": "FMB"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path, root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _normalize_quaternion(q: list[float] | tuple[float, ...]) -> list[float]:
    if len(q) != 4 or not all(math.isfinite(float(x)) for x in q):
        raise RuntimeError("reference quaternion must contain four finite values")
    norm = math.sqrt(sum(float(x) ** 2 for x in q))
    if not math.isfinite(norm) or abs(norm - 1.0) > 2.0e-3:
        raise RuntimeError(f"reference quaternion is not unit length: norm={norm}")
    return [float(x) / norm for x in q]


def _quaternion_multiply(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _quaternion_inverse(q: list[float]) -> list[float]:
    q = _normalize_quaternion(q)
    return [-q[0], -q[1], -q[2], q[3]]


def _quaternion_rotate(q: list[float], vector: list[float]) -> list[float]:
    q = _normalize_quaternion(q)
    result = _quaternion_multiply(
        _quaternion_multiply(q, [float(x) for x in vector] + [0.0]),
        _quaternion_inverse(q),
    )
    return result[:3]


def _quaternion_angle_degrees(q: list[float]) -> float:
    normalized = _normalize_quaternion(q)
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, abs(normalized[3])))))


def _quaternion_slerp(a: list[float], b: list[float], amount: float) -> list[float]:
    a = _normalize_quaternion(a)
    b = _normalize_quaternion(b)
    dot = sum(a[index] * b[index] for index in range(4))
    if dot < 0.0:
        b = [-value for value in b]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        return _normalize_quaternion([
            (1.0 - amount) * a[index] + amount * b[index] for index in range(4)
        ])
    theta = math.acos(dot)
    denominator = math.sin(theta)
    left = math.sin((1.0 - amount) * theta) / denominator
    right = math.sin(amount * theta) / denominator
    return _normalize_quaternion([
        left * a[index] + right * b[index] for index in range(4)
    ])


def _relative_reference_rows(
    observations: dict[str, Any], rigid_bodies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decode FMB/TBB/PTB rigid observations into femur-relative transforms."""
    if (observations.get("status") != "normal_termination_with_complete_observations"
            or observations.get("observations_complete") is not True
            or observations.get("accepted_increment_count") != 140
            or observations.get("time_semantics")
            != "quasi-static continuation parameter, not elapsed physical seconds"):
        raise RuntimeError("archived Open Knee observations do not match the admitted reference")
    body_names = {str(row.get("id")): row.get("name") for row in rigid_bodies}
    if any(body_names.get(body_id) != name
           for body_id, name in OPEN_KNEE_EXPECTED_BODY_NAMES.items()):
        raise RuntimeError("archived Open Knee rigid-body identities changed")

    by_step: dict[int, dict[str, dict[str, Any]]] = {}
    for row in observations.get("records", []):
        field = row.get("field")
        if field not in {"center_of_mass", "rotation_quaternion"}:
            continue
        step = row.get("step")
        if type(step) is not int or step < 1:
            raise RuntimeError("archived Open Knee observation has an invalid step")
        target = by_step.setdefault(step, {})
        if field in target:
            raise RuntimeError(f"duplicate Open Knee observation field at step {step}: {field}")
        target[field] = row

    expected_steps = list(range(1, observations["accepted_increment_count"] + 1))
    if sorted(by_step) != expected_steps:
        raise RuntimeError("archived Open Knee observation steps are incomplete")

    result = []
    for step in expected_steps:
        state = by_step[step]
        if set(state) != {"center_of_mass", "rotation_quaternion"}:
            raise RuntimeError(f"incomplete Open Knee rigid observations at step {step}")
        position_record = state["center_of_mass"]
        rotation_record = state["rotation_quaternion"]
        if position_record.get("units") != "mm":
            raise RuntimeError("archived Open Knee centers of mass are not in millimetres")
        positions = position_record.get("values", {})
        rotations = rotation_record.get("values", {})
        if any(body_id not in positions or body_id not in rotations
               for body_id in OPEN_KNEE_BODIES.values()):
            raise RuntimeError(f"missing Open Knee rigid body at step {step}")

        def vector(body_id: str) -> list[float]:
            value = positions[body_id]
            if not isinstance(value, list) or len(value) != 3:
                raise RuntimeError(f"invalid Open Knee center of mass at step {step}")
            output = [float(x) for x in value]
            if not all(math.isfinite(x) for x in output):
                raise RuntimeError(f"nonfinite Open Knee center of mass at step {step}")
            return output

        def quaternion(body_id: str) -> list[float]:
            value = rotations[body_id]
            if not isinstance(value, list) or len(value) != 4:
                raise RuntimeError(f"invalid Open Knee quaternion at step {step}")
            return _normalize_quaternion([float(x) for x in value])

        pf, pt, pp = (vector(OPEN_KNEE_BODIES[name]) for name in ("femur", "tibia", "patella"))
        qf, qt, qp = (quaternion(OPEN_KNEE_BODIES[name])
                      for name in ("femur", "tibia", "patella"))
        qf_inverse = _quaternion_inverse(qf)
        relative_patella_position = _quaternion_rotate(
            qf_inverse, [pp[i] - pf[i] for i in range(3)]
        )
        relative_tibia_rotation = _quaternion_multiply(qf_inverse, qt)
        relative_patella_rotation = _quaternion_multiply(qf_inverse, qp)
        result.append({
            "step": step,
            "continuation_time": float(position_record["continuation_time"]),
            "knee_flexion_deg": _quaternion_angle_degrees(relative_tibia_rotation),
            "patella_position_relative_femur_com_mm": relative_patella_position,
            "patella_rotation_relative_femur_xyzw": relative_patella_rotation,
        })
    return result


def _interpolate(rows: list[dict[str, Any]], field: str, target_deg: float) -> float:
    ordered = sorted(rows, key=lambda row: row["knee_flexion_deg"])
    x = [float(row["knee_flexion_deg"]) for row in ordered]
    if target_deg < x[0] or target_deg > x[-1]:
        raise RuntimeError(f"requested Open Knee flexion {target_deg} degrees requires extrapolation")
    for index, upper in enumerate(x):
        if upper == target_deg:
            return float(ordered[index][field])
        if upper > target_deg:
            lower = index - 1
            amount = (target_deg - x[lower]) / (upper - x[lower])
            return (1.0 - amount) * float(ordered[lower][field]) + amount * float(ordered[index][field])
    return float(ordered[-1][field])


def _interpolate_reference_pose(
    rows: list[dict[str, Any]], target_deg: float,
) -> tuple[list[float], list[float]]:
    ordered = sorted(rows, key=lambda row: row["knee_flexion_deg"])
    lower_bound = float(ordered[0]["knee_flexion_deg"])
    upper_bound = float(ordered[-1]["knee_flexion_deg"])
    if target_deg < lower_bound or target_deg > upper_bound:
        raise RuntimeError(f"requested Open Knee flexion {target_deg} degrees requires extrapolation")
    for index, row in enumerate(ordered):
        angle = float(row["knee_flexion_deg"])
        if angle == target_deg:
            return (list(row["patella_position_relative_femur_com_mm"]),
                    list(row["patella_rotation_relative_femur_xyzw"]))
        if angle > target_deg:
            prior = ordered[index - 1]
            prior_angle = float(prior["knee_flexion_deg"])
            amount = (target_deg - prior_angle) / (angle - prior_angle)
            prior_position = prior["patella_position_relative_femur_com_mm"]
            next_position = row["patella_position_relative_femur_com_mm"]
            position = [
                (1.0 - amount) * float(prior_position[axis])
                + amount * float(next_position[axis])
                for axis in range(3)
            ]
            rotation = _quaternion_slerp(
                prior["patella_rotation_relative_femur_xyzw"],
                row["patella_rotation_relative_femur_xyzw"], amount,
            )
            return position, rotation
    return (list(ordered[-1]["patella_position_relative_femur_com_mm"]),
            list(ordered[-1]["patella_rotation_relative_femur_xyzw"]))


def _relative_reference_metrics(
    rows: list[dict[str, Any]], baseline_flexion_deg: float,
) -> list[dict[str, Any]]:
    p0, q0 = _interpolate_reference_pose(rows, baseline_flexion_deg)
    q0_inverse = _quaternion_inverse(q0)
    output = []
    for row in rows:
        position_delta = [
            float(row["patella_position_relative_femur_com_mm"][i]) - float(p0[i])
            for i in range(3)
        ]
        q_delta = _quaternion_multiply(
            q0_inverse, row["patella_rotation_relative_femur_xyzw"]
        )
        output.append({
            **row,
            "patella_origin_displacement_from_baseline_mm": math.sqrt(
                sum(value * value for value in position_delta)
            ),
            "patella_orientation_change_from_baseline_deg": (
                _quaternion_angle_degrees(q_delta)
            ),
        })
    return output


def _path_length(rows: list[dict[str, Any]], field: str) -> float:
    return sum(
        math.sqrt(sum((float(rows[i][field][axis]) - float(rows[i - 1][field][axis])) ** 2
                      for axis in range(3)))
        for i in range(1, len(rows))
    )


def _valid_source_segment_summary(
    rows: list[dict[str, Any]], step_deg: float,
) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for row in rows:
        if (not groups or float(row["knee_flexion_deg"])
                - float(groups[-1][-1]["knee_flexion_deg"]) > step_deg + 1.0e-8):
            groups.append([row])
        else:
            groups[-1].append(row)
    return [
        {
            "start_flexion_deg": float(group[0]["knee_flexion_deg"]),
            "end_flexion_deg": float(group[-1]["knee_flexion_deg"]),
            "sample_count": len(group),
            "body_origin_travel_mm": 1000.0 * _path_length(
                group, "patella_position_relative_femur_body_origin_m"
            ),
        }
        for group in groups
    ]


def _myosim_rows(sources: Path, np: Any, mujoco: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from myo_sim.build.compose import build_model
    from .myosim_export import export_fullbody
    from .upper_limb_pose_audit import _pose_qpos_with_unit_metrics

    exported = export_fullbody(sources.resolve())
    source = exported.get("source", {})
    if (source.get("revision") != MYOSIM_REVISION
            or source.get("archive_sha256") != MYOSIM_ARCHIVE_SHA256):
        raise RuntimeError("MyoSim patellofemoral source revision or archive identity changed")
    source_files = {}
    for relative, expected in PATELLA_REBASE.items():
        path = sources.resolve() / "myosim" / relative
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(f"MyoSim patella source overlay changed: {relative} sha256={actual}")
        source_files[relative] = {"sha256": actual}

    model = build_model("myofullbody")
    knee_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "knee_angle_l"))
    femur_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "femur_l"))
    patella_body = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "patella_l"))
    if min(knee_joint, femur_body, patella_body) < 0:
        raise RuntimeError("pinned MyoSim left patellofemoral source identities changed")
    q_index = int(model.jnt_qposadr[knee_joint])
    source_range = [float(value) for value in model.jnt_range[knee_joint]]
    if q_index != 120 or source_range != [0.0, 2.0944]:
        raise RuntimeError("pinned MyoSim left-knee driver identity or range changed")

    data = mujoco.MjData(model)
    theta_values = np.linspace(0.0, math.pi / 2.0, MYOSIM_SAMPLE_COUNT)
    rows = []
    invalid_angles = []
    for theta in theta_values:
        qpos, _, _, _ = _pose_qpos_with_unit_metrics(
            model, ((q_index, float(theta)),), mujoco, np
        )
        range_violations = []
        for joint in range(model.njnt):
            if (bool(model.jnt_limited[joint])
                    and int(model.jnt_type[joint]) in (
                        int(mujoco.mjtJoint.mjJNT_SLIDE),
                        int(mujoco.mjtJoint.mjJNT_HINGE),
                    )):
                index = int(model.jnt_qposadr[joint])
                lower, upper = (float(value) for value in model.jnt_range[joint])
                value = float(qpos[index])
                if value < lower - 1.0e-12 or value > upper + 1.0e-12:
                    range_violations.append({
                        "joint": mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_JOINT, joint
                        ),
                        "value": value,
                        "range": [lower, upper],
                    })
        if range_violations:
            invalid_angles.append({
                "flexion_deg": math.degrees(float(theta)),
                "violations": range_violations,
            })
            continue
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        qf_wxyz = [float(value) for value in data.xquat[femur_body]]
        qp_wxyz = [float(value) for value in data.xquat[patella_body]]
        qf = _normalize_quaternion([qf_wxyz[1], qf_wxyz[2], qf_wxyz[3], qf_wxyz[0]])
        qp = _normalize_quaternion([qp_wxyz[1], qp_wxyz[2], qp_wxyz[3], qp_wxyz[0]])
        qf_inverse = _quaternion_inverse(qf)
        position = _quaternion_rotate(
            qf_inverse,
            (data.xpos[patella_body] - data.xpos[femur_body]).tolist(),
        )
        rotation = _quaternion_multiply(qf_inverse, qp)
        rows.append({
            "knee_flexion_deg": math.degrees(float(theta)),
            "patella_position_relative_femur_body_origin_m": position,
            "patella_rotation_relative_femur_xyzw": rotation,
        })

    if not rows:
        raise RuntimeError("MyoSim has no source-range-valid samples in the comparison interval")
    p0 = rows[0]["patella_position_relative_femur_body_origin_m"]
    q0_inverse = _quaternion_inverse(rows[0]["patella_rotation_relative_femur_xyzw"])
    for row in rows:
        delta = [row["patella_position_relative_femur_body_origin_m"][i] - p0[i]
                 for i in range(3)]
        row["patella_origin_displacement_from_baseline_mm"] = 1000.0 * math.sqrt(
            sum(value * value for value in delta)
        )
        row["patella_orientation_change_from_baseline_deg"] = _quaternion_angle_degrees(
            _quaternion_multiply(q0_inverse, row["patella_rotation_relative_femur_xyzw"])
        )
    invalid_angle_values = [row["flexion_deg"] for row in invalid_angles]
    invalid_ranges = []
    for angle_value in invalid_angle_values:
        if (not invalid_ranges
                or angle_value - invalid_ranges[-1]["end_deg"]
                > 90.0 / (MYOSIM_SAMPLE_COUNT - 1) + 1.0e-8):
            invalid_ranges.append({"start_deg": angle_value, "end_deg": angle_value,
                                   "sample_count": 1})
        else:
            invalid_ranges[-1]["end_deg"] = angle_value
            invalid_ranges[-1]["sample_count"] += 1
    return {"source": source, "source_files": source_files,
            "mujoco_version": str(mujoco.__version__),
            "left_knee_qpos_index": q_index,
            "left_knee_source_range_rad": source_range,
            "source_range_valid_sample_count": len(rows),
            "source_range_invalid_sample_count": len(theta_values) - len(rows),
            "source_range_valid_start_deg": rows[0]["knee_flexion_deg"],
            "source_range_valid_end_deg": rows[-1]["knee_flexion_deg"],
            "source_range_invalid_ranges": invalid_ranges,
            "source_range_invalid_samples": invalid_angles,
            "sample_count": len(theta_values),
            "sampling_step_deg": 90.0 / (MYOSIM_SAMPLE_COUNT - 1)}, rows


def compare_patellofemoral_reference_path(
    *, sources: Path, observations_path: Path, archive_audit_path: Path,
    intersection_receipt_path: Path,
) -> dict[str, Any]:
    """Compare gross path magnitudes without fitting or promoting either model."""
    try:
        import mujoco
        import numpy as np
    except ImportError as error:  # pragma: no cover - requires pinned source environment
        raise RuntimeError(
            "patellofemoral reference comparison requires pinned MyoSim, NumPy, and MuJoCo"
        ) from error

    sources = Path(sources).resolve()
    observations_path = Path(observations_path).resolve()
    archive_audit_path = Path(archive_audit_path).resolve()
    intersection_receipt_path = Path(intersection_receipt_path).resolve()
    archive_audit = json.loads(archive_audit_path.read_text(encoding="utf-8"))
    archive_info = archive_audit.get("archived_reference", {})
    if (archive_info.get("observations_complete") is not True
            or archive_info.get("normal_termination_reported") is not True
            or archive_info.get("error_termination_reported") is not False
            or archive_info.get("local_reproduction") is not False):
        raise RuntimeError("Open Knee archive audit is incomplete or has changed identity")
    intersection = json.loads(intersection_receipt_path.read_text(encoding="utf-8"))
    if (intersection.get("schema")
            != "numi.human.patellofemoral-pose-intersection-audit.v1"
            or intersection.get("source", {}).get("source_model", {}).get("revision")
            != MYOSIM_REVISION
            or intersection.get("qualification", {}).get(
                "source_myo_sim_patella_femur_surfaces_range_valid_intersections_absent"
            ) is not False):
        raise RuntimeError("current patellofemoral intersection receipt does not bind the known failure")

    observations_bytes = observations_path.read_bytes()
    try:
        observations = json.loads(gzip.decompress(observations_bytes))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot decode retained Open Knee observations: {error}") from error
    reference_raw_rows = _relative_reference_rows(
        observations, archive_audit["inventory"]["rigid_bodies"]
    )
    flexions = [float(row["knee_flexion_deg"]) for row in reference_raw_rows]
    flexion_increments = [flexions[index] - flexions[index - 1]
                          for index in range(1, len(flexions))]

    myosim_identity, myosim_rows = _myosim_rows(sources, np, mujoco)
    reference_baseline_deg = float(reference_raw_rows[0]["knee_flexion_deg"])
    reference_rows = _relative_reference_metrics(reference_raw_rows, reference_baseline_deg)
    endpoint_flexion_deg = max(flexions)
    target_angles = (*FLEXION_TARGETS_DEG[:2], endpoint_flexion_deg)
    summary = []
    for target in target_angles:
        summary.append({
            "flexion_deg": float(target),
            "myosim_patella_body_origin_delta_from_qpos0_mm": (
                _interpolate(
                    myosim_rows,
                    "patella_origin_displacement_from_baseline_mm", target,
                )
            ),
            "myosim_patella_orientation_change_from_qpos0_deg": (
                _interpolate(
                    myosim_rows,
                    "patella_orientation_change_from_baseline_deg", target,
                )
            ),
            "open_knee_patella_com_delta_from_first_observation_mm": _interpolate(
                reference_rows,
                "patella_origin_displacement_from_baseline_mm", target,
            ),
            "open_knee_patella_orientation_change_from_first_observation_deg": _interpolate(
                reference_rows,
                "patella_orientation_change_from_baseline_deg", target,
            ),
            "interpolation": "linear_between_adjacent_observed_scalar_samples_without_angle_extrapolation",
        })
    result = {
        "schema": SCHEMA,
        "status": "exploratory_only_no_qualification",
        "question": (
            "How do gross patellar orientation and relative-origin path magnitudes "
            "compare between the pinned MyoSim left knee and one retained Open Knee specimen?"
        ),
        "method": {
            "myosim": (
                "MyoSim full-body left knee; source joint equalities projected; "
                "only source-range-valid coordinates retained from a 0.5 degree sample grid "
                "covering 0 to 90 degrees"
            ),
            "open_knee": (
                "Archived FEBio 2.9.1 oks003 rigid observations; femur FMB=4, "
                "tibia TBB=3, patella PTB=1; all 140 accepted quasi-static states"
            ),
            "comparison": (
                "Patellar orientation excursion is the shortest relative-quaternion angle "
                "from each model's own starting state. Translation is the change in relative "
                "frame origin, expressed as a Euclidean magnitude from that state. MyoSim uses "
                "qpos0; Open Knee uses its first accepted observation."
            ),
            "baseline_states": {
                "myosim": {
                    "flexion_deg": 0.0,
                    "source_range_valid": True,
                    "state": "equality-projected qpos0",
                },
                "open_knee": {
                    "flexion_deg": reference_baseline_deg,
                    "step": reference_raw_rows[0]["step"],
                    "continuation_parameter": reference_raw_rows[0]["continuation_time"],
                    "state": "first accepted archived observation; not exact zero flexion",
                },
            },
        },
        "sources": {
            "myosim": {
                "repository": myosim_identity["source"].get("repository"),
                "revision": myosim_identity["source"].get("revision"),
                "archive_sha256": myosim_identity["source"].get("archive_sha256"),
                "mujoco_version": myosim_identity["mujoco_version"],
                "left_knee_qpos_index": myosim_identity["left_knee_qpos_index"],
                "left_knee_source_range_rad": myosim_identity["left_knee_source_range_rad"],
                "projected_sample_count": myosim_identity["sample_count"],
                "source_range_valid_sample_count": myosim_identity[
                    "source_range_valid_sample_count"
                ],
                "source_range_invalid_sample_count": myosim_identity[
                    "source_range_invalid_sample_count"
                ],
                "source_range_valid_start_deg": myosim_identity[
                    "source_range_valid_start_deg"
                ],
                "source_range_valid_end_deg": myosim_identity[
                    "source_range_valid_end_deg"
                ],
                "source_range_invalid_samples": myosim_identity[
                    "source_range_invalid_samples"
                ],
                "sample_step_deg": myosim_identity["sampling_step_deg"],
                "overlay_files": myosim_identity["source_files"],
            },
            "open_knee": {
                "source_doi": archive_info["manifest"]["source_doi"],
                "febio_version_from_log": archive_info["version"],
                "observations_complete": archive_info["observations_complete"],
                "accepted_increment_count": archive_info["accepted_increment_count"],
                "negative_jacobian_trial_diagnostics": archive_info[
                    "negative_jacobian_trial_diagnostics"
                ],
                "warning_blocks": archive_info["warning_blocks"],
                "local_solver_reproduction": archive_info["local_reproduction"],
                "original_binary_identity": archive_info["manifest"]["original_binary_identity"],
                "observations_compressed_sha256": _sha256(observations_path),
                "observations_uncompressed_sha256": hashlib.sha256(
                    gzip.decompress(observations_bytes)
                ).hexdigest(),
                "source_audit_sha256": _sha256(archive_audit_path),
            },
            "existing_patellofemoral_intersection_audit_sha256": _sha256(
                intersection_receipt_path
            ),
        },
        "results": {
            "open_knee_flexion_range_deg": [min(flexions), max(flexions)],
            "open_knee_flexion_order_diagnostic": {
                "accepted_state_order_preserved": True,
                "reverse_increment_count": sum(value < 0.0 for value in flexion_increments),
                "maximum_reverse_increment_deg": max(
                    (abs(value) for value in flexion_increments if value < 0.0),
                    default=0.0,
                ),
                "interpolation_order": "sorted_by_measured_flexion_within_observed_range",
            },
            "matched_flexion_comparison": summary,
            "myosim_source_range_valid_sample_count": myosim_identity[
                "source_range_valid_sample_count"
            ],
            "myosim_source_range_invalid_sample_count": myosim_identity[
                "source_range_invalid_sample_count"
            ],
            "myosim_source_range_invalid_flexion_ranges_deg": myosim_identity[
                "source_range_invalid_ranges"
            ],
            "myosim_valid_source_interval_motion": _valid_source_segment_summary(
                myosim_rows, myosim_identity["sampling_step_deg"]
            ),
            "myosim_full_0_to_90_continuous_path_available": False,
        },
        "finding": (
            "All three matched target poses pass MyoSim source joint limits. Their relative-origin "
            "displacement proxies are close near the archived endpoint, while MyoSim's patellar "
            "orientation excursion is larger. The 0.5 degree sweep also includes source-limit "
            "violations at intermediate angles, so it does not establish a continuous valid path. "
            "The translation origins differ (MyoSim body origin versus Open Knee rigid-body "
            "center of mass), and the specimens and mechanics differ, so these values cannot "
            "calibrate or select a MyoSim path. The source-mesh intersection failure remains."
        ),
        "limitations": {
            "different_specimens": True,
            "myosim_origin_is_body_frame_origin": True,
            "open_knee_translation_uses_rigid_body_center_of_mass": True,
            "open_knee_continuation_is_quasi_static_not_elapsed_time": True,
            "quadriceps_active_loading_in_reference": False,
            "original_febio_binary_identity_supplied": False,
            "local_febio_reproduction_completed": False,
            "clinical_validation": False,
            "anatomical_frame_registration_between_specimens": False,
            "source_path_correction_performed": False,
        },
        "qualification": {
            "clinical_anatomy_qualified": False,
            "loaded_tracking_qualified": False,
            "continuous_angle_interval_qualified": False,
            "source_mesh_intersection_gate_passed": False,
            "source_path_correction_selected": False,
            "whole_human_qualification": False,
        },
        "reproduction_command": [
            sys.executable, "-m", "numilab_human.patellofemoral_reference_path_comparison",
            "--sources", str(sources),
            "--observations", str(observations_path),
            "--archive-audit", str(archive_audit_path),
            "--intersection-receipt", str(intersection_receipt_path),
            "--output", "<new-receipt-path>",
        ],
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--archive-audit", type=Path, required=True)
    parser.add_argument("--intersection-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compare_patellofemoral_reference_path(
            sources=args.sources,
            observations_path=args.observations,
            archive_audit_path=args.archive_audit,
            intersection_receipt_path=args.intersection_receipt,
        )
        output = args.output.resolve()
        if output.exists():
            raise RuntimeError(f"refusing to overwrite existing comparison receipt: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"patellofemoral reference path comparison: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "output": str(output),
                      "comparison_count": len(result["results"]["matched_flexion_comparison"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
