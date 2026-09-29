"""Source-oracle and full-driver-domain diagnostics for joint equalities.

These checks do not rewrite source ranges, equality laws or native mechanics.
Finite polynomial witnesses can prove a conflict. Numerically found stationary
points are not an interval-arithmetic proof of global constraint feasibility.
"""
from __future__ import annotations

import math
from typing import Any

from . import model as human


def source_equality_projection_oracle(
    model: Any, data: Any, mujoco: Any, tolerance: float,
) -> dict[str, Any]:
    """Read MuJoCo's constraint residuals after the owning mj_forward call."""
    expected = {i for i in range(model.neq) if bool(model.eq_active0[i])
                and int(model.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT)}
    rows = []
    for row in range(data.nefc):
        if int(data.efc_type[row]) != int(mujoco.mjtConstraint.mjCNSTR_EQUALITY):
            continue
        index = int(data.efc_id[row])
        if index not in expected:
            raise RuntimeError("pose audit source oracle encountered an unexpected equality")
        residual = float(data.efc_pos[row])
        if not math.isfinite(residual):
            raise RuntimeError("pose audit source oracle encountered a nonfinite equality residual")
        dependent = int(model.eq_obj1id[index])
        driver = int(model.eq_obj2id[index])
        rows.append({
            "source_equality_id": index,
            "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, index),
            "dependent_q_index": int(model.jnt_qposadr[dependent]),
            "driver_q_index": int(model.jnt_qposadr[driver]) if driver >= 0 else None,
            "residual": residual,
            "unit": "m" if int(model.jnt_type[dependent]) == int(mujoco.mjtJoint.mjJNT_SLIDE) else "rad",
            "passed": abs(residual) <= tolerance,
        })
    ids = [row["source_equality_id"] for row in rows]
    covered = len(ids) == len(expected) and set(ids) == expected
    return {
        "method": "independent_MuJoCo_mj_forward_efc_pos_not_importer_polynomial_recalculation",
        "expected_active_joint_equalities": len(expected),
        "measured_joint_equalities": len(rows),
        "complete_unique_coverage": covered,
        "maximum_absolute_residual": max((abs(row["residual"]) for row in rows), default=0.0),
        "maximum_allowed_residual": tolerance,
        "tolerance_basis": "existing_projected_joint_range_coordinate_arithmetic_admission",
        "rows": rows,
        "passed": covered and all(row["passed"] for row in rows),
        "boundary": "Kinematic projection at this pose; not soft-constraint dynamics or loaded qualification.",
    }


def polynomial_driver_witnesses(
    coefficients: list[float], domain: list[float], driver_reference: float,
    dependent_reference: float, np: Any,
) -> list[dict[str, Any]]:
    """Endpoints and numerical stationary points of the source polynomial."""
    if (
        not coefficients or len(coefficients) > 5 or len(domain) != 2
        or not all(math.isfinite(x) for x in [*coefficients, *domain, driver_reference, dependent_reference])
        or domain[0] > domain[1]
    ):
        raise RuntimeError("joint constraint domain has invalid polynomial or driver bounds")
    lower, upper = domain
    points = {lower: "driver_lower_endpoint", upper: "driver_upper_endpoint"}
    derivative = [degree * coefficients[degree] for degree in range(1, len(coefficients))]
    while derivative and derivative[-1] == 0.0:
        derivative.pop()
    if len(derivative) > 1:
        for root in np.polynomial.polynomial.polyroots(derivative):
            real, imaginary = float(root.real), float(root.imag)
            if not math.isfinite(real) or not math.isfinite(imaginary):
                raise RuntimeError("joint constraint stationary-point solve produced a nonfinite root")
            if abs(imaginary) > 1.0e-8 * max(1.0, abs(real)):
                continue
            driver = real + driver_reference
            if lower < driver < upper:
                points[driver] = "numerical_polynomial_stationary_point"
    witnesses = []
    for driver, kind in sorted(points.items()):
        value = dependent_reference + float(np.polynomial.polynomial.polyval(
            driver - driver_reference, coefficients,
        ))
        if not math.isfinite(value):
            raise RuntimeError("joint constraint polynomial produced a nonfinite witness")
        witnesses.append({"driver_value": driver, "dependent_value": value, "kind": kind})
    return witnesses


def joint_equality_driver_domain_audit(
    exported: dict[str, Any], joined_joints: list[dict[str, Any]],
    np: Any, tolerance: float,
) -> dict[str, Any]:
    """Inspect all source equalities; retain unsupported domains explicitly."""
    source_joints = {row["id"]: row for row in exported["joints"]}
    native_joints = {row["source_joint_id"]: row for row in joined_joints}
    equalities = exported["joint_equalities"]
    dependent_ids = {row["dependent_joint"] for row in equalities}
    if len(dependent_ids) != len(equalities):
        raise RuntimeError("joint constraint domain audit has duplicated dependent owners")
    rows = []
    for equality in equalities:
        dependent = source_joints[equality["dependent_joint"]]
        joined = native_joints.get(dependent["id"])
        if joined is None:
            raise RuntimeError("joint constraint domain audit has no native dependent owner")
        driver_id = equality["master_joint"]
        driver = source_joints.get(driver_id) if driver_id >= 0 else None
        if driver_id >= 0 and driver is None:
            raise RuntimeError("joint constraint domain audit has no source driver")
        row = {
            "source_equality_id": equality["id"], "name": equality["name"],
            "dependent_joint_name": dependent["name"],
            "dependent_q_index": dependent["qpos_address"],
            "driver_joint_name": driver["name"] if driver else None,
            "driver_q_index": driver["qpos_address"] if driver else None,
            "polycoef": equality["polycoef"],
            "dependent_reference": equality["dependent_reference"],
            "driver_reference": equality["master_reference"],
            "source_position_limit_enabled": bool(dependent["limited"]),
            "source_range": list(dependent["range"]) if dependent["limited"] else None,
            "native_position_limit_enabled": bool(joined["native_dof"]["flags"] & human._MR_DOF_POSITION_LIMIT),
            "compiler_limit_status": joined["core_limit_status"],
            "unit": "m" if dependent["type"] == 2 else "rad",
        }
        if driver_id in dependent_ids or (driver is not None and not driver["limited"]):
            row.update(status="unverified_driver_domain", reason=(
                "driver_is_itself_equality_dependent_requires_composition" if driver_id in dependent_ids
                else "source_driver_has_no_finite_enforced_range"))
            rows.append(row)
            continue
        domain = list(driver["range"]) if driver is not None else [0.0, 0.0]
        witnesses = polynomial_driver_witnesses(equality["polycoef"], domain,
            equality["master_reference"], equality["dependent_reference"], np)
        native_range = joined["native_dof"]["position_range"] if row["native_position_limit_enabled"] else None
        for witness in witnesses:
            value = witness["dependent_value"]
            consumed = float(np.float32(value))
            source_range = row["source_range"]
            source_violation = max(0.0, source_range[0] - value, value - source_range[1]) if source_range else 0.0
            native_violation = max(0.0, native_range[0] - consumed, consumed - native_range[1]) if native_range else 0.0
            witness.update(projected_fp32_value=consumed,
                source_range_violation=source_violation, native_range_violation=native_violation,
                source_range_conflict=source_violation > tolerance,
                native_range_conflict=native_violation > tolerance)
        source_conflict = any(w["source_range_conflict"] for w in witnesses)
        native_conflict = any(w["native_range_conflict"] for w in witnesses)
        row.update(driver_domain=domain, witnesses=witnesses, native_position_range=native_range,
            dependent_witness_extrema=[min(w["dependent_value"] for w in witnesses), max(w["dependent_value"] for w in witnesses)],
            source_range_conflict=source_conflict, native_range_conflict=native_conflict,
            status="witnessed_projection_range_conflict" if source_conflict or native_conflict else "no_conflict_at_polynomial_witnesses")
        rows.append(row)
    source_conflicts = sum(bool(row.get("source_range_conflict")) for row in rows)
    native_conflicts = sum(bool(row.get("native_range_conflict")) for row in rows)
    unverified = sum(row["status"] == "unverified_driver_domain" for row in rows)
    return {
        "schema": "numi.human.source-joint-equality-driver-domain-audit.v1",
        "method": "FP64_source_polynomial_endpoints_and_numerical_stationary_witnesses",
        "expected_source_equalities": len(equalities), "evaluated_equalities": len(rows),
        "source_range_conflict_count": source_conflicts, "native_range_conflict_count": native_conflicts,
        "unverified_driver_domain_count": unverified,
        "maximum_allowed_range_violation": tolerance,
        "tolerance_basis": "existing_projected_joint_range_coordinate_arithmetic_admission",
        "status": "witnessed_projection_range_conflicts" if source_conflicts or native_conflicts else
            "unverified_driver_domains" if unverified else "no_conflict_at_polynomial_witnesses",
        "rows": rows,
        "boundary": (
            "All declared independent driver ranges are inspected without changing any source law or gate. "
            "A witness proves incompatibility of exact equality projection with the declared bound at that driver value. "
            "MuJoCo source equality/limit dynamics are compliant; these witnesses do not prove those soft dynamics infeasible. "
            "FP32 native checks round source-projected coordinates against consumed range bytes, not native polynomial execution. "
            "Numerical stationary points do not constitute interval-arithmetic proof or clinical joint-range calibration."
        ),
    }
