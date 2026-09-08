#!/usr/bin/env python3
"""Independent algebraic audit of the offline source-stop certificate.

This does not integrate physics or qualify NumanX/Metal dynamics. The analytic
native fixtures separately verify the full mass equation and recruitment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "support-stance-20260908/source-manifest.json"
MANIFEST_SHA = "2cea81e64a3c8da63731ae67c40ddc39aa61d9c4d0436000e806d25005e5b6d8"
VECTOR_KEYS = (
    "acceleration", "limit_force", "equality_force", "muscle_force",
    "passive_force", "support_force", "gravity_target", "force_residual",
)


def fp32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def parse(log: str) -> tuple[list[float], dict, dict]:
    def unique(prefix: str) -> str:
        rows = [line[len(prefix):] for line in log.splitlines() if line.startswith(prefix)]
        if len(rows) != 1:
            raise ValueError(f"expected exactly one {prefix} record")
        return rows[0]
    q = json.loads(unique("compiled_equilibrium_q="))
    reactions = json.loads(unique("compiled_equilibrium_reactions="))
    summary = dict(token.split("=", 1) for token in unique(
        "numi_human_whole_body_support_wrench=").split() if "=" in token)
    return q, reactions, summary


def audit(q: list[float], r: dict, manifest: dict) -> dict:
    if r.get("schema") != "numi.human.offline-reactions.v1":
        raise ValueError("wrong offline reaction schema")
    nv = manifest["core_tree"]["nv"]
    nq = manifest["core_tree"]["nq"]
    for values, count in [(q, nq)] + [(r.get(key), nv) for key in VECTOR_KEYS]:
        if not isinstance(values, list) or len(values) != count or any(
            type(value) not in (float, int) or not math.isfinite(value) for value in values
        ):
            raise ValueError("invalid reaction vector dimensions or finite values")
    a, force = r["acceleration"], r["limit_force"]
    violations = []
    maximum_penetration = maximum_complementarity = maximum_inward = 0.0
    active = 0
    limited_dofs = set()
    for joint in manifest["core_tree"]["source_joint_map"]:
        v, qi = joint["core_v_index"], joint["core_q_index"]
        if not joint["source_limited"]:
            continue
        limited_dofs.add(v)
        lower, upper = map(fp32, joint["source_range"])
        gaps = (q[qi] - lower, upper - q[qi])
        penetration = max(0.0, -min(gaps))
        maximum_penetration = max(maximum_penetration, penetration)
        if penetration > 1e-7:
            violations.append(f"source range: {joint['source_name']}")
        for gap, direction in zip(gaps, (1.0, -1.0)):
            if gap <= 1e-7:
                maximum_inward = max(maximum_inward, -direction * a[v])
                if direction * a[v] < -1e-7:
                    violations.append(f"inward stop acceleration: {joint['source_name']}")
        if force[v] != 0.0:
            active += 1
            loaded_gap = gaps[0] if force[v] > 0.0 else gaps[1]
            if loaded_gap > 1e-7:
                violations.append(f"reaction away from source stop: {joint['source_name']}")
            product = abs(force[v] * a[v])
            maximum_complementarity = max(maximum_complementarity, product)
            if product > 1e-5:
                violations.append(f"physical complementarity: {joint['source_name']}")
    if any(force[v] != 0.0 for v in range(nv) if v not in limited_dofs):
        violations.append("reaction on an unlimited coordinate")

    maximum_force_sum = 0.0
    for v in range(nv):
        terms = [r[key][v] for key in (
            "muscle_force", "passive_force", "support_force", "limit_force", "equality_force")]
        terms += [-r["gravity_target"][v]]
        error = abs(math.fsum(terms) - r["force_residual"][v]) / (1.0 + sum(map(abs, terms)))
        maximum_force_sum = max(maximum_force_sum, error)
    if maximum_force_sum > 1e-11:
        violations.append("physical force sum")

    # Zero source velocity makes Jdot*v zero. Check every source polynomial's
    # position and acceleration tangent, including locked dependents. T'feq=0
    # checks ideal equality reactions without assuming their individual values.
    virtual_work = r["equality_force"].copy()
    maximum_position_error = maximum_acceleration_error = 0.0
    for row in manifest["joint_equalities"]["records"]:
        qi, vi = row["dependent_core_q"], row["dependent_core_v"]
        mq, mv = row["master_core_q"], row["master_core_v"]
        x = 0.0 if mq is None else q[mq] - fp32(row["master_reference"])
        coefficients = list(map(fp32, row["polycoef"]))
        position = fp32(row["dependent_reference"]) + sum(c * x ** i for i, c in enumerate(coefficients))
        derivative = sum(i * coefficients[i] * x ** (i - 1) for i in range(1, 5))
        tangent_acceleration = 0.0 if mv is None else derivative * a[mv]
        maximum_position_error = max(maximum_position_error, abs(q[qi] - position))
        maximum_acceleration_error = max(maximum_acceleration_error, abs(a[vi] - tangent_acceleration))
        if mv is not None:
            virtual_work[mv] += derivative * r["equality_force"][vi]
        virtual_work[vi] = 0.0
    maximum_virtual_work = max(map(abs, virtual_work))
    if maximum_position_error > 1e-8 or maximum_acceleration_error > 1e-8 or maximum_virtual_work > 1e-8:
        violations.append("source equality tangent or virtual work")
    return {
        "status": "offline_reactions_passed" if not violations else "failed",
        "violations": violations,
        "limited_source_coordinates": len(limited_dofs),
        "loaded_source_stops": active,
        "maximum_source_range_violation": maximum_penetration,
        "maximum_inward_stop_acceleration": maximum_inward,
        "maximum_force_times_acceleration": maximum_complementarity,
        "maximum_relative_force_sum_error": maximum_force_sum,
        "maximum_equality_position_error": maximum_position_error,
        "maximum_equality_acceleration_error": maximum_acceleration_error,
        "maximum_equality_virtual_work_coefficient": maximum_virtual_work,
        "dynamic_limits_qualified": False,
        "sustained_standing_qualified": False,
        "walking_qualified": False,
        "full_release_qualified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_manifest = MANIFEST.read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != MANIFEST_SHA:
        raise ValueError("pinned source manifest changed")
    raw_log = args.log.read_bytes()
    q, reactions, summary = parse(raw_log.decode())
    result = audit(q, reactions, json.loads(raw_manifest))
    result.update({
        "schema": "numi.human.offline-reaction-audit.v1",
        "log_sha256": hashlib.sha256(raw_log).hexdigest(),
        "source_manifest_sha256": MANIFEST_SHA,
        "native_normalized_residual_rms": float(summary["internal_normalized_residual_rms"]),
        "native_balanced": summary["internal_balanced"] == "true",
    })
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))
    return 0 if result["status"] == "offline_reactions_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
