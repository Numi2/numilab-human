#!/usr/bin/env python3
"""Recompute descriptive row-wise metrics from the retained standing logs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_parser():
    path = ROOT / "remote_stand_trial.py"
    spec = importlib.util.spec_from_file_location("stand_progress_parser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load calibrated progress parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def terminal(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("stand_terminal_state="):
            return json.loads(line.split("=", 1)[1])
    raise ValueError("missing terminal-state record")


def arm_summary(path: Path, parser, arm_json: dict, *, threshold_m: float = 0.01) -> dict:
    lines = path.read_text().splitlines()
    rows = [parser.parse_progress_line(line) for line in lines
            if line.startswith(parser.PROGRESS_PREFIX)]
    if len(rows) != 200 or int(rows[-1]["step"]) != 1600:
        raise ValueError(f"{path}: expected 200 rows ending at step 1600")
    first, final = rows[0], rows[-1]

    def drift(row):
        return math.hypot(
            float(row["root_xyz_m"][0]) - float(first["root_xyz_m"][0]),
            float(row["root_xyz_m"][1]) - float(first["root_xyz_m"][1]),
        )

    state = terminal("\n".join(lines))
    q_sha, v_sha = canonical_sha(state["initial_q"]), canonical_sha(state["initial_v"])
    if q_sha != arm_json["initial_q_sha256"] or v_sha != arm_json["initial_v_sha256"]:
        raise ValueError(f"{path}: terminal-state q/v hashes differ from trial output")
    if abs(drift(final) - arm_json["root_horizontal_drift_m"]) > 1e-12:
        raise ValueError(f"{path}: recomputed endpoint drift differs from recorded observable")

    crossing = next((row for row in rows if drift(row) > threshold_m), None)
    return {
        "stdout_sha256": sha256(path),
        "accepted_progress_rows": len(rows),
        "step_count": int(final["step"]),
        "duration_s": float(final["simulated_seconds"]),
        "root_start_xyz_m": [float(value) for value in first["root_xyz_m"]],
        "root_end_xyz_m": [float(value) for value in final["root_xyz_m"]],
        "horizontal_drift_m": drift(final),
        "first_drift_gt_10mm": None if crossing is None else {
            "step": int(crossing["step"]),
            "simulated_seconds": float(crossing["simulated_seconds"]),
            "drift_m": drift(crossing),
            "feedback_max_excitation_delta": float(crossing["muscle_feedback_max_excitation_delta"]),
        },
        "max_root_speed_m_s": max(float(row["root_linear_speed_m_s"]) for row in rows),
        "final_root_speed_m_s": float(final["root_linear_speed_m_s"]),
        "max_progress_acceleration_mixed_units": max(
            float(row["max_generalized_acceleration"]) for row in rows
        ),
        "support_force_min_n": min(float(row["support_force_n"]) for row in rows),
        "support_force_max_n": max(float(row["support_force_n"]) for row in rows),
        "contact_count_min": min(int(row["contact_count"]) for row in rows),
        "contact_count_max": max(int(row["contact_count"]) for row in rows),
        "max_penetration_m": max(float(row["penetration_m"]) for row in rows),
        "max_feedback_excitation_delta": max(
            float(row["muscle_feedback_max_excitation_delta"]) for row in rows
        ),
        "root_assistance_force_n": float(final["root_assistance_force_n"]),
        "root_assistance_torque_nm": float(final["root_assistance_torque_nm"]),
        "initial_q_sha256": q_sha,
        "initial_v_sha256": v_sha,
    }


def main() -> int:
    parser = load_parser()
    control_path = ROOT / "out-of-band-control/native-stdout.txt"
    treatment_path = ROOT / "study-followup-v3/trials/source-stance-feedback-on-clean-wrapper/output/native-stdout.txt"
    failed_treatment_path = ROOT / "out-of-band-treatment-wrapper-failure/remote-native-stdout.txt"
    control_output = json.loads((ROOT / "study-followup-v3/trials/source-stance-feedback-off-imported/output/stdout.json").read_text())
    treatment_output = json.loads((ROOT / "study-followup-v3/trials/source-stance-feedback-on-clean-wrapper/output/stdout.json").read_text())
    control = arm_summary(control_path, parser, control_output)
    treatment = arm_summary(treatment_path, parser, treatment_output)

    failed_lines = [line for line in failed_treatment_path.read_text().splitlines()
                    if line.startswith(parser.PROGRESS_PREFIX)]
    treatment_lines = [line for line in treatment_path.read_text().splitlines()
                       if line.startswith(parser.PROGRESS_PREFIX)]
    analysis = json.loads((ROOT / "study-followup-v3/analysis.json").read_text())["payload"]
    registration = json.loads((ROOT / "study-followup-v3/registration.json").read_text())
    plan = registration["payload"]["plan"]
    difference = treatment["horizontal_drift_m"] - control["horizontal_drift_m"]
    if not (control["initial_q_sha256"] == treatment["initial_q_sha256"] and
            control["initial_v_sha256"] == treatment["initial_v_sha256"]):
        raise ValueError("paired arms do not share exact initial q/v state hashes")
    if abs(difference - analysis["mean_difference"]) > 1e-12:
        raise ValueError("row-wise recomputation differs from registered analysis")
    if analysis["verdict"] != "contradicted":
        raise ValueError("registered analysis verdict differs from expected fixed prediction")

    result = {
        "schema": "numi.human.stand-feedback-rowwise-comparison.v1",
        "registration_sha256": registration["sha256"],
        "analysis_sha256": sha256(ROOT / "study-followup-v3/analysis.json"),
        "model_prediction_m": [plan["prediction"]["minimum"], plan["prediction"]["maximum"]],
        "control_native_exit_status_captured": False,
        "treatment_native_exit_code": treatment_output["native_exit_code"],
        "initial_q_matches": True,
        "initial_v_matches": True,
        "failed_wrapper_attempt_progress_matches_clean_run": failed_lines == treatment_lines,
        "control": control,
        "treatment": treatment,
        "treatment_minus_control_m": difference,
        "verdict": analysis["verdict"],
        "interpretation": "Single source-specific simulation pair. The imported control lacks its native exit status; do not use as standing qualification or population inference.",
    }
    output = ROOT / "trajectory-comparison.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
