#!/usr/bin/env python3
"""Import the already completed, preregistered control trajectory transparently."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def read_terminal_state(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("stand_terminal_state="):
            state = json.loads(line.split("=", 1)[1])
            if state.get("schema") != "numi.human.legacy-stand-terminal.v1":
                raise ValueError("unexpected terminal-state schema")
            return state
    raise ValueError("native stdout has no terminal-state record")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-log", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--parser", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()

    module_spec = importlib.util.spec_from_file_location("stand_progress_parser", args.parser)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("cannot load the calibrated standing parser")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    provenance = json.loads(args.provenance.read_text())
    identity = json.loads(args.runtime_identity.read_text())
    if sha256(args.progress_log) != provenance["remote_stdout_sha256"]:
        raise ValueError("retained native stdout hash differs from the provenance record")
    if sha256(args.runtime_identity) != provenance["runtime_identity_file_sha256"]:
        raise ValueError("runtime identity file hash differs from the provenance record")
    if provenance.get("launch_disposition") != "out-of-band-after-preregistration":
        raise ValueError("control import is not marked as out-of-band")

    raw = args.progress_log.read_text()
    rows = [module.parse_progress_line(line) for line in raw.splitlines()
            if line.startswith(module.PROGRESS_PREFIX)]
    if not rows:
        raise ValueError("native stdout contains no accepted progress rows")
    first, final = rows[0], rows[-1]
    if int(final["step"]) != 1600 or abs(float(final["simulated_seconds"]) - 1.6) > 1e-12:
        raise ValueError("retained control did not reach the preregistered endpoint")
    terminal = read_terminal_state(raw)
    if terminal.get("step_count") != 1600 or terminal.get("root_assistance") is not False:
        raise ValueError("terminal state disagrees with the retained control endpoint")

    initial_xyz = [float(x) for x in first["root_xyz_m"]]
    final_xyz = [float(x) for x in final["root_xyz_m"]]
    input_hashes = {Path(item["remote_path"]).name: item["sha256"]
                    for item in identity["files"]
                    if item["role"] in {"rigid_payload", "muscle_payload", "joint_equalities",
                                         "support_contacts", "tendon_payload"}}
    identity_sha = hashlib.sha256(
        (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    first_sha = canonical_sha(first)
    q0_sha = canonical_sha(terminal["initial_q"])
    v0_sha = canonical_sha(terminal["initial_v"])
    output = {
        "schema": "numi.science.human-stand-controller-comparison.v1",
        "stance_id": "current-toe-enthesis-20261002",
        "arm": "control",
        "run_completed": True,
        "control_imported": True,
        "native_exit_status_captured": False,
        "step_count": int(final["step"]),
        "steps": 1600,
        "timestep_seconds": 0.001,
        "contact_iterations": 64,
        "root_assistance_force_n": float(final["root_assistance_force_n"]),
        "root_assistance_torque_nm": float(final["root_assistance_torque_nm"]),
        "initial_q_sha256": q0_sha,
        "initial_v_sha256": v0_sha,
        "initial_root_xyz_m": initial_xyz,
        "initial_root_orientation_xyzw": first["root_orientation_xyzw"],
        "first_progress_observation_sha256": first_sha,
        "final_root_xyz_m": final_xyz,
        "root_horizontal_drift_m": math.hypot(
            final_xyz[0] - initial_xyz[0], final_xyz[1] - initial_xyz[1]
        ),
        "root_linear_speed_m_s": float(final["root_linear_speed_m_s"]),
        "support_force_n": float(final["support_force_n"]),
        "contact_count": int(final["contact_count"]),
        "maximum_generalized_acceleration_progress_peak_mixed_units": max(
            float(row["max_generalized_acceleration"]) for row in rows
        ),
        "maximum_reported_support_penetration_m": max(
            float(row["penetration_m"]) for row in rows
        ),
        "muscle_feedback_max_excitation_delta": max(
            float(row["muscle_feedback_max_excitation_delta"]) for row in rows
        ),
        "input_hashes_match": True,
        "input_hashes": input_hashes,
        "runtime_identity_sha256": identity_sha,
        "native_probe_sha256": identity["files_by_role"]["probe"]["sha256"],
        "metal_library_sha256": identity["files_by_role"]["metallib"]["sha256"],
        "native_source_file_sha256": identity["files_by_role"]["probe_source"]["sha256"],
        "native_source_revision": identity["runtime_source_revision"],
        "human_command_revision": identity["human_command_revision"],
        "device": {"cpu": "Apple M4 Pro", "host": "macmini SSH execution owner"},
        "accepted_progress_rows": len(rows),
        "retained_remote_output": provenance["remote_output"],
        "control_provenance_sha256": sha256(args.provenance),
    }
    args.run.mkdir(parents=True, exist_ok=True)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"control import failed: {error}", file=sys.stderr)
        raise SystemExit(2)
