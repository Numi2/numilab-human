#!/usr/bin/env python3
"""Run a standing arm through a stdin-fed zsh wrapper with preserved native status."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "remote_stand_trial.py"


def load_base():
    spec = importlib.util.spec_from_file_location("remote_stand_trial_base", BASE)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the bound standing parser and argv builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def fixed_remote_script(base, identity: dict, arm: str, trial_id: str) -> tuple[str, Path]:
    script, remote_output = base.remote_script(identity, arm, trial_id)
    if "status=$?" not in script or '"$status"' not in script:
        raise ValueError("expected legacy zsh status assignment was not found")
    # zsh reserves $status as a read-only special parameter; keep the native exit code in rc.
    return script.replace("status=$?", "native_rc=$?").replace('"$status"', '"$native_rc"'), remote_output


def run_remote(script: str, *, syntax_only: bool, timeout: int):
    zsh_args = ["/bin/zsh", "-f", "-n", "-s"] if syntax_only else ["/bin/zsh", "-f", "-s"]
    return subprocess.run(
        ["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "macmini", *zsh_args],
        input=script, capture_output=True, text=True, timeout=timeout, check=False,
    )


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
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--shell-syntax-only", action="store_true")
    args = parser.parse_args()
    base = load_base()
    identity = json.loads((ROOT / "runtime-identity.json").read_text())
    remote_command, remote_output = fixed_remote_script(base, identity, args.arm, args.trial_id)

    if args.shell_syntax_only:
        result = run_remote(remote_command, syntax_only=True, timeout=30)
        if result.returncode:
            print(result.stderr, file=sys.stderr, end="")
            return result.returncode
        print(json.dumps({"shell_syntax_passed": True,
                          "remote_command_sha256": hashlib.sha256(remote_command.encode()).hexdigest(),
                          "remote_output_path": str(remote_output)}, sort_keys=True))
        return 0

    if args.run is None:
        parser.error("--run is required unless --shell-syntax-only is set")
    run_dir = args.run.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result = run_remote(remote_command, syntax_only=False, timeout=1100)
    (run_dir / "native-stdout.txt").write_text(result.stdout)
    (run_dir / "native-stderr.txt").write_text(result.stderr)
    (run_dir / "runtime-identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"native owner or remote wrapper exited {result.returncode}; raw streams retained in {run_dir}"
        )
    if f"REMOTE_EXIT_CODE=0" not in result.stderr:
        raise RuntimeError("remote script did not return its captured native exit code")

    rows = [base.parse_progress_line(line) for line in result.stdout.splitlines()
            if line.startswith(base.PROGRESS_PREFIX)]
    if not rows:
        raise RuntimeError("native owner returned no accepted progress rows")
    first, final = rows[0], rows[-1]
    if int(final["step"]) != 1600 or abs(float(final["simulated_seconds"]) - 1.6) > 1e-12:
        raise RuntimeError("native owner did not reach the preregistered endpoint")
    terminal = read_terminal_state(result.stdout)
    if terminal.get("step_count") != 1600 or terminal.get("root_assistance") is not False:
        raise RuntimeError("native terminal state disagrees with the registered horizon")

    initial_xyz = [float(x) for x in first["root_xyz_m"]]
    final_xyz = [float(x) for x in final["root_xyz_m"]]
    input_hashes = {Path(item["remote_path"]).name: item["sha256"]
                    for item in identity["files"]
                    if item["role"] in {"rigid_payload", "muscle_payload", "joint_equalities",
                                         "support_contacts", "tendon_payload"}}
    identity_sha = hashlib.sha256(
        (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    output = {
        "schema": "numi.science.human-stand-controller-comparison.v1",
        "stance_id": "current-toe-enthesis-20261002",
        "arm": args.arm,
        "run_completed": True,
        "native_exit_code": result.returncode,
        "steps": 1600,
        "step_count": int(final["step"]),
        "timestep_seconds": 0.001,
        "contact_iterations": 64,
        "root_assistance_force_n": float(final["root_assistance_force_n"]),
        "root_assistance_torque_nm": float(final["root_assistance_torque_nm"]),
        "initial_q_sha256": canonical_sha(terminal["initial_q"]),
        "initial_v_sha256": canonical_sha(terminal["initial_v"]),
        "initial_root_xyz_m": initial_xyz,
        "initial_root_orientation_xyzw": first["root_orientation_xyzw"],
        "first_progress_observation_sha256": canonical_sha(first),
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
        "retained_remote_output": str(remote_output),
        "remote_command_sha256": hashlib.sha256(remote_command.encode()).hexdigest(),
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"corrected native owner adapter failed: {error}", file=sys.stderr)
        raise SystemExit(2)
