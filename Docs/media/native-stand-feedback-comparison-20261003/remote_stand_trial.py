#!/usr/bin/env python3
"""Run one registered, bounded Human standing arm on the Numi M4 Pro owner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IDENTITY_PATH = ROOT / "runtime-identity.json"
CALIBRATION_PATH = ROOT / "parser-calibration.json"
PROGRESS_PREFIX = "human_standing_progress="
TOKEN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=((?:\[[^\]]*\])|[^\s]+)")


def parse_progress_line(line: str) -> dict:
    if not line.startswith(PROGRESS_PREFIX):
        raise ValueError("not a Human standing progress line")
    values: dict[str, object] = {}
    for key, raw in TOKEN.findall(line[len(PROGRESS_PREFIX):]):
        if key == "root_xyz_m" or key == "root_orientation_xyzw":
            values[key] = json.loads(raw)
        elif key == "contact_count" or key == "step":
            values[key] = int(raw)
        else:
            try:
                values[key] = float(raw)
            except ValueError:
                values[key] = raw
    if "step" not in values or "root_xyz_m" not in values:
        raise ValueError("progress row lacks step or root position")
    return values


def run_parser_calibration() -> None:
    calibration = json.loads(CALIBRATION_PATH.read_text())
    observed = parse_progress_line(calibration["input_line"])
    expected = calibration["expected"]
    for key, value in expected.items():
        actual = observed[key]
        if isinstance(value, list):
            if any(abs(float(a) - float(b)) > 1.0e-13 for a, b in zip(actual, value, strict=True)):
                raise AssertionError(f"parser calibration mismatch for {key}")
        elif isinstance(value, float):
            if abs(float(actual) - value) > 1.0e-13:
                raise AssertionError(f"parser calibration mismatch for {key}")
        elif actual != value:
            raise AssertionError(f"parser calibration mismatch for {key}")


def remote_script(identity: dict, arm: str, run_id: str) -> tuple[str, Path]:
    remote_root = "/Users/n/human-standing-20260922"
    build = "/Users/n/numi-human-standing-build-20260922"
    input_dir = f"{remote_root}/current-enthesis-input-20261002"
    output_dir = f"{remote_root}/science-stand-feedback-comparison-20261003-{run_id}"
    probe = f"{build}/bin/metalrobo_numilab_human_myosim_visual_probe"
    q = shlex.quote
    checks = []
    for item in identity["files"]:
        checks.append(
            f"actual=$(shasum -a 256 {q(item['remote_path'])} | awk '{{print $1}}'); "
            f"test \"$actual\" = {q(item['sha256'])} || "
            f"{{ echo 'runtime identity mismatch: {item['remote_path']}' >&2; exit 70; }}"
        )
    native_argv = [
        probe,
        f"{input_dir}/myosim-fullbody-core-reference.nhrigid",
        f"{input_dir}/myosim-fullbody-muscle-reference.nhmyo",
        f"{output_dir}/mechanics",
        "--mechanics-only",
        "--tendon-payload", f"{input_dir}/numi-human-tendon-attachments.nhtendon",
        "--support-contact-payload", f"{input_dir}/myosim-fullbody-support-contact.nhcnt",
        "--joint-equality-payload", f"{input_dir}/myosim-fullbody-joint-equalities.nheq",
        "--support-stance-dof", "2", "0.02",
        "--support-stance-dof", "108", "0.1",
        "--support-stance-dof", "109", "0.1",
        "--support-stance-dof", "110", "0.1",
        "--support-stance-dof", "122", "0.1",
        "--support-stance-dof", "123", "0.1",
        "--support-stance-dof", "124", "0.1",
        "--support-stance-contact", "2",
        "--support-stance-contact", "3",
        "--support-stance-contact", "4",
        "--support-stance-contact", "5",
        "--support-stance-contact", "6",
        "--support-stance-contact", "7",
        "--persistent-metal-stand",
        "--persistent-source-passive-joint-tissue",
        "--stand-contact-iterations", "64",
        "--muscle-step-count", "1600",
        "--muscle-step-seconds", "0.001",
        "--dimension", "512",
    ]
    if arm == "treatment":
        native_argv.extend(["--stand-muscle-path-feedback", "10", "1"])
    quoted_argv = " \\\n  ".join(q(arg) for arg in native_argv)
    script = "\n".join([
        "set -u",
        *checks,
        "cpu=$(sysctl -n machdep.cpu.brand_string)",
        "test \"$cpu\" = 'Apple M4 Pro' || { echo 'unexpected native execution CPU' >&2; exit 71; }",
        "printf 'REMOTE_DEVICE_CPU=%s\\n' \"$cpu\" >&2",
        f"out={q(output_dir)}",
        "if [ -e \"$out\" ]; then echo 'remote output already exists' >&2; exit 73; fi",
        "mkdir -p \"$out\"",
        "export NUMI_HUMAN_EXECUTION_STAGES=1",
        f"/usr/bin/time -l {quoted_argv} > \"$out/stdout.txt\" 2> \"$out/stderr.txt\"",
        "status=$?",
        "cat \"$out/stdout.txt\"",
        "cat \"$out/stderr.txt\" >&2",
        "printf '\\nREMOTE_EXIT_CODE=%s\\n' \"$status\" >&2",
        "exit \"$status\"",
    ])
    return script, Path(output_dir)


def execute_arm(arm: str, run_dir: Path, trial_id: str) -> dict:
    identity = json.loads(IDENTITY_PATH.read_text())
    remote_command, remote_output = remote_script(identity, arm, trial_id)
    result = subprocess.run(
        ["/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
         "macmini", "/bin/zsh", "-lc", remote_command],
        capture_output=True, text=True, timeout=1100, check=False,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "native-stdout.txt").write_text(result.stdout)
    (run_dir / "native-stderr.txt").write_text(result.stderr)
    (run_dir / "runtime-identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"native owner exited {result.returncode}; raw output retained in {run_dir}"
        )

    rows = [parse_progress_line(line) for line in result.stdout.splitlines()
            if line.startswith(PROGRESS_PREFIX)]
    if not rows:
        raise RuntimeError("native owner returned no accepted progress rows")
    first, final = rows[0], rows[-1]
    if int(final["step"]) != 1600 or abs(float(final["simulated_seconds"]) - 1.6) > 1.0e-12:
        raise RuntimeError("native owner did not reach the preregistered 1.6 s endpoint")
    initial_xyz = [float(x) for x in first["root_xyz_m"]]
    final_xyz = [float(x) for x in final["root_xyz_m"]]
    input_hashes = {Path(item["remote_path"]).name: item["sha256"]
                    for item in identity["files"]
                    if Path(item["remote_path"]).name != "numilab_human_myosim_visual_probe.mm"
                    and Path(item["remote_path"]).name not in {
                        "compile_commands.json", "CMakeLists.txt", "REVISION.in"}}
    identity_sha = hashlib.sha256(
        (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    first_observation_sha = hashlib.sha256(
        (json.dumps(first, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    output = {
        "schema": "numi.science.human-stand-controller-comparison.v1",
        "stance_id": "current-toe-enthesis-20261002",
        "arm": arm,
        "run_completed": True,
        "step_count": int(final["step"]),
        "steps": 1600,
        "timestep_seconds": 0.001,
        "contact_iterations": 64,
        "root_assistance_force_n": float(final["root_assistance_force_n"]),
        "root_assistance_torque_nm": float(final["root_assistance_torque_nm"]),
        "initial_root_xyz_m": initial_xyz,
        "initial_root_orientation_xyzw": first["root_orientation_xyzw"],
        "first_progress_observation_sha256": first_observation_sha,
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
        "muscle_feedback_max_excitation_delta": float(
            final["muscle_feedback_max_excitation_delta"]
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
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "treatment"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--trial-id")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_parser_calibration()
        print(json.dumps({"calibration_passed": True,
                          "schema": "numi.science.human-stand-progress-parser-calibration.v1"}))
        return 0
    if not args.arm or args.run is None or not args.trial_id:
        parser.error("--arm, --run, and --trial-id are required for an owner run")
    result = execute_arm(args.arm, args.run.resolve(), args.trial_id)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"human standing study adapter failed: {error}", file=sys.stderr)
        sys.exit(2)
