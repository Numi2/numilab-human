"""Retain failed subject-scaled native long-horizon execution attempts.

The bounded subject-scaled replay already proves that the patched NHRIGID2
input can be consumed for 64 steps.  Longer runs attempted with the same
payload and the current Mac mini build stopped in the native Metal operator
before publishing a result.  This compiler keeps that negative evidence
immutable and fail-closed, while binding short completed controls so a timeout
is not mistaken for an immediate state failure or a mechanics result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "Docs/media/native-subject-scaled-runtime-128-failure-20260915"
EVIDENCE_512 = ROOT / "Docs/media/native-subject-scaled-runtime-512-failure-20260915"
CONTROL = ROOT / "Docs/media/native-subject-scaled-runtime-1-20260915"
CONTROL_2 = ROOT / "Docs/media/native-subject-scaled-runtime-2-20260915"
CONTROL_8 = ROOT / "Docs/media/native-subject-scaled-runtime-8-20260915"
OUTPUT = ROOT / "Docs/media/native-subject-scaled-long-horizon-failure-20260915/receipt-v4.json"
SCHEMA = "HumanPack.native-subject-scaled-long-horizon-failure.v1"
SCALED_RIGID_SHA256 = "303722b1f50ba603d75c52796aad0072f8a33776d5c3a24557ad9b393db3efb4"
BINARY_SHA256 = "47b6426c66f81042e167ad3c17c46d3d620ed8e05fd377225a0b45739402c50b"
MUSCLE_SHA256 = "9a988f19a6fd8e533cd0f2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05"
TENDON_SHA256 = "a594194f510eb4aa990a8767f868f999a10b4fedb745c8665368a231ed39b555"
SUPPORT_SHA256 = "4d54f8155cd83baaee7af536099824ac0da61e5d5e77544b42c6e5ce1b48c907"
EQUALITY_SHA256 = "b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a"
SOURCE_HEAD = "86c24d8a024fbb0ea314a376f0f1e112d52b7e9e"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("subject-scaled long-horizon failure: " + message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular(path: Path, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    return path.read_bytes()


def _attempt(root: Path, *, name: str, step_count: int, validation_layer: bool,
             observed_seconds: float) -> dict[str, Any]:
    stdout = root / "native.stdout.txt"
    stderr = root / "native.stderr.txt"
    sample = root / "metal-wait.sample.txt"
    terminated_at = root / "terminated-at-unix.txt"
    stdout_raw = _regular(stdout, f"{name} stdout")
    stderr_raw = _regular(stderr, f"{name} stderr")
    sample_raw = _regular(sample, f"{name} sample")
    terminated_raw = _regular(terminated_at, f"{name} termination timestamp")
    _require(terminated_raw.decode("utf-8").strip().isdigit(),
             f"{name} termination timestamp is not an integer")
    stdout_text = stdout_raw.decode("utf-8")
    sample_text = sample_raw.decode("utf-8")
    _require("tendon_payload=NHTENDON3" in stdout_text,
             f"{name} did not admit the tendon payload")
    _require("joint_equality_payload=NHEQ1" in stdout_text,
             f"{name} did not admit the equality payload")
    _require("compiled_support_geometry=admissible" in stdout_text,
             f"{name} did not admit support geometry")
    _require("myosim_articulated_marker_visual=ok" not in stdout_text,
             f"{name} unexpectedly contains a completed native result")
    _require("MetalArticulatedOperatorSubmission::wait" in sample_text and
             "waitUntilCompleted" in sample_text,
             f"{name} sample does not identify the Metal wait blocker")
    return {
        "name": name,
        "timestep_seconds": 1.25e-5,
        "step_count_requested": step_count,
        "completed": False,
        "termination": "agent_terminated_after_metal_command_buffer_wait",
        "validation_layer": validation_layer,
        "observed_seconds_before_termination": observed_seconds,
        "native_capture": {
            "stdout": {"path": str(stdout.relative_to(ROOT)), "bytes": len(stdout_raw),
                        "sha256": _sha(stdout)},
            "stderr": {"path": str(stderr.relative_to(ROOT)), "bytes": len(stderr_raw),
                        "sha256": _sha(stderr)},
            "sample": {"path": str(sample.relative_to(ROOT)), "bytes": len(sample_raw),
                       "sha256": _sha(sample)},
            "terminated_at_unix": {
                "path": str(terminated_at.relative_to(ROOT)), "bytes": len(terminated_raw),
                "sha256": _sha(terminated_at),
            },
        },
        "observed": {
            "payload_admission_reached": True,
            "completed_result_line": False,
            "metal_submission_wait_observed": True,
            "stderr_only_validation_banner": validation_layer and len(stderr_raw) > 0,
        },
    }


def _control(root: Path, *, expected_steps: int, label: str) -> dict[str, Any]:
    stdout = root / "native.stdout.txt"
    stderr = root / "native.stderr.txt"
    stdout_raw = _regular(stdout, f"{label} control stdout")
    stderr_raw = _regular(stderr, f"{label} control stderr")
    _require(not stderr_raw, f"{label} control emitted stderr")
    text = stdout_raw.decode("utf-8")
    line = next((row for row in text.splitlines()
                 if row.startswith("myosim_articulated_marker_visual=")), None)
    _require(line is not None and "myosim_articulated_marker_visual=ok" in line,
             f"{label} control did not complete")
    fields = {}
    for token in shlex.split(line):
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value.strip('"')
    _require(fields.get("metal_pose_device") == "Apple M4 Pro",
             f"{label} control device changed")
    _require(fields.get("muscle_step_seconds") == "1.25e-05" and
             fields.get("muscle_step_count") == str(expected_steps) and
             fields.get("persistent_completed_steps") == str(expected_steps),
             f"{label} control horizon changed")
    _require(fields.get("persistent_max_penetration_m") == "0" and
             fields.get("persistent_root_assistance") == "none" and
             fields.get("stand_deterministic_replay") == "bitwise",
             f"{label} control safety boundary changed")
    audit_line = next((row for row in text.splitlines()
                       if row.startswith("persistent_dynamic_force_audit=")), None)
    trace_line = next((row for row in text.splitlines()
                       if row.startswith("persistent_stand_trace=")), None)
    _require(audit_line is not None and trace_line is not None,
             f"{label} control audit or trace is missing")
    try:
        audit = json.loads(audit_line.split("=", 1)[1])
        trace = json.loads(trace_line.split("=", 1)[1])
    except json.JSONDecodeError as error:
        raise HumanImportError(f"{label} control audit/trace is not JSON") from error
    _require(audit.get("schema") == "numi.human.persistent-dynamic-force-audit.v1" and
             len(audit.get("rows", [])) == 128,
             f"{label} control audit coverage changed")
    _require(trace.get("endpoint_equivalent") == "bitwise" and
             trace.get("endpoint_max_q_delta") == 0 and
             trace.get("endpoint_max_v_delta") == 0 and
             len(trace.get("samples", [])) == expected_steps + 1,
             f"{label} control trace is not bitwise or complete")
    required_fields = (
        "persistent_max_acceleration", "persistent_max_penetration_m",
        "persistent_dynamic_force_residual_max_n", "compiled_stand_normalized_residual_rms",
        "compiled_stand_active_support_contacts", "compiled_stand_total_support_force_n",
        "muscle_force_metal_elapsed_ms", "muscle_force_metal_active_records",
    )
    _require(all(key in fields for key in required_fields),
             f"{label} control is missing a required metric")
    _require(int(fields["muscle_force_metal_active_records"]) == 416 * expected_steps,
             f"{label} control did not process all source muscle routes")
    return {
        "label": label,
        "timestep_seconds": 1.25e-5,
        "step_count": expected_steps,
        "persistent_completed_steps": int(fields["persistent_completed_steps"]),
        "persistent_max_acceleration_mps2": float(fields["persistent_max_acceleration"]),
        "persistent_max_penetration_m": float(fields["persistent_max_penetration_m"]),
        "persistent_dynamic_force_residual_max_n": float(fields["persistent_dynamic_force_residual_max_n"]),
        "compiled_stand_normalized_residual_rms": float(fields["compiled_stand_normalized_residual_rms"]),
        "compiled_stand_active_support_contacts": int(fields["compiled_stand_active_support_contacts"]),
        "compiled_stand_total_support_force_n": float(fields["compiled_stand_total_support_force_n"]),
        "dynamic_force_audit_max_abs_residual_n": float(audit["maximum_abs_residual_n"]),
        "dynamic_force_audit_rows": len(audit["rows"]),
        "trace_samples": len(trace["samples"]),
        "muscle_force_metal_elapsed_ms": float(fields["muscle_force_metal_elapsed_ms"]),
        "muscle_force_metal_active_records": int(fields["muscle_force_metal_active_records"]),
        "stand_deterministic_replay": "bitwise",
        "native_capture": {
            "stdout": {"path": str(stdout.relative_to(ROOT)), "bytes": len(stdout_raw),
                        "sha256": _sha(stdout)},
            "stderr": {"path": str(stderr.relative_to(ROOT)), "bytes": len(stderr_raw),
                        "sha256": _sha(stderr)},
        },
    }


def compile_receipt() -> dict[str, Any]:
    attempts = [
        _attempt(EVIDENCE, name="128_steps_without_validation_layer", step_count=128,
                 validation_layer=False, observed_seconds=105.196),
        _attempt(EVIDENCE_512, name="512_steps_with_validation_layer", step_count=512,
                 validation_layer=True, observed_seconds=199.037),
    ]
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.subject-scaled-long-horizon-failure.1",
        "status": "failed",
        "subject": "one adult male source package",
        "source": {
            "checkout": "/Users/n/MetalRobo-human",
            "head": SOURCE_HEAD,
            "worktree_dirty": True,
            "binary": "metalrobo_numilab_human_myosim_visual_probe",
            "binary_sha256": BINARY_SHA256,
            "device": "Mac mini M4 Pro",
        },
        "inputs": {
            "scaled_rigid_sha256": SCALED_RIGID_SHA256,
            "muscle_sha256": MUSCLE_SHA256,
            "tendon_sha256": TENDON_SHA256,
            "support_contact_sha256": SUPPORT_SHA256,
            "joint_equalities_sha256": EQUALITY_SHA256,
        },
        "command": {
            "timestep_seconds": 1.25e-5,
            "muscle_activation": 0.8,
            "persistent_metal_stand": True,
            "persistent_source_passive_joint_tissue": True,
            "persistent_stand_trace": True,
            "stand_contact_iterations": 64,
            "stand_deterministic_replay": True,
            "root_assistance": "none",
        },
        "attempts": attempts,
        "one_step_control": _control(CONTROL, expected_steps=1, label="one-step"),
        "two_step_control": _control(CONTROL_2, expected_steps=2, label="two-step"),
        "eight_step_control": _control(CONTROL_8, expected_steps=8, label="eight-step"),
        "qualification": {
            "scaled_input_admitted": True,
            "long_horizon_completed": False,
            "full_generalized_force_convergence": False,
            "sustained_standing": False,
            "recovery": False,
            "walking": False,
            "subject_calibration": False,
        },
        "blocker": {
            "id": "subject_scaled_long_horizon_native_execution",
            "status": "open",
            "reason": (
                "The 128-step and 512-step 12.5 us attempts admitted the scaled "
                "payloads but were agent-terminated after 105.196 s and 199.037 s "
                "in MetalArticulatedOperatorSubmission::wait before publishing a "
                "native result. Completed 1-, 2-, and 8-step controls show that the "
                "retained blocker is long-horizon native throughput, not an observed "
                "first-step or two-step state break."
            ),
        },
        "boundary": (
            "This receipt retains two failed subject-scaled long-horizon attempts and "
            "completed 1-, 2-, and 8-step controls. The 64-step subject-scaled replay "
            "remains the only released subject-scaled qualification. A Metal command-buffer wait is execution evidence, not a physics, "
            "standing, recovery, walking, activation, anatomy, material, or subject "
            "calibration qualification."
        ),
    }


def immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    receipt = compile_receipt()
    digest = immutable_write(arguments.output.resolve(), receipt)
    print(json.dumps({"schema": SCHEMA, "status": receipt["status"],
                      "output": str(arguments.output.resolve()), "sha256": digest},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        raise SystemExit(run(parser.parse_args()))
    except (HumanImportError, OSError, TypeError, ValueError) as error:
        parser.exit(2, f"subject-scaled long-horizon failure: {error}\n")
