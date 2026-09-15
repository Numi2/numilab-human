"""Audit a native whole-body activation/recruitment sweep.

The native support-wrench probe can close the floating six-coordinate wrench
while leaving articulated coordinates badly unbalanced.  This reader keeps the
two results separate and records the ranked internal coordinates emitted by the
probe.  A global activation sweep is therefore useful diagnostic evidence; it
is never promoted to measured activation calibration or standing qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any

from .model import ImportError
from .physiology import canonical


SCHEMA = "HumanPack.native-activation-sweep-audit.v1"
RESULT_MARKER = "numi_human_whole_body_support_wrench=ok"
ROOT_FORCE_TOLERANCE_N = 1.0e-3
WEIGHT_RELATIVE_TOLERANCE = 1.0e-5
INTERNAL_RESIDUAL_TOLERANCE = 1.0e-3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError("native activation sweep audit: " + message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(fields: dict[str, str], name: str) -> float:
    value = fields.get(name)
    _require(value is not None, f"native result is missing {name}")
    try:
        result = float(value)
    except ValueError as error:
        raise ImportError(f"native result field {name} is not numeric") from error
    _require(math.isfinite(result), f"native result field {name} is not finite")
    return result


def _integer(fields: dict[str, str], name: str) -> int:
    value = _finite_number(fields, name)
    _require(value == int(value), f"native result field {name} is not integral")
    return int(value)


def _boolean(fields: dict[str, str], name: str) -> bool:
    value = fields.get(name)
    _require(value in {"true", "false"}, f"native result field {name} is not boolean")
    return value == "true"


def _fields(path: Path) -> dict[str, str]:
    _require(path.is_file() and not path.is_symlink(), "native stdout is not a regular file")
    lines = path.read_text(encoding="utf-8").splitlines()
    line = next((row for row in reversed(lines) if row.startswith(RESULT_MARKER)), None)
    _require(line is not None, "native stdout has no whole-body support-wrench result")
    fields: dict[str, str] = {}
    for token in shlex.split(line):
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


def _ranked_coordinates(fields: dict[str, str], count: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in range(count):
        prefix = f"residual_rank_{rank}_"
        if prefix + "dof" not in fields:
            continue
        row: dict[str, Any] = {
            "rank": rank,
            "dof": _integer(fields, prefix + "dof"),
            "joint": _integer(fields, prefix + "joint"),
            "child_body": _integer(fields, prefix + "child_body"),
            "force": _finite_number(fields, prefix + "force"),
            "acceleration": _finite_number(fields, prefix + "acceleration"),
            "muscle_force": _finite_number(fields, prefix + "muscle_force"),
            "support_force": _finite_number(fields, prefix + "support_force"),
            "passive_force": _finite_number(fields, prefix + "passive_force"),
            "gravity_target": _finite_number(fields, prefix + "gravity_target"),
            "position": _finite_number(fields, prefix + "position"),
            "limit_force": _finite_number(fields, prefix + "limit_force"),
        }
        name = fields.get(prefix + "name")
        if name is not None:
            row["name"] = name
        rows.append(row)
    _require(rows, "native result contains no ranked internal residuals")
    return rows


def compile_audit(
    *,
    stdout: Path,
    stderr: Path | None = None,
    source_commit: str,
    source_branch: str,
    source_worktree_dirty: bool,
    binary_sha256: str,
    payloads: dict[str, dict[str, str]],
    subject: str = "one adult male source package",
    maximum_internal_residual: float = INTERNAL_RESIDUAL_TOLERANCE,
) -> dict[str, Any]:
    stdout = Path(stdout)
    fields = _fields(stdout)
    _require(fields.get("source_model") == "pinned_MyoSim_full_body", "source model changed")
    _require(fields.get("support_payload") == "NHCNT1", "support payload changed")
    _require(fields.get("joint_manifold") == "NHEQ1", "joint equality payload changed")
    _require(fields.get("replay") == "bitwise", "activation sweep replay is not bitwise")
    _require(isinstance(source_commit, str) and source_commit, "source commit is missing")
    _require(isinstance(source_branch, str) and source_branch, "source branch is missing")
    _require(len(binary_sha256) == 64 and all(c in "0123456789abcdef" for c in binary_sha256),
             "binary SHA-256 is invalid")
    _require(math.isfinite(maximum_internal_residual) and maximum_internal_residual >= 0.0,
             "internal residual tolerance is invalid")

    metrics = {
        "body_mass_kg": _finite_number(fields, "body_mass_kg"),
        "expected_weight_n": _finite_number(fields, "expected_weight_n"),
        "total_support_force_n": _finite_number(fields, "total_support_force_n"),
        "relative_weight_error": _finite_number(fields, "relative_weight_error"),
        "support_contacts": _integer(fields, "support_contacts"),
        "active_support_contacts": _integer(fields, "active_support_contacts"),
        "maximum_root_force_residual_n": _finite_number(fields, "max_root_force_residual"),
        "maximum_root_acceleration_residual": _finite_number(fields, "max_root_acceleration_residual"),
        "internal_normalized_residual_rms": _finite_number(fields, "internal_normalized_residual_rms"),
        "activation_sweeps": _integer(fields, "activation_sweeps"),
        "global_activation_polish_iterations": _integer(fields, "global_activation_polish_iterations"),
        "accepted_global_activation_polish_steps": _integer(fields, "accepted_global_activation_polish_steps"),
        "accepted_pose_steps": _integer(fields, "accepted_pose_steps"),
        "active_position_limits": _integer(fields, "active_position_limits"),
        "internal_balanced": _boolean(fields, "internal_balanced"),
    }
    _require(metrics["activation_sweeps"] > 0, "activation sweep count is empty")
    _require(metrics["support_contacts"] > 0 and metrics["active_support_contacts"] >= 0,
             "support contact counts are invalid")
    _require(metrics["active_support_contacts"] <= metrics["support_contacts"],
             "active support count exceeds authored support count")
    rows = _ranked_coordinates(fields)
    internal_closed = (
        metrics["internal_balanced"]
        and metrics["internal_normalized_residual_rms"] <= maximum_internal_residual
    )
    root_closed = metrics["maximum_root_force_residual_n"] <= ROOT_FORCE_TOLERANCE_N
    weight_closed = metrics["relative_weight_error"] <= WEIGHT_RELATIVE_TOLERANCE
    reasons: list[str] = []
    if not root_closed:
        reasons.append("floating-root wrench exceeds the declared tolerance")
    if not weight_closed:
        reasons.append("support force does not close body weight")
    if not internal_closed:
        reasons.append(
            "global activation polish leaves articulated generalized residuals open; "
            f"RMS={metrics['internal_normalized_residual_rms']:.12g}"
        )
    return {
        "schema": SCHEMA,
        "status": "passed" if internal_closed else "partial",
        "subject": subject,
        "source": {
            "branch": source_branch,
            "commit": source_commit,
            "source_worktree_dirty": source_worktree_dirty,
            "binary_sha256": binary_sha256,
            "device": "Mac mini M4 Pro",
            "clock_nanoseconds": 12500,
            "payloads": payloads,
        },
        "input": {
            "stdout": {"path": str(stdout), "sha256": _sha(stdout)},
            "stderr": ({"path": str(Path(stderr)), "sha256": _sha(Path(stderr))}
                       if stderr is not None else None),
        },
        "metrics": metrics,
        "ranked_internal_residuals": rows,
        "qualification": {
            "activation_sweep_executed": True,
            "global_activation_recruitment_candidate": True,
            "floating_root_wrench_closed": root_closed,
            "body_weight_support_closed": weight_closed,
            "internal_generalized_equilibrium": internal_closed,
            "force_convergence": False,
            "anatomical_supports_loading": False,
            "activation_calibration": False,
            "blood_mass_transfer": False,
            "materials_resolved": False,
            "subject_calibration": False,
            "sustained_standing": False,
            "recovery": False,
            "walking": False,
        },
        "gate": {
            "root_force_tolerance_n": ROOT_FORCE_TOLERANCE_N,
            "weight_relative_tolerance": WEIGHT_RELATIVE_TOLERANCE,
            "maximum_internal_residual": maximum_internal_residual,
            "reasons": reasons,
        },
        "boundary": (
            "A global activation sweep and pose polish was executed on the physical "
            "M4 Pro with source support and equality payloads.  Root wrench and body "
            "weight closure are reported independently from the articulated residual. "
            "This diagnostic does not provide measured activation, anatomical contact "
            "loading, material calibration, blood mechanics, standing, recovery, or walking."
        ),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-branch", required=True)
    parser.add_argument("--source-worktree-dirty", action="store_true")
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--payload", action="append", default=[], metavar="NAME=PATH:SHA256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-internal-residual", type=float, default=INTERNAL_RESIDUAL_TOLERANCE)
    parser.set_defaults(handler=run)


def _payloads(values: list[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for value in values:
        _require("=" in value and ":" in value, "payload must be NAME=PATH:SHA256")
        name, rest = value.split("=", 1)
        path, digest = rest.rsplit(":", 1)
        _require(name and path and len(digest) == 64, "payload identity is malformed")
        result[name] = {"path": path, "sha256": digest}
    return result


def run(arguments: argparse.Namespace) -> int:
    result = compile_audit(
        stdout=arguments.stdout,
        stderr=arguments.stderr,
        source_commit=arguments.source_commit,
        source_branch=arguments.source_branch,
        source_worktree_dirty=arguments.source_worktree_dirty,
        binary_sha256=arguments.binary_sha256,
        payloads=_payloads(arguments.payload),
        maximum_internal_residual=arguments.maximum_internal_residual,
    )
    output = Path(arguments.output).resolve()
    _require(not output.is_symlink(), "output is redirected")
    payload = (canonical(result) + b"\n").decode("utf-8")
    if output.exists():
        _require(output.read_text(encoding="utf-8") == payload, "output is immutable")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                      "output": str(output)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
