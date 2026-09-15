"""Record a failed native support-pose active-set search.

This parser deliberately accepts only the geometry-search diagnostic followed
by the native fail-closed support-wrench error.  It makes rejected pose
candidates and the retained witness set reviewable without treating a failed
support solve as an anatomical loading result.
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


SCHEMA = "HumanPack.native-support-pose-search-failure.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError("native support pose search: " + message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fields(path: Path, error_path: Path | None) -> tuple[dict[str, str], str]:
    _require(path.is_file() and not path.is_symlink(), "native stdout is not a regular file")
    rows = path.read_text(encoding="utf-8").splitlines()
    geometry = next((row for row in rows if row.startswith("compiled_support_geometry=")), None)
    _require(geometry is not None, "support geometry diagnostic is missing")
    error_rows = rows
    if error_path is not None:
        _require(error_path.is_file() and not error_path.is_symlink(),
                 "native stderr is not a regular file")
        error_rows += error_path.read_text(encoding="utf-8").splitlines()
    failure = next((row for row in error_rows
                    if "whole-body unilateral support wrench did not close" in row), None)
    _require(failure is not None, "native run did not fail closed on support wrench")
    fields: dict[str, str] = {}
    for token in shlex.split(geometry):
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields, failure


def _number(fields: dict[str, str], key: str) -> float:
    value = fields.get(key)
    _require(value is not None, f"support geometry is missing {key}")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ImportError(f"support geometry {key} is not numeric") from error
    _require(math.isfinite(parsed), f"support geometry {key} is not finite")
    return parsed


def _integer(fields: dict[str, str], key: str) -> int:
    value = _number(fields, key)
    _require(value == int(value), f"support geometry {key} is not integral")
    return int(value)


def compile_failure(
    *,
    stdout: Path,
    stderr: Path | None,
    source_commit: str,
    source_branch: str,
    binary_sha256: str,
    payloads: dict[str, dict[str, str]],
    source_worktree_dirty: bool = False,
) -> dict[str, Any]:
    stdout = Path(stdout)
    fields, failure = _fields(stdout, Path(stderr) if stderr is not None else None)
    _require(fields.get("compiled_support_geometry") == "admissible",
             "support geometry was not admitted")
    _require(fields.get("compiled_support_gap_tolerance_m") == "1e-06",
             "support gap tolerance changed")
    _require(source_commit and source_branch, "source identity is missing")
    _require(len(binary_sha256) == 64, "binary SHA-256 is invalid")
    metrics = {
        "minimum_gap_m": _number(fields, "compiled_support_min_gap_m"),
        "gap_tolerance_m": _number(fields, "compiled_support_gap_tolerance_m"),
        "separated_witnesses": _integer(fields, "compiled_support_separated_witnesses"),
        "maximum_separated_force_n": _number(fields, "compiled_support_max_separated_force_n"),
        "rejected_pose_candidates": _integer(fields, "compiled_support_rejected_pose_candidates"),
    }
    _require(metrics["separated_witnesses"] > 0 and metrics["rejected_pose_candidates"] > 0,
             "support pose search did not exercise its active-set gates")
    return {
        "schema": SCHEMA,
        "status": "failed",
        "subject": "one adult male source package",
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
        "failure": failure,
        "qualification": {
            "support_geometry_candidate": True,
            "active_set_selected": False,
            "unilateral_support_wrench_closed": False,
            "anatomical_supports_loading": False,
            "force_convergence": False,
            "sustained_standing": False,
            "recovery": False,
            "walking": False,
        },
        "gate": {
            "reasons": [
                "the unilateral support wrench did not close after pose search",
                f"{metrics['rejected_pose_candidates']} pose candidates were rejected",
            ]
        },
        "boundary": (
            "This is a fail-closed anatomical support active-set diagnostic.  It "
            "proves that source geometry was parsed and the search exercised its "
            "separation gates; it does not prove a loaded contact, material, force, "
            "standing, recovery, or walking result."
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
    result = compile_failure(
        stdout=arguments.stdout,
        stderr=arguments.stderr,
        source_commit=arguments.source_commit,
        source_branch=arguments.source_branch,
        source_worktree_dirty=arguments.source_worktree_dirty,
        binary_sha256=arguments.binary_sha256,
        payloads=_payloads(arguments.payload),
    )
    output = Path(arguments.output).resolve()
    payload = canonical(result) + b"\n"
    _require(not output.is_symlink(), "output is redirected")
    if output.exists():
        _require(output.read_bytes() == payload, "output is immutable")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(payload)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "sha256": hashlib.sha256(payload).hexdigest(),
                      "output": str(output)}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
