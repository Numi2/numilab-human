"""Fail-closed preparation boundary for expensive anatomy geometry gates.

A caller must finish input/schema/crosswalk checks and serialize its complete
pre-predicate proof before invoking geometry predicates. The resulting object
binds the proof bytes, canonical input paths and exact runner argv.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FrozenPreparation:
    proof_path: Path
    proof_sha256: str
    runner_argv: tuple[str, ...]
    input_pins: tuple[tuple[str, str], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_pins(input_pins: dict[str, str]) -> dict[str, str]:
    if not isinstance(input_pins, dict) or not input_pins:
        raise ValueError("preparation requires a nonempty input pin map")
    canonical: dict[str, str] = {}
    for raw_path, expected_sha in input_pins.items():
        if (
            not isinstance(raw_path, str)
            or not isinstance(expected_sha, str)
            or not _HEX_SHA256.fullmatch(expected_sha)
        ):
            raise ValueError("malformed preparation input pin")
        path = Path(raw_path).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"pinned input is not a file: {path}")
        key = str(path)
        if key in canonical:
            raise ValueError(f"duplicate canonical input path: {key}")
        if _sha256_file(path) != expected_sha:
            raise ValueError(f"preparation input hash changed: {key}")
        canonical[key] = expected_sha
    return dict(sorted(canonical.items()))


def freeze_preparation(
    proof_path: str | Path,
    proof: dict[str, Any],
    *,
    runner_argv: list[str],
    input_pins: dict[str, str],
) -> FrozenPreparation:
    """Validate and exclusively serialize a complete pre-predicate proof."""
    if not isinstance(proof, dict) or proof.get("predicate_phase_started") is not False:
        raise ValueError("preparation proof must explicitly precede predicates")
    status = proof.get("status")
    if not isinstance(status, str) or not status.startswith("PASS_"):
        raise ValueError("preparation proof does not carry a PASS status")
    if (
        not isinstance(runner_argv, list)
        or not runner_argv
        or any(not isinstance(arg, str) or not arg for arg in runner_argv)
    ):
        raise ValueError("exact nonempty runner argv is required")

    canonical_pins = _canonical_pins(input_pins)
    path = Path(proof_path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen preparation: {path}")
    bound_proof = dict(proof)
    bound_proof["execution_boundary"] = {
        "runner_argv": list(runner_argv),
        "input_pins": canonical_pins,
    }
    encoded = json.dumps(bound_proof, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
    return FrozenPreparation(
        path, _sha256_file(path), tuple(runner_argv), tuple(canonical_pins.items())
    )


def verify_frozen_preparation(preparation: FrozenPreparation) -> dict[str, Any]:
    """Verify frozen bytes and their exact inputs immediately before predicates."""
    path = preparation.proof_path.resolve(strict=True)
    if _sha256_file(path) != preparation.proof_sha256:
        raise ValueError("frozen preparation proof changed after preparation")
    proof = json.loads(path.read_text(encoding="utf-8"))
    boundary = proof.get("execution_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("frozen preparation lacks its execution boundary")
    if tuple(boundary.get("runner_argv", ())) != preparation.runner_argv:
        raise ValueError("frozen runner argv binding differs")
    pins = boundary.get("input_pins")
    if (
        not isinstance(pins, dict)
        or tuple(sorted(pins.items())) != preparation.input_pins
    ):
        raise ValueError("frozen input pin binding differs")
    if proof.get("predicate_phase_started") is not False or not str(
        proof.get("status", "")
    ).startswith("PASS_"):
        raise ValueError("frozen preparation is not a successful pre-predicate proof")
    _canonical_pins(dict(preparation.input_pins))
    return proof


def execute_after_preparation(
    preparation: FrozenPreparation,
    geometry: Callable[[dict[str, Any]], Any],
) -> Any:
    """Revalidate frozen preparation, then call the geometry predicate phase."""
    return geometry(verify_frozen_preparation(preparation))
