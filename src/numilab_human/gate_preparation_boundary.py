"""Fail-closed preparation boundary for expensive anatomy geometry gates.

A caller must finish input/schema/crosswalk checks and serialize its complete
pre-predicate proof before invoking geometry predicates. The resulting object
binds the proof bytes, canonical input paths and exact runner argv.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FrozenPreparation:
    proof_path: Path
    proof_sha256: str
    runner_argv: tuple[str, ...]
    input_pins: tuple[tuple[str, str], ...]
    input_path_bindings: tuple[tuple[str, str], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_pins(
    input_pins: dict[str, str],
) -> tuple[dict[str, str], tuple[tuple[str, str], ...]]:
    if not isinstance(input_pins, dict) or not input_pins:
        raise ValueError("preparation requires a nonempty input pin map")
    canonical: dict[str, str] = {}
    path_bindings: list[tuple[str, str]] = []
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
        path_bindings.append((raw_path, key))
    return dict(sorted(canonical.items())), tuple(sorted(path_bindings))


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

    canonical_pins, path_bindings = _canonical_pins(input_pins)
    path = Path(proof_path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen preparation: {path}")
    bound_proof = dict(proof)
    bound_proof["execution_boundary"] = {
        "runner_argv": list(runner_argv),
        "input_pins": canonical_pins,
        "input_path_bindings": dict(path_bindings),
    }
    encoded = json.dumps(bound_proof, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
    return FrozenPreparation(
        path,
        _sha256_file(path),
        tuple(runner_argv),
        tuple(canonical_pins.items()),
        path_bindings,
    )


def verify_frozen_preparation(preparation: FrozenPreparation) -> dict[str, Any]:
    """Verify frozen bytes and their exact inputs immediately before predicates."""
    path = preparation.proof_path.resolve(strict=True)
    if _sha256_file(path) != preparation.proof_sha256:
        raise ValueError("frozen preparation proof changed after preparation")
    proof = json.loads(path.read_text(encoding="utf-8"))
    boundary = proof.get("execution_boundary")
    if not isinstance(boundary, dict):
        raise TypeError("frozen preparation lacks its execution boundary")
    if tuple(boundary.get("runner_argv", ())) != preparation.runner_argv:
        raise ValueError("frozen runner argv binding differs")
    pins = boundary.get("input_pins")
    if (
        not isinstance(pins, dict)
        or tuple(sorted(pins.items())) != preparation.input_pins
    ):
        raise ValueError("frozen input pin binding differs")
    path_bindings = boundary.get("input_path_bindings")
    if (
        not isinstance(path_bindings, dict)
        or tuple(sorted(path_bindings.items())) != preparation.input_path_bindings
    ):
        raise ValueError("frozen raw-to-canonical path bindings differ")
    for raw_path, canonical_path in preparation.input_path_bindings:
        if str(Path(raw_path).resolve(strict=True)) != canonical_path:
            raise ValueError(f"pinned input path resolution changed: {raw_path}")
    if proof.get("predicate_phase_started") is not False or not str(
        proof.get("status", "")
    ).startswith("PASS_"):
        raise ValueError("frozen preparation is not a successful pre-predicate proof")
    _canonical_pins(dict(preparation.input_pins))
    return proof


def load_frozen_preparation(
    proof_path: str | Path,
    expected_sha256: str,
) -> FrozenPreparation:
    """Load a preparation proof in another process using an external hash pin.

    The caller must obtain expected_sha256 from retained, immutable evidence.
    This function never derives its trust anchor from the proof file itself: it
    reads the exact bytes once, checks that external digest, constructs the
    frozen binding from the validated execution boundary, then verifies that
    proof and every input are still unchanged.
    """
    if not isinstance(expected_sha256, str) or not _HEX_SHA256.fullmatch(expected_sha256):
        raise ValueError("expected frozen preparation SHA-256 is malformed")
    path = Path(proof_path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("frozen preparation proof is not a file")
    encoded = path.read_bytes()
    observed_sha256 = hashlib.sha256(encoded).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("frozen preparation proof differs from external SHA-256 pin")
    try:
        proof = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen preparation proof is not valid UTF-8 JSON") from exc
    if not isinstance(proof, dict):
        raise TypeError("frozen preparation proof must be a JSON object")
    if proof.get("predicate_phase_started") is not False or not str(
        proof.get("status", "")
    ).startswith("PASS_"):
        raise ValueError("frozen preparation is not a successful pre-predicate proof")
    boundary = proof.get("execution_boundary")
    if not isinstance(boundary, dict) or set(boundary) != {
        "runner_argv", "input_pins", "input_path_bindings"
    }:
        raise ValueError("frozen preparation execution boundary is malformed")

    runner_argv = boundary["runner_argv"]
    if (
        not isinstance(runner_argv, list)
        or not runner_argv
        or any(not isinstance(arg, str) or not arg for arg in runner_argv)
    ):
        raise ValueError("frozen preparation runner argv is malformed")

    pins = boundary["input_pins"]
    if not isinstance(pins, dict) or not pins:
        raise ValueError("frozen preparation input pins are malformed")
    for canonical_path, digest in pins.items():
        if (
            not isinstance(canonical_path, str)
            or not Path(canonical_path).is_absolute()
            or not isinstance(digest, str)
            or not _HEX_SHA256.fullmatch(digest)
        ):
            raise ValueError("frozen preparation input pin entry is malformed")

    raw_bindings = boundary["input_path_bindings"]
    if (
        not isinstance(raw_bindings, dict)
        or not raw_bindings
        or any(
            not isinstance(raw_path, str)
            or not raw_path
            or not isinstance(canonical_path, str)
            or not Path(canonical_path).is_absolute()
            for raw_path, canonical_path in raw_bindings.items()
        )
    ):
        raise ValueError("frozen preparation raw-to-canonical path bindings are malformed")
    if (
        len(raw_bindings) != len(pins)
        or len(set(raw_bindings)) != len(raw_bindings)
        or len(set(raw_bindings.values())) != len(raw_bindings)
        or set(raw_bindings.values()) != set(pins)
    ):
        raise ValueError("frozen preparation path bindings do not exactly cover pinned inputs")

    preparation = FrozenPreparation(
        proof_path=path,
        proof_sha256=expected_sha256,
        runner_argv=tuple(runner_argv),
        input_pins=tuple(sorted(pins.items())),
        input_path_bindings=tuple(sorted(raw_bindings.items())),
    )
    verify_frozen_preparation(preparation)
    return preparation


def execute_after_preparation(
    preparation: FrozenPreparation,
    geometry: Callable[[dict[str, Any]], Any],
) -> Any:
    """Revalidate frozen preparation, then call the geometry predicate phase."""
    return geometry(verify_frozen_preparation(preparation))
