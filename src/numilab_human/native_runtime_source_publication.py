"""Publish a hash-locked source/input tuple for the native Human runtime.

The native implementation lives in the Numi Lab source repository, while the
Human owner holds the exact runtime inputs.  This narrow receipt binds the two
immutable public tags and validates the retained fresh public-tag replay.  It
is a delivery and reproducibility record only: it does not promote the
bounded release to force convergence, standing, physiology, or safety.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "Docs/media/native-runtime-source-package-20260915"
INPUT_ROOT = EVIDENCE / "input"
REPLAY_ROOT = EVIDENCE / "replay"
SCHEMA = "HumanPack.native-runtime-source-publication.v1"

RUNTIME_SOURCE = {
    "repository": "https://github.com/Numi2/numi-lab.git",
    "immutable_tag": "human-native-step281-rank-audit-20260915",
    "resolved_commit": "337741b51bfc4a5552a837dacb4d0b5c4268d298",
    "native_target": "metalrobo_numilab_human_myosim_visual_probe",
    "historical_fibre_seed_commit": "7f2410ae5d5e3d8efb44eae26c70a1becf29e2aa",
    "stationary_fibre_root_commit": "f45fcfdc80a05c2226d638227295add7f789c55c",
    "stationary_fibre_root_test_commit": "e2ca10a90554e24894626c27ebf625d56979ddea",
}
SOURCE_INPUT_PACKAGE = {
    "repository": "https://github.com/Numi2/numilab-human.git",
    "immutable_tag": "human-native-runtime-source-inputs-20260915",
    "path": "Docs/media/native-runtime-source-package-20260915",
}
FRESH_PUBLIC_BUILD = {
    "device": "Apple M4 Pro",
    "binary_sha256": "b85669fefaf414a28a9eb44531985dcd83fe549eb5b8d6c7747773bedb3e15bc",
    "historical_binary_sha256": "62648144c20f386b25f625d503fd101f5c846b9ca8b0dac0649e4a064fb04919",
}
INPUTS = {
    "rigid": ("myosim-fullbody-core-reference.nhrigid", 60324,
              "6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44"),
    "muscle": ("myosim-fullbody-muscle-reference.nhmyo", 149372,
               "9a988f19a6fd8e533cd0f2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05"),
    "tendon": ("numi-human-tendon-attachments.nhtendon", 238000,
               "a594194f510eb4aa990a8767f868f999a10b4fedb745c8665368a231ed39b555"),
    "support_contact": ("myosim-fullbody-support-contact.nhcnt", 564,
                        "4d54f8155cd83baaee7af536099824ac0da61e5d5e77544b42c6e5ce1b48c907"),
    "joint_equalities": ("myosim-fullbody-joint-equalities.nheq", 4976,
                         "b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a"),
}
REPLAY = {
    "stdout": ("public-tag-12p5us-64-stdout.txt", 566122,
               "bfb1888bf7dd73a3a94f527cd6ea7fcf3dd55eba85861417b6de1dd5134feac8"),
    "stderr": ("public-tag-12p5us-64-stderr.txt", 111,
               "54f8b2233b3233d10c866e7f98b32da0b978ba4d007ff232c138e61d203a5a74"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("native runtime source publication: " + message)


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _regular(path: Path, label: str) -> bytes:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    return path.read_bytes()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _locked_file(root: Path, specification: tuple[str, int, str], label: str) -> dict[str, Any]:
    name, expected_bytes, expected_sha256 = specification
    path = root / name
    payload = _regular(path, label)
    _require(len(payload) == expected_bytes, f"{label} byte count differs")
    digest = _sha256(payload)
    _require(digest == expected_sha256, f"{label} fingerprint differs")
    return {
        "path": _relative(path),
        "bytes": len(payload),
        "sha256": digest,
    }


def _fields(stdout: Path) -> dict[str, str]:
    text = _regular(stdout, "public-tag replay stdout").decode("utf-8")
    line = next(
        (
            row
            for row in reversed(text.splitlines())
            if "persistent_metal_horizon=true" in row
            and "stand_deterministic_replay=bitwise" in row
            and "myosim_articulated_" in row
        ),
        None,
    )
    _require(line is not None, "public-tag replay lacks a persistent Human result")
    result: dict[str, str] = {}
    for token in shlex.split(line):
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def _number(fields: dict[str, str], key: str) -> float:
    raw = fields.get(key)
    _require(raw is not None, f"public-tag replay lacks {key}")
    try:
        value = float(raw)
    except ValueError as error:
        raise HumanImportError(f"public-tag replay has invalid {key}") from error
    _require(math.isfinite(value), f"public-tag replay has nonfinite {key}")
    return value


def _integer(fields: dict[str, str], key: str) -> int:
    value = _number(fields, key)
    _require(value == int(value), f"public-tag replay has nonintegral {key}")
    return int(value)


def _replay(stdout: Path, stderr: Path) -> dict[str, Any]:
    fields = _fields(stdout)
    stderr_text = _regular(stderr, "public-tag replay stderr").decode("utf-8")
    _require("Metal API Validation Enabled" in stderr_text,
             "public-tag replay does not retain Metal validation evidence")
    _require(fields.get("persistent_metal_horizon") == "true",
             "public-tag replay is not a persistent horizon")
    _require(fields.get("persistent_source_passive_joint_tissue") == "true",
             "public-tag replay does not carry source passive tissue")
    _require(fields.get("persistent_root_assistance") == "none",
             "public-tag replay is assisted")
    _require(fields.get("stand_deterministic_replay") == "bitwise",
             "public-tag replay is not bitwise")
    _require(fields.get("source_support_metal_device") == FRESH_PUBLIC_BUILD["device"],
             "public-tag replay is not physical Mac evidence")
    _require(abs(_number(fields, "muscle_step_seconds") - 1.25e-5) <= 1.0e-15,
             "public-tag replay clock differs")
    _require(_integer(fields, "muscle_step_count") == 64 and
             _integer(fields, "persistent_completed_steps") == 64,
             "public-tag replay horizon differs")
    _require(_number(fields, "persistent_max_penetration_m") == 0.0,
             "public-tag replay has penetration")
    _require(_integer(fields, "compiled_stand_recruited_muscles") == 416,
             "public-tag replay does not retain all accepted fibre records")
    return {
        "stdout": _locked_file(REPLAY_ROOT, REPLAY["stdout"], "public-tag replay stdout"),
        "stderr": _locked_file(REPLAY_ROOT, REPLAY["stderr"], "public-tag replay stderr"),
        "device": fields["source_support_metal_device"],
        "timestep_seconds": _number(fields, "muscle_step_seconds"),
        "step_count": _integer(fields, "muscle_step_count"),
        "duration_seconds": _number(fields, "muscle_step_seconds") * _integer(fields, "muscle_step_count"),
        "persistent_max_acceleration_mps2": _number(fields, "persistent_max_acceleration"),
        "persistent_max_penetration_m": _number(fields, "persistent_max_penetration_m"),
        "recruited_muscles": _integer(fields, "compiled_stand_recruited_muscles"),
        "root_assistance": fields["persistent_root_assistance"],
        "deterministic_replay": fields["stand_deterministic_replay"],
        "metal_validation": True,
    }


def compile_publication(*, evidence: Path = EVIDENCE) -> dict[str, Any]:
    evidence = Path(evidence).resolve()
    _require(evidence.is_relative_to(ROOT), "evidence resolves outside the repository")
    _require(evidence == EVIDENCE.resolve(), "alternate evidence roots are not accepted")
    inputs = {
        name: _locked_file(INPUT_ROOT, specification, f"runtime input {name}")
        for name, specification in INPUTS.items()
    }
    replay = _replay(REPLAY_ROOT / REPLAY["stdout"][0], REPLAY_ROOT / REPLAY["stderr"][0])
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.native-runtime-source-publication.1",
        "status": "partial",
        "subject": "one adult male source package",
        "runtime_source": RUNTIME_SOURCE,
        "source_input_package": SOURCE_INPUT_PACKAGE,
        "inputs": inputs,
        "fresh_public_build": FRESH_PUBLIC_BUILD,
        "fresh_public_tag_replay": replay,
        "qualification": {
            "immutable_runtime_source": True,
            "published_runtime_input_package": True,
            "fresh_public_source_build": True,
            "bounded_unassisted_physical_mac_replay": True,
            "zero_penetration": True,
            "bitwise_replay": True,
            "historical_binary_reproduced": False,
            "temporal_force_convergence": False,
            "sustained_standing": False,
            "physiological_passive_force_calibration": False,
            "recovery": False,
            "walking": False,
        },
        "boundary": (
            "This pins the public native source, the five exact runtime inputs and "
            "one fresh public-tag 0.8 ms replay. The new build hash intentionally "
            "differs from the historical executable, so this receipt does not claim "
            "historical-binary reproduction. It also does not qualify temporal force "
            "convergence, sustained standing, passive-force physiology, recovery or walking."
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
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_publication()
    digest = immutable_write(arguments.output.resolve(), result)
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(arguments.output.resolve()),
        "sha256": digest,
        "temporal_force_convergence": result["qualification"]["temporal_force_convergence"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        raise SystemExit(run(parser.parse_args()))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"native runtime source publication: {error}\n")
