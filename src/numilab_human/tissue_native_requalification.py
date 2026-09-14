"""Bind the current Mac mini costal tissue transaction to a source receipt.

The native check is intentionally admitted as a regional tissue transaction.
It proves source binding, mass partition/rebase and Metal replay for one costal
partition while retaining the unresolved whole-body mass owner, loaded-thorax,
material-calibration and behavior gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.costal-tissue-native-current-requalification.v1"
OUTPUT = ROOT / "Docs/media/tissue-integration-20260908/current-costal-binding-output-20260915.json"

INPUTS = {
    "binding": {
        "path": "/Users/n/numi-human-tissue-ownership-20260908/costal-tissue.nhtbind",
        "sha256": "36213b5a3f183445807b0a37c449c6f7755a6ee27c62347fb9a2224990cfab08",
    },
    "cartilage": {
        "path": "/Users/n/numi-human-tissue-ownership-20260908/bodyparts3d-costal-cartilage.nhcartilage",
        "sha256": "f5c58b4ddf8a97f631fd21aa7c861bbfa57bc8f863ff20a04165be4db0b5e5cc",
    },
    "rigid": {
        "path": "/Users/n/numi-human-tissue-ownership-20260908/myosim-fullbody-core-reference.nhrigid",
        "sha256": "6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44",
    },
    "registration": {
        "path": "/Users/n/numi-human-tissue-ownership-20260908/fullbody-articular-v3.registration.json",
        "sha256": "ebb7adf5a7777f3f811d771ebce7d61a6c578132572dc514910c765d7a31611e",
    },
}
SOURCE = {
    "branch": "numi-human-equilibrium-20260914",
    "commit": "c45fa9622f6c73b58febdc24a7115aecf3d7699f",
    "device": "Mac mini M4 Pro",
    "binding_check_sha256": "cc090d39858ba3f75d4557ae979f5548e21f1f0cb8a7ec8ea0ddc1962fc3bcae",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("native costal tissue requalification: " + message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float), f"{label} is not numeric")
    result = float(value)
    _require(result == result and abs(result) != float("inf"), f"{label} is not finite")
    return result


def _read_output(path: Path) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), "native output is not a regular file")
    raw = path.read_bytes()
    document = read_json(path)
    _require(isinstance(document, dict), "native output is not an object")
    _require(document.get("schema") == "numi.human.tissue-mass-compilation.v1",
             "native output schema changed")
    _require(document.get("status") == "compiled_requires_v5_runtime_admission",
             "native output admission status changed")
    _require(document.get("metal_executed") is True and
             document.get("metal_device") == "Apple M4 Pro" and
             document.get("metal_replay_cases") == 8 and
             document.get("negative_admission_cases") == 7,
             "native output Metal or negative-case evidence changed")
    _require(document.get("whole_body_dynamic_mass_matrix_qualified") is False and
             document.get("production_owner_fraction") == 0 and
             document.get("rib_sternum_articulation_qualified") is False,
             "native output promoted an unresolved owner")
    partition = document.get("partitions")
    _require(isinstance(partition, list) and len(partition) == 1 and isinstance(partition[0], dict),
             "native output partition is incomplete")
    row = partition[0]
    for key, expected in {
        "donor_body": 20,
        "node_count": 13516,
        "source_mass_kg": 18.618999481201172,
        "tissue_mass_kg": 0.11369939548001184,
        "remaining_mass_kg": 18.505300521850586,
    }.items():
        actual = row.get(key)
        if isinstance(expected, int):
            _require(type(actual) is int and actual == expected, f"native output {key} changed")
        else:
            _require(abs(_finite(actual, key) - expected) <= 1.0e-12,
                     f"native output {key} changed")
    _require(abs(row["source_mass_kg"] - row["tissue_mass_kg"] - row["remaining_mass_kg"]) <= 1.0e-6,
             "native costal partition is not conservative")
    for key in ("maximum_attachment_rest_error_m", "maximum_rebase_point_error_m",
                "maximum_rebase_velocity_error_m_s", "maximum_rebase_jacobian_error",
                "maximum_metal_point_error_m", "maximum_metal_jacobian_error",
                "maximum_metal_donor_mass_matrix_scaled_error", "packed_moment_relative_error"):
        _finite(document.get(key, row.get(key)), key)
    return document, _sha(path)


def compile_receipt(*, output: Path = OUTPUT) -> dict[str, Any]:
    output = Path(output)
    document, output_sha = _read_output(output)
    row = document["partitions"][0]
    return {
        "schema": SCHEMA,
        "status": "partial",
        "subject": "one adult male source package",
        "source": SOURCE,
        "inputs": INPUTS,
        "output": {"path": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output),
                    "sha256": output_sha, "schema": document["schema"]},
        "results": {
            "matter_world_fingerprint": document["matter_world_fingerprint"],
            "cooked_nodes": document["cooked_nodes"],
            "cooked_tetrahedra": document["cooked_tetrahedra"],
            "attachments": document["attachments"],
            "metal_replay_cases": document["metal_replay_cases"],
            "negative_admission_cases": document["negative_admission_cases"],
            "source_mass_kg": row["source_mass_kg"],
            "tissue_mass_kg": row["tissue_mass_kg"],
            "remaining_mass_kg": row["remaining_mass_kg"],
            "maximum_attachment_rest_error_m": document["maximum_attachment_rest_error_m"],
            "maximum_rebase_point_error_m": document["maximum_rebase_point_error_m"],
            "maximum_rebase_velocity_error_m_s": document["maximum_rebase_velocity_error_m_s"],
            "maximum_rebase_jacobian_error": document["maximum_rebase_jacobian_error"],
            "maximum_metal_point_error_m": document["maximum_metal_point_error_m"],
            "maximum_metal_jacobian_error": document["maximum_metal_jacobian_error"],
            "maximum_metal_donor_mass_matrix_scaled_error": document["maximum_metal_donor_mass_matrix_scaled_error"],
            "packed_moment_relative_error": row["packed_moment_relative_error"],
        },
        "qualification": {
            "native_metal_replay": True,
            "registered_costal_tissue_mass_and_rebase": True,
            "mass_conservation": True,
            "whole_body_dynamic_mass_matrix": False,
            "loaded_thorax_convergence": False,
            "experimental_material_calibration": False,
            "standing_recovery_walking": False,
        },
        "boundary": (
            "Current source-bound Apple M4 Pro tissue binding, costal mass partition, "
            "COM-frame rebase, attachment preservation and eight Metal replay cases. "
            "The result remains a regional candidate: no whole-body dynamic mass owner, "
            "loaded-thorax convergence, independent rib/sternum articulation, material "
            "calibration, blood/tissue transfer, fat partition, standing, recovery or "
            "walking is admitted."
        ),
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(arguments: argparse.Namespace) -> int:
    result = compile_receipt(output=arguments.output)
    receipt = _immutable_write(arguments.receipt.resolve(), result)
    print(json.dumps({"schema": SCHEMA, "receipt": str(arguments.receipt.resolve()),
                      "sha256": receipt, "status": result["status"]}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    return run(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
