"""Compile the pinned cardiac material, activation, support, and loading contract.

The Rodero configuration contains useful source parameters but omits the
stress-free reference, quantitative Robin/anchor fields, and a complete
activation-time deck.  This compiler preserves the supplied values and turns
each omission into an explicit admission gate.  It never supplies defaults or
promotes the source description to an anatomical native wall owner.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from . import cardiac_wall_source
from .model import ImportError as HumanImportError
from .physiology import canonical, read_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/cardiac-wall-rodero18.v1.json"
SCHEMA = "HumanPack.cardiac-loading-contract.v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("cardiac loading contract: " + message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha256(canonical(value) + b"\n")


def _copy(source: dict[str, Any], key: str) -> Any:
    require(key in source, f"source mechanics is missing {key}")
    return copy.deepcopy(source[key])


def compile_contract(*, config: Path = CONFIG) -> dict[str, Any]:
    config = Path(config)
    raw = config.read_bytes()
    actual_sha = _sha256(raw)
    require(actual_sha == cardiac_wall_source.CONFIG_SHA256,
            "pinned Rodero configuration hash mismatch")
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError) as error:
        raise HumanImportError("invalid Rodero configuration") from error
    require(isinstance(value, dict) and value.get("schema") == "HumanPack.cardiac-wall-rodero18-source.v1",
            "unsupported cardiac wall source configuration")
    mechanics = value.get("source_mechanics")
    require(isinstance(mechanics, dict), "source mechanics contract is absent")
    unresolved = _copy(mechanics, "unresolved_source_inputs")
    require(isinstance(unresolved, list) and len(unresolved) == 6 and all(isinstance(item, str) for item in unresolved),
            "unresolved source input list changed")
    labels = value.get("labels")
    require(isinstance(labels, list) and len(labels) == 24, "source label table changed")
    closure_labels = [row["id"] for row in labels if row.get("role") in {
        "artificial_vessel_closure", "truncated_vessel_border"
    }]
    require(closure_labels == list(range(11, 25)), "closure label range changed")
    active = _copy(mechanics, "active_ventricular")
    loading = _copy(mechanics, "loading")
    passive = {
        "ventricular": _copy(mechanics, "passive_ventricular"),
        "nonventricular": _copy(mechanics, "passive_nonventricular"),
    }
    unresolved_gates = [
        {"id": "stress_free_reference", "status": "missing",
         "source_statement": unresolved[0], "admitted": False},
        {"id": "inertial_density", "status": "missing",
         "source_statement": unresolved[1], "admitted": False},
        {"id": "epicardial_robin_coefficients", "status": "missing",
         "source_statement": unresolved[2], "admitted": False},
        {"id": "venous_anchor_interpretation", "status": "conflicted",
         "source_statement": unresolved[3], "admitted": False},
        {"id": "source_reference_trajectory", "status": "missing",
         "source_statement": unresolved[4], "admitted": False},
        {"id": "zero_forward_valve_admission", "status": "unsupported_native_edge",
         "source_statement": unresolved[5], "admitted": False},
    ]
    forward_zero = [name for name, parameter in loading["valve_resistance"].items()
                    if name.endswith("_forward") and parameter["value"] == 0]
    require(forward_zero == ["aortic_forward", "pulmonary_forward"],
            "source zero-forward valve set changed")
    result = {
        "schema": SCHEMA,
        "compiler": "numilab-human.cardiac-loading-contract.1",
        "source": {
            "config_path": str(config.relative_to(ROOT)) if config.is_relative_to(ROOT) else str(config),
            "config_sha256": actual_sha,
            "source_id": value["id"],
            "archive_sha256": value["source"]["archive"]["sha256"],
            "member_sha256": value["source"]["member"]["sha256"],
            "license": value["source"]["license"],
        },
        "source_materials": passive,
        "source_activation": active,
        "source_loading": loading,
        "closure_labels": closure_labels,
        "unresolved_gates": unresolved_gates,
        "qualification": {
            "source_parameters_hash_bound": True,
            "source_units_preserved": True,
            "passive_material_classes_retained": True,
            "activation_parameters_retained": True,
            "loading_parameters_retained": True,
            "quantitative_robin_support_admitted": False,
            "venous_anchor_support_admitted": False,
            "stress_free_reference_supplied": False,
            "source_activation_time_field_supplied": False,
            "closure_materials_resolved": False,
            "inertial_density_assigned": False,
            "native_anatomical_wall_admitted": False,
            "subject_specific_calibration": False,
            "physical_steps": False,
        },
        "source_qualification": _copy(value, "qualification"),
        "identity_sha256": _canonical_sha({
            "config_sha256": actual_sha,
            "source_materials": passive,
            "source_activation": active,
            "source_loading": loading,
            "closure_labels": closure_labels,
            "unresolved_gates": unresolved_gates,
        }),
        "boundary": (
            "This record preserves the supplied source material, activation, and loading "
            "parameters and names six missing or conflicting admission inputs. It does not "
            "recover the unloaded anatomy, assign closure stiffness or inertial density, "
            "resolve venous/Robin supports, provide a time-resolved activation field, or "
            "qualify native anatomical stepping or subject calibration."
        ),
    }
    canonical(result)
    return result


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    require(not path.is_symlink(), "output is redirected")
    if path.exists():
        require(path.read_bytes() == payload, "output is immutable; choose a new output path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return _sha256(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_contract(config=args.config)
        digest = _immutable_write(args.output.resolve(), result)
        print(json.dumps({"schema": SCHEMA, "output": str(args.output.resolve()),
                          "sha256": digest, "unresolved_gate_count": len(result["unresolved_gates"]),
                          "native_anatomical_wall_admitted": False}, sort_keys=True))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"cardiac loading contract: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
