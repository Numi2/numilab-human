"""Create a subject-mass scaling candidate without admitting it to dynamics.

The current MyoSim export has an authoritative rigid-body tree and inertias,
but its 97.13195176621342 kg mass does not match the acquired 65.5 kg subject.
This compiler produces an auditable uniform mass-only normalization so the
next native owner has an explicit input.  It does not claim segment calibration,
geometry scaling, inertia validation, or production runtime admission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import canonical, read_json

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.subject-mass-scaling-candidate.v1"
COMPILER = "numilab-human.subject-mass-scaling-candidate.1"
PROFILE = ROOT / "config/subject-mass-scaling-candidate.v1.json"
SUBJECT_BINDING = ROOT / "Docs/media/addbiomechanics-subject-binding-20260915/receipt-v1.json"
MASS_OWNER = ROOT / "Docs/media/myosim-mass-owner-20260915/receipt-v1.json"
TARGET_MASS_KG = 65.5


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("subject mass scaling: " + message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _need(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    value = read_json(path)
    return value, _sha256(path)


def _input(path: Path, value: dict[str, Any], digest: str) -> dict[str, str]:
    return {"path": _relative(path), "schema": str(value.get("schema")), "file_sha256": digest}


def _profile(path: Path) -> dict[str, Any]:
    value, _ = _read(path, "scaling profile")
    _need(value.get("schema") == "numi.human.subject-mass-scaling-candidate.v1",
          "scaling profile schema changed")
    _need(value.get("id") == "Falisse2017_subject_1_mass_only_normalization",
          "scaling profile identity changed")
    _need(value.get("subject_binding") == _relative(SUBJECT_BINDING) and
          value.get("mass_owner") == _relative(MASS_OWNER),
          "scaling profile input paths changed")
    _need(value.get("target_mass_kg") == TARGET_MASS_KG,
          "scaling profile target mass changed")
    return value


def compile_candidate(
    *,
    profile: Path = PROFILE,
    subject_binding: Path = SUBJECT_BINDING,
    mass_owner: Path = MASS_OWNER,
) -> dict[str, Any]:
    _profile(Path(profile))
    paths = {"subject_binding": Path(subject_binding), "mass_owner": Path(mass_owner)}
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        documents[name], hashes[name] = _read(path, name.replace("_", " "))

    binding = documents["subject_binding"]
    _need(binding.get("schema") == "HumanPack.addbiomechanics-subject-binding.v1" and
          binding.get("status") == "partial", "subject binding boundary changed")
    binding_qualification = binding.get("qualification", {})
    _need(binding_qualification.get("source_subject_acquired") is True and
          binding_qualification.get("subject_mass_identity_match") is False and
          binding_qualification.get("subject_calibration_ready") is False,
          "subject binding promoted mass calibration")
    subject = binding.get("subject", {})
    _need(subject.get("id") == "Falisse2017:subject_1" and
          subject.get("sex") == "male" and subject.get("age_years") == 43 and
          subject.get("height_m") == 1.78 and subject.get("mass_kg") == TARGET_MASS_KG,
          "subject target changed")

    owner = documents["mass_owner"]
    _need(owner.get("schema") == "HumanPack.myosim-rigid-body-mass-owner-candidate.v1" and
          owner.get("status") == "partial", "rigid-body mass owner boundary changed")
    ledger = owner.get("rigid_body_mass", {})
    _need(ledger.get("body_count") == 103 and ledger.get("mass_bearing_body_count") == 96,
          "rigid-body mass owner counts changed")
    original_total = float(ledger.get("total_mass_kg"))
    _need(math.isfinite(original_total) and original_total > TARGET_MASS_KG,
          "source mass is not above the target subject mass")
    source_bodies = ledger.get("bodies")
    _need(isinstance(source_bodies, list) and len(source_bodies) == 103,
          "rigid-body source rows are incomplete")
    mass_scale = TARGET_MASS_KG / original_total
    _need(math.isfinite(mass_scale) and 0.0 < mass_scale < 1.0,
          "mass-only scaling factor is outside (0,1)")

    rows: list[dict[str, Any]] = []
    scaled_total = 0.0
    positive_rows = 0
    for source in source_bodies:
        _need(isinstance(source, dict), "rigid-body source row is not an object")
        for key in ("id", "name", "parent", "mass_kg", "inertia_kg_m2", "owner_id"):
            _need(key in source, f"rigid-body source row lacks {key}")
        source_mass = float(source["mass_kg"])
        source_inertia = source["inertia_kg_m2"]
        _need(math.isfinite(source_mass) and source_mass >= 0.0,
              f"source mass is invalid for {source['owner_id']}")
        _need(isinstance(source_inertia, list) and len(source_inertia) == 3 and
              all(math.isfinite(float(value)) and float(value) >= 0.0 for value in source_inertia),
              f"source inertia is invalid for {source['owner_id']}")
        scaled_mass = source_mass * mass_scale
        if source_mass > 0.0:
            positive_rows += 1
        scaled_total += scaled_mass
        rows.append({
            "id": source["id"],
            "name": source["name"],
            "parent": source["parent"],
            "owner_id": source["owner_id"],
            "source_mass_kg": source_mass,
            "scaled_mass_kg": scaled_mass,
            "source_inertia_kg_m2": source_inertia,
            "scaled_inertia_kg_m2": [float(value) * mass_scale for value in source_inertia],
        })
    mass_error = scaled_total - TARGET_MASS_KG
    _need(positive_rows == 96 and abs(mass_error) <= 1.0e-12,
          "scaled mass rows do not close the target mass")

    return {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "status": "partial",
        "inputs": {name: _input(paths[name], documents[name], hashes[name]) for name in paths},
        "subject": {
            "id": subject["id"], "sex": subject["sex"], "age_years": subject["age_years"],
            "height_m": subject["height_m"], "target_mass_kg": TARGET_MASS_KG,
        },
        "source_mass": {
            "total_mass_kg": original_total,
            "body_count": ledger["body_count"],
            "mass_bearing_body_count": ledger["mass_bearing_body_count"],
            "owner_ids_sha256": ledger.get("owner_ids_sha256"),
        },
        "scaling": {
            "method": "uniform_mass_only_fixed_geometry",
            "mass_factor": mass_scale,
            "inertia_factor": mass_scale,
            "geometry_factor": 1.0,
            "target_mass_kg": TARGET_MASS_KG,
            "scaled_mass_kg": scaled_total,
            "mass_closure_error_kg": mass_error,
        },
        "bodies": rows,
        "qualification": {
            "source_body_tree_preserved": True,
            "source_mass_rows_preserved": True,
            "target_mass_closure": True,
            "uniform_mass_normalization_candidate": True,
            "segment_mass_calibration": False,
            "geometry_scaling_calibration": False,
            "inertia_calibration": False,
            "mechanical_runtime_admitted": False,
            "organ_mass_owner": False,
            "mechanical_blood_mass_owner": False,
            "skeletal_muscle_tissue_mass_owner": False,
            "fat_volume_and_mass_owner": False,
            "soft_tissue_material_calibration": False,
            "subject_calibration": False,
            "standing": False,
            "recovery": False,
            "walking": False,
        },
        "blockers": [
            {"id": "segment_mass_calibration", "status": "open",
             "reason": "Uniform normalization closes total mass only; no measured segment composition or inertial calibration is supplied."},
            {"id": "geometry_and_inertia_calibration", "status": "open",
             "reason": "The candidate assumes fixed geometry and scales inertia linearly with mass; subject-specific dimensions and inertial validation are absent."},
            {"id": "native_runtime_admission", "status": "open",
             "reason": "The scaled rows have not been consumed by the native articulated solver or compared against held-out measured trials."},
        ],
        "boundary": (
            "This record provides a deterministic uniform mass-only normalization of the "
            "pinned MyoSim rigid-body rows to the acquired Falisse2017 subject mass. It "
            "preserves the source body tree and closes the scalar target mass, but it is "
            "not segment composition, geometry, inertia, material, organ, blood, fat, "
            "muscle, standing, walking, or production runtime qualification."
        ),
    }


def immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    path = Path(path)
    _need(not path.is_symlink(), "output is redirected")
    if path.exists():
        _need(path.read_bytes() == payload, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--subject-binding", type=Path, default=SUBJECT_BINDING)
    parser.add_argument("--mass-owner", type=Path, default=MASS_OWNER)
    parser.add_argument("--output", type=Path, required=True)


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(profile=arguments.profile,
                               subject_binding=arguments.subject_binding,
                               mass_owner=arguments.mass_owner)
    digest = immutable_write(arguments.output, result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "output": str(Path(arguments.output).resolve()), "sha256": digest,
                      "mass_factor": result["scaling"]["mass_factor"],
                      "scaled_mass_kg": result["scaling"]["scaled_mass_kg"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"subject mass scaling: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
