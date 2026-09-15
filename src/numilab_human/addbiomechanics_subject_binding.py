"""Bind the acquired AddBiomechanics subject to the current Human evidence graph.

This compiler is an identity and mass-accounting gate.  It deliberately keeps
the measured subject separate from the current composite MyoSim body until a
subject-scaled mass/inertia fit and Numi prediction comparison exist.
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
SCHEMA = "HumanPack.addbiomechanics-subject-binding.v1"
COMPILER = "numilab-human.addbiomechanics-subject-binding.1"
PROFILE = ROOT / "config/addbiomechanics-subject-binding.v1.json"
ACQUISITION = ROOT / "Docs/media/addbiomechanics-falisse-20260915/acquisition-receipt.json"
CURRENT_EVIDENCE = ROOT / "Docs/media/current-human-evidence-join-20260915/receipt-v3.json"
RIGID_BODY_MASS = ROOT / "Docs/media/myosim-mass-owner-20260915/receipt-v1.json"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("AddBiomechanics subject binding: " + message)


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
    value, _ = _read(path, "binding profile")
    _need(value.get("schema") == "numi.human.addbiomechanics-subject-binding.v1",
          "binding profile schema changed")
    _need(value.get("id") == "Falisse2017_subject_1_to_current_human",
          "binding profile identity changed")
    _need(value.get("expected_subject") == {
        "id": "Falisse2017:subject_1",
        "sex": "male",
        "age_years": 43,
        "height_m": 1.78,
        "mass_kg": 65.5,
    }, "binding profile subject changed")
    _need(value.get("expected_reference_trial_names") == [
        "Gait_5_segment_0", "Gait_7_segment_0", "StairUp_4_segment_0", "StairUp_5_segment_0",
    ], "binding profile trials changed")
    _need(value.get("acquisition") == _relative(ACQUISITION),
          "binding profile acquisition path changed")
    _need(value.get("current_evidence") == _relative(CURRENT_EVIDENCE),
          "binding profile current evidence path changed")
    _need(value.get("rigid_body_mass") == _relative(RIGID_BODY_MASS),
          "binding profile rigid-body mass path changed")
    return value


def compile_binding(
    *,
    profile: Path = PROFILE,
    acquisition: Path = ACQUISITION,
    current_evidence: Path = CURRENT_EVIDENCE,
    rigid_body_mass: Path = RIGID_BODY_MASS,
) -> dict[str, Any]:
    _profile(Path(profile))
    paths = {
        "acquisition": Path(acquisition),
        "current_evidence": Path(current_evidence),
        "rigid_body_mass": Path(rigid_body_mass),
    }
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        documents[name], hashes[name] = _read(path, name.replace("_", " "))

    acquisition_document = documents["acquisition"]
    _need(acquisition_document.get("schema") == "HumanPack.addbiomechanics-acquisition.v1",
          "acquisition schema changed")
    _need(acquisition_document.get("status") == "source_acquired",
          "acquisition is not source_acquired")
    acquisition_qualification = acquisition_document.get("qualification", {})
    for key in ("source_artifact_hash_verified", "adult_male_metadata_verified",
                "measured_reference_tables_extracted", "walking_reference_present"):
        _need(acquisition_qualification.get(key) is True, f"acquisition lacks {key}")
    for key in ("prediction_comparison_complete", "activation_calibration_qualified",
                "anatomical_support_loading_qualified", "materials_qualified",
                "blood_tissue_mass_transfer_qualified", "standing_or_walking_qualified"):
        _need(acquisition_qualification.get(key) is False,
              f"acquisition boundary changed for {key}")
    dataset = acquisition_document.get("dataset", {})
    artifact = dataset.get("source_artifact", {})
    _need(dataset.get("source_artifact_raw_bytes") == artifact.get("bytes") and
          dataset.get("source_artifact_raw_sha256") == artifact.get("sha256"),
          "acquisition source artifact identity is inconsistent")
    subject = acquisition_document.get("subject", {})
    expected_subject = {
        "id": "Falisse2017:subject_1", "sex": "male", "age_years": 43,
        "height_m": 1.78, "mass_kg": 65.5,
    }
    _need(all(subject.get(key) == value for key, value in expected_subject.items()),
          "acquired subject metadata changed")
    trials = acquisition_document.get("trials", [])
    _need([trial.get("name") for trial in trials] == [
        "Gait_5_segment_0", "Gait_7_segment_0", "StairUp_4_segment_0", "StairUp_5_segment_0",
    ], "acquired trial set changed")
    _need(all(trial.get("activity") == "walking" for trial in trials),
          "acquired trial activity changed")
    _need(not acquisition_qualification.get("standing_reference_present") and
          not acquisition_qualification.get("recovery_reference_present"),
          "acquisition unexpectedly supplies standing or recovery data")

    evidence = documents["current_evidence"]
    _need(evidence.get("schema") == "HumanPack.current-human-evidence-join.v2" and
          evidence.get("status") == "partial", "current Human evidence boundary changed")
    evidence_qualification = evidence.get("qualification", {})
    _need(evidence_qualification.get("subject_calibration") is False and
          evidence_qualification.get("integrated_human_qualification") is False,
          "current Human evidence promoted subject or integrated qualification")

    mass_document = documents["rigid_body_mass"]
    _need(mass_document.get("schema") == "HumanPack.myosim-rigid-body-mass-owner-candidate.v1" and
          mass_document.get("status") == "partial", "rigid-body mass owner boundary changed")
    mass = mass_document.get("rigid_body_mass", {})
    _need(mass.get("body_count") == 103 and mass.get("mass_bearing_body_count") == 96,
          "rigid-body mass owner counts changed")
    runtime_mass = float(mass.get("total_mass_kg"))
    measured_mass = float(subject["mass_kg"])
    _need(math.isfinite(runtime_mass) and runtime_mass > 0.0, "runtime mass is not finite and positive")
    mass_delta = runtime_mass - measured_mass
    relative_delta = mass_delta / measured_mass
    _need(mass_delta > 0.0 and relative_delta > 0.1,
          "current rigid-body mass unexpectedly matches the acquired subject")

    blockers = [
        {
            "id": "subject_mass_identity",
            "status": "open",
            "reason": "The acquired 65.5 kg subject and current 97.13195176621342 kg rigid-body owner are not the same scaled mechanical subject; rebuild mass/inertia before fitting.",
        },
        {
            "id": "prediction_comparison",
            "status": "open",
            "reason": "The acquisition has measured references but no Numi prediction tables on disjoint trials.",
        },
        {
            "id": "standing_recovery_reference",
            "status": "open",
            "reason": "The selected subject has gait and stair trials but no standing or recovery trial.",
        },
    ]
    return {
        "schema": SCHEMA,
        "compiler": COMPILER,
        "status": "partial",
        "inputs": {name: _input(paths[name], documents[name], hashes[name]) for name in paths},
        "subject": {
            "id": subject["id"], "sex": subject["sex"], "age_years": subject["age_years"],
            "height_m": subject["height_m"], "mass_kg": subject["mass_kg"],
            "reference_trial_count": len(trials),
            "reference_trial_names": [trial["name"] for trial in trials],
            "walking_reference_present": True,
            "standing_reference_present": False,
            "recovery_reference_present": False,
        },
        "runtime": {
            "current_evidence_status": evidence["status"],
            "native_owner": evidence.get("native_owner", {}),
            "clock_nanoseconds": evidence.get("clock", {}).get("nanoseconds"),
            "rigid_body_count": mass["body_count"],
            "mass_bearing_body_count": mass["mass_bearing_body_count"],
        },
        "mass_comparison": {
            "measured_subject_mass_kg": measured_mass,
            "current_rigid_body_mass_kg": runtime_mass,
            "difference_kg": mass_delta,
            "relative_difference": relative_delta,
            "same_scaled_mechanical_subject": False,
        },
        "qualification": {
            "source_subject_acquired": True,
            "measured_reference_tables_extracted": True,
            "current_runtime_graph_bound": True,
            "subject_identity_bound": False,
            "subject_mass_identity_match": False,
            "prediction_comparison_complete": False,
            "subject_calibration_ready": False,
            "activation_calibration_qualified": False,
            "anatomical_support_loading_qualified": False,
            "blood_tissue_mass_transfer_qualified": False,
            "materials_qualified": False,
            "standing_or_walking_qualified": False,
            "integrated_human_qualification": False,
        },
        "blockers": blockers,
        "boundary": (
            "This receipt binds the verified Falisse2017 adult-male reference artifact to the "
            "current Human evidence graph and compares its measured mass with the current "
            "rigid-body owner. The 31.63195176621342 kg mismatch is retained as a subject-scaling "
            "blocker. No prediction comparison, activation/material fit, anatomical loading, "
            "organ/blood/tissue/fat ownership, standing, recovery, or walking claim is promoted."
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
    parser.add_argument("--acquisition", type=Path, default=ACQUISITION)
    parser.add_argument("--current-evidence", type=Path, default=CURRENT_EVIDENCE)
    parser.add_argument("--rigid-body-mass", type=Path, default=RIGID_BODY_MASS)
    parser.add_argument("--output", type=Path, required=True)


def run(arguments: argparse.Namespace) -> int:
    result = compile_binding(profile=arguments.profile, acquisition=arguments.acquisition,
                             current_evidence=arguments.current_evidence,
                             rigid_body_mass=arguments.rigid_body_mass)
    digest = immutable_write(arguments.output, result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "output": str(Path(arguments.output).resolve()), "sha256": digest,
                      "blockers": len(result["blockers"])}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"AddBiomechanics subject binding: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
