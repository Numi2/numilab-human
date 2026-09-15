"""Join the current Human mechanics and organ/tissue evidence.

This is a source-bound evidence join, not an integrated qualification.  It
ensures that the current native mechanics receipts and the current regional
blood/tissue receipt refer to the same native source owner, binary and clock.
The output keeps the remaining anatomical, material, activation-calibration,
subject-calibration and behavior gates fail-closed.
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
SCHEMA = "HumanPack.current-human-evidence-join.v2"
BODY_COMPOSITION = ROOT / "Docs/media/body-composition-integration-20260914/receipt-v13.json"
FORCE_AUDIT = ROOT / "Docs/media/native-force-audit-pose24-20260915/receipt-v1.json"
FORCE_LEDGER = ROOT / "Docs/media/native-force-audit-pose24-20260915/force-ledger.json"
PASSIVE_STAND = ROOT / "Docs/media/native-passive-stand-pose24-20260915/receipt-v1.json"
REGIONAL_EXCHANGE = ROOT / "Docs/media/native-human-regional-exchange-pose24-20260915/receipt-v1.json"
DYNAMIC_AUDIT = ROOT / "Docs/media/native-dynamic-force-audit-20260915/receipt-v1.json"
BLOOD_MASS_TRANSFER = ROOT / "Docs/media/organ-blood-mass-transfer-20260915/receipt-v3.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("current Human evidence join: " + message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    try:
        value = read_json(path)
    except (OSError, ValueError, UnicodeError) as error:
        raise HumanImportError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    return value, _sha256(path)


def _input(path: Path, value: dict[str, Any], digest: str) -> dict[str, str]:
    return {"path": _relative(path), "schema": str(value.get("schema")), "file_sha256": digest}


def _source(value: dict[str, Any], label: str) -> dict[str, Any]:
    source = value.get("source")
    _require(isinstance(source, dict), f"{label} has no source identity")
    for key in ("branch", "commit", "binary_sha256", "device"):
        _require(isinstance(source.get(key), str) and source[key], f"{label} source lacks {key}")
    return source


def _false(value: dict[str, Any], qualification: str, label: str) -> None:
    _require(value.get("qualification", {}).get(qualification) is False,
             f"{label} promoted {qualification}")


def compile_join(
    *,
    body_composition: Path = BODY_COMPOSITION,
    force_audit: Path = FORCE_AUDIT,
    force_ledger: Path = FORCE_LEDGER,
    passive_stand: Path = PASSIVE_STAND,
    regional_exchange: Path = REGIONAL_EXCHANGE,
    dynamic_audit: Path = DYNAMIC_AUDIT,
    blood_mass_transfer: Path = BLOOD_MASS_TRANSFER,
) -> dict[str, Any]:
    paths = {
        "body_composition": Path(body_composition),
        "force_audit": Path(force_audit),
        "force_ledger": Path(force_ledger),
        "passive_stand": Path(passive_stand),
        "regional_exchange": Path(regional_exchange),
        "dynamic_audit": Path(dynamic_audit),
        "blood_mass_transfer": Path(blood_mass_transfer),
    }
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    for name, path in paths.items():
        documents[name], hashes[name] = _read(path, name.replace("_", " "))

    composition = documents["body_composition"]
    _require(composition.get("schema") == "HumanPack.body-composition-integration-candidate.v1",
             "body composition schema changed")
    _require(composition.get("status") == "partial", "body composition promoted a complete status")
    composition_qualification = composition.get("qualification", {})
    _require(composition_qualification.get("source_identity_graph_bound") is True,
             "body composition source identity graph is not bound")
    _require(composition_qualification.get("integrated_human_qualification") is False,
             "body composition promoted integrated qualification")
    ownership = composition.get("ownership", {})
    _require(isinstance(ownership, dict) and ownership and all(value == 0 for value in ownership.values()),
             "body composition contains a physical owner")

    force = documents["force_audit"]
    _require(force.get("schema") == "HumanPack.native-human-force-audit-current-requalification.v1"
             or force.get("schema") == "numi.human.native-force-audit-current-requalification.v1",
             "force audit schema changed")
    force_source = _source(force, "force audit")
    force_qualification = force.get("qualification", {})
    for key in ("source_identity_bound", "source_coordinate_identity_bound",
                "full_128_dof_component_rows", "authoritative_net_reconstructed",
                "generalized_force_closed", "internal_generalized_balance",
                "per_dof_source_audit", "static_root_wrench_balance"):
        _require(force_qualification.get(key) is True, f"force audit lacks {key}")
    for key in ("force_convergence", "anatomical_support_loading", "activation_calibration",
                "blood_mass_transfer", "material_calibration", "subject_calibration",
                "sustained_standing", "recovery", "walking"):
        _require(force_qualification.get(key) is False, f"force audit boundary changed for {key}")
    force_results = force.get("results", {})
    _require(force_results.get("body_count") == 157 and
             force_results.get("coordinate_identity_rows") == 128 and
             force_results.get("static_active_support_contacts") == 6 and
             force_results.get("maximum_absolute_force_residual") <= 1.0e-3,
             "force audit metrics are outside the admitted static contract")

    ledger = documents["force_ledger"]
    _require(ledger.get("schema") == "numi.human.generalized-force-ledger.v1" and
             ledger.get("status") == "passed", "force ledger is not passing")
    ledger_qualification = ledger.get("qualification", {})
    for key in ("component_assembly_closed", "full_generalized_force_ledger",
                "generalized_force_closed", "per_dof_source_audit"):
        _require(ledger_qualification.get(key) is True, f"force ledger lacks {key}")
    for key in ("force_convergence", "sustained_standing", "walking"):
        _require(ledger_qualification.get(key) is False, f"force ledger boundary changed for {key}")
    ledger_coverage = ledger.get("coverage", {})
    _require(ledger_coverage.get("nv") == 128 and
             ledger_coverage.get("per_dof_source_rows") == 128 and
             ledger_coverage.get("source_contributions_per_dof") == 6,
             "force ledger coverage is incomplete")
    ledger_residual = ledger.get("residual", {})

    stand = documents["passive_stand"]
    _require(stand.get("schema") == "numi.human.native-passive-stand-current-requalification.v1",
             "passive stand schema changed")
    stand_source = _source(stand, "passive stand")
    stand_qualification = stand.get("qualification", {})
    for key in ("bounded_exact_clock_release", "complete_static_generalized_balance",
                "source_support_load_and_replay"):
        _require(stand_qualification.get(key) is True, f"passive stand lacks {key}")
    for key in ("force_convergence", "anatomical_support_loading", "activation_calibration",
                "blood_mass_transfer", "material_calibration", "subject_calibration",
                "sustained_standing", "perturbation_recovery", "walking"):
        _require(stand_qualification.get(key) is False, f"passive stand boundary changed for {key}")
    stand_results = stand.get("results", {})
    _require(stand_results.get("persistent_completed_steps") == 512 and
             stand_results.get("persistent_max_penetration_m") == 0.0 and
             stand_results.get("compiled_stand_active_support_contacts") == 6 and
             stand_results.get("stand_deterministic_replay") == "bitwise",
             "passive stand release evidence is incomplete")

    regional = documents["regional_exchange"]
    _require(regional.get("schema") in {
        "HumanPack.native-human-regional-exchange-current-requalification.v1",
        "HumanPack.native-human-regional-exchange-current-requalification.v2",
    },
             "regional exchange schema changed")
    regional_source = _source(regional, "regional exchange")
    regional_qualification = regional.get("qualification", {})
    for key in ("current_native_replay", "regional_blood_transport", "oxygen_amount_exchange",
                "accepted_step_conservation"):
        _require(regional_qualification.get(key) is True, f"regional exchange lacks {key}")
    for key in ("anatomical_vessel_lumen", "physical_tissue_volume_owner",
                "mechanical_blood_mass_owner", "organ_mechanics", "material_calibration",
                "subject_calibration", "standing_walking"):
        _require(regional_qualification.get(key) is False,
                 f"regional exchange boundary changed for {key}")
    regional_results = regional.get("results", {})
    _require(regional_results.get("timestep_nanoseconds") == 12500 and
             regional_results.get("source_compartment_count") == 21 and
             regional_results.get("source_connection_count") == 24 and
             regional_results.get("regional_bed_count") == 7 and
             regional_results.get("accepted_steps_environment_0") == 511 and
             regional_results.get("replay") == "bitwise" and
             regional_results.get("rollback") == "bitwise",
             "regional exchange exact-clock evidence is incomplete")

    dynamic = documents["dynamic_audit"]
    _require(dynamic.get("schema") == "numi.human.native-dynamic-force-audit-requalification.v1"
             and dynamic.get("status") == "partial",
             "dynamic force audit schema or status changed")
    dynamic_source = _source(dynamic, "dynamic force audit")
    dynamic_qualification = dynamic.get("qualification", {})
    _require(dynamic_qualification.get("source_identity_bound") is True and
             dynamic_qualification.get("coordinate_identity_bound") is True and
             dynamic_qualification.get("full_128_dof_component_rows") is True and
             dynamic_qualification.get("dynamic_component_reconstruction") is True and
             dynamic_qualification.get("dynamic_release") is False and
             dynamic_qualification.get("force_convergence") is False and
             dynamic_qualification.get("activation_calibration") is False and
             dynamic_qualification.get("anatomical_support_loading") is False,
             "dynamic force audit boundary changed")
    dynamic_results = dynamic.get("results", {})
    _require(dynamic_results.get("body_count") == 157 and
             dynamic_results.get("dof_count") == 128 and
             dynamic_results.get("persistent_completed_steps") == 64 and
             dynamic_results.get("source_support_active_contacts") == 6 and
             dynamic_results.get("persistent_max_penetration_m") == 0.0 and
             dynamic_results.get("dynamic_initial_max_abs_residual_n") == 0.03216604835060366 and
             dynamic_results.get("source_dynamic_force_parity_max_delta_n") == 0.0321654636734 and
             dynamic_results.get("persistent_max_acceleration_mps2") == 0.115904301405,
             "dynamic force audit metrics changed")

    mass_transfer = documents["blood_mass_transfer"]
    _require(mass_transfer.get("schema") == "HumanPack.organ-blood-mass-transfer-candidate.v1"
             and mass_transfer.get("status") == "partial" and
             mass_transfer.get("clock", {}).get("nanoseconds") == 12500 and
             mass_transfer.get("counts", {}).get("bed_count") == 7 and
             mass_transfer.get("counts", {}).get("source_blood_owner_count") == 6 and
             mass_transfer.get("accepted_steps") == 511 and
             mass_transfer.get("rejected_steps") == 1 and
             mass_transfer.get("conservation", {}).get("mass_conserved") is True and
             mass_transfer.get("conservation", {}).get("volume_conserved") is True and
             mass_transfer.get("rollback", {}).get("rejected_candidate_state_neutral") is True and
             mass_transfer.get("transfer_counts", {}).get("blood_to_tissue", 0) > 0 and
             mass_transfer.get("transfer_counts", {}).get("tissue_to_blood", 0) > 0,
             "regional blood/tissue mass-transfer evidence changed")
    mass_transfer_qualification = mass_transfer.get("qualification", {})
    _require(mass_transfer_qualification.get("blood_tissue_zeroth_moment_mass_transfer") is True and
             mass_transfer_qualification.get("mechanical_blood_mass_owner") is False and
             mass_transfer_qualification.get("mechanical_tissue_mass_owner") is False and
             mass_transfer_qualification.get("anatomical_exchange_owner") is False and
             mass_transfer_qualification.get("anatomical_vessel_lumen") is False,
             "regional blood/tissue mass-transfer qualification boundary changed")

    for label, source in (("force audit", force_source), ("passive stand", stand_source),
                          ("regional exchange", regional_source),
                          ("dynamic force audit", dynamic_source)):
        _require(source["device"] == "Mac mini M4 Pro", f"{label} is not physical-Mac evidence")
    _require(force_source["commit"] == stand_source["commit"] == regional_source["commit"],
             "native source commits diverge")
    _require(force_source["branch"] == stand_source["branch"], "mechanics source branches diverge")
    _require(force_source["binary_sha256"] == stand_source["binary_sha256"],
             "mechanics binary identities diverge")

    qualification = {
        "source_identity_graph_bound": True,
        "static_generalized_force_closure": True,
        "full_128_dof_force_ledger": True,
        "per_dof_source_audit": True,
        "bounded_passive_release": True,
        "exact_clock": True,
        "dynamic_force_component_audit": True,
        "regional_blood_transport": True,
        "oxygen_amount_exchange": True,
        "blood_tissue_mass_transfer_candidate": True,
        "accepted_step_conservation": True,
        "bitwise_replay": True,
        "anatomical_supports_loading": False,
        "dynamic_foot_contact": False,
        "force_convergence": False,
        "activation_calibration": False,
        "anatomical_blood_mass_transfer": False,
        "mechanical_blood_mass_owner": False,
        "organ_mechanics": False,
        "fat_geometry_and_mass": False,
        "skeletal_muscle_tissue_volume": False,
        "material_calibration": False,
        "subject_calibration": False,
        "sustained_standing": False,
        "recovery": False,
        "walking": False,
        "integrated_human_qualification": False,
    }
    blockers = [
        {"id": "force_convergence", "status": "open",
         "evidence": _relative(force_ledger),
         "reason": "Static rows close, but common-duration force/state refinement is not qualified."},
        {"id": "anatomical_supports_loading", "status": "open",
         "evidence": _relative(paths["body_composition"]),
         "reason": "Foot meshes and support witnesses are source-registered; colliders, exclusions, friction and calibrated loading are absent."},
        {"id": "activation_calibration", "status": "open",
         "evidence": _relative(paths["body_composition"]),
         "reason": "Recruitment is source-bound, but measured activation and held-out force validation are absent."},
        {"id": "blood_mass_transfer", "status": "open",
         "evidence": _relative(paths["blood_mass_transfer"]),
         "reason": "Regional zeroth-moment blood/tissue mass transfer conserves on the exact clock in both directions; anatomical lumen and mechanical blood ownership are absent."},
        {"id": "materials_and_subject_calibration", "status": "open",
         "evidence": _relative(paths["body_composition"]),
         "reason": "Candidate tissue fits and source densities remain uncalibrated and are not admitted to production mechanics."},
        {"id": "standing_recovery_walking", "status": "open",
         "evidence": _relative(paths["passive_stand"]),
         "reason": "The 512-step release is bounded and replayable, but sustained standing, perturbation recovery and walking are not demonstrated."},
    ]
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.current-human-evidence-join.1",
        "status": "partial",
        "subject": "one adult male source package",
        "inputs": {name: _input(paths[name], documents[name], hashes[name]) for name in paths},
        "native_owner": {
            "device": force_source["device"],
            "branch": force_source["branch"],
            "commit": force_source["commit"],
            "binary_sha256": force_source["binary_sha256"],
            "source_identity_agreement": True,
        },
        "clock": {
            "nanoseconds": 12500,
            "seconds": 1.25e-5,
            "regional_transport_exact": True,
            "mechanics_exact": True,
        },
        "static_closure": {
            "body_count": force_results["body_count"],
            "generalized_dof_count": ledger_coverage["nv"],
            "source_rows": ledger_coverage["per_dof_source_rows"],
            "sources_per_dof": ledger_coverage["source_contributions_per_dof"],
            "maximum_absolute_force_residual_n": force_results["maximum_absolute_force_residual"],
            "maximum_assembly_error": ledger_residual["maximum_assembly_error"],
            "maximum_closure_ratio": ledger_residual["maximum_closure_ratio"],
            "internal_normalized_residual_rms": force_results["internal_normalized_residual_rms"],
            "root_force_residual_n": force_results["max_root_force_residual_n"],
            "active_support_contacts": force_results["static_active_support_contacts"],
        },
        "bounded_release": {
            "completed_steps": stand_results["persistent_completed_steps"],
            "peak_acceleration_mps2": stand_results["persistent_max_acceleration_mps2"],
            "maximum_penetration_m": stand_results["persistent_max_penetration_m"],
            "active_support_contacts": stand_results["compiled_stand_active_support_contacts"],
            "replay": stand_results["stand_deterministic_replay"],
        },
        "regional_transport": {
            "source_compartment_count": regional_results["source_compartment_count"],
            "source_connection_count": regional_results["source_connection_count"],
            "regional_bed_count": regional_results["regional_bed_count"],
            "attempted_steps": regional_results["attempted_steps"],
            "accepted_steps": regional_results["accepted_steps_environment_0"],
            "rejected_steps": regional_results["rejected_step_environment_0"],
            "maximum_relative_volume_residual": regional_results["maximum_relative_volume_residual"],
            "maximum_relative_blood_mass_residual": regional_results["maximum_relative_blood_mass_residual"],
            "maximum_relative_oxygen_residual": regional_results["maximum_relative_oxygen_residual"],
            "rollback": regional_results["rollback"],
            "replay": regional_results["replay"],
        },
        "dynamic_force_diagnostic": {
            "completed_steps": dynamic_results["persistent_completed_steps"],
            "initial_max_abs_residual_n": dynamic_results["dynamic_initial_max_abs_residual_n"],
            "persistent_max_acceleration_mps2": dynamic_results["persistent_max_acceleration_mps2"],
            "source_dynamic_force_parity_max_delta_n": dynamic_results["source_dynamic_force_parity_max_delta_n"],
            "active_support_contacts": dynamic_results["source_support_active_contacts"],
            "maximum_penetration_m": dynamic_results["persistent_max_penetration_m"],
            "replay": dynamic_qualification["replay"],
        },
        "blood_tissue_mass_transfer": {
            "accepted_steps": mass_transfer["accepted_steps"],
            "rejected_steps": mass_transfer["rejected_steps"],
            "blood_to_tissue_transfers": mass_transfer["transfer_counts"]["blood_to_tissue"],
            "tissue_to_blood_transfers": mass_transfer["transfer_counts"]["tissue_to_blood"],
            "mass_conserved": mass_transfer["conservation"]["mass_conserved"],
            "volume_conserved": mass_transfer["conservation"]["volume_conserved"],
            "rollback": mass_transfer["rollback"]["rejected_candidate_state_neutral"],
        },
        "ownership": {
            "physical_owner_count": 0,
            "mechanical_blood_mass_owner_count": 0,
            "candidate_mass_budgets_admitted_to_dynamics": False,
        },
        "qualification": qualification,
        "blockers": blockers,
        "boundary": (
            "This receipt joins the current source-composition candidate, pose-24 native "
            "whole-body force audit, complete 128-DoF source ledger, pose-24 passive "
            "release, dynamic force-component diagnostic, exact-clock regional blood/oxygen "
            "replay, and the zeroth-moment blood/tissue mass-transfer candidate for one adult male. "
            "It proves source and runtime identity, static generalized closure, bounded "
            "release, regional amount/mass conservation and replay. It does not admit candidate "
            "tissue or blood mass to rigid-body dynamics and does not prove anatomical "
            "contact, activation calibration, anatomical blood transfer, organ mechanics, "
            "materials, subject calibration, sustained standing, recovery or walking. "
            "The dynamic component result is a diagnostic only; its short release does not "
            "qualify temporal force convergence."
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
    parser.add_argument("--body-composition", type=Path, default=BODY_COMPOSITION)
    parser.add_argument("--force-audit", type=Path, default=FORCE_AUDIT)
    parser.add_argument("--force-ledger", type=Path, default=FORCE_LEDGER)
    parser.add_argument("--passive-stand", type=Path, default=PASSIVE_STAND)
    parser.add_argument("--regional-exchange", type=Path, default=REGIONAL_EXCHANGE)
    parser.add_argument("--dynamic-audit", type=Path, default=DYNAMIC_AUDIT)
    parser.add_argument("--blood-mass-transfer", type=Path, default=BLOOD_MASS_TRANSFER)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = compile_join(
        body_composition=arguments.body_composition,
        force_audit=arguments.force_audit,
        force_ledger=arguments.force_ledger,
        passive_stand=arguments.passive_stand,
        regional_exchange=arguments.regional_exchange,
        dynamic_audit=arguments.dynamic_audit,
        blood_mass_transfer=arguments.blood_mass_transfer,
    )
    digest = immutable_write(arguments.output.resolve(), result)
    print(json.dumps({"schema": SCHEMA, "status": result["status"],
                      "output": str(arguments.output.resolve()), "sha256": digest,
                      "blockers": len(result["blockers"])}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    arguments = parser.parse_args(argv)
    try:
        return run(arguments)
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"current Human evidence join: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
