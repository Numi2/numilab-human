"""Join the current Human organ, blood, tissue, fat and muscle handoffs.

This compiler is deliberately an integration boundary, not a physical-body
qualification.  It proves that the source identities can be joined without
duplicating an owner, and it keeps every missing physical volume, mechanical
mass, material, and calibration authority explicit.
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
SCHEMA = "HumanPack.body-composition-integration-candidate.v1"
PROFILE = ROOT / "config/body-composition-integration.v1.json"

ORGAN_MASS = ROOT / "Docs/media/tissue-mass-candidate-20260914/receipt-v2.json"
REGIONAL_TISSUE = ROOT / "Docs/media/regional-tissue-mass-candidate-20260914/receipt-v1.json"
SURFACES = ROOT / "Docs/media/soft-tissue-surface-candidate-20260914/receipt-v1.json"
ACTIVATION = ROOT / "Docs/media/activation-recruitment-candidate-20260914/receipt-v1.json"
BLOOD_TRANSPORT = ROOT / "Docs/media/organ-blood-tissue-transport-20260914/receipt-v1.json"
TISSUE_EXCHANGE = ROOT / "Docs/media/organ-tissue-exchange-candidate-20260914/receipt-v1.json"
CARDIAC_BLOOD = ROOT / "Docs/media/cardiac-blood-mass-candidate-20260914/receipt.json"
CVSIM21_BLOOD_MASS = ROOT / "Docs/media/cvsim21-blood-mass-step-20260914/receipt-exact-clock.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("body composition integration: " + message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _identity_digest(values: set[str]) -> str:
    return _sha256(canonical(sorted(values)))


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read(path: Path, label: str) -> tuple[dict[str, Any], str]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    value = read_json(path)
    _require(isinstance(value, dict), f"{label} is not an object")
    # The upstream receipts are immutable, source-bound JSON records, but some
    # were intentionally published pretty-printed.  Preserve their exact file
    # digest instead of rewriting or rejecting a valid receipt on formatting.
    return value, _sha256(raw)


def _schema(document: dict[str, Any], expected: str, label: str) -> None:
    _require(document.get("schema") == expected,
             f"{label} schema is {document.get('schema')!r}, expected {expected!r}")


def _read_profile(path: Path) -> dict[str, Any]:
    path = Path(path)
    _require(path.is_file() and not path.is_symlink(), "integration profile is not a regular file")
    raw = path.read_bytes()
    value = read_json(path)
    _require(raw == canonical(value) + b"\n", "integration profile is not canonical")
    _require(set(value) == {
        "schema", "id", "clock_nanoseconds", "expected_source_member_layers",
        "expected_runtime", "inputs", "boundary",
    }, "integration profile fields differ")
    _require(value.get("schema") == "numi.human.body-composition-integration.v1",
             "unsupported integration profile schema")
    _require(value.get("id") == "one_adult_male_cross_domain_source_join",
             "unsupported integration profile")
    _require(value.get("clock_nanoseconds") == 12500,
             "integration profile clock differs")
    _require(value.get("expected_source_member_layers") == {
        "organ_surface_candidates": 378,
        "regional_blood_transport": 329,
        "muscle_tendon_surface_identity": 150,
    }, "integration profile source counts differ")
    _require(value.get("expected_runtime") == {
        "blood_transport_accepted_steps": 511,
        "blood_transport_rejected_steps": 1,
        "cvsim21_mass_accepted_steps": 511,
        "cvsim21_mass_rejected_steps": 1,
    }, "integration profile runtime counts differ")
    _require(value.get("inputs") == [
        "organ_mass", "regional_tissue", "muscle_surfaces", "activation",
        "blood_transport", "tissue_exchange", "cardiac_blood",
        "cvsim21_blood_mass",
    ], "integration profile input order differs")
    return value


def _input(path: Path, document: dict[str, Any], file_sha256: str) -> dict[str, Any]:
    return {
        "path": _relative(path),
        "schema": document["schema"],
        "file_sha256": file_sha256,
    }


def _all_none(rows: list[dict[str, Any]], keys: tuple[str, ...], label: str) -> None:
    for row in rows:
        for key in keys:
            _require(row.get(key) is None,
                     f"{label} assigns {key} for {row.get('member_id', row.get('region_id'))}")


def compile_candidate(
    *,
    profile: Path = PROFILE,
    organ_mass: Path = ORGAN_MASS,
    regional_tissue: Path = REGIONAL_TISSUE,
    surfaces: Path = SURFACES,
    activation: Path = ACTIVATION,
    blood_transport: Path = BLOOD_TRANSPORT,
    tissue_exchange: Path = TISSUE_EXCHANGE,
    cardiac_blood: Path = CARDIAC_BLOOD,
    cvsim21_blood_mass: Path = CVSIM21_BLOOD_MASS,
) -> dict[str, Any]:
    profile = Path(profile)
    profile_document = _read_profile(profile)
    paths = {
        "organ_mass": Path(organ_mass),
        "regional_tissue": Path(regional_tissue),
        "muscle_surfaces": Path(surfaces),
        "activation": Path(activation),
        "blood_transport": Path(blood_transport),
        "tissue_exchange": Path(tissue_exchange),
        "cardiac_blood": Path(cardiac_blood),
        "cvsim21_blood_mass": Path(cvsim21_blood_mass),
    }
    documents: dict[str, dict[str, Any]] = {}
    hashes: dict[str, str] = {}
    labels = {
        "organ_mass": "organ mass candidate",
        "regional_tissue": "regional tissue candidate",
        "muscle_surfaces": "muscle surface candidate",
        "activation": "activation candidate",
        "blood_transport": "blood transport candidate",
        "tissue_exchange": "tissue exchange candidate",
        "cardiac_blood": "cardiac blood candidate",
        "cvsim21_blood_mass": "CVSim21 blood mass receipt",
    }
    for name, path in paths.items():
        documents[name], hashes[name] = _read(path, labels[name])

    _schema(documents["organ_mass"], "HumanPack.tissue-mass-composition-candidate.v1", labels["organ_mass"])
    _schema(documents["regional_tissue"], "HumanPack.regional-tissue-mass-candidate.v1", labels["regional_tissue"])
    _schema(documents["muscle_surfaces"], "HumanPack.soft-tissue-surface-candidate.v1", labels["muscle_surfaces"])
    _schema(documents["activation"], "HumanPack.activation-recruitment-candidate.v1", labels["activation"])
    _schema(documents["blood_transport"], "HumanPack.organ-blood-tissue-transport-candidate.v1", labels["blood_transport"])
    _schema(documents["tissue_exchange"], "HumanPack.organ-tissue-exchange-candidate.v1", labels["tissue_exchange"])
    _schema(documents["cardiac_blood"], "HumanPack.cardiac-blood-mass-candidate.v1", labels["cardiac_blood"])

    organ = documents["organ_mass"]
    organ_counts = organ.get("counts", {})
    _require(organ_counts == {
        "organ_surface_mass_candidates": 342,
        "region_count": 18,
        "shared_source_members": 8,
        "source_members": 378,
        "unresolved_members": 36,
    }, "organ mass inventory counts changed")
    organ_rows = organ.get("candidates")
    _require(isinstance(organ_rows, list) and len(organ_rows) == 378,
             "organ mass candidate rows are incomplete")
    organ_ids = [row.get("member_id") for row in organ_rows]
    _require(all(isinstance(value, str) and value for value in organ_ids),
             "organ mass member identity is invalid")
    _require(len(set(organ_ids)) == len(organ_ids), "organ mass members repeat")
    _all_none(organ_rows, ("mechanical_mass_owner", "physical_volume_owner"), "organ candidate")
    _require(organ.get("totals", {}).get("physical_mass_owner_count") == 0 and
             organ.get("qualification", {}).get("interdomain_disjointness_qualified") is False,
             "organ candidate promoted a physical mass owner")

    regional = documents["regional_tissue"]
    regional_ownership = regional.get("ownership", {})
    _require(regional.get("qualification", {}).get("mass_conservation") is True and
             regional.get("native_evidence", {}).get("metal_device") == "Apple M4 Pro",
             "regional tissue candidate lacks native conservation evidence")
    for key in (
        "production_mechanical_mass_owner", "whole_body_dynamic_mass_matrix_owner",
        "fat_mass_owner", "skeletal_muscle_tissue_partition_owner", "blood_mass_owner",
    ):
        _require(regional_ownership.get(key) is False,
                 f"regional tissue candidate promoted {key}")
    regional_mass = regional.get("region", {}).get("tissue_mass_kg")
    _require(isinstance(regional_mass, (int, float)) and regional_mass > 0.0,
             "regional tissue mass is missing")

    surface = documents["muscle_surfaces"]
    surface_counts = surface.get("counts", {})
    _require(surface_counts.get("source_route_count") == 416 and
             surface_counts.get("muscle_surface_count") == 148 and
             surface_counts.get("tendon_surface_count") == 2 and
             surface_counts.get("routes_without_surface_binding") == 238,
             "muscle surface coverage changed")
    surface_rows = surface.get("surface_rows")
    route_rows = surface.get("route_rows")
    _require(isinstance(surface_rows, list) and len(surface_rows) == 150,
             "muscle/tendon surface rows are incomplete")
    _require(isinstance(route_rows, list) and len(route_rows) == 416,
             "muscle route rows are incomplete")
    route_ids = [row.get("source_actuator_index") for row in route_rows]
    _require(route_ids == list(range(416)), "muscle route identity table is not complete")
    _all_none(route_rows, ("mechanical_mass_owner", "physical_volume_owner",
                           "volumetric_active_force_owner"), "muscle route")
    _require(surface_counts.get("fat_surface_count") == 0 and
             surface_counts.get("skin_surface_count") == 0 and
             surface_counts.get("physical_volume_owner_count") == 0 and
             surface_counts.get("volumetric_active_force_owner_count") == 0,
             "fat/skin or active muscle ownership was promoted")
    _require(surface_counts.get("routes_with_surface_binding") == 178,
             "muscle route binding count changed")

    act = documents["activation"]
    act_counts = act.get("counts", {})
    _require(act_counts.get("source_routes") == 416 and
             act_counts.get("nonzero_routes") == 237 and
             act_counts.get("support_witnesses") == 18,
             "activation route counts changed")
    activation_vector = act.get("recruitment", {}).get("activation_fp64")
    _require(isinstance(activation_vector, list) and len(activation_vector) == 416,
             "activation vector is incomplete")
    _require(act.get("qualification", {}).get("activation_calibration") is False and
             act.get("qualification", {}).get("held_out_force_validation") is False,
             "activation candidate claims calibration")

    transport = documents["blood_transport"]
    transport_counts = transport.get("counts", {})
    _require(transport_counts.get("bed_count") == 7 and
             transport_counts.get("source_blood_owner_count") == 6 and
             transport_counts.get("source_members_bound") == 329 and
             transport.get("clock", {}).get("nanoseconds") == 12500 and
             transport.get("accepted_steps") == 511 and
             transport.get("rejected_steps") == 1,
             "blood transport receipt is not the exact-clock seven-bed run")
    _require(transport.get("conservation", {}).get("mass_conserved") is True and
             transport.get("rollback", {}).get("rejected_candidate_state_neutral") is True,
             "blood transport conservation/rollback is not qualified")
    transport_beds = transport.get("beds")
    _require(isinstance(transport_beds, list) and len(transport_beds) == 7,
             "blood transport beds are incomplete")
    blood_member_ids = [member_id for bed in transport_beds
                        for member_id in bed.get("source_member_ids", [])]
    _require(len(blood_member_ids) == 329 and len(set(blood_member_ids)) == 329,
             "blood transport source members repeat")
    organ_id_set = set(organ_ids)
    _require(set(blood_member_ids) <= organ_id_set,
             "blood transport references an organ member outside the source inventory")
    _all_none(transport_beds, ("mechanical_mass_owner", "physical_volume_owner",
                               "tissue_exchange_owner"), "blood transport bed")

    exchange = documents["tissue_exchange"]
    exchange_counts = exchange.get("counts", {})
    _require(exchange_counts.get("bed_count") == 7 and
             exchange_counts.get("source_blood_owner_count") == 6 and
             exchange_counts.get("source_members_bound") == 329 and
             exchange.get("clock", {}).get("nanoseconds") == 12500 and
             exchange.get("accepted_steps") == 511 and
             exchange.get("rejected_steps") == 1,
             "tissue exchange receipt is not the matching exact-clock run")
    exchange_beds = exchange.get("beds")
    _require(isinstance(exchange_beds, list) and
             {row.get("region_id") for row in exchange_beds} ==
             {row.get("region_id") for row in transport_beds},
             "tissue exchange beds do not match blood transport beds")
    _require(exchange.get("conservation", {}).get("oxygen_conserved") is True and
             exchange.get("rollback", {}).get("rejected_candidate_state_neutral") is True,
             "tissue exchange conservation/rollback is not qualified")
    for row in exchange_beds:
        _require(row.get("physical_tissue_volume_owner") is None and
                 row.get("tissue_exchange_owner") is None,
                 f"tissue exchange promoted an owner for {row.get('region_id')}")

    cardiac = documents["cardiac_blood"]
    _require(cardiac.get("selection") is None and
             cardiac.get("mass_budget", {}).get("both_candidates_share_hydraulic_budget") is True and
             cardiac.get("mass_budget", {}).get("mechanical_mass_owner_assigned") is False and
             cardiac.get("qualification", {}).get("two_way_blood_tissue_transfer") is False,
             "cardiac blood candidate boundary changed")
    cardiac_mass = cardiac.get("mass_budget", {}).get("candidate_mass_kg")
    _require(isinstance(cardiac_mass, (int, float)) and cardiac_mass > 0.0,
             "cardiac blood candidate mass is missing")

    cvsim21 = documents["cvsim21_blood_mass"]
    _require(cvsim21.get("model_id") == "cvsim21_supine_continuous_fixed_rate_upstream_equation" and
             cvsim21.get("attempted_steps") == 512 and
             cvsim21.get("accepted_steps") == 511 and
             cvsim21.get("rejected_steps") == 1 and
             cvsim21.get("clock", {}).get("exact") is True and
             cvsim21.get("clock", {}).get("required_nanoseconds") == 12500 and
             cvsim21.get("clock", {}).get("timestep_nanoseconds") == 12500.0,
             "CVSim21 blood mass receipt is not the exact-clock 512-step run")
    cvsim21_conservation = cvsim21.get("conservation", {})
    _require(cvsim21_conservation.get("mass_conserved") is True and
             cvsim21_conservation.get("volume_conserved") is True and
             abs(cvsim21_conservation.get("mass_residual_kg", float("inf"))) <= 1.0e-12 and
             abs(cvsim21_conservation.get("volume_residual_m3", float("inf"))) <= 1.0e-15,
             "CVSim21 blood mass/volume conservation is not qualified")
    _require(cvsim21.get("rollback", {}).get("accepted_time_excludes_rejections") is True and
             cvsim21.get("rollback", {}).get("rejected_steps") == 1,
             "CVSim21 blood mass rollback boundary changed")
    cvsim21_scope = cvsim21.get("scope", {})
    _require(cvsim21.get("qualification") == "source_absolute_blood_mass_candidate" and
             cvsim21_scope.get("source_aggregate_absolute_blood_mass") is True and
             cvsim21_scope.get("anatomical_registration") is False and
             cvsim21_scope.get("mechanical_mass_owner") is False and
             cvsim21_scope.get("tissue_exchange") is False and
             cvsim21_scope.get("material_density_calibrated") is False and
             cvsim21_scope.get("subject_calibration") is False,
             "CVSim21 blood mass candidate promoted an anatomical owner")
    cvsim21_mass = cvsim21_conservation.get("initial", {}).get("mass_kg")
    _require(isinstance(cvsim21_mass, (int, float)) and cvsim21_mass > 0.0,
             "CVSim21 source mass is missing")

    surface_member_ids = [row.get("member_id") for row in surface_rows]
    _require(all(isinstance(value, str) and value for value in surface_member_ids),
             "muscle/tendon surface identity is invalid")
    surface_ids = set(surface_member_ids)
    _require(len(surface_ids) == 150 and not surface_ids.intersection(organ_id_set),
             "muscle/tendon visual surfaces overlap organ member ownership")
    source_member_layers = {
        "organ_surface_candidates": len(organ_id_set),
        "regional_blood_transport": len(set(blood_member_ids)),
        "muscle_tendon_surface_identity": len(surface_ids),
    }

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.body-composition-integration.1",
        "status": "partial",
        "subject": "one adult male source package",
        "inputs": {
            "profile": _input(profile, profile_document, _sha256(profile.read_bytes())),
            **{
                name: _input(paths[name], documents[name], hashes[name])
                for name in paths
            },
        },
        "source_member_layers": source_member_layers,
        "identity_bindings": {
            "organ_member_ids_sha256": _identity_digest(organ_id_set),
            "blood_transport_member_ids_sha256": _identity_digest(set(blood_member_ids)),
            "muscle_tendon_surface_ids_sha256": _identity_digest(surface_ids),
            "blood_members_subset_of_organ_members": True,
            "surface_ids_disjoint_from_organ_members": True,
            "transport_and_exchange_beds_share_clock": (
                transport["clock"]["nanoseconds"] == exchange["clock"]["nanoseconds"]
            ),
        },
        "candidate_mass_budgets": {
            "organ_surface_candidate_mass_kg": organ["totals"]["candidate_surface_mass_kg"],
            "regional_costal_tissue_candidate_mass_kg": regional_mass,
            "cardiac_hydraulic_blood_candidate_mass_kg": cardiac_mass,
            "cvsim21_aggregate_blood_mass_kg": cvsim21_mass,
            "sum_is_mechanical_body_mass": False,
        },
        "ownership": {
            "organ_physical_volume_owner_count": 0,
            "blood_anatomical_physical_volume_owner_count": 0,
            "blood_mechanical_mass_owner_count": 0,
            "skeletal_muscle_tissue_volume_owner_count": 0,
            "fat_volume_and_mass_owner_count": 0,
            "skin_volume_and_mass_owner_count": 0,
            "tendon_fascia_volume_and_mass_owner_count": 0,
            "whole_body_dynamic_mass_matrix_owner_count": 0,
            "cross_domain_physical_owner_duplicates": 0,
        },
        "runtime_evidence": {
            "clock_nanoseconds": 12500,
            "blood_transport_accepted_steps": transport["accepted_steps"],
            "blood_transport_rejected_steps": transport["rejected_steps"],
            "cvsim21_mass_accepted_steps": cvsim21["accepted_steps"],
            "cvsim21_mass_rejected_steps": cvsim21["rejected_steps"],
            "cvsim21_mass_conserved": cvsim21_conservation["mass_conserved"],
            "cvsim21_volume_conserved": cvsim21_conservation["volume_conserved"],
            "blood_mass_conserved": transport["conservation"]["mass_conserved"],
            "tissue_oxygen_conserved": exchange["conservation"]["oxygen_conserved"],
            "rejected_step_state_neutral": (
                transport["rollback"]["rejected_candidate_state_neutral"] and
                exchange["rollback"]["rejected_candidate_state_neutral"]
            ),
            "muscle_activation_routes": act_counts["source_routes"],
            "muscle_surface_routes_with_binding": surface_counts["routes_with_surface_binding"],
            "muscle_surface_routes_without_binding": surface_counts["routes_without_surface_binding"],
        },
        "qualification": {
            "source_identity_graph_bound": True,
            "organ_surface_mass_candidate_bound": True,
            "regional_blood_transport_bound": True,
            "source_aggregate_blood_mass_bound": True,
            "tissue_oxygen_exchange_candidate_bound": True,
            "muscle_activation_route_identity_bound": True,
            "muscle_surface_identity_bound": True,
            "cross_domain_owner_nonduplication_checked": True,
            "anatomical_physical_volume_owners": False,
            "mechanical_mass_owners": False,
            "skeletal_muscle_tissue_volume": False,
            "fat_geometry_and_mass": False,
            "skin_geometry_and_mass": False,
            "anatomical_blood_mass_transfer": False,
            "organ_mechanics": False,
            "material_calibration": False,
            "subject_calibration": False,
            "integrated_human_qualification": False,
        },
        "boundary": (
            "This record joins source organ candidate moments, regional tissue "
            "mass, the exact-clock CVSim21 aggregate blood mass, regional blood "
            "transport, tissue oxygen exchange, cardiac blood budget, muscle "
            "route activation and NHTISS4 surface identity. It "
            "proves source identity and nonduplicated ownership bookkeeping only. "
            "Fat and skeletal-muscle tissue volumes, anatomical blood/lumen and "
            "organ mechanics, calibrated materials, subject calibration, and the "
            "whole-body mechanical owner remain unresolved. Candidate mass budgets "
            "must not be added to the rigid-body dynamics."
        ),
    }


def _immutable_write(path: Path, value: dict[str, Any]) -> str:
    payload = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output is redirected")
    if path.exists():
        _require(path.read_bytes() == payload,
                 "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def run(arguments: argparse.Namespace) -> int:
    result = compile_candidate(
        profile=arguments.profile,
        organ_mass=arguments.organ_mass,
        regional_tissue=arguments.regional_tissue,
        surfaces=arguments.surfaces,
        activation=arguments.activation,
        blood_transport=arguments.blood_transport,
        tissue_exchange=arguments.tissue_exchange,
        cardiac_blood=arguments.cardiac_blood,
        cvsim21_blood_mass=arguments.cvsim21_blood_mass,
    )
    output = arguments.output.resolve()
    digest = _immutable_write(output, result)
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "sha256": digest,
        "status": result["status"],
        "source_member_layers": result["source_member_layers"],
        "qualification": result["qualification"],
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--organ-mass", type=Path, default=ORGAN_MASS)
    parser.add_argument("--regional-tissue", type=Path, default=REGIONAL_TISSUE)
    parser.add_argument("--surfaces", type=Path, default=SURFACES)
    parser.add_argument("--activation", type=Path, default=ACTIVATION)
    parser.add_argument("--blood-transport", type=Path, default=BLOOD_TRANSPORT)
    parser.add_argument("--tissue-exchange", type=Path, default=TISSUE_EXCHANGE)
    parser.add_argument("--cardiac-blood", type=Path, default=CARDIAC_BLOOD)
    parser.add_argument("--cvsim21-blood-mass", type=Path, default=CVSIM21_BLOOD_MASS)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
