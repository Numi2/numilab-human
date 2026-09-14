"""Compile and step a source-bound regional blood/tissue exchange candidate.

The regional transport candidate already binds seven organ surface-volume
candidates to six CVSim21 blood owners.  This module adds one explicitly
unresolved oxygen amount to those same arterial, transit, venous and tissue
states.  Blood oxygen is advected with the source flow and a conservative
two-way amount exchange is applied in the transit bed.  The tissue volume,
concentration, partition and clearance values are engineering candidates;
they are not capillary, metabolic, material or subject measurements.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import organ_blood_tissue_transport as regional_transport
from . import tissue_mass_candidate
from .model import ImportError as HumanImportError
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.organ-tissue-exchange-candidate.v1"
PROFILE = ROOT / "config/organ-tissue-exchange-candidate.v1.json"
CLOCK_NANOSECONDS = 12_500
CLOCK_SECONDS = CLOCK_NANOSECONDS * 1.0e-9
MAX_STEPS = 1_000_000
REGIONS = regional_transport.REGIONS


class ExchangeError(HumanImportError):
    """A regional tissue exchange candidate cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExchangeError("organ tissue exchange: " + message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{label} is not finite")
    return float(value)


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(result > 0.0, f"{label} is not positive")
    return result


def _fraction(value: Any, label: str) -> float:
    result = _positive(value, label)
    _require(result <= 1.0, f"{label} exceeds one")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _read_canonical(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ExchangeError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _load_profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, profile_sha = _read_canonical(path, "exchange profile")
    required = {
        "schema", "id", "clock_nanoseconds", "initial_blood_oxygen_mol_per_m3",
        "initial_tissue_oxygen_mol_per_m3", "partition_coefficient",
        "clearance_fraction_of_source_flow", "parameter_provenance", "beds",
        "boundary",
    }
    _require(set(profile) == required, "exchange profile fields differ")
    _require(profile["schema"] == "numi.human.organ-tissue-exchange-candidate.v1",
             "unsupported exchange profile schema")
    _require(profile["id"] == "seven_bed_oxygen_exchange_candidate",
             "unsupported exchange profile")
    _require(profile["clock_nanoseconds"] == CLOCK_NANOSECONDS,
             "exchange profile clock differs")
    _positive(profile["initial_blood_oxygen_mol_per_m3"],
              "initial blood oxygen concentration")
    _positive(profile["initial_tissue_oxygen_mol_per_m3"],
              "initial tissue oxygen concentration")
    _positive(profile["partition_coefficient"], "partition coefficient")
    _fraction(profile["clearance_fraction_of_source_flow"],
              "clearance fraction of source flow")
    provenance = profile["parameter_provenance"]
    _require(
        isinstance(provenance, dict)
        and set(provenance) == {"kind", "description"}
        and provenance["kind"] == "engineering_candidate_unresolved"
        and isinstance(provenance["description"], str)
        and provenance["description"].strip(),
        "exchange parameter provenance must remain unresolved",
    )
    beds = profile["beds"]
    _require(isinstance(beds, list) and len(beds) == len(REGIONS),
             "exchange profile must contain seven beds")
    seen: set[str] = set()
    for bed in beds:
        _require(
            isinstance(bed, dict)
            and set(bed) == {"region_id", "clearance_fraction_of_source_flow"},
            "exchange bed fields differ",
        )
        region = bed["region_id"]
        _require(region in REGIONS and region not in seen,
                 f"exchange bed region is duplicated or unknown: {region}")
        seen.add(region)
        _fraction(bed["clearance_fraction_of_source_flow"],
                  f"{region} clearance fraction of source flow")
    _require(seen == set(REGIONS), "exchange beds do not cover seven regions")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "exchange profile boundary is missing")
    return profile, profile_sha


def _region_tissue_volumes(candidate: dict[str, Any]) -> dict[str, float]:
    volumes = {region: 0.0 for region in REGIONS}
    for row in candidate.get("candidates", []):
        region = row.get("region_id")
        if region not in volumes:
            continue
        if row.get("mass_admission") != "surface_candidate_only_physical_volume_unresolved":
            continue
        volume = row.get("candidate_surface_volume_m3")
        _require(type(volume) in (int, float) and math.isfinite(float(volume))
                 and float(volume) > 0.0,
                 f"{region} tissue candidate volume is invalid")
        volumes[region] += float(volume)
    _require(all(value > 0.0 for value in volumes.values()),
             "exchange tissue candidate volumes do not cover seven regions")
    return volumes


def compile_candidate(*, profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _load_profile(Path(profile))
    transport = regional_transport.compile_candidate()
    tissue = tissue_mass_candidate.compile_candidate()
    tissue_volumes = _region_tissue_volumes(tissue)
    bed_profiles = {row["region_id"]: row for row in profile_doc["beds"]}
    initial_blood_concentration = _positive(
        profile_doc["initial_blood_oxygen_mol_per_m3"],
        "initial blood oxygen concentration",
    )
    initial_tissue_concentration = _positive(
        profile_doc["initial_tissue_oxygen_mol_per_m3"],
        "initial tissue oxygen concentration",
    )
    partition = _positive(profile_doc["partition_coefficient"],
                          "partition coefficient")

    beds: list[dict[str, Any]] = []
    initial_state = copy.deepcopy(transport["initial_state"])
    for row in transport["beds"]:
        region = row["region_id"]
        tissue_volume = tissue_volumes[region]
        _require(math.isclose(
            tissue_volume, row["surface_candidate_volume_m3"],
            rel_tol=0.0, abs_tol=1.0e-18,
        ), f"{region} tissue volume disagrees with transport candidate")
        bed_profile = bed_profiles[region]
        fraction = _fraction(
            bed_profile["clearance_fraction_of_source_flow"],
            f"{region} clearance fraction of source flow",
        )
        state = initial_state["beds"][region]
        state.update({
            "arterial_oxygen_mol": initial_blood_concentration * state["arterial_volume_m3"],
            "transit_oxygen_mol": initial_blood_concentration * state["transit_volume_m3"],
            "venous_oxygen_mol": initial_blood_concentration * state["venous_volume_m3"],
            "tissue_oxygen_mol": initial_tissue_concentration * tissue_volume,
        })
        beds.append({
            "region_id": region,
            "group_id": row["group_id"],
            "source_flow_m3_per_s": row["source_flow_m3_per_s"],
            "tissue_candidate_volume_m3": tissue_volume,
            "clearance_fraction_of_source_flow": fraction,
            "partition_coefficient": partition,
            "physical_tissue_volume_owner": None,
            "tissue_exchange_owner": None,
        })

    beds.sort(key=lambda row: row["region_id"])
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-tissue-exchange-candidate.1",
        "status": "partial",
        "source": {
            "transport_schema": transport["schema"],
            "transport_source": transport["source"],
            "transport_candidate_sha256": _digest(transport),
            "tissue_mass_schema": tissue["schema"],
            "tissue_mass_source": tissue["source"],
            "tissue_mass_candidate_sha256": _digest(tissue),
            "exchange_profile": _relative(Path(profile)),
            "exchange_profile_sha256": profile_sha,
        },
        "clock": {
            "nanoseconds": CLOCK_NANOSECONDS,
            "seconds": CLOCK_SECONDS,
            "source": "exchange_profile_and_regional_transport_exact_clock",
        },
        "counts": {
            "bed_count": len(beds),
            "region_count": len(beds),
            "source_blood_owner_count": transport["counts"]["source_blood_owner_count"],
            "source_members_bound": transport["counts"]["source_members_bound"],
            "tissue_candidate_volume_owner_count": 0,
            "tissue_exchange_owner_count": 0,
        },
        "parameter_provenance": profile_doc["parameter_provenance"],
        "beds": beds,
        "initial_state": initial_state,
        "qualification": {
            "regional_blood_transport_bound": True,
            "regional_tissue_surface_volume_bound": True,
            "oxygen_amount_advected_with_source_flow": True,
            "bidirectional_oxygen_amount_exchange_candidate": True,
            "oxygen_amount_conservation_and_rollback": True,
            "anatomical_capillary_or_lumen": False,
            "physical_tissue_volume_owner": False,
            "tissue_exchange_owner": False,
            "metabolic_reaction": False,
            "respiratory_gas_exchange": False,
            "material_calibration": False,
            "subject_calibration": False,
            "organ_mechanics": False,
            "standing_walking": False,
        },
        "boundary": (
            "Seven source organ surface-volume candidates receive a conservative "
            "oxygen amount that is advected through the regional arterial/transit/"
            "venous path and exchanged bidirectionally with an unresolved tissue "
            "candidate. This is a source-bound amount/conservation subgate only; "
            "it does not assign a capillary lumen, tissue volume authority, gas or "
            "metabolic law, material, organ mechanics, or subject physiology."
        ),
    }


def _state_totals(state: dict[str, Any]) -> dict[str, float]:
    return {
        "volume_m3": math.fsum(
            bed[key]
            for bed in state["beds"].values()
            for key in ("arterial_volume_m3", "transit_volume_m3", "venous_volume_m3")
        ),
        "mass_kg": math.fsum(
            bed[key]
            for bed in state["beds"].values()
            for key in ("arterial_mass_kg", "transit_mass_kg", "venous_mass_kg")
        ),
        "oxygen_mol": math.fsum(
            bed[key]
            for bed in state["beds"].values()
            for key in (
                "arterial_oxygen_mol", "transit_oxygen_mol",
                "venous_oxygen_mol", "tissue_oxygen_mol",
            )
        ),
    }


def _state_digest(state: dict[str, Any]) -> str:
    return _digest(state)


def _step(compiled: dict[str, Any], state: dict[str, Any],
          timestep_s: float) -> tuple[dict[str, Any], int]:
    _require(math.isfinite(timestep_s) and timestep_s > 0.0,
             "timestep must be positive")
    candidate = copy.deepcopy(state)
    transfers = 0
    for row in compiled["beds"]:
        bed = candidate["beds"][row["region_id"]]
        volume = row["source_flow_m3_per_s"] * timestep_s
        _require(
            volume < bed["arterial_volume_m3"]
            and volume < bed["transit_volume_m3"],
            f"{row['region_id']} candidate step empties a compartment",
        )

        arterial_density = bed["arterial_mass_kg"] / bed["arterial_volume_m3"]
        incoming_mass = volume * arterial_density
        arterial_oxygen_concentration = (
            bed["arterial_oxygen_mol"] / bed["arterial_volume_m3"]
        )
        incoming_oxygen = volume * arterial_oxygen_concentration
        bed["arterial_volume_m3"] -= volume
        bed["transit_volume_m3"] += volume
        bed["arterial_mass_kg"] -= incoming_mass
        bed["transit_mass_kg"] += incoming_mass
        bed["arterial_oxygen_mol"] -= incoming_oxygen
        bed["transit_oxygen_mol"] += incoming_oxygen

        transit_density = bed["transit_mass_kg"] / bed["transit_volume_m3"]
        outgoing_mass = volume * transit_density
        transit_oxygen_concentration = (
            bed["transit_oxygen_mol"] / bed["transit_volume_m3"]
        )
        outgoing_oxygen = volume * transit_oxygen_concentration
        bed["transit_volume_m3"] -= volume
        bed["venous_volume_m3"] += volume
        bed["transit_mass_kg"] -= outgoing_mass
        bed["venous_mass_kg"] += outgoing_mass
        bed["transit_oxygen_mol"] -= outgoing_oxygen
        bed["venous_oxygen_mol"] += outgoing_oxygen
        transfers += 2

        tissue_volume = row["tissue_candidate_volume_m3"]
        clearance = row["source_flow_m3_per_s"] * row["clearance_fraction_of_source_flow"]
        blood_concentration = bed["transit_oxygen_mol"] / bed["transit_volume_m3"]
        tissue_concentration = bed["tissue_oxygen_mol"] / tissue_volume
        transfer = timestep_s * clearance * (
            blood_concentration - tissue_concentration / row["partition_coefficient"]
        )
        if transfer > 0.0:
            transfer = min(transfer, bed["transit_oxygen_mol"])
        else:
            transfer = -min(-transfer, bed["tissue_oxygen_mol"])
        bed["transit_oxygen_mol"] -= transfer
        bed["tissue_oxygen_mol"] += transfer
        transfers += 1

        for key, value in bed.items():
            if key.endswith("_volume_m3"):
                _require(math.isfinite(value) and value > 0.0,
                         f"{row['region_id']} volume left the positive domain")
            elif key.endswith("_mass_kg") or key.endswith("_oxygen_mol"):
                _require(math.isfinite(value) and value >= 0.0,
                         f"{row['region_id']} amount left the nonnegative domain")
    return candidate, transfers


def simulate(compiled: dict[str, Any], *, steps: int = 512,
             timestep_s: float = CLOCK_SECONDS,
             reject_step: int | None = None) -> dict[str, Any]:
    _require(type(steps) is int and 1 <= steps <= MAX_STEPS, "steps out of range")
    _require(type(timestep_s) in (int, float) and math.isfinite(timestep_s)
             and timestep_s > 0.0, "timestep must be positive")
    if reject_step is not None:
        _require(type(reject_step) is int and 1 <= reject_step <= steps,
                 "reject_step is outside attempted steps")
    _require(abs(float(timestep_s) - CLOCK_SECONDS) <= 1.0e-15,
             "timestep is not the canonical 12.5 us clock")
    state = copy.deepcopy(compiled["initial_state"])
    initial = copy.deepcopy(state)
    initial_totals = _state_totals(state)
    accepted_hashes: list[str] = []
    rejected = 0
    transfer_count = 0
    for attempt in range(1, steps + 1):
        before = copy.deepcopy(state)
        candidate, transfers = _step(compiled, state, float(timestep_s))
        if reject_step == attempt:
            state = before
            rejected += 1
            continue
        state = candidate
        transfer_count += transfers
        accepted_hashes.append(_state_digest(state))
    final_totals = _state_totals(state)
    mass_residual = final_totals["mass_kg"] - initial_totals["mass_kg"]
    volume_residual = final_totals["volume_m3"] - initial_totals["volume_m3"]
    oxygen_residual = final_totals["oxygen_mol"] - initial_totals["oxygen_mol"]
    return {
        "schema": SCHEMA,
        "candidate_schema": compiled["schema"],
        "attempted_steps": steps,
        "accepted_steps": steps - rejected,
        "rejected_steps": rejected,
        "timestep_seconds": float(timestep_s),
        "initial_state": initial,
        "final_state": state,
        "initial_totals": initial_totals,
        "final_totals": final_totals,
        "transfer_count": transfer_count,
        "conservation": {
            "mass_residual_kg": mass_residual,
            "volume_residual_m3": volume_residual,
            "oxygen_residual_mol": oxygen_residual,
            "mass_conserved": abs(mass_residual) <= 1.0e-12,
            "volume_conserved": abs(volume_residual) <= 1.0e-16,
            "oxygen_conserved": abs(oxygen_residual) <= 1.0e-17,
        },
        "rollback": {
            "requested": reject_step is not None,
            "accepted_time_s": (steps - rejected) * float(timestep_s),
            "rejected_candidate_state_neutral": True,
        },
        "accepted_state_trace_sha256": _digest(accepted_hashes),
        "qualification": compiled["qualification"],
        "boundary": compiled["boundary"],
    }


def _write_immutable(path: Path, value: dict[str, Any]) -> str:
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
    compiled = compile_candidate(profile=arguments.profile)
    receipt = simulate(compiled, steps=arguments.steps,
                       timestep_s=arguments.timestep_seconds,
                       reject_step=arguments.reject_step)
    for key in ("source", "clock", "counts", "beds", "parameter_provenance"):
        receipt[key] = compiled[key]
    output = arguments.output.resolve()
    receipt_sha = _write_immutable(output, receipt)
    print(json.dumps({
        "schema": SCHEMA,
        "output": str(output),
        "sha256": receipt_sha,
        "accepted_steps": receipt["accepted_steps"],
        "rejected_steps": receipt["rejected_steps"],
        "transfer_count": receipt["transfer_count"],
        "mass_conserved": receipt["conservation"]["mass_conserved"],
        "volume_conserved": receipt["conservation"]["volume_conserved"],
        "oxygen_conserved": receipt["conservation"]["oxygen_conserved"],
        "qualification": "partial",
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--timestep-seconds", type=float, default=CLOCK_SECONDS)
    parser.add_argument("--reject-step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"organ-tissue-exchange: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
