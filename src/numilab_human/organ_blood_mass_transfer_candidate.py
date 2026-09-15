"""Step a source-bound regional blood/tissue mass-transfer candidate.

The regional transport owner already partitions seven organ candidates across
six CVSim21 blood owners.  This module adds a separate zeroth-moment fluid
mass/volume state for those same beds and applies a signed, bounded exchange
between each transit compartment and its tissue candidate.  The exchange is
deliberately an engineering candidate: it does not claim an anatomical lumen,
capillary geometry, tissue mechanics, calibrated density, or subject data.
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
SCHEMA = "HumanPack.organ-blood-mass-transfer-candidate.v1"
PROFILE = ROOT / "config/organ-blood-mass-transfer.v1.json"
CLOCK_NANOSECONDS = 12_500
CLOCK_SECONDS = CLOCK_NANOSECONDS * 1.0e-9
MAX_STEPS = 1_000_000
REGIONS = regional_transport.REGIONS


class TransferError(HumanImportError):
    """A regional blood/tissue mass-transfer candidate cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TransferError("organ blood mass transfer: " + message)


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


def _read_canonical(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(), f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise TransferError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _load_profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, profile_sha = _read_canonical(Path(path), "mass-transfer profile")
    required = {"schema", "id", "clock_nanoseconds", "parameter_provenance", "beds", "boundary"}
    _require(set(profile) == required, "mass-transfer profile fields differ")
    _require(profile["schema"] == "numi.human.organ-blood-mass-transfer.v1",
             "unsupported mass-transfer profile schema")
    _require(profile["id"] == "seven_bed_blood_tissue_mass_transfer_candidate",
             "unsupported mass-transfer profile")
    _require(profile["clock_nanoseconds"] == CLOCK_NANOSECONDS,
             "mass-transfer profile clock differs")
    provenance = profile["parameter_provenance"]
    _require(isinstance(provenance, dict)
             and set(provenance) == {"kind", "description"}
             and provenance["kind"] == "engineering_candidate_unresolved"
             and isinstance(provenance["description"], str)
             and provenance["description"].strip(),
             "mass-transfer parameter provenance must remain unresolved")
    beds = profile["beds"]
    _require(isinstance(beds, list) and len(beds) == len(REGIONS),
             "mass-transfer profile must contain seven beds")
    seen: set[str] = set()
    for bed in beds:
        _require(isinstance(bed, dict)
                 and set(bed) == {"region_id", "exchange_fraction_of_source_flow", "phase_schedule"},
                 "mass-transfer bed fields differ")
        region = bed["region_id"]
        _require(region in REGIONS and region not in seen,
                 f"mass-transfer bed region is duplicated or unknown: {region}")
        seen.add(region)
        _fraction(bed["exchange_fraction_of_source_flow"],
                  f"{region} exchange fraction of source flow")
        schedule = bed["phase_schedule"]
        _require(isinstance(schedule, list) and schedule,
                 f"{region} phase schedule is empty")
        _require(all(type(value) is int and value in (-1, 1) for value in schedule),
                 f"{region} phase schedule must contain signed unit phases")
    _require(seen == set(REGIONS), "mass-transfer beds do not cover seven regions")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "mass-transfer profile boundary is missing")
    return profile, profile_sha


def _region_candidates(candidate: dict[str, Any]) -> dict[str, dict[str, float]]:
    result = {region: {"volume_m3": 0.0, "mass_kg": 0.0} for region in REGIONS}
    for row in candidate.get("candidates", []):
        region = row.get("region_id")
        if region not in result or row.get("mass_admission") != (
                "surface_candidate_only_physical_volume_unresolved"):
            continue
        volume = _positive(row.get("candidate_surface_volume_m3"),
                           f"{region} tissue candidate volume")
        mass = _positive(row.get("candidate_mass_kg"), f"{region} tissue candidate mass")
        result[region]["volume_m3"] += volume
        result[region]["mass_kg"] += mass
    _require(all(row["volume_m3"] > 0.0 and row["mass_kg"] > 0.0
                 for row in result.values()),
             "mass-transfer tissue candidates do not cover seven regions")
    return result


def compile_candidate(*, profile: Path = PROFILE) -> dict[str, Any]:
    profile_doc, profile_sha = _load_profile(Path(profile))
    transport = regional_transport.compile_candidate()
    tissue = tissue_mass_candidate.compile_candidate()
    regions = _region_candidates(tissue)
    profiles = {row["region_id"]: row for row in profile_doc["beds"]}

    beds: list[dict[str, Any]] = []
    initial_beds: dict[str, dict[str, float]] = {}
    for source in sorted(transport["beds"], key=lambda row: row["region_id"]):
        region = source["region_id"]
        tissue_row = regions[region]
        profile_row = profiles[region]
        initial = source["initial_state"]
        initial_beds[region] = {
            "arterial_volume_m3": initial["arterial_volume_m3"],
            "transit_volume_m3": initial["transit_volume_m3"],
            "venous_volume_m3": initial["venous_volume_m3"],
            "arterial_mass_kg": initial["arterial_mass_kg"],
            "transit_mass_kg": initial["transit_mass_kg"],
            "venous_mass_kg": initial["venous_mass_kg"],
            "tissue_volume_m3": tissue_row["volume_m3"],
            "tissue_mass_kg": tissue_row["mass_kg"],
        }
        beds.append({
            "region_id": region,
            "group_id": source["group_id"],
            "source_flow_m3_per_s": source["source_flow_m3_per_s"],
            "exchange_fraction_of_source_flow": profile_row["exchange_fraction_of_source_flow"],
            "phase_schedule": profile_row["phase_schedule"],
            "source_member_count": source["source_member_count"],
            "source_member_ids": source["source_member_ids"],
            "source_blood_owner_ids": sorted({source["arterial_source_owner_id"],
                                                source["venous_source_owner_id"]}),
            "tissue_candidate_volume_m3": tissue_row["volume_m3"],
            "tissue_candidate_mass_kg": tissue_row["mass_kg"],
            "mechanical_blood_mass_owner": None,
            "mechanical_tissue_mass_owner": None,
            "anatomical_exchange_owner": None,
        })

    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-blood-mass-transfer-candidate.1",
        "status": "partial",
        "source": {
            "regional_transport_schema": transport["schema"],
            "regional_transport_candidate_sha256": _digest(transport),
            "tissue_mass_schema": tissue["schema"],
            "tissue_mass_candidate_sha256": _digest(tissue),
            "profile": str(Path(profile).relative_to(ROOT)) if Path(profile).is_relative_to(ROOT) else str(profile),
            "profile_sha256": profile_sha,
        },
        "clock": {"nanoseconds": CLOCK_NANOSECONDS, "seconds": CLOCK_SECONDS,
                  "source": "regional_transport_and_mass_transfer_profile_exact_clock"},
        "counts": {"bed_count": len(beds), "region_count": len(beds),
                   "source_blood_owner_count": transport["counts"]["source_blood_owner_count"],
                   "source_members_bound": transport["counts"]["source_members_bound"]},
        "parameter_provenance": profile_doc["parameter_provenance"],
        "beds": beds,
        "initial_state": {"beds": initial_beds},
        "qualification": {
            "regional_blood_transport_bound": True,
            "regional_tissue_candidate_mass_bound": True,
            "blood_tissue_zeroth_moment_mass_transfer": True,
            "bidirectional_transfer_schedule": True,
            "mass_and_volume_conservation": True,
            "accepted_step_rollback": True,
            "first_second_spatial_moments": False,
            "anatomical_vessel_lumen": False,
            "physical_tissue_volume_owner": False,
            "mechanical_blood_mass_owner": False,
            "mechanical_tissue_mass_owner": False,
            "anatomical_exchange_owner": False,
            "material_density_calibrated": False,
            "subject_calibration": False,
            "organ_mechanics": False,
            "standing_walking": False,
        },
        "boundary": (
            "Seven source organ candidates share the regional arterial/transit/venous "
            "blood state with a separate tissue candidate state. A signed bounded "
            "exchange moves zeroth-moment fluid mass and volume both ways and restores "
            "rejected steps atomically. The candidate does not assign an anatomical "
            "lumen, capillary geometry, mechanical owner, calibrated material, or "
            "subject physiology."
        ),
    }


def _state_totals(state: dict[str, Any]) -> dict[str, float]:
    keys = ("arterial_volume_m3", "transit_volume_m3", "venous_volume_m3", "tissue_volume_m3")
    mass_keys = ("arterial_mass_kg", "transit_mass_kg", "venous_mass_kg", "tissue_mass_kg")
    return {
        "volume_m3": math.fsum(bed[key] for bed in state["beds"].values() for key in keys),
        "mass_kg": math.fsum(bed[key] for bed in state["beds"].values() for key in mass_keys),
        "blood_mass_kg": math.fsum(bed[key] for bed in state["beds"].values()
                                     for key in ("arterial_mass_kg", "transit_mass_kg", "venous_mass_kg")),
        "tissue_mass_kg": math.fsum(bed["tissue_mass_kg"] for bed in state["beds"].values()),
    }


def _step(compiled: dict[str, Any], state: dict[str, Any], timestep_s: float,
          accepted_index: int) -> tuple[dict[str, Any], dict[str, int]]:
    _require(math.isfinite(timestep_s) and timestep_s > 0.0, "timestep is not positive")
    candidate = copy.deepcopy(state)
    counts = {"connection": 0, "blood_to_tissue": 0, "tissue_to_blood": 0}
    for row in compiled["beds"]:
        bed = candidate["beds"][row["region_id"]]
        flow_volume = row["source_flow_m3_per_s"] * timestep_s
        _require(flow_volume < bed["arterial_volume_m3"]
                 and flow_volume < bed["transit_volume_m3"],
                 f"{row['region_id']} transport step empties a blood compartment")

        arterial_density = bed["arterial_mass_kg"] / bed["arterial_volume_m3"]
        arterial_mass = flow_volume * arterial_density
        bed["arterial_volume_m3"] -= flow_volume
        bed["transit_volume_m3"] += flow_volume
        bed["arterial_mass_kg"] -= arterial_mass
        bed["transit_mass_kg"] += arterial_mass
        transit_density = bed["transit_mass_kg"] / bed["transit_volume_m3"]
        venous_mass = flow_volume * transit_density
        bed["transit_volume_m3"] -= flow_volume
        bed["venous_volume_m3"] += flow_volume
        bed["transit_mass_kg"] -= venous_mass
        bed["venous_mass_kg"] += venous_mass
        counts["connection"] += 2

        phase = row["phase_schedule"][accepted_index % len(row["phase_schedule"])]
        exchange_volume = (row["source_flow_m3_per_s"] * timestep_s
                           * row["exchange_fraction_of_source_flow"])
        if phase > 0:
            _require(exchange_volume < bed["transit_volume_m3"],
                     f"{row['region_id']} blood-to-tissue exchange empties transit")
            density = bed["transit_mass_kg"] / bed["transit_volume_m3"]
            exchange_mass = exchange_volume * density
            bed["transit_volume_m3"] -= exchange_volume
            bed["tissue_volume_m3"] += exchange_volume
            bed["transit_mass_kg"] -= exchange_mass
            bed["tissue_mass_kg"] += exchange_mass
            counts["blood_to_tissue"] += 1
        else:
            _require(exchange_volume < bed["tissue_volume_m3"],
                     f"{row['region_id']} tissue-to-blood exchange empties tissue")
            density = bed["tissue_mass_kg"] / bed["tissue_volume_m3"]
            exchange_mass = exchange_volume * density
            bed["tissue_volume_m3"] -= exchange_volume
            bed["transit_volume_m3"] += exchange_volume
            bed["tissue_mass_kg"] -= exchange_mass
            bed["transit_mass_kg"] += exchange_mass
            counts["tissue_to_blood"] += 1

        for key, value in bed.items():
            if key.endswith("_volume_m3"):
                _require(math.isfinite(value) and value > 0.0,
                         f"{row['region_id']} volume left positive domain")
            elif key.endswith("_mass_kg"):
                _require(math.isfinite(value) and value >= 0.0,
                         f"{row['region_id']} mass left nonnegative domain")
    return candidate, counts


def simulate(compiled: dict[str, Any], *, steps: int = 512,
             timestep_s: float = CLOCK_SECONDS,
             reject_step: int | None = None) -> dict[str, Any]:
    qualification = compiled.get("qualification", {})
    _require(qualification.get("mechanical_blood_mass_owner") is False
             and qualification.get("mechanical_tissue_mass_owner") is False
             and qualification.get("anatomical_exchange_owner") is False,
             "candidate promoted a mechanical or anatomical owner")
    _require(type(steps) is int and 1 <= steps <= MAX_STEPS, "steps out of range")
    _require(type(timestep_s) in (int, float) and math.isfinite(timestep_s)
             and timestep_s > 0.0, "timestep is not positive")
    _require(abs(float(timestep_s) - CLOCK_SECONDS) <= 1.0e-15,
             "timestep is not the canonical 12.5 us clock")
    if reject_step is not None:
        _require(type(reject_step) is int and 1 <= reject_step <= steps,
                 "reject_step is outside attempted steps")
    state = copy.deepcopy(compiled["initial_state"])
    initial = copy.deepcopy(state)
    initial_totals = _state_totals(state)
    accepted_hashes: list[str] = []
    transfer_counts = {"connection": 0, "blood_to_tissue": 0, "tissue_to_blood": 0}
    rejected = 0
    for attempt in range(1, steps + 1):
        before = copy.deepcopy(state)
        candidate, counts = _step(compiled, state, float(timestep_s), state.get("accepted_steps", 0))
        if reject_step == attempt:
            state = before
            rejected += 1
            continue
        candidate["accepted_steps"] = state.get("accepted_steps", 0) + 1
        state = candidate
        for key, value in counts.items():
            transfer_counts[key] += value
        accepted_hashes.append(_digest(state))
    final_totals = _state_totals(state)
    mass_residual = final_totals["mass_kg"] - initial_totals["mass_kg"]
    volume_residual = final_totals["volume_m3"] - initial_totals["volume_m3"]
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
        "transfer_counts": transfer_counts,
        "conservation": {
            "mass_residual_kg": mass_residual,
            "volume_residual_m3": volume_residual,
            "mass_conserved": abs(mass_residual) <= 1.0e-12,
            "volume_conserved": abs(volume_residual) <= 1.0e-16,
        },
        "rollback": {"requested": reject_step is not None,
                     "rejected_candidate_state_neutral": True},
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
    receipt["status"] = compiled["status"]
    for key in ("source", "clock", "counts", "beds", "parameter_provenance"):
        receipt[key] = compiled[key]
    output = arguments.output.resolve()
    receipt_sha = _write_immutable(output, receipt)
    print(json.dumps({"schema": SCHEMA, "output": str(output), "sha256": receipt_sha,
                      "accepted_steps": receipt["accepted_steps"],
                      "rejected_steps": receipt["rejected_steps"],
                      "transfer_counts": receipt["transfer_counts"],
                      "mass_conserved": receipt["conservation"]["mass_conserved"],
                      "volume_conserved": receipt["conservation"]["volume_conserved"],
                      "qualification": "partial"}, sort_keys=True))
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
        parser.exit(2, f"organ-blood-mass-transfer: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
