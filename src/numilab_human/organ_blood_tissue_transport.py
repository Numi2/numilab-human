"""Compile and step a source-bound regional blood-transport candidate.

The pinned BodyParts3D organ moments provide seven explicit parenchymal
region candidates.  CVSim21 provides aggregate arterial/venous source owners
and branch flows.  This module partitions those source owners by the organ
surface-volume candidate and advances an internal arterial -> tissue transit
volume -> venous path at the canonical 12.5 us clock.

The transit volume is an engineering candidate, not an anatomical capillary
volume.  The candidate proves source identity, nonduplicated zeroth-moment
mass transport, conservation, and rejected-step rollback.  It does not
assign vessel tubes, tissue exchange, mechanical mass, calibrated density,
organ mechanics, or subject physiology.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import cvsim21
from . import cvsim21_blood_mass_step as blood_mass
from . import tissue_mass_candidate
from .model import ImportError as HumanImportError
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.organ-blood-tissue-transport-candidate.v1"
PROFILE = ROOT / "config/organ-blood-tissue-transport.v1.json"
BLOOD_OWNER = ROOT / "config/cvsim21-blood-mass-owner.v1.json"
CLOCK_NANOSECONDS = 12_500
CLOCK_SECONDS = CLOCK_NANOSECONDS * 1.0e-9
MAX_STEPS = 1_000_000
REGIONS = (
    "right_lung", "left_lung", "right_kidney", "left_kidney",
    "stomach", "pancreas", "liver",
)


class TransportError(HumanImportError):
    """A regional transport candidate cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TransportError("organ blood tissue transport: " + message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{label} is not finite")
    return float(value)


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(result > 0.0, f"{label} is not positive")
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
        raise TransportError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _load_profile(path: Path) -> tuple[dict[str, Any], str]:
    profile, profile_sha = _read_canonical(path, "transport profile")
    required = {"schema", "id", "clock_nanoseconds", "transit_time_s",
                "parameter_provenance", "beds", "boundary"}
    _require(set(profile) == required, "transport profile fields differ")
    _require(profile["schema"] == "numi.human.organ-blood-tissue-transport.v1",
             "unsupported transport profile schema")
    _require(profile["id"] == "seven_bed_source_regional_transport",
             "unsupported transport profile")
    _require(profile["clock_nanoseconds"] == CLOCK_NANOSECONDS,
             "transport profile clock differs")
    transit_time = _positive(profile["transit_time_s"], "transit time")
    provenance = profile["parameter_provenance"]
    _require(isinstance(provenance, dict)
             and set(provenance) == {"kind", "description"}
             and provenance["kind"] == "engineering_candidate_unresolved"
             and isinstance(provenance["description"], str)
             and provenance["description"].strip(),
             "transit time provenance must remain unresolved")
    beds = profile["beds"]
    _require(isinstance(beds, list) and len(beds) == len(REGIONS),
             "transport profile must contain seven beds")
    seen: set[str] = set()
    for bed in beds:
        _require(isinstance(bed, dict)
                 and set(bed) == {"region_id", "group_id", "arterial_region_id",
                                  "venous_region_id", "flow_connection_id"},
                 "transport bed fields differ")
        region = bed["region_id"]
        _require(region in REGIONS and region not in seen,
                 f"transport bed region is duplicated or unknown: {region}")
        seen.add(region)
        for key in ("group_id", "arterial_region_id", "venous_region_id",
                    "flow_connection_id"):
            _require(isinstance(bed[key], str) and bed[key].strip(),
                     f"transport bed {region} has invalid {key}")
    _require(seen == set(REGIONS), "transport beds do not cover seven regions")
    _require(isinstance(profile["boundary"], str) and profile["boundary"].strip(),
             "transport profile boundary is missing")
    return profile, profile_sha


def _region_candidates(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, list[dict[str, Any]]] = {region: [] for region in REGIONS}
    for row in candidate.get("candidates", []):
        region = row.get("region_id")
        if region in rows and row.get("mass_admission") == (
                "surface_candidate_only_physical_volume_unresolved"):
            rows[region].append(row)
    result: dict[str, dict[str, Any]] = {}
    for region, members in rows.items():
        _require(members, f"organ mass candidate has no admitted rows for {region}")
        result[region] = {
            "source_member_count": len(members),
            "source_member_ids": [row["member_id"] for row in members],
            "surface_candidate_volume_m3": math.fsum(
                _positive(row["candidate_surface_volume_m3"], f"{region} surface volume")
                for row in members
            ),
            "surface_candidate_mass_kg": math.fsum(
                _positive(row["candidate_mass_kg"], f"{region} candidate mass")
                for row in members
            ),
        }
        _require(result[region]["surface_candidate_volume_m3"] > 0.0,
                 f"{region} candidate volume is empty")
    return result


def _source_indices(native: dict[str, Any], owner: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    compartments: dict[str, dict[str, Any]] = {}
    for row in native.get("compartments", []):
        region_value = row.get("anatomical_region_id")
        region = region_value.rsplit(":", 1)[-1] if isinstance(region_value, str) else None
        _require(isinstance(region, str) and region not in compartments,
                 "CVSim21 compartment region identity is not unique")
        compartments[region] = row
    owners = blood_mass._validate_owner(native, owner)
    connections = {row.get("id"): row for row in native.get("connections", [])}
    _require(len(connections) == 24 and all(isinstance(key, str) for key in connections),
             "CVSim21 connection identity is incomplete")
    return compartments, owners["owners"], connections


def compile_candidate(*, profile: Path = PROFILE,
                      blood_owner: Path = BLOOD_OWNER) -> dict[str, Any]:
    profile_doc, profile_sha = _load_profile(Path(profile))
    organ = tissue_mass_candidate.compile_candidate()
    regions = _region_candidates(organ)
    native, lowering = cvsim21.compile_source()
    owner_doc = cvsim21.read_json(Path(blood_owner))
    compartments, owners, connections = _source_indices(native, owner_doc)
    density = _positive(owner_doc["density_kg_per_m3"], "blood density candidate")

    group_totals: dict[str, float] = defaultdict(float)
    for bed in profile_doc["beds"]:
        group_totals[bed["group_id"]] += regions[bed["region_id"]]["surface_candidate_volume_m3"]

    beds: list[dict[str, Any]] = []
    owner_allocations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for bed in sorted(profile_doc["beds"], key=lambda row: row["region_id"]):
        region = bed["region_id"]
        region_candidate = regions[region]
        group_total = group_totals[bed["group_id"]]
        share = region_candidate["surface_candidate_volume_m3"] / group_total
        arterial = compartments.get(bed["arterial_region_id"])
        venous = compartments.get(bed["venous_region_id"])
        flow_row = connections.get(bed["flow_connection_id"])
        _require(arterial is not None and venous is not None,
                 f"{region} source arterial/venous compartments are missing")
        _require(flow_row is not None, f"{region} source flow is missing")
        _require(flow_row["from"] == arterial["stable_identifier"]
                 and flow_row["to"] == venous["stable_identifier"],
                 f"{region} flow connection endpoints differ from source compartments")
        flow = _positive(flow_row["initial_flow_m3_per_s"], f"{region} source flow")
        source_arterial_volume = _positive(arterial["initial_volume_m3"],
                                           f"{region} arterial source volume") * share
        source_venous_volume = _positive(venous["initial_volume_m3"],
                                         f"{region} venous source volume") * share
        transit_volume = flow * float(profile_doc["transit_time_s"])
        _require(transit_volume < source_arterial_volume,
                 f"{region} transit volume consumes arterial source volume")
        arterial_owner = owners[arterial["stable_identifier"]]
        venous_owner = owners[venous["stable_identifier"]]
        arterial_mass = _positive(arterial_owner["initial_mass_kg"],
                                  f"{region} arterial source mass") * share
        venous_mass = _positive(venous_owner["initial_mass_kg"],
                                f"{region} venous source mass") * share
        transit_mass = density * transit_volume
        _require(transit_mass < arterial_mass,
                 f"{region} transit mass consumes arterial source mass")
        bed_row = {
            "region_id": region,
            "group_id": bed["group_id"],
            "allocation_share": share,
            "source_member_count": region_candidate["source_member_count"],
            "source_member_ids": region_candidate["source_member_ids"],
            "surface_candidate_volume_m3": region_candidate["surface_candidate_volume_m3"],
            "surface_candidate_mass_kg": region_candidate["surface_candidate_mass_kg"],
            "arterial_source_region_id": bed["arterial_region_id"],
            "venous_source_region_id": bed["venous_region_id"],
            "flow_connection_id": bed["flow_connection_id"],
            "arterial_source_owner_id": arterial_owner["physical_volume_owner_id"],
            "venous_source_owner_id": venous_owner["physical_volume_owner_id"],
            "source_flow_m3_per_s": flow,
            "transit_volume_candidate_m3": transit_volume,
            "blood_density_kg_per_m3": density,
            "initial_state": {
                "arterial_volume_m3": source_arterial_volume - transit_volume,
                "transit_volume_m3": transit_volume,
                "venous_volume_m3": source_venous_volume,
                "arterial_mass_kg": arterial_mass - transit_mass,
                "transit_mass_kg": transit_mass,
                "venous_mass_kg": venous_mass,
            },
            "physical_volume_owner": None,
            "mechanical_mass_owner": None,
            "tissue_exchange_owner": None,
        }
        beds.append(bed_row)
        owner_allocations[bed_row["arterial_source_owner_id"]].append(
            {"region_id": region, "share": share, "role": "arterial"})
        owner_allocations[bed_row["venous_source_owner_id"]].append(
            {"region_id": region, "share": share, "role": "venous"})

    for owner_id, allocations in owner_allocations.items():
        _require(math.isclose(math.fsum(row["share"] for row in allocations), 1.0,
                              rel_tol=0.0, abs_tol=1.0e-15),
                 f"source owner allocation does not close: {owner_id}")

    source_owner_ids = sorted(owner_allocations)
    initial_state = {row["region_id"]: copy.deepcopy(row["initial_state"]) for row in beds}
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.organ-blood-tissue-transport.1",
        "status": "partial",
        "source": {
            "organ_candidate_schema": organ["schema"],
            "organ_candidate_moments_sha256": organ["source"]["moments_sha256"],
            "organ_candidate_profile_sha256": organ["source"]["profile_sha256"],
            "transport_profile": _relative(Path(profile)),
            "transport_profile_sha256": profile_sha,
            "blood_owner": _relative(Path(blood_owner)),
            "blood_owner_sha256": _digest(owner_doc),
            "native_content_sha256": hashlib.sha256(canonical(native) + b"\n").hexdigest(),
            "source_lowering_manifest_sha256": _digest(lowering),
        },
        "clock": {
            "nanoseconds": CLOCK_NANOSECONDS,
            "seconds": CLOCK_SECONDS,
            "source": "profile_and_CVSim21_exact_clock",
        },
        "counts": {
            "bed_count": len(beds),
            "region_count": len(regions),
            "source_blood_owner_count": len(source_owner_ids),
            "source_members_bound": sum(row["source_member_count"] for row in beds),
        },
        "parameter_provenance": profile_doc["parameter_provenance"],
        "beds": beds,
        "source_owner_partition": [
            {"physical_volume_owner_id": owner_id,
             "allocations": sorted(owner_allocations[owner_id],
                                    key=lambda row: (row["region_id"], row["role"])),
             "allocation_sum": math.fsum(row["share"] for row in owner_allocations[owner_id])}
            for owner_id in source_owner_ids
        ],
        "initial_state": {"beds": initial_state},
        "qualification": {
            "organ_surface_candidate_bound": True,
            "source_blood_zeroth_moment_bound": True,
            "regional_arterial_tissue_venous_transport": True,
            "mass_conservation_and_rollback": True,
            "blood_first_second_spatial_moments": False,
            "anatomical_vessel_tube_or_lumen": False,
            "physical_tissue_volume_owner": False,
            "mechanical_mass_owner": False,
            "two_way_tissue_exchange": False,
            "material_density_calibrated": False,
            "subject_calibration": False,
            "organ_mechanics": False,
            "standing_walking": False,
        },
        "boundary": (
            "Seven source organ surface candidates partition six aggregate CVSim21 "
            "blood owners by surface-volume share and route source branch flow through "
            "an explicit candidate transit volume. This closes only zeroth-moment "
            "regional transport and rollback; it does not establish an anatomical "
            "lumen, capillary volume, tissue exchange, spatial blood moments, "
            "mechanical mass, calibrated material, or subject physiology."
        ),
    }


def _state_totals(state: dict[str, Any]) -> dict[str, float]:
    return {
        "volume_m3": math.fsum(row for key in ("arterial_volume_m3", "transit_volume_m3",
                                                "venous_volume_m3")
                               for bed in state["beds"].values()
                               for row in [bed[key]]),
        "mass_kg": math.fsum(row for key in ("arterial_mass_kg", "transit_mass_kg",
                                              "venous_mass_kg")
                             for bed in state["beds"].values()
                             for row in [bed[key]]),
    }


def _state_digest(state: dict[str, Any]) -> str:
    return _digest(state)


def _step(compiled: dict[str, Any], state: dict[str, Any], timestep_s: float) -> tuple[dict[str, Any], int]:
    _require(math.isfinite(timestep_s) and timestep_s > 0.0,
             "timestep must be positive")
    candidate = copy.deepcopy(state)
    transfers = 0
    for row in compiled["beds"]:
        bed = candidate["beds"][row["region_id"]]
        volume = row["source_flow_m3_per_s"] * timestep_s
        _require(volume < bed["arterial_volume_m3"] and volume < bed["transit_volume_m3"],
                 f"{row['region_id']} candidate step empties a compartment")
        arterial_density = bed["arterial_mass_kg"] / bed["arterial_volume_m3"]
        incoming_mass = volume * arterial_density
        bed["arterial_volume_m3"] -= volume
        bed["transit_volume_m3"] += volume
        bed["arterial_mass_kg"] -= incoming_mass
        bed["transit_mass_kg"] += incoming_mass
        transit_density = bed["transit_mass_kg"] / bed["transit_volume_m3"]
        outgoing_mass = volume * transit_density
        bed["transit_volume_m3"] -= volume
        bed["venous_volume_m3"] += volume
        bed["transit_mass_kg"] -= outgoing_mass
        bed["venous_mass_kg"] += outgoing_mass
        transfers += 2
        for key, value in bed.items():
            if key.endswith("_volume_m3"):
                _require(math.isfinite(value) and value > 0.0,
                         f"{row['region_id']} volume left the positive domain")
            elif key.endswith("_mass_kg"):
                _require(math.isfinite(value) and value >= 0.0,
                         f"{row['region_id']} mass left the nonnegative domain")
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
            "mass_conserved": abs(mass_residual) <= 1.0e-12,
            "volume_conserved": abs(volume_residual) <= 1.0e-16,
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
    compiled = compile_candidate(profile=arguments.profile, blood_owner=arguments.blood_owner)
    receipt = simulate(compiled, steps=arguments.steps,
                       timestep_s=arguments.timestep_seconds,
                       reject_step=arguments.reject_step)
    receipt["source"] = compiled["source"]
    receipt["clock"] = compiled["clock"]
    receipt["counts"] = compiled["counts"]
    receipt["beds"] = compiled["beds"]
    receipt["source_owner_partition"] = compiled["source_owner_partition"]
    receipt["parameter_provenance"] = compiled["parameter_provenance"]
    output = arguments.output.resolve()
    receipt_sha = _write_immutable(output, receipt)
    print(json.dumps({"schema": SCHEMA, "output": str(output),
                      "sha256": receipt_sha,
                      "accepted_steps": receipt["accepted_steps"],
                      "rejected_steps": receipt["rejected_steps"],
                      "transfer_count": receipt["transfer_count"],
                      "mass_conserved": receipt["conservation"]["mass_conserved"],
                      "volume_conserved": receipt["conservation"]["volume_conserved"],
                      "qualification": "partial"}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, default=PROFILE)
    parser.add_argument("--blood-owner", type=Path, default=BLOOD_OWNER)
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
        parser.exit(2, f"organ-blood-tissue-transport: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
