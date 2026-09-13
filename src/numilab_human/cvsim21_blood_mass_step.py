"""Advance a source-bound CVSim21 absolute-blood mass owner.

The pinned CVSim21 graph supplies 21 aggregate blood volumes and 24 hydraulic
flows.  This module gives those source owners one explicit mass state and
advects mass with each accepted hydraulic transfer.  Blood density remains an
unresolved engineering candidate; no anatomy, tissue exchange, mechanical
mass, or subject calibration is inferred from the aggregate graph.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import cvsim21 as source
from .model import ImportError as HumanImportError


SCHEMA = "HumanPack.cvsim21-blood-mass-step-receipt.v1"
OWNER_SCHEMA = "HumanPack.cvsim21-blood-mass-owner.v1"
MAX_STEPS = 1_000_000
CLOCK_NANOSECONDS = 12_500
CLOCK_SECONDS = CLOCK_NANOSECONDS * 1.0e-9
CLOCK_TOLERANCE_SECONDS = 1.0e-15


class StepError(HumanImportError):
    """A CVSim21 blood-mass candidate cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StepError("CVSim21 blood mass step: " + message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{label} must be finite")
    return float(value)


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(result > 0.0, f"{label} must be positive")
    return result


def _positive_numeric_string(value: Any, label: str) -> float:
    _require(isinstance(value, str) and value.strip(), f"{label} must be a numeric string")
    try:
        result = float(value)
    except ValueError as error:
        raise StepError(f"CVSim21 blood mass step: {label} must be numeric") from error
    _require(math.isfinite(result) and result > 0.0, f"{label} must be positive")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(source.canonical(value)).hexdigest()


def _indexed(native: dict[str, Any], table: str, *, allow_empty: bool = False) -> dict[int, dict[str, Any]]:
    rows = native.get(table)
    _require(isinstance(rows, list) and (allow_empty or rows),
             f"native source has no {table}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), f"{table} contains a malformed row")
        identifier = row.get("stable_identifier")
        _require(type(identifier) is int and identifier > 0,
                 f"{table} has an invalid stable identifier")
        _require(identifier not in result,
                 f"{table} repeats stable identifier {identifier}")
        result[identifier] = row
    return dict(sorted(result.items()))


def _validate_native(native: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    _require(native.get("schema") == "HumanPack.physiology-native.v3",
             "unsupported CVSim21 native schema")
    _require(native.get("qualification") == "source_model_variant",
             "native graph is not the pinned CVSim21 source variant")
    _require(native.get("law") == "closed_rational_elastance_regional_volume_v3",
             "unsupported CVSim21 source law")
    _require(native.get("species") == [] and native.get("tissue_reservoirs") == []
             and native.get("exchanges") == [],
             "CVSim21 mass owner cannot invent species or tissue state")
    compartments = _indexed(native, "compartments")
    connections = _indexed(native, "connections")
    _require(len(compartments) == 21 and len(connections) == 24,
             "CVSim21 source topology must contain 21 compartments and 24 connections")
    for identifier, row in compartments.items():
        _positive(row["initial_volume_m3"], f"compartment {row.get('id')} initial volume")
        _positive(row["reference_volume_m3"], f"compartment {row.get('id')} reference volume")
        _require(row["storage_kind"] == "absolute_volume",
                 f"compartment {row.get('id')} is not an absolute-volume owner")
        _require(row["pressure_law"] in {"linear_compliance", "atan_compliance",
                                           "cosine_pulse_elastance"},
                 f"compartment {row.get('id')} pressure law is unsupported")
        if row["pressure_law"] in {"linear_compliance", "atan_compliance"}:
            _positive(row["compliance_m3_per_pa"], f"compartment {row.get('id')} compliance")
        if row["pressure_law"] == "atan_compliance":
            _positive(row["maximum_volume_displacement_m3"],
                      f"compartment {row.get('id')} maximum displacement")
        if row["pressure_law"] == "cosine_pulse_elastance":
            _positive(row["elastance_min_pa_per_m3"], f"compartment {row.get('id')} minimum elastance")
            _positive(row["elastance_max_pa_per_m3"], f"compartment {row.get('id')} maximum elastance")
            _positive_numeric_string(row["period_numerator_seconds"],
                      f"compartment {row.get('id')} period numerator")
            _positive_numeric_string(row["period_denominator"],
                      f"compartment {row.get('id')} period denominator")
            _positive(row["activation_end"], f"compartment {row.get('id')} activation end")
            _finite(row["activation_start"], f"compartment {row.get('id')} activation start")
            _finite(row["phase_delay"], f"compartment {row.get('id')} phase delay")
            _positive(row["source_pi"], f"compartment {row.get('id')} source pi")
    for identifier, row in connections.items():
        _require(type(row.get("from")) is int and row["from"] in compartments
                 and type(row.get("to")) is int and row["to"] in compartments
                 and row["from"] != row["to"],
                 f"connection {row.get('id')} endpoint is invalid")
        _require(row["flow_law"] in {"starling_resistance", "one_way_resistance",
                                      "resistance_inertance"},
                 f"connection {row.get('id')} flow law is unsupported")
        _positive(row["resistance_pa_s_per_m3"], f"connection {row.get('id')} resistance")
        _finite(row["initial_flow_m3_per_s"], f"connection {row.get('id')} initial flow")
        _finite(row["inertance_pa_s2_per_m3"], f"connection {row.get('id')} inertance")
        _require(row["inertance_pa_s2_per_m3"] >= 0.0,
                 f"connection {row.get('id')} inertance is negative")
        if row["flow_law"] == "starling_resistance":
            _finite(row["downstream_pressure_floor_pa"],
                    f"connection {row.get('id')} downstream floor")
    return compartments, connections


def _validate_owner(native: dict[str, Any], owner: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(owner, dict), "owner graph must be an object")
    required = {
        "schema", "id", "native_content_sha256", "source_manifest_sha256", "initial_sha256",
        "volume_coordinates", "qualification", "mass_residual_tolerance_kg",
        "density_kg_per_m3", "density_provenance", "compartment_owners", "boundary",
    }
    _require(set(owner) == required, "owner graph fields differ")
    _require(owner["schema"] == OWNER_SCHEMA, "unsupported owner schema")
    _require(owner["qualification"] == "source_absolute_blood_mass_candidate",
             "owner graph qualification differs")
    compartments, connections = _validate_native(native)
    expected_content = hashlib.sha256(source.canonical(native) + b"\n").hexdigest()
    _require(owner["native_content_sha256"] == expected_content,
             "owner graph does not bind the native CVSim21 content")
    _require(owner["source_manifest_sha256"] == native.get("source_graph_sha256"),
             "owner graph does not bind the CVSim21 source lock")
    _require(owner["initial_sha256"] == source.INITIAL_SHA256,
             "owner graph does not bind the CVSim21 initial state")
    _require(owner["volume_coordinates"] in source.VARIANTS
             and native.get("model_id", "").endswith("_" + owner["volume_coordinates"]),
             "owner graph volume coordinates differ")
    tolerance = _positive(owner["mass_residual_tolerance_kg"], "mass residual tolerance")
    _require(tolerance < 1e-9, "mass residual tolerance is too broad")
    density = _positive(owner["density_kg_per_m3"], "blood density candidate")
    provenance = owner["density_provenance"]
    _require(isinstance(provenance, dict)
             and set(provenance) == {"kind", "description"}
             and provenance["kind"] == "engineering_candidate_unresolved"
             and isinstance(provenance["description"], str)
             and provenance["description"].strip(),
             "blood density must remain explicitly unresolved")
    rows = owner["compartment_owners"]
    _require(isinstance(rows, list) and rows, "owner graph has no compartment owners")
    owners: dict[int, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {
            "stable_identifier", "physical_volume_owner_id", "initial_mass_kg",
            "anatomy_registration", "mechanical_mass_owner"
        }, "compartment owner fields differ")
        identifier = row["stable_identifier"]
        _require(type(identifier) is int and identifier in compartments,
                 "compartment owner has unknown stable identifier")
        _require(identifier not in owners, "compartment owner is duplicated")
        _require(row["physical_volume_owner_id"] == compartments[identifier]["physical_volume_owner_id"],
                 f"compartment {identifier} physical owner differs from source")
        _require(row["anatomy_registration"] is None and row["mechanical_mass_owner"] is None,
                 "aggregate blood owner cannot promote anatomy or mechanical mass")
        mass = _positive(row["initial_mass_kg"], f"compartment {identifier} initial mass")
        expected_mass = density * compartments[identifier]["initial_volume_m3"]
        _require(math.isclose(mass, expected_mass, rel_tol=0.0, abs_tol=tolerance),
                 f"compartment {identifier} initial mass disagrees with candidate density")
        row["initial_mass_kg"] = mass
        owners[identifier] = row
    _require(set(owners) == set(compartments),
             "compartment owners do not cover every CVSim21 volume")
    owner_ids = [row["physical_volume_owner_id"] for row in owners.values()]
    _require(len(owner_ids) == len(set(owner_ids)), "physical volume owner is duplicated")
    return {
        "compartments": compartments,
        "connections": connections,
        "owners": owners,
        "density": density,
        "tolerance": tolerance,
    }


def _activation(row: dict[str, Any], time_s: float) -> float:
    if row["pressure_law"] != "cosine_pulse_elastance":
        return 0.0
    period = float(row["period_numerator_seconds"]) / float(row["period_denominator"])
    phase = math.fmod(time_s - float(row["phase_delay"]), period)
    if phase < 0.0:
        phase += period
    start = float(row["activation_start"])
    duration = float(row["activation_end"])
    relative = math.fmod(phase - start, period)
    if relative < 0.0:
        relative += period
    if relative >= duration:
        return 0.0
    value = 0.5 * (1.0 - math.cos(2.0 * float(row["source_pi"]) * relative / duration))
    _require(math.isfinite(value) and 0.0 <= value <= 1.0 + 1e-12,
             f"compartment {row.get('id')} activation left [0,1]")
    return min(1.0, max(0.0, value))


def _pressures(native: dict[str, Any], state: dict[str, Any]) -> tuple[dict[int, float], dict[str, float]]:
    compartments, _ = _validate_native(native)
    pressures: dict[int, float] = {}
    activation: dict[str, float] = {}
    for identifier, row in compartments.items():
        volume = state["compartments"][identifier]["volume_m3"]
        _require(math.isfinite(volume) and volume > 0.0,
                 f"compartment {row.get('id')} volume is invalid")
        law = row["pressure_law"]
        if law == "linear_compliance":
            pressure = row["external_pressure_pa"] + (
                volume - row["reference_volume_m3"]
            ) / row["compliance_m3_per_pa"]
        elif law == "atan_compliance":
            displacement = volume - row["reference_volume_m3"]
            maximum = row["maximum_volume_displacement_m3"]
            _require(abs(displacement) < maximum,
                     f"compartment {row.get('id')} exceeds atan storage domain")
            pressure = row["external_pressure_pa"] + (
                2.0 * maximum / (math.pi * row["compliance_m3_per_pa"])
            ) * math.tan(math.pi * displacement / (2.0 * maximum))
        else:
            value = _activation(row, state["accepted_time_s"])
            activation[row["id"]] = value
            elastance = row["elastance_min_pa_per_m3"] + value * (
                row["elastance_max_pa_per_m3"] - row["elastance_min_pa_per_m3"]
            )
            pressure = row["reference_pressure_pa"] + elastance * (
                volume - row["reference_volume_m3"]
            )
        _require(math.isfinite(pressure), f"compartment {row.get('id')} pressure is invalid")
        pressures[identifier] = float(pressure)
    return pressures, activation


def _state_from_native(native: dict[str, Any], owner: dict[str, Any]) -> dict[str, Any]:
    compartments, connections = _validate_native(native)
    return {
        "compartments": {
            identifier: {
                "volume_m3": _finite(row["initial_volume_m3"], f"{row.get('id')} volume"),
                "mass_kg": float(owner["owners"][identifier]["initial_mass_kg"]),
            }
            for identifier, row in compartments.items()
        },
        "connections": {
            identifier: {"flow_m3_per_s": _finite(row["initial_flow_m3_per_s"], f"{row.get('id')} flow")}
            for identifier, row in connections.items()
        },
        "accepted_steps": 0,
        "accepted_time_s": 0.0,
    }


def _candidate(native: dict[str, Any], state: dict[str, Any], timestep_s: float) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(math.isfinite(timestep_s) and timestep_s > 0.0,
             "timestep must be finite and positive")
    compartments, connections = _validate_native(native)
    pressures, activation = _pressures(native, state)
    candidate = copy.deepcopy(state)
    transfers: list[dict[str, Any]] = []
    for identifier, row in connections.items():
        drop = pressures[row["from"]] - pressures[row["to"]]
        if row["flow_law"] == "starling_resistance":
            drop = pressures[row["from"]] - max(
                pressures[row["to"]], row["downstream_pressure_floor_pa"]
            )
            flow = max(drop, 0.0) / row["resistance_pa_s_per_m3"]
        elif row["flow_law"] == "one_way_resistance":
            flow = max(drop, 0.0) / row["resistance_pa_s_per_m3"]
        elif row["inertance_pa_s2_per_m3"] == 0.0:
            flow = drop / row["resistance_pa_s_per_m3"]
        else:
            inertance = row["inertance_pa_s2_per_m3"]
            old_flow = state["connections"][identifier]["flow_m3_per_s"]
            flow = (old_flow + timestep_s * drop / inertance) / (
                1.0 + timestep_s * row["resistance_pa_s_per_m3"] / inertance
            )
        _require(math.isfinite(flow), f"connection {row.get('id')} flow is invalid")
        candidate["connections"][identifier]["flow_m3_per_s"] = flow
        if flow == 0.0:
            continue
        source_id = row["from"] if flow > 0.0 else row["to"]
        target_id = row["to"] if flow > 0.0 else row["from"]
        transported = abs(timestep_s * flow)
        available = candidate["compartments"][source_id]["volume_m3"]
        _require(transported < available,
                 f"connection {row.get('id')} empties its source volume")
        candidate["compartments"][source_id]["volume_m3"] -= transported
        candidate["compartments"][target_id]["volume_m3"] += transported
        transfers.append({"connection": identifier, "source": source_id,
                          "target": target_id, "volume_m3": transported})
    candidate["accepted_steps"] += 1
    candidate["accepted_time_s"] += timestep_s
    for row in candidate["compartments"].values():
        _require(math.isfinite(row["volume_m3"]) and row["volume_m3"] > 0.0,
                 "candidate volume left the positive domain")
    return candidate, {"activation": activation, "transfers": transfers}


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_steps": state["accepted_steps"],
        "accepted_time_s": state["accepted_time_s"],
        "compartments": state["compartments"],
        "connections": state["connections"],
    }


def _totals(state: dict[str, Any]) -> dict[str, float]:
    rows = state["compartments"].values()
    return {
        "mass_kg": math.fsum(row["mass_kg"] for row in rows),
        "volume_m3": math.fsum(row["volume_m3"] for row in state["compartments"].values()),
    }


def _apply_mass_transfers(state: dict[str, Any], transfers: list[dict[str, Any]],
                          timestep_s: float) -> dict[str, Any]:
    candidate = copy.deepcopy(state)
    transfer_count = 0
    for transfer in transfers:
        source_id = transfer["source"]
        target_id = transfer["target"]
        volume = transfer["volume_m3"]
        available = candidate["compartments"][source_id]["volume_m3"]
        _require(volume < available, "mass transfer empties its source volume")
        density = candidate["compartments"][source_id]["mass_kg"] / available
        mass = volume * density
        candidate["compartments"][source_id]["volume_m3"] -= volume
        candidate["compartments"][target_id]["volume_m3"] += volume
        candidate["compartments"][source_id]["mass_kg"] -= mass
        candidate["compartments"][target_id]["mass_kg"] += mass
        transfer_count += 1
    for row in candidate["compartments"].values():
        _require(row["volume_m3"] > 0.0 and math.isfinite(row["volume_m3"]),
                 "mass volume left the positive domain")
        _require(row["mass_kg"] >= 0.0 and math.isfinite(row["mass_kg"]),
                 "mass left the nonnegative domain")
    candidate["accepted_steps"] += 1
    candidate["accepted_time_s"] += timestep_s
    return candidate | {"transfer_count": transfer_count}


def simulate(native: dict[str, Any], owner_document: dict[str, Any], *, steps: int,
             timestep_s: float, reject_step: int | None = None) -> dict[str, Any]:
    _require(type(steps) is int and 1 <= steps <= MAX_STEPS, "steps out of range")
    _require(type(timestep_s) in (int, float) and math.isfinite(timestep_s)
             and timestep_s > 0.0, "timestep must be finite and positive")
    if reject_step is not None:
        _require(type(reject_step) is int and 1 <= reject_step <= steps,
                 "reject_step must be within attempted steps")
    owner = _validate_owner(native, owner_document)
    state = _state_from_native(native, owner)
    initial = _public_state(state)
    initial_totals = _totals(state)
    rejected = 0
    transfer_count = 0
    activation_trace: list[dict[str, float]] = []
    accepted_trace: list[str] = []
    for attempt in range(1, steps + 1):
        before = copy.deepcopy(state)
        hydraulic_candidate, diagnostics = _candidate(native, state, float(timestep_s))
        mass_candidate = _apply_mass_transfers(state, diagnostics["transfers"], float(timestep_s))
        for identifier in state["compartments"]:
            _require(math.isclose(
                mass_candidate["compartments"][identifier]["volume_m3"],
                hydraulic_candidate["compartments"][identifier]["volume_m3"],
                rel_tol=0.0, abs_tol=1e-21,
            ), "hydraulic and mass volume owners diverged")
        mass_candidate.pop("transfer_count")
        if reject_step == attempt:
            state = before
            rejected += 1
            continue
        state = mass_candidate
        transfer_count += len(diagnostics["transfers"])
        activation_trace.append(diagnostics["activation"])
        accepted_trace.append(_digest(_public_state(state)))
    final = _public_state(state)
    final_totals = _totals(state)
    mass_residual = final_totals["mass_kg"] - initial_totals["mass_kg"]
    volume_residual = final_totals["volume_m3"] - initial_totals["volume_m3"]
    tolerance = owner["tolerance"]
    return {
        "schema": SCHEMA,
        "model_id": native["model_id"],
        "native_content_sha256": owner_document["native_content_sha256"],
        "source_manifest_sha256": native["source_graph_sha256"],
        "owner_graph_sha256": _digest(owner_document),
        "density": {
            "value_kg_per_m3": owner["density"],
            "provenance": owner_document["density_provenance"],
            "calibrated": False,
        },
        "initial_state": initial,
        "final_state": final,
        "attempted_steps": steps,
        "accepted_steps": state["accepted_steps"],
        "rejected_steps": rejected,
        "timestep_seconds": float(timestep_s),
        "clock": {
            "timestep_nanoseconds": float(timestep_s) * 1.0e9,
            "required_nanoseconds": CLOCK_NANOSECONDS,
            "exact": abs(float(timestep_s) - CLOCK_SECONDS) <= CLOCK_TOLERANCE_SECONDS,
        },
        "conservation": {
            "initial": initial_totals,
            "final": final_totals,
            "mass_residual_kg": mass_residual,
            "volume_residual_m3": volume_residual,
            "mass_conserved": abs(mass_residual) <= tolerance,
            "volume_conserved": abs(volume_residual) <= 1e-17,
        },
        "transfer_count": transfer_count,
        "activation_trace_sha256": _digest(activation_trace),
        "accepted_state_trace_sha256": _digest(accepted_trace),
        "rollback": {
            "rejected_steps": rejected,
            "accepted_time_excludes_rejections": math.isclose(
                state["accepted_time_s"], state["accepted_steps"] * float(timestep_s),
                rel_tol=0.0, abs_tol=1e-15,
            ),
        },
        "qualification": "source_absolute_blood_mass_candidate",
        "scope": {
            "source_aggregate_absolute_blood_mass": True,
            "anatomical_registration": False,
            "tissue_exchange": False,
            "mechanical_mass_owner": False,
            "material_density_calibrated": False,
            "subject_calibration": False,
            "standing": False,
            "walking": False,
        },
        "qualification_boundary": (
            "Pinned CVSim21 aggregate absolute-volume owners with explicit unresolved "
            "density candidate, conservative hydraulic advection, and accepted-step "
            "rollback only. This does not qualify anatomical lumen registration, tissue "
            "exchange, mechanical mass or inertia, material calibration, subject blood "
            "volume, standing, or walking."
        ),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=source.CONFIG)
    parser.add_argument("--owners", type=Path,
                        default=source.ROOT / "config/cvsim21-blood-mass-owner.v1.json")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--timestep-seconds", type=float, default=CLOCK_SECONDS)
    parser.add_argument("--require-clock-nanoseconds", type=int)
    parser.add_argument("--reject-step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    required_clock = getattr(args, "require_clock_nanoseconds", None)
    if required_clock is not None:
        _require(type(required_clock) is int and required_clock > 0,
                 "required clock must be a positive integer number of nanoseconds")
        _require(abs(float(args.timestep_seconds) - required_clock * 1.0e-9)
                 <= CLOCK_TOLERANCE_SECONDS,
                 "timestep does not equal the required nanosecond clock")
    native, lowering = source.compile_source(config=source.read_json(args.config))
    owner_document = source.read_json(args.owners)
    receipt = simulate(native, owner_document, steps=args.steps,
                       timestep_s=args.timestep_seconds, reject_step=args.reject_step)
    receipt["source_lowering_manifest_sha256"] = _digest(lowering)
    payload = source.canonical(receipt) + b"\n"
    output = args.output.resolve()
    if output.exists():
        _require(output.read_bytes() == payload,
                 "output is immutable; choose a new output path")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(payload)
    print(json.dumps({"schema": SCHEMA, "output": str(output),
                      "sha256": hashlib.sha256(payload).hexdigest(),
                      "accepted_steps": receipt["accepted_steps"],
                      "rejected_steps": receipt["rejected_steps"],
                      "mass_conserved": receipt["conservation"]["mass_conserved"],
                      "volume_conserved": receipt["conservation"]["volume_conserved"],
                      "qualification": receipt["qualification"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"cvsim21-blood-mass-step: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
