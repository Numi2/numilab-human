"""Advance an explicit blood-mass ownership sidecar with a physiology graph.

The hydraulic graph owns flow and solute amounts.  This module adds a separate,
hash-bound mass owner for a bounded fixture: compartment and tissue fluid
volumes carry explicit density, connection flow advects mass, and declared
exchange schedules move mass in both directions.  Every candidate is
transactional.  The fixture proves conservation and ownership semantics only;
it is not an anatomical vessel, organ, or subject calibration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError as HumanImportError
from .physiology import ROOT, canonical, compile_graph, read_json
from .physiology_step import (
    MAX_STEPS,
    StepError,
    _candidate,
    _state_from_native,
    _totals,
)


SCHEMA = "HumanPack.blood-mass-transfer-step-receipt.v1"
OWNER_SCHEMA = "HumanPack.blood-mass-transfer-owner.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StepError("blood mass step: " + message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{label} must be finite")
    return float(value)


def _positive(value: Any, label: str) -> float:
    result = _finite(value, label)
    _require(result > 0.0, f"{label} must be positive")
    return result


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _indexed(native: dict[str, Any], table: str, *, allow_empty: bool = False) -> dict[int, dict[str, Any]]:
    rows = native.get(table)
    _require(isinstance(rows, list) and (allow_empty or rows), f"native graph has no {table}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), f"{table} contains a malformed row")
        identifier = row.get("stable_identifier")
        _require(type(identifier) is int and identifier > 0,
                 f"{table} has an invalid stable identifier")
        _require(identifier not in result, f"{table} repeats stable identifier {identifier}")
        result[identifier] = row
    return dict(sorted(result.items()))


def _exact_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    _require(isinstance(value, dict) and set(value) == fields, f"{label} fields differ")
    return value


def _owner_rows(owner: dict[str, Any], key: str, expected: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    rows = owner.get(key)
    _require(isinstance(rows, list) and rows, f"owner graph has no {key}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        _exact_object(row, {"stable_identifier", "physical_volume_owner_id", "density_kg_per_m3",
                            "initial_mass_kg", "provenance"}, key)
        identifier = row["stable_identifier"]
        _require(type(identifier) is int and identifier in expected,
                 f"{key} has an unknown stable identifier")
        _require(identifier not in result, f"{key} repeats stable identifier {identifier}")
        _require(isinstance(row["physical_volume_owner_id"], str)
                 and row["physical_volume_owner_id"].strip(),
                 f"{key} has an invalid physical volume owner")
        density = _positive(row["density_kg_per_m3"], f"{key} density")
        mass = _finite(row["initial_mass_kg"], f"{key} initial mass")
        _require(mass >= 0.0, f"{key} initial mass is negative")
        provenance = row["provenance"]
        _exact_object(provenance, {"kind", "description"}, f"{key} provenance")
        _require(provenance["kind"] == "synthetic_fixture"
                 and isinstance(provenance["description"], str)
                 and provenance["description"].strip(),
                 f"{key} mass provenance is not fixture-bound")
        result[identifier] = row
        row["density_kg_per_m3"] = density
        row["initial_mass_kg"] = mass
    _require(set(result) == set(expected), f"{key} does not cover every native owner")
    return result


def validate_owner_graph(native: dict[str, Any], owner: dict[str, Any], *, steps: int) -> dict[str, Any]:
    """Validate an explicit mass owner against compiled stable identifiers."""
    _exact_object(owner, {"schema", "id", "native_graph_sha256", "qualification",
                          "residual_tolerance_kg", "compartment_owners", "tissue_owners",
                          "exchange_schedules", "boundary"}, "blood mass owner graph")
    _require(owner["schema"] == OWNER_SCHEMA, "unsupported blood mass owner schema")
    _require(owner["qualification"] == "fixture_only", "blood mass owner is not fixture-only")
    _require(owner["native_graph_sha256"] == native.get("authored_graph_sha256"),
             "owner graph does not bind the compiled physiology graph")
    tolerance = _positive(owner["residual_tolerance_kg"], "mass residual tolerance")
    _require(tolerance < 1e-6, "mass residual tolerance is too broad")
    compartments = _indexed(native, "compartments")
    reservoirs = _indexed(native, "tissue_reservoirs")
    compartment_owners = _owner_rows(owner, "compartment_owners", compartments)
    tissue_owners = _owner_rows(owner, "tissue_owners", reservoirs)
    owner_ids = [row["physical_volume_owner_id"] for row in (*compartment_owners.values(),
                                                              *tissue_owners.values())]
    _require(len(owner_ids) == len(set(owner_ids)), "physical volume owner is duplicated")
    for identifier, row in compartment_owners.items():
        expected_mass = row["density_kg_per_m3"] * compartments[identifier]["initial_volume_m3"]
        _require(math.isclose(row["initial_mass_kg"], expected_mass, rel_tol=0.0,
                              abs_tol=tolerance),
                 f"compartment {identifier} initial mass disagrees with density and volume")
    for identifier, row in tissue_owners.items():
        expected_mass = row["density_kg_per_m3"] * reservoirs[identifier]["volume_m3"]
        _require(math.isclose(row["initial_mass_kg"], expected_mass, rel_tol=0.0,
                              abs_tol=tolerance),
                 f"tissue reservoir {identifier} initial mass disagrees with density and volume")

    exchanges = _indexed(native, "exchanges", allow_empty=True)
    schedules = owner["exchange_schedules"]
    _require(isinstance(schedules, list), "exchange schedules must be an array")
    schedule_by_id: dict[int, dict[str, Any]] = {}
    for schedule in schedules:
        _exact_object(schedule, {"exchange_stable_identifier", "flow_schedule_m3_per_s"},
                      "exchange schedule")
        identifier = schedule["exchange_stable_identifier"]
        _require(type(identifier) is int and identifier in exchanges,
                 "exchange schedule has an unknown stable identifier")
        _require(identifier not in schedule_by_id, "exchange schedule is duplicated")
        values = schedule["flow_schedule_m3_per_s"]
        _require(isinstance(values, list) and values, "exchange schedule is empty")
        _require(len(values) <= 4096, "exchange schedule is too long")
        for index, value in enumerate(values):
            flow = _finite(value, f"exchange schedule {identifier}[{index}]")
            # The source owner is a bounded finite-volume fixture.  A schedule
            # that can empty a domain in one accepted step is never admitted.
            _require(abs(flow) <= 0.25, f"exchange schedule {identifier} flow is too large")
            values[index] = flow
        schedule_by_id[identifier] = schedule
    _require(set(schedule_by_id) == set(exchanges),
             "exchange schedules do not cover every native exchange")
    _require(isinstance(owner["boundary"], str) and owner["boundary"].strip(),
             "owner graph has no qualification boundary")
    return {
        "residual_tolerance_kg": tolerance,
        "compartments": compartments,
        "reservoirs": reservoirs,
        "compartment_owners": compartment_owners,
        "tissue_owners": tissue_owners,
        "exchanges": exchanges,
        "schedules": schedule_by_id,
        "steps": steps,
    }


def _initial_mass_state(validated: dict[str, Any]) -> dict[str, Any]:
    return {
        "compartments": {
            identifier: {
                "volume_m3": float(row["initial_volume_m3"]),
                "mass_kg": float(validated["compartment_owners"][identifier]["initial_mass_kg"]),
            }
            for identifier, row in validated["compartments"].items()
        },
        "tissue_reservoirs": {
            identifier: {
                "volume_m3": float(row["volume_m3"]),
                "mass_kg": float(validated["tissue_owners"][identifier]["initial_mass_kg"]),
            }
            for identifier, row in validated["reservoirs"].items()
        },
        "accepted_steps": 0,
        "accepted_time_s": 0.0,
    }


def _mass_totals(state: dict[str, Any]) -> dict[str, float]:
    compartments = state["compartments"].values()
    reservoirs = state["tissue_reservoirs"].values()
    return {
        "mass_kg": math.fsum(row["mass_kg"] for row in (*compartments, *reservoirs)),
        "volume_m3": math.fsum(row["volume_m3"] for row in (*compartments, *reservoirs)),
        "vascular_mass_kg": math.fsum(row["mass_kg"] for row in state["compartments"].values()),
        "tissue_mass_kg": math.fsum(row["mass_kg"] for row in state["tissue_reservoirs"].values()),
    }


def _public_mass_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_steps": state["accepted_steps"],
        "accepted_time_s": state["accepted_time_s"],
        "compartments": state["compartments"],
        "tissue_reservoirs": state["tissue_reservoirs"],
    }


def _mass_candidate(native: dict[str, Any], validated: dict[str, Any], state: dict[str, Any],
                    hydraulic_candidate: dict[str, Any], timestep_s: float) -> tuple[dict[str, Any], dict[str, int]]:
    candidate = copy.deepcopy(state)
    transfers = {"connection": 0, "blood_to_tissue": 0, "tissue_to_blood": 0}
    compartments = validated["compartments"]
    connections = _indexed(native, "connections")
    for identifier, row in connections.items():
        flow = _finite(hydraulic_candidate["connections"][identifier]["flow_m3_per_s"],
                       f"connection {row.get('id')} flow")
        if flow == 0.0:
            continue
        source = row["from"] if flow > 0.0 else row["to"]
        target = row["to"] if flow > 0.0 else row["from"]
        transported_volume = abs(timestep_s * flow)
        source_volume = state["compartments"][source]["volume_m3"]
        _require(transported_volume < source_volume,
                 f"connection {row.get('id')} empties its source owner")
        source_density = state["compartments"][source]["mass_kg"] / source_volume
        transfer = transported_volume * source_density
        candidate["compartments"][source]["volume_m3"] -= transported_volume
        candidate["compartments"][target]["volume_m3"] += transported_volume
        candidate["compartments"][source]["mass_kg"] -= transfer
        candidate["compartments"][target]["mass_kg"] += transfer
        transfers["connection"] += 1

    schedules = validated["schedules"]
    exchanges = validated["exchanges"]
    schedule_index = state["accepted_steps"]
    for identifier, schedule in schedules.items():
        row = exchanges[identifier]
        flow_values = schedule["flow_schedule_m3_per_s"]
        flow = flow_values[schedule_index % len(flow_values)]
        if flow == 0.0:
            continue
        blood = row["compartment"]
        tissue = row["tissue_reservoir"]
        source = blood if flow > 0.0 else tissue
        target = tissue if flow > 0.0 else blood
        source_table = candidate["compartments"] if source in candidate["compartments"] else candidate["tissue_reservoirs"]
        target_table = candidate["compartments"] if target in candidate["compartments"] else candidate["tissue_reservoirs"]
        transported_volume = abs(timestep_s * flow)
        source_volume = state["compartments"].get(source, state["tissue_reservoirs"].get(source))["volume_m3"]
        _require(transported_volume < source_volume,
                 f"exchange {row.get('id')} empties its source owner")
        source_density = state["compartments"].get(source, state["tissue_reservoirs"].get(source))["mass_kg"] / source_volume
        transfer = transported_volume * source_density
        source_table[source]["volume_m3"] -= transported_volume
        target_table[target]["volume_m3"] += transported_volume
        source_table[source]["mass_kg"] -= transfer
        target_table[target]["mass_kg"] += transfer
        transfers["blood_to_tissue" if flow > 0.0 else "tissue_to_blood"] += 1

    for table in (candidate["compartments"], candidate["tissue_reservoirs"]):
        for identifier, row in table.items():
            _require(math.isfinite(row["volume_m3"]) and row["volume_m3"] > 0.0,
                     f"mass owner {identifier} volume left positive domain")
            _require(math.isfinite(row["mass_kg"]) and row["mass_kg"] >= 0.0,
                     f"mass owner {identifier} mass left nonnegative domain")
    candidate["accepted_steps"] += 1
    candidate["accepted_time_s"] += timestep_s
    return candidate, transfers


def simulate(native: dict[str, Any], owner: dict[str, Any], *, steps: int,
             timestep_s: float, reject_step: int | None = None) -> dict[str, Any]:
    """Run accepted hydraulic and explicit mass-owner steps."""
    _require(native.get("qualification") == "fixture_only", "only fixture graphs may run mass fixture")
    _require(type(steps) is int and 1 <= steps <= MAX_STEPS, "steps out of range")
    _require(type(timestep_s) in (int, float) and math.isfinite(timestep_s) and timestep_s > 0.0,
             "timestep must be finite and positive")
    if reject_step is not None:
        _require(type(reject_step) is int and 1 <= reject_step <= steps,
                 "reject_step must be within attempted steps")
    validated = validate_owner_graph(native, owner, steps=steps)
    physiology_state = _state_from_native(native)
    mass_state = _initial_mass_state(validated)
    initial_mass = _mass_totals(mass_state)
    initial_physiology = _totals(native, physiology_state)
    rejected = 0
    accepted_trace: list[str] = []
    transfer_counts = {"connection": 0, "blood_to_tissue": 0, "tissue_to_blood": 0}
    for attempt in range(1, steps + 1):
        before_physiology = copy.deepcopy(physiology_state)
        before_mass = copy.deepcopy(mass_state)
        candidate_physiology = _candidate(native, physiology_state, float(timestep_s))
        candidate_mass, transfers = _mass_candidate(
            native, validated, mass_state, candidate_physiology, float(timestep_s)
        )
        if reject_step == attempt:
            physiology_state = before_physiology
            mass_state = before_mass
            rejected += 1
            continue
        physiology_state = candidate_physiology
        mass_state = candidate_mass
        for key, value in transfers.items():
            transfer_counts[key] += value
        accepted_trace.append(_digest({
            "physiology": physiology_state,
            "mass": _public_mass_state(mass_state),
        }))
    final_mass = _mass_totals(mass_state)
    final_physiology = _totals(native, physiology_state)
    mass_residual = final_mass["mass_kg"] - initial_mass["mass_kg"]
    volume_residual = final_mass["volume_m3"] - initial_mass["volume_m3"]
    positive_steps = sum(
        1 for index in range(mass_state["accepted_steps"])
        for schedule in validated["schedules"].values()
        if schedule["flow_schedule_m3_per_s"][index % len(schedule["flow_schedule_m3_per_s"])] > 0.0
    )
    negative_steps = sum(
        1 for index in range(mass_state["accepted_steps"])
        for schedule in validated["schedules"].values()
        if schedule["flow_schedule_m3_per_s"][index % len(schedule["flow_schedule_m3_per_s"])] < 0.0
    )
    tolerance = validated["residual_tolerance_kg"]
    return {
        "schema": SCHEMA,
        "model_id": owner["id"],
        "native_graph_sha256": native.get("authored_graph_sha256"),
        "owner_graph_sha256": _digest(owner),
        "qualification": "fixture_only",
        "initial_mass_state": _public_mass_state(_initial_mass_state(validated)),
        "final_mass_state": _public_mass_state(mass_state),
        "initial_hydraulic_totals": initial_physiology,
        "final_hydraulic_totals": final_physiology,
        "attempted_steps": steps,
        "accepted_steps": mass_state["accepted_steps"],
        "rejected_steps": rejected,
        "timestep_seconds": float(timestep_s),
        "mass_conservation": {
            "initial": initial_mass,
            "final": final_mass,
            "mass_residual_kg": mass_residual,
            "owned_volume_residual_m3": volume_residual,
            "mass_conserved": abs(mass_residual) <= tolerance,
            "owned_volume_conserved": abs(volume_residual) <= 1e-18,
        },
        "transfer_counts": transfer_counts,
        "two_way_exchange": {
            "positive_accepted_steps": positive_steps,
            "negative_accepted_steps": negative_steps,
            "both_directions_observed": positive_steps > 0 and negative_steps > 0,
        },
        "rollback": {
            "accepted_time_excludes_rejections": math.isclose(
                mass_state["accepted_time_s"], mass_state["accepted_steps"] * float(timestep_s),
                rel_tol=0.0, abs_tol=1e-15,
            ),
            "rejected_steps": rejected,
        },
        "accepted_state_trace_sha256": _digest(accepted_trace),
        "qualification_boundary": (
            "Fixture-only explicit mass ownership, conservative flow/exchange transfer, "
            "two-way exchange, and accepted-candidate rollback. This does not qualify "
            "anatomical vessel tubes or lumens, blood density, organ mechanics, tissue "
            "perfusion, cardiac activation, material calibration, subject calibration, "
            "or standing/walking."
        ),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--owners", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--timestep-seconds", type=float, default=1.0e-4)
    parser.add_argument("--reject-step", type=int)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    graph = read_json(args.graph)
    owner = read_json(args.owners)
    native = compile_graph(graph, sources=args.sources, source_lock=args.source_lock)
    receipt = simulate(native, owner, steps=args.steps, timestep_s=args.timestep_seconds,
                       reject_step=args.reject_step)
    payload = canonical(receipt) + b"\n"
    output = args.output.resolve()
    if output.exists():
        _require(output.read_bytes() == payload, "output is immutable; choose a new output path")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(payload)
    print(json.dumps({"schema": SCHEMA, "output": str(output),
                      "sha256": hashlib.sha256(payload).hexdigest(),
                      "accepted_steps": receipt["accepted_steps"],
                      "rejected_steps": receipt["rejected_steps"],
                      "mass_conserved": receipt["mass_conservation"]["mass_conserved"],
                      "two_way_exchange": receipt["two_way_exchange"]["both_directions_observed"],
                      "qualification": receipt["qualification"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"blood-mass-step: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
