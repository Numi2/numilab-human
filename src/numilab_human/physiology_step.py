"""Advance a compiled organ/circulation graph with accepted-step semantics.

The stepper is deliberately small and conservative.  It owns only the
source-bound graph's hydraulic volumes, edge flows, and declared species
amounts; it does not infer anatomy, blood density, vessel geometry, cardiac
activation, or tissue mechanics.  A graph compiled from the synthetic fixture
is therefore executable evidence for conservation and rollback only.
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


SCHEMA = "HumanPack.physiology-step-receipt.v1"
MAX_STEPS = 1_000_000


class StepError(HumanImportError):
    """A candidate physiology step cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StepError("physiology step: " + message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value),
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


def _state_from_native(native: dict[str, Any]) -> dict[str, Any]:
    compartments = _indexed(native, "compartments")
    connections = _indexed(native, "connections")
    reservoirs = _indexed(native, "tissue_reservoirs", allow_empty=True)
    exchanges = _indexed(native, "exchanges") if native.get("exchanges") else {}
    species = _indexed(native, "species")
    species_ids = tuple(species)
    state = {
        "compartments": {
            key: {
                "volume_m3": _positive(row["initial_volume_m3"],
                                        f"compartment {row.get('id')}.initial_volume_m3"),
                "species_mol": [
                    _finite(amount, f"compartment {row.get('id')}.initial_species_mol[{index}]")
                    for index, amount in enumerate(row["initial_species_mol"])
                ],
            }
            for key, row in compartments.items()
        },
        "connections": {
            key: {"flow_m3_per_s": _finite(row["initial_flow_m3_per_s"],
                                            f"connection {row.get('id')}.initial_flow_m3_per_s")}
            for key, row in connections.items()
        },
        "tissue_reservoirs": {
            key: {
                "species_mol": [
                    _finite(amount, f"reservoir {row.get('id')}.initial_species_mol[{index}]")
                    for index, amount in enumerate(row["initial_species_mol"])
                ]
            }
            for key, row in reservoirs.items()
        },
        "accepted_steps": 0,
        "accepted_time_s": 0.0,
        "species_ids": species_ids,
    }
    _require(all(len(row["species_mol"]) == len(species_ids)
                 for row in state["compartments"].values()),
             "compartment species width disagrees with species table")
    _require(all(len(row["species_mol"]) == len(species_ids)
                 for row in state["tissue_reservoirs"].values()),
             "reservoir species width disagrees with species table")
    _require(all(amount >= 0.0 for row in state["compartments"].values()
                 for amount in row["species_mol"]),
             "compartment species amount is negative")
    _require(all(amount >= 0.0 for row in state["tissue_reservoirs"].values()
                 for amount in row["species_mol"]),
             "reservoir species amount is negative")
    return state


def _pressures(native: dict[str, Any], state: dict[str, Any]) -> dict[int, float]:
    rows = _indexed(native, "compartments")
    pressures = {}
    for identifier, row in rows.items():
        volume = state["compartments"][identifier]["volume_m3"]
        compliance = _positive(row["compliance_m3_per_pa"],
                               f"compartment {row.get('id')}.compliance_m3_per_pa")
        reference = _finite(row["reference_volume_m3"],
                            f"compartment {row.get('id')}.reference_volume_m3")
        pressures[identifier] = (
            _finite(row["reference_pressure_pa"], f"compartment {row.get('id')}.reference_pressure_pa")
            + _finite(row["external_pressure_pa"], f"compartment {row.get('id')}.external_pressure_pa")
            + (volume - reference) / compliance
        )
        _require(math.isfinite(pressures[identifier]),
                 f"compartment {row.get('id')} pressure is not finite")
    return pressures


def _candidate(native: dict[str, Any], state: dict[str, Any], timestep_s: float) -> dict[str, Any]:
    """Return one candidate state, preserving exact table order."""
    _require(timestep_s > 0.0 and math.isfinite(timestep_s), "timestep must be finite and positive")
    compartments = _indexed(native, "compartments")
    connections = _indexed(native, "connections")
    reservoirs = _indexed(native, "tissue_reservoirs", allow_empty=True)
    exchanges = _indexed(native, "exchanges") if native.get("exchanges") else {}
    species_count = len(state["species_ids"])
    pressures = _pressures(native, state)
    candidate = copy.deepcopy(state)
    volume_delta = {identifier: 0.0 for identifier in compartments}

    # Semi-implicit edge flow plus conservative upwind species transport.
    for identifier, row in connections.items():
        source = row["from"]
        target = row["to"]
        _require(source in compartments and target in compartments and source != target,
                 f"connection {row.get('id')} has an invalid endpoint")
        resistance = _positive(row["resistance_pa_s_per_m3"],
                               f"connection {row.get('id')}.resistance_pa_s_per_m3")
        inertance = _finite(row.get("inertance_pa_s2_per_m3", 0.0),
                            f"connection {row.get('id')}.inertance_pa_s2_per_m3")
        _require(inertance >= 0.0, f"connection {row.get('id')} inertance is negative")
        old_flow = state["connections"][identifier]["flow_m3_per_s"]
        pressure_drop = pressures[source] - pressures[target]
        if inertance == 0.0:
            new_flow = pressure_drop / resistance
        else:
            numerator = old_flow + timestep_s * pressure_drop / inertance
            denominator = 1.0 + timestep_s * resistance / inertance
            new_flow = numerator / denominator
        _require(math.isfinite(new_flow), f"connection {row.get('id')} flow is not finite")
        candidate["connections"][identifier]["flow_m3_per_s"] = new_flow
        signed_volume = timestep_s * new_flow
        volume_delta[source] -= signed_volume
        volume_delta[target] += signed_volume

        if new_flow == 0.0:
            continue
        upstream = source if new_flow > 0.0 else target
        downstream = target if new_flow > 0.0 else source
        transported_volume = abs(signed_volume)
        upstream_volume = state["compartments"][upstream]["volume_m3"]
        _require(transported_volume <= upstream_volume,
                 f"connection {row.get('id')} transports more volume than upstream state")
        fraction = transported_volume / upstream_volume
        for species_index in range(species_count):
            amount = state["compartments"][upstream]["species_mol"][species_index] * fraction
            candidate["compartments"][upstream]["species_mol"][species_index] -= amount
            candidate["compartments"][downstream]["species_mol"][species_index] += amount

    # Conservative blood/tissue exchange.  A positive driving concentration
    # moves species from blood to tissue; a negative one reverses direction.
    for identifier, row in exchanges.items():
        compartment = row["compartment"]
        reservoir = row["tissue_reservoir"]
        species_index = row["species"] - 1
        _require(compartment in compartments and reservoir in reservoirs and
                 0 <= species_index < species_count,
                 f"exchange {row.get('id')} has an invalid endpoint")
        clearance = _positive(row["clearance_m3_per_s"],
                              f"exchange {row.get('id')}.clearance_m3_per_s")
        partition = _positive(row.get("partition_coefficient", 1.0),
                              f"exchange {row.get('id')}.partition_coefficient")
        blood_volume = state["compartments"][compartment]["volume_m3"]
        tissue_volume = _positive(
            reservoirs[reservoir]["volume_m3"], f"reservoir {reservoir}.volume_m3"
        )
        blood_concentration = (state["compartments"][compartment]["species_mol"][species_index]
                               / blood_volume)
        tissue_concentration = (state["tissue_reservoirs"][reservoir]["species_mol"][species_index]
                                / tissue_volume)
        transfer = timestep_s * clearance * (blood_concentration - tissue_concentration / partition)
        if transfer > 0.0:
            transfer = min(transfer, state["compartments"][compartment]["species_mol"][species_index])
        else:
            transfer = -min(-transfer,
                            state["tissue_reservoirs"][reservoir]["species_mol"][species_index])
        candidate["compartments"][compartment]["species_mol"][species_index] -= transfer
        candidate["tissue_reservoirs"][reservoir]["species_mol"][species_index] += transfer

    for identifier, row in compartments.items():
        volume = state["compartments"][identifier]["volume_m3"] + volume_delta[identifier]
        _require(math.isfinite(volume) and volume > 0.0,
                 f"compartment {row.get('id')} volume left the positive domain")
        candidate["compartments"][identifier]["volume_m3"] = volume
    for table in (candidate["compartments"], candidate["tissue_reservoirs"]):
        for row in table.values():
            for index, amount in enumerate(row["species_mol"]):
                _require(math.isfinite(amount) and amount >= 0.0,
                         f"species amount {index} left the nonnegative domain")
    candidate["accepted_steps"] += 1
    candidate["accepted_time_s"] += timestep_s
    return candidate


def _totals(native: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    compartments = _indexed(native, "compartments")
    reservoirs = _indexed(native, "tissue_reservoirs", allow_empty=True)
    return {
        "compartment_volume_m3": sum(state["compartments"][key]["volume_m3"] for key in compartments),
        "species_mol": [
            sum(state["compartments"][key]["species_mol"][index] for key in compartments)
            + sum(state["tissue_reservoirs"][key]["species_mol"][index] for key in reservoirs)
            for index in range(len(state["species_ids"]))
        ],
    }


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_steps": state["accepted_steps"],
        "accepted_time_s": state["accepted_time_s"],
        "compartments": state["compartments"],
        "connections": state["connections"],
        "tissue_reservoirs": state["tissue_reservoirs"],
    }


def simulate(native: dict[str, Any], *, steps: int, timestep_s: float,
             reject_step: int | None = None) -> dict[str, Any]:
    """Run bounded accepted steps and return a conservative engineering receipt.

    ``reject_step`` is one-based in attempted-step space and is a deterministic
    rollback exercise.  Rejected candidates never advance accepted time or
    alter the public state.
    """
    _require(type(steps) is int and 1 <= steps <= MAX_STEPS, "steps out of range")
    _require(type(timestep_s) in (int, float) and math.isfinite(timestep_s) and timestep_s > 0.0,
             "timestep must be finite and positive")
    if reject_step is not None:
        _require(type(reject_step) is int and 1 <= reject_step <= steps,
                 "reject_step must be within attempted steps")
    state = _state_from_native(native)
    initial_totals = _totals(native, state)
    initial_state = _public_state(state)
    rejected = 0
    accepted_hashes = []
    for attempt in range(1, steps + 1):
        before = copy.deepcopy(state)
        # A deliberate rejection is an execution decision.  Structural or
        # numerical errors from the candidate builder propagate instead of
        # being misreported as accepted-step rollback.
        candidate = _candidate(native, state, float(timestep_s))
        if reject_step == attempt:
            state = before
            rejected += 1
            continue
        state = candidate
        accepted_hashes.append(_digest(_public_state(state)))
    final_totals = _totals(native, state)
    volume_residual = final_totals["compartment_volume_m3"] - initial_totals["compartment_volume_m3"]
    species_residual = [after - before for after, before in zip(final_totals["species_mol"], initial_totals["species_mol"])]
    final_state = _public_state(state)
    return {
        "schema": SCHEMA,
        "model_id": native.get("model_id"),
        "native_graph_sha256": native.get("authored_graph_sha256"),
        "qualification": "fixture_only" if native.get("qualification") == "fixture_only" else "not_assessed",
        "initial_state": initial_state,
        "final_state": final_state,
        "attempted_steps": steps,
        "accepted_steps": state["accepted_steps"],
        "rejected_steps": rejected,
        "timestep_seconds": float(timestep_s),
        "conservation": {
            "initial": initial_totals,
            "final": final_totals,
            "compartment_volume_residual_m3": volume_residual,
            "species_residual_mol": species_residual,
            "volume_conserved": abs(volume_residual) <= 1e-18,
            "species_conserved": all(abs(value) <= 1e-18 for value in species_residual),
        },
        "rollback": {
            "exercise_requested": reject_step is not None,
            "rejected_steps": rejected,
            "accepted_time_excludes_rejections": math.isclose(
                state["accepted_time_s"], state["accepted_steps"] * float(timestep_s),
                rel_tol=0.0, abs_tol=1e-15,
            ),
        },
        "accepted_state_trace_sha256": _digest(accepted_hashes),
        "qualification_boundary": (
            "Conservative compiled-graph stepping and rejected-candidate rollback only. "
            "This does not qualify anatomical vessel tubes, blood density, cardiac activation, "
            "organ mechanics, material calibration, subject calibration, or standing/walking."
        ),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--timestep-seconds", type=float, default=1.0e-4)
    parser.add_argument("--reject-step", type=int)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    graph = read_json(args.graph)
    native = compile_graph(graph, sources=args.sources, source_lock=args.source_lock)
    receipt = simulate(native, steps=args.steps, timestep_s=args.timestep_seconds,
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
                      "volume_conserved": receipt["conservation"]["volume_conserved"],
                      "species_conserved": receipt["conservation"]["species_conserved"],
                      "qualification": receipt["qualification"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"physiology-step: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
