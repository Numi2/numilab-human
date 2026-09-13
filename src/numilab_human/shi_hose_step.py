"""Step the pinned Shi/Hose closed-loop cardiac source.

``shi_hose`` already compiles the 15-file CellML source into ten storage
compartments and ten source connections.  This module owns only the numerical
source-model reproduction step: source elastance activation, one-way orifice
flow, resistive/inertive flow, conservative volume transfer, and accepted-step
rollback.  It does not add absolute blood volume, tissue exchange, organ
mechanics, material parameters, or subject calibration.
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
from .shi_hose import CONFIG, ROOT, canonical, compile_source, read_json


SCHEMA = "HumanPack.shi-hose-step-receipt.v1"
MAX_STEPS = 1_000_000


class StepError(HumanImportError):
    """A source cardiac candidate cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StepError("Shi/Hose step: " + message)


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


def _indexed(native: dict[str, Any], table: str) -> dict[int, dict[str, Any]]:
    rows = native.get(table)
    _require(isinstance(rows, list) and rows, f"native source has no {table}")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), f"{table} contains a malformed row")
        identifier = row.get("stable_identifier")
        _require(type(identifier) is int and identifier > 0,
                 f"{table} has an invalid stable identifier")
        _require(identifier not in result, f"{table} repeats stable identifier {identifier}")
        result[identifier] = row
    return dict(sorted(result.items()))


def _validate_native(native: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    _require(native.get("schema") == "HumanPack.physiology-native.v2",
             "unsupported Shi/Hose native schema")
    _require(native.get("qualification") == "source_model_reproduction",
             "native source is not source-model reproduction")
    _require(native.get("law") == "closed_periodic_elastance_orifice_v2",
             "unsupported Shi/Hose source law")
    _require(native.get("species") == [] and native.get("tissue_reservoirs") == []
             and native.get("exchanges") == [],
             "source step cannot invent species or tissue state")
    compartments = _indexed(native, "compartments")
    connections = _indexed(native, "connections")
    _require(len(compartments) == 10 and len(connections) == 10,
             "source topology must contain ten compartments and ten connections")
    for identifier, row in compartments.items():
        _positive(row["volume_scale_m3"], f"compartment {row.get('id')} volume scale")
        _positive(row["volume_residual_tolerance"], f"compartment {row.get('id')} tolerance")
        _require(row["storage_kind"] in {"absolute_volume", "storage_displacement"},
                 f"compartment {row.get('id')} storage kind is unsupported")
        if row["storage_kind"] == "absolute_volume":
            _require(row["pressure_law"] in {"atrial_elastance", "ventricular_elastance"},
                     f"compartment {row.get('id')} source elastance is unsupported")
            _positive(row["period_seconds"], f"compartment {row.get('id')} period")
            _positive(row["elastance_min_pa_per_m3"], f"compartment {row.get('id')} minimum elastance")
            _positive(row["elastance_max_pa_per_m3"], f"compartment {row.get('id')} maximum elastance")
            _positive(row["activation_end"], f"compartment {row.get('id')} activation duration")
            _finite(row["activation_start"], f"compartment {row.get('id')} activation start")
            _positive(row["source_pi"], f"compartment {row.get('id')} source pi")
        else:
            _positive(row["compliance_m3_per_pa"], f"compartment {row.get('id')} compliance")
        initial = _finite(row["initial_volume_m3"], f"compartment {row.get('id')} initial volume")
        _require(initial >= 0.0, f"compartment {row.get('id')} initial volume is negative")
    for identifier, row in connections.items():
        _require(type(row.get("from")) is int and row["from"] in compartments
                 and type(row.get("to")) is int and row["to"] in compartments
                 and row["from"] != row["to"],
                 f"connection {row.get('id')} endpoint is invalid")
        _require(row["flow_law"] in {"one_way_orifice", "resistance_inertance"},
                 f"connection {row.get('id')} flow law is unsupported")
        _positive(row["flow_scale_m3_per_s"], f"connection {row.get('id')} flow scale")
        _positive(row["pressure_scale_pa"], f"connection {row.get('id')} pressure scale")
        if row["flow_law"] == "one_way_orifice":
            _positive(row["orifice_coefficient_m3_per_s_sqrt_pa"],
                      f"connection {row.get('id')} orifice coefficient")
        else:
            _positive(row["resistance_pa_s_per_m3"], f"connection {row.get('id')} resistance")
            inertance = _finite(row["inertance_pa_s2_per_m3"],
                                f"connection {row.get('id')} inertance")
            _require(inertance >= 0.0, f"connection {row.get('id')} inertance is negative")
    return compartments, connections


def _activation(row: dict[str, Any], time_s: float) -> float:
    if row["storage_kind"] != "absolute_volume":
        return 0.0
    period = float(row["period_seconds"])
    phase = math.fmod(time_s, period)
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


def _pressures(native: dict[str, Any], state: dict[str, Any]) -> dict[int, float]:
    compartments, _ = _validate_native(native)
    pressures: dict[int, float] = {}
    for identifier, row in compartments.items():
        volume = state["compartments"][identifier]["volume_m3"]
        _require(math.isfinite(volume) and volume >= 0.0,
                 f"compartment {row.get('id')} volume is invalid")
        if row["storage_kind"] == "absolute_volume":
            activation = _activation(row, state["accepted_time_s"])
            elastance = row["elastance_min_pa_per_m3"] + activation * (
                row["elastance_max_pa_per_m3"] - row["elastance_min_pa_per_m3"])
            pressure = row["reference_pressure_pa"] + elastance * (
                volume - row["reference_volume_m3"])
        else:
            pressure = row["external_pressure_pa"] + volume / row["compliance_m3_per_pa"]
        _require(math.isfinite(pressure), f"compartment {row.get('id')} pressure is invalid")
        pressures[identifier] = float(pressure)
    return pressures


def _state_from_native(native: dict[str, Any]) -> dict[str, Any]:
    compartments, connections = _validate_native(native)
    return {
        "compartments": {
            identifier: {"volume_m3": _finite(row["initial_volume_m3"], f"{row.get('id')} volume")}
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
    _require(math.isfinite(timestep_s) and timestep_s > 0.0, "timestep must be finite and positive")
    compartments, connections = _validate_native(native)
    pressures = _pressures(native, state)
    candidate = copy.deepcopy(state)
    deltas = {identifier: 0.0 for identifier in compartments}
    open_valves = 0
    activation = {}
    for identifier, row in compartments.items():
        activation[row["id"]] = _activation(row, state["accepted_time_s"])
    for identifier, row in connections.items():
        drop = pressures[row["from"]] - pressures[row["to"]]
        old_flow = state["connections"][identifier]["flow_m3_per_s"]
        if row["flow_law"] == "one_way_orifice":
            flow = row["orifice_coefficient_m3_per_s_sqrt_pa"] * math.sqrt(max(drop, 0.0))
            if flow > 0.0:
                open_valves += 1
        elif row["inertance_pa_s2_per_m3"] == 0.0:
            flow = drop / row["resistance_pa_s_per_m3"]
        else:
            inertance = row["inertance_pa_s2_per_m3"]
            flow = (old_flow + timestep_s * drop / inertance) / (
                1.0 + timestep_s * row["resistance_pa_s_per_m3"] / inertance)
        _require(math.isfinite(flow), f"connection {row.get('id')} flow is invalid")
        candidate["connections"][identifier]["flow_m3_per_s"] = flow
        signed_volume = timestep_s * flow
        deltas[row["from"]] -= signed_volume
        deltas[row["to"]] += signed_volume
    for identifier, row in compartments.items():
        volume = state["compartments"][identifier]["volume_m3"] + deltas[identifier]
        _require(math.isfinite(volume) and volume >= 0.0,
                 f"compartment {row.get('id')} left the nonnegative volume domain")
        candidate["compartments"][identifier]["volume_m3"] = volume
    candidate["accepted_steps"] += 1
    candidate["accepted_time_s"] += timestep_s
    return candidate, {"open_valves": open_valves, "activation": activation}


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted_steps": state["accepted_steps"],
        "accepted_time_s": state["accepted_time_s"],
        "compartments": state["compartments"],
        "connections": state["connections"],
    }


def _volume_total(state: dict[str, Any]) -> float:
    return math.fsum(row["volume_m3"] for row in state["compartments"].values())


def simulate(native: dict[str, Any], *, steps: int, timestep_s: float,
             reject_step: int | None = None) -> dict[str, Any]:
    """Run source elastance/valve steps with accepted-candidate rollback."""
    _require(type(steps) is int and 1 <= steps <= MAX_STEPS, "steps out of range")
    _require(type(timestep_s) in (int, float) and math.isfinite(timestep_s) and timestep_s > 0.0,
             "timestep must be finite and positive")
    if reject_step is not None:
        _require(type(reject_step) is int and 1 <= reject_step <= steps,
                 "reject_step must be within attempted steps")
    _validate_native(native)
    state = _state_from_native(native)
    initial = _public_state(state)
    initial_volume = _volume_total(state)
    rejected = 0
    open_valve_evaluations = 0
    accepted_trace: list[str] = []
    activation_trace: list[dict[str, float]] = []
    for attempt in range(1, steps + 1):
        before = copy.deepcopy(state)
        candidate, diagnostics = _candidate(native, state, float(timestep_s))
        if reject_step == attempt:
            state = before
            rejected += 1
            continue
        state = candidate
        open_valve_evaluations += diagnostics["open_valves"]
        activation_trace.append(diagnostics["activation"])
        accepted_trace.append(_digest(_public_state(state)))
    final = _public_state(state)
    final_volume = _volume_total(state)
    volume_residual = final_volume - initial_volume
    return {
        "schema": SCHEMA,
        "model_id": native.get("model_id"),
        "qualification": "source_model_reproduction",
        "source_graph_sha256": native.get("source_graph_sha256"),
        "native_graph_sha256": native.get("authored_graph_sha256"),
        "initial_state": initial,
        "final_state": final,
        "attempted_steps": steps,
        "accepted_steps": state["accepted_steps"],
        "rejected_steps": rejected,
        "timestep_seconds": float(timestep_s),
        "conservation": {
            "initial_compartment_volume_m3": initial_volume,
            "final_compartment_volume_m3": final_volume,
            "volume_residual_m3": volume_residual,
            "volume_conserved": abs(volume_residual) <= 1e-15,
        },
        "activation": {
            "accepted_trace_sha256": _digest(activation_trace),
            "open_valve_evaluation_count": open_valve_evaluations,
        },
        "rollback": {
            "accepted_time_excludes_rejections": math.isclose(
                state["accepted_time_s"], state["accepted_steps"] * float(timestep_s),
                rel_tol=0.0, abs_tol=1e-15,
            ),
            "rejected_steps": rejected,
        },
        "accepted_state_trace_sha256": _digest(accepted_trace),
        "qualification_boundary": (
            "Pinned Shi/Hose source-model reproduction only: source elastance, "
            "one-way orifice, resistance/inertance, conservative compartment volume, "
            "and accepted-step rollback. This does not qualify absolute vascular "
            "blood volume, dilution/species transport, organ mechanics, tissue "
            "exchange, material calibration, subject calibration, standing, or walking."
        ),
    }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-directory", type=Path, default=ROOT / "third_party/physiome/shi_hose_2009")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--timestep-seconds", type=float, default=1.0e-4)
    parser.add_argument("--reject-step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    native, lowering = compile_source(directory=args.source_directory, config=read_json(args.config))
    receipt = simulate(native, steps=args.steps, timestep_s=args.timestep_seconds,
                       reject_step=args.reject_step)
    receipt["source_lowering_manifest_sha256"] = _digest(lowering)
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
                      "open_valve_evaluation_count": receipt["activation"]["open_valve_evaluation_count"],
                      "qualification": receipt["qualification"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"shi-hose-step: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
