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
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any

from .model import ImportError as HumanImportError
from .shi_hose import CONFIG, ROOT, canonical, compile_source, read_json


SCHEMA = "HumanPack.shi-hose-step-receipt.v1"
MAX_STEPS = 2_000_000
CLOCK_NANOSECONDS = 12_500
CLOCK_SECONDS = CLOCK_NANOSECONDS * 1.0e-9
CLOCK_TOLERANCE_SECONDS = 1.0e-15


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
    end = float(row["activation_end"])
    pi = float(row["source_pi"])
    if row["pressure_law"] == "atrial_elastance":
        # EAtrium.cellml has a raised-cosine tail that wraps over the beat
        # boundary when Tpwb + Tpww > 1. The wrapped and final-cycle branches
        # are the same waveform, expressed on opposite sides of that boundary.
        wrap_end = (start + end - 1.0) * period
        if 0.0 <= phase <= wrap_end:
            et = 1.0 - math.cos(2.0 * pi * (phase - start * period + period) / (end * period))
        elif wrap_end < phase <= start * period:
            et = 0.0
        elif start * period < phase <= period:
            et = 1.0 - math.cos(2.0 * pi * (phase - start * period) / (end * period))
        else:
            et = 0.0
    elif row["pressure_law"] == "ventricular_elastance":
        # EVentricle.cellml uses Ts1 and Ts2 as absolute phase boundaries:
        # rise on [0, Ts1], fall on (Ts1, Ts2], then zero until the next beat.
        if 0.0 <= phase <= start * period:
            et = 1.0 - math.cos(pi * phase / (start * period))
        elif start * period < phase <= end * period:
            et = 1.0 + math.cos(pi * (phase - start * period) / ((end - start) * period))
        else:
            et = 0.0
    else:
        _require(False, f"compartment {row.get('id')} source activation law is unsupported")
        et = 0.0
    value = 0.5 * et
    _require(math.isfinite(value) and 0.0 <= value <= 1.0 + 1e-12,
             f"compartment {row.get('id')} activation left [0,1]")
    return min(1.0, max(0.0, value))


def _pressures(native: dict[str, Any], state: dict[str, Any], *,
               compartments: dict[int, dict[str, Any]] | None = None) -> dict[int, float]:
    if compartments is None:
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


def _candidate(native: dict[str, Any], state: dict[str, Any], timestep_s: float, *,
               compartments: dict[int, dict[str, Any]] | None = None,
               connections: dict[int, dict[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(math.isfinite(timestep_s) and timestep_s > 0.0, "timestep must be finite and positive")
    if compartments is None or connections is None:
        compartments, connections = _validate_native(native)
    pressures = _pressures(native, state, compartments=compartments)
    # Every leaf in the accepted state is a scalar, so clone only the two
    # mutable record maps instead of recursively walking them on every step.
    candidate = {
        "compartments": {identifier: values.copy()
                         for identifier, values in state["compartments"].items()},
        "connections": {identifier: values.copy()
                        for identifier, values in state["connections"].items()},
        "accepted_steps": state["accepted_steps"],
        "accepted_time_s": state["accepted_time_s"],
    }
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
    candidate["accepted_time_s"] = candidate["accepted_steps"] * timestep_s
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


_TRACE_COLUMNS = (
    "time_s", "activation_la", "activation_lv", "activation_ra", "activation_rv",
    "pressure_la_pa", "pressure_lv_pa", "pressure_ra_pa", "pressure_rv_pa",
    "volume_la_m3", "volume_lv_m3", "volume_ra_m3", "volume_rv_m3",
    "mitral_flow_m3_s", "aortic_flow_m3_s", "tricuspid_flow_m3_s",
    "pulmonary_flow_m3_s", "pulmonary_venous_return_m3_s",
    "systemic_venous_return_m3_s",
)
_TRACE_COMPARTMENTS = {"la": 1, "lv": 2, "ra": 6, "rv": 7}
_TRACE_CONNECTION_NAMES = {
    "mitral": "LA_outflow", "aortic": "LV_outflow",
    "tricuspid": "RA_outflow", "pulmonary": "RV_outflow",
    "pulmonary_venous_return": "Pvn_outflow",
    "systemic_venous_return": "Svn_outflow",
}


def simulate(native: dict[str, Any], *, steps: int, timestep_s: float,
             reject_step: int | None = None, trace_stride: int = 0) -> dict[str, Any]:
    """Run source elastance/valve steps with accepted-candidate rollback."""
    _require(type(steps) is int and 1 <= steps <= MAX_STEPS, "steps out of range")
    _require(type(timestep_s) in (int, float) and math.isfinite(timestep_s) and timestep_s > 0.0,
             "timestep must be finite and positive")
    if reject_step is not None:
        _require(type(reject_step) is int and 1 <= reject_step <= steps,
                 "reject_step must be within attempted steps")
    compartments, connections = _validate_native(native)
    state = _state_from_native(native)
    initial = _public_state(state)
    initial_volume = _volume_total(state)
    rejected = 0
    open_valve_evaluations = 0
    accepted_trace_hasher = hashlib.sha256()
    activation_trace_hasher = hashlib.sha256()
    accepted_trace_hasher.update(b"[")
    activation_trace_hasher.update(b"[")
    trace_digest_count = 0
    _require(type(trace_stride) is int and trace_stride >= 0,
             "trace stride must be a nonnegative integer")
    cycle_metrics: dict[int, dict[str, Any]] = {}
    trace_rows: list[tuple[str, ...]] = []
    period = 0.0
    period_steps = 0
    if trace_stride:
        periods = sorted({float(row["period_seconds"]) for row in compartments.values()
                          if row["storage_kind"] == "absolute_volume"})
        _require(len(periods) == 1, "four cardiac chambers do not share one source period")
        period = periods[0]
        source_period_steps = period / float(timestep_s)
        _require(math.isfinite(source_period_steps) and source_period_steps >= 1.0 and
                 math.isclose(source_period_steps, round(source_period_steps), rel_tol=0.0, abs_tol=1e-8),
                 "trace timestep does not divide the source cardiac period exactly")
        period_steps = int(round(source_period_steps))
        compartment_names = {row["id"].lower(): identifier
                             for identifier, row in compartments.items()}
        connection_names = {row["id"]: identifier
                            for identifier, row in connections.items()}
        _require(all(compartment_names.get(name) == identifier
                     for name, identifier in _TRACE_COMPARTMENTS.items()),
                 "source four-chamber identities changed")
        _require(all(name in connection_names for name in _TRACE_CONNECTION_NAMES.values()),
                 "source cardiac valve/return identities changed")
        trace_flow_ids = {name: connection_names[connection_name]
                          for name, connection_name in _TRACE_CONNECTION_NAMES.items()}
    else:
        trace_flow_ids = {}
    run_started = time.perf_counter()
    for attempt in range(1, steps + 1):
        candidate, diagnostics = _candidate(
            native, state, float(timestep_s),
            compartments=compartments, connections=connections)
        if reject_step == attempt:
            rejected += 1
            continue
        state = candidate
        open_valve_evaluations += diagnostics["open_valves"]
        if trace_digest_count:
            accepted_trace_hasher.update(b",")
            activation_trace_hasher.update(b",")
        accepted_trace_hasher.update(canonical(_digest(_public_state(state))))
        activation_trace_hasher.update(canonical(diagnostics["activation"]))
        trace_digest_count += 1
        if trace_stride:
            cycle_index = (state["accepted_steps"] - 1) // period_steps
            metrics = cycle_metrics.setdefault(cycle_index, {
                "cycle": cycle_index + 1,
                "lv_filling_m3": 0.0, "lv_aortic_ejection_m3": 0.0,
                "rv_filling_m3": 0.0, "rv_pulmonary_ejection_m3": 0.0,
                "pulmonary_venous_return_m3": 0.0,
                "systemic_venous_return_m3": 0.0,
                "lv_pressure_min_pa": math.inf, "lv_pressure_max_pa": -math.inf,
                "rv_pressure_min_pa": math.inf, "rv_pressure_max_pa": -math.inf,
            })
            flow_values = {
                name: state["connections"][identifier]["flow_m3_per_s"]
                for name, identifier in trace_flow_ids.items()
            }
            metrics["lv_filling_m3"] += flow_values["mitral"] * float(timestep_s)
            metrics["lv_aortic_ejection_m3"] += flow_values["aortic"] * float(timestep_s)
            metrics["rv_filling_m3"] += flow_values["tricuspid"] * float(timestep_s)
            metrics["rv_pulmonary_ejection_m3"] += flow_values["pulmonary"] * float(timestep_s)
            metrics["pulmonary_venous_return_m3"] += flow_values["pulmonary_venous_return"] * float(timestep_s)
            metrics["systemic_venous_return_m3"] += flow_values["systemic_venous_return"] * float(timestep_s)
            pressure_values = _pressures(native, state, compartments=compartments)
            for chamber in ("lv", "rv"):
                pressure = pressure_values[_TRACE_COMPARTMENTS[chamber]]
                metrics[f"{chamber}_pressure_min_pa"] = min(
                    metrics[f"{chamber}_pressure_min_pa"], pressure)
                metrics[f"{chamber}_pressure_max_pa"] = max(
                    metrics[f"{chamber}_pressure_max_pa"], pressure)
            if state["accepted_steps"] % trace_stride == 0:
                row_values: list[float] = [state["accepted_time_s"]]
                for chamber in ("la", "lv", "ra", "rv"):
                    identifier = _TRACE_COMPARTMENTS[chamber]
                    compartment = compartments[identifier]
                    row_values.append(_activation(compartment, state["accepted_time_s"]))
                for chamber in ("la", "lv", "ra", "rv"):
                    row_values.append(pressure_values[_TRACE_COMPARTMENTS[chamber]])
                for chamber in ("la", "lv", "ra", "rv"):
                    row_values.append(state["compartments"][_TRACE_COMPARTMENTS[chamber]]["volume_m3"])
                row_values.extend(flow_values[name] for name in _TRACE_CONNECTION_NAMES)
                trace_rows.append(tuple(format(value, ".17g") for value in row_values))
    final = _public_state(state)
    final_volume = _volume_total(state)
    volume_residual = final_volume - initial_volume
    accepted_trace_hasher.update(b"]")
    activation_trace_hasher.update(b"]")
    receipt = {
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
        "clock": {
            "timestep_nanoseconds": float(timestep_s) * 1.0e9,
            "required_nanoseconds": CLOCK_NANOSECONDS,
            "exact": abs(float(timestep_s) - CLOCK_SECONDS) <= CLOCK_TOLERANCE_SECONDS,
        },
        "conservation": {
            "initial_compartment_volume_m3": initial_volume,
            "final_compartment_volume_m3": final_volume,
            "volume_residual_m3": volume_residual,
            "volume_conserved": abs(volume_residual) <= 1e-15,
        },
        "activation": {
            "accepted_trace_sha256": activation_trace_hasher.hexdigest(),
            "open_valve_evaluation_count": open_valve_evaluations,
        },
        "rollback": {
            "accepted_time_excludes_rejections": math.isclose(
                state["accepted_time_s"], state["accepted_steps"] * float(timestep_s),
                rel_tol=0.0, abs_tol=1e-15,
            ),
            "rejected_steps": rejected,
        },
        "accepted_state_trace_sha256": accepted_trace_hasher.hexdigest(),
        "qualification_boundary": (
            "Pinned Shi/Hose source-model reproduction only: source elastance, "
            "one-way orifice, resistance/inertance, conservative compartment volume, "
            "and accepted-step rollback. This does not qualify absolute vascular "
            "blood volume, dilution/species transport, organ mechanics, tissue "
            "exchange, material calibration, subject calibration, standing, or walking."
        ),
    }
    if trace_stride:
        measured_wall_seconds = time.perf_counter() - run_started
        completed_cycles = state["accepted_steps"] // period_steps
        receipt["cardiac_cycle_result"] = {
            "model_period_seconds": period,
            "completed_source_cycles": completed_cycles,
            "cycle_metrics": [cycle_metrics[index] for index in range(completed_cycles)
                              if index in cycle_metrics],
            "trace_columns": list(_TRACE_COLUMNS),
            "trace_rows": trace_rows,
            "trace_stride_accepted_steps": trace_stride,
            "trace_sample_interval_seconds": trace_stride * float(timestep_s),
            "wall_seconds": measured_wall_seconds,
            "accepted_steps_per_second": state["accepted_steps"] /
                max(measured_wall_seconds, 1e-12),
            "mechanical_tissue_coupling": False,
            "anatomical_motion_claim": False,
        }
    return receipt


def _publish_immutable(path: Path, payload: bytes, label: str) -> None:
    output = path.resolve()
    if output.exists():
        _require(output.is_file() and output.read_bytes() == payload,
                 f"{label} exists with different content; choose a new output path")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + f".{os.getpid()}.pending")
    with pending.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, output)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-directory", type=Path, default=ROOT / "third_party/physiome/shi_hose_2009")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--timestep-seconds", type=float, default=1.0e-4)
    parser.add_argument(
        "--require-clock-nanoseconds",
        type=int,
        help="fail closed unless the requested timestep equals this clock in nanoseconds",
    )
    parser.add_argument("--reject-step", type=int)
    parser.add_argument("--trace-csv", type=Path,
                        help="write sampled four-chamber pressure/valve-flow trace")
    parser.add_argument("--trace-stride", type=int, default=1,
                        help="sample the trace every N accepted steps (default: 1)")
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace) -> int:
    required_clock_nanoseconds = getattr(args, "require_clock_nanoseconds", None)
    if required_clock_nanoseconds is not None:
        _require(
            type(required_clock_nanoseconds) is int and required_clock_nanoseconds > 0,
            "required clock must be a positive integer number of nanoseconds",
        )
        required_seconds = required_clock_nanoseconds * 1.0e-9
        _require(
            abs(float(args.timestep_seconds) - required_seconds) <= CLOCK_TOLERANCE_SECONDS,
            "timestep does not equal the required nanosecond clock",
        )
    native, lowering = compile_source(directory=args.source_directory, config=read_json(args.config))
    trace_stride = args.trace_stride if args.trace_csv is not None else 0
    receipt = simulate(native, steps=args.steps, timestep_s=args.timestep_seconds,
                       reject_step=args.reject_step, trace_stride=trace_stride)
    trace_summary = None
    if args.trace_csv is not None:
        cycle_result = receipt["cardiac_cycle_result"]
        rows = cycle_result.pop("trace_rows")
        trace_text = ",".join(_TRACE_COLUMNS) + "\n"
        trace_text += "".join(",".join(row) + "\n" for row in rows)
        trace_payload = trace_text.encode("ascii")
        trace_sha256 = hashlib.sha256(trace_payload).hexdigest()
        _publish_immutable(args.trace_csv, trace_payload, "cardiac trace")
        cycle_result["trace"] = {
            "schema": "HumanPack.shi-hose-cardiac-cycle-trace.v1",
            "sha256": trace_sha256,
            "rows": len(rows),
            "columns": list(_TRACE_COLUMNS),
        }
        cycle_result["stepper_source_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest()
        trace_summary = {"path": str(args.trace_csv.resolve()),
                         "sha256": trace_sha256, "rows": len(rows)}
    receipt["source_lowering_manifest_sha256"] = _digest(lowering)
    payload = canonical(receipt) + b"\n"
    output = args.output.resolve()
    _publish_immutable(output, payload, "output")
    summary = {"schema": SCHEMA, "output": str(output),
               "sha256": hashlib.sha256(payload).hexdigest(),
               "accepted_steps": receipt["accepted_steps"],
               "rejected_steps": receipt["rejected_steps"],
               "volume_conserved": receipt["conservation"]["volume_conserved"],
               "open_valve_evaluation_count": receipt["activation"]["open_valve_evaluation_count"],
               "qualification": receipt["qualification"]}
    if trace_summary is not None:
        heartbeat = receipt["cardiac_cycle_result"]
        summary["cardiac_cycles"] = heartbeat["completed_source_cycles"]
        summary["cycle_metrics"] = heartbeat["cycle_metrics"]
        summary["trace_csv"] = trace_summary
        summary["wall_seconds"] = heartbeat["wall_seconds"]
        summary["accepted_steps_per_second"] = heartbeat["accepted_steps_per_second"]
    print(json.dumps(summary, sort_keys=True))
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
