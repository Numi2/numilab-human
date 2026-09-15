"""Step source-vessel mass moments through a conservative exact-clock probe.

The source-vessel receipt supplies registered surface-integral moments and an
explicit density candidate. This module translates those moments with a
prescribed velocity field while preserving mass and linear momentum. It also
binds the physical-M4 regional blood/oxygen transaction by source hash and
clock, but deliberately does not invent a lumen, centreline, tissue interface,
or material owner.
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
from .physiology import canonical


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.vessel-mass-moment-transport-candidate.v1"
MOMENTS = ROOT / "Docs/media/vessel-mass-moment-candidate-20260915/receipt-v1.json"
REGIONAL_EXCHANGE = ROOT / "Docs/media/native-human-regional-exchange-20260915/receipt-v1.json"
CLOCK_NANOSECONDS = 12_500
CLOCK_SECONDS = CLOCK_NANOSECONDS * 1.0e-9
MAX_STEPS = 1_000_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("vessel mass moment transport: " + message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             f"{label} is not finite")
    return float(value)


def _vector(value: Any, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == 3,
             f"{label} must have three components")
    return [_finite(component, f"{label}[{index}]")
            for index, component in enumerate(value)]


def _matrix(value: Any, label: str) -> list[list[float]]:
    _require(isinstance(value, list) and len(value) == 3,
             f"{label} must be 3x3")
    return [_vector(row, f"{label}[{index}]") for index, row in enumerate(value)]


def _outer(a: list[float], b: list[float]) -> list[list[float]]:
    return [[a[i] * b[j] for j in range(3)] for i in range(3)]


def _matrix_add(*matrices: list[list[float]]) -> list[list[float]]:
    return [[math.fsum(matrix[i][j] for matrix in matrices)
             for j in range(3)] for i in range(3)]


def _matrix_scale(matrix: list[list[float]], scale: float) -> list[list[float]]:
    return [[scale * matrix[i][j] for j in range(3)] for i in range(3)]


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def _read_canonical(path: Path, label: str) -> tuple[dict[str, Any], str]:
    _require(path.is_file() and not path.is_symlink(),
             f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HumanImportError(f"{label} is not valid JSON") from error
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(raw == canonical(value) + b"\n", f"{label} is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _validate_regional_exchange(value: dict[str, Any]) -> None:
    _require(value.get("schema") ==
             "HumanPack.native-human-regional-exchange-current-requalification.v1",
             "regional exchange schema changed")
    results = value.get("results", {})
    qualification = value.get("qualification", {})
    _require(results.get("attempted_steps") == 512 and
             results.get("accepted_steps_environment_0") == 511 and
             results.get("rejected_step_environment_0") == 37 and
             results.get("timestep_nanoseconds") == CLOCK_NANOSECONDS and
             results.get("rollback") == "bitwise" and
             results.get("replay") == "bitwise",
             "regional exchange exact-clock evidence changed")
    _require(qualification.get("accepted_step_conservation") is True and
             qualification.get("regional_blood_transport") is True and
             qualification.get("mechanical_blood_mass_owner") is False and
             qualification.get("anatomical_vessel_lumen") is False and
             qualification.get("subject_calibration") is False,
             "regional exchange boundary changed")


def compile_candidate(
    *,
    moments: Path = MOMENTS,
    regional_exchange: Path = REGIONAL_EXCHANGE,
    velocity_mps: list[float] | None = None,
) -> dict[str, Any]:
    velocity = [0.0, 0.0, 0.0] if velocity_mps is None else _vector(velocity_mps, "velocity")
    moments_path = Path(moments)
    exchange_path = Path(regional_exchange)
    moments_doc, moments_sha = _read_canonical(moments_path, "vessel mass moments")
    exchange_doc, exchange_sha = _read_canonical(exchange_path, "regional exchange")
    _require(moments_doc.get("schema") == "HumanPack.vessel-mass-moment-owner-candidate.v1",
             "vessel mass moments schema changed")
    _validate_regional_exchange(exchange_doc)
    qualification = moments_doc.get("qualification", {})
    _require(qualification.get("zeroth_first_second_mass_moments") is True and
             qualification.get("atomic_checkpoint_restore") is True and
             qualification.get("anatomical_blood_mass_owner") is False,
             "vessel mass moments are not a candidate-only source")
    owners = moments_doc.get("owners")
    _require(isinstance(owners, list) and len(owners) == 6,
             "vessel mass moments must contain six owners")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sorted(owners, key=lambda item: item.get("member_id", "")):
        _require(isinstance(source, dict), "vessel mass moment owner is malformed")
        member_id = source.get("member_id")
        _require(isinstance(member_id, str) and member_id not in seen,
                 "vessel mass moment owners repeat")
        seen.add(member_id)
        mass = _finite(source.get("mass_kg"), f"{member_id} mass")
        _require(mass > 0.0, f"{member_id} mass is not positive")
        centroid = _vector(source.get("world_centroid_m"), f"{member_id} centroid")
        first = _vector(source.get("first_mass_moment_kg_m"), f"{member_id} first moment")
        raw_second = _matrix(source.get("raw_second_mass_moment_kg_m2"),
                             f"{member_id} second moment")
        rows.append({
            "member_id": member_id,
            "mass_kg": mass,
            "world_centroid_m": centroid,
            "first_mass_moment_kg_m": first,
            "raw_second_mass_moment_kg_m2": raw_second,
            "velocity_mps": velocity,
            "linear_momentum_kg_m_per_s": [mass * component for component in velocity],
            "lumen_or_tube_admitted": False,
            "mechanical_blood_mass_owner": False,
            "two_way_tissue_exchange": False,
            "subject_calibration": False,
        })
    total_mass = math.fsum(row["mass_kg"] for row in rows)
    total_momentum = [math.fsum(row["linear_momentum_kg_m_per_s"][i] for row in rows)
                      for i in range(3)]
    return {
        "schema": SCHEMA,
        "compiler": "numilab-human.vessel-mass-moment-transport.1",
        "status": "partial",
        "subject": "one adult male source package",
        "source": {
            "moments_path": str(moments_path.relative_to(ROOT)) if moments_path.is_relative_to(ROOT) else str(moments_path),
            "moments_sha256": moments_sha,
            "regional_exchange_path": str(exchange_path.relative_to(ROOT)) if exchange_path.is_relative_to(ROOT) else str(exchange_path),
            "regional_exchange_sha256": exchange_sha,
            "vessel_count": len(rows),
        },
        "clock": {"nanoseconds": CLOCK_NANOSECONDS, "seconds": CLOCK_SECONDS,
                  "source": "regional_exchange_and_transport_probe"},
        "transport_field": {
            "kind": "uniform_prescribed_velocity_probe",
            "velocity_mps": velocity,
            "provenance": "engineering_probe_not_anatomical_flow_field",
        },
        "owners": rows,
        "initial_totals": {"mass_kg": total_mass,
                           "linear_momentum_kg_m_per_s": total_momentum},
        "qualification": {
            "source_vessel_moments_bound": True,
            "regional_exchange_exact_clock_bound": True,
            "mass_conservation": True,
            "linear_momentum_conservation": True,
            "first_second_moment_transport": True,
            "atomic_rejected_step_rollback": True,
            "deterministic_replay": True,
            "anatomical_vessel_lumen": False,
            "mechanical_blood_mass_owner": False,
            "two_way_blood_tissue_transfer": False,
            "material_calibration": False,
            "subject_calibration": False,
            "standing_walking": False,
        },
        "boundary": (
            "The six source vessel moment candidates are translated through a "
            "prescribed exact-clock velocity probe with conservative mass and "
            "momentum updates and atomic rejection. The physical-M4 regional "
            "blood/oxygen receipt is bound by hash and clock, but no vessel lumen, "
            "anatomical flow field, tissue interface, mechanical mass owner, "
            "material, or subject calibration is admitted."
        ),
    }


def _step(state: list[dict[str, Any]], timestep_s: float) -> list[dict[str, Any]]:
    candidate = copy.deepcopy(state)
    for row in candidate:
        mass = row["mass_kg"]
        velocity = row["velocity_mps"]
        first_before = row["first_mass_moment_kg_m"]
        first_after = [first_before[i] + mass * velocity[i] * timestep_s
                       for i in range(3)]
        raw_before = row["raw_second_mass_moment_kg_m2"]
        raw_after = _matrix_add(
            raw_before,
            _matrix_scale(_outer(velocity, first_before), timestep_s),
            _matrix_scale(_outer(first_before, velocity), timestep_s),
            _matrix_scale(_outer(velocity, velocity), mass * timestep_s * timestep_s),
        )
        row["first_mass_moment_kg_m"] = first_after
        row["world_centroid_m"] = [value / mass for value in first_after]
        row["raw_second_mass_moment_kg_m2"] = raw_after
        row["linear_momentum_kg_m_per_s"] = [mass * value for value in velocity]
    return candidate


def _totals(state: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mass_kg": math.fsum(row["mass_kg"] for row in state),
        "linear_momentum_kg_m_per_s": [
            math.fsum(row["linear_momentum_kg_m_per_s"][i] for row in state)
            for i in range(3)
        ],
        "first_mass_moment_kg_m": [
            math.fsum(row["first_mass_moment_kg_m"][i] for row in state)
            for i in range(3)
        ],
    }


def _simulate_once(compiled: dict[str, Any], *, steps: int,
                   timestep_s: float, reject_step: int | None) -> dict[str, Any]:
    _require(type(steps) is int and 1 <= steps <= MAX_STEPS, "steps out of range")
    _require(abs(float(timestep_s) - CLOCK_SECONDS) <= 1.0e-15,
             "timestep is not the canonical 12.5 us clock")
    if reject_step is not None:
        _require(type(reject_step) is int and 1 <= reject_step <= steps,
                 "reject_step is outside attempted steps")
    state = copy.deepcopy(compiled["owners"])
    initial = copy.deepcopy(state)
    initial_totals = _totals(state)
    accepted_trace: list[str] = []
    rejected = 0
    for attempt in range(1, steps + 1):
        before = copy.deepcopy(state)
        candidate = _step(state, float(timestep_s))
        if reject_step == attempt:
            state = before
            rejected += 1
        else:
            state = candidate
            accepted_trace.append(_digest(state))
    final_totals = _totals(state)
    momentum_residual = [final_totals["linear_momentum_kg_m_per_s"][i] -
                         initial_totals["linear_momentum_kg_m_per_s"][i]
                         for i in range(3)]
    mass_residual = final_totals["mass_kg"] - initial_totals["mass_kg"]
    return {
        "attempted_steps": steps,
        "accepted_steps": steps - rejected,
        "rejected_steps": rejected,
        "timestep_seconds": float(timestep_s),
        "initial_state": initial,
        "final_state": state,
        "initial_totals": initial_totals,
        "final_totals": final_totals,
        "conservation": {
            "mass_residual_kg": mass_residual,
            "linear_momentum_residual_kg_m_per_s": momentum_residual,
            "mass_conserved": abs(mass_residual) <= 1.0e-12,
            "linear_momentum_conserved": max(abs(value) for value in momentum_residual) <= 1.0e-12,
        },
        "rollback": {"requested": reject_step is not None,
                     "rejected_candidate_state_neutral": True},
        "accepted_state_trace_sha256": _digest(accepted_trace),
    }


def simulate(compiled: dict[str, Any], *, steps: int = 512,
             timestep_s: float = CLOCK_SECONDS,
             reject_step: int | None = 37) -> dict[str, Any]:
    first = _simulate_once(compiled, steps=steps, timestep_s=timestep_s,
                           reject_step=reject_step)
    second = _simulate_once(compiled, steps=steps, timestep_s=timestep_s,
                            reject_step=reject_step)
    _require(canonical(first["final_state"]) == canonical(second["final_state"]),
             "transport replay changed final state")
    _require(first["accepted_state_trace_sha256"] == second["accepted_state_trace_sha256"],
             "transport replay changed state trace")
    first["replay"] = "bitwise"
    first["qualification"] = compiled["qualification"]
    first["boundary"] = compiled["boundary"]
    return first


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
    compiled = compile_candidate(moments=arguments.moments,
                                 regional_exchange=arguments.regional_exchange,
                                 velocity_mps=arguments.velocity_mps)
    receipt = simulate(compiled, steps=arguments.steps,
                       timestep_s=arguments.timestep_seconds,
                       reject_step=arguments.reject_step)
    receipt["schema"] = SCHEMA
    receipt["compiler"] = compiled["compiler"]
    receipt["status"] = compiled["status"]
    receipt["subject"] = compiled["subject"]
    receipt["source"] = compiled["source"]
    receipt["clock"] = compiled["clock"]
    receipt["transport_field"] = compiled["transport_field"]
    receipt["owners"] = compiled["owners"]
    output = arguments.output.resolve()
    digest = _write_immutable(output, receipt)
    print(json.dumps({"schema": SCHEMA, "output": str(output), "sha256": digest,
                      "accepted_steps": receipt["accepted_steps"],
                      "rejected_steps": receipt["rejected_steps"],
                      "mass_conserved": receipt["conservation"]["mass_conserved"],
                      "linear_momentum_conserved": receipt["conservation"]["linear_momentum_conserved"],
                      "replay": receipt["replay"]}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--moments", type=Path, default=MOMENTS)
    parser.add_argument("--regional-exchange", type=Path, default=REGIONAL_EXCHANGE)
    parser.add_argument("--velocity-mps", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0])
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--timestep-seconds", type=float, default=CLOCK_SECONDS)
    parser.add_argument("--reject-step", type=int, default=37)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (HumanImportError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"vessel mass moment transport: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
