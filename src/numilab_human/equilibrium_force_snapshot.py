from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .force_ledger import INPUT_SCHEMA, NV
from .model import ImportError, write_json

PREFIX = "compiled_equilibrium_reactions="
PERSISTENT_PREFIX = "persistent_dynamic_force_audit="
INITIAL_REFERENCE_PREFIX = "persistent_initial_force_reference="
INITIAL_REFERENCE_SCHEMA = "numi.human.initial-force-reference-snapshot.v1"
NATIVE_KEYS = (
    "gravity_target", "muscle_force", "equality_force", "limit_force",
    "support_force", "passive_force", "force_residual", "acceleration",
)
RANK_PATTERN = re.compile(
    r'residual_rank_(?P<rank>\d+)_dof=(?P<dof>\d+).*?'
    r'residual_rank_(?P=rank)_name="(?P<name>[^"]+)"'
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError(message)


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
        if path.suffix == ".gz":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as error:
        raise ImportError(f"cannot read native equilibrium log {path}: {error}") from error


def _vector(value: Any, name: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == NV,
             f"native {name} must contain {NV} values")
    result = []
    for index, item in enumerate(value):
        _require(type(item) in (int, float) and math.isfinite(item),
                 f"native {name}[{index}] is not finite")
        result.append(float(item))
    return result


def _json_record(text: str, prefix: str) -> dict[str, Any]:
    rows = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
    _require(len(rows) == 1, f"native log must contain exactly one {prefix[:-1]} record")
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"{prefix[:-1]} contains duplicate JSON key {key}")
            result[key] = value
        return result
    try:
        payload = json.loads(rows[0], object_pairs_hook=unique_object)
    except json.JSONDecodeError as error:
        raise ImportError(f"{prefix[:-1]} is not valid JSON: {error}") from error
    _require(isinstance(payload, dict), f"{prefix[:-1]} must be an object")
    return payload


def _record(text: str) -> dict[str, list[float]]:
    payload = _json_record(text, PREFIX)
    return {key: _vector(payload.get(key), key) for key in NATIVE_KEYS}


def _persistent_record(text: str, *, initial_reference: bool = False) -> dict[str, Any]:
    """Reindex native rows without turning reference reactions into runtime loads.

    The new initial-reference record counts passive_force_n once. Its
    compiled_passive_force_n is a diagnostic of the offline solution, not
    an additional applied force. Legacy v1 audit files retain their historical
    two-term convention; their numerical reconstruction is still checked.
    """
    prefix = INITIAL_REFERENCE_PREFIX if initial_reference else PERSISTENT_PREFIX
    schema = ("numi.human.persistent-initial-force-reference.v1" if initial_reference
              else "numi.human.persistent-dynamic-force-audit.v1")
    payload = _json_record(text, prefix)
    _require(payload.get("schema") == schema, f"{prefix[:-1]} schema mismatch")
    source_rows = payload.get("rows")
    _require(isinstance(source_rows, list) and len(source_rows) == NV,
             f"{prefix[:-1]} must contain {NV} rows")
    fields = (
        "metal_muscle_force_n", "support_force_n", "equality_force_n",
        "limit_force_n", "passive_force_n", "compiled_passive_force_n",
        "gravity_target_n", "residual_n",
    )
    by_dof = {}
    names = {}
    for row in source_rows:
        _require(isinstance(row, dict), f"{prefix[:-1]} rows must be objects")
        dof = row.get("dof")
        _require(type(dof) is int and 0 <= dof < NV and dof not in by_dof,
                 f"{prefix[:-1]} must contain each DoF exactly once")
        values = {}
        for field in fields:
            value = row.get(field)
            _require(type(value) in (int, float) and math.isfinite(value),
                     f"{prefix[:-1]} {field}[{dof}] is not finite")
            values[field] = float(value)
        by_dof[dof] = values
        name = row.get("name")
        if name is not None:
            _require(isinstance(name, str), f"native coordinate name[{dof}] is not text")
            if name.strip() and name.strip() not in {"unnamed", "unknown", "none"}:
                names[dof] = name.strip()
    result = {
        "gravity_target": [by_dof[i]["gravity_target_n"] for i in range(NV)],
        "muscle_force": [by_dof[i]["metal_muscle_force_n"] for i in range(NV)],
        "equality_force": [by_dof[i]["equality_force_n"] for i in range(NV)],
        "limit_force": [by_dof[i]["limit_force_n"] for i in range(NV)],
        "support_force": [by_dof[i]["support_force_n"] for i in range(NV)],
        "passive_force": [by_dof[i]["passive_force_n"] +
                          (0.0 if initial_reference else by_dof[i]["compiled_passive_force_n"])
                          for i in range(NV)],
        "force_residual": [by_dof[i]["residual_n"] for i in range(NV)],
        "acceleration": None,
        "coordinate_names": names,
    }
    if initial_reference:
        result["compiled_passive_reference"] = [by_dof[i]["compiled_passive_force_n"] for i in range(NV)]
        peak = payload.get("maximum_abs_residual_n")
        _require(type(peak) in (int, float) and math.isfinite(peak) and peak >= 0.0,
                 "initial force reference peak residual is not finite and nonnegative")
        actual = max(abs(value) for value in result["force_residual"])
        _require(math.isclose(peak, actual, rel_tol=1.0e-10, abs_tol=1.0e-12),
                 "initial force reference peak does not match its rows")
    return result


def _decode_record(text: str) -> tuple[dict[str, Any], str]:
    kinds = [prefix for prefix in (PREFIX, PERSISTENT_PREFIX, INITIAL_REFERENCE_PREFIX)
             if any(line.startswith(prefix) for line in text.splitlines())]
    _require(len(kinds) == 1, "native log must contain exactly one supported force record kind")
    prefix = kinds[0]
    if prefix == PREFIX:
        return _record(text), prefix[:-1]
    return _persistent_record(text, initial_reference=prefix == INITIAL_REFERENCE_PREFIX), prefix[:-1]


def _rank_names(text: str) -> dict[int, str]:
    result = {}
    for line in text.splitlines():
        if "residual_rank_" not in line:
            continue
        for match in RANK_PATTERN.finditer(line):
            dof = int(match.group("dof"))
            name = match.group("name").strip()
            if 0 <= dof < NV and name:
                _require(result.get(dof) in {None, name}, f"native residual name changed for dof {dof}")
                result[dof] = name
    return result


def _coordinate_map(path: Path | None, text: str) -> tuple[list[str], list[str], dict[str, Any]]:
    if path is None:
        recovered = _rank_names(text)
        names = [f"v_{index:03d}" for index in range(NV)]
        for index, name in recovered.items():
            names[index] = f"v_{index:03d}:{name}"
        return names, ["unknown"] * NV, {
            "source": "native_residual_rank_plus_generic_velocity_order",
            "anatomical_names": bool(recovered), "named_coordinates": len(recovered),
            "coordinate_kinds_known": False,
        }
    resolved = path.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImportError(f"cannot read coordinate map {resolved}: {error}") from error
    _require(isinstance(payload, dict), "coordinate map must be an object")
    names, kinds = payload.get("coordinate_names"), payload.get("coordinate_kinds")
    anatomical_names = payload.get("anatomical_names", True)
    _require(type(anatomical_names) is bool, "coordinate map anatomical_names must be boolean when supplied")
    _require(isinstance(names, list) and len(names) == NV and
             all(isinstance(name, str) and name for name in names),
             "coordinate map must provide 128 non-empty coordinate_names")
    _require(len(set(names)) == NV, "coordinate map names must be unique")
    _require(isinstance(kinds, list) and len(kinds) == NV and
             all(kind in {"translation", "rotation", "unknown"} for kind in kinds),
             "coordinate map must classify all coordinates or explicitly mark them unknown")
    return names, kinds, {
        "source": str(resolved), "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "anatomical_names": anatomical_names, "named_coordinates": NV,
        "coordinate_kinds_known": all(kind != "unknown" for kind in kinds),
    }


def _max_error(lhs: list[float], rhs: list[float]) -> float:
    return max(abs(left - right) for left, right in zip(lhs, rhs))


def _gravity_sign(record: dict[str, Any], tolerance: float) -> tuple[float, list[float], dict[str, float]]:
    non_gravity = [sum(record[key][i] for key in
                      ("muscle_force", "equality_force", "limit_force", "support_force", "passive_force"))
                   for i in range(NV)]
    plus = [non_gravity[i] + record["gravity_target"][i] for i in range(NV)]
    minus = [non_gravity[i] - record["gravity_target"][i] for i in range(NV)]
    plus_error, minus_error = _max_error(plus, record["force_residual"]), _max_error(minus, record["force_residual"])
    plus_valid, minus_valid = plus_error <= tolerance, minus_error <= tolerance
    _require(plus_valid != minus_valid,
             "native gravity sign is not uniquely recoverable from the authoritative residual "
             f"(plus_error={plus_error:.9g}, minus_error={minus_error:.9g}, tolerance={tolerance:.9g})")
    sign = 1.0 if plus_valid else -1.0
    return sign, [sign * value for value in record["gravity_target"]], {
        "plus_error": plus_error, "minus_error": minus_error,
    }


def convert(arguments: argparse.Namespace) -> int:
    source = arguments.log.resolve()
    _require(math.isfinite(arguments.maximum_reconstruction_error) and arguments.maximum_reconstruction_error >= 0.0,
             "maximum reconstruction error must be finite and non-negative")
    text = _read_text(source)
    record, record_kind = _decode_record(text)
    initial = record_kind == INITIAL_REFERENCE_PREFIX[:-1]
    gravity_sign, gravity_bias, sign_errors = _gravity_sign(record, arguments.maximum_reconstruction_error)
    # This new schema has a fixed, source-owned convention. Do not infer a
    # different sign to compensate for an incorrect passive decomposition.
    if initial:
        _require(gravity_sign == -1.0, "initial force reference gravity convention mismatch")
    names, kinds, coordinate_metadata = _coordinate_map(arguments.coordinate_map, text)
    if arguments.coordinate_map is None and record.get("coordinate_names"):
        recovered = _rank_names(text)
        for index, name in record["coordinate_names"].items():
            _require(recovered.get(index) in {None, name}, f"native residual name changed for dof {index}")
            recovered[index] = name
            names[index] = f"v_{index:03d}:{name}"
        coordinate_metadata.update(source="explicit_native_dof_names_plus_generic_velocity_order",
                                   anatomical_names=True, named_coordinates=len(recovered))
    definitions = [
        ("gravity_bias", "gravityTarget", gravity_bias),
        ("muscle_tendon", "generalizedMuscleForce", record["muscle_force"]),
        ("joint_equality", "generalizedJointEqualityForce", record["equality_force"]),
        ("joint_limit", "generalizedPositionLimitForce", record["limit_force"]),
        ("support_contact", "generalizedSupportForce", record["support_force"]),
        ("passive_tissue", "generalizedPassiveCoordinateForce", record["passive_force"]),
    ]
    components = []
    for name, owner, values in definitions:
        if initial:
            owner = ("InitialMetalEvaluation." if name == "muscle_tendon" else
                     "InitialPassiveJointProgram." if name == "passive_tissue" else
                     "CompiledEquilibriumReference.") + owner
        else:
            owner = ("CompiledEquilibrium." if record_kind == PREFIX[:-1] else "LegacyNativeAudit.") + owner
        components.append({"name": name, "owner": owner, "values": values})
    reconstructed = [sum(component["values"][i] for component in components) for i in range(NV)]
    reconstruction_error = _max_error(reconstructed, record["force_residual"])
    _require(reconstruction_error <= arguments.maximum_reconstruction_error,
             f"canonical force snapshot does not reconstruct native residual: {reconstruction_error}")
    schema = INITIAL_REFERENCE_SCHEMA if initial else INPUT_SCHEMA
    boundary = (
        "Initial Metal muscle/passive evaluation combined with static support, equality and limit references. "
        "Not measured runtime constraint reactions, generalized equilibrium, force convergence or standing. "
        "This reference-only schema is intentionally not admitted by the generalized-force ledger."
        if initial else
        "Persistent dynamic force rows are a legacy six-owner audit. Their label alone does not establish "
        "current runtime reaction ownership, force convergence, standing or biological calibration."
        if record_kind == PERSISTENT_PREFIX[:-1] else
        "Compiled equilibrium reaction record converted without physical promotion."
    )
    metadata = {
        "native_log": {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "record": record_kind},
        "coordinate_map": coordinate_metadata,
        "gravity_convention": {"selected_sign": gravity_sign,
                               "meaning": "canonical gravity_bias = selected_sign * native gravity_target",
                               "candidate_reconstruction_errors": sign_errors},
        "maximum_reconstruction_error": reconstruction_error,
        "boundary": boundary,
        "source_owner": "Numi2/numi-lab",
        "source_revision": None,
    }
    if initial:
        metadata.update(reference_only=True, runtime_constraint_forces_measured=False,
                        compiled_passive_reference={"additive": False, "values": record["compiled_passive_reference"]})
    output = {"schema": schema, "nv": NV, "coordinate_names": names, "coordinate_kinds": kinds,
              "components": components, "reported_net": record["force_residual"],
              "generalized_acceleration": record["acceleration"], "metadata": metadata}
    write_json(arguments.output.resolve(), output)
    print(json.dumps({"schema": schema, "gravity_sign": gravity_sign, "record": record_kind,
                      "maximum_reconstruction_error": reconstruction_error,
                      "named_coordinates": coordinate_metadata["named_coordinates"],
                      "output": str(arguments.output.resolve())}, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log", type=Path, required=True,
                        help="native Human log with one compiled, legacy audit, or initial-reference record")
    parser.add_argument("--coordinate-map", type=Path,
                        help="optional JSON with exact 128 coordinate_names and coordinate_kinds")
    parser.add_argument("--maximum-reconstruction-error", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=convert)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert native Human force records without promoting initial references")
    add_arguments(parser)
    return convert(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
