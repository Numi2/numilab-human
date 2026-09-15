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
NATIVE_KEYS = (
    "gravity_target",
    "muscle_force",
    "equality_force",
    "limit_force",
    "support_force",
    "passive_force",
    "force_residual",
    "acceleration",
)
RANK_PATTERN = re.compile(
    r'residual_rank_(?P<rank>\d+)_dof=(?P<dof>\d+).*?'
    r'residual_rank_(?P=rank)_name="(?P<name>[^"]+)"'
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError(message)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        try:
            raw = gzip.decompress(raw)
        except OSError as error:
            raise ImportError(f"cannot decompress native equilibrium log {path}: {error}") from error
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ImportError(f"native equilibrium log is not UTF-8: {path}") from error


def _vector(value: Any, name: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == NV, f"native {name} must contain {NV} values")
    result: list[float] = []
    for index, item in enumerate(value):
        _require(type(item) in (int, float) and math.isfinite(item), f"native {name}[{index}] is not finite")
        result.append(float(item))
    return result


def _record(text: str) -> dict[str, list[float]]:
    rows = [line[len(PREFIX):] for line in text.splitlines() if line.startswith(PREFIX)]
    _require(len(rows) == 1, "native log must contain exactly one compiled_equilibrium_reactions record")
    try:
        payload = json.loads(rows[0])
    except json.JSONDecodeError as error:
        raise ImportError(f"compiled_equilibrium_reactions is not valid JSON: {error}") from error
    _require(isinstance(payload, dict), "compiled_equilibrium_reactions must be an object")
    return {key: _vector(payload.get(key), key) for key in NATIVE_KEYS}


def _persistent_record(text: str) -> dict[str, Any]:
    """Decode the native persistent dynamic per-DoF audit.

    The persistent runner publishes rows sorted by residual magnitude rather
    than by DoF.  Re-indexing by the explicit DoF is therefore mandatory; a
    positional interpretation would silently attach forces to the wrong
    coordinates.  The two passive fields are deliberately combined because
    the canonical snapshot has one passive-tissue owner and the native audit
    already reports their signed sum in ``residual_n``.
    """
    rows = [line[len(PERSISTENT_PREFIX):]
            for line in text.splitlines() if line.startswith(PERSISTENT_PREFIX)]
    _require(len(rows) == 1,
             "native log must contain exactly one persistent_dynamic_force_audit record")
    try:
        payload = json.loads(rows[0])
    except json.JSONDecodeError as error:
        raise ImportError(
            f"persistent_dynamic_force_audit is not valid JSON: {error}"
        ) from error
    _require(isinstance(payload, dict),
             "persistent_dynamic_force_audit must be an object")
    _require(payload.get("schema") == "numi.human.persistent-dynamic-force-audit.v1",
             "persistent_dynamic_force_audit schema mismatch")
    source_rows = payload.get("rows")
    _require(isinstance(source_rows, list) and len(source_rows) == NV,
             f"persistent_dynamic_force_audit must contain {NV} rows")
    fields = (
        "metal_muscle_force_n", "support_force_n", "equality_force_n",
        "limit_force_n", "passive_force_n", "compiled_passive_force_n",
        "gravity_target_n", "residual_n",
    )
    by_dof: dict[int, dict[str, float]] = {}
    for row in source_rows:
        _require(isinstance(row, dict),
                 "persistent_dynamic_force_audit rows must be objects")
        dof = row.get("dof")
        _require(type(dof) is int and 0 <= dof < NV and dof not in by_dof,
                 "persistent_dynamic_force_audit must contain each DoF exactly once")
        values: dict[str, float] = {}
        for field in fields:
            value = row.get(field)
            _require(type(value) in (int, float) and math.isfinite(value),
                     f"persistent_dynamic_force_audit {field}[{dof}] is not finite")
            values[field] = float(value)
        by_dof[dof] = values
    _require(set(by_dof) == set(range(NV)),
             "persistent_dynamic_force_audit DoF coverage is incomplete")
    return {
        "gravity_target": [by_dof[index]["gravity_target_n"] for index in range(NV)],
        "muscle_force": [by_dof[index]["metal_muscle_force_n"] for index in range(NV)],
        "equality_force": [by_dof[index]["equality_force_n"] for index in range(NV)],
        "limit_force": [by_dof[index]["limit_force_n"] for index in range(NV)],
        "support_force": [by_dof[index]["support_force_n"] for index in range(NV)],
        "passive_force": [
            by_dof[index]["passive_force_n"]
            + by_dof[index]["compiled_passive_force_n"]
            for index in range(NV)
        ],
        "force_residual": [by_dof[index]["residual_n"] for index in range(NV)],
        "acceleration": None,
    }


def _decode_record(text: str) -> tuple[dict[str, Any], str]:
    """Select one authoritative native record and reject mixed/empty logs."""
    compiled = [line for line in text.splitlines() if line.startswith(PREFIX)]
    persistent = [line for line in text.splitlines()
                  if line.startswith(PERSISTENT_PREFIX)]
    _require(bool(compiled) ^ bool(persistent),
             "native log must contain exactly one supported force record kind")
    if compiled:
        return _record(text), PREFIX[:-1]
    return _persistent_record(text), PERSISTENT_PREFIX[:-1]


def _rank_names(text: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for line in text.splitlines():
        if "residual_rank_" not in line:
            continue
        for match in RANK_PATTERN.finditer(line):
            dof = int(match.group("dof"))
            name = match.group("name").strip()
            if 0 <= dof < NV and name:
                previous = result.get(dof)
                _require(previous in {None, name}, f"native residual name changed for dof {dof}")
                result[dof] = name
    return result


def _coordinate_map(path: Path | None, text: str) -> tuple[list[str], list[str], dict[str, Any]]:
    if path is None:
        recovered = _rank_names(text)
        names = [f"v_{index:03d}" for index in range(NV)]
        for index, name in recovered.items():
            names[index] = f"v_{index:03d}:{name}"
        return (
            names,
            ["unknown"] * NV,
            {
                "source": "native_residual_rank_plus_generic_velocity_order",
                "anatomical_names": bool(recovered),
                "named_coordinates": len(recovered),
                "coordinate_kinds_known": False,
            },
        )
    resolved = path.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImportError(f"cannot read coordinate map {resolved}: {error}") from error
    _require(isinstance(payload, dict), "coordinate map must be an object")
    names = payload.get("coordinate_names")
    kinds = payload.get("coordinate_kinds")
    anatomical_names = payload.get("anatomical_names", True)
    _require(type(anatomical_names) is bool,
             "coordinate map anatomical_names must be boolean when supplied")
    _require(isinstance(names, list) and len(names) == NV and
             all(isinstance(name, str) and name for name in names),
             "coordinate map must provide 128 non-empty coordinate_names")
    _require(len(set(names)) == NV, "coordinate map names must be unique")
    _require(isinstance(kinds, list) and len(kinds) == NV and
             all(kind in {"translation", "rotation", "unknown"} for kind in kinds),
             "coordinate map must classify all coordinates or explicitly mark them unknown")
    return names, kinds, {
        "source": str(resolved),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "anatomical_names": anatomical_names,
        "named_coordinates": NV,
        "coordinate_kinds_known": all(kind != "unknown" for kind in kinds),
    }


def _max_error(lhs: list[float], rhs: list[float]) -> float:
    return max(abs(left - right) for left, right in zip(lhs, rhs))


def _gravity_sign(record: dict[str, list[float]], tolerance: float) -> tuple[float, list[float], dict[str, float]]:
    non_gravity = [
        record["muscle_force"][index]
        + record["equality_force"][index]
        + record["limit_force"][index]
        + record["support_force"][index]
        + record["passive_force"][index]
        for index in range(NV)
    ]
    plus = [non_gravity[index] + record["gravity_target"][index] for index in range(NV)]
    minus = [non_gravity[index] - record["gravity_target"][index] for index in range(NV)]
    plus_error = _max_error(plus, record["force_residual"])
    minus_error = _max_error(minus, record["force_residual"])
    plus_valid = plus_error <= tolerance
    minus_valid = minus_error <= tolerance
    _require(plus_valid != minus_valid,
             "native gravity sign is not uniquely recoverable from the authoritative residual "
             f"(plus_error={plus_error:.9g}, minus_error={minus_error:.9g}, tolerance={tolerance:.9g})")
    if plus_valid:
        return 1.0, record["gravity_target"], {"plus_error": plus_error, "minus_error": minus_error}
    return -1.0, [-value for value in record["gravity_target"]], {
        "plus_error": plus_error,
        "minus_error": minus_error,
    }


def convert(arguments: argparse.Namespace) -> int:
    source = arguments.log.resolve()
    _require(math.isfinite(arguments.maximum_reconstruction_error) and arguments.maximum_reconstruction_error >= 0.0,
             "maximum reconstruction error must be finite and non-negative")
    text = _read_text(source)
    record, record_kind = _decode_record(text)
    gravity_sign, gravity_bias, sign_errors = _gravity_sign(record, arguments.maximum_reconstruction_error)
    names, kinds, coordinate_metadata = _coordinate_map(arguments.coordinate_map, text)

    components = [
        {"name": "gravity_bias", "owner": "NumiHumanMuscleEquilibrium.gravityTarget", "values": gravity_bias},
        {"name": "muscle_tendon", "owner": "NumiHumanMuscleEquilibrium.generalizedMuscleForce", "values": record["muscle_force"]},
        {"name": "joint_equality", "owner": "NHEQ2/generalizedJointEqualityForce", "values": record["equality_force"]},
        {"name": "joint_limit", "owner": "NHLIM1/generalizedPositionLimitForce", "values": record["limit_force"]},
        {"name": "support_contact", "owner": "NumiHumanStaticSupportContact/generalizedSupportForce", "values": record["support_force"]},
        {"name": "passive_tissue", "owner": "NumiHumanPassiveCoordinateCoupling/generalizedPassiveCoordinateForce", "values": record["passive_force"]},
    ]
    reconstructed = [sum(component["values"][index] for component in components) for index in range(NV)]
    reconstruction_error = _max_error(reconstructed, record["force_residual"])
    _require(reconstruction_error <= arguments.maximum_reconstruction_error,
             f"canonical force snapshot does not reconstruct native residual: {reconstruction_error}")

    output = {
        "schema": INPUT_SCHEMA,
        "nv": NV,
        "coordinate_names": names,
        "coordinate_kinds": kinds,
        "components": components,
        "reported_net": record["force_residual"],
        "generalized_acceleration": record["acceleration"],
        "metadata": {
            "native_log": {
                "path": str(source),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "record": record_kind,
            },
            "coordinate_map": coordinate_metadata,
            "gravity_convention": {
                "selected_sign": gravity_sign,
                "meaning": "canonical gravity_bias = selected_sign * native gravity_target",
                "candidate_reconstruction_errors": sign_errors,
            },
            "maximum_reconstruction_error": reconstruction_error,
            "boundary": (
                "Persistent dynamic force rows are a six-owner generalized-force "
                "audit for the captured horizon. They do not establish a solved "
                "initial equilibrium, long-horizon convergence, standing, recovery, "
                "walking, anatomy, activation calibration, materials, or subject "
                "calibration."
                if record_kind == PERSISTENT_PREFIX[:-1]
                else "Compiled equilibrium reaction record converted without physical promotion."
            ),
            "source_owner": "Numi2/numi-lab:coupled",
        },
    }
    write_json(arguments.output.resolve(), output)
    print(json.dumps({
        "schema": INPUT_SCHEMA,
        "gravity_sign": gravity_sign,
        "record": record_kind,
        "maximum_reconstruction_error": reconstruction_error,
        "named_coordinates": coordinate_metadata["named_coordinates"],
        "output": str(arguments.output.resolve()),
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--log", type=Path, required=True,
                        help="native Human stdout/log containing compiled_equilibrium_reactions")
    parser.add_argument("--coordinate-map", type=Path,
                        help="optional JSON with exact 128 coordinate_names and coordinate_kinds")
    parser.add_argument("--maximum-reconstruction-error", type=float, default=1.0e-4)
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=convert)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert the native 128-DoF Human equilibrium force record into a canonical force snapshot"
    )
    add_arguments(parser)
    return convert(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
