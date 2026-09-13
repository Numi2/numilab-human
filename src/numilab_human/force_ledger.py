from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError, sha256, write_json


INPUT_SCHEMA = "numi.human.generalized-force-snapshot.v1"
SCHEMA = "numi.human.generalized-force-ledger.v1"
NV = 128
REQUIRED_COMPONENTS = (
    "gravity_bias",
    "muscle_tendon",
    "joint_equality",
    "joint_limit",
    "support_contact",
    "passive_tissue",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError(message)


def _vector(value: Any, name: str, count: int) -> list[float]:
    _require(isinstance(value, list) and len(value) == count, f"{name} must contain {count} values")
    result: list[float] = []
    for index, item in enumerate(value):
        _require(type(item) in (int, float) and math.isfinite(item), f"{name}[{index}] is not finite")
        result.append(float(item))
    return result


def _load_snapshot(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImportError(f"cannot read generalized-force snapshot {path}: {error}") from error
    _require(isinstance(payload, dict) and payload.get("schema") == INPUT_SCHEMA,
             "generalized-force snapshot schema mismatch")
    _require(payload.get("nv") == NV, f"generalized-force snapshot must have nv={NV}")
    names = payload.get("coordinate_names")
    _require(isinstance(names, list) and len(names) == NV and
             all(isinstance(name, str) and name for name in names),
             "coordinate_names must contain 128 non-empty names")
    _require(len(set(names)) == NV, "coordinate_names must be unique")
    kinds = payload.get("coordinate_kinds")
    _require(isinstance(kinds, list) and len(kinds) == NV and
             all(kind in {"translation", "rotation", "unknown"} for kind in kinds),
             "coordinate_kinds must classify every coordinate or explicitly mark it unknown")
    components = payload.get("components")
    _require(isinstance(components, list) and components, "components must be a non-empty list")
    seen: set[str] = set()
    decoded = []
    for component in components:
        _require(isinstance(component, dict), "force component must be an object")
        name = component.get("name")
        owner = component.get("owner")
        _require(isinstance(name, str) and name and name not in seen, "force component name is missing or duplicated")
        _require(isinstance(owner, str) and owner, f"force component {name} has no owner")
        seen.add(name)
        decoded.append({"name": name, "owner": owner, "values": _vector(component.get("values"), name, NV)})
    missing = [name for name in REQUIRED_COMPONENTS if name not in seen]
    _require(not missing, "missing required force components: " + ", ".join(missing))
    reported = _vector(payload.get("reported_net"), "reported_net", NV)
    acceleration = payload.get("generalized_acceleration")
    decoded_acceleration = None if acceleration is None else _vector(acceleration, "generalized_acceleration", NV)
    return {
        "coordinate_names": names,
        "coordinate_kinds": kinds,
        "components": decoded,
        "reported_net": reported,
        "generalized_acceleration": decoded_acceleration,
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    }


def audit(arguments: argparse.Namespace) -> int:
    source = arguments.input.resolve()
    snapshot = _load_snapshot(source)
    _require(math.isfinite(arguments.maximum_assembly_error) and arguments.maximum_assembly_error >= 0.0,
             "maximum assembly error must be finite and non-negative")
    _require(math.isfinite(arguments.maximum_closure_ratio) and 0.0 <= arguments.maximum_closure_ratio < 1.0,
             "maximum closure ratio must be finite and in [0, 1)")
    _require(type(arguments.top) is int and 1 <= arguments.top <= NV, "top must be in [1, 128]")

    components = snapshot["components"]
    calculated = [sum(component["values"][index] for component in components) for index in range(NV)]
    assembly_error = [abs(calculated[index] - snapshot["reported_net"][index]) for index in range(NV)]
    maximum_assembly_error = max(assembly_error)

    rows = []
    for index in range(NV):
        magnitudes = [(abs(component["values"][index]), component["name"], component["owner"])
                      for component in components]
        force_scale = sum(item[0] for item in magnitudes)
        net = snapshot["reported_net"][index]
        closure_ratio = abs(net) / force_scale if force_scale > 0.0 else (0.0 if net == 0.0 else math.inf)
        dominant = max(magnitudes, default=(0.0, "none", "none"))
        row = {
            "index": index,
            "coordinate": snapshot["coordinate_names"][index],
            "kind": snapshot["coordinate_kinds"][index],
            "reported_net": net,
            "calculated_net": calculated[index],
            "assembly_error": assembly_error[index],
            "force_scale": force_scale,
            "closure_ratio": closure_ratio,
            "dominant_component": dominant[1],
            "dominant_owner": dominant[2],
            "dominant_magnitude": dominant[0],
        }
        if snapshot["generalized_acceleration"] is not None:
            row["generalized_acceleration"] = snapshot["generalized_acceleration"][index]
        rows.append(row)

    ranked = sorted(rows, key=lambda row: (row["closure_ratio"], abs(row["reported_net"])), reverse=True)
    root = rows[:6]
    internal = rows[6:]
    maximum_closure_ratio = max(row["closure_ratio"] for row in rows)
    maximum_root_closure_ratio = max(row["closure_ratio"] for row in root)
    maximum_internal_closure_ratio = max(row["closure_ratio"] for row in internal)
    rms_closure_ratio = math.sqrt(sum(row["closure_ratio"] ** 2 for row in rows) / NV)
    assembly_closed = maximum_assembly_error <= arguments.maximum_assembly_error
    force_closed = maximum_closure_ratio <= arguments.maximum_closure_ratio
    complete = assembly_closed and force_closed

    receipt = {
        "schema": SCHEMA,
        "status": "passed" if complete else "partial",
        "input": {"path": str(source), "sha256": sha256(source)},
        "coverage": {
            "nv": NV,
            "required_components": list(REQUIRED_COMPONENTS),
            "published_components": [
                {"name": component["name"], "owner": component["owner"]}
                for component in components
            ],
            "authoritative_net_present": True,
            "full_force_coverage": assembly_closed,
        },
        "residual": {
            "maximum_assembly_error": maximum_assembly_error,
            "maximum_closure_ratio": maximum_closure_ratio,
            "maximum_root_closure_ratio": maximum_root_closure_ratio,
            "maximum_internal_closure_ratio": maximum_internal_closure_ratio,
            "rms_closure_ratio": rms_closure_ratio,
        },
        "worst_coordinates": ranked[:arguments.top],
        "qualification": {
            "component_assembly_closed": assembly_closed,
            "generalized_force_closed": force_closed,
            "full_generalized_force_ledger": complete,
            "force_convergence": False,
            "sustained_standing": False,
            "walking": False,
        },
        "gate": {
            "maximum_assembly_error": arguments.maximum_assembly_error,
            "maximum_closure_ratio": arguments.maximum_closure_ratio,
            "reasons": [
                *([] if assembly_closed else ["published force components do not reconstruct the authoritative net force"]),
                *([] if force_closed else ["one or more generalized coordinates exceed the force-closure ratio"]),
            ],
        },
        "metadata": snapshot["metadata"],
    }
    write_json(arguments.output.resolve(), receipt)
    print(json.dumps({
        "status": receipt["status"],
        "maximum_assembly_error": maximum_assembly_error,
        "maximum_closure_ratio": maximum_closure_ratio,
        "worst_coordinate": ranked[0]["coordinate"],
        "worst_component": ranked[0]["dominant_component"],
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True,
                        help="numi.human.generalized-force-snapshot.v1 JSON from the native full-body owner")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-assembly-error", type=float, default=1.0e-4)
    parser.add_argument("--maximum-closure-ratio", type=float, default=0.05)
    parser.add_argument("--top", type=int, default=12)
    parser.set_defaults(handler=audit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit all 128 Human generalized-force owners")
    add_arguments(parser)
    return audit(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
