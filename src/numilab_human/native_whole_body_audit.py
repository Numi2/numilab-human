"""Audit the native full-body generalized-force certificate.

The native Metal owner emits one row for every articulated velocity coordinate.
This reader keeps the root rows separate from internal rows and refuses to
call a state equilibrated while any required force owner is unavailable.  A
closed floating-root wrench is therefore useful diagnostic evidence, never a
standing qualification by itself.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from .model import ImportError, sha256, write_json


INPUT_SCHEMA = "numi.human.whole-body-force-audit.v1"
SCHEMA = "numi.human.native-whole-body-force-audit.v1"
NV = 128
ROOT_DOF_COUNT = 6
TERMS = (
    "gravity",
    "active_muscle",
    "passive_muscle",
    "tendon",
    "ligament_limit",
    "contact",
    "joint_constraint",
    "damping",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ImportError(message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"{label} is not finite")
    return float(value)


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ImportError(f"cannot read native whole-body force audit {path}: {error}") from error
    _require(isinstance(payload, dict) and payload.get("schema") == INPUT_SCHEMA,
             "native whole-body force audit schema mismatch")
    source = payload.get("source")
    equilibrium = payload.get("equilibrium")
    terms_available = payload.get("terms_available")
    active_set = payload.get("contact_active_set")
    fibre = payload.get("fibre_tendon_equilibrium")
    _require(isinstance(source, dict), "native whole-body force audit has no source identity")
    _require(isinstance(equilibrium, dict), "native whole-body force audit has no equilibrium section")
    _require(isinstance(terms_available, dict), "native whole-body force audit has no term availability")
    _require(isinstance(active_set, dict), "native whole-body force audit has no contact active-set record")
    _require(isinstance(fibre, dict), "native whole-body force audit has no fibre/tendon record")
    _require(equilibrium.get("dof_count") == NV, f"native whole-body force audit must have nv={NV}")
    q_sha256 = equilibrium.get("q_sha256")
    _require(isinstance(q_sha256, str) and len(q_sha256) == 64,
             "native whole-body force audit q_sha256 is missing")
    rows = equilibrium.get("rows")
    _require(isinstance(rows, list) and len(rows) == NV,
             f"native whole-body force audit must contain {NV} rows")
    decoded: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"native whole-body force row {index} is not an object")
        _require(row.get("dof_index") == index,
                 f"native whole-body force rows are not ordered at dof {index}")
        values = {term: _finite(row.get(term), f"dof {index} {term}") for term in TERMS}
        net = _finite(row.get("net_residual"), f"dof {index} net_residual")
        scale = math.fsum(abs(value) for value in values.values())
        decoded.append({
            "index": index,
            "name": row.get("dof_name", f"dof_{index}"),
            "terms": values,
            "net_residual": net,
            "force_scale": scale,
            "normalized_residual": abs(net) / scale if scale > 0.0 else (0.0 if net == 0.0 else math.inf),
        })
    availability = {}
    for term in TERMS:
        _require(type(terms_available.get(term)) is bool,
                 f"native whole-body force term {term} has no boolean availability")
        availability[term] = terms_available[term]
    _require(type(active_set.get("selected")) is bool,
             "native whole-body force contact active-set selection is not boolean")
    _require(type(fibre.get("solved")) is bool,
             "native whole-body force fibre/tendon state is not boolean")
    return {
        "source": source,
        "q_sha256": q_sha256,
        "rows": decoded,
        "terms_available": availability,
        "contact_active_set_selected": active_set["selected"],
        "fibre_tendon_equilibrium": fibre["solved"],
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    root = rows[:ROOT_DOF_COUNT]
    internal = rows[ROOT_DOF_COUNT:]

    def maximum(items: list[dict[str, Any]], field: str) -> float:
        return max(float(item[field]) for item in items)

    def rms(items: list[dict[str, Any]], field: str) -> float:
        return math.sqrt(math.fsum(float(item[field]) ** 2 for item in items) / len(items))

    return {
        "root_max_absolute_residual": maximum(root, "net_residual"),
        "root_rms_absolute_residual": rms(root, "net_residual"),
        "root_max_normalized_residual": maximum(root, "normalized_residual"),
        "internal_max_absolute_residual": maximum(internal, "net_residual"),
        "internal_rms_absolute_residual": rms(internal, "net_residual"),
        "internal_max_normalized_residual": maximum(internal, "normalized_residual"),
        "all_max_absolute_residual": maximum(rows, "net_residual"),
        "all_rms_absolute_residual": rms(rows, "net_residual"),
        "all_max_normalized_residual": maximum(rows, "normalized_residual"),
    }


def audit(arguments: argparse.Namespace) -> int:
    source = arguments.input.resolve()
    decoded = _load(source)
    _require(math.isfinite(arguments.maximum_normalized_residual) and
             arguments.maximum_normalized_residual >= 0.0,
             "maximum normalized residual must be finite and non-negative")
    _require(type(arguments.top) is int and 1 <= arguments.top <= NV,
             "top must be in [1, 128]")

    rows = decoded["rows"]
    residual = _summary(rows)
    terms_complete = all(decoded["terms_available"].values())
    residual_closed = (
        residual["all_max_normalized_residual"] <= arguments.maximum_normalized_residual
    )
    equilibrium_closed = (
        terms_complete and
        decoded["fibre_tendon_equilibrium"] and
        residual_closed
    )
    handoff_admissible = (
        equilibrium_closed and decoded["contact_active_set_selected"]
    )
    reasons: list[str] = []
    if not terms_complete:
        reasons.append(
            "required generalized-force terms are unavailable: " +
            ", ".join(term for term in TERMS if not decoded["terms_available"][term])
        )
    if not decoded["fibre_tendon_equilibrium"]:
        reasons.append("fibre/tendon equilibrium state is not solved")
    if not decoded["contact_active_set_selected"]:
        reasons.append("the contact active set was not selected by the native solve")
    if not residual_closed:
        reasons.append(
            "one or more generalized coordinates exceed the normalized residual bound"
        )

    worst = sorted(
        rows,
        key=lambda row: (row["normalized_residual"], abs(row["net_residual"])),
        reverse=True,
    )[: arguments.top]
    receipt = {
        "schema": SCHEMA,
        "status": "passed" if equilibrium_closed else "partial",
        "input": {"path": str(source), "sha256": sha256(source)},
        "source": decoded["source"],
        "equilibrium": {
            "q_sha256": decoded["q_sha256"],
            "dof_count": NV,
            "root_dof_count": ROOT_DOF_COUNT,
            "terms_available": decoded["terms_available"],
            "contact_active_set_selected": decoded["contact_active_set_selected"],
            "fibre_tendon_equilibrium": decoded["fibre_tendon_equilibrium"],
            "maximum_normalized_residual_tolerance": arguments.maximum_normalized_residual,
            "rows": rows,
            "worst_coordinates": worst,
        },
        "residual": residual,
        "qualification": {
            "complete_force_terms": terms_complete,
            "per_dof_source_audit": len(rows) == NV,
            "whole_body_generalized_equilibrium": equilibrium_closed,
            "static_contact_handoff_admissible": handoff_admissible,
            "force_convergence": False,
            "sustained_standing": False,
            "recovery": False,
            "walking": False,
            "anatomical_support_loading": False,
            "activation_calibration": False,
            "blood_mass_transfer": False,
            "material_calibration": False,
            "subject_calibration": False,
        },
        "gate": {
            "maximum_normalized_residual": arguments.maximum_normalized_residual,
            "reasons": reasons,
        },
    }
    write_json(arguments.output.resolve(), receipt)
    print(json.dumps({
        "status": receipt["status"],
        "whole_body_generalized_equilibrium": equilibrium_closed,
        "root_max_absolute_residual": residual["root_max_absolute_residual"],
        "internal_max_absolute_residual": residual["internal_max_absolute_residual"],
        "worst_coordinate": worst[0]["name"],
    }, sort_keys=True))
    return 0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True,
                        help="native numi.human.whole-body-force-audit.v1 JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-normalized-residual", type=float, default=1.0e-3)
    parser.add_argument("--top", type=int, default=12)
    parser.set_defaults(handler=audit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit native Human force balance by root and internal coordinate")
    add_arguments(parser)
    return audit(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
