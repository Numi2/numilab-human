from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from numilab_human.force_ledger import INPUT_SCHEMA, REQUIRED_COMPONENTS, audit
from numilab_human.model import ImportError


def _snapshot(*, residual_index: int | None = None, assembly_error: bool = False) -> dict:
    names = [f"dof_{index}" for index in range(128)]
    kinds = ["translation"] * 3 + ["rotation"] * 125
    component_values = {name: [0.0] * 128 for name in REQUIRED_COMPONENTS}
    component_values["gravity_bias"][2] = -100.0
    component_values["support_contact"][2] = 100.0
    if residual_index is not None:
        component_values["muscle_tendon"][residual_index] = 10.0
    reported = [sum(component_values[name][index] for name in REQUIRED_COMPONENTS) for index in range(128)]
    if assembly_error:
        reported[7] += 1.0
    return {
        "schema": INPUT_SCHEMA,
        "nv": 128,
        "coordinate_names": names,
        "coordinate_kinds": kinds,
        "components": [
            {"name": name, "owner": f"owner:{name}", "values": values}
            for name, values in component_values.items()
        ],
        "reported_net": reported,
        "generalized_acceleration": [0.0] * 128,
        "metadata": {"fixture": True},
    }


def _run(tmp_path: Path, payload: dict) -> dict:
    source = tmp_path / "forces.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "receipt.json"
    arguments = argparse.Namespace(
        input=source,
        output=output,
        maximum_assembly_error=1.0e-4,
        maximum_closure_ratio=0.05,
        top=12,
    )
    assert audit(arguments) == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_balanced_force_ledger_passes(tmp_path: Path) -> None:
    receipt = _run(tmp_path, _snapshot())
    assert receipt["status"] == "passed"
    assert receipt["coverage"]["full_force_coverage"]
    assert receipt["qualification"]["full_generalized_force_ledger"]
    assert receipt["qualification"]["per_dof_source_audit"]
    assert len(receipt["per_dof_audit"]) == 128
    assert {row["name"] for row in receipt["per_dof_audit"][2]["contributions"]} == set(REQUIRED_COMPONENTS)
    assert receipt["residual"]["maximum_closure_ratio"] == 0.0
    assert not receipt["qualification"]["force_convergence"]


def test_internal_residual_is_ranked_and_fails(tmp_path: Path) -> None:
    receipt = _run(tmp_path, _snapshot(residual_index=42))
    assert receipt["status"] == "partial"
    assert not receipt["qualification"]["generalized_force_closed"]
    assert receipt["worst_coordinates"][0]["index"] == 42
    assert receipt["worst_coordinates"][0]["dominant_component"] == "muscle_tendon"
    assert receipt["worst_coordinates"][0]["normalized_residual"] == 1.0
    assert any(row["value"] == 10.0 for row in receipt["per_dof_audit"][42]["contributions"])
    assert receipt["residual"]["maximum_internal_closure_ratio"] == 1.0


def test_unaccounted_authoritative_force_fails_assembly(tmp_path: Path) -> None:
    receipt = _run(tmp_path, _snapshot(assembly_error=True))
    assert receipt["status"] == "partial"
    assert not receipt["coverage"]["full_force_coverage"]
    assert not receipt["qualification"]["component_assembly_closed"]
    assert receipt["residual"]["maximum_assembly_error"] == 1.0


def test_near_zero_force_scale_uses_explicit_absolute_floor(tmp_path: Path) -> None:
    payload = _snapshot()
    payload["components"][1]["values"][51] = 2.0e-6
    payload["reported_net"][51] = 2.0e-6
    receipt = _run(tmp_path, payload)
    assert receipt["status"] == "passed"
    assert receipt["qualification"]["generalized_force_closed"]
    assert receipt["residual"]["maximum_absolute_force_residual"] == 2.0e-6
    assert receipt["residual"]["maximum_closure_ratio"] == 0.002


def test_missing_force_owner_is_rejected(tmp_path: Path) -> None:
    payload = _snapshot()
    payload["components"] = [row for row in payload["components"] if row["name"] != "passive_tissue"]
    source = tmp_path / "forces.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "receipt.json"
    arguments = argparse.Namespace(
        input=source,
        output=output,
        maximum_assembly_error=1.0e-4,
        maximum_closure_ratio=0.05,
        top=12,
    )
    with pytest.raises(ImportError):
        audit(arguments)
