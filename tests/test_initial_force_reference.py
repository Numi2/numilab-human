from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pytest

from numilab_human.equilibrium_force_snapshot import (
    INITIAL_REFERENCE_PREFIX, INITIAL_REFERENCE_SCHEMA, PREFIX, convert,
)
from numilab_human.force_ledger import _load_snapshot
from numilab_human.model import ImportError


def payload() -> dict:
    rows = []
    for dof in reversed(range(128)):
        muscle = 4.0 if dof == 7 else 0.0
        support = 100.0 if dof == 2 else 0.0
        gravity = 100.0 if dof == 2 else 0.0
        passive = -3.5 if dof == 7 else 0.0
        rows.append({
            "dof": dof, "name": "joint_seven" if dof == 7 else "unnamed",
            "metal_muscle_force_n": muscle, "support_force_n": support,
            "equality_force_n": 0.0, "limit_force_n": 0.0,
            "passive_force_n": passive,
            # Deliberately different: this is never an additional load.
            "compiled_passive_force_n": -8.0 if dof == 7 else 0.0,
            "gravity_target_n": gravity,
            "residual_n": muscle + support + passive - gravity,
        })
    return {"schema": "numi.human.persistent-initial-force-reference.v1",
            "maximum_abs_residual_n": 0.5, "rows": rows}


def run(tmp_path: Path, data: dict | None = None, *, compressed: bool = False) -> tuple[dict, Path]:
    data = payload() if data is None else data
    raw = (INITIAL_REFERENCE_PREFIX + json.dumps(data) + "\n").encode()
    source = tmp_path / ("native.log.gz" if compressed else "native.log")
    source.write_bytes(gzip.compress(raw) if compressed else raw)
    output = tmp_path / "reference.json"
    args = argparse.Namespace(log=source, output=output, coordinate_map=None,
                              maximum_reconstruction_error=1e-4)
    assert convert(args) == 0
    return json.loads(output.read_text()), output


@pytest.mark.parametrize("compressed", [False, True])
def test_reference_counts_runtime_passive_once(tmp_path: Path, compressed: bool) -> None:
    result, _ = run(tmp_path, compressed=compressed)
    assert result["schema"] == INITIAL_REFERENCE_SCHEMA
    components = {row["name"]: row for row in result["components"]}
    assert components["passive_tissue"]["values"][7] == -3.5
    assert components["support_contact"]["owner"].startswith("CompiledEquilibriumReference.")
    assert result["reported_net"][7] == 0.5
    metadata = result["metadata"]
    assert metadata["compiled_passive_reference"]["values"][7] == -8.0
    assert metadata["compiled_passive_reference"]["additive"] is False
    assert metadata["reference_only"] is True
    assert metadata["runtime_constraint_forces_measured"] is False
    assert metadata["source_revision"] is None
    assert metadata["source_owner"] == "Numi2/numi-lab"
    assert result["coordinate_names"][7] == "v_007:joint_seven"
    assert result["coordinate_kinds"] == ["unknown"] * 128
    assert metadata["coordinate_map"]["named_coordinates"] == 1


def test_initial_reference_is_not_admitted_to_force_ledger(tmp_path: Path) -> None:
    _, path = run(tmp_path)
    with pytest.raises(ImportError, match="schema mismatch"):
        _load_snapshot(path)


@pytest.mark.parametrize("mutation", ["duplicate_dof", "missing_dof", "nan", "bool", "double_count", "schema", "peak", "name"])
def test_invalid_initial_reference_rejected(tmp_path: Path, mutation: str) -> None:
    data = payload()
    if mutation == "duplicate_dof":
        data["rows"][0]["dof"] = data["rows"][1]["dof"]
    elif mutation == "missing_dof":
        data["rows"].pop()
    elif mutation == "nan":
        data["rows"][0]["compiled_passive_force_n"] = float("nan")
    elif mutation == "bool":
        data["rows"][0]["passive_force_n"] = True
    elif mutation == "double_count":
        row = next(row for row in data["rows"] if row["dof"] == 7)
        row["residual_n"] += row["compiled_passive_force_n"]
        data["maximum_abs_residual_n"] = abs(row["residual_n"])
    elif mutation == "schema":
        data["schema"] = "numi.human.persistent-dynamic-force-audit.v1"
    elif mutation == "peak":
        data["maximum_abs_residual_n"] = 0.0
    else:
        data["rows"][0]["name"] = 42
    with pytest.raises(ImportError):
        run(tmp_path, data)
    assert not (tmp_path / "reference.json").exists()


@pytest.mark.parametrize("mode", ["duplicate_record", "mixed_record", "duplicate_json_key"])
def test_ambiguous_record_rejected(tmp_path: Path, mode: str) -> None:
    text = INITIAL_REFERENCE_PREFIX + json.dumps(payload()) + "\n"
    if mode == "duplicate_record":
        text += text
    elif mode == "mixed_record":
        text += PREFIX + "{}\n"
    else:
        text = text.replace('"maximum_abs_residual_n": 0.5',
                            '"maximum_abs_residual_n": 0.5, "maximum_abs_residual_n": 0.5')
    source = tmp_path / "bad.log"
    source.write_text(text)
    with pytest.raises(ImportError):
        convert(argparse.Namespace(log=source, output=tmp_path / "out.json",
                                   coordinate_map=None, maximum_reconstruction_error=1e-4))


def test_closed_initial_reference_still_cannot_qualify_equilibrium(tmp_path: Path) -> None:
    data = payload()
    row = next(row for row in data["rows"] if row["dof"] == 7)
    row["equality_force_n"] = -0.5
    row["residual_n"] = 0.0
    data["maximum_abs_residual_n"] = 0.0
    result, path = run(tmp_path, data)
    assert result["metadata"]["maximum_reconstruction_error"] == 0.0
    with pytest.raises(ImportError, match="schema mismatch"):
        _load_snapshot(path)
