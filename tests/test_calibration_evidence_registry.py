from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.calibration_evidence_registry import (
    EXPECTED_ROWS, ROOT, compile_registry, immutable_write,
)
from numilab_human.model import ImportError


def test_live_registry_keeps_fit_holdout_and_owner_scopes_separate() -> None:
    result = compile_registry()
    assert result["status"] == "partial"
    assert {row["id"] for row in result["rows"]} == EXPECTED_ROWS
    assert result["counts"] == {
        "domains": 8, "fit_rows": 2, "held_out_rows": 1,
        "production_owner_rows": 0, "blockers": len(result["blockers"]),
    }
    assert result["qualification"]["registry_complete"]
    assert result["qualification"]["fit_and_holdout_scopes_separated"]
    assert not any(row["production_owner"] for row in result["rows"])
    cartilage = next(row for row in result["rows"] if row["id"] == "cartilage_material")
    assert cartilage["fit_evidence"] and cartilage["held_out_evidence"]
    assert cartilage["fit_points"] == 6 and cartilage["held_out_points"] == 3
    assert not result["qualification"]["material_calibration"]
    assert not result["qualification"]["activation_calibration"]


def test_registry_rejects_a_promoted_cartilage_source(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    profile = json.loads((ROOT / "config/calibration-evidence-registry.v1.json").read_text())
    for row in profile["rows"]:
        source = ROOT / row["source"]
        destination = root / row["source"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        value = json.loads(source.read_text())
        if row["id"] == "cartilage_material":
            value["qualified"] = True
        destination.write_text(json.dumps(value))
    profile_path = root / "config/calibration-evidence-registry.v1.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(ImportError, match="cartilage candidate was promoted"):
        compile_registry(profile=profile_path, root=root)


def test_registry_rejects_production_owner_flag(tmp_path: Path) -> None:
    profile = json.loads((ROOT / "config/calibration-evidence-registry.v1.json").read_text())
    next(row for row in profile["rows"] if row["id"] == "blood_tissue_transfer")["production_owner"] = True
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile))
    with pytest.raises(ImportError, match="attempted production ownership"):
        compile_registry(profile=profile_path, root=ROOT)


def test_registry_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_registry()
    output = tmp_path / "receipt.json"
    first = immutable_write(output, result)
    assert immutable_write(output, compile_registry()) == first
    output.write_text('{"forged":true}\n')
    with pytest.raises(ImportError, match="immutable"):
        immutable_write(output, result)
