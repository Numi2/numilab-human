from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.myosim_mass_owner_candidate import compile_candidate
from numilab_human.physiology import canonical


def test_compiled_myosim_mass_owner_binds_exact_body_tree_and_mass() -> None:
    result = compile_candidate()
    ledger = result["rigid_body_mass"]
    assert result["status"] == "partial"
    assert ledger["body_count"] == 103
    assert ledger["mass_bearing_body_count"] == 96
    assert ledger["zero_mass_body_count"] == 7
    assert ledger["total_mass_kg"] == pytest.approx(97.13195176621338)
    assert ledger["bodies"][0]["id"] == 1
    assert ledger["bodies"][0]["parent"] == 0
    assert ledger["bodies"][-1]["id"] == 103
    assert all(row["rigid_body_mass_owner"] for row in ledger["bodies"])
    assert all(row["physical_volume_owner"] is None for row in ledger["bodies"])
    assert all(row["anatomical_tissue_owner"] is None for row in ledger["bodies"])
    assert result["qualification"]["source_rigid_body_mass_nonduplication_checked"]
    assert not result["qualification"]["whole_body_dynamic_mass_matrix_owner"]
    assert not result["qualification"]["standing"]
    assert not result["qualification"]["walking"]


def test_compiled_myosim_mass_owner_rejects_duplicate_body_ids(tmp_path: Path) -> None:
    source = Path("Docs/media/myosim-mass-owner-20260915/source-manifest-v1.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["bodies"][1]["id"] = value["bodies"][0]["id"]
    path = tmp_path / "source.json"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="body ids repeat"):
        compile_candidate(source_manifest=path)


def test_compiled_myosim_mass_owner_rejects_mass_drift(tmp_path: Path) -> None:
    source = Path("Docs/media/myosim-mass-owner-20260915/source-manifest-v1.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["bodies"][1]["mass_kg"] += 0.1
    path = tmp_path / "source.json"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="total mass changed"):
        compile_candidate(source_manifest=path)


def test_compiled_myosim_mass_owner_rejects_orphan_parent(tmp_path: Path) -> None:
    source = Path("Docs/media/myosim-mass-owner-20260915/source-manifest-v1.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["bodies"][10]["parent"] = value["bodies"][10]["id"]
    path = tmp_path / "source.json"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="parent is not an earlier body"):
        compile_candidate(source_manifest=path)
