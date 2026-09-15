from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.body_composition_release_join import (
    _immutable_write,
    compile_candidate,
)
from numilab_human.model import ImportError


def test_release_join_binds_all_cross_domain_receipts() -> None:
    result = compile_candidate()
    assert result["status"] == "partial"
    assert result["subject"] == "one adult male source package"
    assert result["domain_counts"]["muscle_source_routes"] == 416
    assert result["domain_counts"]["organ_blood_tissue_beds"] == 7
    assert result["domain_counts"]["fat_surfaces"] == 0
    assert result["ownership"]["physical_owner_count"] == 0
    assert result["qualification"]["cross_domain_owner_nonduplication"]
    assert result["qualification"]["blood_tissue_conservation_and_rollback_bound"]
    assert not result["qualification"]["organ_mechanics"]
    assert not result["qualification"]["integrated_human_qualification"]


def test_release_join_rejects_promoted_blood_owner(tmp_path: Path) -> None:
    source = Path("Docs/media/organ-blood-mass-transfer-20260915/receipt-v3.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["qualification"]["mechanical_blood_mass_owner"] = True
    path = tmp_path / "blood.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ImportError, match="promoted mechanical_blood_mass_owner"):
        compile_candidate(blood_tissue=path)


def test_release_join_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_candidate()
    path = tmp_path / "receipt.json"
    assert _immutable_write(path, result) == _immutable_write(path, result)
    path.write_text(json.dumps({"forged": True}) + "\n", encoding="utf-8")
    with pytest.raises(ImportError, match="immutable"):
        _immutable_write(path, result)
