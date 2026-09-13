from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.cardiac_blood_mass_candidate import compile_candidate
from numilab_human.model import ImportError as HumanImportError
from numilab_human.physiology import canonical


ROOT = Path(__file__).resolve().parents[1]


def test_both_cavity_candidates_bind_four_hydraulic_chambers_and_conserve_mass() -> None:
    result = compile_candidate()
    assert result["schema"] == "HumanPack.cardiac-blood-mass-candidate.v1"
    assert result["selection"] is None
    assert len(result["candidates"]) == 2
    assert len(result["bindings"]) == 8
    assert result["mass_budget"]["both_candidates_share_hydraulic_budget"]
    assert result["mass_budget"]["candidate_mass_kg"] == pytest.approx(0.39974235600733804)
    assert result["qualification"]["two_disjoint_right_heart_candidates"]
    assert result["qualification"]["four_chamber_hydraulic_binding"]
    assert result["qualification"]["unresolved_density_candidate_explicit"]
    assert not result["qualification"]["physical_volume_owner_assigned"]
    assert not result["qualification"]["mechanical_mass_assigned"]
    assert not result["qualification"]["two_way_blood_tissue_transfer"]
    assert all(row["physical_volume_owner"] is None for row in result["bindings"])
    assert all(row["mechanical_mass_owner"] is None for row in result["bindings"])


def test_partition_selection_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "partition.json"
    value = json.loads((ROOT / "Docs/media/cardiac-partition-20260912/partition.json").read_text())
    value["selection"] = "right_atrium_priority"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(HumanImportError, match="selected or stepped"):
        compile_candidate(partition=path)


def test_density_promotion_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "owner.json"
    value = json.loads((ROOT / "config/cvsim21-blood-mass-owner.v1.json").read_text())
    value["density_provenance"]["kind"] = "source"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(HumanImportError, match="density must remain explicitly unresolved"):
        compile_candidate(blood_owner=path)


def test_bridge_tampering_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bridge.json"
    value = json.loads((ROOT / "Docs/media/organ-blood-cavity-bridge-20260913/bridge.json").read_text())
    value["qualification"]["cavity_domains_disjoint"] = True
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(HumanImportError, match="bridge boundary changed"):
        compile_candidate(bridge=path)
