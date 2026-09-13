from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.cardiac_cavity_ownership import compile_ownership
from numilab_human.model import ImportError as HumanImportError


ROOT = Path(__file__).resolve().parents[1]


def test_both_disjoint_candidates_are_bound_without_selection() -> None:
    result = compile_ownership()
    assert result["schema"] == "HumanPack.organ-cardiac-cavity-ownership-candidates.v1"
    assert result["selection"] is None
    assert len(result["candidates"]) == 2
    assert all(row["four_cavity_interiors_disjoint"] for row in result["candidates"])
    assert result["qualification"]["body_frame_registered"]
    assert not result["qualification"]["biological_valve_interface_selected"]
    assert not result["qualification"]["blood_mass_assigned"]


def test_partition_selection_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "partition.json"
    value = json.loads((ROOT / "Docs/media/cardiac-partition-20260912/partition.json").read_text())
    value["selection"] = "right_atrium_priority"
    path.write_text(json.dumps(value))
    with pytest.raises(HumanImportError, match="already selected"):
        compile_ownership(partition=path)


def test_bridge_geometry_tampering_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bridge.json"
    value = json.loads((ROOT / "Docs/media/organ-blood-cavity-bridge-20260913/bridge.json").read_text())
    value["cvsim_cavity_reference"]["cavity_geometry_sha256"] = "0" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(HumanImportError, match="disagrees with the bridge"):
        compile_ownership(bridge=path)
