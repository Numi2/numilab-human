from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.regional_tissue_mass_candidate import (
    PROFILE,
    compile_candidate,
)
from numilab_human.physiology import canonical


def test_native_regional_partition_is_conservative_but_not_promoted() -> None:
    result = compile_candidate()
    assert result["status"] == "partial"
    assert result["region"]["donor_body"] == 20
    assert result["region"]["cooked_nodes"] == 13516
    assert result["region"]["cooked_tetrahedra"] == 46278
    assert result["region"]["source_member_count"] == 14
    assert result["region"]["tissue_mass_kg"] == pytest.approx(0.11369939548001184)
    assert result["qualification"]["mass_conservation"]
    assert result["qualification"]["native_metal_replay"]
    assert result["qualification"]["com_frame_rebase"]
    assert result["qualification"]["regional_tissue_mass_candidate"]
    assert not result["qualification"]["production_mechanical_mass_owner"]
    assert not result["qualification"]["whole_body_dynamic_mass_matrix"]
    assert not result["qualification"]["material_calibration"]
    assert not result["qualification"]["standing_recovery_walking"]


def test_mass_partition_tampering_is_rejected(tmp_path: Path) -> None:
    source = Path("Docs/media/tissue-ownership-20260908/mass-compilation-metal-final.json")
    mass = json.loads(source.read_text(encoding="utf-8"))
    mass["partitions"][0]["remaining_mass_kg"] += 1.0
    mass_path = tmp_path / "mass.json"
    mass_path.write_bytes(canonical(mass) + b"\n")
    with pytest.raises(ImportError, match="not conservative"):
        compile_candidate(mass_compilation=mass_path)


def test_payload_hash_tampering_is_rejected(tmp_path: Path) -> None:
    payload = tmp_path / "costal-tissue.nhtbind"
    payload.write_bytes(Path("Docs/media/tissue-ownership-20260908/costal-tissue.nhtbind").read_bytes() + b"x")
    with pytest.raises(ImportError, match="payload hash"):
        compile_candidate(binding_payload=payload)


def test_profile_must_remain_canonical(tmp_path: Path) -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    profile["expected"]["donor_body"] = 21
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ImportError, match="not canonical"):
        compile_candidate(profile=path)
