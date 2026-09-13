from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.organ_blood_ownership_ledger import (
    SCHEMA, compile_ledger,
)
from numilab_human.physiology import canonical


ROOT = Path(__file__).resolve().parents[1]


def test_ledger_binds_source_and_hydraulic_identities_without_promoting_mass() -> None:
    result = compile_ledger()
    assert result["schema"] == SCHEMA
    assert result["integrity"]["cross_domain_identity_bound"]
    assert result["integrity"]["duplicate_physical_owner_rejected"]
    assert result["integrity"]["physical_mass_owner_count"] == 0
    assert len(result["hydraulic_ownership"]["cavity_bindings"]) == 4
    assert len(result["vessel_ownership"]["surface_bindings"]) == 6
    assert result["vessel_ownership"]["surface_mass_moment_candidate"]["owner_count"] == 6
    assert result["qualification"]["zeroth_first_second_mass_moments_candidate"]
    assert result["hydraulic_ownership"]["unique_compartment_identifiers"]
    assert not result["qualification"]["anatomical_blood_mass_transfer"]
    assert not result["qualification"]["activation"]
    assert not result["qualification"]["standing"]
    assert not result["qualification"]["walking"]


def test_cavity_source_hash_tampering_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bridge.json"
    value = json.loads((ROOT / "Docs/media/organ-blood-cavity-bridge-20260913/bridge.json").read_text())
    value["bindings"][0]["source_member_sha256"] = "0" * 64
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(HumanImportError, match="source hash disagrees"):
        compile_ledger(cavity_bridge=path)


def test_hidden_mechanical_owner_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "torso.json"
    value = json.loads((ROOT / "Docs/media/organ-torso-body-link-20260913/body-links.json").read_text())
    value["bindings"][0]["mechanical_mass_owner"] = "forged-owner"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(HumanImportError, match="mechanical mass owner"):
        compile_ledger(torso=path)


def test_duplicate_vessel_member_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "vessel.json"
    value = json.loads((ROOT / "Docs/media/organ-vessel-registration-corrected-20260913/registration.json").read_text())
    value["bindings"].append(copy.deepcopy(value["bindings"][0]))
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(HumanImportError, match="must contain 6 bindings"):
        compile_ledger(vessel_registration=path)


def test_mass_moment_candidate_registration_hash_is_bound(tmp_path: Path) -> None:
    path = tmp_path / "mass-moments.json"
    value = json.loads((ROOT / "Docs/media/vessel-mass-moment-owner-corrected-20260913/receipt.json").read_text())
    value["source"]["registration_sha256"] = "0" * 64
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(HumanImportError, match="registration hash disagrees"):
        compile_ledger(vessel_mass_moments=path)
