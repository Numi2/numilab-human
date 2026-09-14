from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.native_regional_exchange_requalification import (
    NATIVE_LOG,
    _immutable_write,
    compile_receipt,
)


def test_current_regional_exchange_binds_exact_clock_and_conservation() -> None:
    result = compile_receipt()
    assert result["status"] == "partial"
    assert result["source"]["commit"] == "c45fa9622f6c73b58febdc24a7115aecf3d7699f"
    assert result["results"]["source_compartment_count"] == 21
    assert result["results"]["regional_bed_count"] == 7
    assert result["results"]["attempted_steps"] == 512
    assert result["results"]["accepted_steps_environment_0"] == 511
    assert result["results"]["timestep_nanoseconds"] == 12500
    assert result["results"]["blood_density_candidate_kg_per_m3"] == 1060.0
    assert result["qualification"]["current_native_replay"]
    assert result["qualification"]["regional_blood_transport"]
    assert result["qualification"]["oxygen_amount_exchange"]
    assert result["qualification"]["accepted_step_conservation"]
    assert not result["qualification"]["mechanical_blood_mass_owner"]
    assert not result["qualification"]["anatomical_vessel_lumen"]


def test_native_log_tampering_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "native.log"
    text = NATIVE_LOG.read_text(encoding="utf-8").replace("accepted_environment0=511", "accepted_environment0=510")
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ImportError, match="native log counters"):
        compile_receipt(native_log=path)


def test_current_regional_exchange_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_receipt()
    path = tmp_path / "receipt.json"
    assert _immutable_write(path, result) == _immutable_write(path, result)
    path.write_text(json.dumps({"forged": True}) + "\n", encoding="utf-8")
    with pytest.raises(ImportError, match="immutable"):
        _immutable_write(path, result)
