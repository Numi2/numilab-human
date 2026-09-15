from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.subject_scaled_native_requalification import (
    STDOUT,
    _immutable_write,
    compile_receipt,
)


def test_subject_scaled_native_requalification_records_bounded_m4_replay() -> None:
    result = compile_receipt()
    assert result["status"] == "partial"
    assert result["source"]["resolved_commit"] == "337741b51bfc4a5552a837dacb4d0b5c4268d298"
    assert result["command"]["timestep_seconds"] == 0.0000125
    assert result["results"]["persistent_completed_steps"] == 64
    assert result["results"]["compiled_stand_recruited_muscles"] == 416
    assert result["results"]["compiled_stand_active_support_contacts"] == 6
    assert result["results"]["persistent_max_acceleration_mps2"] == pytest.approx(0.146436423063)
    assert result["results"]["persistent_max_penetration_m"] == 0.0
    assert result["results"]["dynamic_force_audit_rows"] == 128
    assert result["results"]["trace_endpoint_equivalent"] == "bitwise"
    assert result["qualification"]["scaled_binary_consumed_by_native_runtime"]
    assert result["qualification"]["bounded_12p5_us_m4_replay"]
    assert not result["qualification"]["full_generalized_force_convergence"]
    assert not result["qualification"]["sustained_standing"]
    assert not result["qualification"]["walking"]


def test_subject_scaled_native_requalification_rejects_tampered_stdout(tmp_path: Path) -> None:
    path = tmp_path / "native.stdout.txt"
    path.write_bytes(STDOUT.read_bytes().replace(b"persistent_root_assistance=none",
                                                  b"persistent_root_assistance=borrowed"))
    with pytest.raises(ImportError, match="bounded replay invariants"):
        compile_receipt(stdout=path)


def test_subject_scaled_native_requalification_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_receipt()
    path = tmp_path / "receipt.json"
    assert _immutable_write(path, result) == _immutable_write(path, result)
    path.write_text(json.dumps({"forged": True}) + "\n", encoding="utf-8")
    with pytest.raises(ImportError, match="immutable"):
        _immutable_write(path, result)
