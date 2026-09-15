from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.native_passive_stand_requalification import (
    PASSIVE_STDOUT,
    _immutable_write,
    compile_receipt,
)


def test_passive_equilibrium_requalification_records_bounded_improvement() -> None:
    result = compile_receipt()
    assert result["status"] == "partial"
    assert result["source"]["commit"] == "ef0fc708db0f4de1a07fca426e5a415f62e9da27"
    assert result["command"]["timestep_seconds"] == 0.0000125
    assert result["results"]["passive"]["persistent_completed_steps"] == 512
    assert result["results"]["passive"]["persistent_source_passive_joint_tissue"]
    assert result["results"]["passive"]["persistent_passive_coordinate_couplings"] == 40
    assert result["results"]["passive"]["compiled_stand_balanced"]
    assert result["results"]["passive"]["compiled_stand_normalized_residual_rms"] < 1.0e-4
    assert result["results"]["passive"]["persistent_max_acceleration_mps2"] < 5.0
    assert result["results"]["baseline"]["persistent_max_acceleration_mps2"] > 30.0
    assert result["qualification"]["complete_static_generalized_balance"]
    assert result["qualification"]["bounded_exact_clock_release"]
    assert not result["qualification"]["sustained_standing"]
    assert not result["qualification"]["walking"]


def test_passive_stdout_tampering_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "native-passive-stdout.txt"
    text = PASSIVE_STDOUT.read_text(encoding="utf-8").replace(
        "persistent_source_passive_joint_tissue=true",
        "persistent_source_passive_joint_tissue=false",
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ImportError, match="passive joint-tissue owner"):
        compile_receipt(passive_stdout=path)


def test_passive_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_receipt()
    path = tmp_path / "receipt.json"
    assert _immutable_write(path, result) == _immutable_write(path, result)
    path.write_text(json.dumps({"forged": True}) + "\n", encoding="utf-8")
    with pytest.raises(ImportError, match="immutable"):
        _immutable_write(path, result)
