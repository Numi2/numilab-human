from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.tissue_native_requalification import (
    OUTPUT,
    _immutable_write,
    compile_receipt,
)


def test_current_costal_receipt_binds_native_partition_without_promotion() -> None:
    result = compile_receipt()
    assert result["status"] == "partial"
    assert result["source"]["commit"] == "c45fa9622f6c73b58febdc24a7115aecf3d7699f"
    assert result["results"]["metal_replay_cases"] == 8
    assert result["results"]["tissue_mass_kg"] == pytest.approx(0.11369939548001184)
    assert result["qualification"]["mass_conservation"]
    assert result["qualification"]["registered_costal_tissue_mass_and_rebase"]
    assert not result["qualification"]["whole_body_dynamic_mass_matrix"]
    assert not result["qualification"]["experimental_material_calibration"]


def test_current_costal_output_tampering_is_rejected(tmp_path: Path) -> None:
    value = json.loads(OUTPUT.read_text(encoding="utf-8"))
    value["partitions"][0]["tissue_mass_kg"] += 1.0
    path = tmp_path / "native-output.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ImportError, match="native output tissue_mass_kg changed"):
        compile_receipt(output=path)


def test_current_costal_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_receipt()
    path = tmp_path / "receipt.json"
    assert _immutable_write(path, result) == _immutable_write(path, result)
    path.write_text('{"forged":true}\n', encoding="utf-8")
    with pytest.raises(ImportError, match="immutable"):
        _immutable_write(path, result)
