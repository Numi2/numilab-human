from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.subject_scaled_runtime_input import compile_input


def test_subject_scaled_input_preserves_layout_and_closes_mass(tmp_path: Path) -> None:
    output = tmp_path / "scaled.nhrigid"
    receipt = tmp_path / "receipt.json"
    result = compile_input(output=output, receipt=receipt)
    assert result["schema"] == "HumanPack.subject-scaled-runtime-input.v1"
    assert result["source"]["source_body_count"] == 103
    assert result["source"]["engine_body_count"] == 157
    assert result["mapping"]["mapped_source_rows"] == 103
    assert result["mapping"]["patched_engine_rows"] == 103
    assert result["mapping"]["scaled_binary_mass_kg"] == pytest.approx(65.5, abs=2.0e-5)
    assert result["qualification"]["nhrigid_layout_verified"]
    assert result["qualification"]["body_tree_preserved"]
    assert not result["qualification"]["native_binary_consumed"]
    assert not result["qualification"]["native_replay_qualified"]
    assert len(output.read_bytes()) == 60324
    assert json.loads(receipt.read_text(encoding="utf-8"))["output"]["sha256"] == result["output"]["sha256"]


def test_subject_scaled_input_rejects_tampered_scaling(tmp_path: Path) -> None:
    source = Path("Docs/media/subject-mass-scaling-20260915/receipt-v1.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["bodies"][1]["source_mass_kg"] += 1.0
    path = tmp_path / "scaling.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ImportError, match="disagrees with NHRIGID2"):
        compile_input(scaling=path, output=tmp_path / "scaled.nhrigid", receipt=tmp_path / "receipt.json")


def test_subject_scaled_input_is_immutable(tmp_path: Path) -> None:
    output = tmp_path / "scaled.nhrigid"
    receipt = tmp_path / "receipt.json"
    compile_input(output=output, receipt=receipt)
    first = output.read_bytes()
    compile_input(output=output, receipt=receipt)
    assert output.read_bytes() == first
