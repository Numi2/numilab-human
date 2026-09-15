from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.physiology import canonical
from numilab_human.subject_mass_scaling_candidate import compile_candidate, immutable_write


def test_subject_mass_candidate_closes_scalar_mass_without_runtime_admission() -> None:
    result = compile_candidate()
    assert result["schema"] == "HumanPack.subject-mass-scaling-candidate.v1"
    assert len(result["bodies"]) == 103
    assert result["scaling"]["target_mass_kg"] == 65.5
    assert result["scaling"]["scaled_mass_kg"] == pytest.approx(65.5, abs=1.0e-12)
    assert result["scaling"]["mass_closure_error_kg"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["scaling"]["mass_factor"] == pytest.approx(0.6743404081661175)
    assert result["qualification"]["target_mass_closure"]
    assert result["qualification"]["source_body_tree_preserved"]
    assert not result["qualification"]["segment_mass_calibration"]
    assert not result["qualification"]["inertia_calibration"]
    assert not result["qualification"]["mechanical_runtime_admitted"]
    assert {item["id"] for item in result["blockers"]} == {
        "segment_mass_calibration", "geometry_and_inertia_calibration", "native_runtime_admission",
    }


def test_subject_mass_candidate_rejects_promoted_subject_calibration(tmp_path: Path) -> None:
    source = Path("Docs/media/addbiomechanics-subject-binding-20260915/receipt-v1.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["qualification"]["subject_calibration_ready"] = True
    path = tmp_path / "binding.json"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="promoted mass calibration"):
        compile_candidate(subject_binding=path)


def test_subject_mass_candidate_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_candidate()
    output = tmp_path / "candidate.json"
    first = immutable_write(output, result)
    assert immutable_write(output, compile_candidate()) == first
    with pytest.raises(ImportError, match="immutable"):
        immutable_write(output, {**result, "status": "changed"})
