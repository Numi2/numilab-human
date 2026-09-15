from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.current_human_evidence_join import compile_join, immutable_write
from numilab_human.model import ImportError
from numilab_human.physiology import canonical


def test_current_join_binds_pose24_mechanics_and_regional_exchange() -> None:
    result = compile_join()

    assert result["status"] == "partial"
    assert result["subject"] == "one adult male source package"
    assert result["native_owner"]["commit"] == "7625ec565e086faf0dcd349846dadc2d22d65e86"
    assert result["native_owner"]["source_identity_agreement"]
    assert result["clock"] == {
        "nanoseconds": 12500,
        "seconds": 1.25e-05,
        "regional_transport_exact": True,
        "mechanics_exact": True,
    }
    assert result["static_closure"]["generalized_dof_count"] == 128
    assert result["static_closure"]["source_rows"] == 128
    assert result["static_closure"]["sources_per_dof"] == 6
    assert result["static_closure"]["maximum_absolute_force_residual_n"] < 1.0e-3
    assert result["bounded_release"]["completed_steps"] == 512
    assert result["bounded_release"]["maximum_penetration_m"] == 0.0
    assert result["bounded_release"]["replay"] == "bitwise"
    assert result["regional_transport"]["regional_bed_count"] == 7
    assert result["regional_transport"]["accepted_steps"] == 511
    assert result["regional_transport"]["rejected_steps"] == 37
    assert result["regional_transport"]["rollback"] == "bitwise"
    assert result["qualification"]["static_generalized_force_closure"]
    assert result["qualification"]["regional_blood_transport"]
    assert result["qualification"]["oxygen_amount_exchange"]
    for key in (
        "anatomical_supports_loading", "activation_calibration", "force_convergence",
        "anatomical_blood_mass_transfer", "material_calibration", "subject_calibration",
        "sustained_standing", "recovery", "walking", "integrated_human_qualification",
    ):
        assert result["qualification"][key] is False
    assert {row["id"] for row in result["blockers"]} == {
        "force_convergence", "anatomical_supports_loading", "activation_calibration",
        "blood_mass_transfer", "materials_and_subject_calibration", "standing_recovery_walking",
    }


def test_current_join_rejects_promoted_blood_mass_owner(tmp_path: Path) -> None:
    source = Path("Docs/media/native-human-regional-exchange-pose24-20260915/receipt-v1.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["qualification"]["mechanical_blood_mass_owner"] = True
    path = tmp_path / "regional.json"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="regional exchange boundary changed"):
        compile_join(regional_exchange=path)


def test_current_join_rejects_native_source_divergence(tmp_path: Path) -> None:
    source = Path("Docs/media/native-passive-stand-pose24-20260915/receipt-v1.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["source"]["commit"] = "0" * 40
    path = tmp_path / "stand.json"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="native source commits diverge"):
        compile_join(passive_stand=path)


def test_current_join_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_join()
    output = tmp_path / "receipt.json"
    first = immutable_write(output, result)
    assert immutable_write(output, compile_join()) == first
    changed = {**result, "status": "changed"}
    with pytest.raises(ImportError, match="immutable"):
        immutable_write(output, changed)
