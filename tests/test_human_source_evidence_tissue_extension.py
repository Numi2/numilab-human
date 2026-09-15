import json
from pathlib import Path

import pytest

from numilab_human.human_source_evidence_tissue_extension import (
    BASE,
    BLOOD,
    ORGAN_MASS,
    PROFILE,
    REGIONAL_TISSUE,
    SCHEMA,
    TissueExtensionError,
    _immutable_write,
    compile_candidate,
)
from numilab_human.physiology import canonical


def test_binds_blood_regional_tissue_and_muscle_graph_without_owner_promotion() -> None:
    result = compile_candidate(base=BASE, blood_transfer=BLOOD,
                               regional_tissue=REGIONAL_TISSUE, profile=PROFILE)
    assert result["schema"] == SCHEMA
    assert result["domains"]["organ_blood"]["clock_nanoseconds"] == 12500
    assert result["domains"]["organ_blood"]["bed_count"] == 7
    assert result["domains"]["organ_blood"]["mass_conserved"]
    assert result["domains"]["regional_tissue"]["source_member_count"] == 14
    assert result["domains"]["regional_tissue"]["tissue_mass_kg"] == pytest.approx(0.11369939548001184)
    assert result["qualification"]["blood_transfer_bound"]
    assert result["qualification"]["organ_tissue_mass_bound"]
    assert result["qualification"]["regional_tissue_mass_bound"]
    assert result["qualification"]["exact_clock_bound"]
    assert result["qualification"]["physical_owner_count"] == 0
    assert not result["qualification"]["mechanical_tissue_mass_owner"]
    assert not result["qualification"]["integrated_human_qualification"]


def test_rejects_blood_tissue_hash_divergence(tmp_path: Path) -> None:
    blood = json.loads(BLOOD.read_text())
    blood["source"]["tissue_mass_candidate_sha256"] = "0" * 64
    path = tmp_path / "blood.json"
    path.write_bytes(canonical(blood) + b"\n")
    with pytest.raises(TissueExtensionError, match="blood-transfer and organ tissue-mass hashes diverge"):
        compile_candidate(blood_transfer=path)


def test_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_candidate()
    output = tmp_path / "receipt.json"
    first = _immutable_write(output, result)
    assert first == _immutable_write(output, compile_candidate())
