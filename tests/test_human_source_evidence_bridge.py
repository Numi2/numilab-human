import json
from pathlib import Path

import pytest

from numilab_human.human_source_evidence_bridge import (
    COMPOSITION_EXTENSION,
    CURRENT_EVIDENCE,
    PROFILE,
    SCHEMA,
    EvidenceBridgeError,
    _immutable_write,
    compile_bridge,
)
from numilab_human.physiology import canonical


def test_binds_runtime_organs_blood_contact_and_muscle_graph_without_owner() -> None:
    result = compile_bridge(current_evidence=CURRENT_EVIDENCE,
                            composition_extension=COMPOSITION_EXTENSION, profile=PROFILE)
    assert result["schema"] == SCHEMA
    assert result["source_graph"]["cross_domain_hashes_closed"]
    assert result["source_graph"]["physical_owner_count"] == 0
    assert result["domains"]["organ_blood"]["regional_bed_count"] == 7
    assert result["domains"]["organ_blood"]["source_blood_owner_count"] == 6
    assert result["domains"]["organ_blood"]["mass_conserved"]
    assert result["domains"]["contact"]["proxy_count"] == 30
    assert result["domains"]["muscle"]["source_route_count"] == 416
    assert result["domains"]["muscle"]["candidate_mass_kg"] == pytest.approx(6.859582804936875)
    assert result["domains"]["muscle"]["skeletal_muscle_tissue_mass_owner"] is False
    assert result["qualification"]["blood_tissue_mass_transfer_candidate_bound"]
    assert result["qualification"]["foot_contact_proxy_bound"]
    assert result["qualification"]["muscle_route_volume_incidence_bound"]
    assert result["qualification"]["skeletal_muscle_tissue_mass_candidate_bound"]
    for key in ("anatomical_supports_loading", "activation_calibration",
                "anatomical_blood_mass_transfer", "mechanical_blood_mass_owner",
                "skeletal_muscle_tissue_mass", "fat_geometry_and_mass",
                "material_calibration", "standing", "recovery", "walking",
                "integrated_human_qualification"):
        assert result["qualification"][key] is False


def test_rejects_extension_base_hash_divergence(tmp_path: Path) -> None:
    extension = json.loads(COMPOSITION_EXTENSION.read_text())
    extension["base_receipt"]["sha256"] = "0" * 64
    path = tmp_path / "extension.json"
    path.write_bytes(canonical(extension) + b"\n")
    with pytest.raises(EvidenceBridgeError, match="runtime and composition base hashes diverge"):
        compile_bridge(composition_extension=path)


def test_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_bridge()
    output = tmp_path / "receipt.json"
    first = _immutable_write(output, result)
    assert first == _immutable_write(output, compile_bridge())
