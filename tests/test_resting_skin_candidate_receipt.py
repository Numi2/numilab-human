import copy
import hashlib
import json
from pathlib import Path

import pytest

from numilab_human import resting_anatomy
from numilab_human.resting_anatomy import _compose_skin_candidate_receipt_document


def _write(path, data):
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path):
    base_skin = tmp_path / "base.nhskin"
    skin_sha = _write(base_skin, b"source skin")
    candidate_skin = tmp_path / "candidate.nhskin"
    candidate_sha = _write(candidate_skin, b"candidate skin")
    anatomy_payload = tmp_path / "anatomy.nhanatomy"
    anatomy_sha = _write(anatomy_payload, b"anatomy unchanged")
    base_receipt_path = tmp_path / "base-receipt.json"
    base_receipt_path.write_text("{}")
    candidate_manifest_path = tmp_path / "candidate-manifest.json"
    candidate_manifest = {
        "schema": "numi.human.skin-lower-limb-anchor-rebind-candidate.v1",
        "status": "source_identity_preserved_visual_skin_candidate_pending_native_geometry_audit",
        "inputs": {
            "source_payload": {"path": str(base_skin), "sha256": skin_sha},
            "registration_sha256": "reg-sha",
            "runtime_reference": {"rigid": {"sha256": "rigid-sha"}},
        },
        "output_payload": {"path": str(candidate_skin), "sha256": candidate_sha},
        "nhtiss4_owner_alignment": {"present_lower_limb_owner_records_byte_exact": True},
        "preservation": {"triangle_indices_and_order_byte_identical": True},
        "changed_skin_regions": {"combined_geometry": {"dominant_anatomical_region_breakdown": {"torso": {"maximum": 0.01}}}},
    }
    candidate_manifest_path.write_text(json.dumps(candidate_manifest))
    base = {
        "schema": "numi.human.resting-anatomy-receipt.v1",
        "payload": {"path": str(anatomy_payload), "sha256": anatomy_sha, "vertex_count": 4},
        "functional_bindings": {"anatomy_payload_sha256": anatomy_sha, "rows": [1, 2]},
        "mass_geometry_accounting": {"skin_payload_path": str(base_skin), "skin_payload_sha256": skin_sha},
        "provenance": {"rigid_payload_sha256": "rigid-sha", "bodyparts_registration_sha256": "reg-sha", "other": "untouched"},
        "qualification": {"other_scope": "unchanged"},
    }
    updated_mass = {"skin_payload_path": str(candidate_skin), "skin_payload_sha256": candidate_sha, "closed_skin_volume_m3": None}
    return base, base_receipt_path, candidate_manifest, candidate_manifest_path, candidate_skin, updated_mass, anatomy_sha


def test_skin_candidate_receipt_changes_only_skin_audit_and_provenance(tmp_path):
    base, base_path, candidate, candidate_path, candidate_skin, mass, anatomy_sha = _fixture(tmp_path)
    result = _compose_skin_candidate_receipt_document(
        base, base_path, candidate, candidate_path, candidate_skin, mass, tmp_path / "out.json",
    )
    assert result["payload"] == base["payload"]
    assert result["functional_bindings"] == base["functional_bindings"]
    assert result["qualification"] == base["qualification"]
    assert result["mass_geometry_accounting"] == mass
    assert result["provenance"]["other"] == "untouched"
    assert result["provenance"]["skin_visual_binding_candidate"]["payload_sha256"] == mass["skin_payload_sha256"]
    assert result["payload"]["sha256"] == anatomy_sha


@pytest.mark.parametrize("anchor_matches_candidate", [False, True])
def test_skin_replacement_does_not_present_old_native_anchor_as_current(tmp_path, anchor_matches_candidate):
    base, base_path, candidate, candidate_path, candidate_skin, mass, _ = _fixture(tmp_path)
    anchor = {
        "skin_payload_sha256": (mass["skin_payload_sha256"] if anchor_matches_candidate
                                else base["mass_geometry_accounting"]["skin_payload_sha256"]),
        "posterior_skin_coordinate_m": -0.09247,
        "selected_skin_vertex_count": 5279,
        "initial_pack_sha256": "old-native-pack",
    }
    base["provenance"]["thorax_costal_source_registration"] = {
        "candidate_runtime_anchor": copy.deepcopy(anchor),
        "source_runtime_anchor": {"historical": "preserved"},
    }
    original = copy.deepcopy(base)
    result = _compose_skin_candidate_receipt_document(
        base, base_path, candidate, candidate_path, candidate_skin, mass, tmp_path / "out.json",
    )
    current = result["provenance"]["thorax_costal_source_registration"]["candidate_runtime_anchor"]
    skin_provenance = result["provenance"]["skin_visual_binding_candidate"]
    assert base == original
    assert result["functional_bindings"] == base["functional_bindings"]
    assert result["payload"] == base["payload"]
    assert result["provenance"]["thorax_costal_source_registration"]["source_runtime_anchor"] == {"historical": "preserved"}
    if anchor_matches_candidate:
        assert current == anchor
        assert "historical_thorax_candidate_runtime_anchor" not in skin_provenance
    else:
        assert current["status"] == "derived_by_native_runtime_from_loaded_skin"
        assert current["skin_payload_sha256"] == mass["skin_payload_sha256"]
        assert "posterior_skin_coordinate_m" not in current
        assert "selected_skin_vertex_count" not in current
        assert skin_provenance["historical_thorax_candidate_runtime_anchor"] == anchor
        assert skin_provenance["inherited_native_geometry_audits_cover_candidate_skin"] is False


def test_skin_candidate_receipt_rejects_source_or_rigid_identity_mismatch(tmp_path):
    base, base_path, candidate, candidate_path, candidate_skin, mass, _ = _fixture(tmp_path)
    base["mass_geometry_accounting"]["skin_payload_sha256"] = "wrong-source"
    with pytest.raises(ValueError, match="source differs"):
        _compose_skin_candidate_receipt_document(
            base, base_path, candidate, candidate_path, candidate_skin, mass, tmp_path / "out.json",
        )
    second = tmp_path / "second"
    second.mkdir()
    base, base_path, candidate, candidate_path, candidate_skin, mass, _ = _fixture(second)
    base["provenance"]["rigid_payload_sha256"] = "wrong-rigid"
    with pytest.raises(ValueError, match="rigid/registration identity"):
        _compose_skin_candidate_receipt_document(
            base, base_path, candidate, candidate_path, candidate_skin, mass, second / "out.json",
        )


def test_skin_candidate_composes_receipt_only_lineage_without_mutating_anatomy(tmp_path, monkeypatch):
    source_skin = tmp_path / "source.nhskin"
    source_skin_sha = _write(source_skin, b"source skin")
    candidate_skin = tmp_path / "candidate.nhskin"
    candidate_skin_sha = _write(candidate_skin, b"candidate skin")
    source_nha = tmp_path / "source.nhanatomy"
    source_nha_sha = _write(source_nha, b"source anatomy")
    candidate_nha = tmp_path / "candidate.nhanatomy"
    candidate_nha_sha = _write(candidate_nha, b"composed anatomy")
    predecessor_receipt_path = tmp_path / "predecessor-receipt.json"
    predecessor = {"schema": "numi.human.resting-anatomy-receipt.v1", "payload": {"path": str(source_nha), "sha256": source_nha_sha}}
    predecessor_receipt_sha = _write(predecessor_receipt_path, json.dumps(predecessor).encode())
    base_receipt_path = tmp_path / "receipt-only.json"
    base = {
        "schema": "numi.human.resting-anatomy-receipt.v1",
        "payload": {"path": str(candidate_nha), "sha256": candidate_nha_sha},
        "functional_bindings": {"anatomy_payload_sha256": candidate_nha_sha, "rows": [1]},
        "mass_geometry_accounting": {"skin_payload_path": str(source_skin), "skin_payload_sha256": source_skin_sha},
        "provenance": {
            "rigid_payload_sha256": "rigid-sha",
            "bodyparts_registration_sha256": "registration-sha",
            "airway_sibling_overlap_partition": {
                "base_receipt_path": str(predecessor_receipt_path),
                "base_receipt_sha256": predecessor_receipt_sha,
                "base_payload_path": str(source_nha),
                "base_payload_sha256": source_nha_sha,
                "output_payload_path": str(candidate_nha),
                "output_payload_sha256": candidate_nha_sha,
            },
            "source_id_map": {"1": {"name": "heart"}},
        },
        "qualification": {"source_geometry": True},
        "thorax_source_volume_m3": 0.01,
    }
    common_field = {"schema": "numi.human.cardiac_common_field.v1"}
    common_assets = {}
    for owner, filename, contents in (
        ("map", "common-cardiac-map-f32.bin", b"map bytes"),
        ("polynomials", "common-cardiac-volumes-f32.bin", b"polynomial bytes"),
        ("domain_boxes", "common-cardiac-domains-f32.bin", b"domain bytes"),
    ):
        sidecar = tmp_path / filename
        digest = _write(sidecar, contents)
        common_field[owner] = {"path": filename, "sha256": digest}
        common_assets[owner] = (filename, digest)
    base["provenance"]["cardiac_geometry_binding"] = {"common_field": common_field}
    base_receipt_path.write_text(json.dumps(base))
    candidate_manifest = {
        "schema": "numi.human.skin-lower-limb-anchor-rebind-candidate.v1",
        "status": "pending_native_geometry_audit",
        "inputs": {
            "source_payload": {"path": str(source_skin), "sha256": source_skin_sha},
            "registration_sha256": "registration-sha",
            "runtime_reference": {"rigid": {"sha256": "rigid-sha"}},
        },
        "output_payload": {"path": str(candidate_skin), "sha256": candidate_skin_sha},
        "preservation": {"indices_unchanged": True},
        "changed_skin_regions": {"combined_geometry": {"dominant_anatomical_region_breakdown": {"legs": {"maximum": 0.1}}}},
    }
    candidate_manifest_path = tmp_path / "candidate-manifest.json"
    candidate_manifest_path.write_text(json.dumps(candidate_manifest))
    accounting = {"skin_payload_path": str(candidate_skin), "skin_payload_sha256": candidate_skin_sha, "closed_skin_volume_m3": None}
    monkeypatch.setattr(resting_anatomy, "_mass_and_skin_volume_audit", lambda **_: accounting.copy())
    output = tmp_path / "composed"
    result = resting_anatomy.compose_skin_binding_candidate(
        base_receipt_path, candidate_skin, candidate_manifest_path, output,
    )
    composed = json.loads(Path(result["receipt_path"]).read_text())
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert result["anatomy_payload_unchanged"] and result["functional_bindings_unchanged"]
    assert composed["payload"] == base["payload"]
    assert composed["functional_bindings"] == base["functional_bindings"]
    assert {key: composed["mass_geometry_accounting"][key] for key in accounting} == accounting
    assert manifest["schema"] == "numi.human.resting-anatomy-manifest.v1"
    assert manifest["payload"] == base["payload"]
    assert manifest["source_receipt_lineage"]["predecessor_receipt_sha256"] == predecessor_receipt_sha
    candidate_provenance = composed["provenance"]["skin_visual_binding_candidate"]["base_anatomy_receipt"]
    assert candidate_provenance == {"path": str(base_receipt_path.resolve()), "sha256": result["base_receipt_sha256"]}
    published_common = composed["provenance"]["cardiac_geometry_binding"]["common_field"]
    for owner, (filename, digest) in common_assets.items():
        relative_path = published_common[owner]["path"]
        assert relative_path == filename
        relocated = Path(result["receipt_path"]).parent / relative_path
        assert hashlib.sha256(relocated.read_bytes()).hexdigest() == digest
        assert relocated.read_bytes() == (tmp_path / filename).read_bytes()

    bad_missing = copy.deepcopy(base)
    bad_missing["provenance"]["native_muscle_surfaces"] = {
        "payload_path": str(tmp_path / "missing.nhtissue"), "sha256": "0" * 64,
        "manifest_path": str(tmp_path / "missing.manifest.json"), "manifest_sha256": "0" * 64,
    }
    base_receipt_path.write_text(json.dumps(bad_missing))
    with pytest.raises(ValueError, match="payload or manifest is missing"):
        resting_anatomy.compose_skin_binding_candidate(
            base_receipt_path, candidate_skin, candidate_manifest_path, tmp_path / "missing-muscle",
        )

    tissue_path = tmp_path / "muscle.nhtissue"
    tissue_sha = _write(tissue_path, b"muscle payload")
    tissue_manifest_path = tmp_path / "muscle.manifest.json"
    tissue_manifest_sha = _write(tissue_manifest_path, b"muscle manifest")
    bad_stale = copy.deepcopy(base)
    bad_stale["provenance"]["native_muscle_surfaces"] = {
        "payload_path": str(tissue_path), "sha256": "0" * 64,
        "manifest_path": str(tissue_manifest_path), "manifest_sha256": tissue_manifest_sha,
    }
    base_receipt_path.write_text(json.dumps(bad_stale))
    with pytest.raises(ValueError, match="hash is stale"):
        resting_anatomy.compose_skin_binding_candidate(
            base_receipt_path, candidate_skin, candidate_manifest_path, tmp_path / "stale-muscle",
        )

    missing_manifest_owner = copy.deepcopy(base)
    missing_manifest_owner["provenance"]["native_muscle_surfaces"] = {
        "payload_path": str(tissue_path), "sha256": tissue_sha,
        "manifest_path": str(tmp_path / "missing-manifest.json"), "manifest_sha256": "0" * 64,
    }
    base_receipt_path.write_text(json.dumps(missing_manifest_owner))
    with pytest.raises(ValueError, match="payload or manifest is missing"):
        resting_anatomy.compose_skin_binding_candidate(
            base_receipt_path, candidate_skin, candidate_manifest_path, tmp_path / "missing-muscle-manifest",
        )

    stale_manifest_owner = copy.deepcopy(base)
    stale_manifest_owner["provenance"]["native_muscle_surfaces"] = {
        "payload_path": str(tissue_path), "sha256": tissue_sha,
        "manifest_path": str(tissue_manifest_path), "manifest_sha256": "0" * 64,
    }
    base_receipt_path.write_text(json.dumps(stale_manifest_owner))
    with pytest.raises(ValueError, match="hash is stale"):
        resting_anatomy.compose_skin_binding_candidate(
            base_receipt_path, candidate_skin, candidate_manifest_path, tmp_path / "stale-muscle-manifest",
        )
