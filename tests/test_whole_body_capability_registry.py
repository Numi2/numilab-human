from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.whole_body_capability_registry import (
    PROFILE_SCHEMA,
    SCHEMA,
    SYSTEM_IDS,
    compile_registry,
)


def _fixture_profile(root: Path, *, ready: bool = True,
                     missing_system: str | None = None,
                     missing_source: str | None = None,
                     wrong_schema: str | None = None) -> Path:
    systems = []
    for system_id in sorted(SYSTEM_IDS):
        evidence_id = "source"
        source_path = f"evidence/{system_id}.json"
        if system_id != missing_source:
            (root / source_path).parent.mkdir(parents=True, exist_ok=True)
            (root / source_path).write_text(json.dumps({
                "schema": "fixture.other.v1" if system_id == wrong_schema else "fixture.v1",
                "ready": ready,
                "count": 3,
            }), encoding="utf-8")
        systems.append({
            "id": system_id,
            "domain": system_id,
            "scope": "synthetic registry contract fixture",
            "evidence": [{"id": evidence_id, "path": source_path,
                          "schema": "fixture.v1"}],
            "facts": [{"id": "count", "evidence": evidence_id, "path": "count"}],
            "requirements": [{
                "id": "ready",
                "evidence": evidence_id,
                "path": "ready",
                "operator": "equals",
                "value": True,
                "reason": "fixture closure condition",
            }],
        })
    if missing_system:
        systems = [system for system in systems if system["id"] != missing_system]
    profile = root / "config/profile.json"
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(json.dumps({
        "schema": PROFILE_SCHEMA,
        "id": "fixture-whole-human",
        "systems": systems,
    }), encoding="utf-8")
    return profile


def test_profile_covers_every_required_human_system(tmp_path: Path) -> None:
    profile = _fixture_profile(tmp_path, missing_system="cardiac_electrical")
    with pytest.raises(HumanImportError, match="every required"):
        compile_registry(profile=profile, root=tmp_path)


def test_all_systems_must_close_for_whole_human_qualification(tmp_path: Path) -> None:
    profile = _fixture_profile(tmp_path)
    report = compile_registry(profile=profile, root=tmp_path)
    assert report["schema"] == SCHEMA
    assert report["status"] == "qualified"
    assert report["qualification"]["all_required_subsystems_qualified"]
    assert report["qualification"]["whole_human_capability"]
    assert report["counts"]["qualified_subsystems"] == len(SYSTEM_IDS)


def test_missing_evidence_fails_closed_and_names_the_open_gate(tmp_path: Path) -> None:
    profile = _fixture_profile(tmp_path, missing_source="bloodflow")
    report = compile_registry(profile=profile, root=tmp_path)
    blood = next(item for item in report["systems"] if item["id"] == "bloodflow")
    assert report["status"] == "partial"
    assert not report["qualification"]["whole_human_capability"]
    assert report["counts"]["missing_evidence_sources"] == 1
    assert blood["evidence"][0]["status"] == "missing"
    assert blood["open_requirements"] == ["ready"]


def test_schema_drift_fails_closed_without_discarding_source_hash(tmp_path: Path) -> None:
    profile = _fixture_profile(tmp_path, wrong_schema="cardiac_mechanics")
    report = compile_registry(profile=profile, root=tmp_path)
    heart = next(item for item in report["systems"]
                 if item["id"] == "cardiac_mechanics")
    assert report["status"] == "partial"
    assert report["counts"]["schema_mismatch_sources"] == 1
    assert heart["evidence"][0]["status"] == "schema_mismatch"
    assert heart["evidence"][0]["sha256"]
    assert heart["open_requirements"] == ["ready"]


def test_current_repository_registry_refuses_unqualified_subsystems() -> None:
    report = compile_registry()
    systems = {item["id"]: item for item in report["systems"]}
    assert report["status"] == "partial"
    assert report["counts"]["required_subsystems"] == len(SYSTEM_IDS)
    assert report["counts"]["qualified_subsystems"] < len(SYSTEM_IDS)
    assert not report["qualification"]["whole_human_capability"]
    assert systems["whole_body_anatomy"]["facts"][0]["value"] == 46
    assert systems["bloodflow"]["facts"][0]["value"] == 511
    assert systems["cardiac_mechanics"]["facts"][2]["value"] == 0
    assert not systems["cardiac_electrical"]["requirements"][0]["satisfied"]
    assert not systems["skin"]["requirements"][0]["satisfied"]

    anatomy = systems["whole_body_anatomy"]
    anatomy_facts = {item["id"]: item for item in anatomy["facts"]}
    anatomy_gates = {item["id"]: item for item in anatomy["requirements"]}
    assert next(item for item in anatomy["evidence"]
                if item["id"] == "external_ct_scan001")["status"] == "verified"
    assert anatomy_facts["external_ct_surface_candidates"]["value"] == 36
    assert anatomy_facts["external_ct_closed_meshes"]["value"] == 14
    assert anatomy_facts["external_ct_max_volume_error"]["value"] < 1e-9
    assert anatomy_facts["external_ct_subject_binding"]["value"] is False
    assert not anatomy_gates["external_scan_subject_binding"]["satisfied"]
    assert next(item for item in anatomy["evidence"]
                if item["id"] == "lower_limb_overlay")["status"] == "verified"
    assert anatomy_facts["lower_limb_pose_count"]["value"] == 8
    assert anatomy_facts["lower_limb_source_geometry_checks"]["value"] == 185
    assert anatomy_facts["literal_qpos0_patellar_status"]["value"] == (
        "posterior_or_intersecting_knee_anchor_plane")
    assert anatomy_facts["projected_patella_min_anterior_offset_m"]["value"] > 0.009
    assert anatomy_gates["lower_limb_multi_pose_source_audit"]["satisfied"]
    assert anatomy_gates["projected_patella_anteriority"]["satisfied"]

    organs = systems["internal_organs"]
    organ_facts = {item["id"]: item for item in organs["facts"]}
    organ_gates = {item["id"]: item for item in organs["requirements"]}
    assert organ_facts["external_aorta_label"]["value"] == "Aorta"
    assert organ_facts["external_vci_label"]["value"] == "VCI"
    assert not organ_gates["external_vessel_lumen_connectivity"]["satisfied"]

    muscle = systems["muscle"]
    muscle_facts = {item["id"]: item for item in muscle["facts"]}
    muscle_gates = {item["id"]: item for item in muscle["requirements"]}
    assert muscle_facts["external_ct_muscle_label"]["value"] == "Skeletal-muscle"
    assert muscle_facts["external_ct_individual_muscle_identity"]["value"] is False
    assert not muscle_gates["external_ct_individual_muscle_identity"]["satisfied"]

    tendon = systems["tendon"]
    tendon_facts = {item["id"]: item for item in tendon["facts"]}
    tendon_gates = {item["id"]: item for item in tendon["requirements"]}
    assert tendon_facts["paired_lower_limb_surface_coverage"]["value"] == pytest.approx(0.19951923076923078)
    assert tendon_facts["paired_lower_limb_point_fallbacks"]["value"] == 666
    assert tendon_facts["paired_lower_limb_native_transaction"]["value"] is False
    assert not tendon_gates["paired_tendon_surface_coverage"]["satisfied"]
    assert not tendon_gates["paired_tendon_native_transaction"]["satisfied"]

    bloodflow = systems["bloodflow"]
    bloodflow_gates = {item["id"]: item for item in bloodflow["requirements"]}
    assert not bloodflow_gates["external_ct_vascular_lumen_connectivity"]["satisfied"]
