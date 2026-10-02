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
