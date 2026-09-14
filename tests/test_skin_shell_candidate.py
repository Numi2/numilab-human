from __future__ import annotations

import json

import pytest

from numilab_human.skin_shell_candidate import (
    SkinShellCandidateError,
    _immutable_write,
    compile_candidate,
)


def test_skin_shell_binds_source_geometry_without_promoting_mechanics() -> None:
    result = compile_candidate()

    assert result["status"] == "partial"
    assert result["source"]["bodyparts3d_member_id"] == "FJ2810"
    assert result["source"]["source_vertex_count"] == 102467
    assert result["source"]["outer_surface_vertex_count"] == 54949
    assert result["coverage"]["registered_body_influence_count"] == 86
    assert result["coverage"]["rest_pose_reconstruction_max_error_m"] < 2.0e-5
    assert result["qualification"]["source_skin_member_bound"]
    assert result["qualification"]["registered_visual_influences_bound"]
    assert not result["qualification"]["skin_physical_volume"]
    assert not result["qualification"]["skin_material_calibration"]
    assert not result["qualification"]["fat_geometry"]
    assert not result["ownership"]["skin_mechanical_mass_owner"]


def test_skin_shell_receipt_is_immutable(tmp_path) -> None:
    result = compile_candidate()
    output = tmp_path / "receipt.json"
    first = _immutable_write(output, result)
    assert first == _immutable_write(output, compile_candidate())
    with pytest.raises(SkinShellCandidateError, match="immutable"):
        _immutable_write(output, {**result, "status": "changed"})


def test_skin_shell_manifest_cannot_promote_physics(tmp_path) -> None:
    source = "Docs/media/skin-shell-candidate-20260914/bodyparts3d-myosim-skinned-shell.manifest.json"
    value = json.loads(open(source, encoding="utf-8").read())
    value["status"] = "physical_skin_owner"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(SkinShellCandidateError, match="promoted"):
        compile_candidate(manifest=manifest)
