from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.muscle_surface_geometry_audit import (
    _immutable_write,
    compile_audit,
)


@pytest.fixture(scope="module")
def audit() -> dict:
    return compile_audit()


def test_hash_locked_muscle_geometry_audit_keeps_volume_boundary_explicit(audit: dict) -> None:
    result = audit
    assert result["status"] == "partial"
    assert result["counts"] == {
        "source_surface_count": 150,
        "muscle_surface_count": 148,
        "tendon_surface_count": 2,
        "topology_recomputed_surface_count": 150,
        "single_closed_component_count": 60,
        "closed_multi_component_count": 6,
        "topology_defective_count": 84,
        "surface_volume_candidate_count": 60,
        "physical_volume_owner_count": 0,
        "mechanical_mass_owner_count": 0,
        "material_owner_count": 0,
        "volumetric_active_force_owner_count": 0,
        "total_quotient_vertex_count": 316420,
        "total_quotient_triangle_count": 631464,
        "total_duplicate_face_count": 484,
        "total_vertex_manifold_defect_count": 1452,
    }
    assert result["geometry"]["surface_area_total_m2"] == pytest.approx(3.2682904751011153)
    assert result["geometry"]["algebraic_volume_total_m3"] == pytest.approx(0.006471304532959317)
    assert result["qualification"]["source_member_hashes_bound"]
    assert result["qualification"]["single_closed_surface_volume_candidates_recomputed"]
    assert not result["qualification"]["physical_tissue_volume_owner"]
    assert not result["qualification"]["skeletal_muscle_tissue_mass_owner"]
    assert not result["qualification"]["activation_force_transfer"]
    assert all(row["physical_volume_owner"] is None for row in result["surfaces"])


def test_geometry_audit_rejects_source_member_hash_drift(tmp_path: Path) -> None:
    manifest = Path(
        "Docs/media/numi-human-toe-enthesis-v5-2048/manifests/"
        "bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
    )
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["source"]["surfaces"][0]["member_sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n")
    with pytest.raises(ImportError, match="surface receipt does not bind the surface manifest hash"):
        compile_audit(surface_manifest=path)


def test_geometry_audit_receipt_is_immutable(tmp_path: Path, audit: dict) -> None:
    result = audit
    output = tmp_path / "receipt.json"
    first = _immutable_write(output, result)
    assert first == _immutable_write(output, audit)
    changed = dict(result)
    changed["status"] = "changed"
    with pytest.raises(ImportError, match="immutable"):
        _immutable_write(output, changed)
