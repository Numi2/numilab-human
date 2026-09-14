import json
from pathlib import Path

import pytest

from numilab_human.muscle_volume_owner_candidate import (
    SCHEMA,
    MuscleVolumeError,
    compile_candidate,
)


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "Docs/media/muscle-surface-geometry-audit-20260914/receipt-v1.json"


def test_binds_only_single_closed_muscle_geometry() -> None:
    result = compile_candidate(audit=AUDIT)
    assert result["schema"] == SCHEMA
    assert result["coverage"]["source_muscle_surface_count"] == 148
    assert result["coverage"]["closed_muscle_component_count"] == 60
    assert result["coverage"]["closed_multi_component_count"] == 6
    assert result["coverage"]["topology_defective_muscle_component_count"] == 82
    assert result["geometry"]["closed_muscle_volume_total_m3"] == pytest.approx(
        0.006471304532959317
    )
    assert all(row["owner_id"].startswith("muscle-volume:bodyparts3d:") for row in result["owners"])
    assert all(not row["physical_volume_owner"] for row in result["owners"])
    assert result["qualification"]["single_closed_geometry_volume_bound"]
    assert not result["qualification"]["skeletal_muscle_tissue_mass_owner"]


def test_rejects_promoted_upstream_owner(tmp_path: Path) -> None:
    document = json.loads(AUDIT.read_text())
    document["surfaces"][0]["physical_volume_owner"] = "fem:muscle:1"
    mutated = tmp_path / "audit.json"
    mutated.write_text(json.dumps(document))
    with pytest.raises(MuscleVolumeError, match="already owns a physical field"):
        compile_candidate(audit=mutated)


def test_rejects_volume_total_drift(tmp_path: Path) -> None:
    document = json.loads(AUDIT.read_text())
    document["geometry"]["algebraic_volume_total_m3"] = 1.0
    mutated = tmp_path / "audit.json"
    mutated.write_text(json.dumps(document))
    with pytest.raises(MuscleVolumeError, match="do not reproduce"):
        compile_candidate(audit=mutated)
