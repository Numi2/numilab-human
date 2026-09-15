import json
from pathlib import Path

import pytest

from numilab_human.body_composition_extension_join import (
    BASE,
    FOOT_PROXY,
    MUSCLE_JOIN,
    PROFILE,
    SCHEMA,
    ExtensionJoinError,
    _immutable_write,
    compile_candidate,
)


def test_binds_new_handoffs_to_v14_source_graph_without_owner_promotion() -> None:
    result = compile_candidate(base=BASE, foot_proxy=FOOT_PROXY,
                               muscle_join=MUSCLE_JOIN, profile=PROFILE)
    assert result["schema"] == SCHEMA
    assert result["base_receipt"]["physical_owner_count"] == 0
    assert result["extensions"]["foot_contact_proxy"]["proxy_count"] == 30
    assert result["extensions"]["muscle_route_volume"]["source_route_count"] == 416
    assert result["extensions"]["muscle_route_volume"]["routes_with_closed_geometry"] == 78
    assert result["qualification"]["cross_extension_source_hashes_closed"]
    assert not result["qualification"]["anatomical_supports_loading"]
    assert not result["qualification"]["skeletal_muscle_tissue_mass"]
    assert not result["qualification"]["integrated_human_qualification"]


def test_rejects_base_owner_promotion(tmp_path: Path) -> None:
    base = json.loads(BASE.read_text())
    base["ownership"]["organ_physical_volume_owner_count"] = 1
    mutated = tmp_path / "base.json"
    mutated.write_text(json.dumps(base))
    with pytest.raises(ExtensionJoinError, match="base composition contains a physical owner"):
        compile_candidate(base=mutated, foot_proxy=FOOT_PROXY,
                          muscle_join=MUSCLE_JOIN, profile=PROFILE)


def test_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_candidate()
    output = tmp_path / "receipt.json"
    first = _immutable_write(output, result)
    assert first == _immutable_write(output, compile_candidate())
