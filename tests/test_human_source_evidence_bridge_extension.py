import json
from pathlib import Path

import pytest

from numilab_human.human_source_evidence_bridge_extension import (
    BASE_BRIDGE,
    PROFILE,
    ROUTE_PARTITION,
    ROUTE_VOLUME,
    SCHEMA,
    BridgeExtensionError,
    _immutable_write,
    compile_candidate,
)
from numilab_human.physiology import canonical


def test_binds_route_mass_partition_to_immutable_bridge_without_owner_promotion() -> None:
    result = compile_candidate(base_bridge=BASE_BRIDGE, route_partition=ROUTE_PARTITION,
                               route_volume=ROUTE_VOLUME, profile=PROFILE)
    assert result["schema"] == SCHEMA
    assert result["base_bridge"]["physical_owner_count"] == 0
    assert result["domains"]["muscle"]["source_route_count"] == 416
    assert result["domains"]["muscle"]["closed_surface_count"] == 60
    assert result["domains"]["muscle"]["route_incidence_count"] == 82
    assert result["domains"]["muscle"]["routes_without_surface_binding"] == 238
    assert result["domains"]["muscle"]["candidate_mass_kg"] == pytest.approx(6.859582804936875)
    assert result["qualification"]["route_mass_partition_bound"]
    assert result["qualification"]["candidate_mass_and_volume_close"]
    assert not result["qualification"]["mechanical_mass_owner"]
    assert not result["qualification"]["activation_force_transfer"]
    assert not result["qualification"]["integrated_human_qualification"]


def test_rejects_route_partition_source_hash_divergence(tmp_path: Path) -> None:
    partition = json.loads(ROUTE_PARTITION.read_text())
    partition["source"]["route_volume_receipt_sha256"] = "0" * 64
    path = tmp_path / "partition.json"
    path.write_bytes(canonical(partition) + b"\n")
    with pytest.raises(BridgeExtensionError, match="route-mass source hashes diverge"):
        compile_candidate(route_partition=path)


def test_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_candidate()
    output = tmp_path / "receipt.json"
    first = _immutable_write(output, result)
    assert first == _immutable_write(output, compile_candidate())
