from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.body_composition_integration import (
    _immutable_write,
    compile_candidate,
)
from numilab_human.model import ImportError
from numilab_human.physiology import canonical


def test_cross_domain_candidate_binds_source_layers_without_promoting_owners() -> None:
    result = compile_candidate()

    assert result["status"] == "partial"
    assert result["source_member_layers"] == {
        "organ_surface_candidates": 378,
        "regional_blood_transport": 329,
        "muscle_tendon_surface_identity": 150,
    }
    assert result["runtime_evidence"]["clock_nanoseconds"] == 12500
    assert result["runtime_evidence"]["blood_transport_accepted_steps"] == 511
    assert result["runtime_evidence"]["blood_transport_rejected_steps"] == 1
    assert result["qualification"]["cross_domain_owner_nonduplication_checked"]
    assert not result["qualification"]["integrated_human_qualification"]
    assert result["identity_bindings"]["blood_members_subset_of_organ_members"]
    assert result["identity_bindings"]["surface_ids_disjoint_from_organ_members"]
    assert result["identity_bindings"]["transport_and_exchange_beds_share_clock"]
    assert all(value == 0 for value in result["ownership"].values())


def test_candidate_receipt_is_deterministic_and_immutable(tmp_path: Path) -> None:
    result = compile_candidate()
    output = tmp_path / "receipt.json"
    first = _immutable_write(output, result)
    second = _immutable_write(output, compile_candidate())
    assert first == second
    with pytest.raises(ImportError, match="immutable"):
        _immutable_write(output, {**result, "status": "changed"})


def test_activation_route_count_drift_is_rejected(tmp_path: Path) -> None:
    source = Path("Docs/media/activation-recruitment-candidate-20260914/receipt-v1.json")
    value = json.loads(source.read_text(encoding="utf-8"))
    value["counts"]["source_routes"] = 415
    path = tmp_path / "activation.json"
    path.write_bytes(canonical(value) + b"\n")
    with pytest.raises(ImportError, match="activation route counts changed"):
        compile_candidate(activation=path)
