from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.native_whole_body_audit import INPUT_SCHEMA, audit


TERMS = (
    "gravity",
    "active_muscle",
    "passive_muscle",
    "tendon",
    "ligament_limit",
    "contact",
    "joint_constraint",
    "damping",
)


def _payload(*, complete_terms: bool = True, internal_residual: bool = False) -> dict:
    rows = []
    for index in range(128):
        values = {term: 0.0 for term in TERMS}
        if index == 2:
            values["gravity"] = -100.0
            values["contact"] = 100.0
        if internal_residual and index == 42:
            values["passive_muscle"] = 10.0
        rows.append({
            "dof_index": index,
            "dof_name": f"dof_{index}",
            **values,
            "net_residual": sum(values.values()),
        })
    return {
        "schema": INPUT_SCHEMA,
        "source": {"commit": "fixture", "device": "fixture", "body_count": 157},
        "equilibrium": {
            "dof_count": 128,
            "q_sha256": "a" * 64,
            "rows": rows,
        },
        "terms_available": {term: complete_terms for term in TERMS},
        "contact_active_set": {"selected": True},
        "fibre_tendon_equilibrium": {"solved": True},
    }


def _run(tmp_path: Path, payload: dict) -> dict:
    source = tmp_path / "native-audit.json"
    output = tmp_path / "receipt.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    arguments = argparse.Namespace(
        input=source,
        output=output,
        maximum_normalized_residual=1.0e-3,
        top=12,
    )
    assert audit(arguments) == 0
    return json.loads(output.read_text(encoding="utf-8"))


def test_root_balance_does_not_hide_internal_residual(tmp_path: Path) -> None:
    receipt = _run(tmp_path, _payload(internal_residual=True))
    assert receipt["status"] == "partial"
    assert receipt["residual"]["root_max_absolute_residual"] == 0.0
    assert receipt["residual"]["internal_max_absolute_residual"] == 10.0
    assert receipt["equilibrium"]["worst_coordinates"][0]["index"] == 42
    assert not receipt["qualification"]["whole_body_generalized_equilibrium"]
    assert not receipt["qualification"]["static_contact_handoff_admissible"]


def test_complete_terms_and_residual_are_only_static_handoff_admission(tmp_path: Path) -> None:
    receipt = _run(tmp_path, _payload())
    assert receipt["status"] == "passed"
    assert receipt["qualification"]["complete_force_terms"]
    assert receipt["qualification"]["whole_body_generalized_equilibrium"]
    assert receipt["qualification"]["static_contact_handoff_admissible"]
    assert not receipt["qualification"]["sustained_standing"]
    assert not receipt["qualification"]["walking"]


def test_unavailable_tendon_or_contact_term_fails_closed(tmp_path: Path) -> None:
    receipt = _run(tmp_path, _payload(complete_terms=False))
    assert receipt["status"] == "partial"
    assert not receipt["qualification"]["complete_force_terms"]
    assert not receipt["qualification"]["whole_body_generalized_equilibrium"]
    assert "tendon" in receipt["gate"]["reasons"][0]


def test_malformed_native_audit_is_rejected(tmp_path: Path) -> None:
    payload = _payload()
    del payload["equilibrium"]["rows"][7]["damping"]
    source = tmp_path / "native-audit.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "receipt.json"
    arguments = argparse.Namespace(
        input=source,
        output=output,
        maximum_normalized_residual=1.0e-3,
        top=12,
    )
    with pytest.raises(ImportError):
        audit(arguments)
