from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "Docs/media/native-force-audit-20260915"


def test_current_native_force_audit_binds_all_dofs_without_promoting_equilibrium() -> None:
    receipt = json.loads((EVIDENCE / "receipt-v1.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == "numi.human.native-force-audit-current-requalification.v1"
    assert receipt["status"] == "partial"
    assert receipt["source"]["commit"] == "ef0fc708db0f4de1a07fca426e5a415f62e9da27"
    assert receipt["results"]["dof_count"] == 128
    assert receipt["results"]["coordinate_identity_rows"] == 128
    assert receipt["results"]["passive_coordinate_couplings"] == 40
    assert receipt["results"]["maximum_assembly_error"] <= 1.0e-12
    assert receipt["results"]["maximum_closure_ratio"] == 1.0
    assert receipt["results"]["internal_normalized_residual_rms"] == 0.141417024366
    assert receipt["qualification"]["full_128_dof_component_rows"]
    assert receipt["qualification"]["authoritative_net_reconstructed"]
    assert receipt["qualification"]["per_dof_source_audit"]
    assert receipt["qualification"]["source_coordinate_identity_bound"]
    assert not receipt["qualification"]["internal_generalized_balance"]
    assert not receipt["qualification"]["force_convergence"]
    assert not receipt["qualification"]["sustained_standing"]
    assert not receipt["qualification"]["blood_mass_transfer"]


def test_current_native_force_audit_artifact_hashes_are_bound() -> None:
    receipt = json.loads((EVIDENCE / "receipt-v1.json").read_text(encoding="utf-8"))
    for key in ("native_stdout", "native_stderr", "coordinate_map", "force_snapshot", "force_ledger"):
        artifact = receipt["artifacts"][key]
        expected = receipt["artifacts"][f"{key}_sha256"]
        assert hashlib.sha256((EVIDENCE / artifact).read_bytes()).hexdigest() == expected
    snapshot = json.loads((EVIDENCE / "force-snapshot-passive.json").read_text(encoding="utf-8"))
    assert snapshot["metadata"]["coordinate_map"]["anatomical_names"] is False
    assert snapshot["coordinate_names"][51] == "v_051:joint_63:child_body_64"
    assert len(receipt["results"]["worst_coordinates"]) == 16
