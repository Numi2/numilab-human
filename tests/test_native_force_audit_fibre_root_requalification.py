from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "Docs/media/native-force-audit-fibre-root-20260915"


def test_fibre_root_force_audit_binds_all_rows_without_promoting_dynamic_claims() -> None:
    receipt = json.loads((EVIDENCE / "receipt-v1.json").read_text(encoding="utf-8"))
    assert receipt["schema"] == "numi.human.native-force-audit-fibre-root-requalification.v1"
    assert receipt["status"] == "partial"
    assert receipt["source"]["commit"] == "f45fcfdc80a05c2226d638227295add7f789c55c"
    assert receipt["source"]["binary_sha256"] == "62648144c20f386b25f625d503fd101f5c846b9ca8b0dac0649e4a064fb04919"
    assert receipt["results"]["coordinate_identity_rows"] == 128
    assert receipt["results"]["maximum_assembly_error"] <= 1.0e-12
    assert receipt["results"]["maximum_absolute_force_residual"] <= 1.0e-3
    assert receipt["results"]["static_active_support_contacts"] == 6
    assert receipt["qualification"]["internal_generalized_balance"]
    assert receipt["qualification"]["per_dof_source_audit"]
    assert not receipt["qualification"]["force_convergence"]
    assert not receipt["qualification"]["anatomical_support_loading"]
    assert not receipt["qualification"]["activation_calibration"]
    assert not receipt["qualification"]["blood_mass_transfer"]
    assert not receipt["qualification"]["material_calibration"]
    assert not receipt["qualification"]["subject_calibration"]


def test_fibre_root_force_audit_artifacts_and_component_closure_are_bound() -> None:
    receipt = json.loads((EVIDENCE / "receipt-v1.json").read_text(encoding="utf-8"))
    for key in ("coordinate_map", "force_ledger", "force_snapshot", "native_stdout", "native_stderr"):
        filename = receipt["artifacts"][key]
        expected = receipt["artifacts"][key + "_sha256"]
        assert hashlib.sha256((EVIDENCE / filename).read_bytes()).hexdigest() == expected
    snapshot = json.loads((EVIDENCE / "force-snapshot.json").read_text(encoding="utf-8"))
    ledger = json.loads((EVIDENCE / "force-ledger.json").read_text(encoding="utf-8"))
    assert snapshot["nv"] == 128
    assert len(snapshot["components"]) == 6
    assert all(len(component["values"]) == 128 for component in snapshot["components"])
    assert len(ledger["per_dof_audit"]) == 128
    assert ledger["qualification"]["generalized_force_closed"]
    assert ledger["residual"]["maximum_assembly_error"] <= 1.0e-12
