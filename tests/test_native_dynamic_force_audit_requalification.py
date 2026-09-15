import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "Docs/media/native-dynamic-force-audit-20260915"


def test_dynamic_force_audit_binds_all_dofs_and_keeps_behavior_open() -> None:
    receipt = json.loads((EVIDENCE / "receipt-v1.json").read_text())
    audit = json.loads((EVIDENCE / "dynamic-force-audit.json").read_text())
    assert receipt["schema"] == "numi.human.native-dynamic-force-audit-requalification.v1"
    assert receipt["status"] == "partial"
    assert receipt["qualification"]["full_128_dof_component_rows"]
    assert receipt["qualification"]["dynamic_component_reconstruction"]
    assert receipt["qualification"]["dynamic_release"] is False
    assert receipt["qualification"]["force_convergence"] is False
    assert receipt["qualification"]["sustained_standing"] is False
    assert receipt["qualification"]["walking"] is False
    assert len(audit["rows"]) == 128
    assert audit["rows"][0]["dof"] == 118
    assert audit["maximum_abs_residual_n"] == receipt["results"]["dynamic_initial_max_abs_residual_n"]
    assert receipt["results"]["persistent_max_acceleration_mps2"] < 1.0
    assert receipt["results"]["source_dynamic_force_parity_max_delta_n"] < 0.05


def test_rejected_handoff_experiment_is_preserved_and_not_promoted() -> None:
    experiment = json.loads((EVIDENCE / "rejected-handoff-experiment.json").read_text())
    assert experiment["status"] == "rejected"
    assert experiment["promoted"] is False
    baseline = experiment["baseline_repaired_state"]["max_acceleration_512_mps2"]
    patched = next(row["max_acceleration_mps2"] for row in experiment["patched_state"] if row["steps"] == 512)
    assert patched > baseline


def test_dynamic_force_audit_artifact_manifest_is_bound() -> None:
    manifest = {}
    for line in (EVIDENCE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        manifest[name] = digest
    for name, digest in manifest.items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == digest
