import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "Docs/media/native-coupled-velocity-closure-diagnostic-20260915"


def test_coupled_velocity_closure_diagnostic_keeps_expected_delta_explicit() -> None:
    receipt = json.loads((EVIDENCE / "receipt-v1.json").read_text())
    assert receipt["schema"] == "numi.human.native-coupled-velocity-closure-diagnostic.v1"
    assert receipt["status"] == "partial"
    assert receipt["qualification"]["experimental_candidate"]
    assert receipt["qualification"]["host_verifier_false_failure_fixed"]
    assert receipt["qualification"]["native_trace_endpoint"] == "bitwise"
    assert receipt["qualification"]["force_convergence"] is False
    assert receipt["qualification"]["standing"] is False
    assert receipt["qualification"]["walking"] is False
    assert receipt["results"]["maximum_equality_velocity_error_mps_or_rad_s"] > 0.0
    assert receipt["results"]["persistent_max_acceleration_mps2"] < 1.0
    assert receipt["results"]["persistent_max_penetration_m"] == 0.0


def test_coupled_velocity_closure_artifacts_are_hash_bound() -> None:
    receipt = json.loads((EVIDENCE / "receipt-v1.json").read_text())
    for key, name_key in (
        ("native_source_patch_sha256", "native_source_patch"),
        ("native_stderr_sha256", "native_stderr"),
        ("native_stdout_sha256", "native_stdout"),
    ):
        name = receipt["artifacts"][name_key]
        digest = hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
        assert digest == receipt["artifacts"][key]


def test_dynamic_force_audit_snapshot_and_ledger_keep_internal_gap_visible() -> None:
    snapshot = json.loads((EVIDENCE / "dynamic-force-snapshot-v1.json").read_text())
    ledger = json.loads((EVIDENCE / "dynamic-force-ledger-v1.json").read_text())
    assert snapshot["schema"] == "numi.human.generalized-force-snapshot.v1"
    assert snapshot["metadata"]["native_log"]["record"] == "persistent_dynamic_force_audit"
    assert snapshot["metadata"]["native_log"]["sha256"] == hashlib.sha256(
        (EVIDENCE / "native.stdout.txt").read_bytes()
    ).hexdigest()
    assert ledger["schema"] == "numi.human.generalized-force-ledger.v1"
    assert ledger["status"] == "partial"
    assert ledger["coverage"]["per_dof_source_rows"] == 128
    assert ledger["qualification"]["per_dof_source_audit"] is True
    assert ledger["qualification"]["force_convergence"] is False
    assert ledger["residual"]["maximum_assembly_error"] < 1.0e-12
    assert ledger["residual"]["maximum_closure_ratio"] > 0.001


def test_v2_receipt_binds_dynamic_force_audit_artifacts() -> None:
    receipt = json.loads((EVIDENCE / "receipt-v2.json").read_text())
    assert receipt["schema"] == "numi.human.native-coupled-velocity-closure-diagnostic.v1"
    assert receipt["qualification"]["per_dof_dynamic_force_audit"] is True
    for digest_key, path_key in (
        ("dynamic_force_snapshot_sha256", "dynamic_force_snapshot"),
        ("dynamic_force_ledger_sha256", "dynamic_force_ledger"),
    ):
        path = EVIDENCE / receipt["artifacts"][path_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["artifacts"][digest_key]


def test_long_horizon_timeout_is_preserved_as_failure() -> None:
    timeout = json.loads((EVIDENCE / "long-horizon-timeout-v1.json").read_text())
    assert timeout["schema"] == "numi.human.native-coupled-velocity-closure-timeout.v1"
    assert timeout["status"] == "failed_timeout"
    assert timeout["qualification"]["long_horizon_timeout_preserved"]
    assert timeout["qualification"]["native_result_published"] is False
    for key, name_key in (("native_stdout_sha256", "native_stdout"),
                          ("native_stderr_sha256", "native_stderr")):
        name = timeout["artifacts"][name_key]
        digest = hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
        assert digest == timeout["artifacts"][key]


def test_segmented_horizon_candidate_is_compiled_but_unrequalified() -> None:
    candidate = json.loads((EVIDENCE / "segmented-horizon-candidate-v1.json").read_text())
    assert candidate["schema"] == "numi.human.native-segmented-horizon-candidate.v1"
    assert candidate["status"] == "compiled_unrequalified"
    assert candidate["qualification"]["source_compiled"]
    assert candidate["qualification"]["runtime_requalification"] is False
    assert candidate["candidate"]["maximum_segment_steps"] == 64
    patch = EVIDENCE / candidate["artifacts"]["source_patch"]
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == candidate["artifacts"]["source_patch_sha256"]
