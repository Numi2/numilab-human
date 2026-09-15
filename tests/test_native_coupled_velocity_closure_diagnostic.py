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
