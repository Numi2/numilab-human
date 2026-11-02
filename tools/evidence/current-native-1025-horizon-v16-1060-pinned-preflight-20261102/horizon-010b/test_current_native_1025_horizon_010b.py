from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent
OWNER = ROOT / "prepare_current_native_1025_horizon_010b.py"
spec = importlib.util.spec_from_file_location("horizon_010_test_owner", OWNER)
assert spec and spec.loader
owner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = owner
spec.loader.exec_module(owner)


@pytest.mark.parametrize("steps", [2, 16, 128, 1024, 1000000])
def test_capture_roster_is_small_sorted_and_has_initial_first_terminal(steps):
    roster = owner.capture_steps(steps)
    assert roster == sorted(set(roster))
    assert roster[0] == 0 and roster[1] == 1 and roster[-1] == steps
    assert len(roster) <= 8


def test_exact_progression_is_gated_and_cannot_skip_or_repeat():
    assert owner.next_horizon(2, 16) == 16
    assert owner.next_horizon(16, 128) == 128
    assert owner.next_horizon(128, 1024) == 1024
    for previous, requested in ((2, 128), (16, 1024), (128, 128), (1024, 2048)):
        with pytest.raises(ValueError, match="progression"):
            owner.next_horizon(previous, requested)


def test_duration_horizon_uses_measured_wall_rate_and_explicit_cap():
    assert owner.duration_horizon(1024, 4.0, 10.0) == 2560
    with pytest.raises(ValueError, match="1024-step"):
        owner.duration_horizon(128, 4.0, 10.0)
    with pytest.raises(ValueError, match="does not exceed"):
        owner.duration_horizon(1024, 4.0, 3.0)
    with pytest.raises(ValueError, match="above cap"):
        owner.duration_horizon(1024, 1.0, 2000.0, max_steps=1500)


def test_duration_horizon_rejects_nonfinite_or_nonpositive_measurements():
    for wall, target in ((0, 5), (-1, 5), (float("nan"), 5), (2, 0), (2, float("inf"))):
        with pytest.raises(ValueError):
            owner.duration_horizon(1024, wall, target)


def test_performance_marker_is_parsed_once_from_native_output():
    text = "other output\nresting_integrated_body=completed simulated_s=0.016 wall_s=3.25 real_time_factor=0.0049 physiology_body_clock=matched\n"
    assert owner.parse_performance(text) == {
        "simulated_s": 0.016, "wall_s": 3.25, "real_time_factor": 0.0049
    }
    assert owner.parse_performance("no performance marker") is None
    with pytest.raises(ValueError, match="duplicated"):
        owner.parse_performance(text + text)
    with pytest.raises(ValueError, match="finite and positive"):
        owner.parse_performance("resting_integrated_body=completed simulated_s=1 wall_s=nan real_time_factor=1")


def test_trace_pins_reject_changed_bytes(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text("accepted\n")
    import hashlib
    pin = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    owner._check_trace(str(path), pin)
    path.write_text("rejected\n")
    with pytest.raises(ValueError, match="changed"):
        owner._check_trace(str(path), pin)


def test_max_horizon_matches_native_bound():
    assert owner.capture_steps(owner.MAX_HORIZON_STEPS)[-1] == 1_000_000
    with pytest.raises(ValueError, match="1000000"):
        owner.capture_steps(1_000_001)


def test_orientation_binding_requires_live_pinned_source_only_signed_audit(tmp_path):
    import hashlib, json
    audit = tmp_path / "orientation.json"
    audit.write_text(json.dumps({
        "schema": "numi.h706.1022-role-surface-orientation-audit.v1",
        "status": "PASS_EXACT_1022_SIGNED_ROLE_CONVENTIONS",
        "physical_admission": False, "native_steps": 0,
    }))
    source = tmp_path / "orientation-owner.py"
    source.write_text("# pinned owner\n")
    audit_sha = hashlib.sha256(audit.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    audit_ref, owner_ref = owner.orientation_binding(audit, audit_sha, source, source_sha)
    report = {"geometry_coverage": {"paired_shell_input_coverage": {
        "native_role_orientation_contract": {
            "report_path": audit_ref["path"], "report_sha256": audit_ref["sha256"],
            "validator_path": owner_ref["path"], "validator_sha256": owner_ref["sha256"],
        }
    }}}
    owner.check_report_orientation(report, audit_ref, owner_ref)
    with pytest.raises(ValueError, match="fresh role-orientation audit/owner pin changed"):
        owner.orientation_binding(audit, "0" * 64, source, source_sha)
    report["geometry_coverage"]["paired_shell_input_coverage"]["native_role_orientation_contract"]["report_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="does not bind the selected fresh role-orientation"):
        owner.check_report_orientation(report, audit_ref, owner_ref)
    audit.write_text(json.dumps({
        "schema": "numi.h706.1022-role-surface-orientation-audit.v1",
        "status": "PASS_EXACT_1022_SIGNED_ROLE_CONVENTIONS",
        "physical_admission": True, "native_steps": 0,
    }))
    with pytest.raises(ValueError, match="overclaims physical/native"):
        owner.orientation_binding(audit, hashlib.sha256(audit.read_bytes()).hexdigest(), source, source_sha)


def test_runner_uses_unescaped_argument_expansion_and_fresh_verifier_flags():
    runner = ROOT / "run_current_native_1025_horizon_010b.sh"
    text = runner.read_text()
    assert "[[ $# -ne 15 ]]" in text
    assert 'ORIENTATION_AUDIT=$(resolve_file "${10}")' in text
    assert "STEPS=${SPEC#steps:}" in text
    assert "WALL=${SPEC#wall:}" in text
    assert "\\${10}" not in text and "\\${SPEC#" not in text
    assert '--orientation-audit "$ORIENTATION_AUDIT"' in text
    assert '--orientation-owner "$ORIENTATION_OWNER"' in text
