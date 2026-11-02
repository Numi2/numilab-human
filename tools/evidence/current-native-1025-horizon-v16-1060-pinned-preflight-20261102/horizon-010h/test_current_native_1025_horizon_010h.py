from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent
OWNER = ROOT / "prepare_current_native_1025_horizon_010h.py"
spec = importlib.util.spec_from_file_location("horizon_010h_test_owner", OWNER)
assert spec and spec.loader
owner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = owner
spec.loader.exec_module(owner)


@pytest.mark.parametrize("steps", [2, 16, 128, 1024, 1000000])
def test_sparse_capture_roster_samples_initial_midpoint_and_terminal(steps):
    roster = owner.capture_steps(steps, "sampled-three")
    assert roster == sorted(set(roster))
    assert roster == sorted({0, steps // 2, steps})
    assert roster[0] == 0 and roster[-1] == steps
    assert len(roster) <= 3


@pytest.mark.parametrize("steps", [2, 16, 128, 1024, 1000000])
def test_legacy_quartile_profile_preserves_existing_roster(steps):
    roster = owner.capture_steps(steps, "quartiles-six")
    assert roster == sorted(set(roster))
    assert roster[0] == 0 and roster[1] == 1 and roster[-1] == steps
    assert len(roster) <= 6


def test_sparse_capture_reduces_pack_copies_without_changing_accepted_frames():
    horizon = 1024
    sparse = owner.capture_steps(horizon, "sampled-three")
    sampled = owner.capture_steps(horizon, "quartiles-six")
    assert 1 + len(sparse) == 4
    assert 1 + len(sampled) == 7
    assert len(sparse) < len(sampled)
    source = Path(owner.__file__).read_text()
    assert '"NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS": ",".join(map(str, caps))' in source
    assert "capture_profile" in source
    assert '\"full_fem_endpoint_state_dump_requested\": False' in source


def test_native_physics_and_capture_mode_are_separated_from_frame_selection():
    root = Path(__file__).resolve().parent.parent
    old = (root / "current-native-1025-direct-o-010g-horizon-v17c" / "prepare_current_native_1025_horizon_010g.py").read_text()
    new = Path(owner.__file__).read_text()
    start_text = "    argv, physics, stale = base.safe_scene(cfg, static, route, install, steps)"
    end_text = "    route.validate_full_scene_mode(argv[1:], env)"
    def launch_context(source):
        start = source.index(start_text)
        end = source.index(end_text, start)
        block = source[start:end]
        capture_line = "        \"NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS\": \",\".join(map(str, caps)),\n"
        return block.replace(capture_line, "")
    assert launch_context(old) == launch_context(new)
    assert '\"NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS\": ",".join(map(str, caps))' in new
    assert '\"NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE\": \"0\"' in new


def test_capture_profile_validator_rejects_missing_or_misaligned_frames():
    expected = [0, 8, 16]
    assert owner.validate_capture_roster(16, "sampled-three", expected, expected,
        {str(x): {} for x in expected}, expected) == expected
    with pytest.raises(ValueError, match="motion/stage"):
        owner.validate_capture_roster(16, "sampled-three", [0, 8, 16], [0, 16],
            {"0": {}, "8": {}, "16": {}}, expected)
    with pytest.raises(ValueError, match="observer/verifier"):
        owner.validate_capture_roster(16, "sampled-three", expected, expected,
            {"0": {}, "16": {}}, expected)
    with pytest.raises(ValueError, match="observer/verifier"):
        owner.validate_capture_roster(16, "sampled-three", expected, expected,
            {"0": {}, "8": {}, "16": {}}, [0, 16])
    with pytest.raises(ValueError, match="invalid step key"):
        owner.validate_capture_roster(16, "sampled-three", expected, expected,
            {"initial": {}, "8": {}, "16": {}}, expected)


def test_capture_profile_rejects_unrecognized_values():
    with pytest.raises(ValueError, match="capture profile"):
        owner.capture_steps(16, "endpoints-only-unverified")


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
    runner = ROOT / "run_current_native_1025_horizon_010h.sh"
    text = runner.read_text()
    assert "[[ $# -ne 15 ]]" in text
    assert "NUMI_HUMAN_CAPTURE_PROFILE:-sampled-three" in text
    assert '--capture-profile "$CAPTURE_PROFILE"' in text
    assert 'ORIENTATION_AUDIT=$(resolve_file "${10}")' in text
    assert "STEPS=${SPEC#steps:}" in text
    assert "WALL=${SPEC#wall:}" in text
    assert "\\${10}" not in text and "\\${SPEC#" not in text
    assert '--orientation-audit "$ORIENTATION_AUDIT"' in text
    assert '--orientation-owner "$ORIENTATION_OWNER"' in text


def test_v17c_geometry_report_schema_is_required():
    owner.validate_v17c_geometry_report({
        "schema": "numi.human.current-motion-geometry-verification.v11", "status": "PASS"
    })
    with pytest.raises(ValueError, match="pinned v17c schema"):
        owner.validate_v17c_geometry_report({
            "schema": "numi.human.current-motion-geometry-verification.v9", "status": "PASS"
        })
    with pytest.raises(ValueError, match="pinned v17c schema"):
        owner.validate_v17c_geometry_report({
            "schema": "numi.human.current-motion-geometry-verification.v11", "status": "FAIL"
        })


def test_horizon_runner_points_to_its_own_versioned_prepare_owner():
    text=(ROOT/"run_current_native_1025_horizon_010h.sh").read_text()
    assert "PREP=/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-native-1025-launch/current-native-1025-direct-o-010h-sparse-capture-v17c/prepare_current_native_1025_horizon_010h.py" in text
    assert Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-native-1025-launch/current-native-1025-direct-o-010h-sparse-capture-v17c/prepare_current_native_1025_horizon_010h.py").is_file()
