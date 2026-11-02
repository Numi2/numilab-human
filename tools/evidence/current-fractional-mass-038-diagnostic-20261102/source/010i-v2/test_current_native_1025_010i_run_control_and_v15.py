from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
OWNER = ROOT / "prepare_current_native_1025_direct_o_011.py"
spec = importlib.util.spec_from_file_location("native_direct_o_011_test_owner", OWNER)
assert spec and spec.loader
owner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = owner
spec.loader.exec_module(owner)

MATERIALIZER = ROOT / "materialize_current_native_1025_038_identity_011.py"
materializer_spec = importlib.util.spec_from_file_location("native_identity_011_test_materializer", MATERIALIZER)
assert materializer_spec and materializer_spec.loader
materializer = importlib.util.module_from_spec(materializer_spec)
sys.modules[materializer_spec.name] = materializer
materializer_spec.loader.exec_module(materializer)

def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def _source(tmp_path: Path, *, mutate: str | None = None):
    run_config = tmp_path / "run-config.json"
    source_control = tmp_path / "source-control.json"
    run_config.write_text('{"run":1}\n')
    source_control.write_text('{"control":1}\n')
    run_sha, control_sha = _sha(run_config), _sha(source_control)
    argv = [
        "NUMI_H706_FRACTIONAL_ENABLE=1",
        "NUMI_H706_FRACTIONAL_OUTPUT_DIR=" + str((tmp_path / "output").resolve()),
        "NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256=" + run_sha,
        "NUMI_H706_FRACTIONAL_SOURCE_CONTROL_SHA256=" + control_sha,
    ]
    if mutate == "wrong_run":
        argv[2] = "NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256=" + "0" * 64
    elif mutate == "partial":
        argv.pop(3)
    elif mutate == "unknown":
        argv.append("NUMI_H706_FRACTIONAL_SURPRISE=1")
    elif mutate == "duplicate":
        argv.append(argv[3])
    return {
        "argv": argv,
        "output_dir": str((tmp_path / "output").resolve()),
        "run_config": {"path": str(run_config.resolve()), "sha256": run_sha, "bytes": run_config.stat().st_size},
        "source_control_manifest": {"path": str(source_control.resolve()), "sha256": control_sha},
    }

def test_exact_run_control_hash_pair_is_validated_and_not_forwarded(tmp_path):
    expected = {"ENABLE": "1"}
    app, cli = owner.verify_h706_argv_environment(_source(tmp_path), expected)
    assert app == expected
    assert cli["OUTPUT_DIR"].endswith("/output")

@pytest.mark.parametrize("case", ["wrong_run", "partial", "unknown", "duplicate"])
def test_unrecognized_or_mismatched_run_control_metadata_fails_closed(tmp_path, case):
    with pytest.raises(ValueError):
        owner.verify_h706_argv_environment(_source(tmp_path, mutate=case), {"ENABLE": "1"})

def test_changed_run_control_file_is_rejected(tmp_path):
    doc = _source(tmp_path)
    (tmp_path / "run-config.json").write_text('{"run":2}\n')
    with pytest.raises(ValueError, match="run_config pin changed"):
        owner.verify_h706_argv_environment(doc, {"ENABLE": "1"})

def test_current_038_manifest_has_only_expected_app_env_differences():
    base = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
    cfg_path = base / "current-native-1025-launch/current-native-1025-direct-o-010i-envelope-bound-v1/final-controls-004/current-native-launch-010-build014b.template.json"
    run_path = base / "current-fractional-semantic-native-1047/postpass-030d-canonical-038/out-006/run-manifest.json"
    template = json.loads(cfg_path.read_text())
    run = json.loads(run_path.read_text())
    env = materializer.env_from_manifest(run, template["h706_fractional_env"])
    assert env["CARRIERS_SKIN_NPZ"].endswith("skin-adaptive-source-quadrature-032d.npz")
    assert env["CARRIERS_FAT_NPZ"].endswith("fat-adaptive-source-quadrature-032d.npz")
    assert env["CARRIER_WITNESS_SKIN_CSV"].endswith("skin-witnesses-currentR-sparse-032dplus-v15.csv")
    assert env["CARRIER_WITNESS_FAT_CSV"].endswith("fat-witnesses-currentR-sparse-032dplus-v15.csv")

def test_current_038_identity_materialization_preserves_geometry_and_mass_gate(tmp_path):
    base = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
    template_path = base / "current-native-1025-launch/current-native-1025-direct-o-010i-envelope-bound-v1/final-controls-004/current-native-launch-010-build014b.template.json"
    run_path = base / "current-fractional-semantic-native-1047/postpass-030d-canonical-038/out-006/run-manifest.json"
    preflight_path = run_path.with_name("preflight.json")
    cfg, receipt = materializer.materialize(template_path, run_path, preflight_path, tmp_path / "final.json", tmp_path / "future-native-root")
    assert cfg["mass_admission"] is None
    assert "recorded_flow_bundle" not in cfg
    assert cfg["geometry_admission"] == json.loads(template_path.read_text())["geometry_admission"]
    assert set(cfg["h706_fractional_auxiliary_path_pins"]) == {"projection_report"}
    assert "source_projection_015" not in cfg["h706_fractional_auxiliary_path_pins"]
    assert cfg["h706_fractional_env"]["CARRIERS_SKIN_SHA256"] == "7f648262440539761f5cc6f14c9a2a7c5bf73c5cf75b310ec747872d83c0efb3"
    assert cfg["h706_fractional_env"]["CARRIERS_FAT_SHA256"] == "12068b0a15bcc24dd8adae0d452ad6bf49b443a8a4b78712093aaa26713126a8"
    assert set(cfg["h706_fractional_run_control_pins"]) == {"run_config", "source_control_manifest"}
    assert cfg["h706_fractional_run_control_pins"]["run_config"]["sha256"] == "0d10ba16cef08cc18dc006c6aa0757e428041f7e857a2ecbaec2dd05b41aac26"
    assert cfg["h706_fractional_run_control_pins"]["source_control_manifest"]["sha256"] == "3ab84015892b4937f4920101d18a0429a616da0b6086f606faaa0792496d20d0"
    assert cfg["h706_fractional_auxiliary_path_pins"]["projection_report"]["sha256"] == "772bd092fae235a366de9683c584e1a6672e33f2bb07886bd4b4b511fbf418c4"
    assert receipt["status"] == "PASS_SOURCE_ONLY_IDENTITY_READY_MASS_PENDING"
    assert receipt["physical_admission"] is False and receipt["native_steps"] == 0
