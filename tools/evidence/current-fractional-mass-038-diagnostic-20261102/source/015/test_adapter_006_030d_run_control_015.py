from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("adapter_006_030d_run_control_015_tested", HERE / "adapter_006_030d_run_control_015.py")
assert spec and spec.loader
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_case(tmp_path: Path):
    run_config = tmp_path / "run-config.json"
    source_control = tmp_path / "source-control.json"
    run_config.write_text('{"kind":"run-config"}\n')
    source_control.write_text('{"kind":"source-control"}\n')
    run_config_sha = file_sha(run_config)
    source_control_sha = file_sha(source_control)
    app_env = {key: f"value-{key}" for key in adapter._ORIGINAL.H706_FRACTIONAL_ENV_KEYS}
    app_env["ENABLE"] = "1"
    actual = dict(app_env)
    actual.update({
        "OUTPUT_DIR": str(tmp_path / "output"),
        "RUN_CONFIG_SHA256": run_config_sha,
        "SOURCE_CONTROL_SHA256": source_control_sha,
    })
    argv = [f"NUMI_H706_FRACTIONAL_{key}={value}" for key, value in sorted(actual.items())]
    manifest = {
        "run_config": {"path": str(run_config), "sha256": run_config_sha},
        "source_control_manifest": {"path": str(source_control), "sha256": source_control_sha},
        "argv": argv,
    }
    source = tmp_path / "run-manifest.json"
    source.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    cfg = {
        "h706_fractional_env": app_env,
        "h706_fractional_environment_source": {
            "path": str(source), "resolved_path": str(source.resolve()),
            "sha256": file_sha(source), "argv_sha256": adapter.base.json_digest(argv),
        },
        "h706_fractional_run_control_pins": {
            "run_config": {"path": str(run_config), "sha256": run_config_sha},
            "source_control_manifest": {"path": str(source_control), "sha256": source_control_sha},
        },
    }
    return cfg, source, argv


def test_full_actual_argv_and_separate_controls_pass(tmp_path):
    cfg, source, original = make_case(tmp_path)
    source_path, filtered = adapter.validate_run_control_bindings(cfg)
    assert source_path == source.resolve()
    assert filtered == [x for x in original if not any(
        x.startswith(adapter.PREFIX + key + "=") for key in adapter.CONTROL_VARIABLES)]
    assert len(filtered) == len(original) - 2


def test_run_control_hash_mismatch_rejected(tmp_path):
    cfg, _, _ = make_case(tmp_path)
    cfg["h706_fractional_run_control_pins"]["run_config"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from actual argv/manifest"):
        adapter.validate_run_control_bindings(cfg)


def test_argv_digest_mismatch_rejected(tmp_path):
    cfg, _, _ = make_case(tmp_path)
    cfg["h706_fractional_environment_source"]["argv_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="full argv digest"):
        adapter.validate_run_control_bindings(cfg)


def test_app_environment_mismatch_rejected(tmp_path):
    cfg, _, _ = make_case(tmp_path)
    cfg["h706_fractional_env"]["ENABLE"] = "0"
    with pytest.raises(ValueError, match="app env differs"):
        adapter.validate_run_control_bindings(cfg)
