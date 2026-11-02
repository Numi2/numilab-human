from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("receipt_builder_015_tested", HERE / "build_current_fractional_mass_receipt_015.py")
assert spec and spec.loader
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)

class Core:
    RUN_SCHEMA = "numi.h706.current-fractional-mass-cli-run.v1"
    class ReceiptError(RuntimeError): pass
    def __init__(self, control): self.control, self.delegated = control, 0
    def read_json(self, path): return self.control
    def path_sha(self, ref, label): return Path(ref["path"]).resolve(), ref["sha256"]
    def validate_control_manifest(self, run, path): self.delegated += 1; return {"legacy": True}

class Flow:
    def __init__(self): self.nested, self.closure = 0, None
    def verify_nested_control_refs(self, control): self.nested += 1
    class Base:
        def __init__(self, owner): self.owner = owner
        def verify_source_closure(self, *args): self.owner.closure = args
    @property
    def BASE(self): return self.Base(self)

def bound_case(tmp_path):
    control_path, closure_path = tmp_path / "control.json", tmp_path / "closure.json"
    control_path.write_text("{}"); closure_path.write_text("{}")
    control = {"variant": owner.VARIANT_030D, "status": owner.STATUS_030D,
               "source": {"path": "/source.cpp", "sha256": "a" * 64},
               "binary": {"path": "/binary", "sha256": "b" * 64},
               "admission_adapter": {"path": "/frozen/006.py",
                                     "sha256": "249ebe4e548da2d4f1633c03dd994de313b1c591b47d82591fb8c5ea2a12609d"}}
    run = {"schema": Core.RUN_SCHEMA, "physical_admission": False, "native_steps": 0,
           "source_control_manifest": {"path": str(control_path), "sha256": hashlib.sha256(b"{}").hexdigest()},
           "source_closure": {"path": str(closure_path), "sha256": hashlib.sha256(b"{}").hexdigest()},
           "source": control["source"], "binary": control["binary"]}
    return run, control_path, control

def test_legacy_p029_delegates_unchanged(tmp_path):
    path = tmp_path / "control.json"; path.write_text("{}")
    core, flow = Core({"variant": "029-canonical-source-tet-sites"}), Flow()
    assert owner.validate_control_manifest_015({}, path, core=core, flow=flow) == {"legacy": True}
    assert core.delegated == 1 and flow.nested == 0

def test_030d_replays_bound_control_and_closure(tmp_path):
    run, path, control = bound_case(tmp_path)
    core, flow = Core(control), Flow()
    adapted = owner.validate_control_manifest_015(run, path, core=core, flow=flow)
    assert adapted["variant"] == control["variant"]
    assert adapted["admission_adapter"] == {
        "path": str(owner.RUN_CONTROL_ADAPTER_PATH),
        "sha256": owner.RUN_CONTROL_ADAPTER_SHA256,
    }
    assert control["admission_adapter"]["path"] == "/frozen/006.py"
    assert flow.nested == 1 and flow.closure is not None

def test_030d_rejects_failed_source_control(tmp_path):
    run, path, control = bound_case(tmp_path)
    control["status"] = "REJECTED"
    with pytest.raises(Exception, match="full-allocator NOT_RUN"):
        owner.validate_control_manifest_015(run, path, core=Core(control), flow=Flow())

def test_030d_rejects_source_binary_mismatch(tmp_path):
    run, path, control = bound_case(tmp_path)
    run["binary"] = {"path": "/other", "sha256": "c" * 64}
    with pytest.raises(Exception, match="source/binary"):
        owner.validate_control_manifest_015(run, path, core=Core(control), flow=Flow())

def test_030d_rejects_physical_claims(tmp_path):
    run, path, control = bound_case(tmp_path)
    run["native_steps"] = 1
    with pytest.raises(Exception, match="native/physical"):
        owner.validate_control_manifest_015(run, path, core=Core(control), flow=Flow())


def test_output_paths_are_fresh_distinct_and_projection_is_required(tmp_path):
    output, projection = tmp_path / "mass.json", tmp_path / "projection.json"
    assert owner.validate_output_paths(output, projection, projection_required=True) == (output, projection)
    with pytest.raises(ValueError, match="must differ"):
        owner.validate_output_paths(output, output, projection_required=True)
    with pytest.raises(ValueError, match="report path"):
        owner.validate_output_paths(output, None, projection_required=True)
    projection.write_text("{}")
    with pytest.raises(ValueError, match="overwrite existing source"):
        owner.validate_output_paths(output, projection, projection_required=True)


def test_runtime_parser_removes_only_the_two_control_hashes():
    values = {
        "NUMI_H706_FRACTIONAL_ENABLE": "1",
        "NUMI_H706_FRACTIONAL_OUTPUT_DIR": "/tmp/output",
        "NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256": "a" * 64,
        "NUMI_H706_FRACTIONAL_SOURCE_CONTROL_SHA256": "b" * 64,
    }
    env, args = owner.parse_runtime_015({}, lambda _run: (dict(values), {"--flag": ""}))
    assert env == {key: value for key, value in values.items()
                   if key not in owner.RUN_CONTROL_VARS}
    assert args == {"--flag": ""}


def test_runtime_parser_rejects_a_single_control_hash():
    values = {"NUMI_H706_FRACTIONAL_RUN_CONFIG_SHA256": "a" * 64}
    with pytest.raises(ValueError, match="both separate run-control hashes"):
        owner.parse_runtime_015({}, lambda _run: (dict(values), {}))
