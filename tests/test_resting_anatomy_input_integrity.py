"""A completed geometry pass must not certify inputs changed during the audit."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "resting_verification_integrity_tool",
    Path(__file__).resolve().parents[1] / "tools/verify_resting_anatomy.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

@pytest.mark.parametrize("changed", [None, "pack", "receipt", "source", "removed_source"])
def test_audit_rechecks_explicit_inputs_before_publishing_pass(tmp_path, monkeypatch, changed):
    paths = {name: tmp_path / name for name in ("pack", "receipt", "source")}
    for name, path in paths.items():
        path.write_bytes(("original-" + name).encode())
    digests = {name: hashlib.sha256(path.read_bytes()).hexdigest()
               for name, path in paths.items()}
    monkeypatch.setattr(cli.verify, "load_capture",
                        lambda *args: ({(51005, 71): object()}, {"fixture": True}))
    def audited(*args, **kwargs):
        if changed == "removed_source":
            paths["source"].unlink()
        elif changed is not None:
            paths[changed].write_bytes(b"changed-after-preflight")
        return {"passed": True, "rows": [{"stage": "self", "passed": True}]}
    monkeypatch.setattr(cli.verify, "audit_changed_surfaces", audited)
    output = tmp_path / "report.json"
    code = cli.main([
        "--pack", str(paths["pack"]), "--pack-sha256", digests["pack"],
        "--capture-receipt", str(paths["receipt"]), "--receipt-sha256", digests["receipt"],
        "--source-pin", str(paths["source"]) + "=" + digests["source"],
        "--changed", "51005:71", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["rows"] == [{"stage": "self", "passed": True}]
    assert report["anatomical_admission"] is False
    assert report["native_admission"] is False
    if changed is None:
        assert code == 0 and report["passed"] is True
        assert report["input_integrity_after_audit"]["passed"] is True
        assert report["input_integrity_after_audit"]["revalidated_input_count"] == 3
    else:
        assert code == 1 and report["passed"] is False
        assert report["status"] == "REJECTED_INPUT_CHANGED_DURING_AUDIT"
        assert report["failed_stage"] == "source_integrity"
        assert report["input_integrity_after_audit"]["passed"] is False
        assert report["input_integrity_after_audit"]["geometry_passed_before_recheck"] is True
