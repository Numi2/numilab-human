"""Bounded joint checks retain completed witnesses without declaring success."""
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import time

import pytest

spec = importlib.util.spec_from_file_location(
    "resting_verification_budget_tool",
    Path(__file__).resolve().parents[1] / "tools/verify_resting_anatomy.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def arguments(tmp_path):
    pack, receipt, output = [tmp_path / name for name in ("pack", "receipt", "result.json")]
    pack.write_bytes(b"pack")
    receipt.write_bytes(b"receipt")
    return [
        "--pack", str(pack), "--pack-sha256", hashlib.sha256(pack.read_bytes()).hexdigest(),
        "--capture-receipt", str(receipt),
        "--receipt-sha256", hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "--changed", "51005:71", "--output", str(output)], output


def test_timeout_preserves_every_completed_row_and_never_admits(tmp_path, monkeypatch):
    args, output = arguments(tmp_path)
    rows = [{"stage": "self", "identity": [51005, 71], "passed": True},
            {"stage": "nearby_pair", "identities": [[51005, 71], [51004, 5]],
             "passed": False, "unexpected_triangle_pairs": [[17, 31]]}]
    monkeypatch.setattr(cli.verify, "load_capture",
                        lambda *args: ({(51005, 71): object()}, {}))
    def interrupted(*args, **kwargs):
        for row in rows:
            kwargs["progress"](row)
        raise cli.GeometryBudgetExceeded("fixture geometry deadline")
    monkeypatch.setattr(cli.verify, "audit_changed_surfaces", interrupted)
    assert cli.main(args + ["--collect-failures", "--maximum-seconds", "1"]) == 1
    report = json.loads(output.read_text())
    assert report["passed"] is False
    assert report["status"] == "REJECTED_GEOMETRY_BUDGET_EXHAUSTED"
    assert report["failed_stage"] == "geometry_budget"
    assert report["collection_complete"] is False
    assert report["anatomical_admission"] is False
    assert report["native_admission"] is False
    assert report["input_integrity_after_audit"]["passed"] is True
    ledger = output.with_suffix(".gates.jsonl")
    assert [json.loads(line) for line in ledger.read_text().splitlines()] == rows
    assert report["completed_gate_log"]["completed_row_count"] == 2
    assert report["completed_gate_log"]["sha256"] == hashlib.sha256(ledger.read_bytes()).hexdigest()
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_real_geometry_deadline_restores_signal_owner():
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(cli.GeometryBudgetExceeded):
        with cli.geometry_budget(0.005):
            time.sleep(0.05)
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL)[0] == 0


def test_does_not_replace_another_owners_timer(monkeypatch):
    monkeypatch.setattr(signal, "getitimer", lambda *_: (2., 0.))
    with pytest.raises(ValueError, match="another owner"):
        with cli.geometry_budget(1):
            pytest.fail("nested geometry must not start")


@pytest.mark.parametrize("budget", ["0", "-1", "nan", "inf"])
def test_invalid_budgets_fail_before_loading_geometry(tmp_path, monkeypatch, budget):
    args, _ = arguments(tmp_path)
    monkeypatch.setattr(cli.verify, "load_capture", lambda *args: pytest.fail("no geometry"))
    with pytest.raises(SystemExit):
        cli.main(args + ["--maximum-seconds", budget])


def test_never_overwrites_a_retained_gate_log(tmp_path, monkeypatch):
    args, output = arguments(tmp_path)
    ledger = output.with_suffix(".gates.jsonl")
    ledger.write_bytes(b"retained-witness\n")
    monkeypatch.setattr(cli.verify, "load_capture", lambda *args: pytest.fail("no geometry"))
    with pytest.raises(SystemExit):
        cli.main(args)
    assert ledger.read_bytes() == b"retained-witness\n"

def test_deadline_during_checkpoint_leaves_complete_valid_json_row(tmp_path, monkeypatch):
    args, output = arguments(tmp_path)
    row = {"stage": "self", "identity": [51005, 71], "passed": True}
    monkeypatch.setattr(cli.verify, "load_capture",
                        lambda *args: ({(51005, 71): object()}, {}))
    dumps = cli.json.dumps
    def deadline_during_encoding(value, *args, **kwargs):
        if value is row:
            signal.raise_signal(signal.SIGALRM)
        return dumps(value, *args, **kwargs)
    monkeypatch.setattr(cli.json, "dumps", deadline_during_encoding)
    def interrupted(*args, **kwargs):
        kwargs["progress"](row)
        pytest.fail("deadline must be delivered after checkpoint")
    monkeypatch.setattr(cli.verify, "audit_changed_surfaces", interrupted)
    assert cli.main(args + ["--maximum-seconds", "1"]) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "REJECTED_GEOMETRY_BUDGET_EXHAUSTED"
    assert report["completed_gate_log"]["completed_row_count"] == 1
    assert output.with_suffix(".gates.jsonl").read_text() == dumps(row, sort_keys=True) + "\n"
