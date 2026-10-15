"""Bad declaration inputs must fail before expensive capture or exact geometry work."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "resting_verification_preflight_tool",
    Path(__file__).resolve().parents[1] / "tools/verify_resting_anatomy.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.mark.parametrize("preflight", [False, True])
@pytest.mark.parametrize("second", [
    ("51005:71", "51005:77"),
    ("51005:77", "51005:71"),
])
def test_duplicate_partition_stops_before_capture_and_predicates(tmp_path, monkeypatch, preflight, second):
    pin = Mock(side_effect=AssertionError("must not read bytes"))
    load = Mock(side_effect=AssertionError("must not load capture"))
    audit = Mock(side_effect=AssertionError("must not audit geometry"))
    monkeypatch.setattr(cli, "pin", pin)
    monkeypatch.setattr(cli.verify, "load_capture", load)
    monkeypatch.setattr(cli.verify, "audit_changed_surfaces", audit)
    output = tmp_path / "report.json"
    args = ["--pack", "unused", "--pack-sha256", "0"*64,
            "--capture-receipt", "unused", "--receipt-sha256", "0"*64,
            "--changed", "51005:71", "--output", str(output),
            "--shared-partition", "51005:71", "51005:77", "first", "1"*64,
            "--shared-partition", *second, "different-source", "2"*64]
    if preflight:
        args += ["--preflight-only"]
    assert cli.main(args) == 1
    result = json.loads(output.read_text())
    assert result["failed_stage"] == "preparation"
    assert result["error"] == "duplicate shared partition declaration"
    pin.assert_not_called()
    load.assert_not_called()
    audit.assert_not_called()


def test_partition_keeps_source_order_but_rejects_self_pair():
    expected = ((51005, 77), (51005, 71), Path("source"), "1"*64)
    assert cli.shared_partition_declarations(
        [("51005:77", "51005:71", "source", "1"*64)]) == [expected]
    with pytest.raises(ValueError, match="two distinct"):
        cli.shared_partition_declarations(
            [("51005:71", "51005:71", "source", "1"*64)])
