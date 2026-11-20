"""Verify that scoped diagnostics leave the accepted replay and rejection unchanged."""
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "resting-clearance-and-atrium-015/motion-extension"
TRACE_NAMES = (
    "resting-coupled.csv",
    "resting-com-support-impulses.csv",
    "resting-com-momentum-diagnostic.csv",
)
MINIMA = "fem_deformable_limit_minimum="
SELECTED = "fem_deformable_limit_trace="


def extract(log, prefix):
    return [json.loads(line.removeprefix(prefix)) for line in log.splitlines()
            if line.startswith(prefix)]


def main():
    run = HERE / sys.argv[1]
    execution = json.loads((run / "execution.json").read_text())
    assert execution["exit"] == 1
    for name in TRACE_NAMES:
        assert (run / "native" / name).read_bytes() == (REFERENCE / "native" / name).read_bytes(), name
    assert (run / "failure.json").read_bytes() == (REFERENCE / "failure.json").read_bytes()
    failure = json.loads((run / "failure.json").read_text())
    stdout = (run / "stdout.log").read_text()
    assert "human_staged_solver_rollback step=22 tier=1 rollback=byte_identical" in stdout
    with (run / "native" / TRACE_NAMES[0]).open(newline="") as stream:
        accepted = list(csv.DictReader(stream))
    assert len(accepted) == 21
    log = (run / "stderr.log").read_text()
    minima = extract(log, MINIMA)
    selected = [row for row in extract(log, SELECTED) if row["contacts"]]
    assert all(row["root"] == 21 and row["microtick"] == 191 for row in minima + selected)
    result = {
        "accepted_trace_byte_identity": list(TRACE_NAMES),
        "failure_receipt_byte_identity": True,
        "tier_one_rollback": "byte_identical",
        "completed_steps": len(accepted),
        "accepted_seconds": 0.042,
        "wall_seconds": execution["wall_seconds"],
        "end_to_end_headless_rtf": 0.042 / execution["wall_seconds"],
        "failed_root": failure["segment_start"],
        "successful_microsteps_in_failed_root": failure["matter_microsteps"],
        "rejection_diagnostics": failure["matter_diagnostics"],
    }
    if minima:
        assert not selected
        assert all(row["tiled"] and row["reason"] == 8 for row in minima)
        assert all(row["prior_bound"] == 1 for row in minima)
        result["reported_contact_bounds"] = minima
        result["reported_bounds_under_0_05"] = sum(row["bound"] < 0.05 for row in minima)
        result["prediction"] = "Only one reported restriction below 0.05 in the 24-pass retry"
    else:
        assert selected
        for row in selected:
            assert row["iteration"] == 1
            assert all(contact["contact_slot"] == 410 for contact in row["contacts"])
        result["selected_feature"] = selected
    (run / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: val for key, val in result.items() if key != "selected_feature"}, indent=2))


if __name__ == "__main__":
    main()
