"""Verify the existing staged replay prefix and measure its motion extension."""
import csv
import json
from pathlib import Path

reference = Path("staged-accuracy")
extension = Path("motion-extension")
names = (
    "resting-coupled.csv",
    "resting-com-support-impulses.csv",
    "resting-com-momentum-diagnostic.csv",
)

traces = {}
for name in names:
    before = (reference / "native" / name).read_bytes()
    after = (extension / "native" / name).read_bytes()
    prefix_length = min(len(before), len(after))
    traces[name] = {
        "reference_bytes": len(before),
        "extension_bytes": len(after),
        "prefix_bytes": prefix_length,
        "prefix_identical": before[:prefix_length] == after[:prefix_length],
    }

execution = json.loads((extension / "execution.json").read_text())
comparison = json.loads((extension / "comparison.json").read_text())
log = (extension / "stdout.log").read_text()
completed = comparison["accepted_steps"]
with (extension / "native/resting-coupled.csv").open(newline="") as stream:
    trace = list(csv.DictReader(stream))
assert len(trace) == len(completed)
launch = json.loads((extension / "launch.json").read_text())
argv = launch["argv"]
requested_steps = int(argv[argv.index("--muscle-step-count") + 1])
initialization = next(line for line in log.splitlines()
                      if line.startswith("resting_initialization=dynamic_release"))
result = {
    "requested_steps": requested_steps,
    "completed_steps": len(completed),
    "accepted_seconds": comparison["accepted_seconds"],
    "simulated_seconds_per_wall_second": comparison["end_to_end_rtf"],
    "execution": execution,
    "initialization": initialization,
    "pressure_pa": comparison["pressure_pa"],
    "peak_bed_penetration_m": comparison["peak_bed_penetration_m"],
    "peak_normal_support_impulse_ns": max((float(row["normal_impulse_ns"]) for row in trace), default=0),
    "max_root_assistance_n": comparison["max_root_assistance_n"],
    "terminal_audit": comparison["terminal_audit"],
    "terminal_error": comparison["terminal_error"],
    "retries_with_verified_rollback": comparison["verified_rollbacks"],
    "staged_attempts_after_prefix": [row for row in comparison["staged_attempts"] if row[0] > 18],
    "prefix_traces": traces,
    "native_surface_audit": [
        row for row in (extension / "native/resting-surface-audit.csv").read_text().splitlines()[1:]
        if row.strip()
    ],
}
assert all(traces[name]["prefix_identical"] for name in names)
if execution["exit"] == 0:
    assert completed == list(range(1, requested_steps + 1))
    assert comparison["terminal_audit"]
    assert all(t["prefix_bytes"] == t["reference_bytes"] for t in traces.values())
    assert result["max_root_assistance_n"] == 0
    assert result["pressure_pa"][0] < result["pressure_pa"][1]
(extension / "verified-extension.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
