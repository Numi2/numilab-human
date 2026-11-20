"""Check that the sampled native root preserves the retained physical trace."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "resting-clearance-and-atrium-017/floor-native-live-parent-026"
run = Path(sys.argv[1]).resolve()
config = json.loads((run / "launch.json").read_text())
root = int(config["environment"]["NUMI_MATTER_GPU_TIMING_ROOT"])
count = int(config["argv"][config["argv"].index("--muscle-step-count") + 1])

def trace(directory):
    with (directory / "native/resting-coupled.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows

execution = json.loads((run / "execution.json").read_text())
assert execution["exit"] == 0
reference = trace(BASE)
trial = trace(run)
assert len(trial) == count and len(reference) >= count
mismatches = []
for baseline, profiled in zip(reference[:count], trial, strict=True):
    assert baseline.keys() == profiled.keys()
    fields = [key for key in baseline if baseline[key] != profiled[key]]
    if fields:
        mismatches.append(dict(step=profiled["step"], fields=fields))
stages = defaultdict(list)
err = (run / "stderr.log").read_text()
for name, step, duration in re.findall(r"^matter_gpu_stage=(\S+) step=(\d+) elapsed_ns=(\d+)$",
                                        err, re.MULTILINE):
    assert int(step) == root
    stages[name].append(int(duration))
stats = {}
for name, values in stages.items():
    stats[name] = dict(spans=len(values), sum_ms=sum(values)/1.e6,
                       max_span_ms=max(values)/1.e6, duration_ns=values)
throughput = [line for line in (run / "stdout.log").read_text().splitlines()
              if line.startswith("resting_integrated_throughput ")]
output = dict(exit=execution["exit"], wall_seconds=execution["wall_seconds"],
    accepted_rows=len(trial), fields_per_row=len(trial[0]), root_profiled=root,
    exact_physical_trace_match=not mismatches, mismatches=mismatches,
    pleural_pressure_pa_range=[min(float(row["pleural_pa"]) for row in trial),
                             max(float(row["pleural_pa"]) for row in trial)],
    stage_totals=stats, integrated_window=throughput,
    count_unavailable_markers=err.count("matter_gpu_timing_span_unavailable"),
    scope="Instrumented headless profile of one native root. Stage splits perturb encoding. Setup and presentation are outside stage spans.")
(run / "profile-summary.json").write_text(json.dumps(output, indent=2) + "\n")
assert not mismatches, mismatches
assert stats, "The device supplied no stage samples"
print(json.dumps({**output, "stage_totals":{k:v["sum_ms"] for k,v in stats.items()}}))
