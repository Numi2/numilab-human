"""Compare accepted mechanics and GPU cost for the paired short solver sample."""
import csv
import json
from pathlib import Path
import re

root = Path.cwd()
control = root/"glide-refresh-ccd-gap-trial"
sample = root/"glide-refresh-solver-work-sample"
def read_trace(run):
    with (run/"native/resting-coupled.csv").open() as stream:
        return list(csv.DictReader(stream))[:2]
def gpu_ms(run):
    pat = r"resting_native_profile step=[12] command_ms=[0-9.]+ gpu_ms=([0-9.]+)"
    return [float(v) for v in re.findall(pat,(run/"stdout.log").read_text())]

baseline, candidate = read_trace(control), read_trace(sample)
assert len(baseline) == len(candidate) == 2
diffs = {key: max(abs(float(a[key])-float(b[key])) for a,b in zip(baseline,candidate))
         for key in baseline[0]}
a, b = gpu_ms(control), gpu_ms(sample)
report = {
    "scope": "Two accepted 2 ms steps of the unchanged horizon, stopped intentionally before the third step completed",
    "control_step_gpu_ms": a,
    "candidate_step_gpu_ms": b,
    "control_total_gpu_ms": sum(a),
    "candidate_total_gpu_ms": sum(b),
    "short_gpu_time_drop_fraction": 1.-sum(b)/sum(a),
    "matched_rows_byte_identity": baseline==candidate,
    "nonzero_trace_field_differences": {k:v for k,v in diffs.items() if v != 0.0},
    "candidate_execution": json.loads((sample/"execution.json").read_text()),
}
(root/"solver-work-sample-comparison.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
