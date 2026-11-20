"""Compare numeric and timing evidence with the exact swept-clearance control."""
import csv
import json
from pathlib import Path
import re
import sys

root = Path.cwd()
reference = root.parent/"resting-clearance-and-atrium-013/glide-refresh-ccd-gap-trial"
candidate = root/sys.argv[1]
def trace(run,name):
    path = run/"native"/name
    with path.open() as stream:
        return list(csv.DictReader(stream))

def metrics(run):
    rows = trace(run,"resting-coupled.csv")
    gpu = [float(value) for value in re.findall(
        r"resting_native_profile step=\d+ command_ms=[0-9.]+ gpu_ms=([0-9.]+)",
        (run/"stdout.log").read_text())]
    execution = json.loads((run/"execution.json").read_text())
    simulated = float(rows[-1]["time_s"]) if rows else 0
    output = {
        "accepted_steps":len(rows),
        "simulated_seconds":simulated,
        "end_to_end_real_time_factor":simulated/execution["wall_seconds"],
        "gpu_ms_per_step":gpu, "gpu_total_ms":sum(gpu), "execution":execution,
    }
    if rows:
        output.update(
            pleural_range_pa=[min(float(x["pleural_pa"]) for x in rows),max(float(x["pleural_pa"]) for x in rows)],
            peak_bed_impulse_ns=max(float(x["normal_impulse_ns"]) for x in rows),
            peak_bed_penetration_m=max(float(x["peak_penetration_m"]) for x in rows),
            maximum_root_force_n=max(abs(float(x["root_assistance_n"])) for x in rows),
            maximum_root_torque_nm=max(abs(float(x["root_assistance_nm"])) for x in rows),
            peak_mass_residual_kg=max(abs(float(x["max_abs_body_mass_residual_kg"])) for x in rows),
        )
    if (run/"failure.json").exists():
        fail = json.loads((run/"failure.json").read_text())
        output["failure"] = {key: value for key,value in fail.items() if key.startswith("matter_")}
        output["Brain_accepted_time_us"] = fail["brain_input"]["source_time_us"]
        output["Brain_rejected_time_us"] = fail["brain_candidate"]["timestamp_us"]
    return rows,output

r,a = metrics(reference)
c,b = metrics(candidate)
matches = {}
for name in ("resting-coupled.csv", "resting-com-support-impulses.csv", "resting-com-momentum-diagnostic.csv"):
    p = reference/"native"/name
    q = candidate/"native"/name
    x,y = p.read_text().splitlines(),q.read_text().splitlines()
    common = min(len(x),len(y))
    matches[name] = {"both_row_counts": [len(x),len(y)], "shared_lines_identical":x[:common]==y[:common]}
shared = min(len(r),len(c),len(a["gpu_ms_per_step"]),len(b["gpu_ms_per_step"]))
f = {}
for into,got in zip(r,c):
    for key in into:
        difference = abs(float(into[key])-float(got[key]))
        if difference > f.get(key,0.0):
            f[key] = difference
report = {
    "reference":a, "candidate":b, "shared_profiled_accepted_steps":shared,
    "shared_trace_comparisons":matches,
    "differing_trace_field_max_abs":f,
    "paired_GPU_speed_ratio":sum(a["gpu_ms_per_step"][:shared])/sum(b["gpu_ms_per_step"][:shared]) if shared else None,
}
(root/f"{candidate.name}-comparison.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
