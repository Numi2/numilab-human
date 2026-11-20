import csv
import hashlib
import json
from pathlib import Path
import re

root = Path.cwd()
trial = root / "glide-refresh-ccd-gap-trial"
control = root / "glide-refresh-trial"

def summarize(run):
    with (run/"native/resting-coupled.csv").open() as stream:
        steps = list(csv.DictReader(stream))
    assert steps
    log = (run/"stdout.log").read_text()
    rows = [dict(step=int(step), command_ms=float(command), gpu_ms=float(gpu))
            for step,command,gpu in re.findall(
            r"resting_native_profile step=(\d+) command_ms=([\d.]+) gpu_ms=([\d.]+)",log)]
    run_time = json.loads((run/"execution.json").read_text())
    accepted_step = int(steps[-1]["step"])
    accepted_end = re.search(
        rf"human_execution_stage=authoritative_segment_end wall_elapsed_ms=([\d.]+) stage_step={accepted_step}\b",log)
    horizon = re.search(r"human_execution_stage=native_horizon_begin wall_elapsed_ms=([\d.]+)",log)
    native_span = (float(accepted_end[1])-float(horizon[1]))/1000 if horizon and accepted_end else None
    wall = run_time["wall_seconds"]
    simulated = float(steps[-1]["time_s"])
    # In the existing trace, support is normal impulse divided by 2 ms.
    pressure = [float(s["pleural_pa"]) for s in steps]
    summary = {
        "accepted_steps":int(steps[-1]["step"]),
        "accepted_seconds":simulated,"exit":run_time["exit"],"process_seconds":wall,
        "process_real_time_factor":simulated/wall,
        "accepted_native_span_seconds":native_span,
        "accepted_native_span_real_time_factor":simulated/native_span if native_span else None,
        "pleural_pressure_pa_min_max":[min(pressure),max(pressure)],
        "peak_bed_normal_impulse_ns":max(float(s["normal_impulse_ns"]) for s in steps),
        "max_bed_penetration_m":max(float(s["peak_penetration_m"]) for s in steps),
        "max_root_force_n":max(abs(float(s["root_assistance_n"])) for s in steps),
        "max_root_torque_nm":max(abs(float(s["root_assistance_nm"])) for s in steps),
        "max_abs_body_mass_residual_kg":max(abs(float(s["max_abs_body_mass_residual_kg"])) for s in steps),
        "per_accepted_step_gpu":rows,
    }
    if (run/"failure.json").exists():
        f=json.loads((run/"failure.json").read_text())
        summary["failure"] = {k:v for k,v in f.items() if k.startswith("matter_")}
        summary["accepted_brain_us"] = f["brain_input"]["source_time_us"]
        summary["rejected_brain_us"] = f["brain_candidate"]["timestamp_us"]
    return steps,summary


control_rows, before = summarize(control)
candidate_rows, after = summarize(trial)
fields = ("time_s","pleural_pa","normal_impulse_ns","peak_penetration_m","root_assistance_n",
          "root_assistance_nm","max_abs_body_mass_residual_kg")
summary = {
    "device": "Apple M6 on SSH numi",
    "change": "existing numerical normal-gap globalization uses swept collision thickness",
    "control": before,
    "collision_gap": after,
    "common_accepted_rows": min(len(control_rows), len(candidate_rows)),
    "matched_control_prefix": control_rows == candidate_rows[:len(control_rows)],
    "last_candidate_row": {f: float(candidate_rows[-1][f]) for f in fields},
    "scope": "Short native mechanics-only motion with 26 fascia and variable pleural; no fine-volume install, whole-scene crossings, or duration qualification",
}
(root/"glide-ccd-gap-comparison.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps(summary, indent=2))
