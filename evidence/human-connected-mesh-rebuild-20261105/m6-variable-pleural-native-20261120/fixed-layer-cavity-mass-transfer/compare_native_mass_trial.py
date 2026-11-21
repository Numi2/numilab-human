"""Compare the fixed-cavity run with the same instrumented unconditioned scene."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

control, trial = map(Path, sys.argv[1:])

def records(root):
    stdout = (root / "stdout.log").read_text()
    stderr = (root / "stderr.log").read_text()
    trace = root / "resting-coupled.csv"
    rows = []
    if trace.exists():
        with trace.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    timings = [float(ms) for ms in re.findall(
        r"^resting_native_profile step=\d+ .*? gpu_ms=(\S+)", stdout, re.M)]
    stages = defaultdict(int)
    for name, root_id, ns in re.findall(
            r"^matter_gpu_stage=(\w+) step=(\d+) elapsed_ns=(\d+)", stderr, re.M):
        stages[(int(root_id), name)] += int(ns)
    preparation = [line for line in stdout.splitlines() if line.startswith((
        "registered_volume_fixed_cavities=", "whole_body_skin_fat_registered_volume_install=",
        "resting_tissue_mass_partition=", "resting_initialization=",
        "resting_vascular_external_pressure=", "resting_fascia_fem_terminal_audit=",
        "resting_fem_skin_bed_accepted_root_reaction "))]
    failures = [line for line in stderr.splitlines() if line.startswith((
        "skin_fat_cavity_transport_failures=", "myosim_articulated_visual=failed",
        "matter_gpu_timing_invalid_records", "matter_gpu_stage_incomplete"))]
    active_work_prefix = "matter_fgmres_active_work="
    active_work = [json.loads(line[len(active_work_prefix):]) for line in stderr.splitlines()
                   if line.startswith(active_work_prefix)]
    mass_files = {}
    for file in root.glob("registered-volume-fractional-mass-*.json"):
        with file.open("rb") as stream:
            mass_files[file.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    execution = json.loads((root / "execution.json").read_text())
    return dict(execution=execution, accepted_roots=len(rows), rows=rows,
                total_simulated_seconds=float(rows[-1]["time_s"]) if rows else 0.0,
                gpu_root_ms=timings, gpu_stages=[
                    dict(root=i, name=name, milliseconds=ns / 1e6)
                    for (i, name), ns in sorted(stages.items())],
                preparation=preparation, failures=failures, source_mass_receipts=mass_files,
                active_work=active_work)

a, b = records(control), records(trial)
a_rows, b_rows = a.pop("rows"), b.pop("rows")
if b_rows:
    assert len(a_rows) >= len(b_rows)
    assert [row["time_s"] for row in a_rows[:len(b_rows)]] == [row["time_s"] for row in b_rows]
    assert a["source_mass_receipts"] == b["source_mass_receipts"]
delta = {}
if b_rows:
    for name in b_rows[0]:
        error = max(abs(float(one[name]) - float(two[name])) for one, two in zip(a_rows, b_rows))
        if error:
            delta[name] = error
seconds = b["total_simulated_seconds"]
print(json.dumps(dict(control=a, trial=b,
    nonidentical_trace_columns_max_absolute_delta=delta,
    measured_launch_realtime_factor=seconds / b["execution"]["wall_seconds"],
    integrated_motion_qualification=False), indent=2))
