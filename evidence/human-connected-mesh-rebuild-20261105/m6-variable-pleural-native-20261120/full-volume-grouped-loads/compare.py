"""Compare the grouped and dense contact dispatch on the same accepted roots."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


def read_run(directory):
    native = directory / "native" if (directory / "native").is_dir() else directory
    with (native / "resting-coupled.csv").open(newline="") as stream:
        trace = list(csv.DictReader(stream))
    report, = native.glob("registered-volume-fractional-mass-*.json")
    stdout = (directory / "stdout.log").read_text()
    stderr = (directory / "stderr.log").read_text()
    runtime = json.loads((directory / "execution.json").read_text())
    evidence = json.loads((directory / "launch.json").read_text())
    spans = {int(root): float(ms) for root, ms in re.findall(
        r"^resting_native_profile step=(\d+) command_ms=[\d.]+ gpu_ms=([\d.]+)", stdout, re.M)}
    work = {row["root"] + 1: row for row in (json.loads(line.split("=", 1)[1]) for line
        in stderr.splitlines() if line.startswith("matter_fgmres_active_work="))}
    return trace, report, stdout, stderr, runtime, evidence, spans, work


def compare(control, candidate):
    a, mass_a, log_a, _, r_a, e_a, gpu_a, work_a = read_run(control)
    b, mass_b, log_b, stderr_b, r_b, e_b, gpu_b, work_b = read_run(candidate)
    if str(Path(e_b["environment"]["DYLD_LIBRARY_PATH"]) / "libmetalrobo.dylib") not in stderr_b:
        raise ValueError("The new host runtime was not observed by dyld")
    if not a or [row["step"] for row in a] != [str(i) for i in range(1, len(a) + 1)]:
        raise ValueError("Control trace does not contain contiguous accepted roots")
    if [row["step"] for row in b] != [str(i) for i in range(1, len(b) + 1)]:
        raise ValueError("Candidate trace has missing or repeated roots")
    if [r["time_s"] for r in b] != [r["time_s"] for r in a[:len(b)]]:
        raise ValueError("Candidate trace has different accepted physical times")
    for key in ("NUMI_HUMAN_FEM_TRIAL_BED_PLANE", "NUMI_HUMAN_CONTACT_ROW_CACHE",
                "NM_MIXED_CERTIFICATE_TILED", "NM_FGMRES_INEXACT_NEWTON_MAX_ETA"):
        if e_a["environment"][key] != e_b["environment"][key]:
            raise ValueError(f"Different physical or solver setting: {key}")
    with mass_a.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    if mass_a.read_bytes() != mass_b.read_bytes():
        raise ValueError("Candidate changed installed mass allocation")

    used_roots = list(range(1, len(b) + 1))
    profiled = all(root in gpu_a and root in gpu_b for root in used_roots)
    if profiled and used_roots:
        profiles = [{"root": root, "control_gpu_ms":gpu_a[root], "grouped_gpu_ms":gpu_b[root],
                     "control_fgmres_columns":work_a.get(root, {}).get("used_columns_sum"),
                     "grouped_fgmres_columns":work_b.get(root, {}).get("used_columns_sum")}
                    for root in used_roots]
        speedup = sum(gpu_a[root] for root in used_roots) / sum(gpu_b[root] for root in used_roots)
    else:
        profiles, speedup = [], None

    certifications = {}
    for tag in ("resting_fem_skin_bed_accepted_root_reaction",
                "resting_fascia_fem_terminal_audit=passed"):
        certifications[tag] = {"control": next((line for line in log_a.splitlines()
                                               if line.startswith(tag + " ")), None),
                               "grouped": next((line for line in log_b.splitlines()
                                               if line.startswith(tag + " ")), None)}
    differing_fields = sorted({key for x, y in zip(a, b) for key in x if x[key] != y[key]})
    trace_time = float(b[-1]["time_s"]) if b else 0.0
    if not math.isfinite(trace_time): raise ValueError("Nonfinite accepted time")
    return {
        "controlled_variable": "Grouped-force host and Metal dispatch overlay",
        "control_execution": r_a, "candidate_execution": r_b,
        "accepted_roots": len(b),
        "accepted_simulated_s": trace_time,
        "full_range_pass": len(b) == len(a) and r_a["exit"] == r_b["exit"] == 0,
        "identical_trace_columns": not differing_fields,
        "differing_trace_columns": differing_fields,
        "identical_installed_mass_sha256": sha,
        "by_root": profiles,
        "matched_gpu_speedup": speedup,
        "candidate_inclusive_realtime_factor": trace_time/r_b["wall_seconds"],
        "terminal_audits": certifications,
        "scope": "Single unpresented matched release trial. No sustained performance or geometry qualification."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.control, args.candidate), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
