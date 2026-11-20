"""Compare certified trajectories and timings with the existing native FEM gates."""
from argparse import ArgumentParser
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "next-stage-profile"

def summarize(trial, *, require_domain_gate=True):
    status = json.loads((trial / "execution.json").read_text())
    launch = json.loads((trial / "launch.json").read_text())
    rows = []
    metrics = {}
    stages = []
    with (trial / "stderr.log").open() as stream:
        for line in stream:
            if line.startswith("matter_solver_domain_norms="):
                rows.append(json.loads(line.partition("=")[2]))
            elif line.startswith("matter_fgmres_active_work="):
                work = json.loads(line.partition("=")[2])
                for key in ("used_columns_sum", "budget_exhausted_solves",
                            "cycles_with_work", "encoded_cycles", "active_newton_begins"):
                    metrics[key] = metrics.get(key, 0) + work[key]
            elif line.startswith("matter_gpu_"):
                stages.append(line.rstrip())
            elif line.startswith("myosim_articulated_visual=failed"):
                status["rejection"] = line.rstrip()
    status["roots_certified"] = len(rows)
    status["work"] = metrics
    status["gpu_timing"] = stages
    status["instrumented_root"] = launch.get("instrumented_root")
    status["lean_observers"] = launch.get("lean_observers", False)
    if status["exit"] != 0:
        return status
    declared_steps = int(launch["argv"][launch["argv"].index("--muscle-step-count") + 1])
    assert [row["root"] for row in rows] == list(range(declared_steps))
    status["maximum_mechanical_residual_over_limit"] = max(
        r["nonvascular_norm"] / r["mechanical_threshold"] for r in rows)
    status["maximum_vascular_residual_over_limit"] = max(
        r["vascular_norm"] / r["vascular_threshold"] for r in rows)
    status["domain_target_misses"] = [
        r for r in rows if r["nonvascular_norm"] > r["mechanical_threshold"]
        or r["vascular_norm"] > r["vascular_threshold"]
    ]
    if require_domain_gate:
        assert not status["domain_target_misses"], status["domain_target_misses"]
    stdout = trial / "stdout.log"
    steps = 0
    min_j = float("inf")
    with stdout.open() as stream:
        for line in stream:
            if line.startswith("resting_integrated_body=completed "):
                status["simulated_seconds"], status["body_wall_seconds"], status["realtime_factor"] = (
                    float(v) for v in re.search(r"simulated_s=(\S+) wall_s=(\S+) real_time_factor=(\S+)", line).groups())
                assert "physiology_body_clock=matched root_assistance=false" in line
            elif line.startswith("resting_FEM_tet,object=BodyParts3D whole-body skin-fat bonded shell,"):
                steps += 1
                min_j = min(min_j, float(re.search(r",J=([^,]+)", line)[1]))
    assert steps == 28632 and min_j > 0
    status["terminal_skin_fat_minimum_J"] = min_j
    with (trial / "native/resting-coupled.csv").open() as stream:
        trace = list(csv.DictReader(stream))
    assert len(trace) == declared_steps
    assert all(float(row["root_assistance_n"]) == 0 for row in trace)
    status["pleural_pressure_pa_endpoints"] = [float(trace[i]["pleural_pa"]) for i in (0, -1)]
    return status

def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    control = summarize(args.control)
    candidate = summarize(args.candidate)
    report = {"control": control, "candidate": candidate}
    if not candidate["exit"]:
        assert not control["exit"] and control["simulated_seconds"] == candidate["simulated_seconds"]
        report["body_time_reduction"] = 1 - candidate["body_wall_seconds"] / control["body_wall_seconds"]
        for filename in ("initial-skin-fem.csv", "skin-bone-witness.csv", "native/resting-coupled.csv"):
            key = filename.replace("/", "_").replace(".", "_")
            report[f"same_{key}"] = sha256(args.control / filename) == sha256(args.candidate / filename)
        with TemporaryDirectory(dir=HERE) as scratch:
            endpoints = []
            for trial, label in ((args.control, "control"), (args.candidate, "candidate")):
                endpoint = Path(scratch) / (label + ".gz")
                with (trial / "stdout.log").open("rb") as source, gzip.open(endpoint, "wb", compresslevel=1) as target:
                    shutil.copyfileobj(source, target)
                endpoints.append(str(endpoint))
            field_path = Path(scratch) / "fields.json"
            command = [sys.executable, str(TOOLS / "compare_solver_fields.py"),
                       *endpoints, str(field_path)]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            fields = json.loads(field_path.read_text())
            args.output.with_suffix(".fields.json").write_text(field_path.read_text())
            for key in ("force_regions_over_1pct", "strain_metrics_over_1pct", "strain_metrics_over_5pct"):
                fields[key] = len(fields[key])
            report["regional_comparison"] = fields
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, allow_nan=False))

if __name__ == "__main__":
    main()
