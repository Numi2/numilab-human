"""Compare matched release prefixes using accepted traces and native receipts."""
from argparse import ArgumentParser
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_bank_trials import endpoint_checks

PREFIX = "matter_fgmres_active_work="
KEYS = (
    "time_s", "lung_volume_ml", "airflow_ml_s", "pleural_pa", "PaO2_mmhg",
    "PaCO2_mmhg", "lv_mmhg", "ra_target_ml", "blood_ml", "blood_error_ml",
    "body_mass_residual_kg", "normal_impulse_ns", "root_assistance_n",
    "root_assistance_nm", "peak_penetration_m",
)
PHASE = re.compile(r"^human_execution_stage=(\w+) wall_elapsed_ms=([.0-9]+) stage_step=(\d+)")
TIMING = re.compile(r"^matter_gpu_stage=(\w+) step=(\d+) elapsed_ns=(\d+)")


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def field_comparison(a, b, keys):
    assert len(a) == len(b) != 0
    out = {}
    for key in keys:
        left, right = ([float(row[key]) for row in records] for records in (a, b))
        assert all(map(math.isfinite, [*left, *right])), key
        delta = max(abs(x-y) for x, y in zip(left, right, strict=True))
        scale = max(map(abs, left))
        out[key] = {"baseline_min": min(left), "baseline_max": max(left),
                    "candidate_min": min(right), "candidate_max": max(right),
                    "max_absolute_delta": delta,
                    "relative_to_baseline_peak": delta/scale if scale else None}
    return out


def receipt(path, endpoint):
    status = json.loads((path / "execution.json").read_text())
    assert status["exit"] == 0, status
    starts, ends, messages, work, stages = {}, {}, {}, [], {}
    with (path / "stdout.log").open() as stream:
        for line in stream:
            match = PHASE.match(line)
            if match:
                name, millis, step = match.groups()
                if name == "authoritative_segment_begin":
                    starts[int(step)+1] = float(millis)
                if name == "authoritative_segment_end":
                    ends[int(step)] = float(millis)
            for prefix in ("resting_FEM_terminal_audit=", "resting_fascia_fem_terminal_audit=",
                           "resting_FEM_bed_release_initialization=", "whole_body_tissue_mass_reference="):
                if line.startswith(prefix):
                    messages[prefix[:-1]] = line.strip()
            if line.startswith("human_staged_solver_attempt "):
                fields = dict(item.split("=", 1) for item in line.split()[1:])
                messages.setdefault("staged_attempts", []).append(fields)
    with (path / "stderr.log").open() as stream:
        for line in stream:
            if line.startswith(PREFIX):
                work.append(json.loads(line[len(PREFIX):]))
            match = TIMING.match(line)
            if match:
                name, step, nanos = match.groups()
                record = stages.setdefault(int(step), {}).setdefault(name, {"spans": 0, "duration_ns": 0})
                record["spans"] += 1
                record["duration_ns"] += int(nanos)
    durations = [ends[k]-starts[k] for k in sorted(ends)]
    trace = rows(path / "native/resting-coupled.csv")
    assert len(durations) == len(work) == len(trace), (len(durations), len(work), len(trace))
    assert sorted(ends) == list(range(1,len(trace)+1))
    elapsed = float(trace[-1]["time_s"])
    return {
        "execution": status, "segments": len(durations), "simulated_seconds": elapsed,
        "process_realtime_factor": elapsed/status["wall_seconds"],
        "segment_sum_wall_s": sum(durations)/1000,
        "segment_median_after_first_ms": statistics.median(durations[1:]),
        "per_segment_wall_ms": durations,
        "total_krylov_columns": sum(x["used_columns_sum"] for x in work),
        "total_active_newton_begins": sum(x["active_newton_begins"] for x in work),
        "budget_exhausted_solves": sum(x["budget_exhausted_solves"] for x in work),
        "krylov_columns_by_root": [x["used_columns_sum"] for x in work],
        "timed_gpu_stages": stages, "native_diagnostics": messages,
        **({"terminal_checks": endpoint_checks(path)} if endpoint else {}),
    }


def paired(a, b, endpoint):
    baseline, candidate = receipt(a, endpoint), receipt(b, endpoint)
    p = "native/resting-coupled.csv"
    trace_a, trace_b = rows(a / p), rows(b / p)
    dyn_fields = tuple(dict.fromkeys((*KEYS, *trace_a[0].keys())))
    coupled = field_comparison(trace_a, trace_b, (k for k in dyn_fields if k in trace_a[0]))
    assert coupled["time_s"]["max_absolute_delta"] == 0.
    p = "native/resting-body-motion-partition.csv"
    motion_a, motion_b = rows(a / p), rows(b / p)
    assert [(x["accepted_step"], x["body_index"]) for x in motion_a] == [
        (x["accepted_step"], x["body_index"]) for x in motion_b]
    motion = field_comparison(motion_a, motion_b,
                              (k for k in motion_a[0].keys() if k != "body_name"))
    return {
        "baseline": baseline, "candidate": candidate,
        "process_wall_speed_ratio": baseline["execution"]["wall_seconds"]/candidate["execution"]["wall_seconds"],
        "native_segment_speed_ratio": baseline["segment_sum_wall_s"]/candidate["segment_sum_wall_s"],
        "coupled_trace": coupled, "body_motion": motion,
        "coupled_csv_byte_equal": (a / "native/resting-coupled.csv").read_bytes()
                                  == (b / "native/resting-coupled.csv").read_bytes(),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--endpoint", action="store_true")
    args = parser.parse_args()
    with args.report.open("x") as stream:
        json.dump(paired(args.baseline,args.candidate,args.endpoint),stream,indent=2,allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
