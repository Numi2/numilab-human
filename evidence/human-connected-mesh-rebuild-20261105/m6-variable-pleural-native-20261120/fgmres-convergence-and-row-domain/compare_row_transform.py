"""Check matched native work, accepted state and stage timing."""
import hashlib
import json
from pathlib import Path
import sys

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def run(directory):
    launch = json.loads((directory / "launch.json").read_text().replace(
        str(directory.resolve()), "<run-output>"))
    summary = json.loads((directory / "profile-summary.json").read_text())
    key = "NM_FGMRES_ROW_TRANSFORM_WORK_DOMAIN"
    policy = launch["environment"].pop(key)
    assert summary["exact_physical_trace_match"] and summary["accepted_rows"] == 3
    works = {}
    for line in (directory / "stderr.log").read_text().splitlines():
        if not line.startswith("matter_fgmres_active_work="):
            continue
        event = json.loads(line.partition("=")[2])
        works[event["root"]] = event
    assert list(sorted(works)) == [0, 1, 2]
    archives = {p.name: digest(p) for p in directory.joinpath("native").glob("*.csv")}
    return launch, policy, summary, works, archives

def main():
    on, off = [run(Path(p)) for p in sys.argv[1:]]
    on_plan, on_policy, on_timing, on_work, on_traces = on
    off_plan, off_policy, off_timing, off_work, off_traces = off
    assert on_policy == "1" and off_policy == "0"
    assert on_plan == off_plan, "Expected identical launch pins and inputs"
    assert on_work == off_work, "FGMRES activity or convergence diverged"
    assert on_traces == off_traces and len(on_traces) == 10
    key = "matter_newton_linear"
    gpu_on = on_timing["stage_totals"][key]["sum_ms"]
    gpu_off = off_timing["stage_totals"][key]["sum_ms"]
    result = {
        "scope": "Single matched three-root headless pair with root 1 stage profile",
        "accepted_steps": 3,
        "exact_physical_trace_fields": 3 * 77,
        "exact_trace_sha256": on_traces,
        "krylov_columns_by_root": {i: w["used_columns_sum"] for i, w in on_work.items()},
        "root1_linear_gpu_ms_control": gpu_off,
        "root1_linear_gpu_ms_candidate": gpu_on,
        "root1_linear_gpu_percent_reduction": (1 - gpu_on / gpu_off) * 100,
        "control": off_timing,
        "candidate": on_timing,
    }
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
