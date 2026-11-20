"""Summarize matched hardware times and record trace differences separately."""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

root = Path(__file__).resolve().parent / "fgmres-pair-ramp8"
numbers = re.compile(r"resting_native_profile step=(\d+) .*?gpu_ms=([\d.]+)")
def sha(path):
    with path.open("rb") as data:
        return hashlib.file_digest(data, "sha256").hexdigest()

pairs = []
for trial in range(3):
    variants = {}
    for variant in ("control", "distributed"):
        run = root / f"{trial}-{variant}"
        elapsed = json.loads((run / "execution.json").read_text())
        log = (run / "stdout.log").read_text()
        coupled = run / "native/resting-coupled.csv"
        with coupled.open() as stream:
            trace = list(csv.DictReader(stream))
        timings = {int(step): float(ms) for step, ms in numbers.findall(log)}
        assert elapsed["exit"] == 0 and sorted(timings) == list(range(1,9))
        assert log.count("human_standing_progress=accepted") == 8
        pleural = [float(row["pleural_pa"]) for row in trace]
        variants[variant] = {
            "transaction_gpu_ms": [timings[i] for i in range(1,9)],
            "first_four_gpu_ms": sum(timings[i] for i in range(1,5)),
            "entire_launch_wall_s": elapsed["seconds"],
            "realtime_factor_including_setup": .016 / elapsed["seconds"],
            "pleural_pa_range": [min(pleural), max(pleural)],
            "coupled_trace_sha256": sha(coupled),
            "trace": trace,
        }
    control, test = variants["control"], variants["distributed"]
    relative = 1. - test["first_four_gpu_ms"] / control["first_four_gpu_ms"]
    control_trace, test_trace = control.pop("trace"), test.pop("trace")
    assert len(control_trace) == len(test_trace)
    assert all(a.keys() == b.keys() and a["time_s"] == b["time_s"] for a,b in zip(control_trace,test_trace))
    columns = {}
    for key in control_trace[0]:
        x, y = ([float(item[key]) for item in rows] for rows in (control_trace,test_trace))
        error = max(abs(a-b) for a,b in zip(x,y))
        relative_trace_scale = max(abs(v) for v in x)
        if error:
            columns[key] = {
                "maximum_absolute_difference": error,
                "relative_to_maximum_absolute_control": error / relative_trace_scale if relative_trace_scale else None,
            }
    pairs.append({"trial":trial,"primary_window_gpu_time_reduction":relative,**variants,"changed_columns":columns})
median = statistics.median(row["primary_window_gpu_time_reduction"] for row in pairs)
result = {"gpu_primary_window_median_reduction":median,"target_gpu_time_reduction":0.20,
          "promote": False, "pairs":pairs, "scope":"26 fascia regions and two ACL objects, eight 2 ms release roots, headless. The paired calculation tests the first four roots."}
(root.parent / "fgmres-pair-ramp8-summary.json").write_text(json.dumps(result, indent=2)+"\n")
print(median)
