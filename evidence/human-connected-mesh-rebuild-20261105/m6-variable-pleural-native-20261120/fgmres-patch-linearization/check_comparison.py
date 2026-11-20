"""Verify the measured cache experiment against exact accepted native traces."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
retained = HERE.parent / "muscle-shell-registration-018"
trial = HERE / "cached-root-1"
subprocess.run([sys.executable, str(retained / "check_profile_matched_roots.py"), str(trial)],check=True)
control = Path(sys.argv[1]).resolve() if len(sys.argv)>1 else retained/"profile-root-1"
a = json.loads((control/"profile-summary.json").read_text())
b = json.loads((trial/"profile-summary.json").read_text())
assert a["root_profiled"] == b["root_profiled"] == 1
assert a["accepted_rows"] == b["accepted_rows"] == 3
assert a["fields_per_row"] == b["fields_per_row"] == 77
assert a["exact_physical_trace_match"] and b["exact_physical_trace_match"]

def counters(folder):
    lines = (folder/"stderr.log").read_text().splitlines()
    prefix = "matter_fgmres_active_work="
    telemetry = [json.loads(line.removeprefix(prefix)) for line in lines if line.startswith(prefix)]
    return dict((row["root"], {
        "columns": row["used_columns_sum"], "restarts": row["cycles_with_work"],
        "newton_begins": row["active_newton_begins"], "exhausted_solves": row["budget_exhausted_solves"],
    }) for row in telemetry)
assert counters(control) == counters(trial), "Krylov work counts changed"

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

native_a, native_b = (d/"native" for d in (control,trial))
trace_files = {p.name for p in native_a.glob("*.csv")}
assert trace_files == {p.name for p in native_b.glob("*.csv")}
verified = {}
for name in sorted(trace_files):
    actual = digest(native_b/name)
    assert digest(native_a/name) == actual, name
    verified[name] = actual

column = "matter_newton_linear"
linear_a, linear_b = (s["stage_totals"][column]["sum_ms"] for s in (a,b))
err = (trial/"stderr.log").read_text()
cache = re.findall(r"^matter_patch_linearization_cache bytes=(\d+)$",err,re.MULTILINE)
assert cache and all(int(value)>0 for value in cache), "Native solver did not admit the patch cache"
assert "matter_fgmres_operator_icb=1 preconditioner_prefix=0" in err
report = dict(
    scope="One stage-instrumented root per run, with identical accepted physiology/contact fields and Krylov work counts.",
    control=str(control),
    accepted_simulated_s=3*0.000015625,
    gpu_linear_control_ms=linear_a, gpu_linear_cache_ms=linear_b,
    relative_gpu_linear_time_reduction=1-linear_b/linear_a,
    wall_s_control=a["wall_seconds"], wall_s_cache=b["wall_seconds"],
    cache_bytes=int(cache[-1]),
    exact_shared_trace_fields=3*77,
    exact_native_trace_sha256=verified,
    work_counts=counters(trial)
)
(HERE / "comparison.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
