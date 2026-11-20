"""Compare the same-cadence staged native run against its retained paired control."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

here = Path.cwd()
baseline = here.parent / "resting-clearance-and-atrium-014/certified-budget-edge-compensated-complete-audit-steps18"
trial = here / sys.argv[1]

def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))

def match_csv(name):
    base = baseline/"native"/name
    test = trial/"native"/name
    before, after = read_csv(base), read_csv(test)
    first_difference = None
    field_changes = {}
    mismatch_by_field = Counter()
    for step, (reference, observed) in enumerate(zip(before,after),1):
        for key in reference.keys() | observed.keys():
            a,b = reference[key],observed[key]
            if a == b:
                continue
            if first_difference is None:
                first_difference = step
            mismatch_by_field[key] += 1
            if not a or not b:
                continue
            try:
                p,q = float(a),float(b)
            except ValueError:
                continue
            if abs(p-q) > field_changes.get(key, {}).get("max_abs", -1):
                field_changes[key] = {"max_abs":abs(p-q), "at_row":step,
                                      "baseline":p, "staged":q}
    left,right = base.read_bytes(),test.read_bytes()
    return {"row_counts":[len(before),len(after)],
            "byte_identical":left == right,
            "staged_sha256":hashlib.sha256(right).hexdigest(),
            "first_changed_row":first_difference,
            "changed_rows_by_field":dict(mismatch_by_field),
            "numeric_difference":field_changes}

run = json.loads((trial/"execution.json").read_text())
control = json.loads((baseline/"execution.json").read_text())
stdout = (trial/"stdout.log").read_text()
stderr = (trial/"stderr.log").read_text()
rows = read_csv(trial/"native/resting-coupled.csv")
simulated = float(rows[-1]["time_s"]) if rows else 0.0
starts = [tuple(map(int,m)) for m in re.findall(
    r"^human_staged_solver_attempt step=(\d+) tier=(\d+) newton=(\d+) linear=(\d+) accepted=(\d+) published=(\d+)",
    stdout, re.M)]
rollbacks = [tuple(map(int,m)) for m in re.findall(
    r"^human_staged_solver_rollback step=(\d+) tier=(\d+) rollback=byte_identical",
    stdout, re.M)]
accepted = [int(step) for step in re.findall(r"^human_standing_progress=accepted step=(\d+)",stdout,re.M)]
owner = re.search(r"^resting_FEM_terminal_audit=passed .+",stdout,re.M)
pressure = [float(r["pleural_pa"]) for r in rows]
comparison = {
    "control":control,"staged":run,
    "accepted_steps":accepted,
    "accepted_seconds":simulated,
    "staged_attempts":starts,
    "verified_rollbacks":rollbacks,
    "terminal_audit":owner.group() if owner else None,
    "end_to_end_rtf":simulated/run["wall_seconds"],
    "wall_speedup_same_scope":control["wall_seconds"]/run["wall_seconds"] if len(rows)==18 else None,
    "pressure_pa":[min(pressure),max(pressure)] if pressure else None,
    "peak_bed_penetration_m":max((float(r["peak_penetration_m"]) for r in rows),default=0),
    "max_root_assistance_n":max((abs(float(r["root_assistance_n"])) for r in rows),default=0),
    "traces":{n:match_csv(n) for n in (
        "resting-coupled.csv","resting-com-support-impulses.csv",
        "resting-com-momentum-diagnostic.csv")},
    "terminal_error":next((line for line in reversed(stderr.splitlines())
                           if line.startswith("myosim_articulated_visual=failed")),None),
}
if run["exit"] == 0:
    assert owner and accepted == list(range(1,len(rows)+1))
    assert [(a[0],a[1]) for a in starts if a[-2:] == (0,0)] == rollbacks
    assert [a[0] for a in starts if a[-2:] == (1,1)] == accepted[1:]
    assert comparison["max_root_assistance_n"] == 0
    assert pressure and max(pressure)>min(pressure)
comparison["parity"] = all(trace["byte_identical"] for trace in comparison["traces"].values())
(trial/"comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
print(json.dumps({k:v for k,v in comparison.items() if k!="traces"}))
