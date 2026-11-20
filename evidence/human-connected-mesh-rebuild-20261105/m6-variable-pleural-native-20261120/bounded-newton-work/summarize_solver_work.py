"""Summarize observed solver work per native root without counting post-rejection rows."""
import collections
import csv
import json
from pathlib import Path
import re

root = Path.cwd()
run = root / "active-work"
with (run/"native/resting-coupled.csv").open() as stream:
    accepted = {int(r["step"])-1 for r in csv.DictReader(stream)}
by_root = collections.defaultdict(list)
for line in (run/"stderr.log").read_text().splitlines():
    prefix = "matter_fgmres_microtick_work="
    if line.startswith(prefix):
        row = json.loads(line[len(prefix):])
        by_root[row["root"]].append(row)
t = dict((int(step)-1,float(ms)) for step,ms in re.findall(
    r"resting_native_profile step=(\d+) command_ms=[0-9.]+ gpu_ms=([0-9.]+)",
    (run/"stdout.log").read_text()))
report = {}
for root_id, rows in sorted(by_root.items()):
    if len(rows) != 256 or len({r["microtick"] for r in rows}) != 256:
        raise RuntimeError(f"expected 256 unique scheduled microticks for root {root_id}")
    columns = collections.Counter(r["used_columns"] for r in rows)
    newton = collections.Counter(r["active_newton_begins"] for r in rows)
    latched = collections.Counter(r["final_newton_latched"] for r in rows)
    report[root_id] = dict(
        accepted=root_id in accepted,
        gpu_ms=t.get(root_id),
        microticks=len(rows),
        actual_Krylov_columns=sum(r["used_columns"] for r in rows),
        encoded_cycles=sum(r["encoded_cycles"] for r in rows),
        cycles_with_work=sum(r["cycles_with_work"] for r in rows),
        budget_exhausted_solves=sum(r["budget_exhausted_solves"] for r in rows),
        max_active_Newton_begins=max(newton),
        microticks_by_active_Newton_begins=dict(sorted(newton.items())),
        microticks_by_used_columns=dict(sorted(columns.items())),
        final_Newton_latched_counts=dict(latched),
    )
(root/"active-work-summary.json").write_text(json.dumps(report,indent=2)+"\n")
for key,val in report.items():
    print(key,val["accepted"],val["gpu_ms"],val["actual_Krylov_columns"],val["encoded_cycles"],
          val["cycles_with_work"],val["microticks_by_active_Newton_begins"],
          val["final_Newton_latched_counts"])
