"""Retain the original slot witness and measure the energy-slope discrepancy."""
import json
from pathlib import Path

here = Path.cwd()
tiled = here/"certified-budget-late-witness-steps16"
detail = here/"certified-budget-late-barrier-steps16"

def records(path,key):
    prefix=key+"="
    return [json.loads(line.removeprefix(prefix))
            for line in (path/"stderr.log").read_text().splitlines()
            if line.startswith(prefix)]

t = records(tiled,"fem_deformable_limit_minimum")
selected, = [result for result in records(detail,"fem_deformable_limit_trace")
             if result["contacts"]]
contact, = selected["contacts"]
assert contact["contact_slot"]==614 and len(t)==24
assert all(x["contact_slot"]==614 and x["reason"]==8 for x in t)
assert t[0]["bound"] == contact["lane_bound_after"]
assert contact["ccd_certified"] and not contact["ccd_hit"]
failure=json.loads((tiled/"failure.json").read_text())
detailed_failure=json.loads((detail/"failure.json").read_text())
assert failure["matter_diagnostics"]==detailed_failure["matter_diagnostics"]
assert failure["matter_microsteps"]==183
trace_names=("resting-coupled","resting-com-support-impulses","resting-com-momentum-diagnostic")
for trace in trace_names:
    path=Path("native")/(trace+".csv")
    assert (tiled/path).read_bytes()==(detail/path).read_bytes()
trials=contact["per_alpha_trace"]["trials"]
assert len(trials)==8
assert all(x["tested_full_fraction"] and x["ccd_certified"] and not x["ccd_hit"]
           for x in trials)
assert all(x["reason_code"]==6 for x in trials[:-1]) and trials[-1]["reason_code"]==7
result={
    "accepted_trace_byte_identity":list(trace_names),
    "failed_matter_step":failure["matter_microsteps"],
    "rejection_certificate":failure["matter_diagnostics"],
    "scope":"32 ms candidate from 30 ms predecessor, full 24-pass rollback retry",
    "root":selected["root"], "microtick":selected["microtick"],
    "edges":contact["nodes"][:4],
    "objects":contact["identity"][:2],
    "type":"deformable edge-edge",
    "stored_slope_j":trials[0]["base"][1],
    "trial_changes_j":[(r["energy"][0]-r["base"][0]) for r in trials],
    "trial_fractions":[r["ccd"][0] for r in trials],
    "full_chord_clearance_m":contact["ccd_separation_m"],
    "ccd_boundary_m":contact["collision_thickness_m"],
    "full_budget_iteration_minima":t,
    "unmodified_detail":selected,
}
(here/"late-barrier-witness.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({k:v for k,v in result.items()
                  if k not in ("full_budget_iteration_minima","unmodified_detail")},indent=2))
