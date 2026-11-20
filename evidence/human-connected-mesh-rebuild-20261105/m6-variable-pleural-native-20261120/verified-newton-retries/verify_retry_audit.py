"""Verify the exact native attempt ledger and physical trace continuity."""
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import sys

target = Path(sys.argv[1])
control = Path("certified-budget-edge-compensated-parameters-steps18")
execution = json.loads((target/"execution.json").read_text())
stdout = (target/"stdout.log").read_text()
stderr = (target/"stderr.log").read_text()
rejections = [int(step) for step in re.findall(
    r"^human_verified_newton_retry step=(\d+) .*rollback=byte_identical$", stdout, re.M
)]
accepted = [int(step) for step in re.findall(
    r"^human_standing_progress=accepted step=(\d+) ", stdout, re.M
)]
completed_retries = [tuple(map(int, fields)) for fields in re.findall(
    r"^human_verified_newton_retry_outcome step=(\d+) completed_steps=(\d+) accepted=(\d+) published=(\d+)$",
    stdout, re.M
)]
traces = {}
for name in ("resting-coupled", "resting-com-support-impulses", "resting-com-momentum-diagnostic"):
    original = (control/"native"/f"{name}.csv").read_bytes()
    current = (target/"native"/f"{name}.csv").read_bytes()
    traces[name] = {"identical":current == original, "sha256":hashlib.sha256(current).hexdigest()}
with (target/"native/resting-coupled.csv").open() as stream:
    rows = list(csv.DictReader(stream))
assert rows and len(rows) == len(accepted)
assert all(math.isclose(float(row["time_s"]),step*.002,abs_tol=2e-8)
           for step,row in zip(accepted,rows))
ply = [float(row["pleural_pa"]) for row in rows]
root_assistance = [max(abs(float(row["root_assistance_n"])),abs(float(row["root_assistance_nm"])))
                   for row in rows]
simulated = float(rows[-1]["time_s"])
terminal = re.search(r"^resting_FEM_terminal_audit=passed .*$", stdout, re.M)
assert terminal is not None
audit = dict(re.findall(r"([A-Za-z_]+)=([^ ]+)", terminal.group()))
assert int(audit["accepted_steps"]) == 18
assert int(audit["adapter_passes"]) == int(audit["expected_adapter_passes"]) == 21
assert float(audit["min_J"]) > 0.35
report = {
    "terminal_fem_audit":audit,
    "peak_bed_penetration_m":max(float(row["peak_penetration_m"]) for row in rows),
    "status":"verified",
    "execution":execution,
    "accepted_physical_steps":accepted,
    "verified_rejected_attempt_steps":rejections,
    "completed_gpu_attempts":len(accepted)+len(rejections),
    "full_budget_retries":completed_retries,
    "simulated_seconds":simulated,
    "end_to_end_real_time_factor":simulated/execution["wall_seconds"],
    "pleural_range_pa":[min(ply),max(ply)],
    "peak_bed_impulse_ns":max(float(row["normal_impulse_ns"]) for row in rows),
    "peak_mass_residual_kg":max(abs(float(row["max_abs_body_mass_residual_kg"])) for row in rows),
    "assistance_zero":max(root_assistance) == 0,
    "physical_traces_against_pre_audit_fix":traces,
    "crossing_verification":"whole-scene trajectory not measured",
}
assert execution["exit"] == 0
assert "persistent Human borrowed tendon-load snapshot disagreed" not in stderr
assert accepted == list(range(1,19)) and rejections == [16,17,18]
assert completed_retries == [(s,s,1,1) for s in rejections]
assert not (target/"failure.json").exists()
assert max(ply) > min(ply) and report["assistance_zero"]
assert all(row["identical"] for row in traces.values())
path = target/"verified-retry-audit.json"
path.write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report))
