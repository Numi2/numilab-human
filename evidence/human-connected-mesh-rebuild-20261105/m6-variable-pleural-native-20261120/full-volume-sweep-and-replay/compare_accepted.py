"""Check matching accepted kinetic and physiological history at the physical time indices."""
import csv
import hashlib
import json
from pathlib import Path
import sys

control, candidate = map(Path, sys.argv[1:3])
def read(run, name):
    with (run/"native"/name).open(newline="") as stream:
        return list(csv.DictReader(stream))
def accepted(run):
    return [row for row in read(run,"resting-coupled.csv") if float(row["time_s"]) > 0]

a, b = accepted(control), accepted(candidate)
assert a and b and len(a) <= len(b), (len(a), len(b))
assert a == b[:len(a)]
for row in b:
    assert float(row["body_mass_residual_kg"]) == 0
    assert float(row["root_assistance_n"]) == 0
    assert float(row["root_assistance_nm"]) == 0
assert all(float(x["time_s"]) < float(y["time_s"]) for x,y in zip(b,b[1:]))
def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()
receipt, = (control/"native").glob("registered-volume-fractional-mass-*.json")
mass = candidate/"native"/receipt.name
assert digest(mass) == digest(receipt)
cfg = json.loads((candidate/"launch.json").read_text())
reference = json.loads((control/"launch.json").read_text())
unchanged_env = ("NM_VASCULAR_STEP_EXTERNAL_PRESSURE", "NUMI_HUMAN_RESTING_AORTIC_WALL_OUTER_CONTACT",
                 "NUMI_HUMAN_RESTING_WHOLE_BODY_SKIN_FAT_FEM",
                 "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_RECORDED_CARRIER_ALLOCATION_CSV")
for key in unchanged_env:
    assert cfg["environment"][key] == reference["environment"][key] == "1"
view = json.loads((candidate/"compact-motion-evidence.json").read_text())
base_report = json.loads((control/"compact-motion-evidence.json").read_text())
result = {"matched_accepted_rows": len(a), "exact_csv_field_count": len(a[0]),
          "accepted_rows": len(b), "accepted_time_s": float(b[-1]["time_s"]),
          "recorded_mass_receipt_sha256": digest(mass),
          "accepted_sample_ranges": view["accepted_sample_ranges"],
          "full_wall_real_time_factor": view["verified_end_to_end_simulated_per_wall_second"],
          "matched_contact_work_first_16": view["contact_work"][:len(a)] == base_report["contact_work"][:len(a)],
          "matched_solver_work_first_16": view["solver_work"][:len(a)] == base_report["solver_work"][:len(a)],
          "candidate_terminal": view["native_terminal"]}
assert view["execution"]["exit"] == 0 and len(b) == int(cfg["argv"][cfg["argv"].index("--muscle-step-count")+1])
print(json.dumps(result,indent=2))
