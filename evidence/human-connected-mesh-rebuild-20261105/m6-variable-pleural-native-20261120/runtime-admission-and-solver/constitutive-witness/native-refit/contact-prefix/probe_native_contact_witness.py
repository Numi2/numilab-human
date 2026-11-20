"""Check the exact captured feature sweep with the existing Metal proof."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

run = Path(sys.argv[1]).resolve()
script_dir = Path(__file__).resolve().parent
launch = json.loads((run/"launch.json").read_text())
metallib = Path(launch["environment"]["NUMI_MATTER_METALLIB_OVERRIDE"])
with metallib.open("rb") as stream:
    assert hashlib.file_digest(stream,"sha256").hexdigest() == launch["pins"][str(metallib)]
tag = "fem_deformable_limit_trace="
traces = [json.loads(line[len(tag):]) for line in (run/"stderr.log").read_text().splitlines()
          if line.startswith(tag)]
found = [(trace,contact) for trace in traces for contact in trace["contacts"]
         if contact["contact_slot"] == 2222094 and trace["root"] == 12 and trace["iteration"] == 15]
assert len(found) == 1, f"Expected one exact contact record, got {len(found)}"
trace, row = found[0]
assert row["geometry_coordinate_frame"] == "reference_local"
assert row["identity"][0] == row["identity"][1] == 15
assert launch["environment"]["NM_FEM_CLEARANCE_ALIGNED_BARRIER"] == "1"
kind = row["identity"][2]
margin = row["numerical_clearance_offset_m"]
assert kind in (0,1,2) and margin > 0.
def points(endpoint):
    return [high[:3]+low[:3] for high,low in zip(
        row[f"potential_feature_{endpoint}_high"],
        row[f"potential_feature_{endpoint}_low"],strict=True)]
start,finish = points("start"), points("end")
crossing_end = finish.copy()
crossing_end[0] = finish[2 if kind == 1 else 1]
cases = [
    ("unchanged_accepted_feature",margin,start),
    ("full_proposed_correction_at_native_margin",margin,finish),
    ("full_proposed_correction_at_positive_picometre_margin",1e-12,finish),
    ("deliberately_crossing_endpoint",margin,crossing_end),
]
records = [str(len(cases))]
for _,clearance,end in cases:
    records.append(f"{clearance:.9g} {kind}")
    records.extend(" ".join(format(value,".9g") for value in point) for point in start+end)
probe = subprocess.run([script_dir/"check_native_contact_witness",metallib],
                       input="\n".join(records)+"\n",text=True,capture_output=True,check=True)
lines = probe.stdout.splitlines()
assert len(lines)==len(cases)
summary = {name:bool(int(line.split("=")[-1])) for (name,_,_),line in zip(cases,lines,strict=True)}
assert not summary["deliberately_crossing_endpoint"]
print(json.dumps({"contact":row,"metal_interval_proof":summary,
                  "metallib_sha256": launch["pins"][str(metallib)],
                  "line_search_alpha":trace["line_search_alpha"]},indent=2))
