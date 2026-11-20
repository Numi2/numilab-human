"""Replay the captured near-margin feature using the native sweep limiter."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

witness, *libraries = map(Path, sys.argv[1:])
contact = json.loads(witness.read_text())["contact"]
margin = contact["numerical_clearance_offset_m"]
kind = contact["identity"][2]
normal = contact["normal"]
assert kind == 0
def endpoint(where):
    return [a[:3]+b[:3] for a,b in zip(contact[f"potential_feature_{where}_high"],
                    contact[f"potential_feature_{where}_low"],strict=True)]
start, finish = endpoint("start"), endpoint("end")
crossing = finish.copy()
crossing[0] = finish[3]
cases = {"observed": finish, "unchanged": start, "crossing": crossing}
records = [str(len(cases))]
for end in cases.values():
    records.append(f"{margin:.9g} {kind}")
    records.extend(" ".join(f"{value:.9g}" for value in point) for point in start+end)
records.extend(" ".join(f"{value:.9g}" for value in normal) for _ in cases)
results = []
for library in libraries:
    probe = Path(__file__).with_name("check_native_contact_witness")
    out = subprocess.run([probe, library, "--prefix"], text=True, capture_output=True,
                         input="\n".join(records)+"\n", check=True)
    numbers = [tuple(map(float,line.split())) for line in out.stdout.splitlines()]
    by_case = dict(zip(cases,numbers,strict=True))
    assert all(clearance == margin and 0. <= alpha <= 1.
               for clearance,alpha in numbers), by_case
    assert by_case["unchanged"][1] == 1. and by_case["crossing"][1] < 1.
    with library.open("rb") as stream:
        sha = hashlib.file_digest(stream,"sha256").hexdigest()
    results.append(dict(library=str(library),sha256=sha,
                        cases={name:dict(margin_m=m,fraction=f)
                               for name,(m,f) in by_case.items()}))
assert len(results)==2
print(json.dumps(results,indent=2))
