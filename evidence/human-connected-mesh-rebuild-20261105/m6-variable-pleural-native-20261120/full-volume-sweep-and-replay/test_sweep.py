"""Compare the two real Metal sweep paths on captured and analytical controls."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

old_dir, candidate, witness = map(Path, sys.argv[1:])
old = old_dir / "NumiMatter.metallib"
runner = old_dir / "check_native_contact_witness"
contact = json.loads(witness.read_text())["contact"]
margin = contact["numerical_clearance_offset_m"]
def endpoint(which):
    return [a[:3]+b[:3] for a,b in zip(
        contact[f"potential_feature_{which}_high"],contact[f"potential_feature_{which}_low"],strict=True)]
def pose(*points):
    return [list(point)+[0,0,0] for point in points]
def invoke(library, cases):
    lines = [str(len(cases))]
    for _, size, kind, start, finish, _ in cases:
        lines.append(f"{size:.9g} {kind}")
        lines.extend(" ".join(f"{value:.9g}" for value in point) for point in start+finish)
    output = subprocess.run([runner, library], text=True, check=True, capture_output=True,
                            input="\n".join(lines)+"\n").stdout
    return dict(zip((case[0] for case in cases),
                    (int(row.split("=")[-1]) for row in output.splitlines()),strict=True))
zero = (0,0,0)
controls = []
for kind, label in enumerate(("point-triangle","edge-edge","point-point")):
    def fixture(z):
        if kind == 0:
            return pose((.25,.25,z), zero, (1,0,0), (0,1,0))
        if kind == 1:
            return pose((0,-1,z), (0,1,z), (-1,0,0), (1,0,0))
        return pose((0,0,z), zero, zero, zero)
    for suffix,a,b,result in (("separated",.3,.2,1),("crossing",.3,-.3,0),
                              ("clearance-violated",.3,.05,0),("unchanged",.3,.3,1)):
        controls.append((label+"-"+suffix,.1,kind,fixture(a),fixture(b),result))
start, finish = endpoint("start"),endpoint("end")
controls += [("captured-closing",margin,0,start,finish,0),
             ("captured-unchanged",margin,0,start,start,1),
             ("captured-clear-activation-test",.25*margin,0,start,finish,1)]
previous, patched = invoke(old,controls), invoke(candidate,controls)
assert previous == patched, (previous, patched)
expected = {name:known for name,_,_,_,_,known in controls}
assert previous == expected, (previous, expected)
prefix = subprocess.run([sys.executable,old_dir/"check_prefix.py",witness,old,candidate],
                        text=True,check=True,capture_output=True)
fractional = json.loads(prefix.stdout)
assert fractional[0]["cases"] == fractional[1]["cases"], fractional
print(json.dumps(dict(whole_sweeps=patched, fractional_sweeps=fractional,
                      witness_sha256=hashlib.sha256(witness.read_bytes()).hexdigest()),indent=2))
