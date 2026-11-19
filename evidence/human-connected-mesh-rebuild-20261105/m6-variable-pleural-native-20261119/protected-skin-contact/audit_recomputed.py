import json
import runpy
import sys
from pathlib import Path
from register_tissue_bellies import install_for_float32
import measure_crossing_curves
import mmap
from contextlib import contextmanager
from register_tissue_bellies import hash_file

@contextmanager
def capture_retained_raw(path):
    receipt=json.loads(path.with_suffix(".receipt.json").read_text())
    bodies={row["body_index"]:row for row in receipt["accepted_body_poses"]}
    assert hash_file(path)==receipt["pack_file_sha256"]
    with path.open("rb") as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as mapped:
        yield mapped,bodies,receipt

here=Path(__file__).resolve().parent
run=here/"recomputed-support-native"
raw=run/"native/accepted-geometry/step-8000.mrvpack"
install_for_float32()
out=run/"step-8000-skin-exact.json"
if not out.exists():
    runpy.run_path(str(here.parent/"resting-surface-clearance-001/audit.py"))["run"](raw,out)
measure_crossing_curves.capture=capture_retained_raw
curves=measure_crossing_curves.audit(run,out)
(run/"curves.json").write_text(json.dumps(curves,indent=2)+"\n")
r=json.loads(out.read_text())
print(list(r),flush=True)
print("targeted patches",curves["tissue"].get("(51005, 119)"),curves["tissue"].get("(51004, 89)"),flush=True)
print("tissue curve length",curves["tissue_curve_length_m"], "self",curves["self_crossing"],flush=True)
