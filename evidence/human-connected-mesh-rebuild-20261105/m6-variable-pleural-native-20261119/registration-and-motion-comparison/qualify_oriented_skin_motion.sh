#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PY=/Users/numi/numi-human-aortic-interface-20261015/.venv/bin/python
NAME=cardiac-split-native-8000-trimmed-oriented
"$PY" run_cardiac_split_trial.py 8000 --capture 8000 --suffix=-trimmed-oriented --skin-trial trimmed-oriented-pressure-skin
"$PY" summarize_cardiac_native_run.py "$NAME" "$NAME-summary.json" > "$NAME-summary.log"
"$PY" verify_native_cardiac_split.py "$NAME/native/accepted-geometry/step-8000.receipt.json" "$NAME-split-verification.json" > "$NAME-split-verification.log"
"$PY" - <<'PY'
import json, os, sys
from fit_cardiac_volume_grid import HERE, PREVIOUS, hash_file
sys.path.insert(0,str(PREVIOUS))
from archive_bed_preflight import archive
n=HERE/"cardiac-split-native-8000-trimmed-oriented/native"
for f in ("skin-source-influences.bin","resting-human.mrvpack"):
    archive(n/f)
PY
"$PY" - <<'PY' > "$NAME-skin-exact.log"
from pathlib import Path
import runpy
from fast_skin_intersection_audit import install_for_float32
from fit_cardiac_volume_grid import HERE
install_for_float32()
run=HERE/"cardiac-split-native-8000-trimmed-oriented"
audit_path=HERE.parent/"resting-surface-clearance-001/audit.py"
runpy.run_path(str(audit_path))["run"](run/"native/accepted-geometry/step-8000.mrvpack",
    HERE/(run.name+"-skin-exact.json"))
PY
"$PY" compare_guarded_variable_pressure.py --variant trimmed-oriented
"$PY" - <<'PY'
import json, os, sys
from fit_cardiac_volume_grid import HERE, PREVIOUS, hash_file
sys.path.insert(0,str(PREVIOUS))
from archive_bed_preflight import archive
run=HERE/"cardiac-split-native-8000-trimmed-oriented/native/accepted-geometry"
for f in ("step-8000.mrvpack","step-8000.common-position-low.bin"):
    p=run/f
    old=HERE/"cardiac-split-native-8000/native/accepted-geometry"/(f+".zst")
    expected=json.loads(old.with_suffix(".zst.recovery.json").read_text())["sha256"]
    if hash_file(p)==expected:
        os.link(old,p.with_suffix(p.suffix+".zst"))
    archive(p)
PY
