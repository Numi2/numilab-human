#!/bin/bash
set -euo pipefail
root=$1; py=$2; cav=$3; work=$4; reader=$5; predicate=$6; recruit=$7
case="$root/bonded-equilibrium"; final="$case/free-interface"; paired="$case/bed-stop/paired"
n=$("$py" - "$final/three" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1])
if (p/'failure.json').exists():print(json.loads((p/'failure.json').read_text())['segment_start'])
else:print(json.loads((p/'terminal.receipt.json').read_text())['accepted_step'])
PY
)
check_frame() {
    local dir="$1"
    "$py" "$cav/check_cavity_endpoint.py" "$dir" "$work" "$reader" "$predicate" --require-clear >"$dir/endpoint-check.log" 2>&1 || test -s "$dir/verified-endpoint.json"
    "$py" "$cav/measure_cavity_step.py" "$dir" "$dir/motion.json"
    "$py" "$cav/archive_cavity_endpoint.py" "$dir" "$dir/endpoint-recovery.json"
}
if test -s "$final/three/terminal.receipt.json"; then check_frame "$final/three"; fi
if (( n>0 )); then
    steps=$((n<2?n:2));dir="$final/accepted-$steps"
    if ! test -s "$dir/terminal.receipt.json"; then
        "$py" -c 'import shutil; assert shutil.disk_usage(".").free>=12*2**30'
        "$py" "$cav/run_cavity_motion.py" "$paired/parent" "$final/owner" "$cav/typed-fans/six-margin.npz" "$dir" --steps "$steps" --capture-fem-body-load --recruitment-reference "$recruit" >"$final/matched.log" 2>&1
    fi
    check_frame "$dir"
fi
"$py" "$case/check_trial.py" "$root" --free-interface >"$final/result.json"
