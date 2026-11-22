#!/bin/bash
set -euo pipefail
root=$1; py=$2; cavity=$3; work=$4; reader=$5; predicate=$6; recruit=$7
out="$root/bonded-equilibrium"; final="$out/bed-stop/paired"; full="$final/three"
check_frame() {
    local dir="$1"
    if test -s "$dir/motion.json" && test -s "$dir/endpoint-recovery.json" && "$py" -c 'import sys,json,hashlib,pathlib; p=pathlib.Path(sys.argv[1]); a=json.loads((p/"endpoint-recovery.json").read_text()); assert a["verified_member_sha256"]["verified-endpoint.json"]==hashlib.sha256((p/"verified-endpoint.json").read_bytes()).hexdigest()' "$dir"; then return; fi
    "$py" "$cavity/check_cavity_endpoint.py" "$dir" "$work" "$reader" "$predicate" --require-clear >"$dir/endpoint-check.log" 2>&1 || test -s "$dir/verified-endpoint.json"
    "$py" "$cavity/measure_cavity_step.py" "$dir" "$dir/motion.json"
    "$py" "$cavity/archive_cavity_endpoint.py" "$dir" "$dir/endpoint-recovery.json"
}
if test -s "$full/terminal.receipt.json"; then check_frame "$full"; fi
n=$("$py" - "$full" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if (p/'failure.json').exists():print(json.loads((p/'failure.json').read_text())['segment_start'])
else:print(json.loads((p/'terminal.receipt.json').read_text())['accepted_step'])
PY
)
if (( n > 0 )); then
    steps=$(( n < 2 ? n : 2 ))
    dir="$final/accepted-$steps"
    if ! test -s "$dir/terminal.receipt.json"; then
        "$py" -c 'import shutil; assert shutil.disk_usage(".").free>=12*2**30'
        "$py" "$cavity/run_cavity_motion.py" "$final/parent" "$final/owner" "$cavity/typed-fans/six-margin.npz" "$dir" --steps "$steps" --capture-fem-body-load --recruitment-reference "$recruit" >"$final/matched-endpoint.log" 2>&1
    fi
    check_frame "$dir"
fi
"$py" "$out/check_trial.py" "$root" >"$out/result.json"
