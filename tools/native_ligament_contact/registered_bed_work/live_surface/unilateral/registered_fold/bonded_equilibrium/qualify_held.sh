#!/bin/bash
set -euo pipefail
root=$1; py=$2; cav=$3; work=$4; reader=$5; predicate=$6; recruit=$7
case="$root/bonded-equilibrium"; dir="$case/held-volume"; run="$dir/three"
n=$("$py" - "$run" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if (p/'failure.json').exists():print(json.loads((p/'failure.json').read_text())['segment_start'])
else:print(json.loads((p/'terminal.receipt.json').read_text())['accepted_step'])
PY
)
if (( n>0 )); then
    if ! test -s "$run/terminal.receipt.json"; then
        run="$dir/accepted-$n"
        "$py" -c 'import shutil;assert shutil.disk_usage(".").free>=12*2**30'
        "$py" "$cav/run_cavity_motion.py" "$case/bed-stop/paired/parent" "$dir/observed-owner" "$cav/typed-fans/six-margin.npz" "$run" --steps "$n" --capture-fem-body-load --recruitment-reference "$recruit" >"$dir/accepted-endpoint.log" 2>&1
    fi
    "$py" "$cav/check_cavity_endpoint.py" "$run" "$work" "$reader" "$predicate" --require-clear >"$run/endpoint-check.log" 2>&1 || test -s "$run/verified-endpoint.json"
    "$py" "$cav/measure_cavity_step.py" "$run" "$run/motion.json"
    "$py" "$cav/archive_cavity_endpoint.py" "$run" "$run/endpoint-recovery.json"
fi
"$py" "$case/check_trial.py" "$root" --held >"$dir/result.json"
