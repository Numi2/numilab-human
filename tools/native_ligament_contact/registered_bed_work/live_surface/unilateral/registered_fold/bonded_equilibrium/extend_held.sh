#!/bin/bash
set -euo pipefail
root=$1; py=$2; cav=$3; work=$4; reader=$5; predicate=$6; recruit=$7
case="$root/bonded-equilibrium"; dir="$case/held-volume"; out="$dir/six"
"$py" - "$dir" <<'PY'
import json,sys,shutil
from pathlib import Path
p=Path(sys.argv[1]);v=json.loads((p/'result.json').read_text())['measured_endpoints']['held-volume-three']['volume']
if v['parent_floor_failures'] or v['new_floor_failures']:raise ValueError('Third step missed a fixed material-volume bound')
if shutil.disk_usage(p).free<12*2**30:raise ValueError('Insufficient disk for native trial')
PY
"$py" "$cav/run_cavity_motion.py" "$case/bed-stop/paired/parent" "$dir/observed-owner" "$cav/typed-fans/six-margin.npz" "$out" --steps 6 --trace-root 3 --capture-fem-body-load --recruitment-reference "$recruit" >"$dir/six.log" 2>&1 || test -s "$out/failure.json"
"$py" "$case/six_summary.py" "$dir" >"$dir/six-summary.json"
if test -s "$out/terminal.receipt.json"; then
    "$py" "$cav/check_cavity_endpoint.py" "$out" "$work" "$reader" "$predicate" --require-clear >"$out/endpoint-check.log" 2>&1 || test -s "$out/verified-endpoint.json"
    "$py" "$cav/measure_cavity_step.py" "$out" "$out/motion.json"
    "$py" "$cav/archive_cavity_endpoint.py" "$out" "$out/endpoint-recovery.json"
fi
