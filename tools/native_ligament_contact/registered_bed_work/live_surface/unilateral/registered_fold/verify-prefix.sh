#!/bin/bash
set -euo pipefail
case_dir=$1; py=$2; cavity=$3; recruitment=$4; work=$5; reader=$6; predicate=$7
cd "$case_dir"
if test -s three/terminal.receipt.json; then
    job=three
else
    n=$("$py" -c "import json; d=json.load(open('three/failure.json')); assert d['segment_start']>0; print(d['segment_start'])")
    job="accepted-$n"
    "$py" -c 'import shutil; assert shutil.disk_usage(".").free>=12*2**30'
    "$py" "$cavity/run_cavity_motion.py" "$case_dir/parent" "$case_dir/owner" "$cavity/typed-fans/six-margin.npz" "$case_dir/$job" --steps "$n" --capture-fem-body-load --recruitment-reference "$recruitment" >"$case_dir/replay-motion.log" 2>&1
fi
"$py" "$cavity/check_cavity_endpoint.py" "$case_dir/$job" "$work" "$reader" "$predicate" --require-clear >"$case_dir/endpoint.log" 2>&1 || test -s "$job/verified-endpoint.json"
"$py" "$cavity/measure_cavity_step.py" "$case_dir/$job" "$case_dir/motion.json"
"$py" "$case_dir/../../associate_skin_witnesses.py" "$case_dir/$job" >"$case_dir/crossing-owners.json"
"$py" "$cavity/archive_cavity_endpoint.py" "$case_dir/$job" "$case_dir/endpoint-recovery.json"
