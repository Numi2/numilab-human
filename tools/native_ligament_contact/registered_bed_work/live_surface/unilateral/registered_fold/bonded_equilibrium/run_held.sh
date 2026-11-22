#!/bin/bash
set -euo pipefail
root=$1;py=$2;cav=$3;recruit=$4
case="$root/bonded-equilibrium"; dir="$case/held-volume"
test -s "$dir/verified-mass.json"
if ! test -s "$dir/observed-owner/identities.json" || ! grep -q '"observed_owner"' "$dir/observed-owner/identities.json"; then
    "$py" "$case/observe_release.py" "$dir" >"$dir/observed-build.log" 2>&1
fi
"$py" -c 'import shutil;assert shutil.disk_usage(".").free>=12*2**30'
"$py" "$cav/run_cavity_motion.py" "$case/bed-stop/paired/parent" "$dir/observed-owner" "$cav/typed-fans/six-margin.npz" "$dir/three" --steps 3 --trace-root 2 --capture-fem-body-load --recruitment-reference "$recruit" >"$dir/native.log" 2>&1 || test -s "$dir/three/failure.json"
