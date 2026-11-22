#!/bin/bash
set -euo pipefail
root=$1; py=$2; cavity=$3; work=$4; predicate=$5; recruitment=$6
out="$root/bonded-equilibrium"
"$py" "$out/prepare.py" "$root" "$work/audit_installed_layer_boundaries.py" "$predicate" >"$out/build.log" 2>&1
"$py" -c 'import shutil; assert shutil.disk_usage(".").free >= 12*2**30'
"$py" "$cavity/run_cavity_motion.py" "$root/material-direction/release/parent-2" "$out/owner" "$cavity/typed-fans/six-margin.npz" "$out/three" --steps 3 --trace-root 2 --capture-fem-body-load --recruitment-reference "$recruitment" >"$out/native.log" 2>&1
