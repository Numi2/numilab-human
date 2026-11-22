#!/bin/bash
set -euo pipefail
root=$1; py=$2; cavity=$3; recruitment=$4; work=$5; reader=$6; pred=$7
cd "$root/material-direction"
"$py" compare.py "$root" "$PWD/dynamic/three" "$PWD/coupled/three" "$PWD/release/three-1" "$PWD/release/three-2" >comparison.json
"$py" release/verify.py "$PWD" >release/result.json
if test -s release/three-2/terminal.receipt.json; then
    "$py" "$cavity/check_cavity_endpoint.py" "$PWD/release/three-2" "$work" "$reader" "$pred" --require-clear >release/endpoint.log 2>&1 || test -s release/three-2/verified-endpoint.json
    "$py" "$cavity/measure_cavity_step.py" "$PWD/release/three-2" "$PWD/release/motion.json"
    "$py" "$cavity/archive_cavity_endpoint.py" "$PWD/release/three-2" "$PWD/release/endpoint-recovery.json"
fi
"$py" archive.py "$root" >archive.log 2>&1
