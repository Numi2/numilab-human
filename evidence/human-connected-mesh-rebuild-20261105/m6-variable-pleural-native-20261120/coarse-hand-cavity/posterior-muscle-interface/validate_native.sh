#!/bin/bash
set -euo pipefail
W="$HOME/Work/human-integrated-gaps-20261104"
T="$W/coarse-hand-cavity-024"
V="$T/posterior-gluteal-continuum-041"
J="$T/posterior-load-path-042"
P="$T/posterior-muscle-interface-045"
PY="$HOME/numi-human-prep-venv-20261005/bin/python3"
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$T:$T/posterior-load-interface-040:$V:$J:$W/resting-clearance-and-atrium-017/floor-scene-build:$W/resting-clearance-and-atrium-006:$HOME/Work/skin-fat-native-integration-209/tools"
cd "$P"
run=paired-native-128
[[ -s "$run/execution.json" ]]
[[ -s "$run/verified-mass.json" ]] || "$PY" "$V/verify_transfer.py" "$V/native-candidate-128" paired-06 "$T" "$run/verified-mass.json" --native "$run"
[[ -s "$run/verified-cells.json" ]] || "$PY" check_native_endpoint.py "$run" "$V/native-candidate-128" "$run/verified-cells.json"
"$PY" "$J/audit_shoulder_prefix.py" "$run/launch.json" "$W/resting-clearance-and-atrium-017/fixed-shoulder-anchors.json" "$run/native/resting-body-motion-partition.csv" "$run/verified-shoulders.json" --steps 1 32 64 96 128
"$PY" "$J/audit_boundary_bones.py" "$run" "$T/thickness-rebuild-034/inputs/physical-bone-initial.bin" "$run/bone-endpoints.json" --step 128 > "$run/bone-endpoints.log"
