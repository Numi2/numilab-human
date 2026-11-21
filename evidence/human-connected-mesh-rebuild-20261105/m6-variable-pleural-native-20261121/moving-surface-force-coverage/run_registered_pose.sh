#!/bin/bash
set -euo pipefail
cfg=$1
z=$2
steps=$3
p="$PWD/repartition/rounded-work"
a=${5:-"$p/motion-p1-boundary"}
b=${4:-"$PWD/repartition/motion-registry-build"}
bash run_cortical_trial.sh "$cfg/config.json" "$a" "$b/numi-human-native" "$z" "$PY" "$steps" 1 || test -f "$z/failure.json"
if ! test -s "$z/accepted-pelvic-state.csv"; then exit 0; fi
"$PY" audit_complete_pelvic_forces.py "$p" "$p/response" "$a" "$z/accepted-pelvic-state.csv" "$z/bulk-force.json"
"$PY" audit_finite_cortical_crossings.py "$a" "$z/accepted-pelvic-surfaces.csv" "$prior/changed_surface_pairs" "$z/native-crossings.json"
"$PY" audit_accepted_cortical_volume.py "$p" "$a" "$z/accepted-pelvic-state.csv" "$z/solid-overlap.json"
"$PY" audit_constitutive_boundary.py "$p" "$a" "$z" "$prior/changed_surface_pairs" "$z/constitutive-boundary.json"
"$PY" audit_accepted_cortical_volume.py "$p" "$a" "$z/accepted-pelvic-state.csv" "$z/constitutive-volume.json" --surface "$z/accepted-pelvic-surfaces.csv"
"$PY" evaluate_pelvic_trial.py "$z" >/dev/null
