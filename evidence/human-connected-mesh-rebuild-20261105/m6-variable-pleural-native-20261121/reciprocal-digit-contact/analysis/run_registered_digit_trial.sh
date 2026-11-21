#!/bin/bash
set -euo pipefail
z=$1
steps=$2
build=$3
p="$PWD/repartition"; a="$p/full-surface-force-map";work="$PWD"
cfg="$z-config";export W PY prior
"$PY" prepare_digit_trial.py "$p/with-tiled-forward" "$work/digit-contact" "$z" "$cfg"
bash multicapture_cortical_trial.sh "$cfg/config.json" "$a" "$build/numi-human-native" "$z" "$PY" "$steps" 1 "" "$work/inspect_registered_digit_endpoint.py" "$W/resting-surface-clearance-001/audit.py" || test -f "$z/failure.json"
if test -s "$z/accepted-pelvic-state.csv";then
 c="$p/rounded-work"
 "$PY" audit_complete_pelvic_forces.py "$c" "$c/response" "$a" "$z/accepted-pelvic-state.csv" "$z/bulk-force.json"
 "$PY" audit_finite_cortical_crossings.py "$a" "$z/accepted-pelvic-surfaces.csv" "$prior/changed_surface_pairs" "$z/native-crossings.json"
 "$PY" audit_accepted_cortical_volume.py "$c" "$a" "$z/accepted-pelvic-state.csv" "$z/solid-overlap.json"
 "$PY" audit_constitutive_boundary.py "$c" "$a" "$z" "$prior/changed_surface_pairs" "$z/constitutive-boundary.json"
 "$PY" audit_accepted_cortical_volume.py "$c" "$a" "$z/accepted-pelvic-state.csv" "$z/constitutive-volume.json" --surface "$z/accepted-pelvic-surfaces.csv"
 "$PY" evaluate_pelvic_trial.py "$z" >/dev/null
fi
