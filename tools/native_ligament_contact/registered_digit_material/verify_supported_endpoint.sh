#!/bin/bash
set -euo pipefail
runtime=$1
casepath=$2
runpath=$3
reference=$4
treatment_parent=$5
split=$6
bed=$7
work=$8
allwork=$9
trialdir=${10}
lib=${11}
step=$(jq -er 'select(.exit_code==0) | .accepted_steps' "$runpath/native-outcome.json")
PYTHONPATH="$(dirname "$lib"):$PYTHONPATH" "$runtime" "$work/audit_installed_layer_boundaries.py" "$runpath/launch.json" "$lib" "$runpath/accepted-layers.json" --accepted-nodes "$runpath/physical-reference.node" "$runpath/physical-accepted.node" > "$runpath/accepted-layers.log" 2>&1
"$runtime" "$trialdir/compare_posterior_motion.py" "$runpath" "$casepath" "$reference" "$treatment_parent" "$split" "$bed" "$runpath/physical-response.json" > "$runpath/physical-response.log"
"$runtime" "$work/audit_registered_initial_envelope.py" --terminal "$allwork/resting-surface-clearance-001/audit.py" "$runpath/terminal.mrvpack.zst" "$runpath/native/accepted-geometry/step-$step.receipt.json" "$runpath/admission.json" "$runpath/launch.json" "$runpath/envelope" >"$runpath/envelope.log" 2>&1
"$runtime" "$work/classify_registered_envelope.py" "$runpath/envelope/accepted-envelope.json" "$runpath/stdout.log" "$runpath/envelope/physical-banks.json" --window-receipt "$runpath/native/resting-ocular-sclera-window.json" >> "$runpath/envelope.log" 2>&1
