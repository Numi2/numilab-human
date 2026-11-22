#!/bin/bash
set -euo pipefail
root=$1; py=$2; cav=$3; work=$4; reader=$5; predicate=$6; recruit=$7
case="$root/bonded-equilibrium"; dir="$case/held-volume"; native="$dir/three"
"$py" "$case/opening.py" "$dir" "$reader" "$work" --executed "$native" --scan >"$dir/executed-envelope.log" 2>&1
# If motion rejects, the eye-display source identities still come from the
# identical opening pack; the accepted capture is retained and SHA-checked.
window="$native/native-traces/resting-ocular-sclera-window.json"
if ! test -s "$window"; then window="$case/free-interface/three/native-traces/resting-ocular-sclera-window.json"; fi
"$py" "$work/classify_registered_envelope.py" "$dir/executed-envelope/initial-envelope.json" "$native/stdout.log" "$dir/executed-envelope/physical-banks.json" --window-receipt "$window" >"$dir/executed-classification.log"
bash "$case/qualify_held.sh" "$root" "$py" "$cav" "$work" "$reader" "$predicate" "$recruit"
for end in "$native" "$dir/accepted-2" "$dir/accepted-1"; do
    if test -s "$end/verified-endpoint.json"; then
        "$py" "$case/compare_opening.py" "$dir" "$end" >"$dir/executed-to-terminal.json"
        "$py" "$case/bank_motion.py" "$dir" "$end" "$reader" >"$dir/bank-motion.json"
        break
    fi
done
