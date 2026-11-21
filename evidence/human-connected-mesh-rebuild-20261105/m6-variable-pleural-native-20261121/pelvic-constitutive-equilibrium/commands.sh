#!/bin/bash
set -euo pipefail
# Retained Numi workspace, with the parent closed-contact command's certified-assets already prepared.
work="$HOME/Work/human-integrated-gaps-20261104/resting-clearance-and-atrium-017/extended-equilibrium-027/pelvic-contact-response-048"
cd "$work"
e=$(dirname "$work")
part="$e/pelvic-material-interface-047"
prior="$e/anatomical-extensor-and-bed-045"
prev="$e/material-bed-equilibrium-046"
app="$HOME/Work/skin-fat-native-integration-209"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
ref="$prev/native-release-motion/step-8.receipt.json"
export PYTHONPATH="$work:$part:$(/usr/bin/python3 -c 'import json,sys;print(":".join(json.load(open(sys.argv[1]))["authoring_pythonpath"]))' "$part/tool-versions.json"):${PYTHONPATH:-}"
# Place these authoring tools beside the parent revision's posterior_registration scripts.
# The four build inputs in constitutive/ refer to this candidate only.
"$py" prepare_pelvic_quotient.py repartition repartition/response repartition/certified-assets "$ref" repartition/quotient-assets
"$py" prepare_pelvic_reference_frame.py repartition repartition/quotient-assets "$ref" repartition/relative-assets
"$py" constitutive/build_pelvic_coupling.py "$prior/heart-build" "$work/repartition/relative-assets" "$app" "$work/repartition/replay-constitutive-build"
xcrun clang++ -std=c++17 -Wall -Wextra -Werror -O2 -fobjc-arc -framework Metal -framework Foundation verify_pelvic_material_kernel.mm -o verify-pelvic-material
"$py" test_pelvic_material_kernel.py "$work/verify-pelvic-material" "$work/repartition/replay-constitutive-build/metalrobo/HumanRestingSkin.metallib" repartition/replay-constitutive-build/kernel-parity.json
run="$work/repartition/replay-constitutive-1"
test ! -e "$run"
bash run_cortical_trial.sh "$prev/supported-release/config.json" "$work/repartition/relative-assets" "$work/repartition/replay-constitutive-build/numi-human-native" "$run" "$py" 1 1 || test -f "$run/failure.json"
if test -s "$run/accepted-pelvic-state.csv";then
    "$py" audit_complete_pelvic_forces.py repartition repartition/response repartition/relative-assets "$run/accepted-pelvic-state.csv" "$run/bulk-force.json"
fi
