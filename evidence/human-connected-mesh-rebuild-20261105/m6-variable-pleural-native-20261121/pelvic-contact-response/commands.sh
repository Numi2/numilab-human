#!/bin/bash
set -euo pipefail
root="$HOME/Work/human-integrated-gaps-20261104/resting-clearance-and-atrium-017/extended-equilibrium-027"
prev="$root/material-bed-equilibrium-046"
p47="$root/pelvic-material-interface-047"
work="$root/pelvic-contact-response-048"
base="$root/native-material-ligament-044"
parent="$root/anatomical-extensor-and-bed-045/heart-build"
app="$HOME/Work/skin-fat-native-integration-209"
material="$app/matter/materials/passive_skeletal_muscle_unqualified.nmatter"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
cd "$work"
export PYTHONPATH="$p47:$(/usr/bin/python3 -c 'import json,sys;print(":".join(json.load(open(sys.argv[1]))["authoring_pythonpath"]))' "$p47/tool-versions.json"):${PYTHONPATH:-}"
clang++ -std=c++17 -O3 -Wall -Wextra -Werror -shared -fPIC cortical_distance.cpp -o cortical_distance.dylib
"$py" pelvic_elastic_response.py "$p47" "$prev" "$material" response .022
"$py" prepare_native_contact.py "$p47" "$prev" "$base" response native-assets
"$py" prepare_native_cells.py "$p47" "$prev" response "$material" native-assets
"$py" build_pelvic_coupling.py "$parent" "$work/native-assets" "$app" "$work/feature-build"
bash run_cortical_trial.sh "$prev/supported-release/config.json" "$work/native-assets" "$work/feature-build/numi-human-native" "$work/feature-32" "$py" 32 1
bash run_cortical_trial.sh "$prev/supported-release/config.json" "$work/native-assets" "$work/feature-build/numi-human-native" "$work/contact-off-32-replay" "$py" 32 0
"$py" audit_accepted_cortical_volume.py "$p47" native-assets feature-32/accepted-pelvic-state.csv feature-32/solid-overlap.json
"$py" evaluate_pelvic_trial.py feature-32 contact-off-32-replay
bash run_cortical_trial.sh "$prev/supported-release/config.json" "$work/native-assets" "$work/feature-build/numi-human-native" "$work/feature-16s" "$py" 8000 1
