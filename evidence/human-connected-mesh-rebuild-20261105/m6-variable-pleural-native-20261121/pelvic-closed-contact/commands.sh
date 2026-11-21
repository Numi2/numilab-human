#!/bin/bash
set -euo pipefail
# Run on SSH Numi after the parent revision's authoring stage through enriched-pool.
work="$HOME/Work/human-integrated-gaps-20261104/resting-clearance-and-atrium-017/extended-equilibrium-027/pelvic-contact-response-048"
cd "$work"
e=$(dirname "$work")
part="$e/pelvic-material-interface-047"
prev="$e/material-bed-equilibrium-046"
prior="$e/anatomical-extensor-and-bed-045"
app="$HOME/Work/skin-fat-native-integration-209"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
ref="$prev/native-release-motion/step-8.receipt.json"
law="$app/matter/materials/passive_skeletal_muscle_unqualified.nmatter"
export PYTHONPATH="$work:$part:$(/usr/bin/python3 -c 'import json,sys;print(":".join(json.load(open(sys.argv[1]))["authoring_pythonpath"]))' "$part/tool-versions.json"):${PYTHONPATH:-}"
"$py" prepare_cortical_vertex_faces.py repartition "$prev" repartition/enriched-pool repartition/face-geometry
"$py" factor_contact_columns.py repartition repartition/response repartition/face-geometry "$ref" "$law" repartition/face-assets
"$py" close_cut_surface.py repartition "$prev" repartition/conforming-boundary
"$py" prepare_reverse_surface.py repartition repartition/conforming-boundary repartition/enriched-pool repartition/closed-geometry
"$py" use_registered_force_columns.py repartition/face-assets repartition/closed-geometry repartition/closed-qualified
"$py" prepare_free_residual.py repartition repartition/response repartition/closed-qualified "$ref" "$law" repartition/certified-assets
"$py" build_pelvic_coupling.py "$prior/heart-build" "$work/repartition/certified-assets" "$app" "$work/repartition/replay-certified-build"
run="$work/repartition/replay-certified"
test ! -e "$run"
bash run_cortical_trial.sh "$prev/supported-release/config.json" "$work/repartition/certified-assets" "$work/repartition/replay-certified-build/numi-human-native" "$run" "$py" 32 1 || test -f "$run/failure.json"
state="$run/accepted-pelvic-state.csv"
"$py" audit_pelvic_equilibrium.py repartition repartition/response repartition/certified-assets "$ref" "$law" "$state" "$run/force-equilibrium.json"
"$py" audit_reverse_clearance.py repartition/certified-assets repartition/conforming-boundary "$state" cortical_distance.dylib "$run/native-reverse.json"
"$py" audit_accepted_cortical_volume.py repartition repartition/certified-assets "$state" "$run/solid-overlap.json"
"$py" count_accepted_cortical_crossings.py repartition/certified-assets repartition/conforming-boundary "$state" "$prior/changed_surface_pairs" "$run/crossings.json"
"$py" evaluate_pelvic_trial.py "$run"
