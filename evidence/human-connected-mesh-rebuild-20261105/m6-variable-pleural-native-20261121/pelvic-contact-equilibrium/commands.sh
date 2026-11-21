#!/bin/bash
set -euo pipefail
work="$HOME/Work/human-integrated-gaps-20261104/resting-clearance-and-atrium-017/extended-equilibrium-027/pelvic-contact-response-048"
cd "$work"
e=$(dirname "$work")
part="$e/pelvic-material-interface-047"
prev="$e/material-bed-equilibrium-046"
app="$HOME/Work/skin-fat-native-integration-209"
law="$app/matter/materials/passive_skeletal_muscle_unqualified.nmatter"
ref="$prev/native-release-motion/step-8.receipt.json"
base="$e/native-material-ligament-044"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
# Source scripts from this revision belong in the posterior registration tool directory.
export PYTHONPATH="$work:$part:$(/usr/bin/python3 -c 'import json,sys;print(":".join(json.load(open(sys.argv[1]))["authoring_pythonpath"]))' "$part/tool-versions.json"):${PYTHONPATH:-}"
"$py" repair_cut_carrier.py "$part" "$prev/posterior" carrier-repair 30 1557
mkdir -p repartition
if test ! -e repartition/partition-layers;then ln -s "$part/partition-layers" repartition/partition-layers;fi
"$py" "$part/project_material_cells.py" "$prev/native-release-motion" carrier-repair "$part/partition-layers" "$law" repartition/projected
"$py" "$part/bind_material_ports.py" "$prev/release-posterior/config.json" "$ref" "$prev/native-release-motion" "$base" "$part/partition-layers" repartition/projected repartition/ports
"$py" "$part/prepare_pelvic_mass.py" "$prev/release-posterior/config.json" "$ref" repartition/projected "$law" repartition/mass
"$py" "$part/test_pelvic_port_operator.py" "$part/partition-layers" repartition/projected repartition/ports "$law" repartition/operator-checks.json
"$py" pelvic_elastic_response.py repartition "$prev" "$law" repartition/response .022
"$py" prepare_native_contact.py repartition "$prev" "$base" repartition/response repartition/native-assets
"$py" prepare_native_cells.py repartition "$prev" repartition/response "$law" repartition/native-assets
clang++ -std=c++17 -O3 -Wall -Wextra -Werror -shared -fPIC cortical_distance.cpp -o cortical_distance.dylib
"$py" prepare_contact_pool.py repartition "$prev" repartition/response repartition/native-assets repartition/pool-assets cortical_distance.dylib
"$py" enrich_cortical_modes.py repartition repartition/response repartition/pool-assets "$ref" "$law" cortical_distance.dylib repartition/enriched
"$py" prepare_native_contact.py repartition "$prev" "$base" repartition/enriched repartition/enriched-assets
"$py" prepare_native_cells.py repartition "$prev" repartition/enriched "$law" repartition/enriched-assets
"$py" prepare_contact_pool.py repartition "$prev" repartition/enriched repartition/enriched-assets repartition/enriched-pool cortical_distance.dylib
"$py" factor_contact_columns.py repartition repartition/response repartition/enriched-pool "$ref" "$law" repartition/exact-assets
"$py" build_pelvic_coupling.py "$e/anatomical-extensor-and-bed-045/heart-build" "$work/repartition/exact-assets" "$app" "$work/repartition/exchange-build"
for steps in 32 8000;do
  tag=exchange-32
  if test "$steps" = 8000;then tag=exchange-16s;fi
  run="$work/repartition/$tag"
  bash run_cortical_trial.sh "$prev/supported-release/config.json" "$work/repartition/exact-assets" "$work/repartition/exchange-build/numi-human-native" "$run" "$py" "$steps" 1 || test -f "$run/failure.json"
  "$py" audit_accepted_cortical_volume.py repartition repartition/exact-assets "$run/accepted-pelvic-state.csv" "$run/solid-overlap.json"
  "$py" audit_pelvic_equilibrium.py repartition repartition/response repartition/exact-assets "$ref" "$law" "$run/accepted-pelvic-state.csv" "$run/force-equilibrium.json"
  "$py" check_omitted_points.py repartition "$prev" repartition/exact-assets "$run/accepted-pelvic-state.csv" cortical_distance.dylib "$run/omitted-points.json"
  "$py" evaluate_pelvic_trial.py "$run"
done
