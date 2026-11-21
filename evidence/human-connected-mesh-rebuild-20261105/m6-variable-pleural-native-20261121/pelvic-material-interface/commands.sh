#!/bin/bash
set -euo pipefail
# Run on the retained Numi checkout with mounted parents from the hashed native launch.
W="$HOME/Work/human-integrated-gaps-20261104"
E="$W/resting-clearance-and-atrium-017/extended-equilibrium-027"
parent="$E/material-bed-equilibrium-046"
base="$E/native-material-ligament-044"
prior="$E/anatomical-extensor-and-bed-045"
owner="$HOME/Work/skin-fat-native-integration-209"
material="$owner/matter/materials/passive_skeletal_muscle_unqualified.nmatter"
entry="$parent/native-release-motion"
cfg="$parent/release-posterior/config.json"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
# Source directories use the same existing PYTHONPATH as the parent posterior registration replay.
manifest="$(cd "$(dirname "$0")" && pwd)/tool-versions.json"
export PYTHONPATH="$(/usr/bin/python3 -c 'import json,sys;print(":".join(json.load(open(sys.argv[1]))["authoring_pythonpath"]))' "$manifest"):${PYTHONPATH:-}"
"$py" prepare_material_interface.py "$cfg" "$entry/step-8.receipt.json" "$entry" "$base" partition-layers
"$py" certify_partition.py "$cfg" "$entry/step-8.receipt.json" "$entry" "$base" partition-layers "$prior/changed_surface_pairs" intersections.json
"$py" project_material_cells.py "$entry" "$parent/posterior" partition-layers "$material" projected
"$py" bind_material_ports.py "$cfg" "$entry/step-8.receipt.json" "$entry" "$base" partition-layers projected ports
"$py" prepare_pelvic_mass.py "$cfg" "$entry/step-8.receipt.json" projected "$material" mass
"$py" test_pelvic_port_operator.py partition-layers projected ports "$material" operator-checks.json
"$py" qualify_material_motion.py "$entry" "$cfg" "$base" "$parent/supported-release-60" projected ports "$material" motion
"$py" build_pelvic_preflight.py "$parent/moment-owner" "$PWD/native-owner" "$owner"
bash run_native_mass.sh "$cfg" "$PWD/native-mass" "$PWD/mass/mass-reference.txt" "$PWD/native-owner" "$py"
"$py" check_native_partition.py "$entry" native-mass mass/result.json native-mass/comparison.json
