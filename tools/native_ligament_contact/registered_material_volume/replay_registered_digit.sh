#!/bin/bash
set -euo pipefail
case_dir=$1
parent=$2
work=$3
base=$4
py=$5
here=$(cd "$(dirname "$0")" && pwd)
compiler="$base/../captured-fat-sliver-495/proposal/compiler"
"$py" "$here/rebuild_torso_native_role_support.py" "$base" "$parent" "$case_dir/proposal.npz" "$case_dir" --prior-mesh "$parent/mesh" >"$case_dir/p1.log" 2>&1
cmp "$case_dir/reference.node" "$case_dir/mesh/mesh.node"
"$py" "$work/bind_torso_primary_groups.py" "$base" "$case_dir" "$case_dir/proposal.npz" "$case_dir/mesh/current-child.labels.csv" >"$case_dir/labels.log" 2>&1
"$py" "$work/emit_torso_registered_surfaces.py" "$parent/launch.json" "$parent" "$case_dir/proposal.npz" "$case_dir" >"$case_dir/surface.log" 2>&1
record="$compiler/mass-allocation/run-manifest.json"
frame=$(jq -er '.argv[(.argv|index("--coordinate-frame-sha256"))+1]' "$record")
body=$(jq -er '.argv[(.argv|index("--source-body-count"))+1]' "$record")
reg="$case_dir/material/registration"
NUMI_HUMAN_VERIFY_VOLUME_FLOOR_ONLY="$case_dir/replayed.volumes" "$base/verify_repaired_registered_volume" "$reg" "$compiler" "$base/paired-native-parent-ledger/compiled-parent-cell-mass.csv" "$case_dir/mesh" "$case_dir/mesh/current-child.labels.csv" "$reg/facet-ancestry-current.csv" "$frame" "$body" "$reg/source-domain-pins.tsv" "$reg/all-body-current-rest.csv" b1704fb7b846254583f4b6a08c394ed3cad1603554244078706ca4b3811ac674 >"$case_dir/import-volume.log" 2>&1
cmp "$case_dir/replayed.volumes" "$case_dir/native.volumes"
"$py" "$work/replay_current_carrier_witnesses.py" "$base" "$case_dir" "$case_dir/material" >"$case_dir/carrier.log" 2>&1
"$py" "$work/run_selected_mass_replay.py" "$base" "$case_dir" "$case_dir/mesh" "$case_dir/mesh/current-child.labels.csv" >"$case_dir/role.log" 2>&1
"$py" "$work/install_registered_torso_volume.py" "$parent/launch.json" "$parent/material" "$case_dir" "$case_dir/launch.json" >"$case_dir/install.log" 2>&1
