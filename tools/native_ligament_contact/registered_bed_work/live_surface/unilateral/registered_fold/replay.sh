#!/bin/bash
set -euo pipefail
# All paths are to the restored, pinned owners and run evidence on SSH Numi.
root=$1; parent_material=$2; digit=$3; work=$4; base=$5; cavity=$6; recruitment=$7; py=$8
case_dir="$root/fold-repair/material"
cd "$root"
"$py" fold-repair/fit.py feasible-bed "$parent_material/fine-mass-routed" fold-repair/source.json fold-repair/recheck.json --route-mass --connected --joint --replay fold-repair/joint.json
bash fold-repair/compile-registration.sh "$root" "$parent_material" "$digit" "$work" "$base" "$py"
"$py" fold-repair/compile_mass.py "$root/feasible-bed/two" "$parent_material" "$case_dir"
"$py" "$work/replay_current_carrier_witnesses.py" "$base" "$case_dir" "$case_dir/material"
"$py" "$work/run_selected_mass_replay.py" "$base" "$case_dir" "$case_dir/mesh" "$case_dir/mesh/current-child.labels.csv"
mkdir -p fold-repair/parent
"$py" "$work/install_registered_torso_volume.py" "$root/feasible-bed/parent/launch.json" "$parent_material/material" "$case_dir" "$root/fold-repair/parent/launch.json"
"$py" fold-repair/build_owner.py "$root/fold-repair"
"$py" "$cavity/run_cavity_motion.py" "$root/fold-repair/parent" "$root/fold-repair/owner" "$cavity/typed-fans/six-margin.npz" "$root/fold-repair/three" --steps 3 --trace-root 2 --capture-fem-body-load --recruitment-reference "$recruitment"
