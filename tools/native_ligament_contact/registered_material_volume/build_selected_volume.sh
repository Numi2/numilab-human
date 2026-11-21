#!/bin/bash
set -euo pipefail
if [[ $# -ne 6 ]]; then
    echo 'Usage: build_selected_volume.sh PYTHON APP BASELINE_DIR TILED_DIGIT_BUILD SELECTED_LAUNCH OUTPUT' >&2
    exit 2
fi
py=$1
app=$2
base=$3
digit=$4
selection=$5
out=$6
scripts=$(cd "$(dirname "$0")" && pwd)
case "$out" in /*) ;; *) echo 'Use an absolute output path' >&2; exit 2 ;; esac
test ! -e "$out"
mkdir -p "$out"
old="$(dirname "$base")/resting-clearance-and-atrium-010"
ref="$base/refit-motion-qualification"
work="$out/registration"
full="$base/extended-equilibrium-027/native-candidate-restored-24"
export PYTHONPATH="$(dirname "$base")/resting-clearance-and-atrium-006:$old${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd "$scripts"
"$py" capture_selected_skin_reference.py "$old/aligned-reference-capture-launch.json" "$selection" "$old/frozen-source-frame.csv" "$work"
"$py" "$old/rebind_current_skin.py" --owner "$app" --identity "$work/identity.json" --frozen-capture "$old/frozen-source-frame.csv" --current-capture "$work/source-frame.csv" --out "$work/rebound"
(
    cd "$work"
    ln -s rebound skin-source-rebind-002
    for job in refit_paired_skin_fat finalize_paired_registration remap_skin_patch_parents; do "$py" "$old/$job.py"; done
)
"$py" emit_skin_source_registration.py "$work" "$old" "$ref/p1/current-paired-volume-state.json" "$work/native"
"$py" build_registered_source.py "$digit" "$work/native" "$out/current-full-source-build" "$app"
"$py" prepare_registered_seed_launch.py "$full/launch.json" "$full/stdout.log" "$work/registered-seed-launch.json"
"$py" enable_volume_first_touch.py "$out/current-full-source-build" "$out/fine-first-touch-build" "$app"
"$py" build_paired_material_source.py "$out/fine-first-touch-build" "$work" "$out/paired-cell-source-build" "$app"
"$py" restore_paired_role_replay.py "$out/paired-cell-source-build" "$base/floor-scene-build" "$out/paired-mass-source-build" "$app"
"$py" capture_current_parent.py "$out/paired-mass-source-build" "$out/paired-mass-export-build" "$app" "$base/floor-live-parent-capture-002/launch.json" "$selection" "$work/registered-seed-launch.json" "$work/current-parent-capture"
"$py" mass_and_cell_transfer.py "$base" "$work/current-parent-capture" "$work/fine-mass-routed"
"$py" rebind_physical_volume.py "$base" "$work" "$work/material"
"$py" replay_current_carrier_witnesses.py "$base" "$work" "$work/material"
"$py" run_selected_mass_replay.py "$base" "$work"
"$py" prepare_live_volume_install.py "$work/registered-seed-launch.json" "$base" "$work" "$work/current-mass-launch.json"
"$py" build_installed_volume_motion.py "$out/paired-mass-source-build" "$out/live-registered-volume-build" "$app"
bash run_volume_admission.sh "$work/current-mass-launch.json" "$selection" "$out/live-registered-volume-build/numi-human-native" "$out/native-preparation" "$py"
