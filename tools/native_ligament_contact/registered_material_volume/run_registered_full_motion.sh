#!/bin/bash
set -euo pipefail
# The existing full-volume build and registered material ledgers are prerequisites.
here=$(cd "$(dirname "$0")" && pwd)
current_launch=$1
regional_selection=$2
full_release=$3
scene_build=$4
base=$5
app=$6
py=$7
out=$8
mkdir -p "$out/lowered"
base=$(cd "$base" && pwd)
out=$(cd "$out" && pwd)
"$py" "$here/link_registered_material_owner.py" "$scene_build" "$full_release" "$out/paired-host" "$app"
set +e
NUMI_SELECTED_TRIAL_MODE=motion NM_FEM_WRITE_NATIVE_TANGENT_SOURCE_ONLY=1 NM_FEM_WRITE_NATIVE_TANGENT_SOURCE="$out/lowered/generated_stateless_tangent.metalinc" bash "$here/run_volume_admission.sh" "$current_launch" "$regional_selection" "$out/paired-host/numi-human-native" "$out/prepare-lowering" "$py"
rc=$?
set -e
test "$rc" -ne 0
test -s "$out/lowered/generated_stateless_tangent.metalinc"
rg -q 'matter_native_tangent_source prepared=.* eligible=20 materials=20 source_only=1' "$out/prepare-lowering/stderr.log"
"$py" "$here/build_selected_material_specialization.py" "$base" "$out/paired-host" "$out/lowered" "$app" "$out/specialized" "$current_launch"
"$py" "$here/integrate_registered_pleural_owner.py" "$scene_build" "$out/specialized" "$base/floor-scene-build" "$out/whole" "$app"
"$base/thoracic_pressure_gpu_check" "$out/whole/metalrobo/HumanRestingSkin.metallib"
NUMI_SELECTED_TRIAL_MODE=motion bash "$here/run_volume_admission.sh" "$out/specialized/launch.json" "$regional_selection" "$out/whole/numi-human-native" "$out/motion" "$py"
