#!/bin/bash
set -euo pipefail
# Run in the retained Numi authoring workspace with the parent source locations in the launch.
: "${PY:?}" "${prior:?}" "${prev:?}" "${p47:?}" "${w44:?}" "${app:?}"
work=$PWD
cap=$prev/native-release-motion
ref=$cap/step-8.receipt.json
cfg=$prev/release-posterior/config.json
law=$app/matter/materials/passive_skeletal_muscle_unqualified.nmatter
prefix=$work/repartition/replay-rounded
part=$prefix/work
mkdir -p "$part"
"$PY" prepare_rounded_partition.py "$cfg" "$ref" "$cap" "$w44" "$prefix/partition"
ln -s "$prefix/partition" "$part/partition-layers"
"$PY" "$p47/project_material_cells.py" "$cap" carrier-repair "$prefix/partition" "$law" "$part/projected"
"$PY" bind_separated_ports.py "$cfg" "$ref" "$cap" "$w44" "$prefix/partition" "$part/projected" "$part/ports"
"$PY" "$p47/prepare_pelvic_mass.py" "$cfg" "$ref" "$part/projected" "$law" "$part/mass"
"$PY" close_cut_surface.py "$part" "$prev" "$part/conforming-boundary"
"$PY" pelvic_elastic_response.py "$part" "$prev" "$law" "$part/response" .022
bash compile_rounded_native_assets.sh "$part" "$prev" "$w44" "$app" "$PY"
for kind in raw compiled; do
  args=();if test "$kind" = raw; then args+=("$prefix/partition"); fi
  "$PY" audit_unloaded_interface.py "$part" "$part/native-features" "$cfg" "$ref" "$cap" "$w44" "$prior/changed_surface_pairs" "$prefix/$kind-crossings.json" ${args[@]+"${args[@]}"}
done
"$PY" constitutive/build_pelvic_coupling.py "$prior/heart-build" "$part/native-features" "$app" "$prefix/build"
"$PY" test_stable_features.py "$work/verify-stable-features" "$prefix/build/metalrobo/HumanRestingSkin.metallib" "$part/native-features" "$prefix/build/stable-feature-parity.json"
for steps in 1 8 64; do
  bash run_pose_cohort.sh refreshed "$steps" "$part/native-features" "$prefix/build" "$prefix/steps-$steps" "$PY" "$prior/changed_surface_pairs" "$part"
done
