#!/bin/bash
set -euo pipefail
: "${PY:?}" "${prior:?}" "${prev:?}" "${app:?}"
part=${1:?rebuilt physical regional directory}
build=${2:?unused build directory}
assets=$build-assets
src=$PWD
"$PY" prepare_material_surface_reference.py "$part" "$part/native-features" "$prev/native-release-motion/step-8.receipt.json" "$assets-material"
"$PY" prepare_cortical_certificate.py "$assets-material" "$assets"
"$PY" constitutive/build_pelvic_coupling.py "$prior/heart-build" "$assets" "$app" "$build"
"$PY" test_stable_features.py "$src/verify-stable-features" "$build/metalrobo/HumanRestingSkin.metallib" "$assets" "$build/stable-feature-parity.json"
xcrun -sdk macosx metal -fno-fast-math -std=metal3.1 -I constitutive -c test_geometry_signs.metal -o "$build/verify.air"
xcrun -sdk macosx metallib "$build/verify.air" -o "$build/verify.metallib"
xcrun clang++ -Wall -Wextra -Werror -O2 -std=c++17 -fobjc-arc -framework Metal -framework Foundation verify_geometry_signs.mm -o "$build/check-signs"
"$PY" test_geometry_signs.py "$build/check-signs" "$build/verify.metallib" "$build/predicate-parity.json"
for n in 4 8;do
  bash run_pose_cohort.sh refreshed "$n" "$assets" "$build" "$build-steps-$n" "$PY" "$prior/changed_surface_pairs" "$part"
done
cmp "$build-steps-4/accepted-pelvic-state.csv" "$build-steps-8/accepted-pelvic-state.csv"
cmp "$build-steps-4/accepted-pelvic-surfaces.csv" "$build-steps-8/accepted-pelvic-surfaces.csv"
