#!/bin/bash
set -euo pipefail
# Run from the existing posterior registration workspace on the pinned Numi checkout.
: "${PY:?}" "${prior:?}" "${prev:?}" "${app:?}"
part=${1:?registered muscle work directory}
build=${2:?unused absolute build path}
short=${3:?qualified endpoint step}
long=${4:?longer candidate window}
"$PY" prepare_shared_facets.py "$part" "$part/certified-rest-surface" "$build-facets"
"$PY" prepare_forward_surface.py "$part" "$prev" "$build-facets" "$build-assets"
"$PY" constitutive/build_pelvic_coupling.py "$prior/heart-build" "$build-assets" "$app" "$build"
xcrun clang++ -std=c++17 -Wall -Wextra -Werror -O2 -fobjc-arc -framework Metal -framework Foundation verify_shared_features.mm -o "$build/verify-shared-features"
NUMI_VERIFY_ACTUAL_FACETS=1 "$PY" test_stable_features.py "$build/verify-shared-features" "$build/metalrobo/HumanRestingSkin.metallib" "$build-assets" "$build/native-features-parity.json"
for n in "$long" "$short";do
 bash run_pose_cohort.sh refreshed "$n" "$build-assets" "$build" "$build-steps-$n" "$PY" "$prior/changed_surface_pairs" "$part"
done
for f in accepted-pelvic-state.csv accepted-pelvic-surfaces.csv;do cmp "$build-steps-$short/$f" "$build-steps-$long/$f";done
"$PY" audit_forward_samples.py "$build-assets" "$build-steps-$short/accepted-pelvic-surfaces.csv" cortical_distance.dylib "$build-steps-$short/independent-forward.json"
