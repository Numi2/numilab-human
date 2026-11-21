#!/bin/bash
set -euo pipefail
# Run in the pinned Numi authoring workspace with the dependencies from the parent checkpoint.
: "${PY:?Numi authoring interpreter}"
: "${prior:?pinned anatomical-extensor-and-bed-045 directory}"
: "${app:?pinned skin-fat-native-integration-209 checkout}"
work=$PWD
"$PY" prepare_feature_facets.py repartition/facet-assets repartition/replay-feature-assets
"$PY" constitutive/build_pelvic_coupling.py "$prior/heart-build" "$work/repartition/replay-feature-assets" "$app" "$work/repartition/replay-feature-owner-build"
xcrun clang++ -std=c++17 -Wall -Wextra -Werror -O2 -fobjc-arc -framework Metal -framework Foundation verify_stable_features.mm -o verify-stable-features
"$PY" test_stable_features.py "$work/verify-stable-features" "$work/repartition/replay-feature-owner-build/metalrobo/HumanRestingSkin.metallib" repartition/replay-feature-assets repartition/replay-feature-owner-build/stable-feature-parity.json
for n in 60 256;do
    bash run_pose_cohort.sh refreshed "$n" "$work/repartition/replay-feature-assets" "$work/repartition/replay-feature-owner-build" "$work/repartition/replayed-features-$n" "$PY" "$prior/changed_surface_pairs"
done
