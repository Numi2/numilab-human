#!/bin/bash
set -euo pipefail
: "${PY:?}" "${prior:?}" "${app:?}"
partition=${1:?absolute path to retained regional partition}
shared=${2:?shared forward and finite facet assets}
out=${3:?unused destination}
short=${4:?last accepted root}
long=${5:?continuation root count}
mkdir -p "$out"
"$PY" prepare_constitutive_boundary.py "$partition" "$shared" "$out/physical"
"$PY" prepare_motion_registry.py "$out/physical" "$out/registration"
a="$out/registration";b="$out/build"
"$PY" constitutive/build_pelvic_coupling.py "$prior/heart-build" "$a" "$app" "$b"
xcrun clang++ -std=c++17 -Wall -Wextra -Werror -fobjc-arc -O2 -framework Metal -framework Foundation verify_cortical_registry.mm -o "$b/verify-registry"
xcrun clang++ -std=c++17 -Wall -Wextra -Werror -fobjc-arc -O2 -framework Metal -framework Foundation verify_shared_features.mm -o "$b/verify-facets"
lib="$b/metalrobo/HumanRestingSkin.metallib"
"$b/verify-registry" "$lib" >"$b/registry-fixtures.csv"
NUMI_VERIFY_PHYSICAL_P1=1 NUMI_VERIFY_ACTUAL_FACETS=1 "$PY" test_stable_features.py "$b/verify-facets" "$lib" "$a" "$b/native-features-parity.json"
for n in "$long" "$short";do
    z="$out/roots-$n"
    bash run_pose_cohort.sh refreshed "$n" "$a" "$b" "$z" "$PY" "$prior/changed_surface_pairs" "$partition"
    "$PY" audit_constitutive_boundary.py "$partition" "$a" "$z" "$prior/changed_surface_pairs" "$z/constitutive-boundary.json"
    "$PY" audit_accepted_cortical_volume.py "$partition" "$a" "$z/accepted-pelvic-state.csv" "$z/constitutive-volume.json" --surface "$z/accepted-pelvic-surfaces.csv"
    "$PY" audit_forward_samples.py "$a" "$z/accepted-pelvic-surfaces.csv" cortical_distance.dylib "$z/independent-forward.json"
    "$PY" audit_registered_motion_forces.py "$a" "$z" "$z/registered-forces.json"
    "$PY" evaluate_pelvic_trial.py "$z"
done
for f in accepted-pelvic-state.csv accepted-pelvic-surfaces.csv accepted-cortical-registry.csv;do
    cmp "$out/roots-$long/$f" "$out/roots-$short/$f"
done
