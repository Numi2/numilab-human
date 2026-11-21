#!/bin/bash
set -euo pipefail
: "${PY:?}" "${prior:?}" "${app:?}"
part=${1:?absolute path to reference partition}
base=${2:?shared forward and finite facet assets}
out=${3:?unused build and evidence path}
rows=${4:?optional retained swept-face list or none}
short=${5:?dedicated horizon steps}
long=${6:?continuation horizon steps}
mkdir -p "$out"
"$PY" prepare_constitutive_boundary.py "$part" "$base" "$out/physical-assets"
assets="$out/physical-assets"
if [[ "$rows" != none ]];then
    "$PY" compile_swept_contact.py "$assets" "$rows" "$out/covered-assets"
    assets="$out/covered-assets"
fi
"$PY" constitutive/build_pelvic_coupling.py "$prior/heart-build" "$assets" "$app" "$out/build"
xcrun clang++ -std=c++17 -Wall -Wextra -Werror -fobjc-arc -O2 -framework Metal -framework Foundation verify_shared_features.mm -o "$out/verify-facets"
NUMI_VERIFY_PHYSICAL_P1=1 NUMI_VERIFY_ACTUAL_FACETS=1 "$PY" test_stable_features.py "$out/verify-facets" "$out/build/metalrobo/HumanRestingSkin.metallib" "$assets" "$out/facet-fixtures.json"
"$PY" build_split_fixture.py constitutive "$out/signs"
xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math -I constitutive -I "$out/signs" -c "$out/signs/test-split.metal" -o "$out/signs/test.air"
xcrun -sdk macosx metallib "$out/signs/test.air" -o "$out/signs/test.metallib"
xcrun clang++ -O2 -std=c++17 -Wall -Wextra -Werror -fobjc-arc -framework Foundation -framework Metal "$out/signs/verify.mm" -o "$out/signs/verify"
"$PY" verify_split_geometry.py "$out/signs/verify" "$out/signs/test.metallib" "$out/signs/qualification.json"
for n in "$long" "$short";do
    run="$out/steps-$n"
    bash run_pose_cohort.sh refreshed "$n" "$assets" "$out/build" "$run" "$PY" "$prior/changed_surface_pairs" "$part"
    "$PY" audit_constitutive_boundary.py "$part" "$assets" "$run" "$prior/changed_surface_pairs" "$run/constitutive-boundary.json"
    "$PY" audit_accepted_cortical_volume.py "$part" "$assets" "$run/accepted-pelvic-state.csv" "$run/constitutive-volume.json" --surface "$run/accepted-pelvic-surfaces.csv"
    "$PY" audit_forward_samples.py "$assets" "$run/accepted-pelvic-surfaces.csv" cortical_distance.dylib "$run/independent-forward.json"
    "$PY" evaluate_pelvic_trial.py "$run"
done
for f in accepted-pelvic-state.csv accepted-pelvic-surfaces.csv;do cmp "$out/steps-$short/$f" "$out/steps-$long/$f";done
