#!/bin/bash
set -euo pipefail
: "${PY:?}" "${prior:?}" "${app:?}"
a=${1:?registered motion data}
capture=${2:?retained native terminal evidence directory}
p=${3:?original partition}
out=${4:?unused output directory}
n=${5:?native horizon in two millisecond steps}
test ! -e "$out"
mkdir -p "$out"
"$PY" prepare_block_factor.py "$a" "$out/panels"
"$PY" prepare_runtime_panels.py "$a" "$out/panels" "$out/assets"
"$PY" panel_fixture_inputs.py "$a" "$out/panels" "$capture" "$out/coupon"
xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math -c "$out/panels/bench.metal" -o "$out/panels/bench.air"
xcrun -sdk macosx metallib "$out/panels/bench.air" -o "$out/panels/standalone.metallib"
xcrun clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wextra -Werror -framework Foundation -framework Metal compare_pelvic_factors.mm -o "$out/panels/compare"
"$PY" constitutive/build_pelvic_coupling.py "$prior/heart-build" "$a" "$app" "$out/original-build"
"$PY" constitutive/build_pelvic_coupling.py "$prior/heart-build" "$out/assets" "$app" "$out/panel-build"
for k in accepted random localized; do
    f="$out/coupon/$k"
    "$out/panels/compare" "$out/original-build/metalrobo/HumanRestingSkin.metallib" "$out/panels/standalone.metallib" "$a" "$out/panels" "$f/rhs.bin" "$f" >"$f/timing.csv"
done
"$PY" audit_pelvic_panels.py "$a" "$out/panels" "$out/coupon"
for owner in original panel; do
    data="$a";if [[ "$owner" = panel ]]; then data="$out/assets";fi
    z="$out/$owner"
    bash run_pose_cohort.sh refreshed "$n" "$data" "$out/$owner-build" "$z" "$PY" "$prior/changed_surface_pairs" "$p"
    "$PY" audit_constitutive_boundary.py "$p" "$data" "$z" "$prior/changed_surface_pairs" "$z/constitutive-boundary.json"
    "$PY" audit_accepted_cortical_volume.py "$p" "$data" "$z/accepted-pelvic-state.csv" "$z/constitutive-volume.json" --surface "$z/accepted-pelvic-surfaces.csv"
    "$PY" evaluate_pelvic_trial.py "$z"
done
"$PY" compare_motion_factors.py "$out/original" "$out/panel" "$out/comparison.json"
