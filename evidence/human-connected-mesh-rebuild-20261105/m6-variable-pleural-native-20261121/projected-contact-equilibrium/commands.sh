#!/bin/bash
set -euo pipefail
# Run in the registered posterior-contact workspace on SSH Numi.
: "${PY:?}" "${prior:?}" "${prev:?}" "${app:?}"
part=${1:?absolute path to the rounded regional partition}
build=${2:?unused absolute build path}
short=${3:?number of accepted 2 ms steps}
long=${4:?requested continuation steps}
bash replay_shared_pressure.sh "$part" "$build" "$short" "$long"
assets="$build-assets"
xcrun clang++ -O2 -std=c++17 -Wall -Wextra -Werror -fobjc-arc -framework Foundation -framework Metal test_pelvic_traction_fit.mm -o "$build/test-pelvic-traction-fit"
nodes=$(jq '[.contact_flexibility[].nodes]|add' "$assets/manifest.json")
"$build/test-pelvic-traction-fit" "$build/metalrobo/HumanRestingSkin.metallib" "$nodes"
xcrun clang++ -O2 -std=c++17 -Wall -Wextra -Werror -fobjc-arc -framework Foundation -framework Metal test_pelvic_dual_increment.mm -o "$build/test-pelvic-dual-increment"
width=$(jq '.contact_flexibility[0].nodes' "$assets/manifest.json")
"$build/test-pelvic-dual-increment" "$build/metalrobo/HumanRestingSkin.metallib" "$width" active
"$PY" test_pelvic_closed_solid.py "$prior/changed_surface_pairs"
for n in "$short" "$long";do
    out="$build-steps-$n"
    "$PY" audit_accepted_cortical_volume.py "$part" "$assets" "$out/accepted-pelvic-state.csv" "$out/constitutive-volume.json" --surface "$out/accepted-pelvic-surfaces.csv"
    "$PY" evaluate_pelvic_trial.py "$out"
done
"$PY" collect_equilibrium_study.py "$build-steps-$short" "$build-steps-$long" "$build/measured-comparison.json"
