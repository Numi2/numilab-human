#!/bin/bash
set -euo pipefail
# Run against the same pinned full-volume launch used by the native constructor.
here=$(cd "$(dirname "$0")" && pwd)
launch=$1
predicate_sources=$2
app=$3
out=$4
py=$5
mkdir -p "$out"
echo "6c05956f6b3348e8d71c668099ee42e7dc29e2325335f4b06df821b09e524f98  $app/apps/NumiHumanRegisteredSurfaceIntersection.hpp" | shasum -a 256 -c -
clang++ -std=c++17 -O3 -ffp-contract=off -I"$app/apps" -dynamiclib "$predicate_sources/boundary_pairs.cpp" -o "$out/predicates.dylib"
export PYTHONPATH="$predicate_sources:$here${PYTHONPATH:+:$PYTHONPATH}"
"$py" "$here/audit_registered_material_closure.py" "$launch" "$out/cell-boundaries.json"
"$py" "$here/audit_installed_layer_boundaries.py" "$launch" "$out/predicates.dylib" "$out/closed-boundaries.json"
