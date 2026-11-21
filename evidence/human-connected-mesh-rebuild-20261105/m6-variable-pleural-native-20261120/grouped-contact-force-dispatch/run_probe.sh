#!/bin/bash
set -euo pipefail
E=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "${1:?pass the absolute numi-lab checkout path}" && pwd)
out=$(mktemp -d "${TMPDIR:-/tmp}/numi-grouped-contact.XXXXXX")
printf '%s\n' "$out"
build() {
    local label=$1 commit=$2
    local source="$out/$label-src"
    mkdir -p "$source"
    git -C "$repo" archive "$commit" matter/src/matter_kernels.metal matter/src/metal matter/include | tar -xf - -C "$source"
    xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math -I "$source/matter/include" -c "$source/matter/src/matter_kernels.metal" -o "$out/$label.air"
    xcrun -sdk macosx metallib "$out/$label.air" -o "$out/$label.metallib"
}
build control "$(cat "$E/control-commit.txt")"
build candidate "$(cut -d' ' -f1 "$E/matter-commit.txt")"
xcrun -sdk macosx clang++ -O3 -std=c++17 -fobjc-arc \
    -I "$out/candidate-src/matter/include" "$E/measure_grouped_force.mm" \
    -framework Foundation -framework Metal -o "$out/measure-grouped-force"
"$out/measure-grouped-force" "$out/control.metallib" "$out/candidate.metallib"
