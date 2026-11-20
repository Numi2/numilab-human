#!/bin/bash
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
dir="$(cd -- "$(dirname -- "$0")" && pwd)"
owner="$HOME/Work/skin-fat-native-integration-209"
generated="$dir/complete-fine-p1-current-parent"
state="$generated/current-paired-volume-state.json"
skin="$generated/skin-p1-receipt.json"
fat="$generated/fat-p1-receipt.json"
mass="$(jq -er '.source330_identity.source330_mass.path' "$state")"
labels="$(jq -er '.source330_identity.source330_labels.path' "$state")"
elements="$dir/../resting-clearance-and-atrium-016/repaired-tetgen/mesh.ele"
for role in "$skin" "$fat"; do
    actual="$(shasum -a 256 "$(jq -er .role_geometry_path "$role")" | cut -d ' ' -f 1)"
    test "$actual" = "$(jq -er .role_geometry_sha256 "$role")"
done
/usr/bin/xcrun clang++ -std=c++23 -O3 -Wall -Wextra -Wpedantic -Werror \
    -I"$owner/apps" -I"$owner/include" -I"$owner/matter/include" \
    "$dir/validate_complete_fine_p1.cpp" -lz -o "$dir/validate_complete_fine_p1"
/usr/bin/time -l "$dir/validate_complete_fine_p1" "$mass" "$labels" "$elements" \
    "$(shasum -a 256 "$state" | cut -d ' ' -f 1)" \
    "$(jq -er .source330_identity.source330_mass.sha256 "$state")" \
    "$(jq -er .p1_one_hop_path "$skin")" "$(jq -er .p1_one_hop_sha256 "$skin")" \
    "$(jq -er .role_geometry_sha256 "$skin")" \
    "$(jq -er .p1_one_hop_path "$fat")" "$(jq -er .p1_one_hop_sha256 "$fat")" \
    "$(jq -er .role_geometry_sha256 "$fat")" \
    "$(jq -er .cells "$skin" "$fat" | awk '{n+=$1} END{print n}')"
