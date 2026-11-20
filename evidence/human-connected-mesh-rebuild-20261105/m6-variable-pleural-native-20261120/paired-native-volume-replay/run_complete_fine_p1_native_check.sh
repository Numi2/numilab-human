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
/usr/bin/xcrun clang++ -DNUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY=1 -std=c++23 -O3 -Wall -Wextra -Wpedantic -Werror \
    -I"$owner/apps" -I"$owner/include" -I"$owner/matter/include" \
    "$dir/validate_complete_fine_p1.cpp" -lz -o "$dir/validate_complete_fine_p1"
extra=()
if [ "${1:-}" = "--primary-carriers" ] || [ "${1:-}" = "--flow-witnesses" ] || [ "${1:-}" = "--child-labels" ] || [ "${1:-}" = "--full-replay" ]; then
    carrier_receipt="$dir/paired-carrier-packet-audit/receipt.json"
    report="$dir/archived-compensation-independent-check.json"
    if [ "$1" != "--primary-carriers" ]; then
        witnesses="$dir/paired-native-flow-support-audit.json"
        skin_csv="$(jq -er '.generated_witnesses[0].path' "$witnesses")"
        skin_sha="$(jq -er '.generated_witnesses[0].sha256' "$witnesses")"
        fat_csv="$(jq -er '.generated_witnesses[1].path' "$witnesses")"
        fat_sha="$(jq -er '.generated_witnesses[1].sha256' "$witnesses")"
    else
        skin_csv="$(jq -er '.reports[0].primary_witness_path' "$carrier_receipt")"
        skin_sha="$(jq -er '.reports[0].primary_witness_sha256' "$carrier_receipt")"
        fat_csv="$(jq -er '.reports[1].primary_witness_path' "$carrier_receipt")"
        fat_sha="$(jq -er '.reports[1].primary_witness_sha256' "$carrier_receipt")"
    fi
    extra=("$dir/../resting-clearance-and-atrium-016/repaired-tetgen/mesh.node"
        "$(jq -er .source330_identity.source330_tets.path "$state")"
        "$(jq -er .source330_identity.source330_nodes.path "$state")"
        "$(jq -er '.reports[0].current_packet_path' "$carrier_receipt")"
        "$(jq -er '.reports[0].current_packet_sha256' "$carrier_receipt")"
        "$(jq -er '.reports[1].current_packet_path' "$carrier_receipt")"
        "$(jq -er '.reports[1].current_packet_sha256' "$carrier_receipt")"
        "$skin_csv" "$skin_sha" "$fat_csv" "$fat_sha"
        "$(jq -er .transport_distance_limit_m "$report")"
        "$(jq -er .source330_identity.source330_tets.sha256 "$state")"
        "$(jq -er .source330_identity.source330_nodes.sha256 "$state")")
    if [ "$1" = "--child-labels" ] || [ "$1" = "--full-replay" ]; then
        label_report="$dir/paired-current-child-labels-receipt.json"
        old_manifest="$dir/../captured-fat-sliver-495/proposal/compiler/mass-allocation/run-manifest.json"
        extra+=("$(jq -er .path "$label_report")" "$(jq -er .sha256 "$label_report")"
                "$(jq -er '.argv[(.argv|index("--source-body-count"))+1]' "$old_manifest")"
                "$(jq -er '.argv[(.argv|index("--coordinate-frame-sha256"))+1]' "$old_manifest")")
    fi
    if [ "$1" = "--full-replay" ]; then
        reflow="$dir/archived-compensation-warm-reflow.json"
        original_moves="$(jq -er '.augmentations + ([.density_capacity.moves[]] | add)' "$reflow")"
        if [ "${NUMI_HUMAN_NATIVE_DENSITY_CONTROL:-0}" = "1" ]; then
            extra+=("$dir/paired-density-closed-flow-candidate.npz"
                    "$(jq -er .mass_sha256 "$report")" "$original_moves")
        else
            closure="$dir/paired-native-density-closure.json"
            extra+=("$dir/paired-native-density-closed-flow.npz"
                    "$(jq -er .candidate_sha256 "$closure")"
                    "$((original_moves + $(jq -er .route_adjustments "$closure")))")
        fi
    fi
fi
/usr/bin/time -l "$dir/validate_complete_fine_p1" "$mass" "$labels" "$elements" \
    "$(shasum -a 256 "$state" | cut -d ' ' -f 1)" \
    "$(jq -er .source330_identity.source330_mass.sha256 "$state")" \
    "$(jq -er .p1_one_hop_path "$skin")" "$(jq -er .p1_one_hop_sha256 "$skin")" \
    "$(jq -er .role_geometry_sha256 "$skin")" \
    "$(jq -er .p1_one_hop_path "$fat")" "$(jq -er .p1_one_hop_sha256 "$fat")" \
    "$(jq -er .role_geometry_sha256 "$fat")" \
    "$(jq -er .cells "$skin" "$fat" | awk '{n+=$1} END{print n}')" ${extra[@]+"${extra[@]}"}
