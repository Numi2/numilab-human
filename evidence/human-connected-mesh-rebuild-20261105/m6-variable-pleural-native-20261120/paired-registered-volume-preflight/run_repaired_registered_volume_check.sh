#!/bin/bash
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
V="$(cd "$(dirname "$0")" && pwd)"
W="$(dirname "$V")"
C="$W/captured-fat-sliver-495/proposal/compiler"
A="$HOME/Work/skin-fat-native-integration-209"
M="${NUMI_HUMAN_REGISTERED_MESH_OVERRIDE:-$W/resting-clearance-and-atrium-016/repaired-tetgen}"
F="$W/resting-clearance-and-atrium-016/repaired-tetgen"
S="${NUMI_HUMAN_REGISTERED_SIDECAR_OVERRIDE:-$V/paired-repaired-registration}"
P="$V/paired-native-parent-ledger/compiled-parent-cell-mass.csv"
deps="$C/volume-export/source-domain-registry.json"
jq -r '.roles[] | [.source_domain,.source_asset_sha256,.source_point_domain_count,.source_face_count]|@tsv'     "$deps" > "$S/source-domain-pins.tsv"
if [ ! -f "$S/facet-ancestry.csv" ]; then
    zstd -dc "$F/facet-ancestry.csv.zst" > "$S/facet-ancestry.csv"
fi
test "$(shasum -a 256 "$S/facet-ancestry.csv" | cut -d' ' -f1)" =     "$(jq -er .emitted_csv_sha256 "$F/facet-ancestry-receipt.json")"
facets="${NUMI_HUMAN_REGISTERED_FACETS_OVERRIDE:-$S/facet-ancestry.csv}"
labels="${NUMI_HUMAN_REGISTERED_LABELS_OVERRIDE:-$V/paired-current-child.labels.csv}"
capture="$(jq -er '.command[(.command|index("--capture"))+1]' "$C/volume-export.command.json")"
allBodyFrames="$S/all-body-current-rest.csv"
/usr/bin/python3 "$V/emit_retained_all_body_frames.py" "$capture/registered-skin-poses-100.json"     "$C/volume-export/registered/donor-frames.csv" "$allBodyFrames"
existing="$C/mass-allocation/run-manifest.json"
frame="$(jq -er '.argv[(.argv|index("--coordinate-frame-sha256"))+1]' "$existing")"
body="$(jq -er '.argv[(.argv|index("--source-body-count"))+1]' "$existing")"
/usr/bin/xcrun clang++ -DNUMI_HUMAN_FULL_REFERENCE_SKIN_CAPACITY=1 -std=c++23 -O3 -Wall -Wextra     -Wpedantic -Werror -I"$A/apps" -I"$A/include" -I"$A/matter/include"     "$V/verify_repaired_registered_volume.cpp" -o "$V/verify_repaired_registered_volume"
/usr/bin/time -l "$V/verify_repaired_registered_volume" "$S" "$C" "$P" "$M"     "$labels" "$facets" "$frame" "$body"     "$S/source-domain-pins.tsv" "$allBodyFrames" b1704fb7b846254583f4b6a08c394ed3cad1603554244078706ca4b3811ac674
