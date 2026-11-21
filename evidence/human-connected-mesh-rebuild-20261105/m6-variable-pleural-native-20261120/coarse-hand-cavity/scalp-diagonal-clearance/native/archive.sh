#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
trial=root-128-01
build=build-01
pack="$trial/native/resting-human.mrvpack"
stem=20261121-numi-scalp-derived-native
repo=Numi2/workstation-checkpoint-archive
tag=numi-human-bed-column-20261120
gh=/opt/homebrew/bin/gh
zstd=/opt/homebrew/bin/zstd
sum() { shasum -a 256 "$1" | cut -d' ' -f1; }
[[ -s "$trial/comparison.json" && -s "$trial/verified-mass.json" ]]
[[ -s "$trial/verified-cells.json" && -s "$trial/verified-shoulders.json" ]]
pack_sha=$(sum "$pack")
prior_pack_sha=2912968fb564757773201f97c4dcac3dc0a713f46d1ce5b93888aa249147c889
assets=()
pack_asset=20261121-numi-posterior-fat-four-cell.mrvpack.zst
if [[ "$pack_sha" != "$prior_pack_sha" ]]; then
    pack_asset="$stem.mrvpack.zst"
    "$zstd" -q -4 -T1 "$pack" -o "$pack_asset"
    assets+=("$pack_asset")
fi
run_asset="$stem.tar.gz"
/usr/bin/tar --exclude='*/native/resting-human.mrvpack' -czf "$run_asset" \
    "$trial" "$build/PosteriorBedReference.inc" "$build/registration-provenance.json" \
    "$build/build-commands.json" "$build/build-identity.json" "$build/build.log" \
    "$build/numi-human-native" "$build/numilab_human_myosim_visual_probe.mm"
"$gh" release upload "$tag" -R "$repo" "$run_asset" "${assets[@]}"
"$gh" release download "$tag" -R "$repo" --pattern "$run_asset" -O - | cmp - "$run_asset"
"$gh" release download "$tag" -R "$repo" --pattern "$pack_asset" -O - | "$zstd" -q -dc | cmp - "$pack"
/usr/bin/tar -xOzf "$run_asset" "$build/numi-human-native" | cmp - "$build/numi-human-native"
/usr/bin/tar -xOzf "$run_asset" "$trial/native/resting-coupled.csv" | cmp - "$trial/native/resting-coupled.csv"
/usr/bin/tar -xOzf "$run_asset" "$trial/bone-endpoints.json" | cmp - "$trial/bone-endpoints.json"
{
    printf 'release=https://github.com/%s/releases/tag/%s\n' "$repo" "$tag"
    printf 'asset=%s sha256=%s exact_download=true member_recovery=true\n' "$run_asset" "$(sum "$run_asset")"
    printf 'asset=%s exact_decompressed_sha256=%s independent_download=true\n' "$pack_asset" "$pack_sha"
} > archive-recovery.txt
cat archive-recovery.txt
