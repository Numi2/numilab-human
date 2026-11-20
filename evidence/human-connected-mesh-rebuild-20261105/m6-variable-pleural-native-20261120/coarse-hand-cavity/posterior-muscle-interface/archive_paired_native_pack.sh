#!/bin/bash
set -euo pipefail
P="$HOME/Work/human-integrated-gaps-20261104/coarse-hand-cavity-024/posterior-muscle-interface-045"
repo=Numi2/workstation-checkpoint-archive
tag=numi-human-bed-column-20261120
asset=20261120-numi-paired-deep-fat-native-128.mrvpack.zst
run="$P/paired-native-128"
pack="$run/native/resting-human.mrvpack"
ram=/Volumes/NumiPairedArchive045
[[ -s "$P/published-commit.txt" && -s "$run/verified-mass.json" && -s "$pack" && ! -L "$pack" ]]
[[ ! -e "$ram" ]]
! lsof "$pack"
disk=$(hdiutil attach -nomount ram://1500000 | awk 'NR==1 {print $1}')
[[ "$disk" =~ ^/dev/disk[0-9]+$ ]]
trap 'diskutil eject "$disk" >/dev/null' EXIT
diskutil eraseDisk APFS NumiPairedArchive045 GPT "$disk" >/dev/null
diskutil info "$ram" | rg 'Protocol:.*Disk Image' >/dev/null
raw=$(shasum -a 256 "$pack" | cut -d' ' -f1)
/opt/homebrew/bin/zstd -q -4 -T1 "$pack" -o "$ram/$asset"
zipped=$(shasum -a 256 "$ram/$asset" | cut -d' ' -f1)
/opt/homebrew/bin/gh release upload "$tag" -R "$repo" "$ram/$asset"
/opt/homebrew/bin/gh release download "$tag" -R "$repo" --pattern "$asset" -O "$ram/recovered.zst"
cmp "$ram/$asset" "$ram/recovered.zst"
/opt/homebrew/bin/zstd -q -dc "$ram/recovered.zst" | cmp - "$pack"
{
    printf 'release=https://github.com/%s/releases/tag/%s\n' "$repo" "$tag"
    printf 'asset=%s sha256=%s original_mrvpack_sha256=%s independent_restore=exact\n' "$asset" "$zipped" "$raw"
    printf 'evidence_commit=%s\n' "$(cat "$P/published-commit.txt")"
    stat -f 'verified_size=%z path=%N' "$pack"
    df -k "$P"
} > "$P/paired-native-pack-recovery.txt"
rm "$pack"
df -k "$P" >> "$P/paired-native-pack-recovery.txt"
cat "$P/paired-native-pack-recovery.txt"
