#!/bin/bash
set -euo pipefail
T="$HOME/Work/human-integrated-gaps-20261104/coarse-hand-cavity-024"
D="$T/load-transfer-capacity-048"
ram=/Volumes/NumiEngagement048
run="$ram/native-engagement-128-02"
baseline_failed="$ram/native-engagement-128"
repo=Numi2/workstation-checkpoint-archive
tag=numi-human-bed-column-20261120
pack="$run/native/resting-human.mrvpack"
prefix=20261120-numi-posterior-engagement-native-128
asset="$prefix.mrvpack.zst"
witness="$prefix-evidence.tar.gz"
gh=/opt/homebrew/bin/gh
zstd=/opt/homebrew/bin/zstd
raw_signature() { shasum -a 256 "$1" | cut -d' ' -f1; }
cd "$D"
[[ -s "$run/execution.json" && -s "$baseline_failed/execution.json" ]]
archive_files=(native-engagement-128-02 native-engagement-128)
if [[ -s "$pack" ]]; then
    /usr/bin/tar --exclude='*/native/resting-human.mrvpack' -czf "$ram/$witness" \
        -C "$ram" "${archive_files[@]}"
else
    /usr/bin/tar -czf "$ram/$witness" -C "$ram" "${archive_files[@]}"
fi
if [[ -s "$pack" ]]; then
    "$zstd" -q -4 -T1 "$pack" -o "$ram/$asset"
    "$gh" release upload "$tag" -R "$repo" "$ram/$witness" "$ram/$asset"
    "$gh" release download "$tag" -R "$repo" --pattern "$asset" -O - | "$zstd" -q -dc | cmp - "$pack"
else
    "$gh" release upload "$tag" -R "$repo" "$ram/$witness"
fi
"$gh" release download "$tag" -R "$repo" --pattern "$witness" -O - | cmp - "$ram/$witness"
/usr/bin/tar -tzf "$ram/$witness" | LC_ALL=C sort > archive-members.txt
if [[ -s "$pack" ]]; then
    ! rg 'resting-human.mrvpack$' archive-members.txt
fi
for a in "${archive_files[@]}"; do
    /usr/bin/tar -xOzf "$ram/$witness" "$a/execution.json" | cmp - "$ram/$a/execution.json"
    /usr/bin/tar -xOzf "$ram/$witness" "$a/stdout.log" | cmp - "$ram/$a/stdout.log"
    /usr/bin/tar -xOzf "$ram/$witness" "$a/stderr.log" | cmp - "$ram/$a/stderr.log"
done
{
    printf 'release=https://github.com/%s/releases/tag/%s\n' "$repo" "$tag"
    printf 'asset=%s sha256=%s independent_restore=exact\n' "$witness" "$(raw_signature "$ram/$witness")"
    if [[ -s "$pack" ]]; then
        printf 'asset=%s sha256=%s original_mrvpack_sha256=%s independent_restore=exact\n' \
            "$asset" "$(raw_signature "$ram/$asset")" "$(raw_signature "$pack")"
    fi
} > archive-recovery.txt
cat archive-recovery.txt
