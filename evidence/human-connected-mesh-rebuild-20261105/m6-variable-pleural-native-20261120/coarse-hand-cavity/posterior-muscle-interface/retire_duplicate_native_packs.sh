#!/bin/bash
set -euo pipefail
shopt -s nullglob
T="$HOME/Work/human-integrated-gaps-20261104/coarse-hand-cavity-024"
P="$T/posterior-muscle-interface-045"
archive=Numi2/workstation-checkpoint-archive
tag=numi-human-bed-column-20261120
asset=20261120-numi-older-bed-normal-bank-native.mrvpack
checksum=129caadf7baa062451eb82f24494eafb8fec8822151ab41a36b1a634740f77b0
volume=/Volumes/NumiPackVerify045
[[ ! -e "$volume" ]]
root="$T/active-operator-pair-025"
cd "$root"
files=({assembled,borrowed,control-long,control,graph,six-long,six}/native/resting-human.mrvpack)
for file in "${files[@]}"; do
    [[ -f "$file" && ! -L "$file" && "$(stat -f %l "$file")" == 1 ]]
    ! lsof "$file"
done
disk=$(hdiutil attach -nomount ram://770000 | awk 'NR==1 {print $1}')
[[ "$disk" =~ ^/dev/disk[0-9]+$ ]]
trap 'diskutil eject "$disk" >/dev/null' EXIT
diskutil eraseDisk APFS NumiPackVerify045 GPT "$disk" >/dev/null
diskutil info "$volume" | rg 'Protocol:.*Disk Image' >/dev/null
/opt/homebrew/bin/gh release download "$tag" -R "$archive" --pattern "$asset" -O "$volume/$asset"
printf '%s  %s\n' "$checksum" "$volume/$asset" | shasum -a 256 -c
for file in "${files[@]}"; do cmp "$volume/$asset" "$file"; done
{
    printf 'directory=%s\n' "$root"
    printf 'release=https://github.com/%s/releases/tag/%s\n' "$archive" "$tag"
    printf 'asset=%s sha256=%s independent_download_exact_comparison=passed\n' "$asset" "$checksum"
    df -k .
    for name in "${files[@]}"; do
        stat -f 'verified_size=%z disk_blocks=%b path=%N' "$name"
    done
} > "$P/native-pack-recovery.txt"
rm "${files[@]}"
df -k . >> "$P/native-pack-recovery.txt"
cat "$P/native-pack-recovery.txt"
