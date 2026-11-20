#!/bin/bash
set -euo pipefail
base="$HOME/Work/human-integrated-gaps-20261104/resting-clearance-and-atrium-017/refit-motion-qualification"
checkpoint="$HOME/Work/human-integrated-gaps-20261104/coarse-hand-cavity-024/effective-newton-terminal-044"
repo=Numi2/workstation-checkpoint-archive
tag=numi-human-bed-column-20261120
asset=20261120-numi-superseded-refit-native-role-flows.tar.zst
ram=/Volumes/NumiRolesArchive044
[[ ! -e "$ram" ]]
disk=$(hdiutil attach -nomount ram://1100000 | awk 'NR==1 {print $1}')
[[ "$disk" =~ ^/dev/disk[0-9]+$ ]]
trap 'diskutil eject "$disk"' EXIT
diskutil eraseDisk APFS NumiRolesArchive044 GPT "$disk" > "$checkpoint/role-archive-volume.log"
diskutil info "$ram" | rg 'Protocol:.*Disk Image'
cd "$base"
files=(native-role-flow/skin-source-carrier-flow.csv native-role-flow/fat-source-carrier-flow.csv)
for file in "${files[@]}"; do
    [[ -f "$file" && ! -L "$file" ]]
    ! lsof "$file"
done
/usr/bin/shasum -a 256 "${files[@]}" > "$ram/members.sha256"
/usr/bin/tar -cf - "${files[@]}" | /opt/homebrew/bin/zstd -q -4 -T1 -o "$ram/$asset"
sha=$(/usr/bin/shasum -a 256 "$ram/$asset" | awk '{print $1}')
/opt/homebrew/bin/gh release upload "$tag" -R "$repo" "$ram/$asset"
/opt/homebrew/bin/gh release download "$tag" -R "$repo" --pattern "$asset" -O "$ram/restored.tar.zst"
/usr/bin/cmp "$ram/$asset" "$ram/restored.tar.zst"
for file in "${files[@]}"; do
    /opt/homebrew/bin/zstd -q -dc "$ram/restored.tar.zst" | /usr/bin/tar -xOf - "$file" | /usr/bin/cmp - "$file"
done
/usr/bin/shasum -a 256 -c "$ram/members.sha256"
{
    echo "release=https://github.com/$repo/releases/tag/$tag"
    echo "asset=$asset sha256=$sha independent_member_restore=exact"
    cat "$ram/members.sha256"
    /bin/df -k /System/Volumes/Data
    for file in "${files[@]}"; do /usr/bin/stat -f '%z %b %N' "$file"; done
} > "$checkpoint/role-archive-recovery.txt"
rm "${files[@]}"
/bin/df -k /System/Volumes/Data >> "$checkpoint/role-archive-recovery.txt"
cat "$checkpoint/role-archive-recovery.txt"
