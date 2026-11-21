#!/bin/bash
set -euo pipefail
T="$HOME/Work/human-integrated-gaps-20261104/coarse-hand-cavity-024"
S="$T/sacral-interface-046"
archive=Numi2/workstation-checkpoint-archive
tag=numi-human-bed-column-20261120
asset=20261120-numi-witness-reach-native-prefix-initial.mrvpack
canonical="$T/witness-reach-026/native-prefix/native/resting-human.mrvpack"
duplicate="$T/witness-reach-026/native-retry/native/resting-human.mrvpack"
checksum=1ffebb91f2d75387b82aba4873c848253978f8b84f171561e0e3bc80413798f9
gh=/opt/homebrew/bin/gh
[[ -f "$canonical" && ! -L "$canonical" && -f "$duplicate" && ! -L "$duplicate" ]]
[[ "$(stat -f %l "$duplicate")" == 1 ]]
! lsof "$duplicate"
printf '%s  %s\n' "$checksum" "$canonical" | shasum -a 256 -c
cmp "$canonical" "$duplicate"
assets=$("$gh" release view "$tag" -R "$archive" --json assets --jq '.assets[].name')
if ! grep -Fxq "$asset" <<< "$assets"; then
    upload=$(mktemp -d "$S/archive-link.XXXXXX")
    trap 'rm -f "$upload/$asset"; rmdir "$upload"' EXIT
    ln "$canonical" "$upload/$asset"
    "$gh" release upload "$tag" -R "$archive" "$upload/$asset"
fi
"$gh" release download "$tag" -R "$archive" --pattern "$asset" -O - | cmp - "$canonical"
! lsof "$duplicate"
cmp "$canonical" "$duplicate"
{
    printf 'archive=https://github.com/%s/releases/download/%s/%s\n' "$archive" "$tag" "$asset"
    printf 'sha256=%s independent_download_exact_comparison=passed\n' "$checksum"
    stat -f 'verified_size=%z disk_blocks=%b path=%N' "$duplicate"
    df -k "$S"
} > "$S/witness-pack-recovery.txt"
rm "$duplicate"
df -k "$S" >> "$S/witness-pack-recovery.txt"
cat "$S/witness-pack-recovery.txt"
