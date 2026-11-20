#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
work=$(dirname "$PWD")
reference="$work/resting-clearance-and-atrium-017/contact-prefix-certification/NumiMatter.metallib"
./check_patch_linearization "$reference" build/NumiMatter.metallib | tee fixture.log
"$HOME/numi-human-prep-venv-20261005/bin/python3" check_comparison.py "$@"
raw=$PWD/cached-root-1/native/resting-human.mrvpack
canonical=$(realpath "$work/muscle-shell-registration-018/profile-root-1/native/resting-human.mrvpack.zst")
if [[ -f "$raw" && ! -e "$raw.zst" ]]; then
    bytes=$(stat -f %z "$raw")
    sha=$(shasum -a 256 "$raw" | cut -d" " -f1)
    /opt/homebrew/bin/zstd -qt "$canonical"
    recover=$(/opt/homebrew/bin/zstd -qdc "$canonical" | shasum -a 256 | cut -d" " -f1)
    if [[ "$sha" == "$recover" ]]; then
        ln -s "$canonical" "$raw.zst"
        printf '{"raw_bytes":%s,"raw_sha256":"%s","recovery_path":"%s","recovered_raw_sha256":"%s"}\n' "$bytes" "$sha" "$canonical" "$recover" > initial-pack-recovery.json
        rm "$raw"
    fi
fi
