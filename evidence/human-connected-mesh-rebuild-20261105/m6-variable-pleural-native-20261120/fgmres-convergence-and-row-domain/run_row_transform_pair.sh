#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
W=$(dirname "$PWD")
python="$HOME/numi-human-prep-venv-20261005/bin/python3"
archive=$(realpath "$W/resting-clearance-and-atrium-017/floor-native-live-parent-020/native/resting-human.mrvpack.zst")
/opt/homebrew/bin/zstd -qt "$archive"
for arm in on off; do
    output=row-transform-$arm
    "$python" run_native.py --row-transform "$arm" "$output"
    "$python" "$W/muscle-shell-registration-018/check_profile_matched_roots.py" "$output" > "$output/profile-compact.json"
    raw="$PWD/$output/native/resting-human.mrvpack"
    sha=$(shasum -a 256 "$raw" | cut -d" " -f1)
    recovered=$(/opt/homebrew/bin/zstd -qdc "$archive" | shasum -a 256 | cut -d" " -f1)
    if [[ "$sha" == b9179f1e87e4bb7990762e20bcf87f209d2303f1d3735c7bda1e463b57a2b91e && "$sha" == "$recovered" ]]; then
        ln -s "$archive" "$raw.zst"
        printf '{"raw_bytes":%s,"raw_sha256":"%s","recovery_path":"%s","recovered_raw_sha256":"%s"}\n' "$(stat -f %z "$raw")" "$sha" "$archive" "$recovered" > "$output/initial-pack-recovery.json"
        rm "$raw"
    fi
done
