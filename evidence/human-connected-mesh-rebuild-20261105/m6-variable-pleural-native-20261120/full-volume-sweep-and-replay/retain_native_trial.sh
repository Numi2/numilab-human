#!/bin/bash
set -euo pipefail
N=$(cd "$(dirname "$0")" && pwd)
D=$(dirname "$N")
PY="$HOME/numi-human-prep-venv-20261005/bin/python3"
ZSTD=/opt/homebrew/bin/zstd
src=$1
dest=$2
mkdir -p "$dest/native"
[[ -f "$src/execution.json" ]]
cp "$src/"{launch,execution}.json "$dest/"
for name in stdout.log stderr.log failure.json; do
    if [[ -f "$src/$name" ]]; then cp "$src/$name" "$dest/"; fi
done
for name in resting-coupled.csv resting-surface-audit.csv; do
    if [[ -f "$src/native/$name" ]]; then cp "$src/native/$name" "$dest/native/"; fi
done
find "$src/native" -maxdepth 1 -name 'registered-volume-fractional-mass-*.json' -exec cp {} "$dest/native/" \;
if [[ -f "$src/native/resting-human.mrvpack" ]]; then
    initial="$src/native/resting-human.mrvpack"
    sha=$(shasum -a 256 "$initial" | cut -d' ' -f1)
    retained=$(python3 - "$D/contact-prefix-certification/terminal-026-storage-recovery.jsonl" <<'INLINE'
import json,sys
for line in open(sys.argv[1]):
    row=json.loads(line)
    if row["archive"].endswith("/resting-human.mrvpack.zst"):
        print(row["archive"])
        break
INLINE
)
    if [[ -f "$retained" ]] && "$ZSTD" -qdc "$retained" | cmp -s - "$initial"; then
        printf 'sha256=%s retained_identical=%s\n' "$sha" "$retained" > "$dest/initial-pack.txt"
    else
        output="$dest/native/resting-human.mrvpack.zst"
        "$ZSTD" -q -4 -T1 "$initial" -o "$output"
        "$ZSTD" -qdc "$output" | cmp - "$initial"
        printf 'sha256=%s retained_verified=%s\n' "$sha" "$output" > "$dest/initial-pack.txt"
    fi
fi
# This volume is an experiment workspace. Preserve full accepted endpoint
# bytes, including residual words, before ejecting the temporary image.
for point in "$src/native/accepted-geometry/"*.receipt.json; do
    [[ -f "$point" ]] || continue
    base=${point%.receipt.json}
    mkdir -p "$dest/native/accepted-geometry"
    dir="$dest/native/accepted-geometry"
    for file in "$base".*; do
        case "$file" in
            *.common-position-low.bin|*.mrvpack)
                output="$dir/$(basename "$file").zst"
                "$ZSTD" -q -4 -T1 "$file" -o "$output"
                "$ZSTD" -qdc "$output" | cmp - "$file"
                ;;
            *)
                cp "$file" "$dir/"
                ;;
        esac
    done
    if [[ -f "$base.mrvpack" ]]; then
        "$PY" "$D/audit_native_skin_endpoint.py" "$base.mrvpack" "$dest/exact-skin-surface.json" > "$dest/exact-surface-audit.log"
    fi
done
"$PY" "$D/summarize_live_parent_motion.py" "$dest"
