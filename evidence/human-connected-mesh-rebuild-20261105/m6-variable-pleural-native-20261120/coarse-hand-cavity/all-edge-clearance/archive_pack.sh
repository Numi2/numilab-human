#!/bin/bash
set -euo pipefail
pack=$1
asset=$2
receipt=$3
repo=Numi2/workstation-checkpoint-archive
tag=numi-human-bed-column-20261120
gh=/opt/homebrew/bin/gh
test -f "$pack" && test ! -L "$pack"
test ! -e "$receipt"
test "$("$gh" repo view "$repo" --json visibility --jq .visibility)" = PRIVATE
size=$(stat -f %z "$pack")
sha=$(shasum -a 256 "$pack" | cut -d ' ' -f1)
alias="$(dirname "$pack")/$asset"
if ! "$gh" release view "$tag" --repo "$repo" --json assets --jq '.assets[].name' | grep -Fxq "$asset"; then
    ln "$pack" "$alias"
    "$gh" release upload "$tag" "$alias" --repo "$repo"
fi
fetched=$("$gh" release download "$tag" --repo "$repo" --pattern "$asset" --output - | shasum -a 256 | cut -d ' ' -f1)
test "$fetched" = "$sha"
if [[ -e "$alias" ]]; then
    [[ "$alias" -ef "$pack" ]]
    rm "$alias"
fi
/usr/bin/printf '{\n  "path": "%s",\n  "asset": "%s",\n  "private_release": "https://github.com/%s/releases/tag/%s",\n  "sha256": "%s",\n  "size_bytes": %s,\n  "verified_live_download": true\n}\n' "$pack" "$asset" "$repo" "$tag" "$sha" "$size" >"$receipt"
