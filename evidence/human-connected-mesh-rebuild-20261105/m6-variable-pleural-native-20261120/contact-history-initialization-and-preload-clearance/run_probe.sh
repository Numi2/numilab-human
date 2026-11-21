#!/bin/bash
set -euo pipefail
repo=$(cd "${1:?path to numi-lab checkout at tested revision}" && pwd)
here=$(cd "$(dirname "$0")" && pwd)
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
python3 - "$repo/matter/src/runtime.mm" "$out/upload_plan_fixture.hpp" <<'PY'
from pathlib import Path
import sys

source, destination = map(Path, sys.argv[1:])
text = source.read_text()
start = text.index("[[nodiscard]] NSUInteger checkedBytes(")
end = text.index("\n}\n", start) + 3
owner = text.index("class UploadPlan {")
owner_end = text.index("\n};", owner) + 3
destination.write_text(text[start:end] + "\n" + text[owner:owner_end] + "\n")
PY
/usr/bin/clang++ -O2 -std=c++23 -fobjc-arc -I"$out" -I"$repo/matter/include" -I"$repo/include" \
    -framework Foundation -framework Metal "$here/measure.mm" -o "$out/measure"
export NUMI_HUMAN_PREPARATION_MEMORY=1 NUMI_MATTER_UPLOAD_STAGING_BYTES=8388608
"$out/measure" candidate 0 1
"$out/measure" candidate 51 3
for mode in control candidate candidate control control candidate; do
    "$out/measure" "$mode" 4207296 1
done
