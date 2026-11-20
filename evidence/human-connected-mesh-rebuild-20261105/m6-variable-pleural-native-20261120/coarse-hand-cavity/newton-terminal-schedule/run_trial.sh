#!/bin/bash
set -euo pipefail
T="$HOME/Work/human-integrated-gaps-20261104/coarse-hand-cavity-024"
J="$T/posterior-load-path-042"
N="$T/newton-terminal-schedule-043"
V="$T/posterior-gluteal-continuum-041"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
cd "$N"
test -s app/identity.json && test -s library/identity.json && test -s shader/identity.json
shasum -a 256 -c preregistration.sha256
vol=NumiNewtonTerminal043
ram="/Volumes/$vol"
test ! -e "$ram"
disk=$(hdiutil attach -nomount ram://900000 | awk 'NR==1 {print $1}')
[[ "$disk" =~ ^/dev/disk[0-9]+$ ]]
diskutil eraseDisk APFS "$vol" GPT "$disk" > ram-volume.log
diskutil info "$ram" | rg 'Protocol:.*Disk Image'
/opt/homebrew/bin/gh release download numi-human-bed-column-20261120 \
    -R Numi2/workstation-checkpoint-archive \
    --pattern 20261120-numi-posterior-ilium-native.mrvpack -O "$ram/initial.mrvpack"
"$py" - "$V/native-candidate-128/launch.json" "$N/terminal-stage-source.json" <<'PY'
import json
import sys
from pathlib import Path
reference = json.loads(Path(sys.argv[1]).read_text())
reference["environment"]["NUMI_MATTER_KEEP_ORIGINAL_NEWTON_TERMINAL"] = "1"
with Path(sys.argv[2]).open("x") as stream:
    json.dump(reference, stream, indent=2)
    stream.write("\n")
PY
for run in full staged; do
    flags=()
    input="$V/native-candidate-128/launch.json"
    if [[ "$run" = staged ]]; then
        input="$N/terminal-stage-source.json"
        flags=(--staged)
    fi
    code=0
    "$py" "$J/replay_outlier.py" "$input" "$ram/initial.mrvpack" "$ram/$run" \
        --shader "$N/shader/guarded-short.metallib" --binary "$N/app/numi-human-native" \
        --library "$N/library/libmetalrobo.dylib" --fields ${flags[@]+"${flags[@]}"} || code=$?
    mkdir "$run" "$run/native"
    for file in stdout.log.gz stderr.log launch.json execution.json native/resting-coupled.csv; do
        if [[ -f "$ram/$run/$file" ]]; then
            cp "$ram/$run/$file" "$run/$file"
            cmp "$ram/$run/$file" "$run/$file"
        fi
    done
    gzip -t "$run/stdout.log.gz"
    echo "$run status=$code"
done
diskutil eject "$disk"
"$py" "$J/compare_solver_prefixes.py" "$J/loaded-dispatch-full" full full-parity.json
"$py" "$J/compare_solver_prefixes.py" full staged staged-prefix.json
if rg -q '"returncode": 0' full/execution.json && rg -q '"returncode": 0' staged/execution.json; then
    "$py" "$J/compare_latch.py" full staged staged-work.json
    "$py" "$T/next-stage-profile/compare_solver_fields.py" "$J/loaded-dispatch-full/stdout.log.gz" full/stdout.log.gz full-fields.json > full-fields.log
    "$py" "$T/next-stage-profile/compare_solver_fields.py" full/stdout.log.gz staged/stdout.log.gz staged-fields.json > staged-fields.log
fi
