#!/bin/bash
set -euo pipefail
T=$HOME/Work/human-integrated-gaps-20261104/coarse-hand-cavity-024
J=$T/posterior-load-path-042
cd "$J"
name=NumiLoadedDispatchLatch1120
ram="/Volumes/$name"
test ! -e "$ram"
disk=$(hdiutil attach -nomount ram://900000 | awk 'NR==1 {print $1}')
[[ "$disk" =~ ^/dev/disk[0-9]+$ ]]
diskutil eraseDisk APFS "$name" GPT "$disk" > loaded-dispatch-ram-volume.log
diskutil info "$ram" | rg 'Protocol:.*Disk Image'
/opt/homebrew/bin/gh release download numi-human-bed-column-20261120 \
    -R Numi2/workstation-checkpoint-archive \
    --pattern 20261120-numi-posterior-ilium-native.mrvpack -O "$ram/initial.mrvpack"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
for run in loaded-dispatch-full loaded-dispatch-short; do
    flags=()
    if [[ "$run" = loaded-dispatch-short ]]; then flags=(--staged); fi
    "$py" replay_outlier.py "$T/posterior-gluteal-continuum-041/native-candidate-128/launch.json" \
        "$ram/initial.mrvpack" "$ram/$run" --shader guarded-dispatch-shader/guarded-short.metallib \
        --binary guarded-dispatch-app-02/numi-human-native \
        --library guarded-dispatch-library/libmetalrobo.dylib --fields ${flags[@]+"${flags[@]}"}
    mkdir "$run"
    mkdir "$run/native"
    for file in stdout.log.gz stderr.log launch.json execution.json native/resting-coupled.csv; do
        cp "$ram/$run/$file" "$run/$file"
        cmp "$ram/$run/$file" "$run/$file"
    done
    gzip -t "$run/stdout.log.gz"
done
diskutil eject "$disk"
"$py" compare_solver_prefixes.py control loaded-dispatch-full loaded-dispatch-control-comparison.json
"$py" compare_solver_prefixes.py loaded-dispatch-full loaded-dispatch-short loaded-dispatch-staged-comparison.json
"$py" compare_latch.py loaded-dispatch-full loaded-dispatch-short loaded-dispatch-comparison.json
"$py" "$T/next-stage-profile/compare_solver_fields.py" control/stdout.log.gz loaded-dispatch-full/stdout.log.gz loaded-dispatch-control-fields.json > loaded-dispatch-control-fields.log
"$py" "$T/next-stage-profile/compare_solver_fields.py" loaded-dispatch-full/stdout.log.gz loaded-dispatch-short/stdout.log.gz loaded-dispatch-staged-fields.json > loaded-dispatch-staged-fields.log
"$py" verify_loaded_guard.py loaded-dispatch-comparison.json loaded-dispatch-control-comparison.json
