#!/bin/bash
set -euo pipefail
T=$HOME/Work/human-integrated-gaps-20261104/coarse-hand-cavity-024
J=$T/posterior-load-path-042
cd "$J"
name=NumiDispatchLatch1120
ram="/Volumes/$name"
if [[ -f "$ram/initial.mrvpack" ]]; then
    disk=$(diskutil info "$ram" | awk '/Part of Whole/{print "/dev/"$NF}')
    diskutil info "$disk" | rg 'Protocol:.*Disk Image'
else
    test ! -e "$ram"
    disk=$(hdiutil attach -nomount ram://900000 | awk 'NR==1{print $1}')
    [[ "$disk" =~ ^/dev/disk[0-9]+$ ]]
    diskutil eraseDisk APFS "$name" GPT "$disk" > dispatch-ram-volume.log
    diskutil info "$ram" | rg 'Protocol:.*Disk Image'
    /opt/homebrew/bin/gh release download numi-human-bed-column-20261120 -R Numi2/workstation-checkpoint-archive --pattern 20261120-numi-posterior-ilium-native.mrvpack -O "$ram/initial.mrvpack"
fi
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
declare -a flags
for run in dispatch-full-budget dispatch-short-budget; do
    flags=()
    if [[ "$run" = dispatch-short-budget ]]; then flags=(--staged); fi
    "$py" replay_outlier.py "$T/posterior-gluteal-continuum-041/native-candidate-128/launch.json" \
        "$ram/initial.mrvpack" "$run" --shader guarded-dispatch-shader/guarded-short.metallib \
        --binary guarded-dispatch-app-02/numi-human-native --fields --scratch "$ram/$run" ${flags[@]+"${flags[@]}"} || rc=$?
    mkdir "$run/native-verified"
    for f in "$ram/$run/"*; do
        if [[ "$f" = *.mrvpack ]]; then
            test "$(shasum -a 256 "$f" | cut -d ' ' -f 1)" = 214d94f8e953a270a2e41c47c10687ee64b710d6e4d9f9ce22b970ecf056b486
            continue
        fi
        test -f "$f" || continue
        cp "$f" "$run/native-verified/"
        cmp "$f" "$run/native-verified/${f##*/}"
    done
    unlink "$run/native"
    mv "$run/native-verified" "$run/native"
    echo "$run status=${rc:-0}"
    rc=0
done
diskutil eject "$disk"
"$py" compare_solver_prefixes.py control dispatch-full-budget dispatch-control-comparison.json
"$py" compare_solver_prefixes.py dispatch-full-budget dispatch-short-budget dispatch-staged-comparison.json
if rg -q '"returncode": 0' dispatch-full-budget/execution.json &&
   rg -q '"returncode": 0' dispatch-short-budget/execution.json; then
    "$py" compare_latch.py dispatch-full-budget dispatch-short-budget dispatch-comparison.json
    "$py" "$T/next-stage-profile/compare_solver_fields.py" control/stdout.log.gz dispatch-full-budget/stdout.log.gz dispatch-control-fields.json > dispatch-control-fields.log
    "$py" "$T/next-stage-profile/compare_solver_fields.py" dispatch-full-budget/stdout.log.gz dispatch-short-budget/stdout.log.gz dispatch-staged-fields.json > dispatch-staged-fields.log
fi
