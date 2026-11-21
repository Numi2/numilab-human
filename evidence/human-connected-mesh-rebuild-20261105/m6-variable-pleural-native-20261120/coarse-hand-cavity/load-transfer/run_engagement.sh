#!/bin/bash
set -euo pipefail
W="$HOME/Work/human-integrated-gaps-20261104"
T="$W/coarse-hand-cavity-024"
D="$T/load-transfer-capacity-048"
P="$T/posterior-muscle-interface-045"
V="$T/posterior-gluteal-continuum-041"
PY="$HOME/numi-human-prep-venv-20261005/bin/python3"
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$T:$T/posterior-load-interface-040:$V:$T/posterior-load-path-042:$W/resting-clearance-and-atrium-017/floor-scene-build:$W/resting-clearance-and-atrium-006:$HOME/Work/skin-fat-native-integration-209/tools"
cd "$D"
volume=/Volumes/NumiEngagement048
build=${1:-native-engagement-build-02}
case "$build" in
    native-engagement-build|native-engagement-build-02) ;;
    *) exit 2 ;;
esac
evidence="${build/build/128}"
[[ ! -e "$evidence" && -x "$build/numi-human-native" ]]
if [[ ! -e "$volume" ]]; then
    disk=$(hdiutil attach -nomount ram://1330000 | awk 'NR==1 {print $1}')
    [[ "$disk" =~ ^/dev/disk[0-9]+$ ]]
    diskutil eraseDisk APFS NumiEngagement048 GPT "$disk" >/dev/null
    printf '%s\n' "$disk" > scratch-device.txt
fi
diskutil info "$volume" | rg 'Protocol:.*Disk Image' >/dev/null
run="$volume/$evidence"
step_count=128
status=0
"$PY" "$V/native_trial.py" run "$P/paired-native-128" "$build" "$run" --steps "$step_count" || status=$?
mkdir "$evidence"
cp "$run/"{launch.json,execution.json} "$evidence"/
if [[ $status != 0 ]]; then
    tail -40 "$run/stderr.log" > "$evidence"/rejection.txt
    tail -50 "$run/stdout.log" >> "$evidence"/rejection.txt
    cat "$evidence"/rejection.txt
    exit "$status"
fi
rg '^(posterior_reference_mass_install=|resting_integrated_(body|throughput)|resting_fem_skin_bed_accepted_root_reaction |resting_fascia_fem_terminal_audit=|resting_FEM_terminal_audit=)' \
  "$run/stdout.log" | tee "$evidence"/terminal.log
"$PY" "$V/verify_transfer.py" "$V/native-candidate-128" engagement-fit-02 "$T" \
 "$evidence"/verified-mass.json --native "$run"
"$PY" "$P/check_native_endpoint.py" "$run" "$P/paired-native-128" "$evidence"/verified-cells.json
"$PY" "$T/posterior-load-path-042/audit_shoulder_prefix.py" "$run/launch.json" \
 "$W/resting-clearance-and-atrium-017/fixed-shoulder-anchors.json" "$run/native/resting-body-motion-partition.csv" \
 "$evidence"/verified-shoulders.json --steps 1 32 64 96 128
cp "$run/native/resting-coupled.csv" "$evidence"/
"$PY" "$T/posterior-load-path-042/audit_boundary_bones.py" "$run" \
 "$T/thickness-rebuild-034/inputs/physical-bone-initial.bin" "$run/bone-endpoints.json" \
 --step 128 > "$evidence"/bone-audit.log
"$PY" compare_native.py "$P/paired-native-128" "$run" "$evidence/verified-comparison.json"
