#!/bin/bash
set -euo pipefail
W="$HOME/Work/human-integrated-gaps-20261104"
T="$W/coarse-hand-cavity-024"
N="$T/scalp-diagonal-clearance-052"
B="$T/preload-cavity-topology-051"
P="$T/posterior-muscle-interface-045"
V="$T/posterior-gluteal-continuum-041"
PY="$HOME/numi-human-prep-venv-20261005/bin/python3"
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$B:$T:$T/posterior-load-interface-040:$V:$T/posterior-load-path-042:$W/resting-clearance-and-atrium-017/floor-scene-build:$W/resting-clearance-and-atrium-006:$HOME/Work/skin-fat-native-integration-209/tools:$T/thickness-rebuild-034"
cd "$N"
build="$N/native/${1:-build-01}"
run="$N/native/${2:-root-128-01}"
test -x "$build/numi-human-native"
status=0
"$PY" "$V/native_trial.py" run "$P/paired-native-128" "$build" "$run" --steps 128 || status=$?
if [[ "$status" != 0 ]]; then
    tail -15 "$run/stderr.log" | cut -c1-320
    tail -10 "$run/stdout.log" | cut -c1-320
    exit "$status"
fi
# Run registration and motion checks even if one of the independent gates finds a failure.
failure=0
"$PY" native/verify_mass.py "$run" "$B/native-root-128" volume-02/candidate.npz "$run/verified-mass.json" || failure=1
rg '^(posterior_reference_mass_install=|resting_integrated_(body|throughput)|resting_fem_skin_bed_accepted_root_reaction |resting_fascia_fem_terminal_audit=|resting_FEM_terminal_audit=)' "$run/stdout.log" > "$run/terminal.log"
"$PY" "$P/check_native_endpoint.py" "$run" "$B/native-root-128" "$run/verified-cells.json" || failure=1
"$PY" "$T/posterior-load-path-042/audit_shoulder_prefix.py" "$run/launch.json" \
    "$W/resting-clearance-and-atrium-017/fixed-shoulder-anchors.json" "$run/native/resting-body-motion-partition.csv" \
    "$run/verified-shoulders.json" --steps 1 32 64 96 128 || failure=1
"$PY" "$T/posterior-load-path-042/audit_boundary_bones.py" "$run" \
    "$T/thickness-rebuild-034/inputs/physical-bone-initial.bin" "$run/bone-endpoints.json" --step 128 > "$run/bone-audit.log" || failure=1
"$PY" "$T/load-transfer-capacity-048/compare_native.py" "$B/native-root-128" "$run" "$run/comparison.json" || failure=1
exit "$failure"
