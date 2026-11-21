#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
parent=$(cd "$here/.." && pwd)
art=$(cd "$parent/.." && pwd)
girdle=$(cd "$art/.." && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
work="$HOME/Work/human-integrated-gaps-20261104"
p="$work/resting-clearance-and-atrium-017"
e="$p/extended-equilibrium-027"
d="$e/supported-shoulder-motion-040"
prior="$e/paired-crossing-ownership-035"
scene="$p/floor-native-live-parent-026"
receipt="$prior/native-step-zero/step-0.receipt.json"
j="$e/source-shoulder-registration-041"
bone="$j/balanced-cap-bone/bodyparts3d-myosim-major-bones.nhbones"
retained="$HOME/numi-human-retained-delivery-20261009/anatomy-completion-1276"
z="$retained/vein-owner-blend-source-append-dryrun-012-inputs/Sources/isa_BP3D_4.0_obj_99.zip"
myo="$retained/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/fit-source-root-001/myosim/checkout"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="$repo/src:$here:$parent:$art:$girdle:$d/publish:$d/publication-connected:$work/coarse-hand-cavity-024/posterior-load-path-042:$HOME/Work/skin-fat-native-integration-209/tools:$work/resting-clearance-and-atrium-005:$work/resting-surface-clearance-001/src:$myo"
args=("$d/native-97kg/launch.json" "$p/fixed-shoulder-anchors.json" "$d/compact-body-motion.csv")
capture=("$scene" "$prior/load_complete_reference.py" "$receipt" "$bone")
out=$(mktemp -d)
cd "$out"
run() {
    local script=$1 result=$2
    shift 2
    "$py" "$here/$script.py" "$@" "$result"
    cmp "$result" "$here/$result"
}
run shared_thoracic_reference shared-torso.json "$z" "${capture[@]}" "$parent/joint-center.json" "$parent/emitted-sites.json"
run fit_humeral_head_center head-centers.json "${args[@]}" "$z" "$girdle/source-routes.json" "$parent/emitted-sites.json" "$parent/joint-center.json" shared-torso.json "$receipt"
run preserve_glenoid_spacing spacing.json "${args[@]}" "$z" head-centers.json shared-torso.json
run close_shoulder_arm_chain chain.json "${args[@]}" head-centers.json shared-torso.json "$receipt" spacing.json
run verify_articular_motion kinematics.json "${args[@]}" "$z" "$parent/emitted-sites.json" head-centers.json shared-torso.json spacing.json "$here/material-features.json"
run register_connected_arm free-arm.json "${args[@]}" "$z" "${capture[@]}" shared-torso.json head-centers.json chain.json spacing.json
run transfer_connected_volume volume-screen.json "${capture[@]}" head-centers.json free-arm.json
"$py" -m unittest discover -s "$repo/tests" -p test_shoulder_registration.py
echo "Reproduced thoracic registration, connected arms, physical skin/fat and capsular checks in $out"
