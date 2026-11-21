#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ref=$(cd "$here/.." && pwd)
original="$ref/connected-volume"
checker="$ref/contact-qualification"
cap=$(cd "$ref/.." && pwd)
art=$(cd "$cap/.." && pwd)
girdle=$(cd "$art/.." && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
work="$HOME/Work/human-integrated-gaps-20261104"
p="$work/resting-clearance-and-atrium-017"
e="$p/extended-equilibrium-027"
d="$e/supported-shoulder-motion-040"
prior="$e/paired-crossing-ownership-035"
owner="$p/refit-motion-qualification"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
retained="$HOME/numi-human-retained-delivery-20261009/anatomy-completion-1276"
myo="$retained/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/fit-source-root-001/myosim/checkout"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="$repo/src:$here:$checker:$original:$ref:$cap:$art:$girdle:$d/publish:$d/publication-connected:$work/coarse-hand-cavity-024/posterior-load-path-042:$HOME/Work/skin-fat-native-integration-209/tools:$work/resting-clearance-and-atrium-005:$work/resting-surface-clearance-001/src:$myo"
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
gzip -dc "$original/candidate.node.gz" >"$out/candidate.node"
gzip -dc "$here/reference.patch.gz" | patch -s "$out/candidate.node"
"$py" "$here/test_joint_contact_rows.py"
"$py" "$original/emit_register.py" "$out/candidate.node" "$p/constitutive-witness-repaired-tetgen/mesh.ele" "$owner/read_reference_volumes" "$p/close_live_parent_density_compiled" "$owner" "$p/floor-native-live-parent-026" "$prior/load_complete_reference.py" "$prior/native-step-zero/step-0.receipt.json" "$e/source-shoulder-registration-041/balanced-cap-bone/bodyparts3d-myosim-major-bones.nhbones" "$ref/free-arm.json" "$out"
cmp "$here/reference.json" "$out/receipt.json"
header="$HOME/Work/skin-fat-native-integration-209/apps"
echo "6c05956f6b3348e8d71c668099ee42e7dc29e2325335f4b06df821b09e524f98  $header/NumiHumanRegisteredSurfaceIntersection.hpp" | shasum -a 256 -c -
clang++ -std=c++17 -O3 -ffp-contract=off -I"$header" -dynamiclib "$checker/boundary_pairs.cpp" -o "$out/predicates.dylib"
"$py" "$checker/layer_contacts.py" "$out/positions.npy" "$out/boundaries.npz" "$out/material-layers.json" "$out/predicates.dylib"
"$py" "$checker/check_boundary_witnesses.py" "$out/positions.npy" "$out/boundaries.npz" "$out/material-layers.json"
cmp "$here/material-layers.json" "$out/material-layers.json"
echo "Reproduced the full tissue reference checks and remaining defects on the same geometry."
