#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
ref=$(cd "$here/.." && pwd)
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
export PYTHONPATH="$repo/src:$here:$ref:$cap:$art:$girdle:$d/publish:$d/publication-connected:$work/coarse-hand-cavity-024/posterior-load-path-042:$HOME/Work/skin-fat-native-integration-209/tools:$work/resting-clearance-and-atrium-005:$work/resting-surface-clearance-001/src:$myo"
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
gzip -dc "$here/candidate.node.gz" >"$out/candidate.node"
"$py" "$here/cofactor_test.py"
"$py" "$here/emit_register.py" "$out/candidate.node" "$p/constitutive-witness-repaired-tetgen/mesh.ele" "$owner/read_reference_volumes" "$p/close_live_parent_density_compiled" "$owner" "$p/floor-native-live-parent-026" "$prior/load_complete_reference.py" "$prior/native-step-zero/step-0.receipt.json" "$e/source-shoulder-registration-041/balanced-cap-bone/bodyparts3d-myosim-major-bones.nhbones" "$ref/free-arm.json" "$out"
for name in receipt.json carrier-capacity.json crossing-pairs.json; do
    cmp "$out/$name" "$here/$name"
done
echo "Reproduced the material reference and its recorded contact and density failures."
