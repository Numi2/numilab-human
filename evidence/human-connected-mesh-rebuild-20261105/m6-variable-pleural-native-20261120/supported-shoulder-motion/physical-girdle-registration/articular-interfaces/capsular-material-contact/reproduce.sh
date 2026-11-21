#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(git -C "$here" rev-parse --show-toplevel)
art=$(cd "$here/.." && pwd)
girdle=$(cd "$art/.." && pwd)
work="$HOME/Work/human-integrated-gaps-20261104"
p="$work/resting-clearance-and-atrium-017"
e="$p/extended-equilibrium-027"
d="$e/supported-shoulder-motion-040"
retained="$HOME/numi-human-retained-delivery-20261009/anatomy-completion-1276"
z="$retained/vein-owner-blend-source-append-dryrun-012-inputs/Sources/isa_BP3D_4.0_obj_99.zip"
myo="$retained/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/fit-source-root-001/myosim/checkout"
host="$HOME/Work/skin-fat-native-integration-209"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="$repo/src:$here:$art:$girdle:$d/publish:$d/publication-connected:$work/coarse-hand-cavity-024/posterior-load-path-042:$host/tools:$work/resting-clearance-and-atrium-005:$work/resting-surface-clearance-001/src:$myo"
args=("$d/native-97kg/launch.json" "$p/fixed-shoulder-anchors.json" "$d/compact-body-motion.csv")
parent=$(jq -r '.argv[3]' "${args[0]}")
j="$e/source-shoulder-registration-041"
bone="$j/balanced-cap-bone/bodyparts3d-myosim-major-bones.nhbones"
tendon="$j/final-tendon/numi-human-tendon-attachments.nhtendon"
receipt="$e/paired-crossing-ownership-035/native-step-zero/step-0.receipt.json"
scratch=$(mktemp -d)
cd "$scratch"
"$py" -m numilab_human.capsular_material_registration "$bone" "$parent" "$z" "$girdle/source-routes.json" "$girdle/rigid.json" "$girdle/material-sites.json" emitted-sites.json
cmp emitted-sites.json "$here/emitted-sites.json"
"$py" "$girdle/cook_material_sites.py" "$d" "$p" "$host" "$bone" "${args[0]}" "${args[1]}" emitted-sites.json "$scratch/catalog" > catalog.log
cmp catalog/result.json "$here/catalog.json"
"$py" "$here/compare_native_material_points.py" catalog "$p" "$host" "$bone" emitted-sites.json native-parity.json
cmp native-parity.json "$here/native-parity.json"
"$py" "$here/diagnose_reference_frames.py" emitted-sites.json "$girdle/source-routes.json" "$girdle/rigid.json" frames.json
cmp frames.json "$here/frames.json"
"$py" "$here/qualify_joint_center.py" "${args[@]}" "$j/source-bones.npz" "$girdle/rigid.json" emitted-sites.json "$receipt" joint-center.json
cmp joint-center.json "$here/joint-center.json"
"$py" "$here/verify_donor_shoulder.py" "${args[@]}" "$j/source-bones.npz" "$girdle/source-routes.json" "$receipt" donor-check.json
cmp donor-check.json "$here/donor-check.json"
"$py" "$here/certify_material_branch.py" "${args[@]}" "$bone" emitted-sites.json "$receipt" "$here/branches.json" certified.json
cmp certified.json "$here/certified.json"
"$py" "$art/run_articular_scene.py" "$d/native-scene-admission/launch.json" "$d" catalog "$bone" "$tendon" native > launch.log
"$py" - native "$here/native.json" <<'PY'
import json,sys
from pathlib import Path
run,ref=map(Path,sys.argv[1:])
doc=json.loads((run/'result.json').read_text())
stored=json.loads(ref.read_text())
assert doc['exit_code']==stored['exit_code']==1
assert [e for e in doc['errors'] if e.startswith('registered ligament build failed')]==stored['errors']
assert len((run/'native/resting-coupled.csv').read_text().splitlines())-1==stored['accepted_trace_rows']==0
PY
echo "Reproduced shoulder material registration, force checks and the native entrance rejection in $scratch"
