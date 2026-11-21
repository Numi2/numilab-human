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
scene="$p/floor-native-live-parent-026"
bone="$e/source-shoulder-registration-041/balanced-cap-bone/bodyparts3d-myosim-major-bones.nhbones"
pose="$prior/native-step-zero/step-0.receipt.json"
retained="$HOME/numi-human-retained-delivery-20261009/anatomy-completion-1276"
atlas="$retained/vein-owner-blend-source-append-dryrun-012-inputs/Sources/isa_BP3D_4.0_obj_99.zip"
myo="$retained/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/fit-source-root-001/myosim/checkout"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="$repo/src:$here:$ref/connected-volume:$ref:$cap:$art:$girdle:$d/publish:$d/publication-connected:$work/coarse-hand-cavity-024/posterior-load-path-042:$HOME/Work/skin-fat-native-integration-209/tools:$work/resting-clearance-and-atrium-005:$work/resting-surface-clearance-001/src:$myo"
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
args=("$d/native-97kg/launch.json" "$p/fixed-shoulder-anchors.json" "$d/compact-body-motion.csv" "$atlas" "$cap/emitted-sites.json" "$ref/head-centers.json")
"$py" "$here/solve_retained_strips.py" "${args[@]}" "$ref/spacing.json" "$here/feature-seeds.json" "$out/strip-domain.json" "$out/strip-features.json"
"$py" "$here/transition_work.py" "${args[@]}" "$ref/spacing.json" "$ref/shared-torso.json" "$out/strip-domain.json" "$out/transition-work.json"
"$py" "$ref/verify_articular_motion.py" "${args[@]}" "$ref/shared-torso.json" "$ref/spacing.json" "$out/strip-features.json" "$out/motion.json"
cmp "$here/motion.json" "$out/motion.json"
cmp "$here/transition-work.json" "$out/transition-work.json"
header="$HOME/Work/skin-fat-native-integration-209/apps"
echo "6c05956f6b3348e8d71c668099ee42e7dc29e2325335f4b06df821b09e524f98  $header/NumiHumanRegisteredSurfaceIntersection.hpp" | shasum -a 256 -c -
clang++ -std=c++17 -O3 -ffp-contract=off -I"$header" -dynamiclib "$here/boundary_pairs.cpp" -o "$out/predicates.dylib"
"$py" "$ref/connected-volume/prepare_contacts.py" "$scene" "$prior/load_complete_reference.py" "$pose" "$bone" "$ref/free-arm.json" "$out/boundaries.npz"
"$py" - "$scene/launch.json" "$ref/connected-volume" "$out" <<'PY'
import gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
launch,folder,out=map(Path,sys.argv[1:])
env=json.loads(launch.read_text())['environment']
p=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE'])
r=json.loads((folder/'receipt.json').read_text())
with p.open('rb') as source:
    assert hashlib.file_digest(source,'sha256').hexdigest()==r['source']['nodes']
with gzip.open(folder/'candidate.node.gz','rb') as source:
    assert hashlib.file_digest(source,'sha256').hexdigest()==r['candidate_node_sha256']
with gzip.open(folder/'candidate.node.gz','rt') as source:
    np.save(out/'candidate.npy',np.loadtxt(source,skiprows=1,usecols=(1,2,3),dtype=np.float32))
np.save(out/'source.npy',np.loadtxt(p,skiprows=1,usecols=(1,2,3),dtype=np.float32))
PY
for mesh in source candidate; do
    "$py" "$here/layer_contacts.py" "$out/$mesh.npy" "$out/boundaries.npz" "$out/$mesh-layers.json" "$out/predicates.dylib"
    "$py" "$here/check_boundary_witnesses.py" "$out/$mesh.npy" "$out/boundaries.npz" "$out/$mesh-layers.json"
    cmp "$here/$mesh-layers.json" "$out/$mesh-layers.json"
done
echo "Reproduced both physical references and the bilateral prescribed shoulder qualification."
