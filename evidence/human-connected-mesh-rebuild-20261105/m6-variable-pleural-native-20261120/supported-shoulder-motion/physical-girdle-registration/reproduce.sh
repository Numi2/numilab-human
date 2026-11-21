#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root="$HOME/Work/human-integrated-gaps-20261104"
p="$root/resting-clearance-and-atrium-017"
d="$p/extended-equilibrium-027/supported-shoulder-motion-040"
src="$root/resting-surface-clearance-001/src"
g="$HOME/numi-human-retained-delivery-20261009/anatomy-completion-1276"
atlas="$g/vein-owner-blend-source-append-dryrun-012-inputs/Sources/isa_BP3D_4.0_obj_99.zip"
arc="$g/input-retention-1380/verified-inputs/147-myo_sim-33c89c2b.tar.gz"
myo="$g/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/fit-source-root-001/myosim/checkout"
out=$(mktemp -d)
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
export OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="$here:$d/publish:$d/publication-connected:$root/coarse-hand-cavity-024/posterior-load-path-042:$HOME/Work/skin-fat-native-integration-209/tools:$root/resting-clearance-and-atrium-005:$src:$myo"
cd "$out"
args=("$d/native-97kg/launch.json" "$p/fixed-shoulder-anchors.json" "$d/compact-body-motion.csv")
"$py" "$here/verify_sources.py" "$arc" "$myo" sources.json
cmp sources.json "$here/checkout-sources.json"
"$py" "$here/model_surfaces.py" "$arc" "$out"
"$py" "$here/fit_source_surfaces.py" rigid "${args[@]}" source-bones.npz rigid.json
cmp rigid.json "$here/rigid.json"
"$py" "$here/connected_registration.py" "${args[@]}" "$atlas" rigid.json connected.json
cmp connected.json "$here/connected.json"
"$py" "$here/emit_registration.py" "${args[@]}" "$atlas" "$(jq -r '.parent_registration.file' "$here/pins.json")" connected.json bone-candidate
b=bone-candidate/bodyparts3d-myosim-major-bones.nhbones
test "$(shasum -a 256 "$b" | cut -d' ' -f1)" = "$(jq -r --arg b "$b" '.outputs[$b]' "$here/pins.json")"
"$py" "$here/verify_bone_motion.py" "${args[@]}" "$b" motion.json
cmp motion.json "$here/emitted-motion.json"
"$py" "$here/source_ligament_routes.py" "${args[@]}" "$b" source-bones.npz routes.json
"$py" "$here/verify_source_wraps.py" "${args[@]}" "$b" routes.json wraps.json
cmp wraps.json "$here/registered-wraps.json"
PYTHONPATH="$HOME/numilab-human/src:$PYTHONPATH" "$py" - "${args[0]}" <<'PY'
from pathlib import Path
import sys,json
from numilab_human.model import numi_human_tendon_attachment_envelope_payload as compile
old=Path(json.load(open(sys.argv[1]))['argv'][3])
source=json.loads(old.with_suffix('.manifest.json').read_text())['source']['runtime_reference']['manifest']
import hashlib
art=Path(source['file'])
assert hashlib.sha256(art.read_bytes()).hexdigest()==source['sha256']
for folder,target in ((old.parent,'base'),(Path('bone-candidate'),'regional')):
    compile(art.parent,folder,Path(target),migrate_semantic_rigid_foot_endpoints=True)
PY
"$py" "$here/rebind_regional_tendon.py" "${args[0]}" bone-candidate base regional bound
cmp bound/numi-human-tendon-attachments.nhtendon "$here/numi-human-tendon-attachments.nhtendon"
"$py" "$here/fit_material_sites.py" "${args[@]}" "$atlas" "$b" "$d/named-regions.json" sites.json
cmp sites.json "$here/material-sites.json"
"$py" "$here/cook_material_sites.py" "$d" "$p" "$HOME/Work/skin-fat-native-integration-209" "$b" "${args[0]}" "${args[1]}" sites.json cook
cmp cook/result.json "$here/catalog-result.json"
u="$p/extended-equilibrium-027/paired-crossing-ownership-035"
"$py" "$here/audit_tissue_neighbors.py" "$p/floor-native-live-parent-026" "$u/load_complete_reference.py" "$u/native-step-zero/step-0.receipt.json" "$b" tissues.json
cmp tissues.json "$here/tissue-neighbors.json"
"$py" "$here/fit_source_wraps.py" "${args[@]}" "$b" routes.json footprints.json
cmp footprints.json "$here/footprint-wraps.json"
echo "Retained girdle, tissue, tendon and catalog results reproduced in $out"
