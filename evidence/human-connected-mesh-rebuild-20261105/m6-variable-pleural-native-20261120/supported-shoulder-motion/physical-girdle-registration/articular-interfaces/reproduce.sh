#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
base=$(cd "$here/.." && pwd)
root="$HOME/Work/human-integrated-gaps-20261104"
project="$root/resting-clearance-and-atrium-017"
extended="$project/extended-equilibrium-027"
shoulder="$extended/supported-shoulder-motion-040"
full="$project/floor-native-live-parent-026"
prior="$extended/paired-crossing-ownership-035"
source="$HOME/numi-human-retained-delivery-20261009/anatomy-completion-1276"
atlas="$source/vein-owner-blend-source-append-dryrun-012-inputs/Sources/isa_BP3D_4.0_obj_99.zip"
myo="$source/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/fit-source-root-001/myosim/checkout"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
out=$(mktemp -d)
cd "$out"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 PATH="/opt/homebrew/bin:$PATH"
export PYTHONPATH="$out:$here:$base:$shoulder/publish:$shoulder/publication-connected:$root/coarse-hand-cavity-024/posterior-load-path-042:$HOME/Work/skin-fat-native-integration-209/tools:$root/resting-clearance-and-atrium-005:$root/resting-surface-clearance-001/src:$myo"
for name in verify_bone_motion rebind_regional_tendon;do
  cp "$base/$name.py" "$name.py"
  patch -s -p1 < "$here/$name.patch"
done
args=("$shoulder/native-97kg/launch.json" "$project/fixed-shoulder-anchors.json" "$shoulder/compact-body-motion.csv")
capture=("$full" "$prior/load_complete_reference.py" "$prior/native-step-zero/step-0.receipt.json")
b=bodyparts3d-myosim-major-bones.nhbones
t=numi-human-tendon-attachments.nhtendon
"$py" "$here/audit_common_reference.py" "${args[0]}" "$atlas" raw.json
cmp raw.json "$here/common-reference.json"
"$py" "$here/test_local_glenoid_reference.py" "${args[@]}" "$atlas" "${capture[2]}" glenoid.json
cmp glenoid.json "$here/local-glenoid.json"
"$py" "$here/fit_contact_footprint.py" "${args[@]}" "$atlas" "${capture[2]}" footprint
"$py" "$here/conserve_articular_volume.py" "${capture[@]}" "footprint/$b" balanced
cmp balanced/balance.json "$here/balance.json"
arc="$source/input-retention-1380/verified-inputs/147-myo_sim-33c89c2b.tar.gz"
"$py" "$base/verify_sources.py" "$arc" "$myo" source-verification.json
cmp source-verification.json "$base/checkout-sources.json"
"$py" "$base/model_surfaces.py" "$arc" "$out"
"$py" "$here/screen_material_transport.py" "${capture[@]}" "balanced/$b" source-bones.npz model-source.json supported.json
cmp supported.json "$here/final-support.json"
# Verify the short standard Git recovery path against the independently cooked asset.
mkdir recovery
git -C recovery init -q
parent_bone=$(jq -r '.argv[3]' "${args[0]}")
test "$(shasum -a 256 "$parent_bone" |cut -d' ' -f1)" = "$(jq -r '.bones_parent_sha256' "$here/pins.json")"
cp "$parent_bone" "recovery/$b"
git -C recovery apply --check "$here/clavicle-bones.patch"
git -C recovery apply "$here/clavicle-bones.patch"
cmp "recovery/$b" "balanced/$b"
"$py" "$here/screen_glenoid_neighbors.py" "${args[@]}" "$atlas" "${capture[@]}" "balanced/$b" neighbors-glenoid.json
cmp neighbors-glenoid.json "$here/glenoid-neighbors.json"
"$py" "$here/audit_articular_material.py" "${args[0]}" "balanced/$b" geometry.json
cmp geometry.json "$here/final-material.json"
"$py" verify_bone_motion.py "${args[@]}" "balanced/$b" motion.json > motion.log
cmp motion.json "$here/final-motion.json"
"$py" "$base/audit_tissue_neighbors.py" "${capture[@]}" "balanced/$b" tissues.json
cmp tissues.json "$here/final-tissues.json"
"$py" "$here/compare_contact_witnesses.py" "${capture[@]}" "balanced/$b" witnesses.json
cmp witnesses.json "$here/exact-neighbors.json"
PYTHONPATH="$HOME/numilab-human/src:$PYTHONPATH" "$py" - "${args[0]}" <<'PY'
import json,hashlib,sys
from pathlib import Path
from numilab_human.model import numi_human_tendon_attachment_envelope_payload as cook
old=Path(json.load(open(sys.argv[1]))['argv'][3])
m=json.loads(old.with_suffix('.manifest.json').read_text())['source']['runtime_reference']['manifest']
src=Path(m['file']);assert hashlib.sha256(src.read_bytes()).hexdigest()==m['sha256']
for folder,out in ((old.parent,'parent-regional'),(Path('balanced'),'candidate-regional')):
    cook(src.parent,folder,Path(out),migrate_semantic_rigid_foot_endpoints=True)
PY
"$py" rebind_regional_tendon.py "${args[0]}" balanced parent-regional candidate-regional bound
cmp "bound/$t" "$here/$t"
cmp bound/comparison.json "$here/comparison.json"
"$py" "$base/cook_material_sites.py" "$shoulder" "$project" "$HOME/Work/skin-fat-native-integration-209" "balanced/$b" "${args[0]}" "${args[1]}" - catalog
cmp catalog/result.json "$here/native-catalog.json"
"$py" "$here/run_articular_scene.py" "$shoulder/native-scene-admission/launch.json" "$shoulder" catalog "balanced/$b" "bound/$t" native
"$py" - "$here/native-result.json" native/result.json <<'PY'
import json,sys
a,b=[json.load(open(p)) for p in sys.argv[1:]]
assert a['exit_code']==b['exit_code']!=0
for failure in ('side=0 stable_id=13','side=1 stable_id=12'):
    assert any(failure in e for e in a['errors']) and any(failure in e for e in b['errors'])
PY
echo "Clavicle geometry, force bindings and complete-scene admission reproduced in $out"
