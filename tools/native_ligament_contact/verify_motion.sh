#!/bin/bash
set -euo pipefail
tools=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$tools/../.." && pwd)
work=${1:?directory built by verify.sh}
package=${2:?composed current-package.json}
output=${3:?new native run directory with room for both geometry packs}
work=$(cd "$work" && pwd)
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
project="$HOME/Work/human-integrated-gaps-20261104"
scene="$project/resting-clearance-and-atrium-017"
e="$scene/extended-equilibrium-027"
c="$e/capsule-material-contact-042"
export PATH="/opt/homebrew/bin:$PATH" OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
# Retained evidence tools contain the material-branch and exact triangle predicates.
export PYTHONPATH="$project/resting-clearance-and-atrium-017/extended-equilibrium-027/capsule-material-contact-042/reference-delivery/evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/supported-shoulder-motion/physical-girdle-registration/articular-interfaces/capsular-material-contact/thoracic-reference:$project/resting-clearance-and-atrium-017/extended-equilibrium-027/capsule-material-contact-042/qualified-owner:$project/resting-clearance-and-atrium-017/extended-equilibrium-027/capsule-material-contact-042/delivery/evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/supported-shoulder-motion/physical-girdle-registration/articular-interfaces/capsular-material-contact:$project/resting-clearance-and-atrium-017/extended-equilibrium-027/source-shoulder-registration-041:/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/fit-source-root-001/myosim/checkout:$project/resting-clearance-and-atrium-017/extended-equilibrium-027/capsule-material-contact-042/qualified-owner:$project/resting-clearance-and-atrium-017/extended-equilibrium-027/capsule-material-contact-042/delivery/evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/supported-shoulder-motion/physical-girdle-registration/articular-interfaces/capsular-material-contact:/Users/numi/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/current-manifest-recovery-004/fit-source-root-001/myosim/checkout:$project/resting-clearance-and-atrium-017/extended-equilibrium-027/source-shoulder-registration-041:$project/resting-clearance-and-atrium-017/extended-equilibrium-027/supported-shoulder-motion-040/publish:$project/resting-clearance-and-atrium-017/extended-equilibrium-027/supported-shoulder-motion-040/publication-connected:$project/resting-clearance-and-atrium-017/extended-equilibrium-027/supported-shoulder-motion-040:$project/resting-clearance-and-atrium-017/extended-equilibrium-027/supported-shoulder-motion-040/publish:$project/coarse-hand-cavity-024/posterior-load-path-042:/Users/numi/Work/skin-fat-native-integration-209/tools:$project/resting-clearance-and-atrium-005:$project/resting-surface-clearance-001/src:$repo/src"
cfg=$(mktemp)
trap 'rm -f "$cfg"' EXIT
"$py" - "$package" "$cfg" <<'PY'
import hashlib,json,sys
from pathlib import Path
p,dst=map(Path,sys.argv[1:])
j=json.loads(p.read_text())
j['argv'][j['argv'].index('--muscle-step-count')+1]='8'
j['environment']['NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS']='8'
j['hashes'][str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
dst.write_text(json.dumps(j,indent=2)+'\n')
PY
"$py" "$work/run_scene.py" "$cfg" "$output"
cp "$cfg" "$output/motion-package.json"
capture="$output/native/accepted-geometry/step-8.mrvpack"
bones=$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1]))["argv"][3])' "$package")
"$py" "$tools/audit_motion_routes.py" "$project/resting-surface-clearance-001/audit.py" "$bones" "$c/emitted-sites.json" "$work/strip.json" "$c/emitted-native/stderr.log" "$capture" "$output/native/resting-body-motion-partition.csv" "$output/shoulder-motion.json"
"$py" - "$project/resting-surface-clearance-001/audit.py" "$capture" "$output/exact-skin.json" <<'PY'
import sys,runpy
from fast_skin_intersection_audit import install_for_float32
install_for_float32()
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
PY
"$py" "$e/paired-crossing-ownership-035/classify_ocular_presentation.py" "$output/exact-skin.json" "$output/native/resting-ocular-sclera-window.json" "$output/typed-skin.json"
"$py" - "$output/typed-skin.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
n=r['typed_scope']['counts']['other_tissue']
s=r['self_intersections']['count']
print(f'Outer skin has {s} self-intersection pairs and {n} nonocular tissue pairs')
if n or s:raise SystemExit(1)
PY
