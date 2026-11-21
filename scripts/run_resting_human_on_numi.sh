#!/bin/bash
# Launch the registered supine scene and its existing native acceptance observers on Numi.
set -euo pipefail
root="$HOME/Work/human-integrated-gaps-20261104"
base="$root/resting-clearance-and-atrium-017/extended-equilibrium-027"
work="$base/material-bed-equilibrium-046"
prior="$base/anatomical-extensor-and-bed-045"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
cfg="$work/supported-release/config.json"
binary="$prior/heart-build/numi-human-native"
mode=${1:-16}
case "$mode" in
16) steps=8000 ;;
60) steps=30000 ;;
--check) steps=0 ;;
*) echo "Usage: $0 [16|60|--check]" >&2; exit 2 ;;
esac
test -f "$cfg" && test -x "$binary" && test -x "$py"
"$py" - "$cfg" "$binary" <<'PY'
import hashlib,json,sys
from pathlib import Path
cfg,binary=map(Path,sys.argv[1:])
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
c=json.loads(cfg.read_text());a=c['argv'];p=Path(a[a.index('--soft-tissue-payload')+1])
if sha(binary)!='6bab4c5be982ce0d5edd4ecdc73c91ced5b0043d4de5a95fefd51978c521ce38':raise SystemExit('Native binary identity mismatch')
if sha(p)!='e3e189d0c69412a985aa75663a3959b95ad98a4b8027f5e7d050f86ef2eb454c':raise SystemExit('Muscle payload identity mismatch')
if c['headless'] or c['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':raise SystemExit('Wrong native scene')
if sha(cfg)!='57466a57719ba2a18ebc665175bab665d63e0c6ab77d0ab777da10b7dd3499bb':raise SystemExit('Launch configuration identity mismatch')
print('Verified supported native viewer, current bilateral muscle registration and variable pleural pressure.')
PY
if [[ "$steps" = 0 ]];then exit 0;fi
cd "$work"
export PYTHONPATH="$work:$prior:$base/capsule-material-contact-042/qualified-owner:$root/resting-surface-clearance-001/src"
# The run directory is distinct even when two launch commands arrive together.
out=$(mktemp -d "$work/viewer-${mode}s-XXXXXX")
echo "Native evidence directory: $out"
exec bash "$work/run_supported_duration.sh" "$binary" "$root/resting-surface-clearance-001" "$prior/changed_surface_pairs" supported-release "${out##*/}" "$steps"
