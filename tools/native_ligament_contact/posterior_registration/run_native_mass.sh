#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
prior=$1
run_dir=$2
ledger=$3
owner=$4
py=$5
volume=NumiPelvicMassCheck047
mkdir -p "$run_dir"
test ! -e "/Volumes/$volume"
device=$(hdiutil attach -nomount ram://6553600 | awk 'NR==1{print $1}')
trap 'hdiutil detach "$device" > "$run_dir/ram-disk-released.log"' EXIT
diskutil info -plist "$device" >"$run_dir/ram-disk-info.plist"
"$py" - "$device" "$run_dir/ram-disk-info.plist" <<'PY'
import plistlib,sys
with open(sys.argv[2],'rb') as f:r=plistlib.load(f)
assert r['DeviceNode']==sys.argv[1] and r['BusProtocol']=='Disk Image' and r['IOKitSize']==3355443200
assert not r['Internal'] and not r['Content'] and not r['MountPoint']
PY
diskutil eraseDisk APFS "$volume" "$device" > "$run_dir/ram-disk-format.log"
"$py" - "$prior" "$run_dir/config.json" "$ledger" <<'PY'
import json,sys,hashlib
from pathlib import Path
p,out,ledger=map(Path,sys.argv[1:]);cfg=json.loads(p.read_text())
cfg['environment']['NUMI_HUMAN_PELVIC_MASS_PREFLIGHT']=str(ledger)
cfg['hashes'][str(ledger)]=hashlib.sha256(ledger.read_bytes()).hexdigest()
out.write_text(json.dumps(cfg,indent=2)+'\n')
PY
scratch="/Volumes/$volume/scene"
"$py" "$owner/run_scene.py" "$run_dir/config.json" "$scratch"
cp "$scratch"/{launch.json,result.json,stdout.log,stderr.log} "$run_dir"
if test -f "$scratch/native/resting-coupled.csv";then cp "$scratch/native/resting-coupled.csv" "$run_dir";fi
if test -f "$scratch/native/accepted-geometry/step-8.receipt.json";then cp "$scratch/native/accepted-geometry/step-8.receipt.json" "$run_dir";fi
"$py" - "$run_dir" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
r=json.loads((p/'result.json').read_text())
print(r,flush=True)
if r['exit_code'] or r['trace_rows']!=8:raise SystemExit(1)
PY
