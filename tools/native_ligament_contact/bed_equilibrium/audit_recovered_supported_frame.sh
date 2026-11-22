#!/bin/bash
set -euo pipefail
run=$1
step=$2
root=$3
py=$4
work=$5
auditdir="$run/frame-$step-audit"
mkdir -p "$auditdir"
vol=NumiPelvicResponse048
test ! -e "/Volumes/$vol"
device=$(hdiutil attach -nomount ram://1677722 | awk 'NR==1{print $1}')
trap 'hdiutil detach "$device" > "$auditdir/ram-disk-released.log"' EXIT
diskutil info -plist "$device" > "$auditdir/device.plist"
"$py" - "$device" "$auditdir/device.plist" <<'PY'
import plistlib,sys
with open(sys.argv[2],'rb') as f:r=plistlib.load(f)
assert r['BusProtocol']=='Disk Image' and r['DeviceNode']==sys.argv[1]
assert not r['Internal'] and not r['Content'] and not r['MountPoint']
assert 800*1024**2<=r['IOKitSize']<821*1024**2
PY
diskutil eraseDisk APFS "$vol" "$device" > "$auditdir/ram-disk-format.log"
dir="/Volumes/$vol/native/accepted-geometry"
mkdir -p "$dir"
pack="$dir/step-$step.mrvpack"
cp "$run/step-$step.receipt.native.json" "$dir/step-$step.receipt.json"
/opt/homebrew/bin/zstd -qdc "$run/step-$step.mrvpack.zst" > "$pack"
"$py" - "$pack" <<'PY'
from pathlib import Path
import sys,json,hashlib
p=Path(sys.argv[1]);r=json.loads(p.with_suffix('.receipt.json').read_text())
with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert str(p)==r['accepted_pack_path'] and digest==r['pack_file_sha256']
PY
"$py" "$root/resting-clearance-and-atrium-017/audit_native_skin_endpoint.py" "$pack" "$auditdir/exact-skin-surface.json" > "$auditdir/skin-audit.log"
"$py" "$work/classify_native_skin_witnesses.py" "$root/resting-surface-clearance-001/audit.py" "$pack" "$auditdir/exact-skin-surface.json" "$auditdir/skin-intersection-types.json"
