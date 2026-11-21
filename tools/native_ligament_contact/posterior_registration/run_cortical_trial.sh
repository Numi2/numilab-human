#!/bin/bash
set -euo pipefail
src=$1
assets=$2
bin=$3
run_dir=$4
py=$5
steps=$6
enabled=$7
volume=NumiPelvicResponse048
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

"$py" "$(dirname "$0")/run_cortical_trial.py" "$src" "$assets" "$bin" "$run_dir" "/Volumes/$volume" "$steps" "$enabled"
