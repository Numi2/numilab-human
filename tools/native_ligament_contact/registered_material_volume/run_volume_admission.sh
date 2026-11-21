#!/bin/bash
set -euo pipefail
source_launch=$1
selection=$2
app_binary=$3
output=$4
py=$5
mkdir -p "$output"
mount=/Volumes/NumiVolumeAdmission
test ! -e "$mount"
disk=$(hdiutil attach -nomount ram://6553600 | awk 'NR==1{print $1}')
trap 'hdiutil detach "$disk" >"$output/ram-disk-released.log"' EXIT
diskutil info -plist "$disk" >"$output/ram-disk-info.plist"
"$py" - "$disk" "$output/ram-disk-info.plist" <<'PY'
import plistlib,sys
r=plistlib.load(open(sys.argv[2],'rb'))
assert r['DeviceNode']==sys.argv[1] and r['BusProtocol']=='Disk Image' and r['IOKitSize']==3355443200 and not r['Internal'] and not r['MountPoint']
PY
diskutil eraseDisk APFS NumiVolumeAdmission "$disk" >"$output/ram-disk-format.log"
"$py" "$(dirname "$0")/admit_current_volume.py" "$source_launch" "$selection" "$app_binary" "$mount" "$output"
