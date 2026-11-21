#!/bin/bash
set -euo pipefail
export PATH="/opt/homebrew/bin:$PATH"
test "$(/bin/hostname | tr '[:upper:]' '[:lower:]' | cut -d. -f1)" = numi || [[ "$USER" == numi ]]
folder=$(cd "$(dirname "$0")" && pwd)
extended=$(dirname "$folder")
project=$(dirname "$(dirname "$extended")")
control="$project/resting-clearance-and-atrium-006/source-clearance-presentation-timing"
fine="$extended/full-volume-startup-032/fixed-layer-conditioned-033/local-conservative-transfer-034/native-witness-filtered-trial"
volume=NumiBound040
evidence="$folder/native-joint-bound"
[[ ! -e "/Volumes/$volume" && ! -e "$evidence" ]]
[[ "$(df -k /System/Volumes/Data | tail -1 | awk '{print $4}')" -ge 1572864 ]]
mkdir "$evidence"
device=$(/usr/bin/hdiutil attach -nomount ram://5767168 | awk 'NR==1{print $1}')
[[ "$device" =~ ^/dev/disk[0-9]+$ ]]
trap '/usr/bin/hdiutil detach "$device" > "$evidence/scratch-ejected.log" 2>&1' EXIT
/usr/sbin/diskutil info -plist "$device" > "$evidence/scratch-device.plist"
python="$HOME/numi-human-prep-venv-20261005/bin/python3"
"$python" - "$evidence/scratch-device.plist" "$device" <<'PYTHON'
import plistlib, sys
with open(sys.argv[1], "rb") as stream:
    d = plistlib.load(stream)
assert d["DeviceNode"] == sys.argv[2] and d["BusProtocol"] == "Disk Image"
assert not d["Internal"] and not d["Content"] and not d["MountPoint"]
assert d["IOKitSize"] == 2952790016
PYTHON
/usr/sbin/diskutil eraseDisk APFS "$volume" "$device" > "$evidence/scratch-format.log"
"$python" "$folder/replay_joint.py" "$folder/native-97kg" "/Volumes/$volume" "$evidence" "$folder/joint-source-bound.nhrigid" "$folder/joint-bound-receipt.json"
