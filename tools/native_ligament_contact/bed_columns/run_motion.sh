#!/bin/bash
set -euo pipefail
trial=$(cd "$(dirname "$0")" && pwd)
volume=NumiMaterialBed045
output="$trial/${2:-native-motion}"
[[ ! -e "/Volumes/$volume" ]]
trap 'if [[ -n "${device:-}" ]]; then /usr/bin/hdiutil detach "$device" > "$output/ram-disk-released.log"; fi' EXIT
device=$(/usr/bin/hdiutil attach -nomount ram://6553600 | awk 'NR==1{print $1}')
[[ "$device" =~ ^/dev/disk[0-9]+$ ]]
/usr/sbin/diskutil info -plist "$device" > "$output/ram-disk-info.plist"
python3 - "$device" "$output/ram-disk-info.plist" <<'PY'
import plistlib
import sys
with open(sys.argv[2], "rb") as f:
    info = plistlib.load(f)
assert info["DeviceNode"] == sys.argv[1] and info["BusProtocol"] == "Disk Image"
assert not info["Internal"] and not info["Content"] and not info["MountPoint"]
assert info["IOKitSize"] == 3355443200
PY
/usr/sbin/diskutil eraseDisk APFS "$volume" "$device" > "$output/ram-disk-format.log"
cd "$trial"
"$HOME/numi-human-prep-venv-20261005/bin/python3" run_supported.py "$1" "$trial/native-build/numi-human-native" "/Volumes/$volume" "$output" "${3:-8000}"
