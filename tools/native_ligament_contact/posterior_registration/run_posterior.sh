#!/bin/bash
set -euo pipefail
trial=$(cd "$(dirname "$0")" && pwd)
volume=NumiGluteal046
input="$trial/${5:-native-posterior}"
output="$trial/${6:-native-posterior-motion}"
mkdir -p "$output"
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
"$HOME/numi-human-prep-venv-20261005/bin/python3" run_posterior_replay.py "$trial/moment-owner" "$input/config.json" "$1" "$2" "$3" "$4" "$trial/posterior" "$trial/full-motion" "/Volumes/$volume" "$output"
