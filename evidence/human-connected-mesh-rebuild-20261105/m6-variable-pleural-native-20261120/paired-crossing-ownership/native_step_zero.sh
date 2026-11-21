#!/bin/bash
set -euo pipefail
trial=$(cd "$(dirname "$0")" && pwd)
e=$(dirname "$trial")
p=$(dirname "$e")
source="$p/floor-native-live-parent-026"
volume=NumiReference035
output="$trial/native-step-zero"
mkdir "$output"
finish() {
  set +e
  run="/Volumes/$volume/pilot"
  if [[ -d "$run" ]]; then
    for name in launch.json execution.json stdout.log stderr.log; do
      if [[ -f "$run/$name" ]]; then cp "$run/$name" "$output/"; fi
    done
    capture="$run/native/accepted-geometry/step-0.receipt.json"
    if [[ -f "$capture" ]]; then
        cp "$capture" "$output/"
    fi
    for pack in "$run/native/resting-human.mrvpack" "$run/native/accepted-geometry/step-0.mrvpack"; do
      if [[ -f "$pack" ]]; then shasum -a 256 "$pack"; fi
    done > "$output/capture-sha256.txt"
  fi
  if [[ -n "${device:-}" ]]; then
    /usr/bin/hdiutil detach "$device" > "$output/ram-disk-released.log" 2>&1
  fi
}
[[ ! -e "/Volumes/$volume" ]]
[[ "$(df -k "$HOME" | tail -1 | awk '{print $4}')" -ge 1258291 ]]
trap finish EXIT
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
metal=$(jq -er ".environment.NUMI_MATTER_METALLIB_OVERRIDE" "$source/launch.json")
"$HOME/numi-human-prep-venv-20261005/bin/python3" "$trial/run_step_zero.py" \
    "$source" "$metal" "/Volumes/$volume/pilot" --roots 1 --capture-step 0 --budget-wall-seconds 410
