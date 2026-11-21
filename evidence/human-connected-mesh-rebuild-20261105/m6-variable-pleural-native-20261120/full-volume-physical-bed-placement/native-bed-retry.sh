#!/bin/bash
set -euo pipefail
trial=$(cd "$(dirname "$0")" && pwd)
d=$(dirname "$trial")
p=$(dirname "$d")
volume=NumiBedRetry030
finish() {
  set +e
  if [[ -d "/Volumes/$volume/pilot" ]]; then
    run="/Volumes/$volume/pilot"
    mkdir -p "$trial/pilot-native-ram-retry"
    for name in launch.json execution.json stdout.log stderr.log; do
       if [[ -f "$run/$name" ]]; then cp "$run/$name" "$trial/pilot-native-ram-retry/"; fi
    done
    if [[ -f "$run/native/resting-coupled.csv" ]]; then
       cp "$run/native/resting-coupled.csv" "$trial/pilot-native-ram-retry/"
    fi
    for name in resting-surface-audit.csv resting-com-momentum-diagnostic.csv; do
      if [[ -f "$run/native/$name" ]]; then cp "$run/native/$name" "$trial/pilot-native-ram-retry/"; fi
    done
    if [[ -f "$run/native/resting-human.mrvpack" ]]; then
      shasum -a 256 "$run/native/resting-human.mrvpack" > "$trial/pilot-native-ram-retry/initial-pack.txt"
    fi
    if compgen -G "$run/native/registered-volume-fractional-mass-*.json" >/dev/null; then
       cp "$run/native"/registered-volume-fractional-mass-*.json "$trial/pilot-native-ram-retry/"
    fi
  fi
  if [[ -n "${device:-}" ]]; then /usr/bin/hdiutil detach "$device" > "$trial/pilot-native-ram-retry-disk-released.log" 2>&1; fi
}
[[ ! -e "/Volumes/$volume" && ! -e "$trial/pilot-native-ram-retry" ]]
[[ "$(df -k "$HOME" | tail -1 | awk '{print $4}')" -ge 1258291 ]]
trap finish EXIT
device=$(/usr/bin/hdiutil attach -nomount ram://5767168 | awk 'NR==1{print $1}')
[[ "$device" =~ ^/dev/disk[0-9]+$ ]]
/usr/sbin/diskutil info -plist "$device" > "$trial/ram-disk-info.plist"
python3 - "$device" "$trial/ram-disk-info.plist" <<'PY'
import plistlib,sys
with open(sys.argv[2],'rb') as f:
 d=plistlib.load(f)
assert d["DeviceNode"]==sys.argv[1] and d["BusProtocol"]=="Disk Image"
assert not d["Internal"] and not d["Content"] and not d["MountPoint"]
assert d["IOKitSize"] == 2952790016
PY
/usr/sbin/diskutil eraseDisk APFS "$volume" "$device" > "$trial/pilot-native-ram-retry-format.log"
"$HOME/numi-human-prep-venv-20261005/bin/python3" "$trial/run_native_bed.py"     "$p/floor-native-live-parent-026" "$d/NumiMatter.metallib" "/Volumes/$volume/pilot"     --roots 8 --headless --budget-wall-seconds 490 --bed-placement "$trial/bed-placement.json"
