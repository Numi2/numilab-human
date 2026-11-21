#!/bin/bash
set -euo pipefail
trial=$(cd "$(dirname "$0")" && pwd)
startup=$(dirname "$(dirname "$trial")")
d=$(dirname "$startup")
proposal=${1:-native-proposal}
evidence=${2:-native-trial}
p=$(dirname "$d")
volume=NumiCavity034
finish() {
  set +e
  if [[ -d "/Volumes/$volume/pilot" ]]; then
    run="/Volumes/$volume/pilot"
    mkdir -p "$trial/$evidence"
    for name in launch.json execution.json stdout.log stderr.log; do
       if [[ -f "$run/$name" ]]; then cp "$run/$name" "$trial/$evidence/"; fi
    done
    if [[ -f "$run/native/resting-coupled.csv" ]]; then
       cp "$run/native/resting-coupled.csv" "$trial/$evidence/"
    fi
    for name in resting-surface-audit.csv resting-com-momentum-diagnostic.csv; do
      if [[ -f "$run/native/$name" ]]; then cp "$run/native/$name" "$trial/$evidence/"; fi
    done
    if [[ -f "$run/native/resting-human.mrvpack" ]]; then
      shasum -a 256 "$run/native/resting-human.mrvpack" > "$trial/$evidence/initial-pack.txt"
    fi
    if compgen -G "$run/native/registered-volume-fractional-mass-*.json" >/dev/null; then
       cp "$run/native"/registered-volume-fractional-mass-*.json "$trial/$evidence/"
    fi
  fi
  if [[ -n "${device:-}" ]]; then /usr/bin/hdiutil detach "$device" > "$trial/$evidence-disk-released.log" 2>&1; fi
}
[[ ! -e "/Volumes/$volume" && ! -e "$trial/$evidence" ]]
[[ "$(df -k "$HOME" | tail -1 | awk '{print $4}')" -ge 1258291 ]]
trap finish EXIT
device=$(/usr/bin/hdiutil attach -nomount ram://5767168 | awk 'NR==1{print $1}')
[[ "$device" =~ ^/dev/disk[0-9]+$ ]]
/usr/sbin/diskutil info -plist "$device" > "$trial/$evidence-ram-disk-info.plist"
python3 - "$device" "$trial/$evidence-ram-disk-info.plist" <<'PY'
import plistlib,sys
with open(sys.argv[2],'rb') as f:
 d=plistlib.load(f)
assert d["DeviceNode"]==sys.argv[1] and d["BusProtocol"]=="Disk Image"
assert not d["Internal"] and not d["Content"] and not d["MountPoint"]
assert d["IOKitSize"] == 2952790016
PY
/usr/sbin/diskutil eraseDisk APFS "$volume" "$device" > "$trial/$evidence-format.log"
"$HOME/numi-human-prep-venv-20261005/bin/python3" "$trial/run_native_candidate.py"     "$p/floor-native-live-parent-026" "$d/full-volume-grouped-loads-031/NumiMatter.metallib" --host-library "$startup/libmetalrobo.dylib" "/Volumes/$volume/pilot"     --cavities "$trial/$proposal/fixed-boundary-volume-proposal.npz" --roots 3 --profile-root 1 --headless --budget-wall-seconds 490 --bed-placement "$d/full-volume-bed-placement-030/bed-placement.json"
