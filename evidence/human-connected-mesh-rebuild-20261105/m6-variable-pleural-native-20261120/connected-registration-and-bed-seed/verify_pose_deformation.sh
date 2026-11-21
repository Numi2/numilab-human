#!/bin/bash
set -euo pipefail
if [[ $# != 3 ]]; then
  echo 'Usage: verify_pose_deformation.sh RUNTIME_WORKTREE TETGEN_DIRECTORY PYTHON' >&2
  exit 2
fi
cd "$(dirname "$0")"
runtime=$1
mesh=$2
py=$3
xcrun -sdk macosx metal -std=metal4.0 -O3 -fno-fast-math \
    -I "$runtime/matter/include" -I "$runtime/include" \
    -c pose_displacement_probe.metal -o pose_displacement_probe.air
xcrun -sdk macosx metallib pose_displacement_probe.air -o pose_displacement_probe.metallib
xcrun clang++ -std=c++20 -O3 -fno-fast-math -fobjc-arc \
    -I "$runtime/matter/include" -I "$runtime/include" -framework Foundation \
    -framework Metal pose_displacement_probe.mm -o pose_displacement_probe
"$py" audit_bed_pose_volume.py "$mesh" bed-pose-fit.json --metal-fixture \
    | ./pose_displacement_probe "$PWD/pose_displacement_probe.metallib"
