#!/bin/bash
set -euo pipefail
tools=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$tools/../.." && pwd)
ev="$repo/evidence/human-connected-mesh-rebuild-20261105/m6-variable-pleural-native-20261120/native-ligament-contact"
p="$HOME/Work/human-integrated-gaps-20261104/resting-clearance-and-atrium-017"
e="$p/extended-equilibrium-027"
c="$e/capsule-material-contact-042"
app="$HOME/Work/skin-fat-native-integration-209"
py="$HOME/numi-human-prep-venv-20261005/bin/python3"
work=${1:?pass a new directory for the build and native run}
mkdir -p "$work"
work=$(cd "$work" && pwd)
export PATH="/opt/homebrew/bin:$PATH" PYTHONPATH="$repo/src" OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$py" "$tools/test_rebind.py"
"$py" "$tools/test_loaded_sections.py"
"$py" "$tools/prepare_sources.py" "$p" "$e" "$c" "$app" "$ev" "$work"
cd "$work"
"$py" build_native.py "$p" "$e" "$c" "$app"
clang++ -O2 -std=c++23 -fobjc-arc -Ibuild/include -I"$app/matter/include" -I"$app/include" audit_gpu.mm -o audit_gpu -framework Foundation -framework Metal
"$py" test_source_rows.py
"$py" test_straight_rows.py
"$py" compose_current.py "$c/emitted-native/launch.json" "$p/floor-native-live-parent-026/launch.json" current.json
"$py" rebind_current.py current.json "$p/floor-native-live-parent-026/launch.json" "$work/current-package.json"
"$py" run_scene.py "$work/current-package.json" "$work/native"
