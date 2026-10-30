#!/usr/bin/env bash
set -euo pipefail
OUT="$(cd -- "$(dirname -- "$0")" && pwd)"
ROOT="$(cd -- "$OUT/../../../.." && pwd)"
CXX="${CXX:-clang++}"
TETGEN_VENDOR_DIR="${TETGEN_VENDOR_DIR:?set TETGEN_VENDOR_DIR to the pinned TetGen v1.6.0 header/dependency directory}"
PREDICATES_OBJ="${PREDICATES_OBJ:-$TETGEN_VENDOR_DIR/predicates.o}"
sha_check() {
  local expected="$1" path="$2"
  printf '%s  %s\n' "$expected" "$path" | shasum -a 256 -c - >/dev/null
}
sha_check 255bd025b7b7730ce81348c30df45fe5ad98b7160d5cea29c3bb03c91dae8b1a \
  "$ROOT/tools/evidence/human-physical-mesher-controls-20261030/mesher-054/source/tetgen-explicit-no-improvement-validation.cxx"
sha_check 69b012da7c0b01e6327f5f281ac2d10fc3fb42503d22366c7bd0bc9181bd833b \
  "$TETGEN_VENDOR_DIR/tetgen.h"
sha_check d15c7e1a4dc798fa0b48204e346a9fe031070ccb7789d8b74b910dfafba26f4c \
  "$PREDICATES_OBJ"
TEST_TMP="$OUT/.control-tests.stdout.tmp"
trap 'rm -f "$TEST_TMP"' EXIT
"$CXX" -std=c++11 -O2 "$OUT/test_hull_region_attribute_controls.cpp" \
  -o "$OUT/test_hull_region_attribute_controls" 2> "$OUT/control-test-compile.stderr.txt"
"$OUT/test_hull_region_attribute_controls" > "$TEST_TMP"
cmp "$TEST_TMP" "$OUT/control-tests.stdout.txt"
"$CXX" -std=c++11 -O3 -I "$OUT" -I "$TETGEN_VENDOR_DIR" \
  "$OUT/tetgen-hull-attribute-fix-minimal.cxx" "$PREDICATES_OBJ" -lm \
  -o "$OUT/tetgen-hull-attribute-fix-minimal" 2> "$OUT/tetgen-compile.stderr.txt"
printf 'PASS dependency pins, six focused controls, and TetGen build\n'
