#!/bin/bash
set -euo pipefail
py=$1;root=$2
cd "$root"
while ! test -s incremental-three/execution.json; do
 if ! pgrep -f '[P]ython .*/run_cavity_motion.py.*incremental-three' >/dev/null; then test -s incremental-three/execution.json || exit 1; fi
 sleep 4
done
direct-tests/check direct-tests/control.metallib control >direct-tests/control-roundtrip.log
direct-tests/check direct-tests/incremental.metallib incremental >direct-tests/incremental-roundtrip.log
while ! test -s incremental-three/terminal.receipt.json && ! test -s incremental-three/failure.json; do
 if ! pgrep -f '[P]ython .*/run_cavity_motion.py.*incremental-three' >/dev/null; then exit 1; fi
 sleep 2
done
"$py" compare_incremental_release.py "$root" incremental-comparison.json
