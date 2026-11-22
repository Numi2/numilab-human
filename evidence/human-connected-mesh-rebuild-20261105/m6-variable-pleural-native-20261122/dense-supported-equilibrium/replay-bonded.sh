#!/bin/bash
# Use the recovered owner, existing native runner and source hashes in launch.json.
set -euo pipefail
trial=$1
out=$2
: "${PY:?}" "${u:?}" "${z:?}" "${h:?}" "${work:?}" "${W:?}" "${E:?}"
case "$trial" in
  bonded-prefix) owner=bonded-owner; steps=2 ;;
  bonded-rejection) owner=bonded-owner; steps=4 ;;
  conserved-prefix) owner=conserved-owner; steps=2 ;;
  conserved-third) owner=conserved-owner; steps=3 ;;
  *) exit 2 ;;
esac
"$PY" -c 'import shutil,sys;sys.exit(shutil.disk_usage(sys.argv[1]).free<12*2**30)' "$u"
"$PY" "$z/run_cavity_motion.py" "$u/cell-trace-parent" "$u/$owner/native-owner" "$z/typed-fans/six-margin.npz" "$out" --steps "$steps" --capture-fem-body-load --recruitment-reference "$h/drive-control-60"
"$PY" "$z/check_cavity_endpoint.py" "$out" "$work" "$W/resting-surface-clearance-001/audit.py" "$E/connected-arm-volume-043/boundary_pairs.dylib" --require-clear
