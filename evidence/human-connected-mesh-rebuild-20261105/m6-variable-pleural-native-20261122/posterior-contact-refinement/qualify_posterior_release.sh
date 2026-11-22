#!/bin/bash
set -euo pipefail
case_dir=$1; parent=$2; runtime=$3; proposal=$4; recruitment=$5
audit_dir=$6; implementation=$7; predicate=$8; py=$9
cd "$case_dir"
run_native() {
  "$py" -c 'import shutil; assert shutil.disk_usage(".").free>=12*2**30, "prelaunch disk reserve"'
  "$py" "$runtime/run_cavity_motion.py" "$parent" "$case_dir/owner/native-owner" "$proposal" "$case_dir/$1" --steps "$2" --capture-fem-body-load --recruitment-reference "$recruitment"
}
run_native native-three 3 || test -s native-three/failure.json
case_name=native-three
if test -e native-three/failure.json; then
  steps=$("$py" -c 'import json; s=json.load(open("native-three/failure.json"))["segment_start"]; assert s in [1,2]; print(s)')
  case_name=native-prefix
  run_native "$case_name" "$steps"
fi
"$py" "$runtime/check_cavity_endpoint.py" "$case_dir/$case_name" "$audit_dir" "$implementation" "$predicate" --require-clear
