#!/bin/bash
set -euo pipefail
variant=$1
steps=$2
assets=$3
build=$4
out=$5
python_exec=$6
predicate=$7
part=${8:-repartition}
src="$PWD/repartition/surface-trial-config"
lib="$PWD/repartition/endpoint-owner-build/$variant/libmetalrobo.dylib"
base="$PWD/repartition/pose-config-$variant"
"$python_exec" - "$src" "$base" "$lib" <<'PY'
import json,sys
from pathlib import Path
from material_context import sha
src,base,lib=map(Path,sys.argv[1:]);base.mkdir(exist_ok=True)
d=json.loads((src/'config.json').read_text());e=d['environment']
prior=Path(e['DYLD_LIBRARY_PATH'])/'libmetalrobo.dylib'
assert d['pins'][str(prior)]==sha(prior)
d['pins'][str(lib)]=sha(lib);e['DYLD_LIBRARY_PATH']=str(lib.parent);e['DYLD_PRINT_LIBRARIES']='1'
(base/'config.json').write_text(json.dumps(d,indent=2)+'\n')
if not (base/'run_supported.py').exists():(base/'run_supported.py').hardlink_to(src/'run_supported.py')
PY
test ! -e "$out"
bash run_cortical_trial.sh "$base/config.json" "$assets" "$build/numi-human-native" "$out" "$python_exec" "$steps" 1 || test -f "$out/failure.json"
rg -F "$lib" "$out/stderr.log" > "$out/loaded-dylib.log"
if test -e "$out/terminal.receipt.json";then
    "$python_exec" audit_endpoint_pose.py "$out/accepted-pelvic-state.csv" "$out/terminal.receipt.json" "$out/endpoint-pose.json"
fi
if test -s "$out/accepted-pelvic-state.csv";then
    "$python_exec" audit_complete_pelvic_forces.py "$part" "$part/response" "$assets" "$out/accepted-pelvic-state.csv" "$out/bulk-force.json"
    "$python_exec" audit_finite_cortical_crossings.py "$assets" "$out/accepted-pelvic-surfaces.csv" "$predicate" "$out/native-crossings.json"
    "$python_exec" audit_accepted_cortical_volume.py "$part" "$assets" "$out/accepted-pelvic-state.csv" "$out/solid-overlap.json"
    "$python_exec" evaluate_pelvic_trial.py "$out"
fi
