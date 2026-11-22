#!/bin/bash
# Replay an installed registered volume in the coupled native owner and check its sampled endpoint.
set -euo pipefail
if [[ $# -lt 8 || $# -gt 9 ]]; then
    echo "usage: $0 INPUTS SELECTION BINARY OUTPUT STEPS PYTHON LAYER_PREDICATE ENVELOPE_AUDIT [CAPTURE_DIRECTORY]" >&2
    exit 2
fi
inputs=$1
selection=$2
binary=$3
out=$4
steps=$5
py=$6
predicate=$7
envelope=$8
capture=${9:-$inputs}
tools=$(cd "$(dirname "$0")" && pwd)
[[ ! -e "$out" ]]
mkdir -p "$capture"
[[ ! -e "$capture/endpoint-reference.node" && ! -e "$capture/endpoint-accepted.node" ]]
"$py" - "$inputs" "$steps" "$capture" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve()
steps=int(sys.argv[2])
if steps<1:raise ValueError('A physical prefix must contain a positive number of steps')
read=lambda name:json.loads((p/name).read_bytes())
if not read('material/native-role-verification.json')['passed']:raise ValueError('Mass writer or reader failed')
if any(sum(row) for row in read('reference-layers.json')['pair_counts']):raise ValueError('Physical reference has intersecting layer boundaries')
launch=read('launch.json')
if launch['environment'].get('NM_VASCULAR_STEP_EXTERNAL_PRESSURE')!='1':raise ValueError('Variable pleural pressure required')
launch['environment'].update(NUMI_HUMAN_REGISTERED_CELL_QUALITY='1',NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS=str(steps),NUMI_HUMAN_REGISTERED_CELL_NODE_PREFIX=str(Path(sys.argv[3]).resolve()/'endpoint'))
(p/('verified-motion-'+str(steps)+'.json')).write_text(json.dumps(launch,indent=2)+'\n')
PY
NUMI_SELECTED_TRIAL_MODE=motion NUMI_SELECTED_TRIAL_ROOTS="$steps" NUMI_SELECTED_RETAIN_TERMINAL_PACK=1 bash "$tools/run_volume_admission.sh" "$inputs/verified-motion-$steps.json" "$selection" "$binary" "$out" "$py"
"$py" - "$out/admission.json" "$steps" <<'PY'
import sys,json
r=json.load(open(sys.argv[1]))
if r['exit_code']!=0 or r['watchdog'] or r['accepted_steps']!=int(sys.argv[2]):raise ValueError('Physical prefix did not finish')
PY
PYTHONPATH="$(dirname "$predicate")${PYTHONPATH:+:$PYTHONPATH}" "$py" "$tools/audit_installed_layer_boundaries.py" "$out/launch.json" "$predicate" "$capture/accepted-layers.json" --accepted-nodes "$capture/endpoint-reference.node" "$capture/endpoint-accepted.node" >"$capture/accepted-layers.log" 2>&1
"$py" "$tools/audit_registered_initial_envelope.py" --terminal "$envelope" "$out/terminal.mrvpack.zst" "$out/native/accepted-geometry/step-$steps.receipt.json" "$out/admission.json" "$out/launch.json" "$out/envelope" >"$out/envelope.log" 2>&1
"$py" "$tools/classify_registered_envelope.py" "$out/envelope/accepted-envelope.json" "$out/stdout.log" "$out/envelope/physical-banks.json" --window-receipt "$out/native/resting-ocular-sclera-window.json"
"$py" - "$capture/accepted-layers.json" "$out/envelope/physical-banks.json" "$out/stdout.log" <<'PY'
from pathlib import Path
import sys,json
layer,body,log=map(Path,sys.argv[1:])
rows=[dict(p.split('=',1) for p in l.split()[1:]) for l in log.read_text().splitlines() if l.startswith('registered_cell_shape ')]
if len(rows)!=4:raise ValueError('Incomplete native material-cell audit')
counts=sum(int(r[k]) for r in rows for k in ('nonfinite','inverted','volume_le_1e_18','shape_lt_1e_5'))
pairs=sum(map(sum,json.loads(layer.read_bytes())['pair_counts']))
b=json.loads(body.read_bytes())
print('material cell failure counts:',counts,'material layer pairs:',pairs,'nonocular skin-to-structure pairs:',b['physical_nonocular_pairs'],'ocular source pairs:',b['ocular_source_pairs'])
if counts or pairs or b['physical_nonocular_pairs'] or b['ocular_source_pairs']:sys.exit(1)
PY
