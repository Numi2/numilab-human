"""Verify carrier eligibility on every expanded receiver and retain source-bound role proofs."""
import sys,json
from pathlib import Path
from emit_skin_source_registration import sha
import numpy as np
baseline,reg,out=map(Path,sys.argv[1:]);fine=baseline/'refit-motion-qualification';m=reg/'fine-mass-routed';source=baseline/'certify_live_parent_flow_witnesses.py'
flow=m/'current-density-closed-flow.npz';r=json.loads((m/'mass-transfer.json').read_text());assert sha(flow)==r['new_flow_sha256'] and r['density_outliers']==0
code=source.read_text();a=code.index('root = Path(');b=code.index('labels_path =',a)
code=code[:a]+'''root=baseline
args=type('Options',(),dict(emit_witnesses=True))()
closure=dict(transport_distance_limit_m=mass['distance_bound_m'])
closed=dict(candidate_sha256=mass['new_flow_sha256'])
candidate=flow
p1=destination/'p1'
state=json.loads((p1/'current-paired-volume-state.json').read_text())
carrier_receipt=json.loads((fine/'witnesses/receipt.json').read_text())
'''+code[b:]
a=code.index('with np.load(root/"live-parent-route-geometry.npz")');b=code.index('if args.emit_witnesses:',a)
code=code[:a]+'''centers=np.empty((len(cell_groups),3))
for data in carriers_by_role:
    centers[data['children']]=data['centroids']
sources=np.concatenate([data['source_position'] for data in carriers_by_role])
distance=np.linalg.norm(sources[edge_carrier[edge_repeat]]-centers[child],axis=1)
if not np.all(np.isfinite(distance)&(distance<=closure['transport_distance_limit_m'])):raise ValueError('Expanded native role distance failed')
report['maximum_carrier_distance_m']=float(distance.max())
'''+code[b:]
code=code.replace('out = root/"paired-live-parent-flow-witness"',"out=destination/'witnesses'")
exec(compile(code,str(source),'exec'),dict(baseline=baseline,mass=r,flow=flow,destination=out,fine=fine,sha=lambda p:sha(Path(p))))
