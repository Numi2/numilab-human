"""Limit vertical registration to the existing physical parent donor capture and residual budget."""
import csv,json,sys
from pathlib import Path
import numpy as np
from emit_skin_source_registration import sha
source,launch,out=map(Path,sys.argv[1:]);s=np.load(source);nodes=s['nodes'];rank={int(i):j for j,i in enumerate(nodes)}
env=json.loads(launch.read_text())['environment'];p='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_'
fp=Path(env[p+'COMBINED_NODE_MAP_CSV']);assert sha(fp)==env[p+'COMBINED_NODE_MAP_SHA256']
limit=np.full(len(nodes),np.inf);seen=np.zeros(len(nodes),bool)
for r in csv.DictReader(fp.open()):
    n=int(r['joint_node_id'])-1
    if n not in rank:continue
    j=rank[n];seen[j]=True
    for prefix in ['parent_current_residual_','operational_reference_residual_','source_residual_']:
        v=np.array([float(r[prefix+k+'_m']) for k in 'xyz'])
        rem=.02**2-np.dot(v[:2],v[:2])
        if rem<0:raise ValueError('Current mapping exceeds native residual bound')
        limit[j]=min(limit[j],max(0.,(np.sqrt(rem)-v[2])*1000-.002))
assert seen.all();np.savez_compressed(out,limit_mm=limit,nodes=nodes,input_sha256=sha(fp))
