"""Identify when the same source-material point has incompatible child positions."""
from pathlib import Path
import json,sys,itertools
import numpy as np
from audit_current_tissue_parents import records
from compile_connected_joint import digest
def run(scene,out):
    load=lambda p:json.loads(p.read_bytes());args=load(scene/'launch.json')['argv']
    receipt=Path(args[args.index('--resting-anatomy-receipt')+1]);binding=load(receipt)['passive_tissue_parent_bindings'];surfaces=binding['surfaces'];ids={s['stable_id'] for s in surfaces};child=Path(args[args.index('--soft-tissue-payload')+1])
    pose=load(scene/'registered-initial-bodies.json')['all_bodies'];world=records(child,[p['rest'] for p in pose],ids)
    groups={}
    for s in surfaces:
        sid=s['stable_id']
        for i,z in enumerate(s['vertices']):
            groups.setdefault(tuple(sorted(tuple(t) for t in z)),[]).append((sid,i))
    repeated=sum(len(v)>1 for v in groups.values());bad=[]
    for stencil,ids in groups.items():
        if len(ids)<2:continue
        pairs=itertools.combinations(ids,2)
        def distance(pair):
            (a,i),(b,j)=pair
            return float(np.linalg.norm(world[a][0][i]-world[b][0][j]))
        p=max(pairs,key=distance);gap=distance(p)
        if gap>4e-6:bad.append(dict(children=p,separation_m=gap,source=stencil))
    bad.sort(key=lambda v:-v['separation_m']);result=dict(receipt=digest(receipt),child=digest(child),repeated_stencils=repeated,incompatible_over_4um=len(bad),maximum_gap_m=bad[0]['separation_m'] if bad else 0.,witnesses=bad[:16]);out.write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k not in ('witnesses','receipt','child')})
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
