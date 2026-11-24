"""Measure the closest source-supported reconstruction available on the retained tissue stencils."""
from pathlib import Path
import json,sys,itertools,struct
import numpy as np
from scipy.spatial import cKDTree
from audit_current_tissue_parents import records
from compile_connected_joint import digest
def closest(points,target):
    # A closest point in a convex subset of R3 needs at most four vertices.
    best=(float('inf'),None)
    for n in range(1,min(4,len(points))+1):
        for idx in itertools.combinations(range(len(points)),n):
            q=points[list(idx)];w0=np.linalg.lstsq((q[1:]-q[0]).T,target-q[0],rcond=1e-12)[0] if n>1 else []
            w=np.r_[1-np.sum(w0),w0]
            if np.min(w)<-1e-10:continue
            w=np.maximum(w,0);w/=sum(w);err=float(np.linalg.norm(w@q-target))
            if err<best[0]:
                full=np.zeros(len(points));full[list(idx)]=w;best=err,full
    return best
def run(scene,audit,out):
    rd=lambda p:json.loads(p.read_bytes());a=rd(scene/'launch.json')['argv'];receipt=Path(a[a.index('--resting-anatomy-receipt')+1]);binding=rd(receipt)['passive_tissue_parent_bindings']
    poses=[v['rest'] for v in rd(scene/'registered-initial-bodies.json')['all_bodies']]
    failures=rd(audit/'result.json')['surfaces'];bad=[s for s in binding['surfaces'] if failures[str(s['stable_id'])]['failing']];ids={t[1] for s in bad for row in s['vertices'] for t in row}
    pool=[records(Path(p['path']),poses,ids) for p in binding['source_payloads']]
    items=[]
    for surface in bad:
        sid=surface['stable_id'];z=np.load(audit/f'{sid}.npz')
        for node,target in zip(z['nodes'],z['current']):
            terms=surface['vertices'][node];vecs=np.array([pool[s][b][0][v] for s,b,v,_ in terms]);err,w=closest(vecs,target)
            items.append(dict(id=sid,node=int(node),terms=len(terms),groups=len({(p,b) for p,b,v,c in terms}),best_error_um=err*1e6))
    summary={}
    for sid in sorted({v['id'] for v in items}):
        subset=[v for v in items if v['id']==sid];summary[sid]=dict(failing=len(subset),in_span_at_2um=sum(v['best_error_um']<=2 for v in subset),max_best_error_um=max(v['best_error_um'] for v in subset))
    out.write_text(json.dumps(dict(method='Optimistic convex envelope of exactly the authored parents. Independent nonnegative weights sum to one. Retaining original material proportions can only restrict the result.',receipt=digest(receipt),summary=summary,worst=sorted(items,key=lambda v:-v['best_error_um'])[:12]),indent=2)+'\n');print(summary)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:]))
