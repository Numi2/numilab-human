"""Enforce the original coordinates in marginal cells at the remote edge of a repair."""
import json,sys
from pathlib import Path
import numpy as np
from transfer_connected_volume import geometry
def run(material,candidate,out):
    z=np.load(material);x=z['x'];t=z['tets'];y=np.load(candidate);history=[]
    v,q=geometry(y,t);invalid=(v<=1e-18)|(q<=1e-5)
    restored=set()
    for i in range(40):
        bad=np.flatnonzero(invalid)
        if not len(bad):break
        ids=np.unique(t[bad]);new=set(ids)-restored
        if not new:raise ValueError('Unchanged witness')
        displacement=np.max(np.linalg.norm(y[ids].astype(float)-x[ids],axis=1))
        if displacement>1e-5:raise ValueError('Failed cells extend into the intended anatomical displacement')
        restored.update(ids);y[ids]=x[ids]
        near=np.flatnonzero(np.any(np.isin(t,ids),axis=1))
        a,b=geometry(y,t[near]);invalid[near]=(a<=1e-18)|(b<=1e-5)
        history.append([len(bad),len(new),displacement,int(invalid.sum())])
    a,b=geometry(y,t);assert a.min()>1e-18 and b.min()>1e-5
    out.with_suffix('.json').write_text(json.dumps(dict(restored_nodes=len(restored),history=history,cells=len(t),inversions=0,subthreshold=0,minimum_volume_m3=float(a.min()),minimum_quality=float(b.min()),moved_nodes=int(np.any(x!=y,axis=1).sum())),indent=2)+'\n')
    np.save(out,y)
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
