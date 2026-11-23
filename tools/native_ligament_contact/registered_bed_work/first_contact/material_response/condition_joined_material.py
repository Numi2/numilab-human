"""Resolve element-quality failures without moving the registered material boundaries."""
from itertools import combinations
from pathlib import Path
import json,sys
import numpy as np
from material_cavity_flips import replace_pass,boundary
from material_star_cavity import grow_pass
from supported_material_admission import geom
from register_material_surfaces import sha

def run(path,out):
    pkg=np.load(path);p=pkg['x'].astype(float);t=pkg['t'];roles=pkg['role']
    locked=pkg['locked'];result=[];materials=[];log=[]
    for role in (15,16):
        c=t[roles==role];f0=boundary(c);neighbors=t[roles!=role]
        occupied={e for row in neighbors for e in combinations(sorted(map(int,row)),2)}
        class BoundaryEdges:
            def __contains__(self,edge):
                return bool(locked[edge[0]] and locked[edge[1]]) or edge in occupied
        forbidden=BoundaryEdges()
        def quality(cell):
            v,q=geom(p,cell);return np.minimum(v/1e-18,q/1e-5)
        for i in range(35):
            count=int(sum(quality(c)<1))
            if not count:break
            c,ops=replace_pass(p,c,forbidden,quality);method='edge'
            if not ops:c,ops=grow_pass(p,c,forbidden,quality);method='star'
            after=int(sum(quality(c)<1));log.append(dict(material=role,iteration=i,before=count,after=after,method=method,operations=len(ops)));print(log[-1],flush=True)
            if not ops:break
        if f0!=boundary(c):raise ValueError('Changed registered material boundary')
        result.append(c);materials.extend([role]*len(c))
    out.mkdir();c=np.concatenate(result);v,q=geom(p,c);record=dict(parent=sha(path),iterations=log,cells=len(c),invalid=int(((v<=1e-18)|(q<1e-5)).sum()),nonpositive=int(np.sum(v<=0)),min_volume_m3=float(v.min()),min_shape=float(q.min()))
    np.savez_compressed(out/'domain.npz',**{k:pkg[k] for k in pkg if k not in ('t','role')},t=c,role=materials)
    (out/'quality.json').write_text(json.dumps(record,indent=2)+'\n')
if __name__=='__main__':run(*(Path(k).resolve() for k in sys.argv[1:]))
