"""Retriangulate the material volume controlling the failed physical direction."""
from itertools import combinations
from pathlib import Path
import json
import numpy as np
from material_cavity_flips import replace_pass,boundary
from material_star_cavity import grow_pass
from supported_material_admission import geom


def fit(path,out):
    data=np.load(path); rows,labels,T=data['rows'],data['labels'],data['local']
    pos=data['xyz'];i=int(np.where(rows==data['selected'])[0][0]);source=T[i]
    walls={tuple(sorted(np.searchsorted(data['ids'],face))) for k in 'OIDC' for face in data[k] if np.isin(face,data['ids']).all()}
    same=(labels[:,1:]==labels[i,1:]).all(1)
    blocked={e for cell in T[~same] for e in combinations(sorted(cell),2)}
    def geometric_margin(c):
        volumes=[];margins=[]
        for frame in pos:
            v,q=geom(frame,c)
            volumes.append(v);margins.extend([v/1e-18,q/1e-5])
        for v,lo in zip(volumes[1:],(.4,.35)):
            j=v/np.maximum(volumes[0],1e-30);margins.extend([j/lo,2.5/np.maximum(j,1e-30)])
        return np.min(margins,axis=0)
    def margin(c):
        baseline=geometric_margin(c)
        hit=np.isin(c,source).all(1)
        return np.where(hit,0.,baseline)
    selected=T[same]
    # Earlier quality-only selection missed physically stiff cells above its shape threshold.
    results=[]
    for f in (replace_pass,grow_pass):
        _,new=f(pos[0],selected,blocked,margin)
        for cells,after in new:
            mask=np.isin(cells,source).all(1)
            if not mask.any():continue
            key={tuple(sorted(c)):r for c,r in zip(T,rows)}
            original=[int(key[tuple(sorted(c))]) for c in cells]
            internal={tuple(sorted(face))for c in cells for face in combinations(c,3)}-{tuple(sorted(q)) for q in boundary(cells)}
            if internal&walls or boundary(cells)!=boundary(after):continue
            if np.min(geometric_margin(after))<1:continue
            vol=[];shape=[]
            for x in pos:
                v,q=geom(x,after);v0,_=geom(x,cells)
                if abs(v.sum()-v0.sum())>1e-9*v0.sum():raise ValueError('Cavity volume changes')
                vol.append(v);shape.append(q.min())
            results.append(dict(alg=f.__name__,old=original,new=data['ids'][after].tolist(),label=labels[i,1:].tolist(),minimum_shape=shape,
                                endpoint_J=(vol[2]/vol[0]).tolist()))
    out.write_text(json.dumps(results,indent=2)+'\n');print(len(results),[(r['alg'],len(r['old']),len(r['new']),r['minimum_shape'])for r in results])

if __name__=='__main__':fit(Path('cavity.npz'),Path('proposals.json'))
