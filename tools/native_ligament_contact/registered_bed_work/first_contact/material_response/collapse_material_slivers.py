"""Collapse small edges only if the joined material and all interface links are preserved."""
from itertools import combinations
from pathlib import Path
import json,sys
import numpy as np
from fit_carpal_clearance import geom
from audit_registered_skeleton import digest

EDGES=np.array(list(combinations(range(4),2)))
def link(rows,vertices):
    linkset=set()
    for row in rows:
        simplex=set(row)
        if vertices<=simplex:
            opposite=sorted(simplex-vertices)
            for n in range(1,len(opposite)+1):
                linkset.update(combinations(opposite,n))
    return linkset
def manifold(rows,a,b):
    return (link(rows,{a}) & link(rows,{b}))==link(rows,{a,b})
def collapse(rows,a,b):
    updated=rows.copy();updated[updated==a]=b
    keep=np.all(np.diff(np.sort(updated,axis=1),axis=1)>0,axis=1)
    return updated,keep
def quality(points,cells):
    v,q=geom(points[cells])
    return np.minimum(v/1e-18,q/1e-5)
def score(quality):
    return (int(np.sum(quality<1)),float(np.maximum(0.,1.-quality).sum()))
def run(source,out,*,interior=False):
    data=np.load(source); x=data['x'].astype(float); t=data['t'].copy();roles=data['role'].copy()
    f=data['facets'].copy();mk=data['markers'].copy()
    frozen=data['locked'].copy();frozen[np.unique(f[mk==15])]=True
    if interior:frozen[np.unique(f[mk!=1000])]=True
    results=[];scale=1.e6
    for it in range(100):
        q=quality(x,t);bad=t[q<1]
        if not len(bad):break
        pairs=np.unique(np.sort(bad[:,EDGES].reshape(-1,2),axis=1),axis=0)
        lengths=np.linalg.norm(x[pairs[:,0]]-x[pairs[:,1]],axis=1)*scale
        best=None
        for i in np.argsort(lengths):
            if lengths[i]>(4500. if interior else 95.):break
            endpoints=tuple(map(int,pairs[i]))
            for a,b in (endpoints,endpoints[::-1]):
                if frozen[a]:continue
                incident=np.any(t==a,axis=1)|np.any(t==b,axis=1);star=t[incident];mats=roles[incident]
                if not manifold(star,a,b) or any(not manifold(star[mats==m],a,b) for m in np.unique(mats)):continue
                boundary=(mk!=1000)&(np.any(f==a,axis=1)|np.any(f==b,axis=1))
                faces=f[boundary];markers=mk[boundary]
                if any(not manifold(faces[markers==m],a,b) for m in np.unique(markers)):continue
                if any(not np.any(faces[markers==m]==b) for m in np.unique(markers[np.any(faces==a,axis=1)])):continue
                changed,keep=collapse(star,a,b);newq=quality(x,changed[keep])
                if not len(newq) or np.min(newq)<=0:continue
                old=score(q[incident]);new=score(newq)
                if new[0]>old[0] or new[1]>=old[1]:continue
                if new[0]==old[0] and (not interior or old[1]-new[1]<.05):continue
                cf,valid=collapse(faces,a,b);keepface=valid&np.any(faces==a,axis=1)
                orig=x[faces[keepface]];trial=x[cf[keepface]]
                mis=0.
                if len(orig):
                    normal=np.cross(orig[:,1]-orig[:,0],orig[:,2]-orig[:,0])
                    newnormal=np.cross(trial[:,1]-trial[:,0],trial[:,2]-trial[:,0])
                    if np.any(np.einsum('ij,ij->i',normal,newnormal)<=0):continue
                    normal/=np.linalg.norm(normal,axis=1)[:,None]
                    mis=float(np.max(np.abs(normal@(x[b]-x[a])))*scale)
                    if mis>0.5:continue
                rank=(new[0]-old[0],new[1]-old[1],mis,lengths[i],a,b)
                if best is None or rank<best[0]:best=(rank,a,b)
        if best is None:break
        (gain,_,miss,l,_,_),a,b=best
        t,keep=collapse(t,a,b);t=t[keep];roles=roles[keep]
        f,keep=collapse(f,a,b);f=f[keep];mk=mk[keep]
        # Interior retained source faces can coincide after a collapse.
        _,keep=np.unique(np.column_stack((np.sort(f,axis=1),mk)),axis=0,return_index=True);f=f[keep];mk=mk[keep]
        results.append(dict(drop=a,keep=b,edge_um=l,normal_change_um=miss,cells_cleared=-gain));print(it,len(bad)+gain,flush=True)
    if len(np.unique(np.sort(t,axis=1),axis=0))!=len(t):raise ValueError('duplicate physical tetrahedron')
    from verify_carpal_mesh import classify
    classified,_=classify(x,t,f[mk!=1000],mk[mk!=1000])
    if not np.array_equal(classified,roles):raise ValueError('material interface labeling changed')
    q=quality(x,t);out.mkdir();np.savez_compressed(out/'domain.npz',**{k:data[k] for k in data.files if k not in ('t','role','facets','markers')},t=t,role=roles,facets=f,markers=mk)
    (out/'result.json').write_text(json.dumps(dict(source=digest(source),operations=results,remaining=int((q<1).sum()),nonpositive=int((q<=0).sum())),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
