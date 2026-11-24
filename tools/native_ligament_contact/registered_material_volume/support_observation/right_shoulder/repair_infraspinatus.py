"""Retriangulate the inverted material star against both actual native positions."""
from itertools import combinations
import json
from pathlib import Path
import sys
import numpy as np
from prepare_infraspinatus import dets


def boundary(t):
    fs=np.concatenate([t[:,idx] for idx in combinations(range(4),3)])
    faces,count=np.unique(np.sort(fs,axis=1),axis=0,return_counts=True)
    if np.max(count)>2:raise ValueError('Nonmanifold material region')
    return {tuple(x)for x in faces[count==1]}


def orient(x,t):
    t=t.copy();idx=dets(x,t)<0;t[idx,:2]=t[idx,1::-1];return t


def assess(frames,old,new):
    if boundary(old)!=boundary(new):return None
    quality=[]
    for x in frames:
        det=dets(x,new)
        if np.min(det)<=0 or not np.isclose(det.sum(),dets(x,old).sum(),atol=1e-16,rtol=1e-10):return None
        edge=np.stack([np.linalg.norm(x[new[:,i]]-x[new[:,j]],axis=1) for i,j in combinations(range(4),2)])
        quality.append(float(np.min(det/edge.max(0)**3)))
    return min(quality)


def repair(src,out):
    z=dict(np.load(src));t=z['tets'];x=z['ref'];frames=[x,z['opening'],z['accepted']]
    bad=np.flatnonzero((dets(frames[1],t)<=0)|(dets(frames[2],t)<=0))
    if len(bad)!=1:raise ValueError('Unexpected source witness')
    start=t[bad[0]];candidates=[]
    for size in (2,3):
        for cells in combinations(start,5-size):
            ids=np.flatnonzero(np.isin(t,cells).sum(1)==len(cells))
            if len(ids)!=size:continue
            old=t[ids]; nodes=np.unique(old)
            if len(nodes)!=5:continue
            axis=np.setdiff1d(nodes,cells)
            new=orient(x,np.array([list(axis)+list(f) for f in combinations(cells,len(cells)-1)]))
            q=assess(frames,old,new)
            if q is None:continue
            kept=np.delete(t,ids,axis=0);updated=np.vstack([kept,new])
            if len(np.unique(np.sort(updated,axis=1),axis=0))!=len(updated):continue
            if boundary(updated)!=boundary(t):continue
            candidates.append((q,ids,new,updated))
    if not candidates:raise ValueError('No compatible bistellar cavity')
    score,ids,new,updated=max(candidates,key=lambda v:v[0]);out.mkdir(exist_ok=True)
    before={};after={}
    for n in ('ref','opening','accepted'):
        vals=dets(z[n],updated)/dets(x,updated);prev=dets(z[n],t)/dets(x,t)
        before[n]=dict(min_J=float(prev.min()),inverted=int((prev<=0).sum()))
        after[n]=dict(min_J=float(vals.min()),max_J=float(vals.max()),inverted=int((vals<=0).sum()),
                      volume_m3=float(dets(z[n],updated).sum()/6))
        if after[n]['inverted']:raise ValueError('Other inverted cells')
    # The entire chord between the two native states is a cubic determinant.
    # Bound every value by the Bernstein control hull over that actual motion.
    a=z['opening'][updated[:,1:]]-z['opening'][updated[:,:1]]
    b=z['accepted'][updated[:,1:]]-z['accepted'][updated[:,:1]]
    v=[]
    for s in (0.,1/3,2/3,1.):v.append(np.linalg.det(a*(1-s)+b*s)/dets(x,updated))
    values=np.array(v)
    M=np.array([[(1-s)**3,3*s*(1-s)**2,3*s*s*(1-s),s**3]for s in (0.,1/3,2/3,1.)])
    controls=np.linalg.solve(M,values)
    if controls.min()<=0:raise ValueError('Inverted material between native states')
    report=dict(original=int(bad[0]),original_cells=ids.tolist(),original_cell_nodes=t[ids].tolist(),replacement_nodes=new.tolist(),
                alternatives=len(candidates),cavity_min_shape=score,before=before,after=after,faces_preserved=len(boundary(t)),
                motion_min_J_bound=float(controls.min()),motion_max_J_bound=float(controls.max()),vertices=len(x),cells=len(updated))
    z['tets']=updated;np.savez_compressed(out/'registration.npz',**z)
    (out/'quality.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)

if __name__=='__main__':repair(*(Path(s).resolve()for s in sys.argv[1:]))
