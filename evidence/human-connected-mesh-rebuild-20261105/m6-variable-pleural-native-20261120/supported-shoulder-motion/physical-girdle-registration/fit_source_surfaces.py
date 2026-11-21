"""Register intact atlas bones to their corresponding native MyoSim meshes in body-local coordinates."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from source_pair_audit import inputs, world


def sample(v,f,n,seed):
    t=v[f]; ab=t[:,1]-t[:,0]; ac=t[:,2]-t[:,0]
    area=np.linalg.norm(np.cross(ab,ac),axis=1)
    rng=np.random.default_rng(seed)
    # Stratify by cumulative area. Duplicate authored seam vertices add no area.
    face=np.searchsorted(np.cumsum(area)/area.sum(),(np.arange(n)+rng.random(n))/n)
    uv=rng.random((n,2)); uv[uv.sum(axis=1)>1]=1-uv[uv.sum(axis=1)>1]
    return t[face,0]+uv[:,:1]*ab[face]+uv[:,1:]*ac[face]


def align(x,y,scalable):
    a=x.mean(axis=0);b=y.mean(axis=0)
    x=x-a;y=y-b
    U,S,V=np.linalg.svd(x.T@y)
    signs=np.ones(3);signs[-1]=np.sign(np.linalg.det(U@V))
    matrix=(U*signs)@V
    scale=float((S@signs)/np.sum(x*x)) if scalable else 1.
    return matrix, scale, b-scale*a@matrix


def principal(x):
    v=np.linalg.eigh(np.cov(x.T))[1][:,::-1]
    v[:,2]=np.cross(v[:,0],v[:,1]);return v


def objective(x,y,tree):
    first=tree.query(x)[0];second=cKDTree(x).query(y)[0]
    return float((np.mean(first**2)+np.mean(second**2))/2)


def iteration(x,y,rt,scale,tr,tree_x,tree_y,scalable):
    current=(x@rt)*scale+tr
    ix=tree_y.query(current)[1]
    back=(y-tr)@rt.T/scale
    iy=tree_x.query(back)[1]
    return align(np.concatenate([x,x[iy]]),np.concatenate([y[ix],y]),scalable)


def fit(x,y,scalable=False):
    tx,ty=cKDTree(x),cKDTree(y);xa=x.mean(axis=0);ya=y.mean(axis=0)
    px,py=principal(x),principal(y)
    starts=[np.eye(3)]
    for sx,sy in ((1,1),(-1,1),(1,-1),(-1,-1)):
        starts.append(px@np.diag([sx,sy,sx*sy])@py.T)
    candidates=[]
    for m in starts:
        r,s,t=m,1.,ya-xa@m
        for _ in range(30):
            prev=r,s,t
            r,s,t=iteration(x,y,r,s,t,tx,ty,scalable)
            if max(np.max(np.abs(r-prev[0])),abs(s-prev[1]),np.max(np.abs(t-prev[2])))<1e-9:break
        candidates.append((objective(s*x@r+t,y,ty),r,s,t))
    _,r,s,t=min(candidates,key=lambda k:k[0])
    return r,s,t


def run(launch,anchors,motion,source,out,scalable=False):
    ms,poses,default,owners=inputs(launch,anchors,motion)
    bones=np.load(source);rows=[]
    for k,name in enumerate(sorted(owners)):
        sid,b=owners[name];v,f,*_=ms[sid]
        target=bones[name+'_local'];g=bones[name+'_faces']
        x=sample(v,f,1200,911);y=sample(target,g,1200,710)
        rot,s,t=fit(x,y,scalable)
        # Refine on denser independent samples, then use held-out locations
        # for the separately reported bidirectional distances.
        x=sample(v,f,4200,1991);y=sample(target,g,4200,722);tx,ty=cKDTree(x),cKDTree(y)
        for _ in range(28):
            old=rot,s,t
            rot,s,t=iteration(x,y,rot,s,t,tx,ty,scalable)
            if np.max(np.abs(rot-old[0]))<1e-8 and abs(s-old[1])<1e-8 and np.max(np.abs(t-old[2]))<1e-8:break
        x=sample(v,f,12000,37);y=sample(target,g,12000,45);ty=cKDTree(y)
        pre=objective(x,y,ty)**.5;post=objective(s*x@rot+t,y,ty)**.5
        a=Rotation.from_matrix(rot.T)
        rows.append(dict(bone=str(name),sid=sid,body=b,scale=s,rotation_xyzw=a.as_quat().tolist(),translation_m=t.tolist(),
                         heldout_before_rms_m=pre,heldout_after_rms_m=post,rotation_deg=float(a.magnitude()*180/np.pi)))
        print(name,round(pre*1000,2),round(post*1000,2),round(rows[-1]['rotation_deg'],2),round(s,4),flush=True)
    out.write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__':
    run(*map(Path,sys.argv[2:]),scalable={'rigid':False,'similarity':True}[sys.argv[1]])
