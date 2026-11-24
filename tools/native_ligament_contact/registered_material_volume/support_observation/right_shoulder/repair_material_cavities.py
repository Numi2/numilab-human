"""Retriangulate registered material cavities with a common positive Jacobian in every pose."""
from itertools import combinations
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import linprog
from prepare_infraspinatus import dets,exterior
from repair_infraspinatus import boundary,orient,assess
from compile_connected_joint import digest
from prepare_shoulder import MOVING

SIDES=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
def exposed(t):
    faces=t[:,SIDES].reshape(-1,3)
    _,ii,n=np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True)
    if n.max()>2:raise ValueError('Nonmanifold cavity')
    return faces[ii[n==1]]


def fan(X,old):
    b=exposed(old);ix=np.unique(old);K=len(ix);ref=X[-1]
    if set(ix)!=set(b.ravel()):return None
    # The same material interpolation defines a Steiner point in all poses.
    # Maximize the minimum oriented cone determinant divided by its local edge scale.
    planes=[]
    for x in X:
        v=x[b];n=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0])
        h=np.maximum.reduce([np.linalg.norm(v[:,(i+1)%3]-v[:,i],axis=1) for i in range(3)])
        q=np.einsum('fij,fj->fi',x[ix][None,:,:]-v[:,:1],n)/h[:,None]**3
        planes.extend(q)
    pr=linprog(np.r_[np.zeros(K),-1],A_ub=np.column_stack([planes,np.ones(len(planes))]),
               b_ub=np.zeros(len(planes)),A_eq=[np.r_[np.ones(K),0]],b_eq=[1],bounds=[(0,1)]*K+[(0,None)],method='highs')
    if not pr.success or pr.x[-1]<1e-9:return None
    weights=pr.x[:-1];coords=[weights@x[ix] for x in X];Xnew=[np.vstack([x,v])for x,v in zip(X,coords)]
    new=np.column_stack((np.full(len(b),len(ref)),b))
    q=assess(Xnew,old,new)
    if q is None:return None
    return q,new,Xnew,dict(type='common_material_steiner',nodes=ix.tolist(),weights=weights.tolist())


def choices(X,t,id):
    V=t[id];ref=X[-1];ans=[]
    near=set([id]);edge_stars=[]
    for m in (3,2):
        for key in combinations(V,m):
            ids=np.flatnonzero(np.isin(t,key).sum(1)==m)
            if m==3:near.update(ids.tolist())
            else:edge_stars.append(ids)
            if len(ids)!=5-m:continue
            old=t[ids];nodes=np.unique(old)
            if len(nodes)!=5:continue
            axis=np.setdiff1d(nodes,key);new=orient(ref,np.array([list(axis)+list(f) for f in combinations(key,len(key)-1)]))
            q=assess(X,old,new)
            if q is not None:ans.append((q,ids,new,X,dict(type='bistellar')))
    if ans:return ans
    groups=[np.array(sorted(near))]+edge_stars
    # A fan around a convex local cavity can repair a boundary-only diagonal
    # which cannot be removed by a single bistellar operation.
    for size in (1,2,3):
        groups.extend(np.array(sorted([id]+list(select))) for select in combinations(near-{id},size))
    for ids in groups:
        if len(ids)<2:continue
        z=fan(X,t[ids])
        if z is not None:
            q,n,Y,record=z;ans.append((q,ids,n,Y,record))
    return ans


def repair(geom,cages,out):
    out.mkdir(exist_ok=True);result={}
    for sid in MOVING:
        src=geom/f'bank-{sid}.npz';data=np.load(src)
        keys=('reference','rig_reference','opening','accepted');X=[data[s] for s in keys];t=data['t'];initial=t.copy()
        f=np.load(cages/f'bank-{sid}.npz')['f'];target={tuple(r) for r in np.sort(f,axis=1)}
        hist=[]
        for _ in range(100):
            bad=np.where(np.any([dets(x,t)<=0 for x in X],axis=0))[0]
            if not len(bad):break
            edit=None
            for it in bad:
                cand=choices(X,t,it)
                for q,ids,new,xx,r in sorted(cand,key=lambda a:-a[0]):
                    fresh=np.vstack([np.delete(t,ids,axis=0),new])
                    if len(np.unique(np.sort(fresh,axis=1),axis=0))!=len(fresh):continue
                    try:outer=boundary(fresh)
                    except ValueError:continue
                    if outer!=target:continue
                    edit=(fresh,xx,r|dict(replaced=t[ids].tolist(),inserted=new.tolist(),minimum_shape=q));break
                if edit:break
            if not edit:break
            t,X,rec=edit;hist.append(rec)
        if boundary(t)!=target:raise ValueError('Anatomical boundary changed')
        # Account for signed volume and integrated P1 support on the shared material coordinates.
        summary={}
        for k,x in zip(keys,X):
            D=dets(x,t);base=dets(x,initial);delta=(D.sum()-base.sum())/6
            if abs(delta)>2e-15:raise ValueError('Nonconservative cavity')
            summary[k]=dict(nonpositive=int((D<=0).sum()),min_signed_mm3=float(D.min()*1e9/6),volume_change_mm3=float(delta*1e9))
        d=dets(X[0],t)
        samples=[0.,1/3,2/3,1.]
        controls=np.linalg.solve(np.array([[(1-s)**3,3*s*(1-s)**2,3*s*s*(1-s),s**3]for s in samples]),np.array([dets(X[2]*(1-s)+X[3]*s,t)for s in samples]))
        result[sid]=dict(cells=len(t),source=digest(src),faces=len(f),native_motion_min_J=float((controls/d).min())if np.all(d>0)else None,stages=summary,edits=hist)
        np.savez_compressed(out/f'bank-{sid}.npz',t=t,**dict(zip(keys,X)))
        print(sid,len(hist),len(t),[v['nonpositive']for v in summary.values()],result[sid]['native_motion_min_J'],flush=True)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':repair(*(Path(a).resolve() for a in sys.argv[1:]))
