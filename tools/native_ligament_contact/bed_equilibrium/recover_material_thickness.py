"""Correct the signed thickness of deforming tetrahedra without rigidly tying their faces."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsmr
from screen_registered_dense_posture import load_geometry, measure
from construct_registered_donor_field import errors
from untangle_registered_donor_field import differentials


def audit(y,cells,ref):
    j=[];shape=[];det=[]
    for i in range(0,len(cells),32768):
        d,_,long3=differentials(y,cells[i:i+32768])
        det.append(d);j.append(d/ref[i:i+len(d)]);shape.append(d/long3)
    det=np.concatenate(det);j=np.concatenate(j);shape=np.concatenate(shape)
    bad=(j<.4)|(j>2.5)|(shape<1e-5)|(det<=6e-18)
    return j,shape,det,bad


def correct(x,y,t,ref,desired_vol):
    X=x[t];Y=y[t]
    # Choose the widest reference face, which has a well-conditioned in-plane basis.
    _,cof,_=differentials(x,t)
    apex=np.argmax(np.sum(cof*cof,axis=2),axis=1)
    face=(apex[:,None]+np.arange(1,4))%4
    r=np.arange(len(t));base=X[r[:,None],face]
    a,b=base[:,1]-base[:,0],base[:,2]-base[:,0]
    normal=np.cross(a,b);area=np.linalg.norm(normal,axis=1)
    normal/=area[:,None]
    off=X[r,apex]-base[:,0]
    h=np.einsum('ij,ij->i',off,normal)
    aa=np.einsum('ij,ij->i',a,a);bb=np.einsum('ij,ij->i',b,b);ab=np.einsum('ij,ij->i',a,b)
    oa=np.einsum('ij,ij->i',off,a);ob=np.einsum('ij,ij->i',off,b)
    # Area squared avoids subtraction of almost equal products.
    v=(bb*oa-ab*ob)/(area*area);w=(aa*ob-ab*oa)/(area*area)
    coeff=np.zeros((len(t),4));coeff[r,apex]=1.
    coeff[r[:,None],face]=-np.column_stack([1.-v-w,v,w])
    current=Y[r[:,None],face]
    ny=np.cross(current[:,1]-current[:,0],current[:,2]-current[:,0]);area_now=np.linalg.norm(ny,axis=1)
    if np.any(area_now==0):raise ValueError('degenerate supporting face in pose')
    ny/=area_now[:,None]
    projected=np.einsum('ni,nij->nj',coeff,Y)
    rhs=np.sign(h)*desired_vol/area_now-np.einsum('nj,nj->n',projected,ny)
    nodes=np.unique(t);packed=np.searchsorted(nodes,t)
    B=csr_matrix(((coeff[:,:,None]*ny[:,None,:]).ravel(),
                 (3*packed[:,:,None]+np.arange(3)).ravel(),np.arange(0,len(t)*12+1,12)),
                 shape=(len(t),len(nodes)*3))
    result=lsmr(B,rhs,atol=1e-11,btol=1e-10,conlim=1e9,maxiter=600)
    return nodes,result[0].reshape(-1,3),float(result[3]/max(np.linalg.norm(rhs),1e-30))


def run(full,donor,seed,out,iters):
    x,target,owner,t,mat,outer,_,pins=load_geometry(full,donor)
    if json.loads((seed/'result.json').read_text())['source']!=pins:raise ValueError('source mismatch')
    y=x+np.load(seed/'displacement.npy');initial=y.copy()
    ref=np.concatenate([differentials(x,t[i:i+32768])[0] for i in range(0,len(t),32768)])
    seen=np.zeros(len(t),bool);history=[]
    for it in range(iters+1):
        J,shape,det,bad=audit(y,t,ref)
        h=dict(iteration=it,failed=int(bad.sum()),inverted=int((det<=0).sum()),min_J=float(J.min()),max_J=float(J.max()))
        history.append(h);print(h,flush=True)
        if not bad.any() or it==iters:break
        seen|=bad
        # Previously recovered cells keep their volume if still comfortably admissible.
        sel=np.flatnonzero(seen);use=t[sel];_,_,long3=differentials(y,use)
        desired=np.clip(np.where(bad[sel],ref[sel],det[sel]),
                np.maximum.reduce([.50*ref[sel],np.full(len(sel),7.2e-18),1.12e-5*long3]),2.30*ref[sel])
        nodes,dy,rel=correct(x,y,use,ref[sel],desired)
        moved=np.zeros(len(x),bool);moved[nodes]=True
        near=moved[t].any(axis=1);part=t[near];rvol=ref[near]
        def score(j,shape,det,b):
            penalty=np.maximum.reduce([.4-j,j-2.5,(1e-5-shape)/1e-5,(6e-18-det)/rvol,np.zeros(len(j))])
            return (int(b.sum()),float(np.log1p(penalty).sum()))
        best=score(J[near],shape[near],det[near],bad[near]);alpha=0.
        orig=y[nodes].copy()
        for k in (1.,.75,.5,.25,.125,.0625,.03125):
            y[nodes]=orig+k*dy;key=score(*audit(y,part,rvol))
            if key<best:best=key;alpha=k
        y[nodes]=orig+alpha*dy
        h.update(linear_relative_residual=rel,fraction=alpha,nodes=int(len(nodes)))
        if not alpha:break
    out.mkdir(exist_ok=False);np.save(out/'displacement.npy',y-x)
    high=y.astype('f4');lo=(y-high.astype(float)).astype('f4')
    r=dict(source=pins,seed=str(seed),history=history,quality=measure(y,owner,t,mat),
           compensated_emission=measure(high.astype(float)+lo.astype(float),owner,t,mat),
           float32_world_emission=measure(high.astype(float),owner,t,mat),
           correction_to_seed=errors(y,initial),discrepancy_to_rigid_donors=errors(y,target),
           outer_error=errors(y[outer],target[outer]),outer_below_plane=int((y[outer,2]<0).sum()),
           scope='A construction for the deformed coordinates with unchanged constitutive reference. An accepted native loaded replay and attachment checks remain necessary.')
    (out/'result.json').write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('full','donor','seed','out'):p.add_argument(k,type=Path)
    p.add_argument('--iters',type=int,default=24);run(**vars(p.parse_args()))
