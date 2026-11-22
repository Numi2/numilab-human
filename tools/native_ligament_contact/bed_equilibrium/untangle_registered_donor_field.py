"""Untangle the initial posed coordinates while keeping reference, cell masses and topology fixed."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from screen_registered_dense_posture import load_geometry, measure
from construct_registered_donor_field import errors


def differentials(x, t):
    v=x[t]
    b,c,d=(v[:, k]-v[:, 0] for k in (1,2,3))
    grad=np.empty_like(v)
    grad[:,1]=np.cross(c,d);grad[:,2]=np.cross(d,b);grad[:,3]=np.cross(b,c)
    grad[:,0]=-grad[:,1:].sum(1)
    det=np.einsum('ij,ij->i', b, grad[:,1])
    span=np.maximum.reduce([np.einsum('ij,ij->i',e,e) for e in (b,c,d,b-c,b-d,c-d)])
    return det,grad,span*np.sqrt(span)


def deficits(x, t, reference, lo, hi):
    ds=[];sc=[]
    for begin in range(0,len(t),32768):
        sl=slice(begin,begin+32768)
        det, grad, length3=differentials(x,t[sl])
        desired=np.maximum.reduce([np.full(len(det),6.03e-18),1.005e-5*length3,lo*reference[sl]])
        ds.append(np.maximum(desired-det,det-hi*reference[sl]) / reference[sl])
        sc.append(det/reference[sl])
    return np.concatenate(ds),np.concatenate(sc)


def step(x, cells, ref, bad, lower, upper):
    active_nodes=np.unique(cells[bad])
    index=np.full(len(x),-1,np.int32);index[active_nodes]=np.arange(len(active_nodes))
    ix=np.flatnonzero((index[cells]>=0).any(1))
    t=cells[ix];det,grad,l3=differentials(x,t)
    idx=index[t];grad[idx<0]=0.
    norm=np.maximum(np.sqrt((grad*grad).sum((1,2))),np.finfo(float).tiny)
    # Every incident tetrahedron enters both conservative bounds. A fresh global
    # measurement and backtrack also certify the nonlinear longest-edge term.
    unit=grad/norm[:,None,None]
    sign=np.vstack((unit.reshape(len(t),12),-unit.reshape(len(t),12)))
    dofs=(3*np.maximum(idx,0)[:,:,None]+np.arange(3)).reshape(len(t),12)
    dofs=np.vstack((dofs,dofs))
    bounds=np.r_[np.maximum.reduce([np.full(len(det),6.12e-18),1.02e-5*l3,lower*ref[ix]*1.006])-det,
                 det-upper*ref[ix]*.996]/np.tile(norm,2)
    # The exact dual of minimum nodal displacement under the linearized inequalities.
    scale=max(float(bounds.max()),1e-10)
    A=csr_matrix((sign.ravel(),dofs.ravel(),np.arange(0,sign.size+1,12)),shape=(2*len(t),3*len(active_nodes)))
    A.eliminate_zeros()
    # Only rows reachable within a conservative trust box need enter this step.
    possible=bounds>=-4.*scale
    A=A[possible];b=bounds[possible]/scale;AT=A.T.tocsr()
    def dual(v):
        u=AT@v
        return .5*(u@u)-b@v,A@u-b
    sol=minimize(dual,np.zeros(len(b)),jac=True,method='L-BFGS-B',bounds=[(0.,None)]*len(b),
                 options={'maxiter':300,'ftol':1e-15,'gtol':1e-7,'maxcor':8})
    delta=(AT@sol.x).reshape(-1,3)*scale
    if not np.isfinite(delta).all():raise ValueError('nonfinite untangling step')
    return active_nodes,delta,ix


def run(full,donor,seed,out,limit):
    x,target,owner,cells,mat,outer,_,pins=load_geometry(full,donor)
    previous=json.loads((seed/'result.json').read_text())
    if previous['source']!=pins:raise ValueError('pose/cell source mismatch')
    y=x+np.load(seed/'displacement.npy');initial=y.copy()
    ref=np.concatenate([differentials(x,cells[i:i+32768])[0] for i in range(0,len(cells),32768)])
    if not np.all(ref>0):raise ValueError('reference must be orientation preserving')
    lower,upper=.40,2.5
    log=[]
    for trial in range(limit+1):
        ds,ratio=deficits(y,cells,ref,lower,upper)
        bad=ds>1e-10
        log.append(dict(iteration=trial,violations=int(bad.sum()),max_deficit=float(ds.max()),
                        min_J=float(ratio.min()),max_J=float(ratio.max())))
        print(log[-1],flush=True)
        if not bad.any() or trial==limit:break
        nodes,delta,affected=step(y,cells,ref,bad,lower,upper)
        before=y[nodes].copy()
        best_score=np.linalg.norm(np.maximum(ds[affected],0))
        winner=0.
        for alpha in (1.,.75,.5,.25,.125,.0625,.03125):
            y[nodes]=before+alpha*delta
            local,_=deficits(y,cells[affected],ref[affected],lower,upper)
            score=np.linalg.norm(np.maximum(local,0))
            if score<best_score*(1.-1.e-9):
                winner=alpha;best_score=score
        y[nodes]=before+winner*delta
        if not winner:break
    r=dict(source=pins,seed=str(seed),iterations=log,J_bounds=[lower,upper],
           quality=measure(y,owner,cells,mat),coordinates_rounded_float32_quality=measure(y.astype(np.float32).astype(float),owner,cells,mat),
           correction_to_seed=errors(y,initial),discrepancy_to_rigid_donors=errors(y,target),
           outer_below_plane=int((y[outer,2]<0).sum()),
           scope='Deformed initial geometry only, with original material reference. Contact, attachment errors and native stored energy remain to be qualified.')
    out.mkdir(exist_ok=False);np.save(out/'displacement.npy',y-x);(out/'result.json').write_text(json.dumps(r,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('full','donor','seed','out'):p.add_argument(k,type=Path)
    p.add_argument('--iterations',type=int,default=10)
    a=p.parse_args();run(a.full,a.donor,a.seed,a.out,a.iterations)
