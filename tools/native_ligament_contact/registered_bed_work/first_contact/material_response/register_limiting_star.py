"""Fit a connected, bonded material star around the coupled solver's volume witness."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import minimize
from compile_connected_joint import digest
from certify_transported_material import measure

def run(job,vol,out,witness,offset):
    out.mkdir();r=lambda p:np.loadtxt(p,usecols=(1,2,3),skiprows=1).astype('f4').astype(float)
    x,y=r(vol/'mesh.node'),r(vol/'current.node')
    e=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    c=witness-offset;t0=e[c];start=y.copy()
    b=np.load(job/'registered-boundary/registered-boundary.npz');cortex=np.load(job/'verified-connected-material/contact/material-contact.npz')['samples'][:,0]
    reg=json.loads((job.parent/'ligament-equilibrium-070/scene-registration/registration.json').read_bytes())
    pins=np.r_[b['O_node_joint'],b['D_node_joint'],cortex,reg['first_contact_new'],[a[2] for a in reg['anchors']]].astype(int)
    vs=np.setdiff1d(np.unique(e[np.isin(e,t0).any(1)]),pins)
    rows=np.flatnonzero(np.isin(e,vs).any(1));t=e[rows];target=int(np.flatnonzero(rows==c)[0])
    loc=np.full(len(x),-1,'i4');loc[vs]=np.arange(len(vs));ix,k=np.nonzero(loc[t]>=0)
    edges=x[t[:,1:]]-x[t[:,:1]];iv=np.linalg.inv(edges);rest=np.linalg.det(edges)
    D=np.concatenate([-iv.sum(2)[:,:,None],iv],2);N=len(t);p=len(vs);mu=1e-6
    def states(u):
        a=y[t].copy();a[ix,k]+=u.reshape(-1,3)[loc[t[ix,k]]]*mu
        F=iv@(a[:,1:]-a[:,:1]);j=np.linalg.det(F);return a,F,j
    zero=np.zeros(3*p);v,F0,j0=states(zero);U,s0,V=np.linalg.svd(F0);R=U@V
    def grad(B):
        d=np.einsum('ndk,ndj->nkj',D,B)
        dense=np.zeros((N,p,3));dense[ix,loc[t[ix,k]]]=d[ix,k]*mu
        return dense.reshape(N,-1)
    def cof(F):return np.stack([np.cross(F[:,:,1],F[:,:,2]),np.cross(F[:,:,2],F[:,:,0]),np.cross(F[:,:,0],F[:,:,1])],axis=2)
    le=np.array([[i,k] for i in range(4) for k in range(i)])
    lo=np.maximum(.4000005,np.minimum(.70,j0)-.015)
    hi=np.minimum(2.4999,np.maximum(1.65,j0)+.015)
    def obj(u):
        a,F,j=states(u);diff=F-F0;w=np.ones(N);w[target]=100
        # Use a local registration objective. The surrounding elements penalize distortion
        # and the actual simulation subsequently rebuilds the material forces at this state.
        per=(j-1)**2*w+.15*np.sum(diff**2,(1,2))
        value=np.sum(per)+.000002*(u@u)
        derivative=(2*w*(j-1))@grad(cof(F))+.3*grad(diff).sum(0)+.000004*u
        return value,derivative
    def constraint(u,derivative=False):
        a,F,j=states(u);dv=a[:,le[:,0]]-a[:,le[:,1]];s2=(dv*dv).sum(2);ids=np.argmax(s2,1);jrow=np.arange(N)
        scale=s2[jrow,ids]**1.5
        floor=1.000001e-5*scale/rest;tiny=6.01e-18/rest
        if not derivative:return np.r_[j-lo,hi-j,j-floor,j-tiny]
        jac=grad(cof(F));p0,p1=le[ids].T;dl=np.zeros((N,4,3))
        dl[jrow,p0]=3*dv[jrow,ids]*np.sqrt(s2[jrow,ids])[:,None]*1.000001e-5/rest[:,None];dl[jrow,p1]=-dl[jrow,p0]
        ds=np.zeros((N,p,3));ds[ix,loc[t[ix,k]]]=dl[ix,k]*mu
        return np.vstack((jac,-jac,jac-ds.reshape(N,-1),jac))
    solutions=[];report=[]
    for limit in [25.,100.]:
        s=minimize(obj,zero,method='SLSQP',bounds=[(-limit,limit)]*len(zero),jac=True,constraints=[dict(type='ineq',fun=constraint,jac=lambda u:constraint(u,True))],options=dict(maxiter=100,ftol=1e-10))
        for back in range(14):
            moved=(start[vs]+(s.x*mu*2**-back).reshape(-1,3)).astype('f4').astype(float)
            u=((moved-start[vs])/mu).ravel();a,F,j=states(u);sv=np.linalg.svd(F,compute_uv=False);severe=(sv[:,0]>2)|(sv[:,2]<.5)
            if np.array_equal(start[vs],moved) or np.min(constraint(u)) < -1e-7 or severe.sum()>((s0[:,0]>2)|(s0[:,2]<.5)).sum():continue
            if j[target]<j0[target]:
                solutions.append((float(j[target]),moved.copy()))
                report.append(dict(trust_um=limit,fraction=2**-back,solver_message=s.message,cells=len(t),severe=int(severe.sum()),target_J=float(j[target])));break
    if not solutions:raise ValueError('No admissible improvement')
    chosen=min(solutions,key=lambda s:s[0]);y[vs]=chosen[1];ch=np.flatnonzero(np.any(y!=start,axis=1));cs=set(ch.tolist())
    np.savez_compressed(out/'candidate.npz',nodes=ch,positions=y[ch].astype('f4'))
    with (vol/'current.node').open() as f,(out/'current.node').open('x') as g:
        g.write(next(f))
        for i,line in enumerate(f):g.write(str(i+1)+' '+' '.join(format(v,'.17g') for v in y[i])+'\n' if i in cs else line)
    rec=dict(source=digest(vol/'current.node'),elements=digest(vol/'mesh.ele'),current=digest(out/'current.node'),witness=witness,provisional_offset=offset,target_nodes=t0.tolist(),initial_J=float(j0[target]),fitted_J=chosen[0],moved=len(ch),maximum_change_m=float(np.linalg.norm(y-start,axis=1).max()),trials=report)
    (out/'result.json').write_text(json.dumps(rec,indent=2)+'\n');print(rec,flush=True)
if __name__=='__main__':run(*(Path(v).resolve() for v in sys.argv[1:4]),*(int(v) for v in sys.argv[4:]))
