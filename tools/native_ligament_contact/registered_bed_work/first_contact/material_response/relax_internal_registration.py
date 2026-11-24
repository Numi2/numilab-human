"""Condition the starting displacement field inside fixed outer and deep tissue boundaries."""
from pathlib import Path
import sys,json
import numpy as np
from compile_connected_joint import digest
from certify_transported_material import measure

def run(job,vol,out):
    out.mkdir();read=lambda f:np.loadtxt(f,usecols=(1,2,3),skiprows=1).astype('f4').astype(float)
    x,y=read(vol/'mesh.node'),read(vol/'current.node');initial=y.copy()
    tet=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4),dtype='i4')-1
    b=np.load(job/'registered-boundary/registered-boundary.npz')
    reg=json.loads((job.parent/'ligament-equilibrium-070/scene-registration/registration.json').read_bytes())
    protected=np.zeros(len(x),bool)
    cortical=np.load(job/'verified-connected-material/contact/material-contact.npz')['samples'][:,0].astype(int)
    protected[np.r_[b['O_node_joint'],b['D_node_joint'],cortical,reg['first_contact_new'],[a[2] for a in reg['anchors']]].astype(int)]=True
    def deformation(coords,t):return np.linalg.solve(x[t[:,1:]]-x[t[:,:1]],coords[t[:,1:]]-coords[t[:,:1]])
    J=np.concatenate([np.linalg.det(deformation(y,t)) for t in np.array_split(tet,64)])
    # Refit the complete set of high and low volume-ratio stars, in their common material frame.
    targets=np.flatnonzero((J>1.70)|(J<.65));centers=np.unique(tet[targets]);centers=centers[~protected[centers]]
    mask=np.isin(tet,centers);rows=np.flatnonzero(mask.any(1));t=tet[rows];em=x[t[:,1:]]-x[t[:,:1]]
    inv=np.linalg.inv(em);g=np.concatenate([-inv.sum(2)[:,:,None],inv],axis=2);det0=np.linalg.det(em)
    def metrics(points):
        F=inv@(points[t[:,1:]]-points[t[:,:1]]);sv=np.linalg.svd(F,compute_uv=False);jj=np.linalg.det(F)
        return dict(high=int((jj>1.7).sum()),low=int((jj<.65).sum()),severe=int(np.sum((sv[:,0]>2)|(sv[:,2]<.5))),jacobian=[float(jj.min()),float(jj.max())],det_deviation=float(np.sum((jj-1)**2)))
    base=metrics(y);baseJ=J[rows]
    low=np.minimum(baseJ,.70)-1e-9;high=np.maximum(baseJ,1.5)+1e-9
    dr,sc=measure(y,t);minshape=np.minimum(dr/sc,1e-4)
    pairs=np.argwhere(np.isin(t,centers));order=np.argsort(t[pairs[:,0],pairs[:,1]],kind='stable');pairs=pairs[order];ns=t[pairs[:,0],pairs[:,1]]
    groups=np.split(pairs,np.flatnonzero(np.diff(ns))+1)
    groups=sorted(groups,key=lambda r:-max(abs(baseJ[r[:,0]]-1)))
    history=[]
    for iteration in range(9):
        updates=0
        for group in groups:
            a,k=group.T;n=t[a[0],k[0]];v=y[t[a]];F=inv[a]@(v[:,1:]-v[:,:1]);j=np.linalg.det(F)
            U,sv,V=np.linalg.svd(F);R=U@V
            if np.any(np.linalg.det(R)<0):raise ValueError('Improper registration')
            G=g[a,:,k];C=j[:,None,None]*np.linalg.inv(F).transpose(0,2,1)
            dJ=np.einsum('ni,nij->nj',G,C)
            # Values in material coordinates are dimensionless. This is an initial
            # registration fit. The constitutive law in the simulation is unaltered.
            wt=np.maximum(1,np.maximum((j-1)/.5,(1-j)/.3)**2)
            shear=.12
            res=F-R;A=dJ.T@(wt[:,None]*dJ)+np.eye(3)*shear*np.sum(G*G)
            rhs=-dJ.T@(wt*(j-1))-shear*np.einsum('ni,nij->j',G,res)
            delta=np.linalg.solve(A,rhs)
            if not np.isfinite(delta).all():raise ValueError('Fit failed')
            previous=np.sum(wt*(j-1)**2)+shear*np.sum(res**2)
            lower=np.minimum(low[a],j);upper=np.maximum(high[a],j)
            slope=dJ@delta;alpha=1.
            neg=slope<0;pos=slope>0
            if neg.any():alpha=min(alpha,np.min((lower[neg]-j[neg])/slope[neg]))
            if pos.any():alpha=min(alpha,np.min((upper[pos]-j[pos])/slope[pos]))
            if alpha<1e-9:continue
            for back in range(12):
                trial=(y[n]+delta*alpha*2**(-back)).astype('f4').astype(float)
                if np.array_equal(trial,y[n]):break
                change=trial-y[n];jj=j+dJ@change
                if np.any(jj<low[a]-1e-7) or np.any(jj>high[a]+1e-7):continue
                vv=v.copy();vv[np.arange(len(a)),k]=trial
                edges=vv[:,1:]-vv[:,:1];dd=np.linalg.det(edges)
                longest=np.max(np.linalg.norm(vv[:,:,None,:]-vv[:,None,:,:],axis=3),axis=(1,2))
                if np.any(dd<6e-18) or np.any(dd<minshape[a]*longest**3):continue
                FF=F+G[:,:,None]*change[None,None,:]
                vs=np.linalg.svd(FF,compute_uv=False)
                next_energy=np.sum(wt*(jj-1)**2)+shear*np.sum((vs-1)**2)
                if next_energy>=previous or np.sum((vs[:,0]>2)|(vs[:,2]<.5))>np.sum((sv[:,0]>2)|(sv[:,2]<.5)):continue
                y[n]=trial;updates+=1;break
        m=metrics(y);history.append(m|dict(updates=updates));print(iteration,history[-1],flush=True)
        if not updates:break
    changed=np.flatnonzero(np.any(initial!=y,axis=1))
    if np.any(protected[changed]):raise ValueError('Prescribed geometry moved')
    np.savez_compressed(out/'candidate.npz',nodes=changed,positions=y[changed].astype('f4'))
    with (vol/'current.node').open() as f,(out/'current.node').open('x') as o:
        o.write(next(f));sel=set(changed.tolist())
        for i,line in enumerate(f):o.write(str(i+1)+' '+' '.join(format(d,'.17g') for d in y[i])+'\n' if i in sel else line)
    report=dict(reference=digest(vol/'mesh.node'),initial=digest(vol/'current.node'),elements=digest(vol/'mesh.ele'),result=digest(out/'current.node'),total_cells=len(tet),free=len(centers),star_cells=len(t),changed=len(changed),maximum_change_m=float(np.max(np.linalg.norm(y-initial,axis=1))),initial_patch=base,iterations=history)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':run(*(Path(s).resolve() for s in sys.argv[1:]))
