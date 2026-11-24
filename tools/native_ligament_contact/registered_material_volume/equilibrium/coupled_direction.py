"""Compute a shared displacement direction for connected registration patches with fixed exterior data."""
import numpy as np
from scipy.sparse import coo_matrix,eye,diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

def fit(y,src,x,t,free,inv,vr,mu,bulk,guards,interval,radii,local,energy,geom,witness):
    # H approximates the positive elastic shear and compressive contributions.
    # Trial energies and feasibility use the original nonlinear material law.
    free=np.asarray(free)
    index=np.full(len(y),-1,dtype=int);index[free]=np.arange(len(free))
    ix=index[t];valid=ix>=0; every=np.arange(len(t))
    rr,cc=np.nonzero(valid)
    incid=coo_matrix((np.ones(len(rr)),(rr,ix[rr,cc])),shape=(len(t),len(free))).tocsr()
    _,component=connected_components(incid.T@incid,directed=False)
    if np.any(incid.sum(0)==0):raise ValueError('Disconnected free coordinate')
    groups=[]
    for k in np.unique(component):
        nodes=np.flatnonzero(component==k); rows=np.flatnonzero(np.asarray(incid[:,nodes].sum(1)).ravel())
        groups.append((nodes,rows))
    g=np.concatenate((-inv.sum(axis=2,keepdims=True),inv),axis=2).transpose(0,2,1)
    shape_grad=coo_matrix(((g[rr,cc]*np.sqrt(vr[rr]*mu[rr])[:,None]).ravel(),((rr[:,None]*3+np.arange(3)).ravel(),np.repeat(ix[rr,cc],3))),shape=(len(t)*3,len(free))).tocsr()
    from scipy.sparse import kron
    shear=kron(shape_grad.T@shape_grad,eye(3),format='csr')
    radius=np.array([radii[int(n)] for n in free])
    def gradient():
        F=local(every,y[t]);lj=np.log(np.linalg.det(F))
        h=(np.linalg.inv(F).transpose(0,2,1)@g.transpose(0,2,1)).transpose(0,2,1)
        fg=(F@g.transpose(0,2,1)).transpose(0,2,1)
        forces=vr[:,None,None]*(mu[:,None,None]*fg+(bulk*lj-mu)[:,None,None]*h)
        grad=np.zeros((len(free),3));np.add.at(grad,ix[rr,cc],forces[rr,cc])
        return h,grad
    def admissible(ytrial,a):
        pts=ytrial[t[a]];F=local(a,pts)
        J=np.linalg.det(F)
        if not np.all((J>=.4)&(J<=2.5)):return False
        vv,qq=geom(ytrial,t[a])
        if np.min(vv)<=1e-18 or np.min(qq)<1e-5:return False
        for disp,lo,hi,qmin,emax in guards:
            z=pts+disp[t[a]];G=local(a,z);j=np.linalg.det(G)
            # Keep a reserve of the control's remaining admissible J interval in
            # each observed pose. Baseline cells at a limit can only move inward.
            lower=.35+.5*(lo[a]-.35);upper=2.5-.5*(2.5-hi[a])
            if not np.all((j>=lower-1e-9)&(j<=upper+1e-9)):return False
            _,qs=geom(z.reshape(-1,3),np.arange(len(a)*4).reshape(-1,4))
            if np.any(qs<np.minimum(qmin[a],1.01e-5)*(1-1e-9)):return False
            logs=np.log(j);W=.5*mu[a]*((G*G).sum((1,2))-3-2*logs)+.5*bulk[a]*logs**2
            if np.any(W>1.05*emax[a]+.01*mu[a]):return False
        values=np.empty((len(a),3));tt=np.ascontiguousarray(t[a],dtype=np.int32)
        if interval(x,ytrial,len(x),tt,len(tt),values):raise ValueError('Native interval checker failed')
        margin=np.minimum(values[:,0]-values[:,1]-float(np.float32(.35)),2.5-values[:,0]-values[:,1])
        return np.all(np.isfinite(margin)&(margin>0))
    report=[]
    for iteration in range(8):
        h,grad=gradient()
        hh=h[rr,cc]*np.sqrt(vr[rr]*bulk[rr])[:,None]
        B=coo_matrix((hh.ravel(),(np.repeat(rr,3),(ix[rr,cc,None]*3+np.arange(3)).ravel())),shape=(len(t),3*len(free))).tocsr()
        H=shear+B.T@B
        scale=1/np.sqrt(H.diagonal());D=diags(scale)
        direction=(-scale*spsolve((D@H@D).tocsc(),scale*grad.ravel())).reshape(-1,3)
        if not np.all(np.isfinite(direction)) or (direction*grad).sum()>=0:raise ValueError('Coupled direction does not descend')
        accepted=0
        for nodes,a in groups:
            nd=free[nodes];old=y[nd].copy()
            oldE=energy(local(a,y[t[a]]),a)
            for step in 2.**(-np.arange(18)):
                trial=(old+step*direction[nodes]).astype('f4').astype(float)
                if np.all(old==trial):break
                if np.any(np.linalg.norm(trial-src[nd],axis=1)>radius[nodes]):continue
                y[nd]=trial
                if admissible(y,a) and energy(local(a,y[t[a]]),a)<oldE:
                    accepted+=1;break
                y[nd]=old
        _,gg=gradient()
        report.append(dict(sweep=iteration+1,components=len(groups),changed_components=accepted,energy_J=float(energy(local(every,y[t]),every)),witness_force_N=float(np.linalg.norm(gg[index[witness]]))))
        if accepted==0:break
    return report
