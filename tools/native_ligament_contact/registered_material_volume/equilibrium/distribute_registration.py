"""Distribute a registered correction subject to both sampled configurations' actual cell limits."""
import numpy as np
from scipy.optimize import minimize
from supported_material_admission import geom

def clip_registration(x,src,delta,end,t,free):
    # Every degree of freedom is a fraction of a source-linked material move.
    # Spatial components are never chosen independently.
    V=geom(x,t)[0]
    baseline_quality=np.minimum(1e-5,geom(end,t)[1])
    def limits(weight,where,emitted):
        z=src.copy();z[free]+=weight[:,None]*delta[free]
        if emitted:z=z.astype('f4').astype(float)
        m=end+(z-src)  # accepted reference-displacement high and low words
        L=t[where];v,q=geom(z,L);w,qm=geom(m,L)
        j=v/V[where];h=w/V[where]
        return np.array([(j-.4),2.5-j,(q/1.e-5-1),h-.35,h-.35*j,2.5-h,(qm-baseline_quality[where])/1.e-5])
    rows=np.arange(len(t));weights=np.ones(len(free));active=np.array([],int)
    history=[]
    for _ in range(16):
        actual=limits(weights,rows,True)
        fail=np.flatnonzero(np.min(actual,axis=0)<-5.e-10)
        history.append(dict(invalid=len(fail),minimum=float(np.min(actual)),active=int(len(active))))
        if not len(fail):return weights,history
        active=np.union1d(active,fail)
        # Counter the *measured* Float32 loss for this represented proposal.
        # Keep the safety reserve for a cell's active limits within its baseline slack.
        exact=limits(weights,active,False);rounded=actual[:,active]
        base=np.maximum(0,limits(np.zeros_like(weights),active,True))
        margin=np.minimum(.35*base,2*np.abs(exact-rounded)+1e-5)
        def bounds(w):return (limits(w,active,False)-margin).ravel()
        def target(w):return .5*np.dot(1-w,1-w)
        opt=minimize(target,weights,jac=lambda w:w-1,method='SLSQP',bounds=[(0.,1.)]*len(weights),constraints=[dict(type='ineq',fun=bounds)],options=dict(maxiter=70,ftol=1.e-12))
        if not opt.success and np.min(bounds(opt.x)) < -1.e-7:return None,history
        nextw=np.clip(opt.x,0,1)
        if np.max(np.abs(nextw-weights))<1.e-10:return None,history
        weights=nextw
    # Fixed represented coordinates can jump past an active limit. Restore
    # just the move owners of violated cells, then recheck all incident rows.
    for _ in range(len(free)+1):
        bad=np.flatnonzero(limits(weights,rows,True).min(axis=0)<-5.e-10)
        if not len(bad):return weights,history
        affected=np.isin(free,t[bad])
        if not np.any(weights[affected]>0):return None,history
        weights[affected]=0
    return None,history
