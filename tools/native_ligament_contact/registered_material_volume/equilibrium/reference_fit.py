"""Fit a connected material hinge against face sidedness, cell shape and volume at three poses."""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

def separation_fit(t,source,poses,faces,pairs,nodes,checker,geom,mass=None):
    cells=t;corner,imap=np.unique(cells,return_inverse=True);T=imap.reshape(-1,4)
    local=np.searchsorted(corner,nodes);start=poses[0][nodes]
    frames=[p[corner] for p in poses];V=geom(source[corner],T)[0]
    ref=[geom(p,T) for p in frames]
    expressions=[]
    for a,b in pairs:
        options=[]
        for tri,opp in [(faces[a],faces[b]),(faces[b],faces[a])]:
            face=np.searchsorted(corner,tri);points=np.searchsorted(corner,np.setdiff1d(opp,tri))
            if np.any(corner[face]!=tri):raise ValueError('Missing reference triangle')
            def distance(p,face=face,points=points):
                s=p[face];n=np.cross(s[1]-s[0],s[2]-s[0])
                return (p[points]-s[0])@n/np.linalg.norm(n)*1e9
            d=distance(frames[0]);sign=1. if np.min(d)>0 else -1.
            options.append((np.min(sign*d),distance,sign))
        best=max(options,key=lambda o:o[0])
        if best[0]<=0:raise ValueError('No registered face-separation witness for this pair')
        expressions.append(best[1:])

    def constraints(u,clearance):
        out=[];delta=u.reshape(-1,3)*1e-9
        if mass is not None:
            base=source[corner].copy();base[local]+=delta
            volume,qref=geom(base,T);rho=mass/volume
            out.extend([(rho-850.)*.1,(950.-rho)*.1,1.e6*(qref-1.e-5)])
        for i,old in enumerate(frames):
            p=old.copy();p[local]+=delta
            J,q=geom(p,T);J=J/(volume if mass is not None else V);v0,q0=ref[i];lower=.35 if i else .4
            out.extend([1e2*(J-lower),1e2*(2.5-J),1e6*(q-np.minimum(1e-5,q0))])
            for distance,sign in expressions:out.append(sign*distance(p)-clearance)
        return np.concatenate(out)
    # Coordinates are ultimately represented as binary32. Optimize directly on
    # that lattice and certify the emitted positions with the nonlinear checks.
    n=3*len(nodes); steps=np.abs(np.spacing(start.astype('f4')).astype(float)).ravel()*1e9
    if not np.all(steps>0):raise ValueError('Invalid emission lattice')
    eye=np.eye(n); bounds=Bounds(np.r_[np.full(n,-48),0],np.r_[np.full(n,48),300])
    integrality=np.r_[np.ones(n),0]; objective=np.r_[np.zeros(n),1.]
    peak=np.vstack([np.c_[np.diag(steps),-np.ones(n)],np.c_[-np.diag(steps),-np.ones(n)]])
    limit=LinearConstraint(peak,-np.inf,0);history=[]
    for clearance in (16.,8.,4.,2.,1.):
        pos=np.zeros(n)
        for it in range(3):
            u=pos*steps; f=constraints(u,clearance)
            h=.125
            jac=np.column_stack([(constraints(u+h*e,clearance)-constraints(u-h*e,clearance))/(2*h) for e in eye])*steps
            mat=LinearConstraint(np.c_[jac,np.zeros(len(f))],jac@pos-f,np.inf)
            res=milp(objective,integrality=integrality,bounds=bounds,constraints=[mat,limit],options=dict(time_limit=7.,mip_rel_gap=.02))
            entry=dict(clearance_nm=clearance,iteration=it,status=int(res.status))
            history.append(entry)
            if res.x is None:break
            pos=np.rint(res.x[:n]);out=(start+(pos*steps).reshape(-1,3)*1e-9).astype('f4').astype(float)
            shift=out-start
            g=constraints(shift.ravel()*1e9,clearance);valid=checker(shift)
            entry.update(valid=bool(valid),minimum_constraint=float(g.min()),maximum_shift_m=float(np.linalg.norm(shift,axis=1).max()))
            if valid and min(g)>=-1.e-7:return out,history
    return None,history
