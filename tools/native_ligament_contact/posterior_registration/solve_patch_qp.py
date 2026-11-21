"""Active-set polish for a strictly convex sparse quadratic with two volume constraints."""
import numpy as np
from scipy.sparse import bmat,csc_matrix
from scipy.sparse.linalg import spsolve
def solve(H,G,lower,upper,warm):
    x=warm.copy()
    if max(np.max(lower-x),np.max(x-upper),np.max(abs(G@x)))>2.e-6:
        raise ValueError('Infeasible starting displacement')
    active=np.zeros(len(x),int)
    active[x-lower<2.e-6]=-1;active[upper-x<2.e-6]=1
    for steps in range(5*len(x)):
        free=np.flatnonzero(active==0);bound=np.flatnonzero(active)
        val=np.where(active[bound]<0,lower[bound],upper[bound])
        B=csc_matrix(G[:,free]);K=bmat([[H[free][:,free],B.T],[B,None]],format='csc')
        rhs=np.r_[-H[free][:,bound]@val,-G[:,bound]@val]
        sol=spsolve(K,rhs)
        if not np.isfinite(sol).all():raise ValueError('Singular patch KKT system')
        target=x.copy();target[free]=sol[:len(free)];target[bound]=val
        d=target-x
        t=np.ones(len(x));low=(active==0)&(target<lower-1e-9);high=(active==0)&(target>upper+1e-9)
        t[low]=(lower[low]-x[low])/d[low];t[high]=(upper[high]-x[high])/d[high]
        if low.any() or high.any():
            i=np.argmin(t);x+=np.clip(t[i],0,1)*d;active[i]=-1 if low[i] else 1
            continue
        x=target;grad=H@x+G.T@sol[len(free):]
        violation=np.where(active==-1,-grad,np.where(active==1,grad,0))
        if violation.max()<1e-7:
            if max(abs(G@x))>1e-6 or max(lower-x)>1e-7 or max(x-upper)>1e-7:raise ValueError('Invalid optimum')
            return x,steps
        active[violation.argmax()]=0
    raise ValueError('Bounded patch solve did not converge')
