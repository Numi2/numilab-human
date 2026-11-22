"""Convex bounds on both extremal principal stretches for a normal-only P1 displacement."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import vstack,hstack,csr_matrix,coo_matrix
def solve(obj,M,f0,A,b,limits,enabled):
    lower,upper=.69,1.46
    k=upper**2/(upper**2-1)
    smooth=1-lower**2
    n=len(f0);size=M.shape[1]
    def append(matrix,rhs):
        nonlocal A,b
        mx=np.maximum(1,np.asarray(abs(matrix).max(axis=1).toarray()).ravel())
        block=matrix.multiply((1./mx)[:,None]).tocsr()
        A=vstack([A,hstack([block,csr_matrix((len(rhs),1))])],format='csr');b=np.r_[b,rhs/mx]
    if enabled:
        axial=M[2::3];base=1+f0[:,2]
        append(vstack([-axial,axial]),np.r_[base-lower,upper-base])
    for step in range(14):
        result=linprog(obj,A_ub=A,b_ub=b,bounds=limits,method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
        if not enabled or not result.success:return result
        g=(M@result.x[:size]).reshape(-1,3)+f0
        norm=(g[:,:2]**2).sum(axis=1);ax=1+g[:,2]
        # Derived from the Schur complements of U^2 I - F F^T and F F^T - L^2 I.
        # The lower constraint is written in its convex second-order-cone form.
        mid=np.sqrt(1+norm/smooth)
        hi=k*norm+ax*ax-upper**2;lo=lower*mid-ax
        ih=np.flatnonzero(hi>1e-6);il=np.flatnonzero(lo>1e-6)
        if not len(ih)+len(il):return result
        indices=np.r_[ih,il]
        dh=np.column_stack((2*k*g[ih,:2],2*ax[ih]));dl=np.column_stack((lower*g[il,:2]/(smooth*mid[il,None]),-np.ones(len(il))))
        tangent=np.vstack([dh,dl]);m=len(indices)
        combine=coo_matrix((tangent.ravel(),(np.repeat(np.arange(m),3),(3*indices[:,None]+np.arange(3)).ravel())),shape=(m,3*n)).tocsr()
        gradient=combine@M
        append(gradient,gradient@result.x[:size]-np.r_[hi[ih],lo[il]])
    raise ValueError('Spectral projection did not certify within its iteration limit')
