"""Choose emitted material coordinates and conservative density changes in one local solve."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, diags, eye, hstack, vstack


def choose(rows, lo, hi, states, cells, groups, mass, volume, G, quantum,
           flags, axis, lower, upper):
    n,m=len(quantum),len(mass)
    # Unknowns are integer coordinate quanta, fractional cell mass changes,
    # and absolute coordinate quanta. Mass bounds below use the moved volume.
    blocks=[];lower_bounds=[];upper_bounds=[]
    def block(a,b,c,l,u):
        M=hstack((csr_matrix(a),csr_matrix(b),csr_matrix(c)),format='csr')
        scale=np.asarray(abs(M).max(axis=1).toarray()).ravel()
        low=np.broadcast_to(l,(M.shape[0],));high=np.broadcast_to(u,(M.shape[0],))
        if np.any((low[scale==0]>0)|(high[scale==0]<0)):
            raise ValueError('Infeasible fixed material constraint')
        use=scale>0
        blocks.append(diags(1./scale[use])@M[use])
        lower_bounds.extend(low[use]/scale[use]);upper_bounds.extend(high[use]/scale[use])
    def zero(r,c):return csr_matrix((r,c))
    block(rows,zero(len(rows),m),zero(len(rows),n),lo,hi)
    block(lower[:,None]*G,-diags(mass),zero(m,n),-np.inf,mass-lower*volume)
    block(upper[:,None]*G,-diags(mass),zero(m,n),mass-upper*volume,np.inf)
    for g in np.unique(groups):
        sel=groups==g
        weights=mass*sel
        block(zero(1,n),weights[None],zero(1,n),0.,0.)
        coeff=.25*(weights@flags)*quantum
        for q in states:
            centers=q[cells].mean(1)
            origin=centers[sel].mean(0)
            for j in range(3):
                block(coeff[None] if j==axis else zero(1,n),
                      (weights*(centers[:,j]-origin[j]))[None],zero(1,n),-8e-15,8e-15)
    I=eye(n)
    block(I,zero(n,m),-I,-np.inf,0)
    block(-I,zero(n,m),-I,-np.inf,0)
    cap=48.
    result=milp(np.r_[np.zeros(n+m),quantum/quantum.max()],integrality=np.r_[np.ones(n),np.zeros(m+n)],
        bounds=Bounds(np.r_[-np.full(n,cap),-np.ones(m),np.zeros(n)],np.r_[np.full(n,cap),np.full(m,2.),np.full(n,cap)]),
        constraints=LinearConstraint(vstack(blocks),lower_bounds,upper_bounds),options={'time_limit':12,'mip_rel_gap':.005})
    return result
