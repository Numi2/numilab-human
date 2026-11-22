"""Preserve a source group's mass and reference, load and endpoint first moments."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, hstack, vstack, csr_matrix
from torso_cell_geometry import geom


def route(states, changed, cells, mass, limits):
    vol=geom(changed[0],cells)[0]
    lower,upper=[rho*vol for rho in limits]
    total=mass.sum()
    span=upper-lower
    if total<=0 or np.any(span<=0):
        raise ValueError('Invalid mass coordinates')
    budget=total-lower.sum()
    # The compiler sums source allocations in float64. Test the resulting
    # error in physical units again after recovering each individual mass.
    mtol=32*np.finfo(float).eps*total
    if budget < -mtol or budget>span.sum()+mtol:
        return None
    budget=np.clip(budget,0.,span.sum())
    n=len(mass)
    Aeq=np.r_[span/span.max(),np.zeros(n)][None]
    rhs=[budget/span.max()]
    rows=[];bounds=[]
    for q,p in zip(states,changed):
        old=q[cells].mean(axis=1);new=p[cells].mean(axis=1)
        origin=old.mean(0)
        target=mass@(old-origin)-lower@(new-origin)
        mat=(new-origin).T*span
        for row,value in zip(mat,target):
            mag=max(float(np.max(abs(row))),abs(value),1e-14)
            rows.extend([row/mag,-row/mag])
            bounds.extend([(value+9.9e-15)/mag,(-value+9.9e-15)/mag])
    prior=(mass-lower)/span
    I=eye(n,format='csr');zero=csr_matrix((len(rows),n))
    A=vstack([hstack([I,-I]),hstack([-I,-I]),hstack([csr_matrix(rows),zero])],format='csr')
    fit=linprog(np.r_[np.zeros(n),span/span.max()],A_ub=A,b_ub=np.r_[prior,-prior,bounds],
                A_eq=Aeq,b_eq=rhs,bounds=[(0,1)]*n+[(0,None)]*n,method='highs',
                options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
    if not fit.success:return None
    result=lower+span*fit.x[:n]
    if abs(result.sum()-total)>max(mtol,1e-20) or np.any(result<lower) or np.any(result>upper+1e-25):
        return None
    for q,p in zip(states,changed):
        old=q[cells].mean(1);new=p[cells].mean(1);origin=old.mean(0)
        if np.max(abs(result@(new-origin)-mass@(old-origin)))>1e-14:
            return None
    return result
