"""Offline connected-reference-fit QP, not the per-step physical solver.

Minimize .5*x.T*Q*x-rhs.T*x subject to A*x>=b by convex hinge
penalty continuation. Every Newton direction has an objective Armijo check.
Rows are scaled to unit gradient norm so reported violations have the same
millimetre-coordinate units as x, rather than mixing volume fractions and mm.
Geometric acceptance and physical mass mapping remain the caller responsibility.
"""
from __future__ import annotations
import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

def solve_hinge_qp(Q,rhs,A,b,penalties=(100.,1000.,10000.,100000.,1000000.),
                   tolerance_mm=1e-3,max_newton=48,max_wall_s=120.,x0=None):
    start=time.monotonic();Q=sparse.csc_matrix(Q);A=sparse.csr_matrix(A)
    rhs=np.asarray(rhs,float);b=np.asarray(b,float)
    n=len(rhs)
    if Q.shape!=(n,n) or A.shape!=(len(b),n):raise ValueError("QP domain mismatch")
    if any(not np.isfinite(v).all() for v in (Q.data,A.data,rhs,b)):
        raise ValueError("QP nonfinite coefficient")
    asym=Q-Q.T
    if np.max(np.abs(asym.data),initial=0)>1e-10*max(1.,np.max(np.abs(Q.data),initial=0)):
        raise ValueError("QP Hessian is not symmetric")
    rownorm=np.sqrt(np.asarray(A.multiply(A).sum(axis=1)).ravel())
    zero=rownorm<=1e-14
    bad=np.flatnonzero(zero&(b>1e-12))
    if len(bad):raise ValueError("fixed-support QP infeasible at zero-gradient constraint rows "+str(bad.tolist()))
    keep=~zero;C=(sparse.diags(1./rownorm[keep])@A[keep]).tocsr();d=b[keep]/rownorm[keep]
    x=np.zeros(n) if x0 is None else np.asarray(x0,float).copy()
    if x.shape!=(n,) or not np.isfinite(x).all():raise ValueError("invalid QP initial vector")
    if not np.all(np.asarray(penalties)>0):raise ValueError("nonpositive QP penalty")
    logs=[]
    def violation(z):return np.maximum(d-C@z,0.)
    def energy(z,rho):
        v=violation(z)
        return .5*float(z@(Q@z))-float(rhs@z)+.5*rho*float(v@v)
    for rho0 in penalties:
        rho=float(rho0);steps=[];stop="iteration_budget"
        for iteration in range(max_newton):
            if time.monotonic()-start>max_wall_s:stop="wall_budget";break
            residual=d-C@x;active=np.flatnonzero(residual>0)
            Ca=C[active];v=residual[active]
            g=Q@x-rhs-rho*(Ca.T@v)
            maxv=float(np.maximum(residual,0).max(initial=0))
            grad=float(np.linalg.norm(g,np.inf))
            if grad<=1e-9*max(1.,float(np.linalg.norm(rhs,np.inf))):stop="stationary_penalty";break
            H=(Q+rho*(Ca.T@Ca)).tocsc()
            fac=splu(H,permc_spec="COLAMD");dx=fac.solve(-g);del fac
            lin=float(np.linalg.norm(H@dx+g)/max(float(np.linalg.norm(g)),1e-30))
            if not np.isfinite(dx).all() or not np.isfinite(lin) or lin>1e-7:
                raise ValueError("QP Newton linear residual certification failed: "+str(lin))
            gd=float(g@dx)
            if gd>=0:raise ValueError("QP Newton direction is not descending")
            f0=energy(x,rho);alpha=1.;chosen=None
            for _ in range(40):
                trial=x+alpha*dx;f1=energy(trial,rho)
                if np.isfinite(f1) and f1<=f0+1e-4*alpha*gd+8*np.finfo(float).eps*max(1.,abs(f0)):
                    chosen=(trial,f1);break
                alpha*=.5
            if chosen is None:stop="line_search_stalled";break
            x,f1=chosen
            steps.append({"iteration":iteration,"active":int(len(active)),"energy_before":f0,
                          "energy_after":f1,"alpha":alpha,"linear_relative_residual":lin,
                          "maximum_violation_mm_before":maxv})
            if np.linalg.norm(alpha*dx,np.inf)<1e-11:stop="small_step";break
        v=violation(x);maxv=float(v.max(initial=0));count=int(np.count_nonzero(v>tolerance_mm))
        logs.append({"rho":rho,"stop":stop,"steps":steps,"maximum_violation_mm":maxv,
                     "unsatisfied_rows":count,"fixed_satisfied_rows":int(zero.sum()),
                     "elapsed_s":time.monotonic()-start})
        if count==0 or stop=="wall_budget":break
    return x,logs,float(violation(x).max(initial=0)),int(np.count_nonzero(violation(x)>tolerance_mm))
