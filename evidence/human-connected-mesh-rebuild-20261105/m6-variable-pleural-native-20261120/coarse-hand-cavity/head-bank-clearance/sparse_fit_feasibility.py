"""Bounded sparse feasibility updates for connected skin/fat patches."""
from time import monotonic

from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csr_matrix, hstack, identity, vstack
import numpy as np


def solve(initial, conditions, bounds, maxiter):
    """Keep nonlinear geometry authoritative and use sparse LPs for proposals."""
    inequalities, equality = conditions
    x = initial.copy()
    count = len(x)
    box = np.array(bounds)
    nonnegative = [(0., None)] * (count+1)
    objective = np.r_[np.zeros(count), np.full(count,1.e-6), 1.]
    delta_rows = identity(count,format="csr")
    absolute_delta = vstack((
        hstack((delta_rows,-delta_rows,csr_matrix((count,1)))),
        hstack((-delta_rows,-delta_rows,csr_matrix((count,1)))),
    ),format="csr")

    def values(q):
        ineq, eq = inequalities["fun"](q), equality["fun"](q)
        score = max(0.,-float(ineq.min()),float(np.abs(eq).max())*10.)
        return ineq,eq,score

    margin, conservation, score = values(x)
    trust = .35
    deadline = monotonic() + 30.
    for step in range(maxiter):
        if margin.min() >= -1.e-8 and np.abs(conservation).max() <= 5.e-9:
            return OptimizeResult(x=x,success=True,message="Exact sparse feasibility check passed",
                                  nit=step)
        jac = csr_matrix(inequalities["jac"](x))
        eq_jac = csr_matrix(equality["jac"](x))
        matrix = vstack((
            hstack((-jac,csr_matrix((len(margin),count)),csr_matrix(-np.ones((len(margin),1))))),
            absolute_delta,
        ),format="csr")
        equalities = hstack((eq_jac,csr_matrix((len(conservation),count+1))),format="csr")
        remaining = deadline-monotonic()
        if remaining <= 0.:
            break
        limits = list(zip(np.maximum(-trust,box[:,0]-x), np.minimum(trust,box[:,1]-x)))
        trial = linprog(objective,A_ub=matrix,b_ub=np.r_[margin,np.zeros(2*count)],
                        A_eq=equalities,b_eq=-conservation,bounds=limits+nonnegative,
                        method="highs",options={"time_limit":min(remaining,5.0)})
        if not trial.success:
            return OptimizeResult(x=x,success=False,message=f"Sparse proposal failed: {trial.message}",
                                  nit=step)
        p = x+trial.x[:count]
        new_margin,new_conservation,new_score=values(p)
        if not np.isfinite(new_score) or new_score >= score-1.e-12:
            trust *= .5
            if trust < 2.e-5:
                break
            continue
        x, margin, conservation, score = p,new_margin,new_conservation,new_score
        trust=min(.60,trust*1.4)
    return OptimizeResult(x=x,success=False,message="Bounded sparse feasibility search stopped")
