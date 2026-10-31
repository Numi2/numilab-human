#!/usr/bin/env python3
"""Tiny equivalence and bounded-fallback controls for Q629 v15 row scaling."""
from pathlib import Path
import importlib.util
from unittest.mock import patch
import numpy as np
from scipy.optimize import linprog, OptimizeResult
from scipy.sparse import csr_matrix

HERE=Path(__file__).resolve().parent
HELPER=HERE/"quality_lp_629_v15_rowscaled_witness.py"
spec=importlib.util.spec_from_file_location("q629_v15_control_subject",HELPER)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)

def check_case(name,A,b,c,bounds,expected_status=0):
    A=csr_matrix(np.asarray(A,dtype=np.float64));b=np.asarray(b,dtype=np.float64)
    Ae,be,diag=q._equilibrate_inequality_rows(A,b)
    rowmax=np.asarray(np.abs(A).max(axis=1).toarray()).reshape(-1)
    magnitude=np.maximum(rowmax,np.abs(b))
    factors=np.ones(len(b),dtype=np.float64)
    nz=magnitude>0.
    factors[nz]=1./np.maximum(magnitude[nz],np.finfo(np.float64).tiny)
    assert np.all(factors>0.) and np.isfinite(factors).all()
    assert np.allclose(Ae.toarray(),A.toarray()*factors[:,None],rtol=1.e-14,atol=0.)
    assert np.allclose(be,b*factors,rtol=1.e-14,atol=0.)
    original=linprog(c,A_ub=A,b_ub=b,bounds=bounds,method="highs-ds")
    equilibrated=linprog(c,A_ub=Ae,b_ub=be,bounds=bounds,method="highs-ds")
    assert original.status==equilibrated.status,(name,original.status,equilibrated.status)
    assert original.status==expected_status,(name,original.status,expected_status)
    original_normalized_violation=None
    scaled_normalized_violation=None
    if original.success:
        rowmag=np.maximum(np.asarray(np.abs(A).max(axis=1).toarray()).reshape(-1),np.abs(b))
        floor=np.finfo(float).tiny
        original_normalized_violation=float(np.max(np.maximum(A@original.x-b,0.)/np.maximum(rowmag,floor),initial=0.))
        scaled_normalized_violation=float(np.max(np.maximum(A@equilibrated.x-b,0.)/np.maximum(rowmag,floor),initial=0.))
        assert scaled_normalized_violation<=1.e-7,(name,"scaled LP solution violates original rows",scaled_normalized_violation)
        if original_normalized_violation<=1.e-7:
            assert abs(original.fun-equilibrated.fun)<=1.e-9,(name,original.fun,equilibrated.fun)
            assert np.allclose(original.x,equilibrated.x,atol=1.e-9,rtol=1.e-9),(name,original.x,equilibrated.x)
    assert diag["method"].startswith("positive full-row")
    assert diag["post_equilibration"]["all_finite"]
    post=diag["post_equilibration"]["row_max_coefficient_or_rhs_nonzero"]
    assert post["max"] is None or post["max"]<=1.+1.e-12,(name,post)
    return {"case":name,"status":original.status,"objective":original.fun if original.success else None,
            "original_normalized_violation":original_normalized_violation,
            "row_scaled_solution_normalized_violation":scaled_normalized_violation,
            "pre_coeff_span":diag["pre_equilibration"]["coefficient_abs_nonzero"],
            "post_row_span":post,"zero_rows_left_unscaled":diag["zero_rows_left_unscaled"]}

reports=[]
# Empty inequality matrix and empty row both stay valid and untouched.
reports.append(check_case("empty_matrix",np.zeros((0,2)),[],[1.,0.],[(1.,None),(0.,None)]))
# A tiny coefficient row with an explicit auxiliary/slack variable, a large row,
# and a zero row; positive row factors must retain the same optimum.
reports.append(check_case("slack_dynamic_range_and_empty_row",
    [[-1.e-12,-1.e-12],[1.e6,0.],[1.e-15,0.],[0.,0.]],
    [-1.e-12,7.5e5,1.e-15,0.],[-1.,.1],[(0.,None),(0.,None)]))
# Opposing large coefficients exercise cancellation while preserving the feasible set.
reports.append(check_case("coefficient_cancellation",
    [[-1.e-12,-1.e-12],[1.e6,0.],[1.e12,-1.e12]],
    [-1.e-12,7.5e5,0.],[-1.,.1],[(0.,None),(0.,None)]))
# An all-zero row with negative RHS remains infeasible; it is not silently discarded.
reports.append(check_case("empty_infeasible_row",[[0.]],[-1.],[1.],[(None,None)],expected_status=2))

# Inject status 4 on the first call, then run the actual unchanged LP once with
# highs-ipm. Both calls must use identical costs, rows, RHS, and bounds; fallback
# time limit is only the remaining portion of the original total budget.
A=csr_matrix([[-1.,-1.],[1.,0.]])
b=np.array([-1.,.75]);cost=np.array([-1.,.1]);bounds=[(0.,None),(0.,None)]
real_linprog=q.linprog;seen=[]
def injected(*args,**kwargs):
    seen.append({"cost":np.array(args[0],copy=True),"A":kwargs["A_ub"].copy(),
                 "b":np.array(kwargs["b_ub"],copy=True),"bounds":kwargs["bounds"],
                 "method":kwargs["method"],"limit":kwargs["options"]["time_limit"]})
    if len(seen)==1:return OptimizeResult(status=4,success=False,x=None,fun=None,message="synthetic Not Set")
    return real_linprog(*args,**kwargs)
with patch.object(q,"linprog",side_effect=injected):
    solved,attempts,primary,elapsed=q._solve_highs_with_status4_fallback(cost,A,b,bounds,2.)
assert primary==4 and len(attempts)==2 and attempts[1]["method"]=="highs-ipm",attempts
assert solved.success and solved.status==0,(solved.status,solved.message)
assert seen[0]["method"]=="highs-ds" and seen[1]["method"]=="highs-ipm"
assert np.array_equal(seen[0]["cost"],seen[1]["cost"])
assert (seen[0]["A"]!=seen[1]["A"]).nnz==0
assert np.array_equal(seen[0]["b"],seen[1]["b"]) and seen[0]["bounds"]==seen[1]["bounds"]
assert 0.<seen[1]["limit"]<=2.
reports.append({"case":"bounded_status4_ipm_retry","primary_status":primary,
                "fallback_status":int(solved.status),"elapsed_s":elapsed,
                "fallback_time_limit_s":seen[1]["limit"],"same_LP_cost_rows_rhs_bounds":True})
print("PASS_ROW_EQUILIBRATION_CONTROLS")
for row in reports:print(row)
