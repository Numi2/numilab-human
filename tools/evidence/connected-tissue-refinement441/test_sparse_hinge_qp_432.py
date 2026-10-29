import importlib.util
from pathlib import Path
import numpy as np
from scipy import sparse
f=Path(__file__).with_name("solve_sparse_hinge_qp_432.py");s=importlib.util.spec_from_file_location("fitqp",f);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def solve(Q,rhs,A,b):
 x,logs,v,c=m.solve_hinge_qp(sparse.csc_matrix(Q),np.array(rhs),sparse.csr_matrix(A),np.array(b),tolerance_mm=1e-4)
 for stage in logs:
  for p in stage["steps"]:assert p["energy_after"]<=p["energy_before"]+1e-8*max(1.,abs(p["energy_before"]))
 return x,v,c
x,v,c=solve([[1.]],[2.],[[-1.]],[-1.]);assert c==0 and abs(x[0]-1)<1e-4
x,v,c=solve([[1.,0],[0,4.]],[0.,0.],[[1,1],[1,0],[0,1],[1+1e-12,1+1e-12],[0,0]],[2,0,0,2+2e-12,-1]);assert c==0 and np.allclose(x,[1.6,.4],atol=2e-4),(x,v,c)
x,v,c=solve([[1.]],[0.],[[1.],[-1.]],[1.,0.]);assert c==2 and v>.49
try:solve([[1.]],[0.],[[0.]],[1.])
except ValueError as e:assert "fixed-support" in str(e)
else:raise AssertionError("fixed positive constraint admitted")
try:solve([[1.]],[float("nan")],[[1.]],[1.])
except ValueError:pass
else:raise AssertionError("nonfinite QP admitted")
print("PASS: damped hinge Newton; unilateral bound; coupled nearly-dependent rows; monotone penalty energy; infeasible and fixed support; nonfinite rejection")
