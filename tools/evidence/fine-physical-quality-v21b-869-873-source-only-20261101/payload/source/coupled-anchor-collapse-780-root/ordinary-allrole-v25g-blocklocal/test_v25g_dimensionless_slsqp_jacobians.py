from pathlib import Path
import importlib.util, sys, time
import numpy as np
from scipy import optimize

HERE=Path(__file__).resolve()
spec=importlib.util.spec_from_file_location("v25g_jacobian_test",HERE.parent/"repair_coupled_quality_points_722_v25g_ordinary_blocklocal.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
# The moving vertex makes one dynamic edge uniquely longest. The cell is a thin,
# positive tetrahedron, so this exercises both determinant and longest-edge math.
p=np.asarray([[0.,0.,0.],[1.3,0.,0.],[0.1,1.7,0.],[9.,6.,1.e-3]],np.float32)
t=np.asarray([[0,1,2,3]],np.int64)
d,q=m.Q._M.measurements(p,t)
assert np.all(d>0) and not bool(np.asarray(m.Q._M.passes(d,q))[0])
good=np.asarray([False])
original=optimize.minimize
checked={}
def maxerr(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    return float(np.max(np.abs(a-b)/np.maximum(1.,np.maximum(np.abs(a),np.abs(b)))))
def checking_minimize(fun,x0,*args,**kwargs):
    jac=kwargs["jac"];constraint=kwargs["constraints"][0]
    z=np.asarray([0.13,-0.09,0.07,0.31,0.28,0.30],dtype=np.float64)
    h=2.e-6
    numeric=np.empty(6,dtype=np.float64)
    for j in range(6):
        zp=z.copy();zm=z.copy();zp[j]+=h;zm[j]-=h
        numeric[j]=(fun(zp)-fun(zm))/(2*h)
    checked["objective_relerr"]=maxerr(jac(z),numeric)
    cj=constraint["jac"](z);base=np.asarray(constraint["fun"](z),dtype=np.float64)
    numeric_c=np.empty_like(cj)
    for j in range(6):
        zp=z.copy();zm=z.copy();zp[j]+=h;zm[j]-=h
        numeric_c[:,j]=(np.asarray(constraint["fun"](zp))-np.asarray(constraint["fun"](zm)))/(2*h)
    checked["constraint_relerr"]=maxerr(cj,numeric_c)
    checked["constraint_shape"]=list(cj.shape)
    assert checked["objective_relerr"]<2.e-4,checked
    assert checked["constraint_relerr"]<2.e-4,checked
    return original(fun,x0,*args,**kwargs)
optimize.minimize=checking_minimize
try:
    candidate,diag=m._single_node_quality_candidate(p,t,3,np.asarray([0]),5.e-4,
        good,good,None,time.monotonic()+2.)
finally:
    optimize.minimize=original
assert checked["constraint_shape"][1]==6 and checked["constraint_shape"][0]>0,checked
assert candidate is not None,diag
nd,nq=m.Q._M.measurements(candidate,t)
assert np.all(nd>0) and np.isfinite(nq).all()
assert float(np.abs(candidate[3].astype(np.float64)-p[3].astype(np.float64)).sum())<=5.e-4
assert diag["variable_parameterization"]=="normalized_delta_over_cap"
assert diag["constraint_row_scaling"]=="L1_over_cap_and_det_over_per_cell_native_scale"
print("PASS dimensionless one-node objective/constraint Jacobians and final exact/F32 gates",checked,diag)
