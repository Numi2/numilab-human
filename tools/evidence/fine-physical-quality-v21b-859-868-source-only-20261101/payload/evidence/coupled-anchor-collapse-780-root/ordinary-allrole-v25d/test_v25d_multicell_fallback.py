from pathlib import Path
import importlib.util, sys, time
import numpy as np

HERE=Path(__file__).resolve()
spec=importlib.util.spec_from_file_location("v25d_blocklocal",HERE.parent/"repair_coupled_quality_points_722_v25d_inner_back_remainder_blocklocal.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
p=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,2.e-5]],np.float32)
# Two incident local rows force the (N,6) edge lookup path that failed in v25c.
t=np.asarray([[0,1,2,3],[0,1,2,3]],np.int64)
d,q=m.Q._M.measurements(p,t);g=np.asarray(m.Q._M.passes(d,q),bool)
candidate,diag=m._single_node_quality_candidate(p,t,3,np.asarray([0,1]),5.e-5,g,g,None,time.monotonic()+2.)
assert candidate is not None,diag
nd,nq=m.Q._M.measurements(candidate,t)
assert np.all(nd>0) and np.all(m.Q._M.passes(nd,nq)),diag
assert np.array_equal(candidate[[0,1,2]].view(np.uint32),p[[0,1,2]].view(np.uint32))
assert float(np.abs(candidate[3].astype(np.float64)-p[3].astype(np.float64)).sum())<=5.e-5
print("PASS multi-cell single-node fallback index and exact/native gates",diag)
