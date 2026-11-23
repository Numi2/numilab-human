"""Select adjacent Float32 coordinates subject to the original tissue mass moments."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


def emit(points,base,free,W,tolerance=7.5e-11,protected=None):
    # At a binade boundary the upward and downward spacings differ. Build
    # the actual two floats that bracket each coordinate.
    out=points.astype('f4');p=points[free]
    v=out[free]
    lo=np.where(v.astype(float)>p,np.nextafter(v,np.float32(-np.inf)),v)
    hi=np.where(v.astype(float)<p,np.nextafter(v,np.float32(np.inf)),v)
    if protected is not None:
        lo[protected]=hi[protected]=v[protected]
    for axis in range(3):
        raw=v[:,axis].astype(float)
        if np.max(abs(W@(raw-base[free,axis])))<=tolerance:continue
        low=lo[:,axis].astype(float);up=hi[:,axis].astype(float)
        step=up-low
        valid=(step>0)&np.any(W!=0,axis=0)
        if not valid.any():continue
        A=W[:,valid]*step[valid]/tolerance
        constant=(W@(low-base[free,axis]))/tolerance
        active=np.any(A!=0,axis=1)
        if np.max(abs(constant[~active]),initial=0)>1.:continue
        # Find any bracketing selection. Continuous coordinates already meet
        # these same physical moments; integer variables enforce exact storage.
        res=milp(np.zeros(valid.sum()),integrality=np.ones(valid.sum()),
                 bounds=Bounds(0,1),constraints=LinearConstraint(csr_matrix(A[active]),-constant[active]-.98,-constant[active]+.98),
                 options={'time_limit':2.0,'presolve':True})
        if res.x is None:continue
        cand=low.copy();cand[valid]+=step[valid]*np.rint(res.x)
        if np.max(abs(W@(cand-base[free,axis])))<=tolerance:out[free,axis]=cand.astype('f4')
    return out.astype(float)
