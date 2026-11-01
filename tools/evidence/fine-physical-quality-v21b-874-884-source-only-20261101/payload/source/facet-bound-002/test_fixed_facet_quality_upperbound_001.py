from pathlib import Path
import importlib.util,sys
import numpy as np

HERE=Path(__file__).resolve();spec=importlib.util.spec_from_file_location("facet_bound_owner",HERE.parent/"fixed_facet_quality_upperbound_001.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
# A skinny, fixed triangle has a global q upper bound below the unchanged native threshold.
base=np.asarray([[0.,0.,0.],[1.,0.,0.],[1.,1.e-7,0.]],np.float64)
area,L,qbound=m.facet_upper_bound(base,[0,1,2])
assert qbound<1.e-5 and abs(area-5.e-8)<1.e-18 and abs(L-1.)<1.e-14
for x in ([0.2,0.4,1.e-8],[0.2,0.4,1.e-4],[0.2,0.4,2.]):
    p=np.vstack((base,np.asarray(x,np.float64)))
    a,b,c,d=p
    det=float(np.dot(b-a,np.cross(c-a,d-a)))
    longest=max(float(np.linalg.norm(p[i]-p[j])) for i in range(4) for j in range(i+1,4))
    assert det>0 and det/longest**3<=qbound*(1.+1.e-12),(det/longest**3,qbound)
# A well-shaped fixed triangle does not trigger an impossibility claim.
tri=np.asarray([[0.,0.,0.],[1.,0.,0.],[0.5,0.866025403784,0.]],np.float64)
_,_,good_bound=m.facet_upper_bound(tri,[0,1,2])
assert good_bound>1.e-5
print("PASS fixed-facet q upper-bound controls",{"skinny_q_upper_bound":qbound,"good_triangle_q_upper_bound":good_bound})
