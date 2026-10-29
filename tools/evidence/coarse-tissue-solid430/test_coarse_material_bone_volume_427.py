import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).with_name("audit_coarse_material_bone_volume_427.py")
spec=importlib.util.spec_from_file_location("coarse_material_audit",p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=np.array([[0.,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]])
t=np.array([[0,1,2,3],[0,2,1,4]])
assert np.all(m.volumes(p,t)>0)
f=m.exterior(t);assert f.shape==(6,3)
s=m.solid(p,f);assert s.status()==m.mf.Error.NoError
assert abs(s.volume()-1/3)<1e-14
upper=m.solid(p[:4],m.exterior(t[:1]));assert abs((s^upper).volume()-1/6)<1e-14
outside=m.solid(p[:4]+[3,0,0],m.exterior(t[:1]));assert (s^outside).is_empty()
try:m.exterior(np.vstack([t,t[:1],t[:1]]))
except ValueError:pass
else:raise AssertionError("nonmanifold incidence admitted")
print("PASS: outward tet boundary; shared-face cancellation; exact positive/empty intersections; nonmanifold incidence rejection")
