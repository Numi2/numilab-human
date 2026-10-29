from pathlib import Path
import importlib.util,json
import numpy as np
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("capacity",P/"audit_regional_mass_capacity_464.py");m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=np.array([[.1,.2,.3],[1.3,.1,.2],[.2,1.4,.1],[.1,.4,1.5],[1.2,1.1,1.4]])
t=np.array([[0,1,2,3],[1,4,2,3]])
v,g=m.tet_volumes_gradients(p,t)
assert np.all(v>0)
d=np.random.default_rng(464).normal(size=p.shape)
eps=1e-6
fd=(m.tet_volumes_gradients(p+eps*d,t)[0]-m.tet_volumes_gradients(p-eps*d,t)[0])/(2*eps)
ad=np.einsum("tij,tij->t",g,d[t])
assert np.allclose(fd,ad,rtol=1e-9,atol=1e-10),(fd,ad)
assert np.max(np.abs(g.sum(axis=1)))<1e-15
keys=np.array([[15,1,2],[16,3,4]])
supply={(15,1,2):float(v[0]*1000),(16,3,4):float(v[1]*900)}
rows,_,gr=m.capacity_rows(p,t,keys,supply)
assert all(r["mass_capacity_feasible"] for r in rows)
bad=dict(supply);bad[(16,3,4)]=float(v[1]*1000)
rows2,_,_=m.capacity_rows(p,t,keys,bad)
assert not rows2[1]["mass_capacity_feasible"]
assert rows2[1]["volume_deficit_to_nearest_bound_m3"]>0
empty,_,_=m.capacity_rows(p,t,keys,{**supply,(16,99,99):1.})
assert not empty[-1]["mass_capacity_feasible"] and empty[-1]["candidate_cells"]==0
print(json.dumps({"status":"passed","checks":["analytic directional volume derivative","translation cancellation","both fixed material density bounds","missing child group capacity failure","signed nearest-bound volume correction"],"max_directional_error":float(np.max(np.abs(fd-ad)))}))
