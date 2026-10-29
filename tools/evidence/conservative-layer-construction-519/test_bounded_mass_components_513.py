from pathlib import Path
import importlib.util,numpy as np,json
p=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location("graph510",p/"audit_bounded_mass_components_513.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
pc=np.array([[0.,0.,0.],[1.,0.,0.]]);pk=np.array([[15,1,1],[15,1,1]]);pm=np.array([1.,1.]);cv=np.array([.0005,.0015])
r,e=m.components(pc,pm,pk,pc,cv,pk,.01)
assert len(r)==2 and e==2 and sum(not x["component_capacity_feasible"] for x in r)==2
r,e=m.components(pc,pm,pk,pc,cv,pk,2.)
assert len(r)==1 and e==4 and r[0]["component_capacity_feasible"]
r,e=m.components(pc,pm,pk,pc+3,cv,pk,.01)
assert len(r)==4 and e==0 and sum(x["isolated_parent"] for x in r)==2 and sum(x["isolated_child"] for x in r)==2
r,e=m.components(pc[:1],pm[:1],pk[:1],np.array([[.01,0.,0.]]),np.array([.001]),pk[:1],.01)
assert len(r)==1 and e==1 and r[0]["component_capacity_feasible"]
try:m.components(pc,pm,pk,pc,cv*0,pk,.01)
except ValueError:pass
else:raise AssertionError("zero-volume child accepted")
for bad in [pk.astype(float),np.array([[15,-1,1],[15,1,1]])]:
 try:m.components(pc,pm,bad,pc,cv,pk,.01)
 except ValueError:pass
 else:raise AssertionError("untyped or invalid anatomical IDs accepted")
print(json.dumps({"status":"passed","checks":["aggregate_pass_disconnected_components_fail","connected_capacity_control","isolated_source_and_child","exact_radius_boundary","zero_volume_rejected","typed_nonnegative_anatomical_ids"]}))
