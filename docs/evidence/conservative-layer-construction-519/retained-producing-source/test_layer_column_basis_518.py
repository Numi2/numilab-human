from pathlib import Path
import csv,json,importlib.util
import numpy as np
P=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location("cap518",P/"audit_regional_mass_capacity_464.py");cap=importlib.util.module_from_spec(s);s.loader.exec_module(cap)
BASE=P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk"
with np.load(BASE/"candidate.npz") as z:x=z["points_reference_world_m_f64"].astype(np.float32).astype(float);t=z["volume_tetrahedra"];pins=z["pinned_volume_node_rows"]
n=len(x)//3;assert len(x)==3*n
with (BASE/"coarse-node-to-fine-surface.csv").open() as f:rows=list(csv.DictReader(f))
assert len(rows)==len(x)
for layer,role in enumerate(["O","I","D"]):
 for i in range(n):
  r=rows[layer*n+i];assert int(r["volume_node_row"])==layer*n+i and r["role"]==role and int(r["role_row"])==i
rng=np.random.default_rng(518);q=rng.normal(size=(n,3));q/=np.linalg.norm(q,axis=1)[:,None];q[np.unique(pins%n)]=0
d=q[np.arange(len(x))%n]
assert np.array_equal(d[pins],np.zeros((len(pins),3)))
step=1e-7
v,g=cap.tet_volumes_gradients(x,t);vp=cap.tet_volumes_gradients(x+step*d,t)[0];vm=cap.tet_volumes_gradients(x-step*d,t)[0]
fd=(vp-vm)/(2*step);analytic=np.einsum("ijk,ijk->i",g,d[t])
scale=np.linalg.norm(g,axis=2).sum(axis=1)
err=np.abs(fd-analytic)/np.maximum(scale,1e-30)
assert err.max()<1e-7,float(err.max())
after=x+step*d
assert np.allclose(after[n:2*n]-after[:n],x[n:2*n]-x[:n],atol=2e-16,rtol=0)
assert np.allclose(after[2*n:]-after[:n],x[2*n:]-x[:n],atol=2e-16,rtol=0)
source=P/"connected-layer-column-fit-514/run-001/candidate.npz"
with np.load(source) as z:
 y=z["points_reference_world_m_f64"].astype(np.float32).astype(float);ct=z["volume_tetrahedra"];cp=z["child_parent_tet_row"]
 assert np.array_equal(ct,t);assert np.isfinite(y).all();assert np.array_equal(y[pins],x[pins])
 material=z["material_index"]
rows=list(csv.DictReader((P/"native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv.tets.csv").open()))
pm=np.array([float(r["parent_mass_kg"]) for r in rows])[cp];cy=cap.tet_volumes_gradients(y,t)[0];rho=pm/cy
for k,limits in cap.DENSITY_BOUNDS.items():
 selected=material==k;assert np.all(rho[selected]>=limits[0]) and np.all(rho[selected]<=limits[1])
r={"status":"passed","column_count":n,"runtime_node_count_unchanged":len(x),"maximum_reduced_gradient_error":float(err.max()),"source_layer_and_role_order":True,"layer_offsets_preserved_in_authoring_basis":True,"all_pins_unchanged":True,"per_original_cell_density_bounds_passed":True,"actual_candidate_sha256":cap.sha(source),"script_sha256":cap.sha(__file__),"physical_admission":False}
out=P/"connected-layer-column-fit-514/run-001/column-basis-tests-518.json";out.write_text(json.dumps(r,indent=2)+"\n");print(json.dumps(r))
