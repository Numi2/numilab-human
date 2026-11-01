import importlib.util,sys,json
from pathlib import Path
import numpy as np
script=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/quality-endpoint-collapse-001/fixed_boundary_cavity_884b_root.py")
spec=importlib.util.spec_from_file_location("fixed884b",script);m=importlib.util.module_from_spec(spec);sys.modules["fixed884b"]=m;spec.loader.exec_module(m)
root=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/quality-endpoint-collapse-001/run-884b-bipyramid-fixture-003")
if root.exists():raise RuntimeError("fixture output exists")
inp=root/"inputs";inp.mkdir(parents=True)
base=m.HERE/"run-883e-synthetic-fixture-001"/"inputs";st=json.loads((base/"state.json").read_text())
p=np.asarray([[0,0,200],[0,0,-200],[1,0,0],[-.5,.8660254,0],[-.5,-.8660254,0]],np.float32)
raw=np.asarray([[0,1,2,3],[0,1,3,4],[0,1,4,2]],np.int64);t=m.M.orient(p,raw);det,q=m.M.measurements(p,t)
if np.any(det<=0) or np.count_nonzero(m.M.passes(det,q))!=0:raise AssertionError("fixture input must be positive with three native failures")
b=np.asarray(m.T.m.boundary(t),np.int64)
arr={"points_world_m":p,"points_world_m_f32":p.copy(),"tetrahedra":t,"material_indices":np.full(3,16,np.int32),"boundary_faces":b,
"desired_boundary_faces":b.copy(),"mapped_current_boundary_faces":b.copy(),"proposed_current_desired_boundary_faces":b.copy(),
"retained_input_cell_rows":np.arange(3,dtype=np.int64),"output_cell_to_input_cell_rows":np.arange(3,dtype=np.int64),"output_cell_to_input_cell":np.arange(3,dtype=np.int64),
"input_cell_to_output_cell_rows":np.arange(3,dtype=np.int64),"input_cell_to_output_cell":np.arange(3,dtype=np.int64),
"input_node_to_output_node_rows":np.arange(5,dtype=np.int64),"input_node_to_output_node":np.arange(5,dtype=np.int64),
"output_node_to_input_node_rows":np.arange(5,dtype=np.int64),"output_node_to_input_node":np.arange(5,dtype=np.int64)}
vp=inp/"fat-bipyramid.npz";np.savez_compressed(vp,**arr);st["Fat_current"]={"path":str(vp),"sha256":m.sha(vp),"cells":3,"nodes":5,"native_quality_failures":3}
sp=inp/"state.json";m.E.write_json(sp,st);out=root/"proposal";r=m.run(sp,"Fat",(0,1),out,fixture=True,seconds=3.)
if r["quality"]["bad_before"]!=3 or r["quality"]["bad_after"]!=0:raise AssertionError("expected 3->2 native quality gain")
receipt={"schema":"h706.fixed-boundary-cavity-fixture.v1","status":"PASS","producer":m.ref(script),"input_state":m.ref(sp),"input_volume":m.ref(vp),
"proposal_report":m.ref(out/"report.json"),"candidate_volume":m.ref(out/"fat-fixed-boundary-cavity-volume-proposal.npz"),
"independent_replay":r["independent_replay"],"edge":[0,1],"old_cells":3,"new_cells":2,"old_bad":3,"new_bad":0,
"old_q":r["quality"]["q_before"],"new_q":r["quality"]["q_after"],"exact_det6_delta":"0","oriented_boundary_fixed":True,
"all_volume_points_fixed":True,"source_surfaces_unchanged":True,"all185":"NOT_RUN","mass_mapping":"NOT_RUN","physical_admission":False,"native_steps":0}
rp=root/"fixture-receipt.json";m.E.write_json(rp,receipt)
print(json.dumps({"status":"PASS","receipt":str(rp),"receipt_sha256":m.sha(rp),"producer_sha256":m.sha(script),
"quality":r["quality"],"independent":r["independent_replay"]},indent=2))
