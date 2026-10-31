#!/usr/bin/env python3
from pathlib import Path
import hashlib, importlib.util, json, sys
import numpy as np

D=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

adapter=load("connected_cavity_candidate_625",HERE/"connected_cavity_candidate_625.py")
owner=load("owner198_for_625_cavity_control",D/"fixed_boundary_volume_flip_198_root.py")
solver=load("solver220_for_625_cavity_control",D/"complete-surface-cavity-220-root/solve_oriented_cavity_milp_220.py")
batch=load("batch230_for_625_cavity_control",D/"batch_fixed_boundary_quality_milp_230_root.py")
exact_path=D.parent/"deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py"
sys.path.insert(0,str(exact_path.parents[1]))
exact=__import__("numilab_human.cardiac_cavity_intersections",fromlist=["_audit_pair"])

def self_check(p,t):
    faces=t[:,owner.PAT].reshape(-1,3)
    _,ix=np.unique(np.sort(faces,axis=1),axis=0,return_index=True)
    records,_=exact.float32_surface_records(p,faces[ix])
    return exact._audit_pair(records,records,same_surface=True)

def no_retained(p,all_tets,old_rows,new_tets,prior=()):
    keep=np.ones(len(all_tets),bool);keep[np.asarray(old_rows,np.int64)]=False
    return {"pass":not bool(np.any(keep)),"retained_cells":int(np.count_nonzero(keep))}

p=np.asarray(((0,0,0),(1,0,0),(0,1,0),(.2,.2,1e-7),(.2,.2,-1)),np.float32)
old=owner.orient(p,np.asarray(((0,1,2,3),(0,2,1,4)),np.int64))
mat=np.full(2,16,np.int32)
mrep=adapter.solve_bounded_fixed_vertex_milp(
    owner,p,old,mat,(0,1),coverage_fn=batch.coverage,solve_fn=solver.solve,
    seconds=2,max_points=5,max_candidate_tets=5,
    exact_replacement_self_check=self_check,
    exact_retained_intersection_check=no_retained,
)
assert mrep["accepted"],mrep
assert mrep["native_bad_before_local"]==1 and mrep["native_bad_after_local"]==0
assert np.array_equal(mrep["output_cell_to_input_cell"],np.full(len(mrep["tetrahedra"]),-1,np.int64))
assert np.array_equal(mrep["replacement_parent_cell_rows"],np.tile((0,1),len(mrep["replacement_tetrahedra"])))

new2,_=owner.propose_pair(p,*old)
t0=new2[0]
center=p[t0].astype(np.float64).mean(axis=0).astype(np.float32)
p_ext=np.vstack((p,center))
split=owner.orient(p_ext,np.column_stack((t0[owner.PAT],np.full(4,len(p),np.int64))))
candidate=np.vstack((new2[1:],split))
srep=adapter.evaluate_connected_cavity(
    owner,p,old,mat,(0,1),candidate,output_points=p_ext,new_point_rows=(len(p),),
    point_inside=lambda x: bool(np.array_equal(x.view(np.uint32),center.view(np.uint32))),
    exact_replacement_self_check=self_check,
    exact_retained_intersection_check=no_retained,
)
assert srep["accepted"],srep
assert srep["native_bad_before_local"]==1 and srep["native_bad_after_local"]==0
assert np.array_equal(srep["points_world_m"].view(np.uint32),p_ext.view(np.uint32))
assert len(srep["new_point_rows"])==1
assert np.array_equal(srep["complete_parent_union_rows"],np.asarray((0,1),np.int64))
assert np.array_equal(srep["replacement_parent_cell_rows"],np.tile((0,1),len(candidate)))

rejected={}
try:
    adapter.evaluate_connected_cavity(
        owner,p,old,mat,(0,1),candidate,output_points=p_ext,new_point_rows=(len(p),),
        exact_replacement_self_check=self_check,
        exact_retained_intersection_check=no_retained,
    )
except ValueError as e: rejected["missing_point_location"]=str(e)
else: raise AssertionError("missing exact point locator did not fail closed")
try:
    adapter.evaluate_connected_cavity(
        owner,p,old,mat,(0,1),candidate,output_points=p_ext,new_point_rows=(len(p),),
        point_inside=lambda x: True,
        exact_replacement_self_check=self_check,
        exact_retained_intersection_check=lambda *args: {"pass":False},
    )
except ValueError as e: rejected["retained_sat_failure"]=str(e)
else: raise AssertionError("retained SAT failure did not fail closed")
assert adapter.solve_bounded_fixed_vertex_milp(
    owner,p,old,mat,(0,1),coverage_fn=batch.coverage,solve_fn=solver.solve,
    seconds=.1,max_points=5,max_candidate_tets=4,
    exact_replacement_self_check=self_check,
    exact_retained_intersection_check=no_retained,
)["status"]=="REJECT_CANDIDATE_BUDGET"

source_paths={
 "625_adapter":HERE/"connected_cavity_candidate_625.py",
 "220_solver":D/"complete-surface-cavity-220-root/solve_oriented_cavity_milp_220.py",
 "230_bounded_driver":D/"batch_fixed_boundary_quality_milp_230_root.py",
 "303_connected_remesher":D/"remesh_quality_witness_cavities_303_root.py",
 "251_polygon_replay":D/"replay_current_polygon_quality_251_root.py",
 "261_lazy_ears":D/"quality_floor_lazy_ears_261_root.py",
 "198_exact_volume_owner":D/"fixed_boundary_volume_flip_198_root.py",
 "349_exact_intersection_owner":exact_path,
}
pins={name:{"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()} for name,path in source_paths.items()}
result={
 "status":"PASS_CONNECTED_CAVITY_625_SYNTHETIC_CONTROLS",
 "controls":{
  "bounded_230_220_fixed_vertex_milp":{"native_bad_before":mrep["native_bad_before_local"],"native_bad_after":mrep["native_bad_after_local"],"replacement_cells":len(mrep["replacement_tetrahedra"]),"coverage":mrep["coverage"],"solver_status":mrep["solver"]["status"]},
  "303_style_steiner_replacement":{"native_bad_before":srep["native_bad_before_local"],"native_bad_after":srep["native_bad_after_local"],"replacement_cells":len(candidate),"new_points":len(srep["new_point_rows"]),"complete_parent_union_rows":srep["complete_parent_union_rows"].tolist()},
  "fail_closed":rejected,
  "candidate_budget":"5 choose 4 admitted; 5 choose 4 with cap 4 rejected before solver",
 },
 "api":{
  "evaluate_connected_cavity":"owner, points, tetrahedra, materials, old_rows, replacement_tetrahedra; optional output_points/new_point_rows/point_inside; mandatory exact_replacement_self_check(points,new_tets) and exact_retained_intersection_check(points,parent_tets,old_rows,new_tets,prior_replacements); optional complete parent CSR.",
  "solve_bounded_fixed_vertex_milp":"owner, points, tetrahedra, materials, old_rows; caller supplies existing 230 coverage_fn and 220 solve_fn; hard limits of 36 points, 60,000 candidate tets, 50,000 cavity cells and 20 seconds then runs the same evaluator."
 },
 "thresholds_changed":False,"surface_or_point_prefix_changed":False,
 "exact_local_volume_and_oriented_chain":"existing owner.certify plus owner.chain",
 "source_context_and_full_target_checks":"external required gates",
 "physical_admission":False,"native_steps":0,
 "source_pins":pins,
}
out=HERE/"connected-cavity-controls.json"
out.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))

