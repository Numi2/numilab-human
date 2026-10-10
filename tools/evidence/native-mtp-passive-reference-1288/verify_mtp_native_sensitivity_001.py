from pathlib import Path
import sys,json,hashlib,csv,importlib.util
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
D=Path(sys.argv[1]).resolve();assert D.parent==A
N=D/"baseline/native-run";O=D/"native-anatomy-verification-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,Path(ca.__file__),Path(ci.__file__),h.INV,D/"baseline/execution.json",D/"baseline/run-declaration.json",N/"run-metadata.json",N/"resting-coupled.csv",N/"resting-com-q-integration.csv"]}
declaration=json.load(open(D/"baseline/run-declaration.json"));execution=json.load(open(D/"baseline/execution.json"));meta=json.load(open(N/"run-metadata.json"))
assert execution["returncode"]==0 and not execution["changed_inputs"] and execution["native_argv_matches_prepared_cli_preview"] and execution["native_environment_matches_prepared_cli_preview"]
assert meta["loaded_metal_runtime"]["verified"]
K=declaration["mtp_passive_stiffness_nm_per_rad"]
out={"scope":"Bounded native MTP anatomical-mechanics sensitivity; exact surface self checks, state-owned physiological trace and unassisted contact. Not full-body, five-minute, or clinical qualification.","pins":pins,"stiffness_nm_per_rad":K,"poses":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
trace=list(csv.DictReader(open(N/"resting-coupled.csv")))
q=list(csv.DictReader(open(N/"resting-com-q-integration.csv")))
qs=np.array([[float(v) for v in r["q_accepted_f32_semicolon"].split(";")] for r in q])
imp=np.array([[float(v) for v in r["gpu_stand_source_limit_impulses_f32_by_local_v_index_semicolon"].split(";")] for r in q])
assert len(q)==1000 and [int(r["accepted_step"]) for r in q]==list(range(7001,8001)) and np.isfinite(qs).all()
out["mtp_window"]={"first_step":7001,"last_step":8000,"q_indices":[111,125],"v_indices":[110,124],"angle_min_rad":qs[:,[111,125]].min(0).tolist(),"angle_max_rad":qs[:,[111,125]].max(0).tolist(),"angle_mean_rad":qs[:,[111,125]].mean(0).tolist(),"limit_impulse_abs_max":np.abs(imp[:,[110,124]]).max(0).tolist(),"limit_nonzero_steps":np.count_nonzero(imp[:,[110,124]],axis=0).tolist(),"restoring_torque_at_endpoint_nm":(-K*qs[-1,[111,125]]).tolist()}
assert all(float(r["root_assistance_n"])==float(r["root_assistance_nm"])==0 for r in trace)
out["trace"]={"rows":len(trace),"final_step":int(trace[-1]["step"]),"final_time_s":float(trace[-1]["time_s"]),"all_finite":all(np.isfinite(float(v)) for r in trace for v in r.values()),"maximum_contact_penetration_m":max(float(r["peak_penetration_m"]) for r in trace),"maximum_blood_error_ml":max(abs(float(r["blood_error_ml"])) for r in trace),"final":trace[-1],"root_assistance_zero":True}
B=A/"twenty-six-surface-native-composition-001/baseline/native-run"
if K==0:
 bt=list(csv.DictReader(open(B/"resting-coupled.csv")));pins[str(B/"resting-coupled.csv")]=sha(B/"resting-coupled.csv")
 assert len(trace)==len(bt)==1000
 differences=[{"step":int(b["step"]),"columns":[k for k in b if b[k]!=n[k]]} for b,n in zip(bt,trace) if b!=n]
 allowed={"min_contact_gap_m","peak_penetration_m","pre_projection_contact_residual_m_s","pre_projection_limit_residual_generalized_s","pre_projection_equality_residual_generalized_s","post_projection_contact_residual_m_s","post_projection_limit_residual_generalized_s","post_projection_equality_residual_generalized_s","equality_position_projection_max_generalized","equality_velocity_projection_max_generalized_s"}
 assert all(d["step"]>=7001 and set(d["columns"])<=allowed for d in differences),differences[:3]
 out["zero_control"]={"all_physiology_and_other_trace_columns_exact":True,"observer_window_diagnostic_difference_rows":len(differences),"differing_columns":sorted(set(k for d in differences for k in d["columns"])),"explanation":"Final1000steps use existing individual-step accepted-Q observation, changing scope of residual/geometry diagnostics; all other trace columns stay exact. Actual accepted body poses and rendered triangles checked independently below."}
keys,*_=ca._load_target_inventory(h.INV)
for step in (0,8000):
 pack=N/f"accepted-geometry/step-{step}.mrvpack";rp=pack.with_suffix(".receipt.json")
 pins[str(pack)]=sha(pack);pins[str(rp)]=sha(rp)
 rc=json.load(open(rp));assert rc["pack_file_sha256"]==pins[str(pack)] and rc["physical_endpoint"]=="accepted"
 p,s,_=ca._pack_surfaces(pack,keys)
 pose={"step":step,"surface_count":len(s),"foot_muscles":[]};out["poses"].append(pose)
 if K==0:
  bp=B/f"accepted-geometry/step-{step}.mrvpack";brp=bp.with_suffix(".receipt.json");pins[str(bp)]=sha(bp);pins[str(brp)]=sha(brp);bc=json.load(open(brp))
  assert rc["accepted_body_poses"]==bc["accepted_body_poses"]
  b,bs,_=ca._pack_surfaces(bp,keys);assert set(s)==set(bs)
  assert all(np.array_equal(p[np.asarray(sf["faces"],int)],b[np.asarray(bs[k]["faces"],int)]) for k,sf in s.items())
  pose["all157_accepted_body_poses_exact_baseline"]=True;pose["all860_referenced_surface_triangles_exact_baseline"]=True
 for sid in range(21,29):
  f=np.asarray(s[(51005,sid)]["faces"],int);used=np.unique(f);v=p[used];f=np.searchsorted(used,f)
  u,iv=np.unique(v,axis=0,return_inverse=True);f=iv[f]
  er,deg=h.exact_rows(u,f,ci);assert not deg
  a=ci._audit_pair(er,er,same_surface=True);top=analyze_topology(u.tolist(),f.tolist())
  pose["foot_muscles"].append({"stable_id":sid,"self_count":a["count"],"self_pairs":a["triangle_pairs"],"closed_oriented_manifold_candidate":top["closed_oriented_manifold_candidate"]})
 save();print(json.dumps({"step":step,"foot_self":{x["stable_id"]:x["self_count"] for x in pose["foot_muscles"]}}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save();assert out["inputs_unchanged"]
print(json.dumps({"output":str(O),"mtp_window":out["mtp_window"],"trace_rows":len(trace),"wall_seconds":execution["wall_seconds"]}),flush=True)

