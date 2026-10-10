#!/Users/n/numi-human-prep-venv-20261005/bin/python3.13
from __future__ import annotations
import os
for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[n]="1"
import hashlib,importlib.util,inspect,json,sys,time
from pathlib import Path
import numpy as np
B=Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218")
FIT=B/"fit-attempt-002-adapter-revision-003/fit_17_pose_source_clearance.py"
OWNER=Path("/Users/n/numi-human-conforming-composition-source-1216/src/numilab_human/common_atlas_skin_clearance.py")
CAND=B/"bilateral-source-qp-001/qp-dual-localcheck-018-root/result-001/unadmitted-compact-source.npy"
GATE=B/"local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd/candidate-verification.json"
TABLES=B/"local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd/candidate-target-pair-tables.json"
ATT4=B/"fit-attempt-002/attempts/attempt-0004-source.npy"
QP=B/"bilateral-source-qp-001/qp-dual-localcheck-018-root/result-001/local-feasibility.json"
SKIN_MAN=B/"package-preparation-002/composed-candidate/common-atlas-skin-geometry-registration.manifest.json"
OUT=B/"resume-prepare-d319-004/conditioning-vertex-30204-004"
OUT.mkdir(parents=True,exist_ok=True)
REPORT=OUT/"diagnosis.json"; ARR=OUT/"vertex-constraints.npz"; V=30204; C=.500001
class Stop(Exception): pass
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
 return h.hexdigest()
def loadmod(path,name):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
def need(x,msg):
 if not x: raise RuntimeError(msg)
def writej(p,j): p.write_text(json.dumps(j,indent=2,sort_keys=True,allow_nan=False)+"\n")
def diag(m,L):
 t=time.monotonic(); c=m.c; current_skin=m.CURRENT_SKIN; ref=np.asarray(m.REFERENCE_IDS,dtype=np.int64); source_faces=np.asarray(L["skin_faces"],dtype=np.int64)
 need(source_faces.shape[1]==3 and source_faces.max()<len(m.CURRENT_SKIN["pos"]),"full source faces differ")
 faces=np.searchsorted(ref,source_faces)
 need(np.array_equal(ref[faces],source_faces),"source-to-compact face remap differs")
 cand=np.load(CAND,allow_pickle=False); need(cand.dtype==np.dtype("<f4") and cand.shape==(len(ref),3),"candidate shape/dtype mismatch")
 full=np.asarray(m.CURRENT_SKIN["pos"],dtype="<f4").copy(); full[ref]=cand
 fw=L["combined_forward"](full); need(fw["diagnostics"].get("admissible") is True,"candidate forward rejected")
 world=np.asarray(fw["world_positions_by_pose"],dtype="<f4"); maps=np.asarray(fw["jacobians_by_pose"],dtype=np.float64)
 need(world.shape==(17,len(ref),3) and maps.shape==(17,len(ref),3,3),"forward shape mismatch")
 gate=json.loads(GATE.read_text()); tables=json.loads(TABLES.read_text())
 need(gate.get("status")=="candidate_target_and_non_target_gate_scan_complete" and all(gate.get("candidate_target_tables_cover_all_859_targets_by_pose", [])) and len(gate.get("candidate_target_tables_cover_all_859_targets_by_pose", [])) == 17,"candidate full17 audit not complete")
 need(len(gate["candidate_non_target_gate_checks"])==17 and len(tables["poses"])==17,"pose table count mismatch")
 _,nonocular,_,_=c._load_target_inventory(m.INV); nonoc={f"{int(a)}:{int(b)}" for a,b in nonocular}
 for r in gate["candidate_non_target_gate_checks"]:
  need(all(int(x.get("inside_vertex_count",-1))==0 for x in r["closed_target_inside"].values()),"nonzero closed-target interior seeds")
 af=set(); per_pose=[]
 for p,tab in enumerate(tables["poses"]):
  s=set()
  for key in nonoc:
   row=tab.get(key); need(isinstance(row,dict),f"missing target {p}:{key}")
   s.update(int(pair[0]) for pair in row.get("triangle_pairs",[]))
  per_pose.append(len(s)); af.update(s)
 af=np.asarray(sorted(af),dtype=np.int64); need(len(af)>0 and af.min()>=0 and af.max()<len(faces),"active face union invalid")
 ac=faces[af]; il=np.flatnonzero(np.any(ac==V,axis=1)); incident=af[il]; iv=faces[incident]
 need(len(incident)>0,"no incident active faces for failed vertex")
 tri=world[:,iv,:].astype(np.float64); av=np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0]); al=np.linalg.norm(av,axis=2)
 need(np.isfinite(al).all() and (al>0).all(),"degenerate incident face")
 normals=av/al[:,:,None]; j=maps[:,V,:,:]; sv=np.linalg.svd(j,compute_uv=False); cond=sv[:,0]/sv[:,-1]
 need(np.isfinite(sv).all() and (sv[:,-1]>1e-8).all(),"singular J")
 pnorm=np.stack([c._area_weighted_vertex_normals(world[p],faces) for p in range(17)])
 smooth=np.stack([c._smooth_vertex_directions(pnorm[p],faces)[V] for p in range(17)])
 sd,_,_,sdrep=c._shared_source_directions_from_pose_normals(maps[:,V:V+1,:,:],smooth[:,None,:]); preferred=sd[0]
 mapped=np.einsum("pij,j->pi",j,preferred); ml=np.linalg.norm(mapped,axis=1); mu=mapped/ml[:,None]
 vm=np.repeat(j[:,None,:,:],len(incident),axis=1).reshape(-1,3,3); vn=normals.reshape(-1,3)
 def cons(d):
  w=np.einsum("kij,j->ki",vm,d); return np.einsum("ki,ki->k",vn,w)-C*np.linalg.norm(w,axis=1)
 def jac(d):
  w=np.einsum("kij,j->ki",vm,d); wl=np.maximum(np.linalg.norm(w,axis=1),1e-15); return np.einsum("ki,kij->kj",vn-C*w/wl[:,None],vm)
 from scipy.optimize import minimize
 result=minimize(lambda d:.5*float(np.dot(d-preferred,d-preferred)),preferred,jac=lambda d:d-preferred,constraints={"type":"ineq","fun":cons,"jac":jac},method="SLSQP",options={"maxiter":64,"ftol":1e-12})
 residual=cons(result.x); threshold=2*C*C-1
 pairrows=[]
 for p in range(17):
  if len(incident)>1:
   dots=normals[p]@normals[p].T; i,k=np.triu_indices(len(incident),1); z=int(np.argmin(dots[i,k])); a,b=int(i[z]),int(k[z]); md=float(dots[a,b])
   pairrows.append({"pose_index":p,"accepted_step":int(L["poses"][p]["step"]),"min_normal_dot":md,"face_rows":[int(incident[a]),int(incident[b])],"max_pairwise_min_projection":float(np.sqrt(max(0,(1+md)/2))),"pair_proves_empty_cone":md<threshold})
  else: pairrows.append({"pose_index":p,"accepted_step":int(L["poses"][p]["step"]),"min_normal_dot":1.,"face_rows":[],"max_pairwise_min_projection":1.,"pair_proves_empty_cone":False})
 hashes=[hashlib.sha256(np.ascontiguousarray(world[p],dtype="<f4").tobytes()).hexdigest() for p in range(17)]
 need(hashes==[r["candidate_world_f32_sha256"] for r in gate["candidate_non_target_gate_checks"]],"candidate forward differs from full17 gate")
 srcid=int(ref[V]); base=np.load(ATT4,allow_pickle=False); need(base.shape==cand.shape,"attempt4 compact shape mismatch")
 delta=cand[V].astype(float)-base[V].astype(float); qpj=json.loads(QP.read_text()); varids={int(x) for x in qpj.get("variable_compact_ids",[])}
 sm=json.loads(SKIN_MAN.read_text()); checks=sm["common_atlas_binding_runtime_rest_validation"]["checks"]; need(len(checks)==current_skin["nb"],"skin binding label count mismatch")
 w=current_skin["weights"][srcid]; binding_rows=[{"binding_index":i,"body_name":checks[i]["body_name"],"core_body_index":checks[i]["core_body_index"],"weight_f32":float(w[i]),"binding_record_f32":current_skin["bind"][i].astype(float).tolist()} for i in range(len(w)) if float(w[i])>0.0]
 pins=list(L["tracked"]); known={x["path"] for x in pins}
 for p in (FIT,OWNER,CAND,GATE,TABLES,ATT4,QP,SKIN_MAN,B/"resume-prepare-d319-004/resume-execution.json",B/"fit-attempt-003-resume-d319-001/input-postcheck.json"):
  s=str(p.resolve())
  if s not in known: pins.append(m.pin(p)); known.add(s)
 np.savez_compressed(ARR,
  vertex_compact_id=np.asarray(V,dtype="<i8"),vertex_full_source_id=np.asarray(srcid,dtype="<i8"),
  incident_skin_face_rows=incident.astype("<i8"),incident_face_compact_vertex_ids=iv.astype("<i8"),
  unit_outward_face_normals_by_pose=normals.astype("<f8"),jacobian_by_pose=j.astype("<f8"),
  jacobian_by_constraint=vm.astype("<f8"),unit_normal_by_constraint=vn.astype("<f8"),
  jacobian_singular_values_by_pose=sv.astype("<f8"),preferred_source_direction=preferred.astype("<f8"),
  slsqp_result_x=np.asarray(result.x,dtype="<f8"),slsqp_constraint_residuals=residual.astype("<f8"),
  preferred_constraint_residuals=cons(preferred).astype("<f8"))
 report={"schema":"numi.human.vertex-active-face-direction-conditioner-diagnostic.v1","status":"completed_single_vertex_reproduction",
  "qualification":"Read-only single-vertex conditioning reproduction. It makes no asset/native change and does not relax the owner's 0.5 admission bound.",
  "owner":{"path":str(OWNER),"sha256":sha(OWNER)},"fit_adapter":{"path":str(FIT),"sha256":sha(FIT)},
  "resume_execution":{"path":str(B/"resume-prepare-d319-004/resume-execution.json"),"sha256":sha(B/"resume-prepare-d319-004/resume-execution.json")},
  "input_postcheck":{"path":str(B/"fit-attempt-003-resume-d319-001/input-postcheck.json"),"sha256":sha(B/"fit-attempt-003-resume-d319-001/input-postcheck.json")},
  "candidate_gate":{"path":str(GATE),"sha256":sha(GATE)},"candidate_tables":{"path":str(TABLES),"sha256":sha(TABLES)},
  "candidate_source":{"path":str(CAND),"sha256":sha(CAND)},"candidate_world_f32_sha256_by_pose":hashes,
  "compact_vertex_id":V,"full_source_vertex_id":srcid,"source_vertex_30204_compact_id_if_present":int(np.searchsorted(ref,30204)) if 30204 in ref else None,
  "attempt4_source_position_m":base[V].astype(float).tolist(),"d319_source_position_m":cand[V].astype(float).tolist(),
  "d319_minus_attempt4_delta_m":delta.tolist(),"d319_minus_attempt4_delta_norm_m":float(np.linalg.norm(delta)),
  "current_source_position_m":current_skin["pos"][srcid].astype(float).tolist(),
  "anatomical_binding_weights":binding_rows,
  "skin_registration_manifest":{"path":str(SKIN_MAN),"sha256":sha(SKIN_MAN)},
  "qp_variable_support_contains_vertex":V in varids,"qp_variable_support_count":len(varids),
  "active_face_union_count":int(len(af)),"incident_active_face_count":int(len(incident)),
  "incident_active_face_rows":incident.tolist(),"active_faces_per_pose_count":per_pose,
  "active_set":"Union of nonocular target-pair skin face rows over all17 candidate tables; owner applies the union to each pose. Both closed targets have zero inside vertices at all17, so none added from interior seeds.",
  "candidate_world_position_m_by_pose":world[:,V,:].astype(float).tolist(),"jacobian_by_pose":j.tolist(),
  "jacobian_singular_values_by_pose":sv.tolist(),"jacobian_condition_number_by_pose":cond.tolist(),"jacobian_determinant_by_pose":np.linalg.det(j).tolist(),
  "preferred_source_direction":preferred.tolist(),"preferred_world_alignment_min_by_pose":np.min(np.einsum("pi,pni->pn",mu,normals),axis=1).tolist(),
  "preferred_direction_report":sdrep,"selection_bound":C,"pairwise_sufficient_empty_cone_dot_threshold":threshold,
  "minimum_normal_pair_by_pose":pairrows,"any_pose_pair_proves_cone_empty":any(x["pair_proves_empty_cone"] for x in pairrows),
  "owner_slsqp":{"method":"SLSQP","maxiter":64,"ftol":1e-12,"success":bool(result.success),"status":int(result.status),"message":str(result.message),
   "iterations":int(getattr(result,"nit",-1)),"objective":float(result.fun),"result_x":np.asarray(result.x).tolist(),"result_x_norm":float(np.linalg.norm(result.x)),
   "constraint_count":int(len(residual)),"minimum_residual":float(residual.min()),"maximum_residual":float(residual.max()),"failed_constraints":int(np.count_nonzero(residual<0))},
  "constraint_arrays":{"path":str(ARR),"keys":["incident_skin_face_rows","incident_face_compact_vertex_ids","unit_outward_face_normals_by_pose","jacobian_by_pose","jacobian_by_constraint","unit_normal_by_constraint","preferred_source_direction","slsqp_result_x","slsqp_constraint_residuals"]},
  "input_pins":pins,"input_pin_count":len(pins),"elapsed_wall_seconds":time.monotonic()-t,
 }
 writej(REPORT,report)
 before={x["path"]:x["sha256"] for x in pins}; after={p:sha(p) for p in before}
 need(after==before,"pinned inputs changed during diagnostic")
 report["inputs_unchanged"]=True; report["input_hashes_after"]=[{"path":p,"sha256":v} for p,v in sorted(after.items())]
 writej(REPORT,report)
 return report
mod=loadmod(FIT,"fit17_conditioner_diagnostic")
src,first=inspect.getsourcelines(mod.main)
line=next(first+i for i,x in enumerate(src) if x.strip()=="if args.prepare_check:")
seen=False
def tracer(frame,event,arg):
 global seen
 if not seen and event=="line" and frame.f_code is mod.main.__code__ and frame.f_lineno==line:
  seen=True; sys.settrace(None); r=diag(mod,frame.f_locals)
  print(json.dumps({"status":r["status"],"report":str(REPORT),"report_sha256":sha(REPORT),"arrays":str(ARR),"arrays_sha256":sha(ARR),"incident_active_faces":r["incident_active_face_count"],"any_pose_pair_proves_cone_empty":r["any_pose_pair_proves_cone_empty"],"slsqp":r["owner_slsqp"]},sort_keys=True),flush=True)
  raise Stop()
 return tracer
sys.argv=[str(FIT),"--prepare-check"]; sys.settrace(tracer)
try: mod.main()
except Stop: pass
finally: sys.settrace(None)
if not seen: raise RuntimeError("preparation interception did not fire")
