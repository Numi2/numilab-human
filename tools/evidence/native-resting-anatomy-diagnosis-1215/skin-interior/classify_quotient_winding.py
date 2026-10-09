#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,importlib.util,sys,struct,gc,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009"); E=Path("/Users/n/numi-human-resting-evidence-20261005")
B=R/"skin-resting-multipose-clearance-1206"; OUT=B/"inside-vertex-diagnosis-001"
Q=OUT/"report-coordinate-quotient-stop.json"
FIT=B/"fit-attempt-002"; S=E/"native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin"
C=FIT/"attempts/attempt-0009-source.npy"; CA=FIT/"attempts/attempt-0009-target-audits.json"; F=E/"native-common-skin-multipose-forward-model-915.py"
P0=R/"native-lung1178-thumb1187-smoke-1191/native-run"; PBASE=R/"native-flat-reference-40s-1201/native-run"
AUDIT=R/"native-flat-reference-40s-1201/skin-audit-1201-terminal"
PACKS=[P0/"accepted-geometry/step-0.mrvpack",P0/"accepted-geometry/step-10000.mrvpack",PBASE/"accepted-geometry/step-20000.mrvpack"]
RECEIPTS=[p.with_suffix(".receipt.json") for p in PACKS]; TARGETS=[(51005,63),(51005,64)]; start=time.monotonic()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p): return {"path":str(p),"sha256":sha(p)}
def readj(p): return json.loads(Path(p).read_text())
quot=readj(Q)
assert quot["status"]=="stopped_after_exact_quotient_topology_or_self_audit_gate"
assert all(quot["targets"][f"{k[0]}:{k[1]}"]["exact_coordinate_quotient"]["topology"]["closed_oriented_2_manifold"] for k in TARGETS)
assert all("exact_self_intersection_audit" in quot["targets"][f"{k[0]}:{k[1]}"] for k in TARGETS)
assert quot["targets"]["51005:63"]["exact_self_intersection_audit"]["unallowed_pair_count"]==0
assert quot["targets"]["51005:64"]["exact_self_intersection_audit"]["unallowed_pair_count"]==5
sys.path.insert(0,"/Users/n/numi-human-free-apex-two-family-1178/src")
from numilab_human import common_atlas_skin_clearance as c
raw=S.read_bytes(); magic,abi,nb,nv,ni,fp,arch=struct.unpack_from("<8s5I32s",raw); assert magic==b"NHSKIN1\0" and abi==5
vo=60+36*nb; io=vo+56*nv; wo=io+4*ni
vr=np.frombuffer(raw,"<f4",14*nv,vo).reshape(nv,14); faces=np.frombuffer(raw,"<u4",ni,io).reshape(-1,3).astype(np.int64)
pos0=vr[:,:3].astype(np.float64); ref=np.unique(faces)
skin={"nb":nb,"nv":nv,"bind_u":np.frombuffer(raw,"<u4",9*nb,60).reshape(nb,9).copy(),"bind":np.frombuffer(raw,"<f4",9*nb,60).reshape(nb,9).copy(),"pos":pos0,"faces":faces,"weights":np.frombuffer(raw,"<f4",nv*nb,wo).reshape(nv,nb).astype(float)}
captures=[]; states=[]; targets=[]; initial=None; baseline_full=None; packpins=[]
for i,(pack,rp) in enumerate(zip(PACKS,RECEIPTS)):
 rec=readj(rp); step=[0,10000,20000][i]; assert rec["accepted_step"]==step and rec["physical_endpoint"]=="accepted"
 if initial is None: initial=rec
 packpos,surfs,_=c._pack_surfaces(pack,set(TARGETS)); base=int(surfs[(51007,1)]["faces"].min())
 assert np.array_equal(surfs[(51007,1)]["faces"]-base,faces)
 skinpos=np.asarray(packpos[base:base+nv],dtype="<f4").copy()
 if step==20000: baseline_full=skinpos
 captures.append(skinpos[ref].astype(np.float64))
 targets.append({k:np.asarray(packpos[surfs[k]["faces"]],dtype="<f4").astype(np.float64) for k in TARGETS})
 states.append({"step":step,"body_poses":rec["accepted_registered_body_poses"],"respiratory_motion":rec["accepted_respiratory_motion"]})
 packpins.extend([pin(pack),pin(rp)]); del packpos,surfs; gc.collect()
captures=np.asarray(captures,dtype="<f4")
spec=importlib.util.spec_from_file_location("forward915",F); fm=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fm;spec.loader.exec_module(fm)
forward,map_report,_=fm.build_forward(skin=skin,source_positions=pos0,referenced_ids=ref,captured_by_pose=captures,state_receipts=states,initial_body_poses=initial["initial_anatomical_registration"]["body_poses"],map_receipt=initial,anatomy_parameters_path=initial["skin_source_mapping"]["anatomy_parameters"]["path"],respiration_source=Path("/Users/n/numi-human-resting-resp-source-20261006"),clearance_module=c)
zero=forward(pos0.astype("<f4")); assert np.array_equal(np.asarray(zero["world_positions_by_pose"],dtype="<f4"),captures)
source_candidate=pos0.astype("<f4").copy(); source_candidate[ref]=np.load(C,allow_pickle=False)
fitres=forward(source_candidate); fit_refs=np.asarray(fitres["world_positions_by_pose"],dtype="<f4")
fit_full=np.asarray(baseline_full,dtype="<f4").copy(); fit_full[ref]=fit_refs[2]
fit_audits=readj(CA); assert len(fit_audits)==3
fit_cross={k:set(int(p[0]) for p in fit_audits[2][f"{k[0]}:{k[1]}"]["triangle_pairs"]) for k in TARGETS}
base_cross={k:set() for k in TARGETS}
for line in (AUDIT/"step-20000.crossing-witnesses.jsonl").open():
 row=json.loads(line); key=tuple(row.get("target_surface",[]))
 if key in base_cross and row.get("role")=="surface_crossing": base_cross[key].add(int(row["skin_source_face_row"]))

def winding(points,tris,batch=20):
 out=np.empty(len(points),dtype=np.float64); t=np.asarray(tris,dtype=np.float64)
 for i in range(0,len(points),batch):
  p=np.asarray(points[i:i+batch],dtype=np.float64); v=t[None,:,:,:]-p[:,None,None,:]
  a,b,d=v[:,:,0],v[:,:,1],v[:,:,2]; la=np.linalg.norm(a,axis=2); lb=np.linalg.norm(b,axis=2); ld=np.linalg.norm(d,axis=2)
  num=np.einsum("bfi,bfi->bf",a,np.cross(b,d)); ab=np.einsum("bfi,bfi->bf",a,b); bd=np.einsum("bfi,bfi->bf",b,d); da=np.einsum("bfi,bfi->bf",d,a)
  den=la*lb*ld+ab*ld+bd*la+da*lb; out[i:i+len(p)]=np.sum(2*np.arctan2(num,den),axis=1)/(4*np.pi)
 return out

def patch_metrics(ids,points):
 ids=np.unique(np.asarray(ids,dtype=np.int64))
 if not len(ids): return {"vertex_count":0,"bbox_world_m":None,"extent_mm_xyz":None,"largest_adjacency_components":[]}
 xyz=points[ids]; lo=xyz.min(0); hi=xyz.max(0)
 set_ids=set(map(int,ids)); adj={i:set() for i in set_ids}
 for tri in faces:
  a,b,d=map(int,tri)
  if a in set_ids and b in set_ids: adj[a].add(b);adj[b].add(a)
  if b in set_ids and d in set_ids: adj[b].add(d);adj[d].add(b)
  if d in set_ids and a in set_ids: adj[d].add(a);adj[a].add(d)
 unseen=set_ids; comps=[]
 while unseen:
  stack=[unseen.pop()]; comp=[]
  while stack:
   v=stack.pop();comp.append(v);nxt=adj[v]&unseen;unseen-=nxt;stack.extend(nxt)
  cp=points[comp]; clo=cp.min(0);chi=cp.max(0); comps.append({"vertex_count":len(comp),"bbox_world_m":{"min":clo.tolist(),"max":chi.tolist()},"extent_mm_xyz":((chi-clo)*1000).tolist()})
 comps.sort(key=lambda q:q["vertex_count"],reverse=True)
 return {"vertex_count":len(ids),"bbox_world_m":{"min":lo.tolist(),"max":hi.tolist()},"extent_mm_xyz":((hi-lo)*1000).tolist(),"largest_adjacency_components":comps[:10]}

results={}
for cohort,points,cross in (("baseline_native_1201",baseline_full,base_cross),("fit_iteration8_counterfactual",fit_full,fit_cross)):
 results[cohort]={}
 for k in TARGETS:
  key=f"{k[0]}:{k[1]}"; tris=targets[2][k]; lo=tris.min((0,1));hi=tris.max((0,1)); in_aabb=np.flatnonzero(np.all((points>=lo)&(points<=hi),axis=1)); w=winding(points[in_aabb],tris); aw=np.abs(w)
  inside=in_aabb[aw>.5]; ambiguous=in_aabb[(aw>=.45)&(aw<=.55)]; outside=in_aabb[aw<.45]
  incident=set()
  for fr in cross[k]: incident.update(map(int,faces[fr]))
  clean=np.asarray([v for v in inside if int(v) not in incident],dtype=np.int64)
  results[cohort][key]={"target_world_aabb_m":{"min":lo.tolist(),"max":hi.tolist()},"skin_vertices_in_target_aabb":len(in_aabb),"winding_abs_quantiles":{"min":float(aw.min()) if len(aw) else None,"p10":float(np.quantile(aw,.1)) if len(aw) else None,"median":float(np.median(aw)) if len(aw) else None,"p90":float(np.quantile(aw,.9)) if len(aw) else None,"max":float(aw.max()) if len(aw) else None},"inside_abs_winding_gt_0_5":len(inside),"outside_abs_winding_lt_0_45":len(outside),"ambiguous_abs_winding_0_45_to_0_55":len(ambiguous),"exact_crossing_skin_face_rows":len(cross[k]),"vertices_incident_to_exact_crossing_faces":len(incident),"inside_vertices_not_incident_to_exact_crossing_faces":patch_metrics(clean,points),"all_inside_vertices_patch":patch_metrics(inside,points),"inside_vertex_ids_sample":inside[:32].astype(int).tolist()}
  print(json.dumps({"cohort":cohort,"target":key,"aabb":len(in_aabb),"inside":len(inside),"clean_inside":len(clean),"ambiguous":len(ambiguous),"extent_mm":results[cohort][key]["inside_vertices_not_incident_to_exact_crossing_faces"]["extent_mm_xyz"]}),flush=True)
report={"schema":"numi.human.skin-muscle-inside-vertex-diagnostic.v1","status":"bounded_winding_occupancy_diagnostic_not_admission","inputs":[pin(Q),pin(S),pin(C),pin(CA),pin(F),pin(AUDIT/"step-20000.targets.jsonl"),pin(AUDIT/"step-20000.crossing-witnesses.jsonl"),*packpins,pin(Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/common_atlas_skin_clearance.py"))],"target_shell_gates":{k: {"bit_exact_float32_coordinate_quotient_closed_oriented":quot["targets"][k]["exact_coordinate_quotient"]["topology"]["closed_oriented_2_manifold"],"exact_self_unallowed_pairs":quot["targets"][k]["exact_self_intersection_audit"]["unallowed_pair_count"]} for k in ["51005:63","51005:64"]},"forward_replay":{"source915_map_report":map_report,"baseline_zero_delta_world_f32_bit_exact":True,"fit_iteration8_diagnostics":fitres["diagnostics"],"fit_is_offline_counterfactual_not_native":True},"results":results,"method":{"coordinate_weld":"target quotient uses bit-exact Float32 xyz only, no tolerance weld/cap; inside winding uses original Float32 triangle soup, geometrically identical to quotient","generalized_winding":"absolute solid-angle winding; >0.5 labeled inside-occupancy, <0.45 outside, [0.45,0.55] ambiguous","boundary_exclusion":"inside skin vertices not incident to any exact crossing skin face row for corresponding cohort/target","extent":"bounds and face-adjacency connected components among selected NHSKIN vertex rows"},"interpretation_limits":["51005:64 quotient is closed/oriented but has 5 exact unallowed self-intersection pairs; its winding occupancy is reported diagnostically and is not an unambiguous solid interior.","A vertex classified within a closed muscle surface does not alone prove triangle-surface intersection or muscle volume occupancy by the skin shell.","Iteration-8 positions are offline forward replay at the unchanged native baseline step-20000 state; no candidate native run was made."],"elapsed_seconds":time.monotonic()-start}
out=OUT/"report-coordinate-quotient-winding.json";out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"report":str(out),"sha256":sha(out),"elapsed_seconds":report["elapsed_seconds"]}),flush=True)
