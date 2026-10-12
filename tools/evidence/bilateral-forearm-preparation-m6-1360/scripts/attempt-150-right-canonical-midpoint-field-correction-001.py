#!/usr/bin/env python3
import ast, copy, hashlib, importlib.util, json, math, pickle, sys
from pathlib import Path
import numpy as np
ROOT=Path("/Users/n/numi-human-fds132-m6-1350")
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
CAND142=ROOT/"attempt-142-right-current8000-witness-edge-midpoint-refinement-001/result-001/candidate.pkl"
CAND122=ROOT/"attempt-122-right-cap-boundary-directed-contractions-001/result-001/candidate.pkl"
ATTEMPT142=ROOT/"attempt-142-right-current8000-witness-edge-midpoint-refinement-001.py"
CHECK142=ROOT/"attempt-142-right-current8000-witness-edge-midpoint-refinement-001/result-001/construction-checkpoint.json"
DIAG148=ROOT/"attempt-148-candidate142-route-field-convention-diagnostic-001/report.json"
FIELD=ROOT/"attempt-095-right-registered-route-field-001/result-001/right-route-anchor-reference-field-001.npz"
SRC095=ROOT/"attempt-095-right-registered-route-field-001.py"
ROUTES=ROOT/"attempt-050/arrangement-001/diagnostic-061/fds-fdp-source-routes-061.json"
REG=ROOT/"attempt-088-registered-route-field-refined-2mm-001/result-001/registration-frame-proof.json"
BASE=A/"forearm-shared-field-native-012-diagonal097/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
OUT=ROOT/"attempt-150-right-canonical-midpoint-field-correction-001"
RES=OUT/"result-001"
GS=1.0076111869732585
GT=np.asarray([-.024862083488424228,.24627952500546427,.12388234159326461],np.float64)
START=__import__("time").monotonic()

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x:raise RuntimeError(m)
def load_attempt142():
 spec=importlib.util.spec_from_file_location("attempt142_owner",ATTEMPT142)
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 return m
def exact095():
 tree=ast.parse(SRC095.read_text())
 node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="route_field")
 ns={"np":np,"math":math,"req":req}
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(SRC095),"exec"),ns)
 return ns["route_field"]
def main():
 req(not OUT.exists(),"candidate149 output exists")
 OUT.mkdir();RES.mkdir()
 inputs=(Path(__file__),ATTEMPT142,CAND142,CAND122,CHECK142,DIAG148,FIELD,SRC095,ROUTES,REG,BASE)
 for p in inputs:req(p.exists(),f"missing input {p}")
 hashes={str(p):sha(p) for p in inputs}
 req(hashes[str(CAND142)]=="11e7ae7f16b61c33d803359778a6bf69222f7e5a40f3bdc019555a4dc4c11c66","candidate142 bytes changed")
 req(hashes[str(CAND122)]=="aeeaac9444ef873624d7a0c9e667ffa694b0cb1509ee89e3ccdea8acbc665f94","candidate122 bytes changed")
 d148=json.loads(DIAG148.read_text())
 req(d148["candidate142_sha256"]==hashes[str(CAND142)] and d148["candidate122_sha256"]==hashes[str(CAND122)],"diagnostic148 input identity mismatch")
 req(all(d148["rows"][str(s)]["candidate122_inherited_weights_bit_identical"] for s in (131,133)),"candidate142 changed inherited field weights")
 cp142=pickle.loads(CAND142.read_bytes());cp122=pickle.loads(CAND122.read_bytes())
 cp=copy.deepcopy(cp142);route_field=exact095();ff=np.load(FIELD,allow_pickle=False)
 ctrl=np.asarray(ff["route_control_world_positions_m"],np.float64);cc=np.asarray(ff["route_control_core_body_indices"],np.int64)
 core=list(map(int,cp["core_body_indices"]));req(core==list(map(int,cp122["core_body_indices"])) and core==sorted(set(core)),"canonical 21-core order changed")
 repairs=[]
 for sid in (131,133):
  s=cp["surfaces"][sid];old=cp122["surfaces"][sid]
  oldids=set(map(int,old["global_vertex_ids"]));ids=list(map(int,s["global_vertex_ids"]))
  new=[g for g in ids if g not in oldids]
  req(len(new)==1,f"expected one new midpoint in stable{sid}, found {len(new)}")
  loc={g:i for i,g in enumerate(ids)};gi=new[0];i=loc[gi]
  point=np.asarray(s["vertices_float32_m"][i],np.float32).astype(np.float64)
  source_world=GS*point+GT
  w=route_field(source_world[None,:],ctrl,cc,core)[0]
  oldw=np.asarray(s["core_weights_float64"][i],np.float64)
  req(np.all(w>0) and np.isfinite(w).all() and abs(math.fsum(map(float,w))-1)<1e-12,"canonical midpoint field invalid")
  req(np.array_equal(np.asarray([float(x) for x in w]),np.asarray(w,dtype=np.float64)),"canonical midpoint conversion changed double coefficients")
  s["core_weights_float64"][i]=tuple(map(float,w))
  repairs.append({"stable_id":sid,"global_vertex_id":gi,"source_xyz_m":point.tolist(),"registered_world_m":source_world.tolist(),"old_weight_sha256":hashlib.sha256(np.asarray(oldw,dtype="<f8").tobytes()).hexdigest(),"new_weight_sha256":hashlib.sha256(np.asarray(w,dtype="<f8").tobytes()).hexdigest(),"max_abs_weight_change":float(np.max(np.abs(oldw-w))),"old_vs_canonical_max_abs":float(np.max(np.abs(oldw-w))),"new_min_positive_weight":float(np.min(w))})
 candidate=RES/"candidate.pkl"
 with candidate.open("wb") as f:pickle.dump(cp,f,protocol=pickle.HIGHEST_PROTOCOL);f.flush()
 ch=sha(candidate)
 checkpoint={"schema":"numi.human.right-fds131-fdp133-canonical-midpoint-field-correction-150.checkpoint.v1","status":"field_corrected_candidate_saved_before_predicates","candidate_path":str(candidate),"candidate_sha256":ch,"input_candidate142_path":str(CAND142),"input_candidate142_sha256":hashes[str(CAND142)],"structural_parent_candidate122_path":str(CAND122),"structural_parent_candidate122_sha256":hashes[str(CAND122)],"canonical_route_field_source_path":str(SRC095),"canonical_route_field_source_sha256":hashes[str(SRC095)],"canonical_route_field_npz_path":str(FIELD),"canonical_route_field_npz_sha256":hashes[str(FIELD)],"method":"Preserve candidate142 geometry and all inherited candidate122 coordinates/21-weight tuples; replace only the two midpoint 21-weight tuples with exact attempt095 route_field output at their registered source coordinates. No pose fit, geometry nudge, or weight truncation.","corrected_midpoints":repairs}
 cpout=RES/"construction-checkpoint.json";cpout.write_text(json.dumps(checkpoint,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"event":"candidate_saved","path":str(candidate),"sha256":ch,"midpoints":repairs},sort_keys=True),flush=True)
 m=load_attempt142();loaded=m.loadref();_,_,load_core,_,_,_,_,_,_,ci,bind,dpose,poses,ctx,_,_=loaded
 req(list(map(int,load_core))==core,"pose/helper core binding order mismatch")
 rows={};gpos={};weights={}
 for sid in (131,133):
  s=cp["surfaces"][sid];ids=list(map(int,s["global_vertex_ids"]));pts=np.asarray(s["vertices_float32_m"],np.float32);ww=np.asarray(s["core_weights_float64"],np.float64);faces=np.asarray(s["faces"],np.int64);fg=np.asarray(s["faces_global"],np.int64)
  req(np.array_equal(np.asarray(ids,dtype=np.uint64)[faces],np.asarray(fg,dtype=np.uint64)),"candidate local/global face mapping mismatch")
  req(ww.shape==(len(ids),21) and np.all(ww>0) and np.max(np.abs(ww.sum(axis=1)-1))<1e-12,"candidate full21 positive partition invalid")
  for gid,p,w in zip(ids,pts,ww):
   if gid in gpos:req(np.array_equal(gpos[gid][0],p) and np.array_equal(weights[gid],w),"shared interface point/field differs")
   else:gpos[gid]=(p,p.astype(np.float64));weights[gid]=w
  rows[sid]={"ids":ids,"points":pts,"weights":ww,"faces":faces,"fg":fg}
 source_rows={str(sid):m.exactaudit(rows[sid]["points"],rows[sid]["faces"],ci,f"source-{sid}") for sid in (131,133)}
 source_pos={gid:gpos[gid][1] for gid in gpos}
 source_union=m.unionaudit(rows,source_pos,ci,"source-union")
 sr={"candidate_sha256":ch,"rows":source_rows,"union":source_union}
 source_path=RES/"source-critical.json";source_path.write_text(json.dumps(sr,sort_keys=True,indent=2)+"\n")
 pose=poses["step8000"]
 pos={gid:m.world(gpos[gid][1],weights[gid],core,bind,pose) for gid in gpos}
 f64_rows={str(sid):m.exactaudit(np.asarray([pos[g] for g in rows[sid]["ids"]],np.float64),rows[sid]["faces"],ci,f"8000-f64-{sid}") for sid in (131,133)}
 f64_union=m.unionaudit(rows,pos,ci,"8000-f64-union")
 f64={"schema":"numi.human.right-fds131-fdp133-canonical-midpoint-field-correction-150.current8000-f64.v1","candidate_sha256":ch,"pose_pack_sha256":ctx["step8000"]["pack_sha256"],"pose_receipt_sha256":ctx["step8000"]["receipt_sha256"],"arithmetic":"CPU Float64 binding and weight accumulation; exact triangle predicates on Float64 points; not native GPU.","rows":f64_rows,"union":f64_union}
 f64_path=RES/"current8000-f64-critical.json";f64_path.write_text(json.dumps(f64,sort_keys=True,indent=2)+"\n")
 restrows=[]
 for r in repairs:
  sid=r["stable_id"];gid=r["global_vertex_id"];i=rows[sid]["ids"].index(gid)
  q=m.world(rows[sid]["points"][i],rows[sid]["weights"][i],core,bind,dpose)
  target=GS*rows[sid]["points"][i].astype(np.float64)+GT
  restrows.append({"stable_id":sid,"global_vertex_id":gid,"registered_rest_error_um":float(np.linalg.norm(q-target)*1e6)})
 rest={"candidate_sha256":ch,"input_candidate122_rest_gate_um":2.0,"new_midpoint_errors":restrows,"new_midpoint_max_um":max(x["registered_rest_error_um"] for x in restrows),"source_frame":"G(x_source) using pinned recovered registered transform; CPU arithmetic; not native."}
 restpath=RES/"new-midpoint-rest-check.json";restpath.write_text(json.dumps(rest,sort_keys=True,indent=2)+"\n")
 f32results={}
 for mode in ("separate","fma"):
  xyz={gid:(tuple(map(int,gpos[gid][0].view(np.uint32))),gpos[gid][0]) for gid in gpos}
  posed=m.f32_worlds(sorted(gpos),xyz,weights,core,bind,pose,mode)
  rr={str(sid):m.exact_audit(np.asarray([posed[g] for g in rows[sid]["ids"]],np.float32),rows[sid]["faces"],ci,f"8000-f32-{mode}-{sid}") for sid in (131,133)}
  uu=m.unionaudit(rows,{gid:np.asarray(v,float) for gid,v in posed.items()},ci,f"8000-f32-{mode}-union")
  f32results[mode]={"arithmetic":"CPU ordinary Float32 model; not native GPU.","rows":rr,"union":uu}
  (RES/f"current8000-f32-{mode}-critical.json").write_text(json.dumps(f32results[mode],sort_keys=True,indent=2)+"\n")
 summary={"schema":"numi.human.right-fds131-fdp133-canonical-midpoint-field-correction-150.summary.v1","status":"corrected_field_candidate_not_admitted","candidate_admission":False,"gpu_launched":False,"candidate_path":str(candidate),"candidate_sha256":ch,"checkpoint_path":str(cpout),"checkpoint_sha256":sha(cpout),"field_convention_correction":repairs,"source_gate":{"path":str(source_path),"sha256":sha(source_path),"row_self_pairs":{k:v["self_count"] for k,v in source_rows.items()},"union_pairs":source_union["self_count"],"union_components":source_union["components"]},"current8000_float64":{"path":str(f64_path),"sha256":sha(f64_path),"row_self_pairs":{k:v["self_count"] for k,v in f64_rows.items()},"union_pairs":f64_union["self_count"]},"current8000_cpu_float32":{mode:{"path":str(RES/f"current8000-f32-{mode}-critical.json"),"sha256":sha(RES/f"current8000-f32-{mode}-critical.json"),"row_self_pairs":{k:v["self_count"] for k,v in item["rows"].items()},"row_degenerates":{k:v.get("degenerate_count","not_reported") for k,v in item["rows"].items()},"union_pairs":item["union"]["self_count"],"union_degeneracy":"unavailable_not_computed"} for mode,item in f32results.items()},"new_midpoint_rest":{"path":str(restpath),"sha256":sha(restpath),"max_um":rest["new_midpoint_max_um"],"gate_um":2.0},"inputs_sha256":hashes,"elapsed_seconds":__import__("time").monotonic()-START}
 sp=RES/"summary.json";sp.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n")
 print(json.dumps({"event":"summary_saved","path":str(sp),"sha256":sha(sp),"candidate_sha256":ch,"source":{"rows":summary["source_gate"]["row_self_pairs"],"union":summary["source_gate"]["union_pairs"]},"current8000_f64":{"rows":summary["current8000_float64"]["row_self_pairs"],"union":summary["current8000_float64"]["union_pairs"]},"cpu_f32":summary["current8000_cpu_float32"],"rest_um":rest["new_midpoint_max_um"]},sort_keys=True),flush=True)
if __name__=="__main__":main()
