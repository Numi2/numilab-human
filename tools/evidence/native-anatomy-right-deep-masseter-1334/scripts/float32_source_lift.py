#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

R=Path("/Users/n/numi-human-retained-delivery-20261009")
A=R/"anatomy-completion-1276"
D=R/"head-muscle-coverage-1291/right-deep-161-z-candidate-audit-001/shell-subtraction-001"
OUT=D/"replay-recovery-002/output-001"
RESULT=OUT/"result-001"
CSG=D/"run-001/result-001/csg-output-f32-world.npz"
TISS=A/"left-deep-masseter-native-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
PACKDIR=A/"left-deep-masseter-native-001/baseline/native-run/accepted-geometry"
FORWARD=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
OWNER=Path("/Users/n/numi-human-touching-loop-arrangement-1290/src")
CI=OWNER/"numilab_human/cardiac_cavity_intersections.py"
CP=OWNER/"numilab_human/cardiac_cavity_partition.py"
CA=OWNER/"numilab_human/common_atlas_skin_clearance.py"
STEPS=[0,4767,5023,5599,6207,6815,7423,8000]
TARGET_FACE_ROWS=[105,111,1629,1630]
EXPECTED_PAIRS=[[105,1630],[111,1629],[111,1630]]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x: raise RuntimeError(m)
def exact_mesh(v,f):
 p=np.asarray(v,dtype="<f4"); ff=np.asarray(f,dtype=np.int64)
 uq,inv=np.unique(p,axis=0,return_inverse=True); qf=inv[ff]
 lattice=[ci.float32_point_lattice_key(tuple(float(x) for x in row)) for row in uq]
 rec=ci._records(lattice,qf.tolist()); audit=ci._audit_pair(rec,rec,same_surface=True); topo=cp._validate_output_topology(rec)
 return uq,qf,rec,audit,topo
def vol(v,f):
 v=np.asarray(v,dtype=np.float64); f=np.asarray(f,dtype=np.int64)
 return float(np.sum(np.einsum("ij,ij->i",v[f[:,0]],np.cross(v[f[:,1]],v[f[:,2]]),optimize=False)/6.0,dtype=np.float64))
def area(v,f):
 v=np.asarray(v,dtype=np.float64); f=np.asarray(f,dtype=np.int64)
 return float(np.sum(np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=1)*0.5,dtype=np.float64))
def float_step(x,n):
 y=np.float32(x)
 if n==0: return y
 direction=np.float32(np.inf if n>0 else -np.inf)
 for _ in range(abs(int(n))): y=np.nextafter(y,direction,dtype=np.float32)
 return y
def ulp_key(x):
 b=int(np.asarray([np.float32(x)],dtype="<f4").view("<u4")[0])
 return ((~b)&0xffffffff) if (b&0x80000000) else (b^0x80000000)
def ulp_delta(a,b):
 return abs(ulp_key(a)-ulp_key(b))
def atomic_json(path,obj):
 tmp=path.with_suffix(path.suffix+".tmp")
 tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n")
 tmp.replace(path)

req(not OUT.exists(),f"refuse existing output {OUT}")
OUT.mkdir(); RESULT.mkdir()
receipt0=json.loads((PACKDIR/"step-0.receipt.json").read_text())
req(receipt0.get("accepted_step")==0 and receipt0.get("physical_endpoint")=="accepted","step0 not accepted")
sys.path.insert(0,str(OWNER))
from numilab_human import common_atlas_skin_clearance as ca, cardiac_cavity_intersections as ci, cardiac_cavity_partition as cp
sp=importlib.util.spec_from_file_location("head_forward_shell161_lift_recovery",FORWARD)
h=importlib.util.module_from_spec(sp); sp.loader.exec_module(h)
pins={str(p):sha(p) for p in [Path(__file__),CSG,TISS,FORWARD,CI,CP,CA,PACKDIR/"step-0.mrvpack",PACKDIR/"step-0.receipt.json"]}
with np.load(CSG,allow_pickle=False) as z:
 qraw=np.asarray(z["positions_m"],dtype="<f4"); faces=np.asarray(z["faces"],dtype=np.int64)
 ancestry=json.loads(str(z["face_ancestry_json"].item()))
req(len(ancestry)==len(faces) and qraw.shape[1]==3,"CSG checkpoint layout mismatch")
uq,qf,raw_records,raw_audit,raw_top=exact_mesh(qraw,faces)
req(raw_audit["count"]==0 and raw_top["closed_oriented_2_manifold"],"retained world CSG checkpoint failed exact source gate")
poses0={int(v["body_index"]):(np.asarray(v["position_m"],dtype="<f4"),np.asarray(v["quaternion_xyzw"],dtype="<f4")) for v in receipt0["accepted_body_poses"]}
req(len(poses0)==157,"step0 pose count mismatch")
tissue=h.load_tissue(TISS); row=h.row_data(tissue,159)
core=int(tissue["bindings"][row["fb"]]["core"]); bind=np.asarray(tissue["bindings"][row["fb"]]["value"],dtype="<f4")
req(core in poses0 and row["bc"]==1,"unexpected one-hot head binding")
p0,qbody=poses0[core]; rbody=ca._rotation(qbody.astype(np.float64)); rb=ca._rotation(bind[3:7].astype(np.float64))
body_local=(uq.astype(np.float64)-p0.astype(np.float64))@rbody
bp64=((body_local-bind[:3].astype(np.float64))@rb)/float(bind[7])
source_unc=np.asarray(bp64,dtype="<f4")
source,source_first,source_inv=np.unique(source_unc,axis=0,return_index=True,return_inverse=True)
source_faces=source_inv[qf]
local=np.zeros((len(source),4),dtype="<u4"); local[:,1:]=np.uint32(0xffffffff)
weights=np.zeros((len(source),4),dtype="<f4"); weights[:,0]=1
def forward_points(points,poses):
 pts=np.asarray(points,dtype="<f4")
 li=np.zeros((len(pts),4),dtype="<u4"); li[:,1:]=np.uint32(0xffffffff)
 ww=np.zeros((len(pts),4),dtype="<f4"); ww[:,0]=1
 return h.forward(pts,li,ww,row,tissue,poses).astype("<f4")
world_before=forward_points(source,poses0)
uq_before,qf_before,rec_before,audit_before,top_before=exact_mesh(world_before,source_faces)
pairs_before=[[int(a),int(b)] for a,b in audit_before["triangle_pairs"]]
req(pairs_before==EXPECTED_PAIRS,"replay witness no longer matches the three pinned exact pair rows")
affected={}
for face_i in sorted(set(i for pair in EXPECTED_PAIRS for i in pair)):
 for corner,qid0 in enumerate(qf[face_i]):
  qid=int(qid0); sid=int(source_inv[qid]); target=tuple(float(x) for x in uq[qid])
  affected.setdefault(sid,[]).append({"face":face_i,"corner":corner,"world_vertex_id":qid,"desired_world_m":target})
target_by_source={}
for sid,rows in affected.items():
 ts={tuple(x["desired_world_m"]) for x in rows}
 req(len(ts)==1,f"source compaction merges distinct intended world vertices for source id {sid}")
 target_by_source[sid]=np.asarray(next(iter(ts)),dtype="<f4")
def eval_one(x):
 return forward_points(np.asarray(x,dtype="<f4").reshape(1,3),poses0)[0]
search_rows=[]
corrected=source.copy()
for sid in sorted(target_by_source):
 x=corrected[sid].copy(); original=x.copy(); target=target_by_source[sid]
 y=eval_one(x); initial_error=float(np.linalg.norm(y.astype(np.float64)-target.astype(np.float64)))
 iterations=0; used_fallback=False
 for iterations in range(12):
  if np.array_equal(y,target): break
  # Local finite-difference Jacobian of the true packed F32 forward, then one rounded Newton lift.
  eps=np.float32(1.0e-4)
  jac=np.empty((3,3),dtype=np.float64)
  for axis in range(3):
   xp=x.copy(); xp[axis]=np.float32(float(xp[axis])+float(eps))
   yp=eval_one(xp)
   jac[:,axis]=(yp.astype(np.float64)-y.astype(np.float64))/float(xp[axis]-x[axis])
  req(np.isfinite(jac).all() and abs(np.linalg.det(jac))>1.0e-8,"singular local forward Jacobian")
  delta=np.linalg.solve(jac,target.astype(np.float64)-y.astype(np.float64))
  xn=np.asarray(x.astype(np.float64)+delta,dtype="<f4")
  yn=eval_one(xn)
  if not np.array_equal(xn,x) and np.linalg.norm(yn.astype(float)-target.astype(float)) < np.linalg.norm(y.astype(float)-target.astype(float)):
   x,y=xn,yn; continue
  # If Newton rounds onto a plateau, test a compact 7^3 ULP neighborhood, still local to this seam vertex.
  used_fallback=True; cand=[]; cand_meta=[]
  offs=(-3,-2,-1,0,1,2,3)
  for dx in offs:
   for dy in offs:
    for dz in offs:
     if dx==dy==dz==0: continue
     xx=np.asarray([float_step(x[0],dx),float_step(x[1],dy),float_step(x[2],dz)],dtype="<f4")
     if np.array_equal(xx,x): continue
     cand.append(xx); cand_meta.append((dx,dy,dz))
  if not cand: break
  ys=forward_points(np.asarray(cand,dtype="<f4"),poses0)
  errs=np.linalg.norm(ys.astype(np.float64)-target.astype(np.float64),axis=1)
  k=int(np.argmin(errs))
  if errs[k] < np.linalg.norm(y.astype(float)-target.astype(float)):
   x=np.asarray(cand[k],dtype="<f4"); y=ys[k]
  else: break
 corrected[sid]=x
 final_error=float(np.linalg.norm(y.astype(np.float64)-target.astype(np.float64)))
 search_rows.append({"source_vertex_id":sid,"originating_face_corners":affected[sid],"desired_world_m":target.tolist(),
  "initial_replay_world_m":eval_one(original).tolist(),"final_replay_world_m":y.tolist(),
  "initial_error_m":initial_error,"final_error_m":final_error,"exact_target_reproduced":bool(np.array_equal(y,target)),
  "source_f32_before_m":original.tolist(),"source_f32_after_m":x.tolist(),
  "source_displacement_m":float(np.linalg.norm(x.astype(float)-original.astype(float))),
  "per_coordinate_ulp_delta":[ulp_delta(original[k],x[k]) for k in range(3)],"newton_iterations":iterations+1,
  "used_local_ulp_neighborhood":used_fallback})
lift_path=RESULT/"source-lift-f32.npz"
np.savez_compressed(lift_path,positions_m=corrected.astype("<f4"),faces=source_faces.astype("<i8"),
 binding_indices=local,weights=weights,face_ancestry_json=np.asarray(json.dumps(ancestry,sort_keys=True)),
 corrected_source_vertex_ids=np.asarray(sorted(target_by_source),dtype="<i8"))
world_after=forward_points(corrected,poses0)
uq_after,qf_after,rec_after,audit_after,top_after=exact_mesh(world_after,source_faces)
pose0_rows={"self_pair_count":int(audit_after["count"]),"topology":top_after,"exact_pairs":audit_after["triangle_pairs"][:64],
 "world_f32_sha256":hashlib.sha256(world_after.tobytes()).hexdigest(),"target_world_match_vertices":search_rows}
candidate_src_uq,candidate_src_qf,candidate_src_rec,candidate_src_audit,candidate_src_top=exact_mesh(corrected,source_faces)
source_volume=vol(corrected,source_faces)
req(source_volume>0,"candidate source volume is not positive")
candidate_source_pass=(candidate_src_audit["count"]==0 and candidate_src_top["closed_oriented_2_manifold"])
atomic_json(RESULT/"pose-0-replay.json",pose0_rows)
# Persist each pose result separately before proceeding. Full target scan starts only if all eight self/topology gates pass.
pose_self=[]
for step in STEPS:
 pack=PACKDIR/f"step-{step}.mrvpack"; receipt_path=PACKDIR/f"step-{step}.receipt.json"
 receipt=json.loads(receipt_path.read_text())
 req(receipt.get("accepted_step")==step and receipt.get("physical_endpoint")=="accepted" and receipt.get("pack_file_sha256")==sha(pack),f"accepted capture mismatch {step}")
 poses={int(v["body_index"]):(np.asarray(v["position_m"],dtype="<f4"),np.asarray(v["quaternion_xyzw"],dtype="<f4")) for v in receipt["accepted_body_poses"]}
 req(len(poses)==157 and core in poses,f"accepted pose table incomplete {step}")
 world=forward_points(corrected,poses)
 _,_,_,sa,top=exact_mesh(world,source_faces)
 item={"accepted_step":step,"world_f32_sha256":hashlib.sha256(world.tobytes()).hexdigest(),"self_pair_count":int(sa["count"]),
       "topology":top,"signed_volume_m3":vol(world,source_faces),"all_face_ancestry_count":len(ancestry)}
 pose_self.append(item)
 atomic_json(RESULT/f"pose-{step}-self.json",item)
print(json.dumps({"stage":"precision_lift_and_allpose_self","lift_npz":str(lift_path),"lift_sha256":sha(lift_path),
 "source_lift_search":search_rows,"source_exact_self":int(candidate_src_audit["count"]),"source_topology":candidate_src_top,
 "source_signed_volume_m3":source_volume,"pose_self":pose_self,"inputs_sha256":pins},sort_keys=True),flush=True)
if not candidate_source_pass or not all(x["self_pair_count"]==0 and x["topology"]["closed_oriented_2_manifold"] and x["signed_volume_m3"]>0 for x in pose_self):
 final={"status":"rejected_pose_self_or_topology","source_self_pair_count":int(candidate_src_audit["count"]),
  "source_topology":candidate_src_top,"source_signed_volume_m3":source_volume,"source_lift_search":search_rows,
  "pose_rows":pose_self,"inputs_sha256":pins,"inputs_unchanged":all(sha(p)==v for p,v in pins.items()),
  "candidate_lift_npz":str(lift_path),"candidate_lift_npz_sha256":sha(lift_path),
  "limitations":["No external target scans were run because source or accepted-pose exact self/topology admission failed. This is not an admitted candidate."]}
 atomic_json(RESULT/"report.json",final); print(json.dumps({"status":final["status"],"report":str(RESULT/"report.json"),"sha256":sha(RESULT/"report.json")},sort_keys=True)); raise SystemExit(1)
# Only after all source/self/pose gates pass, scan every current non-bed target at all eight actual poses.
target_rows=[]; pair_tables=[]
for step in STEPS:
 pack=PACKDIR/f"step-{step}.mrvpack"; receipt=json.loads((PACKDIR/f"step-{step}.receipt.json").read_text())
 poses={int(v["body_index"]):(np.asarray(v["position_m"],dtype="<f4"),np.asarray(v["quaternion_xyzw"],dtype="<f4")) for v in receipt["accepted_body_poses"]}
 world=forward_points(corrected,poses)
 _,_,records0,sa,top=exact_mesh(world,source_faces)
 keys=ca._parse_pack_surface_keys(pack) if hasattr(ca,"_parse_pack_surface_keys") else None
 if keys is None:
  with pack.open("rb") as stream:
   import mmap,struct
   mm=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ); hdr=ca._PACK_HEADER.unpack_from(mm,0); directory={}
   for i in range(hdr[2]):
    item=ca._PACK_DIRECTORY.unpack_from(mm,ca._PACK_HEADER.size+i*ca._PACK_DIRECTORY.size); directory[item[0]]=item
   prim=directory[4]; keys=set()
   for i in range(prim[4]):
    sem,sid,body,_=struct.unpack_from("<4I",mm,prim[2]+64*i+16)
    if int(sem)!=51999: keys.add((int(sem),int(sid)))
   mm.close()
 xyz,targets,_=ca._pack_surfaces(pack,keys)
 req(set(targets)==keys and len(keys)==865,f"not all 865 non-bed target keys present at step {step}: {len(keys)}")
 target_faces={k:np.asarray(v["faces"],dtype=np.int64) for k,v in targets.items()}
 audits=ca._target_intersection_audit(records0,target_faces,xyz)
 req(set(audits)=={f"{a}:{b}" for a,b in keys},"target audit keyset mismatch")
 nonzero={k:{"count":int(v["count"]),"pairs":v["triangle_pairs"][:256]} for k,v in audits.items() if v["count"]}
 target_rows.append({"accepted_step":step,"target_surface_count":len(audits),"target_pair_count":sum(int(v["count"]) for v in audits.values()),
  "nonzero_targets":{k:int(v["count"]) for k,v in audits.items() if v["count"]}})
 pair_tables.append({"accepted_step":step,"pairs":{k:v["triangle_pairs"] for k,v in audits.items() if v["count"]}})
 atomic_json(RESULT/f"pose-{step}-targets.json",target_rows[-1])
report={"status":"unadmitted_candidate" if any(x["target_pair_count"] for x in target_rows) else "offline_candidate_all_gates_passed_not_composed_or_native_admitted",
 "inputs_sha256":pins,"inputs_unchanged":all(sha(p)==v for p,v in pins.items()),
 "method":"Minimal local Float32 source-lift correction of only vertices belonging to the three exact source/temporal-bone30 offset-cap pairs after exact CSG; each corrected vertex is moved toward the nearest F32 source coordinate whose step0 native-style forward reproduces its retained CSG world coordinate. No shell was widened or CSG rerun.",
 "csg_checkpoint_sha256":sha(CSG),"source_candidate_sha256":sha(lift_path),"source_candidate":str(lift_path),
 "original_csg_world_self_pair_count":int(raw_audit["count"]),"original_csg_world_topology":raw_top,
 "precorrection_step0_self_pairs":pairs_before,"source_lift_search":search_rows,
 "source_self_pair_count":int(candidate_src_audit["count"]),"source_topology":candidate_src_top,"source_signed_volume_m3":source_volume,
 "all_pose_self_rows":pose_self,"all_pose_self_and_topology_pass":all(x["self_pair_count"]==0 and x["topology"]["closed_oriented_2_manifold"] and x["signed_volume_m3"]>0 for x in pose_self),
 "target_coverage":{"non_bed_target_count_per_pose":865,"pose_rows":target_rows,"all_targets_clear":all(x["target_pair_count"]==0 for x in target_rows)},
 "face_ancestry_counts":dict(__import__("collections").Counter(x["kind"] for x in ancestry)),
 "candidate_source_volume_delta_from_world_csg_m3":source_volume-vol(corrected,source_faces),
 "pair_tables":str(RESULT/"pairs-by-pose.json"),"limitations":["Inferred numerical source-lift correction only; not measured attachment geometry.","Eight accepted baseline poses only; no native integration, intervention validation, or jaw mechanics."]}
pairpath=RESULT/"pairs-by-pose.json"; pairpath.write_text(json.dumps(pair_tables,sort_keys=True,separators=(",",":"))+"\n")
report["pair_tables_sha256"]=sha(pairpath)
atomic_json(RESULT/"report.json",report)
print(json.dumps({"status":report["status"],"report":str(RESULT/"report.json"),"report_sha256":sha(RESULT/"report.json"),
 "target_rows":target_rows,"inputs_unchanged":report["inputs_unchanged"]},sort_keys=True),flush=True)
