#!/usr/bin/env python3
from __future__ import annotations
import hashlib, importlib.util, json, mmap, signal, struct, sys, time
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np

R=Path("/Users/n/numi-human-retained-delivery-20261009")
A=R/"anatomy-completion-1276"
SRCROOT=R/"head-muscle-coverage-1291/alternate-head-source-inventory-001/z-anatomy-range"
BASE=SRCROOT/"head-bone-triangle-registration-001"
RUN_ROOT=Path("/Users/n/numi-human-retained-delivery-20261009/head-muscle-coverage-1291/right-deep-161-z-candidate-audit-001/shell-subtraction-001")
OUT=RUN_ROOT/"run-001"
PREFLIGHT_OUT=RUN_ROOT/"preflight-001"
PRE_ONLY="--preflight" in sys.argv
RESULT=(PREFLIGHT_OUT if PRE_ONLY else OUT)/"result-001"
NPZ=SRCROOT/"head-main-meshes.npz"; META=SRCROOT/"head-main-meshes.json.meta"
REG=BASE/"registration-and-interface-report-003.json"
TISS=A/"left-deep-masseter-native-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MAN=TISS.with_suffix(".manifest.json")
PACKDIR=A/"left-deep-masseter-native-001/baseline/native-run/accepted-geometry"
STEPS=[0,4767,5023,5599,6207,6815,7423,8000]
FORWARD=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
OWNER=Path("/Users/n/numi-human-touching-loop-arrangement-1290/src")
PARTITION=OWNER/"numilab_human/cardiac_cavity_partition.py"
CERT=OWNER/"numilab_human/cardiac_partition_certificate.py"
CI=OWNER/"numilab_human/cardiac_cavity_intersections.py"
CA=OWNER/"numilab_human/common_atlas_skin_clearance.py"
OFFSET_M=1.0e-5
STABLE_ID=161; OBJ="Deep part of masseter.r"; TARGET_BONES=[(51004,30),(51004,35),(51005,159)]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def req(x,msg):
 if not x: raise RuntimeError(msg)
def vol(v,f):
 v=np.asarray(v,dtype=np.float64); f=np.asarray(f,dtype=np.int64)
 a=v[f[:,0]]; b=v[f[:,1]]; c=v[f[:,2]]
 return float(np.sum(np.einsum("ij,ij->i",a,np.cross(b,c),optimize=False)/6.0,dtype=np.float64))
def exact_rational(v):
 return [tuple(Fraction.from_float(float(x)) for x in row) for row in np.asarray(v,dtype="<f4")]
def exact_volume(v,f):
 s=Fraction(0)
 for tri in np.asarray(f,dtype=np.int64):
  a,b,c=(v[int(i)] for i in tri)
  s+=(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6
 return s
def parse_keys(pack,ca):
 with Path(pack).open("rb") as stream: mm=mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ)
 h=ca._PACK_HEADER.unpack_from(mm,0); d={}
 for i in range(h[2]):
  item=ca._PACK_DIRECTORY.unpack_from(mm,ca._PACK_HEADER.size+i*ca._PACK_DIRECTORY.size); d[item[0]]=item
 prim=d[4]; keys=set()
 for i in range(prim[4]):
  sem,sid,body,_=struct.unpack_from("<4I",mm,prim[2]+64*i+16)
  if int(sem)!=51999: keys.add((int(sem),int(sid)))
 mm.close(); return keys
def exact_mesh(v,f,ci,cp):
 p=np.asarray(v,dtype="<f4"); ff=np.asarray(f,dtype=np.int64)
 uq,inv=np.unique(p,axis=0,return_inverse=True); qf=inv[ff]
 lattice=[ci.float32_point_lattice_key(tuple(float(x) for x in row)) for row in uq]
 rec=ci._records(lattice,qf.tolist()); audit=ci._audit_pair(rec,rec,same_surface=True); topo=cp._validate_output_topology(rec)
 return rec,uq,qf,audit,topo
def alarm(_sig,_frame): raise TimeoutError("bounded exact shell arrangement exceeded 1200 seconds")

if PRE_ONLY:
 if PREFLIGHT_OUT.exists(): raise SystemExit(f"refuse existing preflight directory {PREFLIGHT_OUT}")
 PREFLIGHT_OUT.mkdir(parents=True); RESULT.mkdir()
else:
 if OUT.exists(): raise SystemExit(f"refuse existing run output directory {OUT}")
 if not (PREFLIGHT_OUT/"result-001"/"preflight.json").is_file():
  raise SystemExit(f"required fresh preflight is missing: {PREFLIGHT_OUT}")
 OUT.mkdir(parents=True); RESULT.mkdir()
CAND_DIR=R/"head-muscle-coverage-1291/right-deep-161-z-candidate-audit-001"
CANDIDATE=CAND_DIR/"stable-161-right-deep-z-row-candidate.npz"
CAND_REPORT=CAND_DIR/"report.json"
OLD_TISS=A/"sartorius50-precision-native-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
inputs=[Path(__file__),NPZ,META,REG,TISS,MAN,FORWARD,PARTITION,CERT,CI,CA,CANDIDATE,CAND_REPORT,OLD_TISS]
packs={}
for step in STEPS:
 pack=PACKDIR/f"step-{step}.mrvpack"; receipt=PACKDIR/f"step-{step}.receipt.json"
 req(pack.is_file() and receipt.is_file(),f"missing accepted capture {step}")
 packs[step]=(pack,receipt); inputs.extend([pack,receipt])
pins={str(p):sha(p) for p in inputs}
expected_pins={
 str(CANDIDATE):"eedc0e15c726ab1c43be2578523bb5692b0f18d2b26c4b29a7bffa6b69918024",
 str(CAND_REPORT):"37f0a0607a38fc6857c802d3b5648264595930f6426c838aa5ba1bb2b2871b19",
 str(NPZ):"2a13e4fc4488aa2c4ceca15a82a32ba22e3bea4b54797bb650c59f707c0e344b",
 str(META):"cd7d30b4b8f8b448c9e433169b7e79a1cbabce3b5d8fdab376d38e4c5783c4a7",
 str(REG):"5b8a5959c4a0e3e7ff559a3dfa2f3687afd1ff363aa6c0f8c686189a23e62ccf",

 str(FORWARD):"ee755eabac267563862c95552524c715bc8dd9897a2bcac7e29e483f5b4a57c5",
 str(PARTITION):"48dd0d826e5afc907706909b8e277421c700d2be7a94a5bc2454b967d1a50360",
 str(CERT):"22c6f349119fc06fcfbd2457e21d266578cb259fb0668c9c510b46c8c451be3a",
 str(CI):"90c7a7fb0ce4383397f97519dd5fe163201d86a0c97e18e3a9eb3fa9858cd561",
 str(CA):"fd152a157cd698094592e64cb746a81a32fc29381a6e6ffc4d5524400041f6b5",
}
for path,expected in expected_pins.items():
 req(pins.get(path)==expected,f"pinned input SHA mismatch: {path}")
if not PRE_ONLY:
 pfpath=PREFLIGHT_OUT/"result-001"/"preflight.json"
 req(pfpath.is_file(),f"required fresh preflight is missing: {pfpath}")
 prior_pf=json.loads(pfpath.read_text())
 req(prior_pf.get("inputs_sha256")==pins and prior_pf.get("inputs_unchanged") is True,
     "preflight does not match exact current script/input hashes")
sys.path.insert(0,str(OWNER))
from numilab_human import common_atlas_skin_clearance as ca, cardiac_cavity_intersections as ci, cardiac_cavity_partition as cp
sp=importlib.util.spec_from_file_location("head_forward_shell161",FORWARD); h=importlib.util.module_from_spec(sp); sp.loader.exec_module(h)
tissue=h.load_tissue(TISS); row159=h.row_data(tissue,159); row160=h.row_data(tissue,160)
req(row159["bc"]==row160["bc"]==1,"superficial masseter reference binding count changed")
req(row159["binding_bytes"]==row160["binding_bytes"],"bilateral superficial masseter head binding bytes differ")
core=int(tissue["bindings"][row159["fb"]]["core"]); req(core==23,"head binding core changed")
bindval=np.asarray(tissue["bindings"][row159["fb"]]["value"],dtype="<f4")
old_tissue=h.load_tissue(OLD_TISS); old159=h.row_data(old_tissue,159)
req(old159["binding_bytes"]==row159["binding_bytes"],"registered candidate binding differs from current scene binding")
candidate_report=json.loads(CAND_REPORT.read_text())
req(candidate_report.get("status")=="offline_candidate_audited_not_composed_or_native_admitted","source candidate report status mismatch")
req(candidate_report.get("candidate_npz_sha256")==sha(CANDIDATE),"candidate report does not bind exact NPZ")
req(candidate_report.get("inputs_sha256",{}).get(str(OLD_TISS))==sha(OLD_TISS),"candidate audit source TISS provenance mismatch")
z=np.load(NPZ,allow_pickle=False); objects={o["object_name"]:o for o in json.loads(META.read_text())["objects"]}
req(OBJ in objects,f"missing current Z source object {OBJ}")
o=objects[OBJ]; raw=np.asarray(z[o["prefix"]+"_world_vertices_f64"],dtype="<f4")
source_faces=np.asarray(z[o["prefix"]+"_triangles_i32"],dtype=np.int64)
req(raw.ndim==2 and raw.shape[1]==3 and source_faces.ndim==2 and source_faces.shape[1]==3,"source array layout mismatch")
registration=json.loads(REG.read_text())["source_registration"]
mat=np.asarray(registration["head_refined_source_to_bodyparts_matrix"],dtype=np.float64)
with np.load(CANDIDATE,allow_pickle=False) as cand:
 cand_vertices6=np.asarray(cand["vertices6"],dtype="<f4")
 bp=np.ascontiguousarray(cand_vertices6[:,:3],dtype="<f4")
 faces=np.asarray(cand["faces"],dtype=np.int64)
 local=np.asarray(cand["binding_indices"],dtype="<u4")
 weights=np.asarray(cand["weights"],dtype="<f4")
req(bp.shape==(len(raw),3) and faces.shape==source_faces.shape and np.array_equal(faces,source_faces),"registered candidate changed Z source face identity")
req(local.shape==(len(bp),4) and weights.shape==(len(bp),4) and np.all(local[:,0]==0) and np.all(local[:,1:]==np.uint32(0xffffffff)) and np.all(weights[:,0]==1) and np.all(weights[:,1:]==0),"candidate is not one-hot ABI5 head-core binding")
req(np.isfinite(bp).all() and np.isfinite(weights).all(),"candidate has nonfinite values")
receipt0=json.loads(packs[0][1].read_text()); req(receipt0.get("accepted_step")==0 and receipt0.get("physical_endpoint")=="accepted","step0 is not accepted")
poses0={int(v["body_index"]):(np.asarray(v["position_m"],dtype="<f4"),np.asarray(v["quaternion_xyzw"],dtype="<f4")) for v in receipt0["accepted_body_poses"]}
req(len(poses0)==157 and core in poses0,"step0 accepted body pose table incomplete")
world0=h.forward(bp,local,weights,row159,tissue,poses0).astype("<f4")
source_volume_before=vol(bp,faces)
if source_volume_before<0: faces=faces[:,[0,2,1]]
source_volume_outward=vol(bp,faces)
req(source_volume_outward>0,"winding correction did not make positive signed volume")
rec0,uq0,qf0,self0,top0=exact_mesh(world0,faces,ci,cp)
req(self0["count"]==0 and top0["closed_oriented_2_manifold"],"registered original deep masseter is not exact-self-clear/closed")
all_target_keys0=parse_keys(packs[0][0],ca)
packed0,target0,_=ca._pack_surfaces(packs[0][0],all_target_keys0)
req(set(target0)==all_target_keys0,f"step0 pack adapter omitted target keys: {len(all_target_keys0-set(target0))}")
def make_shell(key):
 xyz=np.asarray(packed0,dtype="<f4"); tf=np.asarray(target0[key]["faces"],dtype=np.int64)
 used=np.unique(tf.reshape(-1)); remap=np.full(len(xyz),-1,dtype=np.int64); remap[used]=np.arange(len(used))
 v0=np.ascontiguousarray(xyz[used],dtype="<f4"); f0=np.ascontiguousarray(remap[tf],dtype=np.int64)
 prep=ca._prepare_closed_clearance_target(v0,f0)
 req(prep["report"]["closed_connected_oriented"] and prep["report"]["self_intersection_audit"]["count"]==0 and prep["report"]["embedded_closed_target"],f"source bone target invalid {key}")
 v=np.asarray(prep["vertices"],dtype="<f4"); f=np.asarray(prep["faces"],dtype=np.int64)
 signed=vol(v,f)
 if signed<0: f=f[:,[0,2,1]]
 cross=np.cross(v[f[:,1]].astype(float)-v[f[:,0]].astype(float),v[f[:,2]].astype(float)-v[f[:,0]].astype(float))
 normals=np.zeros_like(v,dtype=np.float64)
 for j in range(3): np.add.at(normals,f[:,j],cross)
 ln=np.linalg.norm(normals,axis=1); req(np.isfinite(ln).all() and np.all(ln>0),f"undefined bone vertex normal {key}")
 normals/=ln[:,None]
 expanded=np.asarray(v.astype(float)+OFFSET_M*normals,dtype="<f4")
 ep=ca._prepare_closed_clearance_target(expanded,f)
 req(ep["report"]["closed_connected_oriented"] and ep["report"]["self_intersection_audit"]["count"]==0 and ep["report"]["embedded_closed_target"],f"offset bone shell invalid {key}")
 evert=np.asarray(ep["vertices"],dtype="<f4"); eface=np.asarray(ep["faces"],dtype=np.int64)
 ev=vol(evert,eface)
 if ev<0: eface=eface[:,[0,2,1]]
 return {"vertices":evert,"faces":eface,"offset_report":ep["report"],"source_report":prep["report"],
  "signed_volume_m3":ev,"source_signed_volume_m3":signed,
  "target_faces_sha256":hashlib.sha256(np.ascontiguousarray(f0,dtype="<i8").tobytes()).hexdigest(),
  "expanded_faces_sha256":hashlib.sha256(np.ascontiguousarray(eface,dtype="<i8").tobytes()).hexdigest()}
shells={key:make_shell(key) for key in TARGET_BONES}
if PRE_ONLY:
 target_faces_pf={k:np.asarray(v["faces"],dtype=np.int64) for k,v in target0.items()}
 req(len(target_faces_pf)==len(parse_keys(packs[0][0],ca)) and set(TARGET_BONES).issubset(target_faces_pf),
     "step0 pack target face adapter has incomplete key coverage")
 adapter_audit=ca._target_intersection_audit(rec0,target_faces_pf,packed0)
 req(set(adapter_audit)=={f"{a}:{b}" for a,b in target_faces_pf},
     "step0 target audit returned unexpected keyset")
 audit_summary={"target_surface_count":len(adapter_audit),"total_triangle_pairs":sum(int(v["count"]) for v in adapter_audit.values()),
  "per_target_pair_counts":{k:int(v["count"]) for k,v in adapter_audit.items()},
  "record_serialization_bytes":len(json.dumps(adapter_audit,sort_keys=True,separators=(",",":")))}
 smoke=RESULT/"candidate-array-serialization-smoke.npz"
 np.savez_compressed(smoke,positions_m=bp.astype("<f4"),faces=faces.astype("<i8"),
  binding_indices=local.astype("<u4"),weights=weights.astype("<f4"),
  stable_id=np.asarray([STABLE_ID],dtype="<u4"),core_body_index=np.asarray([core],dtype="<u4"))
 with np.load(smoke,allow_pickle=False) as zsm:
  req(np.array_equal(zsm["positions_m"],bp) and np.array_equal(zsm["faces"],faces)
      and np.array_equal(zsm["binding_indices"],local) and np.array_equal(zsm["weights"],weights),
      "candidate NPZ serialization round-trip changed native row arrays")
 out={"status":"preflight_passed","target_count":len(target_faces_pf),"step0_source_self":self0["count"],
      "source_topology":top0,"source_signed_volume_before_m3":source_volume_before,"source_signed_volume_outward_m3":source_volume_outward,
      "step0_registered_source_world_sha256":hashlib.sha256(world0.tobytes()).hexdigest(),
      "binding_core":core,"binding_sha256":hashlib.sha256(row159["binding_bytes"]).hexdigest(),
      "offset_m":OFFSET_M,"shells":{f"{k[0]}:{k[1]}":{"vertices":len(v["vertices"]),"faces":len(v["faces"]),
       "source_report":v["source_report"],"offset_report":v["offset_report"],"signed_volume_m3":v["signed_volume_m3"]} for k,v in shells.items()},
      "target_audit_adapter_smoke":audit_summary,"candidate_npz_roundtrip_sha256":sha(smoke),
      "inputs_sha256":pins,"inputs_unchanged":all(sha(p)==v for p,v in pins.items())}
 dest=RESULT/"preflight.json"; dest.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
 print(json.dumps({"status":out["status"],"path":str(dest),"sha256":sha(dest),"target_count":out["target_count"],
  "shells":{k:(v["vertices"],v["faces"]) for k,v in out["shells"].items()},"inputs_unchanged":out["inputs_unchanged"]},sort_keys=True),flush=True)
 raise SystemExit(0)
def subtract_shell(vin,fin,ain,shell,name):
 src={"muscle":{"vertices":vin,"triangles":np.asarray(fin,dtype=np.int64).tolist()},
      name:{"vertices":exact_rational(shell["vertices"]),"triangles":shell["faces"].tolist()}}
 signal.alarm(1200)
 rows,meta=cp.construct_arrangement(src,source_names=("muscle",name))
 signal.alarm(0)
 selected=[]; ancestry=[]
 for child in rows:
  if child["source"]=="muscle" and child["other_location"]=="outside":
   selected.append(tuple(child["vertices"])); ancestry.append(dict(ain[int(child["source_face"])]))
  elif child["source"]==name and child["other_location"]=="inside":
   tri=tuple(child["vertices"]); selected.append((tri[0],tri[2],tri[1]))
   ancestry.append({"kind":name+"_offset_cap","source_target_face":int(child["source_face"]),"reversed":True})
 req(selected,f"empty difference against {name}")
 mesh=cp.indexed_mesh(selected,convert=lambda x:x); vv=mesh["vertices_m"]; ff=np.asarray(mesh["triangles"],dtype=np.int64)
 _,_,_,sa,top=exact_mesh(np.asarray([[np.float32(float(x)) for x in p] for p in vv],dtype="<f4"),ff,ci,cp)
 req(sa["count"]==0 and top["closed_oriented_2_manifold"],f"{name} exact output topology/self failed")
 vv_f64=np.asarray([[float(x) for x in p] for p in vv],dtype=np.float64); signed=vol(vv_f64,ff)
 if signed<0:
  ff=ff[:,[0,2,1]]
  ancestry=[dict(a,face_order_reversed_for_outward_winding=True) for a in ancestry]
 signed=vol(vv_f64,ff); req(signed>0,f"{name} subtraction output orientation not outward")
 return vv,ff,ancestry,{"target":name,"arrangement":meta,"vertices":len(vv),"faces":len(ff),"exact_f32_self_pairs":int(sa["count"]),
  "topology":top,"signed_volume_m3":signed,"ancestry_counts":dict(Counter(x["kind"] for x in ancestry))}
start=time.monotonic(); source_vertices=exact_rational(world0); source_faces=faces.copy()
ancestry=[{"kind":"registered_original_source","source_face":i,"source_object":OBJ} for i in range(len(faces))]
source_outward_volume_world=vol(world0,source_faces); subtractions={}
for key in TARGET_BONES:
 name=(f"bone_{key[1]}" if key[0]==51004 else "right_superficial_masseter_10um_shell")
 source_vertices,source_faces,ancestry,subsummary=subtract_shell(source_vertices,source_faces,ancestry,shells[key],name)
 subtractions[f"{key[0]}:{key[1]}"]={**subsummary,"offset_shell_report":shells[key]["offset_report"],
  "original_bone_shell_report":shells[key]["source_report"],"original_bone_signed_volume_m3":shells[key]["source_signed_volume_m3"],
  "offset_shell_signed_volume_m3":shells[key]["signed_volume_m3"],"target_faces_sha256":shells[key]["target_faces_sha256"],
  "offset_faces_sha256":shells[key]["expanded_faces_sha256"]}
 print("subtracted",key,"V/F",len(source_vertices),len(source_faces),"ancestry",subsummary["ancestry_counts"],flush=True)
qraw=np.asarray([[np.float32(float(x)) for x in p] for p in source_vertices],dtype="<f4")
csg_checkpoint=RESULT/"csg-output-f32-world.npz"
np.savez_compressed(csg_checkpoint,positions_m=qraw,faces=np.asarray(source_faces,dtype="<i8"),
 face_ancestry_json=np.asarray(json.dumps(ancestry,sort_keys=True)))
_,q0,fout,selfq,topq=exact_mesh(qraw,source_faces,ci,cp)
req(selfq["count"]==0 and topq["closed_oriented_2_manifold"],"step0 F32 candidate exact self/topology failed")
# Canonicalize by exact world point before mapping, then inverse current head binding/pose into BP source coordinates.
p0,qbody=poses0[core]; rbody=ca._rotation(qbody.astype(np.float64)); rb=ca._rotation(bindval[3:7].astype(np.float64))
body_local=(q0.astype(np.float64)-p0.astype(np.float64))@rbody
bp_candidate=((body_local-bindval[:3].astype(np.float64))@rb)/float(bindval[7])
candidate_source=np.asarray(bp_candidate,dtype="<f4")
q0_precompact=q0.copy()
# Compact any source-coordinate duplicates introduced only by inverse Float32 rounding; all maps are identical one-hot.
candidate_source,src_first,src_inv=np.unique(candidate_source,axis=0,return_index=True,return_inverse=True)
fout=src_inv[fout]
# Rebuild exact ABI5 one-hot lane tables after compaction; do not retain stale pre-compaction rows.
cand_local=np.zeros((len(candidate_source),4),dtype="<u4"); cand_local[:,1:]=np.uint32(0xffffffff)
cand_weights=np.zeros((len(candidate_source),4),dtype="<f4"); cand_weights[:,0]=1.
replayed0=h.forward(candidate_source,cand_local,cand_weights,row159,tissue,poses0).astype("<f4")
replay_error=np.linalg.norm(replayed0[src_inv].astype(float)-q0_precompact.astype(float),axis=1)
req(len(replay_error)==len(q0_precompact) and float(replay_error.max())<1.0e-6,
    "step0 inverse/forward replay error exceeds 1um")
src_rec,src_u,src_f,selfsrc,topsrc=exact_mesh(candidate_source,fout,ci,cp)
req(selfsrc["count"]==0 and topsrc["closed_oriented_2_manifold"],"candidate source-frame exact self/topology failed")
source_vol_before=source_volume_outward
source_vol_after=vol(candidate_source,fout); req(source_vol_after>0,"candidate source frame volume not positive")
pose_rows=[]; pair_tables=[]
for step in STEPS:
 pack,receipt_path=packs[step]; receipt=json.loads(receipt_path.read_text())
 req(receipt.get("accepted_step")==step and receipt.get("physical_endpoint")=="accepted" and receipt.get("pack_file_sha256")==sha(pack),f"capture receipt mismatch {step}")
 poses={int(v["body_index"]):(np.asarray(v["position_m"],dtype="<f4"),np.asarray(v["quaternion_xyzw"],dtype="<f4")) for v in receipt["accepted_body_poses"]}
 req(len(poses)==157 and core in poses,f"pose table incomplete {step}")
 world=h.forward(candidate_source,cand_local[:len(candidate_source)],cand_weights[:len(candidate_source)],row159,tissue,poses).astype("<f4")
 # Candidate binding indices and weights remain one-hot after source compaction.
 mr,wu,wf,sa,top=exact_mesh(world,fout,ci,cp); req(sa["count"]==0 and top["closed_oriented_2_manifold"],f"self/topology failure at pose {step}")
 keys=parse_keys(pack,ca); req(set(TARGET_BONES).issubset(keys),"required head bones absent")
 xyz,target_data,_=ca._pack_surfaces(pack,keys)
 target_faces={k:np.asarray(v["faces"],dtype=np.int64) for k,v in target_data.items()}
 req(set(target_faces)==keys,f"target face adapter key coverage failure at {step}")
 audits=ca._target_intersection_audit(mr,target_faces,xyz)
 req(len(audits)==len(keys) and set(audits)=={f"{a}:{b}" for a,b in keys},f"target-key coverage failure at {step}")
 nonzero={k:{"count":int(v["count"]),"pairs":v["triangle_pairs"][:256]} for k,v in audits.items() if v["count"]}
 bone_counts={f"{a}:{b}":int(audits[f"{a}:{b}"]["count"]) for a,b in TARGET_BONES}
 pose_rows.append({"accepted_step":step,"accepted_body_state_sha256":receipt["accepted_body_state_sha256"],
  "candidate_world_f32_sha256":hashlib.sha256(world.tobytes()).hexdigest(),"self_pair_count":int(sa["count"]),
  "topology":top,"target_surface_count":len(audits),"total_target_pairs":int(sum(v["count"] for v in audits.values())),
  "cut_target_pair_counts":bone_counts,"nonzero_external_targets":nonzero})
 pair_tables.append({"step":step,"pairs":{k:v["triangle_pairs"] for k,v in audits.items() if v["count"]}})
 print("pose",step,"self",sa["count"],"pairs",sum(v["count"] for v in audits.values()),"bones",bone_counts,"nonzero",len(nonzero),flush=True)
req(all(sha(p)==v for p,v in pins.items()),"input changed during run")
npzout=RESULT/"stable-161-right-deep-offset-shell-10um.npz"
np.savez_compressed(npzout,positions_m=candidate_source.astype("<f4"),faces=fout.astype("<i8"),binding_indices=cand_local.astype("<u4"),
 weights=cand_weights.astype("<f4"),step0_world_f32=replayed0,face_ancestry_json=np.asarray(json.dumps(ancestry,sort_keys=True)),
 stable_id=np.asarray([STABLE_ID],dtype="<u4"),core_body_index=np.asarray([core],dtype="<u4"),
 binding_record_sha256=np.asarray([hashlib.sha256(row159["binding_bytes"]).hexdigest()]))
pairs_path=RESULT/"pairs-by-pose.json"; pairs_path.write_text(json.dumps(pair_tables,sort_keys=True,separators=(",",":"))+"\n")
def marker_summary():
 out={}
 for marker in json.loads(META.read_text()).get("objects",[]):
  name=str(marker.get("object_name",""))
  if name not in ("Masseter.o1l","Masseter.ol","Masseter.el"): continue
  arr=np.asarray(z[marker["prefix"]+"_world_vertices_f64"],dtype="<f4")
  reg=np.asarray(arr.astype(float)@mat[:3,:3].T+mat[:3,3],dtype="<f4")
  out[name]={"marker_vertices":len(arr),"registered_source_sha256":hashlib.sha256(reg.tobytes()).hexdigest(),
   "marker_position_changed":False,"meaning":"source marker only; not a certified attachment footprint"}
 return out
report={"schema":"offline.deep-masseter-offset-shell-reference.v1","status":"unadmitted_candidate",
 "scope":"Stable161 right deep masseter from the frozen registered Z-Anatomy candidate, sequentially subtracted against 10um outward shells at accepted step0 for temporal bone30, mandible35, and right superficial masseter159.",
 "inputs_sha256":pins,"inputs_unchanged":all(sha(p)==v for p,v in pins.items()),
 "source":{"object":OBJ,"stable_id":STABLE_ID,"semantic_member":"FMA49004","source_family":"Z-Anatomy Startup.blend evaluated object; FMA49004 is semantic key only, not a BP3D tessellation claim","source_vertex_count":len(raw),"source_face_count":len(faces),
  "source_volume_before_orientation_m3":source_volume_before,"source_volume_outward_registered_m3":source_volume_outward,
  "source_volume_candidate_bp_frame_m3":source_vol_after,"candidate_volume_delta_bp_frame_m3":source_vol_after-source_vol_before,
  "binding_reference_row":159,"binding_semantic":51005,"binding_core_body_index":core,
  "binding_record_sha256":hashlib.sha256(row159["binding_bytes"]).hexdigest(),"binding_equal_to160":True,
  "transform":"Z-Anatomy world -> pinned head residual registration -> unchanged stable159 head binding -> captured accepted body23 pose.",
  "mesh_winding_reversed":source_volume_before<0},
 "clearance_reserve":{"offset_m":OFFSET_M,"offset_um":OFFSET_M*1e6,
  "interpretation":"Inferred visualization reserve above measured 8.5024nm F32 shared-plane rounding; not measured enthesis gap or force.",
  "per_bone_shell_operations":subtractions,"source_volume_removed_fraction":(source_vol_before-source_vol_after)/source_vol_before if source_vol_before else None},
 "geometry":{"source_exact_self_pairs":int(self0["count"]),"source_topology":top0,"source_outward_signed_volume_world_m3":source_outward_volume_world,
  "candidate_source_self_pairs":int(selfsrc["count"]),"candidate_source_topology":topsrc,"step0_F32_world_self_pairs":int(selfq["count"]),
  "step0_F32_world_topology":topq,"candidate_vertices":len(candidate_source),"candidate_faces":len(fout),
  "step0_inverse_forward_max_error_m":float(replay_error.max()),"step0_inverse_forward_p99_error_m":float(np.percentile(replay_error,99)),
  "face_ancestry_counts":dict(Counter(x["kind"] for x in ancestry)),
  "all_8_pose_self_and_all_target_clear":all(x["self_pair_count"]==0 and not x["nonzero_external_targets"] for x in pose_rows)},
 "source_markers":marker_summary(),"capture_pose_count":len(pose_rows),"capture_pose_rows":pose_rows,
 "all_current_surfaces_except_bed_are_scanned":True,"candidate_npz":str(npzout),"candidate_npz_sha256":sha(npzout),
 "pair_tables":str(pairs_path),"pair_tables_sha256":sha(pairs_path),
 "limitations":["No NHTISS/TISS asset was changed or composed.","The 10um shell reserve separates material from temporal bone30, mandible35 and right superficial masseter159, including nominal attachment regions; this is an inferred visualization correction.","The pinned Z export provides no separate endpoint marker objects or certified attachment footprint.","Eight actual baseline accepted poses only; no intervention, jaw mechanics, force, or native qualification."]}
report_path=RESULT/"report.json"; report_path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":report["status"],"report":str(report_path),"report_sha256":sha(report_path),
 "candidate":str(npzout),"candidate_sha256":sha(npzout),"vertices":len(candidate_source),"faces":len(fout),
 "pose_rows":[[x["accepted_step"],x["self_pair_count"],x["total_target_pairs"],x["bone_target_pair_counts"],len(x["nonzero_external_targets"])] for x in pose_rows],
 "all_clear":report["geometry"]["all_8_pose_self_and_all_target_clear"],"volume_delta_m3":source_vol_after-source_vol_before,
 "elapsed_seconds":time.monotonic()-start},sort_keys=True),flush=True)
