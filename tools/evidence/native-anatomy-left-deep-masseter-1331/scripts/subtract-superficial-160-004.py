#!/usr/bin/env python3
import hashlib, importlib.util, json, mmap, signal, struct, sys, time
from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009")
A=R/"anatomy-completion-1276"
BASE=R/"head-muscle-coverage-1291/alternate-head-source-inventory-001/z-anatomy-range/head-bone-triangle-registration-001"
ROOT=BASE/"deep-temporalis-correction-001/deep-left-rib-shell-subtraction-003"
RUN=ROOT/"subtract-superficial-160-004"
SRC=BASE.parent
SCRIPT=BASE/"deep-temporalis-correction-001/deep-left-rib-shell-subtraction-003.py"
PREF=ROOT/"preflight-001/result-001/preflight.json"
BASE_REPORT=ROOT/"run-001/final-recheck-003/report.json"
CHECKPOINT=ROOT/"run-001/result-001/csg-output-f32-world.npz"
CANDIDATE=ROOT/"run-001/result-001/stable-161-deep-left-offset-shell-10um.npz"
PAIRS_OLD=ROOT/"run-001/result-001/pairs-by-pose.json"
TISS=A/"left-scm-masseters-positive-winding-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MAN=TISS.with_suffix(".manifest.json")
PACKDIR=A/"left-scm-masseters-outward-native-001/baseline/native-run/accepted-geometry"
FORWARD=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
OWNER=Path("/Users/n/numi-human-touching-loop-arrangement-1290/src")
PARTITION=Path("/Users/n/numi-human-signed-winding-1273/src/numilab_human/cardiac_cavity_partition.py")
CERT=Path("/Users/n/numi-human-signed-winding-1273/src/numilab_human/cardiac_partition_certificate.py")
CI=OWNER/"numilab_human/cardiac_cavity_intersections.py"
CA=OWNER/"numilab_human/common_atlas_skin_clearance.py"
STEPS=[0,4767,5023,5599,6207,6815,7423,8000]
STABLE=161; SEM=51005; CORE=23; OFFSET=1.0e-5
OUT=RUN/"result-003"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def require(x,msg):
 if not x: raise RuntimeError(msg)
def vol(v,f):
 v=np.asarray(v,dtype=np.float64); f=np.asarray(f,dtype=np.int64)
 a=v[f[:,0]]; b=v[f[:,1]]; c=v[f[:,2]]
 return float(np.sum(np.einsum("ij,ij->i",a,np.cross(b,c),optimize=False)/6.,dtype=np.float64))
def exact_rational(v):
 return [tuple(Fraction.from_float(float(x)) for x in row) for row in np.asarray(v,dtype="<f4")]
def exact_mesh(v,f,ci,cp):
 p=np.asarray(v,dtype="<f4"); ff=np.asarray(f,dtype=np.int64)
 uq,inv=np.unique(p,axis=0,return_inverse=True); qf=inv[ff]
 lattice=[ci.float32_point_lattice_key(tuple(float(x) for x in row)) for row in uq]
 rec=ci._records(lattice,qf.tolist())
 audit=ci._audit_pair(rec,rec,same_surface=True)
 topo=cp._validate_output_topology(rec)
 return rec,uq,qf,audit,topo
def parse_keys(pack,ca):
 with Path(pack).open("rb") as st: mm=mmap.mmap(st.fileno(),0,access=mmap.ACCESS_READ)
 h=ca._PACK_HEADER.unpack_from(mm,0); d={}
 for i in range(h[2]):
  item=ca._PACK_DIRECTORY.unpack_from(mm,ca._PACK_HEADER.size+i*ca._PACK_DIRECTORY.size); d[item[0]]=item
 prim=d[4]; keys=set()
 for i in range(prim[4]):
  sem,sid,body,_=struct.unpack_from("<4I",mm,prim[2]+64*i+16)
  if int(sem)!=51999: keys.add((int(sem),int(sid)))
 mm.close(); return keys
def signed_shell(vertices,faces,ca,name):
 v=np.asarray(vertices,dtype="<f4"); f=np.asarray(faces,dtype=np.int64)
 prep=ca._prepare_closed_clearance_target(v,f)
 require(prep["report"]["closed_connected_oriented"] and prep["report"]["self_intersection_audit"]["count"]==0 and prep["report"]["embedded_closed_target"], name+" original mesh invalid")
 v=np.asarray(prep["vertices"],dtype="<f4"); f=np.asarray(prep["faces"],dtype=np.int64); signed=vol(v,f)
 if signed<0: f=f[:,[0,2,1]]
 cross=np.cross(v[f[:,1]].astype(float)-v[f[:,0]].astype(float),v[f[:,2]].astype(float)-v[f[:,0]].astype(float))
 normals=np.zeros_like(v,dtype=np.float64)
 for j in range(3): np.add.at(normals,f[:,j],cross)
 ln=np.linalg.norm(normals,axis=1); require(np.isfinite(ln).all() and np.all(ln>0),name+" undefined normals")
 normals/=ln[:,None]
 expanded=np.asarray(v.astype(float)+OFFSET*normals,dtype="<f4")
 ep=ca._prepare_closed_clearance_target(expanded,f)
 require(ep["report"]["closed_connected_oriented"] and ep["report"]["self_intersection_audit"]["count"]==0 and ep["report"]["embedded_closed_target"],name+" offset shell invalid")
 ev=np.asarray(ep["vertices"],dtype="<f4"); ef=np.asarray(ep["faces"],dtype=np.int64)
 es=vol(ev,ef)
 if es<0: ef=ef[:,[0,2,1]]
 return ev,ef,prep["report"],ep["report"],signed,es
def alarm(_sig,_frame): raise TimeoutError("bounded exact arrangement exceeded 1200 seconds")
if OUT.exists(): raise SystemExit("refuse existing output "+str(OUT))
OUT.mkdir(parents=True)
start=time.monotonic()
pins=json.loads(PREF.read_text())["inputs_sha256"]
# Assert all prior source/predicate/capture pins are unchanged and checkpoint/report inputs are exact.
changed={p:(v,sha(p)) for p,v in pins.items() if sha(p)!=v}
require(not changed,"preflight source/capture pins changed: "+repr(changed))
base_report=json.loads(BASE_REPORT.read_text())
require(base_report["status"]=="rejected_unadmitted_external_masseter_interface","unexpected prior candidate status")
for p,expected in [(SCRIPT,pins[str(SCRIPT)]),(TISS,pins[str(TISS)]),(MAN,pins[str(MAN)]),
                   (FORWARD,pins[str(FORWARD)]),(CI,pins[str(CI)]),(CA,pins[str(CA)]),
                   (PARTITION,pins[str(PARTITION)]),(CERT,pins[str(CERT)])]:
 require(sha(p)==expected,"pinned owner/source changed: "+str(p))
out_pins={str(p):sha(p) for p in [BASE_REPORT,CHECKPOINT,CANDIDATE,PAIRS_OLD]}
require(base_report["retained_output_sha256"][str(CANDIDATE)]==out_pins[str(CANDIDATE)],"prior candidate output pin changed")
require(base_report["retained_output_sha256"][str(CHECKPOINT)]==out_pins[str(CHECKPOINT)],"prior CSG checkpoint pin changed")
sys.path.insert(0,str(OWNER))
from numilab_human import common_atlas_skin_clearance as ca, cardiac_cavity_intersections as ci, cardiac_cavity_partition as cp
spec=importlib.util.spec_from_file_location("head_forward_shell161",FORWARD)
h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
tissue=h.load_tissue(TISS); row159=h.row_data(tissue,159); row160=h.row_data(tissue,160)
require(row159["bc"]==row160["bc"]==1 and row159["binding_bytes"]==row160["binding_bytes"],"masseter head binding records differ")
require(int(tissue["bindings"][row159["fb"]]["core"])==CORE,"masseter head core mismatch")
with np.load(CHECKPOINT,allow_pickle=False) as z:
 world=np.asarray(z["positions_m"],dtype="<f4")
 faces=np.asarray(z["faces"],dtype=np.int64)
 ancestry=json.loads(str(z["face_ancestry_json"]))
require(world.shape==(659,3) and faces.shape==(1318,3),"post-bone checkpoint layout changed")
pre_rec,pre_u,pre_f,pre_self,pre_top=exact_mesh(world,faces,ci,cp)
require(pre_self["count"]==0 and pre_top["closed_oriented_2_manifold"],"post-bone checkpoint fails exact self/topology")
vol_before=vol(world,faces); require(vol_before>0,"post-bone checkpoint has nonpositive signed volume")
pack0=PACKDIR/"step-0.mrvpack"; rec0=json.loads((PACKDIR/"step-0.receipt.json").read_text())
require(rec0.get("accepted_step")==0 and rec0.get("physical_endpoint")=="accepted","step0 capture invalid")
poses0={int(v["body_index"]):(np.asarray(v["position_m"],dtype="<f4"),np.asarray(v["quaternion_xyzw"],dtype="<f4")) for v in rec0["accepted_body_poses"]}
keys0=parse_keys(pack0,ca); packed0,targets0,_=ca._pack_surfaces(pack0,keys0)
require(set(targets0)==keys0 and (SEM,160) in targets0,"step0 pack missing complete target or stable160")
target_v=np.asarray(packed0,dtype="<f4"); tf=np.asarray(targets0[(SEM,160)]["faces"],dtype=np.int64)
used=np.unique(tf.reshape(-1)); remap=np.full(len(target_v),-1,dtype=np.int64); remap[used]=np.arange(len(used))
target_v=np.ascontiguousarray(target_v[used],dtype="<f4"); tf=np.ascontiguousarray(remap[tf],dtype=np.int64)
shell, shell_faces, source_shell_report, offset_shell_report, source_shell_vol, offset_shell_vol = signed_shell(target_v,tf,ca,"left superficial masseter stable160")
shell_sha=hashlib.sha256(shell.tobytes()+np.asarray(shell_faces,dtype="<i8").tobytes()).hexdigest()
# Existing exact arrangement owner; input checkpoint already contains bone cuts, so only the stable160 cut runs here.
src={"deep_bone_cleared":{"vertices":exact_rational(world),"triangles":faces.tolist()},
     "left_superficial_masseter_10um_shell":{"vertices":exact_rational(shell),"triangles":shell_faces.tolist()}}
signal.alarm(1200)
rows,arr_meta=cp.construct_arrangement(src,source_names=("deep_bone_cleared","left_superficial_masseter_10um_shell"))
signal.alarm(0)
selected=[]; new_ancestry=[]
for child in rows:
 if child["source"]=="deep_bone_cleared" and child["other_location"]=="outside":
  selected.append(tuple(child["vertices"])); new_ancestry.append(dict(ancestry[int(child["source_face"])]))
 elif child["source"]=="left_superficial_masseter_10um_shell" and child["other_location"]=="inside":
  tri=tuple(child["vertices"]); selected.append((tri[0],tri[2],tri[1]))
  new_ancestry.append({"kind":"stable160_offset_cap","source_target_face":int(child["source_face"]),"reversed":True,"offset_um":10.0})
require(bool(selected),"empty difference against stable160")
mesh=cp.indexed_mesh(selected,convert=lambda x:x)
vworld=np.asarray([[np.float32(float(x)) for x in p] for p in mesh["vertices_m"]],dtype="<f4")
fworld=np.asarray(mesh["triangles"],dtype=np.int64)
_,_,_,selfw,topw=exact_mesh(vworld,fworld,ci,cp)
require(selfw["count"]==0 and topw["closed_oriented_2_manifold"],"step0 Float32 output failed exact topology/self gate")
if vol(vworld,fworld)<0:
 fworld=fworld[:,[0,2,1]]
 new_ancestry=[dict(x,face_order_reversed_for_outward_winding=True) for x in new_ancestry]
vol_world_after=vol(vworld,fworld)
require(vol_world_after>0,"third-cut output signed volume not positive")
# Inverse the same accepted head pose/binding operation to store source-local ABI5 row.
p0,q0=poses0[CORE]; rot=ca._rotation(q0.astype(np.float64)); bindval=np.asarray(tissue["bindings"][row159["fb"]]["value"],dtype="<f4")
rb=ca._rotation(bindval[3:7].astype(np.float64))
body_local=(vworld.astype(np.float64)-p0.astype(np.float64))@rot
bp=((body_local-bindval[:3].astype(np.float64))@rb)/float(bindval[7])
pos,first,inv=np.unique(np.asarray(bp,dtype="<f4"),axis=0,return_index=True,return_inverse=True)
fsource=inv[fworld]
bidx=np.zeros((len(pos),4),dtype="<u4"); bidx[:,1:]=np.uint32(0xffffffff)
weights=np.zeros((len(pos),4),dtype="<f4"); weights[:,0]=1.
replay0=h.forward(pos,bidx,weights,row159,tissue,poses0).astype("<f4")
max_replay=float(np.linalg.norm(replay0[inv].astype(float)-vworld.astype(float),axis=1).max())
require(max_replay<1e-6,"inverse/forward step0 error exceeds 1um")
# Check source space closed/self/positive volume.
_,_,_,src_a,src_top=exact_mesh(pos,fsource,ci,cp)
vol_source=vol(pos,fsource)
require(src_a["count"]==0 and src_top["closed_oriented_2_manifold"] and vol_source>0,"source-space third-cut candidate failed exact gates")
npzout=OUT/"stable-161-deep-left-bone-and-superficial-clearance.npz"
np.savez_compressed(npzout,positions_m=pos.astype("<f4"),faces=fsource.astype("<i8"),binding_indices=bidx,weights=weights,
 step0_world_f32=replay0,face_ancestry_json=np.asarray(json.dumps(new_ancestry,sort_keys=True)),stable_id=np.asarray([STABLE],dtype="<u4"),
 core_body_index=np.asarray([CORE],dtype="<u4"),binding_record_sha256=np.asarray([hashlib.sha256(row159["binding_bytes"]).hexdigest()]))
csgout=OUT/"stable-161-deep-left-third-cut-world-f32.npz"
np.savez_compressed(csgout,positions_m=vworld.astype("<f4"),faces=fworld.astype("<i8"),face_ancestry_json=np.asarray(json.dumps(new_ancestry,sort_keys=True)))
pair_rows=[]; pose_rows=[]
for step in STEPS:
 pack=PACKDIR/f"step-{step}.mrvpack"; receipt_path=PACKDIR/f"step-{step}.receipt.json"
 receipt=json.loads(receipt_path.read_text())
 require(receipt.get("accepted_step")==step and receipt.get("physical_endpoint")=="accepted" and receipt.get("pack_file_sha256")==sha(pack),"accepted capture receipt mismatch at "+str(step))
 poses={int(x["body_index"]):(np.asarray(x["position_m"],dtype="<f4"),np.asarray(x["quaternion_xyzw"],dtype="<f4")) for x in receipt["accepted_body_poses"]}
 require(len(poses)==157 and CORE in poses,"accepted pose table incomplete")
 w=h.forward(pos,bidx,weights,row159,tissue,poses).astype("<f4")
 if step==0: require(np.array_equal(w,replay0),"step0 replay changed")
 mr,uq,qf,sa,top=exact_mesh(w,fsource,ci,cp)
 require(sa["count"]==0 and top["closed_oriented_2_manifold"],"pose self/topology gate failed at "+str(step))
 keys=parse_keys(pack,ca); xyz,td,_=ca._pack_surfaces(pack,keys)
 require(set(td)==keys,"target surface pack is incomplete at "+str(step))
 audits=ca._target_intersection_audit(mr,{k:np.asarray(x["faces"],dtype=np.int64) for k,x in td.items()},xyz)
 require(set(audits)=={f"{x[0]}:{x[1]}" for x in keys},"target audit returned incomplete keys at "+str(step))
 nz={k:{"count":int(v["count"]),"pairs":v["triangle_pairs"]} for k,v in audits.items() if int(v["count"])}
 pair_rows.append({"step":step,"pairs":{k:v["triangle_pairs"] for k,v in audits.items() if int(v["count"])}})
 bone_counts={f"51004:{i}":int(audits[f"51004:{i}"]["count"]) for i in (31,35)}
 pose_rows.append({"step":step,"receipt_sha256":sha(receipt_path),"world_f32_sha256":hashlib.sha256(w.tobytes()).hexdigest(),
  "self_pairs":int(sa["count"]),"topology":top,"target_count":len(audits),"total_target_pairs":sum(int(v["count"]) for v in audits.values()),
  "bone31_and35_pairs":bone_counts,"nonzero_targets":{k:int(v["count"]) for k,v in nz.items()}})
 print("pose",step,"self",sa["count"],"target_pairs",pose_rows[-1]["total_target_pairs"],"nonzero",pose_rows[-1]["nonzero_targets"],flush=True)
pairsout=OUT/"pairs-by-pose.json"; pairsout.write_text(json.dumps(pair_rows,sort_keys=True,separators=(",",":"))+"\n")
require(all(sha(p)==v for p,v in pins.items()),"original pinned inputs changed during third cut")
source_ancestry=Counter(str(a.get("kind","unknown")) for a in new_ancestry)
report={"schema":"offline.deep-masseter-third-shell-subtraction.v1",
 "status":"candidate_review_required",
 "operation":"One exact CSG subtraction from the saved F32 world checkpoint after temporal31 and mandible35 cuts. Target is stable160 left superficial masseter expanded outward by10um. The first two cuts were not recomputed.",
 "inputs_sha256":{p:v for p,v in pins.items()},
 "retained_prior_outputs_sha256":out_pins,
 "inputs_unchanged":True,
 "step0_target_surface":{"semantic":SEM,"stable_id":160,"name":"left superficial masseter","raw_face_count":len(np.asarray(targets0[(SEM,160)]["faces"])),"closed_shell_report":source_shell_report,"offset_shell_report":offset_shell_report,"offset_um":10.0,"shell_sha256":shell_sha},
 "prior_bone_cleared_checkpoint":{"path":str(CHECKPOINT),"sha256":sha(CHECKPOINT),"input_vertices":len(world),"input_faces":len(faces),"signed_volume_world_m3":vol_before,"exact_self_pairs":int(pre_self["count"]),"closed_oriented":bool(pre_top["closed_oriented_2_manifold"])},
 "arrangement":{"metadata":arr_meta,"output_vertices_world":len(vworld),"output_faces":len(fworld),"source_face_ancestry_counts":dict(source_ancestry),
  "world_signed_volume_before_m3":vol_before,"world_signed_volume_after_m3":vol_world_after,"world_volume_removed_m3":vol_before-vol_world_after,
  "world_volume_removed_fraction":(vol_before-vol_world_after)/vol_before if vol_before else None,"source_frame_signed_volume_after_m3":vol_source,
  "step0_float32_exact_self_pairs":int(selfw["count"]),"step0_topology":topw,"source_exact_self_pairs":int(src_a["count"]),
  "source_topology":src_top,"step0_inverse_forward_max_error_m":max_replay},
 "pose_audits":{"count":len(pose_rows),"rows":pose_rows,"all_self_clear":all(r["self_pairs"]==0 for r in pose_rows),
  "all_targets_clear":all(r["total_target_pairs"]==0 for r in pose_rows)},
 "candidate":{"path":str(npzout),"sha256":sha(npzout),"world_checkpoint_path":str(csgout),"world_checkpoint_sha256":sha(csgout),
  "faces":len(fsource),"vertices":len(pos),"binding_core":CORE,"one_hot_binding":True,"binding_sha256":hashlib.sha256(row159["binding_bytes"]).hexdigest()},
 "pair_tables":{"path":str(pairsout),"sha256":sha(pairsout)},
 "attachment_markers":{"source_marker_object_sha256":"224a3b164b9ddf3c6f903d6aca771cdf3aaeb53e28167242cf73702cbbb3ce2e",
  "origin_marker_points":55,"insertion_marker_points":36,"marker_geometry_changed":False,"marker_objects_in_candidate_mesh":False,
  "interpretation":"Markers are retained as unchanged source evidence and were not converted into masks. Candidate does not demonstrate preserved attachment footprints."},
 "limitations":["No native scene or source TISS was modified.","The 10um offset is an inferred visualization reserve, not measured anatomical spacing.","This operation trims an overlap between named deep and superficial masseter surfaces; it is not a source measurement or validated shared interface.","No native/GPU qualification; eight MTP0 accepted poses only."]}
report_path=OUT/"report.json"; report_path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":report["status"],"report":str(report_path),"sha256":sha(report_path),"candidate":str(npzout),"candidate_sha256":sha(npzout),
 "faces":len(fsource),"vertices":len(pos),"source_volume_m3":vol_source,"world_volume_removed_m3":vol_before-vol_world_after,
 "poses":[[r["step"],r["self_pairs"],r["total_target_pairs"],r["nonzero_targets"]] for r in pose_rows],
 "elapsed_seconds":time.monotonic()-start},sort_keys=True),flush=True)
