from pathlib import Path
import hashlib,json,math
import numpy as np
OUT=Path("/Users/n/numi-human-retained-delivery-20261009/native-distributed-bed-smoke-1199/terminal-pose-comparison-1199")
RUNS={"1191":Path("/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191/native-run"),"1198":Path("/Users/n/numi-human-retained-delivery-20261009/native-contoured-bed-smoke-1198-attempt002/native-run"),"1199":Path("/Users/n/numi-human-retained-delivery-20261009/native-distributed-bed-smoke-1199/native-run")}
BODY=Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-reference.manifest.json")
MANIFEST=Path("/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/resting-supine-scene-distributed-support.manifest.json")
SUPPORT=Path("/Users/n/numi-human-retained-delivery-20261009/distributed-resting-support-1199/myosim-fullbody-distributed-rigid-digit-support.nhcnt")
BED_AUDIT=Path("/Users/n/numi-human-retained-delivery-20261009/native-distributed-bed-smoke-1199/bed-mesh-capture-audit-1199/bed-mesh-capture-audit.json")
BED_SCRIPT=Path("/Users/n/numi-human-retained-delivery-20261009/native-distributed-bed-smoke-1199/bed-mesh-capture-audit-1199/audit_bed_mesh_captures_1199.py")
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def rot(q):
 x,y,z,w=map(float,q); n=math.sqrt(x*x+y*y+z*z+w*w); x,y,z,w=x/n,y/n,z/n,w/n
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]],float)
def deg(R): return math.degrees(math.acos(max(-1.,min(1.,(np.trace(R)-1.)/2.))))
def load(run,step):
 p=run/"accepted-geometry"/f"step-{step}.receipt.json"; d=read(p)
 assert d["accepted_step"]==step and len(d["accepted_registered_body_poses"])==86
 return p,d,{int(x["body_index"]):x for x in d["accepted_registered_body_poses"]}
data={lab:{s:load(r,s) for s in (0,10000)} for lab,r in RUNS.items()}
first=lambda k:[data[x][0][1][k] for x in ("1191","1198","1199")]
assert len(set(first("accepted_body_state_sha256")))==1
assert len(set(first("accepted_respiration_state_sha256")))==1
assert all(x==first("accepted_registered_body_poses")[0] for x in first("accepted_registered_body_poses")[1:])
order=read(BODY)["core_tree"]["body_order"]
idx={n:order.index(n) for n in ["pelvis","femur_r","femur_l","tibia_r","tibia_l","talus_r","talus_l","calcn_r","calcn_l","torso","humerus_r","humerus_l","ulna_r","ulna_l","radius_r","radius_l"]}
pairs=[("pelvis","femur_r"),("pelvis","femur_l"),("femur_r","tibia_r"),("femur_l","tibia_l"),("tibia_r","talus_r"),("tibia_l","talus_l"),("talus_r","calcn_r"),("talus_l","calcn_l"),("torso","humerus_r"),("torso","humerus_l"),("humerus_r","ulna_r"),("humerus_l","ulna_l"),("ulna_r","radius_r"),("ulna_l","radius_l")]
relative=[]
for pa,ch in pairs:
 pi,ci=idx[pa],idx[ch]; row={"pair":pa+"->"+ch,"parent_index":pi,"child_index":ci}; mats={}
 for lab in RUNS:
  row[lab]={}
  for step in (0,10000):
   pp=data[lab][step][2]; p,c=pp[pi],pp[ci]; Rp,Rc=rot(p["quaternion_xyzw"]),rot(c["quaternion_xyzw"])
   t=Rp.T@(np.array(c["position_m"],float)-np.array(p["position_m"],float)); R=Rp.T@Rc; mats[(lab,step)]=(R,t)
   row[lab][str(step)]={"relative_rotation_deg_from_identity":deg(R),"relative_translation_parent_frame_mm":(t*1000).tolist()}
 row["rotation_change_step0_to_terminal_deg"]={lab:deg(mats[(lab,0)][0].T@mats[(lab,10000)][0]) for lab in RUNS}
 row["terminal_1199_minus_reference"]={}
 for lab in ("1198","1191"):
  Ra,ta=mats[(lab,10000)]; Rb,tb=mats[("1199",10000)]
  row["terminal_1199_minus_reference"][lab]={"relative_rotation_delta_deg":deg(Ra.T@Rb),"relative_translation_parent_frame_delta_mm":((tb-ta)*1000).tolist()}
 relative.append(row)
deltas={}
for ref in ("1198","1191"):
 a,b=data[ref][10000][2],data["1199"][10000][2]; items=[]
 for i in sorted(a):
  dp=(np.asarray(b[i]["position_m"],float)-np.asarray(a[i]["position_m"],float))*1000
  dr=deg(rot(a[i]["quaternion_xyzw"]).T@rot(b[i]["quaternion_xyzw"]))
  items.append({"index":i,"name":order[i],"translation_delta_mm":dp.tolist(),"translation_delta_norm_mm":float(np.linalg.norm(dp)),"orientation_delta_deg":dr})
 items.sort(key=lambda x:x["translation_delta_norm_mm"],reverse=True)
 deltas[ref]={"top12_by_translation":items[:12],"maximum_translation_delta_mm":items[0]["translation_delta_norm_mm"],"maximum_orientation_delta_deg":max(x["orientation_delta_deg"] for x in items)}
runinfo={}; inputs=[]
for lab,r in RUNS.items():
 caps={}
 for s in (0,10000):
  rp,rec,_=data[lab][s]; caps[str(s)]={"body_state_sha256":rec["accepted_body_state_sha256"],"respiration_state_sha256":rec["accepted_respiration_state_sha256"],"accepted_time_s":rec["accepted_time_s"],"root_fingerprint":rec.get("accepted_root_fingerprint"),"receipt_sha256":sha(rp)}
  inputs += [rp,r/"accepted-geometry"/f"step-{s}.mrvpack"]
 inputs += [r/"invocation.json",r/"run-metadata.json",r/"native.log",r.parent/"run-declaration.json"]
 runinfo[lab]={"path":str(r),"captures":caps}
input_hashes={str(p):sha(p) for p in inputs}
for p in (BODY,MANIFEST,SUPPORT,BED_AUDIT,BED_SCRIPT): input_hashes[str(p)]=sha(p)
report={"schema":"numi.human.distributed-bed-terminal-pose-comparison.v1","scope":"Accepted registered body-pose comparison at initial step 0 and terminal step 10000 across runs 1191, 1198, and 1199. It is not a generalized-coordinate q audit or full-cycle anatomical clearance qualification.","runs":runinfo,"initial_identity":{"body_state_equal":True,"respiration_state_equal":True,"all_86_registered_body_pose_records_equal":True,"root_fingerprints_equal":len({data[x][0][1].get("accepted_root_fingerprint") for x in RUNS})==1,"note":"Root fingerprints are recorded separately; body/respiration hashes and all registered pose records are the direct initial identity checks."},"terminal_pose_deltas":deltas,"relative_segment_pose_comparison":relative,"initial_bed_and_nonbed_geometry_transfer":{"audit_path":str(BED_AUDIT),"audit_sha256":sha(BED_AUDIT),"scope":"1199 step-0 pack directly matched all 860 non-bed primitive records, index streams, instance records and non-bed vertex prefix to 1198 step-0. The complete bed records/index/instance matched 1198 step-0 and the retained 1196 runtime Float32 grid; bed remained bit-identical at steps 0 and 10000."},"limitations":["Relative transforms derive from accepted registered rigid-body poses, not scalar joint coordinates or joint-limit tests.","Only steps 0 and 10000 are compared; no intervening anatomy scan is included.","This report does not attribute terminal pose divergence to a particular contact or solver cause."],"source_hashes":input_hashes}
script=Path(__file__); report["analysis_script"]={"path":str(script),"sha256":sha(script)}
OUT.mkdir(parents=True,exist_ok=True)
out=OUT/"pose-comparison.json"; out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"report":str(out),"sha256":sha(out),"initial_identity":report["initial_identity"],"terminal_deltas":{k:{"max_translation_mm":v["maximum_translation_delta_mm"],"max_orientation_deg":v["maximum_orientation_delta_deg"],"top6":[x["name"] for x in v["top12_by_translation"][:6]]} for k,v in deltas.items()}},indent=2))
