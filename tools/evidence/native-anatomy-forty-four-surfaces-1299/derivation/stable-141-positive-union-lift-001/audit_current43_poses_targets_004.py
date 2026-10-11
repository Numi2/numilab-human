from pathlib import Path
import json,hashlib,importlib.util,sys,time,math,os
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009");A=R/"anatomy-completion-1276"
BASE=A/"stable-141-positive-union-lift-001";OUT=BASE/"current43-target-audit-001";OUT.mkdir(exist_ok=False)
CAND=BASE/"local-reference-001/result-001/stable-141-position-correction-iteration-0.npz"
TISS=A/"forty-three-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
TISS42=A/"forty-two-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MAN=TISS.with_suffix(".manifest.json"); INV=Path("/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
CI=Path("/Users/n/numi-human-touching-loop-arrangement-1290/src/numilab_human/cardiac_cavity_intersections.py")
CG=Path("/Users/n/numi-human-touching-loop-arrangement-1290/src/numilab_human/cardiac_cavity_geometry.py")
CA=Path("/Users/n/numi-human-touching-loop-arrangement-1290/src/numilab_human/common_atlas_skin_clearance.py")
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
C43=A/"forty-three-surface-native-composition-001/baseline/native-run/accepted-geometry"
C42=A/"forty-two-surface-native-composition-001/baseline/native-run/accepted-geometry"
E28=A/"twenty-eight-surface-native-composition-001/baseline/native-run/accepted-geometry"
PAIR=R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011"
S42=[0,4767,5023,5599,6207,6815,7423,8000];S28=S42.copy();SLATE=[0,47519,151999,152607,153215,153823,154431,155000]
def sha(p):
 q=hashlib.sha256()
 with Path(p).open("rb") as fh:
  for b in iter(lambda:fh.read(8*1024*1024),b""):q.update(b)
 return q.hexdigest()
def req(x,m):
 if not x:raise RuntimeError(m)
def save(name,obj):
 q=OUT/(name+".tmp")
 with q.open("w") as f:json.dump(obj,f,indent=2,sort_keys=True,default=lambda z:z.item() if isinstance(z,np.generic) else str(z));f.write("\n");f.flush();os.fsync(f.fileno())
 q.replace(OUT/(name+".json"))
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg,common_atlas_skin_clearance as ca
req(sha(CI)=="90c7a7fb0ce4383397f97519dd5fe163201d86a0c97e18e3a9eb3fa9858cd561","predicate pin")
sp=importlib.util.spec_from_file_location("positive_forward",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h);h.TISS=TISS;h.MAN=MAN
tissue=h.load_tissue(TISS);row=h.row_data(tissue,141)
tissue42=h.load_tissue(TISS42); row42=h.row_data(tissue42,141)
for field in ("positions","faces","local","weights"):
 req(np.array_equal(np.asarray(row[field]),np.asarray(row42[field])),f"stable141 row {field} differs between42 and43 composition")
z=np.load(CAND,allow_pickle=False);p=np.ascontiguousarray(z["vertices6"][:,:3],dtype="<f4");local=np.asarray(z["binding_indices"],dtype="<u4");weights=np.asarray(z["weights"],dtype="<f4");faces=np.asarray(z["faces"],dtype="<i8");orig=np.asarray(z["face_origins"],dtype="<i8")
req(p.shape==(len(local),3) and weights.shape==local.shape and faces.shape==(len(orig),3),"candidate arrays")
req(np.all(local[weights>0]<row["bc"]),"active local binding IDs outside row")
req(np.all(local[weights<=0]==np.uint32(0xffffffff)),"inactive local lane sentinel mismatch")
req(np.all(orig>=0)&np.all(orig<len(row["faces"])),"face ancestry out of range")
keys,*_=ca._load_target_inventory(INV);req(len(keys)==859 and (51005,141) in keys,"859 target inventory does not include stable141")
man=json.loads(MAN.read_text());mrow=next(x for x in man["source"]["surfaces"] if int(x["stable_id"])==141)
req(sha(CAND)=="9419cefced52a15fee902fc6d1e47cd20f44c73d7dcfb0a1076e6374d181844d","candidate pin")
input_paths=[Path(__file__),CAND,TISS,TISS42,MAN,INV,CI,CG,CA,H,
 A/"existing-body-141-positive-union-002/result.json",A/"existing-body-141-positive-union-002/stable-141-exact-union.json"]
pin={str(x):sha(x) for x in input_paths}
source_rec,source_deg=h.exact_rows(p,faces,ci)
source_self=None if source_deg else int(ci._audit_pair(source_rec,source_rec,same_surface=True)["count"])
source_top=cg.analyze_topology(p.astype(float).tolist(),faces.tolist())
oldrec,olddeg=h.exact_rows(row["positions"],row["faces"],ci)
oldself=None if olddeg else int(ci._audit_pair(oldrec,oldrec,same_surface=True)["count"])
poses=[]
for grp,arm,steps,dirname in [
 ("current43","baseline",S42,C43),
 ("legacy24-early","baseline",S28,E28),
 ("legacy24-baseline-late","baseline",SLATE,PAIR/"baseline/native-run/accepted-geometry"),
 ("legacy24-intervention-late","intervention",SLATE,PAIR/"intervention/native-run/accepted-geometry")]:
 for step in steps:
  pack=dirname/f"step-{step}.mrvpack";receipt=dirname/f"step-{step}.receipt.json"
  req(pack.is_file() and receipt.is_file(),f"missing accepted capture {grp}/{step}")
  rc=json.loads(receipt.read_text());ph=sha(pack)
  req((rc.get("pack_file_sha256") or rc.get("accepted_pack_file_sha256"))==ph,f"capture pack pin mismatch {grp}/{step}")
  pr=rc.get("accepted_body_poses") or rc.get("accepted_registered_body_poses")
  req(isinstance(pr,list) and len(pr) in (86,157),f"unexpected accepted pose count {grp}/{step}")
  posemap={int(x["body_index"]):(np.asarray(x["position_m"],dtype=np.float32),np.asarray(x["quaternion_xyzw"],dtype=np.float32)) for x in pr}
  needed_cores={int(tissue["bindings"][row["fb"]+int(li)]["core"]) for li in np.unique(local[weights>0])}
  req(needed_cores.issubset(posemap),f"accepted pose map omits active stable141 binding cores {grp}/{step}: {sorted(needed_cores-set(posemap))}")
  poses.append({"group":grp,"arm":arm,"step":step,"pack":pack,"receipt":receipt,"posemap":posemap,"pack_sha":ph,"receipt_sha":sha(receipt),"accepted_pose_count":len(pr)})
  pin[str(pack)]=ph;pin[str(receipt)]=sha(receipt)
rows=[];current43_context={}
start=time.monotonic()
for pose in poses:
 world=np.ascontiguousarray(h.forward(p,local,weights,row,tissue,pose["posemap"]),dtype="<f4")
 rec,deg=h.exact_rows(world,faces,ci)
 cnt=None if deg else int(ci._audit_pair(rec,rec,same_surface=True)["count"])
 top=cg.analyze_topology(world.astype(float).tolist(),faces.tolist())
 brec,bdeg=h.exact_rows(h.forward(row["positions"],row["local"],row["weights"],row,tissue,pose["posemap"]),row["faces"],ci)
 bcnt=None if bdeg else int(ci._audit_pair(brec,brec,same_surface=True)["count"])
 item={"group":pose["group"],"arm":pose["arm"],"step":pose["step"],"pack_sha256":pose["pack_sha"],"receipt_sha256":pose["receipt_sha"],"accepted_pose_count":pose["accepted_pose_count"],"baseline_self_pair_count":bcnt,"candidate_self_pair_count":cnt,"candidate_degenerate_face_count":len(deg),"candidate_closed_oriented":bool(top.get("closed_oriented_manifold_candidate")) and not deg,"candidate_topology":{k:top.get(k) for k in ("closed_oriented_manifold_candidate","face_component_count","euler_characteristic","boundary_edge_count","nonmanifold_edge_count","orientation_conflict_edge_count")}}
 rows.append(item)
 if pose["group"]=="current43":current43_context[pose["step"]]=(world,pose)
 save("progress",{"schema":"stable141-pose-target-progress.v1","last_pose":pose["step"],"group":pose["group"],"rows":rows,"pins":pin})
 print(json.dumps({"group":pose["group"],"step":pose["step"],"baseline_self":bcnt,"candidate_self":cnt,"degenerate":len(deg),"closed":item["candidate_closed_oriented"]},sort_keys=True),flush=True)
self_ok=source_self==0 and not source_deg and bool(source_top.get("closed_oriented_manifold_candidate")) and all(x["candidate_self_pair_count"]==0 and x["candidate_degenerate_face_count"]==0 and x["candidate_closed_oriented"] for x in rows)
target_audits=[]
if self_ok:
 used_src=np.unique(row["faces"]); origcnt=np.bincount(orig,minlength=len(row["faces"]))
 same=np.zeros(len(orig),dtype=bool)
 for shift in range(3):
  cf=np.roll(faces,shift,axis=1)
  same |= np.all(row["positions"][row["faces"][orig]]==p[cf],axis=(1,2)) & np.all(row["local"][row["faces"][orig]]==local[cf],axis=(1,2)) & np.all(row["weights"][row["faces"][orig]]==weights[cf],axis=(1,2))
 changed=set(np.flatnonzero(origcnt!=1).tolist())|set(np.flatnonzero(~same).astype(int).tolist())
 changed.update(set(range(len(row["faces"])))-set(map(int,orig)))
 before=np.asarray(sorted(changed),dtype=np.int64);after=np.flatnonzero(np.isin(orig,before))
 req(len(before)>0 and len(after)>0,"no changed parent face star")
 for step,(world,pose) in current43_context.items():
  pack=pose["pack"];positions,surfaces,_=ca._pack_surfaces(pack,keys)
  req(set(keys).issubset(surfaces) and set(surfaces)-set(keys)=={(51007,1)},f"target keyset differs {step}")
  cface=np.asarray(surfaces[(51005,141)]["faces"],dtype=np.int64);df=cface-row["faces"]
  req(cface.shape==row["faces"].shape and np.all(df==df.flat[0]),f"stable141 capture topology/row offset mismatch {step}")
  actual=positions[int(df.flat[0]):int(df.flat[0])+row["vc"]]
  baseworld=h.forward(row["positions"],row["local"],row["weights"],row,tissue,pose["posemap"])
  replay=float(np.linalg.norm(baseworld[np.unique(row["faces"])].astype(np.float64)-actual[np.unique(row["faces"])].astype(np.float64),axis=1).max(initial=0))
  brec,bdeg=h.exact_rows(actual,row["faces"][before],ci);crec,cdeg=h.exact_rows(world,faces[after],ci)
  req(not bdeg and not cdeg,f"changed-face submesh degenerate {step}")
  pts=np.concatenate([actual[np.unique(row["faces"][before])],world[np.unique(faces[after])]],axis=0);lo=pts.min(0);hi=pts.max(0)
  aud={"step":step,"baseline_forward_capture_max_error_m":replay,"target_keys_scanned":0,"baseline_exact_pairs":0,"candidate_exact_pairs":0,"added_parent_pairs":[],"removed_parent_pairs":[],"per_target":[]}
  for key,tgt in surfaces.items():
   if key==(51005,141):continue
   tf=np.asarray(tgt["faces"],dtype=np.int64);tri=positions[tf]
   selected=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1))
   aud["target_keys_scanned"]+=1
   if not len(selected):continue
   trec,tdeg=h.exact_rows(positions,tf[selected],ci);req(not tdeg,f"target {key} broadphase has degenerate faces")
   ba=ci._audit_pair(brec,trec,same_surface=False);aa=ci._audit_pair(crec,trec,same_surface=False)
   bp={(int(before[x]),int(selected[y])) for x,y in ba["triangle_pairs"]}
   ap={(int(orig[after[x]]),int(selected[y])) for x,y in aa["triangle_pairs"]}
   added=sorted(ap-bp);removed=sorted(bp-ap)
   aud["baseline_exact_pairs"]+=len(bp);aud["candidate_exact_pairs"]+=len(ap)
   if added or removed:
    aud["per_target"].append({"semantic":int(key[0]),"stable_id":int(key[1]),"body":tgt.get("body"),"baseline_parent_pairs":len(bp),"candidate_parent_pairs":len(ap),"added_parent_pairs":added,"removed_parent_pairs":removed})
    aud["added_parent_pairs"].extend([[int(key[0]),int(key[1]),a,b] for a,b in added]);aud["removed_parent_pairs"].extend([[int(key[0]),int(key[1]),a,b] for a,b in removed])
  req(aud["target_keys_scanned"]==859,f"not all 859 targets visited at {step}")
  aud["no_new_parent_pairs"]=len(aud["added_parent_pairs"])==0
  target_audits.append(aud)
  save("progress",{"schema":"stable141-pose-target-progress.v1","last_target_step":step,"rows":rows,"target_audits":target_audits,"pins":pin})
  print(json.dumps({"target_step":step,"targets":aud["target_keys_scanned"],"baseline_pairs":aud["baseline_exact_pairs"],"candidate_pairs":aud["candidate_exact_pairs"],"added":len(aud["added_parent_pairs"]),"removed":len(aud["removed_parent_pairs"]),"replay_m":replay},sort_keys=True),flush=True)
complete={"schema":"stable141-current43-and-legacy24-self-target-audit.v1","scope":"Offline candidate audit only: exact self and topology over current43 eight accepted poses plus the retained older24 pose set (early8 and 8+8 late baseline/intervention); changed-face-star against all859 current43 external target keys. No native/composition admission.","stable_id":141,"label":mrow["label"],"member_id":mrow["member_id"],"candidate_path":str(CAND),"candidate_sha256":sha(CAND),"source_union_candidate_path":str(A/"existing-body-141-positive-union-002/stable-141-exact-union.json"),"source_union_candidate_sha256":sha(A/"existing-body-141-positive-union-002/stable-141-exact-union.json"),"candidate_source_self_pair_count":source_self,"candidate_source_degenerate_faces":source_deg,"candidate_source_closed_oriented":bool(source_top.get("closed_oriented_manifold_candidate")),"baseline_source_self_pair_count":oldself,"poses":rows,"source_and_all32_sampled_pose_self_clear":self_ok,"current43_target_audits":target_audits,"all_current43_859_target_keys_scanned":self_ok and len(target_audits)==8 and all(x["target_keys_scanned"]==859 for x in target_audits),"no_added_parent_pairs_against_current43_targets":self_ok and len(target_audits)==8 and all(x["no_new_parent_pairs"] for x in target_audits),"all_baseline_forward_capture_replays_within_3e-7m":all(x.get("baseline_forward_capture_max_error_m",0)<=3e-7 for x in target_audits),"stable141_row_arrays_identical_42_to_43":True,"stable141_row_42_global_vertex_offset":int(row42["fv"]),"stable141_row_43_global_vertex_offset":int(row["fv"]),"stable141_row_global_face_offset":int(row["fb"]),"legacy24_pose_set_rechecked_for_current_candidate":True,"older310s_pose_maps_use_86_registered_bodies_and_each_candidate_row_active_core_is_required_present":True,"target_pair_count_semantics":"New and removed external rows are source-parent face pairs (candidate descendants map through explicit original face ancestry); exact candidate triangle-pair counts and target IDs are retained per pose.","complete":True,"inputs_unchanged":all(sha(k)==v for k,v in pin.items()),"pins":pin,"elapsed_seconds":time.monotonic()-start,"candidate_admitted":False}
save("report",complete);print(json.dumps({k:complete[k] for k in ("stable_id","candidate_sha256","source_and_all32_sampled_pose_self_clear","all_current43_859_target_keys_scanned","no_added_parent_pairs_against_current43_targets","all_baseline_forward_capture_replays_within_3e-7m","complete","inputs_unchanged","elapsed_seconds")},sort_keys=True),flush=True)
