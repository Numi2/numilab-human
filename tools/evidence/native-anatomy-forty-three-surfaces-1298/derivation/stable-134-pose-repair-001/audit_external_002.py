from pathlib import Path
import sys,json,hashlib,importlib.util,time,signal
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009"); A=R/"anatomy-completion-1276"
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
BASE=A/"remaining-limb-positive-union-lift-001/stable-134-reference-union-row-patch.npz"
CAND=A/"stable-134-pose-repair-001/local-reference-search-001/stable-134-unadmitted-local-reference.npz"
SELF=A/"stable-134-pose-repair-001/local-reference-search-001/report.json"
T=R/"source-seam-connectivity-1247/fhl-current-7b23-count-reconciliation-1258/attempt-002/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
OUT=Path(__file__).with_name("target-audit-002"); OUT.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-self-separation-partial-resume-1267/src")
from numilab_human import cardiac_cavity_intersections as ci, cardiac_cavity_geometry as cg
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import common_atlas_skin_clearance as ca
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sp=importlib.util.spec_from_file_location("fw",H); h=importlib.util.module_from_spec(sp); sp.loader.exec_module(h); h.TISS=T; h.MAN=T.with_suffix(".manifest.json")
t=h.load_tissue(T); row=h.row_data(t,134); z0=np.load(BASE); z=np.load(CAND)
p=z["vertices6"][:,:3].astype(np.float32); local=z["binding_indices"]; weights=z["weights"]; faces=z["faces"].astype(np.int64); origins=z["face_origins"].astype(np.int64)
assert p.shape[0]==len(local)==len(weights) and len(origins)==len(faces)
keys,*_=ca._load_target_inventory(h.INV); assert len(keys)==859
pose_specs=[]; early=A/"twenty-eight-surface-native-composition-001/baseline/native-run/accepted-geometry"
for st in [0,4767,5023,5599,6207,6815,7423,8000]: pose_specs.append(("early",st,early))
pair=R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011"
for arm in ("baseline","intervention"):
 for st in h.STEPS: pose_specs.append((arm,st,pair/arm/"native-run/accepted-geometry"))
pins={str(x):sha(x) for x in [Path(__file__),H,BASE,CAND,SELF,T,T.with_suffix(".manifest.json"),Path(ci.__file__),Path(cg.__file__),Path(ca.__file__),h.INV]}
for r in json.loads(SELF.read_text())["input_sha256"]: pins[r]=json.loads(SELF.read_text())["input_sha256"][r]
report={"schema":"stable-134-positive-union-local-reference-external-audit.v1","scope":"Offline candidate audit only. Uses the exact 24 retained accepted poses, exact current predicate, all 859 external target keys per pose, changed-face ancestry comparison against captured baseline stable134. No composition/native admission.","stable_id":134,"baseline_candidate_sha256":sha(BASE),"candidate_sha256":sha(CAND),"source_tissue_sha256":sha(T),"predicate_sha256":sha(Path(ci.__file__)),"target_inventory_keys":len(keys),"poses":[],"complete":False,"inputs_unchanged":False,"all24_self_clear":False,"all24_external_no_added_parent_pairs":False,"native_admitted":False}
start=time.monotonic();prog=OUT/"progress.jsonl"
def emit(name,**kw):
 obj={"event":name,"elapsed_seconds":time.monotonic()-start,**kw}; prog.open("a").write(json.dumps(obj,sort_keys=True,separators=(",",":"))+"\n"); print(json.dumps(obj,sort_keys=True),flush=True)
def save(): (OUT/"report.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
def alarm(_s,_f): raise TimeoutError("bounded stable134 allpose target audit exceeded 1200 seconds")
signal.signal(signal.SIGALRM,alarm);signal.alarm(1200)
# Changed parent faces include omissions, splits, and any source face whose geometry or attributes changed.
uniq,cnts=np.unique(origins,return_counts=True)
changed=(set(range(len(row["faces"])))-set(map(int,origins)))|set(map(int,uniq[cnts!=1]))
same=np.zeros(len(origins),dtype=bool); rf=row["faces"][origins]
for shift in range(3):
 cf=np.roll(faces,shift,axis=1)
 same |= (np.all(row["positions"][rf]==p[cf],axis=(1,2)) & np.all(row["local"][rf]==local[cf],axis=(1,2)) & np.all(row["weights"][rf]==weights[cf],axis=(1,2)))
changed.update(map(int,origins[~same])); beforeids=np.asarray(sorted(changed),dtype=np.int64); afterids=np.flatnonzero(np.isin(origins,beforeids))
report["changed_face_scope"]={"baseline_face_rows":len(beforeids),"candidate_face_rows":len(afterids),"unique_candidate_parent_face_rows":int(len(np.unique(origins[afterids]))),"candidate_faces_total":len(faces)}
all_self=True; all_external=True
for group,step,directory in pose_specs:
 pack=directory/f"step-{step}.mrvpack"; receipt=directory/f"step-{step}.receipt.json"
 rc=json.loads(receipt.read_text()); assert (rc.get("pack_file_sha256") or rc.get("accepted_pack_file_sha256"))==sha(pack)
 pr=rc.get("accepted_body_poses") or rc.get("accepted_registered_body_poses")
 poses={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in pr}
 world=h.forward(p,local,weights,row,t,poses); crec,cdeg=h.exact_rows(world,faces,ci)
 cself=ci._audit_pair(crec,crec,same_surface=True)["count"] if not cdeg else None
 top=cg.analyze_topology(world.astype(float).tolist(),faces.tolist())
 positions,surfaces,_=ca._pack_surfaces(pack,keys); assert set(keys).issubset(surfaces) and set(surfaces)-set(keys)=={(51007,1)}
 target=surfaces[(51005,134)]; tfull=np.asarray(target["faces"],dtype=np.int64); delta=tfull-row["faces"]; assert np.all(delta==delta.flat[0])
 actual=positions[int(delta.flat[0]):int(delta.flat[0])+row["vc"]]
 base_forward=h.forward(row["positions"],row["local"],row["weights"],row,t,poses)
 used=np.unique(row["faces"]); replay=np.linalg.norm(base_forward[used].astype(float)-actual[used].astype(float),axis=1); replaymax=float(replay.max()) if len(replay) else 0.
 prrow={"group":group,"step":int(step),"pack_sha256":sha(pack),"receipt_sha256":sha(receipt),"self_pair_count":cself,"degenerate_face_count":len(cdeg),"closed_oriented":bool(top.get("closed_oriented_manifold_candidate")),"component_count":top.get("face_component_count"),"euler_characteristic":top.get("euler_characteristic"),"baseline_forward_capture_max_error_m":replaymax,"baseline_replay_pass":replaymax<=3e-7,"target_key_count":len(surfaces),"targets_scanned":0,"added_parent_pair_count":0,"removed_parent_pair_count":0,"interfaces":[]}
 selfok=bool(cself==0 and not cdeg and top.get("closed_oriented_manifold_candidate") and replaymax<=3e-7)
 all_self &= selfok
 if selfok:
  bref,bd=h.exact_rows(actual,row["faces"][beforeids],ci); xrec,xd=h.exact_rows(world,faces[afterids],ci); assert not bd and not xd
  pts=np.concatenate([actual[np.unique(row["faces"][beforeids])],world[np.unique(faces[afterids])]],axis=0); lo=pts.min(axis=0); hi=pts.max(axis=0)
  for key,tgt in surfaces.items():
   if key==(51005,134):continue
   target_faces=np.asarray(tgt["faces"],dtype=np.int64); tri=positions[target_faces]
   sel=np.flatnonzero(np.all(tri.max(axis=1)>=lo,axis=1)&np.all(tri.min(axis=1)<=hi,axis=1)); prrow["targets_scanned"]+=1
   if not len(sel):continue
   trec,td=h.exact_rows(positions,target_faces[sel],ci); assert not td
   ba=ci._audit_pair(bref,trec,same_surface=False); aa=ci._audit_pair(xrec,trec,same_surface=False)
   bp={(int(beforeids[x]),int(sel[y])) for x,y in ba["triangle_pairs"]}
   ap={(int(origins[afterids[x]]),int(sel[y])) for x,y in aa["triangle_pairs"]}
   added=sorted(ap-bp); removed=sorted(bp-ap)
   if added or removed:
    prrow["interfaces"].append({"semantic":int(key[0]),"stable_id":int(key[1]),"body":tgt.get("body"),"baseline_pair_count":len(bp),"candidate_pair_count":len(ap),"added_parent_pairs":added,"removed_parent_pairs":removed})
    prrow["added_parent_pair_count"]+=len(added); prrow["removed_parent_pair_count"]+=len(removed)
  prrow["all_external_keys_scanned"]=prrow["targets_scanned"]==len(surfaces)-1
  extok=bool(prrow["all_external_keys_scanned"] and prrow["added_parent_pair_count"]==0)
 else:
  prrow["all_external_keys_scanned"]=False; extok=False; prrow["external_status"]="skipped_candidate_self_or_replay_gate_failed"
 all_external &= extok
 report["poses"].append(prrow);save()
 emit("pose_complete",group=group,step=int(step),self_count=cself,degenerate=len(cdeg),closed=bool(top.get("closed_oriented_manifold_candidate")),targets=prrow["targets_scanned"],added=prrow["added_parent_pair_count"],removed=prrow["removed_parent_pair_count"],replay_max_m=replaymax)
report.update(complete=True,all24_self_clear=all_self,all24_external_no_added_parent_pairs=all_external,inputs_unchanged=all(sha(k)==v for k,v in pins.items()),input_sha256=pins,elapsed_seconds=time.monotonic()-start)
save();emit("complete",self_clear=all_self,external_no_added=all_external,inputs_unchanged=report["inputs_unchanged"],elapsed=report["elapsed_seconds"])
