from pathlib import Path
import sys,json,importlib.util,traceback,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"pq-left-joint-source-clearance-018";O.mkdir(exist_ok=False)
OWNER=Path("/Users/n/numihuman-fds-joint-active-1321");SNAP=A/"remaining-original-boundary-batch-001/stable-132/joint-target-separation-013/attempt-023/owner-source-snapshot-1321/common_atlas_skin_clearance.py"
sys.path.insert(0,str(OWNER/"src"))
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg
assert Path(ca.__file__).read_bytes()==SNAP.read_bytes()
def mod(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";h=mod(H,"h")
J=A.parent/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-012/solve_terminal_self_pairs_012.py";jo=mod(J,"jo")
N=A/"bilateral-edl-ehl-native-mtp-two-001";T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
C=A/"remaining-original-boundary-batch-001/stable-130/posed-local-separation-004/stable-130-unadmitted-local-candidate.npz"
R=A/"pq-left-current-target-audit-016/report.json";PR=C.parent/"report.json"
z=dict(np.load(C));p=z["vertices6"][:,:3].copy();f=z["faces"];local=z["binding_indices"];w=z["weights"];parents=z["face_origins"]
pins={str(x):h.sha(x) for x in [Path(__file__),H,J,T,C,R,PR,SNAP,Path(ca.__file__),Path(ci.__file__),Path(cg.__file__)]}
t=h.load_tissue(T);row=h.row_data(t,130);audit=json.loads(R.read_text());assert audit["complete"] and audit["inputs_unchanged"] and audit["all_source_and_pose_self_clear"]
fixed=np.flatnonzero(w.max(1)>=.999999)
pairs=sorted({tuple(x) for sample in json.loads(PR.read_text())["pose_samples"] for x in sample["pairs"]})
preserve=np.asarray([pair for pair in pairs if not set(f[pair[0]])&set(f[pair[1]])],int).reshape(-1,2);shared=[pair for pair in pairs if set(f[pair[0]])&set(f[pair[1]])]
samples=[];contexts=[];seedfaces=set(preserve.flat)
report={"scope":"Unadmitted joint source correction of the prior source/eight-pose self-clear PQ130 boundary. Existing local owner with captured current K2 target triangles and actual native parent triangles. Source and pose self-gap preservation are supplied together; one-hot source proxies and all maps fixed. Source increment cap0.5mm; no native or whole-anatomy claim. All25 older contexts and full current targets remain mandatory postchecks.","pins":pins,"complete":False,"contexts":[],"source_preservation_pairs":preserve.tolist(),"shared_pairs_postcheck_only":shared}
def save():
 def cv(v):
  if isinstance(v,np.ndarray):return v.tolist()
  if isinstance(v,np.generic):return v.item()
  raise TypeError(type(v).__name__)
 (O/"report.json").write_text(json.dumps(report,indent=2,default=cv)+"\n")
for e in audit["poses"]:
 step=e["step"];pack=N/f"baseline/native-run/accepted-geometry/step-{step}.mrvpack";rp=pack.with_suffix(".receipt.json");pins[str(pack)]=h.sha(pack);pins[str(rp)]=h.sha(rp)
 rc=json.loads(rp.read_text());assert rc["pack_file_sha256"]==pins[str(pack)]
 pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 cw=h.forward(p,local,w,row,t,pm);rec,deg=h.exact_rows(cw,f,ci);assert not deg and ci._audit_pair(rec,rec,same_surface=True)["count"]==0
 added={(tuple(k),int(i),int(j)) for k,i,j in e["strict_added_parent_pair_ids"]};keys={key for key,_,_ in added}|{(51005,130)}
 pos,surfs,_=ca._pack_surfaces(pack,keys);bfaces=np.asarray(surfs[(51005,130)]["faces"],int);off=bfaces-row["faces"];assert np.all(off==off.flat[0]);bw=pos[int(off.flat[0]):int(off.flat[0])+row["vc"]]
 actual=e["rows"][1]["target_pairs"];ext=[]
 for tr in actual:
  key=tuple(tr["key"])
  for i,j in tr["pairs"]:
   if (key,int(parents[i]),int(j)) not in added:continue
   tri=np.asarray(pos[np.asarray(surfs[key]["faces"][j],int)],np.float32)
   ext.append({"candidate_face_row":int(i),"parent_face_row":int(parents[i]),"target_semantic":key[0],"target_stable_id":key[1],"target_face_id":int(j),"target_triangle_world_positions":tri});seedfaces.add(int(i))
 fi=np.unique(np.r_[np.asarray([v["candidate_face_row"] for v in ext],int),preserve.flat]);ids=np.unique(f[fi]);jac=jo.pose_jacobians(p,local,w,row,t,pm)
 samples.append({"vertex_ids":ids,"baseline_world_positions":cw[ids],"rejected_world_positions":cw[ids],"jacobians":jac[ids],"accepted_parent_world_positions":bw,"accepted_parent_faces":row["faces"],"external_clearance_rows":ext,"clearance_preservation_pair_rows":preserve})
 contexts.append((step,pm));report["contexts"].append({"step":step,"external_child_rows":len(ext),"strict_added_parent_pairs":len(added),"self_preservation_rows":len(preserve)})
 print(json.dumps(report["contexts"][-1]),flush=True);save()
ids=np.unique(f[preserve.flat]);samples.append({"vertex_ids":ids,"baseline_world_positions":p[ids],"rejected_world_positions":p[ids],"jacobians":np.repeat(np.eye(3)[None,:,:],len(ids),axis=0),"clearance_preservation_pair_rows":preserve})
seeds=np.setdiff1d(np.unique(f[np.asarray(sorted(seedfaces),int)]),fixed)
adj=[set() for _ in p]
for a,b,c in f:adj[a].update([b,c]);adj[b].update([a,c]);adj[c].update([a,b])
mov=set(map(int,seeds));mov|={j for i in list(mov) for j in adj[i]};mov=np.setdiff1d(np.asarray(sorted(mov),int),fixed)
report.update(movable_vertex_ids=mov,fixed_onehot_proxy_count=len(fixed));save()
try:
 proposal=ca._propose_local_self_separation_increment(source_positions_m=p,faces=f,movable_vertex_ids=mov,self_patch_seed_vertex_ids=seeds,required_target_seed_vertex_ids=np.empty(0,int),active_target_face_vertex_ids=np.empty(0,int),fixed_support_vertex_ids=fixed,preserved_anchor_vertex_ids=fixed,pose_samples=samples)
 pp=np.asarray(proposal["proposal_source_positions_m_f32"],np.float32);delta=float(np.linalg.norm(pp.astype(float)-p,axis=1).max());report.update(max_source_increment_m=delta,helper={k:v for k,v in proposal.items() if not isinstance(v,np.ndarray)})
 assert delta<=.0005 and np.array_equal(pp[fixed],p[fixed])
 checks=[]
 for label,v in [("source",pp),*[(str(s),h.forward(pp,local,w,row,t,pm)) for s,pm in contexts]]:
  rec,d=h.exact_rows(v,f,ci);a=ci._audit_pair(rec,rec,same_surface=True) if not d else {"count":None,"triangle_pairs":[]};top=cg.analyze_topology(v.astype(float).tolist(),f.tolist())
  checks.append({"pose":label,"self":a["count"],"pairs":a["triangle_pairs"],"degenerate":d,"closed":top["closed_oriented_manifold_candidate"]});print(json.dumps(checks[-1]),flush=True)
 z["vertices6"]=np.column_stack([pp,h.vertex_normals(pp,f)]).astype(np.float32);dest=O/"stable-130-unadmitted-joint-source.npz";np.savez(dest,**z)
 report.update(candidate_path=str(dest),candidate_sha256=h.sha(dest),checks=checks,all_source_and_current_pose_self_clear=all(x["self"]==0 and not x["degenerate"] and x["closed"] for x in checks))
except Exception as e:report.update(error=str(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
report.update(complete=True,inputs_unchanged=all(h.sha(p)==v for p,v in pins.items()));save()
print(json.dumps({k:v for k,v in report.items() if k in ["complete","error","max_source_increment_m","all_source_and_current_pose_self_clear","candidate_path"]}),flush=True)
