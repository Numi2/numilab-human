from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009"); A=R/"anatomy-completion-1276"
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"; C=A/"remaining-limb-positive-union-lift-001/stable-134-reference-union-row-patch.npz"
T=R/"source-seam-connectivity-1247/fhl-current-7b23-count-reconciliation-1258/attempt-002/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
CACHE=A/"stable-134-pose-repair-001/residual-pairs-24pose.json"; OUT=Path(__file__).with_name("local-reference-search-001"); OUT.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-self-separation-partial-resume-1267/src")
from numilab_human import cardiac_cavity_intersections as ci
sys.path.insert(0,"/Users/n/numi-human-local-clearance-preservation-1280/src")
from numilab_human import common_atlas_skin_clearance as ca
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sp=importlib.util.spec_from_file_location("fw",H); h=importlib.util.module_from_spec(sp); sp.loader.exec_module(h); h.TISS=T; h.MAN=T.with_suffix(".manifest.json")
t=h.load_tissue(T); row=h.row_data(t,134); z=np.load(C); p=z["vertices6"][:,:3].astype("<f4"); local=z["binding_indices"].astype("<u4"); w=z["weights"].astype("<f4"); f=z["faces"].astype(np.int64)
cache=json.loads(CACHE.read_text()); assert cache["inputs_unchanged"] and cache["candidate_sha256"]==sha(C) and len(cache["pose_audits"])==24
poses=[]; labels=[]; worlds=[]; baselines=[]
for item in cache["pose_audits"]:
 group,step=item["group"],int(item["step"]); directory=(A/"twenty-eight-surface-native-composition-001/baseline/native-run/accepted-geometry") if group=="early" else (R/"skin-resting-multipose-clearance-1218/native-coupled-pair-preparation-011"/group/"native-run/accepted-geometry")
 rc=json.loads((directory/f"step-{step}.receipt.json").read_text()); prs=rc.get("accepted_body_poses") or rc.get("accepted_registered_body_poses")
 pose={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in prs}
 world=h.forward(p,local,w,row,t,pose); wh=hashlib.sha256(np.ascontiguousarray(world,dtype="<f4").tobytes()).hexdigest(); assert wh==item["world_sha256"]
 tab={"count":int(item["count"]),"triangle_pairs":item["pairs"],"degenerate_face_rows":[]}
 chk=ca.audit_incremental_skin_self_intersections(baseline_world_positions=world,candidate_world_positions=world,baseline_faces=f,candidate_faces=f,baseline_self_audit=tab,expected_baseline_world_f32_sha256=wh,expected_face_index_sha256=ca._face_index_sha256(f),expected_baseline_self_pair_table_sha256=ca._baseline_self_pair_table_sha256(tab,len(f)))
 assert chk["count"]==tab["count"],"cached exact table invalid"
 poses.append(pose); worlds.append(world); baselines.append(tab); labels.append({"group":group,"step":step})
src,deg=h.exact_rows(p,f,ci); assert not deg and ci._audit_pair(src,src,same_surface=True)["count"]==0
pairs=sorted({tuple(map(int,q)) for it in cache["pose_audits"] for q in it["pairs"]}); seedfaces=sorted({j for pair in pairs for j in pair}); seeds=sorted(set(int(v) for fi in seedfaces for v in f[fi]))
pure=np.unique(row["faces"]); pure=pure[row["weights"][pure].max(axis=1)>=.999999]; purebytes={row["positions"][j].tobytes() for j in pure}
locked={j for j in range(len(p)) if p[j].tobytes() in purebytes}; seeds=[j for j in seeds if j not in locked]
def vertex_normals(points,faces):
 acc=np.zeros_like(points,dtype=np.float64); pp=points.astype(float)
 for a,b,c in faces:
  n=np.cross(pp[b]-pp[a],pp[c]-pp[a]); acc[a]+=n; acc[b]+=n; acc[c]+=n
 nn=np.linalg.norm(acc,axis=1); return acc/np.maximum(nn[:,None],1e-30)
vn=vertex_normals(p,f); tri0=np.cross(p[f[:,1]].astype(float)-p[f[:,0]],p[f[:,2]].astype(float)-p[f[:,0]]); len0=np.linalg.norm(tri0,axis=1)
def posed(q,i):
 ch=np.flatnonzero(np.any(q!=p,axis=1)); out=worlds[i].copy()
 if len(ch): out[ch]=h.forward(q[ch],local[ch],w[ch],row,t,poses[i])
 return out
def inc(i,q):
 world=posed(q,i); b=baselines[i]
 return ca.audit_incremental_skin_self_intersections(baseline_world_positions=worlds[i],candidate_world_positions=world,baseline_faces=f,candidate_faces=f,baseline_self_audit=b,expected_baseline_world_f32_sha256=hashlib.sha256(np.ascontiguousarray(worlds[i],dtype="<f4").tobytes()).hexdigest(),expected_face_index_sha256=ca._face_index_sha256(f),expected_baseline_self_pair_table_sha256=ca._baseline_self_pair_table_sha256(b,len(f)))
def score(cs): return (max(cs),sum(cs))
base=[x["count"] for x in baselines]; idx=1
sizes=(0.05,0.1,0.2,0.3); trials=[]; best=None; started=time.monotonic()
for j in seeds:
 ds=[vn[j],-vn[j],*[np.eye(3)[k] for k in range(3)],*[-np.eye(3)[k] for k in range(3)]]
 for di,d in enumerate(ds):
  norm=np.linalg.norm(d)
  if norm<1e-12: continue
  d=d/norm
  for mm in sizes:
   q=p.copy(); q[j]=(p[j].astype(float)+d*(mm*1e-3)).astype("<f4")
   if np.linalg.norm(q[j].astype(float)-p[j].astype(float))>0.00050001: continue
   ch=np.flatnonzero(np.any(q!=p,axis=1)); chf=np.flatnonzero(np.any(np.isin(f,ch),axis=1))
   nn=np.cross(q[f[chf,1]].astype(float)-q[f[chf,0]],q[f[chf,2]].astype(float)-q[f[chf,0]])
   ratio=np.linalg.norm(nn,axis=1)/np.maximum(len0[chf],1e-30); cos=np.einsum("ij,ij->i",nn,tri0[chf])/np.maximum(np.linalg.norm(nn,axis=1)*len0[chf],1e-30)
   if ratio.min()<.2 or cos.min()<=0: continue
   rr,dd=h.exact_rows(q,f,ci)
   if dd or ci._audit_pair(rr,rr,same_surface=True)["count"]!=0: continue
   if inc(idx,q)["count"]>=base[idx]: continue
   counts=[inc(i,q)["count"] for i in range(24)]
   rec={"vertex":int(j),"direction_index":di,"direction":d.tolist(),"requested_mm":mm,"actual_delta_mm":float(np.linalg.norm(q[j].astype(float)-p[j].astype(float))*1000),"changed_faces":int(len(chf)),"min_changed_area_ratio":float(ratio.min()),"min_changed_normal_cosine":float(cos.min()),"counts":counts,"score":[*score(counts)]}
   trials.append(rec)
   if score(counts)<score(base) and (best is None or (score(counts),rec["actual_delta_mm"])<(score(best[0]["counts"]),best[0]["actual_delta_mm"])): best=(rec,q)
   if score(counts)==(0,0): best=(rec,q); break
  if best and score(best[0]["counts"])==(0,0): break
 if best and score(best[0]["counts"])==(0,0): break
 if len(trials)>=120: break
pins={str(x):sha(x) for x in [Path(__file__),H,C,T,T.with_suffix(".manifest.json"),CACHE,Path(ci.__file__),Path(ca.__file__)]}
for path,dig in cache["input_sha256"].items(): pins[path]=dig
report={"schema":"stable-134-bounded-pose-self-reference-search.v1","candidate_path":str(C),"candidate_sha256":sha(C),"predicate_sha256":sha(Path(ci.__file__)),"incremental_owner_sha256":sha(Path(ca.__file__)),"pose_labels":labels,"baseline_counts":base,"source_self_count":0,"seed_faces":seedfaces,"seed_vertices":seeds,"locked_pure_vertex_count":len(locked),"increment_bound_mm":0.5,"trials":trials,"best":None,"diagnostic_only":True,"external_target_audit_performed":False,"inputs_unchanged":all(sha(k)==v for k,v in pins.items()),"input_sha256":pins,"elapsed_seconds":time.monotonic()-started}
if best:
 rec,q=best; out=OUT/"stable-134-unadmitted-local-reference.npz"
 np.savez(out,vertices6=np.column_stack([q,vertex_normals(q,f)]).astype("<f4"),binding_indices=local,weights=w,faces=f.astype("<u4"),face_origins=z["face_origins"])
 report["best"]={"path":str(out),"sha256":sha(out),"trial":rec}
rp=OUT/"report.json"; rp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
print(json.dumps({"report":str(rp),"sha256":sha(rp),"trial_count":len(trials),"baseline_counts":base,"best":report["best"],"inputs_unchanged":report["inputs_unchanged"],"elapsed_seconds":report["elapsed_seconds"]},indent=2),flush=True)
