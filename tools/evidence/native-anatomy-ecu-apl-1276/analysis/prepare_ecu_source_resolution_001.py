from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
R=Path("/Users/n/numi-human-retained-delivery-20261009")
O=R/"anatomy-completion-1276/ecu-source-resolution-001"
O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-completion-1276/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.resting_lung_edge_repair import collapse_midpoint_edge
from numilab_human.resting_anatomy_interface_patch import normals
H=R/"fullbody-source-surface-repair-1276/gluteus-maximus-pilot-001/attempt-010/run_gluteus_union_010.py"
sp=importlib.util.spec_from_file_location("prior",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):sha(p) for p in [Path(__file__),H,h.TISS,h.MAN,Path(sys.modules[collapse_midpoint_edge.__module__].__file__),Path(ci.__file__)]}
t=h.load_tissue(h.TISS)
out={"scope":"Bounded existing topology-preserving source edge-collapse owner, canonical mean binding inference; source and retained-pose diagnostic only; no whole-body or native admission.","pins":pins,"rows":[],"complete":False}
def save(): (O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
def audit(p,f):
 r,d=h.exact_rows(p,f,ci)
 return {"count":len(d) if d else ci._audit_pair(r,r,same_surface=True)["count"],"degenerates":d}
posemaps={}
for step in [0,47519,151999,152607,153215,153823,154431,155000]:
 p=h.PAIR/"baseline/native-run/accepted-geometry"/f"step-{step}.receipt.json"
 pins[str(p)]=sha(p)
 rc=json.loads(p.read_text())
 posemaps[step]={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_registered_body_poses"]}
def canon(ix,w,bc):
 a=np.zeros(bc,np.float64)
 for i,x in zip(ix,w):
  if x>0:a[int(i)]+=float(x)
 return a
for sid in [115,116]:
 row=h.row_data(t,sid);v0=row["positions"];f0=row["faces"]
 # Quotient coincident source vertices only after verifying their binding maps.
 uq,first,inv=np.unique(v0,axis=0,return_index=True,return_inverse=True)
 used=np.unique(inv[f0]);qmap=np.full(len(uq),-1);qmap[used]=np.arange(len(used))
 f=qmap[inv[f0]];p=uq[used];idx=row["local"][first[used]].copy();w=row["weights"][first[used]].copy()
 for j in np.unique(f0):
  q=int(qmap[inv[j]])
  assert np.array_equal(canon(row["local"][j],row["weights"][j],row["bc"]),canon(idx[q],w[q],row["bc"]))
 assert np.all(w[w>0]>2e-6),"frozen forward cutoff would differ from live positive influence path"
 v=np.column_stack((p,normals(p.astype(float),f))).astype(np.float32);orig=np.arange(len(f))
 def scores(v,f,ix,ww,steps=(0,155000)):
  return {"source":audit(v[:,:3],f),**{str(s):audit(h.forward(v[:,:3],ix,ww,row,t,posemaps[s]),f) for s in steps}}
 base=scores(v,f,idx,w)
 rr={"stable_id":sid,"initial_quotient_counts":[len(v),len(f)],"baseline":base,"attempts":[],"operations":[]}
 out["rows"].append(rr);save()
 for iteration in range(5):
  records,degen=h.exact_rows(v[:,:3],f,ci)
  pairs=ci._audit_pair(records,records,same_surface=True)["triangle_pairs"]
  if not pairs: break
  faceids=np.unique(np.asarray(pairs,dtype=int))
  edges=sorted(set(tuple(sorted(map(int,e))) for tr in f[faceids] for e in [tr[[0,1]],tr[[1,2]],tr[[2,0]]]),key=lambda e:float(np.linalg.norm(v[e[0],:3]-v[e[1],:3])))
  accepted=None
  for keep,remove in edges[:12]:
   length=float(np.linalg.norm(v[keep,:3].astype(float)-v[remove,:3].astype(float)))
   if length>.0005:continue
   a={"iteration":iteration,"edge":[keep,remove],"length_m":length};rr["attempts"].append(a)
   try:
    nv,nf,no,rep=collapse_midpoint_edge(v,f,orig,remove_vertex=remove,keep_vertex=keep,max_endpoint_displacement_m=.00025,max_abs_volume_delta_m3=1e-10)
    # Reconstruct the owner's exact compaction map, then infer one canonical mean row.
    replaced=f.copy();replaced[replaced==remove]=keep
    valid=(replaced[:,0]!=replaced[:,1])&(replaced[:,1]!=replaced[:,2])&(replaced[:,2]!=replaced[:,0])
    survived=np.unique(replaced[valid]);ni=idx[survived].copy();nw=w[survived].copy()
    avg=(canon(idx[keep],w[keep],row["bc"])+canon(idx[remove],w[remove],row["bc"]))/2
    nonzero=np.flatnonzero(avg>0);assert len(nonzero)<=4
    k=rep["retained_vertex_after_compaction"];ni[k]=np.uint32(4294967295);nw[k]=0
    ni[k,:len(nonzero)]=nonzero;nw[k,:len(nonzero)]=avg[nonzero].astype(np.float32)
    sc=scores(nv,nf,ni,nw);a["scores"]=sc;a["collapse_report"]=rep
    if sc["source"]["count"]<base["source"]["count"] and all(sc[s]["count"]<=base[s]["count"] and not sc[s]["degenerates"] for s in sc):
     accepted=(nv,nf,no,ni,nw,sc,rep);a["accepted_for_next_offline_iteration"]=True;break
   except Exception as e:a["error"]=type(e).__name__+": "+str(e)
   save()
  if accepted is None:break
  v,f,orig,idx,w,base,rep=accepted;rr["operations"].append(rep);save()
 rr["final"]=scores(v,f,idx,w,posemaps.keys())
 if rr["operations"]:
  npz=O/f"stable-{sid}-reference-resolution-row-patch.npz"
  np.savez(npz,vertices6=v,binding_indices=idx,weights=w,faces=f.astype("<u4"),face_origins=orig)
  rr["candidate_path"]=str(npz);rr["candidate_sha256"]=sha(npz)
 rr["source_and_sampled_pose_self_clear"]=all(a["count"]==0 and not a["degenerates"] for a in rr["final"].values())
 save();print(json.dumps({"sid":sid,"operations":len(rr["operations"]),"final":rr["final"]}),flush=True)
out["complete"]=True;out["inputs_unchanged"]=all(sha(p)==v for p,v in pins.items());save()
