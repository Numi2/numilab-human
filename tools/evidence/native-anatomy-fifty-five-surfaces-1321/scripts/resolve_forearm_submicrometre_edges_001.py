from pathlib import Path
import json,numpy as np,hashlib,sys,importlib.util
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");sid=int(sys.argv[1]);D=A/f"remaining-original-boundary-batch-001/stable-{sid}";O=D/"submicrometre-resolution-001";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-positive-winding-1279/src")
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.resting_lung_edge_repair import collapse_midpoint_edge as component_collapse_owner
from numilab_human.resting_anatomy_interface_patch import normals,signed_volume
from numilab_human.cardiac_cavity_geometry import analyze_topology
wrapper=A/"prepare_bounded_halfmm_source_resolution_001.py";src=wrapper.read_text();exec(src[src.index("def collapse_midpoint_edge("):src.index("\nH=R/")])
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cp=D/f"source-lift-001/stable-{sid}-geometric-reference-row-patch.npz";z=np.load(cp);v=z["vertices6"].copy();f=z["faces"].copy();orig=z["face_origins"].copy();local=z["binding_indices"].copy();weights=z["weights"].copy()
ancestry=[{i} for i in range(len(v))];reference=v[:,:3].copy()
T=A/"fifty-surface-native-composition-001/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";t=h.load_tissue(T);row=h.row_data(t,sid)
audit=json.loads((A/"remaining-original-boundary-batch-001/stable-121/current50-target-audit-001/report.json").read_text());receipts=[Path(x) for x in audit["pins"] if x.endswith(".receipt.json") and "/accepted-geometry/" in x];assert len(receipts)==25
pins={str(x):sha(x) for x in [Path(__file__),cp,H,T,wrapper,Path(ci.__file__),Path(sys.modules[component_collapse_owner.__module__].__file__),*receipts]};contexts=[]
for rp in receipts:
 r=json.loads(rp.read_text());pm={int(x["body_index"]):(np.array(x["position_m"],np.float32),np.array(x["quaternion_xyzw"],np.float32)) for x in r.get("accepted_body_poses",r.get("accepted_registered_body_poses"))};contexts.append((str(rp),pm))
def checks(v,f,local,weights):
 results=[]
 for name,pm in [("source",None),*contexts]:
  p=v[:,:3] if pm is None else h.forward(v[:,:3],local,weights,row,t,pm);rec,dg=h.exact_rows(p,f,ci)
  results.append(dict(context=name,degenerate=dg,self_count=None if dg else ci._audit_pair(rec,rec,same_surface=True)["count"]))
 return results
def shortedges(v,f):
 e=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0);length=np.linalg.norm(v[e[:,0],:3].astype(float)-v[e[:,1],:3],axis=1);order=np.argsort(length);return [(int(e[i,0]),int(e[i,1]),float(length[i])) for i in order if length[i]<5e-7]
out={"scope":"Existing manifold link-condition edge contraction of submicrometre exact-partition features that can collapse under GPU Float32 posing. Keep endpoint and its canonical binding unchanged; removed endpoint ancestry retained. No clinical accuracy or external/native admission.","pins":pins,"stable_id":sid,"trials":[],"accepted_collapses":[],"complete":False,"candidate":None}
def save():(O/"report.json").write_text(json.dumps(out,indent=2)+"\n")
for iteration in range(8):
 edges=shortedges(v,f)
 if not edges:break
 accepted=False
 for a,b,l in edges:
  for remove,keep in [(a,b),(b,a)]:
   if weights[remove].max()>=.999999 or weights[keep].max()>=.999999:continue
   rec={"remove":remove,"keep":keep,"length_m":l,"accepted":False};out["trials"].append(rec)
   try:
    nv,nf,no,rep=collapse_midpoint_edge(v,f,orig,remove_vertex=remove,keep_vertex=keep,position_policy="keep",max_endpoint_displacement_m=5e-7,max_abs_volume_delta_m3=1e-12)
    replaced=f.copy();replaced[replaced==remove]=keep;valid=np.asarray([len(set(map(int,x)))==3 for x in replaced]);used=np.unique(replaced[valid]);ni=local[used];nw=weights[used]
    na=[set(ancestry[i]) for i in used];nk=int(np.searchsorted(used,keep));na[nk]|=ancestry[remove]
    bound=max(float(np.linalg.norm(nv[i,:3].astype(float)-reference[list(ids)],axis=1).max()) for i,ids in enumerate(na));assert bound<=5e-7
    ck=checks(nv,nf,ni,nw);rec.update(owner_report=rep,checks=ck,maximum_original_endpoint_displacement_m=bound)
    if any(x["degenerate"] or x["self_count"] for x in ck):save();continue
    v,f,orig,local,weights,ancestry=nv,nf,no,ni,nw,na;rec["accepted"]=True;out["accepted_collapses"].append(rec);accepted=True;save();print(json.dumps({"sid":sid,"iteration":iteration,"remaining_short_edges":len(shortedges(v,f))}),flush=True);break
   except ValueError as e:rec["error"]=str(e);save()
  if accepted:break
 if not accepted:break
ck=checks(v,f,local,weights);out["checks"]=ck;out["remaining_short_edges"]=shortedges(v,f)
if not out["remaining_short_edges"] and all(not x["degenerate"] and x["self_count"]==0 for x in ck):
 dst=O/f"stable-{sid}-geometric-reference-row-patch.npz";np.savez(dst,vertices6=v,faces=f,face_origins=orig,binding_indices=local,weights=weights)
 out["candidate"]={"path":str(dst),"sha256":sha(dst),"original_vertex_ancestry":[sorted(x) for x in ancestry],"maximum_original_endpoint_displacement_m":max(float(np.linalg.norm(v[i,:3].astype(float)-reference[list(ids)],axis=1).max()) for i,ids in enumerate(ancestry))}
out.update(complete=True,inputs_unchanged=all(sha(p)==v for p,v in pins.items()));save();print(json.dumps({"candidate":None if out["candidate"] is None else {k:v for k,v in out["candidate"].items() if k!="original_vertex_ancestry"},"collapses":len(out["accepted_collapses"]),"short_edges":out["remaining_short_edges"]}),flush=True)
