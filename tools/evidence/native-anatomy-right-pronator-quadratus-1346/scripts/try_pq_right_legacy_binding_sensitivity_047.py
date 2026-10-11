from pathlib import Path
import sys,json,importlib.util,numpy as np,time
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"pq-right-legacy-binding-sensitivity-047";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-touching-loop-arrangement-1290/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci,cardiac_cavity_geometry as cg
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
C=A/"pq-right-fine-binding-sensitivity-041/stable-129-unadmitted-weight--0.0004.npz";N=A/"left-pq-retriangulated-native-mtp-two-002";T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";prior=A/"pq-right-binding-all25-target-audit-043/report.json"
z=dict(np.load(C));p=z["vertices6"][:,:3];f=z["faces"];orig=z["face_origins"];t=h.load_tissue(T);row=h.row_data(t,129)
pins={str(x):h.sha(x) for x in [Path(__file__),H,C,T,prior,Path(ca.__file__),Path(ci.__file__),Path(cg.__file__)]}
def rec(v,ff):
 ids=np.unique(ff);r,d=h.exact_rows(v[ids],np.searchsorted(ids,ff),ci);assert not d;return r
contexts=[]
for path,hashval in json.loads(prior.read_text())["pins"].items():
 if not path.endswith(".mrvpack"):continue
 pack=Path(path);rp=pack.with_suffix(".receipt.json");assert h.sha(pack)==hashval
 rc=json.loads(rp.read_text());pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc.get("accepted_body_poses",rc.get("accepted_registered_body_poses"))}
 pos,s,_=ca._pack_surfaces(pack,{(51005,129),(51004,14)});sf=np.asarray(s[(51005,129)]["faces"]);off=sf-row["faces"];assert np.all(off==off.flat[0]);bv=pos[int(off.flat[0]):int(off.flat[0])+row["vc"]]
 target=rec(pos,np.asarray(s[(51004,14)]["faces"]));base=set(map(tuple,ci._audit_pair(rec(bv,row["faces"]),target,same_surface=False)["triangle_pairs"]))
 contexts.append((str(pack),pm,target,base));pins[path]=hashval;pins[str(rp)]=h.sha(rp)
assert len(contexts)==25
def evaluate(w,check_self=False):
 results=[];worlds=[]
 for path,pm,target,base in contexts:
  v=h.forward(p,z["binding_indices"],w,row,t,pm);r=rec(v,f);pairs=ci._audit_pair(r,target,same_surface=False)["triangle_pairs"];added=sorted({(int(orig[i]),int(j)) for i,j in pairs}-base)
  item={"pack":path,"added_parent_pairs":added};worlds.append(v)
  if check_self:item["self_pairs"]=ci._audit_pair(r,r,same_surface=True)["triangle_pairs"]
  results.append(item)
 return results,worlds
w=z["weights"].copy();initial_worlds=[h.forward(p,z["binding_indices"],w,row,t,pm) for _,pm,_,_ in contexts]
initial,_=evaluate(w,True);assert all(not x["self_pairs"] for x in initial);best=sum(len(x["added_parent_pairs"]) for x in initial)
out={"scope":"Source-fixed inferred binding sensitivity, all25 exact captured-pose ulna targets and candidate self checks. Only already mixed vertices may change; pure attachment proxies unchanged. Not native admission or all-target qualification.","pins":pins,"baseline":initial,"baseline_added":best,"trials":[],"accepted_updates":[]}
def save():(O/"report.json").write_text(json.dumps(out,indent=2)+"\n")
save()
for vid in [867,881,875,885,865,893]:
 if best==0:break
 lanes=np.flatnonzero(w[vid]>0)
 if len(lanes)!=2:continue
 bw=w.copy();winning=None
 for delta in [-.00001,.00001,-.00005,.00005,-.0001,.0001,-.0002,.0002,-.0004,.0004,-.0008,.0008,-.0016,.0016]:
  trial=bw.copy();trial[vid,lanes[0]]=np.float32(float(bw[vid,lanes[0]])+delta);trial[vid,lanes[1]]=np.float32(1-float(trial[vid,lanes[0]]))
  if not np.all(trial[vid,lanes]>0):continue
  rr,ww=evaluate(trial);count=sum(len(x["added_parent_pairs"]) for x in rr)
  item={"vertex":vid,"delta":delta,"weights":trial[vid].tolist(),"total_added":count}
  if count<best:
   selfs=[ci._audit_pair(rec(v,f),rec(v,f),same_surface=True)["count"] for v in ww];item["self_counts"]=selfs
   if not any(selfs):
    best=count;winning=trial;item["improved"]=True
    zz={k:v.copy() for k,v in z.items()};zz["weights"]=trial;dest=O/f"stable-129-unadmitted-v{vid}-delta-{delta}.npz";np.savez(dest,**zz);item.update(candidate_path=str(dest),candidate_sha256=h.sha(dest),poses=rr,max_world_delta_m=max(float(np.linalg.norm(v.astype(float)-b,axis=1).max()) for v,b in zip(ww,initial_worlds)));out["accepted_updates"].append(item.copy())
  out["trials"].append(item);save();print(json.dumps({k:v for k,v in item.items() if k!="poses"}),flush=True)
  if best==0:break
 if winning is not None:w=winning
out.update(complete=True,final_added=best,inputs_unchanged=all(h.sha(x)==v for x,v in pins.items()));save()
