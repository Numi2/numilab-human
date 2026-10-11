from pathlib import Path
import sys,json,hashlib,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"edl-right-refined-ehl-target-supplement-035";O.mkdir(exist_ok=False)
N=A/"ehl-restored-pair-native-mtp-two-001";RN=A/"right-deep-masseter-native-003"
sys.path.insert(0,"/Users/n/numi-human-anatomy-publication-1308/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_geometry as cg,cardiac_cavity_intersections as ci
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";RT=RN/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
C=A/"edl-right-local-binding-sensitivity-029/stable-21-unadmitted-iteration-3.npz";B=A/"edl-right-positive-union-lift-001/stable-21-reference-union-row-patch.npz"
pins={str(p):h.sha(p) for p in [Path(__file__),H,T,RT,C,B,h.INV,Path(ca.__file__),Path(ci.__file__),Path(cg.__file__)]}
t=h.load_tissue(T);rt=h.load_tissue(RT);r=h.row_data(t,21);rh=h.row_data(rt,161)
b=dict(np.load(B));c=dict(np.load(C))
assert np.array_equal(b["vertices6"],c["vertices6"]) and np.array_equal(b["faces"],c["faces"])
fixed=b["weights"].max(1)>=.999999
assert np.array_equal(b["weights"][fixed],c["weights"][fixed]) and np.array_equal(b["binding_indices"][fixed],c["binding_indices"][fixed])
changed=np.flatnonzero(np.any(b["weights"]!=c["weights"],axis=1)|np.any(b["binding_indices"]!=c["binding_indices"],axis=1))
assert len(changed)==4
def exact(p,f):
 rec,dg=h.exact_rows(p,f,ci);assert not dg
 return rec
def aud(p,f):
 rec=exact(p,f);a=ci._audit_pair(rec,rec,same_surface=True)
 v=p.astype(float)-p.mean(0,dtype=float);tri=v[f]
 top=cg.analyze_topology(p.astype(float).tolist(),f.tolist())
 return {"self_pairs":a["triangle_pairs"],"topology":{k:v for k,v in top.items() if not isinstance(v,(list,dict))},"signed_volume_m3":float(np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)}
out={"scope":"Unadmitted right EDL existing-binding correction. Source coordinates/topology and one-hot proxies exact to positive-union candidate; physical routes/transforms fixed. Supplemental eight K2 captured-pose audit against refined EHL23/24 only; all other targets are covered by030; K2 remains an explicit passive-stiffness diagnostic. Existing intersections are not waived.","pins":pins,"changed_vertex_ids":changed.tolist(),"source":aud(c["vertices6"][:,:3],c["faces"]),"poses":[],"complete":False}
def save():(O/"report.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
keys,*_=ca._load_target_inventory(h.INV);keys=set(keys)|{(51005,x) for x in [151,152,159,160,162]}
for step in [0,4767,5023,5599,6207,6815,7423,8000]:
 pack=N/f"baseline/native-run/accepted-geometry/step-{step}.mrvpack";rp=pack.with_suffix(".receipt.json")
 pins[str(pack)]=h.sha(pack);pins[str(rp)]=h.sha(rp);rc=json.loads(rp.read_text());assert rc["pack_file_sha256"]==pins[str(pack)]
 pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 pos,surfs,_=ca._pack_surfaces(pack,keys);assert len(surfs)==865
 # The newly added head row uses the exact existing source/runtime transform at this K2 pose.
 hw=h.forward(rh["positions"],rh["local"],rh["weights"],rh,rt,pm);head=(hw,rh["faces"])
 worlds={name:h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],r,t,pm) for name,z in [("union_parent",b),("candidate",c)]}
 ent={"step":step,"head161_target_source":"admitted RN003 row forwarded at this captured K2 body pose","max_binding_world_change_m":float(np.linalg.norm(worlds["candidate"].astype(float)-worlds["union_parent"],axis=1).max()),"rows":[]};out["poses"].append(ent)
 for name,z in [("union_parent",b),("candidate",c)]:
  w=worlds[name];f=z["faces"];row={"variant":name,**aud(w,f),"target_count":0,"target_pairs":[]};ent["rows"].append(row);rec=exact(w,f);lo=w.min(0);hi=w.max(0)
  targets=[(key,pos,np.asarray(v["faces"],int)) for key,v in surfs.items() if key in {(51005,23),(51005,24)}]
  for key,tp,tf in targets:
   row["target_count"]+=1;tri=tp[tf];sel=np.flatnonzero(np.all(tri.max(1)>=lo,1)&np.all(tri.min(1)<=hi,1))
   if not len(sel):continue
   ids=np.unique(tf[sel]);tr=exact(tp[ids],np.searchsorted(ids,tf[sel]));pairs=ci._audit_pair(rec,tr,same_surface=False)["triangle_pairs"]
   if pairs:row["target_pairs"].append({"key":list(key),"pairs":[[int(i),int(sel[j])] for i,j in pairs]})
  save()
 def pairset(rr):return {(tuple(x["key"]),i,j) for x in rr["target_pairs"] for i,j in x["pairs"]}
 before,after=map(pairset,ent["rows"]);ent["added_target_pairs"]=[[list(k),i,j] for k,i,j in sorted(after-before)];ent["removed_target_pairs"]=[[list(k),i,j] for k,i,j in sorted(before-after)]
 print(json.dumps({"step":step,"self":[len(x["self_pairs"]) for x in ent["rows"]],"targets":[x["target_count"] for x in ent["rows"]],"pairs":[len(before),len(after)],"added":len(after-before),"removed":len(before-after)}),flush=True);save()
out.update(complete=True,inputs_unchanged=all(h.sha(p)==s for p,s in pins.items()),all_pose_self_clear=all(not e["rows"][1]["self_pairs"] for e in out["poses"]),no_new_target_pairs=all(not e["added_target_pairs"] for e in out["poses"]))
save()
