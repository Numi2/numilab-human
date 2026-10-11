from pathlib import Path
import sys,json,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"pq-left-retriangulated-native-target-audit-029";O.mkdir(exist_ok=False)
N=A/"bilateral-edl-ehl-native-mtp-two-001";T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
sys.path.insert(0,"/Users/n/numi-human-anatomy-publication-1308/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_geometry as cg,cardiac_cavity_intersections as ci
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
C=A/"pq-left-native-local-retriangulation-028/stable-130-unadmitted-native-retriangulated.npz";CR=C.parent/"report.json"
pins={str(x):h.sha(x) for x in [Path(__file__),T,C,CR,H,h.INV,Path(ca.__file__),Path(cg.__file__),Path(ci.__file__)]}
cr=json.loads(CR.read_text());assert cr["complete"] and cr["inputs_unchanged"] and cr["all_source_and_native_pose_self_clear"]
t=h.load_tissue(T);row=h.row_data(t,130)
PN=A/"left-pq-native-mtp-two-001";PT=PN/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue";pins[str(PT)]=h.sha(PT);pt=h.load_tissue(PT);pr=h.row_data(pt,130)
b={"vertices6":np.column_stack([row["positions"],np.zeros_like(row["positions"])]),"faces":row["faces"],"binding_indices":row["local"],"weights":row["weights"]};c=dict(np.load(C));parents=c["face_origins"].astype(int)
assert len(parents)==len(c["faces"]) and parents.min()>=0 and parents.max()<len(b["faces"])
def rec(p,f):
 r,d=h.exact_rows(p,f,ci);assert not d;return r
def aud(p,f):
 r=rec(p,f);a=ci._audit_pair(r,r,same_surface=True);v=p.astype(float)-p.mean(0,dtype=float);tr=v[f]
 top=cg.analyze_topology(p.astype(float).tolist(),f.tolist())
 return {"self_pairs":a["triangle_pairs"],"closed":top["closed_oriented_manifold_candidate"],"components":top["face_component_count"],"signed_volume_m3":float(np.einsum("ij,ij->i",tr[:,0],np.cross(tr[:,1],tr[:,2])).sum()/6),"surface_area_m2":float(np.linalg.norm(np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0]),axis=1).sum()/2)}
source={name:aud(z["vertices6"][:,:3],z["faces"]) for name,z in [("native_parent",b),("candidate",c)]}
report={"scope":"Both baseline and candidate use actual native captured positions. Candidate changes two face rows only; its geometry is reassembled at exactly retained native positions, pending a fresh native run. Current866-surface K2 diagnostic of joint corrected left PQ130 candidate. Full source/eight-pose self and all865 neighboring targets compared with the current original native row using its explicit source-face ancestry. Existing baseline self/external intersections remain unresolved; no native admission.","pins":pins,"source":source,"poses":[],"complete":False}
report["source_relative_area_change"]=source["candidate"]["surface_area_m2"]/source["native_parent"]["surface_area_m2"]-1
report["source_relative_volume_change"]=source["candidate"]["signed_volume_m3"]/source["native_parent"]["signed_volume_m3"]-1
def save():(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
keys,*_=ca._load_target_inventory(h.INV);keys=set(keys)|{(51005,x) for x in [151,152,159,160,161,162]}
for step in [0,4767,5023,5599,6207,6815,7423,8000]:
 pack=N/f"baseline/native-run/accepted-geometry/step-{step}.mrvpack";rp=pack.with_suffix(".receipt.json");pins[str(pack)]=h.sha(pack);pins[str(rp)]=h.sha(rp);rc=json.loads(rp.read_text());assert rc["pack_file_sha256"]==pins[str(pack)]
 pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 pos,surfs,_=ca._pack_surfaces(pack,keys);assert len(surfs)==866
 pp=PN/f"baseline/native-run/accepted-geometry/step-{step}.mrvpack";rr=pp.with_suffix(".receipt.json");pins[str(pp)]=h.sha(pp);pins[str(rr)]=h.sha(rr);assert json.loads(rr.read_text())["pack_file_sha256"]==pins[str(pp)]
 vp,ss,_=ca._pack_surfaces(pp,{(51005,130)});sf=np.asarray(ss[(51005,130)]["faces"],int);delta=sf-pr["faces"];assert np.all(delta==delta.flat[0]);candidate_native_world=vp[int(delta.flat[0]):int(delta.flat[0])+pr["vc"]]
 assert np.array_equal(pr["positions"],c["vertices6"][:,:3]) and np.array_equal(pr["local"],c["binding_indices"]) and np.array_equal(pr["weights"],c["weights"])
 bf=np.asarray(surfs[(51005,130)]["faces"],int);off=bf-row["faces"];assert np.all(off==off.flat[0]);native_parent_world=pos[int(off.flat[0]):int(off.flat[0])+row["vc"]]
 ent={"step":step,"rows":[]};report["poses"].append(ent)
 sets=[]
 for name,z in [("native_parent",b),("candidate",c)]:
  p=native_parent_world if name=="native_parent" else candidate_native_world;f=z["faces"];rr=rec(p,f);lo=p.min(0);hi=p.max(0);record={"variant":name,**aud(p,f),"target_count":0,"target_pairs":[]};ent["rows"].append(record)
  idsset=set()
  for key,surf in surfs.items():
   if key==(51005,130):continue
   record["target_count"]+=1;tf=np.asarray(surf["faces"],int);tr=pos[tf];sel=np.flatnonzero(np.all(tr.max(1)>=lo,1)&np.all(tr.min(1)<=hi,1))
   if not len(sel):continue
   ids=np.unique(tf[sel]);records=rec(pos[ids],np.searchsorted(ids,tf[sel]));pairs=ci._audit_pair(rr,records,same_surface=False)["triangle_pairs"]
   if pairs:
    pp=[[int(i),int(sel[j])] for i,j in pairs];record["target_pairs"].append({"key":list(key),"pairs":pp})
    idsset.update((key,int(parents[i]) if name=="candidate" else i,j) for i,j in pp)
  sets.append(idsset);save()
 before,after=sets;ent["strict_added_parent_pair_ids"]=[[list(k),i,j] for k,i,j in sorted(after-before)];ent["strict_removed_parent_pair_ids"]=[[list(k),i,j] for k,i,j in sorted(before-after)]
 print(json.dumps({"step":step,"self":[len(r["self_pairs"]) for r in ent["rows"]],"strict_added":len(after-before),"strict_removed":len(before-after),"targets":ent["rows"][1]["target_count"]}),flush=True);save()
report.update(complete=True,inputs_unchanged=all(h.sha(p)==v for p,v in pins.items()),all_source_and_pose_self_clear=not source["candidate"]["self_pairs"] and all(not e["rows"][1]["self_pairs"] for e in report["poses"]),no_strict_new_parent_pairs=all(not e["strict_added_parent_pair_ids"] for e in report["poses"]));save()
