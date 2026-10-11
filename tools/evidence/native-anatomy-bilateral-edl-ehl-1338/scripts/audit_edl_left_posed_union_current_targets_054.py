from pathlib import Path
import sys,json,importlib.util,numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276");O=A/"edl-left-posed-union-current-target-audit-054";O.mkdir(exist_ok=False)
N=A/"right-edl-ehl-native-mtp-two-001";T=N/"assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
sys.path.insert(0,"/Users/n/numi-human-anatomy-publication-1308/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_geometry as cg,cardiac_cavity_intersections as ci
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py";sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
C=A/"edl-left-posed-union-ancestry-map-sensitivity-053/stable-22-unadmitted-alpha-0.75-0.75-0.75.npz";B=A/"edl-left-positive-union-lift-003/stable-22-reference-union-row-patch.npz";U=A/"edl-left-posed-positive-union-049/stable-22-posed-exact-union.json";CR=A/"try_edl_left_posed_union_ancestry_maps_053.py"
pins={str(x):h.sha(x) for x in [Path(__file__),T,C,B,U,CR,H,h.INV,Path(ca.__file__),Path(cg.__file__),Path(ci.__file__)]}
assert C.is_file() # Candidate bytes are immutable; parent sensitivity run may still be in progress.
b=dict(np.load(B));c=dict(np.load(C));parents=np.array([x["source_face"] for x in json.loads(U.read_text())["ancestry_by_output_face"]],int)
assert len(parents)==len(c["faces"]) and np.array_equal(c["face_origins"],b["face_origins"][parents])
t=h.load_tissue(T);row=h.row_data(t,22)
def rec(p,f):
 r,d=h.exact_rows(p,f,ci);assert not d;return r
def aud(p,f):
 r=rec(p,f);a=ci._audit_pair(r,r,same_surface=True);v=p.astype(float)-p.mean(0,dtype=float);tr=v[f]
 top=cg.analyze_topology(p.astype(float).tolist(),f.tolist())
 return {"self_pairs":a["triangle_pairs"],"closed":top["closed_oriented_manifold_candidate"],"components":top["face_component_count"],"signed_volume_m3":float(np.einsum("ij,ij->i",tr[:,0],np.cross(tr[:,1],tr[:,2])).sum()/6),"surface_area_m2":float(np.linalg.norm(np.cross(tr[:,1]-tr[:,0],tr[:,2]-tr[:,0]),axis=1).sum()/2)}
source={name:aud(z["vertices6"][:,:3],z["faces"]) for name,z in [("union_parent",b),("candidate",c)]}
report={"scope":"All current866-surface native targets against unadmitted posed-union source lift. Exact strict immediate-parent face pair identities distinguish generated child multiplicity. Existing external overlaps remain unresolved. This tests a common0.75 ancestry blend for the three new generated points. The earlier050 ancestry-distance guard failure remains retained; new-source-point inference requires separate qualification, with no admission from this audit.","pins":pins,"source":source,"poses":[],"complete":False}
report["source_relative_area_change"]=source["candidate"]["surface_area_m2"]/source["union_parent"]["surface_area_m2"]-1
report["source_relative_volume_change"]=source["candidate"]["signed_volume_m3"]/source["union_parent"]["signed_volume_m3"]-1
def save():(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
keys,*_=ca._load_target_inventory(h.INV);keys=set(keys)|{(51005,x) for x in [151,152,159,160,161,162]}
for step in [0,4767,5023,5599,6207,6815,7423,8000]:
 pack=N/f"baseline/native-run/accepted-geometry/step-{step}.mrvpack";rp=pack.with_suffix(".receipt.json");pins[str(pack)]=h.sha(pack);pins[str(rp)]=h.sha(rp);rc=json.loads(rp.read_text());assert rc["pack_file_sha256"]==pins[str(pack)]
 pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc["accepted_body_poses"]}
 pos,surfs,_=ca._pack_surfaces(pack,keys);assert len(surfs)==866
 ent={"step":step,"rows":[]};report["poses"].append(ent)
 sets=[]
 for name,z in [("union_parent",b),("candidate",c)]:
  p=h.forward(z["vertices6"][:,:3],z["binding_indices"],z["weights"],row,t,pm);f=z["faces"];rr=rec(p,f);lo=p.min(0);hi=p.max(0);record={"variant":name,**aud(p,f),"target_count":0,"target_pairs":[]};ent["rows"].append(record)
  idsset=set()
  for key,surf in surfs.items():
   if key==(51005,22):continue
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
