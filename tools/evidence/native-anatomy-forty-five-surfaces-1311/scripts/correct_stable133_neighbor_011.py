from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
A=Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
O=A/"stable-133-neighbor-correction-011";O.mkdir(exist_ok=False)
sys.path.insert(0,"/Users/n/numi-human-anatomy-completion-1276/src")
from numilab_human import common_atlas_skin_clearance as ca,cardiac_cavity_intersections as ci
H=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
sp=importlib.util.spec_from_file_location("h",H);h=importlib.util.module_from_spec(sp);sp.loader.exec_module(h)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
C=A/"stable-133-reference-continuation-007/stable-133-unadmitted-iteration-0.npz"
U=A/"remaining-limb-positive-union-lift-001/stable-133-reference-union-row-patch.npz"
T=A/"forty-four-surface-native-composition-002/assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
proofpath=A/"stable-133-reference-continuation-007/current44-independent-audit-001/progress.json";proof=json.loads(proofpath.read_text())
assert len(proof["rows"])==25 and all(x["candidate_self_pair_count"]==0 and x["candidate_closed_oriented"] and not x["candidate_degenerate_face_count"] for x in proof["rows"])
assert sha(C)=="d9bc609dedab25a6b4cbe803cf7d9cef282f70393172c8c37ec09feceeace7e4"
for path,digest in proof["pins"].items():assert sha(path)==digest
pins={str(p):sha(p) for p in [Path(__file__),H,C,U,T,proofpath,Path(ca.__file__),Path(ci.__file__)]}
z=np.load(C);u=np.load(U);p=z["vertices6"][:,:3];origin=u["vertices6"][:,:3];f=z["faces"];local=z["binding_indices"];w=z["weights"];orig=z["face_origins"];t=h.load_tissue(T);row=h.row_data(t,133)
poses=[];worlds=[]
for arm,steps,base in [("early",[0,4767,5023,5599,6207,6815,7423,8000],A/"forty-four-surface-native-composition-002/baseline/native-run"),("baseline",h.STEPS,h.PAIR/"baseline/native-run"),("intervention",h.STEPS,h.PAIR/"intervention/native-run"),("K1",[8000],A/"mtp-passive-native-sensitivity-one-001/baseline/native-run")]:
 for step in steps:
  rp=base/"accepted-geometry"/f"step-{step}.receipt.json";pins[str(rp)]=sha(rp);rc=json.loads(rp.read_text());pm={int(x["body_index"]):(np.asarray(x["position_m"],np.float32),np.asarray(x["quaternion_xyzw"],np.float32)) for x in rc.get("accepted_body_poses",rc.get("accepted_registered_body_poses"))}
  poses.append(pm);worlds.append(h.forward(p,local,w,row,t,pm))
def exact(p,f):
 ids=np.unique(f);rec,deg=h.exact_rows(p[ids],np.searchsorted(ids,f),ci);assert not deg;return rec
src=exact(p,f);assert ci._audit_pair(src,src,same_surface=True)["count"]==0
empty={"count":0,"triangle_pairs":[],"degenerate_face_rows":[]}
def selfcheck(base,cand):
 return ca.audit_incremental_skin_self_intersections(baseline_world_positions=base,candidate_world_positions=cand,baseline_faces=f,candidate_faces=f,baseline_self_audit=empty,expected_baseline_world_f32_sha256=hashlib.sha256(base.tobytes()).hexdigest(),expected_face_index_sha256=ca._face_index_sha256(f),expected_baseline_self_pair_table_sha256=ca._baseline_self_pair_table_sha256(empty,len(f)))["count"]
pack=A/"forty-four-surface-native-composition-002/baseline/native-run/accepted-geometry/step-0.mrvpack";pins[str(pack)]=sha(pack)
pos,surfs,_=ca._pack_surfaces(pack,[(51005,133),(51005,131)])
sf=np.asarray(surfs[(51005,133)]["faces"]);off=sf-row["faces"];assert np.all(off==off.flat[0]);actual=pos[int(off.flat[0]):int(off.flat[0])+row["vc"]]
same=np.zeros(len(f),bool)
for k in range(3):
 cf=np.roll(f,k,axis=1);same|=np.all(row["positions"][row["faces"][orig]]==p[cf],axis=(1,2))&np.all(row["local"][row["faces"][orig]]==local[cf],axis=(1,2))&np.all(row["weights"][row["faces"][orig]]==w[cf],axis=(1,2))
bids=np.unique(orig[~same]);aids=np.flatnonzero(np.isin(orig,bids));tf=np.asarray(surfs[(51005,131)]["faces"])
pts=np.concatenate([actual[np.unique(row["faces"][bids])],worlds[0][np.unique(f[aids])]])
lo=pts.min(0)-.001;hi=pts.max(0)+.001;tri=pos[tf];tids=np.flatnonzero(np.all(tri.max(1)>=lo,axis=1)&np.all(tri.min(1)<=hi,axis=1));target=exact(pos,tf[tids])
bp={(int(bids[a]),int(tids[b])) for a,b in ci._audit_pair(exact(actual,row["faces"][bids]),target,same_surface=False)["triangle_pairs"]}
oldn=np.cross(origin[f[:,1]].astype(float)-origin[f[:,0]],origin[f[:,2]].astype(float)-origin[f[:,0]]);oldlen=np.linalg.norm(oldn,axis=1)
area=lambda x:float(np.linalg.norm(np.cross(x[f[:,1]].astype(float)-x[f[:,0]],x[f[:,2]].astype(float)-x[f[:,0]]),axis=1).sum()/2)
vol=lambda x:float(np.einsum("ij,ij->i",x[f[:,0]].astype(float),np.cross(x[f[:,1]].astype(float),x[f[:,2]].astype(float))).sum()/6)
ba=area(origin);bv=vol(origin)
report={"scope":"Unadmitted vertex1476 sensitivity after one new native44 step0 pair with stable131; all25 self/topology baseline independently audited. Final all859 targets still required.","pins":pins,"trials":[],"selected":None,"complete":False}
def save():(O/"report.json").write_text(json.dumps(report,indent=2)+"\n")
directions=[("revert_fraction_"+str(t),(origin[1476].astype(float)+(p[1476].astype(float)-origin[1476])*t).astype(np.float32)) for t in (0,.125,.25,.375,.5,.625,.75,.875)]
norm=h.vertex_normals(p,f)[1476]
for um in (1,5,10,25,50,100):
 for di,d in enumerate([norm,-norm,*np.eye(3),*(-np.eye(3))]):directions.append((f"delta_{um}um_direction_{di}",(p[1476].astype(float)+d*um*1e-6).astype(np.float32)))
for name,point in directions:
 x=p.copy();x[1476]=point;nn=np.cross(x[f[:,1]].astype(float)-x[f[:,0]],x[f[:,2]].astype(float)-x[f[:,0]]);nl=np.linalg.norm(nn,axis=1)
 bound=float(np.linalg.norm(x.astype(float)-origin,axis=1).max());ar=area(x)/ba-1;vr=vol(x)/bv-1;cos=float((np.einsum("ij,ij->i",nn,oldn)/(nl*oldlen)).min());ratio=float((nl/oldlen).min())
 if bound>.0005 or abs(ar)>.001 or abs(vr)>.001 or cos<=0 or ratio<.2:continue
 world=worlds[0].copy();world[1476]=h.forward(x[1476:1476+1],local[1476:1476+1],w[1476:1476+1],row,t,poses[0])[0]
 ap={(int(orig[aids[a]]),int(tids[b])) for a,b in ci._audit_pair(exact(world,f[aids]),target,same_surface=False)["triangle_pairs"]}
 rr={"proposal":name,"added_parent_pairs":sorted(ap-bp),"maximum_source_displacement_m":bound,"area_relative_change":ar,"volume_relative_change":vr,"minimum_area_ratio":ratio,"minimum_normal_cosine":cos};report["trials"].append(rr)
 if ap-bp:save();continue
 rr["source_self"]=selfcheck(p,x)
 if rr["source_self"]:save();continue
 rr["pose_counts"]=[]
 for i,base in enumerate(worlds):
  world=base.copy();world[1476]=h.forward(x[1476:1476+1],local[1476:1476+1],w[1476:1476+1],row,t,poses[i])[0]
  rr["pose_counts"].append(selfcheck(base,world))
  if rr["pose_counts"][-1]:break
 save();print(json.dumps(rr),flush=True)
 if len(rr["pose_counts"])==25 and not any(rr["pose_counts"]):
  dest=O/"stable-133-neighbor-corrected-unadmitted.npz";np.savez(dest,vertices6=np.column_stack([x,h.vertex_normals(x,f)]).astype("<f4"),faces=f,weights=w,binding_indices=local,face_origins=orig)
  report["selected"]={"candidate_path":str(dest),"candidate_sha256":sha(dest),"proposal":name};break
report.update(complete=True,inputs_unchanged=all(sha(k)==v for k,v in pins.items()));save();print(json.dumps({"selected":report["selected"],"trials":len(report["trials"])}),flush=True)
