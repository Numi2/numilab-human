#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,sys,time
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.spatial import cKDTree
R=Path("/Users/n/numi-human-retained-delivery-20261009"); A=R/"anatomy-completion-1276"
ROOT=A/"scm-vessel-clearance-001"; OUT=ROOT/"run-071-source-attribute-lift-001"; RESULT=OUT/"result-001"
PATCH=A/"right-scm-local-reference-002/stable-151-alpha-0.3.npz"; PATCH_REPORT=A/"right-scm-local-reference-002/report.json"
TISS=A/"neck-back-append-preparation-002/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"; TISS_MAN=TISS.with_suffix(".manifest.json")
FORWARD=A/"neck-back-retained-pose-audit-002/forward_all_positive_weights.py"
INV=Path("/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
R064=ROOT/"run-064-scm-ijv-offset-difference-002"; R064_REPORT=R064/"result-001/report.json"; R064_MESH=R064/"result-001/scm-minus-ijv.exact.json"; R064_SCRIPT=ROOT/"run-064-scm-ijv-offset-difference-002.py"
BASE_ZERO=ROOT/"run-062-scm-ijv-difference-001/result-001/report.json"
OWNER=Path("/Users/n/numi-human-anatomy-publication-1308/src")
CA=OWNER/"numilab_human/common_atlas_skin_clearance.py"; CI=OWNER/"numilab_human/cardiac_cavity_intersections.py"; PART=OWNER/"numilab_human/cardiac_cavity_partition.py"
PAIR=A/"fifty-seven-surface-native-composition-001/baseline/native-run/accepted-geometry"; STEPS=[0,4767,5023,5599,6207,6815,7423,8000]
SCRIPT=Path(__file__).resolve()
if OUT.exists(): raise SystemExit("refuse existing run-071 path")
OUT.mkdir(parents=True); RESULT.mkdir()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x: raise RuntimeError(m)
started=time.monotonic()
pins={str(p):sha(p) for p in [PATCH,PATCH_REPORT,TISS,TISS_MAN,FORWARD,INV,R064_REPORT,R064_MESH,R064_SCRIPT,BASE_ZERO,CA,CI,PART]}
packfiles={}
for step in STEPS:
 packfiles[step]=[PAIR/f"step-{step}.mrvpack",PAIR/f"step-{step}.receipt.json"]
 for p in packfiles[step]:
  req(p.is_file(),f"missing {p}"); pins[str(p)]=sha(p)
sys.path.insert(0,str(OWNER))
from numilab_human import common_atlas_skin_clearance as ca, cardiac_cavity_intersections as ci, cardiac_cavity_partition as cp
spec=importlib.util.spec_from_file_location("scm_forward065",FORWARD); h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
tissue=h.load_tissue(TISS); row=h.row_data(tissue,151); opposite=h.row_data(tissue,152)
req(row["bc"]==3,"appended SCM binding cardinality changed")
with np.load(PATCH,allow_pickle=False) as z:
 parent_src=np.asarray(z["vertices6"][:,:3],dtype="<f4"); parent_faces=np.asarray(z["faces"],dtype=np.int64)
 parent_local=np.asarray(z["binding_indices"],dtype="<u4"); parent_w=np.asarray(z["weights"],dtype="<f4")
req(parent_src.shape==(5160,3) and parent_faces.shape==(10324,3),"patch mesh identity changed")
req(np.all(parent_local[parent_w>0]<row["bc"]),"patch local binding exceeds current row table")
legacy,*_=ca._load_target_inventory(INV); legacy=set(legacy); req(len(legacy)==859,"target inventory count changed")
skin_key=(51007,1); opp_key=(51005,152); target_keys=legacy|{skin_key}
req(skin_key not in legacy and opp_key not in legacy,"target set collision")
def capture(step):
 pack,rp=packfiles[step]; rec=json.loads(rp.read_text())
 req(rec.get("accepted_step")==step and rec.get("physical_endpoint")=="accepted" and rec.get("pack_file_sha256")==sha(pack),"capture receipt mismatch")
 poses={int(x["body_index"]):(np.asarray(x["position_m"],dtype=np.float32),np.asarray(x["quaternion_xyzw"],dtype=np.float32)) for x in rec["accepted_body_poses"]}
 req(len(poses)==157,f"step{step} lacks157 body poses")
 return pack,rec,poses
_,rec0,poses0=capture(0)
oldrec=json.loads((A/"twenty-six-surface-native-composition-001/baseline/native-run/accepted-geometry/step-0.receipt.json").read_text())
req(rec0["accepted_body_state_sha256"]==oldrec["accepted_body_state_sha256"],"current57 q0 body state differs from run064")
world0=h.forward(parent_src,parent_local,parent_w,row,tissue,poses0).astype("<f4")
req(ca._float32_xyz_sha256(world0)==json.loads(R064_REPORT.read_text())["scm_world_f32_sha256"],"q0 source replay differs from run064")
mj=json.loads(R064_MESH.read_text())
def frac(x):
 if isinstance(x,(list,tuple)) and len(x)==2: return Fraction(int(x[0],0) if isinstance(x[0],str) else int(x[0]),int(x[1],0) if isinstance(x[1],str) else int(x[1]))
 raise RuntimeError("bad rational coordinate")
q0=np.asarray([[np.float32(float(frac(c))) for c in p] for p in mj["vertices_m"]],dtype="<f4")
faces=np.asarray(mj["triangles"],dtype=np.int64); ancestry=mj["face_ancestry"]
req(len(faces)==len(ancestry),"ancestry face count mismatch")
world_lookup={p.tobytes():i for i,p in enumerate(world0)}
tri_world=world0[parent_faces].astype(np.float64); tree=cKDTree(tri_world.mean(axis=1))
incident=[set() for _ in range(len(q0))]
for fi,meta in enumerate(ancestry):
 if meta.get("source")=="SCM":
  for vi in faces[fi]: incident[int(vi)].add(int(meta["source_face"]))
def bary_project(p,tri):
 a,b,c=tri; v0=b-a; v1=c-a; v2=p-a
 d00=v0@v0; d01=v0@v1; d11=v1@v1; d20=v2@v0; d21=v2@v1
 den=d00*d11-d01*d01; req(den>0,"degenerate source triangle")
 v=(d11*d20-d01*d21)/den; w=(d00*d21-d01*d20)/den; b=np.maximum([1-v-w,v,w],0.); b/=b.sum()
 return b,b@tri
src=np.zeros_like(q0); local=np.full((len(q0),4),0xffffffff,dtype="<u4"); weights=np.zeros((len(q0),4),dtype="<f4")
support=np.zeros_like(q0,dtype=np.float64); pface=np.full((len(q0),3),-1,dtype=np.int64); barys=np.zeros((len(q0),3))
created=kept=0; maxproj=0.; bind=tissue["bindings"][row["fb"]:row["fb"]+row["bc"]]
req(row["bc"]==3 and [int(x) for x in bind["core"]]==[20,29,23],"SCM core binding table changed")
for vi,p32 in enumerate(q0):
 match=world_lookup.get(p32.tobytes())
 if match is not None:
  src[vi]=parent_src[match]; local[vi]=parent_local[match]; weights[vi]=parent_w[match]; support[vi]=parent_src[match]; kept+=1; continue
 created+=1; p=p32.astype(float)
 cand=sorted(incident[vi]) if incident[vi] else np.asarray(tree.query(p,k=min(48,len(parent_faces)))[1]).reshape(-1).astype(int).tolist()
 best=None
 for sf in cand:
  bw,proj=bary_project(p,tri_world[sf]); d=float(np.linalg.norm(proj-p))
  if best is None or d<best[0]: best=(d,int(sf),bw)
 req(best is not None,"new vertex has no SCM support")
 d,sf,bw=best; maxproj=max(maxproj,d)
 tri=parent_faces[sf]; pface[vi]=tri; barys[vi]=bw; support[vi]=bw@parent_src[tri].astype(float)
 cw=np.zeros(3)
 for k,pvi in enumerate(tri):
  for lane in range(4):
   w=float(parent_w[pvi,lane]); li=int(parent_local[pvi,lane])
   if w>0: req(li<3,"bad local lane"); cw[li]+=bw[k]*w
 req(np.isfinite(cw).all() and np.all(cw>=-1e-8) and cw.sum()>0,"invalid interpolated weights")
 cw=np.maximum(cw,0); cw/=cw.sum()
 local[vi]=[0,1,2,0xffffffff]; weights[vi]=np.asarray([*cw,0],dtype="<f4")
# Invert the weighted affine q0 transform for generated vertices; preserve original records above.
newids=np.flatnonzero(pface[:,0]>=0); n=len(q0); common=np.tile(np.asarray([0,1,2,0xffffffff],dtype="<u4"),(n,1))
y0=h.forward(np.zeros((n,3),dtype="<f4"),common,weights,row,tissue,poses0).astype(float); cols=[]
for axis in range(3):
 probe=np.zeros((n,3),dtype="<f4"); probe[:,axis]=1.
 cols.append(h.forward(probe,common,weights,row,tissue,poses0).astype(float)-y0)
mat=np.stack(cols,axis=2); req(np.isfinite(mat[newids]).all(),"nonfinite weighted affine map")
src[newids]=np.linalg.solve(mat[newids],(q0.astype(float)[newids]-y0[newids])[...,None])[...,0].astype("<f4")
q0_replay=h.forward(src,local,weights,row,tissue,poses0).astype("<f4")
q0_err=np.linalg.norm(q0_replay.astype(float)-q0.astype(float),axis=1); req(float(q0_err.max())<=2e-6,"q0 replay error >2um")
delta=np.linalg.norm(src.astype(float)-support,axis=1); maxdelta=float(delta[newids].max()) if len(newids) else 0.
# Declared 4mm engineering search radius, not an anatomical-accuracy threshold.
within4=maxdelta<=.004
# Recompute normals on the new source mesh.
acc=np.zeros_like(src,dtype=float); sp=src.astype(float)
for f in faces:
 nn=np.cross(sp[f[1]]-sp[f[0]],sp[f[2]]-sp[f[0]])
 for vi in f: acc[vi]+=nn
ln=np.linalg.norm(acc,axis=1); req(np.isfinite(ln).all() and np.all(ln>0),"undefined source normal")
normals=(acc/ln[:,None]).astype("<f4")
def exact_mesh(points,meshfaces=None):
 if meshfaces is None: meshfaces=faces
 meshfaces=np.asarray(meshfaces,dtype=np.int64)
 p=np.asarray(points,dtype="<f4"); uq,inv=np.unique(p,axis=0,return_inverse=True); qf=inv[meshfaces]
 lattice=[ci.float32_point_lattice_key(tuple(float(v) for v in x)) for x in uq]
 rr=ci._records(lattice,qf.tolist()); ss=ci._audit_pair(rr,rr,same_surface=True); tt=cp._validate_output_topology(rr)
 return rr,ss,tt
src_records,src_self,src_topo=exact_mesh(src)
pure=((parent_w==1).sum(axis=1)==1)&((parent_w==0).sum(axis=1)==3); pure_bytes={world0[i].tobytes() for i in np.flatnonzero(pure)}
pure_kept=sum(p.tobytes() in pure_bytes for p in q0)
cache=set(); poses_out=[]
for step in STEPS:
 pack,rec,poses=capture(step); candidate=h.forward(src,local,weights,row,tissue,poses).astype("<f4")
 cr,ss,tt=exact_mesh(candidate)
 xyz,surfs,_=ca._pack_surfaces(pack,target_keys); req(set(surfs)==target_keys,f"step{step} target set mismatch")
 tf={k:np.asarray(v["faces"],dtype=np.int64) for k,v in surfs.items()}
 other=h.forward(opposite["positions"],opposite["local"],opposite["weights"],opposite,tissue,poses).astype("<f4")
 off=len(xyz); xyz=np.concatenate([xyz,other]); tf[opp_key]=opposite["faces"]+off; req(len(tf)==861,"need 861 targets")
 base=h.forward(parent_src,parent_local,parent_w,row,tissue,poses).astype("<f4"); br,bss,btt=exact_mesh(base,parent_faces)
 bt=ca._target_intersection_audit(br,tf,xyz,_validated_target_geometry_cache=cache)
 ct=ca._target_intersection_audit(cr,tf,xyz,_validated_target_geometry_cache=cache)
 changes={}
 for k in sorted(ct):
  if ct[k]["count"] or bt[k]["count"]:
   changes[k]={"base":bt[k]["count"],"candidate":ct[k]["count"],"delta":ct[k]["count"]-bt[k]["count"]}
 poses_out.append({"step":step,"self_pairs":ss["count"],"topology":tt,
  "target_pair_total":sum(v["count"] for v in ct.values()),"baseline_target_pair_total":sum(v["count"] for v in bt.values()),
  "changed_target_surfaces":changes})
NPZ=RESULT/"unadmitted-scm-minus-ijv-source-lift.npz"
np.savez(NPZ,positions_m=src,faces=faces.astype("<i8"),normals=normals,binding_indices=local,weights=weights,
 parent_face_vertices=pface,barycentric=barys,boolean_q0_world_m=q0)
report={"schema":"offline.scm-ijv-source-attribute-lift.v1","status":"unadmitted_candidate",
 "scope":"Inferred lift of q0 SCM-minus-expanded-IJV Boolean mesh into source coordinates with existing SCM binding table; not measured anatomy or admitted asset.",
 "input_pins":pins,"script_sha256":sha(SCRIPT),
 "parent":{"tiss_sha256":pins[str(TISS)],"stable_id":151,"patch_sha256":pins[str(PATCH)],"vertices":len(parent_src),"faces":len(parent_faces),"binding_cores":[int(x) for x in bind["core"]]},
 "boolean":{"mesh_sha256":pins[str(R064_MESH)],"vertices":len(q0),"faces":len(faces),"SCM_child_faces":sum(x.get("source")=="SCM" for x in ancestry),"IJV_cap_faces":sum(x.get("source")=="IJV" for x in ancestry)},
 "lift":{"exact_parent_q0_vertices_retained":kept,"generated_vertices":created,"max_parent_surface_projection_m":maxproj,
 "max_inserted_source_point_offset_from_interpolated_support_m":maxdelta,"4mm_existing_vertex_trust_bound":"not applied to inserted interior cap vertices; maxdelta is depth from parent support, not displacement of a surviving vertex",
 "q0_max_forward_error_m":float(q0_err.max()),"q0_p99_forward_error_m":float(np.quantile(q0_err,.99)),
 "pure_onehot_parent_vertices":int(pure.sum()),"pure_onehot_q0_points_retained":pure_kept,"certified_attachment_mask_available":False,
 "rule":"Exact parent q0 vertices retain source positions and ordered lanes/weights. Generated vertices use nearest q0 parent triangle, barycentric interpolation of existing three core weights normalized only on new rows, and inverse of the weighted q0 rigid transform. Binding table/routes unchanged."},
 "source":{"self_pair_count":src_self["count"],"topology":src_topo,"positions_sha256":ca._float32_xyz_sha256(src),"face_sha256":ca._face_index_sha256(faces)},
 "pose_checks":poses_out,"limitations":["Only eight accepted current57 baseline poses; no new native run.",
 "The cap lift uses nearest SCM support and is inferred; no certified SCM attachment footprint.",
 "Audited 859 inventory surfaces plus native skin plus separately forwarded opposite SCM as 861 external targets.",
 "Target counts compared by surface; face identities differ with triangulation and are not claimed corresponding.",
 "4mm is an engineering search radius, not anatomical accuracy.","No source TISS/runtime/native capture modified."],
 "elapsed_seconds":time.monotonic()-started}
for p,d in pins.items(): req(sha(p)==d,"input changed: "+p)
report=cp.encode_rational(report); rp=RESULT/"report.json"; rp.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":report["status"],"script_sha256":sha(SCRIPT),"npz_sha256":sha(NPZ),"report":str(rp),"report_sha256":sha(rp),
 "generated":created,"maxshift_m":maxdelta,"within4mm":within4,"source_self":src_self["count"],
 "poses":[(x["step"],x["self_pairs"],x["target_pair_total"],x["baseline_target_pair_total"]) for x in poses_out],
 "elapsed":report["elapsed_seconds"]},sort_keys=True),flush=True)
