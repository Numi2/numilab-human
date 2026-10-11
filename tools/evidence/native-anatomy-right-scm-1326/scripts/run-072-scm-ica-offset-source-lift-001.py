#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,importlib.util,json,signal,sys,time
from collections import Counter
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.spatial import cKDTree
R=Path("/Users/n/numi-human-retained-delivery-20261009"); A=R/"anatomy-completion-1276"
ROOT=A/"scm-vessel-clearance-001"; OUT=ROOT/"run-072-scm-ica-offset-source-lift-001"; RESULT=OUT/"result-001"
PREV=ROOT/"run-071-source-attribute-lift-001"
PREV_NPZ=PREV/"result-001/unadmitted-scm-minus-ijv-source-lift.npz"
PREV_REPORT=PREV/"result-001/report.json"
PREV_SCRIPT=ROOT/"run-071-source-attribute-lift-001.py"
NHA=Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-anatomy-manifest.json")
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
if OUT.exists(): raise SystemExit("refuse existing run-072 path")
OUT.mkdir(parents=True); RESULT.mkdir()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()
def req(x,m):
 if not x: raise RuntimeError(m)
started=time.monotonic()
pins={str(p):sha(p) for p in [PATCH,PATCH_REPORT,TISS,TISS_MAN,FORWARD,INV,R064_REPORT,R064_MESH,R064_SCRIPT,BASE_ZERO,CA,CI,PART,PREV_NPZ,PREV_REPORT,PREV_SCRIPT,NHA]}
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
 alpha_src=np.asarray(z["vertices6"][:,:3],dtype="<f4"); alpha_faces=np.asarray(z["faces"],dtype=np.int64)
 alpha_local=np.asarray(z["binding_indices"],dtype="<u4"); alpha_w=np.asarray(z["weights"],dtype="<f4")
with np.load(PREV_NPZ,allow_pickle=False) as z:
 parent_src=np.asarray(z["positions_m"],dtype="<f4"); parent_faces=np.asarray(z["faces"],dtype=np.int64)
 parent_local=np.asarray(z["binding_indices"],dtype="<u4"); parent_w=np.asarray(z["weights"],dtype="<f4")
 parent_pface=np.asarray(z["parent_face_vertices"],dtype=np.int64)
prev_report=json.loads(PREV_REPORT.read_text())
req(prev_report.get("status")=="unadmitted_candidate","previous IJV-only status changed")
req(alpha_src.shape==(5160,3) and alpha_faces.shape==(10324,3),"alpha0.3 patch mesh changed")
req(parent_src.shape==(5264,3) and parent_faces.shape==(10532,3) and parent_pface.shape==(len(parent_src),3),"run071 candidate arrays changed")
req(np.all(alpha_local[alpha_w>0]<row["bc"]) and np.all(parent_local[parent_w>0]<row["bc"]),"SCM local binding out of range")
legacy,*_=ca._load_target_inventory(INV); legacy=set(legacy); req(len(legacy)==859,"target inventory count changed")
skin_key=(51007,1); opp_key=(51005,152); target_keys=legacy|{skin_key}
req(skin_key not in legacy and opp_key not in legacy,"target set collision")
inventory_labels={f"{x['second_semantic']}:{x['second_stable_id']}":x["source_owner_or_label"] for x in csv.DictReader(INV.open(newline=""))}
nha=json.loads(NHA.read_text())
for stable,expected_name,expected_member in [(466,"right internal carotid artery","FJ1682"),(468,"right internal jugular vein","FJ3585")]:
 owner=nha["source_surfaces"][str(stable)]
 req(owner["name"]==expected_name and owner["source_member"]==expected_member,f"NHA stable{stable} identity changed")
 inventory_labels[f"51011:{stable}"]=f"{owner['name']} / {owner['source_member']}"
inventory_labels["51007:1"]="captured native skin surface"
inventory_labels["51005:152"]="opposite SCM stable row152, separately forwarded"
def capture(step):
 pack,rp=packfiles[step]; rec=json.loads(rp.read_text())
 req(rec.get("accepted_step")==step and rec.get("physical_endpoint")=="accepted" and rec.get("pack_file_sha256")==sha(pack),"capture receipt mismatch")
 poses={int(x["body_index"]):(np.asarray(x["position_m"],dtype=np.float32),np.asarray(x["quaternion_xyzw"],dtype=np.float32)) for x in rec["accepted_body_poses"]}
 req(len(poses)==157,f"step{step} lacks157 body poses")
 return pack,rec,poses
pack0,rec0,poses0=capture(0)
oldrec=json.loads((A/"twenty-six-surface-native-composition-001/baseline/native-run/accepted-geometry/step-0.receipt.json").read_text())
req(rec0["accepted_body_state_sha256"]==oldrec["accepted_body_state_sha256"],"current57 q0 state differs")
world0=h.forward(parent_src,parent_local,parent_w,row,tissue,poses0).astype("<f4")
req(np.isfinite(world0).all(),"nonfinite IJV-only replay")
first_mesh=json.loads(R064_MESH.read_text())
def frac(x):
 if isinstance(x,(list,tuple)) and len(x)==2: return Fraction(int(x[0],0) if isinstance(x[0],str) else int(x[0]),int(x[1],0) if isinstance(x[1],str) else int(x[1]))
 raise RuntimeError("bad rational coordinate")
previous_boolean_q0=np.asarray([[np.float32(float(frac(c))) for c in p] for p in first_mesh["vertices_m"]],dtype="<f4")
first_faces=np.asarray(first_mesh["triangles"],dtype=np.int64); first_ancestry=first_mesh["face_ancestry"]
req(len(first_faces)==len(first_ancestry)==len(parent_faces) and np.array_equal(first_faces,parent_faces),"prior Boolean ancestry mismatch")
req(float(np.linalg.norm(world0.astype(float)-previous_boolean_q0.astype(float),axis=1).max())<=2e-6,"run071 q0 replay exceeds2um")
xyz0,surfs0,_=ca._pack_surfaces(pack0,target_keys)
ica_key=(51011,466); ijv_key=(51011,468)
req(ica_key in surfs0 and ijv_key in surfs0,"ICA/IJV stable key missing")
raw_faces=np.asarray(surfs0[ica_key]["faces"],dtype=np.int64)
used=np.unique(raw_faces.reshape(-1)); vmap=np.full(len(xyz0),-1,dtype=np.int64); vmap[used]=np.arange(len(used),dtype=np.int64)
ica=ca._prepare_closed_clearance_target(np.asarray(xyz0[used],dtype="<f4"),vmap[raw_faces])
req(ica["report"]["closed_connected_oriented"] and ica["report"]["self_intersection_audit"]["count"]==0 and ica["report"]["embedded_closed_target"],"right ICA exact shell validation failed")
base_v=np.asarray(ica["vertices"],dtype=float); base_f=np.asarray(ica["faces"],dtype=np.int64)
fcross=np.cross(base_v[base_f[:,1]]-base_v[base_f[:,0]],base_v[base_f[:,2]]-base_v[base_f[:,0]])
vol=float(np.einsum("ij,ij->i",base_v[base_f[:,0]],np.cross(base_v[base_f[:,1]],base_v[base_f[:,2]])).sum()/6)
req(abs(vol)>0,"ICA signed volume is zero")
if vol<0: fcross=-fcross
normals_v=np.zeros_like(base_v)
for c in range(3): np.add.at(normals_v,base_f[:,c],fcross)
nn=np.linalg.norm(normals_v,axis=1); req(np.isfinite(nn).all() and np.all(nn>0),"ICA vertex normals undefined")
normals_v/=nn[:,None]
offset_m=.0001
offset_v=np.asarray(base_v+offset_m*normals_v,dtype="<f4")
offset_d=np.linalg.norm(offset_v.astype(float)-base_v,axis=1)
offset_ica=ca._prepare_closed_clearance_target(offset_v,base_f)
req(offset_ica["report"]["closed_connected_oriented"] and offset_ica["report"]["self_intersection_audit"]["count"]==0 and offset_ica["report"]["embedded_closed_target"],"expanded ICA shell invalid")
def rational_vertices(points):
 return [tuple(Fraction.from_float(float(x)) for x in p) for p in np.asarray(points,dtype="<f4")]
sources={"SCM":{"vertices":rational_vertices(world0),"triangles":parent_faces.tolist(),"source_sha256":sha(PREV_NPZ)},
 "ICA":{"vertices":rational_vertices(offset_ica["vertices"]),"triangles":np.asarray(offset_ica["faces"],dtype=np.int64).tolist(),"source_sha256":sha(pack0)}}
signal.alarm(600)
children,arrangement_meta=cp.construct_arrangement(sources,source_names=("SCM","ICA"))
signal.alarm(0)
selected=[]; ancestry=[]
for ch in children:
 if ch["source"]=="SCM" and ch["other_location"]=="outside":
  selected.append(tuple(ch["vertices"]))
  p=first_ancestry[int(ch["source_face"])]
  ancestry.append({"source":"SCM","source_face":int(ch["source_face"]),"previous_source":p["source"],"previous_source_face":int(p["source_face"]),"reversed":False})
 elif ch["source"]=="ICA" and ch["other_location"]=="inside":
  t=tuple(ch["vertices"]); selected.append((t[0],t[2],t[1]))
  ancestry.append({"source":"ICA","source_face":int(ch["source_face"]),"reversed":True})
req(selected,"empty SCM-minus-ICA boundary")
mesh=cp.indexed_mesh(selected,convert=lambda x:x)
exact_q0=mesh["vertices_m"]; faces=np.asarray(mesh["triangles"],dtype=np.int64)
q0=np.asarray([[np.float32(float(x)) for x in p] for p in exact_q0],dtype="<f4")
req(len(faces)==len(ancestry),"ICA face ancestry mismatch")
qr=ci._records(exact_q0,faces.tolist()); qs=ci._audit_pair(qr,qr,same_surface=True); qt=cp._validate_output_topology(qr)
mesh32=cp.indexed_mesh(selected,convert=np.float32); q32=np.asarray(mesh32["vertices_m"],dtype="<f4"); q32f=np.asarray(mesh32["triangles"],dtype=np.int64)
q32r=ci._records([tuple(Fraction.from_float(float(v)) for v in p) for p in q32],q32f.tolist())
q32s=ci._audit_pair(q32r,q32r,same_surface=True); q32t=cp._validate_output_topology(q32r)
req(qs["count"]==0 and qt["closed_oriented_2_manifold"] and q32s["count"]==0 and q32t["closed_oriented_2_manifold"],"ICA Boolean q0 exact/F32 self/topology failed")
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
 alpha_world=h.forward(alpha_src,alpha_local,alpha_w,row,tissue,poses).astype("<f4")
 br0,bss0,btt0=exact_mesh(alpha_world,alpha_faces)
 intermediate=h.forward(parent_src,parent_local,parent_w,row,tissue,poses).astype("<f4")
 br1,bss1,btt1=exact_mesh(intermediate,parent_faces)
 bt0=ca._target_intersection_audit(br0,tf,xyz,_validated_target_geometry_cache=cache)
 bt1=ca._target_intersection_audit(br1,tf,xyz,_validated_target_geometry_cache=cache)
 ct=ca._target_intersection_audit(cr,tf,xyz,_validated_target_geometry_cache=cache)
 expected={f"{k[0]}:{k[1]}" for k in target_keys|{opp_key}}
 req(set(bt0)==set(bt1)==set(ct)==expected,f"step{step} target-key coverage mismatch")
 def pair_ids(audit,kind):
  out=[]
  for skinface,targetface in audit["triangle_pairs"]:
   if kind=="alpha": lin=("alpha_scm",int(skinface))
   elif kind=="intermediate":
    a=first_ancestry[int(skinface)]
    lin=("alpha_scm",int(a["source_face"])) if a["source"]=="SCM" else ("ijv_cap",int(a["source_face"]))
   else:
    a=ancestry[int(skinface)]
    if a["source"]=="ICA": lin=("ica_cap",int(a["source_face"]))
    else:
     p=first_ancestry[int(a["source_face"])]
     lin=("alpha_scm",int(p["source_face"])) if p["source"]=="SCM" else ("ijv_cap",int(p["source_face"]))
   out.append((lin[0],lin[1],int(targetface)))
  return Counter(out)
 def delta(a,b):
  add=b-a; rem=a-b
  return {"added":[[x[0],x[1],x[2],int(n)] for x,n in sorted(add.items())],
          "removed":[[x[0],x[1],x[2],int(n)] for x,n in sorted(rem.items())]}
 changed={}; unexpected=[]
 for k in sorted(ct):
  a=pair_ids(bt0[k],"alpha"); b=pair_ids(bt1[k],"intermediate"); c=pair_ids(ct[k],"candidate")
  d0=delta(a,c); d1=delta(b,c)
  if bt0[k]["count"]!=ct[k]["count"] or d0["added"] or d0["removed"]:
   changed[k]={"label":inventory_labels.get(k,"unlabeled target"),"alpha03":bt0[k]["count"],"ijv_only":bt1[k]["count"],"candidate":ct[k]["count"],
    "added_vs_alpha03":d0["added"],"removed_vs_alpha03":d0["removed"],"added_vs_ijv_only":d1["added"],"removed_vs_ijv_only":d1["removed"]}
   if k not in ("51011:466","51011:468"): unexpected.append(k)
 interest={k:{"label":inventory_labels[k],"alpha03":bt0[k]["count"],"ijv_only":bt1[k]["count"],"candidate":ct[k]["count"]}
  for k in ("51011:466","51011:468","51007:1","51005:152")}
 poses_out.append({"step":step,"self_pairs":ss["count"],"topology":tt,"alpha03_target_total":sum(v["count"] for v in bt0.values()),
  "ijv_only_target_total":sum(v["count"] for v in bt1.values()),"candidate_target_total":sum(v["count"] for v in ct.values()),
  "key_targets":interest,"unexpected_changed_target_keys":unexpected,"changed_target_surfaces":changed})

NPZ=RESULT/"unadmitted-scm-minus-ica-ijv-source-lift.npz"
np.savez(NPZ,positions_m=src,faces=faces.astype("<i8"),normals=normals,binding_indices=local,weights=weights,
 parent_face_vertices=pface,barycentric=barys,boolean_q0_world_m=q0,face_ancestry_json=np.asarray(json.dumps(ancestry,sort_keys=True)))
mesh_path=RESULT/"scm-minus-ica-ijv.exact.json"
mesh_path.write_text(json.dumps({"schema":"offline.scm-minus-expanded-ica-after-ijv.exact-mesh.v1","vertices_m":cp.encode_rational(exact_q0),
 "triangles":faces.tolist(),"face_ancestry":ancestry},sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")
def volume(vertices,triangles):
 value=Fraction(0)
 for f in triangles:
  a,b,c=(vertices[int(i)] for i in f)
  value+=(a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]))/6
 return value
prev_vol=volume(rational_vertices(world0),parent_faces)
alpha_world0=h.forward(alpha_src,alpha_local,alpha_w,row,tissue,poses0).astype("<f4")
alpha_vol=volume(rational_vertices(alpha_world0),alpha_faces); candidate_vol=volume(exact_q0,faces)
alpha_lookup={p.tobytes():i for i,p in enumerate(alpha_world0)}
original_retained=[]; max_original_delta=0.
for i,p in enumerate(q0):
 j=alpha_lookup.get(p.tobytes())
 if j is not None:
  original_retained.append((i,j)); max_original_delta=max(max_original_delta,float(np.linalg.norm(src[i].astype(float)-alpha_src[j].astype(float))))
alpha_pure=((alpha_w==1.0).sum(axis=1)==1)&((alpha_w==0.0).sum(axis=1)==3)
pure_bytes={p.tobytes() for p in alpha_world0[alpha_pure]}; pure_retained=sum(p.tobytes() in pure_bytes for p in q0)
source_vol=volume(rational_vertices(src),faces); alpha_source_vol=volume(rational_vertices(alpha_src),alpha_faces)
report={"schema":"offline.scm-ica-ijv-combined-source-attribute-lift.v1","status":"unadmitted_candidate",
 "scope":"Inferred second Boolean subtraction of the right internal carotid artery from the retained IJV-cleared SCM intermediate, lifted through the existing SCM binding table; not measured anatomy or an admitted asset.",
 "input_pins":pins,"script_sha256":sha(SCRIPT),
 "parent":{"alpha03_patch_sha256":pins[str(PATCH)],"alpha03_tiss_sha256":pins[str(TISS)],"stable_id":151,"alpha03_vertices":len(alpha_src),"alpha03_faces":len(alpha_faces),
  "ijv_intermediate_npz_sha256":pins[str(PREV_NPZ)],"intermediate_vertices":len(parent_src),"intermediate_faces":len(parent_faces),"binding_cores":[int(x) for x in bind["core"]]},
 "targets":{"inventory_path":str(INV),"inventory_sha256":pins[str(INV)],"inventory_surface_count":len(legacy),"native_skin_key":"51007:1","opposite_scm_key":"51005:152","per_pose_total":861,
  "right_ica":{"key":"51011:466","label":inventory_labels["51011:466"],"source_member_sha256":nha["source_surfaces"]["466"]["source_sha256"],
   "source_member":nha["source_surfaces"]["466"]["source_member"]},
  "right_ijv":{"key":"51011:468","label":inventory_labels["51011:468"],"source_member_sha256":nha["source_surfaces"]["468"]["source_sha256"],
   "source_member":nha["source_surfaces"]["468"]["source_member"]}},
 "boolean":{"target_key":"51011:466","offset_m":offset_m,"offset_vertex_min_m":float(offset_d.min()),"offset_vertex_max_m":float(offset_d.max()),
  "expanded_shell_validation":offset_ica["report"],"arrangement":arrangement_meta,"output_vertices":len(exact_q0),"output_faces":len(faces),
  "ancestry_counts":{"prior_scm_children":sum(x["source"]=="SCM" and x["previous_source"]=="SCM" for x in ancestry),
   "prior_ijv_cap_children":sum(x["source"]=="SCM" and x["previous_source"]=="IJV" for x in ancestry),"new_ica_cap_faces":sum(x["source"]=="ICA" for x in ancestry)},
  "exact_self_pairs":qs["count"],"exact_topology":qt,"float32_self_pairs":q32s["count"],"float32_topology":q32t,
  "q0_signed_volume_m3":cp.encode_rational(candidate_vol),"q0_volume_removed_by_ica_m3":cp.encode_rational(prev_vol-candidate_vol),
  "q0_volume_removed_vs_alpha03_m3":cp.encode_rational(alpha_vol-candidate_vol)},
 "lift":{"q0_max_forward_error_m":float(q0_err.max()),"q0_p99_forward_error_m":float(np.quantile(q0_err,.99)),
  "intermediate_q0_vertices_retained":kept,"new_ica_cap_vertices":created,"max_parent_surface_projection_m":maxproj,
  "max_inserted_source_offset_from_interpolated_support_m":maxdelta,"alpha03_original_q0_vertices_retained":len(original_retained),
  "max_source_delta_on_retained_alpha03_vertices_m":max_original_delta,"alpha03_pure_onehot_vertices":int(alpha_pure.sum()),
  "alpha03_pure_onehot_q0_points_retained":pure_retained,"certified_attachment_mask_available":False,"binding_table_and_routes_unchanged":True,
  "rule":"Exact q0 vertices from the IJV-only intermediate retain ordered source positions and binding records. New ICA cap vertices use nearest intermediate face, barycentric interpolation of its three existing core weights normalized on new rows only, then inverse weighted q0 transform. Cap support distance is not existing-source displacement."},
 "source":{"self_pairs":src_self["count"],"topology":src_topo,"positions_sha256":ca._float32_xyz_sha256(src),"face_sha256":ca._face_index_sha256(faces),
  "signed_volume_m3":cp.encode_rational(source_vol),"volume_delta_vs_ijv_intermediate_m3":cp.encode_rational(source_vol-volume(rational_vertices(parent_src),parent_faces)),
  "volume_delta_vs_alpha03_m3":cp.encode_rational(source_vol-alpha_source_vol)},
 "pose_checks":poses_out,
 "limitations":["Unadmitted inferred reference correction; 0.10mm expanded ICA shell is a search offset, not a measured tissue gap.",
  "Eight accepted current57 baseline poses only; no new native capture or full-body admission.",
  "861 target surfaces per pose: 859 inventory surfaces plus native skin and separately forwarded opposite SCM row152.",
  "No certified SCM attachment footprint; pure one-hot preservation is only a proxy.",
  "Generated cap vertices can be several mm from nearest source support; this is not applied to existing source vertices.",
  "Pair identities use original source-face ancestry where available and distinguish prior IJV and new ICA cap faces.",
  "No source TISS/runtime/native capture modified."],"elapsed_seconds":time.monotonic()-started}
for p,d in pins.items(): req(sha(p)==d,"input changed during diagnostic: "+p)
report=cp.encode_rational(report); rp=RESULT/"report.json"; rp.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
print(json.dumps({"status":report["status"],"script_sha256":sha(SCRIPT),"npz_sha256":sha(NPZ),"report":str(rp),"report_sha256":sha(rp),
 "mesh_sha256":sha(mesh_path),"generated":created,"max_cap_support_offset_m":maxdelta,"source_self":src_self["count"],
 "poses":[(x["step"],x["self_pairs"],x["alpha03_target_total"],x["candidate_target_total"],x["unexpected_changed_target_keys"]) for x in poses_out],
 "elapsed":report["elapsed_seconds"]},sort_keys=True),flush=True)
