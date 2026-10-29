from __future__ import annotations
import csv, hashlib, importlib.util, json, sys, time, types, traceback, shutil
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
OUT=P/"reduced-physical-boundary-400/attempt-044-minimal-coupled-bone32-buffer/run-007-inner-mask-coupled-buffer"
BASE=P/"reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz"
LEDGER=P/"refined-reference-ledger-534/run-001/child-parent-reference-ledger.npz"
BANK=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/bank-envelopes/bank-023-032-cube2um.npz"
BASE_EXPECTED="888eb12450d0e69fb0ad3f5998159ab7a0a20d3e70b13c42ba416c61cbbede63"
LEDGER_EXPECTED="ed320a79755132a7418cd7dafbb97d034b491ee5ccba587e33324bc7e61b0210"
BANK_EXPECTED="4cb25b1387af6caabb3479ff990ac27b49d1592817af508e8e3d78d9d6f29067"
TOOLS=Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001/tools")
SOURCE=Path(__file__)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wj(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+"\n")
def outface(t,o):
 q=[int(t[j]) for j in range(4) if j!=o]
 if o%2:q[1],q[2]=q[2],q[1]
 return tuple(q)
def label(m,d,r,i): return (int(m[i]),int(d[i]),int(r[i]))
def tet_surface(Pt,T,rows):
 inc=defaultdict(list)
 for ci in map(int,rows):
  tt=T[ci]
  for o in range(4):
   cy=outface(tt,o)
   inc[tuple(sorted(cy))].append((ci,cy))
 ff=[]; parent=[]; opposite=[]
 for k,items in inc.items():
  if len(items)>2: raise ValueError("tet group has nonmanifold face incidence")
  if len(items)==1:
   ci,cy=items[0];ff.append(cy);parent.append(ci)
 F=np.asarray(ff,np.int64)
 if len(F)==0: raise ValueError("empty source group")
 E=np.concatenate((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]),axis=0)
 _,cnt=np.unique(np.sort(E,axis=1),axis=0,return_counts=True)
 if np.any(cnt!=2): raise ValueError(f"group boundary is not closed: edge incidence {Counter(cnt.tolist())}")
 return F,np.asarray(parent,np.int64)
def manifold_mesh(mf,Pp,Ff,runid):
 mesh=mf.Mesh64(vert_properties=np.ascontiguousarray(Pp,np.float64),tri_verts=np.ascontiguousarray(Ff,np.uint64),
  run_index=np.asarray([0,3*len(Ff)],np.uint64),run_original_id=np.asarray([runid],np.uint32),face_id=np.arange(len(Ff),dtype=np.uint64))
 m=mf.Manifold(mesh)
 if m.status()!=mf.Error.NoError: raise RuntimeError(f"Manifold import status {m.status()}")
 return m
def emit(m,path,stage):
 q=m.to_mesh64();p=np.asarray(q.vert_properties,np.float64)[:,:3];f=np.asarray(q.tri_verts,np.int64)
 np.savez_compressed(path,points_world_m_f64=p,faces_f64=f,run_index=np.asarray(q.run_index,np.uint64),
  run_original_id=np.asarray(q.run_original_id,np.uint32),face_id=np.asarray(q.face_id,np.uint64),
  stage=np.asarray([stage]),physical_admission=np.asarray([False]))
 return p,f
def exact_setup():
 pkg=types.ModuleType("numi_tools");pkg.__path__=[str(TOOLS)];sys.modules["numi_tools"]=pkg
 for n,attrs in [("cardiac_cavity_geometry",{"analyze_topology":lambda *a,**k:None}),("model",{"ImportError":RuntimeError}),("physiology",{"canonical":lambda x:x})]:
  z=types.ModuleType("numi_tools."+n);z.__dict__.update(attrs);sys.modules[z.__name__]=z
 from numi_tools import cardiac_cavity_intersections as exact
 return exact
def exact_self(exact,p,f):
 rec,_=exact.float32_surface_records(np.asarray(p,np.float32),np.asarray(f,np.int64))
 r=exact._audit_pair(rec,rec,same_surface=True)
 pairs=np.asarray(r.get("triangle_pairs",[]),np.int64).reshape(-1,2)
 return {"face_records":len(rec),"face_count":len(f),"pair_count":int(r.get("count",len(pairs))),
  "aabb_candidate_pairs":int(r.get("aabb_candidate_pairs",0)),"pairs":pairs.tolist()}
def exact_cross(exact,pa,fa,pb,fb):
 a,_=exact.float32_surface_records(np.asarray(pa,np.float32),np.asarray(fa,np.int64))
 b,_=exact.float32_surface_records(np.asarray(pb,np.float32),np.asarray(fb,np.int64))
 r=exact._audit_pair(a,b,same_surface=False);pairs=np.asarray(r.get("triangle_pairs",[]),np.int64).reshape(-1,2)
 # Distinguish exact coincident faces from true nonshared pair witnesses.
 def vk(PP,tri): return tuple(sorted(tuple(np.asarray(PP[int(j)],np.float32).view(np.uint32).tolist()) for j in tri))
 amap={};bmap={}
 for i,tri in enumerate(fa):amap.setdefault(vk(pa,tri),[]).append(i)
 for j,tri in enumerate(fb):bmap.setdefault(vk(pb,tri),[]).append(j)
 shared={(i,j) for key in set(amap)&set(bmap) for i in amap[key] for j in bmap[key]}
 categories=Counter("shared_face" if tuple(map(int,x)) in shared else "nonshared_exact_intersection" for x in pairs)
 return {"pair_count":int(r.get("count",len(pairs))),"aabb_candidate_pairs":int(r.get("aabb_candidate_pairs",0)),
  "shared_face_pairs":len(shared),"nonshared_pairs":int(len(pairs)-len(shared)),"categories":dict(categories),"pairs":pairs.tolist()}
def main():
 t0=time.monotonic()
 if OUT.exists():raise FileExistsError(OUT)
 for p,h in [(BASE,BASE_EXPECTED),(LEDGER,LEDGER_EXPECTED),(BANK,BANK_EXPECTED)]:
  if not p.is_file() or sha(p)!=h:raise ValueError(f"input pin mismatch: {p}")
 OUT.mkdir(parents=True)
 src=OUT/"authoring-source-chain";src.mkdir()
 if not (src/SOURCE.name).exists():shutil.copy2(SOURCE,src/SOURCE.name)
 receipt={"schema":"numi.minimal-coupled-bone32-buffer.044.v1","status":"RUNNING","physical_admission":False,
  "runtime_requested":False,"inputs":{str(p):sha(p) for p in (BASE,LEDGER,BANK,SOURCE)},
  "operation":"Subtract the pinned 2um body23/bone32 envelope from source current006 material/donor/region 23 solids; transfer an 8.5mL buffer harvested from the adjacent skin23 material band into fat23 through a common derived interface.",
  "scope":{"source_groups":[[15,23,23],[16,23,23]],"outside_groups_untouched_by_construction":True,"source_mass_and_density_limits_unchanged":True,
   "buffer_volume_target_m3":8.5e-6,"buffer_selection":"skin23 intersect (Minkowski expansion of pinned bank32 2um envelope minus the same envelope); radius solved from measured CSG buffer volume, not a fixed physical clearance."}}
 wj(OUT/"receipt.json",receipt)
 try:
  sys.path.insert(0,str(P.parent/"authoring-deps"));import manifold3d as mf
  with np.load(BASE,allow_pickle=False) as z:b={k:np.asarray(z[k]) for k in z.files}
  with np.load(LEDGER,allow_pickle=False) as z:ledger={k:np.asarray(z[k]) for k in z.files}
  p=np.asarray(b["points_reference_world_m_f64"],np.float32).astype(np.float64)
  t=np.asarray(b["volume_tetrahedra"],np.int64);m=np.asarray(b["material_index"],np.int64)
  d=np.asarray(b["geometric_donor_body"],np.int64);r=np.asarray(b["anatomical_region_id"],np.int64)
  if not np.array_equal(ledger["material_index"],m) or not np.array_equal(ledger["geometric_donor_body"],d) or not np.array_equal(ledger["anatomical_region_id"],r):
   raise ValueError("534 source ledger does not match current006 labels")
  group_s=(m==15)&(d==23)&(r==23);group_f=(m==16)&(d==23)&(r==23)
  rs=np.flatnonzero(group_s);rf=np.flatnonzero(group_f)
  Fs,sparents=tet_surface(p,t,rs);Ff,fparents=tet_surface(p,t,rf)
  # Keep only used source vertices and save an exact row map for this regional view.
  def compact(F):
   used=np.unique(F);remap=np.full(len(p),-1,np.int64);remap[used]=np.arange(len(used))
   return p[used],remap[F],used
  ps,fs,sv=compact(Fs);pf,ff,fv=compact(Ff)
  skin=manifold_mesh(mf,ps,fs,440015);fat=manifold_mesh(mf,pf,ff,440016)
  if skin.status()!=mf.Error.NoError or fat.status()!=mf.Error.NoError:raise ValueError("baseline regional solids invalid")
  with np.load(BANK,allow_pickle=False) as z: bp=np.asarray(z["points_world_m_f64"],np.float64);bf=np.asarray(z["faces_f64"],np.int64)
  bone=manifold_mesh(mf,bp,bf,440032)
  base_skin_vol=float(skin.volume());base_fat_vol=float(fat.volume());bone_vol=float(bone.volume())
  skin_mass=float(np.asarray(ledger["child_mass_kg"],np.float64)[group_s].sum())
  fat_mass=float(np.asarray(ledger["child_mass_kg"],np.float64)[group_f].sum())
  bounds={"skin":[skin_mass/1100.,skin_mass/900.],"fat":[fat_mass/950.,fat_mass/850.]}
  bone_skin=skin^bone;bone_fat=fat^bone
  if bone_skin.status()!=mf.Error.NoError or bone_fat.status()!=mf.Error.NoError:raise ValueError("baseline bone regional intersection invalid")
  cut_skin=float(bone_skin.volume());cut_fat=float(bone_fat.volume())
  receipt["baseline"]={"skin23":{"tet_boundary_faces":len(Fs),"used_vertices":len(ps),"solid_volume_m3":base_skin_vol,"immutable_mass_kg":skin_mass,"allowed_volume_m3":bounds["skin"],"bone32_intersection_m3":cut_skin},
   "fat23":{"tet_boundary_faces":len(Ff),"used_vertices":len(pf),"solid_volume_m3":base_fat_vol,"immutable_mass_kg":fat_mass,"allowed_volume_m3":bounds["fat"],"bone32_intersection_m3":cut_fat},
   "bone32_envelope_volume_m3":bone_vol,"combined_bone_cut_m3":cut_skin+cut_fat}
  wj(OUT/"receipt.json",receipt)
  mask_path=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/reduced-physical-boundary-400/attempt-044-minimal-coupled-bone32-buffer/inner-half-skin-clipped-domain-002/clipped-inner-skin-domain.npz')
  if sha(mask_path)!='678dd49df4a03f4426b3d6a745e997324952f9cd52d6556611859ea467052537':raise ValueError("eligibility mask pin mismatch")
  with np.load(mask_path,allow_pickle=False) as iz:
   ip=np.asarray(iz["points_world_m_f64"],np.float64);ifi=np.asarray(iz["faces_f64"],np.int64)
  inner_mask=manifold_mesh(mf,ip,ifi,440047)
  if inner_mask.status()!=mf.Error.NoError:raise ValueError("inner mask import failed")
  receipt["buffer_eligibility"]={"path":str(mask_path),"sha256":sha(mask_path),"construction":"source positive-tet scalar clip O=1/I=0 <=0.5, protecting outer half of original skin band","volume_m3":float(inner_mask.volume()),"source_boundary_coordinates_modified":False}
  target=8.5e-6;radius=0.002;best=None;trials=[]
  for k in range(4):
   cube=mf.Manifold.cube([2*radius]*3,True)
   dilated=bone.minkowski_sum(cube)
   if dilated.status()!=mf.Error.NoError:raise RuntimeError(f"bone dilation failed at {radius}m: {dilated.status()}")
   band=mf.Manifold.batch_boolean([dilated,bone],mf.OpType.Subtract)
   if band.status()!=mf.Error.NoError:raise RuntimeError(f"bone band failed: {band.status()}")
   buf=mf.Manifold.batch_boolean([skin,band,inner_mask],mf.OpType.Intersect)
   if buf.status()!=mf.Error.NoError:raise RuntimeError(f"skin-band intersection failed: {buf.status()}")
   vol=float(buf.volume())
   trials.append({"radius_m":radius,"dilated_faces":int(dilated.num_tri()),"band_volume_m3":float(band.volume()),"skin_buffer_volume_m3":vol,"buffer_status":str(buf.status())})
   if best is None or abs(vol-target)<abs(best[0]-target):best=(vol,radius,buf,dilated,band)
   if 7.8e-6<=vol<=9.2e-6:break
   if vol<=0:radius*=2
   else:radius=min(0.004,max(0.0001,radius*target/vol))
  bufvol,radius,buffer,dilated,band=best
  receipt["buffer_fit"]={"target_m3":target,"radius_m":radius,"measured_skin23_buffer_m3":bufvol,"trials":trials,
   "buffer_is_closed":str(buffer.status()),"buffer_component_count":len(buffer.decompose())}
  wj(OUT/"buffer-fit.json",receipt["buffer_fit"])
  emit(buffer,OUT/"skin-to-fat-buffer-region23-f64.npz","skin23-to-fat23-buffer")
  # Bone removal and paired material transfer share the exact same buffer and envelope operands.
  skin_cut=mf.Manifold.batch_boolean([skin,bone],mf.OpType.Subtract)
  fat_cut=mf.Manifold.batch_boolean([fat,bone],mf.OpType.Subtract)
  skin_new=mf.Manifold.batch_boolean([skin_cut,buffer],mf.OpType.Subtract)
  fat_plus=mf.Manifold.batch_boolean([fat,buffer],mf.OpType.Add)
  fat_new=mf.Manifold.batch_boolean([fat_plus,bone],mf.OpType.Subtract)
  for name,obj in (("skin-after-bone",skin_cut),("fat-after-bone",fat_cut),("skin-new",skin_new),("fat-new",fat_new)):
   if obj.status()!=mf.Error.NoError:raise RuntimeError(f"{name} invalid: {obj.status()}")
  skin_v=float(skin_new.volume());fat_v=float(fat_new.volume())
  cap={"skin23":{"volume_m3":skin_v,"density_kg_m3":skin_mass/skin_v,"bounds_kg_m3":[900.,1100.],"allowed_volume_m3":bounds["skin"],"pass":bounds["skin"][0]<=skin_v<=bounds["skin"][1]},
       "fat23":{"volume_m3":fat_v,"density_kg_m3":fat_mass/fat_v,"bounds_kg_m3":[850.,950.],"allowed_volume_m3":bounds["fat"],"pass":bounds["fat"][0]<=fat_v<=bounds["fat"][1]}}
  si=mf.Manifold.batch_boolean([skin_new,fat_new],mf.OpType.Intersect);sb=mf.Manifold.batch_boolean([skin_new,bone],mf.OpType.Intersect);fb=mf.Manifold.batch_boolean([fat_new,bone],mf.OpType.Intersect)
  if any(x.status()!=mf.Error.NoError for x in (si,sb,fb)):raise RuntimeError("final set-intersection screen failed")
  receipt["candidate"]={"skin_volume_m3":skin_v,"fat_volume_m3":fat_v,"skin_component_count":len(skin_new.decompose()),"fat_component_count":len(fat_new.decompose()),
   "bone32_skin_intersection_m3":float(sb.volume()),"bone32_fat_intersection_m3":float(fb.volume()),"skin_fat_overlap_m3":float(si.volume()),
   "capacity":cap,"joint_material_volume_delta_m3":skin_v+fat_v-base_skin_vol-base_fat_vol,
   "net_fat_change_from_bone_cut_and_transfer_m3":fat_v-base_fat_vol}
  skp,skf=emit(skin_new,OUT/"candidate-skin-region23-f64.npz","skin23-bone32-buffered")
  ftp,ftf=emit(fat_new,OUT/"candidate-fat-region23-f64.npz","fat23-bone32-buffered")
  np.savez_compressed(OUT/"source-view-maps.npz",skin_global_volume_node_rows=sv,fat_global_volume_node_rows=fv,
    skin_source_tet_rows=sparents,fat_source_tet_rows=fparents,selected_skin_source_tet_rows=rs,selected_fat_source_tet_rows=rf,
    current006_sha256=np.asarray([BASE_EXPECTED]),ledger534_sha256=np.asarray([LEDGER_EXPECTED]))
  np.savez_compressed(OUT/"buffer-mesh.npz",points_world_m_f64=np.asarray(buffer.to_mesh64().vert_properties,np.float64)[:,:3],
    faces_f64=np.asarray(buffer.to_mesh64().tri_verts,np.int64),source_bank32_sha256=np.asarray([BANK_EXPECTED]))
  np.savez_compressed(OUT/"mask-inputs.npz",bone32_points_world_m_f64=bp,bone32_faces=bf,
    source_skin23_faces_global_node_rows=Fs,source_fat23_faces_global_node_rows=Ff,
    source_skin23_used_global_nodes=sv,source_fat23_used_global_nodes=fv)
  cheap={}
  for role,pp,ff in [("skin",skp,skf),("fat",ftp,ftf)]:
   qq=np.asarray(pp,np.float32).astype(np.float64);tt=qq[ff]
   zero=int(np.sum(np.all(np.cross(tt[:,1]-tt[:,0],tt[:,2]-tt[:,0])==0,axis=1)))
   cheap[role]={"vertices":len(pp),"faces":len(ff),"f32_zero_area_faces":zero}
  # Face ancestry tags identify old pure-OOO source facets independently of material row IDs.
  rolemap={}
  with (P/"conforming-fit-006-native-assets-545/run-002-operational-reference/registered-assets/volume/combined-node-map.csv").open() as cf:
   for row in csv.DictReader(cf):rolemap[int(row["joint_node_id"])-1]=row["role"]
  ooos={i for i,tri in enumerate(Fs) if all(rolemap[int(n)]=="O" for n in tri)}
  with np.load(OUT/"candidate-fat-region23-f64.npz",allow_pickle=False) as az:
   ix=np.asarray(az["run_index"],np.int64)//3;sid=np.asarray(az["run_original_id"],np.int64);fid=np.asarray(az["face_id"],np.int64)
   tags=np.empty(len(ftp[ftf]),np.int64)
   for k,v in enumerate(sid):tags[ix[k]:ix[k+1]]=v
   ex=np.flatnonzero((tags==440015)&np.isin(fid,list(ooos)))
   qq=ftp[ftf[ex]];area=float(np.linalg.norm(np.cross(qq[:,1]-qq[:,0],qq[:,2]-qq[:,0]),axis=1).sum()*.5) if len(ex) else 0.
  receipt["cheap_geometry"]=cheap
  receipt["fat_old_outer_skin_facet_inventory"]={"face_count":len(ex),"area_m2":area,"source_face_rows":sorted(set(map(int,fid[ex]))),"does_not_alone_establish_finite_cover":True}
  allcap=all(v["pass"] for v in cap.values())
  receipt.update({"status":"SOURCE_ONLY_CANDIDATE_CAPACITY_AND_CHEAP_GEOMETRY" if allcap else "LOCAL_BUFFER_VOLUME_CAPACITY_FAIL","physical_admission":False,"runtime_requested":False,"runtime_created":0,"accepted_steps":0,"exact_geometry_audit":"NOT_RUN_PENDING_F32_NONDEGENERACY_AND_INTERFACE_REPAIR","elapsed_s":time.monotonic()-t0,
   "notes":["This reconstructs a reduced physical layer boundary only; detailed accepted O/back sources are unchanged.","The new buffer is limited to original scalar-clipped inner Skin cells; all material masses/donor/region labels and density bounds remain fixed.","No TetGen/native mass flow/runtime or motion check was run."]})
  wj(OUT/"receipt.json",receipt)
  print(json.dumps({"status":receipt["status"],"buffer_mL":bufvol*1e6,"radius_mm":radius*1e3,"skin_density":skin_mass/skin_v,"fat_density":fat_mass/fat_v,"cheap_geometry":cheap,"fat_ooo_faces":len(ex),"elapsed_s":time.monotonic()-t0,"out":str(OUT)},indent=2),flush=True)
 except Exception as e:
  receipt.update({"status":"FAILED","exception":repr(e),"traceback":traceback.format_exc(),"elapsed_s":time.monotonic()-t0})
  wj(OUT/"receipt.json",receipt);raise
if __name__=="__main__":main()
