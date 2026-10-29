from __future__ import annotations
import csv, hashlib, json, sys, types, time
from pathlib import Path
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
A=P/"reduced-physical-boundary-400/attempt-044-minimal-coupled-bone32-buffer"
R=A/"run-007-inner-mask-coupled-buffer"
BASE=P/"reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz"
SKIN=R/"candidate-skin-region23-f64.npz"
FAT=R/"candidate-fat-region23-f64.npz"
MASK=R/"mask-inputs.npz"
MAP=P/"conforming-fit-006-native-assets-545/run-002-operational-reference/registered-assets/volume/combined-node-map.csv"
OUT=R/"source-only-audit-004"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def outface(t,o):
 q=[int(t[j]) for j in range(4) if j!=o]
 if o%2:q[1],q[2]=q[2],q[1]
 return tuple(q)
def tet_surface(Pt,T,rows):
 inc={}
 for ci in map(int,rows):
  tt=T[ci]
  for o in range(4):
   cy=outface(tt,o); key=tuple(sorted(cy))
   inc.setdefault(key,[]).append((ci,cy))
 ff=[]; parent=[]
 for items in inc.values():
  if len(items)>2: raise ValueError("nonmanifold source face")
  if len(items)==1: ff.append(items[0][1]);parent.append(items[0][0])
 return np.asarray(ff,np.int64),np.asarray(parent,np.int64)
def import_exact():
 tools=Path("/Users/numi/Work/numi-human-connected-surface-069/D/publication-206-stage-001/tools")
 pkg=types.ModuleType("numi_tools");pkg.__path__=[str(tools)];sys.modules["numi_tools"]=pkg
 for n,attrs in [("cardiac_cavity_geometry",{"analyze_topology":lambda *a,**k:None}),("model",{"ImportError":RuntimeError}),("physiology",{"canonical":lambda x:x})]:
  z=types.ModuleType("numi_tools."+n);z.__dict__.update(attrs);sys.modules[z.__name__]=z
 import importlib; exact=importlib.import_module("numi_tools.cardiac_cavity_intersections")
 return exact
def tags(z,nfaces):
 ix=np.asarray(z["run_index"],np.int64)//3
 ids=np.asarray(z["run_original_id"],np.int64)
 face_id=np.asarray(z["face_id"],np.int64)
 t=np.full(nfaces,-1,np.int64)
 for k,v in enumerate(ids):t[ix[k]:ix[k+1]]=v
 if np.any(t<0): raise ValueError("unassigned output run tag")
 return t,face_id
def areas(points,faces):
 tri=np.asarray(points,np.float64)[np.asarray(faces,np.int64)]
 return .5*np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
def main():
 t0=time.monotonic()
 inp=[BASE,SKIN,FAT,MASK,MAP,A/"raw-fat-skin-source-role-inventory.json"]
 with np.load(BASE,allow_pickle=False) as z:b={k:np.asarray(z[k]) for k in z.files}
 p=np.asarray(b["points_reference_world_m_f64"],np.float32).astype(np.float64)
 T=np.asarray(b["volume_tetrahedra"],np.int64)
 mat=np.asarray(b["material_index"],np.int64);don=np.asarray(b["geometric_donor_body"],np.int64);reg=np.asarray(b["anatomical_region_id"],np.int64)
 rows=np.flatnonzero((mat==15)&(don==23)&(reg==23))
 Fs,_=tet_surface(p,T,rows)
 role={}
 with MAP.open() as f:
  for q in csv.DictReader(f): role[int(q["joint_node_id"])-1]=q["role"]
 ooo=np.array([i for i,f in enumerate(Fs) if all(role[int(n)]=="O" for n in f)],np.int64)
 orig_area=areas(p,Fs)
 with np.load(SKIN,allow_pickle=False) as z: sp=np.asarray(z["points_world_m_f64"],np.float64);sf=np.asarray(z["faces_f64"],np.int64);st,sid=tags(z,len(sf))
 with np.load(FAT,allow_pickle=False) as z: fp=np.asarray(z["points_world_m_f64"],np.float64);ff=np.asarray(z["faces_f64"],np.int64);ft,fid=tags(z,len(ff))
 # Sum output areas by the original source face_id tags. The map is only meaningful when the
 # output run_original_id denotes the pinned old skin operand (440015).
 out_by_source=np.zeros(len(Fs),np.float64)
 sel=(st==440015)&(sid>=0)&(sid<len(Fs))
 np.add.at(out_by_source,sid[sel],areas(sp,sf[sel]))
 records=[]
 for i in ooo:
  records.append({"source_face_row":int(i),"original_area_m2":float(orig_area[i]),"output_skin_area_m2":float(out_by_source[i]),
    "coverage_area_ratio":float(out_by_source[i]/orig_area[i]) if orig_area[i]>0 else None})
 fat_ooo=np.flatnonzero((ft==440016)&np.isin(fid,ooo))
 # Exact input F32 old OOO-vs-bone triangle intersections; diagnostic only.
 with np.load(MASK,allow_pickle=False) as z: bp=np.asarray(z["bone32_points_world_m_f64"],np.float64);bf=np.asarray(z["bone32_faces"],np.int64)
 exact=import_exact()
 def valid_f32(PP,FF):
  q=np.asarray(PP,np.float32)[np.asarray(FF,np.int64)]
  c=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0])
  return np.isfinite(q).all(axis=(1,2)) & np.any(c!=0,axis=1)
 valid_ooo=valid_f32(p,Fs[ooo]); valid_bone=valid_f32(bp,bf)
 valid_ooo_rows=ooo[valid_ooo]; valid_bone_rows=np.flatnonzero(valid_bone)
 a,_=exact.float32_surface_records(np.asarray(p,np.float32),Fs[valid_ooo_rows])
 b_records,_=exact.float32_surface_records(np.asarray(bp,np.float32),bf[valid_bone])
 report=exact._audit_pair(a,b_records,same_surface=False)
 pairs_local=np.asarray(report.get("triangle_pairs",[]),np.int64).reshape(-1,2)
 pairs=np.column_stack((valid_ooo_rows[pairs_local[:,0]],valid_bone_rows[pairs_local[:,1]])) if len(pairs_local) else np.empty((0,2),np.int64)
 np.save(OUT/"old-OOO-vs-bone32-exact-pairs.npy",pairs)
 payload={"status":"SOURCE_ONLY_AREA_AND_PAIR_INVENTORY","inputs":{str(x):sha(x) for x in inp},
  "base_source_geometry":{"point_count":len(p),"tet_count":len(T),"skin23_source_tet_count":len(rows),"skin23_boundary_face_count":len(Fs)},
  "old_pure_OOO":{"face_count":len(ooo),"total_original_area_m2":float(orig_area[ooo].sum()),
    "output_skin_tagged_area_m2":float(out_by_source[ooo].sum()),"area_ratio_total":float(out_by_source[ooo].sum()/orig_area[ooo].sum()),
    "fully_preserved_area_count":int(np.sum(np.isclose(out_by_source[ooo],orig_area[ooo],rtol=1e-10,atol=1e-18))),
    "zero_output_area_count":int(np.sum(out_by_source[ooo]==0)),"per_source_face":records},
  "candidate_fat_ooo":{"count":int(len(fat_ooo)),"area_m2":float(areas(fp,ff[fat_ooo]).sum()) if len(fat_ooo) else 0.0,
    "source_face_rows":sorted(set(map(int,fid[fat_ooo])))},
  "bone32_vs_original_OOO":{"pair_count":int(report.get("count",len(pairs))),
    "aabb_candidate_pairs":int(report.get("aabb_candidate_pairs",0)),"valid_old_ooo_triangles":int(valid_ooo.sum()),"f32_degenerate_old_ooo_rows":ooo[~valid_ooo].tolist(),"valid_bone32_triangles":int(valid_bone.sum()),"f32_degenerate_bone32_rows":np.flatnonzero(~valid_bone).tolist(),"pair_rows_are_original_face_rows":["skin23_source_face_row","bank32_face_row"],"pairs_path":str(OUT/"old-OOO-vs-bone32-exact-pairs.npy"),
    "interpretation":"raw exact surface witnesses; not a finite-cover or admission claim"},
  "candidate_f32_collapse":{"skin_faces":len(sf),"skin_zero_area":int(np.sum(np.all(np.cross(np.asarray(sp[sf],np.float32)[:,1]-np.asarray(sp[sf],np.float32)[:,0],np.asarray(sp[sf],np.float32)[:,2]-np.asarray(sp[sf],np.float32)[:,0])==0,axis=1))),
    "fat_faces":len(ff),"fat_zero_area":int(np.sum(np.all(np.cross(np.asarray(fp[ff],np.float32)[:,1]-np.asarray(fp[ff],np.float32)[:,0],np.asarray(fp[ff],np.float32)[:,2]-np.asarray(fp[ff],np.float32)[:,0])==0,axis=1)))},
  "elapsed_s":time.monotonic()-t0,"physical_admission":False,"runtime_created":0}
 (OUT/"report.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
 print(json.dumps({k:payload[k] for k in ("status","old_pure_OOO","candidate_fat_ooo","bone32_vs_original_OOO","candidate_f32_collapse","elapsed_s") if k!="old_pure_OOO" or True},indent=2)[:16000],flush=True)
if __name__=="__main__":main()
