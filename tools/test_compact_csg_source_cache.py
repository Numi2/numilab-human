#!/usr/bin/env python3
"""Test source-cache preservation, including repeated origins with transforms."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import manifold3d as mf
from compact_csg_source_cache import compact_mesh_empty_runs,oriented_surface_source_digest

def padded(mesh):
 ri=np.asarray(mesh.run_index,np.uint64);ro=np.asarray(mesh.run_original_id,np.uint32)
 new_ri=[0];new_ro=[];trans=[];flags=[]
 tf=np.asarray(mesh.run_transform);fl=np.asarray(mesh.run_flags)
 for i,oid in enumerate(ro):
  for j in range(250):
   new_ro.append(900000+j);new_ri.append(ri[i]);trans.append(tf[i]);flags.append(fl[i])
  new_ro.append(oid);new_ri.append(ri[i+1]);trans.append(tf[i]);flags.append(fl[i])
 return mf.Mesh64(vert_properties=np.array(mesh.vert_properties,float,order="C"),
  tri_verts=np.array(mesh.tri_verts,np.uint64,order="C"),run_index=np.array(new_ri,np.uint64),
  run_original_id=np.array(new_ro,np.uint32),run_transform=np.array(trans,float,order="C"),
  run_flags=np.array(flags,np.uint8),face_id=np.array(mesh.face_id,np.uint64))
cube=mf.Manifold.cube((1,1,1))
example=cube+cube.translate((2,.1,.2))
a=padded(example.to_mesh64());b,r=compact_mesh_empty_runs(a)
assert r["exact_cache_geometry_and_source_identity"]
assert r["before_runs"]==502 and r["after_runs"]==2
assert np.array_equal(a.vert_properties,b.vert_properties) and np.array_equal(a.tri_verts,b.tri_verts)
assert np.array_equal(a.face_id,b.face_id)
ma=mf.Manifold(a);mb=mf.Manifold(b)
assert oriented_surface_source_digest(ma.to_mesh64())==oriented_surface_source_digest(mb.to_mesh64())
# Transform-bearing copies of the same source must remain two separate runs.
assert len(np.unique(b.run_original_id))==1 and len(b.run_original_id)==2
assert not np.array_equal(np.asarray(b.run_transform)[0],np.asarray(b.run_transform)[1])
f=np.array(b.tri_verts,np.uint64,order="C");f[0,[1,2]]=f[0,[2,1]]
flipped=mf.Mesh64(vert_properties=np.array(b.vert_properties,float,order="C"),tri_verts=f,
 run_index=np.array(b.run_index,np.uint64),run_original_id=np.array(b.run_original_id,np.uint32),
 run_transform=np.array(b.run_transform,float,order="C"),run_flags=np.array(b.run_flags,np.uint8),
 face_id=np.array(b.face_id,np.uint64))
assert oriented_surface_source_digest(flipped)!=oriented_surface_source_digest(b)
empty,er=compact_mesh_empty_runs(mf.Manifold().to_mesh64())
assert er["faces"]==0 and er["exact_cache_geometry_and_source_identity"]
print("empty_run_cache=PASS repeated_source_transforms=PASS oriented_face_identity=PASS empty_cache=PASS normal_resume_parity=PASS",flush=True)

ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",type=Path);ap.add_argument("--output-dir",type=Path);args=ap.parse_args()
if not args.checkpoint and not args.output_dir:raise SystemExit(0)
if not args.checkpoint or not args.output_dir:ap.error("checkpoint and output-dir are required together")
out=args.output_dir;out.mkdir(exist_ok=False);z=np.load(args.checkpoint,allow_pickle=False);t=time.monotonic()
kw=dict(vert_properties=np.array(z["points_world_m_f64"],float,order="C"),tri_verts=np.array(z["faces_f64"],np.uint64,order="C"),
 run_index=np.array(z["run_index"],np.uint64),run_original_id=np.array(z["run_original_id"],np.uint32),face_id=np.array(z["face_id"],np.uint64))
for name,dtype in (("run_transform",np.float64),("run_flags",np.uint8)):
 if name in z.files:kw[name]=np.array(z[name],dtype,order="C")
raw=mf.Mesh64(**kw);compacted,report=compact_mesh_empty_runs(raw,True)
original_resume=mf.Manifold(raw);compacted_resume=mf.Manifold(compacted)
d0=oriented_surface_source_digest(original_resume.to_mesh64());d1=oriented_surface_source_digest(compacted_resume.to_mesh64())
assert d0==d1
data={k:z[k] for k in z.files};data["run_index"]=np.asarray(compacted.run_index);data["run_original_id"]=np.asarray(compacted.run_original_id)
for name in ("run_transform","run_flags"):
 if name in data:data[name]=np.asarray(getattr(compacted,name))
np.savez_compressed(out/"compacted-checkpoint.npz",**data)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={**report,"input_path":str(args.checkpoint),"input_sha256":sha(args.checkpoint),"output_sha256":sha(out/"compacted-checkpoint.npz"),
 "normal_resume_parity":True,"normal_resume_digest":d0,"normal_resume_faces":original_resume.num_tri(),
 "normal_resume_compacted_faces":compacted_resume.num_tri(),"helper_sha256":sha(Path(__file__).with_name("compact_csg_source_cache.py")),
 "elapsed_s":time.monotonic()-t,"physical_admission":False,"simulation_steps":0,
 "note":"Raw cache geometry/source identity is exact; existing Manifold recovery may normalize topology. This operation adds no normalization beyond ordinary recovery."}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt),flush=True)
