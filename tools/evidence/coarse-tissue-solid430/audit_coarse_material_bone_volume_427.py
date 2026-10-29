#!/usr/bin/env python3
"""Inspect actual canonical-Float32 tissue volume against pinned bone solids.
No displacement, clipping, toleranced admission, or mass redistribution is performed.
"""
import argparse,hashlib,json,time,sys
from pathlib import Path
import numpy as np
R=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207")
sys.path.insert(0,str(R/"authoring-deps"))
import manifold3d as mf

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def solid(v,f):
 return mf.Manifold(mf.Mesh64(vert_properties=np.ascontiguousarray(v,np.float64),tri_verts=np.ascontiguousarray(f,np.uint64)))

def exterior(t):
 # Positive oriented tetrahedron outward faces; retain original global node IDs.
 f=t[:,np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])].reshape(-1,3)
 keys,inv,cnt=np.unique(np.sort(f,axis=1),axis=0,return_inverse=True,return_counts=True)
 if np.any(cnt>2):raise ValueError("nonmanifold tetrahedral face incidence")
 return f[cnt[inv]==1]

def volumes(p,t):
 a,b,c,d=(p[t[:,j]] for j in range(4))
 return np.einsum("ij,ij->i",b-a,np.cross(c-a,d-a))/6

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--bank-manifest",required=True);ap.add_argument("--output",required=True);ap.add_argument("--banks",default="");ap.add_argument("--coordinates",choices=("current","reference"),default="reference")
 args=ap.parse_args();start=time.monotonic();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
 cand=Path(args.candidate);manifest=Path(args.bank_manifest)
 with np.load(cand,allow_pickle=False) as z:
  p=np.asarray(z["points_current_world_m_f32" if args.coordinates=="current" else "points_reference_world_m_f64"],np.float32).astype(np.float64)
  t=np.asarray(z["volume_tetrahedra"],np.int64);mat=np.asarray(z["material_index"],np.int64)
 if not np.isfinite(p).all() or t.min()<0 or t.max()>=len(p):raise ValueError("invalid candidate domain")
 vol=volumes(p,t)
 if np.any(vol<=0):raise ValueError("nonpositive canonical Float32 cells")
 layers={};surfaces={};metadata={}
 for m,name in ((15,"skin"),(16,"fat")):
  tt=t[mat==m];ff=exterior(tt);used=np.unique(ff);remap=np.full(len(p),-1,np.int64);remap[used]=np.arange(len(used));v=p[used];f=remap[ff]
  a=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
  if np.any(np.linalg.norm(a,axis=1)==0):raise ValueError("degenerate boundary face")
  s=solid(v,f);expected=float(vol[mat==m].sum());actual=float(s.volume())
  md={"material_index":m,"tetrahedra":len(tt),"boundary_vertices":len(v),"boundary_faces":len(f),"status":str(s.status()),"tet_volume_sum_m3":expected,"solid_volume_m3":actual,"volume_closure_m3":actual-expected}
  metadata[name]=md
  if s.status()!=mf.Error.NoError or abs(actual-expected)>max(1e-12,abs(expected)*1e-8):
   (out/"receipt.json").write_text(json.dumps({"status":"invalid_material_boundary","candidate_sha256":sha(cand),"layers":metadata},indent=2)+"\n")
   raise ValueError("material boundary invalid or volume closure failed: "+name)
  layers[name]=s;surfaces[name]=(v,f)
  np.savez_compressed(out/(name+"-material-boundary.npz"),points_world_m_f64=v,faces_f64=f,global_volume_node_rows=used)
 indices=set(map(int,args.banks.split(","))) if args.banks else None
 rows=json.loads(manifest.read_text())["operands"]
 if indices is not None:rows=[b for b in rows if b["index"] in indices]
 results=[]
 for b in rows:
  path=Path(b["operand_path"])
  if sha(path)!=b["operand_sha256"]:raise ValueError("bone operand hash changed")
  with np.load(path,allow_pickle=False) as z:v=z["points_world_m_f64"];f=z["faces_f64"].astype(np.int64)
  bank=solid(v,f)
  if bank.status()!=mf.Error.NoError:raise ValueError("invalid bone operand")
  row={"body":b["body"],"bone":b["bone"],"index":b["index"],"operand_sha256":b["operand_sha256"],"layers":{}}
  for name,s in layers.items():
   inter=s^bank
   if inter.status()!=mf.Error.NoError:raise ValueError("Boolean failure "+str(b["index"]))
   mesh=inter.to_mesh64();vv=np.asarray(mesh.vert_properties)[:,:3];ff=np.asarray(mesh.tri_verts,np.int64)
   info={"intersection_m3":float(inter.volume()),"vertices":len(vv),"faces":len(ff),"status":str(inter.status())}
   if len(vv):info["bbox_world_m"]=[vv.min(0).tolist(),vv.max(0).tolist()]
   row["layers"][name]=info
  results.append(row)
 receipt={"status":"coarse_material_bone_volume_inventory","coordinate_field":args.coordinates,"scope":"canonical Float32 physical material solids; positive intersection volumes are unresolved, no admission tolerance applied; per-bank volumes overlap and must not be summed as a union","candidate_sha256":sha(cand),"bank_manifest_sha256":sha(manifest),"script_sha256":sha(__file__),"point_count":len(p),"tet_count":len(t),"minimum_cell_volume_m3":float(vol.min()),"layers":metadata,"banks":results,"elapsed_s":time.monotonic()-start}
 (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
 print(json.dumps({"status":receipt["status"],"layers":metadata,"banks_checked":len(results),"positive_pairs":sum(r["layers"][n]["intersection_m3"]>0 for r in results for n in layers),"elapsed_s":receipt["elapsed_s"]}),flush=True)
if __name__=="__main__":main()
