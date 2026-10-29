#!/usr/bin/env python3
"""Extract measured coarse material/bone boundary witnesses for connected repair.
This is an offline inventory: it neither changes geometry nor admits contact.
"""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np
R=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207")
P=R/"physical-transfer-209"
sys.path.insert(0,str(R/"authoring-deps"))
import manifold3d as mf
spec=importlib.util.spec_from_file_location("coarse_gate427",P/"audit_coarse_material_bone_volume_427.py")
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def sha(p):return gate.sha(p)
def boundary(t, rows):
 f=t[:,np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])].reshape(-1,3)
 owners=np.repeat(rows,4)
 _,inv,cnt=np.unique(np.sort(f,axis=1),axis=0,return_inverse=True,return_counts=True)
 gate.exterior(t) # includes the paired-face orientation check
 keep=cnt[inv]==1
 return f[keep],owners[keep]
def tagged(p,f):
 used=np.unique(f);inv=np.full(len(p),-1,np.int64);inv[used]=np.arange(len(used))
 ident=mf.Manifold.reserve_ids(1)
 s=mf.Manifold(mf.Mesh64(np.ascontiguousarray(p[used]),np.ascontiguousarray(inv[f],np.uint64),
    run_index=np.array([0,len(f)*3],np.uint64),
    run_original_id=np.array([ident],np.uint32),face_id=np.arange(len(f),dtype=np.uint64)))
 if s.status()!=mf.Error.NoError:raise ValueError("invalid tagged material solid")
 return s,ident
def witness_faces(mesh,ident,nface):
 ids=np.asarray(mesh.face_id,np.int64);runs=np.asarray(mesh.run_index,np.int64)//3
 out=[]
 for j,original in enumerate(mesh.run_original_id):
  if int(original)==ident:out.extend(ids[runs[j]:runs[j+1]])
 out=np.unique(out)
 if len(out) and (out[0]<0 or out[-1]>=nface):raise ValueError("Boolean face ancestry out of domain")
 return out
def components(nodes,tets):
 # Connected components of the complete tetrahedral vertex stars, not disjoint per-bank edits.
 active=np.any(np.isin(tets,nodes),axis=1)
 rows=np.flatnonzero(active);parent={int(i):int(i) for i in rows};first={}
 def find(a):
  while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
  return a
 for i in rows:
  for n in tets[i]:
   n=int(n)
   if n in first:
    a,b=find(int(i)),find(first[n])
    if a!=b:parent[max(a,b)]=min(a,b)
   else:first[n]=int(i)
 groups={}
 for i in rows:groups.setdefault(find(int(i)),[]).append(int(i))
 return sorted(groups.values(),key=lambda x:x[0])
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--bank-manifest",required=True)
 ap.add_argument("--out",required=True);ap.add_argument("--coordinates",choices=["current","reference"],default="reference")
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
 z=np.load(a.candidate,allow_pickle=False)
 p=np.asarray(z["points_current_world_m_f32" if a.coordinates=="current" else "points_reference_world_m_f64"],np.float32).astype(np.float64)
 t=np.asarray(z["volume_tetrahedra"],np.int64);mat=np.asarray(z["material_index"],np.int64)
 if not np.isfinite(p).all() or t.min()<0 or t.max()>=len(p) or np.any(gate.volumes(p,t)<=0):raise ValueError("invalid Float32 physical cells")
 manifests=json.loads(Path(a.bank_manifest).read_text())["operands"];layers={};rows=[];allnodes=set();allcells=set();unresolved=[]
 for m,name in ((15,"skin"),(16,"fat")):
  rr=np.flatnonzero(mat==m);f,owners=boundary(t[rr],rr);s,ident=tagged(p,f)
  layers[name]=(f,owners,s,ident)
  np.savez_compressed(out/(name+"-boundary-ancestry.npz"),volume_node_rows=f,volume_cell_rows=owners)
 for b in manifests:
  path=Path(b["operand_path"])
  if sha(path)!=b["operand_sha256"]:raise ValueError("bone operand changed")
  bankz=np.load(path,allow_pickle=False);bv=np.asarray(bankz["points_world_m_f64"],np.float64);bf=np.asarray(bankz["faces_f64"],np.int64)
  bank=gate.solid(bv,bf)
  if bank.status()!=mf.Error.NoError:raise ValueError("invalid bank")
  for name,(f,owners,s,ident) in layers.items():
   inter=s^bank
   if inter.status()!=mf.Error.NoError:raise ValueError("failed Boolean")
   volume=float(inter.volume())
   if volume<=0:continue
   mesh=inter.to_mesh64();faces=witness_faces(mesh,ident,len(f))
   nodes=np.unique(f[faces]) if len(faces) else np.empty(0,np.int64)
   cells=np.unique(owners[faces]) if len(faces) else np.empty(0,np.int64)
   item={"bank_index":int(b["index"]),"body":int(b["body"]),"bone":int(b["bone"]),"layer":name,
     "intersection_volume_m3":volume,"boundary_face_rows":faces.tolist(),"boundary_cell_rows":cells.tolist(),
     "boundary_node_rows":nodes.tolist(),"face_ancestry_complete":bool(len(faces))}
   if not len(faces):unresolved.append({"bank_index":int(b["index"]),"layer":name,"reason":"occupied solid has no surviving material-boundary witness; containment requires separate cell localization"})
   rows.append(item);allnodes.update(map(int,nodes));allcells.update(map(int,cells))
 groups=components(np.array(sorted(allnodes),np.int64),t)
 report={"status":"measured_connected_cavity_seed_inventory","physical_admission":False,"geometry_modified":False,
  "candidate_sha256":sha(a.candidate),"bank_manifest_sha256":sha(a.bank_manifest),"script_sha256":sha(__file__),
  "coordinates":a.coordinates,"float32_emission":True,"banks_checked":len(manifests),"positive_material_bank_pairs":len(rows),
  "boundary_witness_cells":len(allcells),"boundary_witness_nodes":len(allnodes),"connected_cavity_count":len(groups),
  "cavities":[{"index":i,"volume_cell_rows":g,"volume_node_count":len(np.unique(t[g]))} for i,g in enumerate(groups)],
  "unlocalized_containments":unresolved,"bank_witnesses":rows,"elapsed_s":time.monotonic()-start,
  "limits":"Positive volumes have no admission tolerance. Boundary provenance identifies repair seeds only; no face/volume/physical validity is inferred. Overlapping stars are merged, so independent patches cannot silently share volume cells."}
 (out/"receipt.json").write_text(json.dumps(report,indent=2)+"\n")
 print(json.dumps({k:report[k] for k in ["status","positive_material_bank_pairs","boundary_witness_cells","boundary_witness_nodes","connected_cavity_count","unlocalized_containments","elapsed_s"]}),flush=True)
if __name__=="__main__":main()
