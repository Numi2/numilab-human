#!/usr/bin/env python3
"""Actual Boolean material-boundary contact witnesses for offline mesh construction."""
from pathlib import Path
import argparse,importlib.util,json,time
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
def load(name,file):
 s=importlib.util.spec_from_file_location(name,P/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
w=load("witness448","inventory_connected_cavity_witnesses_448.py")
import igl
def barycentric(points,triangles):
 a=triangles[:,0];u=triangles[:,1]-a;v=triangles[:,2]-a;d=points-a
 uu=np.einsum("ij,ij->i",u,u);uv=np.einsum("ij,ij->i",u,v);vv=np.einsum("ij,ij->i",v,v)
 du=np.einsum("ij,ij->i",d,u);dv=np.einsum("ij,ij->i",d,v);det=uu*vv-uv*uv
 if np.any(det<=0):raise ValueError("degenerate source face")
 b=(du*vv-dv*uv)/det;c=(dv*uu-du*uv)/det
 return np.column_stack([1-b-c,b,c])

def extract(candidate,manifest):
 with np.load(candidate) as z:p=z["points_reference_world_m_f64"].astype(np.float32).astype(float);t=z["volume_tetrahedra"];mat=z["material_index"]
 if np.any(w.gate.volumes(p,t)<=0):raise ValueError("nonpositive physical cells")
 layers=[]
 for m in [15,16]:
  rr=np.flatnonzero(mat==m);f,owners=w.boundary(t[rr],rr);solid,ident=w.tagged(p,f);layers.append((m,f,owners,solid,ident))
 banks=json.loads(Path(manifest).read_text())["operands"]
 records=[];details=[];unresolved=[];maxerr=0.;positive_pairs=0
 for b in banks:
  path=Path(b["operand_path"])
  if w.sha(path)!=b["operand_sha256"]:raise ValueError("bank changed")
  with np.load(path) as z:bv=z["points_world_m_f64"];bf=z["faces_f64"].astype(np.int64)
  bank=w.gate.solid(bv,bf)
  for material,f,owners,solid,ident in layers:
   intersection=solid^bank;volume=float(intersection.volume())
   if volume<=0:continue
   positive_pairs+=1
   mesh=intersection.to_mesh64();mp=np.asarray(mesh.vert_properties)[:,:3];mf=np.asarray(mesh.tri_verts,np.int64);faceids=np.asarray(mesh.face_id,np.int64);runs=np.asarray(mesh.run_index,np.int64)//3
   rows=np.concatenate([np.arange(runs[j],runs[j+1]) for j,origin in enumerate(mesh.run_original_id) if int(origin)==ident]) if ident in mesh.run_original_id else np.array([],np.int64)
   if not len(rows):unresolved.append([int(b["index"]),material,volume]);continue
   q=mp[mf[rows]].mean(axis=1);ids=faceids[rows]
   distance,_,closest,_=igl.signed_distance(np.ascontiguousarray(q),np.ascontiguousarray(bv),np.ascontiguousarray(bf),igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
   selected=[]
   for face in np.unique(ids):
    ix=np.flatnonzero(ids==face);k=ix[np.argmin(distance[ix])]
    if distance[k]<0:selected.append(k)
   if not selected:unresolved.append([int(b["index"]),material,volume]);continue
   selected=np.asarray(selected,np.int64);q=q[selected];ids=ids[selected];distance=distance[selected];closest=closest[selected]
   bary=barycentric(q,p[f[ids]]);err=np.linalg.norm(np.einsum("ij,ijk->ik",bary,p[f[ids]])-q,axis=1)
   maxerr=max(maxerr,float(err.max()))
   if np.any(err>1e-9) or np.any(bary < -1e-7) or np.any(bary>1+1e-7):raise ValueError("Boolean ancestry barycentric reconstruction failed")
   direction=closest-q;norm=np.linalg.norm(direction,axis=1);direction/=norm[:,None]
   for k in range(len(q)):
    records.append((f[ids[k]],bary[k],direction[k],float(-distance[k]+2e-6),material,int(b["index"]),int(owners[ids[k]]),q[k],float(distance[k])))
   details.append({"material":material,"bank":int(b["index"]),"body":int(b["body"]),"bone":int(b["bone"]),"intersection_m3":volume,"constraints":len(q),"minimum_signed_distance_m":float(distance.min())})
 arrays={"volume_node_rows":np.asarray([r[0] for r in records],np.int64).reshape(-1,3),"barycentric":np.asarray([r[1] for r in records],float).reshape(-1,3),"outward_bank_normal":np.asarray([r[2] for r in records],float).reshape(-1,3),"required_normal_correction_m":np.array([r[3] for r in records]),"material_index":np.array([r[4] for r in records]),"bank_index":np.array([r[5] for r in records]),"source_cell_row":np.array([r[6] for r in records]),"point_world_m":np.asarray([r[7] for r in records],float).reshape(-1,3),"signed_distance_m":np.array([r[8] for r in records])}
 return arrays,{"constraints":len(records),"positive_pairs":positive_pairs,"localized_positive_pairs":len(details),"unlocalized_containments_or_contact_rows":unresolved,"max_barycentric_reconstruction_error_m":maxerr,"banks":details}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--candidate",required=True);ap.add_argument("--out",required=True);args=ap.parse_args()
 out=Path(args.out);out.mkdir(exist_ok=False,parents=True);start=time.monotonic()
 manifest=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json"
 arrays,r=extract(args.candidate,manifest);np.savez_compressed(out/"contact-constraints.npz",**arrays)
 r.update(status="measured_Boolean_boundary_contact_construction_constraints",candidate_sha256=w.sha(args.candidate),bank_manifest_sha256=w.sha(manifest),script_sha256=w.sha(__file__),elapsed_s=time.monotonic()-start,physical_admission=False,clearance_construction_target_m=2e-6,scope="recomputed measured material-boundary intersections; normal halfspaces are local construction proposals, not a global collision certificate")
 (out/"receipt.json").write_text(json.dumps(r,indent=2)+"\n");print(json.dumps({k:v for k,v in r.items() if k!="banks"}))
if __name__=="__main__":main()
