#!/usr/bin/env python3
"""Audit topology of pinned mechanical and detailed skin/fat boundaries."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(points,faces):
 p=np.asarray(points,float);f=np.asarray(faces,np.int64)
 if f.ndim!=2 or f.shape[1]!=3 or f.min()<0 or f.max()>=len(p):raise ValueError("invalid domain")
 directed=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
 edge,inv,cnt=np.unique(np.sort(directed,axis=1),axis=0,return_inverse=True,return_counts=True)
 signed=np.bincount(inv,weights=np.where(directed[:,0]<directed[:,1],1,-1),minlength=len(edge))
 used=np.unique(f)
 graph=coo_matrix((np.ones(2*len(edge)),(np.r_[edge[:,0],edge[:,1]],np.r_[edge[:,1],edge[:,0]])),shape=(len(p),len(p))).tocsr()
 _,labels=connected_components(graph,directed=False)
 comps,counts=np.unique(labels[used],return_counts=True)
 boundary=edge[cnt==1];brows=[]
 if len(boundary):
  bg=coo_matrix((np.ones(2*len(boundary)),(np.r_[boundary[:,0],boundary[:,1]],np.r_[boundary[:,1],boundary[:,0]])),shape=(len(p),len(p))).tocsr()
  _,bl=connected_components(bg,directed=False)
  for key in np.unique(bl[np.unique(boundary)]):
   rows=np.flatnonzero(bl==key);rows=rows[np.asarray(bg.getnnz(axis=1))[rows]>0]
   degree=np.asarray(bg.getnnz(axis=1))[rows]
   brows.append({"vertex_count":len(rows),"closed_loop_degree2":bool(np.all(degree==2)),"degree_min":int(degree.min()),"degree_max":int(degree.max()),"bbox_world_m":[p[rows].min(0).tolist(),p[rows].max(0).tolist()],"node_rows":rows.tolist()})
 return {"used_vertices":len(used),"faces":len(f),"edges":len(edge),"euler_characteristic":int(len(used)-len(edge)+len(f)),"connected_components":len(comps),"component_vertex_counts":sorted(counts.tolist(),reverse=True),"boundary_edges":len(boundary),"nonmanifold_edges":int(np.count_nonzero(cnt>2)),"equal_oriented_paired_edges":int(np.count_nonzero((cnt==2)&(signed!=0))),"boundary_components":brows}

def main():
 out=P/"layer-boundary-topology-467";out.mkdir(exist_ok=False)
 coarse=P/"reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz"
 deep=P/"geometry-worker/matched-nesting-390/run-038-run037-fat-occupancy/D-geometry-run037-f32.npz"
 with np.load(coarse) as z:
  pts=z["points_reference_world_m_f64"].astype(np.float32).astype(float)
  c={role:audit(pts,z["coarse_boundary_faces_"+role]) for role in ["O","I","D"]}
 with np.load(deep) as z:
  d=audit(z["points_world_m"],z["faces"])
  closed=audit(z["points_world_m"],np.concatenate([z["faces"],z["closed_cap_faces"]]))
 tetra=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
 face=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
 test=audit(tetra,face)
 assert test["euler_characteristic"]==2 and test["connected_components"]==1 and test["boundary_edges"]==0 and test["nonmanifold_edges"]==0 and test["equal_oriented_paired_edges"]==0
 opened=audit(tetra,face[:-1]);assert opened["euler_characteristic"]==1 and opened["boundary_edges"]==3 and len(opened["boundary_components"])==1 and opened["boundary_components"][0]["closed_loop_degree2"]
 same=audit(tetra,np.concatenate([face,face[[0]]]))
 assert same["nonmanifold_edges"]==3
 r={"status":"topology_inventory","coarse_candidate_sha256":sha(coarse),"D037_geometry_sha256":sha(deep),"script_sha256":sha(__file__),"coarse":c,"detailed_D_open":d,"detailed_D_with_explicit_head_caps":closed,"tests":"closed/open/duplicated tetrahedron surface passed","physical_admission":False,"scope":"identifies topological approximation mismatch and open boundaries separately; no geometry movement or anatomical interface classification"}
 (out/"receipt.json").write_text(json.dumps(r,indent=2)+"\n")
 print(json.dumps({k:{s:v[s] for s in ["used_vertices","faces","euler_characteristic","connected_components","boundary_edges","nonmanifold_edges","equal_oriented_paired_edges"]} for k,v in [("coarse_D",c["D"]),("detailed_D_open",d),("detailed_D_capped",closed)]}))
if __name__=="__main__":main()
