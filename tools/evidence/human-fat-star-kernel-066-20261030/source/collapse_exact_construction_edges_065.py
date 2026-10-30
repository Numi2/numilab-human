#!/usr/bin/env python3
"""Exact, zero-displacement contraction of duplicate construction-node edges.
No anatomical source node removed, no nonzero-volume cell discarded.
"""
from pathlib import Path
import numpy as np,json,itertools,hashlib,time
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
S=H/'fat009-batch-cavity-064/fat-partial-proposal.npz';D=H/'fat009-construction-node-quotient-065';D.mkdir(exist_ok=False)
with np.load(S)as z:a={k:z[k]for k in z.files}
p=a['points_world_m'];t=a['tetrahedra'].copy();materials=a['material_indices'];original_count=245996;pat=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
_,inv,count=np.unique(p.view(np.uint32),axis=0,return_inverse=True,return_counts=True);groups=[np.flatnonzero(inv==i).tolist()for i in np.flatnonzero(count>1)]
def simplex_link(cells,vertices):
 vertices=set(vertices);link=set()
 for cell in cells:
  if not vertices<=set(map(int,cell)):continue
  rest=sorted(set(map(int,cell))-vertices)
  for k in range(len(rest)+1):
   link.update(tuple(x)for x in itertools.combinations(rest,k))
 return link
steps=[];mapping=np.arange(len(p));removed_rows=[]
initial=t.copy();keep=np.ones(len(t),bool)
for group in groups:
 originals=[x for x in group if x<original_count];assert len(originals)<=1
 u=originals[0]if originals else min(group)
 for v in group:
  if v==u:continue
  assert v>=original_count and np.array_equal(p[u].view(np.uint32),p[v].view(np.uint32))
  cells=t[keep];touch=np.any((cells==u)|(cells==v),axis=1);local=cells[touch]
  lu=simplex_link(local,[u]);lv=simplex_link(local,[v]);le=simplex_link(local,[u,v]);link_ok=(lu&lv)==le
  # Explicit full per-step link evidence retained even if it fails; no admission.
  t[t==v]=u;mapping[mapping==v]=u
  repeated=np.any(np.diff(np.sort(t,axis=1),axis=1)==0,axis=1)&keep
  rows=np.flatnonzero(repeated)
  q=p[initial[rows]]
  assert all(len({tuple(x.view(np.uint32))for x in row})<4 for row in q)
  keep[rows]=False;removed_rows.extend(rows.tolist())
  steps.append({'retained_node_row':int(u),'contracted_construction_node_row':int(v),'exact_F32_bit_identity':True,'simplicial_link_condition':link_ok,'link_intersection_size':len(lu&lv),'edge_link_size':len(le),'removed_repeated_node_cell_rows':rows.tolist(),'removed_materials':materials[rows].tolist(),'removed_cells_all_exactly_zero_by_repeated_positions':True})
assert np.all(mapping[:original_count]==np.arange(original_count))
t=t[keep];mat=materials[keep]
used=np.unique(t);unused_source=np.setdiff1d(np.arange(original_count),used);assert len(unused_source)==0
# All original rows stay in order; compact only unused construction rows.
rows=np.r_[np.arange(original_count),used[used>=original_count]];compact=np.full(len(p),-1,np.int64);compact[rows]=np.arange(len(rows));pp=p[rows];tt=compact[t]
q=pp[tt].astype(np.float64);d=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),q[:,3]-q[:,0])
keys=np.sort(tt,axis=1);unique_cells=len(np.unique(keys,axis=0))
faces=tt[:,pat].reshape(-1,3);sortfaces=np.sort(faces,axis=1);_,first,counts=np.unique(sortfaces,axis=0,return_index=True,return_counts=True);b=faces[first[counts==1]]
edges=np.sort(np.concatenate([b[:,[0,1]],b[:,[1,2]],b[:,[2,0]]]),axis=1);_,ec=np.unique(edges,axis=0,return_counts=True)
r={'status':'EXACT_CONSTRUCTION_EDGE_COLLAPSE_PARTIAL_NOT_PHYSICAL_ADMISSION','physical_admission':False,'runtime':0,'accepted_steps':0,'source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'source_cells':len(initial),'candidate_cells':len(tt),'removed_cells':len(removed_rows),'source_nodes':len(p),'candidate_nodes':len(pp),'all_original_source_nodes_and_F32_bits_preserved':bool(np.array_equal(pp[:original_count],p[:original_count])),'no_nonzero_cell_removed':True,'exact_total_volume_preservation_basis':'Every retained tetrahedron has identical F32 corner coordinates; every removed simplex has repeated exact F32 corner positions, so its exact determinant is zero.','steps':steps,'all_link_conditions_pass':all(x['simplicial_link_condition']for x in steps),'F32_zero_cells':int(np.sum(d==0)),'F32_negative_cells':int(np.sum(d<0)),'duplicate_cells':len(tt)-unique_cells,'nonmanifold_tet_faces':int(np.sum(counts>2)),'boundary_edges_not_two_incident':int(np.sum(ec!=2)),'boundary_faces':len(b),'original_PLC_facets':'PENDING_CONNECTED_RESTORATION'}
np.savez_compressed(D/'fat-quotient-proposal.npz',points_world_m=pp,tetrahedra=tt,material_indices=mat,source_candidate_node_to_quotient_node=compact[mapping],retained_source_candidate_cell_rows=np.flatnonzero(keep),removed_exact_zero_source_candidate_cell_rows=np.asarray(removed_rows,np.int64),boundary_faces=b)
(D/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items()if k!='steps'}))
