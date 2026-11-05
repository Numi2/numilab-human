from pathlib import Path
import numpy as np,json,hashlib
W=Path('/Users/numi/Work/human-integrated-gaps-20261104')
D=W/'contact-stall-360';D.mkdir(exist_ok=False)
volume=W/'witness-cluster-rebuild-356/proposal/compiler/rich433/candidate-current-typed-volume.npz'
ids=np.array([61202,61204,61325,61213],np.int64)-1666
out={'native_object':15,'global_node_offset':1666,'native_nodes':(ids+1666).tolist(),'rich_nodes':ids.tolist()}
with np.load(volume,allow_pickle=False) as z:
 points=z['points_reference_world_m_f64'];t=z['volume_tetrahedra']
 out['reference_positions_world_m']=points[ids].tolist()
 out['exact_tetrahedron_rows']=np.flatnonzero(np.isin(t,ids).sum(1)==4).tolist()
 node_neighbors=np.unique(t[np.any(t==ids[0],axis=1)])
 out['opposite_triangle_node_edge_neighbors_of_vertex']=[bool(n in node_neighbors) for n in ids[1:]]
 out['roles']={}
 for role in 'OID':
  m=z['current_typed_role_source_row_to_joint_node_'+role]
  out['roles'][role]=[np.flatnonzero(m==i).tolist() for i in ids]
 out['incident_tet_counts']=[int(np.any(t==i,axis=1).sum()) for i in ids]
 # Small connected volume star around the witness.
 rows=np.flatnonzero(np.any(np.isin(t,ids),axis=1))
 out['star_tet_rows']=rows.tolist()
 out['materials']=np.unique(z['material_index'][rows]).tolist()
 out['anatomical_regions']=np.unique(z['anatomical_region_id'][rows]).tolist()
out['volume_asset']={'path':str(volume),'sha256':hashlib.file_digest(volume.open('rb'),'sha256').hexdigest()}
for role in 'OID':
 path=W/f'witness-cluster-rebuild-356/proposal/candidate-{role}-typed-current.npz'
 with np.load(path,allow_pickle=False) as z:
  m=out['roles'][role]
  role_ids=[int(a[0]) if len(a)==1 else -1 for a in m]
  if min(role_ids)<0:continue
  f=z['faces'];v=role_ids[0];tri=np.array(role_ids[1:])
  rows=np.flatnonzero(np.all(np.isin(f,tri),axis=1))
  star=np.flatnonzero(np.any(f==v,axis=1))
  adj=np.unique(f[star])
  out['role_surface']={'role':role,'source_nodes':role_ids,'triangle_source_face_rows':rows.tolist(),'vertex_star_faces':star.tolist(),'triangle_vertices_edge_neighbors':[bool(x in adj) for x in tri],'vertex_star_triangle_nodes':f[star].tolist()}
  out['triangle_ancestry']={k:z[k][rows].tolist() for k in ['face_source_layer','face_source_body','face_source_bone','face_source_face_id']}
(D/'witness-source.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
