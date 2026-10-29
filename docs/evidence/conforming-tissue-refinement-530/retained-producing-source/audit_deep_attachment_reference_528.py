from pathlib import Path
import numpy as np,csv,json,hashlib,time,sys
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');R=P.parent
sys.path.insert(0,str(R/'authoring-deps'));import igl
out=P/'deep-attachment-reference-witnesses-528';out.mkdir(exist_ok=False)
source=P/'reduced-physical-boundary-400/attempt-002-parent-ancestry-walk/candidate.npz'
fine=P/'geometry-worker/matched-nesting-390/run-037-component21-f32-flips/candidate-D-component21-hull-f32-conditioned.npz'
banks=P/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz'
attachment=P/'native-physical-parent-mass-330/native-source-identity-control-003/parent-deep-attachments.csv'
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
with attachment.open() as f:rows=list(csv.DictReader(f))
with np.load(source) as z:x=z['points_reference_world_m_f64'].astype(np.float32).astype(float)
with np.load(fine) as z:v=z['points_world_m'].astype(np.float32).astype(float);faces=z['faces'].astype(np.int64);fb=z['face_source_body'];fs=z['face_source_id'];ff=z['face_source_face_id']
with np.load(banks) as z:b={k:z[k].copy() for k in z.files}
query=np.asarray([x[int(r['volume_node_row'])] for r in rows]);dist,face,closest=igl.point_mesh_squared_distance(query,v,faces)
witnesses=[]
for i,row in enumerate(rows):
 body=int(row['donor_body']);vi=int(row['volume_node_row']);mask=np.flatnonzero(fb==body)
 if not len(mask):raise ValueError('missing typed fine D source body')
 sd,sf,sc=igl.point_mesh_squared_distance(query[i:i+1],v,faces[mask]);target=np.asarray(sc[0]);frow=int(mask[int(sf[0])])
 ids=faces[frow];E=np.c_[v[ids[1]]-v[ids[0]],v[ids[2]]-v[ids[0]]];uv=np.linalg.lstsq(E,target-v[ids[0]],rcond=None)[0];bary=np.r_[1-uv.sum(),uv]
 scans=[]
 for bank in np.flatnonzero(b['bank_body']==body):
  lo,hi=map(int,b['bank_face_offsets'][bank:bank+2]);F=b['faces'][lo:hi].astype(np.int64);V=b['points_world_f32'].astype(float)
  # Only source/target points inside the bank AABB can be contained.
  bounds=np.vstack([V[np.unique(F)].min(0),V[np.unique(F)].max(0)])
  Q=np.vstack([query[i],target])
  inside_box=np.all(Q>=bounds[0],axis=1)&np.all(Q<=bounds[1],axis=1)
  if not inside_box.any():continue
  D,FI,C=igl.point_mesh_squared_distance(Q,V,F);W=np.asarray(igl.winding_number(V,F,Q)).reshape(-1)
  for j,name in enumerate(['original','fine_surface_proposal']):
   if inside_box[j] and abs(W[j])>.5:
    scans.append({'position':name,'body':body,'bone':int(b['bank_bone'][bank]),'distance_to_boundary_m':float(np.sqrt(D[j])),'absolute_winding':float(abs(W[j])),'bank_face_row':int(FI[j])})
 witnesses.append({'site_row':int(row['site_row']),'source_vertex_row':int(row['source_vertex_row']),'volume_node_row':vi,'donor_body':body,'frame_owner_body':int(row['frame_owner_body']),'original_reference_world_m':query[i].tolist(),'nearest_full_surface_distance_m':float(np.sqrt(dist[i])),'nearest_full_surface_body':int(fb[int(face[i])]),'same_body_fine_surface_distance_m':float(np.sqrt(sd[0])),'same_body_fine_surface_point_m':target.tolist(),'fine_face_row':frow,'fine_source_id':int(fs[frow]),'fine_source_face_id':int(ff[frow]),'fine_face_barycentric':bary.tolist(),'containment_witnesses':scans})
result={'status':'read_only_deep_attachment_reference_witnesses','source_candidate_sha256':sha(source),'fine_D_sha256':sha(fine),'source_attachment_packet_sha256':sha(attachment),'bank_packet_sha256':sha(banks),'bank_coordinate_contract':'native-source-order Float32 replay emulation, not fresh GPU capture','query_contract':'canonical Float32 source reference points, promoted to double for offline proximity/winding diagnostics','attachment_count':len(rows),'witnesses':witnesses,'physical_admission':False,'authored_attachment_changes':False,'scope':'nearest-surface points are repair proposals only; no mechanics or attachment identity/target change; exact solid/contact and connected-cell gates remain required'}
(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'attachments':len(rows),'original_contained_sites':[r['site_row'] for r in witnesses if any(w['position']=='original' for w in r['containment_witnesses'])],'proposal_contained_sites':[r['site_row'] for r in witnesses if any(w['position']=='fine_surface_proposal' for w in r['containment_witnesses'])],'max_same_body_projection_m':max(r['same_body_fine_surface_distance_m'] for r in witnesses),'receipt_sha256':sha(out/'receipt.json')}),flush=True)
