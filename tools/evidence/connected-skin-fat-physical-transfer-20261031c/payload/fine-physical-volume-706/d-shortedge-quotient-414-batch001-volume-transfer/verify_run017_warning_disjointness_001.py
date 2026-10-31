import hashlib,json,pathlib,numpy as np
H=pathlib.Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
W=H/'fat-comp0-connected-warning-repair-412/run-017-current-comp0-remesh/output/warning-witnesses.json'
A=H/'d-shortedge-quotient-414-next-001-preflight-attempt004/allowed-D-edge-map.npz'
C=H/'fat-comp0-connected-warning-repair-412/run-014-current-comp0-edge-star-cavity/comp0-full-edge-star-exclusion.npz'
def sha(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
for p,h in ((W,'2fb6806c4ec2686dc7e077b139992ea94462d971c35cc9208582fd04ae6c3c69'),(A,'439a1df81c608ebb153992d884cf8356e507cd228e20346e97a2ef46cce0aa95'),(C,'bfb88500119d48ff09c3af5358a405302ea29f152252d7b8dccbd6b4275ac893')):
 if sha(p)!=h:raise ValueError(f'pin mismatch {p}')
w=json.loads(W.read_text());a=np.load(A,allow_pickle=False);c=np.load(C,allow_pickle=False)
endpoint=set(map(int,a['D_allowed_orientation_edge_rows'].ravel()));protected=set(map(int,c['comp0_full_protected_point_rows']))
rec=[];fail=[]
for cl in w['warning_clusters']:
 dpoints=set();vpoints=set();faces=[]
 for seg in cl['segments']:
  vpoints.update(map(int,seg['volume_point_rows']))
  for f in seg['surface_faces']:
   faces.append({'source_group':f.get('source_group'),'source_face_row':f.get('source_face_row'),'target_face_row':f.get('target_face_row'),'D_point_rows':f.get('D_point_rows_in_volume_order'),'volume_face_point_rows':f.get('volume_face_point_rows')})
   dpoints.update(map(int,f.get('D_point_rows_in_volume_order') or []))
 direct=sorted(dpoints&endpoint);unprotected=sorted(vpoints-protected)
 if direct or unprotected:fail.append({'warning_group':cl['warning_group'],'D_endpoint_overlap':direct,'unprotected_volume_points':unprotected})
 rec.append({'warning_group':cl['warning_group'],'classification':cl['classification'],'D_point_rows':sorted(dpoints),'volume_points':sorted(vpoints),'D_rows_overlapping_preflight_endpoints':direct,'volume_points_in_comp0_full_protection':len(vpoints)-len(unprotected),'volume_point_count':len(vpoints),'faces':faces})
out={'schema':'h706.run017-warning-vs-D414-preflight-disjointness.v1','status':'PASS_WARNING_PATCH_DISJOINT_FROM_D414_PREFLIGHT' if not fail else 'FAIL_OVERLAP','physical_admission':False,'native_steps':0,'warning_witness_path':str(W),'warning_witness_sha256':sha(W),'preflight_edge_map_path':str(A),'preflight_edge_map_sha256':sha(A),'comp0_mask_path':str(C),'comp0_mask_sha256':sha(C),'authorized_orientation_count':len(a['D_allowed_orientation_edge_rows']),'unique_authorized_D_endpoint_count':len(endpoint),'protected_comp0_point_count':len(protected),'warnings':rec,'failures':fail,'meaning':'No selected D contraction endpoint intersects a D point on any warned target face, and every warned volume-side endpoint/face point belongs to the authenticated comp0 full-point protection mask. Full post-transfer exact context remains required.'}
O=H/'d-shortedge-quotient-414-batch001-volume-transfer/run017-warning-disjointness.json';O.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':out['status'],'path':str(O),'sha256':sha(O),'warnings':len(rec),'failures':len(fail)},indent=2))
