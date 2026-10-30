from pathlib import Path
import numpy as np, csv,json,time,hashlib,igl
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
D=P/'canonical-attachment-fat-containment-694'
packet=P/'protected-149-current-surface-crosswalk-001/run-009-correct-witness-operand-packet/protected-149-targets.csv'
with packet.open(newline="") as f:rows=[r for r in csv.DictReader(f) if r['stable_attachment_id']]
assert len(rows)==14
Q=np.asarray([[float(r['target_world_'+a+'_m']) for a in ('x','y','z')] for r in rows],dtype=np.float64)
inputs=[
 ('baseline635',P/'paired-fine-headwall-rim-635/run-010/output/fine-fat-closed-boundary.npz','points_world_m_f64','faces'),
 ('head014',P/'paired-head-reference-fit-684/run-019-witness-taper-exact-audit/candidate-closed-surfaces-reconstructed.npz','fat_points_world_m_f64','fat_faces')
]
results=[]
for name,path,pkey,fkey in inputs:
 data=np.load(path,allow_pickle=False)
 if pkey not in data:
  raise RuntimeError((str(path),data.files))
 points=np.asarray(data[pkey],dtype=np.float64);faces=np.asarray(data[fkey],dtype=np.int64)
 t0=time.monotonic();dist,face,closest=igl.point_mesh_squared_distance(Q,points,faces);W=igl.fast_winding_number(points,faces,Q)
 out={'name':name,'input':str(path),'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'seconds':time.monotonic()-t0,'diagnostic_method':'floating nearest triangle distance and approximate fast winding number; exact support/material-cell admission remains mandatory','rows':[]}
 for k,r in enumerate(rows):
  out['rows'].append({'stable_attachment_id':int(r['stable_attachment_id']),'mapping_kind':r['mapping_kind'],'target_world_m':Q[k].tolist(),'nearest_distance_m':float(np.sqrt(dist[k])),'nearest_closed_fat_face_row':int(face[k]),'nearest_point_m':closest[k].tolist(),'approximate_winding':float(W[k]),'physical_admission':False})
 results.append(out)
 (D/'report.json').write_text(json.dumps({'inputs':results,'runtime':0,'accepted_steps':0,'physical_admission':False},indent=2)+'\n')
 print(json.dumps(out),flush=True)
