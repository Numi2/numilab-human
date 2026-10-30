from pathlib import Path
import numpy as np,json,hashlib,collections,time
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
O=H/'complete-surface-quality-026';O.mkdir(exist_ok=False)
paths={'O':H/'microedge-csr-canonical-011/O/candidate-O-csr-canonical.npz','I':H/'local-microedge-saved-operation-replay-011/I/candidate-I-saved-op-replay.npz','D':H/'current-D-proof-015/candidate-D-saved-op-replay-source-mapped.npz'}
out={'scope':'complete current Float32 source-surface triangle quality; no cell or physical-admission claim','physical_admission':False,'accepted_steps':0,'surfaces':{}}
for role,path in paths.items():
 with np.load(path,allow_pickle=False) as z:
  x=np.asarray(z['points_world_m'],np.float32).astype(np.float64);f=np.asarray(z['faces'],np.int64)
  fields={k:np.asarray(z[k]).copy() for k in ['face_source_original_id','face_source_body','face_source_bone'] if k in z.files}
 tri=x[f];e=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,1],tri[:,0]-tri[:,2]],axis=1)
 lengths=np.linalg.norm(e,axis=2);a2=np.linalg.norm(np.cross(e[:,0],-e[:,2]),axis=1)
 altitude=a2/lengths.max(axis=1)
 # Stable min interior angle: atan2 of common cross area and dot at each corner.
 dots=np.stack([-np.einsum('ij,ij->i',e[:,0],e[:,2]),-np.einsum('ij,ij->i',e[:,1],e[:,0]),-np.einsum('ij,ij->i',e[:,2],e[:,1])],axis=1)
 minangle=np.min(np.arctan2(a2[:,None],dots),axis=1)*180/np.pi
 thresholds=[1e-10,1e-9,1e-8,2e-8,5e-8,1e-7,1e-6,1e-5]
 subset=np.flatnonzero(altitude<1e-7)
 np.savez_compressed(O/(role+'-below100nm.npz'),face_rows=subset,altitude_m=altitude[subset],minimum_angle_degrees=minangle[subset],faces=f[subset],**{k:v[subset] for k,v in fields.items()})
 worst=np.argsort(altitude)[:50]
 r={'input':str(path),'input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'faces':len(f),'vertices':len(x),'zero_area':int(np.count_nonzero(a2==0)),'altitude_counts_below_m':{str(t):int(np.count_nonzero(altitude<t)) for t in thresholds},'angle_counts_below_degrees':{str(t):int(np.count_nonzero(minangle<t)) for t in [1e-5,1e-4,1e-3,1e-2,0.1,1.0]},'altitude_quantiles_m':dict(zip(['min','q001','q01','q50'],map(float,np.quantile(altitude,[0,.001,.01,.5])))),'source_ids_below100nm':dict(collections.Counter(map(int,fields.get('face_source_original_id',np.full(len(f),-1))[subset]))),'worst_faces':[{'row':int(i),'vertices':f[i].tolist(),'altitude_m':float(altitude[i]),'angle_degrees':float(minangle[i]),**{k:int(v[i]) for k,v in fields.items()}} for i in worst]}
 out['surfaces'][role]=r
 del tri,e,lengths,a2,altitude,minangle,x,f
(O/'report.json').write_text(json.dumps(out,indent=2)+chr(10))
print(json.dumps({'report':str(O/'report.json'),'sha256':hashlib.sha256((O/'report.json').read_bytes()).hexdigest(),'surfaces':{role:{k:v for k,v in r.items() if k in ('altitude_counts_below_m','angle_counts_below_degrees','zero_area','faces','vertices','source_ids_below100nm')} for role,r in out['surfaces'].items()}}))
