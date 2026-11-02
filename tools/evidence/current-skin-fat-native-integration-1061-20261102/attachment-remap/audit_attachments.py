from pathlib import Path
import csv,json,hashlib,numpy as np,itertools,time
S=Path(__file__).resolve().parent.parent; H=S.parent; P=H.parent; out=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
oldrun=P/'native-registered-volume-parent-identity-physical-control-365'; base=json.loads((oldrun/'launch.json').read_text()); env=base['environment']; oldcsv=Path(env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_D_NODES_CSV']); old=list(csv.DictReader(oldcsv.open()))
assert sha(oldcsv)==env['NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_D_NODES_SHA256']
reqpath=oldrun/'inputs/deep-attachment-requests.csv'; requests=list(csv.DictReader(reqpath.open())); assert len(requests)==14
export=H/'current-native-1024-root/current-433-1044/volume-export'; newcsv=export/'registered/registered-D.csv'; new=list(csv.DictReader(newcsv.open())); src=export/'roles/D-source.json'; doc=json.loads(src.read_text()); nodes=np.array([r['child_volume_node_row'] for r in doc['vertices']]); ids=np.array([int(r['source_vertex_row']) for r in new]); donor=np.array([int(r['donor']) for r in new]); xyz=np.array([[float(r[k]) for k in ['wx','wy','wz']] for r in new]); assert np.array_equal(ids,np.array([r['source_vertex_row'] for r in doc['vertices']]))
vol=H/'current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz'; assert sha(vol)=='5d31c141b66628801a711c2e745f744f745611882a269e173eddfcd0e99d3372'
z=np.load(vol,allow_pickle=False); points=z['points_reference_world_m_f64'].astype(np.float32).astype(np.float64); tets=z['volume_tetrahedra']; assert np.array_equal(xyz,points[nodes]); weights=z['full86_weights']; lanes={int(r['donor']):int(r['binding']) for r in csv.DictReader((export/'registered/donor-frames.csv').open())}; confidence=np.array([weights[n,lane] for n,lane in zip(nodes,[lanes[int(d)] for d in donor])]); altitude=np.full(len(points),np.inf)
for lo in range(0,len(tets),100000):
 t=tets[lo:lo+100000];p=points[t]; six=np.abs(np.einsum('ij,ij->i',p[:,1]-p[:,0],np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0])))
 for j in range(4):
  f=[k for k in range(4) if k!=j];area=np.linalg.norm(np.cross(p[:,f[1]]-p[:,f[0]],p[:,f[2]]-p[:,f[0]]),axis=1);assert np.all(area>0);np.minimum.at(altitude,t[:,j],six/area)

np.savez_compressed(out/'prepared-quality.npz',altitude=altitude,confidence=confidence,nodes=nodes)
report=[]
for r in requests:
 body=int(r['donor_body']);v=old[int(r['stable_source_vertex'])];x=np.array([float(v[k]) for k in ['wx','wy','wz']]);distance=np.linalg.norm(xyz-x,axis=1)
 masks={'same_donor':donor==body,'donor_weight':(donor==body)&(confidence>=0.85),'all_native_quality':(donor==body)&(confidence>=0.85)&(altitude[nodes]>=0.00005)}
 row={'request':r}
 for k,m in masks.items():
  idx=np.flatnonzero(m);j=idx[np.argmin(distance[idx])];row[k]={'source_vertex':int(ids[j]),'compact_row':int(j),'distance_m':float(distance[j]),'confidence':float(confidence[j]),'altitude_m':float(altitude[nodes[j]])}
 report.append(row)
(out/'nearest-eligible-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
