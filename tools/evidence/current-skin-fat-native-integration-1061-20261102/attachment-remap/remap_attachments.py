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
maximum_move_m=0.005; minimum_spacing_m=0.025; min_alt_m=0.00005; min_confidence=0.85
selected={};audit=[]
for body in sorted({int(r['donor_body']) for r in requests}):
 group=[(k,r) for k,r in enumerate(requests) if int(r['donor_body'])==body]; assert len(group)<=3; lists=[]; oldpoints=[]
 for k,r in group:
  v=old[int(r['stable_source_vertex'])];assert int(v['donor'])==body and v['source_domain']==r['source_domain'];x=np.array([float(v[c]) for c in ['wx','wy','wz']]);oldpoints.append(x);distance=np.linalg.norm(xyz-x,axis=1);valid=(donor==body)&(confidence>=min_confidence)&(altitude[nodes]>=min_alt_m)&(distance<=maximum_move_m);cand=np.flatnonzero(valid);cand=cand[np.lexsort((ids[cand],distance[cand]))];assert len(cand)>0,(r,'no eligible local site');lists.append([(int(j),float(distance[j])) for j in cand[:64]])
 best=None
 for combo in itertools.product(*lists):
  if any(np.linalg.norm(xyz[combo[i][0]]-xyz[combo[j][0]])<minimum_spacing_m for i in range(len(combo)) for j in range(i)):continue
  key=(sum(d*d for j,d in combo),tuple(int(ids[j]) for j,d in combo))
  if best is None or key<best[0]:best=(key,combo)
 assert best is not None,('cannot retain spacing within declared displacement',body)
 for (k,r),(j,d),oldpoint in zip(group,best[1],oldpoints):
  selected[k]={'stable_attachment_id':r['stable_attachment_id'],'donor_body':body,'stable_source_vertex':int(ids[j]),'source_domain':new[j]['source_domain']}
  others=[xyz[a[0]] for a in best[1] if a[0]!=j];audit.append({'stable_attachment_id':int(r['stable_attachment_id']),'donor_body':body,'old_source_vertex':int(r['stable_source_vertex']),'new_source_vertex':int(ids[j]),'new_compact_surface_row':j,'volume_node_row':int(nodes[j]),'old_world_m':oldpoint.tolist(),'current_world_m':xyz[j].tolist(),'displacement_m':d,'donor_weight':float(confidence[j]),'minimum_incident_altitude_m':float(altitude[nodes[j]]),'minimum_same_donor_spacing_m':min([float(np.linalg.norm(xyz[j]-v)) for v in others],default=None)})
path=out/'deep-attachment-requests.csv'
with path.open('w') as f:
 w=csv.DictWriter(f,fieldnames=['stable_attachment_id','donor_body','stable_source_vertex','source_domain']);w.writeheader();w.writerows(selected[i] for i in range(14))
receipt={'status':'PASS_REFERENCE_ATTACHMENT_REMAP_ONLY','physical_admission':False,'native_steps':0,'method':'deterministic minimum-squared-displacement donor-preserving current D selection; all-pairs spacing checked jointly per donor; no hard constraints added','limits':{'maximum_displacement_m':maximum_move_m,'minimum_spacing_m':minimum_spacing_m,'minimum_donor_weight':min_confidence,'minimum_incident_altitude_m':min_alt_m,'maximum_attachments_per_donor':3},'inputs':{str(p):sha(p) for p in [oldrun/'launch.json',oldcsv,reqpath,src,newcsv,vol,export/'registered/donor-frames.csv']},'output':{'path':str(path),'sha256':sha(path)},'rows':sorted(audit,key=lambda a:a['stable_attachment_id']),'geometry_changed':False,'mass_changed':False,'stable_attachment_ids_preserved':True}
(out/'remap-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
