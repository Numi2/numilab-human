from pathlib import Path
import hashlib,json,pickle,os
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
B=Path('/Users/n/numi-human-fds132-m6-1350/attempt-156-right-candidate150-carrier-assets-001')
C=Path('/Users/n/numi-human-fds132-m6-1350/attempt-166-right-capB-ancestry-correction-001/result-001/candidate.pkl')
O=A/'right166-topology-bundle-001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,m):
 if not x:raise RuntimeError(m)
def save(p,j):p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
require(not O.exists(),'preserve output')
require(sha(C)=='25d759377d4861660f12c6458cd5bde4d48e152462048f15aea6410b70b2c17d','candidate changed')
prep=json.loads((B/'preparation.json').read_text())
oldpath=Path(prep['candidate']['candidate150_path']);require(sha(oldpath)==prep['candidate']['candidate150_sha256'],'base changed')
old=pickle.load(oldpath.open('rb'));new=pickle.load(C.open('rb'))
require(old['core_body_indices']==new['core_body_indices'],'core order changed')
require(old['global_vertices_exact_m']==new['global_vertices_exact_m'],'source positions changed')
pins=dict(prep['inputs_sha256'])
for p in (B/'preparation.json',C,oldpath,Path(__file__)):pins[str(p)]=sha(p)
require(all(sha(p)==v for p,v in pins.items()),'input changed')
ops=[]
for op in new['local_diagonal_conditioning_158']['operations']:
 ops.append({'sid':op['sid'],'faces':op['face_indices'],'old':op['old_faces'],'new':op['new_faces'],'ancestry':[op['parent_identity']],'detail':op})
groups={}
for op in new['local_diagonal_conditioning_159']['operations']:
 groups.setdefault(op['sid'],[]).append(op)
for sid,gg in groups.items():
 ops.append({'sid':sid,'faces':[x['face_index'] for x in gg],'old':[x['old_global_face'] for x in gg],'new':[x['new_global_face'] for x in gg],'ancestry':[[131,5230]],'detail':gg})
groups={}
for op in new['local_diagonal_conditioning_165']['operations']:
 groups.setdefault((op['stable_id'],op['operation']),[]).append(op)
for (sid,name),gg in groups.items():
 ancestors=sorted(set(tuple(x['stored_primary_parent']) for x in gg))
 ops.append({'sid':sid,'faces':[x['face_index'] for x in gg],'old':[x['old_global_vertex_ids'] for x in gg],'new':[x['new_global_vertex_ids'] for x in gg],'ancestry':[list(x) for x in ancestors],'detail':gg})
O.mkdir();(O/'assets').mkdir();rows={};lineage=[]
for sid in (131,133):
 a=old['surfaces'][sid];b=new['surfaces'][sid]
 for key in ('global_vertex_ids','vertices_exact_m','vertices_float32_m','core_weights_float64','parent_ids','parent_faces'):
  require(np.array_equal(np.asarray(a[key]),np.asarray(b[key])),'non-topological change '+key)
 p=Path(prep['geometry_rows'][str(sid)]['path']);require(sha(p)==prep['geometry_rows'][str(sid)]['sha256'],'base compact changed');pins[str(p)]=sha(p)
 with np.load(p,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 d['parent_input_local_face_indices']=d['parent_input_local_face_indices'].astype(np.int64)
 basef=np.asarray(a['faces_global'],dtype=np.int64);newf=np.asarray(b['faces_global'],dtype=np.int64)
 require(np.array_equal(d['faces_global'],basef),'base face order')
 replay=basef.copy();roles=np.asarray(a['roles']).copy()
 for op in ops:
  if op['sid']!=sid:continue
  fs=op['faces'];require(np.array_equal(replay[fs],op['old']),'operation predecessor mismatch')
  replay[fs]=op['new']
  lineage.append({'stable_id':sid,'derived_face_indices':fs,'predecessor_input_local_face_indices':d['parent_input_local_face_indices'][fs].tolist(),'pooled_source_parent_ancestry':op['ancestry'],'operation':op})
  d['parent_input_local_face_indices'][fs]=-1
 for op in new['capB_interface_extension_overlay']['role_overrides']:
  if op['stable_id']!=sid:continue
  require(roles[op['face_index']]==op['from_role'],'role predecessor')
  roles[op['face_index']]=op['to_role']
 require(np.array_equal(replay,newf),'undeclared topology change')
 require(np.array_equal(roles,np.asarray(b['roles'])),'undeclared role change')
 ix={int(g):i for i,g in enumerate(d['global_vertex_ids'])};require(set(newf.ravel())==set(ix),'referenced vertex inventory changed')
 d['faces_global']=newf;d['faces_local']=np.asarray([[ix[int(g)] for g in f] for f in newf],dtype=np.int64);d['roles']=roles.astype(d['roles'].dtype)
 q=O/'assets'/f'stable-{sid}-candidate166-source-geometry-field.npz';np.savez_compressed(q,**d)
 rows[str(sid)]={'path':str(q),'sha256':sha(q),'face_count':len(newf),'face_incident_vertex_count':len(ix),'changed_face_indices':np.flatnonzero(np.any(basef!=newf,axis=1)).tolist(),'normal_source':'Unchanged candidate150 source normals; inferred tessellation changes with bounds in ordered lineage.','direct_input_face_sentinel':-1,'derived_face_lineage_path':str(O/'derived-face-lineage.json')}
source=Path(prep['parent_terms']['native013_source_index1_path']);require(sha(source)==prep['parent_terms']['native013_source_index1_sha256'],'terms changed');pins[str(source)]=sha(source)
os.link(source,O/'assets/right-parent-surface-terms-native013-source-index-1.json')
save(O/'derived-face-lineage.json',{'candidate':str(C),'candidate_sha256':sha(C),'scope':'Pooled predecessor ancestry for inferred tessellation, not exact prior patch coverage. Ordered operations replay candidate150 exactly to candidate165.','operations':lineage,'interface_extension':new['capB_interface_extension_overlay']})
save(O/'preparation.json',{'schema':prep['schema'],'status':'prepared_native_pose_validation_pending','candidate_admission':False,'inputs_sha256':pins,'carrier_payload':prep['carrier_payload'],'geometry_rows':rows,'parent_terms':prep['parent_terms'],'candidate_path':str(C),'candidate_sha256':sha(C),'baseline_bundle_path':str(B),'baseline_bundle_sha256':sha(B/'preparation.json'),'derived_face_lineage_path':str(O/'derived-face-lineage.json'),'derived_face_lineage_sha256':sha(O/'derived-face-lineage.json'),'coordinates_weights_and_vertex_inventory_unchanged':True,'interface_extension':new['capB_interface_extension_overlay'],'known_failure':'Exact row133 self pair3323/7167 remains at actual7423/8000; not admitted.','gpu_launched':False})
require(all(sha(p)==v for p,v in pins.items()),'input changed during preparation')
print(json.dumps({'output':str(O),'preparation_sha256':sha(O/'preparation.json'),'changed_face_count':sum(len(x['changed_face_indices']) for x in rows.values())}))

