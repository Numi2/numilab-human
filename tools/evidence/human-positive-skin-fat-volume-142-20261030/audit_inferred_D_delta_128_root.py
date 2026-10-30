from pathlib import Path
import json,hashlib,numpy as np,sys,importlib.util,time
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706');ROOT=H.parent;out=H/'inferred-D-boundary-exact-delta-128-root';out.mkdir(exist_ok=False);started=time.monotonic()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in z.files}
def imp(name,p):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
baseline=H/'quality-warning-connected-union-038/merge-run004-run005-O/full-check/independent-verification-001/report.json';assert sha(baseline)=='af18d3e4d3f02284acc4a27740c5c9256464c17fb99ca443a7186716a836cdf7'
v=imp('verify034',H/'quality-proposal-verification-034/verify_quality_proposal_034.py')
assert sha(v.OWNER)==v.OWNER_SHA and sha(v.BONE)==v.BONE_SHA
sys.path.insert(0,str(v.OWNER.parent.parent));import numilab_human.cardiac_cavity_intersections as pred
bone=imp('delta128bone',v.BONE)
base=H/'quality-warning-connected-union-038-typed-metadata-005-001'
paths={'O':base/'O-merged-run004-run005-001/typed-output/candidate-O-connected-union-typed.npz','I':base/'I-run-002/candidate-I-connected-union-typed.npz','D':H/'fat-inferred-boundary-positive-volume-127-root/candidate-D-inferred-boundary-f32-repair.npz'}
oldD=load(base/'D-run-001/candidate-D-connected-union-typed.npz');D=load(paths['D'])
changed=np.flatnonzero(np.any(D['points_world_m'].view(np.uint32)!=oldD['points_world_m'].view(np.uint32),axis=1));assert changed.tolist()==[19084]
for k in oldD:
 if k not in ['points_world_m','points_world_m_f32']:assert D[k].dtype==oldD[k].dtype and D[k].shape==oldD[k].shape and D[k].tobytes()==oldD[k].tobytes(),k
assert D['points_world_m'].tobytes()==D['points_world_m_f32'].tobytes()
df=np.flatnonzero(np.any(D['faces']==19084,axis=1));q=D['points_world_m'][D['faces'][df]].astype(float);p=oldD['points_world_m'][D['faces'][df]].astype(float)
n=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);on=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);areas=np.linalg.norm(n,axis=1);oa=np.linalg.norm(on,axis=1)
assert np.all(areas>0);cos=np.einsum('ij,ij->i',n,on)/areas/oa;assert np.all(cos>0)
prior=json.loads(baseline.read_text());targets=[]
for role in ['O','I','D']:
 a=load(paths[role])
 for t in prior['roles'][role]['protected_targets']:
  row=int(t['output_row']);ref=(oldD if role=='D'else a)['points_world_m'][row]
  assert a['points_world_m'][row].tobytes()==ref.tobytes()
  targets.append([role,int(t['canonical_protected_row']),row])
assert len(targets)==149
report={'status':'STARTED','physical_admission':False,'runtime':0,'accepted_steps':0,'baseline_complete_delta_proof':{'path':str(baseline),'sha256':sha(baseline)},'source_exact_owner_sha256':sha(v.OWNER),'coordinate_change_m':(D['points_world_m'][19084].astype(float)-oldD['points_world_m'][19084]).tolist(),'changed_D_vertices':changed.tolist(),'changed_D_faces':df.tolist(),'all149_attachment_coordinates_unchanged':targets,'changed_face_min_normal_cosine':float(cos.min()),'changed_face_area_ratio_range':[float((areas/oa).min()),float((areas/oa).max())],'sources':{role:{'path':str(path),'sha256':sha(path)}for role,path in paths.items()}}
(out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
allrecords=[];selected=set();facebase=0;pointbase=0
for role in ['O','I','D']:
 a=load(paths[role]);rr,_=pred.float32_surface_records(a['points_world_m'].astype(float),a['faces'],source_seam_groups=())
 records=[(r[0],r[1],r[2],facebase+i,tuple(pointbase+int(x)for x in r[4]))for i,r in enumerate(rr)]
 allrecords.extend(records)
 if role=='D':selected.update(map(int,df+facebase))
 facebase+=len(records);pointbase+=len(a['points_world_m'])
assert len(allrecords)==646159
report['complete_current_surface_face_count']=len(allrecords)
report['changed_vs_complete_current_surfaces']=pred._audit_self_pairs_touching_faces(allrecords,selected)
assert report['changed_vs_complete_current_surfaces']['count']==0,report['changed_vs_complete_current_surfaces']
report['status']='CHECKING_ALL185_BONE_DOMAINS';(out/'progress.json').write_text(json.dumps(report,indent=2)+'\n')
bones,breceipt=bone.load_bone_barrier_records(pred,root=ROOT,vertex_offset=pointbase,expected_native_sha256='e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5',expected_manifest_sha256='0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c')
assert len(bones)==938272
changedrecords=[r for r in allrecords if int(r[3])in selected]
report['changed_vs_all185_both_domains']=pred._audit_pair_prepared_first(pred._prepare_surface_aabb(bones),changedrecords,same_surface=False)
assert report['changed_vs_all185_both_domains']['count']==0,report['changed_vs_all185_both_domains']
report['bone_sources']=breceipt;report['status']='PASS_EXACT_CURRENT_D_ONE_ULP_DELTA';report['elapsed_s']=time.monotonic()-started
report['interpretation']='Complete current geometry coverage consists of pinned unchanged-record baseline proof plus exact changed ten-face checks against all current O/I/D records and all185 bone domains. This is geometry only, not mass/native/motion admission.'
(out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':report['status'],'changed_faces':len(df),'elapsed_s':report['elapsed_s'],'surface_crossings':report['changed_vs_complete_current_surfaces']['count'],'bone_crossings':report['changed_vs_all185_both_domains']['count'],'min_normal_cosine':float(cos.min()),'area_ratio':report['changed_face_area_ratio_range']}),flush=True)
