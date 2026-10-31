
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,subprocess,time
H=Path(__file__).resolve().parent;PY='/Users/numi/numi-human-prep-venv-20261005/bin/python'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def mod(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def load(p):
 with np.load(p) as z:return {k:z[k] for k in z.files}
T=mod('bind498transfer',H/'transfer_surface_contractions_indexed_379_root.py')
out=H/'skin-paired-I-flip-target-498-root';out.mkdir(exist_ok=False);start=time.monotonic();commands=[]
ip=H/'outer-O-quality-exchange-484-root/exchange/qualification435001/skin-target437/skin-bound-current-O-target-volume-proposal.npz'
assert sha(ip)=='f683337f12859af5f5b1239bd137e0eba2c86352fd432a65c5c87d33d4634b7d'
joint=H/'joint-current-source-497-root/audit354/report.json'
assert sha(joint)=='e051b6b879210147c76916d9fa19a3a6125d4bb7a8e977d67553b36d59ce4e01'
ar=json.loads(joint.read_text());assert ar['status']=='PASS_EXACT_JOINT_CHANGED_OID_VS_RETAINED_WALLS_ALL185'
cfgp=Path(ar['config']['path']);assert sha(cfgp)==ar['config']['sha256'];cfg=json.loads(cfgp.read_text())
oldI=Path(cfg['baseline_candidates']['I']['path']);I=Path(cfg['candidate_surfaces']['I']['path'])
assert sha(oldI)=='0e48a2f42dc6eb7ee1f1b0ce6fc63213e70dde6cc9f398ef58b03e7dae0d85b5'
assert sha(I)=='81b285a4e12f11dfd4156bb2545bb86f5ccee900ef6021a11beff36954b00774'
base=load(ip);bI=load(oldI);nI=load(I);p=base['points_world_m'];t=base['tetrahedra']
assert np.array_equal(bI['points_world_m_f32'].view(np.uint32),nI['points_world_m_f32'].view(np.uint32))
assert bI['faces'].shape==nI['faces'].shape
changed=np.flatnonzero(np.any(bI['faces']!=nI['faces'],axis=1));assert len(changed)==2
actual=T.boundary(t);target=base['desired_boundary_faces'].copy()
assert np.array_equal(T.boundary_identity(actual),T.boundary_identity(target))
lookup={T.key(x):i for i,x in enumerate(p)};assert len(lookup)==len(p)
mp=np.asarray([lookup[T.key(x)] for x in bI['points_world_m_f32']],np.int64)
before=mp[bI['faces'][:,[0,2,1]]];after=mp[nI['faces'][:,[0,2,1]]]
def facekey(f):return tuple(map(int,T.boundary_identity(np.asarray(f).reshape(1,3))[0]))
tl={facekey(f):i for i,f in enumerate(target)};assert len(tl)==len(target)
changes=[]
for row in changed:
 idx=tl[facekey(before[row])]
 changes.append({'I_output_face_row':int(row),'target_face_row':int(idx),'old_oriented_volume_vertices':target[idx].tolist(),'new_oriented_volume_vertices':after[row].tolist()})
 target[idx]=after[row]
assert len(np.unique(np.sort(target,axis=1),axis=0))==len(target)
keys={facekey(f) for f in target}
assert all(facekey(f) in keys for f in after)
edges=np.concatenate([target[:,[0,1]],target[:,[1,2]],target[:,[2,0]]]);se=np.sort(edges,axis=1);order=np.lexsort((se[:,1],se[:,0]));ss=se[order]
starts=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];counts=np.diff(np.r_[starts,len(ss)]);sign=np.where(edges[:,0]<edges[:,1],1,-1)
assert np.all(counts==2) and np.all(np.add.reduceat(sign[order],starts)==0)
base.update(desired_boundary_faces=target,typed_I_candidate_path=np.asarray([str(I)]),typed_I_candidate_sha256=np.asarray([sha(I)]))
candidate=out/'skin-current-I-flip-target-bound.npz';np.savez_compressed(candidate,**base)
with np.load(candidate) as z:
 for k in base:assert np.array_equal(z[k],base[k])
beforekeys={facekey(f) for f in actual};targetkeys={facekey(f) for f in target}
missing=np.asarray([f for f in target if facekey(f) not in beforekeys],np.int64);extra=np.asarray([f for f in actual if facekey(f) not in targetkeys],np.int64)
assert len(missing)==len(extra)==2
br={'status':'PASS_TYPED_I_LOCAL_FLIP_SKIN_TARGET_REBIND_SOURCE_ONLY','physical_admission':False,'native_steps':0,'source':str(ip),'source_sha256':sha(ip),'candidate':str(candidate),'candidate_sha256':sha(candidate),'old_I':str(oldI),'old_I_sha256':sha(oldI),'current_I':str(I),'current_I_sha256':sha(I),'full_context':str(joint),'full_context_sha256':sha(joint),'changes':changes,'source_point_coordinates_unchanged':True,'actual_positive_volume_geometry_unchanged':True,'target_oriented_chain_closed':True,'missing_target_faces':2,'extra_actual_faces':2,'native_quality_failures':1111,'mass_mapping':'NOT_ADMITTED_CURRENT_SOURCE330_REBIND_REQUIRED','producer_sha256':sha(__file__)}
(out/'target-bind-report.json').write_text(json.dumps(br,indent=2)+'\n')
# Existing304 exact witness closure, seeded only by the changed physical boundary.
Q=mod('prepare498',H/'prepare_quality_neighbor_cavities_304_root.py')
comps,prep=Q.prepare(p,t,base['material_indices'],target,actual,missing,extra,maximum_cells=4000)
pd=out/'preparation304';pd.mkdir(exist_ok=False);po=[0];fo=[0];co=[0]
for c in comps:po.append(po[-1]+len(c['global_points']));fo.append(fo[-1]+len(c['faces_local']));co.append(co[-1]+len(c['rows']))
shell=pd/'connected-cavity-shells.npz'
np.savez_compressed(shell,point_offsets=po,face_offsets=fo,cell_offsets=co,global_point_rows=np.concatenate([c['global_points'] for c in comps]),local_faces=np.concatenate([c['faces_local'] for c in comps]),input_cell_rows=np.concatenate([c['rows'] for c in comps]))
pr={'status':'PASS_COMPACT_TARGET_CAVITY_PREPARATION' if prep.get('all_target_components_admissible',False) else 'COMPACT_TARGET_CAVITY_WITNESSES_REMAIN','physical_admission':False,'native_steps':0,'input':str(candidate),'input_sha256':sha(candidate),'script_sha256':sha(__file__),'preparation':prep,'components':[{k:v for k,v in c.items() if k not in ('rows','global_points','faces_local')} for c in comps],'output':str(shell),'output_sha256':sha(shell),'elapsed_s':time.monotonic()-start}
(pd/'report.json').write_text(json.dumps(pr,indent=2)+'\n');print(json.dumps({'target':br,'preparation':prep}),flush=True)
result={'status':'SKIN_I_FLIP_TARGET_PREPARED','physical_admission':False,'native_steps':0,'target_bind_report':str(out/'target-bind-report.json'),'target_bind_report_sha256':sha(out/'target-bind-report.json'),'preparation_report':str(pd/'report.json'),'preparation_report_sha256':sha(pd/'report.json')}
if prep.get('all_target_components_admissible',False):
 cmd=[PY,str(H/'remesh_compact_target_cavities_293_root.py'),'--input',str(candidate),'--output',str(out/'remesh293-S0'),'--prepared-dir',str(pd),'--prepared-sha256',sha(shell),'--seconds','30','--component-seconds','5']
 commands.append(cmd)
 with (out/'remesh293-S0.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 rp=out/'remesh293-S0/report.json';r=json.loads(rp.read_text())
 result.update(status='SKIN_I_FLIP_BOUNDARY_REPAIR_TRIAL_FINISHED',remesh_report=str(rp),remesh_report_sha256=sha(rp),passing_components=r['passing_components'],complete_boundary_matches_desired=r['complete_boundary_matches_desired'],remaining_quality_failures=r['remaining_quality_failures'],candidate=r['output'],candidate_sha256=r['output_sha256'])
result.update(commands=commands,producer_sha256=sha(__file__),elapsed_s=time.monotonic()-start)
(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
