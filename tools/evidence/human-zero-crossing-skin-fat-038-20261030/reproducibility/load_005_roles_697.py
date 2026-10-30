from pathlib import Path
import hashlib,importlib.util,json,time
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
TASK=H/'quality-warning-connected-union-038-typed-metadata-005-001'
HELPER=H/'current-D-proof-015/handoff-source004/prepare_tetgen_handoff_697_typedD.py'
HELPER_SHA='83d82a37cc4ceacccd073e54375128bb748ca6e0aec717d7d34dd8954aa0b80f'
OUT=TASK/'actual-697-I-D-load-preflight.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
if sha(HELPER)!=HELPER_SHA:raise ValueError('pinned 697 helper changed')
spec=importlib.util.spec_from_file_location('handoff697_005',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
owner=helper.load_owner()
results={}
for role,rel in [('I','I-run-002/candidate-I-connected-union-typed.npz'),('D','D-run-001/candidate-D-connected-union-typed.npz')]:
 p=TASK/rel;t=time.time();s=helper.load_surface_217(owner,p,role,None)
 points=np.asarray(s['points'],np.float32);faces=np.asarray(s['faces'],np.int64)
 if points.ndim!=2 or points.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(points).all():raise ValueError(role+' loaded surface shape/finiteness')
 if len(faces)==0 or faces.min()<0 or faces.max()>=len(points):raise ValueError(role+' loaded face rows out of point domain')
 for k in ('ancestry','ancestry_original_ids','ancestry_parent_rows'):
  if len(s[k])!=len(faces):raise ValueError(role+' loaded '+k+' count mismatch')
 lengths=[len(x) for x in s['ancestry']]
 if not lengths or min(lengths)<1:raise ValueError(role+' has empty loaded ancestry')
 results[role]={'path':str(p),'sha256':sha(p),'points':int(len(points)),'faces':int(len(faces)),'loaded_ancestry_tuple_count':int(sum(lengths)),'ancestry_kind_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(s['ancestry_kind'],return_counts=True))},'identity_schema_count':len(s['typed_identity_schema']),'exact_source_rows_nonnegative':int(np.count_nonzero(np.asarray(s['exact_source_rows'])>=0)),'used_point_count':int(len(np.unique(faces.reshape(-1)))),'elapsed_s':time.time()-t}
report={'schema':'quality-warning-connected-union-005-actual-697-role-load-preflight.v1','status':'PASS_217_I_D_TYPED_SURFACE_LOAD_ONLY','helper_path':str(HELPER),'helper_sha256':sha(HELPER),'owner_build_path':str(helper.BUILD),'owner_build_sha256':sha(helper.BUILD),'owner_build_expected_sha256':helper.BUILD_SHA,'roles':results,'physical_admission':False,'tetgen_run':False,'native_runtime_run':False,'limitations':['This directly invokes the pinned 697/217 loader for current typed I and D only. The merged O role is pending its own independent verification and typed build. No PLC, volume, mass, or physical/native admission is established.']}
OUT.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':report['status'],'report':str(OUT),'sha256':sha(OUT),'roles':{r:{'points':x['points'],'faces':x['faces'],'ancestry':x['loaded_ancestry_tuple_count']} for r,x in results.items()}},sort_keys=True))
