from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
H=Path(sys.argv[1])
helper=H/'current-D-proof-015/handoff-source004/prepare_tetgen_handoff_697_typedD.py'
collar=H/'fixed-boundary-handoff-036/warning037-rim/collar-current.npz'
roles={
 'O':H/'quality-warning-connected-union-037-typed-metadata-001/O-retry1/candidate-O-connected-union-typed.npz',
 'I':H/'quality-warning-connected-union-037-typed-metadata-001/I/candidate-I-connected-union-typed.npz',
 'D':H/'quality-warning-connected-union-037-typed-metadata-001/D/candidate-D-connected-union-typed.npz',
}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 if sha(helper)!='83d82a37cc4ceacccd073e54375128bb748ca6e0aec717d7d34dd8954aa0b80f': raise ValueError('697 handoff helper changed')
 if sha(collar)!='eef1d75c8ab6ab3e4eeaa839a72dfd50c53aa43b0c02df1c98e32b2076630394': raise ValueError('037 collar changed')
 spec=importlib.util.spec_from_file_location('handoff697',helper); h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
 owner=h.load_owner()
 out={'schema':'quality-warning-037-actual-697-surface-loader-preflight.v1','status':'PASS_217_TYPED_SURFACE_LOAD_ONLY','helper_path':str(helper),'helper_sha256':sha(helper),'owner_build_path':str(h.BUILD),'owner_build_sha256':sha(h.BUILD),'owner_build_expected_sha256':h.BUILD_SHA,'collar_path':str(collar),'collar_sha256':sha(collar),'roles':{},'tetgen_run':False,'physical_admission':False,'native_runtime_run':False,'limitations':['This invokes the pinned 697/217 typed surface loader only. It does not construct a PLC, run TetGen, establish mass, or admit physical/native geometry.']}
 for role,path in roles.items():
  t=time.time(); surface=h.load_surface_217(owner,path,role,collar if role=='O' else None)
  pts=np.asarray(surface['points'],np.float32); faces=np.asarray(surface['faces'],np.int64)
  if pts.ndim!=2 or pts.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(pts).all(): raise ValueError(f'{role} actual 697 surface shape invalid')
  if int(faces.min())<0 or int(faces.max())>=len(pts): raise ValueError(f'{role} actual 697 face point index outside range')
  for field in ('ancestry','ancestry_original_ids','ancestry_parent_rows'):
   if len(surface[field])!=len(faces): raise ValueError(f'{role} loaded {field} row count mismatch')
  lens=[len(x) for x in surface['ancestry']]
  if not lens or min(lens)<1: raise ValueError(f'{role} loaded ancestry has empty faces')
  out['roles'][role]={'path':str(path),'sha256':sha(path),'points':int(len(pts)),'faces':int(len(faces)),'loaded_ancestry_tuple_count':int(sum(lens)),'ancestry_kind_counts':{str(int(k)):int(v) for k,v in zip(*np.unique(surface['ancestry_kind'],return_counts=True))},'loaded_identity_schema_count':len(surface['typed_identity_schema']),'exact_source_row_nonnegative':int(np.count_nonzero(np.asarray(surface['exact_source_rows'])>=0)),'used_point_count':int(len(np.unique(faces.reshape(-1)))),'load_elapsed_s':time.time()-t}
 outpath=H/'quality-warning-connected-union-037-typed-metadata-001/actual-697-load-preflight/report.json'; outpath.parent.mkdir(parents=True,exist_ok=True)
 outpath.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':out['status'],'report':str(outpath),'sha256':sha(outpath),'roles':{k:{'points':v['points'],'faces':v['faces'],'ancestry':v['loaded_ancestry_tuple_count']} for k,v in out['roles'].items()}}))
if __name__=='__main__':main()
