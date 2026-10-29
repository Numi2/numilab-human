from pathlib import Path
import importlib.util,json,hashlib,time
import numpy as np
A=Path(__file__).resolve().parent.parent
P=A.parent.parent
BASE=P/'reduced-physical-boundary-400/attempt-030-connected-constrained-edge-fit/run-006/candidate.npz'
SRC=A/'authoring-source-chain/build_minimal_coupled_bone32_buffer_003.py'
MASK=A/'run-003/mask-inputs.npz'
OUT=A/'baseline-regional-solid-audit-003'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir()
 spec=importlib.util.spec_from_file_location('regional_owner',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 exact=m.exact_setup();b=np.load(BASE,allow_pickle=False);z=np.load(MASK,allow_pickle=False)
 p=b['points_current_world_m_f32'].astype(np.float64)
 r={'status':'SOURCE_ONLY_BASELINE_REGIONAL_SELF_AUDIT','physical_admission':False,'runtime_created':0,'accepted_steps':0,'inputs':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [BASE,SRC,MASK,Path(__file__)]},'surfaces':{}}
 for name in ['skin','fat']:
  f=z['source_'+name+'23_faces_global_node_rows'].astype(np.int64)
  start=time.time();a=m.exact_self(exact,p,f)
  r['surfaces'][name]={'faces':len(f),'exact_self':a,'seconds':time.time()-start}
 (OUT/'report.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps({name:{'faces':v['faces'],'self_pairs':v['exact_self']['pair_count'],'seconds':v['seconds']} for name,v in r['surfaces'].items()}))
if __name__=='__main__':main()
