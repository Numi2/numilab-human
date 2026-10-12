from __future__ import annotations
import ast,hashlib,json,pickle,sys,resource,os
from pathlib import Path
from fractions import Fraction as F
import numpy as np
ROOT=Path('/Users/n/numi-human-fds132-m6-1350')
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
PUB=Path('/Users/n/numi-human-left-pq-publication-1344/src')
C=ROOT/'attempt-177-right-pentagon-option2-checkpoint-001/result-002/candidate.pkl'
OWNER=ROOT/'attempt-158-right-candidate150-local-diagonal-conditioning-001/audit_candidate158_010.py'
CI=PUB/'numilab_human/cardiac_cavity_intersections.py'
NAT=A/'back-forearm-native-014-012-right166/integrated-right-back-audit-001'
NATREPORT=NAT/'report.json'
OUT=ROOT/'attempt-180-right177-native-all8-exact-audit-001/result-001'
OUT.mkdir(parents=True,exist_ok=False)
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
try:
 lim=int(2.5*1024**3);resource.setrlimit(resource.RLIMIT_AS,(lim,lim))
except Exception:pass
sys.path.insert(0,str(PUB))
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(C)=='2be23c705be2a6589a709481f1dcf57a09b87481e6524909ea95c6048818e5e9'
assert sha(CI)=='05ba7856fc373f55a1270eb5f3ef887a4eac493ad9fb4590a618caee952d2a32'
want={'req','exact_records','topology_volume','union_faces','exact_geometry'}
nodes=[n for n in ast.parse(OWNER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in want]
ns={'F':F,'np':np,'ci':ci,'analyze_topology':analyze_topology};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OWNER),'exec'),ns)
cp=pickle.loads(C.read_bytes()); basecp=pickle.loads((ROOT/'attempt-166-right-capB-ancestry-correction-001/result-001/candidate.pkl').read_bytes()); faces={sid:[tuple(map(int,t)) for t in cp['surfaces'][sid]['faces_global']] for sid in (131,133)}
active={sid:set(int(g) for t in faces[sid] for g in t) for sid in (131,133)}
ids={sid:list(map(int,cp['surfaces'][sid]['global_vertex_ids'])) for sid in (131,133)}
steps=(0,3807,4767,5023,5599,6207,7423,8000)
fullreport=json.loads(NATREPORT.read_text())
# Verify the retained native report explicitly identifies the eight actual accepted packs and original candidate166.
assert fullreport['complete'] and fullreport['immutable_pin_count']>=150
pose_results={};pins={str(C):sha(C),str(OWNER):sha(OWNER),str(CI):sha(CI),str(NATREPORT):sha(NATREPORT)}
for step in steps:
 poses={};step_npz_pins={}
 for sid in (131,133):
  p=NAT/f'step-{step}-row-{sid}.npz'; b=p.read_bytes();pins[str(p)]=hashlib.sha256(b).hexdigest();step_npz_pins[str(sid)]={'path':str(p),'sha256':pins[str(p)]}
  z=np.load(p,allow_pickle=False)
  npz_ids=list(map(int,z['global_vertex_ids'])); xyz=np.asarray(z['positions'],dtype='<f4'); packed_faces=np.asarray(z['faces'],dtype=np.int64)
  # The NPZ face layout is the actual candidate166 native row; vertex IDs are retained, independent of compact order.
  base_faces=basecp['surfaces'][sid]['faces_global']
  native_local={g:i for i,g in enumerate(npz_ids)}
  assert np.array_equal(packed_faces,np.asarray([[native_local[int(g)] for g in t] for t in base_faces],dtype=np.int64)),(step,sid,'native NPZ face/global-ID mapping differs from pinned166')
  # The immutable report NPZ is expected to contain every referenced candidate177 global ID; only unreferenced listed IDs can be omitted.
  assert active[sid] == set(npz_ids), (step,sid,len(active[sid]),len(npz_ids),len(active[sid]-set(npz_ids)),len(set(npz_ids)-active[sid]))
  assert packed_faces.shape==(len(cp['surfaces'][sid]['faces_global']),3)
  # Reconstruct full candidate vertex map in candidate global-ID order; native capture fills every referenced ID.
  coord={g:tuple(float(v) for v in xyz[i]) for i,g in enumerate(npz_ids)}
  for i,g in enumerate(ids[sid]):
   if g not in coord: coord[g]=tuple(float(v) for v in cp['surfaces'][sid]['vertices_float32_m'][i])
  poses[sid]=coord
  z.close()
 result=ns['exact_geometry'](cp,faces,poses,f'native-step-{step}')
 # Persist each numerical result immediately before evaluating gates/report enrichment.
 rows={k:{'exact_degenerate_count':v['exact_degenerate_count'],'exact_topology_component_count':v['exact_topology_component_count'],'exact_topology_error':v['exact_topology_error'],'exact_self_pair_count':v['exact_self_pair_count'],'exact_self_pairs':v['exact_self_pairs'],'float32_topology_volume':v['float32_topology_volume']} for k,v in result['rows'].items()}
 u=result['union']
 out={'schema':'numi.human.right177-native-pose-exact-audit-180-step.v1','step':step,'candidate_path':str(C),'candidate_sha256':sha(C),'source_native_report_path':str(NATREPORT),'source_native_report_sha256':sha(NATREPORT),'positions_npz_pins':step_npz_pins,'rows':rows,'union':u,'candidate_admission':False}
 op=OUT/f'step-{step}.json';op.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 pose_results[str(step)]={'report':str(op),'report_sha256':sha(op),'row_self_pairs':{k:v['exact_self_pair_count'] for k,v in result['rows'].items()},'row_degenerates':{k:v['exact_degenerate_count'] for k,v in result['rows'].items()},'union_pair_count':u['union_exact_pair_count'],'union_degenerates':u['union_exact_degenerate_count'],'union_shared_face_pairs':u['shared_face_pairs'],'union_topology_error':u['union_exact_topology_error']}
 summary={'schema':'numi.human.right177-native-all8-exact-audit-180.v1','candidate_path':str(C),'candidate_sha256':sha(C),'source_native_report_path':str(NATREPORT),'source_native_report_sha256':sha(NATREPORT),'helper_owner_path':str(OWNER),'helper_owner_sha256':sha(OWNER),'intersection_owner_path':str(CI),'intersection_owner_sha256':sha(CI),'pose_results':pose_results,'completed_steps':list(map(int,pose_results.keys())),'candidate_admission':False,'complete':False}
 (OUT/'progress.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'step':step,'rows':pose_results[str(step)]['row_self_pairs'],'degenerates':pose_results[str(step)]['row_degenerates'],'union_pairs':u['union_exact_pair_count'],'union_degenerates':u['union_exact_degenerate_count'],'shared':u['shared_face_pairs'],'topology_error':u['union_exact_topology_error']},sort_keys=True),flush=True)
summary['complete']=len(pose_results)==len(steps)
summary['all_pose_geometry_gates_pass']=summary['complete'] and all(all(v==0 for v in x['row_self_pairs'].values()) and all(v==0 for v in x['row_degenerates'].values()) and x['union_pair_count']==0 and x['union_degenerates']==0 and x['union_topology_error'] is None for x in pose_results.values())
summary['candidate_admission']=False
path=OUT/'report.json';path.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps({'report':str(path),'report_sha256':sha(path),'all_pose_geometry_gates_pass':summary['all_pose_geometry_gates_pass']},sort_keys=True),flush=True)
