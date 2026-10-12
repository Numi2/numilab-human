from __future__ import annotations
import ast,hashlib,json,pickle,sys,resource
from pathlib import Path
from fractions import Fraction as F
import numpy as np
os_env=None
ROOT=Path('/Users/n/numi-human-fds132-m6-1350')
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
PUB=Path('/Users/n/numi-human-left-pq-publication-1344/src')
C=ROOT/'attempt-177-right-pentagon-option2-checkpoint-001/result-002/candidate.pkl'
PARENT=ROOT/'attempt-166-right-capB-ancestry-correction-001/result-001/candidate.pkl'
OWNER=ROOT/'attempt-158-right-candidate150-local-diagonal-conditioning-001/audit_candidate158_010.py'
CI=PUB/'numilab_human/cardiac_cavity_intersections.py'
OUT=ROOT/'attempt-179-right177-exact-source-gate-001/result-001'
OUT.mkdir(parents=True,exist_ok=False)
try:
 lim=int(2.5*1024**3);resource.setrlimit(resource.RLIMIT_AS,(lim,lim))
except Exception:pass
sys.path.insert(0,str(PUB))
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(C)=='2be23c705be2a6589a709481f1dcf57a09b87481e6524909ea95c6048818e5e9'
assert sha(PARENT)=='25d759377d4861660f12c6458cd5bde4d48e152462048f15aea6410b70b2c17d'
want={'req','exact_records','topology_volume','union_faces','exact_geometry'}
nodes=[n for n in ast.parse(OWNER.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in want]
ns={'F':F,'np':np,'ci':ci,'analyze_topology':analyze_topology};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(OWNER),'exec'),ns)
cp=pickle.loads(C.read_bytes());base=pickle.loads(PARENT.read_bytes())
faces={sid:[tuple(map(int,t)) for t in cp['surfaces'][sid]['faces_global']] for sid in (131,133)}
positions={sid:{int(g):p for g,p in zip(cp['surfaces'][sid]['global_vertex_ids'],cp['surfaces'][sid]['vertices_float32_m'])} for sid in (131,133)}
# Exact source predicates use candidate Fraction coordinates; F32 topology uses retained packed vertices.
r=ns['exact_geometry'](cp,faces,positions,'source-exact177')
basefaces={sid:[tuple(map(int,t)) for t in base['surfaces'][sid]['faces_global']] for sid in (131,133)}
b=ns['exact_geometry'](base,basefaces,positions,'source-base166')
# Store critical evidence immediately, including a failed outcome.
for key in ('rows','union'): pass
source_rows={k:{'exact_degenerate_count':v['exact_degenerate_count'],'exact_topology_component_count':v['exact_topology_component_count'],'exact_topology_error':v['exact_topology_error'],'exact_self_pair_count':v['exact_self_pair_count'],'exact_self_pairs':v['exact_self_pairs'],'float32_topology_volume':v['float32_topology_volume']} for k,v in r['rows'].items()}
report={'schema':'numi.human.right177-exact-source-gate-179.v1','candidate_path':str(C),'candidate_sha256':sha(C),'parent_candidate_path':str(PARENT),'parent_candidate_sha256':sha(PARENT),'helper_owner_path':str(OWNER),'helper_owner_sha256':sha(OWNER),'intersection_module_path':str(CI),'intersection_module_sha256':sha(CI),'rows':source_rows,'union':r['union'],'baseline_source':{'rows':{k:{'self_pair_count':v['exact_self_pair_count'],'topology_error':v['exact_topology_error']} for k,v in b['rows'].items()},'union_pair_count':b['union']['union_exact_pair_count'],'union_shared_face_pairs':b['union']['shared_face_pairs']},'candidate_admission':False}
report['source_gate_pass']=all(v['exact_degenerate_count']==0 and v['exact_self_pair_count']==0 and v['exact_topology_error'] is None and v['float32_topology_volume']['topology']['closed_oriented_manifold_candidate'] for v in r['rows'].values()) and r['union']['union_exact_degenerate_count']==0 and r['union']['union_exact_pair_count']==0 and r['union']['union_exact_topology_error'] is None and not r['union']['interface_errors'] and r['union']['shared_face_pairs']==2276
(OUT/'source-critical-report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'source_gate_pass':report['source_gate_pass'],'rows':{k:{'self':v['exact_self_pair_count'],'degenerate':v['exact_degenerate_count'],'topology':v['exact_topology_error'],'components':v['float32_topology_volume']['topology']['face_component_count'],'signs':v['float32_topology_volume']['component_signs']} for k,v in r['rows'].items()},'union':{'pairs':r['union']['union_exact_pair_count'],'degen':r['union']['union_exact_degenerate_count'],'topology':r['union']['union_exact_topology_error'],'shared':r['union']['shared_face_pairs'],'interface_errors':r['union']['interface_errors']},'report':str(OUT/'source-critical-report.json')}),flush=True)
