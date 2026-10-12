import ast,json,pickle,hashlib,sys
from pathlib import Path
from fractions import Fraction as F
import numpy as np
P=Path('/Users/n/numi-human-left-pq-publication-1344/src');sys.path.insert(0,str(P))
from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.cardiac_cavity_geometry import analyze_topology
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
H=Path('/Users/n/numi-human-fds132-m6-1350/attempt-158-right-candidate150-local-diagonal-conditioning-001/audit_candidate158_010.py')
C=Path('/Users/n/numi-human-fds132-m6-1350/attempt-166-right-capB-ancestry-correction-001/result-001/candidate.pkl')
wanted={'req','exact_records','topology_volume','union_faces','exact_geometry'}
nodes=[n for n in ast.parse(H.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in wanted]
ns={'F':F,'np':np,'ci':ci,'analyze_topology':analyze_topology};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(H),'exec'),ns)
cp=pickle.loads(C.read_bytes());faces={sid:cp['surfaces'][sid]['faces_global'] for sid in (131,133)}
positions={sid:{int(g):p for g,p in zip(cp['surfaces'][sid]['global_vertex_ids'],cp['surfaces'][sid]['vertices_float32_m'])} for sid in (131,133)}
r=ns['exact_geometry'](cp,faces,positions,'exact-source166')
r['candidate_path']=str(C);r['candidate_sha256']=hashlib.sha256(C.read_bytes()).hexdigest()
r['source_gates_pass']=all(v['exact_self_pair_count']==0 and v['exact_topology_error'] is None and v['float32_topology_volume']['topology']['closed_oriented_manifold_candidate'] for v in r['rows'].values()) and r['union']['union_exact_pair_count']==0 and r['union']['union_exact_topology_error'] is None and r['union']['shared_face_pairs']==2275 and not r['union']['interface_errors']
r['candidate_admission']=False
out=A/'right166-source-exact-audit-001.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'source_gates_pass':r['source_gates_pass'],'rows':{k:v['exact_self_pair_count'] for k,v in r['rows'].items()},'union':r['union']['union_exact_pair_count'],'report':str(out)}),flush=True)
assert r['source_gates_pass']

