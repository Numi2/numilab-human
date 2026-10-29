#!/usr/bin/env python3
"""Topology-only 1..8 edge-ring inventory around actual candidate025 witnesses."""
from pathlib import Path
import collections,csv,hashlib,importlib.util,json,sys,time
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
BASE=P/'geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024'
REPORT=BASE/'full-native-self-028/residual-pair-attribution-029/report.json'
WITNESSES=BASE/'full-native-self-028/residual-pair-attribution-029/witness-pairs.csv'
BOUNDARY=BASE/'../closed603-merge-020/authoring-source-chain/boundary_helper.py'
OUT=P/'boundary-feasibility-612/current025-collar-rings-615/run-001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def loadmod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def edge(a,b):return (min(int(a),int(b)),max(int(a),int(b)))
started=time.monotonic();rr=json.loads(REPORT.read_text());cand=Path(rr['candidate_path'])
if sha(cand)!=rr['candidate_sha256']:raise SystemExit('candidate025 pin mismatch')
if sha(REPORT)=='' or not WITNESSES.is_file():raise SystemExit('witness evidence missing')
with np.load(cand,allow_pickle=False) as z:
 F=np.asarray(z['faces'],np.int64)
if len(F)!=int(rr.get('output_faces',len(F))):raise SystemExit('candidate025 face count mismatch')
# Build face adjacency through shared edges in the candidate025 row domain.
edge_faces=collections.defaultdict(list)
for row,(a,b,c) in enumerate(F):
 for u,v in ((a,b),(b,c),(c,a)):edge_faces[edge(u,v)].append(int(row))
adj=[set() for _ in range(len(F))]
for rows in edge_faces.values():
 for r in rows:
  adj[r].update(q for q in rows if q!=r)
seeds=collections.defaultdict(set);pair_indices=collections.defaultdict(list)
with WITNESSES.open(newline='') as f:
 for r in csv.DictReader(f):
  pi=int(r['pair_index'])
  for side in ('a','b'):
   tx=r[f'failed_transaction_{side}']
   if tx in ('','None','null'):continue
   tx=int(tx);face=int(r[f'candidate_face_row_{side}'])
   if face<0 or face>=len(F):raise SystemExit(f'witness face row out of candidate025 domain: {face}')
   seeds[tx].add(face);pair_indices[tx].append(pi)
expected={'34':83,'80':48}
for tx,n in expected.items():
 if len(set(pair_indices[int(tx)]))!=n:raise SystemExit(f'tx{tx} actual witness count mismatch')
seed_union={t:set(v) for t,v in seeds.items()}
helper=loadmod('boundary_helper_615',BOUNDARY)
summary=[]
for tx in (34,80):
 selected=set(seeds[tx]);frontier=set(selected);ring_rows=[]
 for ring in range(0,9):
  if ring>0:
   nxt=set()
   for r in frontier:nxt.update(adj[r])
   nxt-=selected;selected.update(nxt);frontier=nxt
  ec=collections.Counter()
  vertices=set()
  for r in selected:
   a,b,c=map(int,F[r]);vertices.update((a,b,c))
   for u,v in ((a,b),(b,c),(c,a)):ec[edge(u,v)]+=1
  boundary_edge_count=sum(n==1 for n in ec.values());nonmanifold_edges={str(e):n for e,n in ec.items() if n>2}
  chi=len(vertices)-len(ec)+len(selected)
  try:
   loops,oriented=helper.boundary_cycles(F,sorted(selected));boundary={'status':'PASS','loop_count':len(loops),'loop_lengths':sorted(map(len,loops),reverse=True),'loops':loops}
  except Exception as exc:boundary={'status':'FAIL','error':str(exc),'loop_count':None,'loop_lengths':[],'loops':[]}
  overlaps={str(other):len(selected&other_rows) for other,other_rows in seed_union.items() if other!=tx and selected&other_rows}
  ring_rows.append({'ring':ring,'new_faces':len(frontier) if ring>0 else len(seeds[tx]),'selected_faces':len(selected),'selected_vertices':len(vertices),'unique_edges':len(ec),'euler_characteristic':chi,'boundary_edge_count':boundary_edge_count,'nonmanifold_edge_count':len(nonmanifold_edges),'nonmanifold_edge_examples':list(nonmanifold_edges.items())[:20],'exact_boundary_cycles':boundary,'selected_face_rows_sha256':hashlib.sha256(np.asarray(sorted(selected),dtype='<i8').tobytes()).hexdigest(),'overlap_with_other_failed_transaction_seeds':overlaps})
 summary.append({'transaction_id':tx,'candidate025_seed_pair_count':len(set(pair_indices[tx])),'seed_face_count':len(seeds[tx]),'seed_face_rows':sorted(seeds[tx]),'seed_pair_indices':sorted(set(pair_indices[tx])),'rings':ring_rows})
result={'schema':'candidate025-witness-collar-ring-inventory.v1','candidate025_path':str(cand),'candidate025_sha256':sha(cand),'candidate025_faces':len(F),'pair_report_path':str(REPORT),'pair_report_sha256':sha(REPORT),'witness_csv_path':str(WITNESSES),'witness_csv_sha256':sha(WITNESSES),'boundary_cycle_helper_path':str(BOUNDARY),'boundary_cycle_helper_sha256':sha(BOUNDARY),'rows_are_candidate025_output_face_rows':True,'neighborhood':'face adjacency by shared undirected edge; no coordinate tolerance or geometry mutation','transactions':summary,'scope':'topology-only collar feasibility inventory; no exact collision, containment, bone/I, physical, or motion gate','elapsed_s':time.monotonic()-started}
(OUT/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'output':str(OUT/'result.json'),'candidate_sha256':result['candidate025_sha256'],'face_count':len(F),'transactions':[{'id':x['transaction_id'],'pair_count':x['candidate025_seed_pair_count'],'seed_faces':x['seed_face_count'],'rings':[{'ring':r['ring'],'faces':r['selected_faces'],'vertices':r['selected_vertices'],'boundary':r['exact_boundary_cycles']['loop_lengths'],'boundary_status':r['exact_boundary_cycles']['status'],'chi':r['euler_characteristic'],'overlap':r['overlap_with_other_failed_transaction_seeds']} for r in x['rings']]} for x in summary],'elapsed_s':result['elapsed_s']},indent=2))
