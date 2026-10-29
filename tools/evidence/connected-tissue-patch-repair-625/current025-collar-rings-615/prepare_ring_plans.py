from pathlib import Path
import collections,csv,hashlib,importlib.util,json,sys
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
BASE=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh"
CAND=BASE/"fine-I-bone-barrier-run-024/closed603-merge-025/candidate-f32-linked-remesh-typed.npz"
WIT=BASE/"fine-I-bone-barrier-run-024/full-native-self-028/residual-pair-attribution-029/witness-pairs.csv"
INV=P/"boundary-feasibility-612/current025-collar-rings-615/run-001/result.json"
HELP=BASE/"closed603-merge-020/authoring-source-chain/boundary_helper.py"
OUT=P/"boundary-feasibility-612/current025-collar-rings-615/prepared-ring-plans-002.json"
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def edge(a,b):return (min(int(a),int(b)),max(int(a),int(b)))
assert sha(CAND)=="9790ed1fa49bacff36ea169c7b8354344e40f1df33f604c006bcfb7991723a2a"
inv=json.loads(INV.read_text()); assert inv["candidate025_sha256"]==sha(CAND)
with np.load(CAND,allow_pickle=False) as z:F=np.asarray(z["faces"],np.int64)
adj=[set() for _ in range(len(F))]; ef=collections.defaultdict(list)
for row,tri in enumerate(F):
 for a,b in zip(tri,np.roll(tri,-1)):ef[edge(a,b)].append(row)
for rows in ef.values():
 for r in rows:adj[r].update(q for q in rows if q!=r)
seeds=collections.defaultdict(set); pairs=collections.defaultdict(set)
with WIT.open(newline="") as f:
 for row in csv.DictReader(f):
  pi=int(row["pair_index"])
  for side in ("a","b"):
   raw=row[f"failed_transaction_{side}"]
   if raw in ("","None","null"):continue
   tx=int(raw);face=int(row[f"candidate_face_row_{side}"])
   assert 0<=face<len(F)
   seeds[tx].add(face);pairs[tx].add(pi)
assert len(pairs[34])==83 and len(pairs[80])==48
helper=load("boundaryplan615",HELP)
plans=[]
for tx,rings in ((34,(2,3,4,5,6)),(80,(1,2,3))):
 for ring in rings:
  selected=set(seeds[tx]);front=set(selected)
  for _ in range(ring):
   nxt=set()
   for r in front:nxt.update(adj[r])
   nxt-=selected;selected.update(nxt);front=nxt
  loops,_=helper.boundary_cycles(F,sorted(selected))
  ec=collections.Counter();verts=set()
  for r in selected:
   tri=F[r];verts.update(map(int,tri))
   for a,b in zip(tri,np.roll(tri,-1)):ec[edge(a,b)]+=1
  chi=len(verts)-len(ec)+len(selected)
  assert chi==1 and len(loops)==1,(tx,ring,chi,len(loops))
  face_rows=sorted(selected)
  plans.append({"transaction_id":tx,"ring":ring,"face_rows":face_rows,"boundary_loops":loops,
   "seed_face_rows":sorted(seeds[tx]),"pair_indices":sorted(pairs[tx]),
   "face_rows_sha256":hashlib.sha256(np.asarray(face_rows,dtype="<i8").tobytes()).hexdigest(),
   "faces":len(face_rows),"vertices":len(verts),"loop_lengths":sorted(map(len,loops),reverse=True),"euler_characteristic":chi})
payload={"schema":"candidate025-exact-ear-ring-plans.v1","candidate_path":str(CAND),"candidate_sha256":sha(CAND),
 "witness_path":str(WIT),"witness_sha256":sha(WIT),"inventory_path":str(INV),"inventory_sha256":sha(INV),
 "boundary_helper_path":str(HELP),"boundary_helper_sha256":sha(HELP),"rows_are_candidate025_output_face_rows":True,
 "plans":plans,"physical_admission":False}
OUT.write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps({"output":str(OUT),"sha256":sha(OUT),"choices":[{k:p[k] for k in ("transaction_id","ring","faces","vertices","loop_lengths","face_rows_sha256")} for p in plans]},indent=2))
