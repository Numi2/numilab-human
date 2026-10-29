#!/usr/bin/env python3
from pathlib import Path
import collections,hashlib,importlib.util,itertools,json,sys,time
import numpy as np
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
BASE=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh"
PLAN=P/"boundary-feasibility-612/current025-collar-rings-615/prepared-ring-plans-002.json"
CAND=BASE/"fine-I-bone-barrier-run-024/closed603-merge-025/candidate-f32-linked-remesh-typed.npz"
REPLAY=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/all185-source-order-native-f32-replay.npz"
BANK=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union"
BONE_TOOL=P/"bone-barrier-remeshing-609/bone_barrier.py"
MAN=BANK/"exact-set-union-remaining-013/all185-bank-operand-manifest.json"
FINE_I=BANK/"dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz"
DP=P/"constraint-disk-triangulation-616/constraint_disk.py"
OUT=P/"boundary-feasibility-612/current025-collar-rings-615/run-003-exact-dp-first"
EXPECTED={"candidate":"9790ed1fa49bacff36ea169c7b8354344e40f1df33f604c006bcfb7991723a2a",
"replay":"e9f0fffefa68f274e8d1c0a565d3d6cab1cfba31d13e085b8fdc79ca362185e5",
"manifest":"0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c",
"fineI":"5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410",
"dp":"eab86c0a1060a1292db4a9a01e0a766be8739ea2fa2c575a5658082a453d87ea",
"owner":"f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d"}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def edge(a,b):return(min(int(a),int(b)),max(int(a),int(b)))
def main():
 for k,p in (("candidate",CAND),("replay",REPLAY),("manifest",MAN),("fineI",FINE_I),("dp",DP)):
  assert sha(p)==EXPECTED[k],(k,sha(p))
 plan=json.loads(PLAN.read_text());assert plan["candidate_sha256"]==EXPECTED["candidate"]
 with np.load(CAND,allow_pickle=False) as z:
  points=np.asarray(z["points_world_m"],np.float32);faces=np.asarray(z["faces"],np.int64)
 predmod=load("exact615",P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/closed603-merge-017b/authoring-source-chain/cardiac_cavity_intersections.py") if False else None
 # Use the same exact owner loader as the validated 614 driver.
 ears=load("ears615",P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py")
 pred=ears.load_exact_owner();assert sha(pred.__file__)==EXPECTED["owner"]
 bone=load("bone615",BONE_TOOL)
 lattice=tuple(pred.float32_point_lattice_key(v)for v in points)
 edge_rows=collections.defaultdict(list)
 for row,tri in enumerate(faces):
  for a,b in zip(tri,np.roll(tri,-1)):edge_rows[edge(a,b)].append(row)
 with np.load(REPLAY,allow_pickle=False) as b:
  native_xyz=b["points_world_f32"].astype(float);native_faces=b["faces"].astype(np.int64)
 manifest=json.loads(MAN.read_text());negative=[-1];barrier_counts={}
 def overlap(xyz,fs,lo,hi):
  tris=xyz[fs]
  return np.flatnonzero(np.all(tris.max(axis=1)>=lo,axis=1)&np.all(tris.min(axis=1)<=hi,axis=1))
 def add_records(label,xyz,fs,offset,lo,hi,exact64=False):
  rows=overlap(xyz,fs,lo,hi);barrier_counts.setdefault(label,{"total_faces":len(fs),"queried_faces":0})
  barrier_counts[label]["queried_faces"]=max(barrier_counts[label]["queried_faces"],int(len(rows)))
  if not len(rows):return []
  selected=fs[rows];used=np.unique(selected);mapping=np.full(len(xyz),-1,np.int64);mapping[used]=np.arange(len(used))
  keys=tuple(bone.exact_common_lattice_key(v)if exact64 else pred.float32_point_lattice_key(v)for v in xyz[used])
  raw=pred._records(keys,mapping[selected]);out=[]
  for rec in raw:
   out.append((rec[0],rec[1],rec[2],negative[0],tuple(int(used[v])+offset for v in rec[4])));negative[0]-=1
  return out
 # First, test the smallest requested topologically valid collars.
 choices=[p for p in plan["plans"] if (p["transaction_id"],p["ring"]) in ((34,2),(80,1))]
 assert len(choices)==2
 OUT.mkdir(parents=True,exist_ok=False);attempts=[];started=time.monotonic()
 for choice in choices:
  tx,ring=choice["transaction_id"],choice["ring"];piece={"face_rows":choice["face_rows"],"boundary_loops":choice["boundary_loops"]}
  removed=set(piece["face_rows"]);allowed=np.unique(faces[sorted(removed)])
  lo=np.nextafter(points[allowed].astype(float).min(axis=0),-np.inf);hi=np.nextafter(points[allowed].astype(float).max(axis=0),np.inf)
  fixed=add_records("all185-native-F32",native_xyz,native_faces,len(points),lo,hi)
  vo=len(points)+1000000
  for bnk in manifest["operands"]:
   path=Path(bnk["operand_path"]);assert sha(path)==bnk["operand_sha256"]
   with np.load(path,allow_pickle=False)as b:
    fixed.extend(add_records(f"bone-{bnk['body']}-{bnk['bone']}",b["points_world_m_f64"],b["faces_f64"],vo,lo,hi,True));vo+=len(b["points_world_m_f64"])
  with np.load(FINE_I,allow_pickle=False)as b:
   fixed.extend(add_records("fine-I-open",b["points_world_m"].astype(float),b["faces"].astype(np.int64),vo,lo,hi))
  for row in overlap(points.astype(float),faces,lo,hi):
   if int(row) in removed:continue
   rec=pred._records(lattice,[tuple(map(int,faces[row]))])[0];fixed.append((rec[0],rec[1],rec[2],int(row),rec[4]))
  base_index=pred._prepare_surface_aabb(fixed)
  external={e for e,rows in edge_rows.items()if any(r not in removed for r in rows)}
  try:
   tris,records,info=load("dp615",DP).collision_aware_dp_disk(piece["boundary_loops"][0],points.astype(float),points,external,base_index,pred,lattice,[],615000000+tx*10000+ring,max_seconds=12,max_branches=512)
   against=pred._audit_pair_prepared_first(base_index,records,same_surface=True)
   selfaudit=pred._audit_record_pairs(itertools.combinations(records,2),same_surface=True)
   assert against["count"]==0 and selfaudit["count"]==0,(against["count"],selfaudit["count"])
   old=faces[sorted(removed)];new=np.asarray(tris,np.int64)
   def boundary(fs):
    ec=collections.Counter((int(a),int(b))for f in fs for a,b in zip(f,np.roll(f,-1)))
    out=collections.Counter()
    for (a,b),n in ec.items():
     d=n-ec[(b,a)]
     if d>0:out[(a,b)]=d
    return out
   assert boundary(old)==boundary(new),"directed boundary changed"
   op={"schema":"pair-linked-connected-remesh-compact-operations.v1","candidate_sha256":EXPECTED["candidate"],
    "transaction_id":tx,"patch_index":0,"ring":ring,"parent_face_rows":sorted(removed),"new_faces":tris,
    "boundary_loop_lengths":choice["loop_lengths"],"coordinate_edits":0,"exact_changed_vs_current_and_structures_pairs":0,
    "exact_new_self_pairs":0,"exact_recheck":{"against_count":against["count"],"self_count":selfaudit["count"]},
    "method":info["method"],"statistics":info["statistics"],"selection_cost":info["selection_cost"],
    "physical_admission":False}
   op_path=OUT/f"tx{tx}-ring{ring}.operation.json";op_path.write_text(json.dumps(op,indent=2)+"\n")
   attempts.append({"tx":tx,"ring":ring,"status":"PASS","new_faces":len(tris),"metrics":info,"operation":str(op_path)})
  except Exception as e:
   attempts.append({"tx":tx,"ring":ring,"status":"FAIL","error":repr(e)})
 result={"schema":"candidate025-near-barrier-exact-dp-first.v1","candidate_sha256":EXPECTED["candidate"],
  "plan_sha256":sha(PLAN),"dp_sha256":sha(DP),"exact_owner_sha256":EXPECTED["owner"],"replay_sha256":EXPECTED["replay"],
  "manifest_sha256":EXPECTED["manifest"],"fine_I_sha256":EXPECTED["fineI"],"barrier_counts":barrier_counts,
  "attempts":attempts,"elapsed_s":time.monotonic()-started,"physical_admission":False,"runtime":0,"accepted_steps":0}
 (OUT/"result.json").write_text(json.dumps(result,indent=2)+"\n")
 print(json.dumps({"result":str(OUT/"result.json"),"attempts":attempts,"elapsed_s":result["elapsed_s"]},indent=2),flush=True)
if __name__=="__main__":main()
