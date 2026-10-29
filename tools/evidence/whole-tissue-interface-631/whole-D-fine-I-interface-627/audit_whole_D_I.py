from pathlib import Path
import sys,json,hashlib,time,signal,importlib.util
import numpy as np
P=Path(__file__).resolve().parent.parent
D=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/full-native-self-028/current-collar-plan-032/engine-20-ops-one-sided624-run-006/candidate-f32-linked-remesh-typed.npz"
I=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz"
OUT=Path(__file__).resolve().parent/"run-001"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False);t=time.monotonic()
 expected={D:"c17794a1012f76924dcd9e4e754bb026c774deb5a3d37e2f1c79635ccfe9dc83",I:"5d46ee99df2613e9f30a43075af95a6402beffe26392cd9f017d8c1a154bf410"}
 for p,h in expected.items():
  if sha(p)!=h:raise ValueError("input changed "+str(p))
 loader=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py"
 s=importlib.util.spec_from_file_location("loader627",loader);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);pred=m.load_exact_owner()
 if sha(pred.__file__)!="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d":raise ValueError("exact owner changed")
 def progress(stage,**data):
  r={"stage":stage,"elapsed_s":time.monotonic()-t,**data}
  (OUT/"progress.json").write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
 def expired(*_):raise TimeoutError("bounded whole-interface audit exceeded 180 seconds")
 signal.signal(signal.SIGALRM,expired);signal.alarm(180)
 try:
  with np.load(D)as z:dx=z["points_world_m"];df=z["faces"]
  with np.load(I)as z:ix=z["points_world_m"];iff=z["faces"]
  progress("building complete original-row records",D_faces=len(df),I_faces=len(iff))
  dr,_=pred.float32_surface_records(dx,df,source_seam_groups=())
  ir,_=pred.float32_surface_records(ix,iff,source_seam_groups=())
  index=pred._prepare_surface_aabb(dr)
  progress("auditing complete D versus complete open I")
  def pairs():
   count=0
   for second,first in pred._query_surface_aabb(ir,index):
    count+=1
    if count%100000==0:progress("narrow phase",aabb_candidate_pairs_seen=count)
    yield first,second
  report=pred._audit_record_pairs(pairs(),same_surface=False)
  for p,h in expected.items():
   if sha(p)!=h:raise ValueError("input changed during audit")
  np.savez_compressed(OUT/"witness-pairs.npz",D_I_face_rows=np.asarray(report["triangle_pairs"],dtype=np.int64).reshape(-1,2))
  result={"status":"PASS_SURFACE_PAIR_GATE"if report["count"]==0 else"FAILED_SURFACE_PAIR_GATE",
    "complete":True,"D_path":str(D),"D_sha256":expected[D],"I_path":str(I),"I_sha256":expected[I],
    "D_faces":len(df),"I_faces":len(iff),"exact_owner_sha256":sha(pred.__file__),
    "scope":"complete open D versus complete open fine-I; no closure caps; surface intersection only, not inside/outside",
    "exact_triangle_pair_count":report["count"],"aabb_candidate_pairs":report["aabb_candidate_pairs"],
    "triangle_pairs":report["triangle_pairs"],"elapsed_s":time.monotonic()-t,"physical_admission":False,"runtime":0,"accepted_steps":0}
 except Exception as exc:
  result={"status":"INCOMPLETE","complete":False,"error":repr(exc),"elapsed_s":time.monotonic()-t,"physical_admission":False}
 finally:signal.alarm(0)
 (OUT/"report.json").write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k!="triangle_pairs"}),flush=True)
 if not result.get("complete"):raise SystemExit(2)
if __name__=="__main__":main()
