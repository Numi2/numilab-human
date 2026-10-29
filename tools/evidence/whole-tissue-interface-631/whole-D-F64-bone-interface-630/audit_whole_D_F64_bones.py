from pathlib import Path
import sys,json,hashlib,time,signal,importlib.util
import numpy as np
P=Path(__file__).resolve().parent.parent
D=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/full-native-self-028/current-collar-plan-032/engine-20-ops-one-sided624-run-006/candidate-f32-linked-remesh-typed.npz"
M=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/exact-set-union-remaining-013/all185-bank-operand-manifest.json"
OUT=Path(__file__).resolve().parent/"run-001"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
 OUT.mkdir(exist_ok=False);t=time.monotonic()
 if sha(D)!="c17794a1012f76924dcd9e4e754bb026c774deb5a3d37e2f1c79635ccfe9dc83"or sha(M)!="0e56a99c9f85be5995480fc2b150cedf000d3df40862732a5e32fa1d3040398c":raise ValueError("input pin mismatch")
 pred=load("loader630",P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py").load_exact_owner()
 if sha(pred.__file__)!="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d":raise ValueError("exact owner changed")
 helper=P/"bone-barrier-remeshing-609/bone_barrier.py";helper_sha=sha(helper);bone=load("bone630",helper)
 operands=json.loads(M.read_text())["operands"]
 if len(operands)!=185 or len({(o["body"],o["bone"])for o in operands})!=185 or len({o["index"]for o in operands})!=185:raise ValueError("incomplete bank inventory")
 pins={str(D):sha(D),str(M):sha(M),str(helper):helper_sha}
 for op in operands:
  if sha(op["operand_path"])!=op["operand_sha256"]:raise ValueError("operand changed")
  pins[op["operand_path"]]=op["operand_sha256"]
 (OUT/"input-pins.json").write_text(json.dumps(pins,indent=2))
 def progress(stage,**data):
  r={"stage":stage,"elapsed_s":time.monotonic()-t,**data};(OUT/"progress.json").write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
 def expired(*_):raise TimeoutError("bounded F64 interface query exceeded 180 seconds")
 signal.signal(signal.SIGALRM,expired);signal.alarm(180)
 completed=[];witness=[]
 try:
  with np.load(D)as z:x=z["points_world_m"];faces=z["faces"]
  progress("building complete D index",D_faces=len(faces))
  dr,_=pred.float32_surface_records(x,faces,source_seam_groups=());index=pred._prepare_surface_aabb(dr)
  for oi,op in enumerate(operands):
   with np.load(op["operand_path"])as z:xx=z["points_world_m_f64"];ff=z["faces_f64"]
   if xx.dtype!=np.float64 or len(ff)!=op["face_count"]:raise ValueError("F64 operand domain mismatch")
   keys=tuple(bone.exact_common_lattice_key(v)for v in xx)
   records=pred._records(keys,ff)
   def pairs():
    for second,first in pred._query_surface_aabb(records,index):yield first,second
   report=pred._audit_record_pairs(pairs(),same_surface=False)
   completed.append({"manifest_index":op["index"],"body":op["body"],"bone":op["bone"],"face_count":len(ff),"exact_triangle_pair_count":report["count"],"aabb_candidate_pairs":report["aabb_candidate_pairs"]})
   witness.extend((op["index"],a,b)for a,b in report["triangle_pairs"])
   if (oi+1)%20==0:progress("completed operand queries",banks_complete=oi+1,contact_pairs=len(witness))
  for path,h in pins.items():
   if sha(path)!=h:raise ValueError("input changed during audit")
  np.savez_compressed(OUT/"witness-pairs.npz",bank_index_D_row_bone_row=np.asarray(witness,dtype=np.int64).reshape(-1,3))
  result={"status":"PASS_SURFACE_PAIR_GATE"if not witness else"SURFACE_CONTACTS_REQUIRE_CLASSIFICATION","complete":True,
   "candidate_path":str(D),"candidate_sha256":pins[str(D)],"manifest_path":str(M),"manifest_sha256":pins[str(M)],
   "scope":"complete open D vs all185 original Float64 operands on exact common lattice; no Float32 rounding of bone operands; surface pair gate not closed solid classification",
   "exact_triangle_pair_count":len(witness),"aabb_candidate_pairs":sum(r["aabb_candidate_pairs"]for r in completed),"banks_complete":len(completed),
   "banks":completed,"all_pairs_archive":"witness-pairs.npz","elapsed_s":time.monotonic()-t,"physical_admission":False,"runtime":0,"accepted_steps":0}
 except Exception as exc:
  result={"status":"INCOMPLETE","complete":False,"error":repr(exc),"banks_complete":len(completed),"banks":completed,"elapsed_s":time.monotonic()-t,"physical_admission":False}
 finally:signal.alarm(0)
 (OUT/"report.json").write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items()if k!="banks"}),flush=True)
 if not result.get("complete"):raise SystemExit(2)
if __name__=="__main__":main()
