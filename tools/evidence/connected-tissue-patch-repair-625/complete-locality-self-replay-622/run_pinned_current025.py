from pathlib import Path
import sys,importlib.util,json,hashlib,time,argparse
import numpy as np
from stable_face_self_replay import stable_face_ids,persistent_records,report_pairs_to_rows
P=Path(__file__).resolve().parent.parent
F=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024"
BASE=F/"closed603-merge-025/candidate-f32-linked-remesh-typed.npz"
REPORT=F/"full-native-self-028/self.json"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def owner():
 p=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/D-witness-contract-001/run-042-abstract-cycle-coupled-remesh/remesh.py"
 s=importlib.util.spec_from_file_location("ears622",p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
 pred=m.load_exact_owner()
 if sha(pred.__file__)!="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d":raise ValueError("exact owner changed")
 return pred
def main():
 a=argparse.ArgumentParser();a.add_argument("candidate",type=Path);a.add_argument("output",type=Path);args=a.parse_args()
 if args.output.exists():raise ValueError("do not overwrite audit")
 if sha(BASE)!="9790ed1fa49bacff36ea169c7b8354344e40f1df33f604c006bcfb7991723a2a":raise ValueError("baseline changed")
 if sha(REPORT)!="f932dac6f102958f9fd0a734f61a5d2cdaf89f4a9de51243a723172c5cf6a5b0":raise ValueError("full baseline report changed")
 before_sha=sha(args.candidate);old=np.load(BASE);new=np.load(args.candidate)
 x0=np.asarray(old["points_world_m"]);x=np.asarray(new["points_world_m"])
 if x0.dtype!=np.float32 or x.dtype!=np.float32 or x0.shape!=x.shape or x0.tobytes()!=x.tobytes():raise ValueError("locality replay forbids any coordinate or vertex-domain edit")
 f0=np.asarray(old["faces"]);f=np.asarray(new["faces"])
 full=json.loads(REPORT.read_text())
 if full.get("complete") is not True or full.get("exact_triangle_pair_count")!=237 or full.get("candidate_sha256")!=sha(BASE):raise ValueError("incomplete or mismatched full baseline")
 pairs=tuple(tuple(map(int,p))for p in full["triangle_pairs"])
 if len(pairs)!=237 or len(set(pairs))!=237 or any(not(0<=a<b<len(f0))for a,b in pairs):raise ValueError("invalid full baseline pairs")
 pred=owner();t=time.monotonic()
 r0,e0=pred.float32_surface_records(x0,f0,source_seam_groups=())
 r1,e1=pred.float32_surface_records(x,f,source_seam_groups=())
 if e0!=e1:raise ValueError("shared-identity policy changed")
 stable=stable_face_ids(f0,f)
 baseline=pred._CompleteSelfPairBaseline(pred._immutable_self_pair_records(r0),pairs)
 updated,report=pred._update_complete_self_pair_baseline(baseline,persistent_records(r1,stable))
 report["triangle_pairs_persistent_ids"]=report["triangle_pairs"]
 report["triangle_pairs"]=report_pairs_to_rows(report["triangle_pairs"],stable)
 if sha(args.candidate)!=before_sha:raise ValueError("candidate changed during audit")
 result={"status":"PASS_SELF_GATE"if not report["count"]else"FAILED_SELF_GATE",
   "candidate_path":str(args.candidate),"candidate_sha256":before_sha,"baseline_path":str(BASE),
   "baseline_sha256":sha(BASE),"baseline_full_report_path":str(REPORT),"baseline_full_report_sha256":sha(REPORT),
   "exact_owner_path":str(pred.__file__),"exact_owner_sha256":sha(pred.__file__),
   "method":"existing exact owner complete-locality update; reused unchanged pairs from pinned full scan; not a second full scan",
   "complete":True,"exact_triangle_pair_count":report["count"],"self":report,
   "elapsed_s":time.monotonic()-t,"physical_admission":False,"native_motion":"NOT_RUN"}
 args.output.mkdir(parents=True)
 (args.output/"report.json").write_text(json.dumps(result,indent=2))
 np.savez_compressed(args.output/"persistent-face-row-map.npz",current_row_to_persistent_face_id=np.asarray(stable,dtype=np.int64))
 print(json.dumps({k:result[k]for k in["status","exact_triangle_pair_count","elapsed_s"]}),flush=True)
if __name__=="__main__":main()
