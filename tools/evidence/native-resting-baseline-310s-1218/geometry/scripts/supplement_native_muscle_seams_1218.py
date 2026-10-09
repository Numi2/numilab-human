#!/usr/bin/env python3
"""Apply the existing source-bound seam contract to retained raw muscle audits."""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path
B=Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/native-accepted-geometry-audit-001")
SCRIPT=B/"audit_native_candidate_1218_root003.py"
SCRIPT_SHA="246b9c28eeb4c80ded2271dfa3cd340e63ee059568052416748d5c7791761568"
PARENTS = {19999: {'directory': 'early-40s-002', 'sha256': '84ee3729387e33b30d55b41a6e58a552ea67d2c4ca96b8d44928c2d8ae6fc920'}, 47519: {'directory': 'early-95s-001', 'sha256': '4cfdfb841ebd2afabb0bf4d29809b4f0479ff6796248644e79481f17d88e5ee8'}}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def need(x,m):
 if not x:raise ValueError(m)
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+"\n")
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("--out",type=Path,required=True)
 a=parser.parse_args();out=a.out.resolve()
 need(out.parent==B and not out.exists(),"fresh direct audit-root child required")
 need(sha(SCRIPT)==SCRIPT_SHA,"audit script changed")
 spec=importlib.util.spec_from_file_location("_native1218_source_seam_audit",SCRIPT)
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 _,owner,ci,cap,val,core,outer,codec,owner_files=m.load_owners()
 static,source_rows=m.validate_candidate(codec)
 tracked={str(p.resolve()):h for p,h in static.items()}
 tracked.update({str(p.resolve()):h for p,h in owner_files.items()})
 tracked[str(SCRIPT)]=SCRIPT_SHA;tracked[str(Path(__file__).resolve())]=sha(__file__)
 parents={}
 for step,info in PARENTS.items():
  p=B/info["directory"]/"summary.json"
  need(sha(p)==info["sha256"],"retained summary changed")
  doc=json.loads(p.read_text())
  need(doc["input_hashes_unchanged"] and doc["capture_steps_requested"]==[step],"parent audit incomplete or changed")
  rows=[r for r in doc["pose_results"] if r["accepted_step"]==step]
  need(len(rows)==1 and rows[0]["pair_coverage_complete"],"full skin audit missing")
  tracked.update(doc["input_hashes_before"]);tracked[str(p)]=info["sha256"]
  parents[step]=(p,rows[0])
 before={p:sha(p) for p in tracked}
 need(before==tracked,"retained input pin changed")
 out.mkdir();start=time.monotonic()
 write(out/"declaration.json",{"schema":"numi.human.native-muscle-raw-self-seam-diagnosis.v1",
  "status":"running_source_seam_contract_supplement","steps":list(PARENTS),
  "parent_skin_audits_preserved":{str(s):str(p) for s,(p,r) in parents.items()},
  "audit_script_sha256":SCRIPT_SHA,"inputs_before":before})
 results=[]
 for step,(p,old) in parents.items():
  r=m.audit_muscles(step,m.RUN,out,owner,ci,cap,val,core,outer,source_rows,m.NHA_SHA)
  oldm=old["muscle_geometry"]
  need(r["capture_sha256"]==oldm["capture_sha256"] and r["receipt_sha256"]==oldm["receipt_sha256"],"capture identity differs")
  for now,prev in zip(r["surfaces"],oldm["surfaces"]):
   need(now["surface"]==prev["surface"] and now["raw_indexed_contact_pair_count"]==prev["self_intersection_count"]
    and now["quotient_self_intersection_count"]==prev["quotient_self_intersection_count"]
    and now["face_count"]==prev["face_count"],"raw or quotient predicate result differs")
  results.append({"accepted_step":step,"parent_skin_summary":str(p),
   "skin_target_pairs_unchanged":old["all_skin_crossing_pair_count"],
   "skin_self_pairs_unchanged":old["skin_self_crossing_pair_count"],"muscle_geometry":r})
 after={p:sha(p) for p in tracked};need(after==before,"inputs changed")
 report={"schema":"numi.human.native-muscle-raw-self-seam-diagnosis.v1",
  "status":"complete_source_seam_contract_supplement","steps":list(PARENTS),
  "pose_results":results,"all_muscle_self_and_enclosure_gates_clear":all(r["muscle_geometry"]["all_muscle_self_zero"]
    and r["muscle_geometry"]["all_outer_envelopes_clear"] and r["muscle_geometry"]["stable_63_64_cross_pair_count"]==0 for r in results),
  "inputs_before":before,"inputs_after":after,"inputs_unchanged":True,"elapsed_wall_seconds":time.monotonic()-start,
  "qualification":"Muscle source-seam classification only. All raw findings and all full-skin audit findings retained. No full-run anatomical admission."}
 write(out/"summary.json",report)
 print(json.dumps({"status":report["status"],"muscle_gates_clear":report["all_muscle_self_and_enclosure_gates_clear"],
  "elapsed_wall_seconds":report["elapsed_wall_seconds"]},sort_keys=True))
if __name__=="__main__":main()
