#!/usr/bin/env python3
import hashlib,importlib.util,json,os,resource,sys,time
from pathlib import Path
import numpy as np
for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[n]="1"
BASE=Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/stable63-64-self-topology-audit-1217-attempt2")
PREV=BASE/"result/report.json"
PREV_SHA="38b498e8f356e3dbca9e7a759af6d9a6569f70c632a7db38693961288b9869a3"
SCRIPT=BASE/"stable64-outer-envelope-supplement-001/audit_stable64_outer.py"
DECL=BASE/"stable64-outer-envelope-supplement-001/audit-declaration.json"
OUT=BASE/"stable64-outer-envelope-supplement-001"
EXPECTED_TISS="1cd0c3d5bd1ff6d163f6544be4d0fa169e8bf696bf6c729ef0b18879b0856c48"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def require(c,m):
 if not c:raise RuntimeError(m)
def main():
 start=time.monotonic(); cpu0=time.process_time()
 require(sha(PREV)==PREV_SHA,"completed self/topology report changed")
 prev=json.loads(PREV.read_text())
 require(prev["inputs_unchanged"] is True and prev["candidate"]["payload_sha256"]==EXPECTED_TISS,"prior audit not clean/current")
 require(not (OUT/"report.json").exists(),"supplement report already exists")
 expected_inputs=prev["input_hashes_after"]
 inputs_before={p:sha(p) for p in expected_inputs}
 require(inputs_before==expected_inputs,"one or more prior frozen run/source inputs changed")
 spec=importlib.util.spec_from_file_location("_stable64_outer_audit_base",BASE/"audit_stable63_64.py")
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
 ci,clearance=mod.load_predicates()
 source=mod.source_rows()[64]
 results=[]
 for item in prev["captures"]:
  step=int(item["step"])
  pack=Path(item["pack_path"])
  require(sha(pack)==item["pack_sha256"],f"step {step} accepted pack changed")
  positions,surfaces,counts=mod.pack_row_vertices_faces(pack,{(51005,64)})
  faces_global=np.asarray(surfaces[(51005,64)]["faces"],dtype=np.int64)
  ids=np.unique(faces_global);base=int(ids[0])
  require(len(ids)==source["vertex_count"] and np.array_equal(ids,np.arange(base,base+len(ids))),f"step {step} local vertex order is not exact")
  local_faces=faces_global-base
  require(np.array_equal(local_faces,source["faces"]),f"step {step} captured local faces differ from final NHTISS")
  xyz=np.asarray(positions[base:base+len(ids)],dtype="<f4")
  target=clearance._prepare_closed_clearance_target(xyz,local_faces,allow_nested_enclosure=True)
  report=target["report"]
  components=report["component_face_rows"]
  pairproofs=report["component_pair_proofs"]
  eligible=(report.get("embedded_closed_target") is True
            and report.get("external_skin_enclosure_only") is True
            and report.get("all_components_closed_oriented_unused_free") is True
            and report.get("self_intersection_audit",{}).get("count")==0
            and report.get("face_count")==6690
            and report.get("outer_component_face_count")+sum(x["face_count"] for x in components if x["component_id"]!=report.get("outer_component_id"))==6690
            and len(pairproofs)==3
            and all(x.get("cross_component_intersection_count")==0 for x in pairproofs)
            and all(x.get("relation") in ("first_contains_second","second_contains_first","disjoint") for x in pairproofs))
  results.append({"accepted_step":step,"accepted_time_s":step*0.002,"pack_path":str(pack),"pack_sha256":item["pack_sha256"],
   "receipt_path":item["receipt_path"],"receipt_sha256":item["receipt_sha256"],
   "stable_id":64,"label":"left vastus lateralis","faces_match_candidate_exactly":True,"vertex_count":len(ids),"face_count":len(local_faces),
   "allow_nested_enclosure":True,"eligible_as_external_skin_enclosure":bool(eligible),"helper_report":report})
 inputs_after={p:sha(p) for p in expected_inputs}
 unchanged=inputs_before==inputs_after
 require(unchanged,"prior pinned inputs changed during supplement")
 wall=time.monotonic()-start;cpu=time.process_time()-cpu0
 result={"schema":"numi.human.stable64-outer-envelope-supplement.v1",
  "status":"all_three_captures_unique_outer_envelope_eligible" if unchanged and all(x["eligible_as_external_skin_enclosure"] for x in results) else "outer_envelope_failed_or_input_changed",
  "prior_audit_report":{"path":str(PREV),"sha256":PREV_SHA},
  "candidate_payload_sha256":EXPECTED_TISS,"candidate_manifest_sha256":"f052eff736f040f2f6ca1a1a99d2939ca0ed5c0bead055dfb39d3141f05a3cac",
  "target":"NHTISS stable_id 64, left vastus lateralis; full 6690 face surface retained in report and helper input",
  "policy":"Explicit allow_nested_enclosure=True only for the external-skin enclosure interpretation. No component faces are removed. This does not label inner components as cavities or disposable structures.",
  "inputs_before":inputs_before,"inputs_after":inputs_after,"inputs_unchanged":unchanged,
  "script_path":str(SCRIPT),"script_sha256":sha(SCRIPT),"declaration_path":str(DECL),"declaration_sha256":sha(DECL),
  "elapsed_wall_seconds":wall,"elapsed_process_cpu_seconds":cpu,"max_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  "captures":results}
 out=OUT/"report.json";out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"status":result["status"],"report":str(out),"sha256":sha(out),"wall_s":wall,"cpu_s":cpu,"steps":[{"step":x["accepted_step"],"eligible":x["eligible_as_external_skin_enclosure"],"outer_component":x["helper_report"].get("outer_component_id"),"outer_faces":x["helper_report"].get("outer_component_face_count"),"components":[{"id":c["component_id"],"faces":c["face_count"]} for c in x["helper_report"]["component_face_rows"]],"relations":[p["relation"] for p in x["helper_report"]["component_pair_proofs"]]} for x in results]},sort_keys=True))
 return 0
if __name__=="__main__":raise SystemExit(main())
