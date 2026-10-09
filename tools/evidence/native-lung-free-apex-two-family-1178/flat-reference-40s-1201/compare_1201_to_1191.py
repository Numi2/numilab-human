#!/usr/bin/env python3
"""Read-only post-run CSV/physiology summary for the closed 1201 control."""
import csv, hashlib, io, json, math, sys
from pathlib import Path

BASE = Path("/Users/n/numi-human-retained-delivery-20261009/native-lung1178-thumb1187-smoke-1191")
CANDIDATE = Path("/Users/n/numi-human-retained-delivery-20261009/native-flat-reference-40s-1201")
BASE_RUN = BASE / "native-run"
RUN = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else CANDIDATE / "native-run"
CUT = 10000
DECL = "c907c5ea7d1ed6027da60f27a0b07f498fd6499989b9051349688241b5656085"
BASE_DECL = "e9f7ada8ea4eb52760a790edb3fb3a3c05953c5ad06fac3f605b943b6405f05a"
BASE_META = "eb84c9d25714d2e6f163be664c26d1e30708e97a03f2d7dfaf0c22e4d4347069"
BASE_LOG = "40a2aa81b8635a20b49d94d70b174efcdb8182936b70202ee110345517ddefba"
BASE_INV = "d085fd4604033f9de4fea9819218a8773350cde55c4e6254f120719952ddb995"
CSV_PINS = {
 "resting-com-momentum-diagnostic.csv":"be8708957990c08b357e54b28ca070ef93f0495183db57c7adf6041a7f80de32",
 "resting-com-support-impulses.csv":"82eeb9dfb1336a7e576df87cd6305679e673dd1759d315c12de8d0d16148480c",
 "resting-coupled.csv":"5c4f129103fbf23010cfe47b99367c30d8497f70e525b4c97ca61b53c3ca90e6",
 "resting-surface-audit.csv":"8076b081e33b85a17dd07a5322e8b1625ecaab57f3dd5adf113253b254c1cb5e",
}
STEP_KEY = {
 "resting-com-momentum-diagnostic.csv":"accepted_step",
 "resting-com-support-impulses.csv":"accepted_step",
 "resting-coupled.csv":"step",
 "resting-surface-audit.csv":"step",
}
RANGE_COLS = ("lung_volume_ml","airflow_ml_s","alveolar_pa","pleural_pa","diaphragm_mm","rib_mm","PaO2_mmhg","PaCO2_mmhg","SaO2","tidal_ml","lv_mmhg","rv_mmhg","aorta_mmhg","pulmonary_artery_mmhg","lv_ml","rv_ml","blood_ml","aortic_ejected_ml","pulmonary_ejected_ml")
RESIDUAL_COLS = ("oxygen_balance_error_stpd_ml","co2_balance_error_stpd_ml","respiratory_net_volume_ml","respiratory_volume_balance_ml","blood_error_ml","blood_continuity_residual_accum_ml","blood_physical_delta_accum_ml","blood_residual_minus_physical_ml","blood_endpoint_minus_physical_ml")

def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for block in iter(lambda:f.read(1<<20), b""): h.update(block)
 return h.hexdigest()

def pin(path):
 path=Path(path); return {"path":str(path),"sha256":sha(path),"bytes":path.stat().st_size}

def checked(path, expected):
 actual=pin(path)
 if actual["sha256"] != expected: raise RuntimeError("pinned input changed: "+str(path))
 return actual

def csv_data(path):
 raw=Path(path).read_bytes(); lines=raw.splitlines(keepends=True)
 parsed=list(csv.reader(io.StringIO(raw.decode("utf-8"),newline="")))
 if not parsed or len(parsed)!=len(lines) or any(len(r)!=len(parsed[0]) for r in parsed[1:]):
  raise RuntimeError("empty/multiline/malformed CSV: "+str(path))
 return parsed[0],parsed[1:],lines

def prefix(path, key):
 header, rows, lines=csv_data(path)
 if key not in header: raise RuntimeError(f"{path.name}: missing {key}")
 ix=header.index(key); out=[]; raw=[lines[0]]; steps=[]; passed=False
 for row,line in zip(rows,lines[1:]):
  step=int(row[ix])
  if steps and step < steps[-1]: raise RuntimeError("nonmonotone step in "+str(path))
  steps.append(step)
  if step<=CUT:
   if passed: raise RuntimeError("step returned below cutoff in "+str(path))
   out.append(row); raw.append(line)
  else: passed=True
 return header,out,raw,len(rows),steps[-1] if steps else None

def stats(rows, name):
 if not rows or name not in rows[0]: raise RuntimeError("missing coupled column "+name)
 values=[]
 for i,row in enumerate(rows):
  x=float(row[name])
  if not math.isfinite(x): raise RuntimeError(f"nonfinite {name} row {i}")
  values.append(x)
 return {"min":min(values),"max":max(values),"terminal":values[-1],"max_abs":max(abs(x) for x in values)}

def log_report(path):
 lines=Path(path).read_text(errors="replace").splitlines()
 done=[x for x in lines if x.startswith("resting_integrated_body=completed ")]
 rejected=[x for x in lines if x.startswith("resting_integrated_rejection=")]
 terminal=[x for x in lines if x.startswith("stand_terminal_state=")]
 if len(done)!=1 or len(rejected)!=1 or len(terminal)!=1: raise RuntimeError("unique completion/probe/terminal log records not found")
 def fields(line):
  d={}
  for token in line.split():
   if "=" in token:
    k,v=token.split("=",1); d[k]=v
  return d
 d=fields(done[0]); r=fields(rejected[0]); t=json.loads(terminal[0].split("=",1)[1])
 if not rejected[0].startswith("resting_integrated_rejection=pass "): raise RuntimeError("rejection/replay probe failed")
 if d.get("root_assistance")!="false": raise RuntimeError("completion log does not report root_assistance=false")
 return {
  "integrated_line":done[0],
  "integrated":{"simulated_s":float(d["simulated_s"]),"internal_wall_s":float(d["wall_s"]),"real_time_factor":float(d["real_time_factor"]),"physiology_body_clock":d.get("physiology_body_clock"),"root_assistance":d["root_assistance"],"presentation_qualification":d.get("presentation_qualification")},
  "terminal":{"schema":t.get("schema"),"step_count":t.get("step_count"),"timestep_seconds":t.get("timestep_seconds"),"root_assistance":t.get("root_assistance"),"q_count":len(t.get("q",[]))},
  "rejected_step_replay_probe":{"raw_line":rejected[0],"status":r.get("resting_integrated_rejection"),"rejected_after_accepted_predecessor":r.get("rejected_after_accepted_predecessor"),"body_q_v_root_myo_unchanged":r.get("body_q_v_root_myo_unchanged"),"circulation_and_clock_unchanged":r.get("circulation_and_clock_unchanged"),"respiration_brain_history_unchanged":r.get("respiration_brain_history_unchanged"),"retry_matches_uninterrupted_replay":r.get("retry_matches_uninterrupted_replay"),"rejected_step":r.get("rejected_step"),"rejected_global_step":r.get("rejected_global_step"),"all_owners_match_uninterrupted_prefix":r.get("all_owners_match_uninterrupted_prefix"),"retry":r.get("retry")}
 }

def main():
 bp=BASE/"run-declaration.json"; cp=CANDIDATE/"run-declaration.json"
 bmeta=BASE_RUN/"run-metadata.json"; cmeta=RUN/"run-metadata.json"
 blog=BASE_RUN/"native.log"; clog=RUN/"native.log"; binv=BASE_RUN/"invocation.json"
 inputs=[checked(bp,BASE_DECL),checked(bmeta,BASE_META),checked(blog,BASE_LOG),checked(binv,BASE_INV)]
 bd=json.loads(bp.read_text()); cd=json.loads(cp.read_text())
 if sha(cp)!=DECL: raise RuntimeError("1201 declaration hash changed")
 if (cd.get("seconds"),cd.get("dt"),cd.get("capture_steps"))!=(40,0.002,[0,20000]): raise RuntimeError("1201 duration/dt/capture declaration changed")
 ba=bd.get("immutable_assets",{}); ca=cd.get("immutable_assets",{})
 if len(ba)!=25 or any(ca.get(p)!=h for p,h in ba.items()): raise RuntimeError("candidate does not preserve all 25 original asset pins")
 extras=sorted(set(ca)-set(ba))
 if extras!=sorted({str(BASE/"run.py"),str(BASE/"run-declaration.json")}): raise RuntimeError("unexpected candidate declaration extra pins")
 inputs.append(pin(cp))
 bm=json.loads(bmeta.read_text()); cm=json.loads(cmeta.read_text())
 if bm.get("exit_code")!=0 or cm.get("exit_code")!=0: raise RuntimeError("wrapper exit code is not zero")
 if cm.get("source_files_changed_during_run") not in ([],None): raise RuntimeError("source files changed during candidate run")
 if any(cm.get("asset_sha256",{}).get(p)!=h for p,h in ba.items()): raise RuntimeError("candidate metadata fails to confirm unchanged 25 input asset hashes")
 for p in (bmeta,cmeta,blog,clog,binv):
  inputs.append(pin(p))
 ci=RUN/"invocation.json"
 if ci.exists(): inputs.append(pin(ci))
 lg=log_report(clog)
 if lg["terminal"]["step_count"]!=20000 or lg["terminal"]["timestep_seconds"]!=0.002 or lg["terminal"]["root_assistance"] is not False or abs(lg["integrated"]["simulated_s"]-40.0)>.002:
  raise RuntimeError("candidate terminal proof does not match 40s/step20000/dt.002")
 csvs={}
 for name,key in STEP_KEY.items():
  base=BASE_RUN/name; cand=RUN/name
  if sha(base)!=CSV_PINS[name]: raise RuntimeError("1191 CSV hash changed: "+name)
  h1,r1,b1,n1,last1=prefix(base,key); h2,r2,b2,n2,last2=prefix(cand,key)
  if h1!=h2 or last1!=CUT: raise RuntimeError("baseline prefix header/coverage mismatch: "+name)
  if not r2: raise RuntimeError("candidate prefix empty: "+name)
  k=h2.index(key); candidate_last=int(r2[-1][k])
  if candidate_last>CUT or candidate_last<CUT-32: raise RuntimeError("candidate lacks expected near-20s observer coverage: "+name)
  common1=[row for row in r1 if int(row[k])<=candidate_last]
  common1_bytes=[b1[0]]+[line for row,line in zip(r1,b1[1:]) if int(row[k])<=candidate_last]
  be=(common1_bytes==b2); fe=(common1==r2)
  baseline_only_steps=[int(row[k]) for row in r1 if int(row[k])>candidate_last]
  common_step_set={int(x[k]) for x in common1}
  candidate_only_steps=[int(row[k]) for row in r2 if int(row[k]) not in common_step_set]
  csvs[name]={"step_key":key,"header":h1,"baseline_full_sha256":sha(base),"candidate_full_sha256":sha(cand),"baseline_common_prefix_sha256":hashlib.sha256(b"".join(common1_bytes)).hexdigest(),"candidate_prefix_sha256":hashlib.sha256(b"".join(b2)).hexdigest(),"baseline_rows_through_10000":len(r1),"candidate_rows_through_10000":len(r2),"common_sample_rows":len(common1),"common_sample_last_step":candidate_last,"baseline_terminal_only_rows_after_common_last_step":baseline_only_steps,"candidate_only_steps":candidate_only_steps,"serialized_common_prefix_bytes_exact":be,"fieldwise_common_prefix_exact":fe,"candidate_total_rows":n2,"candidate_final_step":last2}
  inputs.extend((pin(base),pin(cand)))
  if not be or not fe: raise RuntimeError("common first-20-second serialized/field rows differ: "+name)
 with (RUN/"resting-coupled.csv").open(newline="") as f:
  reader=csv.DictReader(f); rows=list(reader); headers=reader.fieldnames
 if not rows or int(rows[-1]["step"])!=20000: raise RuntimeError("coupled trace lacks terminal step20000")
 ranges={x:stats(rows,x) for x in RANGE_COLS}
 residuals={x:stats(rows,x) for x in RESIDUAL_COLS}
 assist={x:stats(rows,x) for x in ("root_assistance_n","root_assistance_nm")}
 breaths=stats(rows,"breaths"); cycles=stats(rows,"complete_filling_ejection_cycles")
 phys={"coupled_columns":headers,"coupled_trace_rows":len(rows),"first_step":int(rows[0]["step"]),"terminal_step":int(rows[-1]["step"]),"complete_breaths_terminal":int(round(breaths["terminal"])),"complete_breaths_max_observed":int(round(breaths["max"])),"complete_filling_ejection_cycles_terminal":int(round(cycles["terminal"])),"complete_filling_ejection_cycles_max_observed":int(round(cycles["max"])),"selected_physiology_ranges":ranges,"gas_and_blood_volume_residuals":residuals,"root_assistance_csv":assist,"root_assistance_zero_on_every_coupled_row":all(v["min"]==0 and v["max"]==0 for v in assist.values()),"terminal_coupled_state":rows[-1]}
 wrapper=cm.get("wall_seconds")
 if wrapper is None: raise RuntimeError("run-metadata lacks wrapper wall_seconds")
 report={"scope":"Closed 40-second flat-reference native run; exact first-20-second observer CSV comparison with retained 1191; bounded CPU post-run summary.","status":"pass","candidate_run":str(RUN),"candidate_wrapper_exit_code":cm["exit_code"],"wrapper_wall_s":wrapper,"internal_integrated_body":lg["integrated"],"accepted_terminal":lg["terminal"],"rejected_step_and_replay_probe":lg["rejected_step_replay_probe"],"first_20_second_prefix_comparison":{"baseline_run":str(BASE_RUN),"cutoff_step":CUT,"nominal_cutoff_time_s":20.0,"baseline_capture_steps":bd.get("capture_steps"),"candidate_capture_steps":cd.get("capture_steps"),"all_four_common_sample_prefixes_exact":all(v["serialized_common_prefix_bytes_exact"] and v["fieldwise_common_prefix_exact"] for v in csvs.values()),"csvs":csvs},"physiology_40s":phys,"physical_input_asset_pins":{"baseline_asset_count":len(ba),"all_25_baseline_assets_confirmed_by_candidate_declaration_and_metadata":True,"additional_candidate_declaration_pins":extras},"input_pins":inputs,"limits":["Exact CSV comparison covers steps through 10000 only and does not determine the entire 40-second anatomy result.","MRVPACK capture schedules differ ([0,4991,5375,5759,6111,6495,7743,10000] vs [0,20000]); exact shared observer rows test whether that schedule perturbed physics at sampled states through 20 seconds.","Wrapper wall includes setup and concurrent CPU build/test activity; do not interpret it as a performance benchmark.","This 40-second flat-reference control is not a 310-second/full-cycle qualification."]}
 report["analysis_script"]=pin(Path(__file__))
 dest=RUN.parent/"cpu-comparison-1191.json"
 tmp=dest.with_suffix(".json.tmp"); tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); tmp.replace(dest)
 print(json.dumps({"status":report["status"],"report":str(dest),"prefixes":{k:{"rows":v["common_sample_rows"],"exact":v["serialized_common_prefix_bytes_exact"]} for k,v in csvs.items()},"breaths":phys["complete_breaths_terminal"],"filling_ejection_cycles":phys["complete_filling_ejection_cycles_terminal"],"wrapper_wall_s":wrapper,"internal_rtf":lg["integrated"]["real_time_factor"]},indent=2,sort_keys=True))

if __name__=="__main__": main()
