#!/usr/bin/env python3
"""Read-only exact-trace comparison for passive NHTISS midpoint run 1217 attempt2."""
import csv, hashlib, io, json, math
from pathlib import Path

BASE = Path("/Users/n/numi-human-retained-delivery-20261009/native-flat-reference-40s-1201")
CANDIDATE = Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/native-refinement-1217-attempt2")
BR = BASE / "native-run"
CR = CANDIDATE / "native-run"
OUT = CANDIDATE / "review-001"
CSV_NAMES = ("resting-coupled.csv", "resting-com-momentum-diagnostic.csv", "resting-com-support-impulses.csv", "resting-surface-audit.csv")
STEP_KEYS = {"resting-coupled.csv":"step", "resting-com-momentum-diagnostic.csv":"accepted_step", "resting-com-support-impulses.csv":"accepted_step", "resting-surface-audit.csv":"step"}


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha(path): return sha_bytes(Path(path).read_bytes())
def pin(path):
 p=Path(path); b=p.read_bytes(); return {"path":str(p),"sha256":sha_bytes(b),"bytes":len(b)}
def readj(p): return json.loads(Path(p).read_text())
def parse_csv(p):
 raw=Path(p).read_bytes(); text=raw.decode("utf-8"); rows=list(csv.reader(io.StringIO(text,newline="")))
 if not rows or any(len(r)!=len(rows[0]) for r in rows[1:]): raise RuntimeError(f"malformed CSV {p}")
 return raw,rows[0],rows[1:]
def fields(line):
 out={}
 for token in line.split():
  if "=" in token:
   k,v=token.split("=",1); out[k]=v
 return out
def stat(values):
 vals=[float(x) for x in values]
 if not vals or not all(math.isfinite(x) for x in vals): raise RuntimeError("empty/nonfinite series")
 return {"min":min(vals),"max":max(vals),"terminal":vals[-1],"max_abs":max(abs(x) for x in vals)}
def summarize_csv(path, header, rows):
 key=STEP_KEYS[path.name]
 if key not in header: raise RuntimeError(f"missing {key} in {path}")
 ki=header.index(key); steps=[int(r[ki]) for r in rows]
 if not steps or steps[-1]!=20000 or any(b<a for a,b in zip(steps,steps[1:])): raise RuntimeError(f"bad accepted-step coverage in {path}")
 out={"rows":len(rows),"first_step":steps[0],"terminal_step":steps[-1],"step_key":key}
 if path.name=="resting-coupled.csv":
  result={}
  for col in header:
   if any(t in col.lower() for t in ("breath","cycle","heart","beat")):
    ix=header.index(col)
    vals=[float(r[ix]) for r in rows]
    result[col]={"initial":vals[0],"terminal":vals[-1],"min":min(vals),"max":max(vals),"increments":sum(1 for a,b in zip(vals,vals[1:]) if b>a)}
  out["cycle_counters"]=result
  for col in ("PaO2_mmhg","PaCO2_mmhg","SaO2","lung_volume_ml","airflow_ml_s","blood_ml","aortic_ejected_ml","pulmonary_ejected_ml","complete_filling_ejection_cycles","breaths","root_assistance_n","root_assistance_nm"):
   if col in header: out.setdefault("selected_ranges",{})[col]=stat([r[header.index(col)] for r in rows])
 return out

bd=readj(BASE/"run-declaration.json"); cd=readj(CANDIDATE/"run-declaration.json")
bm=readj(BR/"run-metadata.json"); cm=readj(CR/"run-metadata.json")
bi=readj(BR/"invocation.json"); ci=readj(CR/"invocation.json")
be=readj(BASE/"execution.json"); ce=readj(CANDIDATE/"execution.json")
if bm.get("exit_code")!=0 or cm.get("exit_code")!=0: raise RuntimeError("native owner exit code not zero")
if be.get("returncode")!=0 or ce.get("returncode")!=0: raise RuntimeError("wrapper return code not zero")
if be.get("changed_inputs")!={} or ce.get("changed_inputs")!={}: raise RuntimeError("run wrapper reports changed inputs")
if be.get("declaration_sha256")!=sha(BASE/"run-declaration.json") or ce.get("declaration_sha256")!=sha(CANDIDATE/"run-declaration.json"): raise RuntimeError("execution receipt declaration binding mismatch")
if cm.get("source_files_changed_during_run") not in ([],None): raise RuntimeError("candidate source drift")
if (bd.get("seconds"),bd.get("dt"),bd.get("capture_steps"))!=(40,0.002,[0,20000]): raise RuntimeError("1201 declaration drift")
if (cd.get("seconds"),cd.get("dt"),cd.get("capture_steps"))!=(40,0.002,[0,9983,20000]): raise RuntimeError("1217 attempt2 declaration drift")
if bd.get("contact_iterations")!=cd.get("contact_iterations") or bd.get("contact_iterations")!=64: raise RuntimeError("contact-iteration mismatch")
base_assets=bd["immutable_assets"]; cand_assets=cd["immutable_assets"]
shared=[]; declared_mismatches=[]; missing=[]
for p,h in base_assets.items():
 if cand_assets.get(p)!=h: declared_mismatches.append({"path":p,"base":h,"candidate":cand_assets.get(p)})
 if not Path(p).is_file(): missing.append(p)
 elif sha(p)!=h: declared_mismatches.append({"path":p,"expected":h,"actual":sha(p)})
 else: shared.append(p)
if declared_mismatches or missing: raise RuntimeError("shared 1201 immutable input pins changed")
candidate_bad=[]
for p,h in cand_assets.items():
 if not Path(p).is_file() or sha(p)!=h: candidate_bad.append({"path":p,"expected":h,"actual":sha(p) if Path(p).is_file() else None})
if candidate_bad: raise RuntimeError("1217 declared immutable asset hash mismatch")
# Verify actual invocation pin maps agree with run metadata. The candidate intentionally relocates
# common-field files and replaces the passive muscle-surface payload, manifest, and receipt.
base_run_assets=bm.get("asset_sha256",{}); cand_run_assets=cm.get("asset_sha256",{})
old_relocated={
 "/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/package-003/attempt-010/admission-output/composed-anatomy/common-cardiac-map-f32.bin": "c0b0811fe2e2fb45dc37201205e36af39cf092739cfb52c56fdbad9a67dbb858",
 "/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/package-003/attempt-010/admission-output/composed-anatomy/common-cardiac-volumes-f32.bin": "7be4f62996c996c6cfdd49480d27766c8253387bb24aeb504fef73cd553ae699",
 "/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/package-003/attempt-010/admission-output/composed-anatomy/common-cardiac-domains-f32.bin": "7652785c9ea164b16ee96f64a3dcf940398af36cfd535a5f9b7a7d6bd2c7a611",
 "/Users/n/numi-human-resting-evidence-20261005/passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue": "b3d0381f73e05b04ed7ea1ba23eb6559073ad20aa0d9e2ad0f97a18fab3359bd",
 "/Users/n/numi-human-resting-evidence-20261005/passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json": "82cdd937e0f2daf0a8704fb21246353c38148527f8602d674ac865f935264dde",
 "/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/package-003/attempt-010/admission-output/composed-anatomy/resting-anatomy-receipt.json": "1d3c364a8f5b66f6c88412a193b801d57a75de1ff797f9d61247da684810e2cd",
}
new_relocated={
 "/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/common-cardiac-map-f32.bin": "c0b0811fe2e2fb45dc37201205e36af39cf092739cfb52c56fdbad9a67dbb858",
 "/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/common-cardiac-volumes-f32.bin": "7be4f62996c996c6cfdd49480d27766c8253387bb24aeb504fef73cd553ae699",
 "/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/common-cardiac-domains-f32.bin": "7652785c9ea164b16ee96f64a3dcf940398af36cfd535a5f9b7a7d6bd2c7a611",
 "/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue": "1cd0c3d5bd1ff6d163f6544be4d0fa169e8bf696bf6c729ef0b18879b0856c48",
 "/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json": "f052eff736f040f2f6ca1a1a99d2939ca0ed5c0bead055dfb39d3141f05a3cac",
 "/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/resting-anatomy-receipt.json": "10ecac382b42e08b8ba296f361d419a0a7db1fb6e458448596e127acbea14d62",
}
base_core={p:h for p,h in base_run_assets.items() if p not in old_relocated}
cand_core={p:h for p,h in cand_run_assets.items() if p not in new_relocated}
if base_core!=cand_core: raise RuntimeError("candidate changed a non-muscle runtime asset pin")
if any(base_run_assets.get(p)!=h for p,h in old_relocated.items()): raise RuntimeError("1201 runtime replacement source pins differ from declaration")
if any(cand_run_assets.get(p)!=h for p,h in new_relocated.items()): raise RuntimeError("candidate runtime replacement pins differ from declared lineage")
if any(ci.get("asset_sha256",{}).get(p)!=h for p,h in cand_run_assets.items()): raise RuntimeError("candidate invocation/runtime asset maps disagree")
# Compare native arguments after replacing the explicitly allowed paths and omitting only the two new passive-surface args.
def canonical_argv(inv, run_dir, soft, receipt, movie):
 a=list(inv["argv"])
 repl={str(run_dir):"<RUN>",str(soft):"<MUSCLE_SURFACE>",str(receipt):"<ANATOMY_RECEIPT>",str(run_dir/"native-viewer.mov"):"<MOVIE>"}
 return [repl.get(x,x) for x in a]
base_soft=Path("/Users/n/numi-human-resting-evidence-20261005/passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue")
cand_soft=Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue")
base_receipt=Path("/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/package-003/attempt-010/admission-output/composed-anatomy/resting-anatomy-receipt.json")
cand_receipt=Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/resting-anatomy-receipt.json")
ba=canonical_argv(bi,BR,base_soft,base_receipt,BR/"native-viewer.mov")
ca=canonical_argv(ci,CR,cand_soft,cand_receipt,CR/"native-viewer.mov")
# New native args are the intended muscle surface manifest/payload only; retain them explicitly in the report.
base_canon=ba
cand_canon=ca
extra_tokens=[x for x in cand_canon if x not in base_canon]
missing_tokens=[x for x in base_canon if x not in cand_canon]
allowed_extra=[]
for flag in ("--native-muscle-surfaces", "--native-muscle-surfaces-manifest", "--native-muscle-surfaces-payload"):
 if flag in extra_tokens: allowed_extra.append(flag)
# Some builds use one payload flag and include a separate manifest flag; print all actual argv edits later.
# Byte-compare the complete physical and physiology CSV streams.
csv_report={}; csv_equal=True
for name in CSV_NAMES:
 bp=BR/name; cp=CR/name
 bb,bh,brows=parse_csv(bp); cb,ch,crows=parse_csv(cp)
 exact=bb==cb; csv_equal &= exact
 first=None
 if not exact:
  for i,(x,y) in enumerate(zip(bb,cb)):
   if x!=y: first={"byte_offset":i,"base_byte":x,"candidate_byte":y}; break
  if first is None: first={"length_base":len(bb),"length_candidate":len(cb)}
 if bh!=ch: raise RuntimeError(f"CSV header differs: {name}")
 csv_report[name]={"base":pin(bp),"candidate":pin(cp),"byte_exact":exact,"field_header_exact":bh==ch,"base_summary":summarize_csv(bp,bh,brows),"candidate_summary":summarize_csv(cp,ch,crows),"first_difference":first}
# Native logs: compare exact transaction replay/rejection-probe evidence and terminal q/v state.
def native_log_summary(path):
 lines=Path(path).read_text(errors="replace").splitlines()
 done=[x for x in lines if x.startswith("resting_integrated_body=completed ")]
 reject=[x for x in lines if x.startswith("resting_integrated_rejection=")]
 term=[x for x in lines if x.startswith("stand_terminal_state=")]
 if len(done)!=1 or len(reject)!=1 or len(term)!=1: raise RuntimeError(f"native log completion/probe/terminal records are not unique: {path}")
 d=fields(done[0]); r=fields(reject[0]); t=json.loads(term[0].split("=",1)[1])
 return {"completion_line":done[0],"completion_fields":d,"rejection_line":reject[0],"rejection_fields":r,"terminal_state":{"schema":t.get("schema"),"step_count":t.get("step_count"),"timestep_seconds":t.get("timestep_seconds"),"root_assistance":t.get("root_assistance"),"q_count":len(t.get("q",[])),"v_count":len(t.get("v",[])),"q_sha256":sha_bytes(json.dumps(t.get("q",[]),separators=(",",":"),ensure_ascii=True).encode()),"v_sha256":sha_bytes(json.dumps(t.get("v",[]),separators=(",",":"),ensure_ascii=True).encode()),"q":t.get("q"),"v":t.get("v")}}
bl=native_log_summary(BR/"native.log"); cl=native_log_summary(CR/"native.log")
# Receipts bind complete rendered identity as well as physical state; separate the two.
receipt_report={}
receipt_state_keys=("accepted_step","accepted_time_s","accepted_timestamp_microseconds","accepted_body_state_sha256","accepted_respiration_state_sha256","accepted_registered_body_poses","accepted_respiratory_motion","physical_endpoint","surface_audit_endpoint")
for step in (0,20000):
 bpath=BR/f"accepted-geometry/step-{step}.receipt.json"; cpath=CR/f"accepted-geometry/step-{step}.receipt.json"
 br=readj(bpath); cr=readj(cpath)
 diffs={k:{"base":br.get(k),"candidate":cr.get(k)} for k in receipt_state_keys if br.get(k)!=cr.get(k)}
 fingerprints={k:{"base":br.get(k),"candidate":cr.get(k),"equal":br.get(k)==cr.get(k)} for k in ("accepted_root_fingerprint","accepted_transaction_fingerprint")}
 receipt_report[str(step)]={"base_receipt":pin(bpath),"candidate_receipt":pin(cpath),"accepted_body_and_respiration_state_fields_exact":not diffs,"state_field_differences":diffs,"composite_identity_fingerprints":fingerprints,"base_capture":{k:br.get(k) for k in ("captured_vertex_buffer_sha256","index_count","primitive_count","instance_count","vertex_count","surface_audit")},"candidate_capture":{k:cr.get(k) for k in ("captured_vertex_buffer_sha256","index_count","primitive_count","instance_count","vertex_count","surface_audit")}}
# Record the additional passive midpoint receipt separately; there is no same-step 1201 pack.
mid_path=CR/"accepted-geometry/step-9983.receipt.json"
mid_receipt=readj(mid_path)
if mid_receipt.get("accepted_step")!=9983 or mid_receipt.get("physical_endpoint")!="accepted": raise RuntimeError("midpoint receipt is not the declared accepted state")
mid_capture={"receipt":pin(mid_path),"accepted_step":mid_receipt.get("accepted_step"),"accepted_time_s":mid_receipt.get("accepted_time_s"),"accepted_body_state_sha256":mid_receipt.get("accepted_body_state_sha256"),"accepted_respiration_state_sha256":mid_receipt.get("accepted_respiration_state_sha256"),"root_fingerprint":mid_receipt.get("accepted_root_fingerprint"),"transaction_fingerprint":mid_receipt.get("accepted_transaction_fingerprint")}
# Compare deterministic replay probe fields; the rejection line itself should be identical when path-independent.
probe_keys=("resting_integrated_rejection","rejected_after_accepted_predecessor","body_q_v_root_myo_unchanged","circulation_and_clock_unchanged","respiration_brain_history_unchanged","retry_matches_uninterrupted_replay","same_command_buffer_reject","accepted_prefix","rejected_step","inert_suffix","multistep_reject","accepted_prefix","rejected_global_step","inert_suffix_global_step","all_owners_match_uninterrupted_prefix","retry","same_context_retry")
probe_equal=all(bl["rejection_fields"].get(k)==cl["rejection_fields"].get(k) for k in probe_keys)
terminal_equal=(bl["terminal_state"]["q"]==cl["terminal_state"]["q"] and bl["terminal_state"]["v"]==cl["terminal_state"]["v"] and all(bl["terminal_state"].get(k)==cl["terminal_state"].get(k) for k in ("step_count","timestep_seconds","root_assistance")))
physical_csv_names=("resting-coupled.csv","resting-com-momentum-diagnostic.csv","resting-com-support-impulses.csv")
physical_csv_equal=all(csv_report[n]["byte_exact"] for n in physical_csv_names)
# The new mesh has one vertex and two faces; require the surface-audit stream to differ only in its mesh triangle count.
sbase,bh,brows=parse_csv(BR/"resting-surface-audit.csv"); scand,ch,crows=parse_csv(CR/"resting-surface-audit.csv")
if bh!=ch or len(brows)!=len(crows): raise RuntimeError("surface audit schema/coverage mismatch")
mesh_ix=bh.index("mesh_triangles_checked")
other_surface_differences=[]; triangle_deltas=[]
for i,(x,y) in enumerate(zip(brows,crows)):
 for j,(a,b) in enumerate(zip(x,y)):
  if j==mesh_ix:
   try: triangle_deltas.append(int(b)-int(a))
   except ValueError: other_surface_differences.append({"row":i,"column":bh[j],"base":a,"candidate":b})
  elif a!=b: other_surface_differences.append({"row":i,"column":bh[j],"base":a,"candidate":b})
surface_audit_geometry_only_delta=(not other_surface_differences and len(set(triangle_deltas))==1 and triangle_deltas[0]==2)
# Runtime/path and output timing.
loaded={"base":bm.get("loaded_metal_runtime"),"candidate":cm.get("loaded_metal_runtime")}
runtime_same_verified=bool(loaded["base"] and loaded["candidate"] and loaded["base"].get("verified") is True and loaded["candidate"].get("verified") is True and loaded["base"].get("expected_sha256")==loaded["candidate"].get("expected_sha256") and loaded["base"].get("expected_path")==loaded["candidate"].get("expected_path"))
old_attempt=Path("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/native-refinement-1217")
old_log=(old_attempt/"native-run/native.log").read_text(errors="replace")
old_failure=next((line for line in old_log.splitlines() if "myosim_articulated_visual=failed" in line), "")
old_meta=readj(old_attempt/"native-run/run-metadata.json")
report={
 "schema":"numi.human.passive-muscle-midpoint-physics-regression.v1",
 "scope":"40-second native run with a passive NHTISS geometry refinement and additional accepted-state capture; read-only trace comparison against flat 1201 control.",
 "status":"physical_physiology_trace_parity" if physical_csv_equal and surface_audit_geometry_only_delta and probe_equal and terminal_equal and runtime_same_verified and all(x["accepted_body_and_respiration_state_fields_exact"] for x in receipt_report.values()) else "difference_detected",
 "base_declaration":pin(BASE/"run-declaration.json"),"candidate_declaration":pin(CANDIDATE/"run-declaration.json"),
 "base_run_metadata":pin(BR/"run-metadata.json"),"candidate_run_metadata":pin(CR/"run-metadata.json"),
 "base_invocation":pin(BR/"invocation.json"),"candidate_invocation":pin(CR/"invocation.json"),
 "shared_original_immutable_assets":{"declared_count":len(base_assets),"actual_hashes_verified":len(shared),"all_exact":True,"candidate_extra_asset_count":len(cand_assets)-len(base_assets),"candidate_all_declared_asset_hashes_exact":True,"base_runtime_assets":len(base_run_assets),"candidate_runtime_assets":len(cand_run_assets),"candidate_all_original_runtime_asset_pins_exact":True},
 "verified_input_pin_maps":{"1201_declared_immutable_assets":base_assets,"1217_declared_immutable_assets":cand_assets,"1201_native_runtime_assets":base_run_assets,"1217_native_runtime_assets":cand_run_assets},
 "execution_status":{"1201_wrapper_returncode":be.get("returncode"),"1217_wrapper_returncode":ce.get("returncode"),"1201_wrapper_changed_inputs":be.get("changed_inputs"),"1217_wrapper_changed_inputs":ce.get("changed_inputs"),"1201_owner_exit_code":bm.get("exit_code"),"1217_owner_exit_code":cm.get("exit_code"),"1217_source_files_changed_during_run":cm.get("source_files_changed_during_run"),"same_loaded_native_runtime_verified":runtime_same_verified},
 "candidate_geometry_assets":{"muscle_payload":{"path":str(cand_soft),"sha256":sha(cand_soft)},"muscle_manifest":{"path":"/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json","sha256":sha("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json")},"anatomy_receipt":{"path":str(cand_receipt),"sha256":sha(cand_receipt)},"candidate_composition_report":{"path":"/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/report.json","sha256":sha("/Users/n/numi-human-retained-delivery-20261009/muscle-conforming-refinement-1216/compose-final-001/candidate/report.json")}},
 "declared_physics_configuration":{"seconds":cd["seconds"],"dt":cd["dt"],"steps":cd["capture_steps"],"contact_iterations":cd["contact_iterations"],"postural_activation_cap":cd["postural_activation_cap"],"release_initialization":cd["release_initialization"],"rigid_hands_enabled":cd["rigid_hands_enabled"],"hip_reference_enabled":cd["hip_reference_enabled"],"parent_1201_declaration_sha256":cd.get("parent_declaration_sha256"),"capture_schedule_difference":"attempt2 adds midpoint at accepted step 9983 (19.966 s); 1201 has [0,20000]"},
 "actual_native_configuration":{"argv_common_prefix_exact_after_path_normalization":base_canon==cand_canon,"base_runtime":loaded["base"],"candidate_runtime":loaded["candidate"],"native_invocation_base_argv":bi["argv"],"native_invocation_candidate_argv":ci["argv"]},
 "trace_comparison":{"coupled_com_support_csvs_byte_exact":physical_csv_equal,"geometry_surface_audit_only_expected_added_triangle_count":surface_audit_geometry_only_delta,"surface_audit_triangle_count_delta_per_row":sorted(set(triangle_deltas)),"surface_audit_other_field_differences":other_surface_differences[:10],"files":csv_report},
 "accepted_endpoint_comparison":receipt_report,
 "additional_midpoint_capture":mid_capture,
 "failed_attempt1":{"declaration":pin(old_attempt/"run-declaration.json"),"execution":pin(old_attempt/"execution.json"),"run_metadata":pin(old_attempt/"native-run/run-metadata.json"),"native_log":pin(old_attempt/"native-run/native.log"),"owner_exit_code":old_meta.get("exit_code"),"failure_line":old_failure,"accepted_state_packs_present":sorted(p.name for p in (old_attempt/"native-run/accepted-geometry").glob("*.mrvpack"))},
 "replay_and_rejection_probe":{"base_line":bl["rejection_line"],"candidate_line":cl["rejection_line"],"structured_probe_fields_exact":probe_equal,"terminal_q_v_and_step_exact":terminal_equal,"base_terminal":{"step_count":bl["terminal_state"]["step_count"],"dt":bl["terminal_state"]["timestep_seconds"],"root_assistance":bl["terminal_state"]["root_assistance"],"q_count":bl["terminal_state"]["q_count"],"v_count":bl["terminal_state"]["v_count"],"q_sha256":bl["terminal_state"]["q_sha256"],"v_sha256":bl["terminal_state"]["v_sha256"]},"candidate_terminal":{"step_count":cl["terminal_state"]["step_count"],"dt":cl["terminal_state"]["timestep_seconds"],"root_assistance":cl["terminal_state"]["root_assistance"],"q_count":cl["terminal_state"]["q_count"],"v_count":cl["terminal_state"]["v_count"],"q_sha256":cl["terminal_state"]["q_sha256"],"v_sha256":cl["terminal_state"]["v_sha256"]}},
 "timing":{"base_owner_wall_seconds":bm.get("wall_seconds"),"candidate_owner_wall_seconds":cm.get("wall_seconds"),"base_wrapper_elapsed_seconds":be.get("elapsed_s",be.get("wall_seconds")),"candidate_wrapper_elapsed_seconds":ce.get("elapsed_s",ce.get("wall_seconds")),"base_integrated":{"simulated_s":float(bl["completion_fields"]["simulated_s"]),"wall_s":float(bl["completion_fields"]["wall_s"]),"rtf":float(bl["completion_fields"]["real_time_factor"])},"candidate_integrated":{"simulated_s":float(cl["completion_fields"]["simulated_s"]),"wall_s":float(cl["completion_fields"]["wall_s"]),"rtf":float(cl["completion_fields"]["real_time_factor"])},"qualification":"Timing is descriptive; runs are not an isolated performance comparison."},
 "physiology_counters":{"base":csv_report["resting-coupled.csv"]["base_summary"].get("cycle_counters"),"candidate":csv_report["resting-coupled.csv"]["candidate_summary"].get("cycle_counters"),"base_ranges":csv_report["resting-coupled.csv"]["base_summary"].get("selected_ranges"),"candidate_ranges":csv_report["resting-coupled.csv"]["candidate_summary"].get("selected_ranges")},
 "limits":["This is a 40-second physics/trace regression for the passive muscle-surface refinement, not anatomical clearance or full-cycle qualification.","The extra 19.966 s accepted capture is observational; exact traces and common endpoint states determine whether it left dynamics unchanged.","Native invocation, composite identity fingerprints, and rendered pack hashes differ because the candidate binds a different passive muscle-surface asset; explicit body/respiration state hashes and physical/physiology traces are compared separately.","The surface audit records the expected added two triangles in each sampled mesh count; its remaining values match exactly.","Attempt1 is preserved separately as a failed capture-cadence startup and is not included as a successful run."]
}
# Avoid embedding full terminal vectors; remove them from internal summaries before write.
for x in (bl,cl): x["terminal_state"].pop("q",None); x["terminal_state"].pop("v",None)
report["analysis_script"]=pin(Path(__file__))
OUT.mkdir(parents=True,exist_ok=True)
target=OUT/"comparison.json"
tmp=target.with_suffix(".json.tmp"); tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); tmp.replace(target)
print(json.dumps({"status":report["status"],"report":str(target),"report_sha256":sha(target),"csv":{k:{"byte_exact":v["byte_exact"],"base_sha":v["base"]["sha256"],"candidate_sha":v["candidate"]["sha256"],"rows":v["candidate_summary"]["rows"]} for k,v in csv_report.items()},"probes":{"exact":probe_equal,"terminal_qv_exact":terminal_equal},"timing":report["timing"],"cycle_counters":report["physiology_counters"]},indent=2,sort_keys=True))
