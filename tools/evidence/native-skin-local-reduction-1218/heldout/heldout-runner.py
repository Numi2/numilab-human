#!/usr/bin/env python3
"""Prepare/run held-out 1218 skin audit with offline refined stable64 target.
This wrapper does not launch native physics. --scan is legal only after fit-attempt-001 succeeds.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
import numpy as np
for n in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"): os.environ[n]="1"
R=Path("/Users/n/numi-human-retained-delivery-20261009"); E=Path("/Users/n/numi-human-resting-evidence-20261005")
ROOT=Path("/Users/n/numi-human-conforming-composition-source-1216")
HERE=R/"skin-resting-multipose-clearance-1218/heldout-001"
OUT=HERE/"scan-output"
BASE_AUDIT=HERE/"audit_six_holdouts_1218.py"
FIT=R/"skin-resting-multipose-clearance-1218/fit-attempt-001"
HELDOUT=(4991,5375,5759,6111,6495,7743); KEY=(51005,64)
TISS=R/"muscle-conforming-refinement-1216/compose-final-001/candidate/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
MAN=TISS.with_name("bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json")
BASE_TISS=E/"passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
CTX=R/"muscle-conforming-refinement-1216/producer-attempt-002/producer-subset-001"
CTISS=CTX/"bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"; CMAN=CTX/"bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
TARGET_VALIDATION=HERE/"refined-target-1217-validation.json"
VALIDATOR=HERE/"validate-refined-target-1217.py"
HELPER=Path("/Users/n/numi-human-conforming-composition-source-1216/src/numilab_human/model.py")
RUN=R/"native-lung1178-thumb1187-smoke-1191/native-run"
NATIVE_DECL=R/"muscle-conforming-refinement-1216/native-refinement-1217-attempt2/run-declaration.json"
PRODUCER=R/"muscle-conforming-refinement-1216/producer-attempt-002"
PRODUCER_DECL=PRODUCER/"producer-declaration.json"
PRODUCER_EXEC=PRODUCER/"producer-execution.json"
BASE_MAN=E/"passive-tissue-achilles-clearance-039-composed-001/bodyparts3d-myosim-fullbody-muscle-surfaces.manifest.json"
COMMON_CLEARANCE=ROOT/"src/numilab_human/common_atlas_skin_clearance.py"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<22),b""): h.update(b)
 return h.hexdigest()
def pin(p): return {"path":str(Path(p).resolve()),"sha256":sha(p),"bytes":Path(p).stat().st_size}
def need(x,msg):
 if not x: raise RuntimeError(msg)
def verify_hash_map(label,mapping):
 need(isinstance(mapping,dict) and mapping,label+" has no path-to-SHA map")
 checked=[]
 for raw,expected in sorted(mapping.items()):
  p=Path(raw)
  need(p.is_file() and not p.is_symlink(),label+" input missing or symlinked: "+str(p))
  need(sha(p)==expected,label+" input hash mismatch: "+str(p))
  checked.append(pin(p))
 return checked
def verify_pin_list(label,pins):
 need(isinstance(pins,list) and pins,label+" has no input-pin list")
 checked=[]
 for item in pins:
  need(isinstance(item,dict) and "path" in item and "sha256" in item,label+" has malformed input pin")
  p=Path(item["path"])
  need(p.is_file() and not p.is_symlink(),label+" input missing or symlinked: "+str(p))
  need(sha(p)==item["sha256"],label+" input hash mismatch: "+str(p))
  if "bytes" in item: need(p.stat().st_size==int(item["bytes"]),label+" input byte count mismatch: "+str(p))
  checked.append(pin(p))
 return checked
def import_path(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 need(spec is not None and spec.loader is not None,"cannot load "+str(path))
 mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument("--prepare-check",action="store_true")
 ap.add_argument("--candidate-check",action="store_true")
 ap.add_argument("--scan",action="store_true")
 ap.add_argument("--out",type=Path,default=OUT)
 a=ap.parse_args()
 need(sum(bool(x) for x in (a.prepare_check,a.candidate_check,a.scan))==1,"select exactly one of --prepare-check, --candidate-check, or --scan")
 need(a.out.resolve()==OUT.resolve(),"output path is fixed to the designated fresh sibling")
 need(not OUT.exists() if a.scan else True,"refuse existing scan output")
 target_report=json.loads(TARGET_VALIDATION.read_text())
 need(target_report.get("status")=="pass_within_1um","1217 refined-target validation has not passed")
 need(target_report.get("inputs_unchanged") is True,"1217 refined-target validation did not verify input stability")
 target_validation_inputs=verify_hash_map("1217 target validation",target_report.get("inputs"))
 need(target_report["inputs"].get(str(TISS.resolve()))==sha(TISS),"refined NHTISS drifted after 1217 validation")
 need(target_report["inputs"].get(str(MAN.resolve()))==sha(MAN),"refined NHTISS manifest drifted after 1217 validation")
 producer_decl=json.loads(PRODUCER_DECL.read_text())
 producer_exec=json.loads(PRODUCER_EXEC.read_text())
 need(producer_exec.get("returncode")==0 and producer_exec.get("changed_inputs")==[],"stable64 source producer did not complete with unchanged inputs")
 need(producer_exec.get("declaration_sha256")==sha(PRODUCER_DECL),"producer execution does not bind its declaration")
 producer_argv=producer_decl.get("argv",[])
 def option_value(option):
  return producer_argv[producer_argv.index(option)+1] if option in producer_argv and producer_argv.index(option)+1<len(producer_argv) else None
 need(Path(option_value("--output") or "/").resolve()==CTX.resolve(),"producer declaration does not point to the context directory")
 need(option_value("--stable-id")=="64" and option_value("--conforming-edge-refinement")=="64:2597:3054","producer declaration does not bind the selected stable64 edge")
 context_manifest=json.loads(CMAN.read_text())
 context_payload=context_manifest.get("payload",{})
 need(context_manifest.get("coverage",{}).get("selected_stable_ids")==[64] and context_manifest.get("coverage",{}).get("emitted_surface_count")==1,"producer context is not the bounded stable64 subset")
 need(context_payload.get("file")==CTISS.name and int(context_payload.get("bytes",-1))==CTISS.stat().st_size and context_payload.get("sha256")==sha(CTISS),"producer manifest does not bind the emitted context payload")
 final_manifest=json.loads(MAN.read_text())
 final_source=final_manifest.get("source",{})
 comp=final_source.get("conforming_edge_refinement_composition",{})
 final_surface=next((x for x in final_source.get("surfaces",[]) if int(x.get("stable_id",-1))==64),None)
 context_surface=next((x for x in context_manifest.get("source",{}).get("surfaces",[]) if int(x.get("stable_id",-1))==64),None)
 need(final_surface is not None and context_surface is not None,"stable64 operation provenance is missing")
 operation=final_surface.get("conforming_edge_refinement")
 context_operation=context_surface.get("conforming_edge_refinement")
 need(operation==context_operation,"final stable64 operation differs from producer-derived context")
 need(comp.get("source_payload_path")==str(BASE_TISS) and comp.get("source_payload_sha256")==sha(BASE_TISS),"composition does not bind the accepted039 source payload")
 need(comp.get("source_manifest_path")==str(BASE_MAN) and comp.get("source_manifest_sha256")==sha(BASE_MAN),"composition does not bind the accepted039 source manifest")
 need(comp.get("context_payload_path")==str(CTISS) and comp.get("context_payload_sha256")==sha(CTISS),"composition context payload pin mismatch")
 need(comp.get("context_manifest_path")==str(CMAN) and comp.get("context_manifest_sha256")==sha(CMAN),"composition context manifest pin mismatch")
 need(comp.get("helper_source_path")==str(HELPER) and comp.get("helper_source_sha256")==sha(HELPER),"refinement helper pin mismatch")
 native_decl=json.loads(NATIVE_DECL.read_text())
 candidate_record=native_decl["candidate"]["candidate_native_muscle_surfaces_record"]
 need(Path(candidate_record["payload_path"]).resolve()==TISS.resolve() and candidate_record["sha256"]==sha(TISS),"1217 declaration does not bind the refined NHTISS")
 need(Path(native_decl["candidate"]["candidate_manifest"]).resolve()==MAN.resolve() and native_decl["candidate"]["candidate_manifest_sha256"]==sha(MAN),"1217 declaration does not bind the refined manifest")
 fit_preflight_path=FIT/"preflight.json"
 need(fit_preflight_path.is_file(),"1218 fit preflight is missing")
 fit_preflight=json.loads(fit_preflight_path.read_text())
 fit_preflight_inputs=verify_pin_list("1218 fit preflight",fit_preflight.get("input_pins"))
 solve_start_path=FIT/"solve-start.json"
 solve_start=json.loads(solve_start_path.read_text())
 need(solve_start.get("preflight_sha256")==sha(fit_preflight_path),"fit solve-start does not bind its preflight")
 fit_pins={str(Path(x["path"]).resolve()):x["sha256"] for x in fit_preflight.get("input_pins",[]) if isinstance(x,dict) and "path" in x}
 need(str(NATIVE_DECL.resolve()) in fit_pins and fit_pins[str(NATIVE_DECL.resolve())]==sha(NATIVE_DECL),"fit preflight does not bind the 1217 NHTISS declaration")
 audit=import_path(BASE_AUDIT,"_audit_six_holdouts_1206_for_1218")
 audit.FIT=FIT; audit.OUT_DEFAULT=OUT
 if a.prepare_check:
  wrapper=audit.load_wrapper(); static=wrapper.validate_static_sources()
  runner=wrapper.load_runner(); keys,_=runner.inventory(); ci,clearance,validator,core=runner.load_predicates()
  need(len(keys)==859,"existing exact runner inventory is not859")
  predicate_modules=[pin(Path(m.__file__).resolve()) for m in (ci,clearance,validator,core)]
  holdout_checks=[]
  for step in HELDOUT:
   pack=RUN/"accepted-geometry"/f"step-{step}.mrvpack"; rec=pack.with_suffix(".receipt.json")
   accepted=runner.verify_pack(step,pack,rec,validator,wrapper.NHA_SHA)
   holdout_checks.append({"step":step,"pack":pin(pack),"receipt":pin(rec),"verified":bool(accepted)})
  need(all(x["verified"] for x in holdout_checks),"one or more heldout accepted-state pack receipts failed existing 1172 verification")
  print(json.dumps({"status":"heldout_runner_prepared_not_scanned","target_forward_status":target_report["status"],"target_validation_inputs_verified":len(target_validation_inputs),"target_validation_script":pin(VALIDATOR),"target_pack_decoder":pin(COMMON_CLEARANCE),"producer_declaration":pin(PRODUCER_DECL),"producer_execution":pin(PRODUCER_EXEC),"producer_context_payload":pin(CTISS),"producer_context_manifest":pin(CMAN),"refinement_operation_exact_match":True,"fit_preflight_inputs_verified":len(fit_preflight_inputs),"fit_solve_start":pin(solve_start_path),"heldout_pack_receipts":holdout_checks,"heldout_steps":list(HELDOUT),"target_faces":6690,"scan_output":str(OUT),"audit_base":pin(BASE_AUDIT),"static_source_pins":[pin(x) for x in static],"predicate_module_pins":predicate_modules},sort_keys=True))
  return 0
 if a.candidate_check:
  need(FIT.joinpath("candidate-report.json").is_file() and FIT.joinpath("candidate-source-positions-f32.npy").is_file(),"fit candidate is not complete; candidate-check did not run")
  sys.argv=[str(BASE_AUDIT),"--candidate-check","--out",str(OUT)]
  rc=audit.main()
  need(rc==0,"1218 no-scan candidate check returned nonzero")
  return 0
 # Candidate is read only after the fit owner has completed its successful report.
 report_path=FIT/"candidate-report.json"; cand_path=FIT/"candidate-source-positions-f32.npy"
 need(report_path.is_file() and cand_path.is_file(),"fit candidate is not complete; held-out scan remains unlaunched")
 report=json.loads(report_path.read_text())
 need(report.get("status")=="inferred_engineering_clearance_candidate_pending_native_replay","fit-attempt-001 did not succeed with the expected source-only status")
 # Reuse the exact stable64 decoded row from the separately pinned 1217 target validator.
 target_module=import_path(VALIDATOR,"_refined_stable64_1217_validation")
 candidate=target_module.row(TISS,64); base=target_module.row(BASE_TISS,64)
 target_faces=np.asarray(candidate["f"],dtype=np.int64); old_faces=np.asarray(base["f"],dtype=np.int64)
 need(candidate["counts"]==[3975,6690] and base["counts"]==[3974,6688],"NHTISS target row count changed")
 need(np.array_equal(candidate["v"][:-1].view("u1"),base["v"].view("u1")),"old stable64 NHTISS vertex rows changed")
 need(np.array_equal(candidate["binds"].view("u1"),base["binds"].view("u1")),"old stable64 NHTISS binding rows changed")
 need(target_report["inputs_unchanged"] is True,"1217 row validation input inventory was not stable")
 # Pin current fit event using the existing attempt checker before the long audit starts.
 skin=audit.load_skin(audit.SKIN); ref=np.unique(skin["faces"])
 wrapper0=audit.load_wrapper(); runner0=wrapper0.load_runner(); keys0,_=runner0.inventory()
 fit_event=audit.validate_final_accepted_trial(np.load(cand_path,allow_pickle=False),ref,report,runner0,keys0)
 audit.fit_event=fit_event
 injections=[]
 # Wrap the existing captured-pack reader. Preserve all old captured coordinates and append one predicted midpoint.
 def patched_load_wrapper():
  w=audit.load_wrapper_original()
  old_load_runner=w.load_runner
  def patched_load_runner():
   runner=old_load_runner(); old_load_predicates=runner.load_predicates
   def patched_load_predicates():
    ci,clearance,validator,core=old_load_predicates()
    original_pack_reader=clearance._pack_surfaces
    def read_with_refined_64(pack, requested):
     positions,surfaces,counts=original_pack_reader(pack,requested)
     p=Path(pack).resolve()
     if KEY not in requested or p.parent!=RUN/"accepted-geometry":
      return positions,surfaces,counts
     step=int(p.stem.removeprefix("step-"))
     if step not in HELDOUT: return positions,surfaces,counts
     receipt_path=p.with_suffix(".receipt.json"); receipt=json.loads(receipt_path.read_text())
     need(int(receipt.get("accepted_step",-1))==step,"heldout NHTISS target prediction lacks accepted pose receipt")
     packed_faces=np.asarray(surfaces[KEY]["faces"],dtype=np.int64)
     old_map=target_module.face_map(old_faces,packed_faces,len(base["v"]))
     original_positions=np.asarray(positions)
     midpoint_world=target_module.mapped_vertices(candidate["v"][-1:],candidate["binds"],receipt)[0]
     new_global_id=len(original_positions)
     local_to_global=np.concatenate([old_map,np.asarray([new_global_id],dtype=np.int64)])
     refined_global_faces=local_to_global[target_faces]
     need(refined_global_faces.shape==(6690,3),"refined target face remap is not6690 rows")
     appended=np.concatenate([original_positions,midpoint_world.astype(original_positions.dtype)[None,:]],axis=0)
     need(np.array_equal(appended[:-1].view("u1"),original_positions.view("u1")),"pack old points changed while appending predicted midpoint")
     new_surfaces=dict(surfaces); new_target=dict(surfaces[KEY]); new_target["faces"]=refined_global_faces; new_surfaces[KEY]=new_target
     injections.append({"step":step,"pack_sha256":sha(p),"receipt_sha256":sha(receipt_path),"preserved_captured_old_vertex_count":3974,"appended_offline_midpoint_id":new_global_id,"target_face_count":6690,"source_face_lineage":"final 1217 stable64 row, bound to 1217 validation report"})
     return appended,new_surfaces,counts
    clearance._pack_surfaces=read_with_refined_64
    return ci,clearance,validator,core
   runner.load_predicates=patched_load_predicates
   return runner
  w.load_runner=patched_load_runner
  return w
 # Preserve the original loader as the base, then patch only runtime modules returned by the existing 1172 owner.
 audit.load_wrapper_original=audit.load_wrapper
 audit.load_wrapper=patched_load_wrapper
 sys.argv=[str(Path(__file__).resolve()),"--scan","--out",str(OUT)]
 target_inputs=[Path(__file__),TISS,MAN,BASE_TISS,BASE_MAN,CTISS,CMAN,HELPER,TARGET_VALIDATION,VALIDATOR,BASE_AUDIT,PRODUCER_DECL,PRODUCER_EXEC,COMMON_CLEARANCE]
 before={str(Path(p).resolve()):sha(p) for p in target_inputs}
 rc=audit.main()
 need(rc==0,"existing 1206 scan owner returned nonzero "+str(rc))
 summary_path=OUT/"summary.json"; summary=json.loads(summary_path.read_text())
 need(len(summary.get("pose_results",[]))==len(HELDOUT),"existing scan did not finish all six heldout poses")
 # Include the newly used target assets in a separate bounded binding receipt; original scan summary stays untouched.
 after={p:sha(p) for p in before}
 need(before==after,"refined target input changed during heldout scan")
 artifacts={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
 binding={"schema":"numi.human.refined-stable64-heldout-target-binding.v1","status":"complete_offline_target_bound_to_exact_heldout_scan","audit_summary":pin(summary_path),"audit_summary_status":summary.get("status"),"heldout_steps":list(HELDOUT),"target_key":[51005,64],"target_counts":{"vertices":3975,"faces":6690},"target_nhtiss":pin(TISS),"target_manifest":pin(MAN),"source_nhtiss_039":pin(BASE_TISS),"producer_context_payload":pin(CTISS),"producer_context_manifest":pin(CMAN),"helper_source":pin(HELPER),"1217_actual_capture_validation":pin(TARGET_VALIDATION),"target_injection_calls":injections,"target_coordinate_policy":"For each held-out 1191 state, keep the 3974 original GPU-captured stable64 vertex coordinates byte-for-byte. Append only midpoint vertex predicted offline from the final NHTISS source record, its local binding weights, and that step's accepted registered-body transforms. Replace stable64 faces in-memory with the exact 6690-face refined NHTISS stream mapped through old source-face identity. This is offline prediction, not a new native capture.","scan_artifacts":artifacts,"extra_inputs_before":before,"extra_inputs_after":after,"extra_inputs_unchanged":True,"qualification":"Held-out exact skin geometry audit only; no native run, continuous-time claim, or physics qualification."}
 side=HERE/"heldout-binding.json";side.write_text(json.dumps(binding,indent=2,sort_keys=True,allow_nan=False)+"\n")
 print(json.dumps({"status":binding["status"],"summary":str(summary_path),"binding":str(side),"binding_sha256":sha(side),"pose_results":[{k:r.get(k) for k in ("step","all_skin_crossing_pair_count","skin_self_crossing_pair_count","pair_coverage_complete")} for r in summary["pose_results"]]},sort_keys=True),flush=True)
 return 0
if __name__=="__main__": raise SystemExit(main())
