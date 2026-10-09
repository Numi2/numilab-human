#!/usr/bin/env python3
"""Fail-closed standalone 20 s preflight wrapper around the retained 1171 exact audit core.

This validates attempt3's closed run and 8 accepted captures. It does not pretend the run is
a registered P18 arm and does not modify the 1171 registered-arm CLI gate.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path

E=Path("/Users/n/numi-human-resting-evidence-20261005")
ATTEMPT=E/"final-native-scene-preflight-936/skin-927-lung-1178-viewer-018-v015-attempt3"
RUN=ATTEMPT/"native-run"
COMP=E/"native-lung-free-apex-two-family-composition-1178/composition-024-attempt5"
NHA=COMP/"final/resting-thorax.nhanatomy"
RESP=COMP/"final/resting-reference-respiration.json"
COMP_REPORT=COMP/"composition-report.json"
LINEAGE=COMP/"current-reciprocal-map-report-v2.json"
CORE=E/"native-lung-late-pose-audit-runner-1171/audit_lung_cycle_1159.py"
CORE_SHA="37bd87c0ac870e59141422db5f6cd8be9b16452155472e61bd84840bc8f8583a"
BASE_SHA="791e2acdfdf917cd8554d83887f78682121a181216a07ff96c7ca2b985214345"
STEPS=[0,4991,5375,5759,6111,6495,7743,10000]
DT=0.0020000000949949026
PINNED={
 str(ATTEMPT/"assembly-preflight.json"):"b08a3f0a0f86addde7d88220cca7cccaaab999f9176a1b94a8be3a6c45e915d0",
 str(ATTEMPT/"launch-command.sh"):"62f044769e9a3c3ce5cfb1ef50aaff77f8ecf9d0d39c600d2aa0f92d27d547e2",
 str(RUN/"invocation.json"):"4e99e30555a4fa64cfcdc3611413964ecc899924e815642d4e23dc0fc3faa76c",
 str(RUN/"run-metadata.json"):"488a9b2906aae625acb7026b141046e6c4cce3ec72711cca80b506b7ce02024f",
 str(RUN/"native.log"):"320a81061eacaa006c96fdf889e786381df6bcb9bb62b465f2f153671972d4a7",
 str(RUN/"resting-coupled.csv"):"5c4f129103fbf23010cfe47b99367c30d8497f70e525b4c97ca61b53c3ca90e6",
 str(ATTEMPT/"comparison-1173/comparison.json"):"bf8a4ead892f5cd319a28316b6703e4cddfd415f4faae0ed0ceb093c13304500",
 str(NHA):"1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241",
 str(RESP):"70f4d235b988f9466b80e795ac0e793a82d6b13d03ebb01b84a313f0ad398e8a",
 str(COMP_REPORT):"ddbfae12a5acd5c35521cb913121f08e49fb8c336c89b7dd60df3272ec0a7fbc",
 str(LINEAGE):"fceac1a821d6ad7c368433dd9d73c0071423275ab645b6a49e2d50e18ee5e4ac",
}
RUNTIME_PATH="/Users/n/numi-human-performance-build-014/lib/libmetalrobo.dylib"
RUNTIME_SHA="6bccfc4044d825423e66bc2f60ba3cf59eaa9a4936773ab08182f58a09927092"
REFERENCE="/Users/n/numi-human-retained-delivery-20261009/native-terminal-cycle-1173"

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(4*1024*1024),b""): h.update(b)
 return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())
def option(argv,name):
 if argv.count(name)!=1: raise ValueError("expected one "+name)
 i=argv.index(name)
 if i+1>=len(argv): raise ValueError("missing "+name)
 return argv[i+1]

def validate_documents(meta,inv,cmp,expected_csv_sha,asset_hasher=None):
 if meta.get("exit_code")!=0: raise ValueError("native run did not exit 0")
 if meta.get("argv")!=inv.get("argv"): raise ValueError("invocation differs from closed metadata")
 if meta.get("source_files_changed_during_run")!=[]: raise ValueError("source changed during run")
 rt=meta.get("loaded_metal_runtime",{})
 if rt.get("verified") is not True or rt.get("expected_path")!=RUNTIME_PATH or rt.get("expected_sha256")!=RUNTIME_SHA:
  raise ValueError("loaded runtime is not the pinned unchanged runtime")
 argv=inv.get("argv",[])
 if not argv or Path(argv[0]).resolve()!=Path("/Users/n/numi-human-retired-alias-visibility-build-018-attempt2/bin/numi-human-native"):
  raise ValueError("viewer executable mismatch")
 if int(option(argv,"--muscle-step-count"))!=10000 or abs(float(option(argv,"--muscle-step-seconds"))-0.002)>1e-12:
  raise ValueError("not the pinned 10000-root, 2 ms standalone preflight")
 if len(argv)<5 or Path(argv[4]).resolve()!=RUN.resolve(): raise ValueError("run output path differs")
 assets=inv.get("asset_sha256")
 if not isinstance(assets,dict) or not assets: raise ValueError("invocation lacks asset hashes")
 for path,digest in assets.items():
  actual=asset_hasher(path) if asset_hasher else sha(path)
  if actual!=digest: raise ValueError("asset hash mismatch: "+path)
 if Path(cmp.get("candidate","")).resolve()!=RUN.resolve(): raise ValueError("comparison candidate mismatch")
 if cmp.get("status")!="measured_no_parity_assumed" or cmp.get("candidate_observations")!=1250:
  raise ValueError("1183 cadence comparison incomplete")
 if cmp.get("candidate_csv_sha256")!=expected_csv_sha: raise ValueError("CSV hash mismatch")
 identity=cmp.get("reference_identity",{})
 if identity.get("run_path")!=REFERENCE or identity.get("historical_metadata_reconstructed") is not False:
  raise ValueError("comparison is not bound to independent 1173 reference")
 cadence=cmp.get("trace_cadence_comparison",{})
 seq=cadence.get("candidate_accepted_steps",[])
 if cadence.get("all_values_exact") is not True or len(seq)!=1250 or seq[0]!=8 or seq[-1]!=10000 or any(b-a!=8 for a,b in zip(seq,seq[1:])):
  raise ValueError("1250-row accepted cadence comparison failed")
 terminal=cmp.get("terminal_log_proof",{})
 if terminal.get("accepted_steps")!=10000 or terminal.get("terminal_presentations")!=1:
  raise ValueError("single terminal accepted state not proven")
 return True

def validate_attempt():
 for p,h in PINNED.items():
  if not Path(p).is_file() or Path(p).is_symlink() or sha(p)!=h: raise ValueError("pinned input mismatch: "+p)
 if sha(CORE)!=CORE_SHA: raise ValueError("1171 audit core hash mismatch")
 meta,inv,cmp=read(RUN/"run-metadata.json"),read(RUN/"invocation.json"),read(ATTEMPT/"comparison-1173/comparison.json")
 validate_documents(meta,inv,cmp,sha(RUN/"resting-coupled.csv"))
 if not 0<meta.get("wall_seconds",0)<1200: raise ValueError("invalid bounded-run wall time")
 assembly=read(ATTEMPT/"assembly-preflight.json")
 if assembly.get("status")!="assembled_owner_cli_validated_native_not_run": raise ValueError("preflight assembly status mismatch")
 if assembly.get("native_receipt",{}).get("NHA_sha256")!=PINNED[str(NHA)]: raise ValueError("assembly NHA mismatch")
 if assembly.get("respiration_config",{}).get("sha256")!=PINNED[str(RESP)]: raise ValueError("assembly config mismatch")
 accepted=RUN/"accepted-geometry"
 receipts=sorted(accepted.glob("step-*.receipt.json"),key=lambda p:int(p.name.split("-")[1].split(".")[0]))
 steps=[int(p.name.split("-")[1].split(".")[0]) for p in receipts]
 if steps!=STEPS: raise ValueError("accepted capture receipt set mismatch")
 receipt_pins=[]; pack_pins=[]
 for step,rp in zip(STEPS,receipts):
  d=read(rp)
  if d.get("accepted_step")!=step or d.get("physical_endpoint")!="accepted": raise ValueError("capture not accepted: "+str(step))
  if d.get("surface_audit_endpoint")!="passed" or d.get("surface_audit",{}).get("physical_endpoint")!="accepted": raise ValueError("capture surface gate failed: "+str(step))
  audit=d.get("surface_audit",{})
  if audit.get("mesh_zero_area_triangles")!=0 or audit.get("mesh_nonfinite_area_triangles")!=0: raise ValueError("captured mesh has invalid triangles: "+str(step))
  pack=accepted/("step-%d.mrvpack"%step)
  if not pack.is_file() or sha(pack)!=d.get("pack_file_sha256"): raise ValueError("capture pack hash mismatch: "+str(step))
  if Path(d.get("accepted_pack_path","")).resolve()!=pack.resolve(): raise ValueError("capture receipt points elsewhere")
  if abs(float(d.get("accepted_time_s",-1))-step*DT)>1e-6: raise ValueError("capture time mismatch: "+str(step))
  receipt_pins.append({"step":step,"path":str(rp),"sha256":sha(rp),"captured_vertex_buffer_sha256":d.get("captured_vertex_buffer_sha256")})
  pack_pins.append({"step":step,"path":str(pack),"sha256":sha(pack),"bytes":pack.stat().st_size})
 pins={p:sha(p) for p in PINNED}
 pins.update({p:sha(p) for p in inv["asset_sha256"]})
 pins.update({x["path"]:x["sha256"] for x in receipt_pins})
 pins.update({x["path"]:x["sha256"] for x in pack_pins})
 return {"schema":"numi.human.standalone-short-preflight-admission.v1","status":"PASS_standalone_20s_only",
  "registered_trial":False,"scope":"Exact discrete scan over the closed standalone 20 s attempt3; not a registered study arm and not 310 s qualification.",
  "accepted_steps":STEPS,"roots":10000,"dt_seconds":0.002,"wall_seconds":meta["wall_seconds"],
  "cadence_rows":1250,"comparison_reference":REFERENCE,"loaded_runtime":meta["loaded_metal_runtime"],
  "nha":{"path":str(NHA),"sha256":sha(NHA)},"respiration_config":{"path":str(RESP),"sha256":sha(RESP)},
  "composition_report":{"path":str(COMP_REPORT),"sha256":sha(COMP_REPORT)},
  "current_reciprocal_map_report_v2":{"path":str(LINEAGE),"sha256":sha(LINEAGE)},
  "run_metadata_sha256":sha(RUN/"run-metadata.json"),"invocation_sha256":sha(RUN/"invocation.json"),
  "comparison_sha256":sha(ATTEMPT/"comparison-1173/comparison.json"),"capture_receipts":receipt_pins,
  "capture_packs":pack_pins,"pinned_inputs":pins}

def import_core():
 if sha(CORE)!=CORE_SHA: raise ValueError("1171 exact audit core changed")
 spec=importlib.util.spec_from_file_location("native_lung_audit_1171",CORE)
 if not spec or not spec.loader: raise ImportError("cannot import 1171 core")
 mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
 if sha(mod.BASE)!=BASE_SHA: raise ValueError("1171 exact intersection base changed")
 return mod

def self_test():
 goodmeta={"exit_code":0,"argv":["/Users/n/numi-human-retired-alias-visibility-build-018-attempt2/bin/numi-human-native","rigid","muscle","bones",str(RUN),"--muscle-step-seconds","0.002","--muscle-step-count","10000"],
  "source_files_changed_during_run":[],"loaded_metal_runtime":{"verified":True,"expected_path":RUNTIME_PATH,"expected_sha256":RUNTIME_SHA}}
 goodinv={"argv":list(goodmeta["argv"]),"asset_sha256":{"fixture":"fixture"}}
 goodcmp={"candidate":str(RUN),"status":"measured_no_parity_assumed","candidate_observations":1250,"candidate_csv_sha256":"fixture",
  "reference_identity":{"run_path":REFERENCE,"historical_metadata_reconstructed":False},
  "trace_cadence_comparison":{"all_values_exact":True,"candidate_accepted_steps":list(range(8,10001,8))},
  "terminal_log_proof":{"accepted_steps":10000,"terminal_presentations":1}}
 import copy
 validate_documents(goodmeta,goodinv,goodcmp,"fixture",asset_hasher=lambda p:"fixture")
 cases=[
  (lambda m,i,c:m.update(exit_code=1),"failed exit"),
  (lambda m,i,c:m.update(loaded_metal_runtime={"verified":False}),"runtime mismatch"),
  (lambda m,i,c:i.update(argv=i["argv"]+["changed"]),"invocation mismatch"),
  (lambda m,i,c:i.update(asset_sha256={"fixture":"wrong"}),"asset hash mismatch"),
  (lambda m,i,c:c.update(candidate_observations=1249),"incomplete cadence"),
  (lambda m,i,c:c.update(status="incomplete"),"failed comparison"),
  (lambda m,i,c:c.update(candidate_csv_sha256="wrong"),"CSV mismatch"),
  (lambda m,i,c:c["trace_cadence_comparison"].update(all_values_exact=False),"cadence mismatch"),
  (lambda m,i,c:c["terminal_log_proof"].update(accepted_steps=9999),"terminal mismatch"),
  (lambda m,i,c:c["reference_identity"].update(historical_metadata_reconstructed=True),"historical reconstruction"),
 ]
 for mutate,label in cases:
  m,i,c=copy.deepcopy(goodmeta),copy.deepcopy(goodinv),copy.deepcopy(goodcmp); mutate(m,i,c)
  try: validate_documents(m,i,c,"fixture",asset_hasher=lambda p:"fixture")
  except ValueError: continue
  raise AssertionError("accepted "+label)
 return {"status":"PASS","same_validator_used":True,"negative_cases":len(cases)}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--validate-only",action="store_true")
 p.add_argument("--out",type=Path)
 p.add_argument("--self-test",action="store_true")
 a=p.parse_args()
 if a.self_test:
  print(json.dumps(self_test(),sort_keys=True)); return 0
 admission=validate_attempt()
 core=import_core()
 cfg={"run":RUN,"out":a.out or E/"native-lung-standalone-short-preflight-1184/unused",
  "nha_path":NHA,"nha_sha":PINNED[str(NHA)],"map_reports":[],"d_map_composition_report":COMP_REPORT,
  "lobe_lineage_report":LINEAGE,"geometry_only_area_mismatch":False,"workers":1,"probe_report":None,"registered_arm":None,"base":core.BASE}
 base,adapters,ctx=core.prepare_worker_context(cfg)
 if ctx.get("registered_trial_path") is not None or ctx.get("registered_context") is not None: raise ValueError("standalone adapter entered registered context")
 if ctx["steps"]!=STEPS or ctx["nha_sha"]!=PINNED[str(NHA)]: raise ValueError("owner context differs from standalone pins")
 if ctx["area_binding"]["status"]!="PASS_exact_geometry_and_config_binding": raise ValueError("selected NHA/config area binding failed")
 admission["owner_context"]={"registered_trial":False,"steps":ctx["steps"],"nha_sha256":ctx["nha_sha"],
  "D_map_pairs":ctx["d_map_doc"]["declared_pairs"],"D_map_counts_by_lobe":ctx["d_map_doc"]["counts_by_lobe"],
  "lobe_lineage_pairs":ctx["lineage"]["declared_pairs"],"source_lineage_validation":ctx["lineage"]["source_simplex_validation"],
  "area_binding_status":ctx["area_binding"]["status"]}
 if a.validate_only:
  print(json.dumps({"status":"PASS_standalone_short_preflight_admitted_no_scan","admission":admission},sort_keys=True)); return 0
 if not a.out:p.error("--out required for a scan")
 out=a.out.resolve()
 if E not in out.parents or out.exists():raise ValueError("scan output must be a fresh directory under evidence root")
 tracked=set(Path(p).resolve() for p in admission["pinned_inputs"])
 tracked.update(Path(p).resolve() for p in ctx["tracked"])
 tracked.update((CORE.resolve(),Path(core.BASE).resolve(),Path(__file__).resolve()))
 for item in admission["capture_receipts"]+admission["capture_packs"]:
  tracked.add(Path(item["path"]).resolve())
 before={str(p):sha(p) for p in sorted(tracked)}
 cfg["out"]=out
 core.prepare_final(cfg)
 after={str(p):sha(p) for p in sorted(tracked)}
 rep=json.loads((out/"report.json").read_text())
 input_unchanged=(before==after)
 scan_complete=rep["complete_pair_coverage"] is True and len(rep["pose_results"])==8
 geometry_pass=(rep["geometry_scan_status"]=="PASS_exact_discrete_predicates"
  and rep["physiology_integration_status"]=="PASS_area_binding"
  and rep["unallowed_self_pair_total"]==0 and rep["unclassified_cross_pair_total"]==0
  and rep["degenerate_face_seen"] is False and scan_complete and input_unchanged)
 admission["status"]="COMPLETE_standalone_short_scan" if scan_complete else "INCOMPLETE_standalone_short_scan"
 admission["registered_trial"]=False
 admission["geometry_scan_status"]="PASS_exact_discrete_predicates" if geometry_pass else "FAIL_or_incomplete"
 admission["physiology_integration_status"]=rep["physiology_integration_status"]
 admission["input_hashes_unchanged"]=input_unchanged
 admission["tracked_input_count"]=len(before)
 admission["tracked_hashes_before"]=before
 admission["tracked_hashes_after"]=after
 admission["audit_core"]={"path":str(CORE),"sha256":sha(CORE),"base_path":str(core.BASE),"base_sha256":sha(core.BASE),
  "wrapper_path":str(Path(__file__).resolve()),"wrapper_sha256":sha(__file__),"report_path":str(out/"report.json"),
  "report_sha256":sha(out/"report.json"),"status":rep["status"],"accepted_steps":rep["accepted_steps"],
  "full_pair_coverage":rep["complete_pair_coverage"],"self_unallowed":rep["unallowed_self_pair_total"],
  "cross_unclassified":rep["unclassified_cross_pair_total"],"degenerate_face_seen":rep["degenerate_face_seen"]}
 (out/"standalone-admission.json").write_text(json.dumps(admission,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"status":admission["status"],"geometry_scan_status":admission["geometry_scan_status"],
  "registered_trial":False,"out":str(out),"full_pair_coverage":rep["complete_pair_coverage"],
  "self_unallowed":rep["unallowed_self_pair_total"],"cross_unclassified":rep["unclassified_cross_pair_total"],
  "input_hashes_unchanged":input_unchanged},sort_keys=True))
 return 0 if geometry_pass else 3

if __name__=="__main__": raise SystemExit(main())

