#!/usr/bin/env python3
"""Screen one inferred Dirichlet skin-weight field at the retained 1201 terminal capture.

This is counterfactual only: starting from the captured 1187 skin coordinates, it adds
(candidate LBS - baseline LBS) reconstructed with the pinned 1182 transform owner, then
runs the unchanged exact 1172 all-target and self-intersection predicates. It does not
modify a native pack or claim an actual capture under the candidate weights.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, os, resource, sys, time
from pathlib import Path
for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","VECLIB_MAXIMUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
 os.environ[k]="1"
import numpy as np

ROOT=Path("/Users/n/numi-human-retained-delivery-20261009/skin-source-binding-1202")
OUT=ROOT/"counterfactual-screen-terminal-1201"
RUN=Path("/Users/n/numi-human-retained-delivery-20261009/native-flat-reference-40s-1201/native-run")
BASE_SKIN=Path("/Users/n/numi-human-resting-evidence-20261005/native-skin-epl143-clearance-1187/attempt-006/bodyparts3d-myosim-skinned-shell-epl143-candidate.nhskin")
BASE_SKIN_SHA="b2d235e32c1c7d7f753eb83d1e8e9d045a1fd62be9c5c6da65dfde8844e2622b"
CANDIDATE=ROOT/"counterfactual-screen-input/candidate-dirichlet.nhskin"
CANDIDATE_SHA="f7c8fdb1990bed9ab1fe8355bfc432043d268bec1b858e379d7f893ff51567df"
CANDIDATE_BUILD=ROOT/"counterfactual-screen-input/candidate-build-report.json"
DIR_NPZ=ROOT/"dirichlet/bodyparts3d-skin-binding-solution.npz"
DIR_NPZ_SHA="8ee440a0e346ba253c32cf693661b889c622908cd7355c13ef50ae1d9187aae8"
DIR_REPORT=ROOT/"dirichlet/dirichlet-reconstruction-report.json"
DIR_REPORT_SHA="a979ed813e20544f4f0b9e9e03784d652abb5440dd60939107c360e5ca978803"
GRAPH=ROOT/"source-surface-binding-inputs.npz"
GRAPH_SHA="85ceca04ba63fc463a4515579719fe82ad03137c98891647b75e472461c96625"
BONES=ROOT/"source-surface-binding-bones.json"
BONES_SHA="4a7c19baa4f3e36d6422288e0c91d9e326d393b1a3493a2b2097f5ac6e4df42c"
AUDIT_MODULE=Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/skin_surface_audit.py")
AUDIT_MODULE_SHA="dbd72470bd2cfdaa398ba5d253393b755d84b5cdfd25b9dd1952bde1120d5f46"
BUILD_SCRIPT=ROOT/"build_dirichlet_candidate.py"; BUILD_SCRIPT_SHA="d7b9c60bc7cb2abed0ef852a7036ac5ff3d2482b87128594eeb7666d30be835e"
DIR_SOURCE_SCRIPT=ROOT/"reconstruct_dirichlet_solution.py"; DIR_SOURCE_SCRIPT_SHA="741d42e9d866b53581dc32004e73835dbdfa6c72b1099720d80700abd4824b13"
BINDING_OWNER=Path("/Users/n/numi-human-free-apex-two-family-1178/src/numilab_human/skin_surface_binding.py"); BINDING_OWNER_SHA="33039db6031150e20a1fc1f3bba7e3d594f608dc0adcfc4a6de368a79c4102e7"
RECON=Path("/Users/n/numi-human-resting-evidence-20261005/native-vastus-lateralis-weight-row-erratum-1182/reproduce.py")
RUNNER=Path("/Users/n/numi-human-resting-evidence-20261005/native-lung-late-skin-audit-runner-1172/revision-003/audit_full_skin_cycle_1172.py")
RUNNER_SHA="b1b9be5959610325067f703c32bd55e2f568c11124ec232f0d066ed75a6fdcbb"
INV=Path("/Users/n/numi-human-resting-evidence-20261005/native-complete-skin-containment-audit-890/pair-summary-v3.csv")
INV_SHA="a43484aa8d65b207cc498bb470097ec5f06ef2bb9872837bde9f197e19e815e3"
RUN_DECL=Path("/Users/n/numi-human-retained-delivery-20261009/native-flat-reference-40s-1201/run-declaration.json")
RUN_DECL_SHA="c907c5ea7d1ed6027da60f27a0b07f498fd6499989b9051349688241b5656085"
BASE_RESULT=Path("/Users/n/numi-human-retained-delivery-20261009/native-flat-reference-40s-1201/skin-audit-1201-terminal/step-20000.result.json")
BASE_RESULT_SHA="9703c87b6e0fcb2e8a7fc0b8f25eaf8124659d3db0712f4b9a5f83a9cad1ede6"
NHA_SHA="1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241"
STEP=20000

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(4*1024*1024),b""): h.update(block)
 return h.hexdigest()
def require(ok,msg):
 if not ok: raise RuntimeError(msg)
def checked(p, expected=None):
 p=Path(p).resolve()
 require(p.is_file() and not p.is_symlink(),f"missing/nonregular input: {p}")
 actual=sha(p)
 if expected is not None: require(actual==expected,f"hash mismatch: {p}: {actual} != {expected}")
 return actual
def load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 require(spec is not None and spec.loader is not None,f"cannot import {path}")
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def write_json(p,obj):
 Path(p).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")

require(not OUT.exists(),"refusing to overwrite terminal screen output")
preflight_command=Path("/Users/n/numi-human-retained-delivery-20261009/native-flat-reference-40s-1201/audit_skin_1201.py")
pins={
 str(BASE_SKIN):BASE_SKIN_SHA,str(CANDIDATE):CANDIDATE_SHA,str(CANDIDATE_BUILD):checked(CANDIDATE_BUILD),
 str(DIR_NPZ):DIR_NPZ_SHA,str(DIR_REPORT):DIR_REPORT_SHA,str(GRAPH):GRAPH_SHA,str(BONES):BONES_SHA,
 str(AUDIT_MODULE):AUDIT_MODULE_SHA,str(BUILD_SCRIPT):BUILD_SCRIPT_SHA,str(DIR_SOURCE_SCRIPT):DIR_SOURCE_SCRIPT_SHA,str(BINDING_OWNER):BINDING_OWNER_SHA,str(RECON):checked(RECON),str(RUNNER):RUNNER_SHA,str(INV):INV_SHA,
 str(RUN_DECL):RUN_DECL_SHA,str(BASE_RESULT):BASE_RESULT_SHA,str(preflight_command):checked(preflight_command),
 str(RUN/"run-metadata.json"):checked(RUN/"run-metadata.json"),
 str(RUN/"invocation.json"):checked(RUN/"invocation.json"),
 str(RUN/"native.log"):checked(RUN/"native.log"),
 str(RUN/"accepted-geometry"/f"step-{STEP}.mrvpack"):checked(RUN/"accepted-geometry"/f"step-{STEP}.mrvpack"),
 str(RUN/"accepted-geometry"/f"step-{STEP}.receipt.json"):checked(RUN/"accepted-geometry"/f"step-{STEP}.receipt.json"),
}
with open(RUN/"invocation.json") as f: invocation=json.load(f)
assets=invocation.get("asset_sha256")
require(isinstance(assets,dict) and len(assets)==25,"1201 native invocation asset map changed")
for p,digest in assets.items():
 pins[p]=checked(Path(p),digest)

build=json.loads(CANDIDATE_BUILD.read_text())
require(build.get("status")=="source_verified_offline_candidate_not_native_admitted","candidate source-build report status mismatch")
require(build.get("output",{}).get("sha256")==CANDIDATE_SHA,"candidate payload/hash not bound")
candidate_bytes=CANDIDATE.read_bytes()
npz=__import__("numpy").load(DIR_NPZ,allow_pickle=False)
candidate_skin=load_module(RECON,"transform_owner_1182_screen")
base=candidate_skin.parse_skin(BASE_SKIN)
cand=candidate_skin.parse_skin(CANDIDATE)
require(base["binding_count"]==cand["binding_count"]==86 and base["vertex_count"]==cand["vertex_count"]==54949,"NHSKIN ABI5 dimensions changed")
require(base["indices"]==cand["indices"],"candidate triangle index sequence changed")
require(base["bindings"]==cand["bindings"],"candidate binding table changed")
require(base["source_sha256"]==cand["source_sha256"],"candidate NHSKIN header/source identity changed")
require(np.array_equal(np.asarray(base["vertices"],dtype=object)[:,:6],np.asarray(cand["vertices"],dtype=object)[:,:6]),"candidate position/normal records changed")
with np.load(DIR_NPZ,allow_pickle=False) as z:
 w=np.asarray(z["full_weights"],dtype="<f8")
require(w.shape==(cand["vertex_count"],cand["binding_count"]),"Dirichlet full matrix dimensions mismatch")
require(np.array_equal(np.frombuffer(cand["raw"],dtype="<f4",count=cand["vertex_count"]*cand["binding_count"],offset=cand["full_offset"]).reshape(w.shape),np.asarray(w,dtype="<f4")),"candidate ABI5 full field does not match Dirichlet matrix")
receipt=RUN/"accepted-geometry"/f"step-{STEP}.receipt.json"
body_pose,receipt_doc=candidate_skin.pose_map(receipt)
require(receipt_doc.get("accepted_step")==STEP and abs(float(receipt_doc.get("accepted_time_s",-1))-40.0)<1e-5,"1201 terminal receipt is not accepted step 20000 at 40 s")
require(len(body_pose)==86,"1201 terminal receipt body-pose inventory is not 86")
for binding in base["bindings"]:
 require(int(binding[0]) in body_pose,"NHSKIN binding body index absent from accepted pose")
# Owner transform reconstruction at the one accepted pose, both baseline and candidate weights.
t0=time.monotonic()
baseline_lbs=np.empty((base["vertex_count"],3),dtype=np.float64)
candidate_lbs=np.empty_like(baseline_lbs)
for i in range(base["vertex_count"]):
 baseline_lbs[i]=candidate_skin.skin_point(base,i,body_pose)[0]
 candidate_lbs[i]=candidate_skin.skin_point(cand,i,body_pose)[0]
transform_elapsed=time.monotonic()-t0
# Load the exact native pack's rows and verify pack/NHSKIN topology correspondence before changing positions in memory.
audit=load_module(RUNNER,"audit_core_1172_terminal_screen")
audit.SKIN=CANDIDATE
audit.SKIN_SHA=CANDIDATE_SHA
keys,expected=audit.inventory()
require(len(keys)==859,"pinned target inventory is not 859")
ci,clearance,validator,core=audit.load_predicates()
pack=RUN/"accepted-geometry"/f"step-{STEP}.mrvpack"
receipt_path=pack.with_suffix(".receipt.json")
positions,surfaces,pack_counts=clearance._pack_surfaces(pack,set(keys))
skin_key=audit.EXPECTED_SKIN_KEY
require(skin_key in surfaces,"terminal MRVPACK lacks NHSKIN surface")
skin_faces=np.asarray(surfaces[skin_key]["faces"],dtype=np.int64)
skin_base=int(receipt_doc["skin_source_mapping"]["pack_first_vertex"])
source_indices=np.asarray(cand["indices"],dtype=np.uint32).reshape(-1,3)
require(len(skin_faces)==len(source_indices) and int(skin_faces.min())>=skin_base and int(skin_faces.max())<skin_base+cand["vertex_count"],"captured NHSKIN topology/vertex range differs from source mapping")
require(np.array_equal(skin_faces-skin_base,source_indices),"captured NHSKIN face indices differ from source payload/mapping")
source_ids=np.arange(skin_base,skin_base+cand["vertex_count"],dtype=np.int64)
positions=np.asarray(positions)
require(int(source_ids.max())<len(positions),"NHSKIN source IDs exceed pack position array")
captured_skin=np.asarray(positions[source_ids],dtype=np.float64)
baseline_residual=captured_skin-baseline_lbs
candidate_delta=candidate_lbs-baseline_lbs
counterfactual=np.asarray(captured_skin+candidate_delta,dtype="<f4")
# Counterfactual position adapter: preserve the captured common-field/respiratory residual, change only LBS delta.
_original_pack_surfaces=clearance._pack_surfaces
def pack_surfaces_with_counterfactual(pack_path,target_keys):
 p,s,c=_original_pack_surfaces(pack_path,target_keys)
 require(Path(pack_path).resolve()==pack.resolve(),"unexpected pack requested during terminal screen")
 p=np.asarray(p).copy()
 p[source_ids]=counterfactual
 return p,s,c
clearance._pack_surfaces=pack_surfaces_with_counterfactual
OUT.mkdir()
screen_declaration={
 "schema":"numi.human.dirichlet-skin-weight-terminal-counterfactual.declaration.v1",
 "status":"screen_started",
 "classification":"counterfactual offline screen; not actual native capture, not admission, not a full-cycle result",
 "candidate_skin":{"path":str(CANDIDATE),"sha256":CANDIDATE_SHA,"build_report":str(CANDIDATE_BUILD),"build_report_sha256":pins[str(CANDIDATE_BUILD)]},
 "base_skin":{"path":str(BASE_SKIN),"sha256":BASE_SKIN_SHA},
 "dirichlet_solution":{"path":str(DIR_NPZ),"sha256":DIR_NPZ_SHA,"report":str(DIR_REPORT),"report_sha256":DIR_REPORT_SHA},
 "native_reference":{"run":str(RUN),"step":STEP,"time_s":float(receipt_doc["accepted_time_s"]),"pack":str(pack),"pack_sha256":pins[str(pack)],"receipt":str(receipt),"receipt_sha256":pins[str(receipt)],"invocation_sha256":pins[str(RUN/"invocation.json")],"NHA_sha256":NHA_SHA,"original_skin_sha256":BASE_SKIN_SHA},
 "transform_reconstruction":{"owner":str(RECON),"owner_sha256":pins[str(RECON)],"method":"1182 skin_point(): source NHSKIN rest point -> ABI5 binding transform -> accepted registered body pose -> full F32 binding-weight sum; double accumulation in owner","counterfactual_rule":"captured_skin_F32 + (candidate_LBS - baseline_LBS), then one F32 materialization; preserves the captured residual from common-field/respiratory/runtime sources instead of reconstructing them"},
 "predicates":{"runner":str(RUNNER),"runner_sha256":RUNNER_SHA,"audit_core_sha256":audit.AUDIT_CORE_SHA,"accepted_pack_validator_sha256":audit.HELPER_SHA,"exact_intersection_sha256":audit.PRED_SHA,"skin_pack_reader_sha256":audit.CLEARANCE_SHA,"target_inventory_sha256":INV_SHA,"surface_targets":len(keys),"contact_exemptions":[]},
 "tracked_inputs_before":pins,
 "baseline_reconstruction_residual":{"max_abs_component_m":float(np.max(np.abs(baseline_residual))),"rms_component_m":float(np.sqrt(np.mean(baseline_residual**2))),"p99_vertex_norm_m":float(np.quantile(np.linalg.norm(baseline_residual,axis=1),.99))},
 "candidate_LBS_delta":{"changed_vertex_count":int(np.count_nonzero(np.any(candidate_delta!=0,axis=1))),"max_vertex_shift_m":float(np.max(np.linalg.norm(candidate_delta,axis=1))),"rms_vertex_shift_m":float(np.sqrt(np.mean(candidate_delta**2))),"p99_vertex_shift_m":float(np.quantile(np.linalg.norm(candidate_delta,axis=1),.99))},
 "transform_reconstruction_elapsed_s":transform_elapsed,
 "scope":"One actual accepted 1201 terminal pose only. Candidate coordinates are counterfactual with captured common-field/respiratory residual held fixed. All 859 exact target surfaces, ocular subset, complete skin self-pair coverage and degeneracy checks use unchanged 1172/908 predicates. No actual candidate capture, no source admission, no temporal or physiology qualification."
}
write_json(OUT/"declaration.json",screen_declaration)
print(json.dumps({"status":"scan_started","step":STEP,"targets":len(keys),"transform_s":transform_elapsed,"rss_peak_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)
# Restore and apply exact 1172 accepted pack validation/triangle intersection implementation.
try:
 result=audit.run_pose(STEP,RUN,OUT,ci,clearance,validator,core,keys,expected,NHA_SHA)
finally:
 clearance._pack_surfaces=_original_pack_surfaces
after={p:sha(Path(p)) for p in pins}
unchanged=pins==after
write_json(OUT/"summary.json",{
 "schema":"numi.human.dirichlet-skin-weight-terminal-counterfactual.summary.v1",
 "status":"complete_counterfactual_exact_pair_screen" if result["pair_coverage_complete"] and unchanged else "incomplete_or_input_changed",
 "declaration_sha256":sha(OUT/"declaration.json"),"candidate_skin_sha256":CANDIDATE_SHA,"native_reference_step":STEP,
 "pair_coverage_complete":result["pair_coverage_complete"],"all_skin_crossing_pair_count":result["all_skin_crossing_pair_count"],
 "ocular_crossing_pair_count":result["ocular_crossing_pair_count"],"nonocular_crossing_pair_count":result["nonocular_crossing_pair_count"],
 "skin_self_crossing_pair_count":result["skin_self_crossing_pair_count"],"skin_degenerate_face_count":len(result["skin_degenerate_face_rows"]),
 "invalid_target_surface_count":result["invalid_target_surface_count"],"invalid_target_triangle_count":result["invalid_target_triangle_count"],
 "input_hashes_unchanged":unchanged,"inputs_before":pins,"inputs_after":after,
 "result_path":str(OUT/f"step-{STEP}.result.json"),"result_sha256":sha(OUT/f"step-{STEP}.result.json"),
 "qualification":"Counterfactual terminal pose only. This does not establish actual native candidate geometry, nor a full-cycle clearance result."
})
print(json.dumps({"status":"complete","all_crossings":result["all_skin_crossing_pair_count"],"ocular":result["ocular_crossing_pair_count"],"self":result["skin_self_crossing_pair_count"],"degenerate":len(result["skin_degenerate_face_rows"]),"targets":result["surface_target_count"],"inputs_unchanged":unchanged,"elapsed_s":result["elapsed_s"]},sort_keys=True),flush=True)
