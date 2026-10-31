#!/usr/bin/env python3
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
RUN=H/"d-shortedge-quotient-414-batch001-volume-transfer"
REP=H/"d-shortedge-quotient-414-batch001-surface-proposal/report.json"
RAW=H/"d-shortedge-quotient-414-batch001-surface-proposal/candidate-D-short-edge-collapse-raw.npz"
PARENT=H/"d-shortedge-quotient-414-next-001-preflight-attempt004/identity-seed-D.npz"
ULTIMATE=H/"fat-comp0-connected-warning-repair-412/run-009-final-D411-comp1-flips-exact-adapter/candidate-D-for-exact-appended-audit.npz"
BINDER=H/"d-shortedge-quotient-351-run002/bind_short_edge_typed_351.py"
OUT=RUN/"typed-rebind-419"
PINS={
 REP:"4d4be7ee3d713f8cc7a4a4e52e0adff63e1c021d710a74015aa24da223899653",
 RAW:"60f89988727fd00d272aaf2bed798e0f19e9f91f98067c308408a935c183cacc",
 PARENT:"fba94572f52f51e58902063da181d24fcdf47473d0b322d68370277d9ab32c03",
 ULTIMATE:"707bdf3ac95fd0dc9539c148b9bad8e31207dab515d75cd21ebe039a38cb2703",
}
def sha(p):
 with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
for p,h in PINS.items():
 got=sha(p)
 if got!=h:raise ValueError(f"pinned input changed {p}: {got} != {h}")
seed=np.load(PARENT,allow_pickle=False);parent=np.load(ULTIMATE,allow_pickle=False);raw=np.load(RAW,allow_pickle=False)
assert np.array_equal(seed["points_world_m"],parent["points_world_m"])
assert np.array_equal(seed["points_world_m_f32"],parent["points_world_m_f32"])
assert np.array_equal(seed["faces"],parent["faces"])
assert str(raw["immediate_parent_candidate_sha256"][0])==PINS[PARENT]
assert str(raw["typed_parent_candidate_sha256"][0])==PINS[ULTIMATE]
report=json.loads(REP.read_text())
assert report["status"]=="PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL"
assert report["physical_admission"] is False and report["native_steps"]==0
assert report["accepted_collapse_count"]==2363 and len(report["accepted_operations"])==2363
assert {x["role"] for x in report["accepted_operations"]}=={"D"}
ctx=report["joint_changed_vs_complete_OID_walls_all185"]
assert ctx["pair_count"]==0 and ctx["new_face_self_pairs"]==0
adapt=json.loads(json.dumps(report))
adapt["outputs"]={"D":{"path":str(RAW),"sha256":PINS[RAW],"parent":str(PARENT),"parent_sha256":PINS[PARENT]}}
adapt["joint_complete_surface_walls_all185_pairs"]=int(ctx["pair_count"])
adapt["joint_changed_self_pairs"]=int(ctx["new_face_self_pairs"])
adapt["quality_metric"]="joint_volume_shape_apex_upper_bound"
adapt["construction_surface_floor"]=1e-5
adapt["joint_bound_owner_sha256"]="5df00334a92311b4cc0dd24dc7f7d1127f55fd354777ff888eabc607484c8e82"
adapt["typed_binding_scope"]="D only; O/I are exact identity layers"
adapt["typed_parent_chain"]={"immediate_identity_seed":{"path":str(PARENT),"sha256":PINS[PARENT]},"ultimate_typed_D414_parent":{"path":str(ULTIMATE),"sha256":PINS[ULTIMATE]}}
for op in adapt["accepted_operations"]:
 op["typed_parent_union"]=op["complete_typed_parent_union"]
 op["new_faces"]=op["new_faces_current_point_rows"]
 rows=np.asarray(op["new_output_face_rows"],dtype=np.int64)
 assert np.array_equal(raw["faces"][rows],np.asarray(op["new_faces"],dtype=np.int64))
OUT.mkdir(parents=True,exist_ok=False)
config=OUT/"adapter-input.json"
config.write_text(json.dumps(adapt,sort_keys=True,indent=2,allow_nan=False)+"\n")
spec=importlib.util.spec_from_file_location("typed_binder_414",BINDER)
binder=importlib.util.module_from_spec(spec);sys.modules[spec.name]=binder;spec.loader.exec_module(binder)
binder.apply(config,OUT/"typed-owner")
typereceipt=OUT/"typed-owner/receipt.json"
tr=json.loads(typereceipt.read_text())
entry=tr["candidate_roles"]["D"]
receipt={"schema":"h706.d-short-edge-414-typed-rebind.v1","status":"PASS_D_TYPED_FACE_UNION_BINDING","source_script_path":str(Path(__file__)),"source_script_sha256":sha(Path(__file__)),"surface_report_path":str(REP),"surface_report_sha256":PINS[REP],"raw_candidate_path":str(RAW),"raw_candidate_sha256":PINS[RAW],"immediate_typed_identity_seed_path":str(PARENT),"immediate_typed_identity_seed_sha256":PINS[PARENT],"ultimate_typed_D_path":str(ULTIMATE),"ultimate_typed_D_sha256":PINS[ULTIMATE],"adapter_input_path":str(config),"adapter_input_sha256":sha(config),"existing_typed_binder_path":str(BINDER),"existing_typed_binder_sha256":sha(BINDER),"binder_receipt_path":str(typereceipt),"binder_receipt_sha256":sha(typereceipt),"typed_candidate_path":entry["path"],"typed_candidate_sha256":entry["sha256"],"accepted_operation_count":2363,"generated_face_count":8320,"coordinates_changed":False,"source_context_pair_count":0,"replacement_self_pair_count":0,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
rp=OUT/"receipt.json";rp.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
print(json.dumps({"status":receipt["status"],"receipt_path":str(rp),"receipt_sha256":sha(rp),"typed_candidate_path":entry["path"],"typed_candidate_sha256":entry["sha256"],"typed_binder_receipt_sha256":sha(typereceipt)},sort_keys=True))
