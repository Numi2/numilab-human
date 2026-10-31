#!/usr/bin/env python3
from pathlib import Path
import hashlib, importlib.util, json, time
import numpy as np
H=Path(__file__).resolve().parents[1]
RUN=H/"d-shortedge-quotient-411-batch-final"
REP=RUN/"run001/report.json"
RAW=RUN/"run001/candidate-D-short-edge-collapse-raw.npz"
PARENT=H/"d-shortedge-quotient-409-run001/identity-seed-D.npz"
ULTIMATE=H/"d-shortedge-quotient-385-run006/typed-rebind-003/candidate-D-typed-current.npz"
BINDER=H/"d-shortedge-quotient-351-run002/bind_short_edge_typed_351.py"
OUT=RUN/"typed-rebind-002"
def sha(p):
    with Path(p).open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
pins={
    REP:"adbdea6519969ee1ee77f24e09448d3a2ef53e0971b4fee1a1eebbc689fa8db4",
    RAW:"fd81696c70df46e090bcb81f27f87b593d34207ab2f8f4a58ad32b191e8faf42",
    PARENT:"ba35fe84b620ad235dd9751f3ef23e231c38e994b9e99ec2c27817f20275c729",
    ULTIMATE:"9a5ca486d7b2025a7a2140798155df79c208871703f1eaf75171761b388ba7eb",
}
for p,want in pins.items():
    got=sha(p)
    if got!=want: raise ValueError(f"pinned input changed: {p}: {got}")
seed=np.load(PARENT,allow_pickle=False);parent=np.load(ULTIMATE,allow_pickle=False);raw=np.load(RAW,allow_pickle=False)
assert str(seed["typed_parent_candidate_sha256"][0])==pins[ULTIMATE]
assert np.array_equal(seed["points_world_m"],parent["points_world_m"])
assert np.array_equal(seed["points_world_m_f32"],parent["points_world_m_f32"])
assert np.array_equal(seed["faces"],parent["faces"])
assert str(raw["immediate_parent_candidate_sha256"][0])==pins[PARENT]
assert str(raw["typed_parent_candidate_sha256"][0])==pins[ULTIMATE]
report=json.loads(REP.read_text())
assert report["status"]=="PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL"
assert report["physical_admission"] is False and report["native_steps"]==0
assert report["accepted_collapse_count"]==2015 and len(report["accepted_operations"])==2015
assert {x["role"] for x in report["accepted_operations"]}=={"D"}
ctx=report["joint_changed_vs_complete_OID_walls_all185"]
assert ctx["pair_count"]==0 and ctx["new_face_self_pairs"]==0
adapt=json.loads(json.dumps(report))
adapt["outputs"]={"D":{"path":str(RAW),"sha256":pins[RAW],"parent":str(PARENT),"parent_sha256":pins[PARENT]}}
adapt["joint_complete_surface_walls_all185_pairs"]=int(ctx["pair_count"])
adapt["joint_changed_self_pairs"]=int(ctx["new_face_self_pairs"])
adapt["quality_metric"]="normalized_shape_apex_bound"
adapt["construction_surface_floor"]=1e-5
adapt["typed_binding_scope"]="D only; O/I are exact identity layers"
adapt["typed_parent_chain"]={"immediate_identity_seed":{"path":str(PARENT),"sha256":pins[PARENT]},"ultimate_typed_D185":{"path":str(ULTIMATE),"sha256":pins[ULTIMATE]}}
for op in adapt["accepted_operations"]:
    op["typed_parent_union"]=op["complete_typed_parent_union"]
    op["new_faces"]=op["new_faces_current_point_rows"]
    rows=np.asarray(op["new_output_face_rows"],dtype=np.int64)
    assert np.array_equal(raw["faces"][rows],np.asarray(op["new_faces"],dtype=np.int64))
OUT.mkdir(parents=True,exist_ok=False)
config=OUT/"adapter-input.json"
config.write_text(json.dumps(adapt,sort_keys=True,indent=2,allow_nan=False)+"\n")
spec=importlib.util.spec_from_file_location("typed_binder_411_v2",BINDER)
binder=importlib.util.module_from_spec(spec);spec.loader.exec_module(binder)
binder.apply(config,OUT/"typed-owner")
typed_receipt=OUT/"typed-owner/receipt.json"
receipt={
 "schema":"h706.d-short-edge-quotient-typed-rebind-envelope.v1",
 "status":"PASS_D_TYPED_FACE_UNION_BINDING",
 "source_script_path":str(Path(__file__)),"source_script_sha256":sha(__file__),
 "original_surface_report_path":str(REP),"original_surface_report_sha256":pins[REP],
 "raw_candidate_path":str(RAW),"raw_candidate_sha256":pins[RAW],
 "immediate_typed_identity_seed_path":str(PARENT),"immediate_typed_identity_seed_sha256":pins[PARENT],
 "ultimate_typed_D185_path":str(ULTIMATE),"ultimate_typed_D185_sha256":pins[ULTIMATE],
 "adapter_input_path":str(config),"adapter_input_sha256":sha(config),
 "existing_typed_binder_path":str(BINDER),"existing_typed_binder_sha256":sha(BINDER),
 "binder_receipt_path":str(typed_receipt),"binder_receipt_sha256":sha(typed_receipt),
 "accepted_operation_count":2015,"coordinates_changed":False,
 "surface_context_pair_count":0,"replacement_self_pair_count":0,
 "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
 "prior_failed_adapter_attempt":{"path":str(RUN/"typed-rebind-001"),"reason":"legacy binder field names did not include typed_parent_union/new_faces; raw source unchanged"}
}
rp=OUT/"binding-envelope.json"
rp.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
print(json.dumps({"status":receipt["status"],"receipt":str(rp),"receipt_sha256":sha(rp),"typed_candidate":json.loads(typed_receipt.read_text())["candidate_roles"]["D"]},sort_keys=True))
