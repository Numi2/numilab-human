#!/usr/bin/env python3
"""Bind the frozen run006 D topology through the existing typed-face-union owner."""
from pathlib import Path
import hashlib, importlib.util, json, time
import numpy as np
H=Path(__file__).resolve().parents[1]
RUN=H/"d-shortedge-quotient-385-run006"
raw_report=RUN/"surface-run001/report.json"
raw_npz=RUN/"surface-run001/candidate-D-short-edge-collapse-raw.npz"
parent=H/"d-shortedge-quotient-351-run002/typed-rebind-003/candidate-D-typed-current.npz"
binder=H/"d-shortedge-quotient-351-run002/bind_short_edge_typed_351.py"
expected={
 str(raw_report):"557b3978592b5febdef39d6e238ab38ee69bddc95680ee816abd7cd85e340d62",
 str(raw_npz):"f8220c4a021acca0d15eea29c02ce7b51f7102954e242e8403ebf8adf9f971b7",
 str(parent):"18517e7dc0b1b25d67eea9691ad98bdbcfd07d18b9e1fe45f25b95515ee99ade",
}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""): h.update(block)
 return h.hexdigest()
for p,d in expected.items():
 if sha(p)!=d: raise ValueError(f"pinned input changed: {p}")
rep=json.loads(raw_report.read_text())
assert rep["status"]=="PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL"
assert rep["physical_admission"] is False and rep["native_steps"]==0
assert len(rep["accepted_operations"])==309 and set(x["role"] for x in rep["accepted_operations"])=={"D"}
assert rep["joint_changed_vs_complete_OID_walls_all185"]["pair_count"]==0
assert rep["joint_changed_vs_complete_OID_walls_all185"]["new_face_self_pairs"]==0
raw_parent=np.load(raw_npz,allow_pickle=False)
assert str(raw_parent["typed_parent_candidate_sha256"][0])==expected[str(parent)]
assert str(raw_parent["immediate_parent_candidate_sha256"][0])=="cf27a4d3485724b33f884061fcc94529d8d0f8f980aa46ae496d713dcd59c94c"
# Make a D-only adapter input. O/I raw files were unchanged and are handled by their current typed owners.
rep["outputs"]={"D":{"path":str(raw_npz),"sha256":sha(raw_npz),"parent":str(parent),"parent_sha256":sha(parent)}}
rep["joint_complete_surface_walls_all185_pairs"]=int(rep["joint_changed_vs_complete_OID_walls_all185"]["pair_count"])
rep["joint_changed_self_pairs"]=int(rep["joint_changed_vs_complete_OID_walls_all185"]["new_face_self_pairs"])
rep["quality_metric"]="normalized_shape_apex_bound"
rep["construction_surface_floor"]=1e-5
rep["typed_binding_scope"]="D only; raw O/I were identity in this run and are not re-bound here"
config=RUN/"typed-rebind-input-001.json"
config.write_text(json.dumps(rep,sort_keys=True,indent=2,allow_nan=False)+"\n")
out=RUN/"typed-rebind-001"
spec=importlib.util.spec_from_file_location("typed_binder385",binder)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.apply(config,out)
receipt_path=out/"receipt.json"
envelope={"schema":"h706.d-short-edge-quotient-typed-rebind-envelope.v1","status":"PASS_D_TYPED_FACE_UNION_BINDING","source_script_path":str(Path(__file__)),"source_script_sha256":sha(Path(__file__)),"original_surface_report_path":str(raw_report),"original_surface_report_sha256":sha(raw_report),"raw_candidate_path":str(raw_npz),"raw_candidate_sha256":sha(raw_npz),"parent_typed_candidate_path":str(parent),"parent_typed_candidate_sha256":sha(parent),"adapter_input_path":str(config),"adapter_input_sha256":sha(config),"existing_typed_binder_path":str(binder),"existing_typed_binder_sha256":sha(binder),"binder_receipt_path":str(receipt_path),"binder_receipt_sha256":sha(receipt_path),"accepted_operation_count":309,"coordinates_changed":False,"surface_context_pair_count":0,"replacement_self_pair_count":0,"physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
path=out/"binding-envelope.json";path.write_text(json.dumps(envelope,sort_keys=True,indent=2)+"\n")
print(json.dumps({"status":envelope["status"],"typed_candidate":mod.json.loads(receipt_path.read_text())["candidate_roles"]["D"],"binder_receipt_sha256":sha(receipt_path),"envelope_path":str(path),"envelope_sha256":sha(path)},sort_keys=True))
