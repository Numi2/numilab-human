"""Actual source lineage checks; no geometry or physical admission."""
from pathlib import Path
import importlib.util,json,tempfile,numpy as np,time,hashlib
P=Path(__file__).resolve().parent.parent
F=Path(__file__).resolve().parent
H=P/"paired-region-rebuild-667/tetgen-handoff-697"
def load(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
m=load("lineage_current",F/"prepare_reduced_surface_lineage_current.py")
adapter=load("registered_owner",H/"registered_surface_handoff.py")
O=P/"reduced-physical-boundary-400/attempt-046-fine-role-simplification/run-081-OI-local-topology-repair/candidate-O-self-and-OI-repaired.npz"
OS=P/"surface-transfer-checkpoint-392/payload/O/candidate.npz"
OC=P/"reduced-surface-lineage-702/O081-lineage-prepared.npz"
I=P/"reduced-physical-boundary-400/attempt-046-fine-role-simplification/run-093-I020-local-witness-flips-005/candidate-I-local-flipped.npz"
IS=P/"geometry-worker/fixed-physical-O-local-nested-001/pilot-004-union/dnew-vs-i296-inventory-001/i328-cavity-audit-001/fill-003/I-f32-conditioning-367/condition-288-pair-connected-remesh-001-run-002/result/candidate-f32-typed.npz"
t=time.monotonic()
with tempfile.TemporaryDirectory(prefix="lineage-real-asset-parity-") as td:
    td=Path(td)
    rr=m.prepare(O,OS,td/"O.npz",td/"O-receipt.json")
    with np.load(td/"O.npz",allow_pickle=False) as z:a={k:z[k] for k in ("points_world_m","faces",*m.CSR,"face_ancestry_offsets")}
    with np.load(OC,allow_pickle=False) as z:
        for k in a:np.testing.assert_array_equal(a[k],z[k])
    assert rr["source_had_full_declared_typed_CSR"]
    ir=m.prepare(I,IS,td/"I.npz",td/"I-receipt.json")
    result=adapter.load_prepared_reduced_surface_217(None,td/"I.npz","I")
    with np.load(td/"I.npz",allow_pickle=False) as z:
        parents=z["face_ancestry_source_face_rows"]
        parent_lane=z["face_ancestry_parent_face_rows"]
        orig_lane=z["face_ancestry_source_original_ids"]
    with np.load(IS,allow_pickle=False) as z:
        np.testing.assert_array_equal(parent_lane,z["face_source_face_id"][parents])
        np.testing.assert_array_equal(orig_lane,z["face_source_original_id"][parents])
    assert not ir["source_had_full_declared_typed_CSR"]
receipt=dict(status="PASS_REAL_O_PARITY_AND_I_NATIVE_IDENTITY_LANES",
             owner_sha256=m.sha(F/"prepare_reduced_surface_lineage_current.py"),
             test_sha256=m.sha(__file__),registered_surface_owner_sha256=m.sha(H/"registered_surface_handoff.py"),
             O_complete_typed_CSR_bitwise_equal=True,I_ancestry_tuples=len(parents),
             I_declared_parent_face_ids_recomputed=True,I_loader_faces=len(result["faces"]),
             input_hashes={str(p):m.sha(p) for p in [O,OS,OC,I,IS]},wall_seconds=time.monotonic()-t,
             physical_admission=False,runtime=0,accepted_steps=0,I_self_pairs_remaining=66)
F.joinpath("test_actual_reduced_lineage.receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
