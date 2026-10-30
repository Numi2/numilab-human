"""Actual D071 full-union and prepared-native-route regression; no admission."""
from pathlib import Path
import importlib.util,json,numpy as np,types,tempfile
P=Path(__file__).resolve().parent.parent
F=Path(__file__).resolve().parent
H=P/"paired-region-rebuild-667/tetgen-handoff-697"
def load(n,p):
    sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
m=load("lineage_current",F/"prepare_reduced_surface_lineage_current.py")
h=load("handoff_current",H/"prepare_tetgen_handoff_697_typedD.py")
c=P/"reduced-physical-boundary-400/attempt-046-fine-role-simplification/run-071-bone60-source-patch-apply-001/candidate-D023-bone60-source-patched.npz"
s=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/full-native-self-028/current-collar-plan-032/engine-20-ops-one-sided624-run-006/candidate-f32-linked-remesh-typed.npz"
with tempfile.TemporaryDirectory(prefix="lineage-D071-parity-") as td:
    out=Path(td)/"D.npz";rec=Path(td)/"receipt.json"
    r=m.prepare(c,s,out,rec)
    def fail(*args,**kwargs):raise AssertionError("prepared D may not take legacy scalar or union aliases")
    result=h.load_surface_217(types.SimpleNamespace(load_surface=fail),out,"D")
    with np.load(c,allow_pickle=False) as z:
        cp=z["points_world_m"].astype(np.float32);cf=z["faces"]
        union_ids=z["topology_flip_parent_union_face_ids"]
        union_runs=z["topology_flip_parent_union_run_original_ids"]
        union_offsets=z["topology_flip_parent_union_offsets"]
        full_source_rows=z["topology_flip_parent_candidate_face_rows"]
        full_source_offsets=z["topology_flip_parent_candidate_face_offsets"]
    with np.load(out,allow_pickle=False) as z:
        a={k:z[k] for k in ["points_world_m","faces","face_ancestry_offsets","face_ancestry_source_original_ids","face_ancestry_parent_face_rows","face_ancestry_source_face_rows"]}
    np.testing.assert_array_equal(a["points_world_m"].astype(np.float32),cp)
    np.testing.assert_array_equal(a["faces"],cf)
    np.testing.assert_array_equal(a["face_ancestry_offsets"],full_source_offsets)
    np.testing.assert_array_equal(a["face_ancestry_source_face_rows"],full_source_rows)
    # The prior archive deduplicates native-ID pairs separately from its full
    # source-row CSR. Preserve every source tuple, then compare native pair sets.
    for row in range(len(cf)):
        lo,hi=a["face_ancestry_offsets"][row:row+2]
        ulo,uhi=union_offsets[row:row+2]
        expected=set(zip(union_runs[ulo:uhi].tolist(),union_ids[ulo:uhi].tolist()))
        actual=set(zip(a["face_ancestry_source_original_ids"][lo:hi].tolist(),
                       a["face_ancestry_parent_face_rows"][lo:hi].tolist()))
        assert actual==expected,(row,actual,expected)
    assert len(result["ancestry"])==len(cf)
    assert r["repair_union_representatives_selected"]==2
    prepared_sha=m.sha(out)
receipt=dict(status="PASS_REAL_D071_NORMALIZED_FULL_UNION_AND_NATIVE_ROUTE",
             candidate_sha256=m.sha(c),prepared_sha256=prepared_sha,source_sha256=m.sha(s),
             owner_sha256=m.sha(F/"prepare_reduced_surface_lineage_current.py"),
             handoff_sha256=m.sha(H/"prepare_tetgen_handoff_697_typedD.py"),
             test_sha256=m.sha(__file__),faces=len(cf),ancestry_tuples=len(full_source_rows),native_pair_tuples=len(union_ids),
             all_prior_native_pair_sets_equal=True,full_prior_source_row_CSR_bitwise_equal=True,geometry_bitwise_equal=True,
             legacy_route_not_used=True,physical_admission=False,runtime=0,accepted_steps=0)
F.joinpath("test_actual_D071_lineage.receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
