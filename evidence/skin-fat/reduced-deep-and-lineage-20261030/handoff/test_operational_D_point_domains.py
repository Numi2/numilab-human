"""Actual D source-coordinate-domain tests; no physical admission."""
from pathlib import Path
import importlib.util,json,tempfile,numpy as np,copy
P=Path(__file__).resolve().parent.parent;F=Path(__file__).resolve().parent
H=P/"paired-region-rebuild-667/tetgen-handoff-697"
sp=importlib.util.spec_from_file_location("operational_point_owner",H/"canonical_d_operational_targets.py")
m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
D624=P/"geometry-worker/matched-nesting-390/run-044-native-f32-replay-185/cube2um-union/conditioning-run-007-source-retype/pair-geometry-014-linked-remesh/fine-I-bone-barrier-run-024/full-native-self-028/current-collar-plan-032/engine-20-ops-one-sided624-run-006/candidate-f32-linked-remesh-typed.npz"
D010=P/"canonical-D-boundary-binding-696/run-010-combined-operational-reference-candidate/candidate-D-canonical14-reference-remap.npz"
D071=P/"reduced-physical-boundary-400/attempt-046-fine-role-simplification/run-071-bone60-source-patch-apply-001/candidate-D023-bone60-source-patched.npz"
packet=dict(mode="authenticated-source-coordinate-subset",
            sources=[dict(role=role,path=str(path),sha256=m.POINT_SOURCE_SHA256[role])
                     for role,path in [("D624",D624),("D010",D010)]])
with np.load(D624,allow_pickle=False) as z:A=z["points_world_m"].astype(np.float32)
with np.load(D010,allow_pickle=False) as z:B=z["points_world_m"].astype(np.float32)
points=np.vstack((A[[0,1]],B[[128292,19850]]))
asset=np.asarray([0,0,1,1],np.int64);rows=np.asarray([0,1,128292,19850],np.int64)
checks=[]
with tempfile.TemporaryDirectory(prefix="point-domain-regression-") as td:
    path=Path(td)/"mapped.npz"
    def emit(v=points,a=asset,r=rows):
        np.savez_compressed(path,points_world_m=v,
            point_preservation_source_asset_index=a,point_preservation_source_vertex_row=r)
    emit();m.validate_point_preservation(packet,path,points);checks.append("two authenticated source domains")
    cases=[]
    bad=copy.deepcopy(packet);bad["sources"][0]["sha256"]="wrong"
    cases.append(("wrong_source_hash",bad,points,asset,rows,points))
    bad=copy.deepcopy(packet);bad["sources"].reverse()
    cases.append(("reversed_source_domains",bad,points,asset,rows,points))
    bad=points.copy();bad[0,0]=np.nextafter(bad[0,0],np.float32(np.inf))
    cases.append(("changed_F32_coordinate",packet,bad,asset,rows,bad))
    bad=rows.copy();bad[0]=len(A)
    cases.append(("outside_source_row",packet,points,asset,bad,points))
    bad=asset.copy();bad[0]=2
    cases.append(("outside_source_asset",packet,points,bad,rows,points))
    cases.append(("float_row_mapping",packet,points,asset,rows.astype(float),points))
    cases.append(("foreign_emitted_surface",packet,points,asset,rows,points+np.float32(.001)))
    for name,p,v,a,r,passed in cases:
        emit(v,a,r)
        try:m.validate_point_preservation(p,path,passed)
        except ValueError:checks.append("reject:"+name)
        else:raise AssertionError(name)
    np.savez_compressed(path,points_world_m=points)
    try:m.validate_point_preservation(packet,path,points)
    except ValueError:checks.append("reject:missing_maps")
    else:raise AssertionError("missing maps")
    with np.load(D071,allow_pickle=False) as z:
        v=z["points_world_m"].astype(np.float32)
        off=z["vertex_parent_source_candidate_offsets"]
        r=z["vertex_parent_source_candidate_rows"]
    assert np.all(np.diff(off)==1)
    emit(v,np.zeros(len(v),np.int64),r)
    actual=m.validate_point_preservation(packet,path,v)
    assert actual["source_domain_counts"]==[113824,0]
    checks.append("all113824_D071_rows_use_declared_D624_vertex_CSR")
receipt=dict(status="PASS_EXACT_AUTHENTICATED_POINT_DOMAINS_ONLY",test_count=len(checks),checks=checks,
             actual_D071=actual,owner_sha256=m.sha(H/"canonical_d_operational_targets.py"),
             test_sha256=m.sha(__file__),D071_sha256=m.sha(D071),
             physical_admission=False,runtime=0,accepted_steps=0)
F.joinpath("test_operational_D_point_domains.receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
