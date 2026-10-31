from pathlib import Path
import hashlib, importlib.util, json
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
BUILDER=ROOT/"repair_coupled_quality_points_722_inner_back_v23d_role_balanced_fixed_fixed.py"
spec=importlib.util.spec_from_file_location("v23d_builder",BUILDER)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
fn=mod._typed_archive_scalar_text
field="typed_parent_candidate_path"
accepted=[
    ("unicode_0d",np.asarray("parent.npz",dtype="<U10")),
    ("unicode_1d",np.asarray(["parent.npz"],dtype="<U10")),
    ("bytes_0d",np.asarray(b"sha",dtype="|S3")),
    ("bytes_1d",np.asarray([b"sha"],dtype="|S3")),
]
values={}
for name,value in accepted:
    values[name]=fn(value,field,"control.npz")
assert values["unicode_0d"]==values["unicode_1d"]=="parent.npz"
assert values["bytes_0d"]==values["bytes_1d"]=="sha"
rejected=[
    ("empty_array",np.asarray([],dtype="<U1")),
    ("two_items",np.asarray(["a","b"],dtype="<U1")),
    ("two_dimensional",np.asarray([["a"]],dtype="<U1")),
    ("object_dtype",np.asarray("a",dtype=object)),
    ("empty_text",np.asarray("",dtype="<U1")),
]
reject_results={}
for name,value in rejected:
    try:
        fn(value,field,"control.npz")
    except ValueError:
        reject_results[name]="REJECTED"
    else:
        raise AssertionError("accepted invalid scalar: "+name)
receipt={
 "schema":"h706.v23d-typed-parent-scalar-control.v1",
 "status":"PASS",
 "builder_sha256":hashlib.sha256(BUILDER.read_bytes()).hexdigest(),
 "accepted_forms":values,
 "rejected_forms":reject_results,
}
receipt_path=ROOT/"all-quality-846-root/v23d-scalar-control-001/receipt.json"
receipt_path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,sort_keys=True))
