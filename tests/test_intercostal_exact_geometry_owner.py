"""Exact historical cap identity can survive face-slot reindexing."""
import hashlib
import io
import json

import numpy as np
import pytest

from numilab_human.resting_intercostal_mesh_arrangement import (
    _normalize_local_reconstruction_parent_face_owners,
)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def raw_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

def npz(**values):
    out = io.BytesIO()
    np.savez_compressed(out, **values)
    return out.getvalue()

def fixture(*, changed=None):
    original = {
        "positions_world_f64": np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype="<f8"),
        "faces": np.asarray([[0,1,2],[0,2,3]], dtype="<u4"),
        "source_original_id": np.asarray([900000048,51010313], dtype="<u4"),
        "source_face_id": np.asarray([17,0], dtype="<u4"),
    }
    points = original["positions_world_f64"].astype("<f4")
    faces = np.asarray([[1,2,0]], dtype="<u4")
    source = {
        "faces": [{"output_face":7, "classification":"unchanged_parent_face",
                   "exact_source_face_claim":True, "parent_output_face_ids":[0],
                   "source_rows":[[900000048,17]]}],
    }
    evidence = {"method":"exact_geometry_authenticated_parent_patch",
                "source_output_face_id":7, "source_face_ids":[17], "rib_stable_id":48}
    if changed:
        changed(original, points, faces, source, evidence)
    original_raw = npz(**original)
    source["parent_candidate_sha256"] = sha(original_raw)
    source_raw = raw_json(source)
    registry = raw_json({"rows":[{"source_bone":[51004,48], "derived_source_id":900000048,
                                  "faces":20,"tool_sha256":"a"*64}]})
    parent = npz(positions_world_f32=points, faces=faces)
    operation = b"exact geometry evidence test operation"
    evidence["source_report_sha256"] = sha(source_raw)
    report = raw_json({
        "schema":"numi.human.intercostal-local-reconstruction-parent-face-owners.v1",
        "stable_id":313, "parent_payload_sha256":sha(parent),
        "operation_ancestry_sha256":sha(operation),
        "parent_faces_sha256":sha(faces.tobytes()), "parent_face_count":1,
        "parent_face_owners":[{"parent_face_id":0,"rib_stable_ids":[48],"evidence":[evidence]}],
    })
    kwargs = dict(stable_id=313,parent_faces=faces,parent_payload_raw=parent,
                  operation_ancestry_raw=operation,
                  authenticated_source_report_raws=[source_raw,registry],
                  authenticated_source_report_sha256s=[sha(source_raw),sha(registry)],
                  authenticated_source_parent_npz_raws=[original_raw],
                  authenticated_source_parent_npz_sha256s=[sha(original_raw)])
    return report,kwargs

def run(changed=None, **overrides):
    report,kwargs=fixture(changed=changed)
    kwargs.update(overrides)
    return _normalize_local_reconstruction_parent_face_owners(report,sha(report),**kwargs)

def test_exact_oriented_historical_cap_inherits_across_reindexing():
    owners,summary=run()
    assert owners == {0:48}
    assert summary["exact_geometry_inherited_parent_face_owner_count"] == 1
    assert summary["direct_typed_source_face_owner_count"] == 0

def test_missing_independent_original_npz_pin_rejected():
    with pytest.raises(ValueError,match="independent pins"):
        run(authenticated_source_parent_npz_sha256s=[])

def test_drifted_original_npz_rejected():
    with pytest.raises(ValueError,match="independent pins"):
        run(authenticated_source_parent_npz_raws=[b"drift"])

def test_unprovided_source_parent_rejected():
    with pytest.raises(ValueError,match="pinned source parent"):
        run(authenticated_source_parent_npz_raws=[],authenticated_source_parent_npz_sha256s=[])

def test_one_ulp_current_displacement_is_not_exact_ownership():
    def change(o,p,f,s,e):
        p[0,0]=np.nextafter(np.float32(0),np.float32(1))
    with pytest.raises(ValueError,match="differs from its historical typed source"):
        run(change)

def test_opposite_orientation_is_not_an_inherited_patch():
    def change(o,p,f,s,e):
        f[0]=f[0,::-1]
    with pytest.raises(ValueError,match="differs from its historical typed source"):
        run(change)

@pytest.mark.parametrize("kind",["inferred","mixed_support","wrong_original_row","wrong_original_face","wrong_output","float_ids"])
def test_inferred_or_forged_historical_evidence_rejected(kind):
    def change(o,p,f,s,e):
        row=s["faces"][0]
        if kind=="inferred":
            row["exact_source_face_claim"]=False
            row["classification"]="inferred_local_precision_patch"
        elif kind=="mixed_support":
            row["parent_output_face_ids"]=[0,1]
        elif kind=="wrong_original_row":
            o["source_original_id"][0]=900000049
        elif kind=="wrong_original_face":
            o["source_face_id"][0]=18
        elif kind=="wrong_output":
            e["source_output_face_id"]=0
        elif kind=="float_ids":
            o["source_face_id"]=o["source_face_id"].astype("<f8")+0.5
    with pytest.raises(ValueError):
        run(change)

def test_old_method_cannot_claim_a_different_historical_output_slot():
    def change(o,p,f,s,e):
        e["method"]="inherited_authenticated_parent_patch"
    with pytest.raises(ValueError,match="not proven by the pinned typed source-face ancestry"):
        run(change)
