import hashlib
import json
import struct
from pathlib import Path
import numpy as np
import pytest
from numilab_human import passive_attachment_composition as pac
from numilab_human.passive_reference_binding_extension import extend_reference_bindings


def fixture(tmp_path, conflict=False, count=3):
    out = tmp_path/"source.nhtissue"
    records, bindings, vertices, indices, rows = [], [], [], [], []
    for j, cores in enumerate(((41, 25), (41, 29), (20,)) + ((20,),)*(count-3)):
        sid = j+1
        records.append([len(bindings), len(cores), j*4, 4, j*12, 12, sid, 1])
        for core in cores:
            bindings.append(struct.pack("<I8f", core, int(conflict and sid == 2 and core == 41), 0, 0, 0, 0, 0, 1, 1))
        vb = bytearray(4*56)
        np.ndarray((4, 6), "<f4", buffer=vb, strides=(56, 4))[:] = [[0,0,0,0,0,1],[1,0,0,0,0,1],[0,1,0,0,0,1],[0,0,1,0,0,1]]
        np.ndarray((4, 4), "<f4", buffer=vb, offset=40, strides=(56, 4))[:, 0] = 1
        vertices.append(bytes(vb))
        indices.extend((np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], "<u4")+j*4).ravel())
        rows.append({"stable_id": sid, "vertex_count": 4, "triangle_count": 4,
                     "body_weight_count": len(cores), "matched_muscles": [{"name": "unchanged"}],
                     "body_bindings": [{"core_body_index": c, "myosim_body": str(c)} for c in cores]})
    raw = (struct.pack("<8s6I32s", b"NHTISS4\0", 5, count, len(bindings), count*4, count*12, 17, b"x"*32)
           + np.array(records, "<u4").tobytes()+b"".join(bindings)+b"".join(vertices)
           + np.array(indices, "<u4").tobytes())
    out.write_bytes(raw)
    manifest = {"schema": "numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1",
                "source": {"surfaces": rows, "myosim_source_archive_sha256": (b"x"*32).hex()},
                "payload": {"file": out.name, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                            "surface_count": count, "binding_count": len(bindings), "vertex_count": count*4,
                            "index_count": count*12, "registration_fingerprint32": "00000011"}}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    return out


def test_appends_only_missing_existing_registered_support(tmp_path):
    source = fixture(tmp_path)
    before = source.read_bytes()
    report = extend_reference_bindings(source, tmp_path/"out", {1: 2})
    child = tmp_path/"out"/source.name
    old, new = pac._read_nhtiss4(source), pac._read_nhtiss4(child)
    assert source.read_bytes() == before
    assert new["binding_count"] == old["binding_count"]+1
    for a, b in zip(old["records"], new["records"]):
        x, y = pac._row_slices(old, a), pac._row_slices(new, b)
        assert x["vertex_bytes"] == y["vertex_bytes"] and x["local_faces"] == y["local_faces"]
        assert y["binding_bytes"].startswith(x["binding_bytes"])
    assert report["rows"][0]["added_body_bindings"] == [{"core_body_index": 29, "myosim_body": "29"}]
    m = json.loads(child.with_suffix(".manifest.json").read_text())
    assert m["source"]["surfaces"][0]["matched_muscles"] == [{"name": "unchanged"}]
    assert not report["native_admission"]


def test_rejects_conflicting_transform_for_same_core(tmp_path):
    source = fixture(tmp_path, conflict=True)
    with pytest.raises(Exception, match="inconsistent registered transforms"):
        extend_reference_bindings(source, tmp_path/"out", {1: 2})
    assert not (tmp_path/"out").exists()


def test_rejects_missing_or_identical_donor(tmp_path):
    source = fixture(tmp_path)
    for mapping in ({1: 1}, {1: 99}, {}):
        with pytest.raises(Exception):
            extend_reference_bindings(source, tmp_path/"out", mapping)
    assert not (tmp_path/"out").exists()


def test_receipt_rebind_preserves_parent_rows_and_mass(tmp_path):
    from numilab_human.passive_reference_receipt_composition import compose_reference_receipt
    source = fixture(tmp_path, count=150)
    anatomy = tmp_path/"body.nhanatomy"
    anatomy.write_bytes(b"retained anatomical source")
    old = pac._read_nhtiss4(source)
    a = pac._nhtiss_row_arrays(old, old["records"][0])
    patch = tmp_path/"patch.npz"
    moved = a["positions"].copy()
    moved[0, 0] += np.float32(.001)
    np.savez_compressed(patch, vertices6=np.c_[moved, a["normals"]].astype("<f4"),
                        faces=a["faces"], binding_indices=a["binding_indices"], weights=a["weights"])
    report = tmp_path/"preparation.json"
    report.write_text('{"native_admission": false}')
    metadata = json.loads(source.with_suffix(".manifest.json").read_text())["payload"]
    receipt = {"schema": "numi.human.resting-anatomy-receipt.v1",
               "payload": {"path": str(anatomy), "sha256": hashlib.sha256(anatomy.read_bytes()).hexdigest()},
               "provenance": {"native_muscle_surfaces": {"payload_path": str(source), "sha256": metadata["sha256"]}},
               "mass_geometry_accounting": {"reference_total_mass_kg": 72},
               "passive_tissue_parent_bindings": {"child_payload_sha256": metadata["sha256"],
                   "surfaces": [{"stable_id": 3, "vertices": [[[0,3,0,1.0]]]}],
                   "source_payloads": [{"path": "retained", "sha256": "retained"}]}}
    rp = tmp_path/"receipt.json"
    rp.write_text(json.dumps(receipt))
    result = compose_reference_receipt(rp, tmp_path/"bound", [(1, patch, report)], extensions={1: 2})
    new = json.loads(Path(result["receipt_path"]).read_text())
    assert new["mass_geometry_accounting"] == receipt["mass_geometry_accounting"]
    assert new["passive_tissue_parent_bindings"]["surfaces"] == receipt["passive_tissue_parent_bindings"]["surfaces"]
    assert new["passive_tissue_parent_bindings"]["source_payloads"] == receipt["passive_tissue_parent_bindings"]["source_payloads"]
    assert new["passive_tissue_parent_bindings"]["child_payload_sha256"] == result["payload_sha256"]
    actual = pac._read_nhtiss4(Path(result["payload_path"]))
    assert pac._row_slices(old, old["records"][2]) == {
        **pac._row_slices(actual, actual["records"][2]),
        "first_binding": int(old["records"][2][0]),
        "first_vertex": int(old["records"][2][2]),
        "first_index": int(old["records"][2][4])}
    # A subsequent geometry-only repair must reuse the already admitted
    # support table while preserving the same parent-map/source ownership.
    again = compose_reference_receipt(Path(result["receipt_path"]), tmp_path/"rebound",
                                      [(1, patch, report)], extensions={})
    rebound = json.loads(Path(again["receipt_path"]).read_text())
    assert rebound["passive_tissue_parent_bindings"]["surfaces"] == receipt["passive_tissue_parent_bindings"]["surfaces"]
    assert rebound["provenance"]["passive_reference_receipt_compositions"][-1]["binding_extension"]["status"] == "EXISTING_REGISTERED_SUPPORTS_REUSED"
    assert not (tmp_path/"rebound"/"supports").exists()
    receipt["passive_tissue_parent_bindings"]["surfaces"][0]["stable_id"] = 1
    rp.write_text(json.dumps(receipt))
    with pytest.raises(Exception, match="material-parent reconstruction"):
        compose_reference_receipt(rp, tmp_path/"rejected", [(1, patch, report)], extensions={1: 2})
    assert not (tmp_path/"rejected").exists()
