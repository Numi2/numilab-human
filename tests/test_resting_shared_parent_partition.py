import copy
import hashlib
import json
import numpy as np
import pytest
from tests.test_resting_shared_partition_verify import scene
from numilab_human import passive_attachment_composition as pac
from numilab_human import passive_tissue_parent_carriers as pc
from numilab_human import resting_shared_partition_verify as shared


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent_scene(tmp_path):
    source, digest, captures = scene(tmp_path)
    data = pac._read_nhtiss4(source)
    binding = pac._row_slices(data, data["records"][0])["binding_bytes"]
    carriers = []
    for record in data["records"][:2]:
        arrays = pac._nhtiss_row_arrays(data, record)
        weights = np.tile(np.array([.2, .8], dtype="<f8"), (4, 1))
        weights[3] = [0., 1.]  # authentic structural zero; no epsilon floor
        carriers.append(pc.CarrierRow(int(record[6]),
            np.c_[arrays["positions"], arrays["normals"]].astype("<f4"),
            arrays["faces"], binding, weights))
    payload, surfaces, _ = pc.encode_carriers(
        carriers, registration_fingerprint=data["fingerprint"],
        source_digest=data["source_digest"])
    parent = tmp_path/"parent.nhtissue"
    parent.write_bytes(payload)
    receipt = {"passive_tissue_parent_bindings": {
        "child_payload_sha256": digest,
        "source_payloads": [{"path": str(parent), "sha256": sha(parent)}],
        "surfaces": surfaces}}
    path = tmp_path/"source-receipt.json"

    def context(value=None):
        path.write_text(json.dumps(receipt if value is None else value))
        return {"parent_bindings_context": {"path": str(path), "sha256": sha(path)}}
    return source, digest, captures, parent, receipt, context


def test_complete_parent_carriers_own_interface_even_if_child_fallback_differs(tmp_path):
    source, _, captures, _, receipt, context = parent_scene(tmp_path)
    data = pac._read_nhtiss4(source)
    raw = bytearray(data["raw"])
    record = data["records"][1]
    slots = np.ndarray((4, 4), "<u4", buffer=raw,
        offset=data["vertex_start"]+int(record[2])*56+24, strides=(56, 4))
    slots[:, 0] = 1
    source.write_bytes(raw)
    receipt["passive_tissue_parent_bindings"]["child_payload_sha256"] = sha(source)
    proof = shared.certify_shared_muscle_partition(
        captures, (51005, 1), (51005, 2), source, sha(source), capture_context=context())
    assert proof["shared_interface_face_count"] == 1
    assert proof["exact_native_partition_certificate"]["interiors_disjoint"]
    bindings = proof["ordered_source_bindings"].values()
    assert all(row["full_parent_carrier_material"]["complete_positive_terms_retained"]
               for row in bindings)
    assert all(not row["full_parent_carrier_material"]["coefficient_normalization_or_truncation"]
               for row in bindings)


@pytest.mark.parametrize("mutation,match", [
    ("child", "child payload differs"),
    ("source_hash", "source hash differs"),
    ("allocation", "surface allocation differs"),
    ("negative", "invalid parent term identity"),
    ("zero", "invalid parent term identity"),
    ("sum", "partition invalid"),
    ("duplicate", "duplicate parent material support"),
])
def test_parent_receipt_fails_closed_before_certificate(tmp_path, monkeypatch, mutation, match):
    source, digest, captures, _, receipt, context = parent_scene(tmp_path)
    receipt = copy.deepcopy(receipt)
    extension = receipt["passive_tissue_parent_bindings"]
    if mutation == "child":
        extension["child_payload_sha256"] = "0"*64
    elif mutation == "source_hash":
        extension["source_payloads"][0]["sha256"] = "0"*64
    elif mutation == "allocation":
        extension["surfaces"][0]["vertices"].pop()
    elif mutation == "negative":
        extension["surfaces"][0]["vertices"][0][0][3] = -.1
    elif mutation == "zero":
        extension["surfaces"][0]["vertices"][0][0][3] = 0.
    elif mutation == "sum":
        extension["surfaces"][0]["vertices"][0][0][3] = .3
    else:
        terms = extension["surfaces"][0]["vertices"][0]
        terms.append(copy.deepcopy(terms[0]))
    monkeypatch.setattr(shared.cert, "audit_shared_interface_partition",
                        lambda *a, **k: pytest.fail("predicate reached after invalid source"))
    with pytest.raises(Exception, match=match):
        shared.preflight_shared_muscle_partition(captures, (51005, 1), (51005, 2),
            source, digest, capture_context=context(receipt))


def test_parent_receipt_file_hash_is_required(tmp_path):
    source, digest, captures, _, _, context = parent_scene(tmp_path)
    value = context()
    value["parent_bindings_context"]["sha256"] = "0"*64
    with pytest.raises(Exception, match="receipt hash differs"):
        shared.preflight_shared_muscle_partition(captures, (51005, 1), (51005, 2),
            source, digest, capture_context=value)


def test_parent_term_order_and_actual_emission_cannot_be_reinterpreted(tmp_path):
    source, digest, captures, _, receipt, context = parent_scene(tmp_path)
    for terms in receipt["passive_tissue_parent_bindings"]["surfaces"][1]["vertices"]:
        terms.reverse()
    with pytest.raises(Exception, match="no reciprocal source/material"):
        shared.preflight_shared_muscle_partition(captures, (51005, 1), (51005, 2),
            source, digest, capture_context=context())
    for terms in receipt["passive_tissue_parent_bindings"]["surfaces"][1]["vertices"]:
        terms.reverse()
    captures[(51005, 2)][0][0, 0] = np.float32(1e-8)
    with pytest.raises(Exception, match="different Float32 positions"):
        shared.preflight_shared_muscle_partition(captures, (51005, 1), (51005, 2),
            source, digest, capture_context=context())


def test_parent_cold_preflight_never_calls_exact_certificate(tmp_path, monkeypatch):
    source, digest, captures, _, _, context = parent_scene(tmp_path)
    monkeypatch.setattr(shared.cert, "audit_shared_interface_partition",
                        lambda *a, **k: pytest.fail("predicate in cold preparation"))
    result = shared.preflight_shared_muscle_partition(captures, (51005, 1), (51005, 2),
        source, digest, capture_context=context())
    assert result["shared_interface_face_count"] == 1
    assert not result["exact_predicates_executed"]
