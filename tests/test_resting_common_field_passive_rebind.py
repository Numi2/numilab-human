from __future__ import annotations

import copy
import hashlib

import numpy as np
import pytest

from numilab_human.resting_anatomy_interface_patch import encode_payload
from numilab_human.resting_common_field_passive_rebind import rebind_passive_rows


def fixture():
    vertices = np.array([[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
                         [0, 1, 0, 0, 0, 1]], dtype="<f4")
    rows = {sid: {"body_index": 23, "layer": 1, "flags": 0,
                  "vertices6": vertices.copy(), "faces": np.array([[0, 1, 2]])}
            for sid in (1, 323, 345)}
    header = (b"NHANAT1\0", 5, 3, 9, 9, 42, b"a" * 32)
    raw = encode_payload(header, rows)
    sha = hashlib.sha256(raw).hexdigest()
    receipt = {
        "payload": {"sha256": sha},
        "functional_bindings": {"anatomy_payload_sha256": sha, "drive": [1, 2]},
        "mass_accounting": {"kg": 72},
        "passive_tissue_parent_bindings": {"anatomy_payload_sha256": sha,
                                          "anatomy_surfaces": []},
        "provenance": {
            "source_id_map": {str(sid): {"source": str(sid)} for sid in rows},
            "cardiac_geometry_binding": {"common_field": {
                "schema": "numi.human.cardiac_common_field.v1",
                "anatomy_payload_sha256": sha, "volume_stable_ids": [1],
                "vertex_ranges": [{"stable_id": 1, "vertex_count": 3, "first_vertex": 0}],
                "passive_attachment_ranges": [],
                **{k: {"path": k, "sha256": "b" * 64}
                   for k in ("map", "domain_boxes", "polynomials")},
            }},
        },
    }
    new_rows = copy.deepcopy(rows)
    for sid in (323, 345):
        new_rows[sid]["vertices6"][0, 0] = 0.25
    return receipt, raw, header, new_rows


def run(receipt, raw, header, rows, ids=(323, 345)):
    return rebind_passive_rows(receipt, raw, encode_payload(header, rows),
                               changed_ids=ids, output_path="/candidate.nhanatomy")


def test_rebind_preserves_function_mass_owner_and_source_identities():
    receipt, raw, h, rows = fixture()
    original = copy.deepcopy(receipt)
    result, audit = run(receipt, raw, h, rows)
    assert receipt == original
    assert result["mass_accounting"] == original["mass_accounting"]
    assert result["functional_bindings"]["drive"] == [1, 2]
    assert result["provenance"]["source_id_map"] == original["provenance"]["source_id_map"]
    assert audit["unchanged_row_count"] == 1
    assert result["passive_tissue_parent_bindings"]["anatomy_payload_sha256"] == result["payload"]["sha256"]


@pytest.mark.parametrize("mutation", ["owner", "extra", "parent", "common", "base", "source"])
def test_rebind_rejects_unproven_changes(mutation):
    r, raw, h, rows = fixture()
    if mutation == "owner":
        rows[323]["body_index"] = 20
    elif mutation == "extra":
        rows[1]["vertices6"][0, 0] = 0.5
    elif mutation == "parent":
        r["passive_tissue_parent_bindings"]["anatomy_surfaces"] = [{"stable_id": 323}]
    elif mutation == "common":
        r["provenance"]["cardiac_geometry_binding"]["common_field"]["anatomy_payload_sha256"] = "c" * 64
    elif mutation == "base":
        r["payload"]["sha256"] = "c" * 64
    else:
        h = (*h[:5], 43, h[6])
    with pytest.raises(ValueError):
        run(r, raw, h, rows)


@pytest.mark.parametrize("ids", [(True, 345), (323.0, 345), (323, 323), (), (323, 999)])
def test_strict_declared_ids(ids):
    r, raw, h, rows = fixture()
    with pytest.raises(ValueError):
        run(r, raw, h, rows, ids)
