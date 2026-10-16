"""Regional anatomy integration must detect overwrites and stale repair parents."""
from pathlib import Path
import hashlib
import numpy as np
import pytest
from numilab_human import resting_anatomy_interface_patch as anatomy
from numilab_human.resting_regional_tissue_composition import merge_anatomy_rows


def payload(tmp_path, name, *, moved=(), resized=(), body=None, fingerprint=7):
    rows = {}
    for sid in (7, 8, 313, 315, 317):
        xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
        if sid in moved:
            xyz[:, 2] = 0.1
        vertices = np.column_stack((xyz, np.tile([0, 0, 1], (3, 1)))).astype("<f4")
        if sid in resized:
            vertices = np.vstack((vertices, vertices[0]))
        rows[sid] = dict(body_index=body if body is not None and sid == 7 else 20,
                         layer=1, flags=0, vertices6=vertices,
                         faces=np.array([[0, 1, 2]], dtype=np.int64))
    p = tmp_path / name
    p.write_bytes(anatomy.encode_payload((b"NHANAT1\0", 5, 5, 15, 15,
                                         fingerprint, b"s" * 32), rows))
    return p


def merge(base, *replacements, pins=None):
    paths = {base, *(p for parent, candidate, ids in replacements for p in (parent, candidate))}
    pins = pins if pins is not None else {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return merge_anatomy_rows(base, replacements, expected_sha256=pins)


def test_joint_repairs_preserve_unrelated_rows_and_existing_abi(tmp_path):
    parent = payload(tmp_path, "parent")
    vessels = payload(tmp_path, "vessels", moved=(7, 8))
    thorax = payload(tmp_path, "thorax", moved=(313,), resized=(313,))
    # Integration base already has independent vessel repairs.
    raw, proof = merge(vessels, (parent, thorax, [313]))
    out = tmp_path / "combined"; out.write_bytes(raw)
    _, rows = anatomy.parse_payload(out)
    _, vr = anatomy.parse_payload(vessels)
    _, tr = anatomy.parse_payload(thorax)
    for sid in (7, 8, 315, 317):
        assert rows[sid]["vertices6"].tobytes() == vr[sid]["vertices6"].tobytes()
        assert rows[sid]["faces"].tobytes() == vr[sid]["faces"].tobytes()
    assert rows[313]["vertices6"].tobytes() == tr[313]["vertices6"].tobytes()
    assert not proof["native_admission"] and not proof["anatomical_admission"]


def test_two_regions_can_be_merged_together(tmp_path):
    base = payload(tmp_path, "base")
    a = payload(tmp_path, "a", moved=(7, 8))
    b = payload(tmp_path, "b", moved=(313, 315, 317))
    _, proof = merge(base, (base, a, [7, 8]), (base, b, [313, 315, 317]))
    assert proof["changed_stable_ids"] == [7, 8, 313, 315, 317]


def test_identical_already_integrated_row_is_idempotent(tmp_path):
    parent = payload(tmp_path, "parent")
    candidate = payload(tmp_path, "candidate", moved=(7,))
    raw, proof = merge(candidate, (parent, candidate, [7]))
    assert raw == candidate.read_bytes()
    assert proof["changed_stable_ids"] == []
    assert proof["selected_stable_ids"] == [7]


@pytest.mark.parametrize("case", ["undeclared", "missing", "conflict", "overlap", "owner", "fingerprint", "pin"])
def test_rejects_unreviewed_or_conflicting_changes(tmp_path, case):
    parent = payload(tmp_path, "parent")
    candidate = payload(tmp_path, "candidate", moved=(7, 8) if case == "undeclared" else (7,),
                        body=91 if case == "owner" else None,
                        fingerprint=8 if case == "fingerprint" else 7)
    base = payload(tmp_path, "base", resized=(7,) if case == "conflict" else ())
    ids = [7, 8] if case == "missing" else [7]
    replacements = [(parent, candidate, ids)]
    if case == "overlap":replacements *= 2
    with pytest.raises(ValueError):
        merge(base, *replacements, pins={} if case == "pin" else None)
