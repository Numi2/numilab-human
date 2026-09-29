from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.torso_body_links import compile_body_links, VISUAL_MANIFEST, NATIVE_RECEIPT


ROOT = Path(__file__).resolve().parents[1]


def test_all_torso_surfaces_have_exact_body_links() -> None:
    receipt = compile_body_links()
    assert receipt["schema"] == "HumanPack.organ-torso-body-link-registration.v1"
    assert len(receipt["bindings"]) == 304
    assert {row["layer"] for row in receipt["bindings"]} == {
        "organ", "vessel", "nerve", "airway", "pulmonary_artery", "pulmonary_vein"}
    assert {row["myosim_body"] for row in receipt["bindings"]} == {"torso", "Abdomen"}
    assert all(row["body_link_registration"] for row in receipt["bindings"])
    assert not receipt["qualification"]["organ_fem_or_mpm"]
    assert not receipt["qualification"]["blood_mass_owner"]
    assert not receipt["qualification"]["subject_calibration"]


def test_map_tampering_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    mapping = json.loads((ROOT / "config/bodyparts3d-myosim-torso-anatomy-map.v1.json").read_text())
    mapping["entries"][0]["myosim_body"] = "Abdomen"
    path.write_text(json.dumps(mapping))
    with pytest.raises(HumanImportError, match="visual payload is not bound"):
        compile_body_links(anatomy_map=path)


def test_visual_member_tampering_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = json.loads(VISUAL_MANIFEST.read_text())
    payload["source"]["surfaces"][0]["member_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(HumanImportError, match="manifest identity"):
        compile_body_links(visual_manifest=path)


def test_rehashed_native_receipt_cannot_admit_forged_member_hash(tmp_path):
    import hashlib
    path = tmp_path/VISUAL_MANIFEST.name
    manifest = json.loads(VISUAL_MANIFEST.read_text());manifest['source']['surfaces'][0]['member_sha256'] = '0'*64
    path.write_text(json.dumps(manifest));native = json.loads(NATIVE_RECEIPT.read_text())
    native['inputs']['payload_manifest']['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = tmp_path/'native.json';receipt.write_text(json.dumps(native))
    with pytest.raises(HumanImportError, match='source member hash/identity'):
        compile_body_links(visual_manifest=path,native_receipt=receipt)


def test_native_source_body_owner_cannot_drift(tmp_path):
    native = json.loads(NATIVE_RECEIPT.read_text());native['rows'][0]['core_body_index'] = 7
    path = tmp_path/'native.json';path.write_text(json.dumps(native))
    with pytest.raises(HumanImportError, match='native source body/member owner'):
        compile_body_links(native_receipt=path)


def test_partial_native_source_coverage_is_rejected(tmp_path):
    native = json.loads(NATIVE_RECEIPT.read_text());native['rows'].pop()
    path = tmp_path/'native.json';path.write_text(json.dumps(native))
    with pytest.raises(HumanImportError, match='audit coverage'):
        compile_body_links(native_receipt=path)
