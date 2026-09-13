from __future__ import annotations

import json
from pathlib import Path

import pytest

from numilab_human.model import ImportError as HumanImportError
from numilab_human.torso_body_links import compile_body_links


ROOT = Path(__file__).resolve().parents[1]


def test_all_torso_surfaces_have_exact_body_links() -> None:
    receipt = compile_body_links()
    assert receipt["schema"] == "HumanPack.organ-torso-body-link-registration.v1"
    assert len(receipt["bindings"]) == 12
    assert {row["layer"] for row in receipt["bindings"]} == {"organ", "vessel", "nerve"}
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
    payload = json.loads((ROOT / "Docs/media/native-torso-anatomy-20260913/bodyparts3d-myosim-torso-anatomy.manifest.json").read_text())
    payload["source"]["surfaces"][0]["member_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(HumanImportError, match="manifest identity"):
        compile_body_links(visual_manifest=path)
