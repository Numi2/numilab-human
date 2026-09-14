from __future__ import annotations

from pathlib import Path

import pytest

from numilab_human.skin_shell_native_visual import (
    SkinShellNativeVisualError,
    _immutable_write,
    compile_candidate,
)


def test_native_skin_shell_visual_is_four_view_admitted() -> None:
    result = compile_candidate()

    assert result["status"] == "qualified"
    assert result["source"]["native_registration_fingerprint32"] == "6a48e223"
    assert result["capture"]["metal_pose_device"] == "Apple M4 Pro"
    assert result["capture"]["renderer_device"] == "Apple M4 Pro"
    assert [row["view"] for row in result["capture"]["views"]] == [
        "front", "oblique", "side", "rear"
    ]
    assert all(row["skin_shell_pixels"] > 0 for row in result["capture"]["views"])
    assert result["qualification"]["native_visual_admission"]
    assert not result["qualification"]["skin_physical_volume"]
    assert not result["qualification"]["skin_deformation"]


def test_native_skin_shell_visual_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_candidate()
    output = tmp_path / "receipt.json"
    first = _immutable_write(output, result)
    assert first == _immutable_write(output, compile_candidate())
    with pytest.raises(SkinShellNativeVisualError, match="immutable"):
        _immutable_write(output, {**result, "status": "changed"})


def test_native_skin_shell_visual_rejects_zero_pixel_view(tmp_path: Path) -> None:
    source = Path("Docs/media/skin-shell-native-visual-20260914/native.stdout.log")
    value = source.read_text(encoding="utf-8").replace(
        "skin_shell_pixels=29661", "skin_shell_pixels=0", 1
    )
    stdout = tmp_path / "native.stdout.log"
    stdout.write_text(value, encoding="utf-8")
    with pytest.raises(SkinShellNativeVisualError, match="no skin-shell pixels"):
        compile_candidate(stdout=stdout)
