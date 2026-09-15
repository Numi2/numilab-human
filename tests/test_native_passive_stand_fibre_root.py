from __future__ import annotations

from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.native_passive_stand_fibre_root import (
    compile_requalification,
    immutable_write,
)


def test_fibre_root_repair_closes_seed_ownership_but_keeps_long_horizon_open() -> None:
    result = compile_requalification()

    assert result["source"]["commit"] == "f45fcfdc80a05c2226d638227295add7f789c55c"
    assert result["repair"]["static_fibre_lengths_carried_into_runtime"]
    assert [row["step_count"] for row in result["cases"]] == [64, 512]
    assert [row["persistent_max_penetration_m"] for row in result["cases"]] == [0.0, 0.0]
    assert result["cases"][0]["persistent_max_acceleration_mps2"] < 1.0
    assert result["cases"][1]["persistent_max_acceleration_mps2"] > 1.0
    assert not result["qualification"]["force_convergence"]
    assert result["blocker"]["status"] == "open"


def test_fibre_root_requalification_rejects_tampered_source_patch(tmp_path: Path) -> None:
    root = tmp_path / "cases"
    root.mkdir()
    source = Path("Docs/media/native-passive-stand-fibre-root-20260915")
    for name in (
        "12p5us-64-stdout.txt", "12p5us-64-stderr.txt",
        "12p5us-512-stdout.txt", "12p5us-512-stderr.txt",
        "source.patch",
    ):
        (root / name).write_bytes((source / name).read_bytes())
    (root / "source.patch").write_text(
        (root / "source.patch").read_text(encoding="utf-8") + "\nchanged\n",
        encoding="utf-8",
    )
    with pytest.raises(ImportError, match="native passive stand fibre root"):
        compile_requalification(case_root=root)


def test_fibre_root_receipt_is_immutable(tmp_path: Path) -> None:
    result = compile_requalification()
    output = tmp_path / "receipt.json"
    digest = immutable_write(output, result)
    assert immutable_write(output, compile_requalification()) == digest
    with pytest.raises(ImportError, match="immutable"):
        immutable_write(output, {**result, "status": "changed"})
