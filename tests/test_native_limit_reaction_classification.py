from __future__ import annotations

from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.native_limit_reaction_classification import (
    SCHEMA,
    compile_receipt,
)


INVALID = str(0xFFFFFFFF)


def _summary(**overrides: str) -> str:
    fields = {
        "myosim_articulated_mechanics": "ok",
        "core_bodies": "157",
        "compiled_stand_recruited_muscles": "416",
        "compiled_stand_balanced": "true",
        "persistent_root_assistance": "none",
        "compiled_stand_active_limits": "30",
        "compiled_stand_active_structural_locks": "24",
        "compiled_stand_active_finite_range_limits": "6",
        "compiled_stand_max_limit_reaction": "1348.0",
        "compiled_stand_max_structural_lock_reaction": "1348.0",
        "compiled_stand_max_structural_lock_reaction_dof": "9",
        "compiled_stand_max_finite_range_limit_reaction": "41.0",
        "compiled_stand_max_finite_range_limit_reaction_dof": "109",
    }
    fields.update(overrides)
    return " ".join(f'{key}="{value}"' for key, value in fields.items()) + "\n"


def _write(path: Path, **overrides: str) -> Path:
    path.write_text("prefix\n" + _summary(**overrides), encoding="utf-8")
    return path


def test_valid_classification_is_partial_and_reconstructs_total(tmp_path: Path) -> None:
    stdout = _write(tmp_path / "stdout.txt")
    receipt = compile_receipt(
        stdout,
        source_commit="a" * 40,
        binary_sha256="b" * 64,
    )
    assert receipt["schema"] == SCHEMA
    assert receipt["status"] == "partial"
    classification = receipt["classification"]
    assert classification["active_total"] == 30
    assert classification["structural_locks"]["active_count"] == 24
    assert classification["finite_range_stops"]["active_count"] == 6
    assert classification["structural_locks"]["maximum_reaction_dof"] == 9
    assert classification["finite_range_stops"]["maximum_reaction_dof"] == 109
    assert receipt["qualification"]["reaction_classes_measured"]
    assert not receipt["qualification"]["finite_range_stop_dependence_resolved"]
    assert not receipt["qualification"]["generalized_force_convergence"]
    assert not receipt["qualification"]["sustained_standing"]


def test_empty_finite_range_class_is_admitted_with_invalid_index(tmp_path: Path) -> None:
    stdout = _write(
        tmp_path / "stdout.txt",
        compiled_stand_active_limits="24",
        compiled_stand_active_finite_range_limits="0",
        compiled_stand_max_finite_range_limit_reaction="0",
        compiled_stand_max_finite_range_limit_reaction_dof=INVALID,
    )
    receipt = compile_receipt(stdout)
    assert receipt["classification"]["finite_range_stops"] == {
        "active_count": 0,
        "maximum_reaction": 0.0,
        "maximum_reaction_dof": None,
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"persistent_root_assistance": "enabled"}, "assistance"),
        ({"core_bodies": "156"}, "157-body"),
        ({"compiled_stand_recruited_muscles": "415"}, "416"),
        ({"compiled_stand_balanced": "false"}, "not balanced"),
        ({"compiled_stand_active_limits": "29"}, "do not reconstruct"),
        ({"compiled_stand_active_structural_locks": "129"}, "outside"),
        ({"compiled_stand_max_structural_lock_reaction": "nan"}, "finite"),
        ({"compiled_stand_max_finite_range_limit_reaction_dof": "128"}, "outside"),
        ({"compiled_stand_max_limit_reaction": "999"}, "maxima"),
    ],
)
def test_invalid_classification_is_rejected(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    stdout = _write(tmp_path / "stdout.txt", **overrides)
    with pytest.raises(ImportError, match=message):
        compile_receipt(stdout)


def test_empty_class_requires_zero_reaction_and_invalid_index(tmp_path: Path) -> None:
    nonzero = _write(
        tmp_path / "nonzero.txt",
        compiled_stand_active_limits="24",
        compiled_stand_active_finite_range_limits="0",
        compiled_stand_max_finite_range_limit_reaction="1",
        compiled_stand_max_finite_range_limit_reaction_dof=INVALID,
    )
    with pytest.raises(ImportError, match="must be zero"):
        compile_receipt(nonzero)

    indexed = _write(
        tmp_path / "indexed.txt",
        compiled_stand_active_limits="24",
        compiled_stand_active_finite_range_limits="0",
        compiled_stand_max_finite_range_limit_reaction="0",
        compiled_stand_max_finite_range_limit_reaction_dof="7",
    )
    with pytest.raises(ImportError, match="invalid index"):
        compile_receipt(indexed)


def test_missing_or_duplicate_summary_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    missing.write_text("no result\n", encoding="utf-8")
    with pytest.raises(ImportError, match="exactly one"):
        compile_receipt(missing)

    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text(_summary() + _summary(), encoding="utf-8")
    with pytest.raises(ImportError, match="exactly one"):
        compile_receipt(duplicate)


def test_source_identities_must_be_full_lowercase_hashes(tmp_path: Path) -> None:
    stdout = _write(tmp_path / "stdout.txt")
    with pytest.raises(ImportError, match="Git SHA"):
        compile_receipt(stdout, source_commit="abc")
    with pytest.raises(ImportError, match="SHA-256"):
        compile_receipt(stdout, binary_sha256="ABC")
