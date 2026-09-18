from __future__ import annotations

from pathlib import Path

import pytest

from numilab_human.model import ImportError
from numilab_human.native_acceleration_force_parity import (
    EXPECTED_SEMANTICS,
    SCHEMA,
    compile_receipt,
)


def _summary(**overrides: str) -> str:
    fields = {
        "myosim_articulated_mechanics": "ok",
        "core_bodies": "157",
        "compiled_stand_recruited_muscles": "416",
        "compiled_stand_balanced": "true",
        "compiled_stand_normalized_residual_rms": "0.00814",
        "persistent_root_assistance": "none",
        "muscle_step_seconds": "0.0001",
        "source_dynamic_force_parity_max_delta_n": "8.0",
        "source_dynamic_force_parity_max_delta_v_mixed_units": "0.002",
        "source_dynamic_force_parity_max_delta_acceleration_mixed_units": "20.0",
        "source_dynamic_force_parity_max_delta_dof": "108",
        "source_dynamic_force_parity_acceleration_semantics": EXPECTED_SEMANTICS,
    }
    fields.update(overrides)
    return " ".join(f'{key}="{value}"' for key, value in fields.items()) + "\n"


def _write(path: Path, **overrides: str) -> Path:
    path.write_text("prefix\n" + _summary(**overrides), encoding="utf-8")
    return path


def test_valid_diagnostic_is_partial_and_never_promotes_standing(tmp_path: Path) -> None:
    stdout = _write(tmp_path / "stdout.txt")
    receipt = compile_receipt(
        stdout,
        source_commit="a" * 40,
        binary_sha256="b" * 64,
    )
    assert receipt["schema"] == SCHEMA
    assert receipt["status"] == "partial"
    assert receipt["diagnostic"]["maximum_acceleration_delta_mixed_units"] == 20.0
    assert receipt["diagnostic"]["maximum_acceleration_delta_dof"] == 108
    assert receipt["qualification"]["same_operator_muscle_force_acceleration_parity_measured"]
    assert not receipt["qualification"]["complete_dynamic_force_assembly"]
    assert not receipt["qualification"]["generalized_force_convergence"]
    assert not receipt["qualification"]["sustained_standing"]


def test_duplicate_summary_metric_is_rejected(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(
        _summary().rstrip() + " source_dynamic_force_parity_max_delta_dof=7\n",
        encoding="utf-8",
    )
    with pytest.raises(ImportError, match="repeats"):
        compile_receipt(stdout)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"persistent_root_assistance": "enabled"}, "assistance"),
        ({"core_bodies": "156"}, "157-body"),
        ({"compiled_stand_recruited_muscles": "415"}, "416"),
        ({"compiled_stand_balanced": "false"}, "not balanced"),
        ({"source_dynamic_force_parity_max_delta_dof": "128"}, "outside"),
        ({"source_dynamic_force_parity_max_delta_acceleration_mixed_units": "nan"}, "finite"),
        ({"source_dynamic_force_parity_acceleration_semantics": "wrong"}, "semantics"),
    ],
)
def test_invalid_execution_contract_is_rejected(
    tmp_path: Path, overrides: dict[str, str], message: str
) -> None:
    stdout = _write(tmp_path / "stdout.txt", **overrides)
    with pytest.raises(ImportError, match=message):
        compile_receipt(stdout)


def test_acceleration_must_equal_velocity_increment_over_timestep(tmp_path: Path) -> None:
    stdout = _write(
        tmp_path / "stdout.txt",
        source_dynamic_force_parity_max_delta_acceleration_mixed_units="19.0",
    )
    with pytest.raises(ImportError, match="inconsistent"):
        compile_receipt(stdout)


def test_missing_or_ambiguous_native_summary_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    missing.write_text("no result\n", encoding="utf-8")
    with pytest.raises(ImportError, match="exactly one"):
        compile_receipt(missing)

    ambiguous = tmp_path / "ambiguous.txt"
    ambiguous.write_text(_summary() + _summary(), encoding="utf-8")
    with pytest.raises(ImportError, match="exactly one"):
        compile_receipt(ambiguous)


def test_source_identities_must_be_full_lowercase_hashes(tmp_path: Path) -> None:
    stdout = _write(tmp_path / "stdout.txt")
    with pytest.raises(ImportError, match="Git SHA"):
        compile_receipt(stdout, source_commit="abc")
    with pytest.raises(ImportError, match="SHA-256"):
        compile_receipt(stdout, binary_sha256="ABC")


@pytest.mark.parametrize("residual", ["0.0824449059063", "0.0500000001", "-0.01", "nan", "inf", "invalid"])
def test_balance_flag_cannot_override_numeric_residual(tmp_path: Path, residual: str) -> None:
    stdout = _write(tmp_path / "stdout.txt", compiled_stand_normalized_residual_rms=residual)
    with pytest.raises(ImportError, match="residual"):
        compile_receipt(stdout)


@pytest.mark.parametrize("residual", ["0", "0.05"])
def test_static_balance_numeric_boundary_is_preserved(tmp_path: Path, residual: str) -> None:
    stdout = _write(tmp_path / "stdout.txt", compiled_stand_normalized_residual_rms=residual)
    receipt = compile_receipt(stdout)
    assert receipt["model"]["compiled_static_residual_rms"] == float(residual)
    assert receipt["model"]["maximum_static_balance_residual_rms"] == 0.05
    assert receipt["qualification"]["static_balance_residual_verified"] is True
    assert receipt["qualification"]["generalized_force_convergence"] is False


def test_missing_balance_residual_is_rejected(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout.txt"
    stdout.write_text(_summary().replace(' compiled_stand_normalized_residual_rms="0.00814"', ''), encoding="utf-8")
    with pytest.raises(ImportError, match="missing.*residual"):
        compile_receipt(stdout)
