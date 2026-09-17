from __future__ import annotations

from pathlib import Path

import pytest

import numilab_human.native_passive_stand_refinement as refinement
from numilab_human.model import ImportError


def _write_case(root: Path, *, omit: str | None = None, pre_delta: float = 0.002) -> None:
    directory = root / "100us"
    directory.mkdir()
    fields = {
        "myosim_articulated_mechanics": "ok",
        "muscle_step_seconds": "0.0001",
        "muscle_step_count": "64",
        "persistent_completed_steps": "64",
        "persistent_metal_horizon": "true",
        "persistent_source_passive_joint_tissue": "true",
        "compiled_stand_balanced": "true",
        "stand_deterministic_replay": "bitwise",
        "source_support_metal_device": "Apple M4 Pro",
        "persistent_max_penetration_m": "0",
        "persistent_max_acceleration": "20",
        "compiled_stand_normalized_residual_rms": "0.001",
        "source_dynamic_force_parity_max_delta_n": "0.01",
        "muscle_step_max_velocity_delta": "0.0001",
        "muscle_step_max_configuration_delta": "0.00001",
        "persistent_max_acceleration_semantics": "pre_projection_total_delta_v_divided_by_timestep",
        "persistent_max_free_acceleration": "1.5",
        "persistent_max_free_acceleration_dof": "7",
        "persistent_max_constraint_delta_v": "0.002",
        "persistent_max_constraint_delta_v_dof": "108",
        "persistent_max_pre_projection_delta_v": str(pre_delta),
        "persistent_max_pre_projection_delta_v_dof": "108",
        "persistent_max_published_delta_v": "0.0021",
        "persistent_max_published_delta_v_dof": "109",
    }
    if omit is not None:
        fields.pop(omit)
    line = " ".join(f'{key}="{value}"' if key == "source_support_metal_device" else f"{key}={value}"
                    for key, value in fields.items())
    (directory / "stdout.txt").write_text(line + "\n")
    (directory / "stderr.txt").write_text("")


def test_velocity_stages_are_bound_without_relabeling_constraint_impulse(tmp_path: Path) -> None:
    _write_case(tmp_path)
    row = refinement._case(tmp_path, "100us", 1.0e-4, 64)
    stage = row["velocity_stage_diagnostics"]
    assert stage["schema"] == "numi.human.velocity-stage-diagnostics.v1"
    assert stage["maximum_free_acceleration_mixed_units"] == 1.5
    assert stage["maximum_constraint_delta_v_mixed_units"] == 0.002
    assert stage["reconstructed_pre_projection_rate_mixed_units"] == 20.0
    assert row["persistent_max_acceleration_mps2"] == 20.0


def test_partial_velocity_stage_record_fails_closed(tmp_path: Path) -> None:
    _write_case(tmp_path, omit="persistent_max_constraint_delta_v_dof")
    with pytest.raises(ImportError, match="partial velocity-stage"):
        refinement._case(tmp_path, "100us", 1.0e-4, 64)


def test_legacy_rate_must_reconstruct_preprojection_delta_v(tmp_path: Path) -> None:
    _write_case(tmp_path, pre_delta=0.003)
    with pytest.raises(ImportError, match="does not reconstruct"):
        refinement._case(tmp_path, "100us", 1.0e-4, 64)
