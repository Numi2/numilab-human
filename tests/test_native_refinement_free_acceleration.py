from __future__ import annotations

from pathlib import Path

import numilab_human.native_passive_stand_refinement as refinement


def test_free_force_acceleration_is_not_relabelled_as_constrained_motion(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(refinement, "CASE_ROOT", tmp_path)

    def case(_root, name, timestep, steps, **_kwargs):
        index = [row[0] for row in refinement.CASE_SPEC].index(name)
        return {
            "name": name,
            "duration_seconds": timestep * steps,
            "timestep_nanoseconds": int(round(timestep * 1.0e9)),
            "step_count": steps,
            "compiled_stand_normalized_residual_rms": 1.0e-5,
            "persistent_max_acceleration_mps2": 20.0,
            "velocity_stage_diagnostics": {
                "maximum_free_acceleration_mixed_units": 100.0 + index,
                "maximum_constraint_delta_v_mixed_units": 0.01,
                "maximum_published_delta_v_mixed_units": 0.002,
            },
        }

    monkeypatch.setattr(refinement, "_case", case)
    result = refinement.compile_refinement(case_root=tmp_path)
    stage = result["convergence"]["velocity_stage"]

    assert result["compiler"].endswith(".4")
    assert "free_force_acceleration_range_over_minimum" in stage
    assert "maximum_free_force_acceleration_mixed_units" in stage
    assert "smooth_force_acceleration_range_over_minimum" not in stage
    assert "unconstrained" in stage["free_force_acceleration_semantics"]
    assert "not a constrained physical body acceleration" in stage[
        "free_force_acceleration_semantics"
    ]
    assert "unconstrained" in result["blocker"]["reason"].lower()
    assert not result["qualification"]["force_convergence"]
