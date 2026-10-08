from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import eye

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "tools/derive_constrained_ocular_skin_registration_candidate_v13.py"
SPEC = importlib.util.spec_from_file_location("eye_skin_candidate_906", RUNNER)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class _AreaConstraint:
    def __init__(self, ratio: float):
        self.ratio = ratio

    def fun(self, _x):
        return np.asarray([self.ratio], dtype=np.float64)


def _problem(area_ratio: float):
    return SimpleNamespace(
        H=eye(3, format="csr", dtype=np.float64),
        base_mm=np.zeros((1, 3), dtype=np.float64),
        patch_vertices=np.asarray([0], dtype=np.int64),
        fixed_delta_mm={},
        free_index={0: 0},
        edge_pairs=np.empty((0, 2), dtype=np.int64),
        edge_weights=np.empty((0,), dtype=np.float64),
        radial_rows=[],
        plan={"area_floor_ratio": 0.05},
        nonlinear_area_constraint=_AreaConstraint(area_ratio),
        linear_constraints=None,
        initial_x=np.zeros((3,), dtype=np.float64),
    )


def _result(*, success: bool, violation: float, area_ratio: float):
    return SimpleNamespace(
        x=np.zeros((3,), dtype=np.float64),
        success=success,
        status=1 if success else 0,
        message="converged" if success else "The maximum number of function evaluations is exceeded.",
        niter=17 if success else 400,
        fun=0.0,
        optimality=0.0 if success else 1.0e-5,
        constr_violation=violation,
        lagrangian_grad=np.zeros((3,), dtype=np.float64),
    )


def test_feasible_capped_iterate_is_returned_only_for_separate_geometry_audit(monkeypatch):
    calls = {}

    def fake_minimize(fun, x0, *, method, jac, hess, constraints, options):
        calls.update(method=method, constraints=constraints, options=options)
        return _result(success=False, violation=0.0, area_ratio=0.05000001)

    monkeypatch.setattr(audit, "minimize", fake_minimize)
    delta, report = audit.solve_local_problem(_problem(0.05000001), np.zeros((3,)), maxiter=400)

    assert calls["method"] == "trust-constr"
    assert calls["options"] == {
        "maxiter": 400,
        "gtol": 1.0e-8,
        "xtol": 1.0e-10,
        "barrier_tol": 1.0e-10,
        "sparse_jacobian": True,
        "verbose": 0,
    }
    assert len(calls["constraints"]) == 1
    assert delta.shape == (1, 3)
    assert report["success"] is False
    assert report["optimizer_converged"] is False
    assert report["feasible_iteration_captured_for_geometry_audit"] is True


def test_infeasible_capped_iterate_is_rejected(monkeypatch):
    monkeypatch.setattr(
        audit, "minimize",
        lambda *args, **kwargs: _result(success=False, violation=1.0e-4, area_ratio=0.06),
    )

    with pytest.raises(RuntimeError, match="no feasible iterate available"):
        audit.solve_local_problem(_problem(0.06), np.zeros((3,)), maxiter=400)


def test_area_floor_violation_is_rejected_even_if_optimizer_says_success(monkeypatch):
    monkeypatch.setattr(
        audit, "minimize",
        lambda *args, **kwargs: _result(success=True, violation=0.0, area_ratio=0.049),
    )

    with pytest.raises(RuntimeError, match="no feasible iterate available"):
        audit.solve_local_problem(_problem(0.049), np.zeros((3,)), maxiter=400)


def test_converged_feasible_iterate_is_labeled_as_converged(monkeypatch):
    monkeypatch.setattr(
        audit, "minimize",
        lambda *args, **kwargs: _result(success=True, violation=0.0, area_ratio=0.05),
    )

    _delta, report = audit.solve_local_problem(_problem(0.05), np.zeros((3,)), maxiter=400)

    assert report["success"] is True
    assert report["optimizer_converged"] is True
    assert report["feasible_iteration_captured_for_geometry_audit"] is False
