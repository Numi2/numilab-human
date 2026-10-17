"""Boundary inventory is diagnostic and never substitutes for solid admission."""
from unittest.mock import patch

import numpy as np
import pytest

from numilab_human import resting_anatomy_verify as v

F = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)


def surface(number, scale=1., offset=0.):
    return v.prepare_surface((51011, number), P * np.float32(scale) + np.float32(offset), F)


def test_boundary_pair_defers_containment_and_never_passes_nested_solids():
    outer, inner = surface(1), surface(2, .2, .1)
    with patch.object(v, "_contained_components", side_effect=AssertionError("deferred")):
        result = v.audit_surface_pair(inner, outer, boundary_only=True)
    assert not result["passed"] and result["boundary_passed"]
    assert result["containment_deferred"] and result["containment"] == []
    assert outer._self_result is None and inner._self_result is None
    full = v.audit_surface_pair(inner, outer)
    assert not full["passed"] and full["containment"][0]["inside_component_witnesses"]


def test_inventory_collects_every_boundary_failure_and_defers_target_self():
    first = surface(1)
    targets = [surface(2, offset=.1), surface(3, offset=.2), surface(4, offset=5.)]
    surfaces = {s.key: s for s in [first, *targets]}
    progress = []
    with patch.object(v, "_contained_components", side_effect=AssertionError("deferred")):
        result = v.audit_changed_surfaces(
            surfaces, [first.key], boundary_inventory_only=True, progress=progress.append)
    assert not result["passed"] and not result["boundary_stage_passed"]
    assert result["containment_pending"] and result["collection_complete"]
    assert result["failure_count"] == 2 and result["failure_row_indices"] == [1, 2]
    assert result["rows"] == progress and first._self_result is not None
    assert all(s._self_result is None for s in targets)
    pairs = result["rows"][1:]
    assert all(r["stage"] == "boundary_pair" and not r["passed"] for r in pairs)
    assert all(r["containment_deferred"] for r in pairs)
    assert all(r["unexpected_triangle_pairs"] for r in pairs[:2])
    assert pairs[-1]["boundary_passed"] and pairs[-1]["aabb_disjoint"]


def test_clean_inventory_cannot_be_admitted_even_when_empty():
    first, far = surface(1), surface(2, offset=5.)
    for targets in ([], [far.key]):
        result = v.audit_changed_surfaces(
            {first.key: first, far.key: far}, [first.key], targets=targets,
            boundary_inventory_only=True)
        assert not result["passed"] and result["boundary_stage_passed"]
        assert result["containment_pending"]
    assert v.audit_changed_surfaces({first.key: first, far.key: far}, [first.key])["passed"]


def test_boundary_inventory_keeps_changed_self_failure_and_typed_identity_checks():
    folded = v.prepare_surface((51005, 1), np.concatenate((P, P + np.float32(.2))),
                               np.concatenate((F, F + 4)))
    result = v.audit_changed_surfaces({folded.key: folded}, [folded.key],
                                     boundary_inventory_only=True)
    assert not result["boundary_stage_passed"] and result["failed_stage"] == "self"
    with pytest.raises(ValueError, match="typed identity"):
        v.audit_changed_surfaces({(51005, 2): folded}, [(51005, 2)],
                                 boundary_inventory_only=True)


def test_boundary_options_require_actual_booleans():
    first, other = surface(1), surface(2)
    for value in (1, [], None, "yes"):
        with pytest.raises(TypeError, match="boundary_only"):
            v.audit_surface_pair(first, other, boundary_only=value)
        with pytest.raises(TypeError, match="boundary_inventory_only"):
            v.audit_changed_surfaces({first.key: first}, [first.key],
                                     boundary_inventory_only=value)
