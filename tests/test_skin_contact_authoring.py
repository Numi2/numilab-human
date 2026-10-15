"""Exact contact preparation regressions, independent of native admission."""
from fractions import Fraction as F
import importlib.util
from pathlib import Path
import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "tools" / "repair_resting_skin_contacts.py"
spec = importlib.util.spec_from_file_location("skin_contact_authoring", path)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)

def test_meter_coordinates_preserve_dimensionless_contact_weights():
    triangle = [(F(0),F(0),F(0)), (F(1,1000),F(0),F(0)), (F(0),F(2,1000),F(0))]
    weights = tool.barycentric(triangle, (F(1,4000),F(1,1000),F(0)))
    np.testing.assert_array_equal(weights, [.25,.25,.5])
    displacement_m = np.array([[0.,0.,0.],[.004,0.,0.],[0.,0.,0.]])
    assert float(weights @ displacement_m @ np.array([1.,0.,0.])) == .001

def test_tiny_nonzero_face_uses_exact_dominant_axis():
    e = F(1,10**20)
    triangle = [(F(1),F(2),F(3)), (F(1),F(2)+e,F(3)), (F(1),F(2),F(3)+e)]
    weights = tool.barycentric(triangle, (F(1),F(2)+e/4,F(3)+e/4))
    np.testing.assert_array_equal(weights, [.5,.25,.25])

def test_contact_edge_has_exact_zero_weight():
    triangle = [(F(0),F(0),F(0)), (F(1),F(0),F(0)), (F(0),F(1),F(0))]
    np.testing.assert_array_equal(tool.barycentric(triangle,(F(1,4),F(3,4),F(0))),[0.,.25,.75])

def test_outside_contact_is_rejected():
    triangle = [(F(0),F(0),F(0)), (F(1),F(0),F(0)), (F(0),F(1),F(0))]
    with pytest.raises(ValueError, match="outside"):
        tool.barycentric(triangle,(F(1),F(1),F(0)))

def test_exact_degenerate_face_is_rejected():
    triangle = [(F(0),F(0),F(0)), (F(1),F(0),F(0)), (F(2),F(0),F(0))]
    with pytest.raises(ValueError, match="degenerate"):
        tool.barycentric(triangle,(F(1),F(0),F(0)))
