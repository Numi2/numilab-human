#!/usr/bin/env python3
"""Deterministic synthetic parity control for the retained vectorized facet bound."""
from pathlib import Path
import importlib.util
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "fixed_facet_quality_upperbound_002_vectorized_root.py"
spec = importlib.util.spec_from_file_location("facet_bound_vectorized_002", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load vectorized facet-bound source")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

rng = np.random.default_rng(20261101)
triangles = rng.normal(size=(100, 3, 3)).astype(np.float64)
triangles = np.concatenate(
    (
        triangles,
        np.asarray(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=np.float64,
        ),
    ),
    axis=0,
)
e01 = triangles[:, 1] - triangles[:, 0]
e02 = triangles[:, 2] - triangles[:, 0]
e12 = triangles[:, 2] - triangles[:, 1]
areas = 0.5 * np.linalg.norm(np.cross(e01, e02), axis=1)
lengths = np.maximum.reduce(
    (np.linalg.norm(e01, axis=1), np.linalg.norm(e02, axis=1), np.linalg.norm(e12, axis=1))
)
upper = np.divide(2.0 * areas, lengths * lengths, out=np.zeros_like(areas), where=lengths > 0)
scalar = np.asarray(
    [module.facet_upper_bound(triangle, (0, 1, 2)) for triangle in triangles],
    dtype=np.float64,
)
np.testing.assert_allclose(scalar[:, 0], areas, rtol=1e-14, atol=1e-15)
np.testing.assert_allclose(scalar[:, 1], lengths, rtol=1e-14, atol=1e-15)
np.testing.assert_allclose(scalar[:, 2], upper, rtol=1e-14, atol=1e-15)
assert areas[-2] == 0.0 and lengths[-2] > 0.0 and upper[-2] == 0.0
assert areas[-1] == 0.0 and lengths[-1] == 0.0 and upper[-1] == 0.0
print("PASS vectorized/scalar facet-bound parity", {"random_triangles": 100, "degenerate_controls": 2, "seed": 20261101})
