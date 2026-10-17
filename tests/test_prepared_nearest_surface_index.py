"""Nearest-feature authoring regressions independent of anatomy admission."""
import numpy as np
import pytest
from numilab_human.bodyparts_adm import prepare_nearest_surface_index


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_query_retains_long_triangle_far_from_its_centroid(scale):
    # The long face is nearest although another centroid is much closer.
    points = np.asarray([[0,0,0],[1000,0,0],[0,1000,0],
                         [0,0,2],[1,0,2],[0,1,2]], dtype=float)*scale
    index = prepare_nearest_surface_index(points, np.asarray([[0,1,2],[3,4,5]]))
    distance, point, face = index.closest(np.asarray([.2,.2,.5])*scale)
    assert face == 0
    assert distance/scale == pytest.approx(.5)
    np.testing.assert_allclose(np.asarray(point)/scale, [.2,.2,0], atol=1e-12)


def test_query_handles_thin_valid_face_and_preserves_source_ordinals():
    p = [[0,0,0],[1,0,0],[0,1e-10,0]]
    index = prepare_nearest_surface_index(p, [[0,1,2],[0,1,2]])
    d, q, face = index.closest([.25,2e-11,.001])
    assert face == 0
    assert d == pytest.approx(.001)
    np.testing.assert_allclose(q, [.25,2e-11,0], atol=1e-14)
    d, q, face = index.closest([2,0,0])
    assert d == pytest.approx(1)
    np.testing.assert_allclose(q, [1,0,0])


def test_query_matches_bruteforce_reference_and_copies_inputs():
    from numilab_human.bodyparts_adm import _nearest_surface_point
    rng = np.random.default_rng(41)
    p = rng.normal(size=(90,3))*100
    f = np.arange(90).reshape(-1,3)
    index = prepare_nearest_surface_index(p,f)
    original = p.copy()
    p[:] = 0
    for point in rng.normal(size=(20,3))*100:
        distance, q, face = index.closest(point)
        expected, target = _nearest_surface_point(tuple(point), original.tolist(), f.tolist())
        assert distance == pytest.approx(expected, abs=1e-10)
        np.testing.assert_allclose(q, target, atol=1e-10)
        assert 0 <= face < len(f)


@pytest.mark.parametrize("defect", ["nan", "flat", "bad_index", "float_index"])
def test_prepare_rejects_invalid_source_geometry(defect):
    p = np.asarray([[0,0,0],[1,0,0],[0,1,0]], dtype=float)
    f = np.asarray([[0,1,2]])
    if defect == "nan": p[0,0] = np.nan
    elif defect == "flat": p[2] = [2,0,0]
    elif defect == "bad_index": f[0,2] = 3
    else: f = f.astype(float)
    with pytest.raises(ValueError):
        prepare_nearest_surface_index(p,f)


def test_query_rejects_nonfinite_point():
    index = prepare_nearest_surface_index([[0,0,0],[1,0,0],[0,1,0]], [[0,1,2]])
    with pytest.raises(ValueError): index.closest([0,np.inf,0])
