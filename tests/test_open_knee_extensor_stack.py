import numpy as np
import pytest

from numilab_human.open_knee import _extensor_stack_metrics


def test_patellar_cartilage_and_tendon_order_cannot_reverse():
    # X is anterior and Z proximal in this independent synthetic frame.
    points = {
        'PTB': np.array([[.050, 0., 0.]]),
        'PTC': np.array([[.040, 0., 0.]]),
        'QAT': np.array([[.050, 0., .040]]),
        'PTL': np.array([[.050, 0., -.040]]),
    }
    anterior = np.array([1., 0., 0.])
    proximal = np.array([0., 0., 1.])
    metrics = _extensor_stack_metrics(points, anterior, proximal, np)
    assert metrics['patellar_cartilage_posterior_to_bone_m'] == pytest.approx(.010)

    for region, wrong in [('PTC', [.060, 0., 0.]),
                          ('QAT', [.050, 0., -.040]),
                          ('PTL', [.050, 0., .040])]:
        candidate = dict(points, **{region: np.array([wrong])})
        with pytest.raises(RuntimeError, match='extensor stack reversed/too close'):
            _extensor_stack_metrics(candidate, anterior, proximal, np)
