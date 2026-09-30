import numpy as np
import pytest
from types import SimpleNamespace

from numilab_human.open_knee import (
    _extensor_stack_metrics, _patellofemoral_surface_winding,
)


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


def test_patellofemoral_surface_winding_cannot_turn_into_the_cartilage():
    # One positive tetrahedron per region; world anterior is negative Y.
    coordinates = np.array([[0., 0., 0.], [1., 0., 0.],
                            [0., .1, 0.], [0., 0., 1.]])
    source = SimpleNamespace(
        regions={name: SimpleNamespace(
            node_ids=ids, elements=[tuple(ids)])
            for name, ids in [('PTC', (1, 2, 3, 4)),
                              ('FMC', (11, 12, 13, 14))]},
        surfaces={
            'PTC_@_FMC_ContactFaces': SimpleNamespace(faces=[(2, 3, 4)]),
            'PTC_@_PTB_TiesFaces': SimpleNamespace(faces=[(1, 2, 4)]),
            'FMC_@_PTC_ContactFaces': SimpleNamespace(faces=[(11, 12, 14)]),
            'FMC_@_FMB_TiesFaces': SimpleNamespace(faces=[(12, 13, 14)]),
        },
    )
    positions = {'PTC': coordinates, 'FMC': coordinates.copy()}
    measured = _patellofemoral_surface_winding(
        source, positions, reflected=False, np=np)
    assert measured['PTC_@_FMC_ContactFaces']['area_weighted_anterior_cosine'] < -.7
    assert measured['PTC_@_PTB_TiesFaces']['area_weighted_anterior_cosine'] > .7

    mirrored = {name: value * np.array([-1., 1., 1.])
                for name, value in positions.items()}
    _patellofemoral_surface_winding(source, mirrored, reflected=True, np=np)

    source.surfaces['PTC_@_FMC_ContactFaces'].faces = [(3, 2, 4)]
    with pytest.raises(RuntimeError, match='inward face'):
        _patellofemoral_surface_winding(source, positions, reflected=False, np=np)
