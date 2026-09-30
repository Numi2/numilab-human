import numpy as np

from tools.verify_patellofemoral_intersections import _intersections


def test_exact_patellofemoral_pair_detects_crossing_and_separation():
    points = np.array([[0., 0., 0.], [1., 0., 0.],
                       [0., 1., 0.], [0., 0., 1.]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    clear = _intersections({'PTC': (points, faces),
                            'FMC': (points + [2., 0., 0.], faces)})
    assert clear['static_noninterpenetration_qualified'] is True
    assert clear['exact_segment_or_polygon_crossing_pairs'] == 0

    intersecting = _intersections({'PTC': (points, faces),
                                   'FMC': (points + [.25, .25, .25], faces)})
    assert intersecting['static_noninterpenetration_qualified'] is False
    assert intersecting['exact_segment_or_polygon_crossing_pairs'] == 3
    assert intersecting['maximum_intersection_segment_m'] > 0

    touching = _intersections({'PTC': (points, faces),
                               'FMC': (points + [1., 0., 0.], faces)})
    assert touching['exact_segment_or_polygon_crossing_pairs'] == 0
    assert touching['exact_point_contact_pairs'] == 9
    assert touching['static_noninterpenetration_qualified'] is False
