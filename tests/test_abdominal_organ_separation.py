"""Cross-surface intersection, nesting and separation are distinct states."""

import numpy as np

from numilab_human.abdominal_organ_separation import audit_pair, exact_integer_meshes


FACES = np.array([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)])
VERTICES = np.array([(0., 0., 0.), (1., 0., 0.),
                     (0., 1., 0.), (0., 0., 1.)])


def compare(second):
    denominator, meshes = exact_integer_meshes({
        'first': (VERTICES, FACES), 'second': (second, FACES),
    })
    return audit_pair(meshes['first'], meshes['second'], denominator)


def test_disjoint_closed_domains_have_zero_crossings_and_outside_parity():
    result = compare(VERTICES + np.array([2., 0., 0.]))
    assert result['status'] == 'separate_closed_domains'
    assert result['disjoint_closed_domains'] is True
    assert result['exact_intersecting_triangle_pairs'] == 0
    assert result['containment']['first_in_second']['location'] == 'outside'
    assert result['containment']['second_in_first']['location'] == 'outside'


def test_crossing_surfaces_fail_without_a_containment_claim():
    result = compare(VERTICES + np.array([.5, 0., 0.]))
    assert result['status'] == 'surface_crossing'
    assert result['disjoint_closed_domains'] is False
    assert result['exact_segment_or_polygon_crossing_pairs'] > 0
    assert result['maximum_intersection_segment_m'] > 0
    assert result['containment']['status'] == 'not_checked_intersecting_surfaces'


def test_nested_surfaces_are_not_separate_despite_zero_crossings():
    result = compare(VERTICES*.2 + np.array([.1, .1, .1]))
    assert result['status'] == 'nested_or_indeterminate'
    assert result['disjoint_closed_domains'] is False
    assert result['exact_intersecting_triangle_pairs'] == 0
    assert result['containment']['second_in_first']['location'] == 'inside'
