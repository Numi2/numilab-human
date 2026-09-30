from fractions import Fraction
import json
from pathlib import Path

import numpy as np
import pytest

from tools.build_patellofemoral_intersection_loop import barycentric
from tools.verify_patellofemoral_surface import payload
from numilab_human.open_knee import parse_source


def test_exact_triangle_intersection_witness_reconstructs_point():
    triangle = ((0, 0, 0), (2, 0, 0), (0, 2, 0))
    point = (Fraction(1, 2), Fraction(1, 2), Fraction(0))
    assert barycentric(point, triangle) == (
        Fraction(1, 2), Fraction(1, 4), Fraction(1, 4))
    with pytest.raises(RuntimeError, match='lacks exact triangle barycentric'):
        barycentric((3, 0, 0), triangle)


def test_compiled_loop_receipt_reconstructs_both_cartilage_surfaces():
    root = Path(__file__).parents[1]
    source = parse_source(root / 'Sources/open-knee-oks003')
    report = json.loads((root /
        'Docs/media/patellofemoral-loop-20260930/receipt.json').read_text())
    intersections = json.loads((root /
        'Docs/media/patellofemoral-intersections-20260930/receipt.json').read_text())
    for side, stem in [('left', 'open-knee-oks003-left'),
                       ('right', 'open-knee-oks003-right-mirrored')]:
        directory = root / 'Build/patellofemoral-surface-20260930' / side
        decoded = payload(directory / f'{stem}.nhknee',
                          json.loads((directory / f'{stem}.manifest.json').read_text()),
                          source)
        row = report['compiled'][side]
        assert row['triangle_pair_indices'] == intersections['compiled'][side][
            'crossing_triangle_pairs']
        assert row['segment_count'] == row['unique_intersection_point_count'] == 18
        graph = {index: set() for index in range(18)}
        for segment in row['segments']:
            first, second = segment['point_ids']
            graph[first].add(second)
            graph[second].add(first)
            assert segment['opposing_face_normal_cosine'] < -.99
            for witness in segment['witnesses']:
                assert witness['point_id'] in (first, second)
                point = np.asarray(witness['position_m'])
                for name in ('patellar', 'femoral'):
                    nodes = segment[f'{name}_node_indices']
                    weights = np.asarray(witness[f'{name}_barycentric'])
                    assert abs(float(weights.sum()) - 1.) < 1e-14
                    actual = weights @ decoded['positions'][nodes]
                    assert float(np.linalg.norm(actual - point)) <= 1e-15
        assert all(len(neighbors) == 2 for neighbors in graph.values())
        cycle = row['cycle_point_ids']
        assert len(cycle) == len(set(cycle)) == 18
        assert all(cycle[(index + 1) % 18] in graph[cycle[index]]
                   for index in range(18))
