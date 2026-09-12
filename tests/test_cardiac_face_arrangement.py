"""Exact geometry regressions for constraint-preserving face subdivision."""
from __future__ import annotations

from collections import Counter, defaultdict
from fractions import Fraction as F
import json
from pathlib import Path
import unittest

from numilab_human.cardiac_face_arrangement import subdivide_triangle
from numilab_human.model import ImportError as HumanImportError

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ((0, 0, 0), (4, 0, 0), (0, 4, 0))


def point(value):
    return tuple(F(x) for x in value)


def subtract(a, b):
    return tuple(x-y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def normal(triangle):
    return cross(subtract(triangle[1], triangle[0]), subtract(triangle[2], triangle[0]))


class ArrangementTests(unittest.TestCase):
    def verify(self, source, result, constraints=()):
        source = tuple(point(p) for p in source)
        reference = normal(source)
        self.assertTrue(result)
        total = [F(0)] * 3
        edge_counts = Counter()
        for triangle in result:
            self.assertTrue(all(isinstance(x, F) for p in triangle for x in p))
            area = normal(triangle)
            self.assertGreater(sum(a*b for a, b in zip(area, reference)), 0)
            total = [a+b for a, b in zip(total, area)]
            for a, b in zip(triangle, triangle[1:] + triangle[:1]):
                edge_counts[tuple(sorted((a, b)))] += 1
        self.assertEqual(tuple(total), reference)
        self.assertTrue(all(n in (1, 2) for n in edge_counts.values()))
        vertices = {p for t in result for p in t}
        # Independently reject any emitted edge passing through another vertex.
        for a, b in edge_counts:
            direction = subtract(b, a)
            axis = next(i for i, d in enumerate(direction) if d)
            for p in vertices - {a, b}:
                if not any(cross(subtract(p, a), direction)):
                    parameter = (p[axis]-a[axis]) / direction[axis]
                    self.assertFalse(0 < parameter < 1)
        # Every source edge and input cut must be a complete path of output edges.
        paths = list(constraints) + list(zip(source, source[1:] + source[:1]))
        for a, b in paths:
            a, b = point(a), point(b)
            direction = subtract(b, a)
            axis = next(i for i, d in enumerate(direction) if d)
            intervals = []
            for x, y in edge_counts:
                if any(cross(subtract(x, a), direction)) or any(cross(subtract(y, a), direction)):
                    continue
                start, end = sorted(((x[axis]-a[axis])/direction[axis], (y[axis]-a[axis])/direction[axis]))
                if 0 <= start < end <= 1:
                    intervals.append((start, end))
            intervals.sort()
            self.assertTrue(intervals)
            self.assertEqual(intervals[0][0], 0)
            self.assertEqual(intervals[-1][1], 1)
            self.assertTrue(all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])))
        return edge_counts

    def test_no_cuts_preserves_source_triangle(self):
        result = subdivide_triangle(SOURCE, [])
        self.assertEqual(len(result), 1)
        self.verify(SOURCE, result)

    def test_crosscut_partitions_without_losing_constraint(self):
        cut = ((2, 0, 0), (0, 2, 0))
        result = subdivide_triangle(SOURCE, [cut])
        self.assertEqual(len(result), 3)
        self.verify(SOURCE, result, [cut])

    def test_crossed_segments_create_exact_shared_interior_vertex(self):
        cuts = [((2, 0, 0), (0, 2, 0)), ((1, 0, 0), (1, 3, 0))]
        result = subdivide_triangle(SOURCE, cuts)
        self.assertEqual(len(result), 7)
        self.assertIn(point((1, 1, 0)), {p for t in result for p in t})
        self.verify(SOURCE, result, cuts)

    def test_segment_endpoint_on_cut_and_collinear_overlap_are_conforming(self):
        cases = [[((0, 1, 0), (3, 1, 0)), ((1, 0, 0), (1, 1, 0))],
                 [((2, 0, 0), (0, 2, 0)), ((F(3, 2), F(1, 2), 0), (F(1, 2), F(3, 2), 0))],
                 [((2, 0, 0), (0, 2, 0)), ((0, 2, 0), (2, 0, 0))]]
        for cuts in cases:
            with self.subTest(cuts=cuts):
                self.verify(SOURCE, subdivide_triangle(SOURCE, cuts), cuts)

    def test_shared_edge_points_propagate_identically_to_neighboring_faces(self):
        other = ((4, 0, 0), (4, 4, 0), (0, 4, 0))
        propagated = [(1, 3, 0), (2, 2, 0), (3, 1, 0)]
        first = subdivide_triangle(SOURCE, [], iter(propagated))
        second = subdivide_triangle(other, [], reversed(propagated))
        first_edges = self.verify(SOURCE, first)
        second_edges = self.verify(other, second)
        shared = set(first_edges) & set(second_edges)
        expected = list(map(point, [(4, 0, 0), (3, 1, 0), (2, 2, 0), (1, 3, 0), (0, 4, 0)]))
        self.assertEqual(shared, {tuple(sorted((a, b))) for a, b in zip(expected, expected[1:])})
        self.assertTrue(all(first_edges[e] == second_edges[e] == 1 for e in shared))

    def test_reordered_constraints_and_points_have_identical_triangulation(self):
        cuts = [((2, 0, 0), (0, 2, 0)), ((1, 0, 0), (1, 3, 0))]
        boundary = [(3, 0, 0), (0, 3, 0)]
        first = subdivide_triangle(SOURCE, cuts, boundary)
        second = subdivide_triangle(SOURCE, [tuple(reversed(c)) for c in reversed(cuts)], reversed(boundary))
        self.assertEqual(first, second)

    def test_winding_and_oblique_plane_are_preserved(self):
        cut = ((2, 0, 0), (0, 2, 0))
        reverse = tuple(reversed(SOURCE))
        result = subdivide_triangle(reverse, [cut])
        self.verify(reverse, result, [cut])
        def transform(p):
            x, y, _ = p
            return (F(3)+2*x-y, F(-7)+x+3*y, F(11)+4*x+2*y)
        source = tuple(transform(p) for p in SOURCE)
        cut = tuple(transform(p) for p in cut)
        self.verify(source, subdivide_triangle(source, [cut]), [cut])

    def test_arbitrarily_small_rational_cuts_are_not_welded_away(self):
        tiny = F(1, 10**80)
        cut = ((tiny, 0, 0), (0, tiny, 0))
        result = subdivide_triangle(SOURCE, [cut])
        self.assertEqual(len(result), 3)
        self.verify(SOURCE, result, [cut])
        self.assertIn(point(cut[0]), {p for t in result for p in t})

    def test_all_collinear_boundary_vertices_survive_nonzero_ears(self):
        boundary = [(i, 0, 0) for i in (1, 2, 3)] + [(0, i, 0) for i in (1, 2, 3)]
        result = subdivide_triangle(SOURCE, [], boundary)
        self.assertEqual(len(result), 7)
        self.verify(SOURCE, result)
        self.assertTrue(set(map(point, boundary)) <= {p for t in result for p in t})

    def test_interior_loop_or_orphan_or_dangling_cut_rejected(self):
        loop = [((1, 1, 0), (2, 1, 0)), ((2, 1, 0), (1, 2, 0)), ((1, 2, 0), (1, 1, 0))]
        cases = [loop, [((1, 1, 0), (2, 1, 0))], [((1, 0, 0), (1, 1, 0))],
                 loop + [((1, 0, 0), (1, 1, 0))]]
        for cuts in cases:
            with self.subTest(cuts=cuts), self.assertRaisesRegex(HumanImportError, 'loop|orphan|dangling|nonsimple'):
                subdivide_triangle(SOURCE, cuts)

    def test_outside_nonplanar_degenerate_and_inexact_inputs_rejected(self):
        cases = [(SOURCE, [((5, 0, 0), (0, 2, 0))], []),
                 (SOURCE, [((1, 0, 1), (0, 2, 0))], []),
                 (SOURCE, [((1, 0, 0), (1, 0, 0))], []),
                 (SOURCE, [], [(1, 1, 0)]),
                 (((0, 0, 0), (1, 0, 0), (2, 0, 0)), [], []),
                 (((0., 0, 0), (1, 0, 0), (0, 1, 0)), [], []),
                 (((False, 0, 0), (1, 0, 0), (0, 1, 0)), [], [])]
        for source, cuts, boundary in cases:
            with self.subTest(source=source, cuts=cuts, boundary=boundary), self.assertRaises(HumanImportError):
                subdivide_triangle(source, cuts, boundary)


class PinnedArrangementTests(unittest.TestCase):
    verify = ArrangementTests.verify
    def test_actual_42_intersections_subdivide_every_affected_source_face(self):
        from numilab_human.cardiac_cavity_geometry import extract_cavity_surfaces
        from numilab_human.cardiac_cavity_intersections import triangle_intersection_points
        cavities = {c['source_id']: c['exact_coordinate_quotient'] for c in extract_cavity_surfaces()['chambers']}
        vertices = {name: [tuple(F.from_float(x) for x in p) for p in q['vertices_m']] for name, q in cavities.items()}
        report = json.loads((ROOT / 'Docs/media/cardiac-cavities-20260912/independent/intersections.json').read_bytes())
        pairs = next(row['triangle_pairs'] for row in report['per_pair'] if row['count'])
        self.assertEqual(len(pairs), 42)
        cuts = defaultdict(list)
        for a, b in pairs:
            first = tuple(vertices['right_atrium'][i] for i in cavities['right_atrium']['triangles'][a])
            second = tuple(vertices['right_ventricle'][i] for i in cavities['right_ventricle']['triangles'][b])
            intersections = sorted(set(triangle_intersection_points(first, second)))
            self.assertEqual(len(intersections), 2)
            cuts[('right_atrium', a)].append(tuple(intersections))
            cuts[('right_ventricle', b)].append(tuple(intersections))
        self.assertEqual(len(cuts), 40)
        total = 0
        for (name, index), segments in cuts.items():
            triangle = tuple(vertices[name][i] for i in cavities[name]['triangles'][index])
            result = subdivide_triangle(triangle, segments)
            self.verify(triangle, result, segments)
            total += len(result)
        self.assertEqual(total, 208)


if __name__ == "__main__":
    unittest.main()
