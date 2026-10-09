"""Complete broad-phase candidate equality against a brute-force oracle."""
from fractions import Fraction
import itertools
import random
import unittest

from numilab_human import cardiac_cavity_intersections as audit


def box(lower, upper, face_id):
    return (None, tuple(lower), tuple(upper), face_id, ())


def brute(first, second, same):
    return {(a[3], b[3]) for a in first for b in second
            if (not same or b[3] > a[3])
            and all(a[2][k] >= b[1][k] and b[2][k] >= a[1][k] for k in range(3))}


class ExactBroadphaseTests(unittest.TestCase):
    def check(self, first, second, same=False):
        pairs = [(a[3], b[3]) for a, b in audit._aabb_candidate_pairs(
            first, second, same_surface=same)]
        self.assertEqual(len(pairs), len(set(pairs)), "duplicated box pair")
        self.assertEqual(set(pairs), brute(first, second, same))
        # Reusing the first surface must preserve both closed overlaps and IDs.
        index = audit._prepare_surface_aabb(first)
        reverse = [(a[3], b[3]) for b, a in audit._query_surface_aabb(second, index)
                   if not same or b[3] > a[3]]
        self.assertEqual(len(reverse), len(set(reverse)))
        self.assertEqual(set(reverse), set(pairs))

    def test_exhaustive_closed_boxes_touching_and_sparse_reordered_ids(self):
        # Includes planar, line and point boxes and all face/edge/point contacts.
        intervals = [(a,b) for a in range(-1,2) for b in range(a,2)]
        rows = [box([x[0] for x in axes], [x[1] for x in axes], 11*i-400)
                for i,axes in enumerate(itertools.product(intervals, repeat=3))]
        first = rows[::2] + rows[1::2]
        second = list(reversed(rows))
        self.check(first, second)
        self.check(first, second, True)
        # Component-filtered inputs keep original sparse face IDs.
        self.check(first[5::7], second[3::5], True)

    def test_long_boxes_and_unit_independent_exact_coordinates(self):
        rng = random.Random(42931)
        rows = []
        for i in range(101):
            lower = [rng.randrange(-1000,1000) for _ in range(3)]
            widths = [rng.choice([0,1,3,25,5000]) for _ in range(3)]
            rows.append(box(lower, [a+b for a,b in zip(lower,widths)], 7*i+13))
        for scale,offset in [(1,0),(1<<200,-(1<<203)),(Fraction(1,17),Fraction(-1,3))]:
            transformed = [box([scale*x+offset for x in r[1]],
                               [scale*x+offset for x in r[2]],r[3]) for r in rows]
            rng.shuffle(transformed)
            self.check(transformed[:73], transformed[14:])
            self.check(transformed, list(reversed(transformed)), True)

    def test_empty_and_identical_bounds(self):
        self.check([],[])
        self.check([], [box((0,0,0),(1,1,1),42)])
        self.check([box((0,0,0),(1,1,1),42)],[])
        rows = [box((0,0,0),(0,0,0),1000-3*i) for i in range(35)]
        self.check(rows,rows)
        self.check(rows,rows,True)

    def test_nested_long_box_and_expired_separated_clusters(self):
        # A prefix or event-sweep shortcut must not lose a long spanning box.
        rows = [box((10*i,-i,0),(10*i+1,-i+1,0),i*31) for i in range(90)]
        rows += [box((-10000,-10000,-1),(10000,10000,1),99999)]
        self.check(list(reversed(rows)),rows,True)
        self.check(rows[1::3],rows)


class PreparedAuditTests(unittest.TestCase):
    def test_reused_index_matches_exact_cross_and_self_audits(self):
        rng = random.Random(7312)
        vertices, faces = [], []
        for _ in range(32):
            while True:
                tri = [tuple(rng.randrange(-5, 6) for _ in range(3)) for _ in range(3)]
                if any(audit._cross(audit._sub(tri[1], tri[0]),
                                    audit._sub(tri[2], tri[0]))):
                    break
            faces.append(tuple(range(len(vertices), len(vertices) + 3)))
            vertices.extend(tri)
        # Include identical geometry with different topology, coplanar overlap,
        # and shared edges/vertices. None may become a blanket exemption.
        vertices.extend([(0,0,0), (4,0,0), (0,4,0), (4,4,0), (2,0,0), (0,2,0)])
        faces.extend([(96,97,98), (97,99,98), (96,100,101), (96,97,98)])
        records = audit._records(vertices, faces)
        records = [(*r[:3], 3*r[3]+11, r[4]) for r in records]
        first = list(reversed(records))
        index = audit._prepare_surface_aabb(first)
        for target in [records, records[::3], records[1::5], []]:
            for same in [False, True]:
                self.assertEqual(
                    audit._audit_pair(first, target, same_surface=same),
                    audit._audit_pair_prepared_first(index, target, same_surface=same))

    def test_index_retains_immutable_record_snapshot(self):
        rows = audit._records([(0,0,0),(2,0,0),(0,2,0)], [(0,1,2)])
        index = audit._prepare_surface_aabb(rows)
        target = list(rows)
        expected = audit._audit_pair(rows, target, same_surface=False)
        rows.clear()
        self.assertEqual(audit._audit_pair_prepared_first(index, target), expected)
        self.assertEqual(audit._audit_pair_prepared_first(
            audit._prepare_surface_aabb([]), target)['count'], 0)


if __name__ == '__main__':
    unittest.main()
