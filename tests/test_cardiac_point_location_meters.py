"""Regress metre/lattice confusion in native-surface containment audits."""
from fractions import Fraction as F
import unittest

from numilab_human import cardiac_cavity_intersections as ci
from numilab_human.model import ImportError as HumanImportError


class Float32PointLocationMetersTests(unittest.TestCase):
    points = ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))

    def test_exact_meter_queries_and_lattice_mismatch_regression(self):
        meter = ci.prepare_float32_point_location_meters(self.points, self.faces)
        controls = (
            ((F(1, 8), F(1, 8), F(1, 8)), "inside"),
            ((F(2), F(2), F(2)), "outside"),
            ((F(1, 7), F(1, 7), F(0)), "boundary"),
        )
        for query, expected in controls:
            with self.subTest(query=query):
                self.assertEqual(
                    ci.point_location_prepared_indexed(query, meter)["location"],
                    expected,
                )
                self.assertEqual(
                    ci.point_location_prepared(query, meter)["location"], expected
                )
        lattice = ci.prepare_point_location(ci._records(
            [ci.float32_point_lattice_key(p) for p in self.points], self.faces
        ))
        outside = controls[1][0]
        # The historical adapter misclassified this outside metre point.
        self.assertEqual(
            ci.point_location_prepared_indexed(outside, lattice)["location"],
            "inside",
        )
        scaled = tuple(q * (1 << 149) for q in outside)
        self.assertEqual(
            ci.point_location_prepared_indexed(scaled, lattice)["location"],
            "outside",
        )

    def test_translated_small_surface_remains_in_meters(self):
        points = tuple(tuple(shift + coordinate / 16 for shift, coordinate in
                             zip((-0.25, 0.5, -0.75), p)) for p in self.points)
        prepared = ci.prepare_float32_point_location_meters(points, self.faces)
        query = tuple(sum(F.from_float(p[k]) for p in points) / 4
                      for k in range(3))
        self.assertEqual(
            ci.point_location_prepared_indexed(query, prepared)["location"],
            "inside",
        )
        self.assertEqual(
            ci.point_location_prepared_indexed(tuple(map(F, (3, 3, 3))),
                                              prepared)["location"],
            "outside",
        )

    def test_rejects_unpacked_geometry_and_invalid_local_faces(self):
        with self.assertRaises(HumanImportError):
            ci.prepare_float32_point_location_meters(
                ((0.1, 0., 0.),) + self.points[1:], self.faces)
        for face in ((-1, 1, 2), (0, 1, 4), (True, 1, 2), (0, 1)):
            with self.subTest(face=face), self.assertRaises(HumanImportError):
                ci.prepare_float32_point_location_meters(self.points, (face,))
        with self.assertRaises(HumanImportError):
            ci.prepare_float32_point_location_meters((), self.faces)
        with self.assertRaises(HumanImportError):
            ci.prepare_float32_point_location_meters(self.points, ())



class ExactAabbScalingTests(unittest.TestCase):
    def test_extreme_dyadic_metres_preserve_bounds_and_stable_ties(self):
        tiny = F(1, 1 << 149)
        huge = F(1 << 128)
        lower = ((-huge, -tiny, F(0)), (F(0), -tiny, F(0)),
                 (F(0), -tiny, F(0)), (huge, -tiny, F(0)))
        upper = tuple((x, tiny, z) for x, _, z in lower)
        root, nodes = ci._prepare_exact_aabb_index(lower, upper, leaf_size=1)
        self.assertEqual(nodes, 7)
        self.assertEqual(root[:2], ((-huge, -tiny, F(0)),
                                   (huge, tiny, F(0))))
        def leaves(node):
            lo, hi, indices, left, right = node
            if indices is not None:
                self.assertEqual(len(indices), 1)
                index = indices[0]
                self.assertEqual((lo, hi), (lower[index], upper[index]))
                return list(indices)
            return leaves(left) + leaves(right)
        self.assertEqual(leaves(root), [0, 1, 2, 3])

    def test_nondyadic_and_float_bounds_retain_exact_values(self):
        for offset in (F(1, 3), 0.25):
            lower = ((offset, offset, offset),
                     (offset + 2, offset, offset))
            upper = tuple(tuple(value + 1 for value in row) for row in lower)
            root, nodes = ci._prepare_exact_aabb_index(lower, upper, leaf_size=1)
            self.assertEqual(nodes, 3)
            self.assertEqual(root[:2], (lower[0], upper[1]))
            self.assertEqual(root[3][:3], (lower[0], upper[0], (0,)))
            self.assertEqual(root[4][:3], (lower[1], upper[1], (1,)))
            self.assertIs(type(root[0][0]), type(offset))

    def test_identical_dyadic_centres_remain_one_sorted_leaf(self):
        lower = tuple((F(-i, 32),) * 3 for i in (1, 2, 3, 4))
        upper = tuple(tuple(-value for value in row) for row in lower)
        root, nodes = ci._prepare_exact_aabb_index(lower, upper, leaf_size=1)
        self.assertEqual(nodes, 1)
        self.assertEqual(root, ((F(-1, 8),) * 3, (F(1, 8),) * 3,
                                (0, 1, 2, 3), None, None))

if __name__ == "__main__":
    unittest.main()
