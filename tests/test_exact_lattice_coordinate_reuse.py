"""Exact arithmetic reuse regressions; no geometry tolerances or physical claims."""
from decimal import Decimal
from fractions import Fraction
import math
from pathlib import Path
import random
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from numilab_human import cardiac_cavity_intersections as audit
from numilab_human.model import ImportError as HumanImportError


def legacy_normalization(points):
    denominator = math.lcm(*(Fraction(value).denominator
                             for point in points for value in point))
    integers = tuple(tuple(Fraction(value).numerator *
                           (denominator // Fraction(value).denominator)
                           for value in point) for point in points)
    origin = integers[0]
    scale = 0
    for point in integers:
        for axis in range(3):
            scale = math.gcd(scale, abs(point[axis] - origin[axis]))
    scale = scale or 1
    return (tuple(tuple((point[axis] - origin[axis]) // scale
                        for axis in range(3)) for point in integers),
            (denominator, origin, scale))


class CoordinateReuseTests(unittest.TestCase):
    def test_mixed_exact_huge_and_float32_lattice_points_match_legacy(self):
        rng = random.Random(20261017)
        for count in (1, 6, 97):
            points = tuple(tuple(Fraction(rng.randrange(-(1 << 190), 1 << 190),
                                           1 << rng.randrange(150))
                                 if (i + j) % 2 else rng.randrange(-(1 << 2900), 1 << 2900)
                                 for j in range(3)) for i in range(count))
            self.assertEqual(audit._integer_lattice_normalization(points),
                             legacy_normalization(points))

    def test_fallback_numeric_representations_keep_legacy_result(self):
        points = ((Decimal("0.125"), "2/7", 0.5), (True, Fraction(3, 5), -0.0))
        self.assertEqual(audit._integer_lattice_normalization(points),
                         legacy_normalization(points))

    def test_repeated_points_retain_origin_scale_and_inverse(self):
        points = ((Fraction(-7, 13), 2, 0),) * 6
        result = audit._integer_lattice_normalization(points)
        self.assertEqual(result, legacy_normalization(points))
        self.assertEqual(result[1][2], 1)
        normalized, (denominator, origin, scale) = result
        self.assertEqual(tuple(tuple(Fraction(origin[k] + scale * point[k], denominator)
                                     for k in range(3)) for point in normalized), points)

    def test_exact_public_coordinate_rejects_float_and_bool(self):
        for value in (True, False, 1.0, float("nan"), "1/2"):
            with self.subTest(value=repr(value)):
                with self.assertRaises(HumanImportError):
                    audit._exact_rational_coordinate(value, "test")

    def test_fraction_and_integer_subclasses_keep_public_semantics(self):
        class Rational(Fraction):
            pass
        class Integer(int):
            pass
        rational = Rational(1, 3)
        self.assertIs(audit._exact_rational_coordinate(rational, "test"), rational)
        self.assertIs(type(audit._exact_rational_coordinate(Integer(7), "test")), int)
        points = ((rational, Integer(7), Fraction(1, 9)), (0, 0, 0))
        self.assertEqual(audit._integer_lattice_normalization(points),
                         legacy_normalization(points))

    def test_triangle_and_prepared_ray_results_match_legacy_normalization(self):
        from unittest.mock import patch
        points = tuple(tuple(Fraction(v, 13) + Fraction(1 << 2890, 7) for v in p)
                       for p in ((0, 0, 0), (4, 0, 0), (0, 4, 0),
                                 (1, 1, -1), (1, 1, 1), (2, 1, 0)))
        actual = audit.triangle_intersection_points(points[:3], points[3:])
        records = audit._records(points[:3], ((0, 1, 2),))
        queries = (points[0], (0, 0, 0))
        prepared = audit.prepare_point_location(records)
        locations = [audit.point_location_prepared_indexed(p, prepared) for p in queries]
        with patch.object(audit, "_integer_lattice_normalization", legacy_normalization):
            self.assertEqual(actual, audit.triangle_intersection_points(points[:3], points[3:]))
            legacy = audit.prepare_point_location(records)
            self.assertEqual(locations, [audit.point_location_prepared_indexed(p, legacy)
                                         for p in queries])


if __name__ == "__main__":
    unittest.main()
