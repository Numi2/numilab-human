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


if __name__ == "__main__":
    unittest.main()
