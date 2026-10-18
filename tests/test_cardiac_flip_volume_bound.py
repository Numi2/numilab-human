"""Synthetic checks for the cardiac flip tetra-volume polynomial and bound."""

import itertools
import unittest

import numpy as np

from tools.audit_cardiac_flip_volume import (
    evaluate_volume_polynomial,
    relative_volume_bounds,
    surface_volume_delta_polynomial,
    tetra_volume_delta_polynomial,
)
from tools.prepare_registered_vertex_collapse import compact_faces_after_collapse
from tools.rebind_common_field_after_vertex_retirement import (
    remap_vertex_indices_after_retirement,
    retire_affine_map_record,
    validate_vertex_retirement_header,
)


def constant_rows(point):
    rows = np.zeros((8, 3), dtype=np.float64)
    rows[0] = point
    return rows


class CardiacFlipVolumeBoundTests(unittest.TestCase):
    def test_constant_tetrahedron_has_only_constant_volume(self):
        anchor = constant_rows((0.0, 0.0, 0.0))
        point_a = constant_rows((1.0, 0.0, 0.0))
        point_b = constant_rows((0.0, 1.0, 0.0))
        point_c = constant_rows((0.0, 0.0, 1.0))
        coefficients, products = tetra_volume_delta_polynomial(
            anchor, point_a, point_b, point_c
        )
        zero_power = (0,) * 7
        self.assertEqual(set(coefficients), {zero_power})
        self.assertAlmostEqual(coefficients[zero_power], 1.0 / 6.0, places=15)
        bound, derivative_bound, rounding = relative_volume_bounds(
            coefficients, products, 2.0
        )
        self.assertAlmostEqual(bound, (1.0 / 6.0 + rounding) / 2.0, places=15)
        self.assertAlmostEqual(derivative_bound, 3.0 * bound, places=15)

    def test_affine_coordinate_matches_direct_tetra_volume(self):
        anchor = constant_rows((0.0, 0.0, 0.0))
        point_a = constant_rows((1.0, 0.0, 0.0))
        point_b = constant_rows((0.0, 1.0, 0.0))
        point_c = constant_rows((0.0, 0.0, 1.0))
        point_c[1, 2] = 0.25
        coefficients, products = tetra_volume_delta_polynomial(
            anchor, point_a, point_b, point_c
        )
        expected = {(0,) * 7: 1.0 / 6.0, (1,) + (0,) * 6: 0.25 / 6.0}
        self.assertEqual(set(coefficients), set(expected))
        for exponent, value in expected.items():
            self.assertAlmostEqual(coefficients[exponent], value, places=15)
        for x in (-1.0, 0.0, 1.0):
            coordinates = np.zeros(7)
            coordinates[0] = x
            direct = (1.0 + 0.25 * x) / 6.0
            self.assertAlmostEqual(
                evaluate_volume_polynomial(coefficients, coordinates), direct, places=15
            )
        bound, _, _ = relative_volume_bounds(coefficients, products, 1.0)
        self.assertGreaterEqual(bound, max(abs(1.0 / 6.0 - 0.25 / 6.0),
                                           abs(1.0 / 6.0 + 0.25 / 6.0)))


    def test_surface_flip_polynomial_matches_signed_surface_volume_delta(self):
        old_faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        new_faces = np.asarray(((0, 1, 3), (1, 2, 3)), dtype=np.int64)
        rng = np.random.default_rng(23)
        vertices = [rng.normal(size=(8, 3)) * 0.1 for _ in range(4)]
        coefficients, _, removed, added = surface_volume_delta_polynomial(
            old_faces, new_faces, lambda index: vertices[index]
        )
        self.assertEqual(sum(item["multiplicity"] for item in removed), 2)
        self.assertEqual(sum(item["multiplicity"] for item in added), 2)

        for coordinates in [np.zeros(7), *[rng.uniform(-1, 1, size=7) for _ in range(20)]]:
            positions = [value[0] + coordinates @ value[1:] for value in vertices]
            def signed_surface_volume(faces):
                total = 0.0
                for a, b, c in faces:
                    matrix = np.stack((positions[a], positions[b], positions[c]), axis=1)
                    total += np.linalg.det(matrix) / 6.0
                return total
            direct = signed_surface_volume(old_faces) - signed_surface_volume(new_faces)
            self.assertAlmostEqual(
                evaluate_volume_polynomial(coefficients, coordinates), direct, places=14
            )


    def test_surface_volume_proof_rejects_boundary_change(self):
        old_faces = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
        new_faces = np.asarray(((0, 1, 3),), dtype=np.int64)
        affine = lambda _: np.zeros((8, 3), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "oriented surface boundaries differ"):
            surface_volume_delta_polynomial(old_faces, new_faces, affine)

    def test_retirement_header_allows_only_one_vertex_count_decrement(self):
        before = (b"NHANAT1\0", 5, 24, 100, 300, b"registration", b"source")
        after = (b"NHANAT1\0", 5, 24, 99, 300, b"registration", b"source")
        validate_vertex_retirement_header(before, after)
        with self.assertRaisesRegex(ValueError, "payload header differs"):
            validate_vertex_retirement_header(
                before, (b"NHANAT1\0", 5, 24, 98, 300, b"registration", b"source")
            )
        with self.assertRaisesRegex(ValueError, "payload header differs"):
            validate_vertex_retirement_header(
                before, (b"NHANAT1\0", 5, 24, 99, 301, b"registration", b"source")
            )

    def test_retired_vertex_and_map_record_remap_stays_byte_exact(self):
        source_faces = np.asarray(
            ((0, 1, 2), (1, 0, 3), (0, 2, 4), (1, 4, 3)), dtype=np.int64
        )
        compact_faces, removed_faces = compact_faces_after_collapse(
            source_faces, keep_vertex=1, drop_vertex=0
        )
        expected_faces = np.asarray(((0, 1, 3), (0, 3, 2)), dtype=np.int64)
        np.testing.assert_array_equal(compact_faces, expected_faces)
        np.testing.assert_array_equal(removed_faces, (0, 1))

        source_map = np.arange(5 * 7 * 4, dtype="<f4").reshape(5, 7, 4)
        compact_map = retire_affine_map_record(source_map, record_index=2)
        np.testing.assert_array_equal(compact_map, source_map[[0, 1, 3, 4]])
        old_indices = np.asarray((0, 1, 3, 4), dtype=np.int64)
        new_indices = remap_vertex_indices_after_retirement(old_indices, retired_vertex=2)
        np.testing.assert_array_equal(new_indices, (0, 1, 2, 3))
        for old_index, new_index in zip(old_indices, new_indices):
            np.testing.assert_array_equal(source_map[old_index], compact_map[new_index])
        with self.assertRaisesRegex(ValueError, "retired or invalid vertex"):
            remap_vertex_indices_after_retirement(np.asarray((2,)), retired_vertex=2)

    def test_cubic_polynomial_matches_direct_determinants_on_coordinate_cube(self):
        anchor = np.zeros((8, 3), dtype=np.float64)
        point_a = np.zeros((8, 3), dtype=np.float64)
        point_b = np.zeros((8, 3), dtype=np.float64)
        point_c = np.zeros((8, 3), dtype=np.float64)
        anchor[0] = (0.2, -0.1, 0.3)
        point_a[0] = (1.1, 0.2, -0.4)
        point_b[0] = (-0.3, 1.2, 0.1)
        point_c[0] = (0.1, -0.2, 0.9)
        anchor[1:] = np.arange(21, dtype=float).reshape(7, 3) * 0.003
        point_a[1:] = np.arange(21, 42, dtype=float).reshape(7, 3) * -0.002
        point_b[1:] = np.arange(42, 63, dtype=float).reshape(7, 3) * 0.001
        point_c[1:] = np.arange(63, 84, dtype=float).reshape(7, 3) * 0.0015
        coefficients, products = tetra_volume_delta_polynomial(
            anchor, point_a, point_b, point_c
        )

        self.assertTrue(any(sum(exponent) == 3 for exponent in coefficients))
        rng = np.random.default_rng(321)
        samples = [rng.uniform(-1.0, 1.0, size=7) for _ in range(32)]
        samples.extend(np.asarray(corner, dtype=float)
                       for corner in itertools.product((-1.0, 1.0), repeat=7))
        reference_volume = 0.75
        bound, _, _ = relative_volume_bounds(coefficients, products, reference_volume)
        for coordinates in samples:
            evaluated = evaluate_volume_polynomial(coefficients, coordinates)
            vertices = [
                value[0] + coordinates @ value[1:]
                for value in (anchor, point_a, point_b, point_c)
            ]
            direct = np.linalg.det(np.stack(
                (vertices[1] - vertices[0], vertices[2] - vertices[0],
                 vertices[3] - vertices[0]), axis=1
            )) / 6.0
            self.assertAlmostEqual(evaluated, direct, places=14)
            self.assertLessEqual(abs(evaluated) / reference_volume, bound + 1e-12)


if __name__ == "__main__":
    unittest.main()
