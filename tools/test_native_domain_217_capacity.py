"""Necessary PLC budget checks; these fixtures do not admit physical tissue."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

spec = importlib.util.spec_from_file_location(
    "native_domain_capacity", Path(__file__).with_name("build_native_domain_217.py"))
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def tetra(offset=(0, 0, 0)):
    return {"points_world_m": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                         dtype=np.float64) + np.array(offset),
            "faces": np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])}


class CapacityTests(unittest.TestCase):
    def test_single_and_disconnected_exact_lower_bounds(self):
        one, two = tetra(), tetra((3, 0, 0))
        joined = {"points_world_m": np.vstack([one["points_world_m"], two["points_world_m"]]),
                  "faces": np.vstack([one["faces"], two["faces"] + 4])}
        report = module.plc_capacity_preflight({"skin": one, "fat": joined}, 3)
        self.assertEqual(report["minimum_combined_tetrahedra"], 3)
        self.assertTrue(report["capacity_possible"])
        self.assertEqual(report["domains"]["fat"]["maximum_volume_components"], 2)
        self.assertFalse(module.plc_capacity_preflight({"skin": one, "fat": joined}, 2)
                         ["capacity_possible"])

    def test_unused_and_float32_coincident_vertices_cannot_inflate_bound(self):
        shell = tetra()
        points = np.vstack([shell["points_world_m"], shell["points_world_m"], [[999, 999, 999]]])
        points[4:, 0] += 1e-50
        same = {"points_world_m": points,
                "faces": np.vstack([shell["faces"], shell["faces"] + 4])}
        before = points.copy()
        report = module.plc_capacity_preflight({"skin": same})["domains"]["skin"]
        self.assertEqual(report["referenced_input_points"], 8)
        self.assertEqual(report["unique_float32_boundary_points"], 4)
        self.assertEqual(report["maximum_volume_components"], 2)
        self.assertEqual(report["minimum_tetrahedra"], 0)
        np.testing.assert_array_equal(before, points)

    def test_invalid_points_indices_and_empty_facets_fail_closed(self):
        for kind in ("nan", "index", "empty"):
            with self.subTest(kind=kind):
                shell = tetra()
                if kind == "nan": shell["points_world_m"][0, 0] = float("nan")
                elif kind == "index": shell["faces"][0, 0] = 4
                else: shell["faces"] = np.empty((0, 3), dtype=np.int64)
                with self.assertRaises(ValueError):
                    module.plc_capacity_preflight({"skin": shell})

    def test_emission_rejects_rounding_collapse_and_distinct_collinear_points(self):
        for points in (np.array([[1., 0, 0], [1. + 1e-8, 0, 0], [1., 1., 0]]),
                       np.array([[0., 0, 0], [1., 1., 0], [2., 2., 0]])):
            with self.subTest(points=points.tolist()):
                before = points.copy()
                report = module.plc_emission_preflight({
                    "skin": {"points_world_m": points, "faces": np.array([[0, 1, 2]])}})
                self.assertFalse(report["emission_possible"])
                self.assertEqual(report["domains"]["skin"]["invalid_face_rows_first_16"], [0])
                np.testing.assert_array_equal(points, before)

    def test_positive_submicron_facets_do_not_trigger_a_construction_threshold(self):
        report = module.plc_emission_preflight({"skin": {
            "points_world_m": np.array([[1., 0, 0], [1. + 2.**-23, 0, 0], [1., 1., 0]]),
            "faces": np.array([[0, 1, 2]])}})
        self.assertTrue(report["emission_possible"])
        self.assertFalse(report["physical_admission"])

    def test_emission_failure_precedes_hole_search_and_tetgen(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            sources = {key: root / (key + ".npz") for key in
                       ("outer", "inner", "deep", "collar", "wall-csv", "parent-cells", "parent-nodes")}
            for path in sources.values(): np.savez(path, placeholder=np.array([0]))
            output = root / "failed-attempt"
            args = ["build_native_domain_217.py", "--run-tetgen", "--output", str(output)]
            for key, path in sources.items(): args += ["--" + key, str(path)]
            invalid = {"points_world_m": np.array([[1., 0, 0], [1. + 1e-8, 0, 0], [1., 1., 0]]),
                       "faces": np.array([[0, 1, 2]])}
            with mock.patch("sys.argv", args), \
                 mock.patch.object(module, "load_surface", return_value={}), \
                 mock.patch.object(module, "read_wall_provenance", return_value={}), \
                 mock.patch.object(module, "assemble_shells", return_value=(invalid, tetra(), {})), \
                 mock.patch.object(module, "plc_capacity_preflight") as capacity, \
                 mock.patch.object(module, "_core_holes") as holes, \
                 mock.patch.object(module, "run_tetgen") as tetgen, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 2)
            capacity.assert_not_called()
            holes.assert_not_called()
            tetgen.assert_not_called()
            receipt = json.loads((output / "adapter-receipt.json").read_text())
            self.assertEqual(receipt["status"], "TETGEN_GEOMETRY_REJECTED")
            self.assertEqual(receipt["emission"]["domains"]["skin"]["invalid_float32_area_faces"], 1)
            self.assertIsNone(receipt["tetgen"])
            self.assertFalse(receipt["physical_admission"])

    def test_capacity_failure_precedes_hole_search_and_tetgen(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            sources = {key: root / (key + ".npz") for key in
                       ("outer", "inner", "deep", "collar", "wall-csv", "parent-cells", "parent-nodes")}
            for path in sources.values(): np.savez(path, placeholder=np.array([0]))
            output = root / "failed-attempt"
            args = ["build_native_domain_217.py", "--run-tetgen", "--output", str(output)]
            for key, path in sources.items(): args += ["--" + key, str(path)]
            original = module.plc_capacity_preflight
            with mock.patch("sys.argv", args), \
                 mock.patch.object(module, "load_surface", return_value={}), \
                 mock.patch.object(module, "read_wall_provenance", return_value={}), \
                 mock.patch.object(module, "assemble_shells", return_value=(tetra(), tetra(), {})), \
                 mock.patch.object(module, "plc_capacity_preflight", side_effect=lambda s: original(s, 1)), \
                 mock.patch.object(module, "_core_holes") as holes, \
                 mock.patch.object(module, "run_tetgen") as tetgen, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(module.main(), 2)
            holes.assert_not_called()
            tetgen.assert_not_called()
            receipt = json.loads((output / "adapter-receipt.json").read_text())
            self.assertEqual(receipt["status"], "TETGEN_CAPACITY_REJECTED")
            self.assertIsNone(receipt["tetgen"])
            self.assertFalse(receipt["physical_admission"])
            self.assertEqual(list(output.iterdir()), [output / "adapter-receipt.json"])


if __name__ == "__main__": unittest.main()
