#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("registered_wall_frame_002", ROOT / "build_native_domain_217.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

class RegisteredWallFrameTests(unittest.TestCase):
    def test_default_source_current_frame_is_unchanged(self):
        source = np.asarray([-0.0715083926916, -0.1469877809, 0.20078103244], dtype=np.float64)
        legacy = mod._wall_reference_point(source, "source-current", None)
        np.testing.assert_array_equal(legacy, source.astype(np.float32))

    def test_registered_frame_applies_se3_once_then_emits_f32(self):
        source = np.asarray([-0.0715083926916, -0.1469877809, 0.20078103244], dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        translation = np.asarray([1.188898923754067e-9, -1.256228454593611e-10, -5.654962409167019e-6])
        frame = {"rotation_row_major": rotation.reshape(-1).tolist(),
                 "translation_m": translation.tolist()}
        legacy = mod._wall_reference_point(source, "source-current", None)
        registered = mod._wall_reference_point(source, "registered-native-current", frame)
        expected = (rotation @ source + translation).astype(np.float32)
        np.testing.assert_array_equal(registered.view(np.uint32), expected.view(np.uint32))
        self.assertFalse(np.array_equal(legacy.view(np.uint32), registered.view(np.uint32)))

    def test_registered_mode_fails_closed_without_frame(self):
        with self.assertRaises(ValueError):
            mod._wall_reference_point(np.asarray([0.0, 0.0, 0.0]), "registered-native-current", None)
        with self.assertRaises(ValueError):
            mod._wall_reference_point(np.asarray([0.0, 0.0, 0.0]), "source-current", {})

    def test_wall_tables_default_to_parent_tables_for_legacy_callers(self):
        cells, nodes = Path("source330-cells.csv.gz"), Path("source330-nodes.csv.gz")
        got = mod.resolve_wall_parent_tables(cells, nodes, None, None, "source-current")
        self.assertEqual(got, (cells, nodes))

    def test_registered_wall_tables_can_be_separate_from_source330_tables(self):
        cells, nodes = Path("capture330-cells.csv.gz"), Path("capture330-nodes.csv.gz")
        wall_cells, wall_nodes = Path("d37-wall-cells.csv.gz"), Path("d37-wall-nodes.csv.gz")
        got = mod.resolve_wall_parent_tables(cells, nodes, wall_cells, wall_nodes,
                                             "registered-native-current")
        self.assertEqual(got, (wall_cells, wall_nodes))

    def test_registered_wall_frame_requires_both_explicit_D37_tables(self):
        cells, nodes = Path("source330-cells.csv.gz"), Path("source330-nodes.csv.gz")
        with self.assertRaises(ValueError):
            mod.resolve_wall_parent_tables(cells, nodes, None, None,
                                           "registered-native-current")
        with self.assertRaises(ValueError):
            mod.resolve_wall_parent_tables(cells, nodes, Path("only-cells.csv.gz"), None,
                                           "registered-native-current")

if __name__ == "__main__":
    unittest.main(verbosity=2)
