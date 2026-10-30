#!/usr/bin/env python3
from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
MODULES = {
    "build": ROOT / "build_native_domain_217.py",
    "register": ROOT / "register_combined_volume_nodes_217.py",
    "labels": ROOT / "export_child_semantic_labels_217.py",
}

def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"capacity_test_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

class CapacityProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mods = {name: load(name, path) for name, path in MODULES.items()}

    def test_explicit_profiles_preserve_defaults_and_parent_caps(self):
        for name, mod in self.mods.items():
            with self.subTest(owner=name):
                self.assertEqual(mod.MAX_NODES, 300_000)
                self.assertEqual(mod.MAX_CELLS, 200_000)
                profile = mod.configure_capacity_profile("full-reference-500k-nodes-2m-cells")
                self.assertEqual((mod.MAX_NODES, mod.MAX_CELLS), (500_000, 2_000_000))
                self.assertEqual(profile["max_nodes_combined"], 500_000)
                self.assertEqual(profile["max_cells_combined"], 2_000_000)
                if name == "labels":
                    self.assertEqual((mod.MAX_PARENT_NODES, mod.MAX_PARENT_CELLS), (300_000, 200_000))
                mod.configure_capacity_profile("legacy")
                self.assertEqual((mod.MAX_NODES, mod.MAX_CELLS), (300_000, 200_000))
                with self.assertRaises(ValueError):
                    mod.configure_capacity_profile("unrecognized")


    @staticmethod
    def tetra(offset):
        import numpy as np
        points = np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64) + np.asarray(offset)
        faces = np.asarray([[0,2,1],[0,1,3],[0,3,2],[1,2,3]], dtype=np.int64)
        return points, faces

    def test_region_seed_for_each_positive_disconnected_component(self):
        import numpy as np
        mod = self.mods["build"]
        p0, f0 = self.tetra((0,0,0))
        p1, f1 = self.tetra((3,0,0))
        points = np.vstack((p0,p1))
        faces = np.vstack((f0,f1+4))
        shell = {"points_world_m": points, "faces": faces,
                 "face_marker": np.ones(len(faces), dtype=np.int32)}
        parent = {15:[
            {"parent_tet_row":7, "point":np.asarray([0.25,0.25,0.25])},
            {"parent_tet_row":9, "point":np.asarray([3.25,0.25,0.25])},
        ]}
        seeds, receipt = mod._region_points(shell,15,parent)
        self.assertEqual(len(seeds),2)
        self.assertEqual([r["source330_parent_tet_row"] for r in receipt],[7,9])
        self.assertTrue(all(r["whole_material_solid_contains_seed"] for r in receipt))

    def test_region_seed_skips_negative_hollow_component(self):
        import numpy as np
        mod = self.mods["build"]
        outer, outer_faces = self.tetra((0,0,0))
        inner, inner_faces = self.tetra((0,0,0))
        inner = (inner - np.asarray([0.25,0.25,0.25])) * 0.2 + np.asarray([0.25,0.25,0.25])
        points = np.vstack((outer,inner))
        faces = np.vstack((outer_faces,inner_faces[:,[0,2,1]]+4))
        shell = {"points_world_m": points, "faces": faces,
                 "face_marker": np.ones(len(faces), dtype=np.int32)}
        parent = {16:[
            {"parent_tet_row":3, "point":np.asarray([0.25,0.25,0.25])},
            {"parent_tet_row":4, "point":np.asarray([0.6,0.1,0.1])},
        ]}
        seeds, receipt = mod._region_points(shell,16,parent)
        self.assertEqual(len(seeds),1)
        self.assertEqual(receipt[0]["source330_parent_tet_row"],4)
        self.assertTrue(receipt[0]["whole_material_solid_contains_seed"])
        self.assertFalse(mod._point_inside(mod._manifold(points,faces), parent[16][0]["point"]))

    def test_child_node_header_limit_is_explicit_for_each_reader(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            node = root / "large.node"
            ele = root / "empty.ele"
            node.write_text("300001 3 0 0\n1 0 0 0\n2 1 0 0\n3 0 1 0\n4 0 0 1\n")
            ele.write_text("1 4 1\n1 1 2 3 4 15\n")
            for name, mod in self.mods.items():
                with self.subTest(owner=name):
                    mod.configure_capacity_profile("legacy")
                    with self.assertRaises(ValueError):
                        if name == "build":
                            mod.read_tetgen_mesh(node, ele, 15)
                        elif name == "register":
                            mod.read_child(node, ele, root / "labels.csv")
                        else:
                            mod.read_tetgen(node, ele)
                    mod.configure_capacity_profile("full-reference-500k-nodes-2m-cells")
                    try:
                        if name == "build":
                            mod.read_tetgen_mesh(node, ele, 15)
                        elif name == "register":
                            mod.read_child(node, ele, root / "labels.csv")
                        else:
                            mod.read_tetgen(node, ele)
                    except ValueError as exc:
                        self.assertRegex(str(exc), "truncated|row count differs")
                    else:
                        self.fail(f"{name} accepted a deliberately truncated 300001-row node file")
                    finally:
                        mod.configure_capacity_profile("legacy")

if __name__ == "__main__":
    unittest.main(verbosity=2)
