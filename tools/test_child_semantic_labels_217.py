#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("export_child_semantic_labels_217.py")
spec = importlib.util.spec_from_file_location("child_semantic_labels_217", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

PARENT_BASE = Path(os.environ.get(
    "NUMI_HUMAN_PARENT_MASS_CSV",
    "/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/"
    "physical-transfer-209/native-parent-label-capture-016/compiled-parent-cell-mass.csv"
))


def write_child_tables(root: Path, parent, rows: list[int], material_override=None):
    used_nodes = sorted({int(n) for row in rows for n in parent["tets"][row]})
    node_map = {old: new for new, old in enumerate(used_nodes)}
    points = parent["points"][used_nodes].astype(np.float32)
    node_ids = [100_003 + 7 * i for i in range(len(used_nodes))]
    node_path, ele_path = root / "child.node", root / "child.ele"
    with node_path.open("w", encoding="utf-8") as stream:
        stream.write(f"{len(points)} 3 0 0\n")
        for point, node_id in zip(points, node_ids):
            stream.write(f"{node_id} " + " ".join(format(float(v), ".9g") for v in point) + "\n")
    with ele_path.open("w", encoding="utf-8") as stream:
        stream.write(f"{len(rows)} 4 1\n")
        for child_row, source_row in enumerate(rows):
            tet = [node_map[int(n)] for n in parent["tets"][source_row]]
            if material_override is None:
                material = int(parent["materials"][source_row])
            elif isinstance(material_override, dict):
                material = int(material_override.get(child_row, parent["materials"][source_row]))
            else:
                material = int(material_override)
            tet_points = points[np.asarray(tet)]
            if module.tet_volume(tet_points.astype(np.float64)) < 0.0:
                tet[0], tet[1] = tet[1], tet[0]
            ids = [node_ids[i] for i in tet]
            stream.write(f"{200_011 + child_row} " + " ".join(map(str, ids)) + f" {material}\n")
    return node_path, ele_path


def write_arbitrary_child_tet(root: Path, points: np.ndarray, material: int):
    points = np.asarray(points, dtype=np.float32)
    tet = [0, 1, 2, 3]
    if module.tet_volume(points.astype(np.float64)) < 0.0:
        tet[0], tet[1] = tet[1], tet[0]
    node_path, ele_path = root / "displaced.node", root / "displaced.ele"
    with node_path.open("w", encoding="utf-8") as stream:
        stream.write("4 3 0 0\n")
        for i, point in enumerate(points):
            stream.write(f"{51_001 + i} " + " ".join(format(float(v), ".9g") for v in point) + "\n")
    with ele_path.open("w", encoding="utf-8") as stream:
        stream.write("1 4 1\n")
        stream.write(f"81_001 " + " ".join(str(51_001 + i) for i in tet) + f" {material}\n")
    return node_path, ele_path


def write_parent_subset(root: Path, actual_base: Path, parent, source_rows: list[int],
                        label_overrides: dict[int, dict[str, str]] | None = None,
                        omit_labels: bool = False) -> Path:
    label_overrides = label_overrides or {}
    with Path(str(actual_base) + ".nodes.csv").open(newline="") as stream:
        node_rows = list(csv.DictReader(stream))
    with actual_base.open(newline="") as stream:
        mass_rows = list(csv.DictReader(stream))
    with Path(str(actual_base) + ".tets.csv").open(newline="") as stream:
        source_tet_rows = list(csv.DictReader(stream))
    with Path(str(actual_base) + ".labels.csv").open(newline="") as stream:
        label_rows = list(csv.DictReader(stream))
    used_nodes = sorted({int(n) for row in source_rows for n in parent["tets"][row]})
    node_remap = {old: new for new, old in enumerate(used_nodes)}
    base = root / "parent.csv"
    with base.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["source_tet_row", "material_index", "mass_kg"])
        for new_row, source_row in enumerate(source_rows):
            row = mass_rows[source_row]
            writer.writerow([new_row, row["material_index"], row["mass_kg"]])
    with Path(str(base) + ".nodes.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(node_rows[0]), lineterminator="\n")
        writer.writeheader()
        for new_node, old_node in enumerate(used_nodes):
            row = dict(node_rows[old_node])
            row["parent_node_index"] = str(new_node)
            writer.writerow(row)
    with Path(str(base) + ".source.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["object_local_node", "material_index", "donor_body", "mass_kg",
                         "local_x_m", "local_y_m", "local_z_m"])
    with Path(str(base) + ".tets.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["parent_tet_row", "material_index", "donor_body", "parent_node0",
                         "parent_node1", "parent_node2", "parent_node3",
                         "parent_reference_volume_m3", "parent_mass_kg"])
        for new_row, source_row in enumerate(source_rows):
            row = source_tet_rows[source_row]
            tet = [node_remap[int(parent["tets"][source_row][c])] for c in range(4)]
            writer.writerow([new_row, row["material_index"], row["donor_body"], *tet,
                             row["parent_reference_volume_m3"], row["parent_mass_kg"]])
    with Path(str(base) + ".labels.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        header = ["parent_tet_row", "material_index", "geometric_donor_body",
                  "anatomical_region_id", "stable_source_face_id"]
        writer.writerow(header)
        if not omit_labels:
            for new_row, source_row in enumerate(source_rows):
                row = dict(label_rows[source_row])
                row["parent_tet_row"] = str(new_row)
                row.update(label_overrides.get(new_row, {}))
                writer.writerow([row[key] for key in header])
    return base


def subset_parent(parent, rows: list[int]):
    used_nodes = sorted({int(n) for row in rows for n in parent["tets"][row]})
    node_remap = {old: new for new, old in enumerate(used_nodes)}
    return {
        "points": parent["points"][used_nodes].copy(),
        "tets": np.asarray([[node_remap[int(n)] for n in parent["tets"][row]]
                            for row in rows], dtype=np.int64),
        "materials": parent["materials"][rows].copy(),
        "donors": parent["donors"][rows].copy(),
        "regions": parent["regions"][rows].copy(),
        "faces": parent["faces"][rows].copy(),
    }


def same_label_neighbor_pair(parent):
    face_rows = {}
    for row, tet in enumerate(parent["tets"]):
        for omitted in range(4):
            face = tuple(sorted(int(n) for i, n in enumerate(tet) if i != omitted))
            face_rows.setdefault(face, []).append(row)
    for face, rows in face_rows.items():
        if len(rows) == 2 and module.semantic(parent, rows[0]) == module.semantic(parent, rows[1]):
            return tuple(rows), face
    raise AssertionError("actual native parent has no shared face with identical complete labels")


def child_with_centroid(center: np.ndarray, material: int = 15) -> dict:
    center = np.asarray(center, dtype=np.float64)
    eps = 1.0e-5
    offsets = np.asarray([[eps, 0, 0], [0, eps, 0], [0, 0, eps],
                          [-eps, -eps, -eps]], dtype=np.float64)
    points = center + offsets  # exact-boundary unit fixture; production node input is Float32
    tet = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
    if module.tet_volume(points.astype(np.float64)) < 0.0:
        tet[0, 0], tet[0, 1] = tet[0, 1], tet[0, 0]
    return {"points": points, "tets": tet,
            "materials": np.asarray([material], dtype=np.uint32)}


class ChildSemanticLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PARENT_BASE.is_file():
            raise unittest.SkipTest(f"retained actual parent packet is absent: {PARENT_BASE}")
        cls.parent = module.read_parent(PARENT_BASE)

    def test_actual_parent_identity_roundtrip(self):
        with tempfile.TemporaryDirectory(prefix="child-labels-217-parent-") as tmp:
            root = Path(tmp)
            node_path, ele_path = write_child_tables(root, self.parent,
                                                     list(range(len(self.parent["tets"]))))
            output = root / "child.labels.csv"
            source_map = root / "child.labels.source.csv"
            receipt_path = root / "child.labels.receipt.json"
            result = module.export(
                PARENT_BASE, node_path, ele_path, output, source_map, receipt_path,
                reference_registration_max_distance_m=0.01,
                reference_registration_ambiguity_tolerance_m=1.0e-10)
            self.assertEqual(result["counts"]["child_tetrahedra"], len(self.parent["tets"]))
            self.assertEqual(result["counts"]["exact_geometry_matches"], len(self.parent["tets"]))
            self.assertEqual(result["counts"]["centroid_containment_matches"], 0)
            self.assertEqual(result["counts"]["bounded_reference_registration_matches"], 0)
            self.assertTrue(result["bounded_reference_registration"]["enabled"])
            with output.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), len(self.parent["tets"]))
            for i, row in enumerate(rows):
                self.assertEqual(int(row["child_tet_row"]), i)
                self.assertEqual(int(row["material_index"]), int(self.parent["materials"][i]))
                self.assertEqual(int(row["geometric_donor_body"]), int(self.parent["donors"][i]))
                self.assertEqual(int(row["anatomical_region_id"]), int(self.parent["regions"][i]))
                self.assertEqual(int(row["stable_source_face_id"]), int(self.parent["faces"][i]))
            with source_map.open(newline="") as stream:
                provenance = list(csv.DictReader(stream))
            child = module.read_tetgen(node_path, ele_path)
            self.assertEqual([int(row["selected_parent_tet_row"]) for row in provenance],
                             list(range(len(self.parent["tets"]))))
            self.assertTrue(all(row["label_match_kind"] == "exact_f32_parent_tet_geometry"
                                for row in provenance))
            self.assertTrue(all(int(row["semantic_tie_count"]) == 0 for row in provenance))
            for child_row, row in enumerate(provenance):
                beta = np.asarray([float(row[f"centroid_barycentric{i}"]) for i in range(4)])
                source_tet = self.parent["tets"][int(row["selected_parent_tet_row"])]
                registered = np.asarray([
                    float(row[f"centroid_registered_reference_{axis}_m"])
                    for axis in ("x", "y", "z")])
                residual = np.asarray([
                    float(row[f"centroid_reference_residual_{axis}_m"])
                    for axis in ("x", "y", "z")])
                centroid = child["points"][child["tets"][child_row]].astype(np.float64).mean(axis=0)
                np.testing.assert_allclose(beta @ self.parent["points"][source_tet] + residual,
                                           centroid, rtol=0.0, atol=1.0e-12)

    def test_bounded_registration_reconstructs_displaced_actual_parent_point(self):
        parent = subset_parent(self.parent, [0])
        tet = parent["tets"][0]
        points = parent["points"][tet]
        face_rows = np.asarray([1, 2, 3], dtype=np.int64)
        face = points[face_rows]
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        if float(np.dot(normal, points[0] - face[0])) > 0.0:
            normal = -normal
        normal /= float(np.linalg.norm(normal))
        displaced = face.mean(axis=0) + 1.0e-3 * normal
        child = child_with_centroid(displaced, int(parent["materials"][0]))
        index = module.build_index(parent)
        with self.assertRaisesRegex(ValueError, "outside all retained parent cells"):
            module.locate(parent, index, child, 0)
        found = module.locate(parent, index, child, 0,
                              reference_registration_max_distance_m=2.0e-3,
                              reference_registration_ambiguity_tolerance_m=1.0e-10)
        self.assertEqual(found["parent_row"], 0)
        self.assertEqual(found["kind"], "bounded_closest_parent_tet_reference_registration")
        registered = found["bary"] @ parent["points"][parent["tets"][0]]
        centroid = child["points"][child["tets"][0]].mean(axis=0)
        residual = centroid - registered
        self.assertAlmostEqual(float(np.linalg.norm(residual)), 1.0e-3, delta=1.0e-10)
        np.testing.assert_allclose(registered + residual, centroid, rtol=0.0, atol=1.0e-12)
        self.assertAlmostEqual(float(found["registration_distance_m"]),
                               float(np.linalg.norm(residual)), delta=1.0e-12)
        with self.assertRaisesRegex(ValueError, "exceeds declared bound"):
            module.locate(parent, index, child, 0,
                          reference_registration_max_distance_m=5.0e-4,
                          reference_registration_ambiguity_tolerance_m=1.0e-10)
        with tempfile.TemporaryDirectory(prefix="child-labels-217-reference-registration-") as tmp:
            root = Path(tmp)
            parent_base = write_parent_subset(root, PARENT_BASE, self.parent, [0])
            node_path, ele_path = write_arbitrary_child_tet(
                root, child["points"], int(parent["materials"][0]))
            labels_path = root / "registered.labels.csv"
            source_path = root / "registered.source.csv"
            receipt_path = root / "registered.receipt.json"
            receipt = module.export(
                parent_base, node_path, ele_path, labels_path, source_path, receipt_path,
                reference_registration_max_distance_m=2.0e-3,
                reference_registration_ambiguity_tolerance_m=1.0e-10)
            self.assertEqual(receipt["counts"]["bounded_reference_registration_matches"], 1)
            self.assertFalse(receipt["labels_are_direct_geometric_ancestry"])
            self.assertIn("not geometric ancestry", receipt["bounded_reference_registration"]["semantics"])
            with labels_path.open(newline="") as stream:
                label = next(csv.DictReader(stream))
            self.assertEqual(int(label["material_index"]), int(parent["materials"][0]))
            self.assertEqual(int(label["geometric_donor_body"]), int(parent["donors"][0]))
            with source_path.open(newline="") as stream:
                source = next(csv.DictReader(stream))
            self.assertEqual(source["label_match_kind"],
                             "bounded_closest_parent_tet_reference_registration")
            child_from_disk = module.read_tetgen(node_path, ele_path)
            beta = np.asarray([float(source[f"centroid_barycentric{i}"]) for i in range(4)])
            registered = np.asarray([
                float(source[f"centroid_registered_reference_{axis}_m"])
                for axis in ("x", "y", "z")])
            residual = np.asarray([
                float(source[f"centroid_reference_residual_{axis}_m"])
                for axis in ("x", "y", "z")])
            centroid = child_from_disk["points"][child_from_disk["tets"][0]].astype(np.float64).mean(axis=0)
            local_parent = module.read_parent(parent_base)
            np.testing.assert_allclose(
                beta @ local_parent["points"][local_parent["tets"][0]] + residual,
                centroid, rtol=0.0, atol=1.0e-12)
            self.assertAlmostEqual(float(source["centroid_reference_residual_norm_m"]),
                                   float(np.linalg.norm(residual)), delta=1.0e-14)

    def test_bounded_registration_rejects_conflicting_closest_semantics(self):
        points = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        parent = {
            "points": points,
            "tets": np.asarray([[0, 2, 1, 3], [0, 1, 2, 4]], dtype=np.int64),
            "materials": np.asarray([15, 15], dtype=np.uint32),
            "donors": np.asarray([43, 62], dtype=np.uint32),
            "regions": np.asarray([43, 62], dtype=np.uint32),
            "faces": np.asarray([100, 200], dtype=np.uint32),
        }
        for tet in parent["tets"]:
            self.assertGreater(module.tet_volume(points[tet]), 0.0)
        child = child_with_centroid(np.asarray([0.8, 0.8, 0.0]), material=15)
        with self.assertRaisesRegex(ValueError, "ambiguous closest-parent semantic labels"):
            module.locate(parent, module.build_index(parent), child, 0,
                          reference_registration_max_distance_m=0.5,
                          reference_registration_ambiguity_tolerance_m=1.0e-10)

    def test_bounded_registration_same_semantic_tie_is_deterministic(self):
        points = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0], [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        parent = {
            "points": points,
            "tets": np.asarray([[0, 2, 1, 3], [0, 1, 2, 4]], dtype=np.int64),
            "materials": np.asarray([15, 15], dtype=np.uint32),
            "donors": np.asarray([43, 43], dtype=np.uint32),
            "regions": np.asarray([43, 43], dtype=np.uint32),
            "faces": np.asarray([100, 100], dtype=np.uint32),
        }
        child = child_with_centroid(np.asarray([0.8, 0.8, 0.0]), material=15)
        found = module.locate(parent, module.build_index(parent), child, 0,
                              reference_registration_max_distance_m=0.5,
                              reference_registration_ambiguity_tolerance_m=1.0e-10)
        self.assertEqual(found["parent_row"], 0)
        self.assertEqual(found["candidate_count"], 2)
        self.assertEqual(found["tie_count"], 1)

    def test_actual_parent_boundary_tie_is_deterministic(self):
        rows, face = same_label_neighbor_pair(self.parent)
        parent = subset_parent(self.parent, list(rows))
        point = self.parent["points"][list(face)].mean(axis=0)
        result = module.locate(parent, module.build_index(parent),
                               child_with_centroid(point), 0)
        self.assertEqual(result["parent_row"], 0)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["tie_count"], 1)
        self.assertEqual(result["kind"], "parent_centroid_containment")

    def test_actual_parent_boundary_tie_with_conflicting_provenance_fails(self):
        rows, face = same_label_neighbor_pair(self.parent)
        parent = subset_parent(self.parent, list(rows))
        parent["regions"][1] = np.uint32((int(parent["regions"][1]) + 1) % 100000)
        point = self.parent["points"][list(face)].mean(axis=0)
        with self.assertRaisesRegex(ValueError, "conflicting parent semantic boundary labels"):
            module.locate(parent, module.build_index(parent),
                          child_with_centroid(point), 0)

    def test_child_material_conflict_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="child-labels-217-material-") as tmp:
            root = Path(tmp)
            row = 0
            other = 16 if int(self.parent["materials"][row]) == 15 else 15
            node_path, ele_path = write_child_tables(root, self.parent, [row], other)
            with self.assertRaisesRegex(ValueError, "conflicting material"):
                module.export(PARENT_BASE, node_path, ele_path, root / "out.csv")

    def test_missing_parent_stable_face_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="child-labels-217-missing-face-") as tmp:
            base = write_parent_subset(Path(tmp), PARENT_BASE, self.parent, [0],
                {0: {"stable_source_face_id": str(module.INVALID)}})
            with self.assertRaisesRegex(ValueError, "missing native parent region or stable source face"):
                module.read_parent(base)

    def test_conflicting_parent_material_provenance_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="child-labels-217-bad-material-") as tmp:
            row = 0
            other = 16 if int(self.parent["materials"][row]) == 15 else 15
            base = write_parent_subset(Path(tmp), PARENT_BASE, self.parent, [row],
                {0: {"material_index": str(other)}})
            with self.assertRaisesRegex(ValueError, "conflicting native parent material provenance"):
                module.read_parent(base)

    def test_missing_parent_label_row_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="child-labels-217-missing-label-") as tmp:
            base = write_parent_subset(Path(tmp), PARENT_BASE, self.parent, [0], omit_labels=True)
            with self.assertRaisesRegex(ValueError, "row counts differ"):
                module.read_parent(base)


if __name__ == "__main__":
    unittest.main(verbosity=2)
