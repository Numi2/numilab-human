import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import numpy as np

ADAPTER_PATH = Path(__file__).with_name("register_current_role_bundle_143_217_p1union.py")
SPEC = importlib.util.spec_from_file_location("pose_compat_adapter", ADAPTER_PATH)
ADAPTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADAPTER
SPEC.loader.exec_module(ADAPTER)
OWNER = ADAPTER.load_owner(ADAPTER.SOURCE_005)


def fixture(root: Path):
    bodies = []
    for i in range(157):
        origin = np.asarray([np.float32(i / 17.0), np.float32(-i / 23.0), np.float32(0.1 + i / 31.0)])
        pose = {
            "position_world_m": [float(x) for x in origin],
            "orientation_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "local_to_world_row_major_f32": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
        bodies.append({"body_index": i, "current": dict(pose), "rest": dict(pose)})
    snapshot = root / "poses.json"
    snapshot.write_text(json.dumps({"all_bodies": bodies}), encoding="utf-8")
    frames = root / "frames.csv"
    with frames.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OWNER.FRAME_HEADER, lineterminator="\n")
        writer.writeheader()
        for donor in range(86):
            pose = bodies[donor]
            row = {"binding": str(donor), "donor": str(donor)}
            for phase in ("current", "rest"):
                for axis, value in zip("xyz", pose[phase]["position_world_m"]):
                    row[f"{phase}_{axis}"] = format(value, ".17g")
                for j, value in enumerate(pose[phase]["local_to_world_row_major_f32"]):
                    row[f"{phase}_R{j // 3}{j % 3}"] = format(value, ".17g")
            writer.writerow(row)
    return snapshot, frames


class PoseCompatTests(unittest.TestCase):
    def test_exact_serialized_f32_frames_pass(self):
        with tempfile.TemporaryDirectory() as td:
            snapshot, frames = fixture(Path(td))
            stats = {}
            poses, bindings = ADAPTER.read_poses_from_serialized_f32(OWNER, snapshot, frames, stats)
            self.assertEqual(len(poses), 157)
            self.assertEqual(len(bindings), 86)
            self.assertTrue(stats["frame_rows_exact"])
            self.assertEqual(stats["pose_matrix_source"], "all_bodies.local_to_world_row_major_f32")

    def test_donor_frame_matrix_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            snapshot, frames = fixture(Path(td))
            with frames.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["current_R01"] = "0.0001"
            with frames.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=OWNER.FRAME_HEADER, lineterminator="\n")
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "differs from exact serialized all_bodies current matrix"):
                ADAPTER.read_poses_from_serialized_f32(OWNER, snapshot, frames, {})

    def test_quaternion_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            snapshot, frames = fixture(Path(td))
            data = json.loads(snapshot.read_text())
            data["all_bodies"][0]["current"]["orientation_world_xyzw"] = [0.0, 0.0, 0.70710678, 0.70710678]
            snapshot.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matrix/quaternion mismatch"):
                ADAPTER.read_poses_from_serialized_f32(OWNER, snapshot, frames, {})

    def test_nonrigid_serialized_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            snapshot, frames = fixture(Path(td))
            data = json.loads(snapshot.read_text())
            data["all_bodies"][0]["current"]["local_to_world_row_major_f32"][0] = 1.01
            snapshot.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a proper rotation"):
                ADAPTER.read_poses_from_serialized_f32(OWNER, snapshot, frames, {})

    def test_surface_matrix_f32_roundoff_is_bounded(self):
        poses = [{"current_R": np.eye(3)}]
        surfaces = {"O": [{"donor": 0, "matrix": np.eye(3) + np.diag([5e-7, 0, 0])}]}
        stats = {}
        ADAPTER.normalize_surface_frame_matrices(surfaces, poses, stats)
        self.assertEqual(stats["surface_matrix_rows_checked"], 1)
        self.assertLessEqual(stats["max_surface_matrix_delta"], 1e-6)
        self.assertTrue(np.array_equal(surfaces["O"][0]["matrix"], np.eye(3)))
        bad = {"O": [{"donor": 0, "matrix": np.eye(3) + np.diag([2e-6, 0, 0])}]}
        with self.assertRaisesRegex(ValueError, "exceeds F32 pose consistency tolerance"):
            ADAPTER.normalize_surface_frame_matrices(bad, poses, {})


class P1UnionTests(unittest.TestCase):
    def write_p1(self, path, semantic=None, scalar=(15, 1, 2, 3)):
        if semantic is None:
            semantic = [[15, 1, 2, 3], [15, 4, 5, 6]]
        np.savez(path,
            schema=np.asarray(ADAPTER.P1_ANCHOR_SCHEMA),
            child_tet_row=np.asarray([0], dtype=np.int64),
            p1_support_role_material=np.asarray([15], dtype=np.uint8),
            p1_support_cell_row=np.asarray([0], dtype=np.int32),
            p1_semantic_tuple_offsets=np.asarray([0, len(semantic)], dtype=np.int64),
            p1_semantic_tuples=np.asarray(semantic, dtype=np.uint32),
            reference_anchor_selected_semantic_tuple=np.asarray([scalar], dtype=np.uint32),
            p1_primary_group_tuple=np.asarray([scalar[:3]], dtype=np.int32))

    def test_union_expands_incidence_without_mutating_scalar_child(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "p1.npz"
            self.write_p1(p)
            child = {"tets": [(0, 1, 2, 3)], "labels": [(15, 1, 2, 3)], "materials": [15]}
            expanded, receipt = ADAPTER.expand_child_with_p1_semantic_union(OWNER, child, p, ADAPTER.sha(p))
            self.assertEqual(list(expanded["tets"]), [(0, 1, 2, 3), (0, 1, 2, 3)])
            self.assertEqual(list(expanded["labels"]), [(15, 1, 2, 3), (15, 4, 5, 6)])
            self.assertEqual(child["labels"], [(15, 1, 2, 3)])
            self.assertTrue(receipt["source_scalar_labels_unchanged"])
            self.assertEqual(receipt["p1_semantic_tuple_rows"], 2)

    def test_scalar_label_must_equal_selected_anchor(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "p1.npz"
            self.write_p1(p)
            child = {"tets": [(0, 1, 2, 3)], "labels": [(15, 1, 9, 3)], "materials": [15]}
            with self.assertRaisesRegex(ValueError, "differs from authenticated selected P1 anchor"):
                ADAPTER.expand_child_with_p1_semantic_union(OWNER, child, p, ADAPTER.sha(p))

    def test_cross_material_support_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "p1.npz"
            self.write_p1(p, semantic=[[16, 1, 2, 3]])
            child = {"tets": [(0, 1, 2, 3)], "labels": [(15, 1, 2, 3)], "materials": [15]}
            with self.assertRaisesRegex(ValueError, "crosses a child material boundary"):
                ADAPTER.expand_child_with_p1_semantic_union(OWNER, child, p, ADAPTER.sha(p))

    def test_wrong_csr_pin_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "p1.npz"
            self.write_p1(p)
            child = {"tets": [(0, 1, 2, 3)], "labels": [(15, 1, 2, 3)], "materials": [15]}
            with self.assertRaisesRegex(ValueError, "CSR pin mismatch"):
                ADAPTER.expand_child_with_p1_semantic_union(OWNER, child, p, "0" * 64)


class Source008TransformTests(unittest.TestCase):
    def test_owner_uses_inverse_of_serialized_f32_matrix_for_world_to_local(self):
        path = ADAPTER.SOURCE_008
        text = path.read_text(encoding="utf-8")
        self.assertIn('local=np.linalg.solve(pose["current_R"],point-pose["current_origin"])', text)
        self.assertNotIn('local=pose["current_R"].T@(point-pose["current_origin"])', text)
        matrix = np.asarray([[1.0, 6.0e-7, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        delta = np.asarray([0.1, -0.2, 0.3])
        local = np.linalg.solve(matrix, delta)
        self.assertLess(np.linalg.norm(matrix @ local - delta), 1e-15)
        self.assertGreater(np.linalg.norm(matrix @ (matrix.T @ delta) - delta), 1e-8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
