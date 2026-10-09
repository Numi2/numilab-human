import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).with_name("audit_delta_1120.py")
SPEC = importlib.util.spec_from_file_location("audit_delta_1120_tested", SCRIPT)
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def make_row(offset=0.0):
    v = np.array([
        [0.0 + offset, 0.0, 0.0, 0.0, 0.0, 1.0],
        [1.0 + offset, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0 + offset, 1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0 + offset, 1.0, 0.0, 0.0, 0.0, 1.0],
    ], dtype="<f4")
    return {
        "body_index": 20,
        "layer": 4,
        "flags": 0,
        "vertices6": v,
        "faces": np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
    }


def make_nha():
    return {sid: make_row(float(sid) / 10000.0) for sid in range(1, 525)}


class CandidateLoader:
    def __init__(self):
        self.args = None

    def load_owner_context(self, *args, **kwargs):
        self.args = (args, kwargs)
        return {"area_binding": {"status": "PASS_exact_geometry_and_config_binding"}}


class FakeBase:
    def adjacency(self, *args):
        return False

    def classify(self, *args):
        return "unclassified_cross_intersection"


class DeltaTests(unittest.TestCase):
    def test_face_flip_and_vertex_motion_are_complete_changed_support(self):
        old = make_nha()
        new = copy.deepcopy(old)
        new[305]["vertices6"][1, 2] += np.float32(0.0001)
        new[305]["faces"][0] = np.array([0, 1, 3])
        delta = D.source_delta(old, new)
        self.assertEqual(delta[305]["vertices"], {1})
        self.assertEqual(delta[305]["faces"], {0, 1})
        self.assertEqual(delta[306]["vertices"], set())
        self.assertEqual(delta[306]["faces"], set())

    def test_row_310_may_change_only_under_separate_exact_copy_proof(self):
        old = make_nha()
        new = copy.deepcopy(old)
        new[310]["vertices6"] = old[305]["vertices6"].copy()
        new[310]["faces"] = old[305]["faces"].copy()
        self.assertNotIn(310, D.source_delta(old, new))
        new[310]["vertices6"][0, 0] += np.float32(1e-6)
        with self.assertRaisesRegex(ValueError, "unique exact lobe-face lineage"):
            D.validate_pleura_source_rows(new, np.array([[305, 0], [305, 1]], dtype="<i4"))

    def test_non_audited_row_topology_edit_refuses_delta(self):
        old = make_nha()
        new = copy.deepcopy(old)
        new[20]["faces"][0] = np.array([0, 1, 3])
        with self.assertRaisesRegex(ValueError, "non-audited row topology"):
            D.source_delta(old, new)

    def test_exact_source_pleura_lineage_is_unique_and_same_winding(self):
        rows = make_nha()
        rows[310]["vertices6"] = rows[305]["vertices6"].copy()
        rows[310]["faces"] = rows[305]["faces"].copy()
        lineage = D.derive_pleura_lineage(rows)
        proof = D.validate_pleura_source_rows(rows, lineage)
        self.assertEqual(lineage.tolist(), [[305, 0], [305, 1]])
        self.assertTrue(proof["source_same_winding_exact"])
        rows[310]["faces"][0] = np.array([0, 2, 1])
        with self.assertRaisesRegex(ValueError, "same-winding exact copied"):
            D.validate_pleura_source_rows(rows, lineage)

    def test_receipt_bound_row310_lineage_accepts_only_current_exact_payload(self):
        rows = make_nha()
        rows[310]["vertices6"] = rows[305]["vertices6"].copy()
        rows[310]["faces"] = rows[305]["faces"].copy()
        lineage = D.derive_pleura_lineage(rows)
        class Base:
            @staticmethod
            def option_many(argv, flag, count):
                return argv[argv.index(flag) + 1:argv.index(flag) + 1 + count]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nha = root / "current.nha"
            nha.write_bytes(b"pinned payload")
            sidecar = root / "row310-lineage.npy"
            np.save(sidecar, lineage, allow_pickle=False)
            receipt = root / "receipt.json"
            payload_sha = D.sha(nha)
            receipt_value = {
                "payload": {"path": str(nha), "sha256": payload_sha},
                "functional_bindings": {"anatomy_payload_sha256": payload_sha},
                "provenance": {
                    "derived_visceral_pleura_proxy": {"output_payload_sha256": payload_sha},
                    "lung_final_shared_interface_rebuild": {
                        "output_payload_sha256": payload_sha,
                        "row310_face_lineage_path": str(sidecar),
                        "row310_face_lineage_sha256": D.sha(sidecar),
                        "row310_face_lineage_face_count": len(lineage),
                    },
                },
            }
            receipt.write_text(json.dumps(receipt_value))
            receipt_sha = D.sha(receipt)
            ctx = {
                "invocation": {"argv": ["native", "--resting-anatomy-receipt", str(receipt)],
                               "asset_sha256": {str(receipt.resolve()): receipt_sha}},
                "nha_path": nha.resolve(), "nha_sha": payload_sha, "nha_rows": rows,
            }
            proof = D.pleura_source_binding(ctx, Base())
            self.assertEqual(proof["lineage"]["sha256"], D.sha(sidecar))
            receipt_value["payload"]["sha256"] = "0" * 64
            receipt.write_text(json.dumps(receipt_value))
            ctx["invocation"]["asset_sha256"][str(receipt.resolve())] = D.sha(receipt)
            with self.assertRaisesRegex(ValueError, "receipt does not bind current NHA"):
                D.pleura_source_binding(ctx, Base())

    def test_native_pleura_copy_check_rejects_any_missing_or_mismatched_face(self):
        class FakeNHA:
            rows = {i: i for i in range(524)}
            def __init__(self, *args): pass
            def close(self): pass

        class FakeReview:
            NHA = FakeNHA

        class FakePack:
            pack_sha = "pack"
            def __init__(self, *args): pass
            def close(self): pass

        class Checker:
            exact = True
            Pack = FakePack
            @staticmethod
            def load_review_module(): return FakeReview
            @staticmethod
            def source_nha_topology_check(nha, pack): return list(range(524))
            @classmethod
            def check_pleura(cls, parent, final, lineage, pack):
                n = len(lineage) if cls.exact else len(lineage) - 1
                return {"faces": len(lineage), "exact_xyz_same_winding_faces": n,
                        "same_winding_mismatches": len(lineage) - n,
                        "maximum_abs_coordinate_delta_m": 0.0 if cls.exact else 1e-8}

        ctx = {"run_path": Path("/run"), "nha_path": Path("/payload")}
        lineage = np.array([[305, 0], [306, 0]], dtype="<i4")
        self.assertTrue(D.pleura_native_copy_check(ctx, 10000, lineage, Checker)["native_all_524_row_mapping"])
        Checker.exact = False
        with self.assertRaisesRegex(ValueError, "not exact same-winding"):
            D.pleura_native_copy_check(ctx, 10000, lineage, Checker)

    def test_target_identity_semantics_must_remain_fixed(self):
        old = make_nha()
        new = copy.deepcopy(old)
        new[311]["layer"] = 9
        with self.assertRaisesRegex(ValueError, "row identity/semantics"):
            D.source_delta(old, new)

    def test_target_regenerated_normals_are_tracked_separately_from_position_support(self):
        old = make_nha()
        new = copy.deepcopy(old)
        new[308]["vertices6"][2, 3] += np.float32(0.01)
        delta = D.source_delta(old, new)
        self.assertEqual(delta[308]["normals"], {2})
        self.assertEqual(delta[308]["vertices"], set())
        self.assertEqual(delta[308]["faces"], set())

    def test_non_audited_normals_must_remain_fixed(self):
        old = make_nha()
        new = copy.deepcopy(old)
        new[20]["vertices6"][2, 3] += np.float32(0.01)
        with self.assertRaisesRegex(ValueError, "non-audited row geometry/attributes"):
            D.source_delta(old, new)

    def test_row_face_count_change_refuses_delta(self):
        old = make_nha()
        new = copy.deepcopy(old)
        new[305]["faces"] = np.vstack([new[305]["faces"], [0, 2, 3]])
        with self.assertRaisesRegex(ValueError, "row count/topology layout"):
            D.source_delta(old, new)

    def test_captured_changed_star_includes_source_and_pose_support(self):
        old = {}
        new = {}
        sd = {}
        for sid in D.ROWS:
            row = make_row()
            old[sid] = {"v": row["vertices6"][:, :3].copy(), "f": row["faces"].copy()}
            new[sid] = {"v": row["vertices6"][:, :3].copy(), "f": row["faces"].copy()}
            sd[sid] = {"vertices": set(), "faces": set()}
        new[305]["v"][0, 2] += np.float32(0.0001)
        sd[305]["vertices"] = {0}
        new[305]["f"][1] = np.array([1, 2, 3])
        changed = D.pose_delta(old, new, sd)
        self.assertEqual(changed[305], {0, 1})

    def test_captured_motion_outside_source_edit_support_refuses_delta(self):
        old = {}
        new = {}
        sd = {}
        for sid in D.ROWS:
            row = make_row()
            old[sid] = {"v": row["vertices6"][:, :3].copy(), "f": row["faces"].copy()}
            new[sid] = {"v": row["vertices6"][:, :3].copy(), "f": row["faces"].copy()}
            sd[sid] = {"vertices": set(), "faces": set()}
        new[305]["v"][0, 0] += np.float32(0.0001)
        with self.assertRaisesRegex(ValueError, "outside source edit support"):
            D.pose_delta(old, new, sd)

    def test_affected_pair_union_covers_both_sides_without_duplicates(self):
        class Product:
            @staticmethod
            def _aabb_candidate_pairs(a, b, same_surface=False):
                if same_surface:
                    return [(x, y) for x in a for y in b if x[3] < y[3]]
                return [(x, y) for x in a for y in b]

        def record(fi):
            t = ((fi, 0, 0), (fi, 1, 0), (fi, 0, 1))
            return (t, (fi, 0, 0), (fi, 1, 1), fi, (fi * 3, fi * 3 + 1, fi * 3 + 2))

        a = [record(i) for i in range(3)]
        b = [record(i) for i in range(4)]
        self_pairs = D.affected_pairs(Product(), a, a, {1}, set(), True)
        cross_pairs = D.affected_pairs(Product(), a, b, {1}, {2}, False)
        self.assertEqual(set(self_pairs), {(0, 1), (1, 2)})
        self.assertEqual(set(cross_pairs), {(0, 2), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2)})

    def test_candidate_loader_never_inherits_geometry_only_area_mismatch(self):
        old_adapter, old_sha = D.SUCCESSOR_DMAP_ADAPTER, D.SUCCESSOR_DMAP_ADAPTER_SHA
        with tempfile.TemporaryDirectory() as td:
            adapter = Path(td) / "successor_adapter.py"
            adapter.write_text("# pinned test adapter\n")
            D.SUCCESSOR_DMAP_ADAPTER = adapter
            D.SUCCESSOR_DMAP_ADAPTER_SHA = D.sha(adapter)
            base = CandidateLoader()
            base.V8_DMAP_ADAPTER = Path("/old/v8_adapter.py")
            base.V8_DMAP_ADAPTER_SHA = "old"
            result = D.load_candidate_context(base, "/run", "/candidate.nha", "a" * 64, "/dmap.json", "/lineage.json")
            self.assertEqual(result["area_binding"]["status"], "PASS_exact_geometry_and_config_binding")
            self.assertIs(base.args[1]["geometry_only_area_mismatch"], False)
            self.assertEqual(base.args[1]["d_map_composition_report"], Path("/dmap.json").resolve())
            self.assertEqual(base.V8_DMAP_ADAPTER, Path("/old/v8_adapter.py"))
            self.assertEqual(base.V8_DMAP_ADAPTER_SHA, "old")
        D.SUCCESSOR_DMAP_ADAPTER, D.SUCCESSOR_DMAP_ADAPTER_SHA = old_adapter, old_sha

    def test_successor_dmap_bridge_requires_exact_registered_v8_triangles(self):
        module_path = D.SUCCESSOR_DMAP_ADAPTER
        spec = importlib.util.spec_from_file_location("successor_dmap_adapter_test", module_path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        rows = {sid: make_row(float(sid) / 10000.0) for sid in (305, 306, 307, 308, 309, 311)}
        maps = {sid: {"d2l": ({0: 0} if sid != 309 else {}), "l2d": ({0: 0} if sid != 309 else {})}
                for sid in (305, 306, 307, 308, 309)}
        parent_doc = {"pairs": maps}
        self.assertTrue(adapter.validate_registered_triangles(parent_doc, rows, copy.deepcopy(rows))["exact_xyz_triangle_match"])
        regenerated_normals = copy.deepcopy(rows)
        regenerated_normals[308]["vertices6"][0, 3] += np.float32(0.01)
        self.assertTrue(adapter.validate_registered_triangles(parent_doc, rows, regenerated_normals)["exact_xyz_triangle_match"])
        changed_mapped_xyz = copy.deepcopy(rows)
        changed_mapped_xyz[308]["vertices6"][0, 0] += np.float32(1e-5)
        with self.assertRaisesRegex(ValueError, "registered triangle differs"):
            adapter.validate_registered_triangles(parent_doc, rows, changed_mapped_xyz)
        changed_mapped_winding = copy.deepcopy(rows)
        changed_mapped_winding[308]["faces"][0] = [0, 2, 1]
        with self.assertRaisesRegex(ValueError, "registered face index triple differs"):
            adapter.validate_registered_triangles(parent_doc, rows, changed_mapped_winding)

    def test_successor_dmap_bridge_rejects_index_identity_change_even_if_xyz_duplicate(self):
        module_path = D.SUCCESSOR_DMAP_ADAPTER
        spec = importlib.util.spec_from_file_location("successor_dmap_adapter_index_test", module_path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        rows = {sid: make_row(float(sid) / 10000.0) for sid in (305, 306, 307, 308, 309, 311)}
        rows[308]["vertices6"][3] = rows[308]["vertices6"][0].copy()
        maps = {sid: {"d2l": ({0: 0} if sid != 309 else {}), "l2d": ({0: 0} if sid != 309 else {})}
                for sid in (305, 306, 307, 308, 309)}
        parent_doc = {"pairs": maps}
        child = copy.deepcopy(rows)
        child[308]["faces"][0] = [3, 1, 2]
        self.assertEqual(
            adapter._tri_bits(rows[308], 0),
            adapter._tri_bits(child[308], 0),
            "fixture must preserve geometric triangle bits through duplicate vertex identity",
        )
        with self.assertRaisesRegex(ValueError, "index triple differs"):
            adapter.validate_registered_triangles(parent_doc, rows, child)

    def test_untouched_rejections_are_reclassified_and_retained(self):
        ev = {
            "type": "cross", "owners": [305, 306], "faces": [7, 9],
            "face_vertex_ids": [[0, 1, 2], [3, 4, 5]],
            "points_exact_lattice_rational": [[[0, 1], [0, 1], [0, 1]]],
            "class": "unclassified_cross_intersection",
        }
        rec = lambda ids: (((0, 0, 0), (0, 1, 0), (1, 0, 0)),
                           (0, 0, 0), (1, 1, 0), ids[0], tuple(ids))
        current = {305: {7: rec([0, 1, 2])}, 306: {9: rec([3, 4, 5])}}
        unchanged = {sid: set() for sid in D.ROWS}
        retained = D.reclass_event(ev, unchanged, {}, current, {}, {}, FakeBase())
        self.assertEqual(retained["class"], "unclassified_cross_intersection")
        self.assertIsNone(D.reclass_event(ev, {**unchanged, 305: {7}}, {}, current, {}, {}, FakeBase()))

    def test_conflicting_shared_input_hash_refuses_merge(self):
        self.assertEqual(D.merge_input_hashes({"x": "1"}, {"x": "1", "y": "2"}),
                         {"x": "1", "y": "2"})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            D.merge_input_hashes({"x": "1"}, {"x": "2"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
