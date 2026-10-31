from __future__ import annotations

import csv
import hashlib
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import incremental_p1_reuse_205 as inc

SOURCE_HEADER = (
    "child_tet_row", "selected_parent_tet_row", "label_match_kind",
    "matching_parent_tet_count", "semantic_tie_count",
    "centroid_barycentric0", "centroid_barycentric1", "centroid_barycentric2", "centroid_barycentric3",
    "centroid_registered_reference_x_m", "centroid_registered_reference_y_m", "centroid_registered_reference_z_m",
    "centroid_reference_residual_x_m", "centroid_reference_residual_y_m", "centroid_reference_residual_z_m",
    "centroid_reference_residual_norm_m", "reference_registration_distance_m",
    "reference_registration_max_distance_m", "reference_registration_ambiguity_tolerance_m",
    "material_index", "geometric_donor_body", "anatomical_region_id", "stable_source_face_id",
)
PROJECTION_HEADER = (
    "child_tet_row", "carrier_child_node_row", "carrier_query_x_m", "carrier_query_y_m", "carrier_query_z_m",
    "selected_parent_tet_row", "selected_parent_barycentric0", "selected_parent_barycentric1",
    "selected_parent_barycentric2", "selected_parent_barycentric3", "registered_world_x_m",
    "registered_world_y_m", "registered_world_z_m", "residual_x_m", "residual_y_m", "residual_z_m",
    "anchor_distance_m", "material_index", "geometric_donor_body", "anatomical_region_id",
    "stable_source_face_id", "primary_group_id", "carrier_node_group_id", "primary_score_numerator",
    "primary_score_denominator", "exact_resolution_used", "exact_min_squared_distance_ratio",
    "tie_parent_rows_json", "tie_semantic_tuples_json",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 15), b""):
            h.update(b)
    return h.hexdigest()


def archive(rows: list[int], carriers: list[int] | None = None) -> dict[str, np.ndarray]:
    if carriers is None:
        carriers = [0] * len(rows)
    n = len(rows)
    result: dict[str, np.ndarray] = {"schema": np.asarray(inc.P1_SCHEMA),
                                     "child_tet_row": np.asarray(rows, np.int64)}
    # All rows have one source parent, one primary parent, one carrier parent,
    # one complete semantic tuple, one exact-tie parent and one near candidate.
    base_parent = np.asarray([500 + int(row) for row in rows], np.int32)
    tuples = np.asarray([[15, 20, 30, 100]], np.uint32).repeat(n, axis=0)
    result.update({
        "p1_support_role_material": np.full(n, 15, np.uint8),
        "p1_support_cell_row": np.asarray(rows, np.int32),
        "p1_primary_group_id": np.zeros(n, np.int32),
        "reference_anchor_carrier_node_group_id": np.zeros(n, np.int32),
        "p1_primary_group_score_numerator": np.ones(n, np.int64),
        "p1_primary_group_score_denominator": np.full(n, 4, np.int64),
        "p1_primary_group_tuple": np.tile(np.asarray([[15, 20, 30]], np.int32), (n, 1)),
        "reference_anchor_near_candidate_projection_separation_m": np.zeros(n, np.float64),
        "reference_anchor_exact_pool_resolution_used": np.zeros(n, np.uint8),
        "reference_anchor_exact_min_squared_distance_ratio": np.asarray([""] * n, dtype="<U78"),
        "reference_anchor_selected_parent_tet_row": base_parent.copy(),
        "reference_anchor_selected_parent_barycentric": np.tile(np.asarray([[1., 0., 0., 0.]]), (n, 1)),
        "reference_anchor_selected_semantic_tuple": tuples.copy(),
        "centroid_diagnostic_parent_tet_row": base_parent.copy(),
        "centroid_diagnostic_barycentric": np.tile(np.asarray([[1., 0., 0., 0.]]), (n, 1)),
        "centroid_diagnostic_registered_world_m": np.zeros((n, 3), np.float64),
        "centroid_diagnostic_residual_norm_m": np.zeros(n, np.float64),
        "reference_anchor_carrier_child_node_row": np.asarray(carriers, np.int64),
        "reference_anchor_carrier_query_world_m": np.zeros((n, 3), np.float64),
        "reference_anchor_centroid_to_projected_anchor_distance_m": np.zeros(n, np.float64),
        "reference_anchor_registered_world_m": np.zeros((n, 3), np.float64),
        "reference_anchor_distance_m": np.zeros(n, np.float64),
        "reference_anchor_tie_max_projection_separation_m": np.zeros(n, np.float64),
        "reference_anchor_tie_max_full86_abs_difference": np.zeros(n, np.float64),
    })
    def csr(offset_name: str, data_names: tuple[str, ...], vals: list[np.ndarray]) -> None:
        offset = [0]
        for v in vals:
            offset.append(offset[-1] + len(v))
        result[offset_name] = np.asarray(offset, np.int64)
        for j, key in enumerate(data_names):
            result[key] = np.concatenate([v if v.ndim > 1 else v.reshape(-1) for v in vals], axis=0) if vals else np.empty(0)
    parent_values = [np.asarray([p], np.int32) for p in base_parent]
    csr("p1_parent_tet_offsets", ("p1_parent_tet_rows",), parent_values)
    csr("p1_primary_parent_tet_offsets", ("p1_primary_parent_tet_rows",), parent_values)
    csr("reference_anchor_carrier_parent_tet_offsets", ("reference_anchor_carrier_parent_tet_rows",), parent_values)
    csr("p1_semantic_tuple_offsets", ("p1_semantic_tuples",), [tuples[i:i+1] for i in range(n)])
    csr("reference_anchor_tie_offsets", (
        "reference_anchor_tie_parent_tet_rows", "reference_anchor_tie_carrier_child_node_rows",
        "reference_anchor_tie_barycentric", "reference_anchor_tie_distance_m"),
        [np.asarray([base_parent[i]], np.int32) for i in range(n)])
    # Replace shared-offset aligned payloads after constructing offsets.
    result["reference_anchor_tie_carrier_child_node_rows"] = np.asarray(carriers, np.int64)
    result["reference_anchor_tie_barycentric"] = np.tile(np.asarray([[1., 0., 0., 0.]]), (n, 1))
    result["reference_anchor_tie_distance_m"] = np.zeros(n, np.float64)
    csr("reference_anchor_tie_semantic_tuple_offsets", ("reference_anchor_tie_semantic_tuples",),
        [tuples[i:i+1] for i in range(n)])
    csr("reference_anchor_near_candidate_offsets", (
        "reference_anchor_near_candidate_parent_tet_rows", "reference_anchor_near_candidate_child_node_rows",
        "reference_anchor_near_candidate_approx_distance_m", "reference_anchor_near_candidate_exact_distance_gap_m"),
        [np.asarray([base_parent[i]], np.int32) for i in range(n)])
    result["reference_anchor_near_candidate_child_node_rows"] = np.asarray(carriers, np.int64)
    result["reference_anchor_near_candidate_approx_distance_m"] = np.zeros(n, np.float64)
    result["reference_anchor_near_candidate_exact_distance_gap_m"] = np.zeros(n, np.float64)
    return result


def write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def label_row(row: int, parent: int = 500) -> list[str]:
    return [str(row), "15", "20", "30", "100"]


def source_row(row: int) -> list[str]:
    vals = {name: "0" for name in SOURCE_HEADER}
    vals.update({"child_tet_row": str(row), "selected_parent_tet_row": str(500 + row),
                 "label_match_kind": "p1_nodal_supported_current_reference_anchor",
                 "matching_parent_tet_count": "1", "semantic_tie_count": "0",
                 "material_index": "15", "geometric_donor_body": "20",
                 "anatomical_region_id": "30", "stable_source_face_id": "100"})
    return [vals[k] for k in SOURCE_HEADER]


def projection_row(row: int, carrier: int = 0) -> list[str]:
    vals = {name: "0" for name in PROJECTION_HEADER}
    vals.update({"child_tet_row": str(row), "carrier_child_node_row": str(carrier),
                 "selected_parent_tet_row": str(500 + row), "material_index": "15",
                 "geometric_donor_body": "20", "anatomical_region_id": "30",
                 "stable_source_face_id": "100", "primary_group_id": "0",
                 "carrier_node_group_id": "0", "primary_score_numerator": "1",
                 "primary_score_denominator": "4", "tie_parent_rows_json": f"[{500 + row}]",
                 "tie_semantic_tuples_json": "[[15,20,30,100]]"})
    return [vals[k] for k in PROJECTION_HEADER]


def write_combined_node_map_gz(path: Path, rows: list[list[str]]) -> None:
    header=("joint_node_id","source_partition","source_tetgen_node_id","registered_I_source_vertex_row",
            "merge_peer_partition","merge_peer_tetgen_node_id","registered_surface_layer","registered_surface_vertex_row")
    with gzip.open(path,"wt",encoding="utf-8",newline="") as f:
        w=csv.writer(f,lineterminator="\n"); w.writerow(header); w.writerows(rows)


def write_role_support(path: Path, material: int, tuples: list[tuple[int,int,int]]) -> None:
    n=len(tuples)
    np.savez_compressed(path,
        node_group_offsets=np.arange(n+1,dtype=np.int64),
        node_group_ids=np.arange(n,dtype=np.int32),
        group_tuples=np.asarray(tuples,dtype=np.int32).reshape((-1,3)),
        node_group_parent_tet_offsets=np.arange(n+1,dtype=np.int64),
        node_group_parent_tet_rows=np.arange(n,dtype=np.int32))


def write_registration_rows(path: Path, rows: list[list[str]]) -> None:
    header=("joint_node_id","source_partition","source_tetgen_node_id","role",
            "current_x_m","current_y_m","current_z_m","incident_material_mask",
            "geometric_donor_body","anatomical_region_id")
    write_csv(path,header,rows)


class IncrementalP1ReuseTests(unittest.TestCase):
    def test_exact_ordered_corner_map_reuses_only_identity(self):
        old_tets = np.asarray([[0, 1, 2, 3], [0, 1, 2, 4]], np.int64)
        old_materials = np.asarray([15, 16])
        # The odd swap and the even cyclic corner permutation both change the
        # ordered identity. We deliberately regenerate both rows instead of
        # reusing cell-indexed barycentric/ancestry fields without a full
        # corner permutation transform.
        new_tets = np.asarray([[0, 1, 2, 3], [1, 0, 2, 3], [1, 2, 0, 3], [0, 1, 2, 4]], np.int64)
        new_materials = np.asarray([15, 15, 15, 16])
        mapping, reverse = inc.build_ordered_tet_row_map(old_tets, old_materials,
                                                          new_tets, new_materials,
                                                          np.arange(5))
        self.assertEqual(mapping.tolist(), [0, -1, -1, 1])
        self.assertEqual(reverse.tolist(), [0, 3])

    def test_csr_merge_reorders_old_rows_and_inserts_changed_row(self):
        base = archive([0, 1, 2], [0, 1, 2])
        fresh = archive([1], [3])
        merged = inc.merge_anchor_archives(base, fresh, np.asarray([2, -1, 0]), {1: 0})
        self.assertEqual(merged["child_tet_row"].tolist(), [0, 1, 2])
        self.assertEqual(merged["p1_support_cell_row"].tolist(), [2, 1, 0])
        self.assertEqual(merged["reference_anchor_carrier_child_node_row"].tolist(), [2, 3, 0])
        self.assertEqual(merged["p1_parent_tet_rows"].tolist(), [502, 501, 500])
        self.assertEqual(merged["p1_parent_tet_offsets"].tolist(), [0, 1, 2, 3])

    def test_full_synthetic_merge_and_177_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            base = root / "base"
            fresh = root / "fresh"
            out = root / "out"
            base.mkdir(); fresh.mkdir()
            points = np.asarray([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,1,1]], np.float32)
            combined_map = base / "combined-node-map.csv.gz"
            write_combined_node_map_gz(combined_map, [
                [str(i+1), "skin", str(i+1), "-1", "", "", "O", str(i)]
                for i in range(len(points))])
            skin_support = base / "skin-support.npz"
            fat_support = base / "fat-support.npz"
            write_role_support(skin_support, 15, [(15,20,30)] * len(points))
            write_role_support(fat_support, 16, [(16,5,6)])
            node = base / "base.node"
            node.write_text("5 3 0 0\n" + "".join(f"{i+1} {x} {y} {z}\n" for i,(x,y,z) in enumerate(points)), encoding="utf-8")
            base_tets = [[0,1,2,3], [0,1,2,4], [1,2,3,4]]
            cand_tets = [[1,2,3,4], [0,1,3,4], [0,1,2,3]]
            def write_ele(path: Path, tets: list[list[int]]) -> None:
                path.write_text(f"{len(tets)} 4 1\n" + "".join(
                    f"{i+1} {a+1} {b+1} {c+1} {d+1} 15\n" for i,(a,b,c,d) in enumerate(tets)), encoding="utf-8")
            ele = base / "base.ele"; candidate_ele = root / "candidate.ele"
            write_ele(ele, base_tets); write_ele(candidate_ele, cand_tets)
            candidate_node = root / "candidate.node"; candidate_node.write_bytes(node.read_bytes())
            node_map = root / "new-to-base.npy"; np.save(node_map, np.arange(5, dtype=np.int64))
            labels = base / "child.labels.csv"; source = base / "child.labels.csv.source.csv"
            projection = base / "p1-anchor-projection.csv"; p1 = base / "p1-anchor.npz"
            write_csv(labels, inc.LABEL_HEADER, [label_row(i) for i in range(3)])
            write_csv(source, SOURCE_HEADER, [source_row(i) for i in range(3)])
            write_csv(projection, PROJECTION_HEADER, [projection_row(i, i) for i in range(3)])
            np.savez_compressed(p1, **archive([0,1,2], [0,1,2]))
            fresh_labels = fresh / "child.labels.csv"; fresh_source = fresh / "child.labels.csv.source.csv"
            fresh_projection = fresh / "p1-anchor-projection.csv"; fresh_p1 = fresh / "p1-anchor.npz"
            write_csv(fresh_labels, inc.LABEL_HEADER, [label_row(1)])
            write_csv(fresh_source, SOURCE_HEADER, [source_row(1)])
            write_csv(fresh_projection, PROJECTION_HEADER, [projection_row(1, 0)])
            fresh_arr = archive([1], [0])
            fresh_arr["reference_anchor_selected_parent_tet_row"][0] = 501
            fresh_arr["centroid_diagnostic_parent_tet_row"][0] = 501
            fresh_arr["p1_parent_tet_rows"][0] = 501
            fresh_arr["p1_primary_parent_tet_rows"][0] = 501
            fresh_arr["reference_anchor_carrier_parent_tet_rows"][0] = 501
            fresh_arr["reference_anchor_tie_parent_tet_rows"][0] = 501
            fresh_arr["reference_anchor_near_candidate_parent_tet_rows"][0] = 501
            np.savez_compressed(fresh_p1, **fresh_arr)
            common_hash = "a" * 64
            base_receipt = {
                "schema": inc.BASE_RECEIPT_SCHEMA,
                "status": "P1_SUPPORTED_REFERENCE_ANCHOR_LABELS_EXPORTED_NOT_PHYSICAL_ADMISSION",
                "full_child_domain_export": True, "physical_admission": False,
                "native_steps": 0, "mass_transport": "NOT_RUN",
                "counts": {"child_tetrahedra": 3, "exported_child_rows": 3},
                "inputs": {"child_node": {"sha256": sha(node)}, "child_elements": {"sha256": sha(ele)},
                           **{key: {"sha256": common_hash} for key in (
                               "parent_mass", "parent_nodes", "parent_source", "parent_tets", "parent_labels",
                               "parent_full86_weights", "current_230_owner", "batched_projection_helper",
                               "skin_p1_support", "skin_p1_support_receipt", "fat_p1_support", "fat_p1_support_receipt")}},
                "outputs": {
                    "labels": {"sha256": sha(labels)}, "parent_row_map": {"sha256": sha(source)},
                    "p1_anchor_projection_csv": {"sha256": sha(projection)}, "p1_anchor_csr": {"sha256": sha(p1)},
                },
            }
            for key, path in (("combined_node_map", combined_map), ("skin_p1_support", skin_support),
                              ("fat_p1_support", fat_support)):
                base_receipt["inputs"][key] = {"path": str(path), "sha256": sha(path)}
            base_receipt_path = base / "receipt.json"
            base_receipt_path.write_text(json.dumps(base_receipt), encoding="utf-8")
            reg_dir = base / "177"; reg_dir.mkdir()
            reg_map = reg_dir / "registered-node-map.csv"
            reg_rows = [[str(i+1), "skin", str(i+1), "O",
                         *(format(float(v), ".9g") for v in points[i]), 32768, 20, 30]
                        for i in range(len(points))]
            write_registration_rows(reg_map, reg_rows)
            reg_receipt = {
                "schema": inc.REGISTRATION_SCHEMA,
                "status": "REGISTERED_NODE_MAP_PREPARED_SOURCE_ONLY_NOT_PHYSICAL_ADMISSION",
                "output": {"sha256": sha(reg_map)},
                "authenticated_p1_semantic_incidence_union": {
                    "schema": inc.INCIDENCE_SCHEMA, "status": "PASS_AUTHENTICATED_COMPLETE_P1_UNION",
                    "child_cells": 3, "selected_anchor_rows_present_in_union": 3,
                    "selected_scalar_anchor_rows_exact": 3, "cross_material_union_rows": 0,
                    "empty_cell_unions": 0, "source_scalar_labels_unchanged": True,
                    "geometry_changed": False, "physical_admission": False,
                    "mass_mapping": "NOT_RUN", "runtime": 0, "p1_csr_sha256": sha(p1),
                },
                "inputs": {"child_labels": {"sha256": sha(labels)}, "tetgen_node": {"sha256": sha(node)},
                           "tetgen_ele": {"sha256": sha(ele)}},
            }
            reg_receipt_path = reg_dir / "receipt.json"
            reg_receipt_path.write_text(json.dumps(reg_receipt), encoding="utf-8")
            fresh_receipt = {
                "schema": inc.FRESH_RECEIPT_SCHEMA, "status": "PASS_SOURCE_ONLY_CHANGED_ROW_SUBSET",
                "full_child_domain_export": False, "physical_admission": False,
                "native_steps": 0, "mass_transport": "NOT_RUN",
                "counts": {"exported_child_rows": 1},
                "inputs": {"candidate_child_node": {"sha256": sha(candidate_node)},
                           "candidate_child_elements": {"sha256": sha(candidate_ele)},
                           **{key: {"sha256": common_hash} for key in (
                               "parent_mass", "parent_nodes", "parent_source", "parent_tets", "parent_labels",
                               "parent_full86_weights", "current_230_owner", "batched_projection_helper",
                               "skin_p1_support", "skin_p1_support_receipt", "fat_p1_support", "fat_p1_support_receipt")}},
                "outputs": {
                    "labels": {"sha256": sha(fresh_labels)}, "parent_row_map": {"sha256": sha(fresh_source)},
                    "p1_anchor_projection_csv": {"sha256": sha(fresh_projection)}, "p1_anchor_csr": {"sha256": sha(fresh_p1)},
                },
            }
            for key, path in (("combined_node_map", combined_map), ("skin_p1_support", skin_support),
                              ("fat_p1_support", fat_support)):
                fresh_receipt["inputs"][key] = {"path": str(path), "sha256": sha(path)}
            fresh_receipt_path = fresh / "receipt.json"
            fresh_receipt_path.write_text(json.dumps(fresh_receipt), encoding="utf-8")
            result = inc.merge_incremental(
                base_node_path=node, base_ele_path=ele, candidate_node_path=candidate_node,
                candidate_ele_path=candidate_ele, node_map_path=node_map,
                base_labels_path=labels, base_source_path=source, base_projection_path=projection,
                base_anchor_path=p1, base_export_receipt_path=base_receipt_path,
                fresh_dir=fresh, fresh_receipt_path=fresh_receipt_path, output_dir=out,
                base_registration_map_path=reg_map, base_registration_receipt_path=reg_receipt_path)
            self.assertEqual(result["counts"]["reused_exact_ordered_cells"], 2)
            self.assertEqual(result["counts"]["fresh_changed_cells"], 1)
            self.assertEqual(result["counts"]["retired_base_cells"], 1)
            self.assertEqual(result["node_registration_reuse"]["status"], "PASS_REINDEXED_BASE_NODE_REGISTRATION")
            with (out / "child.labels.csv").open() as f:
                got = list(csv.DictReader(f))
            self.assertEqual([int(r["child_tet_row"]) for r in got], [0,1,2])
            self.assertEqual([int(r["stable_source_face_id"]) for r in got], [100,100,100])
            with np.load(out / "p1-anchor.npz", allow_pickle=False) as z:
                self.assertEqual(z["p1_support_cell_row"].tolist(), [2,1,0])
                self.assertEqual(z["p1_parent_tet_rows"].tolist(), [502,501,500])
            with (out / "registered-node-map.csv").open() as f:
                nodes = list(csv.DictReader(f))
            self.assertEqual([int(r["joint_node_id"]) for r in nodes], [1,2,3,4,5])

    def test_duplicate_candidate_tet_identity_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate ordered cell identity"):
            inc.build_ordered_tet_row_map(np.asarray([[0,1,2,3]]), np.asarray([15]),
                                          np.asarray([[0,1,2,3],[0,1,2,3]]), np.asarray([15,15]),
                                          np.arange(4))

    def test_node_map_requires_exact_f32_and_one_to_one(self):
        base = np.asarray([[0,0,0],[1,0,0]], np.float32)
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            inc.validate_node_map(base, base, np.asarray([0,0]))
        moved = base.copy(); moved[1,0] = np.nextafter(moved[1,0], np.float32(2))
        with self.assertRaisesRegex(ValueError, "F32 bytes"):
            inc.validate_node_map(base, moved, np.asarray([0,1]))

    def test_changed_row_semantic_cross_material_union_fails(self):
        fresh = archive([0], [0])
        fresh["p1_semantic_tuples"] = np.asarray([[16,20,30,100]], np.uint32)
        labels = {0: ["0", "15", "20", "30", "100"]}
        projection = {0: projection_row(0, 0)}
        with self.assertRaisesRegex(ValueError, "cross-material"):
            inc.validate_changed_semantics(labels, projection, fresh, {0: 0},
                                           np.asarray([[0,1,2,3]]), np.asarray([15]))

    def test_177_interface_scalar_is_preserved_and_role_support_is_material_specific(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            combined=root/"combined.csv.gz"
            write_combined_node_map_gz(combined, [
                ["1","skin","1","0","fat","1","I","0"],
                ["1","fat","1","0","skin","1","I","0"]])
            base=root/"base-177.csv"; output=root/"rebased-177.csv"
            points=np.asarray([[0,0,0]],dtype=np.float32)
            rows=[["1","skin","1","I","0","0","0","98304","65","66"],
                  ["1","fat","1","I","0","0","0","98304","65","66"]]
            write_registration_rows(base,rows)
            inc.rebase_registered_node_map(base,output,np.asarray([0]),True,points)
            skin=root/"skin.npz"; fat=root/"fat.npz"
            write_role_support(skin,15,[(15,65,66)])
            # The preserved geometric donor scalar is 65, but the authenticated
            # Fat support group is (16,66,66); they are separate fields.
            write_role_support(fat,16,[(16,66,66)])
            result=inc.verify_registered_node_role_support(
                base,output,combined,np.asarray([0]),1,skin,fat)
            self.assertEqual(result["status"],
                "PASS_REINDEXED_177_ROLE_ROWS_WITH_MATERIAL_SPECIFIC_171_NODE_GROUPS")
            self.assertEqual(result["role_source_node_records"],{"skin":1,"fat":1})
            with output.open(newline="") as f:
                got=list(csv.DictReader(f))
            self.assertEqual([r["geometric_donor_body"] for r in got],["65","65"])
            self.assertEqual([r["anatomical_region_id"] for r in got],["66","66"])

    def test_177_rejects_role_source_identity_not_in_combined_map(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); combined=root/"combined.csv.gz"
            write_combined_node_map_gz(combined,[
                ["1","skin","1","0","fat","1","I","0"],
                ["1","fat","1","0","skin","1","I","0"]])
            base=root/"base.csv"; output=root/"output.csv"; points=np.asarray([[0,0,0]],np.float32)
            write_registration_rows(base,[
                ["1","skin","1","I","0","0","0","98304","65","66"],
                ["1","fat","1","I","0","0","0","98304","65","66"]])
            inc.rebase_registered_node_map(base,output,np.asarray([0]),True,points)
            with output.open(newline="") as f:
                rows=list(csv.reader(f))
            rows[2][2]="2"
            write_csv(output,tuple(rows[0]),rows[1:])
            skin=root/"skin.npz";fat=root/"fat.npz"
            write_role_support(skin,15,[(15,65,66)])
            write_role_support(fat,16,[(16,66,66),(16,7,8)])
            with self.assertRaisesRegex(ValueError,"role/source identity disagrees"):
                inc.verify_registered_node_role_support(base,output,combined,
                    np.asarray([0]),1,skin,fat)

    def test_177_rejects_material_crossing_node_group_csr(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); combined=root/"combined.csv.gz"
            write_combined_node_map_gz(combined,[
                ["1","skin","1","0","fat","1","I","0"],
                ["1","fat","1","0","skin","1","I","0"]])
            base=root/"base.csv";output=root/"output.csv";points=np.asarray([[0,0,0]],np.float32)
            write_registration_rows(base,[
                ["1","skin","1","I","0","0","0","98304","65","66"],
                ["1","fat","1","I","0","0","0","98304","65","66"]])
            inc.rebase_registered_node_map(base,output,np.asarray([0]),True,points)
            skin=root/"skin.npz";fat=root/"fat.npz"
            write_role_support(skin,15,[(15,65,66)])
            write_role_support(fat,16,[(15,66,66)])
            with self.assertRaisesRegex(ValueError,"outside material 16"):
                inc.verify_registered_node_role_support(base,output,combined,
                    np.asarray([0]),1,skin,fat)

    def test_177_rebase_preserves_multiple_role_rows_per_node(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=root/"source.csv"; output=root/"rebased.csv"
            header=["joint_node_id","role","current_x_m","current_y_m","current_z_m"]
            write_csv(source,tuple(header),[[1,"I-skin",0,0,0],[1,"I-fat",0,0,0],[2,"O",1,0,0]])
            new_points=np.asarray([[1,0,0],[0,0,0]],dtype=np.float32)
            result=inc.rebase_registered_node_map(source,output,np.asarray([1,0]),True,new_points)
            self.assertEqual(result["status"],"PASS_REINDEXED_BASE_NODE_REGISTRATION")
            with output.open(newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([int(r["joint_node_id"]) for r in rows],[1,2,2])
            self.assertEqual([r["role"] for r in rows],["O","I-skin","I-fat"])


if __name__ == "__main__":
    unittest.main()
