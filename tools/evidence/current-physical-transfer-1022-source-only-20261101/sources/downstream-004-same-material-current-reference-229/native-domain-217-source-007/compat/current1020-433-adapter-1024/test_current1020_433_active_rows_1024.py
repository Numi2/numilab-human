#!/usr/bin/env python3
"""Focused active-row, full-crosswalk, and 433 integration controls."""
from __future__ import annotations

import importlib.util
import pathlib
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
MOD_PATH = HERE / "export_refined_coarse_registered_volume_433_p1_anchor_1024_active_rows_gzip.py"
ADAPTER_1021_PATH = HERE.parent / "current1020-433-adapter-1021" / "export_refined_coarse_registered_volume_433_p1_anchor_1021_current_schema.py"

def import_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

mod = import_file(MOD_PATH, "adapter_1024_test")
v1021 = import_file(ADAPTER_1021_PATH, "adapter_1021_test_nested")


def make_asset(path):
    points = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [8., 8., 8.]], dtype=np.float64)
    f32 = points.astype(np.float32)
    payload = {
        "points_world_m": points, "points_world_m_f32": f32,
        "faces": np.asarray([[0, 2, 3], [3, 2, 0]], dtype=np.int64),
        "source_input_vertex_quotient": np.asarray([0, 1, 1, 3], dtype=np.int64),
        "source_input_face_rows": np.asarray([10, 12], dtype=np.int64),
        "quality_star_point_output_to_input_row": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "quality_star_face_output_to_input_row": np.asarray([10, -1], dtype=np.int64),
        "face_source_layer": np.asarray([2, 2], dtype=np.int64),
        "face_source_body": np.asarray([7, 7], dtype=np.int64),
        "face_source_bone": np.asarray([11, 11], dtype=np.int64),
        "face_source_face_id": np.asarray([21, 22], dtype=np.int64),
        "face_source_original_id": np.asarray([31, 32], dtype=np.int64),
        "face_source_identity0": np.asarray([41, 42], dtype=np.int64),
        "face_source_identity1": np.asarray([51, 52], dtype=np.int64),
        "face_ancestry_offsets": np.asarray([0, 2, 3], dtype=np.int64),
        "face_ancestry_source_face_rows": np.asarray([3, 4, 8], dtype=np.int64),
        "face_ancestry_source_original_ids": np.asarray([300, 301, 800], dtype=np.int64),
        "face_ancestry_parent_face_rows": np.asarray([13, 14, 18], dtype=np.int64),
    }
    np.savez(path, **payload)
    return payload


def make_crosswalk(rows=4):
    return {
        "path": "synthetic-crosswalk.npz", "sha256": "a" * 64,
        "schema": "numi.h706.217-registered-surface-node-map.v1",
        "inactive_sentinel": -1,
        "source_row_to_joint_node": {r: np.asarray([9, -1, 10, 11], dtype=np.int64) for r in mod.ROLE_ORDER},
        "active_source_vertex_rows": {r: np.asarray([0, 2, 3], dtype=np.int64) for r in mod.ROLE_ORDER},
        "inactive_source_vertex_rows": {r: np.asarray([1], dtype=np.int64) for r in mod.ROLE_ORDER},
    }


def expect_reject(fn, label):
    try:
        fn()
    except (ValueError, OSError):
        return
    raise AssertionError(f"negative control accepted: {label}")


with tempfile.TemporaryDirectory(prefix="433-current-active-1022-") as tmp:
    root = pathlib.Path(tmp)
    path = root / "current.npz"
    make_asset(path)
    digest = v1021.sha256_file(path)
    cw = make_crosswalk()
    mod.CROSSWALK = cw
    current = mod.load_active_current_typed_role_asset("O", path, digest)
    assert current["points_world_m_f32"].shape == (3, 3)
    assert current["faces"].tolist() == [[0, 1, 2], [2, 1, 0]]
    assert current["active_source_vertex_rows"].tolist() == [0, 2, 3]
    assert current["inactive_source_vertex_rows"].tolist() == [1]
    assert current["source_vertex_to_compact_role_row"].tolist() == [0, -1, 1, 2]
    assert current["typed_face_ancestry_source_face_rows"].tolist() == [3, 4, 8]
    assert current["typed_face_ancestry_source_original_ids"].tolist() == [300, 301, 800]
    assert current["typed_face_ancestry_parent_face_rows"].tolist() == [13, 14, 18]

    crosswalk_path = root / "registered-surface-node-map.npz"
    crosswalk_payload = {"inactive_sentinel": np.asarray(-1, dtype=np.int64)}
    for role in mod.ROLE_ORDER:
        crosswalk_payload[f"{role}_source_row_to_joint_node"] = cw["source_row_to_joint_node"][role]
        crosswalk_payload[f"{role}_active_source_vertex_rows"] = cw["active_source_vertex_rows"][role]
        crosswalk_payload[f"{role}_inactive_source_vertex_rows"] = cw["inactive_source_vertex_rows"][role]
    np.savez(crosswalk_path, **crosswalk_payload)
    crosswalk_sha = mod.sha256_file(crosswalk_path)
    loaded_crosswalk = mod.load_surface_crosswalk(crosswalk_path, crosswalk_sha)
    assert np.array_equal(loaded_crosswalk["source_row_to_joint_node"]["O"], np.asarray([9, -1, 10, 11]))

    owner = mod._load_1021().load_pinned_433_owner()
    mod._install(owner)
    # Existing 938 CSV products may be gzip-compressed. Verify streamed input
    # and keep the existing plain CSV owner path intact.
    import csv
    import gzip
    gz_csv = root / "synthetic-node-map.csv.gz"
    with gzip.open(gz_csv, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["role", "vertex"])
        writer.writeheader()
        writer.writerow({"role": "O", "vertex": "7"})
    assert owner.read_csv(gz_csv) == [{"role": "O", "vertex": "7"}]
    plain_csv = root / "synthetic-node-map.csv"
    with plain_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["role", "vertex"])
        writer.writeheader()
        writer.writerow({"role": "I", "vertex": "4"})
    assert owner.read_csv(plain_csv) == [{"role": "I", "vertex": "4"}]
    loaded_by_patched_owner = owner.load_current_typed_role_asset("O", path, digest)
    assert loaded_by_patched_owner["points_world_m_f32"].shape == (3, 3)
    assert "current_typed_source_surface_vertex_row" in owner.CURRENT_TYPED_NODE_MAP_HEADER

    good_rows = []
    for role in mod.ROLE_ORDER:
        for source_row in (0, 2, 3):
            good_rows.append({
                "role": role, "registered_surface_vertex_row": str(source_row),
                "joint_node_id": str(int(cw["source_row_to_joint_node"][role][source_row]) + 1),
            })
    compact_rows = mod._compact_node_map_rows(good_rows, {r: current for r in mod.ROLE_ORDER})
    assert [int(r["registered_surface_vertex_row"]) for r in compact_rows[:3]] == [0, 1, 2]
    assert [int(r["current_typed_source_surface_vertex_row"]) for r in compact_rows[:3]] == [0, 2, 3]
    assert [int(r["joint_node_id"]) - 1 for r in compact_rows[:3]] == [9, 10, 11]

    # Exercise the installed final-candidate validator with sparse original IDs.
    cw = make_crosswalk()
    cw["source_row_to_joint_node"] = {
        "O": np.asarray([0, -1, 1, 2], dtype=np.int64),
        "I": np.asarray([3, -1, 4, 5], dtype=np.int64),
        "D": np.asarray([6, -1, 7, 8], dtype=np.int64),
    }
    mod.CROSSWALK = cw
    role_assets = {
        role: owner.load_current_typed_role_asset(role, path, digest)
        for role in mod.ROLE_ORDER
    }
    source_rows = (0, 2, 3)
    child_points = np.vstack([role_assets[r]["points_world_m_f32"] for r in mod.ROLE_ORDER]).astype(np.float32)
    candidate_rows = []
    for role_index, role in enumerate(mod.ROLE_ORDER):
        for compact_row, source_row in enumerate(source_rows):
            node = role_index * 3 + compact_row
            candidate_rows.append({
                "volume_node_row": node, "role": role, "role_row": compact_row,
                "source_parent_node_row": -1, "current_surface_vertex_row": compact_row,
                "current_typed_source_surface_vertex_row": source_row,
                "historical_registration_residual_x_m": 0.0,
                "historical_registration_residual_y_m": 0.0,
                "historical_registration_residual_z_m": 0.0,
                "historical_registration_distance_m": 0.0,
            })
    import json
    import hashlib
    node_hash = "c" * 64
    pin_json = {r: {"path": role_assets[r]["path"], "sha256": role_assets[r]["sha256"]}
                for r in mod.ROLE_ORDER}
    candidate = {
        "candidate_mapping_mode": np.asarray([owner.CURRENT_TYPED_VOLUME_SCHEMA]),
        "current_typed_node_map_sha256": np.asarray([node_hash]),
        "current_typed_role_asset_pins_json": np.asarray([json.dumps(pin_json, sort_keys=True)]),
        "points_reference_world_m_f64": child_points.astype(np.float64),
        "parent_volume_node_row": np.full(9, -1, dtype=np.int64),
        "current_typed_historical_registration_residual_world_m": np.zeros((9, 3), dtype=np.float64),
        "current_typed_historical_registration_distance_m": np.zeros(9, dtype=np.float64),
        "child_parent_reference_kind": np.asarray([], dtype="U64"),
        "volume_tetrahedra": np.empty((0, 4), dtype=np.int64),
        "current_typed_child_label_reference_source_sha256": np.asarray(["d" * 64]),
    }
    pins = {r: {"path": role_assets[r]["path"], "sha256": role_assets[r]["sha256"],
                "loaded_asset": role_assets[r]} for r in mod.ROLE_ORDER}
    assert owner.validate_current_typed_candidate_binding(candidate, candidate_rows,
        node_hash, "e" * 64, pins, True, True)
    wrong_source = [dict(r) for r in candidate_rows]
    wrong_source[1]["current_typed_source_surface_vertex_row"] = 1
    expect_reject(lambda: owner.validate_current_typed_candidate_binding(candidate, wrong_source,
        node_hash, "e" * 64, pins, True, True), "inactive original source row in final node map")

    # The native node-registration source_vertex_row is original; surface_vertex_row is dense.
    registration_fixture = (
        "        registration_rows.append([role, local, IDENTITY_KIND, cell, *beta,\n"
        "                                      *parent_residual[node], norm(parent_residual[node]), 0, FRAME_METHOD,\n"
        "                                      ROLE_DOMAIN[role], local, -1, 0, 0, 0, 0, 0, 0])"
    )
    registration_rewritten = mod._replace_node_registration_source_row(registration_fixture)
    assert "registration_rows.append([role, local, IDENTITY_KIND" in registration_rewritten
    assert "ROLE_DOMAIN[role], source_vertex_row, -1" in registration_rewritten
    assert 'source_vertex_row = int(row.get("current_typed_source_surface_vertex_row", local))' in registration_rewritten
    assert "ROLE_DOMAIN[role], local, -1" not in registration_rewritten

    bad = make_crosswalk()
    bad["active_source_vertex_rows"]["O"] = np.asarray([0, 2], dtype=np.int64)
    expect_reject(lambda: mod.validate_role_crosswalk(current, bad), "active set not derived from faces")
    bad = make_crosswalk()
    bad["source_row_to_joint_node"]["O"][1] = 12
    expect_reject(lambda: mod.validate_role_crosswalk(current, bad), "inactive row mapped")
    bad = make_crosswalk()
    bad["source_row_to_joint_node"]["O"][3] = bad["source_row_to_joint_node"]["O"][2]
    expect_reject(lambda: mod.validate_role_crosswalk(current, bad), "duplicate active joint owner")
    badrows = [dict(r) for r in good_rows]
    badrows[0]["joint_node_id"] = "99"
    expect_reject(lambda: mod._compact_node_map_rows(badrows, {r: current for r in mod.ROLE_ORDER}),
                  "joint node differs from full crosswalk")
    missing = [dict(r) for r in good_rows if not (r["role"] == "O" and r["registered_surface_vertex_row"] == "3")]
    expect_reject(lambda: mod._compact_node_map_rows(missing, {r: current for r in mod.ROLE_ORDER}),
                  "face-active row missing from 217 table")
    inactive = [dict(r) for r in good_rows]
    inactive.append({"role": "O", "registered_surface_vertex_row": "1", "joint_node_id": "12"})
    expect_reject(lambda: mod._compact_node_map_rows(inactive, {r: current for r in mod.ROLE_ORDER}),
                  "inactive source point emitted as volume node")
    print("current_active_row_adapter_1024_controls=PASS full-domain=4 active=3 inactive=1 noncontiguous-source-ID-compaction=PASS crosswalk=zero-based joint row; 217 node IDs=one-based negative_controls=8 local-source-row-mapping=PASS gzip-and-plain-csv=PASS")

