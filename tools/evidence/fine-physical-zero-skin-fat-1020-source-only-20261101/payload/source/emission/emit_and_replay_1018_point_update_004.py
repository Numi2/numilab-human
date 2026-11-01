#!/usr/bin/env python3
"""Apply and independently replay the pinned 1018 integer-F32 point fit on current 1009.

This is a coordinate-only candidate emitter. It does not run source admission, native
execution, or mass transfer. Historical ancestry is preserved; only immediate-parent
maps and current parent aliases are rebound for the coordinate-only step.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

SCHEMA = "numi.h706.current1009.integer-f32-1018-point-update-replay.v1"
Q_GATE = 1.0e-5
DET6_FLOOR_M3 = 6.0e-18
SOURCE_MOVE_CAP_M = 50.0e-6
SOURCE_DECLARED_CAP_M = 250.0e-6
INTERIOR_ABS_CAP_M = 1.0e-3
LOCAL_VOLUME_CAP = 1.0e-2
GLOBAL_VOLUME_CAP = 1.0e-4
POLICY_AREA = (0.25, 4.0)
POLICY_COS = 0.1
EXPECTED_FIT_SHA = "ad36a5404221289072a942037fd18a766e0cf945f3d1bb44ffba1faa34d46447"
EXPECTED_PRODUCER_SHA = "5ceca4419a8157e758d417b9a9528650f82435393001faec30ef73ec2f5f325f"
EXPECTED_PRODUCER_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/integer-f32-quality-1018-root/fit_integer_f32_quality.py")
EXPECTED_WARM_SHA = "ae5b4963dcbe5007e00ae944de17eb80c85440a87e6be14db882209c225d20bc"
EXPECTED_STATE_SHA = "413d7ecbf624c0c89c13f3f63905d7c7baf9d222d34d7ce11b706e95a0e4ec2a"
EXPECTED_PREFLIGHT_SHA = "1b58a91f373d23c33e27b7d0a582feb3f1135ed1f92708ab96bf461ecbb236e4"
EXPECTED_FIT_BUNDLE_SHA = "8df1a74c1d9a8f9a6c24e0abf676978a707d9eb9cf6f1cbb89e3d14c81df6517"
PREPARATION_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/integer-f32-quality-1016-root/preparation.json")
HELPER_PATH = Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/all-quality-1009-banks-point-update-001/emit_and_replay_point_updates_001.py")
EXPECTED_HELPER_SHA = "c75a3b5532700df673113ce6ae7b069042244569f4abf03c714ad0953c5098b7"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def pin(path: Path) -> dict:
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha(p)}

def load_owner():
    if sha(HELPER_PATH) != EXPECTED_HELPER_SHA:
        raise ValueError("reusable replay owner source changed")
    spec = importlib.util.spec_from_file_location("point_update_replay_owner_001", HELPER_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod

def scalar_text(mod, payload: dict, key: str) -> str:
    return mod.string_value(payload[key])

def set_scalar(mod, payload: dict, key: str, value: str) -> None:
    mod.set_string_like(payload, key, value)

def policy_screen(old_points: np.ndarray, new_points: np.ndarray, faces: np.ndarray,
                  moved_source_rows: set[int], area_bounds=POLICY_AREA, cosine_min=POLICY_COS):
    old = np.asarray(old_points, dtype=np.float32)
    new = np.asarray(new_points, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    if old.shape != new.shape or old.ndim != 2 or old.shape[1] != 3:
        raise ValueError("source point lane shape mismatch")
    if f.ndim != 2 or f.shape[1] != 3 or (f.size and (f.min() < 0 or f.max() >= len(old))):
        raise ValueError("source faces malformed")
    moved = np.asarray(sorted(moved_source_rows), dtype=np.int64)
    rows = (np.flatnonzero(np.isin(f, moved).any(axis=1)) if len(moved) else np.empty(0, np.int64))
    details = []
    for row in rows:
        tri = f[int(row)]
        a0 = old[tri].astype(np.float64)
        a1 = new[tri].astype(np.float64)
        n0 = np.cross(a0[1] - a0[0], a0[2] - a0[0])
        n1 = np.cross(a1[1] - a1[0], a1[2] - a1[0])
        n0n, n1n = float(np.linalg.norm(n0)), float(np.linalg.norm(n1))
        if not (n0n > 0 and n1n > 0 and math.isfinite(n0n) and math.isfinite(n1n)):
            raise ValueError("updated source face has zero/nonfinite area: " + str(int(row)))
        ratio = n1n / n0n
        cosine = float(np.dot(n0, n1) / (n0n * n1n))
        details.append({"row": int(row), "area_ratio": ratio, "orientation_cosine": cosine})
        if not (area_bounds[0] <= ratio <= area_bounds[1]):
            raise ValueError(f"source face {int(row)} area ratio {ratio} outside {area_bounds}")
        if cosine < cosine_min:
            raise ValueError(f"source face {int(row)} orientation cosine {cosine} below {cosine_min}")
    return {"rows": details, "count": len(details),
            "area_ratio_min": min((x["area_ratio"] for x in details), default=1.0),
            "area_ratio_max": max((x["area_ratio"] for x in details), default=1.0),
            "orientation_cosine_min": min((x["orientation_cosine"] for x in details), default=1.0)}

def apply_rows_preserving_unmoved_f64(points_f32, points_full, updates):
    f32 = np.asarray(points_f32, dtype=np.float32).copy()
    full = np.asarray(points_full).copy()
    if f32.ndim != 2 or f32.shape[1] != 3 or full.shape != f32.shape:
        raise ValueError("coordinate lanes must have equal Nx3 shape")
    seen = set()
    for row, update in updates.items():
        row = int(row)
        if row in seen or row < 0 or row >= len(f32):
            raise ValueError("duplicate/out-of-range coordinate update row")
        seen.add(row)
        old = np.asarray(update["old"], dtype=np.float32)
        new = np.asarray(update["new"], dtype=np.float32)
        if old.shape != (3,) or new.shape != (3,):
            raise ValueError("coordinate update must be a 3-vector")
        if not np.array_equal(f32[row].view(np.uint32), old.view(np.uint32)):
            raise ValueError("coordinate update old endpoint mismatch")
        if np.array_equal(old.view(np.uint32), new.view(np.uint32)):
            raise ValueError("no-op F32 update must be filtered before application")
        f32[row] = new
        full[row] = new.astype(full.dtype)
    return f32, full

def controls():
    p = np.asarray([[0,0,0],[1,0,0],[0,1,0]], np.float32)
    f = np.asarray([[0,1,2]], np.int64)
    scale = np.asarray([[0,0,0],[1.9,0,0],[0,1.9,0]], np.float32)
    ok = policy_screen(p, scale, f, {0}, (0.25,4.0), .1)
    assert ok["count"] == 1 and ok["area_ratio_min"] > 3.0
    for bad, bounds, cos in [
        (np.asarray([[0,0,0],[.4,0,0],[0,.4,0]], np.float32), (.25,4.0), .1),
        (np.asarray([[0,0,0],[0,1,0],[1,0,0]], np.float32), (.25,4.0), .1),
    ]:
        try:
            policy_screen(p, bad, f, {0}, bounds, cos)
        except ValueError:
            pass
        else:
            raise AssertionError("source face policy accepted invalid geometry")
    old32 = np.asarray([[1.,2.,3.],[4.,5.,6.]], np.float32)
    old64 = old32.astype(np.float64)
    old64[0,0] += 2.0**-30
    old64_bytes = old64[0].tobytes()
    upd = {1: {"old": old32[1].copy(), "new": np.asarray([4.1,5,6], np.float32)}}
    got32, got64 = apply_rows_preserving_unmoved_f64(old32, old64, upd)
    assert got64[0].tobytes() == old64_bytes
    assert np.array_equal(got32[0].view(np.uint32), old32[0].view(np.uint32))
    assert np.array_equal(got64[1].astype(np.float32).view(np.uint32), got32[1].view(np.uint32))
    try:
        apply_rows_preserving_unmoved_f64(old32, old64, {1: {"old":old32[1], "new":old32[1]}})
    except ValueError as e:
        assert "no-op" in str(e)
    else:
        raise AssertionError("no-op point update was not filtered")
    return {"declared_area_cosine_policy_accepts_3p61_area_ratio": True,
            "rejects_area_outside_declared_policy": True,
            "rejects_orientation_below_declared_policy": True,
            "unmoved_F64_residual_bytes_preserved": True,
            "moved_F64_lane_matches_emitted_F32": True,
            "no_op_F32_updates_rejected": True}

def check_file_pin(row: dict, label: str, expected_sha: str | None = None):
    if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
        raise ValueError(label + " pin missing")
    p = Path(row["path"]).resolve()
    if not p.is_file() or sha(p) != str(row["sha256"]):
        raise ValueError(label + " pin content mismatch")
    if expected_sha and str(row["sha256"]) != expected_sha:
        raise ValueError(label + " does not match required immutable SHA")
    return {"path": str(p), "sha256": str(row["sha256"])}

def read_path_pin(report: dict, field: str, path_key="path", hash_key="sha256"):
    row = report.get(field)
    if not isinstance(row, dict):
        raise ValueError("missing report pin " + field)
    p = Path(row.get(path_key, "")).resolve()
    h = str(row.get(hash_key, ""))
    if not p.is_file() or sha(p) != h:
        raise ValueError("report pin content mismatch: " + field)
    return p, h

def load_and_validate_inputs(mod, state_path, fit_path, preflight_path):
    state_path, fit_path, preflight_path = map(lambda p: Path(p).resolve(),
                                                (state_path, fit_path, preflight_path))
    state_digest = sha(state_path)
    if state_digest != EXPECTED_STATE_SHA:
        raise ValueError("1009 selected state SHA mismatch")
    if sha(fit_path) != EXPECTED_FIT_SHA:
        raise ValueError("1018 fit report SHA mismatch")
    raw = json.loads(state_path.read_text())
    wrapper, state = mod.unwrap_state(raw)
    assets = mod.current_asset_pins(state)
    for name, row in assets.items():
        mod.assert_pin(row, row, name + " state asset")
    fit = json.loads(fit_path.read_text())
    if fit.get("status") != "LOCAL_ALL_GOOD_PENDING_FULL_REPLAY":
        raise ValueError("1018 report is not the selected local all-good result")
    if fit.get("producer_sha256") != EXPECTED_PRODUCER_SHA:
        raise ValueError("1018 producer SHA mismatch")
    producer = EXPECTED_PRODUCER_PATH
    if not producer.is_file() or sha(producer) != EXPECTED_PRODUCER_SHA:
        raise ValueError("1018 producer file pin mismatch")
    policy = fit.get("construction_policy", {})
    if policy.get("name") != "DECLARED_INNER_I_RECONSTRUCTION_TRIAL":
        raise ValueError("unexpected fit construction policy")
    if policy.get("area_ratio_bounds") != [0.25, 4.0] or float(policy.get("normal_cosine_min", -1)) != 0.1:
        raise ValueError("1018 source-face policy does not match its pinned construction policy")
    if float(policy.get("source_position_cap_m", -1)) != SOURCE_DECLARED_CAP_M:
        raise ValueError("1018 source cap does not match fit policy")
    if float(policy.get("native_q_gate_unchanged", -1)) != Q_GATE or float(policy.get("native_det6_floor_unchanged", -1)) != DET6_FLOOR_M3:
        raise ValueError("1018 native quality policy changed")
    if int(policy.get("required_final_source_crossings", -1)) != 0:
        raise ValueError("1018 report does not require zero final crossings")
    if fit.get("positive") is not True or fit.get("source_constraints_pass") is not True or float(fit.get("final_deficit", math.inf)) != 0.0:
        raise ValueError("1018 fit did not report its local source/quality gates")
    for role in ("Skin", "Fat"):
        d = fit.get("final_roles", {}).get(role, {})
        if int(d.get("bad", -1)) != 0 or int(d.get("oldgood_regressions", -1)) != 0:
            raise ValueError("1018 fit did not report all-good " + role)
    warm = fit.get("warm_start", {})
    warm_path = Path(warm.get("path", "")).resolve()
    if not warm_path.is_file() or sha(warm_path) != str(warm.get("sha256", "")) or str(warm.get("sha256")) != EXPECTED_WARM_SHA:
        raise ValueError("1018 warm-start report pin mismatch")
    warm_report = json.loads(warm_path.read_text())
    bundle_path, bundle_sha = read_path_pin(fit, "input")
    if bundle_sha != EXPECTED_FIT_BUNDLE_SHA:
        raise ValueError("1018 output bundle pin mismatch")
    warm_input = warm_report.get("input", {})
    if Path(warm_input.get("path", "")).resolve() != bundle_path or str(warm_input.get("sha256", "")) != bundle_sha:
        raise ValueError("1016 warm start does not bind to the 1018 input bundle")
    preparation = json.loads(PREPARATION_PATH.read_text())
    if preparation.get("status") != "PASS_DERIVED_LOCAL_INPUT":
        raise ValueError("1016 local bundle preparation did not pass")
    prep_input = preparation.get("input", {})
    prep_output = preparation.get("output", {})
    if Path(prep_output.get("path", "")).resolve() != bundle_path or str(prep_output.get("sha256", "")) != bundle_sha:
        raise ValueError("1016 preparation does not bind to 1018 bundle")
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("schema") != "numi.h706.cap-one-ring-preflight.v1" or preflight.get("status") != "PASS_READONLY_LOCAL_BUNDLE":
        raise ValueError("preflight source/protection report is not a passing owner receipt")
    if preflight.get("input_state") != {"path": str(state_path), "sha256": state_digest}:
        raise ValueError("preflight does not bind to the selected 1009 state")
    prebundle = preflight.get("bundle", {})
    if Path(prebundle.get("path", "")).resolve() != Path(prep_input.get("path", "")).resolve() or str(prebundle.get("sha256", "")) != str(prep_input.get("sha256", "")):
        raise ValueError("preflight and 1016 preparation do not share the exact input bundle")
    if prebundle.get("sha256") != EXPECTED_PREFLIGHT_SHA or prep_input.get("sha256") != EXPECTED_PREFLIGHT_SHA:
        raise ValueError("preflight bundle SHA mismatch")
    preflight["local_bundle_preparation_1016"] = pin(PREPARATION_PATH)
    pprod = preflight.get("producer", {})
    if not pprod.get("path") or not Path(pprod["path"]).is_file() or sha(Path(pprod["path"])) != pprod.get("sha256"):
        raise ValueError("preflight producer source pin invalid")
    return raw, wrapper, state, assets, fit, preflight, bundle_path, state_digest

def actual_updates(mod, fit, state, assets, bundle_path, preflight):
    parents = {role: mod.load_npz(Path(assets[role]["path"])) for role in ("Skin", "Fat")}
    source = mod.load_npz(Path(assets["I"]["path"]))
    maps = mod.check_source_maps(source, parents)
    bundle = mod.load_npz(bundle_path)
    records = fit.get("variable_records")
    bases = np.asarray(fit.get("base_variables_f32"), dtype=np.float32)
    finals = np.asarray(fit.get("final_variables_f32"), dtype=np.float32)
    if not isinstance(records, list) or len(records) != 166 or bases.shape != (166,3) or finals.shape != (166,3):
        raise ValueError("1018 variable arrays do not match the pinned 166-variable roster")
    source_points = np.asarray(source["points_world_m_f32"], dtype=np.float32)
    source_updates = {}
    interior_updates = []
    source_declared_rows = set()
    source_move_values = []
    interior_move_values = []
    seen_variable_keys = set()
    surface_sets = {r: mod.volume_surface_point_rows(parents[r], r) for r in ("Skin", "Fat")}
    protected_other = {r: set().union(*(rows for role, rows in surface_sets[r].items() if role != "I"))
                       if surface_sets[r] else set() for r in ("Skin", "Fat")}
    all_surface = {r: set().union(*surface_sets[r].values()) if surface_sets[r] else set()
                   for r in ("Skin", "Fat")}
    for i, record in enumerate(records):
        base, new = bases[i].copy(), finals[i].copy()
        if not np.isfinite(base).all() or not np.isfinite(new).all():
            raise ValueError("nonfinite F32 variable coordinate")
        if not np.array_equal(base.view(np.uint32), np.asarray(record["base"], dtype=np.float32).view(np.uint32)):
            raise ValueError("variable base and aligned base_variables_f32 disagree")
        kind = record.get("kind")
        if kind == "I_surface":
            sid = int(record["source_row"])
            key = ("I_surface", sid)
            if key in seen_variable_keys or sid < 0 or sid >= len(source_points):
                raise ValueError("duplicate/out-of-range source I variable row")
            seen_variable_keys.add(key)
            source_declared_rows.add(sid)
            if not np.array_equal(source_points[sid].view(np.uint32), base.view(np.uint32)):
                raise ValueError("1018 source variable base does not match current I F32 row " + str(sid))
            for role in ("Skin", "Fat"):
                mapped = int(maps[role][sid])
                if mapped < 0 or mapped != int(record.get("role_rows", {}).get(role, -2)):
                    raise ValueError(role + " source-I role row mismatch at " + str(sid))
                vp = np.asarray(parents[role]["points_world_m_f32"], dtype=np.float32)
                if not np.array_equal(vp[mapped].view(np.uint32), base.view(np.uint32)):
                    raise ValueError(role + " source-I base point mismatch at " + str(sid))
                if mapped in protected_other[role]:
                    raise ValueError(role + " current I variable aliases O/D volume row")
            dist = float(np.linalg.norm(new.astype(np.float64) - base.astype(np.float64)))
            if dist > SOURCE_DECLARED_CAP_M + 1e-15:
                raise ValueError("source point exceeds the 1018 declared 250 um construction cap")
            if dist > SOURCE_MOVE_CAP_M + 1e-15:
                raise ValueError("source point exceeds the root-directed 50 um I motion bound")
            if not np.array_equal(base.view(np.uint32), new.view(np.uint32)):
                source_move_values.append(dist)
                source_updates[sid] = {"old": base, "new": new, "cap_m": SOURCE_MOVE_CAP_M,
                                       "move_m": dist}
        elif kind == "interior_volume":
            role = str(record.get("role", ""))
            row = int(record.get("volume_row", -1))
            key = ("interior_volume", role, row)
            if role not in ("Skin", "Fat") or key in seen_variable_keys or row < 0:
                raise ValueError("invalid/duplicate interior variable identity")
            seen_variable_keys.add(key)
            points = np.asarray(parents[role]["points_world_m_f32"], dtype=np.float32)
            if row >= len(points) or not np.array_equal(points[row].view(np.uint32), base.view(np.uint32)):
                raise ValueError(role + " interior variable base does not match current point row " + str(row))
            if row in all_surface[role]:
                raise ValueError(role + " interior variable is on a mapped source surface")
            cap = float(record.get("cap_m", math.nan))
            median = float(record.get("median_incident_edge_m", math.nan))
            dist = float(np.linalg.norm(new.astype(np.float64) - base.astype(np.float64)))
            if not (math.isfinite(cap) and math.isfinite(median) and cap >= 0 and median > 0):
                raise ValueError("invalid interior variable trust bound")
            if cap > min(INTERIOR_ABS_CAP_M, 0.5 * median) + 1e-15 or dist > cap + 1e-15:
                raise ValueError("interior variable exceeds its existing abs/half-median edge bound")
            if not np.array_equal(base.view(np.uint32), new.view(np.uint32)):
                interior_move_values.append(dist)
                interior_updates.append({"role": role, "row": row, "old": base, "new": new,
                                         "cap_m": cap, "move_m": dist, "median_edge_m": median})
        else:
            raise ValueError("unknown 1018 variable kind: " + str(kind))
    if len(source_declared_rows) != 58 or sum(1 for x in records if x.get("kind") == "interior_volume") != 108:
        raise ValueError("1018 declared variable partition is not 58 I / 108 interior")
    preflight_rows = set(map(int, preflight["cap"]["rows"])) | set(map(int, preflight["one_ring"]["eligible_ring_rows"]))
    if source_declared_rows != preflight_rows:
        raise ValueError("1018 source-variable rows differ from the exact preflight source roster")
    if preflight["cap"]["protected_overlap"] != {
        "O_or_D_coordinate_alias": [], "back": [], "rim": [], "routes": []}:
        raise ValueError("preflight seed protection report is not clear")
    if any(int(v) != 0 for v in preflight["one_ring"]["protected_by_category"].values()):
        raise ValueError("preflight I one-ring protection report is not clear")
    # Recheck explicit source protection masks against current row IDs.
    back_pin = preflight["back_scope"]
    if not Path(back_pin["path"]).is_file() or sha(Path(back_pin["path"])) != back_pin["sha256"]:
        raise ValueError("pinned back-scope row file changed")
    with np.load(back_pin["path"], allow_pickle=False) as z:
        back_i_rows = set(map(int, np.asarray(z["I_vertex_rows"], dtype=np.int64)))
    route_pin = preflight["route_inventory"]
    if not Path(route_pin["path"]).is_file() or sha(Path(route_pin["path"])) != route_pin["sha256"]:
        raise ValueError("pinned route inventory changed")
    routes = json.loads(Path(route_pin["path"]).read_text())
    route_i_rows = {int(x["surface_vertex_row"]) for x in routes.get("surface_routes", [])
                    if x.get("role") == "I" and "surface_vertex_row" in x}
    # The live I rim is the vertex set of source edges with exactly one incident face.
    faces = np.asarray(source["faces"], dtype=np.int64)
    edges = np.concatenate((faces[:,[0,1]], faces[:,[1,2]], faces[:,[2,0]]), axis=0)
    edges.sort(axis=1)
    edge_values, edge_counts = np.unique(edges, axis=0, return_counts=True)
    rim_i_rows = set(map(int, np.unique(edge_values[edge_counts == 1])))
    moved_rows = set(source_updates)
    for label, protected in (("back", back_i_rows), ("routes", route_i_rows), ("rim", rim_i_rows)):
        overlap = sorted(moved_rows & protected)
        if overlap:
            raise ValueError("actual moved I rows overlap protected " + label + ": " + str(overlap[:12]))
    for role in ("Skin", "Fat"):
        mapped_move_rows = {int(maps[role][sid]) for sid in moved_rows}
        if mapped_move_rows & protected_other[role]:
            raise ValueError(role + " moved I points alias current O/D volume rows")
    # Prove the bundle's complete local source-face stencil exactly matches the current source face IDs.
    local_points = np.asarray(bundle["I_local_point_rows"], dtype=np.int64)
    local_faces = np.asarray(bundle["I_local_faces"], dtype=np.int64)
    face_rows = np.asarray(bundle["I_source_face_rows"], dtype=np.int64)
    if local_faces.ndim != 2 or local_faces.shape[1] != 3 or len(face_rows) != len(local_faces):
        raise ValueError("pinned local bundle source face stencil is malformed")
    if len(set(map(int, face_rows))) != len(face_rows) or (len(local_faces) and (local_faces.min() < 0 or local_faces.max() >= len(local_points))):
        raise ValueError("pinned local source stencil has duplicate/out-of-range face rows")
    translated = local_points[local_faces]
    actual = faces[face_rows]
    if not np.array_equal(np.sort(translated, axis=1), np.sort(actual, axis=1)):
        raise ValueError("pinned local source face rows do not match current I face connectivity")
    moved_faces = set(map(int, np.flatnonzero(np.isin(faces, np.asarray(sorted(moved_rows), np.int64)).any(axis=1))))
    if not moved_faces.issubset(set(map(int, face_rows))):
        raise ValueError("fit input bundle omits a source face incident to an actually moved point")
    return source, parents, maps, source_updates, interior_updates, {
        "declared_source_variables": len(source_declared_rows),
        "actual_moved_source_points": len(source_updates),
        "actual_moved_interior_points": len(interior_updates),
        "max_actual_source_move_m": max(source_move_values, default=0.0),
        "max_actual_interior_move_m": max(interior_move_values, default=0.0),
        "source_preflight_roster_rows": len(preflight_rows),
        "back_protected_rows": len(back_i_rows),
        "route_protected_I_rows": len(route_i_rows),
        "rim_protected_I_rows": len(rim_i_rows),
        "source_face_stencil_rows": len(face_rows),
        "actual_moved_source_face_rows": sorted(moved_faces),
    }

def set_parent_aliases(mod, payload, parent_path, parent_sha, output_i_path=None, output_i_sha=None):
    for k in ("typed_parent_candidate_path", "source_current_candidate_path"):
        if k in payload:
            set_scalar(mod, payload, k, str(Path(parent_path).resolve()))
    for k in ("typed_parent_candidate_sha256", "source_current_candidate_sha256"):
        if k in payload:
            set_scalar(mod, payload, k, str(parent_sha))
    if output_i_path is not None:
        if "typed_I_candidate_path" in payload:
            set_scalar(mod, payload, "typed_I_candidate_path", str(Path(output_i_path).resolve()))
        if output_i_sha is not None and "typed_I_candidate_sha256" in payload:
            set_scalar(mod, payload, "typed_I_candidate_sha256", str(output_i_sha))
        for key in list(payload):
            if key.endswith("_current_target_rebind_source_asset_paths_json"):
                obj = json.loads(mod.string_value(payload[key]))
                if "typed_I" in obj:
                    obj["typed_I"] = str(Path(output_i_path).resolve())
                    set_scalar(mod, payload, key, json.dumps(obj, sort_keys=True))
            elif key.endswith("_current_target_rebind_source_asset_sha256_json"):
                obj = json.loads(mod.string_value(payload[key]))
                if "typed_I" in obj:
                    if output_i_sha is None:
                        obj["typed_I"] = "__SOURCE_I_SHA_PENDING__"
                    else:
                        obj["typed_I"] = str(output_i_sha)
                    set_scalar(mod, payload, key, json.dumps(obj, sort_keys=True))

def apply_candidate(mod, source, parents, maps, source_updates, interior_updates, assets, output_paths):
    iout = {k: np.array(v, copy=True) for k,v in source.items()}
    old_i32 = np.asarray(source["points_world_m_f32"], dtype=np.float32)
    old_i_full = np.asarray(source["points_world_m"]).copy()
    i32, ifull = apply_rows_preserving_unmoved_f64(old_i32, old_i_full, source_updates)
    iout["points_world_m_f32"] = i32
    iout["points_world_m"] = ifull
    # Same point/face domains: only direct input-to-output aliases become identity.
    if "source_input_vertex_quotient" in iout and np.asarray(iout["source_input_vertex_quotient"]).shape == (len(i32),):
        iout["source_input_vertex_quotient"] = np.arange(len(i32), dtype=np.int64)
    if "source_input_face_rows" in iout and np.asarray(iout["source_input_face_rows"]).shape == (len(iout["faces"]),):
        iout["source_input_face_rows"] = np.arange(len(iout["faces"]), dtype=np.int64)
    set_parent_aliases(mod, iout, assets["I"]["path"], assets["I"]["sha256"])
    # Retain all absolute source-face ancestry and historical operation records.
    ancestry_i_keys = [k for k in source if k.startswith("face_ancestry") or
                       "quality_star_face_output_to_input_row" in k]
    for k in ancestry_i_keys:
        if not mod.same_array(iout[k], source[k]):
            raise ValueError("source I historical face ancestry was changed: " + k)
    vout = {r: {k: np.array(v, copy=True) for k,v in parents[r].items()} for r in ("Skin","Fat")}
    movement = {}
    for role in ("Skin","Fat"):
        parent, output = parents[role], vout[role]
        old32 = np.asarray(parent["points_world_m_f32"], dtype=np.float32)
        full = np.asarray(parent["points_world_m"]).copy()
        updates = {}
        changed_volume_source_rows = set()
        for sid, u in source_updates.items():
            vr = int(maps[role][sid])
            if vr < 0:
                raise ValueError(role + " source update is not mapped")
            if not np.array_equal(old32[vr].view(np.uint32), u["old"].view(np.uint32)):
                raise ValueError(role + " mapped source update old coordinate mismatch")
            if vr in updates:
                raise ValueError(role + " conflicting updates share a volume point row")
            updates[vr] = {"old": u["old"], "new": u["new"]}
            changed_volume_source_rows.add(vr)
        interior = [u for u in interior_updates if u["role"] == role]
        all_surface_rows = set().union(*mod.volume_surface_point_rows(parent, role).values())
        for u in interior:
            vr = int(u["row"])
            if vr in all_surface_rows or vr in updates:
                raise ValueError(role + " interior point collides with a source-mapped point")
            if not np.array_equal(old32[vr].view(np.uint32), u["old"].view(np.uint32)):
                raise ValueError(role + " interior point old coordinate mismatch")
            updates[vr] = {"old":u["old"], "new":u["new"]}
        new32, newfull = apply_rows_preserving_unmoved_f64(old32, full, updates)
        output["points_world_m_f32"] = new32
        output["points_world_m"] = newfull
        # Recompute cached wall coordinates from row indices without changing wall topology.
        map_key = role.lower() + "_current_target_rebind_wall_faces_volume_rows"
        xyz_key = role.lower() + "_current_target_rebind_wall_face_points_world_m_f32"
        if map_key in parent and xyz_key in parent:
            wall = np.asarray(parent[map_key], dtype=np.int64)
            if wall.ndim != 2 or wall.shape[1] != 3 or (wall.size and (wall.min()<0 or wall.max()>=len(new32))):
                raise ValueError(role + " cached wall row map malformed")
            output[xyz_key] = new32[wall].copy()
        # Rebind only direct step aliases; retain canonical historical ancestry arrays.
        mod.reset_immediate_volume_maps(parent, output)
        set_parent_aliases(mod, output, assets[role]["path"], assets[role]["sha256"],
                           output_paths["I"], None)
        for key, value in (("current_geometry_parent_path", assets[role]["path"]),
                           ("current_geometry_parent_sha256", assets[role]["sha256"])):
            if key in output:
                expected_value = str(Path(value).resolve()) if key.endswith("_path") else str(value)
                set_scalar(mod, output, key, expected_value)
                if scalar_text(mod, output, key) != expected_value:
                    raise ValueError(role + " current geometry parent alias not rebound: " + key)
        for key in ("typed_O_candidate_path","typed_D_candidate_path"):
            if key in output:
                srcrole = key[len("typed_")].split("_")[0]
                expected = assets[srcrole]
                if scalar_text(mod, output, key) != expected["path"]:
                    raise ValueError(role + " current " + srcrole + " alias is stale")
        for key in ("typed_O_candidate_sha256","typed_D_candidate_sha256"):
            if key in output:
                srcrole = key[len("typed_")].split("_")[0]
                if scalar_text(mod, output, key) != assets[srcrole]["sha256"]:
                    raise ValueError(role + " current " + srcrole + " alias hash is stale")
        # Preserve canonical and historical parent unions exactly.
        for k in parent:
            if "canonical_parent_union" in k or "ancestry" in k.lower():
                if not mod.same_array(output[k], parent[k]):
                    raise ValueError(role + " historical ancestry changed: " + k)
        movement[role] = {"source_volume_rows": sorted(changed_volume_source_rows),
                          "interior_volume_rows": sorted(int(u["row"]) for u in interior),
                          "moved_volume_rows": sorted(updates)}
    return iout, vout, movement

def check_expected_diffs(mod, parent, candidate, allowed, label):
    if set(parent) != set(candidate):
        raise ValueError(label + " NPZ key list changed")
    changed = []
    for k in parent:
        if not mod.same_array(parent[k], candidate[k]):
            changed.append(k)
            if k not in allowed:
                raise ValueError(label + " unexpected non-coordinate/ancestry mutation: " + k)
    return changed

def source_face_stencil_check(mod, parent_i, candidate_i, source_updates, bundle, policy):
    old = np.asarray(parent_i["points_world_m_f32"], dtype=np.float32)
    new = np.asarray(candidate_i["points_world_m_f32"], dtype=np.float32)
    faces = np.asarray(parent_i["faces"], dtype=np.int64)
    touched = set(source_updates)
    checked = policy_screen(old, new, faces, touched, tuple(policy["area_ratio_bounds"]),
                            float(policy["normal_cosine_min"]))
    bundle_rows = set(map(int, np.asarray(bundle["I_source_face_rows"], dtype=np.int64)))
    got = {int(x["row"]) for x in checked["rows"]}
    if not got.issubset(bundle_rows):
        raise ValueError("source face screen found an incident face outside the fit stencil")
    return checked

def verify_and_emit(mod, state_path, fit_path, preflight_path, output_dir):
    state_path, fit_path, preflight_path, output_dir = map(
        lambda p: Path(p).resolve(), (state_path, fit_path, preflight_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    raw, wrapper, state, assets, fit, preflight, bundle_path, state_digest = load_and_validate_inputs(
        mod, state_path, fit_path, preflight_path)
    source, parents, maps, source_updates, interior_updates, counts = actual_updates(
        mod, fit, state, assets, bundle_path, preflight)
    bundle = mod.load_npz(bundle_path)
    policy = fit["construction_policy"]
    # Recheck fit baseline parent values before touching output payloads.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".1018-point-update-stage-", dir=output_dir.parent))
    try:
        paths = {"I": output_dir / "candidate-I-point-updated.npz",
                 "Skin": output_dir / "skin-volume-point-updated.npz",
                 "Fat": output_dir / "fat-volume-point-updated.npz"}
        iout, vout, movement = apply_candidate(mod, source, parents, maps, source_updates,
                                               interior_updates, assets, paths)
        source_checks = source_face_stencil_check(mod, source, iout, source_updates, bundle, policy)
        # Verify all active source-I mappings and geometry source protection after application.
        maps_out = mod.check_source_maps(iout, vout)
        for role in ("Skin","Fat"):
            p32 = np.asarray(vout[role]["points_world_m_f32"], dtype=np.float32)
            for sid, u in source_updates.items():
                vr = int(maps_out[role][sid])
                if not np.array_equal(p32[vr].view(np.uint32), u["new"].view(np.uint32)):
                    raise ValueError(role + " emitted I point does not equal candidate source coordinate")
        # Confirm every update has an exact full-star closure before running independent replay.
        fullstar = {}
        for role in ("Skin","Fat"):
            rows = set(maps_out[role][np.asarray(sorted(source_updates), dtype=np.int64)].tolist()) if source_updates else set()
            rows.discard(-1)
            rows |= {int(u["row"]) for u in interior_updates if u["role"] == role}
            tets = np.asarray(parents[role]["tetrahedra"], dtype=np.int64)
            cells = mod.star_cells(tets, rows)
            full_cells = set(map(int, np.flatnonzero(np.isin(tets, np.asarray(sorted(rows), dtype=np.int64)).any(axis=1)))) if rows else set()
            if set(map(int, cells)) != full_cells:
                raise ValueError(role + " moved-point full-star closure is incomplete")
            fullstar[role] = {"moved_volume_point_rows": sorted(rows),
                              "complete_incident_cell_count": len(cells),
                              "complete_star_row_set_sha256": hashlib.sha256(
                                  np.asarray(sorted(full_cells), dtype=np.int64).tobytes()).hexdigest()}
        # Ensure lineage-critical face and canonical ancestry arrays survived; direct map aliases are current.
        input_path_sha = assets["I"]["sha256"]
        if scalar_text(mod, iout, "typed_parent_candidate_path") != assets["I"]["path"] or scalar_text(mod, iout, "typed_parent_candidate_sha256") != input_path_sha:
            raise ValueError("typed parent candidate aliases do not point to current I input")
        if scalar_text(mod, iout, "source_current_candidate_path") != assets["I"]["path"] or scalar_text(mod, iout, "source_current_candidate_sha256") != input_path_sha:
            raise ValueError("source current candidate aliases do not point to current I input")
        # Serialize source I first so volume aliases can bind to its actual bytes.
        i_stage = stage / paths["I"].name
        i_receipt = mod.write_npz_and_verify(i_stage, iout)
        # Now fill all direct source-I aliases and current rebind source-pin JSONs.
        for role in ("Skin","Fat"):
            set_parent_aliases(mod, vout[role], assets[role]["path"], assets[role]["sha256"],
                               paths["I"], i_receipt["sha256"])
        # Rewriting I is unnecessary: only volume aliases refer to it.
        v_receipts = {}
        for role in ("Skin","Fat"):
            v_stage = stage / paths[role].name
            v_receipts[role] = mod.write_npz_and_verify(v_stage, vout[role])
        i_read = mod.load_npz(i_stage)
        v_read = {r: mod.load_npz(stage / paths[r].name) for r in ("Skin","Fat")}
        mod.verify_immediate_parent_pins(i_read, v_read, assets,
            {"path": str(paths["I"].resolve()), "sha256": i_receipt["sha256"]})
        # Check fields outside direct coordinate and immediate-parent maps remain byte-exact.
        i_allowed = {"points_world_m_f32","points_world_m","typed_parent_candidate_path",
                     "typed_parent_candidate_sha256","source_current_candidate_path",
                     "source_current_candidate_sha256","source_input_vertex_quotient","source_input_face_rows"}
        i_changed = check_expected_diffs(mod, source, i_read, i_allowed, "source I")
        for k in source:
            if k.startswith("face_ancestry") or "quality_star_face_output_to_input_row" in k:
                if not mod.same_array(source[k], i_read[k]):
                    raise ValueError("source I absolute/historical face ancestry changed: " + k)
        v_changed = {}
        for role in ("Skin","Fat"):
            allowed = {"points_world_m_f32","points_world_m",
                       role.lower()+"_current_target_rebind_wall_face_points_world_m_f32",
                       "current_geometry_parent_path","current_geometry_parent_sha256",
                       "typed_I_candidate_path","typed_I_candidate_sha256",
                       "output_cell_to_input_cell","output_cell_to_input_cell_rows",
                       "input_cell_to_output_cell","input_cell_to_output_cell_rows",
                       "retained_input_cell_rows","input_node_to_output_node",
                       "input_node_to_output_node_rows","output_node_to_input_node",
                       "output_node_to_input_node_rows","output_point_to_input_point_row",
                       "output_point_to_input_point_rows","source_interior_vertex_quotient",
                       "replacement_parent_offsets","replacement_parent_cell_rows",
                       "replacement_tet_offsets","replacement_tetrahedra",
                       "output_point_parent_offsets","output_point_parent_input_cell_rows",
                       role.lower()+"_current_target_rebind_source_asset_paths_json",
                       role.lower()+"_current_target_rebind_source_asset_sha256_json"}
            v_changed[role] = check_expected_diffs(mod, parents[role], v_read[role], allowed, role)
            for k in parents[role]:
                if "canonical_parent_union" in k or "ancestry" in k.lower():
                    if not mod.same_array(parents[role][k], v_read[role][k]):
                        raise ValueError(role + " absolute/historical ancestry changed: " + k)
            for key in ("typed_I_candidate_path","typed_I_candidate_sha256"):
                expected = str(paths["I"].resolve()) if key.endswith("path") else i_receipt["sha256"]
                if scalar_text(mod, v_read[role], key) != expected:
                    raise ValueError(role + " current typed-I alias is stale: " + key)
        # Full independent all-cell quality, exact changed-star positivity/floor, and paired exact-volume replay.
        replay_state = dict(state)
        replay_state["_candidate_report_source_face_checks"] = source_checks
        replay = mod.replay_pair(parents, v_read, {"I": i_read, **v_read}, replay_state,
                                 source, i_read, source_updates, interior_updates, maps_out, DET6_FLOOR_M3)
        for role in ("Skin","Fat"):
            if replay["roles"][role]["candidate_bad_cells"] != 0:
                raise ValueError(role + " full-body quality replay is not all good")
            if replay["roles"][role]["changed_star_exact_det6_floor_failures"]:
                raise ValueError(role + " exact changed-star det6 floor did not pass")
        if replay["paired_exact_det6_delta_integer"] != "0":
            raise ValueError("paired exact signed-volume change is not zero")
        # Add whole-body determinant diagnostics; exact determinant/floor proof above covers every changed star.
        full_det = {}
        for role in ("Skin","Fat"):
            det = mod.native_det6_float(np.asarray(v_read[role]["points_world_m_f32"], np.float32),
                                        np.asarray(v_read[role]["tetrahedra"], np.int64))
            full_det[role] = {"minimum_float_det6_m3": float(np.min(det)),
                              "cells_below_det6_floor_float_screen": int(np.count_nonzero(det < DET6_FLOOR_M3)),
                              "cell_count": int(len(det))}
            if full_det[role]["cells_below_det6_floor_float_screen"]:
                raise ValueError(role + " full-body float det6 screen found cells below the floor")
        # Construct a new pending-qualification state only after all local replay gates pass.
        core = copy.deepcopy(state)
        source_surfaces = dict(core["source_surfaces"])
        source_surfaces["I"] = {"path": str(paths["I"].resolve()), "sha256": i_receipt["sha256"]}
        core["source_surfaces"] = source_surfaces
        core["source_state"] = {"path": str(state_path), "sha256": state_digest}
        for role in ("Skin","Fat"):
            rr = replay["roles"][role]
            summary = dict(core.get(role, {}))
            summary.update({"path": str(paths[role].resolve()), "sha256": v_receipts[role]["sha256"],
                            "native_quality_failures": rr["candidate_bad_cells"],
                            "nonpositive": len(rr["changed_star_exact_nonpositive_cells"]),
                            "cells": rr["cell_count"], "nodes": rr["point_count"],
                            "geometry_changed_points": len(rr["changed_volume_point_rows"]),
                            "boundary_source_binding_status": "PENDING_FRESH_806_808_AFTER_I_COORDINATE_UPDATE",
                            "boundary_source_metadata": "PENDING_FRESH_SOURCE_QUALIFICATION"})
            current = dict(core.get(role+"_current", {}))
            current.update(summary)
            core[role] = summary
            core[role+"_current"] = current
        for key in ("all185_775","exact_context","source_audit_status","source_crossing_audit_status","typed_217_admission"):
            if key in core:
                core["historical_"+key+"_before_1018_update"] = core[key]
        core["source_crossing_evidence_inherited_from_unchanged_surfaces"] = {
            "status":"INVALIDATED_BY_I_COORDINATE_UPDATE",
            "reason":"current I source coordinates changed; fresh source audit is required"}
        core["exact_context"] = {"status":"PENDING_FRESH_806_808_AFTER_I_COORDINATE_UPDATE"}
        core["source_audit_status"] = "PENDING_FRESH_806_808"
        core["source_crossing_audit_status"] = "PENDING_FRESH_806_808"
        core["all185_775"] = {"status":"PENDING_FRESH_806_808_AFTER_I_COORDINATE_UPDATE"}
        core["typed_217_admission"] = {"status":"PENDING"}
        core["current_metadata_status"] = "PENDING_CANONICAL_REBIND_AFTER_I_COORDINATE_UPDATE"
        core["physical_admission"] = False
        core["native_steps"] = 0
        core["mass_mapping"] = "NOT_RUN"
        core["construction_status"] = "CURRENT_1018_POINT_UPDATE_CANDIDATE_PENDING_FRESH_SOURCE_AUDIT"
        core["quality_point_update_1018"] = {
            "schema": SCHEMA, "status":"PASS_INDEPENDENT_POINT_UPDATE_REPLAY_QUALITY_ONLY",
            "input_state":{"path":str(state_path),"sha256":state_digest},
            "fit_report":pin(fit_path),"fit_producer":{"path":str(EXPECTED_PRODUCER_PATH),"sha256":fit["producer_sha256"]},
            "source_variables_declared":counts["declared_source_variables"],
            "source_points_actually_moved":counts["actual_moved_source_points"],
            "interior_volume_points_actually_moved":counts["actual_moved_interior_points"],
            "max_actual_I_move_m":counts["max_actual_source_move_m"],
            "max_actual_interior_move_m":counts["max_actual_interior_move_m"],
            "construction_policy":{"area_ratio_bounds":policy["area_ratio_bounds"],
                                   "normal_cosine_min":policy["normal_cosine_min"],
                                   "source_cap_report_m":policy["source_position_cap_m"],
                                   "adapter_source_cap_m":SOURCE_MOVE_CAP_M},
            "source_face_screen":source_checks,
            "fullstar":fullstar,"full_body_det6_float":full_det,
            "ancestry":{"absolute_source_face_ancestry_preserved":True,
                        "absolute_volume_parent_unions_preserved":True,
                        "coordinate_only_immediate_maps_rebound":True,
                        "source_and_volume_parent_aliases_rebound":True},
            "full_quality_replay":replay,
            "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN",
            "source_audit":"PENDING_FRESH_806_808"
        }
        if wrapper:
            new_state = copy.deepcopy(raw)
            new_state[wrapper] = core
        else:
            new_state = core
        state_stage = stage / "verified-state.json"
        state_stage.write_text(json.dumps(new_state, indent=2, sort_keys=True)+"\n")
        report_out = {"schema":SCHEMA,"status":"PASS_INDEPENDENT_POINT_UPDATE_REPLAY_QUALITY_ONLY",
                      "producer":pin(Path(__file__)),"input_state":{"path":str(state_path),"sha256":state_digest},
                      "fit_report":pin(fit_path),"fit_producer":{"path":str(EXPECTED_PRODUCER_PATH),"sha256":fit["producer_sha256"]},
                      "preflight_report":pin(preflight_path),
                      "local_bundle_preparation_1016":preflight["local_bundle_preparation_1016"],
                      "inputs":{"I":assets["I"],"Skin":assets["Skin"],"Fat":assets["Fat"],"O":assets["O"],"D":assets["D"]},
                      "source_face_screen":source_checks,"update_counts":counts,
                      "movement":movement,"fullstar":fullstar,"full_body_det6_float":full_det,
                      "source_I_fields_changed":i_changed,"volume_fields_changed":v_changed,
                      "independent_replay":replay,
                      "verified_state":{"path":str(paths["I"].parent/"verified-state.json"),"sha256":sha(state_stage)},
                      "source_audit":"PENDING_FRESH_806_808","typed_217_admission":"PENDING",
                      "physical_admission":False,"native_steps":0,"mass_mapping":"NOT_RUN"}
        i_receipt["path"] = str(paths["I"].resolve())
        for role in ("Skin","Fat"):
            v_receipts[role]["path"] = str(paths[role].resolve())
        report_out["outputs"] = {"I":i_receipt,"Skin":v_receipts["Skin"],"Fat":v_receipts["Fat"]}
        (stage/"point-update-1018-report.json").write_text(json.dumps(report_out,indent=2,sort_keys=True)+"\n")
        os.replace(stage, output_dir)
        final_report_path = output_dir/"point-update-1018-report.json"
        final_state_path = output_dir/"verified-state.json"
        final_report = json.loads(final_report_path.read_text())
        for role, fname in (("I","candidate-I-point-updated.npz"),("Skin","skin-volume-point-updated.npz"),("Fat","fat-volume-point-updated.npz")):
            pp=output_dir/fname
            if not pp.is_file() or sha(pp) != final_report["outputs"][role]["sha256"]:
                raise ValueError("post-rename output pin failed: " + role)
            reread=mod.load_npz(pp); planned=iout if role=="I" else vout[role]
            bad=[k for k in planned if not mod.same_array(planned[k],reread[k])]
            if bad: raise ValueError("post-rename payload mismatch "+role+": "+",".join(bad[:8]))
        if Path(final_report["verified_state"]["path"]).resolve()!=final_state_path.resolve() or sha(final_state_path)!=final_report["verified_state"]["sha256"]:
            raise ValueError("post-rename state pin failed")
        return {"status":report_out["status"],"output_dir":str(output_dir),
                "report":pin(final_report_path),"verified_state":pin(final_state_path),
                "physical_admission":False,"native_steps":0}
    finally:
        if stage.exists():
            shutil.rmtree(stage,ignore_errors=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--state",type=Path)
    ap.add_argument("--fit-report",type=Path)
    ap.add_argument("--preflight",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--controls-only",action="store_true")
    ap.add_argument("--control-output",type=Path)
    args=ap.parse_args()
    got=controls()
    if args.controls_only:
        if args.control_output:
            args.control_output.parent.mkdir(parents=True,exist_ok=True)
            args.control_output.write_text(json.dumps({"schema":SCHEMA,"controls":got},indent=2,sort_keys=True)+"\n")
        print(json.dumps({"status":"PASS_CONTROLS","controls":got},sort_keys=True))
        return
    if not all((args.state,args.fit_report,args.preflight,args.output)):
        ap.error("--state --fit-report --preflight --output are required")
    mod=load_owner()
    result=verify_and_emit(mod,args.state,args.fit_report,args.preflight,args.output)
    result["controls"]=got
    print(json.dumps(result,sort_keys=True))

if __name__=="__main__":
    main()
