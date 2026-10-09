#!/usr/bin/env python3
"""Exact skin-clearance transfer for same-index native lung successors.

Uses the retained 1080/1115 clearance chain, exact 1116-to-successor native
surface comparisons, and a completed 1120 delta report. It does not launch
native simulation or claim whole-body, continuous-time, or physiology clearance.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import struct
import sys
import time
from pathlib import Path
import numpy as np

E = Path("/Users/n/numi-human-resting-evidence-20261005")
HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
BASE_RUN = E / "final-native-scene-preflight-936/skin-927-lung-1116-viewer-018-v015-attempt1/native-run"
BASE_NHA = E / "native-lung-final-metadata-refresh-1116/final/resting-thorax.nhanatomy"
BASE_NHA_SHA = "1b62f569f218cf8ba45f26c393bfaa4c8ed2ff499ab500c002dbbb006747b8bc"
BASE_INV_SHA = "f656d47332c79af4c9ead541d9dff50e747858695d1f0d1d258eac2f7c982b27"
BASE_META_SHA = "7252b46ab1dcc31aa6f7b4bd9a64e1a2100776fa25d318af50a00435c344961b"
BASE_RECEIPT = E / "final-native-scene-preflight-936/skin-927-lung-1116-viewer-018-v015-attempt1/anatomy/resting-anatomy-receipt.json"
BASE_RECEIPT_SHA = "f3753de9c40037011d5233a9a02b55cd434f18f29c62b49a4487b490a92b4293"
BASE_SCENE_MANIFEST = E / "final-native-scene-preflight-936/skin-927-lung-1116-viewer-018-v015-attempt1/resting-scene/resting-supine-scene.manifest.json"
BASE_SCENE_MANIFEST_SHA = "e31d42700ffcc0054b95b929419c397d57a8331bbb755e01030b8aa964ca8ed4"
SKIN = E / "native-common-skin-multipose-clearance-candidate-927/asset-candidate-001/bodyparts3d-myosim-skinned-shell.nhskin"
SKIN_SHA = "bd4bfbbf0e071e24a1bb9eea7b9cd9f34ef862ef20417b5891cd24cec1b009d1"
PRIOR_1080 = E / "native-conditioned-skin-geometry-audit-1080/summary.json"
PRIOR_1080_SHA = "4fb87b0aac2e1d8da3936f79947c1d654cb7ff6c2550946ceb4015423772b82e"
V8_SKIN_REPORT = E / "native-lung-v8-integrity-review-1115/skin-transfer-rerun.json"
V8_SKIN_REPORT_SHA = "04662cf09c83f3be0a4ee2d1f1b74fdb1be77cc24996212030815d911268fb6c"
V8_GEOMETRY_REPORT = E / "native-lung-v8-integrity-review-1115/native-1113-vs-1116-geometry.json"
V8_GEOMETRY_REPORT_SHA = "6b8f6d759af09a74527ee66c86bccc963e6a47ee3d031b07320c519f0465f2d7"
V8_NATIVE_COMPARE = E / "native-lung-v8-integrity-review-1115/native-compare.json"
V8_NATIVE_COMPARE_SHA = "74623574a20e9174bb04e2540a80e63773cf328ab37b2fc0e5d91fda88b50c9c"
V8_SKIN_RUNNER = E / "native-lung-v8-integrity-review-1115/rerun_skin_transfer_exact.py"
V8_SKIN_RUNNER_SHA = "01cb96b22bd3d0d29fa438e6b22d24b34856e48fbcb0f6f165fd4a79723f65c4"
V8_COMPARE_HELPER = HERE / "compare_native.py"
V8_COMPARE_HELPER_SHA = "32f1818241866b6d08d1b559d9f32ed250408ecf09232f86c5c0332abc67247c"
V8_NHA_READER = HERE / "review_v8.py"
V8_NHA_READER_SHA = "d0e74eb333eba379b65394dc4c7b448d6c55116e459d61600e5eb11b789b5431"
V8_PAIR_COMPARE_RUNNER = E / "native-lung-v8-integrity-review-1115/compare_native_1113_1116.py"
V8_PAIR_COMPARE_RUNNER_SHA = "cb9da01d3f136a97139fb3d16e32b7c977c82dcf79f8da41f02f25bb71e0844a"
PREDICATE = REPO / "src/numilab_human/cardiac_cavity_intersections.py"
PREDICATE_SHA = "11f042cc4809a27735c02b66a8fb07ebc92ee66597d7e1814a75231e2cf2d2bb"
DELTA_READER = HERE / "audit_delta_1120.py"
DELTA_READER_SHA = "74ef7863aece7dc2ab121458102ce04db435cc9f3bd7c6aa6d3531726e0465bb"
STEPS = [0, 4991, 5375, 5759, 6111, 6495, 7743, 10000]
TARGET_ROWS = (305, 306, 307, 308, 309, 311)
IDENTITY_ROWS = (305, 306, 307, 308, 309, 310, 311)
ALLOWED_CHANGED_ROWS = {305, 306, 307, 308, 310}
SKIN_KEY = (51007, 1)
TARGET_KEYS = {(51023, sid) for sid in range(305, 310)} | {(51010, 311), (51024, 310)}
EXPECTED_SKIN_TRIANGLES = 109211
EXPECTED_ROW310_FACES = 303656


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_hash(path: Path | str, expected: str, label: str) -> Path:
    p = Path(path).resolve()
    if not p.is_file() or sha(p) != expected:
        raise ValueError(label + " hash mismatch: " + str(p))
    return p


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned module " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_baseline_chain():
    pins = (
        (BASE_RUN / "invocation.json", BASE_INV_SHA, "1116 invocation"),
        (BASE_RUN / "run-metadata.json", BASE_META_SHA, "1116 run metadata"),
        (BASE_NHA, BASE_NHA_SHA, "1116 NHA"),
        (BASE_RECEIPT, BASE_RECEIPT_SHA, "1116 anatomy receipt"),
        (BASE_SCENE_MANIFEST, BASE_SCENE_MANIFEST_SHA, "1116 scene manifest"),
        (SKIN, SKIN_SHA, "skin payload"),
        (PRIOR_1080, PRIOR_1080_SHA, "1080 clearance summary"),
        (V8_SKIN_REPORT, V8_SKIN_REPORT_SHA, "1115 changed-face skin scan"),
        (V8_GEOMETRY_REPORT, V8_GEOMETRY_REPORT_SHA, "1113-to-1116 geometry comparison"),
        (V8_NATIVE_COMPARE, V8_NATIVE_COMPARE_SHA, "1078-to-1113 comparison"),
        (V8_SKIN_RUNNER, V8_SKIN_RUNNER_SHA, "1115 skin scan runner"),
        (V8_COMPARE_HELPER, V8_COMPARE_HELPER_SHA, "1115 pack reader"),
        (V8_NHA_READER, V8_NHA_READER_SHA, "1115 NHA reader"),
        (V8_PAIR_COMPARE_RUNNER, V8_PAIR_COMPARE_RUNNER_SHA, "1113-to-1116 comparison runner"),
        (PREDICATE, PREDICATE_SHA, "exact intersection predicate"),
        (DELTA_READER, DELTA_READER_SHA, "1120 delta reader"),
    )
    for path, digest, label in pins:
        require_hash(path, digest, label)

    prior = json.loads(PRIOR_1080.read_text())
    if not all(prior.get(k) is True for k in (
        "all_requested_steps_complete", "all_target_and_skin_self_pair_coverage_complete",
        "intersection_free",
    )):
        raise ValueError("1080 full baseline skin/self/target coverage is not a pass")
    if prior.get("skin_payload_sha256") != SKIN_SHA or len(prior.get("steps", [])) != 8:
        raise ValueError("1080 baseline does not bind the pinned skin and eight accepted poses")

    v8skin = json.loads(V8_SKIN_REPORT.read_text())
    if (v8skin.get("passed") is not True or len(v8skin.get("steps", [])) != 8
            or v8skin.get("row310", {}).get("faces") != EXPECTED_ROW310_FACES
            or v8skin.get("row310", {}).get("exact_same_winding_xyz_at_all_steps") is not True
            or v8skin.get("row310", {}).get("mismatches") != 0):
        raise ValueError("1115 changed-face skin transfer report is incomplete")
    if v8skin.get("inputs", {}).get("skin", {}).get("sha256") != SKIN_SHA:
        raise ValueError("1115 changed-face scan uses another skin")
    for pose in v8skin["steps"]:
        if (pose.get("step") not in STEPS or pose.get("skin_triangle_count") != EXPECTED_SKIN_TRIANGLES
                or pose.get("exact_crossing_triangle_pairs") != 0
                or pose.get("changed_target_triangles") != 16):
            raise ValueError("1115 changed-face skin clearance evidence does not pass every step")

    native_compare = json.loads(V8_NATIVE_COMPARE.read_text())
    if native_compare.get("passed") is not True:
        raise ValueError("1078-to-1113 native comparison is not passed")

    geometry = json.loads(V8_GEOMETRY_REPORT.read_text())
    if geometry.get("passed") is not True or len(geometry.get("steps", [])) != 8:
        raise ValueError("1113-to-1116 complete capture comparison is incomplete")
    for pose in geometry["steps"]:
        if (pose.get("step_geometry_pass") is not True or pose.get("state_fields_equal") is not True
                or pose.get("primitive_count_1113") != 861 or pose.get("primitive_count_1116") != 861
                or pose.get("global_xyz_diff_vertices") != 0
                or pose.get("global_full_xyz_stream_equal") is not True
                or pose.get("full_index_section_sha256_equal") is not True
                or pose.get("normalized_local_indices_equal") is not True):
            raise ValueError("1113-to-1116 exact surface/state bridge failed")
    if [p["step"] for p in geometry["steps"]] != STEPS:
        raise ValueError("1113-to-1116 accepted steps differ from current cadence")

    base_inv = json.loads((BASE_RUN / "invocation.json").read_text())
    base_asset_sha = {str(Path(k).resolve()): v for k, v in base_inv.get("asset_sha256", {}).items()}
    skin_path = str(SKIN.resolve())
    if base_asset_sha.get(skin_path) != SKIN_SHA:
        raise ValueError("1116 invocation does not bind the selected skin payload")
    skin_args = [i for i, token in enumerate(base_inv.get("argv", [])) if token == "--skin-payload"]
    if len(skin_args) != 1 or Path(base_inv["argv"][skin_args[0] + 1]).resolve() != SKIN.resolve():
        raise ValueError("1116 invocation does not select the selected skin payload")
    base_meta = json.loads((BASE_RUN / "run-metadata.json").read_text())
    if base_meta.get("exit_code") != 0 or not base_meta.get("loaded_metal_runtime", {}).get("verified", False):
        raise ValueError("1116 baseline run was not clean with a verified runtime")
    receipt = json.loads(BASE_RECEIPT.read_text())
    if (Path(receipt.get("payload", {}).get("path", "")).resolve() != BASE_NHA.resolve()
            or receipt.get("payload", {}).get("sha256") != BASE_NHA_SHA):
        raise ValueError("1116 receipt does not bind the pinned NHA")
    return geometry, {"prior_1080": prior, "v8_skin": v8skin, "v8_geometry": geometry,
                      "v8_native_compare": native_compare, "base_invocation": base_inv,
                      "base_metadata": base_meta}


def validate_delta_document(delta, *, delta_path, delta_sha, candidate_run, candidate_nha, candidate_nha_sha):
    if sha(delta_path) != delta_sha:
        raise ValueError("1120 delta report hash mismatch")
    if delta.get("schema") != "numi.human.native-transformed-lung-cycle-delta-audit.summary.v1":
        raise ValueError("unsupported 1120 delta report schema")
    if delta.get("status") not in ("complete_delta_scan_no_unclassified_hits",
                                   "complete_delta_scan_with_geometry_failures"):
        raise ValueError("1120 report is not a completed delta scan")
    if Path(delta.get("candidate_run", "")).resolve() != candidate_run.resolve():
        raise ValueError("1120 report is for a different candidate run")
    current = delta.get("candidate_nha", {})
    if (Path(current.get("path", "")).resolve() != candidate_nha.resolve()
            or current.get("sha256") != candidate_nha_sha or sha(candidate_nha) != candidate_nha_sha):
        raise ValueError("1120 report is not bound to the exact candidate NHA")
    if (delta.get("requested_roots") != STEPS[-1] or delta.get("accepted_steps") != STEPS
            or delta.get("terminal_step_included") is not True):
        raise ValueError("1120 report lacks the complete accepted capture horizon")
    if (delta.get("complete_pair_coverage") is not True
            or delta.get("all_6_self_and_15_cross_per_pose") is not True
            or delta.get("topology_verified_each_pose") is not True
            or delta.get("input_hashes_unchanged") is not True
            or delta.get("worker_failures") != []):
        raise ValueError("1120 delta scan is incomplete or its inputs changed")
    if delta.get("candidate_area_config_binding_status") != "PASS_exact_geometry_and_config_binding":
        raise ValueError("1120 candidate geometry/config area binding is not strict-pass")
    identity = delta.get("run_identity_comparison", {})
    for key in ("argv_equal_except_declared_paths", "environment_equal",
                "loaded_runtime_same_verified", "normalized_respiration_configuration_equal"):
        if identity.get(key) is not True:
            raise ValueError("1120 runtime/physiology identity check missing: " + key)
    poses = delta.get("pose_results", [])
    if len(poses) != 8 or [int(p.get("accepted_step", -1)) for p in poses] != STEPS:
        raise ValueError("1120 report pose list incomplete or out of order")
    for pose in poses:
        proof = pose.get("derived_pleura_native_copy", {})
        if (proof.get("accepted_step") != pose["accepted_step"]
                or proof.get("pack_sha256") != pose.get("pack_sha256")
                or proof.get("lineage_face_count") != EXPECTED_ROW310_FACES
                or proof.get("exact_native_same_winding_copied_faces") != EXPECTED_ROW310_FACES
                or proof.get("same_winding_mismatches") != 0
                or proof.get("maximum_abs_coordinate_delta_m") != 0.0
                or proof.get("native_all_524_row_mapping") is not True):
            raise ValueError("1120 does not prove current row310 exact copy at step " + str(pose.get("accepted_step")))
    # Rehash the exact owner inputs after the 1120 scan; transfer only remains
    # valid while that scan's input set is unchanged.
    hashes = delta.get("input_hashes_before", {})
    if not hashes:
        raise ValueError("1120 report omits its tracked input hash set")
    for path, digest in hashes.items():
        p = Path(path)
        if not p.is_file() or sha(p) != digest:
            raise ValueError("1120 tracked input changed after audit: " + str(p))
    return {int(p["accepted_step"]): p for p in poses}


def source_change_support(base_nha, candidate_nha):
    if (base_nha.nrows, base_nha.nv, base_nha.ni) != (candidate_nha.nrows, candidate_nha.nv, candidate_nha.ni):
        raise ValueError("NHA global index-space dimensions changed")
    if set(base_nha.rows) != set(candidate_nha.rows) or len(candidate_nha.rows) != 524:
        raise ValueError("NHA stable-ID rows changed")
    rows = {}
    changed_source_rows = set()
    for sid in sorted(base_nha.rows):
        a, b = base_nha.rows[sid], candidate_nha.rows[sid]
        for key in ("body", "layer", "flags", "fv", "nv", "fi", "ni"):
            if a[key] != b[key]:
                raise ValueError("NHA row identity/index-space field changed row %d: %s" % (sid, key))
        va = np.frombuffer(base_nha.vertices_raw(sid), dtype="<f4").reshape(-1, 6)
        vb = np.frombuffer(candidate_nha.vertices_raw(sid), dtype="<f4").reshape(-1, 6)
        if va.shape != vb.shape:
            raise ValueError("NHA vertex shape changed row " + str(sid))
        xyz_a = np.ascontiguousarray(va[:, :3]).view("<u4").reshape(-1, 3)
        xyz_b = np.ascontiguousarray(vb[:, :3]).view("<u4").reshape(-1, 3)
        moved = set(np.flatnonzero(np.any(xyz_a != xyz_b, axis=1)).tolist())
        normal_a = np.ascontiguousarray(va[:, 3:]).view("<u4").reshape(-1, 3)
        normal_b = np.ascontiguousarray(vb[:, 3:]).view("<u4").reshape(-1, 3)
        normal_changed = set(np.flatnonzero(np.any(normal_a != normal_b, axis=1)).tolist())
        fa = np.asarray(base_nha.faces(sid), dtype=np.int64).reshape(-1, 3)
        fb = np.asarray(candidate_nha.faces(sid), dtype=np.int64).reshape(-1, 3)
        if fa.shape != fb.shape:
            raise ValueError("NHA face row count changed row " + str(sid))
        face_index_changed = set(np.flatnonzero(np.any(fa != fb, axis=1)).tolist())
        if moved or normal_changed or face_index_changed:
            changed_source_rows.add(sid)
        if sid not in set(range(305, 309)) | {310} and (moved or normal_changed or face_index_changed):
            raise ValueError("source edit escaped the declared 305-308+310 support: row " + str(sid))
        if sid in TARGET_ROWS:
            incident = set(map(int, fa.reshape(-1))) | set(map(int, fb.reshape(-1)))
            if moved - incident:
                raise ValueError("moved source vertex is unused in target row %d: %s" % (sid, sorted(moved - incident)[:10]))
            affected = set(face_index_changed)
            if moved:
                for face_row in range(len(fa)):
                    if moved.intersection(fa[face_row].tolist()) or moved.intersection(fb[face_row].tolist()):
                        affected.add(face_row)
            rows[sid] = {
                "moved_xyz_vertex_ids": sorted(moved),
                "changed_normal_vertex_ids": sorted(normal_changed),
                "face_index_changed_rows": sorted(face_index_changed),
                "source_affected_face_rows": sorted(affected),
            }
    changed_target_rows = [sid for sid in sorted(changed_source_rows) if sid in TARGET_ROWS]
    forbidden = sorted(changed_source_rows - ALLOWED_CHANGED_ROWS)
    if forbidden:
        raise ValueError("changed source rows exceed this transfer contract: " + str(forbidden))
    return {"by_row": rows, "changed_source_rows": sorted(changed_source_rows),
            "changed_lobe_or_diaphragm_rows": changed_target_rows}


def _triangle_identity(pack, surface, face_row):
    ids = tuple(int(x) for x in surface["norm"][face_row * 3:face_row * 3 + 3])
    xyz = tuple(pack.xyz(surface["start"] + idx) for idx in ids)
    return ids, xyz


def changed_face_rows_between_packs(base_pack, candidate_pack, sid, key, face_count):
    a, b = base_pack.surfaces[key], candidate_pack.surfaces[key]
    if a["count"] != face_count * 3 or b["count"] != face_count * 3:
        raise ValueError("native target surface face count differs row %d" % sid)
    changed = set()
    unchanged = 0
    for face_row in range(face_count):
        if _triangle_identity(base_pack, a, face_row) == _triangle_identity(candidate_pack, b, face_row):
            unchanged += 1
        else:
            changed.add(face_row)
    return changed, unchanged


def validate_target_surface_identities(base_pack, candidate_pack, base_nha, candidate_nha):
    """Require stable native primitive identity while allowing declared geometry edits."""
    checked = {}
    for sid in sorted(IDENTITY_ROWS):
        key = (51010, sid) if sid == 311 else ((51024, sid) if sid == 310 else (51023, sid))
        if key not in base_pack.surfaces or key not in candidate_pack.surfaces:
            raise ValueError("target native primitive missing: " + str(key))
        before = base_pack.surfaces[key]
        after = candidate_pack.surfaces[key]
        identity_fields = ("body", "semantic", "instance", "owner", "sid", "primitive_id")
        identity_before = tuple(before[name] for name in identity_fields)
        identity_after = tuple(after[name] for name in identity_fields)
        if identity_before != identity_after:
            raise ValueError("target native primitive identity changed: " + str(key))
        if before["count"] != after["count"]:
            raise ValueError("target native primitive index count changed: " + str(key))
        source_count_before = base_nha.rows[sid]["ni"]
        source_count_after = candidate_nha.rows[sid]["ni"]
        if before["count"] != source_count_before or after["count"] != source_count_after:
            raise ValueError("target native primitive count does not match bound NHA row: " + str(key))
        checked[str(sid)] = {"key": list(key), "identity_fields": list(identity_fields),
                             "identity_equal_1116": True, "index_count_equal_source_rows": True,
                             "index_count": after["count"]}
    return checked


def validate_pack_surface_equivalence(base_pack, candidate_pack, compare_native,
                                      base_nha, candidate_nha):
    if set(base_pack.surfaces) != set(candidate_pack.surfaces):
        raise ValueError("native surface catalogue differs from 1116")
    if len(base_pack.surfaces) != 861 or len(candidate_pack.surfaces) != 861:
        raise ValueError("expected 861 registered native surfaces")
    target_identity = validate_target_surface_identities(
        base_pack, candidate_pack, base_nha, candidate_nha)
    changed_keys = TARGET_KEYS
    unchanged = compare_native.compare_unchanged(base_pack, candidate_pack, changed_keys)
    if unchanged.get("primitive_count") != 861 - len(changed_keys):
        raise ValueError("untouched native surface comparison was incomplete")
    unchanged["target_identity_checks"] = target_identity
    return unchanged


def check_skin_stream_equal(base_pack, candidate_pack):
    a = base_pack.surfaces.get(SKIN_KEY)
    b = candidate_pack.surfaces.get(SKIN_KEY)
    if a is None or b is None:
        raise ValueError("captured skin surface 51007/1 is absent")
    if (a["count"], a["norm"], a["used"]) != (b["count"], b["norm"], b["used"]):
        raise ValueError("candidate captured skin index stream differs from 1116")
    if a["count"] // 3 != EXPECTED_SKIN_TRIANGLES:
        raise ValueError("captured skin triangle count differs from pinned source")
    for local in a["used"]:
        if base_pack.xyz(a["start"] + local) != candidate_pack.xyz(b["start"] + local):
            raise ValueError("captured skin XYZ differs from 1116 at local vertex " + str(local))
    return {"primitive_key": list(SKIN_KEY), "triangle_count": a["count"] // 3,
            "exact_local_index_stream_match": True, "exact_used_vertex_xyz_match": True,
            "used_vertex_count": len(a["used"])}


def build_skin_records(pack, predicate):
    surface = pack.surfaces.get(SKIN_KEY)
    if surface is None:
        raise ValueError("captured skin surface missing")
    ids = sorted(set(surface["norm"]))
    local = {global_local: i for i, global_local in enumerate(ids)}
    faces = np.asarray([local[x] for x in surface["norm"]], dtype=np.int64).reshape(-1, 3)
    points = []
    for local_id in ids:
        xyz = pack.xyz_floats(surface["start"] + local_id)
        if not all(math.isfinite(float(x)) for x in xyz):
            raise ValueError("captured skin has nonfinite XYZ")
        points.append(predicate.float32_point_lattice_key(xyz))
    records = predicate._records(points, faces)
    if len(records) != EXPECTED_SKIN_TRIANGLES:
        raise ValueError("skin record count differs from pinned triangle count")
    return records


def scan_changed_faces_against_skin(pack, predicate, skin_records, target_faces):
    if not target_faces:
        return {"changed_target_triangle_count": 0, "aabb_candidate_pairs": 0,
                "exact_crossing_triangle_pairs": 0, "crossings": []}
    points = []
    faces = []
    source_map = []
    cursor = 0
    for sid, face_row in target_faces:
        key = (51010, sid) if sid == 311 else (51023, sid)
        surface = pack.surfaces.get(key)
        if surface is None:
            raise ValueError("changed target native surface missing: " + str(key))
        ids = tuple(int(x) for x in surface["norm"][face_row * 3:face_row * 3 + 3])
        if len(ids) != 3:
            raise ValueError("target face row is incomplete")
        tri_points = []
        for local_id in ids:
            xyz = pack.xyz_floats(surface["start"] + local_id)
            if not all(math.isfinite(float(x)) for x in xyz):
                raise ValueError("changed target face has nonfinite XYZ")
            tri_points.append(predicate.float32_point_lattice_key(xyz))
        points.extend(tri_points)
        faces.append((cursor, cursor + 1, cursor + 2))
        cursor += 3
        source_map.append((sid, face_row))
    target_records = predicate._records(points, np.asarray(faces, dtype=np.int64))
    audit = predicate._audit_pair(skin_records, target_records, same_surface=False)
    crossings = []
    for skin_row, target_row in audit["triangle_pairs"]:
        sr = skin_records[int(skin_row)]
        tr = target_records[int(target_row)]
        sid, face_row = source_map[int(tr[3])]
        crossings.append({
            "skin_face_row": int(sr[3]),
            "target_stable_id": sid,
            "target_face_row": face_row,
            "intersection_points_lattice": predicate.triangle_intersection_points(sr[0], tr[0]),
        })
    return {"changed_target_triangle_count": len(target_records),
            "aabb_candidate_pairs": int(audit["aabb_candidate_pairs"]),
            "exact_crossing_triangle_pairs": int(audit["count"]), "crossings": crossings}


def validate_input_invocation(run_path, candidate_nha, candidate_nha_sha):
    invocation_path = run_path / "invocation.json"
    metadata_path = run_path / "run-metadata.json"
    if not invocation_path.is_file() or not metadata_path.is_file():
        raise ValueError("candidate native run lacks invocation/run-metadata")
    invocation = json.loads(invocation_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    argv = invocation.get("argv", [])
    for flag, want in (("--torso-anatomy-payload", candidate_nha), ("--skin-payload", SKIN)):
        indices = [i for i, value in enumerate(argv) if value == flag]
        if len(indices) != 1 or indices[0] + 1 >= len(argv):
            raise ValueError("candidate invocation lacks unique " + flag)
        if Path(argv[indices[0] + 1]).resolve() != Path(want).resolve():
            raise ValueError("candidate invocation selects another " + flag)
    assets = {str(Path(k).resolve()): v for k, v in invocation.get("asset_sha256", {}).items()}
    if assets.get(str(candidate_nha.resolve())) != candidate_nha_sha:
        raise ValueError("candidate invocation does not hash-bind current NHA")
    if assets.get(str(SKIN.resolve())) != SKIN_SHA or sha(SKIN) != SKIN_SHA:
        raise ValueError("candidate invocation does not hash-bind pinned skin")
    receipt_indices = [i for i, value in enumerate(argv) if value == "--resting-anatomy-receipt"]
    if len(receipt_indices) != 1 or receipt_indices[0] + 1 >= len(argv):
        raise ValueError("candidate invocation lacks unique anatomy receipt argument")
    receipt_path = Path(argv[receipt_indices[0] + 1]).resolve()
    if assets.get(str(receipt_path)) != sha(receipt_path):
        raise ValueError("candidate invocation does not hash-bind its current anatomy receipt")
    receipt = json.loads(receipt_path.read_text())
    if (Path(receipt.get("payload", {}).get("path", "")).resolve() != candidate_nha.resolve()
            or receipt.get("payload", {}).get("sha256") != candidate_nha_sha):
        raise ValueError("candidate anatomy receipt does not bind candidate NHA")
    if metadata.get("exit_code") != 0 or not metadata.get("loaded_metal_runtime", {}).get("verified", False):
        raise ValueError("candidate native run did not exit successfully with a verified runtime")
    if metadata.get("source_files_changed_during_run") not in (None, [], False):
        raise ValueError("candidate native sources changed during run")
    return {"invocation_path": str(invocation_path), "invocation_sha256": sha(invocation_path),
            "metadata_path": str(metadata_path), "metadata_sha256": sha(metadata_path),
            "anatomy_receipt_path": str(receipt_path), "anatomy_receipt_sha256": sha(receipt_path),
            "skin_asset_sha256": assets[str(SKIN.resolve())],
            "loaded_runtime_verified": True, "native_exit_code": 0}


def run(args):
    if args.out.exists() or not args.out.resolve().is_relative_to(E):
        raise ValueError("output must be a fresh path under the evidence root")
    candidate_run = args.candidate_run.resolve()
    candidate_nha = args.candidate_nha.resolve()
    if sha(candidate_nha) != args.candidate_nha_sha256:
        raise ValueError("candidate NHA hash mismatch")
    baseline_geometry, upstream = validate_baseline_chain()
    delta = json.loads(args.delta_report.read_text())
    delta_poses = validate_delta_document(
        delta, delta_path=args.delta_report.resolve(), delta_sha=args.delta_report_sha256,
        candidate_run=candidate_run, candidate_nha=candidate_nha,
        candidate_nha_sha=args.candidate_nha_sha256)
    current_run_provenance = validate_input_invocation(candidate_run, candidate_nha, args.candidate_nha_sha256)

    review = load_module(V8_NHA_READER, "skin_delta_review_v8")
    compare = load_module(V8_COMPARE_HELPER, "skin_delta_compare_native")
    predicate = load_module(PREDICATE, "skin_delta_exact_predicate")
    base_nha = review.NHA(BASE_NHA)
    candidate_nha_doc = review.NHA(candidate_nha)
    source_delta = source_change_support(base_nha, candidate_nha_doc)
    if not source_delta["changed_source_rows"] or not (set(source_delta["changed_source_rows"]) & {305, 306, 307, 308}):
        raise ValueError("candidate has no declared lung geometry change to transfer")
    if set(source_delta["changed_source_rows"]) - ALLOWED_CHANGED_ROWS:
        raise ValueError("candidate source edits escape supported free-apex rows")
    base_invocation = current_run_provenance
    started = time.monotonic()
    poses = []
    base_index_sha = None
    candidate_index_sha = None
    try:
        for step in STEPS:
            base_dir = BASE_RUN / "accepted-geometry"
            candidate_dir = candidate_run / "accepted-geometry"
            base_pack_path = base_dir / ("step-%d.mrvpack" % step)
            base_receipt_path = base_dir / ("step-%d.receipt.json" % step)
            candidate_pack_path = candidate_dir / ("step-%d.mrvpack" % step)
            candidate_receipt_path = candidate_dir / ("step-%d.receipt.json" % step)
            base_pack = compare.Pack(base_pack_path, base_receipt_path, step)
            candidate_pack = compare.Pack(candidate_pack_path, candidate_receipt_path, step)
            try:
                base_state = base_pack.receipt
                candidate_state = candidate_pack.receipt
                for field in ("accepted_body_state_sha256", "accepted_respiration_state_sha256", "accepted_time_s"):
                    if base_state.get(field) != candidate_state.get(field):
                        raise ValueError("candidate accepted state differs from 1116 at step %d: %s" % (step, field))
                delta_pose = delta_poses[step]
                if delta_pose.get("pack_sha256") != candidate_pack.pack_sha:
                    raise ValueError("1120 report candidate pack does not match skin transfer input at step %d" % step)
                pleura_proof = delta_pose["derived_pleura_native_copy"]
                if pleura_proof.get("pack_sha256") != candidate_pack.pack_sha:
                    raise ValueError("row310 proof references another current pack")
                if base_index_sha is None:
                    base_index_sha = base_pack.index_sha
                    candidate_index_sha = candidate_pack.index_sha
                    base_rows = compare.source_nha_topology_check(base_nha, base_pack)
                    candidate_rows = compare.source_nha_topology_check(candidate_nha_doc, candidate_pack)
                    if len(base_rows) != 524 or len(candidate_rows) != 524:
                        raise ValueError("not all current 524 NHA rows map to native packs")
                if base_pack.index_sha != base_index_sha or candidate_pack.index_sha != candidate_index_sha:
                    raise ValueError("within-run native index section changed across captures")

                untouched = validate_pack_surface_equivalence(base_pack, candidate_pack, compare, base_nha, candidate_nha_doc)
                skin_stream = check_skin_stream_equal(base_pack, candidate_pack)
                skin_records = build_skin_records(candidate_pack, predicate)
                changed_faces = {}
                unchanged_face_counts = {}
                for sid in TARGET_ROWS:
                    key = (51010, sid) if sid == 311 else (51023, sid)
                    face_count = candidate_nha_doc.rows[sid]["ni"] // 3
                    got_changed, unchanged_count = changed_face_rows_between_packs(
                        base_pack, candidate_pack, sid, key, face_count)
                    expected_native = set(delta_pose["native_changed_faces"][str(sid)])
                    if got_changed != expected_native:
                        raise ValueError("captured changed face set disagrees with 1120 at step %d row %d" % (step, sid))
                    source_support = set(source_delta["by_row"][sid]["source_affected_face_rows"])
                    if not source_support.issubset(got_changed):
                        raise ValueError("source-changed face absent from captured changed set at step %d row %d" % (step, sid))
                    changed_faces[sid] = sorted(got_changed)
                    unchanged_face_counts[sid] = unchanged_count
                scanned = [(sid, face) for sid in TARGET_ROWS for face in changed_faces[sid]]
                scan = scan_changed_faces_against_skin(candidate_pack, predicate, skin_records, scanned)
                pose = {
                    "accepted_step": step,
                    "accepted_time_s": candidate_state.get("accepted_time_s"),
                    "accepted_body_state_sha256": candidate_state.get("accepted_body_state_sha256"),
                    "accepted_respiration_state_sha256": candidate_state.get("accepted_respiration_state_sha256"),
                    "baseline_pack_sha256": base_pack.pack_sha,
                    "candidate_pack_sha256": candidate_pack.pack_sha,
                    "baseline_receipt_sha256": sha(base_receipt_path),
                    "candidate_receipt_sha256": sha(candidate_receipt_path),
                    "untouched_native_primitives": untouched,
                    "skin_stream": skin_stream,
                    "target_changed_faces": {str(k): v for k, v in changed_faces.items()},
                    "unchanged_target_face_counts": {str(k): v for k, v in unchanged_face_counts.items()},
                    "delta_audit_row310_copy_proof": pleura_proof,
                    "changed_faces_vs_skin": scan,
                }
                poses.append(pose)
                print(json.dumps({"step": step, "unchanged_primitives": untouched["primitive_count"],
                                  "skin_triangles": skin_stream["triangle_count"],
                                  "changed_faces": sum(len(x) for x in changed_faces.values()),
                                  "skin_exact_hits": scan["exact_crossing_triangle_pairs"]}), flush=True)
            finally:
                base_pack.close()
                candidate_pack.close()
        if [p["accepted_step"] for p in poses] != STEPS:
            raise ValueError("not all eight accepted poses were audited")
        skin_hits = sum(p["changed_faces_vs_skin"]["exact_crossing_triangle_pairs"] for p in poses)
        complete = len(poses) == len(STEPS)
        no_skin_hits = skin_hits == 0
        out = args.out.resolve()
        out.mkdir(parents=True)
        report = {
            "schema": "numi.human.native-lung-skin-clearance-delta-transfer.v1",
            "status": "complete_changed_face_skin_scan_clear" if complete and no_skin_hits else
                      ("complete_changed_face_skin_scan_with_hits" if complete else "incomplete_skin_transfer"),
            "candidate_run": str(candidate_run),
            "candidate_nha": {"path": str(candidate_nha), "sha256": args.candidate_nha_sha256},
            "candidate_1120_delta_report": {"path": str(args.delta_report.resolve()), "sha256": args.delta_report_sha256,
                                              "status": delta["status"],
                                              "area_binding_status": delta["candidate_area_config_binding_status"]},
            "baseline_1116": {"run": str(BASE_RUN), "nha_path": str(BASE_NHA), "nha_sha256": BASE_NHA_SHA,
                              "skin_sha256": SKIN_SHA},
            "baseline_clearance_transfer_chain": {
                "1080_summary": {"path": str(PRIOR_1080), "sha256": PRIOR_1080_SHA,
                                 "complete": True, "intersection_free": True},
                "1115_changed_face_scan": {"path": str(V8_SKIN_REPORT), "sha256": V8_SKIN_REPORT_SHA,
                                           "changed_faces": 16, "skin_triangles_per_pose": EXPECTED_SKIN_TRIANGLES,
                                           "exact_hits_all_poses": 0},
                "1113_to_1116_full_capture_bridge": {"path": str(V8_GEOMETRY_REPORT),
                                                     "sha256": V8_GEOMETRY_REPORT_SHA,
                                                     "primitive_count_per_pose": 861,
                                                     "full_xyz_and_indices_exact": True,
                                                     "state_exact": True},
            },
            "candidate_invocation_provenance": current_run_provenance,
            "source_edit_support": source_delta,
            "steps": poses,
            "accepted_steps": STEPS,
            "complete_eight_pose_coverage": complete,
            "total_changed_face_skin_intersection_pairs": skin_hits,
            "changed_face_skin_clearance_pass": no_skin_hits,
            "skin_identity_and_untouched_surface_transfer": "exact captured skin XYZ/indexes and every non-target primitive match 1116; unchanged target faces are exact per-face; only changed 305-309/311 faces are rescanned",
            "row310_handling": "No separate row310-versus-skin scan: the pinned 1120 per-pose proof requires all 303656 row310 faces to be exact same-winding native copies of current lobe parents.",
            "limitations": [
                "Accepted captures only; no continuous-time clearance guarantee.",
                "This report evaluates skin-pair transfer only. It does not convert the separate lung self/cross geometry result into a pass.",
                "The 1120 geometry delta report must be independently reviewed; remaining lung-to-lung contacts can still fail overall anatomy qualification.",
                "Not physiology, clinical, endurance, or full-body all-pairs acceptance."
            ],
            "elapsed_seconds": time.monotonic() - started,
        }
        report_path = out / "skin-clearance-delta-transfer.json"
        write_json(report_path, report)
        print(json.dumps({"status": report["status"], "report": str(report_path),
                          "poses": len(poses), "skin_hits": skin_hits}))
        if not no_skin_hits:
            raise SystemExit(2)
    finally:
        base_nha.close()
        candidate_nha_doc.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-run", type=Path, required=True)
    p.add_argument("--candidate-nha", type=Path, required=True)
    p.add_argument("--candidate-nha-sha256", required=True)
    p.add_argument("--delta-report", type=Path, required=True)
    p.add_argument("--delta-report-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
