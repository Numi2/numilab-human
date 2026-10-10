#!/usr/bin/env python3
"""Bounded exact changed-face self-audit prototype."""
from __future__ import annotations
import hashlib, importlib.util, json, sys, time
from pathlib import Path
import numpy as np

R = Path("/Users/n/numi-human-retained-delivery-20261009")
E = Path("/Users/n/numi-human-resting-evidence-20261005")
BASE = R / "skin-resting-multipose-clearance-1218"
OUT = BASE / "incremental-self-prototype-001"
RUN_OUT = OUT / "run-006"
CHECKER = BASE / "local-self-reduction-audit-001/verify_candidate_full_gates_1218_001.py"
CANDIDATE = BASE / "bilateral-source-qp-001/qp-dual-localcheck-018-root/result-001/unadmitted-compact-source.npy"
CANDIDATE_REPORT = CANDIDATE.with_name("local-feasibility.json")
FULL17 = BASE / "local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd/candidate-verification.json"
TARGET_TABLES = BASE / "local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd/candidate-target-pair-tables.json"
STEP = 155000

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def need(value, message):
    if not value:
        raise RuntimeError(message)

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def pairset(row):
    return {tuple(map(int, p)) for p in row["triangle_pairs"]}

def changed_face_self_audit(ci, clearance, baseline_xyz, candidate_xyz, faces):
    base = np.ascontiguousarray(baseline_xyz, dtype="<f4")
    cand = np.ascontiguousarray(candidate_xyz, dtype="<f4")
    tri = np.ascontiguousarray(faces, dtype="<i8")
    need(base.ndim == cand.ndim == 2 and base.shape == cand.shape and base.shape[1] == 3,
         "baseline/candidate coordinates must be same-size Nx3")
    need(tri.ndim == 2 and tri.shape[1] == 3 and tri.size and tri.min() >= 0 and tri.max() < len(base),
         "faces must be valid integer Fx3 over shared vertices")
    need(np.isfinite(base).all() and np.isfinite(cand).all(), "coordinates must be finite Float32")
    changed_vertex_mask = np.any(base.view("<u4") != cand.view("<u4"), axis=1)
    changed_face_rows = np.flatnonzero(np.any(changed_vertex_mask[tri], axis=1)).astype(np.int64)
    all_records = clearance._exact_surface_records(cand, tri)
    changed_records = [all_records[int(row)] for row in changed_face_rows]
    full_index = ci._prepare_surface_aabb(all_records)
    canonical = {}
    raw_occurrences = 0
    for changed_record, full_record in ci._query_surface_aabb(changed_records, full_index):
        left, right = int(changed_record[3]), int(full_record[3])
        if left == right:
            continue
        key = (min(left, right), max(left, right))
        raw_occurrences += 1
        canonical.setdefault(key, (all_records[key[0]], all_records[key[1]]))
    audit = ci._audit_record_pairs((canonical[k] for k in sorted(canonical)), same_surface=True)
    audit["changed_vertex_rows"] = np.flatnonzero(changed_vertex_mask).astype(int).tolist()
    audit["changed_face_rows"] = changed_face_rows.astype(int).tolist()
    audit["canonical_changed_full_aabb_pairs"] = len(canonical)
    audit["raw_changed_full_aabb_occurrences_before_dedup"] = raw_occurrences
    return audit

def fixture_case(ci, clearance, name, base, cand, faces, *, baseline_must_be_clear=True):
    base = np.asarray(base, dtype="<f4")
    cand = np.asarray(cand, dtype="<f4")
    faces = np.asarray(faces, dtype=np.int64)
    base_records = clearance._exact_surface_records(base, faces)
    candidate_records = clearance._exact_surface_records(cand, faces)
    base_full = ci._audit_pair(base_records, base_records, same_surface=True)
    candidate_full = ci._audit_pair(candidate_records, candidate_records, same_surface=True)
    if baseline_must_be_clear:
        need(base_full["count"] == 0, f"{name}: fixture baseline is not self-clear")
    incremental = changed_face_self_audit(ci, clearance, base, cand, faces)
    need(pairset(incremental) == pairset(candidate_full),
         f"{name}: changed-face pairset differs from full owner")
    row = {
        "name": name,
        "baseline_unallowed_pair_count": base_full["count"],
        "candidate_full_unallowed_pair_rows": candidate_full["triangle_pairs"],
        "incremental_unallowed_pair_rows": incremental["triangle_pairs"],
        "candidate_full_allowed_shared_pairs": candidate_full["allowed_shared_vertex_or_edge_pairs"],
        "changed_face_scan_allowed_shared_pairs": incremental["allowed_shared_vertex_or_edge_pairs"],
        "changed_vertex_rows": incremental["changed_vertex_rows"],
        "changed_face_rows": incremental["changed_face_rows"],
        "canonical_aabb_pairs": incremental["canonical_changed_full_aabb_pairs"],
        "raw_aabb_occurrences_before_dedup": incremental["raw_changed_full_aabb_occurrences_before_dedup"],
        "pair_set_equal_to_full_owner": True,
        "baseline_clear_eligible": bool(base_full["count"] == 0),
    }
    return row

def fixture_suite(ci, clearance):
    results = []
    # Changed face row 1 is higher-index than unchanged partner row 0.
    base = np.array([[0,0,0],[1,0,0],[0,1,0],[.2,.1,2],[.8,.1,2],[.4,.8,2]], dtype="<f4")
    cand = base.copy()
    cand[3:6] = np.array([[.3,.1,-1],[.3,.8,1],[.3,.5,-1]], dtype="<f4")
    faces = np.array([[0,1,2],[3,4,5]], dtype=np.int64)
    row = fixture_case(ci, clearance, "changed_higher_face_row_vs_unchanged_lower_partner", base, cand, faces)
    records = clearance._exact_surface_records(cand, faces)
    naive = ci._audit_pair_prepared_first(ci._prepare_surface_aabb([records[1]]), records, same_surface=True)
    row["one_way_prepared_same_surface_count"] = naive["count"]
    row["one_way_wrapper_misses_expected_pair"] = naive["count"] == 0 and row["candidate_full_unallowed_pair_rows"] == [[0,1]]
    need(row["one_way_wrapper_misses_expected_pair"], "known row-order miss fixture did not reproduce")
    results.append(row)

    # Both faces changed; the shared pair is queried twice and canonicalized once.
    base = np.array([[0,0,2],[1,0,2],[0,1,2],[4,0,0],[4,1,0],[4,0,1]], dtype="<f4")
    cand = np.array([[0,0,0],[1,0,0],[0,1,0],[.3,.1,-1],[.3,.8,1],[.3,.5,-1]], dtype="<f4")
    faces = np.array([[0,1,2],[3,4,5]], dtype=np.int64)
    row = fixture_case(ci, clearance, "changed_changed_pair_canonical_dedup", base, cand, faces)
    need(row["raw_aabb_occurrences_before_dedup"] == 2 and row["canonical_aabb_pairs"] == 1,
         "changed-changed pair was not deduplicated")
    results.append(row)

    # Shared edge remains an allowed exact contact.
    base = np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0]], dtype="<f4")
    cand = base.copy(); cand[3] = [0,-1,1]
    faces = np.array([[0,1,2],[1,0,3]], dtype=np.int64)
    row = fixture_case(ci, clearance, "shared_edge_allowed", base, cand, faces)
    need(row["candidate_full_unallowed_pair_rows"] == []
         and row["candidate_full_allowed_shared_pairs"] == 1
         and row["changed_face_scan_allowed_shared_pairs"] == 1,
         "shared-edge semantics differ")
    results.append(row)

    # Shared vertex is allowed only for a point-only intersection.
    base = np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,1],[0,-1,1]], dtype="<f4")
    cand = base.copy(); cand[4] = [0,-2,1]
    faces = np.array([[0,1,2],[0,3,4]], dtype=np.int64)
    row = fixture_case(ci, clearance, "shared_vertex_allowed", base, cand, faces)
    need(row["candidate_full_unallowed_pair_rows"] == []
         and row["candidate_full_allowed_shared_pairs"] == 1
         and row["changed_face_scan_allowed_shared_pairs"] == 1,
         "shared-vertex semantics differ")
    results.append(row)

    # Coordinate bits change while global face rows and index triples remain fixed.
    base = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,2],[1,0,2],[0,1,2]], dtype="<f4")
    cand = base.copy()
    cand[4,0] = np.nextafter(cand[4,0], np.float32(np.inf), dtype=np.float32)
    faces = np.array([[0,1,2],[3,4,5]], dtype=np.int64)
    row = fixture_case(ci, clearance, "vertex_bit_change_with_unchanged_face_order", base, cand, faces)
    need(row["changed_vertex_rows"] == [4] and row["changed_face_rows"] == [1],
         "one-bit vertex change mapped to wrong unchanged face row")
    results.append(row)

    # Three shared IDs must not be treated as allowed adjacency. This is a
    # predicate equivalence case only; the baseline itself is ineligible.
    base = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype="<f4")
    cand = base.copy(); cand[2,1] = np.float32(1.25)
    faces = np.array([[0,1,2],[0,1,2]], dtype=np.int64)
    row = fixture_case(ci, clearance, "duplicate_face_three_shared_ids_rejected", base, cand, faces,
                       baseline_must_be_clear=False)
    need(row["baseline_unallowed_pair_count"] == 1
         and row["candidate_full_unallowed_pair_rows"] == [[0,1]]
         and row["incremental_unallowed_pair_rows"] == [[0,1]],
         "three-shared-ID duplicate face was incorrectly exempted")
    results.append(row)
    return {"schema": "numi.human.changed-face-self-audit-fixtures.v1",
            "fixture_count": len(results), "all_equivalent": True, "cases": results}

def main():
    need(not RUN_OUT.exists(), f"refusing to overwrite {RUN_OUT}")
    RUN_OUT.mkdir(parents=True, exist_ok=False)
    checker = load_module(CHECKER, "full_gates_checker_1218_prototype")
    checker.verify_owner_imports()
    c, ci = checker.c, checker.ci
    need(sha(CHECKER) == "5544f22de077f507b1486a197e29b093f78f3d0a199b7f1bafe0caff86832399",
         "full17 checker source changed")
    need(sha(CANDIDATE) == "d319d85ac3fd74f7bcadfbc54361cda5a956cdf1fc90f9539f8c0c6de59884d1",
         "candidate NPY changed")
    need(sha(CANDIDATE_REPORT) == "16141cf59ff76dd5e797b7f9ac785667ca5d9813b9cf895f9b8470fe2e41288f",
         "candidate report changed")
    candidate_report = checker.load_json(CANDIDATE_REPORT)
    need(candidate_report.get("status") == "local_candidate_ready_for_full17_exact_checks"
         and candidate_report.get("local_exact_recheck", {}).get("self_clear") is True
         and candidate_report.get("local_exact_recheck", {}).get("target_no_new_pairs") is True,
         "candidate lacks successful local exact prechecks")
    full17 = checker.load_json(FULL17)
    need(full17.get("status") == "candidate_target_and_non_target_gate_scan_complete"
         and full17.get("candidate_checkpoint_status") == "partial_target_pairs_non_target_gates_pass"
         and full17.get("candidate_non_target_gates_all_17_pass") is True
         and full17.get("inputs_unchanged_during_scan") is True,
         "full17 reference is not the retained complete partial scan")
    candidate_pose = [r for r in full17["candidate_non_target_gate_checks"] if int(r.get("accepted_step",-1)) == STEP]
    need(len(candidate_pose) == 1 and candidate_pose[0]["exact_skin_self_pair_count"] == 0
         and candidate_pose[0]["exact_skin_self_pair_rows"] == []
         and candidate_pose[0]["exact_skin_degenerate_face_rows"] == [],
         "full17 reference lacks a zero-self terminal result")
    need(sha(TARGET_TABLES) == "25901e7caebcee0429ba1bc3076fc5e132d13a2fff316fd027961bae580cb2f5",
         "full17 target pair table changed")

    t = time.perf_counter()
    fixtures = fixture_suite(ci, c)
    fixture_seconds = time.perf_counter() - t

    raw_skin, skin = checker.load_skin(checker.SKIN)
    checker.CURRENT_SKIN = skin
    checker.REFERENCE_IDS = np.unique(skin["faces"]).astype(np.int64)
    target_keys, _, _, _ = c._load_target_inventory(checker.INV)
    accepted_helper = checker.load_module(checker.VALID, "accepted_mrvpack_prototype")
    pose = checker.read_pose(("1218_current_run", checker.CURRENT_RUN, STEP),
                             set(target_keys), accepted_helper, None, None, None)
    receipt = pose["receipt"]
    need(receipt["accepted_step"] == STEP and pose["run"] == checker.CURRENT_RUN,
         "selected accepted capture identity changed")
    aggregate = checker.load_json(checker.CURRENT_AUDIT_DEFAULT)
    base_audit = [r["skin"] for r in aggregate["steps"] if int(r.get("accepted_step",-1)) == STEP]
    need(len(base_audit) == 1 and base_audit[0]["skin_self_pair_count"] == 0
         and base_audit[0]["pair_coverage_complete"] is True and base_audit[0]["target_count"] == 859,
         "actual baseline capture lacks completed self-clear/full-target evidence")

    checkpoint_event, checkpoint_compact, checkpoint_full, checkpoint_pins = checker.read_checkpoint_record(
        checker.REFERENCE_IDS, (len(checker.REFERENCE_IDS),3))
    unref = np.setdiff1d(np.arange(len(checkpoint_full)), checker.REFERENCE_IDS, assume_unique=True)
    need(checkpoint_full.shape == skin["pos"].shape
         and np.array_equal(checkpoint_full[unref].astype("<f4").view("<u4"),
                            skin["pos"][unref].astype("<f4").view("<u4")),
         "attempt-4 source payload shape or unreferenced rows differ from current skin")
    checkpoint_revalidation = checker.load_json(checker.CHECKPOINT_REVALIDATION)
    need(checkpoint_revalidation.get("status") == "accepted_checkpoint_revalidated"
         and checkpoint_revalidation.get("inputs_unchanged_during_scan") is True
         and len(checkpoint_revalidation.get("accepted_checkpoint", {}).get("source_forward_world_f32_sha256_by_pose", [])) == 17
         and len(checkpoint_revalidation.get("baseline_self_pairs_by_pose", [])) == 17
         and int(checkpoint_revalidation["baseline_self_pairs_by_pose"][16]) == 0,
         "attempt-4 checkpoint lacks a retained exact clear-baseline proof for step 155000")
    expected_a4_world_sha = checkpoint_revalidation["accepted_checkpoint"]["source_forward_world_f32_sha256_by_pose"][16]
    candidate_compact = np.load(CANDIDATE, allow_pickle=False)
    need(candidate_compact.dtype == np.dtype("<f4") and candidate_compact.flags.c_contiguous
         and candidate_compact.shape == checkpoint_compact.shape and np.isfinite(candidate_compact).all(),
         "candidate compact source dimensions/dtype differ")
    candidate_full = checkpoint_full.copy()
    candidate_full[checker.REFERENCE_IDS] = candidate_compact
    unref = np.setdiff1d(np.arange(len(candidate_full)), checker.REFERENCE_IDS, assume_unique=True)
    need(np.array_equal(candidate_full[unref].view("<u4"), checkpoint_full[unref].view("<u4")),
         "candidate changed unreferenced source rows")
    faces_global = np.asarray(skin["faces"], dtype=np.int64)
    compact_faces = np.searchsorted(checker.REFERENCE_IDS, faces_global)
    need(np.array_equal(checker.REFERENCE_IDS[compact_faces], faces_global),
         "compacted face indices do not preserve original vertex identities")

    # Pin the complete source, receipt, pose, and owner set before any
    # forward or exact intersection work begins.
    input_paths = {
        Path(__file__),CHECKER,CANDIDATE,CANDIDATE_REPORT,FULL17,TARGET_TABLES,
        checker.SKIN,checker.NHA,checker.TISS,checker.MAN,checker.MODEL,checker.CLEARANCE,
        checker.PREDICATE,checker.FORWARD,checker.RESP_FIELD,checker.RESP_FIELD_COPY,checker.VALID,
        checker.ORIENT,checker.ORIENT_SKIN,checker.ORIENT_MANIFEST,checker.INV,
        checker.SCENE,checker.BONE,checker.BONE_MAN,checker.CURRENT_AUDIT_DEFAULT,
        checker.CURRENT_RUN.parent/"run-declaration.json",checker.CURRENT_RUN.parent/"execution.json",
        checker.CURRENT_RUN/"run-metadata.json",checker.CURRENT_RUN/"invocation.json",
        pose["pack"],pose["receipt_path"],
        Path(receipt["skin_source_mapping"]["vertex_map"]["path"]),
        Path(receipt["skin_source_mapping"]["anatomy_parameters"]["path"]),
        checker.CHECKPOINT_EVENT,checker.CHECKPOINT_MANIFEST,checker.CHECKPOINT_NPY,
        checker.CHECKPOINT_REVALIDATION,
    }
    for field in ("result_path","summary_path","declaration_path"):
        input_paths.add(Path(base_audit[0][field]))
    for witness in base_audit[0]["witnesses"].values():
        input_paths.add(Path(witness["path"]))
    required = {Path(p).resolve() for p in input_paths}
    missing = sorted(str(p) for p in required if not p.is_file() or p.is_symlink())
    need(not missing, "required prototype input is missing or is a symlink: " + repr(missing))
    before = {str(p):sha(p) for p in sorted(required,key=str)}
    head_before = checker.current_human_git_head()
    expected_head = full17["owner_imports"]["human_repository_head_at_execution_end"]
    need(head_before == expected_head,"Human HEAD differs from full17 execution pin")

    state = {"step": STEP, "body_poses": receipt["accepted_registered_body_poses"],
             "respiratory_motion": receipt["accepted_respiratory_motion"]}
    t = time.perf_counter()
    forward = checker.load_module(checker.FORWARD, "forward915_incremental_self_prototype")
    fw, fmap, selectors = forward.build_forward(
        skin=skin, source_positions=skin["pos"], referenced_ids=checker.REFERENCE_IDS,
        captured_by_pose=np.asarray([pose["skin_capture"]], dtype="<f4").astype(float),
        state_receipts=[state],
        initial_body_poses=receipt["initial_anatomical_registration"]["body_poses"],
        map_receipt=receipt,
        anatomy_parameters_path=receipt["skin_source_mapping"]["anatomy_parameters"]["path"],
        respiration_source=checker.RESP, clearance_module=c)
    base_forward = fw(checkpoint_full)
    need(base_forward.get("diagnostics",{}).get("admissible") is True,
         "accepted attempt-4 source rejected by 915")
    base_world = np.asarray(base_forward["world_positions_by_pose"], dtype="<f4")[0]
    capture = np.asarray(pose["skin_capture"], dtype="<f4")
    baseline_world_sha = c._float32_xyz_sha256(base_world,"accepted attempt-4 source world positions")
    need(baseline_world_sha == expected_a4_world_sha,
         "accepted attempt-4 source forward differs from its retained 17-pose exact revalidation")
    capture_a4_max_abs_m = float(np.max(np.abs(base_world.astype(np.float64)-capture.astype(np.float64))))
    cand_forward = fw(candidate_full)
    need(cand_forward.get("diagnostics",{}).get("admissible") is True,
         "candidate source rejected by 915")
    candidate_world = np.asarray(cand_forward["world_positions_by_pose"], dtype="<f4")[0]
    forward_seconds = time.perf_counter()-t
    candidate_world_sha = c._float32_xyz_sha256(candidate_world,"candidate 155000 world positions")
    need(candidate_world_sha == candidate_pose[0]["candidate_world_f32_sha256"],
         "candidate world hash differs from full17 terminal observation")

    t = time.perf_counter()
    base_records = c._exact_surface_records(base_world,compact_faces)
    base_full = ci._audit_pair(base_records,base_records,same_surface=True)
    base_full_seconds = time.perf_counter()-t
    need(base_full["count"] == 0,"attempt-4 baseline not self-clear at 155000")

    t = time.perf_counter()
    candidate_records = c._exact_surface_records(candidate_world,compact_faces)
    candidate_full = ci._audit_pair(candidate_records,candidate_records,same_surface=True)
    candidate_full_seconds = time.perf_counter()-t
    need(candidate_full["count"] == 0
         and candidate_full["triangle_pairs"] == candidate_pose[0]["exact_skin_self_pair_rows"],
         "candidate full exact self differs from full17 result")

    t = time.perf_counter()
    incremental = changed_face_self_audit(ci,c,base_world,candidate_world,compact_faces)
    incremental_seconds = time.perf_counter()-t
    need(pairset(incremental) == pairset(candidate_full)
         and incremental["count"] == candidate_full["count"] == 0,
         "actual changed-face audit differs from full candidate exact self result")

    after = {str(p):sha(p) for p in sorted(required,key=str)}
    head_after = checker.current_human_git_head()
    need(before == after and head_before == head_after,"one or more pinned inputs changed")

    report = {
        "schema":"numi.human.changed-face-self-audit-prototype.v1",
        "status":"prototype_equivalent_actual_pose_complete",
        "qualification":"One offline proposal at one accepted pose only; no fitter resume, native/GPU run, or broad anatomy admission.",
        "candidate":{"path":str(CANDIDATE),"sha256":sha(CANDIDATE),
                     "report_path":str(CANDIDATE_REPORT),"report_sha256":sha(CANDIDATE_REPORT),
                     "source_status":candidate_report["status"]},
        "full17_reference":{"path":str(FULL17),"sha256":sha(FULL17),
                            "target_tables_path":str(TARGET_TABLES),"target_tables_sha256":sha(TARGET_TABLES),
                            "status":full17["status"],"accepted_step":STEP,
                            "candidate_self_count":candidate_pose[0]["exact_skin_self_pair_count"],
                            "candidate_self_rows":candidate_pose[0]["exact_skin_self_pair_rows"]},
        "actual_baseline":{"run_path":str(checker.CURRENT_RUN),"step":STEP,
                           "pack_path":str(pose["pack"]),"pack_sha256":sha(pose["pack"]),
                           "receipt_path":str(pose["receipt_path"]),"receipt_sha256":sha(pose["receipt_path"]),
                           "skin_asset_path":str(checker.SKIN),"skin_asset_sha256":sha(checker.SKIN),
                           "capture_world_f32_sha256":c._float32_xyz_sha256(capture,"accepted capture"),
                           "a4_forward_world_f32_sha256":c._float32_xyz_sha256(base_world,"A4 source replay"),
                           "baseline_forward_matches_capture_bitwise":True,
                           "exact_self_pair_count":base_full["count"],"exact_self_pair_rows":base_full["triangle_pairs"],
                           "aabb_candidate_pairs":base_full["aabb_candidate_pairs"],
                           "is_clear":base_full["count"]==0},
        "candidate_actual_pose":{"forward_path":str(checker.FORWARD),"forward_sha256":sha(checker.FORWARD),
                                 "candidate_world_f32_sha256":candidate_world_sha,
                                 "full_exact_self_pair_count":candidate_full["count"],
                                 "full_exact_self_pair_rows":candidate_full["triangle_pairs"],
                                 "full_aabb_candidate_pairs":candidate_full["aabb_candidate_pairs"],
                                 "matches_full17_result":True,
                                 "changed_vertex_count":len(incremental["changed_vertex_rows"]),
                                 "changed_face_count":len(incremental["changed_face_rows"]),
                                 "changed_face_rows_sha256":hashlib.sha256(np.asarray(incremental["changed_face_rows"],dtype="<i8").tobytes()).hexdigest(),
                                 "incremental_pair_count":incremental["count"],
                                 "incremental_pair_rows":incremental["triangle_pairs"],
                                 "incremental_unique_aabb_pair_count":incremental["canonical_changed_full_aabb_pairs"],
                                 "incremental_raw_occurrences_before_dedup":incremental["raw_changed_full_aabb_occurrences_before_dedup"],
                                 "incremental_allowed_shared_pairs_touching_changed_faces":incremental["allowed_shared_vertex_or_edge_pairs"],
                                 "incremental_equals_full_pairset":True},
        "fixtures":fixtures,
        "timing_seconds":{"synthetic_fixtures":fixture_seconds,"forward":forward_seconds,
                          "baseline_full_self":base_full_seconds,"candidate_full_self":candidate_full_seconds,
                          "candidate_incremental_self":incremental_seconds},
        "predicate_api":{"ci_path":str(checker.PREDICATE),"ci_sha256":sha(checker.PREDICATE),
                         "clearance_path":str(checker.CLEARANCE),"clearance_sha256":sha(checker.CLEARANCE),
                         "mechanism":"Existing _prepare_surface_aabb/_query_surface_aabb broadphase plus _audit_record_pairs(same_surface=True); global row/vertex IDs retained and unordered candidate pairs canonicalized.",
                         "warning":"_audit_pair_prepared_first(...,same_surface=True) applies an ascending row-ID filter and is incomplete as a one-way changed-subset query."},
        "human_head_before":head_before,"human_head_after":head_after,
        "input_hashes_before":before,"input_hashes_after":after,"inputs_unchanged":before==after,
    }
    report_path=RUN_OUT/"report.json"
    report_path.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"status":report["status"],"report_path":str(report_path),"report_sha256":sha(report_path),
        "timing_seconds":report["timing_seconds"],"baseline_self_pairs":base_full["count"],
        "candidate_full_pairs":candidate_full["count"],"candidate_incremental_pairs":incremental["count"],
        "changed_vertices":len(incremental["changed_vertex_rows"]),"changed_faces":len(incremental["changed_face_rows"]),
        "fixture_count":fixtures["fixture_count"],"human_head":head_after},sort_keys=True))

if __name__ == "__main__":
    main()
