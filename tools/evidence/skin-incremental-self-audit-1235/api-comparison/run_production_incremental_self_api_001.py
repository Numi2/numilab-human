#!/usr/bin/env python3
"""Run W's production incremental self-audit API against retained 155000 A4->d319."""
from __future__ import annotations
import hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

R = Path("/Users/n/numi-human-retained-delivery-20261009")
E = Path("/Users/n/numi-human-resting-evidence-20261005")
BASE = R / "skin-resting-multipose-clearance-1218"
OUT = BASE / "incremental-self-prototype-001/api-comparison-001"
WORKTREE = Path("/Users/n/numi-human-incremental-self-audit-1235")
W_SOURCE = WORKTREE / "src/numilab_human/common_atlas_skin_clearance.py"
W_TEST = WORKTREE / "tests/test_common_atlas_skin_incremental_self_audit.py"
CHECKER = BASE / "local-self-reduction-audit-001/verify_candidate_full_gates_1218_001.py"
FORWARD = E / "native-common-skin-multipose-forward-model-915.py"
FULL17 = BASE / "local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd/candidate-verification.json"
TARGET_TABLES = BASE / "local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd/candidate-target-pair-tables.json"
CANDIDATE = BASE / "bilateral-source-qp-001/qp-dual-localcheck-018-root/result-001/unadmitted-compact-source.npy"
CANDIDATE_REPORT = CANDIDATE.with_name("local-feasibility.json")
PROTOTYPE = BASE / "incremental-self-prototype-001/run-006/report.json"
PROTOTYPE_SCRIPT = BASE / "incremental-self-prototype-001/prototype-attempt006.py"
CORRECTION = BASE / "incremental-self-prototype-001/run-006/baseline-forward-capture-correction.json"
CHECKPOINT_EVENT = BASE / "fit-attempt-002-adapter-revision-004/checkpoint-0004/accepted-attempt-0004.event.jsonl"
CHECKPOINT_MANIFEST = BASE / "fit-attempt-002-adapter-revision-004/checkpoint-0004/checkpoint-manifest.json"
CHECKPOINT_NPY = BASE / "fit-attempt-002/attempts/attempt-0004-source.npy"
CHECKPOINT_REVALIDATION = BASE / "fit-attempt-002-adapter-revision-004/checkpoint-check-0004-001/preflight.json"
AGGREGATE = BASE / "native-accepted-geometry-audit-001/eight-pose-aggregate-001/summary.json"
RUNTIME = BASE / "native-baseline-310s-preparation/native-run"
STEP = 155000
H_CLEARANCE_SHA = "cb249ec8fdc57ff95ff8bbbc3236b572eacb2aebee3bad083792bac39f0a67f4"
H_CI_SHA = "423180082e32d002375b612aad36f307ed439af07c04d655e6a8bedc87a338bd"
H_MODEL_SHA = "a12a26af00fcab009d525094f3fb3d87ea575eff30cd15b91b1a0e12f0a316c3"
FORWARD_SHA = "81a5532f634278a45aa9aaf1457d0fdb3be71906256f3490941a4885b74d4cdc"
CANDIDATE_SHA = "d319d85ac3fd74f7bcadfbc54361cda5a956cdf1fc90f9539f8c0c6de59884d1"
CANDIDATE_REPORT_SHA = "16141cf59ff76dd5e797b7f9ac785667ca5d9813b9cf895f9b8470fe2e41288f"
FULL17_SHA = "d7a92b15995a84609e8a56e50c89e6977a75320790065733ac4bca21cb5601b1"
TARGET_TABLES_SHA = "25901e7caebcee0429ba1bc3076fc5e132d13a2fff316fd027961bae580cb2f5"
PROTOTYPE_SHA = "a9560f206aed86c16396d10e64457b77db2bbc89b5a277aeffddc25a9682e4a5"
CORRECTION_SHA = "bd07e1899678b00380994cee7722156d39c9a3ad30455292562630e4b03cb590"
W_SOURCE_SHA = "be37104adf7b99c310b8c27043dba17422eeaca84079d9a12a64170256991977"
W_TEST_SHA = "a405f12d1d2df7001bc5993a38226da8f8c52c95e3599e47b678eab80b7df1fd"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def need(ok, message):
    if not ok:
        raise RuntimeError(message)

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "cannot import " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def pin(path):
    path = Path(path).resolve()
    need(path.is_file() and not path.is_symlink(), "required regular input missing: " + str(path))
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}

def main():
    need(OUT.is_dir(), "precreated output directory is missing")
    need(not (OUT / "report.json").exists(), "refusing to overwrite prior API comparison report")

    checker = load_module(CHECKER, "full_gates_checker_production_api_1235")
    checker.verify_owner_imports()
    need(sha(CHECKER) == "5544f22de077f507b1486a197e29b093f78f3d0a199b7f1bafe0caff86832399",
         "full17 checker source changed")
    need(checker.current_human_git_head() == "c266d2ee6194917a964907d39effc02d5f405b7b",
         "Human HEAD differs from the reviewed full17 execution revision")
    need(sha(W_SOURCE) == W_SOURCE_SHA and sha(W_TEST) == W_TEST_SHA,
         "isolated worktree source/test changed")
    need(sha(FORWARD) == FORWARD_SHA and sha(checker.PREDICATE) == H_CI_SHA
         and sha(checker.MODEL) == H_MODEL_SHA and sha(checker.CLEARANCE) == H_CLEARANCE_SHA,
         "forward/Human owner source pins changed")
    capi = load_module(W_SOURCE, "numilab_human.common_atlas_skin_clearance_incremental_self_1235")
    need(Path(capi.__file__).resolve() == W_SOURCE.resolve(),
         "incremental API did not load from the isolated W worktree")
    need(Path(capi.ci.__file__).resolve() == checker.PREDICATE.resolve()
         and sha(capi.ci.__file__) == H_CI_SHA
         and Path(capi.human.__file__).resolve() == checker.MODEL.resolve()
         and sha(capi.human.__file__) == H_MODEL_SHA,
         "W API resolved its predicate/model dependencies from an unexpected source")
    need(callable(getattr(capi, "audit_incremental_skin_self_intersections", None)),
         "W worktree does not expose the reviewed incremental self API")

    full17 = checker.load_json(FULL17)
    need(sha(FULL17) == FULL17_SHA
         and full17.get("status") == "candidate_target_and_non_target_gate_scan_complete"
         and full17.get("candidate_non_target_gates_all_17_pass") is True
         and full17.get("inputs_unchanged_during_scan") is True,
         "pinned full17 candidate report is not the retained completed result")
    tables_ref = full17.get("candidate_target_pair_tables", {})
    need(tables_ref.get("path") == str(TARGET_TABLES) and tables_ref.get("sha256") == TARGET_TABLES_SHA
         and sha(TARGET_TABLES) == TARGET_TABLES_SHA,
         "full17 candidate target-pair table reference changed")
    candidate_pose = [row for row in full17["candidate_non_target_gate_checks"]
                      if int(row.get("accepted_step", -1)) == STEP]
    need(len(candidate_pose) == 1, "full17 report lacks the unique 155000 candidate row")
    candidate_pose = candidate_pose[0]
    need(candidate_pose.get("candidate_world_f32_sha256") == "e60e58d53d2cec864caf42c5f45d51429bdd24bcab6364748acb91c3ff98d03d"
         and candidate_pose.get("exact_skin_self_pair_count") == 0
         and candidate_pose.get("exact_skin_self_pair_rows") == []
         and candidate_pose.get("exact_skin_degenerate_face_rows") == [],
         "full17 155000 candidate exact self result changed")
    need(sha(CANDIDATE) == CANDIDATE_SHA and sha(CANDIDATE_REPORT) == CANDIDATE_REPORT_SHA,
         "d319 candidate or its local exact-precheck report changed")
    candidate_report = checker.load_json(CANDIDATE_REPORT)
    need(candidate_report.get("status") == "local_candidate_ready_for_full17_exact_checks"
         and candidate_report.get("local_exact_recheck", {}).get("self_clear") is True
         and candidate_report.get("local_exact_recheck", {}).get("target_no_new_pairs") is True,
         "d319 candidate lacks its retained local exact checks")
    prototype = checker.load_json(PROTOTYPE)
    need(sha(PROTOTYPE) == PROTOTYPE_SHA and sha(CORRECTION) == CORRECTION_SHA,
         "retained prototype or capture-correction supplement changed")
    prototype_pose = prototype.get("candidate_actual_pose", {})
    need(prototype_pose.get("candidate_world_f32_sha256") == candidate_pose["candidate_world_f32_sha256"]
         and prototype_pose.get("full_exact_self_pair_rows") == candidate_pose["exact_skin_self_pair_rows"]
         and prototype_pose.get("incremental_pair_rows") == candidate_pose["exact_skin_self_pair_rows"]
         and prototype_pose.get("incremental_equals_full_pairset") is True
         and prototype_pose.get("matches_full17_result") is True,
         "prior prototype is inconsistent with the retained full17 result")
    correction = checker.load_json(CORRECTION)
    need(correction.get("check", {}).get("bitwise_equal") is False
         and correction.get("check", {}).get("a4_forward_sha256") == "8e56ed1c6dfa4f083a8883c8b6ad2538fd62d48488770dd2b959fe6f9735664f"
         and correction.get("check", {}).get("captured_world_sha256") == "fe10675bc9804dcc9d5195c226e524e4d8d7e5f469427b43fef6f670568c07ef",
         "prototype correction does not preserve the baseline-vs-capture mismatch")

    aggregate = checker.load_json(AGGREGATE)
    aggregate_rows = [row for row in aggregate.get("steps", []) if int(row.get("accepted_step", -1)) == STEP]
    need(len(aggregate_rows) == 1, "current run aggregate lacks a unique 155000 baseline capture audit")
    actual_skin_audit = aggregate_rows[0]["skin"]
    need(actual_skin_audit.get("pair_coverage_complete") is True
         and actual_skin_audit.get("target_count") == 859
         and actual_skin_audit.get("skin_self_pair_count") == 0,
         "actual accepted capture is not bound to a complete, self-clear 859-target audit")

    # Pin all source/input files recorded by the 17-pose verifier plus
    # explicitly referenced 155000 inputs and W implementation files.
    input_pins = dict(full17.get("input_hashes_before_scan", {}))
    need(input_pins and isinstance(input_pins, dict), "full17 source input pin map is absent")
    expected_paths = {
        Path(__file__), CHECKER, FORWARD, FULL17, TARGET_TABLES, CANDIDATE, CANDIDATE_REPORT,
        PROTOTYPE, PROTOTYPE_SCRIPT, CORRECTION, W_SOURCE, W_TEST,
        CHECKPOINT_EVENT, CHECKPOINT_MANIFEST, CHECKPOINT_NPY, CHECKPOINT_REVALIDATION,
        AGGREGATE, RUNTIME / "accepted-geometry/step-155000.mrvpack",
        RUNTIME / "accepted-geometry/step-155000.receipt.json",
        RUNTIME / "run-metadata.json", RUNTIME / "invocation.json",
        RUNTIME.parent / "run-declaration.json", RUNTIME.parent / "execution.json",
    }
    expected_paths.update(Path(p) for p in input_pins)
    for field in ("result_path", "summary_path", "declaration_path"):
        expected_paths.add(Path(actual_skin_audit[field]))
    for witness in actual_skin_audit["witnesses"].values():
        expected_paths.add(Path(witness["path"]))
    expected_paths = {Path(path).resolve() for path in expected_paths}
    need(all(path.is_file() and not path.is_symlink() for path in expected_paths),
         "one or more exact source/input files are absent or are symlinks")
    before = {str(path): sha(path) for path in sorted(expected_paths, key=str)}
    for path, expected in input_pins.items():
        need(before[str(Path(path).resolve())] == expected,
             "full17 input changed since its scan: " + str(path))
    need(before[str(W_SOURCE.resolve())] == W_SOURCE_SHA
         and before[str(W_TEST.resolve())] == W_TEST_SHA
         and before[str(FULL17.resolve())] == FULL17_SHA
         and before[str(CANDIDATE.resolve())] == CANDIDATE_SHA,
         "one or more pinned source/output SHA values differ")

    raw_skin, skin = checker.load_skin(checker.SKIN)
    checker.CURRENT_SKIN = skin
    checker.REFERENCE_IDS = np.unique(skin["faces"]).astype(np.int64)
    target_keys, _, _, _ = checker.c._load_target_inventory(checker.INV)
    accepted_helper = checker.load_module(checker.VALID, "accepted_mrvpack_production_api_1235")
    pose = checker.read_pose(("1218_current_run", checker.CURRENT_RUN, STEP),
                             set(target_keys), accepted_helper, None, None, None)
    receipt = pose["receipt"]
    need(receipt.get("accepted_step") == STEP and pose["run"] == checker.CURRENT_RUN,
         "accepted 155000 run identity changed")
    captured_world = np.asarray(pose["skin_capture"], dtype="<f4")
    expected_capture_sha = "fe10675bc9804dcc9d5195c226e524e4d8d7e5f469427b43fef6f670568c07ef"
    capture_sha = capi._float32_xyz_sha256(captured_world, "actual 155000 captured skin")
    need(capture_sha == expected_capture_sha,
         "actual 155000 captured-skin point identity differs from retained correction report")

    manifest = checker.load_json(CHECKPOINT_MANIFEST)
    event_bytes = CHECKPOINT_EVENT.read_bytes()
    event = json.loads(event_bytes)
    revalidation = checker.load_json(CHECKPOINT_REVALIDATION)
    need(sha(CHECKPOINT_REVALIDATION) == "7cc945fd1ed6e6bff455c425265304845626a50cef1aa02dbdee54c74e5d68f0"
         and revalidation.get("status") == "accepted_checkpoint_revalidated"
         and revalidation.get("inputs_unchanged_during_scan") is True,
         "accepted attempt-4 source revalidation report changed")
    need(manifest.get("status") == "accepted" and manifest.get("accepted_attempt") == 4
         and manifest.get("candidate_event_copy_sha256") == sha(CHECKPOINT_EVENT)
         and event_bytes.endswith(b"\n") and event_bytes.count(b"\n") == 1
         and event.get("attempt") == 4 and event.get("status") == "accepted"
         and manifest.get("candidate_npy_path") == str(CHECKPOINT_NPY)
         and event.get("candidate_npy_path") == str(CHECKPOINT_NPY),
         "immutable accepted attempt-4 event/manifest identity changed")
    checkpoint_bytes_sha = sha(CHECKPOINT_NPY)
    need(event.get("candidate_npy_sha256") == checkpoint_bytes_sha
         and manifest.get("candidate_npy_sha256") == checkpoint_bytes_sha
         and event.get("source_positions_f32_sha256") == capi._float32_xyz_sha256(np.load(CHECKPOINT_NPY, allow_pickle=False), "A4 compact source positions"),
         "attempt-4 source array differs from event/manifest")
    checkpoint_compact = np.load(CHECKPOINT_NPY, allow_pickle=False)
    need(checkpoint_compact.dtype == np.dtype("<f4") and checkpoint_compact.flags.c_contiguous
         and checkpoint_compact.shape == (len(checker.REFERENCE_IDS), 3)
         and np.isfinite(checkpoint_compact).all(),
         "accepted attempt-4 compact source array is malformed")
    baseline_source = np.asarray(skin["pos"], dtype="<f4").copy()
    baseline_source[checker.REFERENCE_IDS] = checkpoint_compact
    candidate_compact = np.load(CANDIDATE, allow_pickle=False)
    need(candidate_compact.dtype == np.dtype("<f4") and candidate_compact.flags.c_contiguous
         and candidate_compact.shape == checkpoint_compact.shape and np.isfinite(candidate_compact).all(),
         "d319 compact source array is malformed")
    candidate_source = baseline_source.copy()
    candidate_source[checker.REFERENCE_IDS] = candidate_compact
    unref = np.setdiff1d(np.arange(len(baseline_source)), checker.REFERENCE_IDS, assume_unique=True)
    need(np.array_equal(baseline_source[unref].view("<u4"), skin["pos"][unref].astype("<f4").view("<u4"))
         and np.array_equal(candidate_source[unref].view("<u4"), baseline_source[unref].view("<u4")),
         "source expansion changed one of the 286 unreferenced records")
    global_faces = np.asarray(skin["faces"], dtype=np.int64)
    compact_faces = np.searchsorted(checker.REFERENCE_IDS, global_faces)
    need(np.array_equal(checker.REFERENCE_IDS[compact_faces], global_faces),
         "compact face topology lost original source vertex identities")
    expected_face_sha = capi._face_index_sha256(compact_faces)
    need(len(compact_faces) == 109211 and len(checker.REFERENCE_IDS) == 54663,
         "current selected NHSKIN row counts changed")
    need(revalidation["baseline_self_pairs_by_pose"][16] == 0
         and len(revalidation["accepted_checkpoint"]["source_forward_world_f32_sha256_by_pose"]) == 17,
         "accepted attempt-4 has no retained 17-pose world/self baseline")
    expected_a4_world_sha = revalidation["accepted_checkpoint"]["source_forward_world_f32_sha256_by_pose"][16]
    need(expected_a4_world_sha == "8e56ed1c6dfa4f083a8883c8b6ad2538fd62d48488770dd2b959fe6f9735664f"
         and full17["accepted_checkpoint"]["candidate_incremental_baseline"]["world_f32_sha256_by_pose"][16] == expected_a4_world_sha,
         "A4 accepted checkpoint source world hash differs between full17 and revalidation evidence")

    state = {"step": STEP, "body_poses": receipt["accepted_registered_body_poses"],
             "respiratory_motion": receipt["accepted_respiratory_motion"]}
    started = time.perf_counter()
    forward = checker.load_module(FORWARD, "forward915_incremental_self_api_1235")
    fw, forward_map_report, selector_ids = forward.build_forward(
        skin=skin, source_positions=skin["pos"], referenced_ids=checker.REFERENCE_IDS,
        captured_by_pose=np.asarray([captured_world], dtype="<f4").astype(float),
        state_receipts=[state],
        initial_body_poses=receipt["initial_anatomical_registration"]["body_poses"],
        map_receipt=receipt,
        anatomy_parameters_path=receipt["skin_source_mapping"]["anatomy_parameters"]["path"],
        respiration_source=checker.RESP, clearance_module=capi)
    base_forward = fw(baseline_source)
    need(base_forward.get("diagnostics", {}).get("admissible") is True,
         "accepted A4 source was rejected by pinned forward owner")
    base_world = np.asarray(base_forward["world_positions_by_pose"], dtype="<f4")[0]
    baseline_world_sha = capi._float32_xyz_sha256(base_world, "attempt-4 source world positions")
    need(baseline_world_sha == expected_a4_world_sha,
         "W API forward path does not reproduce the accepted attempt-4 world identity")
    candidate_forward = fw(candidate_source)
    need(candidate_forward.get("diagnostics", {}).get("admissible") is True,
         "d319 source was rejected by the pinned 915 forward owner")
    candidate_world = np.asarray(candidate_forward["world_positions_by_pose"], dtype="<f4")[0]
    candidate_world_sha = capi._float32_xyz_sha256(candidate_world, "d319 candidate world positions")
    need(candidate_world_sha == candidate_pose["candidate_world_f32_sha256"]
         and candidate_world_sha == prototype_pose["candidate_world_f32_sha256"],
         "W API forward candidate world differs from full17/prototype")
    forward_seconds = time.perf_counter() - started

    started = time.perf_counter()
    base_records = capi._exact_surface_records(base_world, compact_faces)
    baseline_audit = capi.ci._audit_pair(base_records, base_records, same_surface=True)
    baseline_full_seconds = time.perf_counter() - started
    need(baseline_audit["count"] == 0 and baseline_audit["triangle_pairs"] == []
         and baseline_audit["count"] == revalidation["baseline_self_pairs_by_pose"][16],
         "attempt-4 baseline at 155000 is not the retained exact self-clear state")

    started = time.perf_counter()
    candidate_records = capi._exact_surface_records(candidate_world, compact_faces)
    candidate_audit = capi.ci._audit_pair(candidate_records, candidate_records, same_surface=True)
    candidate_full_seconds = time.perf_counter() - started
    need(candidate_audit["count"] == candidate_pose["exact_skin_self_pair_count"]
         and candidate_audit["triangle_pairs"] == candidate_pose["exact_skin_self_pair_rows"]
         and candidate_audit["triangle_pairs"] == prototype_pose["full_exact_self_pair_rows"],
         "full exact candidate pair table differs from full17 or prior prototype")

    started = time.perf_counter()
    api_result = capi.audit_incremental_skin_self_intersections(
        baseline_world_positions=base_world,
        candidate_world_positions=candidate_world,
        baseline_faces=compact_faces,
        candidate_faces=compact_faces.copy(),
        baseline_self_audit=baseline_audit,
        expected_baseline_world_f32_sha256=baseline_world_sha,
        expected_face_index_sha256=expected_face_sha,
        expected_baseline_self_pair_table_sha256=
            capi._baseline_self_pair_table_sha256(baseline_audit, len(compact_faces)))
    api_seconds = time.perf_counter() - started
    need(api_result["triangle_pairs"] == candidate_audit["triangle_pairs"]
         and api_result["triangle_pairs"] == candidate_pose["exact_skin_self_pair_rows"]
         and api_result["triangle_pairs"] == prototype_pose["incremental_pair_rows"]
         and api_result["count"] == candidate_audit["count"] == 0
         and api_result["candidate_world_f32_sha256"] == candidate_world_sha
         and len(api_result["changed_skin_vertex_rows"]) == prototype_pose["changed_vertex_count"]
         and len(api_result["changed_skin_face_rows"]) == prototype_pose["changed_face_count"]
         and hashlib.sha256(np.asarray(api_result["changed_skin_face_rows"], dtype="<i8").tobytes()).hexdigest()
             == prototype_pose["changed_face_rows_sha256"],
         "new W production API did not reproduce the full owner/prototype complete pair table")
    need(api_result["fresh_changed_face_aabb_candidate_pairs"] >= api_result["fresh_changed_face_exact_pair_count"]
         and api_result["reused_baseline_exact_pair_count"] == 0,
         "incremental audit fresh-work counters are inconsistent with the clear baseline")

    capture_diff = np.asarray(base_world, dtype=np.float64) - captured_world.astype(np.float64)
    capture_diff_abs = np.abs(capture_diff)
    capture_bitwise_equal = np.array_equal(base_world.view("<u4"), captured_world.view("<u4"))
    need(capture_bitwise_equal is False and
         int(np.count_nonzero(base_world.view("<u4") != captured_world.view("<u4"))) == 7098 and
         float(capture_diff_abs.max()) == 0.0006973538547754288,
         "A4-vs-captured-world difference changed from the retained correction evidence")

    after = {str(path): sha(path) for path in sorted(expected_paths, key=str)}
    need(before == after, "one or more pinned API/prototype/full17 inputs changed during the scan")
    head_after = checker.current_human_git_head()
    need(head_after == "c266d2ee6194917a964907d39effc02d5f405b7b",
         "Human checkout HEAD changed during API comparison")

    report = {
        "schema": "numi.human.incremental-self-api-production-comparison.v1",
        "status": "production_api_equivalent_to_full_exact_owner_and_retained_prototype",
        "qualification": "One offline 155000 A4-to-d319 source transition only. This validates the isolated worktree API against the exact owner and retained 17-pose result; it is not a native run, fitter admission, or whole-body qualification.",
        "worktree": {"path": str(WORKTREE), "head": subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
                     "source_path": str(W_SOURCE), "source_sha256": sha(W_SOURCE),
                     "test_path": str(W_TEST), "test_sha256": sha(W_TEST),
                     "api_path_loaded": str(Path(capi.__file__).resolve()),
                     "predicate_path": str(Path(capi.ci.__file__).resolve()), "predicate_sha256": sha(capi.ci.__file__),
                     "model_path": str(Path(capi.human.__file__).resolve()), "model_sha256": sha(capi.human.__file__)},
        "candidate": {"path": str(CANDIDATE), "sha256": sha(CANDIDATE),
                      "candidate_report_path": str(CANDIDATE_REPORT), "candidate_report_sha256": sha(CANDIDATE_REPORT),
                      "status": candidate_report["status"]},
        "accepted_step": STEP,
        "baseline": {"kind": "accepted_attempt_4 source, not captured candidate source",
                     "world_f32_sha256": baseline_world_sha,
                     "revalidation_expected_world_sha256": expected_a4_world_sha,
                     "full_exact_self_pair_count": baseline_audit["count"],
                     "full_exact_self_pair_rows": baseline_audit["triangle_pairs"],
                     "pair_table_sha256": capi._baseline_self_pair_table_sha256(baseline_audit, len(compact_faces)),
                     "pair_table_matches_retained_clear_baseline_count": True},
        "candidate_world": {"world_f32_sha256": candidate_world_sha,
                            "full_exact_self_pair_count": candidate_audit["count"],
                            "full_exact_self_pair_rows": candidate_audit["triangle_pairs"],
                            "matches_full17_exact_result": True,
                            "matches_prior_prototype_world_and_full_pair_table": True,
                            "matches_prior_prototype_changed_face_rows_sha256": True},
        "production_api": {"schema": api_result["schema"],
                           "count": api_result["count"],
                           "triangle_pairs": api_result["triangle_pairs"],
                           "matches_full_exact_owner": True,
                           "matches_full17_result": True,
                           "matches_prior_prototype_incremental_pair_table": True,
                           "changed_vertex_count": len(api_result["changed_skin_vertex_rows"]),
                           "changed_face_count": len(api_result["changed_skin_face_rows"]),
                           "changed_face_rows_sha256": hashlib.sha256(np.asarray(api_result["changed_skin_face_rows"], dtype="<i8").tobytes()).hexdigest(),
                           "fresh_changed_face_aabb_candidate_pairs": api_result["fresh_changed_face_aabb_candidate_pairs"],
                           "fresh_changed_face_exact_pair_count": api_result["fresh_changed_face_exact_pair_count"],
                           "fresh_changed_face_allowed_shared_vertex_or_edge_pair_count":
                               api_result["fresh_changed_face_allowed_shared_vertex_or_edge_pair_count"],
                           "fresh_changed_face_aabb_raw_occurrences_before_dedup":
                               api_result["fresh_changed_face_aabb_raw_occurrences_before_dedup"],
                           "reused_baseline_exact_pair_count": api_result["reused_baseline_exact_pair_count"],
                           "aabb_work_scope": api_result["aabb_work_scope"]},
        "actual_capture_comparison": {"captured_skin_world_f32_sha256": capture_sha,
                                      "a4_forward_world_f32_sha256": baseline_world_sha,
                                      "bitwise_equal": capture_bitwise_equal,
                                      "different_float32_scalar_count": int(np.count_nonzero(base_world.view("<u4") != captured_world.view("<u4"))),
                                      "max_absolute_coordinate_difference_m": float(capture_diff_abs.max()),
                                      "matches_retained_correction": True,
                                      "note": "A4 uses accepted checkpoint source positions while the native capture used current composed candidate source positions; equality to capture is not claimed or required."},
        "full17_reference": {"path": str(FULL17), "sha256": sha(FULL17),
                             "target_pair_tables_path": str(TARGET_TABLES), "target_pair_tables_sha256": sha(TARGET_TABLES),
                             "terminal_candidate_self_pair_count": candidate_pose["exact_skin_self_pair_count"],
                             "terminal_candidate_self_pair_rows": candidate_pose["exact_skin_self_pair_rows"]},
        "prior_prototype": {"path": str(PROTOTYPE), "sha256": sha(PROTOTYPE),
                            "correction_path": str(CORRECTION), "correction_sha256": sha(CORRECTION),
                            "full_candidate_pair_table": prototype_pose["full_exact_self_pair_rows"],
                            "incremental_pair_table": prototype_pose["incremental_pair_rows"]},
        "timing_seconds": {"forward_A4_and_d319": forward_seconds,
                           "A4_full_exact_self": baseline_full_seconds,
                           "d319_full_exact_self": candidate_full_seconds,
                           "production_incremental_api": api_seconds},
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__,
                    "argv": sys.argv, "human_head_before": checker.current_human_git_head(),
                    "human_head_after": head_after},
        "input_hashes_before": before,
        "input_hashes_after": after,
        "inputs_unchanged": before == after,
    }
    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "report_path": str(report_path),
                      "report_sha256": sha(report_path), "timing_seconds": report["timing_seconds"],
                      "baseline_pairs": baseline_audit["count"], "candidate_full_pairs": candidate_audit["count"],
                      "api_pairs": api_result["count"], "changed_faces": len(api_result["changed_skin_face_rows"]),
                      "input_count": len(before), "human_head": head_after}, sort_keys=True))

if __name__ == "__main__":
    main()
