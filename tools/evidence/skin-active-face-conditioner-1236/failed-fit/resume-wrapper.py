#!/usr/bin/env python3
"""Prepare or launch one bounded resume iteration from audited d319.

--check validates and freezes inputs only. --fit requires that exact preflight
and invokes the existing clearance owner for one new iteration.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
sys.dont_write_bytecode = True
import numpy as np

BASE = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218")
HUMAN = Path("/Users/n/numi-human-free-apex-two-family-1178")
FIT_SRC = BASE / "fit-attempt-002-adapter-revision-003/fit_17_pose_source_clearance.py"
FIT_SHA = "3b7d33b5a697822a0dc93140906e2b6933c013161bbff8dc2f8c2ad6177650e1"
FROZEN_ROOT = Path("/Users/n/numi-human-conforming-composition-source-1216")
FROZEN_CLEARANCE = FROZEN_ROOT / "src/numilab_human/common_atlas_skin_clearance.py"
FROZEN_CLEARANCE_SHA = "ec198693fce562db27c4b761e3dbbc4b1bff4f0b9458ae67ab7ba596a97a000f"
LIVE_CLEARANCE = HUMAN / "src/numilab_human/common_atlas_skin_clearance.py"
LIVE_CLEARANCE_SHA = "cb249ec8fdc57ff95ff8bbbc3236b572eacb2aebee3bad083792bac39f0a67f4"
FULL_GATE = BASE / "local-self-reduction-audit-001/verify_candidate_full_gates_1218_001.py"
FULL_GATE_SHA = "5544f22de077f507b1486a197e29b093f78f3d0a199b7f1bafe0caff86832399"
CANDIDATE = BASE / "bilateral-source-qp-001/qp-dual-localcheck-018-root/result-001/unadmitted-compact-source.npy"
CANDIDATE_SHA = "d319d85ac3fd74f7bcadfbc54361cda5a956cdf1fc90f9539f8c0c6de59884d1"
CANDIDATE_REPORT = BASE / "bilateral-source-qp-001/qp-dual-localcheck-018-root/result-001/local-feasibility.json"
AUDIT_DIR = BASE / "local-self-reduction-audit-001/candidate-full-gates-001/d319d85ac3fd"
AUDIT_REPORT = AUDIT_DIR / "candidate-verification.json"
AUDIT_REPORT_SHA = "d7a92b15995a84609e8a56e50c89e6977a75320790065733ac4bca21cb5601b1"
TABLES = AUDIT_DIR / "candidate-target-pair-tables.json"
TABLES_SHA = "25901e7caebcee0429ba1bc3076fc5e132d13a2fff316fd027961bae580cb2f5"
A4_EVENT = BASE / "fit-attempt-002-adapter-revision-004/checkpoint-0004/accepted-attempt-0004.event.jsonl"
A4_SHA = "42a17786bb3a21a504fcf02e944808d91d58e652c1d41f34e905a4f75e9b7a25"
CURRENT_AUDIT = BASE / "native-accepted-geometry-audit-001/eight-pose-aggregate-001/summary.json"
HEAD_EXPECTED = "c266d2ee6194917a964907d39effc02d5f405b7b"
POSES = [0, 9983, 20000, 4991, 5375, 5759, 6111, 6495, 7743, 0, 9983, 19999, 47519, 152191, 153183, 154143, 155000]
PAIR_COUNTS = [0] * 13 + [614, 624, 645, 638]
PIN_COUNT, A4_TOTAL = 766, 3131
PREP = BASE / "resume-prepare-d319-004"
PREFLIGHT = PREP / "resume-preflight.json"
FIT_OUT = BASE / "fit-attempt-003-resume-d319-001"
EXECUTION = PREP / "resume-execution.json"

def need(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text())

def pin(path):
    p = Path(path).resolve()
    need(p.is_file() and not p.is_symlink(), "required regular input missing: " + str(p))
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}

def head():
    value = subprocess.run(["git", "-C", str(HUMAN), "rev-parse", "HEAD"],
                           check=True, capture_output=True, text=True).stdout.strip()
    need(value == HEAD_EXPECTED, "Human HEAD differs from reviewed c266d2e")
    return value

def keytext(key):
    if isinstance(key, str):
        return key
    need(isinstance(key, (list, tuple)) and len(key) == 2, "malformed target identity")
    return f"{int(key[0])}:{int(key[1])}"

def load_fit():
    need(sha(FIT_SRC) == FIT_SHA and sha(FROZEN_CLEARANCE) == FROZEN_CLEARANCE_SHA
         and sha(LIVE_CLEARANCE) == LIVE_CLEARANCE_SHA and sha(FULL_GATE) == FULL_GATE_SHA,
         "one or more pinned runner/owner sources changed")
    spec = importlib.util.spec_from_file_location("fit_resume_d319_1218", FIT_SRC)
    need(spec is not None and spec.loader is not None, "cannot import frozen fit adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    need(Path(module.CLEARANCE).resolve() == FROZEN_CLEARANCE.resolve()
         and callable(getattr(module.c, "derive_shared_multipose_inferred_clearance", None)),
         "fitter does not use the pinned frozen resume API")
    return module

def validate():
    head0 = head()
    fit = load_fit()
    for path, expected, label in ((CANDIDATE, CANDIDATE_SHA, "d319 source"),
                                  (AUDIT_REPORT, AUDIT_REPORT_SHA, "full audit report"),
                                  (TABLES, TABLES_SHA, "exact target tables"),
                                  (A4_EVENT, A4_SHA, "A4 accepted event")):
        need(sha(path) == expected, label + " hash changed")
    need(not FIT_OUT.exists(), "fresh resume output already exists")
    proposal = read_json(CANDIDATE_REPORT)
    proposal_pin = pin(CANDIDATE_REPORT)
    candidate = np.load(CANDIDATE, allow_pickle=False)
    need(candidate.dtype == np.dtype("<f4") and candidate.flags.c_contiguous
         and candidate.shape == (54663, 3) and np.isfinite(candidate).all(),
         "candidate must be finite compact Float32 (54663,3)")
    pr = proposal.get("proposal", {})
    need(proposal.get("status") == "local_candidate_ready_for_full17_exact_checks"
         and Path(pr.get("path", "")).resolve() == CANDIDATE.resolve()
         and pr.get("sha256") == CANDIDATE_SHA,
         "local candidate report does not bind d319")

    report = read_json(AUDIT_REPORT)
    need(report.get("candidate_checkpoint_status") == "partial_target_pairs_non_target_gates_pass"
         and report.get("candidate_non_target_gates_all_17_pass") is True
         and report.get("target_audit_completed") is True
         and report.get("candidate_target_pairs_remain") is True
         and report.get("candidate_target_tables_cover_all_859_targets_by_pose") == [True] * 17
         and report.get("candidate_target_pair_counts_by_pose") == PAIR_COUNTS,
         "full-gate report is incomplete or differs from root-reviewed 17-pose result")
    need(report.get("inputs_unchanged_during_scan") is True
         and report.get("input_hashes_before_scan") == report.get("input_hashes_after_scan"),
         "full-gate input hashes changed during audit")
    pins = report.get("input_pins")
    before, after = report.get("input_hashes_before_scan"), report.get("input_hashes_after_scan")
    need(isinstance(pins, list) and len(pins) == PIN_COUNT
         and isinstance(before, dict) and len(before) == PIN_COUNT and before == after,
         "audit report does not contain the reviewed 766 unchanged input pins")
    pin_map = {}
    for row in pins:
        need(isinstance(row, dict) and isinstance(row.get("path"), str)
             and isinstance(row.get("sha256"), str) and len(row["sha256"]) == 64,
             "malformed input pin")
        path = str(Path(row["path"]).resolve())
        need(path not in pin_map, "duplicate input pin")
        pin_map[path] = row["sha256"]
    need(pin_map == {str(Path(k).resolve()): v for k, v in before.items()},
         "input pin rows differ from before/after hash map")
    for path, expected in pin_map.items():
        need(Path(path).is_file() and sha(path) == expected, "pinned input changed: " + path)
    owner = report.get("owner_imports", {})
    need(owner.get("human_repository_head_at_execution_start") == HEAD_EXPECTED
         and owner.get("human_repository_head_at_execution_end") == HEAD_EXPECTED
         and head() == HEAD_EXPECTED, "audit did not run on unchanged reviewed Human HEAD")
    checkpoint = report.get("accepted_checkpoint", {})
    need(checkpoint.get("candidate_verification_npy", {}).get("path") == str(CANDIDATE.resolve())
         and checkpoint.get("candidate_verification_npy", {}).get("sha256") == CANDIDATE_SHA
         and checkpoint.get("candidate_verification_report", {}).get("path") == str(CANDIDATE_REPORT.resolve())
         and checkpoint.get("candidate_verification_report", {}).get("sha256") == proposal_pin["sha256"],
         "full-gate accepted checkpoint does not bind candidate and local proposal report")
    need(report.get("candidate_target_pair_tables", {}).get("path") == str(TABLES.resolve())
         and report.get("candidate_target_pair_tables", {}).get("sha256") == TABLES_SHA,
         "full-gate report does not bind exact target table")
    progress_pin = pin(report.get("candidate_scan_progress", {}).get("path", ""))
    need(progress_pin["sha256"] == report["candidate_scan_progress"].get("sha256"),
         "progress journal hash mismatch")
    progress = [json.loads(line) for line in Path(progress_pin["path"]).read_text().splitlines() if line.strip()]
    need(len(progress) == 17 and [int(x.get("accepted_step", -1)) for x in progress] == POSES
         and all(x.get("status") == "pose_complete" and x.get("target_count") == 859 for x in progress),
         "progress journal is not the complete ordered 17 x 859 scan")

    allkeys, nonocular, _, _ = fit.c._load_target_inventory(fit.INV)
    all_text, nonocular_text = {keytext(k) for k in allkeys}, {keytext(k) for k in nonocular}
    need(len(all_text) == 859 and 0 < len(nonocular_text) < 859, "target inventory is malformed")
    tables_doc = read_json(TABLES)
    need(tables_doc.get("schema") == "numi.human.skin-incremental-exact-target-audit.v1"
         and tables_doc.get("candidate_npy", {}).get("path") == str(CANDIDATE.resolve())
         and tables_doc.get("candidate_npy", {}).get("sha256") == CANDIDATE_SHA
         and tables_doc.get("candidate_report", {}).get("path") == str(CANDIDATE_REPORT.resolve())
         and tables_doc.get("candidate_report", {}).get("sha256") == proposal_pin["sha256"],
         "target table does not bind candidate and report")
    pose_tables = tables_doc.get("poses")
    need(isinstance(pose_tables, list) and len(pose_tables) == 17, "target table does not contain 17 poses")
    resume_tables, all_counts = [], []
    for index, table in enumerate(pose_tables):
        need(isinstance(table, dict) and set(table) == all_text, f"target keys incomplete at pose {index}")
        mapped = {}
        for key, row in table.items():
            need(isinstance(row, dict) and type(row.get("count")) is int and row["count"] >= 0
                 and isinstance(row.get("triangle_pairs"), list),
                 f"malformed target row at pose {index}: {key}")
            pairs = [tuple(map(int, pair)) for pair in row["triangle_pairs"]]
            need(len(pairs) == row["count"] == len(set(pairs))
                 and all(a >= 0 and b >= 0 for a, b in pairs),
                 f"exact pair rows/count disagree at pose {index}: {key}")
            mapped[key] = row
        resume_tables.append(mapped)
        all_counts.append(sum(row["count"] for row in mapped.values()))
    need(all_counts == PAIR_COUNTS, "exact target table counts differ from reviewed values")
    residual = sum(row["count"] for table in resume_tables
                   for key, row in table.items() if key in nonocular_text)
    need(0 < residual < A4_TOTAL, f"resume is unwarranted: candidate={residual}, A4={A4_TOTAL}")
    raw = A4_EVENT.read_bytes()
    need(raw.endswith(b"\n") and raw.count(b"\n") == 1, "A4 accepted event is not one JSONL record")
    event = json.loads(raw)
    a4_tables = event.get("target_audits_by_pose")
    need(isinstance(a4_tables, list) and len(a4_tables) == 17, "A4 event lacks 17 exact maps")
    a4_total = sum(int(row["count"]) for table in a4_tables
                   for key, row in table.items() if keytext(key) in nonocular_text)
    need(a4_total == A4_TOTAL, "A4 exact nonocular total is not 3,131")

    gates = report.get("candidate_non_target_gate_checks")
    need(isinstance(gates, list) and len(gates) == 17
         and all(x.get("non_target_gates_pass") is True
                 and x.get("source_fixed_bed_and_thorax_anchors_unchanged") is True
                 and x.get("source_fixed_bed_vertex_count") == 32
                 and x.get("source_thorax_anchor_count") == 15
                 and x.get("exact_skin_self_pair_count") == 0
                 and not x.get("exact_skin_degenerate_face_rows")
                 and not x.get("exact_candidate_source_degenerate_face_rows")
                 and x.get("baseline_relative_orientation_pass") is True
                 and x.get("source_jacobian_winding_pass") is True
                 and x.get("baseline_relative_bed_pass") is True
                 and all(v.get("status") == "complete" and v.get("inside_vertex_count") == 0
                         for v in x.get("closed_target_inside", {}).values())
                 for x in gates),
         "candidate fails one of the 17 non-target gates")
    need(report.get("current_eight_pose_audit_aggregate", {}).get("sha256") == sha(fit.CURRENT_AUDIT_DEFAULT),
         "current native audit aggregate changed")
    need(not FIT_OUT.exists() and head0 == head(), "resume output exists or HEAD changed")
    identity = {
        "resume_wrapper": pin(Path(__file__)),
        "candidate": pin(CANDIDATE), "candidate_report": proposal_pin,
        "full_gate_report": pin(AUDIT_REPORT), "target_tables": pin(TABLES),
        "progress_journal": progress_pin, "attempt4_event": pin(A4_EVENT),
        "fit_adapter": pin(FIT_SRC), "frozen_clearance_owner": pin(FROZEN_CLEARANCE),
        "live_incremental_owner": pin(LIVE_CLEARANCE), "full_gate_runner": pin(FULL_GATE),
        "current_audit": pin(fit.CURRENT_AUDIT_DEFAULT), "human_head": head0,
        "input_pin_count": len(pins), "input_hashes": before,
        "all_target_pairs_by_pose": all_counts, "candidate_nonocular_pair_total": residual,
        "attempt4_nonocular_pair_total": a4_total, "strict_pair_reduction": a4_total - residual,
        "pose_steps": POSES, "fixed_bed_vertices": 32, "preserved_thorax_anchors": 15,
        "max_new_owner_iterations": 1, "backtrack_count": 7,
        "all_17_non_target_gates_pass": True,
    }
    return fit, np.ascontiguousarray(candidate, dtype="<f4"), resume_tables, identity

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    need(PREP.is_dir(), "preparation directory is missing")
    fit, compact, resume_tables, identity = validate()
    identity_bytes = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    identity_sha = hashlib.sha256(identity_bytes).hexdigest()
    if args.check:
        need(not PREFLIGHT.exists(), "preflight already exists; preserve and create a new revision")
        record = {**identity, "schema": "numi.human.skin-source-resume-preflight.v1",
                  "preflight_identity_sha256": identity_sha,
                  "qualification": "Prepare-only; no fit or native run was executed."}
        PREFLIGHT.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({"status": "ready_for_root_review", "preflight": str(PREFLIGHT),
                          "identity_sha256": identity_sha, "remaining_pairs": identity["candidate_nonocular_pair_total"],
                          "A4_pairs": A4_TOTAL, "reduction": identity["strict_pair_reduction"]}, sort_keys=True), flush=True)
        return 0
    need(PREFLIGHT.is_file(), "root-reviewed --check preflight required before fit")
    old = read_json(PREFLIGHT)
    need(old.get("preflight_identity_sha256") == identity_sha
         and {k: old.get(k) for k in identity} == identity, "reviewed preflight differs from current inputs")
    need(not FIT_OUT.exists() and not EXECUTION.exists(), "resume output/execution path already exists")
    launch = {
        "schema": "numi.human.skin-source-resume-launch.v1",
        "status": "launching_one_existing_owner_iteration",
        "preflight_path": str(PREFLIGHT), "preflight_sha256": sha(PREFLIGHT),
        "resume_source_path": str(CANDIDATE), "resume_source_sha256": sha(CANDIDATE),
        "resume_audits_path": str(TABLES), "resume_audits_sha256": sha(TABLES),
        "iteration_budget": 1, "backtrack_count": 7,
        "candidate_nonocular_pairs": identity["candidate_nonocular_pair_total"],
        "attempt4_nonocular_pairs": A4_TOTAL, "human_head": head(),
        "resume_wrapper": pin(Path(__file__)),
        "qualification": "Offline bounded resume only; no native run or final anatomy admission.",
    }
    launch_path = PREP / "resume-launch.json"
    need(not launch_path.exists(), "launch declaration already exists")
    launch_path.write_text(json.dumps(launch, indent=2, sort_keys=True, allow_nan=False) + "\n")
    inputs_before = dict(identity["input_hashes"])
    extra_paths = [CANDIDATE, CANDIDATE_REPORT, AUDIT_REPORT, TABLES, PREFLIGHT, launch_path, Path(__file__)]
    extra_before = {str(p.resolve()): sha(p) for p in extra_paths}

    fit.OUT = FIT_OUT
    original = fit.c.derive_shared_multipose_inferred_clearance
    calls = {"n": 0}
    def resume_once(**kwargs):
        calls["n"] += 1
        need(calls["n"] == 1 and kwargs.get("resume_source_positions") is None
             and kwargs.get("resume_target_audits_by_pose") is None
             and kwargs.get("resume_provenance") is None, "unexpected/multiple resume invocation")
        need(kwargs.get("max_iterations") == 12 and kwargs.get("backtrack_count") == 7,
             "original fit iteration/backtrack parameters changed")
        base = np.asarray(kwargs["source_positions"], dtype="<f4").copy()
        faces = np.asarray(kwargs["faces"], dtype=np.int64)
        referenced = np.unique(faces)
        need(base.shape == (54949, 3) and len(referenced) == 54663
             and np.array_equal(referenced, np.asarray(fit.REFERENCE_IDS, dtype=np.int64)),
             "source-to-compact reference map changed")
        full = base.copy()
        full[referenced] = compact
        unused = np.setdiff1d(np.arange(len(base), dtype=np.int64), referenced, assume_unique=True)
        need(len(unused) == 286 and np.array_equal(full[unused].view("u1"), base[unused].view("u1")),
             "resume lift changed the 286 unreferenced vertices")
        fixed = np.asarray(kwargs["fixed_source_vertex_ids"], dtype=np.int64)
        anchors = np.asarray(kwargs["preserved_source_anchor_vertex_ids"], dtype=np.int64)
        held = np.union1d(fixed, anchors)
        need(len(fixed) == 32 and len(anchors) == 15
             and np.array_equal(full[held].view("u1"), base[held].view("u1")),
             "resume candidate moved fixed bed/thorax anchors")
        kwargs["resume_source_positions"] = full.astype(np.float64)
        kwargs["resume_target_audits_by_pose"] = resume_tables
        kwargs["resume_provenance"] = {
            "source_positions_f32_sha256": hashlib.sha256(compact.tobytes()).hexdigest(),
            "source_positions_path": str(CANDIDATE.resolve()),
            "target_audits_path": str(TABLES.resolve()), "target_audits_sha256": sha(TABLES),
            "candidate_report_path": str(CANDIDATE_REPORT.resolve()),
            "candidate_report_sha256": sha(CANDIDATE_REPORT),
            "full_audit_report_path": str(AUDIT_REPORT.resolve()),
            "full_audit_report_sha256": sha(AUDIT_REPORT),
            "candidate_nonocular_pair_total": identity["candidate_nonocular_pair_total"],
            "attempt4_nonocular_pair_total": A4_TOTAL,
        }
        kwargs["max_iterations"] = 1
        return original(**kwargs)

    fit.c.derive_shared_multipose_inferred_clearance = resume_once
    sys.argv = [str(FIT_SRC), "--fit", "--current-audit-summary", str(fit.CURRENT_AUDIT_DEFAULT)]
    started = time.monotonic()
    rc, error = None, None
    try:
        rc = int(fit.main())
        need(calls["n"] == 1, "existing owner was not invoked exactly once")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        after = {path: sha(path) for path in inputs_before}
        extra_after = {path: sha(path) for path in extra_before}
        head_after = head()
        record = {**launch, "finished_unix_time": time.time(),
                  "elapsed_wall_seconds": time.monotonic() - started, "exit_code": rc, "error": error,
                  "owner_call_count": calls["n"], "input_hashes_before": inputs_before,
                  "input_hashes_after": after, "inputs_unchanged": inputs_before == after,
                  "extra_inputs_before": extra_before, "extra_inputs_after": extra_after,
                  "extra_inputs_unchanged": extra_before == extra_after,
                  "human_head_after": head_after, "human_head_unchanged": head_after == launch["human_head"],
                  "qualification": "One offline owner iteration only; no native run or final admission."}
        EXECUTION.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
