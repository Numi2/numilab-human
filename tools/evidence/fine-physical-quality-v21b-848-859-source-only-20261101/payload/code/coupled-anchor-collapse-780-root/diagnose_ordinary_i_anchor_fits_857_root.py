#!/usr/bin/env python3
"""Read-only diagnostic for 853 ordinary-I anchor-fit rejects.

Replays the bounded candidate edge roster in memory, captures the existing
781/790 fit result, then runs diagnostic-only per-material and no-paired-volume
variants for rejected fits. It writes one JSON report and never writes geometry.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
H = HERE.parent
PRODUCER = HERE / "coupled_anchor_collapse_853_ordinary_i_500um.py"
PRODUCER_SHA = "abd9f62d1269be05270fe117b52b64fe696b3e54ad3662627ab7d347d9a41316"
PASS_ENGINE = HERE / "coupled_anchor_collapse_781.py"
PASS_ENGINE_SHA = "739e41b0fb4a3bd7d7eff2b9688657bad325d294185f7ae22a25af9aa2b186eb"
SCHEMA = "h706.ordinary-i-anchor-fit-diagnostic.v2"

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def pin(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": sha(p)}

def verify_ref(ref, label):
    if not isinstance(ref, dict) or not ref.get("path") or not ref.get("sha256"):
        raise ValueError(label + " lacks path/SHA")
    p = Path(ref["path"]).resolve()
    if not p.is_file() or sha(p) != ref["sha256"]:
        raise ValueError(label + " pin mismatch: " + str(p))
    return p

def load_owner():
    if sha(PRODUCER) != PRODUCER_SHA:
        raise ValueError("pinned 853 producer changed")
    if sha(PASS_ENGINE) != PASS_ENGINE_SHA:
        raise ValueError("pinned 781 engine changed")
    spec = importlib.util.spec_from_file_location("diagnostic_853_pinned", PRODUCER)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import pinned 853 producer")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def fit_summary(fit):
    if not isinstance(fit, dict):
        return {"status": "NO_FIT_RESULT"}
    keys = ("status", "solver_success", "message", "continuous_common_slack_m",
            "continuous_anchor_m", "tested_f32_candidates", "ulp_radius",
            "minimum_new_native_q", "anchor_displacement_m", "move_cap_m",
            "construction_q_margin_multiplier")
    return {k: fit[k] for k in keys if k in fit}

def classify_failure(joint, skin, fat, no_pair):
    js = joint.get("status", "UNKNOWN")
    if js == "PASS":
        return "JOINT_FIT_PASS"
    if skin.get("status") == "PASS" and fat.get("status") == "PASS":
        if no_pair.get("status") == "PASS":
            return "PAIRED_VOLUME_EQUALITY_OR_EXACT_PAIR_F32_GATE_IS_LIMITING"
        return "CROSS_MATERIAL_SHARED_APEX_OR_NUMERICAL_FEASIBILITY_IS_LIMITING"
    if skin.get("status") != "PASS" and fat.get("status") != "PASS":
        return "BOTH_MATERIAL_STARS_FAIL_INDEPENDENTLY_UNDER_CURRENT_CAP_AND_GATES"
    return "ONE_MATERIAL_STAR_FAILS_INDEPENDENTLY_UNDER_CURRENT_CAP_AND_GATES"

def control(output_path):
    cases = [
        ({"status":"REJECT_NO_F32_CANDIDATE"},
         {"status":"PASS"},{"status":"PASS"},{"status":"PASS"},
         "PAIRED_VOLUME_EQUALITY_OR_EXACT_PAIR_F32_GATE_IS_LIMITING"),
        ({"status":"REJECT_NO_F32_CANDIDATE"},
         {"status":"PASS"},{"status":"PASS"},{"status":"REJECT_NO_F32_CANDIDATE"},
         "CROSS_MATERIAL_SHARED_APEX_OR_NUMERICAL_FEASIBILITY_IS_LIMITING"),
        ({"status":"REJECT_NO_F32_CANDIDATE"},
         {"status":"PASS"},{"status":"REJECT_NO_F32_CANDIDATE"},{"status":"REJECT_NO_F32_CANDIDATE"},
         "ONE_MATERIAL_STAR_FAILS_INDEPENDENTLY_UNDER_CURRENT_CAP_AND_GATES"),
    ]
    for joint, skin, fat, no_pair, expected in cases:
        got = classify_failure(joint, skin, fat, no_pair)
        if got != expected:
            raise AssertionError((got, expected))
    report = {"schema": SCHEMA, "status":"PASS_SYNTHETIC_CLASSIFIER_CONTROLS",
              "cases":len(cases), "geometry_loaded":False, "solver_invoked":False,
              "physical_admission":False, "native_steps":0}
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report

def diagnose(run_dir, output_path, time_limit_s=8.0, probe_limit=64):
    start = time.monotonic()
    run_dir = Path(run_dir).resolve()
    state_path = run_dir / "state.json"
    raw = json.loads(state_path.read_text())
    if raw.get("schema") != "h706.coupled-anchor-collapse-batch.v1":
        raise ValueError("candidate state is not the 781 paired raw output schema")
    if raw.get("status") not in ("NO_VALID_COUPLED_ANCHOR_COLLAPSE",
                                  "CANDIDATE_PENDING_775_ALL185"):
        raise ValueError("candidate state is not a recorded 781 candidate/rejection")
    cfg_path = verify_ref(raw.get("run_config"), "run config")
    cfg_raw = json.loads(cfg_path.read_text())
    if cfg_raw.get("ordinary_inner_I_topology_repair", {}).get("enabled") is not True:
        raise ValueError("run config is not the declared ordinary-I 853 mode")
    producer_ref = raw.get("producer_853")
    if not isinstance(producer_ref, dict):
        producer_ref = cfg_raw.get("producer_853")
    owner = load_owner()
    fit_namespace = owner.U.BASE_PREPARE.__globals__
    original_fit = fit_namespace.get("fit_common_anchor")
    if not callable(original_fit):
        raise ValueError("pinned 781 prepare owner exposes no common-anchor fit function")
    if raw.get("accepted_operations", 0) != 0:
        raise ValueError("diagnostic is restricted to zero-operation 853 runs")

    x = owner.M.load_config(cfg_path)
    prot = owner.M.load_pinned_npz(cfg_raw["protected_surface_rows"],
                                   "diagnostic current I protected rows")
    back = np.asarray(prot.get("actual_back_I_rows", []), np.int64).reshape(-1)
    protected = np.asarray(prot.get("protected_surface_rows", []), np.int64).reshape(-1)
    routes = np.asarray(prot.get("route_anchor_rows", []), np.int64).reshape(-1)
    rims = np.asarray(prot.get("source_rim_rows", []), np.int64).reshape(-1)
    if not len(back) or not bool(np.asarray(prot.get("actual_back_I_rows_fixed", [False])).reshape(-1)[0]):
        raise ValueError("853 protected archive lacks authenticated fixed actual-back rows")
    owner.BACK_ROWS = set(map(int, back))
    if not set(map(int, back)).issubset(set(map(int, protected))):
        raise ValueError("actual back rows are not contained in protected I rows")

    edge_path = run_dir / "effective-candidate-edges.npy"
    if not edge_path.is_file():
        raise ValueError("781 effective edge roster is missing")
    edges = np.asarray(np.load(edge_path, allow_pickle=False), np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("effective edge roster is malformed")
    cache = {r: owner.M.cache_volume(x["volumes"][r]) for r in ("Skin", "Fat")}

    engine = owner.M
    fit_calls = []
    def capture_fit(*args, **kwargs):
        result = original_fit(*args, **kwargs)
        fit_calls.append((args, dict(kwargs), result))
        return result
    fit_namespace["fit_common_anchor"] = capture_fit
    deadline = start + float(time_limit_s)
    records = []
    probe_count = 0
    try:
        for idx, edge in enumerate(edges):
            if time.monotonic() >= deadline:
                break
            keep, drop = map(int, edge)
            before = len(fit_calls)
            op = engine.prepare_operation(
                x["surface"], cache, keep, drop,
                protected_surface_rows=x["protected"], route_anchor_rows=x["routes"],
                move_cap_m=float(cfg_raw["move_cap_m"]))
            inner = fit_calls[before] if len(fit_calls) > before else None
            record = {"orientation_index": idx, "keep": keep, "drop": drop,
                      "status": op.get("status")}
            if op.get("status") == "REJECT_ANCHOR_FIT":
                joint = op.get("fit", {})
                record["joint_fit"] = fit_summary(joint)
                if inner is not None and probe_count < int(probe_limit):
                    args, kwargs, captured = inner
                    # Diagnostic-only: duplicate one material's star into the two
                    # required slots; preserve its q/volume/cap constraints while
                    # removing only the pair equality to test single-material fit.
                    single = {}
                    for role in ("Skin", "Fat"):
                        f, old, exact, local, regional = args[1:6]
                        duplicated = {r: f[role] for r in ("Skin", "Fat")}
                        kw = dict(kwargs)
                        kw["paired_volume"] = False
                        solo = original_fit(args[0], duplicated,
                            {r: old[role] for r in ("Skin", "Fat")},
                            {r: exact[role] for r in ("Skin", "Fat")},
                            {r: local[role] for r in ("Skin", "Fat")},
                            {r: regional[role] for r in ("Skin", "Fat")}, **kw)
                        single[role] = fit_summary(solo)
                        probe_count += 1
                        if time.monotonic() >= deadline or probe_count >= int(probe_limit):
                            break
                    record["single_material_fits"] = single
                    no_pair = None
                    if len(single) == 2 and time.monotonic() < deadline and probe_count < int(probe_limit):
                        kw = dict(kwargs)
                        kw["paired_volume"] = False
                        no_pair = original_fit(*args, **kw)
                        record["joint_fit_without_pair_equality"] = fit_summary(no_pair)
                        probe_count += 1
                    record["diagnostic_classification"] = classify_failure(
                        joint,
                        single.get("Skin", {}),
                        single.get("Fat", {}),
                        no_pair or {})
                else:
                    record["diagnostic_classification"] = "JOINT_FIT_CAPTURED; MATERIAL_PROBES_NOT_RUN"
            records.append(record)
            if time.monotonic() >= deadline:
                break
    finally:
        fit_namespace["fit_common_anchor"] = original_fit

    counts = {}
    fit_counts = {}
    classifications = {}
    for r in records:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] == "REJECT_ANCHOR_FIT":
            fs = r.get("joint_fit", {}).get("status", "NO_FIT_STATUS")
            fit_counts[fs] = fit_counts.get(fs, 0) + 1
            cl = r.get("diagnostic_classification", "NO_CLASSIFICATION")
            classifications[cl] = classifications.get(cl, 0) + 1
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": SCHEMA,
        "status": "PASS_READ_ONLY_FIT_DIAGNOSTIC" if len(records) == len(edges) else "PARTIAL_TIME_LIMIT",
        "candidate_state": pin(state_path),
        "run_config": pin(cfg_path),
        "producer_853": pin(PRODUCER),
        "fit_engine_781": pin(PASS_ENGINE),
        "effective_edge_roster": pin(edge_path),
        "raw_rejections": raw.get("rejections", {}),
        "recorded_edge_count": int(len(edges)),
        "diagnosed_edge_count": int(len(records)),
        "top_level_status_counts": counts,
        "joint_fit_status_counts": fit_counts,
        "diagnostic_classification_counts": classifications,
        "fit_probes_run": int(probe_count),
        "probe_limit": int(probe_limit),
        "diagnostic_only_constraint_changes": [
            "per-material probes duplicate that material's full local star into both solver roles",
            "paired-volume-equality probe disables only paired_volume; q, exact positivity checks, per-role local/regional volume budgets and movement cap are unchanged",
            "none of these diagnostic results are candidate acceptance or feasibility certificates"
        ],
        "elapsed_s": time.monotonic() - start,
        "time_limit_s": float(time_limit_s),
        "physical_admission": False,
        "native_steps": 0,
        "edge_results": records,
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    return report

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--probe-limit", type=int, default=64)
    ap.add_argument("--control", action="store_true")
    a = ap.parse_args()
    if a.control:
        print(json.dumps(control(a.output), indent=2))
    else:
        if a.run_dir is None:
            ap.error("--run-dir is required unless --control is used")
        print(json.dumps(diagnose(a.run_dir, a.output, a.seconds, a.probe_limit),
                         indent=2, sort_keys=True, default=str))

if __name__ == "__main__":
    main()

