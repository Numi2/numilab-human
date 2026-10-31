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
SOURCE807 = H / "verify_current_source_evidence_807_root.py"
SOURCE807_SHA = "9a539227c072936a95b3a9457405521998fb40e5df6139f08d7877fbb49c3f62"
SCHEMA = "h706.ordinary-i-anchor-fit-diagnostic.v3"

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

def same_pin(a, b):
    return (isinstance(a, dict) and isinstance(b, dict)
            and str(a.get("sha256", "")) == str(b.get("sha256", ""))
            and Path(str(a.get("path", ""))).resolve() == Path(str(b.get("path", ""))).resolve())

def load_case(run_dir, owner):
    """Authenticate 855 no-op wrapper, config, source parent and protected roster."""
    run_dir = Path(run_dir).resolve()
    state_path = run_dir / "state.json"
    raw = json.loads(state_path.read_text())
    allowed = {"NO_VALID_COUPLED_ANCHOR_COLLAPSE", "CANDIDATE_PENDING_775_ALL185"}
    if raw.get("status") not in allowed:
        raise ValueError("candidate state has unsupported 853 status")
    if raw.get("schema") not in (None, "h706.coupled-anchor-collapse-batch.v1"):
        raise ValueError("candidate state has unsupported 781 schema")
    if int(raw.get("accepted_operations", -1)) != 0:
        raise ValueError("read-only fit diagnostic requires a zero-operation roster")
    if not isinstance(raw.get("rejections"), dict):
        raise ValueError("zero-operation state lacks its rejection summary")

    wrapper_path = run_dir.parent / "state-wrapper.json"
    wrapper = json.loads(wrapper_path.read_text())
    if wrapper.get("schema") != "h706.ordinary-I-853-state-run.v1":
        raise ValueError("853 state wrapper schema mismatch")
    if wrapper.get("physical_admission") is not False or int(wrapper.get("native_steps", -1)) != 0:
        raise ValueError("853 state wrapper carries admission state")
    wrapped_run = wrapper.get("candidate_run", {})
    if not same_pin(wrapped_run, pin(state_path)):
        raise ValueError("853 wrapper does not pin this exact no-op state")
    if wrapped_run.get("status") != raw.get("status") or int(wrapped_run.get("accepted_operations", -1)) != 0:
        raise ValueError("853 wrapper candidate status/count mismatch")
    prep_path = verify_ref(wrapper.get("preparation"), "853 state preparation")
    prep = json.loads(prep_path.read_text())
    if prep.get("schema") != "h706.coupled-anchor-collapse-inner-layer-preparation.v1":
        raise ValueError("853 preparation schema mismatch")
    if prep.get("status") != "PASS_PREFLIGHT_ONLY_NO_SOLVER":
        raise ValueError("853 source preflight did not pass")
    if prep.get("physical_admission") is not False or int(prep.get("native_steps", -1)) != 0:
        raise ValueError("853 source preflight carries admission state")

    cfg_path = verify_ref(raw.get("run_config"), "candidate run config")
    cfg_pin = pin(cfg_path)
    if not same_pin(prep.get("run_config"), cfg_pin):
        raise ValueError("preparation does not pin the candidate run config")
    if not same_pin(raw.get("run_config"), cfg_pin):
        raise ValueError("no-op report run-config pin mismatch")
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("schema") != "h706.coupled-anchor-collapse-run-config.v1":
        raise ValueError("781 run config schema mismatch")
    mode = cfg.get("ordinary_inner_I_topology_repair", {})
    if (mode.get("enabled") is not True or mode.get("role") != "I"
            or mode.get("actual_back_I_rows_fixed") is not True
            or mode.get("source_rims_fixed") is not True
            or mode.get("149_I_routes_fixed") is not True
            or mode.get("outer_O_and_D_exact_current_parent") is not True
            or float(mode.get("maximum_displacement_m", -1)) != 5.0e-4
            or float(cfg.get("move_cap_m", -1)) != 5.0e-4):
        raise ValueError("run config does not bind the ordinary-I 0.5 mm protected policy")
    if "inner_layer_topology_repair" in cfg or "inner_layer_audit_baseline_candidates" in cfg:
        raise ValueError("ordinary-I run config contains a different inner-layer policy")

    producer = raw.get("producer_853", {})
    if (Path(str(producer.get("path", ""))).resolve() != PRODUCER.resolve()
            or producer.get("sha256") != PRODUCER_SHA):
        raise ValueError("no-op report does not pin the exact 853 producer")

    parent_path = verify_ref(prep.get("state"), "immediate source state")
    parent = json.loads(parent_path.read_text())
    if parent.get("physical_admission") is not False or int(parent.get("native_steps", -1)) != 0:
        raise ValueError("immediate source state is not construction-only")
    source_verify = prep.get("source_807_verification", {})
    if (source_verify.get("status") != "PASS_CURRENT_EXACT_SOURCE_EVIDENCE"
            or source_verify.get("audit_mode") != "AUDITED_BOUNDED_INNER_BACK_LAYER_MOVEMENT_630_805"
            or not same_pin(source_verify.get("state"), pin(parent_path))):
        raise ValueError("preparation lacks a matching passing 807 source verification")
    impl = verify_ref(source_verify.get("implementation"), "807 implementation")
    if impl != SOURCE807.resolve() or sha(impl) != SOURCE807_SHA:
        raise ValueError("preparation source verification uses an unexpected 807 owner")
    proof_ref = cfg.get("inner_layer_source_proof", {})
    if (proof_ref.get("mode") != source_verify.get("audit_mode")
            or not same_pin(proof_ref, pin(parent_path))):
        raise ValueError("run config does not bind the verified immediate source state")
    if not same_pin(prep.get("source_807"), proof_ref):
        raise ValueError("preparation and config source verification pins differ")

    parent_sources = parent.get("source_surfaces", {})
    parent_volumes = {r: parent.get(r + "_current") for r in ("Skin", "Fat")}
    if set(parent_sources) != {"O", "I", "D"} or any(not isinstance(v, dict) for v in parent_volumes.values()):
        raise ValueError("immediate source state lacks complete O/I/D and Skin/Fat pins")
    if not same_pin(cfg.get("surface_I"), parent_sources["I"]):
        raise ValueError("run surface I differs from immediate parent")
    for role in ("O", "I", "D"):
        if not same_pin(cfg.get("baseline_surfaces", {}).get(role), parent_sources[role]):
            raise ValueError("run baseline differs from immediate parent " + role)
    for role in ("Skin", "Fat"):
        if not same_pin(cfg.get("volumes", {}).get(role), parent_volumes[role]):
            raise ValueError("run volume differs from immediate parent " + role)
    if not same_pin(prep.get("prior_805_context"), parent.get("exact_context")):
        raise ValueError("preparation context does not match immediate parent exact context")
    inputs = raw.get("source_inputs", {})
    if not same_pin(inputs.get("surface_I"), cfg["surface_I"]):
        raise ValueError("no-op raw source_I pin differs from run config")
    for role in ("O", "I", "D"):
        if not same_pin(inputs.get("baseline_surfaces", {}).get(role), cfg["baseline_surfaces"][role]):
            raise ValueError("no-op raw baseline pin mismatch for " + role)
    for role in ("Skin", "Fat"):
        if not same_pin(inputs.get("volumes", {}).get(role), cfg["volumes"][role]):
            raise ValueError("no-op raw volume pin mismatch for " + role)

    protection_input = verify_ref(prep.get("protection_input"), "authenticated source protection input")
    source_protection = owner.M.load_pinned_npz({"path":str(protection_input),"sha256":sha(protection_input)},
                                                "authenticated source protection input")
    protected_npz = owner.M.load_pinned_npz(cfg["protected_surface_rows"], "853 protected I rows")
    back = np.asarray(source_protection.get("authenticated_inner_back_rows", []), np.int64).reshape(-1)
    if (not len(back) or len(np.unique(back)) != len(back)
            or not bool(np.asarray(source_protection.get("authenticated_inner_back_rows_candidate_eligible", [False])).reshape(-1)[0])):
        raise ValueError("source back rows are not authenticated or candidate-eligible")
    for field, source_field in (("actual_back_I_rows","authenticated_inner_back_rows"),
                                ("source_rim_rows","source_rim_rows"),("route_anchor_rows","route_anchor_rows")):
        if field not in protected_npz or source_field not in source_protection:
            raise ValueError("protected I archive lacks " + field)
        if not np.array_equal(np.asarray(protected_npz[field],np.int64),
                              np.asarray(source_protection[source_field],np.int64)):
            raise ValueError("protected I archive differs from authenticated source rows: " + field)
    if not bool(np.asarray(protected_npz.get("actual_back_I_rows_fixed",[False])).reshape(-1)[0]):
        raise ValueError("actual back I rows are not explicitly fixed")
    expected_protected = np.unique(np.concatenate([
        np.asarray(source_protection["protected_surface_rows"],np.int64).reshape(-1),
        np.asarray(source_protection["authenticated_inner_back_rows"],np.int64).reshape(-1),
        np.asarray(source_protection["source_rim_rows"],np.int64).reshape(-1),
        np.asarray(source_protection["route_anchor_rows"],np.int64).reshape(-1)]))
    if not np.array_equal(np.asarray(protected_npz.get("protected_surface_rows",[]),np.int64), expected_protected):
        raise ValueError("853 protected complete-star union differs from authenticated back/rim/route rows")
    roster_path = verify_ref(raw.get("effective_candidate_edges"), "effective candidate edge roster")
    if roster_path != (run_dir / "effective-candidate-edges.npy").resolve():
        raise ValueError("effective edge roster path is not inside candidate run")
    if int(raw.get("candidate_edge_count", -1)) != len(np.load(roster_path,allow_pickle=False)):
        raise ValueError("reported candidate edge count differs from pinned roster")
    return {"run_dir":run_dir,"state_path":state_path,"raw":raw,"wrapper":wrapper,
            "wrapper_path":wrapper_path,"prep":prep,"prep_path":prep_path,
            "parent_path":parent_path,"parent":parent,"config_path":cfg_path,"config":cfg,
            "protection":protected_npz,"back_rows":back,"protected_rows":np.asarray(protected_npz["protected_surface_rows"],np.int64),
            "routes":np.asarray(protected_npz["route_anchor_rows"],np.int64),
            "rim_rows":np.asarray(protected_npz["source_rim_rows"],np.int64),
            "roster_path":roster_path}

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

def preflight_to_fit_call(run_dir, output_path, time_limit_s=8.0):
    """Load/authenticate actual no-op inputs and stop immediately before SLSQP."""
    start = time.monotonic()
    owner = load_owner()
    case = load_case(run_dir, owner)
    loaded = owner.M.load_config(case["config_path"])
    owner.BACK_ROWS = set(map(int, case["back_rows"]))
    cache = {r: owner.M.cache_volume(loaded["volumes"][r]) for r in ("Skin","Fat")}
    roster = np.asarray(np.load(case["roster_path"],allow_pickle=False),np.int64)
    if roster.ndim != 2 or roster.shape[1] != 2:
        raise ValueError("authenticated effective edge roster has invalid dimensions")
    fit_namespace = owner.U.BASE_PREPARE.__globals__
    original_fit = fit_namespace.get("fit_common_anchor")
    if not callable(original_fit):
        raise ValueError("pinned 781 prepare owner exposes no common-anchor fit function")
    class ReachedFit(Exception):
        def __init__(self, details):
            self.details=details
    def stop_before_solve(*args,**kwargs):
        fixed=args[1]
        raise ReachedFit({
            "fixed_face_count_by_material":{r:int(len(fixed[r])) for r in ("Skin","Fat")},
            "move_cap_m":float(kwargs.get("move_cap_m",loaded["cfg"].get("move_cap_m",float("nan")))),
            "paired_volume":bool(kwargs.get("paired_volume",True)),
            "SLSQP_executed":False})
    fit_namespace["fit_common_anchor"]=stop_before_solve
    prior=[]
    reached=None
    try:
        for idx,(keep,drop) in enumerate(roster):
            if time.monotonic()-start >= float(time_limit_s):
                break
            try:
                result=owner.M.prepare_operation(loaded["surface"],cache,int(keep),int(drop),
                    protected_surface_rows=loaded["protected"],route_anchor_rows=loaded["routes"],
                    move_cap_m=float(loaded["cfg"]["move_cap_m"]))
                prior.append({"index":idx,"keep":int(keep),"drop":int(drop),"status":result.get("status")})
            except ReachedFit as signal:
                reached={"index":idx,"keep":int(keep),"drop":int(drop),**signal.details}
                break
    finally:
        fit_namespace["fit_common_anchor"]=original_fit
    status="PASS_ACTUAL_855_INPUTS_REACHED_FIT_CALL_WITHOUT_SOLVING" if reached else "NO_FIT_CALL_WITHIN_TIME_BOUND"
    report={"schema":"h706.ordinary-i-anchor-fit-preflight.v1","status":status,
        "candidate_state":pin(case["state_path"]),"wrapper":pin(case["wrapper_path"]),
        "preparation":pin(case["prep_path"]),"immediate_parent_state":pin(case["parent_path"]),
        "run_config":pin(case["config_path"]),"effective_edge_roster":pin(case["roster_path"]),
        "authenticated_back_rows":int(len(case["back_rows"])),
        "protected_rows":int(len(case["protected_rows"])),"rim_rows":int(len(case["rim_rows"])),
        "route_rows":int(len(case["routes"])),"pre_fit_edge_statuses":prior,"fit_call":reached,
        "elapsed_s":time.monotonic()-start,"time_limit_s":float(time_limit_s),
        "geometry_written":False,"SLSQP_executed":False,"physical_admission":False,"native_steps":0}
    out=Path(output_path).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+chr(10))
    return report

def diagnose(run_dir, output_path, time_limit_s=8.0, probe_limit=64):
    start = time.monotonic()
    run_dir = Path(run_dir).resolve()
    owner = load_owner()
    case = load_case(run_dir, owner)
    state_path = case["state_path"]
    raw = case["raw"]
    cfg_path = case["config_path"]
    cfg_raw = case["config"]
    fit_namespace = owner.U.BASE_PREPARE.__globals__
    original_fit = fit_namespace.get("fit_common_anchor")
    if not callable(original_fit):
        raise ValueError("pinned 781 prepare owner exposes no common-anchor fit function")
    x = owner.M.load_config(cfg_path)
    back = case["back_rows"]
    protected = case["protected_rows"]
    routes = case["routes"]
    rims = case["rim_rows"]
    owner.BACK_ROWS = set(map(int, back))

    edge_path = case["roster_path"]
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
    ap.add_argument("--preflight-only", action="store_true")
    a = ap.parse_args()
    if a.control:
        print(json.dumps(control(a.output), indent=2))
    else:
        if a.run_dir is None:
            ap.error("--run-dir is required unless --control is used")
        fn=preflight_to_fit_call if a.preflight_only else diagnose
        report=(fn(a.run_dir,a.output,a.seconds) if a.preflight_only
                else fn(a.run_dir,a.output,a.seconds,a.probe_limit))
        print(json.dumps(report,indent=2,sort_keys=True,default=str))

if __name__ == "__main__":
    main()

