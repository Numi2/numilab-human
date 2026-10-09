#!/usr/bin/env python3
"""Read-only correction of contact-iteration sensitivity summary 001."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import pathlib
import struct
import sys

EVIDENCE = pathlib.Path("/Users/n/numi-human-resting-evidence-20261005/native-contact-iteration-sensitivity-1181")
RUNS = {
    "32": EVIDENCE / "iteration-32/output/scene",
    "64": pathlib.Path("/Users/n/numi-human-retained-delivery-20261009/support-drift-baseline-020/native-run"),
}
OUT = EVIDENCE / "sensitivity-summary-003"
OLD = EVIDENCE / "sensitivity-summary-001/summary.json"
OLD_CMP = EVIDENCE / "comparison-32-vs-020/comparison.json"
OLD_REVALIDATION = EVIDENCE / "sensitivity-summary-002/corrected-summary-v2.json"
OLD_ROOT_FRAME = EVIDENCE / "root-frame-32-001/report.json"
OLD_MOTION_ALL = EVIDENCE / "motion-32-10-310/report.json"
OLD_MOTION_LATE = EVIDENCE / "motion-32-250-310/report.json"
OLD_OBSERVER = EVIDENCE / "observer-integrity-32-002/report.json"
BUILD_PINS = pathlib.Path("/Users/n/numi-human-support-drift-observer-build-020/evidence/build-pins.json")
SOURCE = pathlib.Path("/Users/n/numi-human-support-drift-observer-020/apps/numilab_human_myosim_visual_probe.mm")
DT_ARG = 0.002
STEP_COUNT = 155000
POST_INIT_SECONDS = 10.0
LATE_SECONDS = 250.0


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: pathlib.Path):
    return json.loads(path.read_text())


def load_csv(path: pathlib.Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def window_rows(rows, dt):
    result = {"all_run": [], "post_initialization_ge_10s": [], "late_ge_250s": []}
    for row in rows:
        step_field = "step" if "step" in row else "accepted_step"
        step = int(row[step_field])
        actual_time = step * dt
        result["all_run"].append((row, step, actual_time))
        if actual_time >= POST_INIT_SECONDS:
            result["post_initialization_ge_10s"].append((row, step, actual_time))
        if actual_time >= LATE_SECONDS:
            result["late_ge_250s"].append((row, step, actual_time))
    return result


def stats(values):
    vals = [float(v) for v in values]
    if not vals or not all(math.isfinite(v) for v in vals):
        raise ValueError("empty or nonfinite statistic input")
    return {"count": len(vals), "min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals), "max_abs": max(abs(x) for x in vals)}


def extrema_with_step(records, field, mode="max"):
    vals = [(float(row[field]), step, t) for row, step, t in records]
    if not vals or not all(math.isfinite(v) for v, _, _ in vals):
        raise ValueError(f"empty or nonfinite values for {field}")
    chosen = (max if mode == "max" else min)(vals, key=lambda x: x[0])
    return {"count": len(vals), "min": min(x[0] for x in vals), "max": max(x[0] for x in vals),
            "extremum_step": chosen[1], "extremum_time_s": chosen[2], "extremum": chosen[0]}


def argv_value(argv, flag):
    i = argv.index(flag)
    return argv[i + 1]


def summarize_run(label, scene, dt, run32_meta, run64_meta):
    md_path = scene / "run-metadata.json"
    md = load_json(md_path)
    if md.get("exit_code") != 0 or md.get("source_files_changed_during_run") != []:
        raise ValueError(f"{label} native receipt is not a clean successful execution")
    runtime = md.get("loaded_metal_runtime", {})
    if not runtime.get("verified"):
        raise ValueError(f"{label} loaded runtime is not verified")
    argv = md.get("argv", [])
    if float(argv_value(argv, "--muscle-step-seconds")) != DT_ARG or int(argv_value(argv, "--muscle-step-count")) != STEP_COUNT:
        raise ValueError(f"{label} horizon/timestep mismatch")
    iterations = int(argv_value(argv, "--stand-contact-iterations"))
    if iterations != int(label):
        raise ValueError(f"{label} argv iteration count mismatch")
    if len(argv) != len(run32_meta["argv"]) or len(argv) != len(run64_meta["argv"]):
        raise ValueError("native argv length changed")
    return md


def analyze_contact_stream(scene, dt):
    coupled_path = scene / "resting-coupled.csv"
    momentum_path = scene / "resting-com-momentum-diagnostic.csv"
    support_path = scene / "resting-com-support-impulses.csv"
    coupled = load_csv(coupled_path)
    momentum = load_csv(momentum_path)
    if len(coupled) != 19375 or len(momentum) != 19375:
        raise ValueError("unexpected accepted observer cadence/count")
    all_steps = [int(r["step"]) for r in coupled]
    if all_steps != list(range(8, STEP_COUNT + 1, 8)):
        raise ValueError("accepted sample steps are not the declared 8-step cadence")
    windows = window_rows(coupled, dt)
    fields = [
        "pre_projection_contact_residual_m_s", "post_projection_contact_residual_m_s",
        "pre_projection_limit_residual_generalized_s", "post_projection_limit_residual_generalized_s",
        "pre_projection_equality_residual_generalized_s", "post_projection_equality_residual_generalized_s",
        "equality_position_projection_max_generalized", "equality_velocity_projection_max_generalized_s",
    ]
    contact_windows = {}
    for name, records in windows.items():
        contact_windows[name] = {f: extrema_with_step(records, f, "max") for f in fields}
        contact_windows[name]["min_contact_gap_m"] = extrema_with_step(records, "min_contact_gap_m", "min")
        contact_windows[name]["peak_penetration_m"] = extrema_with_step(records, "peak_penetration_m", "max")
        contact_windows[name]["rows"] = len(records)
        contact_windows[name]["time_bounds_s"] = [records[0][2], records[-1][2]]

    # These observer records are emitted every 8 accepted roots, but each impulse is only
    # from the final physical update at that endpoint. The source divides it by Float32(dt).
    grouped_impulse = {}
    grouped_count = {}
    positive = set()
    support_rows = 0
    with support_path.open(newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["accepted_step"])
            if int(row["segment_steps"]) != 8:
                raise ValueError("support impulse stream is not the declared 8-step cadence")
            key = (step, int(row["contact_index"]))
            if key in positive:
                raise ValueError("duplicate support impulse key")
            impulse = float(row["normal_impulse_ns"])
            if not math.isfinite(impulse) or impulse < 0:
                raise ValueError("invalid normal impulse")
            grouped_impulse[step] = grouped_impulse.get(step, 0.0) + impulse
            grouped_count[step] = grouped_count.get(step, 0) + 1
            if impulse > 0:
                positive.add(key)
            support_rows += 1
    if support_rows != 620000 or set(grouped_impulse) != set(all_steps) or any(v != 32 for v in grouped_count.values()):
        raise ValueError("support impulse stream does not cover all 32 contacts at every accepted sample")
    momentum_by_step = {int(r["accepted_step"]): r for r in momentum}
    direct_force = {step: float(r["normal_force_last_physical_step_n"]) for step, r in momentum_by_step.items()}
    sampled_force_from_impulses = {step: grouped_impulse[step] / dt for step in grouped_impulse}
    max_force_crosscheck_error = max(abs(direct_force[s] - sampled_force_from_impulses[s]) for s in all_steps)
    if max_force_crosscheck_error > 1e-3:
        raise ValueError(f"per-contact impulse sum disagrees with owner force field: {max_force_crosscheck_error}")
    forces = {}
    for name, records in windows.items():
        steps = [step for _, step, _ in records]
        force_values = [direct_force[step] for step in steps]
        forces[name] = {
            **stats(force_values),
            "sampled_mean_normal_force_n": sum(force_values) / len(force_values),
            "sampled_mean_from_contact_impulse_over_float32_dt_n": sum(grouped_impulse[s] for s in steps) / (len(steps) * dt),
            "sampled_steps": len(steps),
            "cadence_steps": 8,
            "sample_period_s": 8 * dt,
            "normal_impulse_scope": "final physical step only at each 8-step callback, not a sum over the full interval",
        }
    # Positive normal impulse joined to the same-step pre-step point-J*v stream.
    point_path = scene / "resting-support-point-motion.csv"
    jv = {"all_run": [], "post_initialization_ge_10s": [], "late_ge_250s": []}
    point_rows = 0
    with point_path.open(newline="") as f:
        for row in csv.DictReader(f):
            point_rows += 1
            if row.get("velocity_basis") != "pre_step_point_J_times_stand_previous_velocity":
                raise ValueError("support tangent velocity basis changed")
            step = int(row["accepted_step"])
            key = (step, int(row["contact_index"]))
            if key in positive:
                speed = float(row["tangential_speed_m_s"])
                if not math.isfinite(speed) or speed < 0:
                    raise ValueError("invalid tangent speed")
                jv["all_run"].append(speed)
                if step * dt >= POST_INIT_SECONDS:
                    jv["post_initialization_ge_10s"].append(speed)
                if step * dt >= LATE_SECONDS:
                    jv["late_ge_250s"].append(speed)
    if point_rows != 620000:
        raise ValueError("support point-motion stream coverage changed")
    jv_summary = {name: {**stats(values), "velocity_basis": "pre_step_point_J_times_stand_previous_velocity; not post-solve slip"}
                  for name, values in jv.items()}
    # Finite physiology diagnostics are reported descriptively, not as an acceptance gate.
    physiology = {}
    for name, records in windows.items():
        rows = [r for r, _, _ in records]
        physiology[name] = {field: stats(r[field] for r in rows) for field in
                            ("PaO2_mmhg", "PaCO2_mmhg", "SaO2", "blood_ml",
                             "oxygen_balance_error_stpd_ml", "co2_balance_error_stpd_ml",
                             "root_assistance_n", "root_assistance_nm")}
    return {
        "trace_hashes": {p.name: sha256(p) for p in (coupled_path, momentum_path, support_path, point_path)},
        "accepted_sample_rows": len(coupled),
        "accepted_step_first_last": [all_steps[0], all_steps[-1]],
        "windows": {name: {"rows": len(recs), "first_step": recs[0][1], "last_step": recs[-1][1],
                           "first_actual_time_s": recs[0][2], "last_actual_time_s": recs[-1][2]}
                    for name, recs in windows.items()},
        "contact_residual_and_geometry": contact_windows,
        "support_force_sampling": {
            "method": "per-sample mean of owner normal_force_last_physical_step_n; cross-checked against sum of 32 support contact normal impulses divided by Float32(dt)",
            "max_abs_force_crosscheck_error_n": max_force_crosscheck_error,
            "raw_support_contact_rows": support_rows,
            "contacts_per_sample": 32,
            "windows": forces,
        },
        "positive_impulse_prestep_tangent_speed": {
            "matched_rows": len(jv["all_run"]),
            "raw_support_point_rows": point_rows,
            "windows": jv_summary,
        },
        "physiology_descriptive_only": physiology,
    }


def main():
    if not OUT.is_dir():
        raise SystemExit(f"expected precreated fresh output directory: {OUT}")
    output = OUT / "corrected-summary.json"
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    old_summary = load_json(OLD)
    old_cmp = load_json(OLD_CMP)
    old_revalidation = load_json(OLD_REVALIDATION)
    build_pins = load_json(BUILD_PINS)
    old_summary_sha = sha256(OLD)
    dt = struct.unpack("<f", struct.pack("<f", DT_ARG))[0]
    metas = {label: load_json(scene / "run-metadata.json") for label, scene in RUNS.items()}
    for label, scene in RUNS.items():
        md = summarize_run(label, scene, dt, metas["32"], metas["64"])
    if metas["32"].get("asset_sha256") != metas["64"].get("asset_sha256"):
        raise ValueError("asset hash maps differ")
    argv32, argv64 = metas["32"]["argv"], metas["64"]["argv"]
    argv_diffs = [{"index": i, "iteration_32": a, "iteration_64": b} for i, (a,b) in enumerate(zip(argv32,argv64)) if a != b]
    expected_indices = {4, 41, 47}
    if {x["index"] for x in argv_diffs} != expected_indices:
        raise ValueError(f"unexpected native argv differences: {argv_diffs}")
    if argv32[47] != "32" or argv64[47] != "64":
        raise ValueError("wrong iteration argv difference")
    env32, env64 = metas["32"].get("environment", {}), metas["64"].get("environment", {})
    env_diffs = [{"key": k, "iteration_32": env32.get(k), "iteration_64": env64.get(k)}
                 for k in sorted(set(env32) | set(env64)) if env32.get(k) != env64.get(k)]
    if any(d["key"] != "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT" for d in env_diffs):
        raise ValueError(f"unexpected environment differences: {env_diffs}")
    sim_s = dt * STEP_COUNT
    walls = {label: float(metas[label]["wall_seconds"]) for label in RUNS}
    rtf = {label: sim_s / walls[label] for label in RUNS}
    run_metrics = {label: analyze_contact_stream(scene, dt) for label, scene in RUNS.items()}
    comparison_fields = {}
    for name, item in old_cmp.get("csv_sensitivity_comparison", {}).items():
        comparison_fields[name] = {
            "candidate_rows_complete": item.get("candidate_rows_complete"),
            "reference_rows_complete": item.get("reference_rows_complete"),
            "shared_rows": item.get("shared_rows"),
            "candidate_only_keys": item.get("candidate_only_keys"),
            "reference_only_keys": item.get("reference_only_keys"),
            "numeric_or_text_value_mismatches": item.get("numeric_or_text_value_mismatches"),
            "first_mismatch_step": item.get("first_mismatches", [{}])[0].get("key", [None])[0],
        }
    root_frame = load_json(OLD_ROOT_FRAME)
    input_paths = [OLD, OLD_REVALIDATION, OLD_CMP, OLD_ROOT_FRAME, OLD_MOTION_ALL, OLD_MOTION_LATE, OLD_OBSERVER, BUILD_PINS,
                   SOURCE]
    for scene in RUNS.values():
        input_paths.extend([scene / n for n in ("run-metadata.json", "native.log", "invocation.json",
             "resting-coupled.csv", "resting-com-momentum-diagnostic.csv", "resting-com-support-impulses.csv",
             "resting-support-point-motion.csv", "resting-surface-audit.csv")])
    input_hashes = {}
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        input_hashes[str(path)] = sha256(path)
    source = build_pins["source_checkout"]
    if source.get("path") != str(pathlib.Path(source["path"])):
        raise ValueError("build source path malformed")
    # Verify the principal compiled source file against the frozen build pin, not a newer checkout.
    source_pin = source["files_sha256"].get("apps/numilab_human_myosim_visual_probe.mm")
    source_tree_file = pathlib.Path(source["path"]) / "apps/numilab_human_myosim_visual_probe.mm"
    if not source_pin or sha256(source_tree_file) != source_pin:
        raise ValueError("frozen build source file does not match its build pin")
    profile = load_json(OLD_OBSERVER)
    report = {
        "schema": "numi.human.support-drift-contact-iteration-sensitivity.v2",
        "revision": 3,
        "status": "corrected_measured_iteration_sensitivity_not_adoption",
        "scope": "Read-only revalidation of the already-completed 310 s, 155000-step observer-on runs. Contact iterations are the sole physical argument difference; this is sensitivity evidence, not model qualification or causal proof for drift.",
        "correction_and_supersession": {
            "prior_report_path": str(OLD), "prior_report_sha256": old_summary_sha,
            "previous_revalidation_path": str(OLD_REVALIDATION), "previous_revalidation_sha256": sha256(OLD_REVALIDATION),
            "prior_report_retained_unchanged": True,
            "prior_support_force_error": {
                "prior_field": "time_mean_normal_force_n",
                "prior_values_n": {"32": 88.30346860515755, "64": 88.30107541422399},
                "cause": "The prior calculation divided endpoint-sampled final-physical-step impulses by the full 310 s horizon. The support impulse CSV has one impulse from the last physical step at each 8-step callback, not an impulse integrated across the 8-step segment.",
                "source_confirmation": "The frozen source labels the output as final physical step per segment and writes normal_force_last_physical_step_n = normalImpulse / h, with h = Float32(muscleStepSeconds).",
                "corrected_fields": "support_force_sampling.windows.*.sampled_mean_normal_force_n",
                "denominator": "number_of_endpoint_samples * Float32(0.002 s), not total simulation time",
                "scale_error": 8.0,
            },
            "prior_performance_wording_error": "An earlier status message incorrectly said 32 iterations was slower. The recorded process wall times show 32 completed faster, although its contact residuals and late drift were worse. This correction does not claim the timing difference is an isolated benchmark.",
        },
        "time_and_performance": {
            "requested_step_seconds": DT_ARG,
            "native_float32_step_seconds": dt,
            "accepted_step_count": STEP_COUNT,
            "actual_accepted_time_s": sim_s,
            "normalization": "real-time factor = actual accepted simulation seconds / run-metadata process wall seconds; process-wall/simulation-time is also provided.",
            "process_wall_seconds": walls,
            "real_time_factor_sim_s_per_wall_s": rtf,
            "wall_seconds_per_simulation_second": {label: walls[label] / sim_s for label in RUNS},
            "iteration_32_vs_64": {
                "wall_seconds_saved": walls["64"] - walls["32"],
                "wall_fraction_reduction": 1.0 - walls["32"] / walls["64"],
                "rtf_ratio_32_over_64": rtf["32"] / rtf["64"],
                "interpretation": "32 has higher measured RTF and lower process wall for this run pair. Timing is instrumented and was not isolated from all machine load; it is not a performance qualification.",
            },
        },
        "run_identity": {
            "paths": {label: str(path) for label,path in RUNS.items()},
            "exit_codes": {label: metas[label]["exit_code"] for label in RUNS},
            "source_files_changed_during_run": {label: metas[label]["source_files_changed_during_run"] for label in RUNS},
            "loaded_runtime_verified": {label: metas[label]["loaded_metal_runtime"]["verified"] for label in RUNS},
            "loaded_runtime_sha256": {label: metas[label]["loaded_metal_runtime"]["expected_sha256"] for label in RUNS},
            "argv_differences": argv_diffs,
            "environment_differences": env_diffs,
            "asset_sha256_maps_equal": True,
            "asset_sha256_count": len(metas["32"]["asset_sha256"]),
            "common_physical_inputs": metas["32"]["asset_sha256"],
            "iterations": {"32": 32, "64": 64},
        },
        "measured_run_metrics": run_metrics,
        "trace_comparison_against_64": comparison_fields,
        "observer_profile_timing": profile.get("observer_integrity", {}),
        "root_frame_decomposition": root_frame.get("root_frame_decomposition", {}),
        "interpretation": {
            "decision": "Do not adopt the 32-sweep setting from this sensitivity run. It did run faster (RTF 0.142764 vs 0.112240), but contact and constraint residual maxima were materially worse and the COM/root trajectory diverged from step 8; its 250–310 s COM displacement magnitude was larger. The residual drift is not explained or fixed by reducing contact iterations.",
            "all_run_vs_post_init": "Residual summaries include all observer rows and a separate post-initialization window beginning at accepted step 5000 (Float32-dt time 10.000000475 s); late 250–310 s is also separate. Early penetration and late velocity residuals are not conflated.",
            "contact_force_limit": "Mean support force is a cadence-sampled final-step force, not an interval-integrated force. It agrees between the per-contact normal-impulse sum / Float32(dt) and the owner normal_force_last_physical_step_n field.",
            "jv_limit": "Tangential Jv is pre-step point-J times previous stand velocity at cadence-8 endpoints with positive normal impulse; it is not a post-solve sliding-speed measurement.",
            "physiology_limit": "Gas and blood diagnostics are included as finite descriptive ranges only. Since full traces diverge from the first sampled root, similar ranges do not demonstrate equivalent physiology or qualify either setting.",
        },
        "input_sha256": input_hashes,
        "build_source_binding": {
            "build_pins_path": str(BUILD_PINS),
            "build_pins_sha256": sha256(BUILD_PINS),
            "binary_sha256": build_pins["build"]["binary_sha256"],
            "principal_compiled_source_path": str(source_tree_file),
            "principal_compiled_source_sha256": source_pin,
            "source_checkout_base_revision": source["base_revision"],
            "physical_dylib_sha256": build_pins["unchanged_runtime"]["physical_dylib_sha256"],
            "respiration_metallib_sha256": build_pins["unchanged_runtime"]["respiration_metallib_sha256"],
            "force_formula_source_lines": [22762, 22798, 23440, 23537],
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(output), "sha256": sha256(output),
                      "rtf": rtf, "wall_s": walls,
                      "support_force_mean_n": {k:v["support_force_sampling"]["windows"]["all_run"]["sampled_mean_normal_force_n"] for k,v in run_metrics.items()},
                      "contact_residual_all_post10_late": {k:{w:v["contact_residual_and_geometry"][w]["post_projection_contact_residual_m_s"]["extremum"] for w in ("all_run","post_initialization_ge_10s","late_ge_250s")} for k,v in run_metrics.items()}}, indent=2))

if __name__ == "__main__":
    main()
