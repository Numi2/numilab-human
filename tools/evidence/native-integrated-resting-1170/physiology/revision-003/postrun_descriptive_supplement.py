#!/usr/bin/env python3
"""Read-only descriptive addendum for a closed registered final Human pair.

This keeps the science-v2 registered result authoritative. It reports existing
owner blood/mechanics observations and generic reference comparisons without
changing the protocol, creating a study, running simulation, or qualifying
physiology/anatomy.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

EVIDENCE = Path("/Users/n/numi-human-resting-evidence-20261005")
LAB = Path("/Users/n/numi-human-performance-source-014")
SCIENCE_CLI = LAB / "tools/numi"
P18 = EVIDENCE / "final-integrated-study-readiness-1170/package-v018"
ANALYZER_PATH = P18 / "analyze_final_pair.py"
ANALYZER_SHA256 = "defe06f8ded56d66fe5ba903445bb8dd81ef96ad57fa59e3be4c615c5892cb5c"
PREPARE_PATH = P18 / "prepare_final_plan.py"
PREPARE_SHA256 = "f5d57e46afce90ea756e8eb8e23bfd1f4a6a6e9fcb971659f946ce29768531be"
ADAPTER_PATH = EVIDENCE / "native-lung-late-pose-audit-runner-1171/registered_receipt_adapter_1171.py"
ADAPTER_SHA256 = "ba0b88416dbf9cab892f9daa6f954e931003f9db5075db6b758c209370efc1e8"
OWNER_PATH = LAB / "matter/tools/resting_intervention_study.py"
OWNER_SHA256 = "b06bb5a1cd14d86be362a5d0603f68546ba590bf038f9586c4cf89946fc39c1e"
SMOKE = Path("/Users/n/numi-human-retained-delivery-20261009/terminal-capture-fix-smoke")
SMOKE_VERIFICATION_SHA256 = "24c72cce43add252376ec2d62629ea89aea8c8f98ee5d8c3bcd59c675ec8ce36"
TRACE_FIELDS = (
    "time_s", "step", "blood_ml", "PaO2_mmhg", "PaCO2_mmhg", "SaO2",
    "lung_volume_ml", "alveolar_pa", "pleural_pa", "airflow_ml_s",
    "lv_ml", "rv_ml", "last_lv_stroke_ml", "aortic_ejected_ml",
    "pulmonary_ejected_ml", "complete_filling_ejection_cycles",
)
REQUIRED_NUMERICAL_FIELDS = (
    "respiratory_net_volume_ml", "respiratory_volume_balance_ml",
    "oxygen_balance_error_stpd_ml", "co2_balance_error_stpd_ml",
    "blood_error_ml", "blood_continuity_residual_accum_ml",
    "blood_physical_delta_accum_ml", "blood_residual_minus_physical_ml",
    "blood_endpoint_minus_physical_ml",
)
MECHANICS_FIELDS = (
    "lung_volume_ml", "airflow_ml_s", "alveolar_pa", "pleural_pa",
    "diaphragm_mm", "rib_mm", "diaphragm_excitation", "intercostal_excitation",
    "diaphragm_activation", "intercostal_activation",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot import pinned helper " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_helpers():
    for path, expected in ((ANALYZER_PATH, ANALYZER_SHA256),
                           (PREPARE_PATH, PREPARE_SHA256),
                           (ADAPTER_PATH, ADAPTER_SHA256),
                           (OWNER_PATH, OWNER_SHA256)):
        require(path.is_file() and sha(path) == expected, "pinned helper missing or changed: " + str(path))
    analyzer = load_module(ANALYZER_PATH, "postrun_p18_analyzer")
    prepare = load_module(PREPARE_PATH, "postrun_p18_prepare")
    adapter = load_module(ADAPTER_PATH, "postrun_p18_registered_adapter")
    if str(prepare.HUMAN / "src") not in sys.path:
        sys.path.insert(0, str(prepare.HUMAN / "src"))
    owner = load_module(OWNER_PATH, "postrun_frozen_human_owner")
    require(callable(owner.native_respiration_trace_consistency) and
            callable(owner.resting_reference_comparison), "frozen owner descriptive helpers unavailable")
    return analyzer, prepare, adapter, owner


def number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError("missing or malformed field " + key)
    if not math.isfinite(value):
        raise ValueError("nonfinite field " + key)
    return value


def read_trace(path, required_fields):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        missing = sorted(key for key in required_fields if header.count(key) != 1)
        require(not missing, "trace missing or duplicates required fields: " + ", ".join(missing))
        rows = list(reader)
    require(len(rows) >= 2, "trace must have at least two samples")
    for row in rows:
        for key in required_fields:
            number(row, key)
    times = [number(row, "time_s") for row in rows]
    require(all(b > a for a, b in zip(times, times[1:])), "trace accepted timestamps are not strictly increasing")
    steps = [number(row, "step") for row in rows]
    require(all(float(int(x)) == x for x in steps) and all(b > a for a, b in zip(steps, steps[1:])),
            "trace accepted steps are not strictly increasing integers")
    return header, rows


def select_window(rows, start, end):
    require(math.isfinite(start) and math.isfinite(end) and end > start, "invalid or empty window bounds")
    selected = [row for row in rows if start <= number(row, "time_s") < end]
    require(selected, "no accepted trace samples in [%.6f, %.6f)" % (start, end))
    return selected


def stats(rows, key):
    values = [number(row, key) for row in rows]
    return {"n": len(values), "mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "first": values[0], "last": values[-1]}


def stats_scaled(rows, key, scale, unit):
    require(math.isfinite(scale) and scale > 0.0, "invalid statistic scale")
    values = [number(row, key) * scale for row in rows]
    require(values and all(math.isfinite(value) for value in values), "no finite scaled values for " + key)
    return {"n": len(values), "mean": statistics.fmean(values), "min": min(values), "max": max(values),
            "first": values[0], "last": values[-1], "unit": unit,
            "source_field": key, "source_to_report_scale": scale}


def reference_outlier_counts(rows, field, bounds, *, scale=1.0, unit):
    require(len(bounds) == 2, "reference bounds must have two values")
    lo, hi = bounds
    values = [number(row, field) * scale for row in rows]
    require(values and all(math.isfinite(value) for value in values), "no finite samples for " + field)
    return {"source_field": field, "source_unit": "fraction" if field == "SaO2" else unit,
            "reported_unit": unit, "source_to_report_scale": scale,
            "reference_bounds": list(bounds), "sample_count": len(values),
            "below_lower": sum(value < lo for value in values) if lo is not None else 0,
            "above_upper": sum(value > hi for value in values) if hi is not None else 0,
            "mean": statistics.fmean(values)}


def windows_from_invocation(invocation, steps, dt):
    argv = invocation.get("argv", [])
    try:
        i = argv.index("--start-s")
        start = float(argv[i + 1])
        j = argv.index("--end-s")
        end = float(argv[j + 1])
        k = argv.index("--window-s")
        width = float(argv[k + 1])
    except (ValueError, IndexError):
        raise ValueError("registered invocation lacks exact pre/dose/recovery windows")
    duration = steps * dt
    result = {"pre": [start - width, start], "dose": [end - width, end],
              "recovery": [duration - width, duration]}
    require(result == {"pre": [30.0, 60.0], "dose": [70.0, 100.0], "recovery": [280.0, 310.0]},
            "registered windows differ from P18 frozen final plan")
    return result


def registered_windows(item, steps, dt):
    started = item.get("started", {})
    payload = started.get("payload", {}) if isinstance(started, dict) else {}
    argv = payload.get("argv", [])
    require(argv, "registered trial payload lacks its registered invocation argv")
    return windows_from_invocation({"argv": argv}, steps, dt)


def checked_terminal_time(context, steps):
    geometry = context.get("geometry", {})
    schedule = context.get("schedule", {})
    terminal = schedule.get("terminal", {})
    actual = geometry.get("terminal_accepted_time_s")
    expected = terminal.get("accepted_time_s")
    actual = number({"terminal_time": actual}, "terminal_time")
    expected = number({"schedule_terminal_time": expected}, "schedule_terminal_time")
    require(geometry.get("terminal_step") == steps and abs(actual - expected) <= 1e-12,
            "registered geometry does not expose the exact terminal accepted time")
    return actual


def cycle_multiplicity(rows, start, end):
    selected = [r for r in rows if start <= number(r, "time_s") < end]
    require(len(selected) >= 2, "too few samples for cardiac counter interval summary")
    intervals = []
    for left, right in zip(selected, selected[1:]):
        a = number(left, "complete_filling_ejection_cycles")
        b = number(right, "complete_filling_ejection_cycles")
        increment = b - a
        require(increment >= 0 and float(int(increment)) == increment,
                "cardiac complete-cycle counter is nonmonotonic or noninteger")
        if increment:
            intervals.append({"from_s": number(left, "time_s"), "to_s": number(right, "time_s"),
                              "counter_increment": int(increment)})
    multiplicity = {}
    for event in intervals:
        key = str(event["counter_increment"])
        multiplicity[key] = multiplicity.get(key, 0) + 1
    multi = sum(x["counter_increment"] > 1 for x in intervals)
    if multi:
        interpretation = ("Observed intervals include multiple counter increments (count=%d); do not treat each "
                          "counter-advance interval as exactly one beat." % multi)
    else:
        interpretation = ("Every observed counter-advance interval incremented once; these sampled transition "
                          "intervals remain distinct from the complete-event cycle ledger.")
    return {"counter_advance_intervals": len(intervals),
            "complete_counter_cycles": sum(x["counter_increment"] for x in intervals),
            "interval_multiplicity_histogram": multiplicity,
            "intervals_with_more_than_one_counter_cycle": multi,
            "interpretation": interpretation}


def window_response(rows, windows):
    output = {"descriptive_changes": {}}
    for name, (lo, hi) in windows.items():
        selected = select_window(rows, lo, hi)
        output[name] = {
            "window_s": [lo, hi],
            "accepted_sample_interval_s": [number(selected[0], "time_s"), number(selected[-1], "time_s")],
            "PaO2_mmhg": stats(selected, "PaO2_mmhg"),
            "PaCO2_mmhg": stats(selected, "PaCO2_mmhg"),
            "SaO2": stats(selected, "SaO2"),
            "SaO2_percent": stats_scaled(selected, "SaO2", 100.0, "percent"),
            "lung_volume_ml": stats(selected, "lung_volume_ml"),
            "alveolar_pa": stats(selected, "alveolar_pa"),
            "pleural_pa": stats(selected, "pleural_pa"),
            "airflow_ml_s": stats(selected, "airflow_ml_s"),
            "owner_reference_comparison": None,
        }
    for field in ("PaO2_mmhg", "PaCO2_mmhg", "SaO2", "SaO2_percent"):
        output["descriptive_changes"][field] = {
            "dose_minus_pre": output["dose"][field]["mean"] - output["pre"][field]["mean"],
            "recovery_minus_pre": output["recovery"][field]["mean"] - output["pre"][field]["mean"],
            "unit": output["dose"][field].get("unit", "mmHg" if "mmhg" in field else "fraction"),
        }
    return output


def extract_respiration_config(invocation):
    argv = invocation.get("argv", [])
    try:
        i = argv.index("--resting-scene")
        path = Path(argv[i + 2]).resolve()
    except (ValueError, IndexError):
        raise ValueError("native scene invocation lacks the pinned respiration config argument")
    require(path.is_file() and not path.is_symlink(), "respiration config is missing or symlinked")
    return path


def source_bound_pv_trace(path, rows, windows, arm, output_path):
    lo = min(x[0] for x in windows.values())
    hi = max(x[1] for x in windows.values())
    selected = [r for r in rows if lo <= number(r, "time_s") < hi]
    columns = ["arm", "step", "time_s", "lung_volume_ml", "alveolar_pa", "pleural_pa", "airflow_ml_s",
               "blood_ml", "PaO2_mmhg", "PaCO2_mmhg", "SaO2", "lv_ml", "rv_ml", "last_lv_stroke_ml"]
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow({"arm": arm, **row})
    return {"path": str(path), "sha256": sha(path), "rows": len(selected),
            "source_trace_path": str(Path(path).resolve()),
            "scope": "Raw accepted CSV values within the registered pre/dose/recovery spans; no interpolation or derived flow integration."}


def summarize_registered_arm(context, arm, analyzer, prepare, owner, out):
    item = context["item"]
    rows_path = item["csv"]
    header, rows = read_trace(rows_path, set(TRACE_FIELDS) | set(REQUIRED_NUMERICAL_FIELDS) | set(MECHANICS_FIELDS))
    require(item.get("steps") == 155000 and abs(item.get("dt", 0) - 0.002) <= 1e-12,
            "registered arm must have exactly 155000 accepted steps at requested 2 ms")
    observation = item.get("owner_result", {})
    require(observation.get("accepted_steps") == 155000 and observation.get("duration_valid") is True and
            observation.get("root_assistance_observed") is False,
            "registered owner observation does not prove complete unassisted arm")
    invocation = item.get("inv", {})
    windows = registered_windows(item, item["steps"], item["dt"])
    duration = item["steps"] * item["dt"]
    post = [r for r in rows if number(r, "time_s") >= 10.0 and
            (number(r, "time_s") < duration or number(r, "step") == item["steps"])]
    require(post, "no post-initialization samples including terminal accepted state")
    blood_all = stats(rows, "blood_ml")
    blood_post = stats(post, "blood_ml")
    response = window_response(rows, windows)
    reference = {}
    for name, (lo, hi) in windows.items():
        ref = owner.resting_reference_comparison(rows, lo, hi)
        comparisons = ref.get("general_adult_resting_reference_comparisons", {})
        selected = {}
        for key in ("mean_PaO2_mmhg", "mean_PaCO2_mmhg", "mean_SaO2_percent", "aortic_output_L_min", "mean_last_complete_LV_stroke_ml"):
            if key in comparisons:
                selected[key] = comparisons[key]
        require(selected, "owner reference helper did not expose CO/SV comparisons for " + name)
        response[name]["owner_reference_comparison"] = selected
        reference[name] = ref
    config_path = extract_respiration_config(invocation)
    config_sha = sha(config_path)
    require(invocation.get("asset_sha256", {}).get(str(config_path)) == config_sha,
            "selected respiration config is not bound by the exact native invocation")
    mechanics = owner.native_respiration_trace_consistency(rows_path, config_path, windows)
    post_reference = owner.resting_reference_comparison(rows, 10.0, duration)
    post_general = post_reference.get("general_adult_resting_reference_comparisons", {})
    require(post_general, "owner post-initialization generic reference bounds are missing")
    pao2_bounds = post_general["mean_PaO2_mmhg"]["reference_bounds"]
    paco2_bounds = post_general["mean_PaCO2_mmhg"]["reference_bounds"]
    sao2_bounds = post_general["mean_SaO2_percent"]["reference_bounds"]
    post_gas = {
        "PaO2_mmhg": reference_outlier_counts(post, "PaO2_mmhg", pao2_bounds, unit="mmHg"),
        "PaCO2_mmhg": reference_outlier_counts(post, "PaCO2_mmhg", paco2_bounds, unit="mmHg"),
        "SaO2_percent": reference_outlier_counts(post, "SaO2", sao2_bounds, scale=100.0, unit="percent"),
    }
    numerical = analyzer.numerical_diagnostic_highwater(rows, duration, item["steps"])
    require(set(numerical) == set(REQUIRED_NUMERICAL_FIELDS) and all(math.isfinite(float(v)) for v in numerical.values()),
            "P18 numerical highwater did not return the required finite diagnostic set")
    heart_intervals = {}
    for name, (lo, hi) in windows.items():
        heart_intervals[name] = cycle_multiplicity(rows, lo, hi)
    cycles = analyzer.audit.cardiac(rows)
    breaths = analyzer.audit.respiratory(rows, analyzer.audit.breath_events(rows))
    post_cycles = [c for c in cycles if c["start_s"] >= 10.0 and c["end_s"] <= duration]
    post_breaths = [b for b in breaths if b["start_s"] >= 10.0 and b["end_s"] <= duration]
    breath_windows = {}
    for name, (lo, hi) in windows.items():
        selected_breaths = [b for b in breaths if lo <= b["start_s"] and b["end_s"] <= hi]
        breath_windows[name] = {"complete_intervals": len(selected_breaths),
            "mean_tidal_l": (statistics.fmean(b["vt_l"] for b in selected_breaths) if selected_breaths else None),
            "mean_breath_rate_per_min": (statistics.fmean(b["rr_bpm"] for b in selected_breaths) if selected_breaths else None),
            "scope": "Complete event-ledger cycles wholly contained in the requested window."}
    trace_path = out / (arm + "-registered-pv-trace.csv")
    trace_info = source_bound_pv_trace(rows_path, rows, windows, arm, trace_path)
    return {
        "arm": arm,
        "registered_trial_id": item["started"].get("payload", {}).get("trial", {}).get("id"),
        "accepted_steps": item["steps"], "requested_dt_s": item["dt"],
        "nominal_duration_s": duration,
        "native_terminal_accepted_time_s": checked_terminal_time(context, item["steps"]),
        "root_assistance_observed": observation["root_assistance_observed"],
        "intervention_applied": observation.get("intervention_applied"),
        "delivered_drive_scale": observation.get("delivered_drive_scale"),
        "trace": {"path": str(rows_path), "sha256": sha(rows_path), "columns": len(header), "samples": len(rows)},
        "blood_volume_ml": {"all_samples": blood_all, "post_initialization_10s_to_terminal": blood_post},
        "registered_windows": response,
        "within_arm_gas_changes_mmhg": {
            key: response["descriptive_changes"][key] for key in ("PaO2_mmhg", "PaCO2_mmhg")},
        "respiration_config": {"path": str(config_path), "sha256": config_sha},
        "owner_respiratory_mechanics_identity_check": mechanics,
        "post_initialization_owner_generic_reference_context": post_reference,
        "post_initialization_sample_reference_counts": post_gas,
        "owner_per_step_numerical_diagnostic_highwater": numerical,
        "cardiac_counter_multiplicity_by_window": heart_intervals,
        "sampled_cycle_ledger": {"complete_cardiac_intervals_all_samples": len(cycles),
                                  "complete_cardiac_intervals_post_initialization": len(post_cycles),
                                  "complete_respiratory_intervals_all_samples": len(breaths),
                                  "complete_respiratory_intervals_post_initialization": len(post_breaths),
                                  "respiratory_complete_intervals_by_window": breath_windows,
                                  "scope": "Existing 855 event extraction; cardiac counter intervals can span more than one complete cycle."},
        "owner_generic_reference_context_by_window": reference,
        "source_bound_pv_trace": trace_info,
        "limitations": [
            "Descriptive owner/source context; no clinical, population, anatomy, or physiological qualification.",
            "The registered model does not provide respiratory-to-cardiovascular feedback or full-body modal response.",
            "Cumulative owner-maintained inventory residuals are high-water/error observables and are not summed across exported rows.",
            "CO/SV generic adult ranges are measurement-dependent comparisons, not diagnosis or model admission gates.",
        ],
    }


def run_registered_pair(study, output, analyzer, prepare, adapter, owner):
    study = Path(study).resolve()
    output = Path(output).resolve()
    require(study.is_dir(), "registered study directory is missing")
    require(output != study and study not in output.parents, "supplement output must be outside the registered study")
    require(not output.exists(), "supplement output must be a fresh directory")
    verify = subprocess.run([str(SCIENCE_CLI), "science", "verify", str(study)], cwd=str(LAB),
                            capture_output=True, text=True, check=False)
    require(verify.returncode == 0, "registered science verification failed: " + (verify.stderr or verify.stdout)[-1600:])
    reg_doc, reg = analyzer.audit.read_wrapped(study / "registration.json", "registered study")
    require(reg.get("schema") == "numi.science.registration.v2" and
            reg.get("plan", {}).get("schema") == "numi.science.plan.v2", "unexpected registration schema")
    analysis_doc, analysis = analyzer.audit.read_wrapped(study / "analysis.json", "registered primary analysis")
    require(analysis.get("registration_sha256") == reg_doc.get("sha256") and
            len(analysis.get("paired_differences", [])) == 1,
            "registered primary analysis is missing or does not bind this study")
    trial_map = {"control": study / "trials/resting-baseline", "treatment": study / "trials/resting-drive-half"}
    output.mkdir(parents=True, exist_ok=False)
    per_arm = {}
    contexts = {}
    for arm, trial in trial_map.items():
        context = adapter.load_registered_context(trial, arm)
        contexts[arm] = context
        per_arm[arm] = summarize_registered_arm(context, arm, analyzer, prepare, owner, output)
    official = float(analysis["paired_differences"][0]["treatment_minus_control"])
    require(math.isfinite(official), "registered primary effect is nonfinite")
    manifest_inputs = {
        "registration": {"path": str(study / "registration.json"), "sha256": sha(study / "registration.json")},
        "registered_analysis": {"path": str(study / "analysis.json"), "sha256": sha(study / "analysis.json")},
        "p18_analyzer": {"path": str(ANALYZER_PATH), "sha256": sha(ANALYZER_PATH)},
        "p18_prepare": {"path": str(PREPARE_PATH), "sha256": sha(PREPARE_PATH)},
        "registered_receipt_adapter": {"path": str(ADAPTER_PATH), "sha256": sha(ADAPTER_PATH)},
        "frozen_owner": {"path": str(OWNER_PATH), "sha256": sha(OWNER_PATH)},
        "855_analyzer": {"path": str(analyzer.ANALYZER_PATH), "sha256": sha(analyzer.ANALYZER_PATH)},
    }
    for arm, context in contexts.items():
        manifest_inputs[arm + "_registered_context_files"] = context["input_hashes"]
    result = {
        "schema": "numi.human.postrun-descriptive-supplement.v1",
        "status": "complete_descriptive_addendum",
        "qualification": "Descriptive post-run report only; the registered science result and gates remain authoritative.",
        "registered_study": str(study), "registration_sha256": reg_doc["sha256"],
        "registered_analysis_sha256": analysis_doc["sha256"],
        "registered_primary_treatment_minus_control": official,
        "science_verify": {"command": [str(SCIENCE_CLI), "science", "verify", str(study)],
                            "returncode": verify.returncode, "stdout": verify.stdout.strip()},
        "per_arm": per_arm,
        "pair_context": {"registered_primary_remains_authority": True,
                         "paired_physiology_gates_changed": False,
                         "supplemental_differences_are_descriptive_only": True},
        "input_hashes": manifest_inputs,
        "limitations": [
            "The report describes outputs from the existing registered model and does not establish clinical or population validity.",
            "No respiratory-to-cardiovascular feedback or whole-body modal response is represented by this model.",
            "Geometry, contact, interface, and source limitations remain those stated in registered study evidence.",
        ],
    }
    write_json(output / "supplement.json", result)
    write_markdown(output / "README.md", result)
    outputs = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output / "manifest.json", {"schema": result["schema"], "inputs": manifest_inputs, "outputs": outputs,
                                          "scope": result["qualification"]})
    return result


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_markdown(path, result):
    lines = ["# Descriptive post-run supplement", "",
             "The registered science-v2 result remains authoritative. This addendum adds descriptive blood-volume, P–V, gas, cycle-counter, and source-reference context; it does not alter gates or qualify physiology.", "",
             "Registered primary treatment-minus-control result: **%.6f** (unit and verdict remain as recorded by science-v2)." % result["registered_primary_treatment_minus_control"], "",
             "| Arm | Blood-volume range (mL) | PaO2 dose−pre / recovery−pre (mmHg) | PaCO2 dose−pre / recovery−pre (mmHg) | Terminal accepted time (s) |" , "|---|---:|---:|---:|---:|"]
    for arm in ("control", "treatment"):
        d = result["per_arm"][arm]
        b = d["blood_volume_ml"]["post_initialization_10s_to_terminal"]
        o2 = d["within_arm_gas_changes_mmhg"]["PaO2_mmhg"]
        co2 = d["within_arm_gas_changes_mmhg"]["PaCO2_mmhg"]
        lines.append("| %s | %.3f–%.3f | %+.3f / %+.3f | %+.3f / %+.3f | %.9f |" %
                     (arm, b["min"], b["max"], o2["dose_minus_pre"], o2["recovery_minus_pre"],
                      co2["dose_minus_pre"], co2["recovery_minus_pre"], d["native_terminal_accepted_time_s"]))
    lines += ["", "## Interpretation limits", "",
              "Blood, lung-volume, pressure, flow, gas, and cycle values are native accepted observations checked with existing owner identities. Cumulative gas and blood accounting residuals are running/high-water diagnostics and are not summed across rows. CO/SV comparison bounds and their measurement methods are retained from the frozen Human owner. Per-arm recovery-minus-pre values describe the observed trace; they do not replace the registered paired late-recovery equivalence tests.", "",
              "This addendum does not establish clinical or population validity, anatomy clearance, or a whole-body physiological response. The model includes perfusion-to-gas transport and gas-driven chemoreflex regulation, but respiratory pressure/volume does not feed back into cardiovascular hemodynamics and whole-body modal response is not represented. SaO2 trace values are fractions; comparisons to the owner percent reference convert them to percent. See supplement.json for exact values and hashes.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_smoke(output, analyzer, prepare, owner):
    output = Path(output).resolve()
    require(not output.exists(), "smoke output must be a fresh directory")
    verification = SMOKE / "verification.json"
    require(verification.is_file() and sha(verification) == SMOKE_VERIFICATION_SHA256,
            "20 s smoke verification missing or changed")
    v = json.loads(verification.read_text())
    require(v.get("actual_run_native_exit_code") == 0 and v.get("terminal_capture", {}).get("verified") is True and
            v.get("terminal_capture", {}).get("accepted_step") == 10000,
            "20 s owner smoke did not pass its exact terminal capture")
    scene = SMOKE / "scene"
    trace = scene / "resting-coupled.csv"
    obs = scene / "intervention-observation.json"
    invocation_path = scene / "invocation.json"
    log = scene / "native.log"
    require(sha(trace) == v["physiology_csv_sha256"] and sha(obs) == v["observation_sha256"],
            "20 s smoke physiology/observation hash mismatch")
    header, rows = read_trace(trace, set(TRACE_FIELDS) | set(REQUIRED_NUMERICAL_FIELDS) | set(MECHANICS_FIELDS))
    require(len(rows) == 1250 and number(rows[-1], "step") == 10000,
            "20 s smoke trace does not contain the expected sampled endpoint")
    observation = json.loads(obs.read_text())
    require(observation == json.loads((SMOKE / "stdout.json").read_text()) and
            observation.get("accepted_steps") == 10000 and observation.get("root_assistance_observed") is False,
            "20 s owner observation is not the successful unassisted terminal smoke")
    invocation = json.loads(invocation_path.read_text())
    config = extract_respiration_config(invocation)
    config_sha = sha(config)
    require(invocation.get("asset_sha256", {}).get(str(config)) == config_sha,
            "smoke respiration config is not bound by its native invocation")
    pack = scene / "accepted-geometry/step-10000.mrvpack"
    receipt = scene / "accepted-geometry/step-10000.receipt.json"
    require(sha(pack) == v["terminal_capture"]["pack_sha256"] and
            sha(receipt) == v["terminal_capture"]["receipt_sha256"],
            "smoke terminal geometry files do not match their verification receipt")
    windows = {"smoke_post_initialization": [10.0, 20.0]}
    mechanics = owner.native_respiration_trace_consistency(trace, config, windows)
    heart = analyzer.audit.cardiac(rows)
    breath = analyzer.audit.respiratory(rows, analyzer.audit.breath_events(rows))
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema": "numi.human.postrun-descriptive-supplement-smoke.v1",
        "status": "passed_20s_smoke_only",
        "qualification": "20 s existing-owner CLI smoke only; not a registered study, not a 310 s acceptance, and not physiology qualification.",
        "verification_path": str(verification), "verification_sha256": sha(verification),
        "trace": {"path": str(trace), "sha256": sha(trace), "header_columns": len(header), "samples": len(rows)},
        "observation": {"path": str(obs), "sha256": sha(obs), "accepted_steps": observation["accepted_steps"],
                        "root_assistance_observed": observation["root_assistance_observed"]},
        "native_log": {"path": str(log), "sha256": sha(log)},
        "native_invocation": {"path": str(invocation_path), "sha256": sha(invocation_path)},
        "respiration_config": {"path": str(config), "sha256": config_sha},
        "terminal_capture": {**v["terminal_capture"], "pack_path": str(pack), "receipt_path": str(receipt)},
        "source_pins": {"analyzer_path": str(ANALYZER_PATH), "analyzer_sha256": sha(ANALYZER_PATH),
                         "prepare_path": str(PREPARE_PATH), "prepare_sha256": sha(PREPARE_PATH),
                         "owner_path": str(OWNER_PATH), "owner_sha256": sha(OWNER_PATH)},
        "owner_respiration_identity_check": mechanics,
        "blood_volume_ml": {"all_smoke_samples": stats(rows, "blood_ml"),
                             "post_initialization": stats(select_window(rows, 10.0, 20.0), "blood_ml")},
        "cycle_ledgers": {"cardiac_counter_intervals": len(heart), "complete_breath_intervals": len(breath)},
        "scope_note": "This smoke checks descriptive readers against real closed native 20 s outputs; it makes no claim about the 310 s registered pair.",
    }
    write_json(output / "smoke-validation.json", result)
    (output / "README.md").write_text("# 20 s smoke-only descriptive reader check\n\nThis is an owner CLI smoke, not the registered 310 s pair. It validates the native terminal capture and existing respiratory mechanics identities only. It does not qualify physiology or anatomy.\n", encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", type=Path, help="closed registered P18 study root")
    ap.add_argument("--output", type=Path, required=True, help="fresh evidence directory outside the study")
    ap.add_argument("--smoke", action="store_true", help="validate the retained 20 s owner smoke only")
    args = ap.parse_args()
    analyzer, prepare, adapter, owner = load_helpers()
    if args.smoke:
        result = run_smoke(args.output, analyzer, prepare, owner)
        print("Wrote 20 s smoke-only validation: " + str(args.output.resolve()))
        return 0
    require(args.study is not None, "--study is required for registered-pair analysis")
    result = run_registered_pair(args.study, args.output, analyzer, prepare, adapter, owner)
    print("Wrote descriptive addendum: " + str(args.output.resolve()))
    print("Registered primary result remains unchanged; no study or runtime input was modified.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        sys.stderr.write("REFUSED/FAILED: %s\n" % exc)
        raise SystemExit(2)
