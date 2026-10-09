#!/usr/bin/env python3
"""Read-only descriptive report for the already-declared 310 s native baseline.

This reuses the pinned Human owner checks and direct-pair helpers. It never
launches the simulator and does not admit anatomy or clinical validity.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path("/Users/n/numi-human-retained-delivery-20261009")
PKG = ROOT / "skin-resting-multipose-clearance-1218"
PREP = PKG / "native-baseline-310s-preparation"
RUN_DECL = PREP / "run-declaration.json"
RUNNER = PREP / "run.py"
RUN_DIR = PREP / "native-run"
OUT_DEFAULT = ROOT / "skin-resting-multipose-clearance-1218" / "baseline-analysis-002" / "closed-baseline-analysis.json"

ANALYZER = PKG / "direct-pair-preparation-001" / "analyze_direct_pair.py"
ANALYZER_SHA = "4b273d7c84ab3aba0ebc1a55ab912c4e71c02cfb650493ab7813c7e08a7b31d1"
CLI_ADAPTER = PKG / "direct-pair-preparation-001" / "prepare_owner_cli_treatment.py"
CLI_ADAPTER_SHA = "b4ea409e574535eecbaf39f5b485016f27e8e7d011a5333c1adf41d30368c245"
ANATOMY_REPORT = PKG / "native-accepted-geometry-audit-001" / "early-95s-001" / "summary.json"
ANATOMY_REPORT_SHA = "4cfdfb841ebd2afabb0bf4d29809b4f0479ff6796248644e79481f17d88e5ee8"
OWNER = Path("/Users/n/numi-human-terminal-trace-capture-fix-1162/matter/tools/resting_intervention_study.py")
OWNER_SHA = "ef87b09a17b96c985d6f584932e806ba9e7c6a53785da32a37a56ac846a3ac5f"
SUPPORT = Path("/Users/n/numi-human-retained-delivery-20261009/native-final-pair-postprocessing-1170/revision-003/analyze_final_pair.py")
SUPPORT_SHA = "fa12948701aa0359017d3b9a204bd306afaa78f188f0ca46fc462a907e29eb7c"
STEPS, DT = 155000, 0.002
WINDOWS = {
    "post_initialization_10_40s": [10.0, 40.0],
    "early_30_60s": [30.0, 60.0],
    "middle_70_100s": [70.0, 100.0],
    "terminal_280_310s": [280.0, 310.0],
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or linked required JSON: " + str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def load_pinned(path: Path, expected: str, name: str):
    if path.is_symlink() or not path.is_file() or sha(path) != expected:
        raise ValueError("pinned source changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned source: " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_anatomy_failure(captures: list[dict]) -> dict:
    if sha(ANATOMY_REPORT) != ANATOMY_REPORT_SHA:
        raise ValueError("pinned 95 s anatomy diagnostic changed")
    report = read_json(ANATOMY_REPORT)
    poses = report.get("pose_results")
    if (report.get("schema") != "numi.human.native-accepted-geometry-audit-1218.summary.v1" or
            report.get("capture_steps_requested") != [47519] or
            report.get("capture_times_s") != [95.038] or
            report.get("all_geometry_gates_clear") is not False or
            report.get("all_target_and_skin_self_gates_clear") is not False or
            not isinstance(poses, list) or len(poses) != 1):
        raise ValueError("pinned anatomy diagnostic does not match the recorded failing 95.038 s pose")
    pose = poses[0]
    expected_surfaces = [
        {"pairs": 12, "surface": [51011, 503]},
        {"pairs": 6, "surface": [51005, 36]},
    ]
    capture = next((item for item in captures if item.get("accepted_step") == 47519), None)
    if capture is None or not isinstance(capture.get("accepted_time_s"), (float, int)):
        raise ValueError("the matching native accepted-time receipt is missing for the 95 s anatomy failure")
    actual_time = float(capture["accepted_time_s"])
    if not math.isfinite(actual_time) or abs(actual_time - 47519 * 0.0020000000949949026) > 1e-9:
        raise ValueError("95 s anatomy pose does not bind to the actual accepted Float32 clock")
    if (pose.get("accepted_step") != 47519 or
            pose.get("all_skin_crossing_pair_count") != 18 or
            pose.get("hit_surfaces") != expected_surfaces):
        raise ValueError("pinned anatomy diagnostic crossing partition changed")
    return {
        "status": "failed",
        "report_path": str(ANATOMY_REPORT),
        "report_sha256": ANATOMY_REPORT_SHA,
        "accepted_step": 47519,
        "nominal_time_s": 95.038,
        "actual_native_accepted_time_s": actual_time,
        "actual_time_source_receipt_sha256": capture["receipt_sha256"],
        "skin_crossing_pairs": 18,
        "groups": [
            {"pairs": 12, "surface_ids": [51011, 503]},
            {"pairs": 6, "surface_ids": [51005, 36]},
        ],
        "interpretation": "This remains an open anatomy failure; clean numerical traces do not override it."
    }


def one_flag(argv: list, flag: str) -> str:
    hits = [i for i, item in enumerate(argv) if item == flag]
    if len(hits) != 1 or hits[0] + 1 >= len(argv):
        raise ValueError("expected exactly one native argument " + flag)
    return argv[hits[0] + 1]


def verify_captures(owner, invocation: dict, log: str, directory: Path,
                    steps: list[int]) -> list[dict]:
    raw = invocation.get("environment", {}).get("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS")
    if raw != ",".join(map(str, steps)):
        raise ValueError("actual native capture schedule differs from invocation declaration")
    export_lines = [line for line in log.splitlines() if line.startswith("accepted_geometry_export=")]
    rows = []
    for step in steps:
        pack = directory / "accepted-geometry" / ("step-%d.mrvpack" % step)
        receipt_path = directory / "accepted-geometry" / ("step-%d.receipt.json" % step)
        if pack.is_symlink() or receipt_path.is_symlink() or not pack.is_file() or not receipt_path.is_file():
            raise ValueError("closed baseline is missing accepted capture " + str(step))
        pack_sha, receipt_sha = sha(pack), sha(receipt_path)
        matches = [line for line in export_lines if line.startswith("accepted_geometry_export=" + str(pack) + " ")]
        if len(matches) != 1:
            raise ValueError("native log does not uniquely bind accepted capture " + str(step))
        fields = dict(x.split("=", 1) for x in matches[0].split()[1:] if "=" in x)
        receipt = read_json(receipt_path)
        if (fields.get("pack_sha256") != pack_sha or fields.get("receipt_sha256") != receipt_sha or
                receipt.get("schema") != "numi.human.accepted-render-geometry.v1" or
                receipt.get("accepted_step") != step or receipt.get("pack_file_sha256") != pack_sha or
                receipt.get("accepted_pack_path") != str(pack)):
            raise ValueError("accepted capture receipt/log hashes disagree at step " + str(step))
        rows.append({
            "accepted_step": step,
            "accepted_time_s": receipt.get("accepted_time_s"),
            "pack_path": str(pack), "pack_sha256": pack_sha, "pack_size_bytes": pack.stat().st_size,
            "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha,
            "root_fingerprint": receipt.get("accepted_root_fingerprint_hex"),
            "body_state_sha256": receipt.get("accepted_body_state_sha256"),
            "respiration_state_sha256": receipt.get("accepted_respiration_state_sha256"),
        })
    return rows


def artifact_identity(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("required output artifact missing or linked: " + str(path))
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def require_support_output(summary: dict) -> dict:
    """The owner summary gates receipt/hash/coverage; require its optional series too."""
    count = summary.get("samples")
    if not isinstance(count, int) or count < 100:
        raise ValueError("support summary lacks its required post-initialization sample coverage")
    checked = {}
    for key in ("sampled_aggregate_normal_support_force_last_physical_step_n",
                "sampled_active_contact_count"):
        stats = summary.get(key)
        if not isinstance(stats, dict) or stats.get("n") != count:
            raise ValueError("support diagnostic does not contain a complete finite series: " + key)
        for field in ("mean", "min", "max"):
            value = stats.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("support diagnostic has a missing/nonfinite statistic: " + key + "." + field)
        checked[key] = {"status": "complete_finite_observation_series", "samples": count}
    return checked


def _finite_row(row: dict, key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("missing or malformed drift field " + key) from exc
    if not math.isfinite(value):
        raise ValueError("nonfinite drift field " + key)
    return value


def motion_drift_summary(support_path: Path, body_path: Path, log_path: Path, duration_s: float) -> dict:
    """Describe sampled COM/root motion and contact, without a static-rest gate."""
    with support_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 100:
        raise ValueError("support/COM trace lacks enough rows for drift description")
    rows.sort(key=lambda row: _finite_row(row, "time_s"))
    times = [_finite_row(row, "time_s") for row in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("support/COM sample times are not strictly increasing")
    with body_path.open(newline="") as stream:
        body_rows = list(csv.DictReader(stream))
    if not body_rows:
        raise ValueError("body trace is empty")
    body_rows.sort(key=lambda row: _finite_row(row, "time_s"))

    def window(start: float, end: float) -> dict:
        selected = [row for row in rows if start <= _finite_row(row, "time_s") <= end + 1e-8]
        if len(selected) < 2:
            raise ValueError("too few support rows in drift window %.3f-%.3f s" % (start, end))
        ts = [_finite_row(row, "time_s") for row in selected]
        com = [[_finite_row(row, "com_" + axis + "_m") for row in selected] for axis in "xyz"]
        rv = [[_finite_row(row, "root_v" + axis + "_m_s") for row in selected] for axis in "xyz"]
        delta_com = [axis[-1] - axis[0] for axis in com]
        root_integral = [math.fsum(0.5 * (vel[i] + vel[i + 1]) * (ts[i + 1] - ts[i])
                                   for i in range(len(ts) - 1)) for vel in rv]
        root_speeds = [math.sqrt(math.fsum(rv[axis][i] ** 2 for axis in range(3))) for i in range(len(ts))]
        forces = [_finite_row(row, "normal_force_last_physical_step_n") for row in selected]
        contacts = [_finite_row(row, "active_contact_count") for row in selected]
        masses = [_finite_row(row, "total_body_mass_kg") for row in selected]
        expected_weight = statistics.fmean(masses) * 9.80665
        radii = [math.sqrt(math.fsum((com[axis][i] - com[axis][0]) ** 2 for axis in range(3)))
                 for i in range(len(ts))]
        return {
            "sample_count": len(selected), "time_s": [ts[0], ts[-1]],
            "com_endpoint_delta_mm_xyz": [1000 * x for x in delta_com],
            "com_endpoint_displacement_mm": 1000 * math.sqrt(math.fsum(x * x for x in delta_com)),
            "max_com_excursion_from_window_start_mm": 1000 * max(radii),
            "root_velocity_start_m_s_xyz": [axis[0] for axis in rv],
            "root_velocity_end_m_s_xyz": [axis[-1] for axis in rv],
            "root_speed_max_m_s": max(root_speeds),
            "sampled_root_velocity_trapezoid_integral_mm_xyz": [1000 * x for x in root_integral],
            "root_integral_scope": "Trapezoidal estimate from sampled root velocity; not a direct root-position endpoint difference.",
            "sampled_normal_force_last_physical_step_n": {
                "mean": statistics.fmean(forces), "min": min(forces), "max": max(forces)},
            "sampled_active_contact_count": {
                "mean": statistics.fmean(contacts), "min": min(contacts), "max": max(contacts)},
            "represented_body_mass_kg_mean": statistics.fmean(masses),
            "estimated_body_weight_n_from_measured_mass": expected_weight,
            "sampled_support_force_to_weight_ratio": statistics.fmean(forces) / expected_weight,
            "force_scope": "Sampled total normal contact impulse divided by one physical-step duration; descriptive, not equilibrium proof.",
        }

    def penetration(start: float, end: float) -> dict:
        selected = [row for row in body_rows if start <= _finite_row(row, "time_s") <= end + 1e-8]
        if not selected:
            raise ValueError("no body rows in penetration window")
        penetrations = [_finite_row(row, "peak_penetration_m") for row in selected]
        gaps = [_finite_row(row, "min_contact_gap_m") for row in selected]
        return {"sample_count": len(selected), "time_s": [
                    _finite_row(selected[0], "time_s"), _finite_row(selected[-1], "time_s")],
                "maximum_reported_peak_penetration_um": 1e6 * max(penetrations),
                "minimum_reported_contact_gap_um": 1e6 * min(gaps),
                "qualification": "Finite native contact diagnostics only; this descriptive summary adds no penetration threshold."}

    progress_pattern = re.compile(
        r"human_standing_progress=accepted step=(\d+) simulated_seconds=([^ ]+) "
        r"root_xyz_m=\[([^]]+)\] center_of_mass_xyz_m=\[([^]]+)\] "
        r"root_orientation_xyzw=\[[^]]+\] root_linear_velocity_xyz_m_s=\[([^]]+)\] "
        r"max_generalized_acceleration=[^ ]+ root_linear_speed_m_s=([^ ]+) "
        r"support_force_n=([^ ]+) penetration_m=([^ ]+) contact_count=(\d+)"
    )
    progress = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = progress_pattern.search(line)
        if match is None:
            continue
        step = int(match.group(1))
        t = float(match.group(2))
        vecs = [[float(value.strip()) for value in group.split(",")] for group in match.group(3, 4, 5)]
        if any(len(vec) != 3 or not all(math.isfinite(value) for value in vec) for vec in vecs):
            raise ValueError("accepted progress row has a malformed position/velocity vector")
        speed, force, penetration_m = map(float, match.group(6, 7, 8))
        contacts = int(match.group(9))
        if not all(math.isfinite(value) for value in (t, speed, force, penetration_m)):
            raise ValueError("accepted progress row has a nonfinite drift/contact scalar")
        progress.append({"step": step, "time_s": t, "root_xyz_m": vecs[0],
                         "com_xyz_m": vecs[1], "root_velocity_xyz_m_s": vecs[2],
                         "root_speed_m_s": speed, "support_force_n": force,
                         "penetration_m": penetration_m, "active_contact_count": contacts})
    if len(progress) < 2 or any(b["step"] <= a["step"] or b["time_s"] <= a["time_s"]
                                for a, b in zip(progress, progress[1:])):
        raise ValueError("native log lacks ordered accepted root/COM progress samples")

    def logged_window(start: float, end: float) -> dict:
        selected = [item for item in progress if start <= item["time_s"] <= end + 1e-7]
        if len(selected) < 2:
            raise ValueError("native log has too few progress records in %.3f-%.3f s" % (start, end))
        first, last = selected[0], selected[-1]
        root_delta = [last["root_xyz_m"][i] - first["root_xyz_m"][i] for i in range(3)]
        com_delta = [last["com_xyz_m"][i] - first["com_xyz_m"][i] for i in range(3)]
        times = [item["time_s"] for item in selected]
        root_int = [math.fsum(0.5 * (selected[i]["root_velocity_xyz_m_s"][axis] +
                                     selected[i + 1]["root_velocity_xyz_m_s"][axis]) *
                                    (times[i + 1] - times[i]) for i in range(len(selected) - 1))
                    for axis in range(3)]
        forces = [item["support_force_n"] for item in selected]
        penetrations = [item["penetration_m"] for item in selected]
        contacts = [item["active_contact_count"] for item in selected]
        return {
            "first_record": {"accepted_step": first["step"], "actual_time_s": first["time_s"]},
            "last_record": {"accepted_step": last["step"], "actual_time_s": last["time_s"]},
            "sample_count": len(selected),
            "root_xyz_endpoint_delta_mm": [1000 * value for value in root_delta],
            "root_xyz_endpoint_displacement_mm": 1000 * math.sqrt(math.fsum(v * v for v in root_delta)),
            "center_of_mass_endpoint_delta_mm": [1000 * value for value in com_delta],
            "center_of_mass_endpoint_displacement_mm": 1000 * math.sqrt(math.fsum(v * v for v in com_delta)),
            "root_velocity_trapezoid_integral_mm_xyz_crosscheck": [1000 * value for value in root_int],
            "root_speed_m_s": {"first": first["root_speed_m_s"], "last": last["root_speed_m_s"],
                                "max": max(item["root_speed_m_s"] for item in selected)},
            "support_force_n": {"mean": statistics.fmean(forces), "min": min(forces), "max": max(forces)},
            "contact_penetration_um": {"max": 1e6 * max(penetrations), "min": 1e6 * min(penetrations)},
            "active_contact_count": {"mean": statistics.fmean(contacts), "min": min(contacts), "max": max(contacts)},
            "qualification": "Direct accepted log positions plus sampled velocity/contact diagnostics; not static-rest or penetration-threshold qualification."
        }

    end = duration_s
    return {
        "direct_accepted_log_windows": {
            "10s_to_40s": logged_window(10.0, 40.0),
            "40s_to_95s": logged_window(40.0, 95.0),
            "95s_to_terminal": logged_window(95.0, end),
            "250s_to_terminal": logged_window(250.0, end),
            "280s_to_terminal": logged_window(280.0, end)},
        "sampled_com_support_csv_windows": {
            "post_initialization_10s_to_terminal": window(10.0, end),
            "late_250s_to_terminal": window(250.0, end),
            "terminal_280s_to_terminal": window(280.0, end)},
        "contact_penetration_csv_windows": {
            "post_initialization_10s_to_terminal": penetration(10.0, end),
            "late_250s_to_terminal": penetration(250.0, end),
            "terminal_280s_to_terminal": penetration(280.0, end)},
        "interpretation": "Direct root and COM endpoint motion comes from accepted native log records; velocity integration is a cross-check. Sampled forces and penetration are descriptive. The report does not call settling static rest or infer a cause from correlations."
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("refusing to overwrite an earlier baseline report")

    run_decl = read_json(RUN_DECL)
    execution_path = PREP / "execution.json"
    execution = read_json(execution_path)  # refuses to analyze an open owner-run wrapper
    invocation_path = RUN_DIR / "invocation.json"
    metadata_path = RUN_DIR / "run-metadata.json"
    log_path = RUN_DIR / "native.log"
    trace_path = RUN_DIR / "resting-coupled.csv"
    surface_path = RUN_DIR / "resting-surface-audit.csv"
    support_path = RUN_DIR / "resting-com-momentum-diagnostic.csv"
    invocation = read_json(invocation_path)
    metadata = read_json(metadata_path)
    if (run_decl.get("schema") != "numi.human.resting.native-run.declaration.v1" or
            run_decl.get("seconds") != 310 or run_decl.get("accepted_steps") != STEPS or
            run_decl.get("dt") != DT or run_decl.get("capture_steps") != [0, 9983, 19999, 47519, 152191, 153183, 154143, 155000]):
        raise ValueError("baseline declaration does not describe this 310 s native run")
    declared_output = Path(one_flag(run_decl["argv"], "--output")).resolve()
    if declared_output != RUN_DIR.resolve():
        raise ValueError("native-run declaration points at a different output directory")
    if (execution.get("returncode") != 0 or execution.get("changed_inputs") != {} or
            execution.get("declaration_sha256") != sha(RUN_DECL)):
        raise ValueError("existing Human run.py execution receipt is not a clean successful close")
    runpy_pin = run_decl.get("immutable_assets", {}).get(str(RUNNER))
    if not runpy_pin or sha(RUNNER) != runpy_pin:
        raise ValueError("baseline run.py does not match its declaration pin")

    analyzer = load_pinned(ANALYZER, ANALYZER_SHA, "direct_pair_analyzer_001")
    cli = load_pinned(CLI_ADAPTER, CLI_ADAPTER_SHA, "owner_cli_treatment_adapter_001")
    owner = load_pinned(analyzer.OWNER, analyzer.OWNER_SHA, "resting_intervention_owner_1162")
    if sha(OWNER) != OWNER_SHA:
        raise ValueError("native owner source changed")
    derived_argv, derived_assets, _, cli_sources = cli.native_from_cli(run_decl["argv"])
    if (derived_argv != invocation.get("argv") or derived_assets != invocation.get("asset_sha256") or
            execution.get("returncode") != 0):
        raise ValueError("existing owner CLI no longer reproduces the actual baseline invocation")
    owner.validate_native_310s_invocation(invocation)
    if (metadata.get("exit_code") != 0 or metadata.get("source_files_changed_during_run") != [] or
            metadata.get("argv") != invocation.get("argv") or
            metadata.get("asset_sha256") != invocation.get("asset_sha256") or
            metadata.get("environment") != invocation.get("environment") or
            metadata.get("loaded_metal_runtime", {}).get("verified") is not True):
        raise ValueError("native run-metadata did not verify successful unchanged inputs/runtime")

    for path, expected in invocation["asset_sha256"].items():
        bound = Path(path)
        if bound.is_symlink() or not bound.is_file() or sha(bound) != expected:
            raise ValueError("native invocation asset changed: " + path)

    log = log_path.read_text(encoding="utf-8", errors="replace")
    native = owner.native_scene_summary(log)
    if native["accepted_steps"] != STEPS or abs(native["simulated_s"] - STEPS * DT) > 1e-4:
        raise ValueError("native log does not prove exactly 155000 accepted 2 ms steps")
    terminal = owner.native_terminal_accepted_capture_evidence(invocation, RUN_DIR, native, log, STEPS, DT)
    if terminal.get("verified") is not True:
        raise ValueError("native run lacks exact accepted terminal capture/zero-advance evidence")
    capture_steps = list(run_decl["capture_steps"])
    captures = verify_captures(owner, invocation, log, RUN_DIR, capture_steps)

    rows = owner.read_trace(trace_path)
    cycles = owner.native_cycle_coverage(rows, DT)
    cycle_gate = cycles.get("post_initialization_cycle_coverage", {}).get("passed") is True
    if not cycle_gate:
        raise ValueError("cycle coverage returned without its required passed gate")
    body = owner.native_body_trace_consistency(trace_path, STEPS, DT)
    if body.get("root_assistance_observed") is not False:
        raise ValueError("body trace did not explicitly return zero root assistance")
    if not math.isfinite(float(body.get("maximum_contact_penetration_m"))):
        raise ValueError("body trace returned no finite contact-penetration diagnostic")
    surface = owner.native_surface_trace_consistency(
        surface_path, STEPS, DT, require_whole_mesh=True,
        terminal_accepted_capture=terminal["verified"])
    mesh_audit = surface.get("whole_mesh_area_audit", {})
    if (surface.get("terminal_accepted_capture_included") is not True or
            mesh_audit.get("triangles_checked_per_frame", 0) <= 0 or
            mesh_audit.get("zero_area_triangles") != 0 or
            mesh_audit.get("nonfinite_area_triangles") != 0):
        raise ValueError("surface consistency returned without its whole-mesh/terminal gates")
    resp_path = Path(invocation["argv"][invocation["argv"].index("--resting-scene") + 2])
    if sha(resp_path) != invocation["asset_sha256"].get(str(resp_path)):
        raise ValueError("respiratory configuration does not match the invocation asset pin")
    resp = owner.native_respiration_trace_consistency(trace_path, resp_path, WINDOWS)
    resp_summary = resp.get("respiratory_mechanics", {})
    if (resp_summary.get("accepted_samples") != len(rows) or
            not math.isfinite(float(resp_summary.get("maximum_fraction_of_rounding_allowance", float("nan")))) or
            resp_summary["maximum_fraction_of_rounding_allowance"] > 1.0):
        raise ValueError("respiratory identity helper returned without complete in-bound results")
    windows = {
        name: {
            "samples": owner.window_metrics(rows, pair[0], pair[1]),
            "reference_context": owner.resting_reference_comparison(rows, pair[0], pair[1]),
        }
        for name, pair in WINDOWS.items()
    }
    post = [row for row in rows if row["time_s"] >= 10.0]
    blood = [row["blood_ml"] for row in post]
    physiology = {
        "accepted_trace_sha256": sha(trace_path),
        "accepted_samples": len(rows),
        "post_initialization_time_s": [post[0]["time_s"], post[-1]["time_s"]],
        "blood_volume_ml": {
            "initial_post_init": blood[0], "terminal": blood[-1],
            "min": min(blood), "mean": statistics.fmean(blood), "max": max(blood),
            "max_abs_conservation_error_ml": max(abs(row["blood_error_ml"]) for row in post),
        },
        "full_trace_conservation": {
            "max_abs_O2_balance_error_STPD_ml": max(abs(row["oxygen_balance_error_stpd_ml"]) for row in rows),
            "max_abs_CO2_balance_error_STPD_ml": max(abs(row["co2_balance_error_stpd_ml"]) for row in rows),
        },
        "cycle_coverage": cycles,
        "windows": windows,
        "respiratory_mechanics_identities": resp,
        "units_note": "SaO2 is the fraction exported by the model; the existing reference comparison converts it to percent."
    }
    support_module = load_pinned(SUPPORT, SUPPORT_SHA, "final_pair_support_helper_003")
    support = support_module.support_summary(
        support_path, {"scene/resting-com-momentum-diagnostic.csv": sha(support_path)}, native["simulated_s"])
    support_check = require_support_output(support)
    drift = motion_drift_summary(support_path, trace_path, log_path, native["simulated_s"])
    movie = Path(one_flag(invocation["argv"], "--resting-movie"))
    movie_identity = artifact_identity(movie)
    identity_paths = [RUN_DECL, execution_path, invocation_path, metadata_path, log_path, trace_path,
                      surface_path, support_path, movie]
    identities = [artifact_identity(path) for path in identity_paths]
    anatomy = verify_anatomy_failure(captures)

    wrapper_wall = float(execution["wall_seconds"])
    if not math.isfinite(wrapper_wall) or wrapper_wall <= 0:
        raise ValueError("owner wrapper wall time is not finite and positive")
    report = {
        "analysis_type": "descriptive_closed_native_baseline",
        "status": "closed_run_integrity_checks_passed_anatomy_gate_failed",
        "qualification": "The listed execution, accepted-state, cycle, geometry-consistency, respiratory-identity, and complete-support-observation checks passed. Support adequacy and reference-range comparisons are descriptive; anatomy remains failed. This does not establish physiological calibration or clinical validity.",
        "check_status": {
            "owner_execution_input_and_runtime_identity": "passed",
            "accepted_horizon_terminal_capture_and_capture_receipts": "passed",
            "repeated_breath_and_cardiac_cycle_coverage": "passed",
            "body_trace_finite_clock_and_zero_assistance_integrity": "passed",
            "displayed_surface_and_whole_mesh_numerical_consistency": "passed",
            "respiratory_pressure_volume_flow_identity": "passed",
            "support_com_input_hash_coverage_and_finite_series": "passed",
            "support_force_adequacy_or_static_equilibrium": "not_assessed",
            "physiological_reference_intervals": "descriptive_only_not_a_gate",
            "whole_body_anatomy": "failed",
        },
        "native": {
            "accepted_steps": native["accepted_steps"], "accepted_duration_s": native["simulated_s"],
            "device": native["device"], "world_fingerprint": native["world_fingerprint"],
            "body_source_fingerprint": native["body_source_fingerprint"],
            "coupled_program_fingerprint": native["coupled_program_fingerprint"],
            "root_assistance_in_terminal_state": False,
            "loaded_metal_runtime": metadata["loaded_metal_runtime"],
            "owner_wrapper_wall_seconds": wrapper_wall,
            "native_integrated_wall_s": native["wall_s"],
            "native_integrated_real_time_factor": native["real_time_factor"],
            "owner_wrapper_real_time_factor": native["simulated_s"] / wrapper_wall,
            "timing_context": "The wrapper wall factor includes any overlapping CPU geometry audits and is not an isolated performance benchmark.",
            "native_log_root_assistance": "physiology_body_clock=matched root_assistance=false",
        },
        "numerical_and_physiology_descriptives": physiology,
        "physiological_reference_range_gate": "not_applied; per-window within_reference_bounds fields are descriptive context only",
        "contact_and_surface": {
            "body_trace": body,
            "surface_trace": surface,
            "sampled_support_and_com": support,
            "support_observation_completeness_check": support_check,
            "drift_and_support_diagnostics": drift,
            "zero_assistance": {
                "native_terminal": True,
                "all_retained_physiology_rows_checked": body["root_assistance_observed"] is False,
                "maximum_contact_penetration_m": body["maximum_contact_penetration_m"],
            },
        },
        "accepted_captures": captures,
        "movie_identity": {**movie_identity, "visual_review_performed": False},
        "anatomy_gate": anatomy,
        "replay_rollback_scope": {
            "replay_performed": False, "rollback_performed": False,
            "interpretation": "This direct native run/report did not test deterministic replay, checkpoint restart, or rollback."
        },
        "source_pins": {
            "direct_pair_analyzer": {"path": str(ANALYZER), "sha256": ANALYZER_SHA},
            "owner_cli_adapter": {"path": str(CLI_ADAPTER), "sha256": CLI_ADAPTER_SHA},
            "native_owner": {"path": str(OWNER), "sha256": OWNER_SHA},
            "support_helper": {"path": str(SUPPORT), "sha256": SUPPORT_SHA},
            "owner_cli_sources": cli_sources,
        },
        "native_input_assets": invocation["asset_sha256"],
        "native_input_asset_map_sha256": canonical_sha(invocation["asset_sha256"]),
        "artifact_identities": identities,
        "measured_time_windows_s": WINDOWS,
        "limitations": [
            "These owner identities and conservation checks are numerical consistency checks, not independent physiological calibration.",
            "Reference-range comparisons are descriptive and depend on population, posture, altitude, and measurement method.",
            "The 95.038 s geometry audit remains failed; this baseline is not anatomically admitted even if later physiology and runtime checks pass.",
            "No respiratory-pressure to cardiovascular-hemodynamic feedback, Haldane exchange, or regional ventilation/perfusion validation is established by this report.",
            "Movie bytes are hash-bound but were not visually reviewed here.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError, ImportError, ZeroDivisionError) as exc:
        raise SystemExit("baseline analysis refused: " + str(exc))
