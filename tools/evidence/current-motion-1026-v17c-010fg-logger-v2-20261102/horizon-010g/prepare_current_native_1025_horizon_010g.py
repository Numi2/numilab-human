#!/usr/bin/env python3
"""Gated longer-horizon direct-O motion after frozen 009's two-step probe.

The native owner has no accepted-MRV checkpoint restart path. Each horizon is
therefore a fresh initialization from the same pinned baseline, not a continuous
restart from the preceding accepted state.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, re, sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FIRST_DIR = HERE.parent / "current-native-1025-direct-o-007-recorded-flow-scope"
FIRST_PATH = FIRST_DIR / "prepare_current_native_1025_direct_o_009.py"
sys.path.insert(0, str(FIRST_DIR))
SCHEMA_PLAN = "numi.human.current-native-direct-o-horizon-plan.v1"
SCHEMA_GATE = "numi.human.current-native-direct-o-horizon-gate.v1"
MODE = "fresh_initialization_increasing_horizon_same_pinned_baseline"
PROGRESSION = {2: 16, 16: 128, 128: 1024}
MAX_HORIZON_STEPS = 1_000_000
TERMINAL = re.compile(r"resting_FEM_terminal_audit=passed[^\n]*expected_steps=(\d+)[^\n]*accepted_steps=(\d+)(?:\s|$)")
PERFORMANCE = re.compile(r"resting_integrated_body=completed simulated_s=([^\s]+) wall_s=([^\s]+) real_time_factor=([^\s]+)")

spec = importlib.util.spec_from_file_location("native_direct_o_009_pinned_for_010g", FIRST_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen 009 source/install adapter")
first = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = first
spec.loader.exec_module(first)
base = first.base


def obj(path: Path) -> dict[str, Any]:
    value = base.read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def ref(path: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": base.sha_file(path)}


def check_ref(value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must be an exact path/SHA reference")
    raw, digest = value["path"], value["sha256"]
    if not isinstance(raw, str) or not Path(raw).is_absolute() or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label} has malformed path/SHA")
    path = Path(raw)
    if not path.is_file() or base.sha_file(path) != digest:
        raise ValueError(f"{label} changed or is missing")
    return path.resolve(), digest


def orientation_binding(audit_path: Path, audit_sha: str, owner_path: Path, owner_sha: str) -> tuple[dict[str, str], dict[str, str]]:
    audit_ref = ref(audit_path)
    owner_ref = ref(owner_path)
    if audit_ref["sha256"] != audit_sha or owner_ref["sha256"] != owner_sha:
        raise ValueError("fresh role-orientation audit/owner pin changed")
    audit = obj(audit_path)
    if audit.get("schema") != "numi.h706.1022-role-surface-orientation-audit.v1" or audit.get("status") != "PASS_EXACT_1022_SIGNED_ROLE_CONVENTIONS":
        raise ValueError("fresh role-orientation audit did not pass its exact signed convention gate")
    if audit.get("physical_admission") is not False or audit.get("native_steps") != 0:
        raise ValueError("role-orientation audit overclaims physical/native evidence")
    return audit_ref, owner_ref


def check_report_orientation(report: dict[str, Any], audit_ref: dict[str, str], owner_ref: dict[str, str]) -> None:
    coverage = report.get("geometry_coverage", {}).get("paired_shell_input_coverage", {})
    contract = coverage.get("native_role_orientation_contract", {})
    expected = {
        "report_path": audit_ref["path"], "report_sha256": audit_ref["sha256"],
        "validator_path": owner_ref["path"], "validator_sha256": owner_ref["sha256"],
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("geometry report does not bind the selected fresh role-orientation audit and owner")


def finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be finite and positive") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def capture_steps(horizon: int) -> list[int]:
    if type(horizon) is not int or not 2 <= horizon <= MAX_HORIZON_STEPS:
        raise ValueError("horizon must be an integer in [2, 1000000]")
    steps = sorted({0, 1, horizon // 4, horizon // 2, 3 * horizon // 4, horizon})
    if len(steps) > 8 or steps[0] != 0 or steps[-1] != horizon or 1 not in steps:
        raise ValueError("bounded accepted-frame roster is invalid")
    return steps


def next_horizon(previous_steps: int, requested_steps: int) -> int:
    expected = PROGRESSION.get(previous_steps)
    if expected is None or requested_steps != expected:
        raise ValueError(f"horizon progression must follow {PROGRESSION}; expected {expected} after {previous_steps}")
    return requested_steps


def duration_horizon(previous_steps: int, previous_wall_s: float, target_wall_s: float, max_steps: int = MAX_HORIZON_STEPS) -> int:
    if previous_steps != 1024:
        raise ValueError("duration target is only available after the qualified 1024-step stage")
    wall = finite_positive(previous_wall_s, "previous measured wall seconds")
    target = finite_positive(target_wall_s, "target wall seconds")
    if type(max_steps) is not int or not 1024 <= max_steps <= MAX_HORIZON_STEPS:
        raise ValueError("duration cap must be within the native [1024, 1000000] step range")
    steps = math.ceil(target * previous_steps / wall)
    if steps <= previous_steps:
        raise ValueError("target wall duration does not exceed the measured 1024-step run")
    if steps > max_steps:
        raise ValueError(f"measured target requires {steps} steps, above cap {max_steps}")
    return steps


def parse_performance(stdout: str) -> dict[str, float] | None:
    rows = PERFORMANCE.findall(stdout)
    if not rows:
        return None
    if len(rows) != 1:
        raise ValueError("native integrated-body performance marker is duplicated")
    sim, wall, rtf = (finite_positive(x, "native performance value") for x in rows[0])
    return {"simulated_s": sim, "wall_s": wall, "real_time_factor": rtf}


def context(manifest: Path, source_plan: Path, source_receipt: Path):
    src, cfg, route, ctx = first.read_plan(manifest, source_plan)
    if src.get("status") != "DIRECT_O_SOURCE_PREFLIGHT_READY" or len(src.get("stages", [])) != 1:
        raise ValueError("continuation requires the exact 009 source-preflight plan")
    receipt_path = source_receipt.resolve()
    receipt = obj(receipt_path)
    if receipt.get("status") != "PASS_SOURCE_ONLY_DIRECT_O_NO_LEGACY_DETAIL_BRIDGE":
        raise ValueError("source-preflight receipt is not a passing direct-O source result")
    if receipt.get("input_manifest_sha256") != src["manifest_sha256"] or receipt.get("plan_sha256") != base.sha_file(source_plan):
        raise ValueError("source receipt does not bind the exact current source plan")
    if receipt.get("legacy_detail_bridge_enabled") is not False or receipt.get("legacy_detail_map_required") is not False:
        raise ValueError("source receipt retains legacy detail bridge dependency")
    return src, cfg, route, ctx, receipt, receipt_path


def checkpoint_binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    ck = obj(path)
    result = {"checkpoint": ref(path)}
    for key, outkey in (("source_state", "source_state"), ("217_handoff", "217_handoff"),
                        ("registered_active_source_crosswalk", "registered_active_source_crosswalk")):
        entry = ck.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError(f"checkpoint lacks {key} path/SHA")
        file = Path(entry["path"])
        if not file.is_file() or base.sha_file(file) != entry.get("sha256"):
            raise ValueError(f"checkpoint {key} pin changed")
        result[outkey] = {"path": entry["path"], "sha256": entry["sha256"]}
    return result


def verify_gate(path: Path, manifest: Path, checkpoint: Path, source_plan: Path, source_receipt: Path) -> dict[str, Any]:
    gate = obj(path)
    if gate.get("schema") != SCHEMA_GATE or gate.get("status") != "PASS_NATIVE_DIRECT_O_HORIZON_GATE":
        raise ValueError("previous horizon gate did not pass")
    if gate.get("physical_admission") is not False or gate.get("long_horizon_qualification") is not False:
        raise ValueError("previous gate overclaims physical or long-horizon qualification")
    if gate.get("source_plan") != ref(source_plan) or gate.get("source_preflight_receipt") != ref(source_receipt):
        raise ValueError("previous gate source lineage differs from current config")
    _, cfg, _, ctx, _, _ = context(manifest, source_plan, source_receipt)
    static = ctx["static"]
    if gate.get("manifest_path") != str(manifest.resolve()) or gate.get("manifest_sha256") != base.json_digest(cfg):
        raise ValueError("previous gate config identity differs")
    if gate.get("build_manifest_sha256") != static["manifest_sha256"] or gate.get("native_binary", {}).get("sha256") != static["native_binary_sha256"]:
        raise ValueError("previous gate build/binary identity is stale")
    current = checkpoint_binding(checkpoint)
    if any(gate.get(k) != current[k] for k in current):
        raise ValueError("previous gate checkpoint/source lineage differs")
    refs = {}
    for field in ("motion_plan", "observer", "install_route_receipt", "native_stdout", "native_stderr", "geometry_report",
                  "geometry_orientation_audit", "geometry_orientation_owner"):
        refs[field] = check_ref(gate.get(field), f"previous gate {field}")
    orientation_audit_ref = {"path": str(refs["geometry_orientation_audit"][0]), "sha256": refs["geometry_orientation_audit"][1]}
    orientation_owner_ref = {"path": str(refs["geometry_orientation_owner"][0]), "sha256": refs["geometry_orientation_owner"][1]}
    orientation_binding(refs["geometry_orientation_audit"][0], refs["geometry_orientation_audit"][1],
                        refs["geometry_orientation_owner"][0], refs["geometry_orientation_owner"][1])
    verifier_path, verifier_sha = check_ref(gate.get("geometry_verifier"), "previous geometry verifier")
    exporter_path, exporter_sha = check_ref(gate.get("accepted_geometry_exporter"), "previous accepted-geometry exporter")
    steps = gate.get("requested_accepted_steps")
    if type(steps) is not int or not 2 <= steps <= MAX_HORIZON_STEPS:
        raise ValueError("previous gate horizon is invalid")
    motion_plan_path = refs["motion_plan"][0]
    motion_plan = obj(motion_plan_path)
    if motion_plan.get("schema") == SCHEMA_PLAN:
        if motion_plan.get("status") != "READY_FRESH_INITIALIZATION_EXTENDED_HORIZON" or motion_plan.get("motion", {}).get("requested_accepted_step_count") != steps:
            raise ValueError("previous continuation plan is not the passing recorded horizon")
        if motion_plan.get("motion", {}).get("initialization_policy") != MODE or motion_plan.get("motion", {}).get("checkpoint_restart") is not False:
            raise ValueError("previous plan misrepresents fresh initialization as checkpoint restart")
        if motion_plan.get("geometry_verifier") != {"path": str(verifier_path), "sha256": verifier_sha}:
            raise ValueError("previous plan geometry-verifier pin differs")
        if motion_plan.get("accepted_geometry_exporter") != {"path": str(exporter_path), "sha256": exporter_sha}:
            raise ValueError("previous plan geometry exporter pin differs")
        if motion_plan.get("geometry_orientation_audit") != orientation_audit_ref or motion_plan.get("geometry_orientation_owner") != orientation_owner_ref:
            raise ValueError("previous plan orientation audit/owner pin differs")
        previous_ref = motion_plan.get("previous_gate")
        check_ref(previous_ref, "continuation predecessor gate")
        if gate.get("previous_gate") != previous_ref:
            raise ValueError("gate predecessor differs from planned prior horizon")
    else:
        first_plan, _, _, _ = first.read_plan(manifest, motion_plan_path)
        if first_plan.get("status") != "DIRECT_O_MOTION_PLAN_READY_AFTER_SOURCE_PREFLIGHT" or steps != 2 or first_plan.get("motion", {}).get("requested_accepted_step_count") != 2:
            raise ValueError("first gate accepts only frozen 009's successful two-step motion plan")
        if gate.get("previous_gate") is not None:
            raise ValueError("first motion gate cannot have a continuation predecessor")
    observer_path, _ = refs["observer"]
    route_path, _ = refs["install_route_receipt"]
    stdout_path, _ = refs["native_stdout"]
    stderr_path, _ = refs["native_stderr"]
    report_path, _ = refs["geometry_report"]
    observer, route_receipt, report, _ = verify_motion_bundle(
        observer_path, route_path, stdout_path, stderr_path, motion_plan_path, checkpoint,
        steps, verifier_path, verifier_sha, report_path,
        refs["geometry_orientation_audit"][0], refs["geometry_orientation_audit"][1],
        refs["geometry_orientation_owner"][0], refs["geometry_orientation_owner"][1])
    if report.get("accepted_geometry_exporter") != {"path": str(exporter_path), "sha256": exporter_sha}:
        raise ValueError("previous geometry report exporter identity differs")
    if route_receipt.get("native_binary_sha256") != static["native_binary_sha256"]:
        raise ValueError("previous route receipt does not use current native binary")
    if route_receipt.get("source_preflight_receipt_sha256") != base.sha_file(source_receipt):
        raise ValueError("previous route receipt source-preflight pin differs")
    if route_receipt.get("geometry_admission_receipt_sha256") != motion_plan.get("geometry_admission", {}).get("receipt_sha256"):
        raise ValueError("previous route receipt geometry pin differs")
    if route_receipt.get("mass_admission_receipt_sha256") != motion_plan.get("mass_admission", {}).get("receipt_sha256"):
        raise ValueError("previous route receipt mass pin differs")
    expected_perf = parse_performance(Path(gate["native_stdout"]["path"]).read_text(errors="replace"))
    if gate.get("performance") != expected_perf:
        raise ValueError("previous gate performance does not match its pinned native output")
    if gate.get("geometry_status") != report.get("status") or gate.get("accepted_geometry_steps") != sorted(int(x) for x in observer["accepted_geometry_frames"]):
        raise ValueError("previous gate summary differs from its pinned observer/report")
    return gate


def _check_trace(path_text: str, pin: Any) -> None:
    path = Path(path_text)
    if not isinstance(pin, dict) or set(pin) != {"sha256", "bytes"} or not path.is_file():
        raise ValueError(f"accepted trace pin malformed: {path}")
    if type(pin["bytes"]) is not int or pin["bytes"] <= 0 or pin["bytes"] != path.stat().st_size or base.sha_file(path) != pin["sha256"]:
        raise ValueError(f"accepted trace changed: {path}")



def validate_v17c_geometry_report(report: dict[str, Any]) -> None:
    if report.get("schema") != "numi.human.current-motion-geometry-verification.v11" or report.get("status") != "PASS":
        raise ValueError("selected accepted-state geometry report is not PASS under the pinned v17c schema")

def verify_motion_bundle(observer_path: Path, route_path: Path, stdout_path: Path, stderr_path: Path,
                         plan_path: Path, checkpoint: Path, steps: int, verifier_path: Path,
                         verifier_sha: str, report_path: Path,
                         orientation_audit_path: Path, orientation_audit_sha: str,
                         orientation_owner_path: Path, orientation_owner_sha: str):
    observer, route, report = obj(observer_path), obj(route_path), obj(report_path)
    if observer.get("status") != "PASS_BOUNDED_DIRECT_O_TRACES_AND_MRV" or observer.get("native_runtime_executed") is not True or observer.get("gpu_executed") is not True:
        raise ValueError("motion observer is not a completed accepted-state GPU run")
    if observer.get("terminal_fem_audit_verified") is not True or observer.get("full_fem_endpoint_state_dump_requested") is not False:
        raise ValueError("motion observer lacks compact terminal audit or requests full endpoint dump")
    if observer.get("requested_accepted_steps") != steps:
        raise ValueError("observer horizon differs from the requested step count")
    for key, expected in (("plan_sha256", base.sha_file(plan_path)), ("stdout_sha256", base.sha_file(stdout_path)), ("stderr_sha256", base.sha_file(stderr_path))):
        if observer.get(key) != expected:
            raise ValueError(f"observer {key} differs from current run")
    route_pin = observer.get("install_route_receipt")
    if not isinstance(route_pin, dict) or Path(route_pin.get("path", "")).resolve() != route_path.resolve() or route_pin.get("sha256") != base.sha_file(route_path):
        raise ValueError("observer route receipt pin changed")
    if route.get("status") != "PASS_DIRECT_O_INSTALL_AND_ACCEPTED_STATE_MOTION" or route_pin.get("status") != route.get("status"):
        raise ValueError("install/motion route did not pass")
    if route.get("requested_accepted_steps") != steps or route.get("observed_accepted_steps") != steps or route.get("registered_volume_install_pass") is not True:
        raise ValueError("install/motion route horizon/install claim mismatch")
    if route.get("plan_sha256") != base.sha_file(plan_path) or route.get("stdout_sha256") != base.sha_file(stdout_path) or route.get("stderr_sha256") != base.sha_file(stderr_path):
        raise ValueError("install/motion route does not bind current plan and logs")
    if route.get("physical_admission_claimed") is not False or route.get("long_horizon_qualification_claimed") is not False:
        raise ValueError("route receipt overclaims physical or horizon qualification")
    parsed_direct = first.parse_o_pass(stdout_path.read_text(errors="replace"))
    if route.get("direct_O") != parsed_direct or observer.get("direct_O") != parsed_direct:
        raise ValueError("direct-O installation receipt differs from the pinned native output")
    traces = observer.get("accepted_trace_files")
    if not isinstance(traces, dict) or not traces:
        raise ValueError("accepted-state trace pins are absent")
    for name, pin in traces.items():
        _check_trace(name, pin)
    frames = observer.get("accepted_geometry_frames")
    if not isinstance(frames, dict) or not {"0", str(steps)}.issubset(frames) or len(frames) > 8:
        raise ValueError("accepted MRV frames lack endpoints or exceed the eight-frame cap")
    for key, frame in frames.items():
        step = int(key)
        pack, receipt_path = Path(frame.get("pack_path", "")), Path(frame.get("receipt_path", ""))
        if not pack.is_file() or base.sha_file(pack) != frame.get("pack_sha256") or not receipt_path.is_file() or base.sha_file(receipt_path) != frame.get("receipt_sha256"):
            raise ValueError(f"accepted MRV frame {step} pin changed")
        mr = obj(receipt_path)
        if mr.get("accepted_step") != step or mr.get("physical_endpoint") != "accepted" or mr.get("surface_audit_endpoint") != "passed" or mr.get("pack_file_sha256") != base.sha_file(pack):
            raise ValueError(f"accepted MRV frame {step} receipt failed")
        if mr.get("position_normal_tangent_source") != "accepted-state renderer mesh buffer copied on the same Metal command buffer":
            raise ValueError(f"accepted MRV frame {step} lacks same-command-buffer state provenance")

    if report.get("schema") != "numi.human.current-motion-geometry-verification.v11" or report.get("status") != "PASS":
        raise ValueError("selected accepted-state geometry report is not PASS")
    if report.get("physical_admission") is not False or report.get("accepted_motion_measurement", {}).get("movement_is_qualification") is not False:
        raise ValueError("geometry report overclaims physical qualification")
    checkpoint_info = checkpoint_binding(checkpoint)
    for key, value in (("checkpoint_sha256", checkpoint_info["checkpoint"]["sha256"]),
                       ("source_state_sha256", checkpoint_info["source_state"]["sha256"]),
                       ("217_receipt_sha256", checkpoint_info["217_handoff"]["sha256"]),
                       ("217_map_sha256", checkpoint_info["registered_active_source_crosswalk"]["sha256"]),
                       ("stdout_sha256", base.sha_file(stdout_path)), ("stderr_sha256", base.sha_file(stderr_path))):
        if report.get(key) != value:
            raise ValueError(f"geometry report {key} differs from current run/checkpoint")
    obs_ref = report.get("observer", {})
    if Path(obs_ref.get("path", "")).resolve() != observer_path.resolve() or obs_ref.get("sha256") != base.sha_file(observer_path):
        raise ValueError("geometry report does not bind the current observer")
    verifier_ref = report.get("verifier_source", {})
    if Path(verifier_ref.get("path", "")).resolve() != verifier_path.resolve() or verifier_ref.get("sha256") != verifier_sha or base.sha_file(verifier_path) != verifier_sha:
        raise ValueError("geometry report does not bind the selected frozen verifier")
    audit_ref, owner_ref = orientation_binding(orientation_audit_path, orientation_audit_sha,
                                               orientation_owner_path, orientation_owner_sha)
    check_report_orientation(report, audit_ref, owner_ref)
    exporter_ref = report.get("accepted_geometry_exporter", {})
    exporter_path = Path(exporter_ref.get("path", ""))
    if not exporter_path.is_file() or base.sha_file(exporter_path) != exporter_ref.get("sha256"):
        raise ValueError("geometry report accepted-geometry exporter path/SHA changed")
    expected_frames = sorted(int(k) for k in frames)
    if report.get("accepted_frame_ids") != expected_frames or 0 not in expected_frames or steps not in expected_frames or len(expected_frames) > 8:
        raise ValueError("geometry report frame roster differs from observer or exceeds cap")
    terminal = report.get("native_terminal_audit", {})
    if terminal.get("status") != "PASS_NATIVE_COMPACT_TERMINAL_AUDIT" or terminal.get("expected_steps") != steps or terminal.get("accepted_steps") != steps:
        raise ValueError("compact terminal FEM audit does not match requested horizon")
    try:
        minimum, maximum = float(terminal["minimum_J_global"]), float(terminal["maximum_J_global"])
    except (TypeError, ValueError, KeyError):
        raise ValueError("compact terminal FEM audit lacks J bounds") from None
    if not math.isfinite(minimum) or minimum <= 0 or not math.isfinite(maximum) or maximum < minimum:
        raise ValueError("compact terminal FEM J bounds are invalid")
    if report.get("terminal_cell_J", {}).get("mode") != "NATIVE_COMPACT_ALL_TET_AUDIT":
        raise ValueError("geometry report changes terminal J evidence scope")
    if report.get("motion_coverage") not in ("SAMPLED_ACCEPTED_MRV_FRAMES", "ALL_ACCEPTED_STEP_IDS"):
        raise ValueError("unknown accepted-motion coverage")
    return observer, route, report, checkpoint_info


def _verify_source_claims(stage_plan: dict[str, Any], source_plan: dict[str, Any],
                         source_receipt: dict[str, Any], source_receipt_path: Path,
                         source_plan_path: Path) -> None:
    if stage_plan.get("schema") == SCHEMA_PLAN:
        if stage_plan.get("source_plan") != ref(source_plan_path) or stage_plan.get("source_preflight_receipt") != ref(source_receipt_path):
            raise ValueError("continuation plan source lineage changed")
    else:
        if stage_plan.get("source_preflight_receipt_path") != str(source_receipt_path.resolve()) or stage_plan.get("source_preflight_receipt_sha256") != base.sha_file(source_receipt_path):
            raise ValueError("motion plan source receipt pin changed")
    if source_receipt.get("status") != "PASS_SOURCE_ONLY_DIRECT_O_NO_LEGACY_DETAIL_BRIDGE":
        raise ValueError("source preflight receipt no longer passes")
    if stage_plan.get("geometry_admission") != source_plan.get("geometry_admission") or stage_plan.get("mass_admission") != source_plan.get("mass_admission"):
        raise ValueError("motion geometry/mass claims differ from source plan")


def verify_stage(manifest: Path, checkpoint: Path, source_plan_path: Path, source_receipt_path: Path,
                 motion_plan_path: Path, observer_path: Path, route_path: Path, stdout_path: Path,
                 stderr_path: Path, report_path: Path, verifier_path: Path, verifier_sha: str,
                 orientation_audit_path: Path, orientation_audit_sha: str,
                 orientation_owner_path: Path, orientation_owner_sha: str, out: Path):
    source_plan, cfg, route_module, ctx, source_receipt, source_receipt_path = context(manifest, source_plan_path, source_receipt_path)
    static = ctx["static"]
    stage_plan = obj(motion_plan_path)
    if stage_plan.get("manifest_path") != str(manifest.resolve()) or stage_plan.get("manifest_sha256") != base.json_digest(cfg):
        raise ValueError("motion plan belongs to a different exact config")
    if stage_plan.get("build_manifest_sha256") != static["manifest_sha256"] or stage_plan.get("native_binary", {}).get("sha256") != static["native_binary_sha256"]:
        raise ValueError("motion plan build/binary is stale")
    _verify_source_claims(stage_plan, source_plan, source_receipt, source_receipt_path, source_plan_path)
    steps = stage_plan.get("motion", {}).get("requested_accepted_step_count")
    if type(steps) is not int or steps < 2:
        raise ValueError("motion plan horizon is invalid")
    if stage_plan.get("schema") == SCHEMA_PLAN:
        motion = stage_plan["motion"]
        if motion.get("initialization_policy") != MODE or motion.get("checkpoint_restart") is not False:
            raise ValueError("continuation plan falsely claims a checkpoint restart")
        prev_ref = stage_plan.get("previous_gate")
        prev_path, _ = check_ref(prev_ref, "previous horizon gate")
        previous = verify_gate(prev_path, manifest, checkpoint, source_plan_path, source_receipt_path)
        audit_ref, owner_ref = orientation_binding(orientation_audit_path, orientation_audit_sha,
                                                   orientation_owner_path, orientation_owner_sha)
        if stage_plan.get("geometry_orientation_audit") != audit_ref or stage_plan.get("geometry_orientation_owner") != owner_ref:
            raise ValueError("continuation plan fresh orientation audit/owner pins differ")
        if previous.get("geometry_orientation_audit") != audit_ref or previous.get("geometry_orientation_owner") != owner_ref:
            raise ValueError("continuation changed the authenticated role-orientation context")
        if stage_plan.get("checkpoint", {}).get("checkpoint", {}).get("sha256") != previous.get("checkpoint_sha256"):
            raise ValueError("continuation plan changed checkpoint lineage")
        if previous.get("geometry_verifier") != {"path": str(verifier_path.resolve()), "sha256": verifier_sha}:
            raise ValueError("continuation changed the verified geometry verifier")
        if previous.get("accepted_geometry_exporter") != stage_plan.get("accepted_geometry_exporter"):
            raise ValueError("continuation changed the verified geometry exporter")
        if motion.get("horizon_kind") == "progression":
            next_horizon(previous["requested_accepted_steps"], steps)
        elif motion.get("horizon_kind") == "measured_duration_target":
            timing = previous.get("performance")
            if previous.get("requested_accepted_steps") != 1024 or not isinstance(timing, dict):
                raise ValueError("duration target requires a qualified 1024-step measured predecessor")
            expected = duration_horizon(1024, timing.get("wall_s"), motion.get("target_wall_seconds"), motion.get("maximum_steps", MAX_HORIZON_STEPS))
            if steps != expected:
                raise ValueError("duration target horizon differs from measured-performance derivation")
        else:
            raise ValueError("unknown continuation horizon kind")
        if stage_plan.get("geometry_verifier") != {"path": str(verifier_path.resolve()), "sha256": verifier_sha}:
            raise ValueError("continuation geometry-verifier pin differs from plan")
    elif stage_plan.get("status") != "DIRECT_O_MOTION_PLAN_READY_AFTER_SOURCE_PREFLIGHT" or steps != 2:
        raise ValueError("the initial gate accepts only frozen 009's one- or two-step plan, specifically two here")
    if stage_plan.get("motion", {}).get("legacy_detail_bridge_enabled") is not False or stage_plan.get("motion", {}).get("maximum_mrv_captures") != 8 or stage_plan.get("motion", {}).get("full_fem_endpoint_state_dump_requested") is not False:
        raise ValueError("motion plan violates direct-O capture limits")
    audit_ref, owner_ref = orientation_binding(orientation_audit_path, orientation_audit_sha,
                                               orientation_owner_path, orientation_owner_sha)
    observer, route_receipt, report, checkpoint_info = verify_motion_bundle(
        observer_path, route_path, stdout_path, stderr_path, motion_plan_path, checkpoint, steps,
        verifier_path, verifier_sha, report_path, orientation_audit_path, orientation_audit_sha,
        orientation_owner_path, orientation_owner_sha)
    if stage_plan.get("schema") == SCHEMA_PLAN and stage_plan.get("accepted_geometry_exporter") != report.get("accepted_geometry_exporter"):
        raise ValueError("continuation plan accepted-geometry exporter differs from verifier report")
    if route_receipt.get("native_binary_sha256") != stage_plan.get("native_binary", {}).get("sha256"):
        raise ValueError("native route binary differs from the plan")
    if route_receipt.get("source_preflight_receipt_sha256") != base.sha_file(source_receipt_path):
        raise ValueError("native route receipt source pin differs")
    if route_receipt.get("geometry_admission_receipt_sha256") != stage_plan.get("geometry_admission", {}).get("receipt_sha256") or route_receipt.get("mass_admission_receipt_sha256") != stage_plan.get("mass_admission", {}).get("receipt_sha256"):
        raise ValueError("native route receipt geometry/mass pins differ")
    gate = {
        "schema": SCHEMA_GATE, "status": "PASS_NATIVE_DIRECT_O_HORIZON_GATE",
        "measurement_scope": "accepted-state motion geometry measurement; not physical or settled-rest qualification",
        "initialization_policy": MODE, "native_runtime_executed": True, "gpu_executed": True,
        "physical_admission": False, "long_horizon_qualification": False, "requested_accepted_steps": steps,
        "accepted_geometry_steps": sorted(int(k) for k in observer["accepted_geometry_frames"]),
        "maximum_mrv_captures": 8, "manifest_path": str(manifest.resolve()),
        "manifest_sha256": base.json_digest(cfg), "build_manifest_sha256": static["manifest_sha256"],
        "native_binary": {"path": static["build"]["build"]["binary"], "sha256": static["native_binary_sha256"]},
        "checkpoint_sha256": checkpoint_info["checkpoint"]["sha256"],
        "source_state": checkpoint_info["source_state"], "217_handoff": checkpoint_info["217_handoff"],
        "registered_active_source_crosswalk": checkpoint_info["registered_active_source_crosswalk"],
        "source_plan": ref(source_plan_path), "source_preflight_receipt": ref(source_receipt_path),
        "motion_plan": ref(motion_plan_path), "observer": ref(observer_path),
        "install_route_receipt": ref(route_path), "native_stdout": ref(stdout_path),
        "native_stderr": ref(stderr_path), "geometry_report": ref(report_path),
        "geometry_verifier": {"path": str(verifier_path.resolve()), "sha256": verifier_sha},
        "geometry_orientation_audit": audit_ref,
        "geometry_orientation_owner": owner_ref,
        "accepted_geometry_exporter": report["accepted_geometry_exporter"],
        "geometry_status": report["status"], "terminal_fem_audit": report["native_terminal_audit"],
        "motion_coverage": report["motion_coverage"],
        "omitted_accepted_step_id_count": report["omitted_accepted_step_id_count"],
        "performance": parse_performance(stdout_path.read_text(errors="replace")),
        "previous_gate": stage_plan.get("previous_gate"),
    }
    base.write_once(out, gate)
    return gate


def make_plan(manifest: Path, checkpoint: Path, source_plan_path: Path, source_receipt_path: Path,
              previous_gate_path: Path, out: Path, steps: int, verifier_path: Path, verifier_sha: str,
              orientation_audit_path: Path, orientation_audit_sha: str,
              orientation_owner_path: Path, orientation_owner_sha: str,
              exporter_path: Path, exporter_sha: str, horizon_kind: str,
              target_wall_seconds: float | None = None, maximum_steps: int = MAX_HORIZON_STEPS):
    if not verifier_path.is_file() or base.sha_file(verifier_path) != verifier_sha:
        raise ValueError("geometry verifier source path/SHA changed")
    if not exporter_path.is_file() or base.sha_file(exporter_path) != exporter_sha:
        raise ValueError("accepted-geometry exporter path/SHA changed")
    audit_ref, owner_ref = orientation_binding(orientation_audit_path, orientation_audit_sha,
                                               orientation_owner_path, orientation_owner_sha)
    previous = verify_gate(previous_gate_path, manifest, checkpoint, source_plan_path, source_receipt_path)
    if previous.get("geometry_verifier") != {"path": str(verifier_path.resolve()), "sha256": verifier_sha}:
        raise ValueError("continuation must reuse the verified prior-stage geometry verifier")
    if previous.get("accepted_geometry_exporter") != {"path": str(exporter_path.resolve()), "sha256": exporter_sha}:
        raise ValueError("continuation must reuse the verified prior-stage geometry exporter")
    if previous.get("geometry_orientation_audit") != audit_ref or previous.get("geometry_orientation_owner") != owner_ref:
        raise ValueError("continuation must reuse the verified prior-stage orientation audit and owner")
    if horizon_kind == "progression":
        next_horizon(previous["requested_accepted_steps"], steps)
        target = None
    elif horizon_kind == "measured_duration_target":
        if previous.get("requested_accepted_steps") != 1024 or not isinstance(previous.get("performance"), dict):
            raise ValueError("duration target requires actual measured performance from 1024 steps")
        expected = duration_horizon(1024, previous["performance"].get("wall_s"), target_wall_seconds, maximum_steps)
        if steps != expected:
            raise ValueError(f"duration target step count must be {expected} based on measured native performance")
        target = finite_positive(target_wall_seconds, "target wall seconds")
    else:
        raise ValueError("unsupported continuation horizon kind")
    caps = capture_steps(steps)
    source, cfg, route, ctx, _, receipt_path = context(manifest, source_plan_path, source_receipt_path)
    static = ctx["static"]
    root = first.root_path(cfg, allow=True)
    attempt = re.sub(r"[^A-Za-z0-9_-]+", "-", out.parent.name).strip("-")
    if not attempt:
        raise ValueError("continuation job directory must have a nonempty safe name")
    install = root / f"install-horizon-{steps}-{attempt}"
    if install.exists():
        raise FileExistsError(f"refusing to reuse continuation install directory: {install}")
    argv, physics, stale = base.safe_scene(cfg, static, route, install, steps)
    scene = dict(physics)
    for key in stale:
        scene.pop(key, None)
    env = first.apply_contact_capacity_trial(
        first.apply_direct_env(route, scene, ctx["env"], False, False, ctx["fractional"], ctx["recorded"]))
    env.update({
        "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT": "1",
        "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS": "1",
        "NUMI_HUMAN_ACCEPTED_BODY_MOTION_AUDIT": "1",
        "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT": "1",
        "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_FIRST_STEP": "1",
        "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT_LAST_STEP": str(steps),
        "NUMI_HUMAN_CAPTURE_FEM_ENDPOINT_STATE": "0",
        "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS": ",".join(map(str, caps)),
    })
    route.validate_full_scene_mode(argv[1:], env)
    stage = {"name": "direct_O_install_and_fresh_initialized_extended_horizon_motion",
             "argv": argv, "environment": env, "expected_runtime": True,
             "expected_accepted_steps": steps, "accepted_geometry_steps": caps,
             "cwd": source.get("cwd")}
    plan = {
        "schema": SCHEMA_PLAN, "status": "READY_FRESH_INITIALIZATION_EXTENDED_HORIZON",
        "manifest_path": str(manifest.resolve()), "manifest_sha256": base.json_digest(cfg),
        "source_plan": ref(source_plan_path), "source_preflight_receipt": ref(receipt_path),
        "previous_gate": ref(previous_gate_path), "checkpoint": checkpoint_binding(checkpoint),
        "build_manifest_sha256": static["manifest_sha256"],
        "native_binary": {"path": static["build"]["build"]["binary"], "sha256": static["native_binary_sha256"]},
        "geometry_admission": source.get("geometry_admission"), "mass_admission": source.get("mass_admission"),
        "geometry_verifier": {"path": str(verifier_path.resolve()), "sha256": verifier_sha},
        "geometry_orientation_audit": audit_ref,
        "geometry_orientation_owner": owner_ref,
        "accepted_geometry_exporter": {"path": str(exporter_path.resolve()), "sha256": exporter_sha},
        "stage": stage,
        "motion": {"requested_accepted_step_count": steps, "horizon_kind": horizon_kind,
                   "target_wall_seconds": target, "maximum_steps": maximum_steps,
                   "initialization_policy": MODE, "checkpoint_restart": False,
                   "measurement_scope": "accepted-state motion geometry measurement; not physical or settled-rest qualification",
                   "accepted_geometry_steps": caps, "maximum_mrv_captures": 8,
                   "trace_files_required": list(base.TRACE_FILES), "terminal_fem_audit_required": True,
                   "full_fem_endpoint_state_dump_requested": False, "viewer_path_required": True,
                   "legacy_detail_bridge_enabled": False},
        "execution_policy": {"launch_performed": False, "native_runtime_executed": False,
                             "gpu_executed": False, "physical_admission": False},
    }
    base.write_once(out, plan)
    shell = out.with_suffix(".stage.sh")
    if shell.exists():
        raise FileExistsError(f"refusing to overwrite continuation command: {shell}")
    shell.write_text(base.stage_shell(route, stage, stage["cwd"]))
    shell.chmod(0o755)
    return plan


def record_run(manifest: Path, source_plan_path: Path, source_receipt_path: Path,
               motion_plan_path: Path, stdout_path: Path, stderr_path: Path,
               exit_code: int, observer_out: Path):
    plan = obj(motion_plan_path)
    if plan.get("schema") != SCHEMA_PLAN or plan.get("status") != "READY_FRESH_INITIALIZATION_EXTENDED_HORIZON":
        raise ValueError("continuation plan is not ready")
    if exit_code != 0:
        raise ValueError(f"native extended-horizon run exited with {exit_code}")
    source, cfg, _, ctx, _, receipt_path = context(manifest, source_plan_path, source_receipt_path)
    if plan.get("manifest_path") != str(manifest.resolve()) or plan.get("manifest_sha256") != base.json_digest(cfg):
        raise ValueError("continuation config identity changed")
    if plan.get("source_plan") != ref(source_plan_path) or plan.get("source_preflight_receipt") != ref(receipt_path):
        raise ValueError("continuation source lineage changed")
    if plan.get("build_manifest_sha256") != ctx["static"]["manifest_sha256"] or plan.get("native_binary", {}).get("sha256") != ctx["static"]["native_binary_sha256"]:
        raise ValueError("continuation build/binary identity changed")
    for key in ("geometry_orientation_audit", "geometry_orientation_owner"):
        path, digest = check_ref(plan.get(key), f"motion plan {key}")
        if base.sha_file(path) != digest:
            raise ValueError(f"motion plan {key} changed")
    audit_ref, owner_ref = plan["geometry_orientation_audit"], plan["geometry_orientation_owner"]
    orientation_binding(Path(audit_ref["path"]), audit_ref["sha256"],
                        Path(owner_ref["path"]), owner_ref["sha256"])
    motion = plan["motion"]; steps = motion["requested_accepted_step_count"]
    stdout, stderr = stdout_path.read_text(errors="replace"), stderr_path.read_text(errors="replace")
    if "whole_body_skin_fat_registered_volume_install=PASS" not in stdout or "whole_body_skin_fat_registered_volume_install=FAIL" in stdout + stderr:
        raise ValueError("registered-volume install PASS absent or FAIL present")
    direct = first.parse_o_pass(stdout)
    if (steps, steps) not in [tuple(map(int, x)) for x in TERMINAL.findall(stdout)] or "resting_FEM_terminal_audit=failed" in stdout + stderr:
        raise ValueError("native terminal FEM audit lacks requested accepted-step pass")
    outdir = Path(plan["stage"]["argv"][4]); traces = {}
    for name in base.TRACE_FILES:
        path = outdir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"accepted-state trace missing or empty: {path}")
        traces[str(path)] = {"sha256": base.sha_file(path), "bytes": path.stat().st_size}
    frames = {}; accepted = outdir / "accepted-geometry"
    for step in motion["accepted_geometry_steps"]:
        pack, rec_path = accepted / f"step-{step}.mrvpack", accepted / f"step-{step}.receipt.json"
        if not pack.is_file() or not rec_path.is_file():
            raise ValueError(f"accepted MRV frame/receipt missing at step {step}")
        rec = obj(rec_path); digest = base.sha_file(pack)
        if rec.get("accepted_step") != step or rec.get("physical_endpoint") != "accepted" or rec.get("surface_audit_endpoint") != "passed" or rec.get("pack_file_sha256") != digest:
            raise ValueError(f"accepted MRV receipt invalid at step {step}")
        if rec.get("position_normal_tangent_source") != "accepted-state renderer mesh buffer copied on the same Metal command buffer":
            raise ValueError(f"accepted MRV frame {step} lacks same-command-buffer provenance")
        frames[str(step)] = {"pack_path": str(pack), "pack_sha256": digest,
                             "receipt_path": str(rec_path), "receipt_sha256": base.sha_file(rec_path)}
    if len(frames) > 8:
        raise ValueError("MRV capture count exceeds eight")
    route_path = observer_out.with_name(observer_out.stem + ".route.json")
    if route_path.exists():
        raise FileExistsError(f"refusing to overwrite continuation route receipt: {route_path}")
    route_receipt = {
        "schema": "numi.human.current-native-direct-o-motion-receipt.v1",
        "status": "PASS_INSTALL_AND_ACCEPTED_STATE_MOTION",
        "plan_sha256": base.sha_file(motion_plan_path), "stdout_sha256": base.sha_file(stdout_path),
        "stderr_sha256": base.sha_file(stderr_path), "native_exit_code": exit_code,
        "native_binary_sha256": plan["native_binary"]["sha256"],
        "geometry_admission_receipt_sha256": plan["geometry_admission"]["receipt_sha256"],
        "geometry_qualification_scope": "source_geometry_only", "physical_admission": False,
        "mass_admission_receipt_sha256": plan["mass_admission"]["receipt_sha256"],
        "source_preflight_receipt_sha256": base.sha_file(receipt_path),
        "registered_volume_install_pass": True, "direct_O": direct,
        "legacy_detail_bridge_enabled": False, "requested_accepted_steps": steps,
        "observed_accepted_steps": steps, "native_runtime_executed": True, "gpu_executed": True,
        "physical_admission_claimed": False, "long_horizon_qualification_claimed": False,
        "initialization_policy": MODE,
    }
    base.write_once(route_path, route_receipt)
    observer = {
        "schema": "numi.human.current-native-direct-o-measurement-observer-receipt.v1",
        "status": "PASS_BOUNDED_DIRECT_O_TRACES_AND_MRV", "plan_sha256": base.sha_file(motion_plan_path),
        "stdout_sha256": base.sha_file(stdout_path), "stderr_sha256": base.sha_file(stderr_path),
        "native_stdout_path": str(stdout_path.resolve()), "native_stderr_path": str(stderr_path.resolve()),
        "requested_accepted_steps": steps, "measurement_scope": motion["measurement_scope"],
        "direct_O": direct, "accepted_trace_files": traces, "accepted_geometry_frames": frames,
        "terminal_fem_audit_verified": True, "full_fem_endpoint_state_dump_requested": False,
        "native_runtime_executed": True, "gpu_executed": True, "physical_admission_claimed": False,
        "long_horizon_qualification_claimed": False, "initialization_policy": MODE,
        "performance": parse_performance(stdout),
        "install_route_receipt": {"path": str(route_path), "sha256": base.sha_file(route_path), "status": route_receipt["status"]},
    }
    base.write_once(observer_out, observer)
    return observer


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest="cmd", required=True)
    common = ("manifest", "checkpoint", "source-plan", "source-receipt", "previous-gate", "out", "verifier", "exporter")
    for name in ("plan-horizon", "plan-duration"):
        q = sub.add_parser(name)
        for flag in common: q.add_argument("--" + flag, required=True, type=Path)
        q.add_argument("--verifier-sha256", required=True); q.add_argument("--exporter-sha256", required=True)
        q.add_argument("--orientation-audit", required=True, type=Path)
        q.add_argument("--orientation-audit-sha256", required=True)
        q.add_argument("--orientation-owner", required=True, type=Path)
        q.add_argument("--orientation-owner-sha256", required=True)
        if name == "plan-horizon": q.add_argument("--steps", required=True, type=int, choices=(16, 128, 1024))
        else:
            q.add_argument("--target-wall-seconds", required=True, type=float)
            q.add_argument("--maximum-steps", type=int, default=MAX_HORIZON_STEPS)
    q = sub.add_parser("record")
    for flag in ("manifest", "source-plan", "source-receipt", "motion-plan", "stdout", "stderr", "observer-out"):
        q.add_argument("--" + flag, required=True, type=Path)
    q.add_argument("--exit-code", required=True, type=int)
    q = sub.add_parser("verify-stage")
    for flag in ("manifest", "checkpoint", "source-plan", "source-receipt", "motion-plan", "observer", "route", "stdout", "stderr", "geometry-report", "verifier", "out"):
        q.add_argument("--" + flag, required=True, type=Path)
    q.add_argument("--verifier-sha256", required=True)
    q.add_argument("--orientation-audit", required=True, type=Path)
    q.add_argument("--orientation-audit-sha256", required=True)
    q.add_argument("--orientation-owner", required=True, type=Path)
    q.add_argument("--orientation-owner-sha256", required=True)
    a = p.parse_args()
    try:
        if a.cmd in ("plan-horizon", "plan-duration"):
            if a.cmd == "plan-horizon":
                gate = verify_gate(a.previous_gate, a.manifest, a.checkpoint, a.source_plan, a.source_receipt)
                steps, kind, target, maximum = next_horizon(gate["requested_accepted_steps"], a.steps), "progression", None, MAX_HORIZON_STEPS
            else:
                gate = verify_gate(a.previous_gate, a.manifest, a.checkpoint, a.source_plan, a.source_receipt)
                if not isinstance(gate.get("performance"), dict): raise ValueError("predecessor lacks actual measured performance")
                maximum = a.maximum_steps
                steps = duration_horizon(gate["requested_accepted_steps"], gate["performance"].get("wall_s"), a.target_wall_seconds, maximum)
                kind, target = "measured_duration_target", a.target_wall_seconds
            result = make_plan(a.manifest, a.checkpoint, a.source_plan, a.source_receipt, a.previous_gate, a.out, steps,
                a.verifier, a.verifier_sha256,
                a.orientation_audit, a.orientation_audit_sha256, a.orientation_owner, a.orientation_owner_sha256,
                a.exporter, a.exporter_sha256, kind, target, maximum)
            print(json.dumps({"status": result["status"], "steps": steps, "plan": str(a.out),
                              "command": str(a.out.with_suffix(".stage.sh")), "capture_steps": result["motion"]["accepted_geometry_steps"]}, sort_keys=True))
        elif a.cmd == "record":
            result = record_run(a.manifest, a.source_plan, a.source_receipt, a.motion_plan, a.stdout, a.stderr, a.exit_code, a.observer_out)
            print(json.dumps({"status": result["status"], "observer": str(a.observer_out)}, sort_keys=True))
        else:
            result = verify_stage(a.manifest, a.checkpoint, a.source_plan, a.source_receipt, a.motion_plan,
                a.observer, a.route, a.stdout, a.stderr, a.geometry_report, a.verifier, a.verifier_sha256,
                a.orientation_audit, a.orientation_audit_sha256, a.orientation_owner, a.orientation_owner_sha256, a.out)
            print(json.dumps({"status": result["status"], "steps": result["requested_accepted_steps"], "gate": str(a.out)}, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
