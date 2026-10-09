#!/usr/bin/env python3
"""Analyze a completed direct native control/treatment pair with existing owner physiology checks."""
from __future__ import annotations
import argparse, csv, gc, hashlib, importlib.util, json, math, statistics, struct
from pathlib import Path

OWNER = Path("/Users/n/numi-human-terminal-trace-capture-fix-1162/matter/tools/resting_intervention_study.py")
OWNER_SHA = "ef87b09a17b96c985d6f584932e806ba9e7c6a53785da32a37a56ac846a3ac5f"
SUPPORT_OWNER = Path("/Users/n/numi-human-retained-delivery-20261009/native-final-pair-postprocessing-1170/revision-003/analyze_final_pair.py")
SUPPORT_OWNER_SHA = "fa12948701aa0359017d3b9a204bd306afaa78f188f0ca46fc462a907e29eb7c"
STEPS, DT = 155000, 0.002
START, END, SCALE, WINDOW = 60.0, 100.0, 0.5, 30.0

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_module(path: Path, expected: str, name: str):
    if path.is_symlink() or not path.is_file() or sha(path) != expected:
        raise ValueError(f"pinned owner source changed: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ValueError(f"cannot load owner module: {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file(): raise ValueError(f"missing or linked evidence: {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict): raise ValueError(f"expected JSON object: {path}")
    return result


def canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def verify_owner_cli_binding(run_decl_path: Path, direct_decl_path: Path, direct_decl: dict,
                             treatment_path: Path) -> dict:
    run_decl_path = run_decl_path.resolve(strict=True)
    run_decl = read_json(run_decl_path)
    if run_decl.get("schema") != "numi.human.resting.native-run.declaration.v1":
        raise ValueError("treatment does not use the existing Human native-run declaration schema")
    if Path(run_decl["argv"][run_decl["argv"].index("--output") + 1]).resolve() != treatment_path.resolve():
        raise ValueError("native-run declaration output path differs from the treatment run")
    direct_ref = run_decl.get("direct_treatment_declaration", {})
    if (direct_ref.get("path") != str(direct_decl_path.resolve()) or
            direct_ref.get("sha256") != sha(direct_decl_path)):
        raise ValueError("native-run declaration does not bind this exact direct treatment declaration")
    preflight_ref = run_decl.get("owner_cli_preflight", {})
    preflight_path = Path(preflight_ref.get("path", ""))
    if (preflight_path.is_symlink() or not preflight_path.is_file() or
            sha(preflight_path) != preflight_ref.get("sha256")):
        raise ValueError("owner CLI native-command preflight is missing or changed")
    preflight = read_json(preflight_path)
    if preflight.get("schema") != "numi.human.owner-cli-native-command-preflight.v1":
        raise ValueError("unexpected owner CLI preflight schema")
    if (preflight.get("direct_treatment_declaration", {}).get("path") != str(direct_decl_path.resolve()) or
            preflight.get("direct_treatment_declaration", {}).get("sha256") != sha(direct_decl_path)):
        raise ValueError("owner CLI preflight is not bound to this direct treatment declaration")
    expected_argv = direct_decl["treatment"]["argv"]
    expected_assets = direct_decl["treatment"]["immutable_assets"]
    if (preflight.get("owner_cli_native_argv") != expected_argv or
            preflight.get("direct_expected_native_argv_sha256") != canonical_sha(expected_argv) or
            preflight.get("owner_cli_native_argv_sha256") != canonical_sha(expected_argv) or
            preflight.get("native_asset_sha256") != expected_assets or
            preflight.get("argv_equal_direct_declaration") is not True or
            preflight.get("native_assets_equal_direct_declaration") is not True):
        raise ValueError("owner CLI preflight does not prove exact native command/asset parity")
    argv_digest = canonical_sha(run_decl["argv"])
    if preflight.get("top_level_treatment_argv_sha256") != argv_digest:
        raise ValueError("preflight top-level CLI command differs from native-run declaration")
    execution_path = run_decl_path.parent / "execution.json"
    execution = read_json(execution_path)
    if (execution.get("returncode") != 0 or execution.get("changed_inputs") != {} or
            execution.get("declaration_sha256") != sha(run_decl_path)):
        raise ValueError("existing Human run.py execution receipt failed or changed its declaration inputs")
    return {"native_run_declaration_path": str(run_decl_path),
            "native_run_declaration_sha256": sha(run_decl_path),
            "owner_cli_preflight_path": str(preflight_path),
            "owner_cli_preflight_sha256": sha(preflight_path),
            "run_execution_path": str(execution_path), "run_execution_sha256": sha(execution_path),
            "expected_native_argv_sha256": canonical_sha(expected_argv),
            "actual_invocation_expected_argv": expected_argv,
            "owner_cli_preflight_passed": True}

def check_run(directory: Path, arm: str, declaration: dict, owner):
    directory = directory.resolve(strict=True)
    inv_path = directory / "invocation.json"; metadata_path = directory / "run-metadata.json"
    log_path = directory / "native.log"; trace_path = directory / "resting-coupled.csv"
    invocation, metadata = read_json(inv_path), read_json(metadata_path)
    if (metadata.get("exit_code") != 0 or metadata.get("source_files_changed_during_run") != [] or
            metadata.get("argv") != invocation.get("argv") or metadata.get("asset_sha256") != invocation.get("asset_sha256") or
            metadata.get("environment") != invocation.get("environment") or
            metadata.get("loaded_metal_runtime", {}).get("verified") is not True):
        raise ValueError(f"{arm} is not a completed unchanged native Human run")
    if not log_path.is_file() or log_path.is_symlink() or not trace_path.is_file() or trace_path.is_symlink():
        raise ValueError(f"{arm} lacks native log or accepted physiology trace")
    for path, expected in invocation.get("asset_sha256", {}).items():
        if not Path(path).is_file() or sha(Path(path)) != expected:
            raise ValueError(f"{arm} input asset changed: {path}")
    if arm == "control":
        pinned = declaration["baseline"]
        if (str(directory) != pinned["run_path"] or sha(inv_path) != pinned["invocation_sha256"] or
                sha(metadata_path) != pinned["run_metadata_sha256"] or sha(log_path) != pinned["native_log_sha256"] or
                sha(trace_path) != pinned["trace_sha256"]):
            raise ValueError("historical control differs from the closed baseline declaration pins")
    native = owner.native_scene_summary(log_path.read_text(encoding="utf-8", errors="replace"))
    if native["accepted_steps"] != STEPS or abs(native["simulated_s"] - STEPS * DT) > 1e-4:
        raise ValueError(f"{arm} did not complete 155000 accepted 2 ms roots")
    if native["device"] != declaration["baseline"]["device"] or native["world_fingerprint"] != declaration["baseline"]["world_fingerprint"]:
        raise ValueError(f"{arm} device/world differs from the direct baseline")
    if native["body_source_fingerprint"] != declaration["baseline"]["body_source_fingerprint"]:
        raise ValueError(f"{arm} body source fingerprint differs from the matched baseline")
    if arm == "control":
        owner.validate_native_310s_invocation(invocation)
        capture_steps = [int(x) for x in invocation["environment"]["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"].split(",")]
        if capture_steps != declaration["baseline"]["capture_steps"]:
            raise ValueError("historical control capture schedule differs from its closed declaration")
    else:
        expected = declaration["treatment"]
        if invocation.get("argv") != expected["argv"] or invocation.get("environment") != expected["environment"] or invocation.get("asset_sha256") != expected["immutable_assets"]:
            raise ValueError("treatment invocation differs from the prepared treatment declaration")
        if metadata.get("loaded_metal_runtime") != declaration["baseline"]["loaded_runtime"]:
            raise ValueError("treatment loaded runtime differs from the direct baseline runtime identity")
        if "--resting-drive-intervention" not in invocation["argv"]:
            raise ValueError("treatment invocation omits the declared intervention")
        i = invocation["argv"].index("--resting-drive-intervention")
        if invocation["argv"][i+1:i+4] != ["60.0", "100.0", "0.5"]:
            raise ValueError("treatment drive settings differ from [60,100) at 0.5")
        capture_steps = expected["capture_steps"]
    # Every requested accepted pack and its receipt must be explicitly named and hash-matched by native.log.
    export_lines = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    if line.startswith("accepted_geometry_export=")]
    for step in capture_steps:
        pack = directory / "accepted-geometry" / f"step-{step}.mrvpack"
        receipt_path = directory / "accepted-geometry" / f"step-{step}.receipt.json"
        if not pack.is_file() or not receipt_path.is_file(): raise ValueError(f"{arm} missing accepted capture {step}")
        matching = [line for line in export_lines if line.startswith(f"accepted_geometry_export={pack} ")]
        if len(matching) != 1: raise ValueError(f"{arm} native log does not uniquely bind capture {step}")
        fields = dict(x.split("=",1) for x in matching[0].split()[1:] if "=" in x)
        receipt = read_json(receipt_path)
        if (fields.get("pack_sha256") != sha(pack) or fields.get("receipt_sha256") != sha(receipt_path) or
                receipt.get("accepted_step") != step or receipt.get("pack_file_sha256") != sha(pack) or
                receipt.get("accepted_pack_path") != str(pack)):
            raise ValueError(f"{arm} accepted capture receipt/hash mismatch at step {step}")
    terminal = owner.native_terminal_accepted_capture_evidence(
        invocation, directory, native, log_path.read_text(encoding="utf-8", errors="replace"), STEPS, DT)
    args = argparse.Namespace(steps=STEPS, dt=DT, arm=arm, start_s=START, end_s=END,
                              scale=SCALE, window_s=WINDOW, unit_id=declaration["matched_unit_id"])
    obs = owner.observation(args, trace_path, native, log_path.read_text(encoding="utf-8", errors="replace"))
    body = owner.native_body_trace_consistency(trace_path, STEPS, DT)
    surface = owner.native_surface_trace_consistency(directory / "resting-surface-audit.csv", STEPS, DT,
                require_whole_mesh=True, terminal_accepted_capture=terminal["verified"])
    resp_path = Path(invocation["argv"][invocation["argv"].index("--resting-scene") + 2])
    respiratory = owner.native_respiration_trace_consistency(
        trace_path, resp_path, {name: obs[name + "_window_s"] for name in ("pre", "dose", "recovery")})
    cycle = owner.native_cycle_coverage(owner.read_trace(trace_path), DT)
    support = None
    support_path = directory / "resting-com-momentum-diagnostic.csv"
    if support_path.is_file():
        p18 = load_module(SUPPORT_OWNER, SUPPORT_OWNER_SHA, "p18_support_helper")
        support = p18.support_summary(support_path, {"scene/resting-com-momentum-diagnostic.csv": sha(support_path)}, STEPS*DT)
    result = {"directory": str(directory), "invocation_sha256": sha(inv_path), "run_metadata_sha256": sha(metadata_path),
              "native_log_sha256": sha(log_path), "trace_sha256": sha(trace_path),
              "asset_sha256": invocation["asset_sha256"], "native": native,
              "native_return_code": metadata.get("exit_code"),
              "loaded_metal_runtime": metadata.get("loaded_metal_runtime"),
              "owner_observation": obs, "body_trace_consistency": body,
              "surface_trace_consistency": surface, "respiratory_identity": respiratory,
              "cycle_coverage": cycle, "support_com": support,
              "timings_s": {"owner_run_metadata_wall_seconds": metadata.get("wall_seconds"),
                             "native_integrated_wall_s": native["wall_s"], "native_gpu_s": native["gpu_s"],
                             "native_rtf": native["real_time_factor"]},
              "capture_steps": capture_steps}
    return result

TRACE_STEP_STRIDE = 8

def verify_matched_accepted_grid(control_rows, treatment_rows, *, dt: float, steps: int,
                                 stride: int = TRACE_STEP_STRIDE) -> dict:
    """Fail closed unless both traces contain the same complete accepted grid."""
    if not math.isfinite(dt) or dt <= 0 or steps <= 0 or stride <= 0 or steps % stride:
        raise ValueError("invalid declared accepted grid parameters")
    expected_steps = tuple(range(stride, steps + 1, stride))

    def parse(rows, arm):
        parsed = []
        previous = 0
        for index, row in enumerate(rows):
            try:
                raw_step = float(row["step"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{arm} accepted grid has missing/invalid step at row {index}") from exc
            if not math.isfinite(raw_step) or not raw_step.is_integer():
                raise ValueError(f"{arm} accepted grid has nonfinite/noninteger step at row {index}")
            step = int(raw_step)
            if step <= previous:
                raise ValueError(f"{arm} accepted steps are duplicate or out of order at row {index}")
            if step > steps or step % stride:
                raise ValueError(f"{arm} accepted step {step} is outside the declared cadence")
            try:
                time_s = float(row["time_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{arm} accepted grid has missing/invalid time at row {index}") from exc
            if not math.isfinite(time_s):
                raise ValueError(f"{arm} accepted grid has nonfinite time at row {index}")
            native_dt = struct.unpack("<f", struct.pack("<f", dt))[0]
            expected_time_s = step * native_dt
            # The pinned native CSV writes 12 significant digits. Require the exact
            # serialized Float32-dt time so a shared row shift cannot pass by tolerance.
            expected_serialized_time_s = float(format(expected_time_s, ".12g"))
            if time_s != expected_serialized_time_s:
                raise ValueError(f"{arm} accepted time is shifted from step*Float32(dt) at step {step}")
            parsed.append((step, time_s))
            previous = step
        actual_steps = tuple(step for step, _ in parsed)
        if actual_steps != expected_steps:
            mismatch = next((i for i, (actual, expected) in enumerate(zip(actual_steps, expected_steps))
                             if actual != expected), min(len(actual_steps), len(expected_steps)))
            actual = actual_steps[mismatch] if mismatch < len(actual_steps) else None
            expected = expected_steps[mismatch] if mismatch < len(expected_steps) else None
            raise ValueError(f"{arm} accepted grid is incomplete or shifted at index {mismatch}: expected {expected}, got {actual}")
        return tuple(parsed)

    control = parse(control_rows, "control")
    treatment = parse(treatment_rows, "treatment")
    if control != treatment:
        mismatch = next(i for i, pair in enumerate(zip(control, treatment)) if pair[0] != pair[1])
        raise ValueError(f"matched arms have different accepted step/time at index {mismatch}: {control[mismatch]} vs {treatment[mismatch]}")
    return {"passed": True, "row_count": len(control), "step_stride": stride,
            "first_step": control[0][0], "last_step": control[-1][0],
            "first_time_s": control[0][1], "last_time_s": control[-1][1],
            "exact_step_time_sequence_sha256": canonical_sha([[step, time_s] for step, time_s in control]),
            "comparison": "All expected accepted rows are present in strict order; each time agrees with step times the declared Float32 dt using the pinned 12-significant-digit trace serialization; arms have exactly equal parsed (step,time_s) sequences."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--declaration", type=Path, required=True)
    p.add_argument("--native-run-declaration", type=Path, required=True,
                   help="existing Human owner-CLI run declaration and execution receipt")
    p.add_argument("--treatment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.is_symlink(): raise ValueError("refusing to overwrite a prior analysis")
    owner = load_module(OWNER, OWNER_SHA, "resting_intervention_study_1162")
    declaration = read_json(a.declaration)
    if declaration.get("status") != "prepared_not_launched" or declaration.get("schema") != "numi.human.direct-matched-treatment-declaration.v1":
        raise ValueError("unexpected direct treatment declaration")
    declaration["matched_unit_id"] = hashlib.sha256(json.dumps(
        {"assets": declaration["treatment"]["immutable_assets"], "steps": STEPS, "dt": DT,
         "world_fingerprint": declaration["baseline"]["world_fingerprint"]},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    owner_cli_binding = verify_owner_cli_binding(a.native_run_declaration, a.declaration, declaration, a.treatment)
    control_trace = Path(declaration["baseline"]["run_path"]) / "resting-coupled.csv"
    treatment_trace = a.treatment / "resting-coupled.csv"
    accepted_time_grid = verify_matched_accepted_grid(
        owner.read_trace(control_trace), owner.read_trace(treatment_trace), dt=DT, steps=STEPS)
    control = check_run(Path(declaration["baseline"]["run_path"]), "control", declaration, owner)
    treatment = check_run(a.treatment, "treatment", declaration, owner)
    if control["asset_sha256"] != treatment["asset_sha256"]:
        raise ValueError("matched arms do not use byte-identical native assets")
    c, t = control["owner_observation"], treatment["owner_observation"]
    pair = {
        "primary_delta_PaCO2_mmhg": t["primary_delta_PaCO2_mmhg"] - c["primary_delta_PaCO2_mmhg"],
        "dose_ventilation_treatment_minus_control_L_min": t["inspiratory_minute_ventilation_dose_L_min"] - c["inspiratory_minute_ventilation_dose_L_min"],
        "dose_PaCO2_treatment_minus_control_mmhg": t["PaCO2_dose_mean_mmhg"] - c["PaCO2_dose_mean_mmhg"],
        "dose_PaO2_treatment_minus_control_mmhg": t["PaO2_dose_mean_mmhg"] - c["PaO2_dose_mean_mmhg"],
        "recovery_PaCO2_treatment_minus_control_mmhg": t["PaCO2_recovery_mean_mmhg"] - c["PaCO2_recovery_mean_mmhg"],
        "recovery_PaO2_treatment_minus_control_mmhg": t["PaO2_recovery_mean_mmhg"] - c["PaO2_recovery_mean_mmhg"],
        "recovery_ventilation_treatment_minus_control_L_min": t["inspiratory_minute_ventilation_recovery_L_min"] - c["inspiratory_minute_ventilation_recovery_L_min"],
    }
    report = {"schema": "numi.human.direct-native-pair-analysis.v2", "status": "measured",
              "qualification": "One matched deterministic native baseline/treatment pair; not population inference, anatomy qualification, or clinical validation.",
              "registration_scope": "Direct baseline reused as historical control; this was not prospectively registered as a paired allocation.",
              "declaration_sha256": sha(a.declaration), "owner_cli_binding": owner_cli_binding,
              "owner": {"path": str(OWNER), "sha256": OWNER_SHA},
              "support_helper": {"path": str(SUPPORT_OWNER), "sha256": SUPPORT_OWNER_SHA},
              "accepted_time_grid": accepted_time_grid,
              "arm_results": {"control": control, "treatment": treatment}, "paired_changes": pair,
              "program_fingerprint_relation": "equal" if control["native"]["coupled_program_fingerprint"] == treatment["native"]["coupled_program_fingerprint"] else "different",
              "limitations": ["Native accepted-state and conservation checks are numerical consistency checks.",
                              "Support/COM values are sampled diagnostics, not proof of static equilibrium.",
                              "Mechanics identities and measured pressure/flow do not establish independent physiological calibration or clinical validity.",
                              "No anatomical interface qualification is implied by this analysis."]}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(a.output)

if __name__ == "__main__":
    try: main()
    except (OSError, ValueError, KeyError, IndexError) as exc: raise SystemExit(f"direct pair analysis refused: {exc}")
