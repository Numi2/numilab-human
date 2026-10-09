#!/usr/bin/env python3
"""Prepare (never launch) a matched 310 s half-drive native treatment from a closed direct baseline."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path

OWNER = Path("/Users/n/numi-human-terminal-trace-capture-fix-1162/matter/tools/resting_intervention_study.py")
OWNER_SHA = "ef87b09a17b96c985d6f584932e806ba9e7c6a53785da32a37a56ac846a3ac5f"
BASELINE_DEFAULT = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/native-baseline-310s-preparation/native-run")
TREATMENT_DEFAULT = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/native-treatment-310s-drive60-100-scale050/native-run")
CAPTURES, STEPS, DT = [0, 49999, 55007, 155000], 155000, 0.002
START, END, SCALE, WINDOW = 60.0, 100.0, 0.5, 30.0

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_owner():
    if not OWNER.is_file() or sha(OWNER) != OWNER_SHA:
        raise ValueError("pinned native paired-run helper changed")
    spec = importlib.util.spec_from_file_location("resting_intervention_study_1162", OWNER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def read_object(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required direct-run artifact is missing or linked: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

def build_template(baseline: Path, treatment_output: Path) -> dict:
    if (CAPTURES != sorted(set(CAPTURES)) or CAPTURES[0] != 0 or CAPTURES[-1] != STEPS or
            any(x not in (0, STEPS) and (x + 1) % 16 for x in CAPTURES)):
        raise ValueError("treatment capture schedule is not ordered and cadence-valid")
    owner = load_owner(); baseline = baseline.resolve(strict=True)
    inv_path = baseline / "invocation.json"; invocation = read_object(inv_path)
    owner.validate_native_310s_invocation(invocation)
    if treatment_output.exists() or treatment_output.is_symlink():
        raise ValueError("treatment output path must be new")
    resolved_output = treatment_output.resolve(strict=False)
    if not treatment_output.is_absolute() or resolved_output == baseline or baseline in resolved_output.parents:
        raise ValueError("treatment output must be a fresh absolute path outside the immutable baseline run")
    if "--resting-drive-intervention" in invocation["argv"]:
        raise ValueError("baseline invocation already has an intervention")
    baseline_steps = [int(x) for x in invocation["environment"].get("NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS", "").split(",") if x]
    if (baseline_steps != sorted(set(baseline_steps)) or not baseline_steps or
            baseline_steps[0] != 0 or baseline_steps[-1] != STEPS or
            any(x not in (0, STEPS) and (x + 1) % 16 for x in baseline_steps)):
        raise ValueError("baseline capture schedule is malformed or misses a cadence-valid terminal")
    meta_path = baseline / "run-metadata.json"
    try:
        metadata = read_object(meta_path)
    except ValueError as exc:
        raise ValueError("baseline is still open or has no completed owner run-metadata; no declaration written") from exc
    if (metadata.get("exit_code") != 0 or metadata.get("source_files_changed_during_run") != [] or
            metadata.get("argv") != invocation.get("argv") or metadata.get("asset_sha256") != invocation.get("asset_sha256") or
            metadata.get("environment") != invocation.get("environment") or
            metadata.get("loaded_metal_runtime", {}).get("verified") is not True):
        raise ValueError("closed baseline metadata does not verify its invocation/runtime")
    log_path = baseline / "native.log"
    if log_path.is_symlink() or not log_path.is_file():
        raise ValueError("closed baseline native log is missing")
    native = owner.native_scene_summary(log_path.read_text(encoding="utf-8", errors="replace"))
    if native["accepted_steps"] != STEPS or abs(native["simulated_s"] - STEPS * DT) > 1e-4:
        raise ValueError("baseline did not complete exactly 155000 accepted 2 ms roots")
    bindings = invocation.get("asset_sha256")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("baseline invocation has no immutable asset hash map")
    changed = [name for name, digest in bindings.items() if not Path(name).is_file() or sha(Path(name)) != digest]
    if changed:
        raise ValueError("baseline asset drift prevents declaration: " + ", ".join(changed))
    trace = baseline / "resting-coupled.csv"
    if trace.is_symlink() or not trace.is_file():
        raise ValueError("closed baseline physiology trace is missing")
    export_lines = [line for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    if line.startswith("accepted_geometry_export=")]
    baseline_capture_details = []
    for step in baseline_steps:
        pack = baseline / "accepted-geometry" / f"step-{step}.mrvpack"
        receipt_path = baseline / "accepted-geometry" / f"step-{step}.receipt.json"
        if pack.is_symlink() or receipt_path.is_symlink() or not pack.is_file() or not receipt_path.is_file():
            raise ValueError(f"baseline capture is incomplete at accepted step {step}")
        pack_sha, receipt_sha = sha(pack), sha(receipt_path)
        matching = [line for line in export_lines if line.startswith(f"accepted_geometry_export={pack} ")]
        if len(matching) != 1:
            raise ValueError(f"baseline log does not uniquely bind capture {step}")
        fields = dict(x.split("=",1) for x in matching[0].split()[1:] if "=" in x)
        receipt = read_object(receipt_path)
        if (fields.get("pack_sha256") != pack_sha or fields.get("receipt_sha256") != receipt_sha or
                receipt.get("accepted_step") != step or receipt.get("pack_file_sha256") != pack_sha or
                receipt.get("accepted_pack_path") != str(pack)):
            raise ValueError(f"baseline accepted capture receipt/hash mismatch at step {step}")
        baseline_capture_details.append({"accepted_step": step, "pack_path": str(pack), "pack_sha256": pack_sha,
                                         "receipt_path": str(receipt_path), "receipt_sha256": receipt_sha})
    args = argparse.Namespace(steps=STEPS, dt=DT, arm="treatment", start_s=START,
                              end_s=END, scale=SCALE, window_s=WINDOW)
    argv = owner.native_scene_command(invocation, treatment_output, args)
    env = dict(invocation["environment"])
    env["NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"] = ",".join(map(str, CAPTURES))
    failure_key = "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"
    if failure_key in env:
        env[failure_key] = str(treatment_output / "common-field-failure.json")
    allowed_env = {"NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS", failure_key}
    changed_env = {k for k in set(env) | set(invocation["environment"]) if env.get(k) != invocation["environment"].get(k)}
    if not changed_env.issubset(allowed_env):
        raise ValueError("treatment environment changes a non-capture/output setting")
    argv_base = list(argv); movie_i = argv_base.index("--resting-movie")
    argv_base[movie_i + 1] = invocation["argv"][invocation["argv"].index("--resting-movie") + 1]
    argv_base[4] = invocation["argv"][4]
    intervention_i = argv_base.index("--resting-drive-intervention")
    del argv_base[intervention_i:intervention_i + 4]
    if argv_base != invocation["argv"]:
        raise ValueError("treatment argv has an unapproved physical/configuration difference")
    pack_sizes = [(baseline / "accepted-geometry" / f"step-{x}.mrvpack").stat().st_size for x in baseline_steps]
    max_pack = max(pack_sizes)
    return {
        "schema": "numi.human.direct-matched-treatment-declaration.v1",
        "status": "prepared_not_launched",
        "interpretation": "The already completed direct baseline is the exact control and will not be rerun. This is a retrospective matched comparison, not prospectively registered paired allocation.",
        "owner": {"path": str(OWNER), "sha256": OWNER_SHA},
        "baseline": {"run_path": str(baseline), "invocation_path": str(inv_path), "invocation_sha256": sha(inv_path),
                     "run_metadata_path": str(meta_path), "run_metadata_sha256": sha(meta_path),
                     "native_log_sha256": sha(log_path), "trace_sha256": sha(trace),
                     "accepted_steps": STEPS, "dt_s": DT, "capture_steps": baseline_steps,
                     "accepted_captures": baseline_capture_details,
                     "device": native["device"], "world_fingerprint": native["world_fingerprint"],
                     "body_source_fingerprint": native["body_source_fingerprint"],
                     "program_fingerprint": native["coupled_program_fingerprint"],
                     "loaded_runtime": metadata["loaded_metal_runtime"]},
        "treatment": {"run_path": str(treatment_output), "argv": argv, "environment": env,
                      "intervention": {"start_s": START, "end_s": END, "scale": SCALE,
                                        "semantics": "delivered diaphragm/intercostal excitation scaled on [60,100); Brain chemoreflex remains active"},
                      "analysis_windows_s": {"pre": [START-WINDOW, START], "dose": [END-WINDOW, END],
                                             "recovery": [STEPS*DT-WINDOW, STEPS*DT]},
                      "capture_steps": CAPTURES,
                      "capture_rationale": "Initial, last cadence-valid root inside intervention, early recovery at 110.014 s, and terminal. Full accepted physiology and 64 ms surface traces remain enabled.",
                      "program_fingerprint": "record actual value from completed treatment native log",
                      "immutable_assets": bindings},
        "output_budget": {"baseline_pack_size_bytes": pack_sizes,
                          "treatment_accepted_pack_count": len(CAPTURES),
                          "estimated_treatment_accepted_pack_bytes": len(CAPTURES)*max_pack,
                          "estimated_root_resting_pack_bytes": max_pack,
                          "estimate_note": "Additional traces/movie are small relative to packs; recheck free disk before launch."}}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, default=BASELINE_DEFAULT)
    p.add_argument("--treatment-output", type=Path, default=TREATMENT_DEFAULT)
    p.add_argument("--declaration", type=Path, required=True)
    a = p.parse_args()
    if a.declaration.exists() or a.declaration.is_symlink():
        raise ValueError("refusing to overwrite a declaration")
    value = build_template(a.baseline, a.treatment_output)
    a.declaration.parent.mkdir(parents=True, exist_ok=True)
    with a.declaration.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
    print(a.declaration)

if __name__ == "__main__":
    try: main()
    except (OSError, ValueError, KeyError, IndexError) as exc: raise SystemExit(f"treatment declaration refused: {exc}")
