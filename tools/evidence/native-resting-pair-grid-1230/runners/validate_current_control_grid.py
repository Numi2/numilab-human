#!/usr/bin/env python3
"""Validate the current closed control trace's accepted time grid only."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json
from pathlib import Path

ROOT = Path("/Users/n/numi-human-free-apex-two-family-1178")
BUNDLE = ROOT / "tools/evidence/native-resting-pair-grid-1230"
ANALYZER = BUNDLE / "source/analyze_direct_pair_revision-002.py"
ANALYZER_SHA = "3856e0a0c2de8a95d5f9ebf62354d827946b4a375fc348b777b015ac6d423523"
ORIGINAL_ANALYZER = BUNDLE / "source/analyze_direct_pair.py"
ORIGINAL_ANALYZER_SHA = "4b273d7c84ab3aba0ebc1a55ab912c4e71c02cfb650493ab7813c7e08a7b31d1"
BASE = Path("/Users/n/numi-human-retained-delivery-20261009/skin-resting-multipose-clearance-1218/native-baseline-310s-preparation/native-run")
PINS = {
    "invocation.json": "34f757ccda069de88df8c025ee024e681c9d6da5a2ac8599dfce0a14eced577e",
    "run-metadata.json": "401ff87d7b4c52a3bb6c0b5a1d119939eccef7d37a15d6ef4a101a4ffcde29b3",
    "native.log": "3d0b9ae0c2265f763be895d18623d622c8e563e3a4d4f223792a34c0522e9a38",
    "resting-coupled.csv": "e06988dbcf8a187bab35b854d80d1a0e5111a4b7f90a286af9d9614fa5d952d5",
}
STEPS = 155000
DT = 0.002

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if out.exists() or out.is_symlink():
        raise ValueError(f"refusing existing output: {out}")
    if sha(ANALYZER) != ANALYZER_SHA or sha(ORIGINAL_ANALYZER) != ORIGINAL_ANALYZER_SHA:
        raise ValueError("analyzer source pin mismatch")
    mod = load_module(ANALYZER, "direct_pair_revision_002_grid_check")
    owner = mod.load_module(mod.OWNER, mod.OWNER_SHA, "resting_intervention_study_1162_grid_check")
    before = {name: sha(BASE / name) for name in PINS}
    if before != PINS:
        raise ValueError(f"closed baseline file pin mismatch: {before}")
    invocation = mod.read_json(BASE / "invocation.json")
    metadata = mod.read_json(BASE / "run-metadata.json")
    if (metadata.get("exit_code") != 0 or metadata.get("source_files_changed_during_run") != [] or
            metadata.get("argv") != invocation.get("argv") or
            metadata.get("asset_sha256") != invocation.get("asset_sha256") or
            metadata.get("environment") != invocation.get("environment") or
            metadata.get("loaded_metal_runtime", {}).get("verified") is not True):
        raise ValueError("closed baseline invocation/runtime receipt failed")
    owner.validate_native_310s_invocation(invocation)
    native = owner.native_scene_summary((BASE / "native.log").read_text(encoding="utf-8", errors="replace"))
    if native["accepted_steps"] != STEPS or abs(native["simulated_s"] - STEPS * DT) > 1.0e-4:
        raise ValueError("baseline did not reach the declared physical duration")
    rows = owner.read_trace(BASE / "resting-coupled.csv")
    grid = mod.verify_matched_accepted_grid(rows, rows, dt=DT, steps=STEPS)
    grid["comparison"] = "Single current control trace validated for the full expected accepted-step cadence and step-to-time correspondence; it was not compared with a treatment trace."
    after = {name: sha(BASE / name) for name in PINS}
    if after != before:
        raise ValueError("baseline evidence changed during read-only validation")
    report = {
        "schema": "numi.human.direct-control-grid-validation.v1",
        "status": "pass",
        "scope": "Current closed 1218 control accepted-time grid validation only.",
        "paired_grid_comparison_performed": False,
        "current_treatment_analyzed": False,
        "current_paired_outcome_claimed": False,
        "analyzer": {"path": str(ANALYZER), "sha256": ANALYZER_SHA,
                     "original_path": str(ORIGINAL_ANALYZER), "original_sha256": ORIGINAL_ANALYZER_SHA,
                     "owner_path": str(mod.OWNER), "owner_sha256": mod.OWNER_SHA},
        "baseline": {"run_path": str(BASE), "files": before,
                     "immutable_asset_sha256": invocation.get("asset_sha256"),
                     "accepted_steps": native["accepted_steps"], "simulated_s": native["simulated_s"],
                     "declared_dt_s": DT, "effective_float32_dt_s": 0.0020000000949949026},
        "control_grid": grid,
        "input_bytes_unchanged": before == after,
        "limitation": "The 95 s anatomy audit for this current scene reports 18 skin-target crossings. No direct treatment declaration/run is authorized or represented here. The retained 1170 pair is separate prior registered evidence with older scene/assets; it is not a current-scene paired outcome.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(report, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print(out)

if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise SystemExit(f"current control grid validation refused: {exc}")
