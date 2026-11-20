"""Pin the compact control, candidate and geometry evidence for publication."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import sys

root=Path(sys.argv[1])
stage=Path(sys.argv[2])
source_paths = {
    "capture-evidence.py": "capture-evidence.py",
    "audit_shoulder_prefix.py": "audit_shoulder_prefix.py",
    "shoulder-prefix-audit.json": "shoulder-prefix-audit.json",
    "compare_solver_prefixes.py": "compare_solver_prefixes.py",
    "verify_dispatch_guard.sh": "verify_dispatch_guard.sh",
    "build_short_guard_dispatch.py": "build_short_guard_dispatch.py",
    "build_guard_dispatch_app.py": "build_guard_dispatch_app.py",
    "guarded-dispatch-shader/latch-required-short-trial.patch": "dispatch-latch-guard-shader.patch",
    "guarded-dispatch-shader/build.json": "dispatch-guard-shader-build.json",
    "guarded-dispatch-app-02/short-guard-app-runtime.patch": "dispatch-latch-guard-host.patch",
    "guarded-dispatch-app-02/commands.json": "dispatch-guard-host-build.json",
    "guarded-dispatch-app-02/binary.sha256": "dispatch-guard-binary.sha256",
    "guarded-buffer-control-comparison.json": "guarded-buffer-control-comparison.json",
    "guarded-buffer-staged-comparison.json": "guarded-buffer-staged-comparison.json",
    "dispatch-comparison.json": "dispatch-comparison.json",
    "dispatch-control-comparison.json": "dispatch-control-comparison.json",
    "dispatch-staged-comparison.json": "dispatch-staged-comparison.json",
    "dispatch-control-fields.json": "dispatch-control-fields.json",
    "dispatch-staged-fields.json": "dispatch-staged-fields.json",
    "older-bed-bank-recovery.json": "older-bed-bank-recovery.json",
    "audit_boundary_bones.py": "audit_boundary_bones.py",
    "screen_gluteal_columns.py": "screen_gluteal_columns.py",
    "replay_outlier.py": "replay_outlier.py",
    "build_latch.py": "build_latch.py",
    "build_short_guard.py": "build_short_guard.py",
    "build_guard_app.py": "build_guard_app.py",
    "guarded-short-build-04/latch-required-short-trial.patch": "latch-required-short-trial.patch",
    "guarded-short-build-04/build.json": "guarded-short-shader-build.json",
    "guarded-short-app/short-guard-app-runtime.patch": "short-guard-app-runtime.patch",
    "guarded-short-app/commands.json": "guarded-short-host-build.json",
    "guarded-short-app/binary.sha256": "guarded-short-binary.sha256",
    "compare_latch.py": "compare_latch.py",
    "boundary-and-load-summary.json": "boundary-and-load-summary.json",
    "outlier-stage-summary.json": "outlier-stage-summary.json",
    "witness-recovery.json": "witness-recovery.json",
    "older-pack-reclamation.tsv": "older-pack-reclamation.tsv",
    "latch-build-03/build.json": "shader-build.json",
    "latch-build-03/one-correction-domain-latch.patch": "one-correction-domain-latch.patch",
    "comparison.json": "comparison.json",
    "field-comparison.json": "field-comparison.json",
    "solver-outlier/launch.json": "instrumented-launch.json",
    "solver-outlier/execution.json": "instrumented-execution.json",
}
for f, dest in source_paths.items():
    shutil.copyfile(root / f, stage / dest)
for name in ("guarded-comparison.json", "guarded-field-comparison.json"):
    if (root / name).exists():
        shutil.copyfile(root / name, stage / name)
for mode in ("control", "one-correction", "guarded-prefix", "guarded-full-budget",
             "dispatch-full-budget", "dispatch-short-budget"):
    base=root/mode
    for name in ("launch.json","execution.json","native/resting-coupled.csv"):
        if (base / name).is_file():
            shutil.copyfile(base / name, stage / f"{mode}-{Path(name).name}")
    with (base/"stderr.log").open() as f, (stage/f"{mode}-residuals.log").open("x") as out:
        for ln in f:
            if ln.startswith(("matter_solver_domain_norms=", "resting_common_failure ",
                              "myosim_articulated_visual=failed")):
                out.write(ln)
    prefixes=("human_staged_solver_attempt", "human_staged_solver_rollback", "resting_integrated_body=", "resting_fem_skin_bed_terminal_history ",
              "resting_fascia_fem_terminal_audit=passed", "resting_FEM_terminal_audit=")
    with gzip.open(base/"stdout.log.gz","rt") as f, (stage/f"{mode}-terminal.log").open("x") as out:
        for ln in f:
            if ln.startswith(prefixes):
                out.write(ln)
manifest={}
for p in sorted(stage.iterdir()):
    if p.is_file():
        manifest[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
(stage/"SHA256SUMS").write_text("".join(f"{h}  {name}\n" for name, h in manifest.items()))
