"""Verify a paired native replay and restore the tested certificate from GitHub."""
import json
from pathlib import Path
import sys
from archive_cavity_trial import archive_files
from compare_surface_release import outcome
from material_context import sha


def read(p):
    return json.loads(p.read_text())


def single_record(p, key):
    rows = [json.loads(line.split("=", 1)[1]) for line in p.read_text().splitlines()
            if line.startswith(key + "=")]
    if len(rows) != 1:
        raise ValueError(f"Expected one {key} in {p}")
    return rows[0]


def verify(case):
    trial = case/"one"
    parent = case.parent/"normal-row"
    old = parent/"one"
    control, run = outcome(old), outcome(trial)
    for k in ("pleural_coupling", "installed_material_area", "iterations", "residual",
              "accepted_steps", "accepted_force_history", "common_coordinates_restored", "dynamic_library"):
        if control[k] != run[k]:
            raise ValueError("The matched physical calculation differs: " + k)
    launch, base = [read(p/"launch.json") for p in (trial, old)]
    if launch["argv"][0] != base["argv"][0]:
        raise ValueError("Different native runtime")
    device = read(case/"device.json")
    shader = Path(launch["environment"]["NUMI_MATTER_METALLIB_OVERRIDE"])
    if device["metallib"] != sha(shader) or sha(shader)!=launch["pins"][str(shader)]:
        raise ValueError("Metal library identity mismatch")
    for k in ("registered_contact_raw_row", "registered_plane_limit_witness"):
        if single_record(old/"stderr.log", k) != single_record(trial/"stderr.log", k):
            raise ValueError("Limiting physical state changed: " + k)
    cert_before, cert_after = [read(p/"failure.json") for p in (old,trial)]
    if cert_before["matter_diagnostics"] != cert_after["matter_diagnostics"]:
        raise ValueError("The final physical residual or rejection code changed")
    summary = dict(device=device["device"], step_s=read(parent/"result.json")["step_s"],
                   passed_floor_cases=device["cases"], old_floor_m=device["witness_old_floor_m"],
                   corrected_floor_m=device["witness_new_floor_m"], limiting_gap_m=device["observed_gap_m"],
                   accepted_steps=run["accepted_steps"], fresh_physical_residual=run["residual"],
                   mechanical_failure_code=cert_after["matter_diagnostics"][3],
                   physically_matched_iterations=len(run["iterations"]),reproduced_momentum_row=True,
                   pleural_coupling=run["pleural_coupling"],wall_s=run["elapsed_s"],
                   first_reporting_objects=[control["matter_object"],run["matter_object"]],
                   scope="The registered bed certificate now uses the step limiter's fixed-plane arithmetic "
                         "scale. The independent sweep and the common momentum acceptance gate still run.")
    (case/"comparison.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    files={k:case/k for k in (
        "build_and_check.py", "check_floor.metal", "certificate_floor.metalinc", "verify_and_archive.py",
        "check_floor.mm","check-floor","test.metallib","device.json","build/NumiMatter.metallib",
        "comparison.json","support_distribution.py","support-distribution.json","parent/launch.json","one/stdout.log","one/stderr.log","one/execution.json",
        "one/launch.json","one/failure.json")}
    for p in sorted((trial/"native-traces").glob("*.csv")):
        files[str(p.relative_to(case))] = p
    files["parent-row-recovery.json"]=parent/"recovery.json"
    archive_files(files, "registered-plane-clearance-"+sha(shader)[:16]+".tar.zst",
                  case/"recovery.json","M6 certificate correction, independent signed-distance checks, "
                  "and matched whole-scene first-step failure. Parent archive supplies the unchanged native runtime.")


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
