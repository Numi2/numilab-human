"""Pair a single native FEM momentum row with the same nonlinear trial's collision witness."""
import json
import math
import sys
from pathlib import Path

from compare_surface_release import outcome
from material_context import sha


def read(path):
    return json.loads(path.read_text())


def record(path, key):
    matches = [json.loads(line.split("=", 1)[1]) for line in path.read_text().splitlines()
               if line.startswith(key + "=")]
    if len(matches) != 1:
        raise ValueError(f"Expected one {key}, found {len(matches)}")
    return matches[0]


def verify(root):
    case = root / "normal-row"
    trial = case / "one"
    baseline = root / "limit-witness/one"
    data, control = outcome(trial), outcome(baseline)
    for k in ("pleural_coupling", "installed_material_area", "accepted_steps", "residual",
              "iterations", "accepted_force_history", "common_coordinates_restored"):
        if data[k] != control[k]:
            raise ValueError("The diagnostic changed the integrated result: " + k)
    if not data["pleural_coupling"] or not data["installed_material_area"]:
        raise ValueError("Missing native pressure or registered bed")
    # Each continuum certifies an environment-wide residual. Compare every stored
    # field apart from the identity of the first reporter, and preserve both IDs.
    failures = [read(p / "failure.json") for p in (baseline, trial)]
    reporters = []
    for failure in failures:
        reporter = failure.pop("matter_object_index")
        reporters.append(reporter)
        for key in ("matter_status", "stand_status"):
            status = failure["first_failure_latch"][key]
            if status[1] != reporter:
                raise ValueError("Inconsistent first-reporter receipt")
            status[1] = None
    if failures[0] != failures[1]:
        raise ValueError("Other full-scene rejection or rollback fields changed")
    launch = read(trial / "launch.json")
    exe = Path(launch["argv"][0])
    lib = Path(data["dynamic_library"])
    if (exe.resolve() != (case / "owner/numi-human-native").resolve()
        or lib.resolve() != (case / "build/libmetalrobo.dylib").resolve()
        or any(sha(p) != launch["pins"][str(p)] for p in (lib, exe))):
        raise ValueError("Native artifact mismatch")
    r = record(trial / "stderr.log", "registered_contact_raw_row")
    w = record(trial / "stderr.log", "registered_plane_limit_witness")
    if w != record(baseline / "stderr.log", "registered_plane_limit_witness"):
        raise ValueError("Contact witness changed")
    dt = float(launch["argv"][launch["argv"].index("--muscle-step-seconds") + 1])
    if (r["node"] != w["node"] or r["root"] != w["root"] or r["newton"] != w["iteration"]
        or any(not math.isfinite(x) for key in ("velocity_m_s", "point_m", "raw_rhs_Ns")
               for x in r[key]) or r["mass_kg"] <= 0 or r["point_m"] != w["point_xyz_m"]):
        raise ValueError("Mismatched coupled and collision rows")
    report = dict(step_s=dt, node=r, limiter=w, already_applied_normal_N=w["normal_impulse_Ns"]/dt,
                  raw_net_signed_normal_rhs_over_dt_N=r["raw_rhs_Ns"][2]/dt,
                  fresh_physical_residual=data["residual"], accepted_steps=data["accepted_steps"],
                  variable_pleural_pressure=data["pleural_coupling"], matched_iteration_history=True,
                  all_other_failure_fields_match=True, first_reporting_objects=reporters,
                  binary=sha(exe), dylib=sha(lib), metallib=data["library"], launch_sha256=sha(trial/"launch.json"),
                  scope="The signed momentum imbalance includes inertia and the existing contact reaction. "
                        "It is measured at the failed Newton candidate, not an accepted contact multiplier.")
    (case / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())
