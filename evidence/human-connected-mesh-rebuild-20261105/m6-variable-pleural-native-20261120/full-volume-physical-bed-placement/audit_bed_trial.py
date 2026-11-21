"""Compare the fixed-bed reaction against a matched flat-bed prefix."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def read_accepted_trace(native):
    with (native / "resting-coupled.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    steps = [int(row["step"]) for row in rows]
    times = [float(row["time_s"]) for row in rows]
    if steps != list(range(1, len(rows) + 1)):
        raise ValueError("Accepted trace has missing or repeated steps")
    if any(not math.isfinite(t) or t <= prev for prev, t in zip([0.0, *times], times)):
        raise ValueError("Accepted time is nonfinite or did not advance")
    return rows


def rec(log, tag):
    lines = [line for line in log.splitlines() if line.startswith(tag + " ")]
    if len(lines) > 1:
        raise ValueError(f"Ambiguous native receipt: {tag}")
    return lines[0] if lines else None


def scalar(line, key):
    name, separator, value = line.partition(" " + key + "=")
    if not separator:
        raise ValueError(f"Required native value absent: {key}")
    result = float(value.split()[0])
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite measurement: {key}")
    return result


def mass_digest(native):
    path, = native.glob("registered-volume-fractional-mass-*.json")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trial(folder):
    native = folder / "native" if (folder / "native").is_dir() else folder
    receipt = json.loads((folder / "execution.json").read_text())
    return read_accepted_trace(native), mass_digest(native), receipt, (folder / "stdout.log").read_text()


def trace_range(rows, field):
    array = tuple(float(row[field]) for row in rows)
    if not all(math.isfinite(value) for value in array):
        raise ValueError(f"Nonfinite trace samples: {field}")
    return [min(array), max(array)] if array else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("plane_receipt", type=Path)
    args = parser.parse_args()
    plane = json.loads(args.plane_receipt.read_text())
    control, original_mass, _, control_log = load_trial(args.control)
    test, trial_mass, execution, log = load_trial(args.candidate)
    if original_mass != trial_mass:
        raise ValueError("Bed-only trial changed registered mass")
    tag = "resting_initial_support_alignment=applied"
    control_pose, trial_pose = rec(control_log, tag), rec(log, tag)
    if not control_pose or control_pose != trial_pose:
        raise ValueError("Bed-only trial changed initial body pose")
    flat = control[:len(test)]
    if len(flat) != len(test) or [row["time_s"] for row in flat] != [row["time_s"] for row in test]:
        raise ValueError("Candidate times do not match the control prefix")
    terminal_reaction = rec(log, "resting_fem_skin_bed_accepted_root_reaction")
    if terminal_reaction:
        samples = scalar(terminal_reaction, "valid_contact_sample_count")
        force = scalar(terminal_reaction, "mean_normal_force_N")
        impulse = scalar(terminal_reaction, "normal_impulse_Ns")
        dt = scalar(terminal_reaction, "root_dt_s")
        if min(samples, force, impulse, dt) < 0.0 or dt == 0.0:
            raise ValueError("Native bed receipt has an invalid unilateral reaction")
        if not math.isclose(force * dt, impulse, rel_tol=1e-4, abs_tol=1e-11):
            raise ValueError("Native force and impulse disagree")
        if "articulated_support_rows=0" not in terminal_reaction:
            raise ValueError("Additional support owner in tilted-bed trial")
        reaction = {"force_N":force, "impulse_Ns":impulse, "samples":samples,
                    "fraction_of_full_weight":force/plane["weight_N"]}
    else:
        reaction = None

    def trace_metrics(rows):
        return {name:trace_range(rows, name) for name in
                ("airflow_ml_s", "lung_volume_ml", "pleural_pa", "right_atrium_ml", "body_mass_residual_kg")}

    time_s = float(test[-1]["time_s"]) if test else 0.0
    data = {
        "accepted_roots": len(test),
        "accepted_simulated_time_s": time_s,
        "inclusive_wall_time_s": execution["wall_seconds"],
        "inclusive_realtime_factor": time_s/execution["wall_seconds"],
        "native_exit": execution["exit"],
        "native_stop": execution["stop"],
        "mass_receipt_sha256": original_mass,
        "executed_body_placement": trial_pose,
        "physical_bed_initial_geometry": rec(log, "resting_fem_skin_bed_compiled=validated"),
        "native_ligament_admission": rec((args.candidate/"stderr.log").read_text(), "registered_ligament_source_geometry=admitted"),
        "native_ligament_terminal_audit": rec(log, "resting_fascia_fem_terminal_audit=passed"),
        "initial_equilibrium": rec(log, "resting_initialization=dynamic_release"),
        "last_accepted_root_bed_reaction": reaction,
        "candidate_trace_range": trace_metrics(test),
        "matched_control_trace_range": trace_metrics(flat),
        "continuous_motion_geometry_verified": False,
    }
    print(json.dumps(data, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
