"""Measure skeletal support and bound the capacity of contacts present in an accepted dense Human release."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_momentum(run):
    trace = run / "native-traces/resting-com-momentum-diagnostic.csv"
    with trace.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError("Two consecutive accepted momentum samples are required")
    previous, last = rows[-2:]
    step = int(last["accepted_step"])
    dt = float(last["time_s"]) - float(previous["time_s"])
    if dt <= 0 or last["delta_com_valid"] != "1" or step != int(previous["accepted_step"]) + 1:
        raise ValueError("The terminal momentum samples do not cover one physical step")

    def vector(template):
        return np.array([float(last[template.format(axis)]) for axis in "xyz"])

    normal = vector("support_normal_{}")
    mass = float(last["total_body_mass_kg"])
    force = vector("delta_com_p{}_kg_m_s") / dt
    gravity = vector("gravity_impulse_{}_ns") / dt
    center = vector("com_{}_m")
    if mass <= 0 or not np.isfinite([*force, *gravity, *center, mass]).all():
        raise ValueError("Invalid accepted momentum")
    if abs(np.linalg.norm(normal) - 1) > 1e-5 or gravity @ normal >= 0:
        raise ValueError("Invalid support plane or gravity orientation")
    return step, dt, mass, force, gravity, center, normal, digest(trace)


def skin_to_bone(run, step):
    dump = run / f"contact-{step - 1}.rows"
    counts = np.fromfile(run / "contact-counts.u32", "<u4")
    if not 1 <= step <= len(counts) or counts[step - 1] > 8192:
        raise ValueError("The terminal contact record exceeds the native observation window")
    rows = np.fromfile(dump, "<f4").reshape(-1, 7, 4)
    if len(rows) != counts[step - 1]:
        raise ValueError("Incomplete native contact observations")
    pattern = r"registered_material_span object=15 element_offset=\d+ element_count=\d+ state_offset=(\d+) state_count=(\d+)"
    matches = set(re.findall(pattern, (run / "stderr.log").read_text()))
    if len(matches) != 1:
        raise ValueError("Missing or ambiguous registered skin/fat node range")
    offset, count = map(int, matches.pop())
    ids = rows[:, 0].copy().view("<u4")
    use = (ids[:, 1] >= offset) & (ids[:, 1] < offset + count)
    ids, data = ids[use], rows[use].astype(float)
    if len(np.unique(ids[:, 0])) != len(data) or not np.isfinite(data[:, 1:]).all():
        raise ValueError("Duplicate or nonfinite native observations")
    closure, area, pressure = data[:, 1, 3], data[:, 2, 3], data[:, 6, 3]
    if np.any((closure <= 0) | (area <= 0) | (pressure <= 0)):
        raise ValueError("Invalid compressive contact")
    force = data[:, 4, :3]
    per_pascal = force / pressure[:, None]
    if not np.allclose(np.linalg.norm(per_pascal, axis=1), area, rtol=1e-5, atol=1e-15):
        raise ValueError("Pressure times area differs from observed cortical force")
    if not np.allclose(force, -data[:, 3, :3], rtol=1e-6, atol=1e-7):
        raise ValueError("Material and skeletal contact forces do not cancel")
    return data[:, 2, :3], force, per_pascal, closure, pressure, digest(dump)


def peak_required(operator, target):
    count = operator.shape[1]
    if count == 0:
        return 0.0 if np.linalg.norm(target) == 0 else None
    # The final optimization variable bounds every nonnegative contact coefficient.
    equal = sparse.csr_matrix(np.column_stack((operator, np.zeros(len(target)))))
    limits = sparse.hstack((sparse.eye(count), -np.ones((count, 1))), format="csr")
    result = linprog(np.r_[np.zeros(count), 1.0], A_ub=limits, b_ub=np.zeros(count),
                     A_eq=equal, b_eq=target, bounds=(0, None), method="highs")
    if result.status == 2:
        return None
    if not result.success:
        raise ValueError(f"Contact bound solver failed ({result.message})")
    coeff, peak = result.x[:-1], float(result.x[-1])
    if np.max(abs(operator @ coeff - target)) > 2e-5 or np.max(coeff) > peak + 1e-7:
        raise ValueError("Contact bound did not reproduce the target")
    return peak


def assess(run):
    step, dt, mass, total, gravity, center, normal, trace_hash = read_momentum(run)
    points, f, per_pa, closure, pressure, dump_hash = skin_to_bone(run, step)
    measured = f.sum(axis=0)
    other = total - gravity - measured
    target = -gravity - other
    arm = points - center
    bounds = {}
    for units, scale in (("peak_pressure_kPa", 1000.0), ("peak_closure_mm", 0.001 * pressure / closure)):
        per_unit = per_pa * np.broadcast_to(scale, (len(f),))[:, None]
        wrench = np.vstack((per_unit.T, np.cross(arm, per_unit).T))
        bounds[units] = {
            "bed_normal_only": peak_required((per_unit @ normal)[None], np.array([target @ normal])),
            "three_force_components": peak_required(per_unit.T, target),
            "with_current_interface_moment": peak_required(wrench, np.r_[target, np.cross(arm, f).sum(axis=0)]),
        }
    bed = run / "physical-posterior-bed-history.csv"
    with bed.open() as stream:
        impulses = [float(row["terminal_microstep_normal_impulse_Ns"]) for row in csv.DictReader(stream)]
    if not np.isfinite(impulses).all() or any(v < 0 for v in impulses):
        raise ValueError("Invalid terminal bed impulse")
    return {
        "launch_sha256": digest(run / "launch.json"), "momentum_sha256": trace_hash,
        "skin_bone_rows_sha256": dump_hash, "bed_impulses_sha256": digest(bed),
        "accepted_step": step, "physical_step_s": dt, "skeletal_mass_kg": mass,
        "skeletal_normal_acceleration_m_s2": float(total @ normal / mass),
        "skeletal_gravity_normal_N": float(gravity @ normal),
        "physical_skin_bed_terminal_impulse_Ns": sum(impulses),
        "skin_bone_rows": len(f), "skin_bone_normal_N": float(measured @ normal),
        "other_owners_inferred_normal_N": float(other @ normal),
        "required_skin_bone_normal_N": float(target @ normal),
        "measured_peak_pressure_kPa": float(pressure.max(initial=0.0) / 1000),
        "measured_peak_closure_mm": float(closure.max(initial=0.0) * 1000), "bounds": bounds,
        "scope": "Frozen measured contact areas, normals and slopes. Independent nonnegative pressures allow arbitrary redistribution. The torque case holds the observed interface moment fixed. A new contact set or different physical owners require a new measurement.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(assess(args.run), indent=2, allow_nan=False) + "\n")
