"""Fit the supported opening posture to native bed contact and gravity observations."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from compile_connected_joint import digest
from fit_lower_support import read, records


def joint_motion_basis(payload, mapping, position, coordinates, velocity_count):
    """Differentiate the native equality polynomials for the independent coordinates."""
    count = struct.unpack_from("<I", payload, 20)[0]
    if len(payload) != 80 + 96 * count:
        raise ValueError("Invalid native joint-equality payload")
    velocity = {int(r["local_q_index"]): int(r["local_v_index"]) for r in mapping
                if r["local_q_index"] and r["local_v_index"]}
    basis = np.zeros((velocity_count, len(coordinates)))
    basis[[velocity[k] for k in coordinates], range(len(coordinates))] = 1
    followers = {}
    for i in range(count):
        y, v, z, parent, _, ref, _, a1, a2, a3, a4, _ = struct.unpack_from("<4I8f", payload, 80 + 96*i)
        if z == 0xffffffff:
            continue
        if v in followers or y in coordinates or not 0 <= z < len(position):
            raise ValueError("Invalid dependent coordinate in joint equalities")
        if velocity.get(y) != v or velocity.get(z) != parent:
            raise ValueError("Equality payload and runtime coordinate map disagree")
        delta = position[z] - ref
        followers[v] = (parent, a1 + 2*a2*delta + 3*a3*delta**2 + 4*a4*delta**3)

    active = set()
    resolved = set()

    def resolve(v):
        if v in followers and v not in resolved:
            if v in active:
                raise ValueError("Cycle in joint equality graph")
            active.add(v)
            parent, derivative = followers[v]
            basis[v] = derivative * resolve(parent)
            active.remove(v)
            resolved.add(v)
        return basis[v]

    for v in followers:
        resolve(v)
    return basis


def log_records(path, prefix):
    key = prefix + "="
    return [json.loads(line[len(key):]) for line in path.read_text().splitlines() if line.startswith(key)]


def fit(base, measured, prior, out, knee_flexion=False):
    launch_path = base / "launch.json"
    launch = read(launch_path)
    args = launch["argv"]
    report = read(prior / "construction.json")
    measurement = measured / "stdout.log"
    if report["jacobian_source"] != digest(measurement) or report["launch"] != digest(launch_path):
        raise ValueError("Mismatched native observations")
    bed = log_records(measurement, "initial_bed_posture_row")
    gravity, = log_records(measurement, "initial_gravity_total")
    force, = log_records(base / "stdout.log", "persistent_initial_force_reference")
    rows = sorted(force["rows"], key=lambda row: row["dof"])
    trace = base / "native-traces"
    sections = records(trace / "resting-local-bed-sections.csv")
    if [row["dof"] for row in rows] != list(range(len(rows))):
        raise ValueError("Incomplete native gravity force vector")
    if len(sections) != 32 or [r["row"] for r in bed] != list(range(len(sections))):
        raise ValueError("Expected the 32 registered contact sections")
    targets = np.array([r["gravity_target_n"] for r in rows])
    center = np.array(gravity["center_m"])
    points = np.array([r["xyz"] for r in bed])
    gaps = np.array([r["gap"] for r in bed])
    jacobian = np.array([r["Jnormal"] for r in bed])
    compliance = np.array([float(r["skin_compliance_m_per_n"]) + float(r["fat_compliance_m_per_n"])
                           for r in sections])
    thickness = np.array([float(r["skin_thickness_m"]) + float(r["fat_thickness_m"]) for r in sections])
    if not np.isfinite([compliance, thickness]).all() or np.any(compliance <= 0) or np.any(thickness <= 0):
        raise ValueError("Invalid registered material column")
    normal_row = records(trace / "resting-com-support-impulses.csv")[0]
    normal = np.array([float(normal_row["normal_" + k]) for k in "xyz"])
    first_state = records(trace / "resting-com-q-integration.csv")[0]
    position = np.fromstring(first_state["q_before_f32_semicolon"], sep=";")
    axes = [7, 101, 115, 106, 120] if knee_flexion else [7, 101, 115]
    pose_axes = axes[3:] if knee_flexion else axes
    payload = Path(args[args.index("--joint-equality-payload") + 1]).read_bytes()
    basis = joint_motion_basis(payload, records(trace / "resting-com-q-index-map.csv"),
                              position, axes, len(targets))
    contact_derivative = jacobian @ basis
    required_moments = targets @ basis

    def evaluate(variables):
        height, roll, pitch, *angles = variables
        if knee_flexion:
            angles = [0., 0., 0., *angles]
        world = Rotation.from_rotvec([roll, pitch, 0]).apply(points-center) + center + height*normal
        gap = gaps + (world-points) @ normal + contact_derivative @ angles
        loads = np.maximum(0, -gap) / compliance
        torque = (np.cross(world-center, normal) * loads[:, None]).sum(axis=0)
        residual = np.r_[loads.sum() - 9.81*gravity["mass_kg"]*normal[2], torque[:2],
                         contact_derivative.T @ loads - required_moments]
        return residual, loads

    if knee_flexion:
        # The source knees extend to zero. Preserve the original hip and spine
        # coordinates while fitting a small positive knee flexion on each side.
        lower = [-.025, -.07, -.025, 0., 0.]
        upper = [.025, .07, .025, .14, .14]
        scales = [.005, .035, .012, .05, .05]
        seeds = [[0., p, 0., k, k] for p in (-.02, 0., .02) for k in (.04, .065, .09)]
        weights = np.array([8., 8., 8., 1., 1., 1., 1., 1.])
    else:
        lower = [-.025, -.07, -.025, -.13, -.0017, -.0017]
        upper = [.025, .07, .025, .13, .0017, .0017]
        scales = [.005, .035, .012, .05, .001, .001]
        seeds = [[0., p, 0., s, 0., 0.] for p, s in ((.03, .03), (.02, .05), (.04, .06), (.025, .07))]
        weights = 1.

    def objective(s):
        return evaluate(s)[0] * weights

    trials = [least_squares(objective, seed, bounds=(lower, upper), x_scale=scales, max_nfev=600,
                            ftol=1e-13, xtol=1e-13, gtol=1e-10) for seed in seeds]
    solution = min(trials, key=lambda trial: np.linalg.norm(trial.fun)).x
    residual, loads = evaluate(solution)
    stock = np.max(loads * compliance / thickness)
    if (np.max(abs(residual[:3])) > 6 or stock > .96 or
            min(loads[[10,20,21]].sum(), loads[[11,30,31]].sum()) < 20):
        raise ValueError("Insufficient bilateral gravity support")
    root = read(base / "native-initial-bodies.json")["all_bodies"][0]["current"]
    rotation = Rotation.from_rotvec([solution[1], solution[2], 0.])
    xyz = center + rotation.apply(np.array(root["position_world_m"])-center) + solution[0]*normal
    quat = rotation * Rotation.from_quat(root["orientation_world_xyzw"])
    start = args.index("--root-pose") + 1
    args[start:start+7] = list(map(str, np.r_[xyz, quat.as_quat()]))
    for k, delta in zip(pose_axes, solution[3:], strict=True):
        args.extend(["--pose-q", str(k), str(position[k]+delta)])
    record = dict(observation=report["jacobian_source"], source=digest(launch_path), q=pose_axes,
                  measured_moment_q=axes, deltas_rad=solution[3:].tolist(), translation_m=solution[0],
                  world_rotation=solution[1:3].tolist(), rows=loads.tolist(),
                  force_and_moment_residual=residual.tolist(), native_gravity_moment_Nm=required_moments.tolist(),
                  maximum_stock_fraction=float(stock))
    out.mkdir()
    construction = out / "construction.json"
    construction.write_text(json.dumps(record, indent=2) + "\n")
    launch["pins"][str(construction.resolve())] = digest(construction)
    (out / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "measured", "prior", "out"):
        parser.add_argument(key, type=lambda value: Path(value).resolve())
    parser.add_argument("--knee-flexion", action="store_true")
    fit(**vars(parser.parse_args()))
