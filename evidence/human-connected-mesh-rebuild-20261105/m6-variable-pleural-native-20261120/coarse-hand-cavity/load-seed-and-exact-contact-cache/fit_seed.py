"""Design the position of a fixed-bed IPC preload. No trajectory is computed."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, root
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "operator-resident-030/parallel-operators-128"
FOOTPRINT = HERE.parent / "terminal-support-footprint.json"
WIDTH = 1e-5

def normal_force(clearance, mass, timestep):
    if clearance >= WIDTH:
        return 0.0
    distance = max(clearance, WIDTH * 1e-4)
    s = distance * distance
    shat = WIDTH * WIDTH
    difference = s - shat
    stiffness = 10 / max(timestep * timestep * shat / mass, 1.e-20)
    return 2 * distance * stiffness * (
        2 * difference * np.log(s / shat) + difference**2 / s)

def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    scene = PARENT / "launch.json"
    launch = json.loads(scene.read_text())
    argv = launch["argv"]
    dt = float(argv[argv.index("--muscle-step-seconds") + 1])
    root_slot = argv.index("--root-pose") + 1
    origin = np.asarray(argv[root_slot:root_slot+3], dtype=float)
    quaternion = Rotation.from_quat(np.asarray(argv[root_slot+3:root_slot+7], dtype=float))
    node_path = PARENT / "initial-skin-fem.csv"
    with node_path.open() as stream:
        outer = [r for r in csv.DictReader(stream) if int(r["layer"]) == 0]
    points = np.array([[float(row[f"{axis}_m"]) for axis in "xyz"] for row in outer])
    masses = np.array([float(row["mass_kg"]) for row in outer])
    contact = np.flatnonzero(points[:, 2] - points[:, 2].min() < 1e-7)
    assert len(contact) == 3
    physical = json.loads(FOOTPRINT.read_text())
    center = np.asarray(physical["body_com_m"])
    weight = float(physical["weight_N"])
    assert np.isfinite(points).all() and (masses[contact] > 0).all()
    local = points[contact] - origin
    def geometry(variables):
        dx, dy, dz = variables
        rotation = Rotation.from_rotvec([dx, dy, 0])
        vertices = rotation.apply(local) + origin + [0, 0, dz]
        com = rotation.apply(center - origin) + origin + [0, 0, dz]
        bary = np.linalg.solve(np.vstack([vertices[:, :2].T, np.ones(3)]),
                               np.r_[com[:2], 1])
        assert min(bary) > 0
        return rotation, vertices, bary
    def constraint(variables):
        _, verts, bary = geometry(variables)
        goals = np.array([brentq(lambda distance: normal_force(distance, mass, dt) - fraction * weight,
                                  1.e-8, WIDTH, xtol=1.e-16)
                          for mass, fraction in zip(masses[contact], bary, strict=True)])
        return verts[:, 2] - goals
    result = root(constraint, [0, 0, 3e-6], tol=1e-9)
    assert result.success and np.max(np.abs(constraint(result.x))) < 1e-12, result
    rotation, verts, weights = geometry(result.x)
    moved = rotation.apply(points-origin) + origin + [0, 0, result.x[2]]
    assert np.array_equal(np.flatnonzero(moved[:,2] < WIDTH), contact)
    assert moved[:,2].min() > 1.e-8 and np.linalg.norm(rotation.as_rotvec()) < 1.e-3
    table = [{"outer_row": int(outer[j]["row"]), "native_node": int(outer[j]["global_node"]),
              "mass_kg": float(masses[j]), "target_gap_m": float(verts[i, 2]),
              "gravity_support_n": float(force * weight),
              "static_barrier_force_n": float(normal_force(verts[i, 2], masses[j], dt))}
             for i, (j, force) in enumerate(zip(contact, weights, strict=True))]
    prediction = {
        "source_launch_sha256": sha256(scene), "source_outer_sha256": sha256(node_path),
        "model": "Existing Matter IPC normal response at zero incoming velocity, under one shared rigid alignment",
        "gravity_n": weight, "timestep_s": dt, "barrier_width_m": WIDTH,
        "root_pose": [*origin.tolist(), *(rotation * quaternion).as_quat().tolist()],
        "small_angle_to_control_rad": float(np.linalg.norm(rotation.as_rotvec())),
        "support": table, "minimum_clearance_m": moved[:,2].min(),
        "next_inactive_clearance_m": float(np.sort(moved[:,2])[3]),
        "not_equilibrium": "The calculation does not balance tissue internal forces, articulation, muscle or pressure. Native stepped certificates and measured reactions control the verdict.",
    }
    (HERE / "prediction.json").write_text(json.dumps(prediction, indent=2) + "\n")
    print(json.dumps(prediction))

if __name__ == "__main__":
    main()
