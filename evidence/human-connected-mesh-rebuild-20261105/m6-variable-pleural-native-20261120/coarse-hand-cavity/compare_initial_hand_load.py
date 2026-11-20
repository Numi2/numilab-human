"""Summarize the one-root free-node force balance and solid-witness check."""
import csv
import json
from pathlib import Path

import numpy as np

from build_hand_trial import HERE, digest


def main():
    case = HERE / "connected-interface-replay-contact-rhs"
    target = 3927
    log = case / "stdout.log"
    fields = None
    scope = None
    for line in log.open():
        if line.startswith("resting_FEM_endpoint_capture=begin "):
            scope = dict(item.split("=", 1) for item in line.split()[1:])
        if line.startswith("resting_FEM_node,") and f",global={target}," in line:
            fields = dict(item.split("=",1) for item in line.rstrip().split(",")[1:])
            break
    assert fields and scope and int(scope["steps"]) == 1
    assert fields["constraint_tag"] == "0"
    axes = "xyz"
    mass = float(fields["mass_kg"])
    timestep = float(scope["runtime_dt_s"])
    velocity = np.array([float(fields["v" + axis]) for axis in axes])
    elastic = np.array([float(fields[f"free_internal_force_{axis}_N"]) for axis in axes])
    acceleration_load = mass * velocity / timestep
    inferred = acceleration_load - elastic - mass * np.array([0., 0., -9.81])
    source = HERE / "connected-interface-replay-solid-witness"
    candidates = source / "hand-solid.csv"
    with candidates.open() as f:
        rows = [r for r in csv.DictReader(f) if int(r["node"]) == target]
    asserts = [r for r in rows if int(r["inside_rays"]) == 3]
    assert rows
    forces = np.array([[float(row[f"force_{axis}_N"]) for axis in axes]
                       for row in rows]).sum(axis=0)
    report = {
        "native_endpoint_stdout_sha256": digest(log),
        "solid_witness_launch_sha256": digest(source / "launch.json"),
        "solid_witness_csv_sha256": digest(candidates),
        "outer_surface_node": target,
        "node_mass_kg": mass,
        "runtime_dt_s": timestep,
        "native_lumped_acceleration_force_N": acceleration_load.tolist(),
        "native_bulk_material_force_N": elastic.tolist(),
        "inferred_nonbulk_nongravity_force_N": inferred.tolist(),
        "three_ray_inside_banks": asserts,
        "solid_contact_model_force_N": forces.tolist(),
        "inferred_minus_solid_force_N": (inferred-forces).tolist(),
        "scope": "The load balance uses one backward-Euler step from zero velocity. The solid test reconstructs unilateral body-bank contact at the authored initial pose. It is a diagnostic calculation, not an additional stepping implementation.",
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
