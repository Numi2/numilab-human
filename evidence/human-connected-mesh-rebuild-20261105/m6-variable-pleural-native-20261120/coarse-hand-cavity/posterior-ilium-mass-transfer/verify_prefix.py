"""Verify the accepted native prefix independently of the outer process receipt."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re

import numpy as np

from audit import digest, rows
from audit_bed_motion_shell import terminal_positions
from extension import determinant
def event(path, label):
    with path.open() as log:
        line = next(line for line in log if line.startswith(label + " "))
    return dict(field.split("=", 1) for field in line.split()[1:] if "=" in field)


def points(entries, label):
    return np.array([[r[f"{label}_{a}_m"] for a in "xyz"] for r in entries], dtype=float)


def verify(run):
    stdout = run / "stdout.log"
    stderr = run / "stderr.log"
    with stdout.open() as log:
        accepted = [int(re.search(r"\bstep=(\d+)", line)[1])
                    for line in log if line.startswith("human_standing_progress=accepted ")]
    assert accepted == list(range(1, 129))
    measured = event(stdout, "resting_integrated_throughput")
    terminal = event(stdout, "resting_fascia_fem_terminal_audit=passed")
    bed = event(stdout, "resting_fem_skin_bed_terminal_history")
    assert int(measured["accepted_end_step"]) == int(terminal["accepted_steps"]) == len(accepted)
    with stderr.open() as log:
        norms = [json.loads(line.partition("=")[2]) for line in log
                 if line.startswith("matter_solver_domain_norms=")]
    assert {int(v["root"]) for v in norms} == set(range(len(accepted)))
    mechanical = [v for v in norms if v["nonvascular_norm"] > v["mechanical_threshold"]]
    vascular = [v for v in norms if v["vascular_norm"] > v["vascular_threshold"]]
    nodes = rows(run / "shell-mass.csv.nodes.csv")
    initial = points(nodes, "current").astype(np.float32)
    reference = points(nodes, "rest")
    end = terminal_positions(stdout, initial)
    tets = np.array([[int(r[f"parent_node{i}"]) for i in range(4)]
                     for r in rows(run / "shell-mass.csv.tets.csv")])
    before = determinant(reference, tets)
    ratios = determinant(end, tets) / before
    assert np.all(before > 0.) and np.all(ratios > 0.)
    trace = run / "native/resting-coupled.csv"
    coupled = rows(trace)
    time = np.array([float(v["time_s"]) for v in coupled])
    pressure = np.array([float(v["pleural_pa"]) for v in coupled])
    assert len(time) == len(accepted) and np.all(np.diff(time) > 0.)
    assert pressure.min() < pressure.max()
    assert np.isclose(time[-1]-time[0], float(measured["simulated_s"]), rtol=1e-9, atol=0.)
    return {
        "accepted_steps": len(accepted), "simulated_seconds": float(time[-1]),
        "timed_simulated_seconds": float(measured["simulated_s"]), "timed_wall_seconds": float(measured["wall_s"]),
        "headless_realtime_factor": float(measured["real_time_factor"]),
        "pleural_pressure_range_pa": [float(pressure.min()), float(pressure.max())],
        "mechanical_norm_overruns": mechanical, "vascular_norm_overruns": vascular,
        "positive_terminal_cells": len(ratios), "terminal_jacobian_ratio_minimum": float(ratios.min()),
        "terminal_jacobian_ratio_maximum": float(ratios.max()),
        "maximum_nodal_motion_m": float(np.linalg.norm(end-initial, axis=1).max()),
        "registered_ligament_rows": int(terminal["registered_ligament_force_rows"]),
        "ligament_force_wrench_residual_N": float(terminal["passive_ligament_force_residual_N"]),
        "ligament_moment_wrench_residual_Nm": float(terminal["passive_ligament_moment_residual_Nm"]),
        "terminal_bed_loaded_rows": int(bed["loaded_rows"]),
        "terminal_microstep_bed_normal_reaction_N": float(bed["normal_impulse_sum_Ns"])/(time[1]-time[0]),
        "scope": "Ordered native accepted prefix and complete endpoint node capture. The disk filled during the later redundant tetrahedron text dump and the outer process receipt could not be written. Headless timing excludes load and first accepted step.",
        "inputs_sha256": {p.name: digest(p) for p in (stdout, stderr, trace, run / "launch.json")},
    }


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = verify(args.run)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(report))
