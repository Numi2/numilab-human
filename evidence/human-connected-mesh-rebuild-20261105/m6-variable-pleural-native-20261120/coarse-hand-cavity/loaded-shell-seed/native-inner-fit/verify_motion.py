"""Verify all captured native shell cells and retain the limiting solver roots."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re
import sys

import numpy as np

from build_trial import COARSE, digest

sys.path.insert(0, str(COARSE))
from audit_bed_motion_shell import terminal_positions
from verify_trial import event, points, table


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run = args.trial
    execution = json.loads((run / "execution.json").read_text())
    assert execution["returncode"] == 0
    nodes = table(run / "shell-mass.csv.nodes.csv")
    ref = points(nodes, "rest")
    current = points(nodes, "current").astype(np.float32)
    stdout = run / "stdout.log"
    final = terminal_positions(stdout, current)
    rows = table(run / "shell-mass.csv.tets.csv")
    tets = np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in rows])
    start_tets = ref[tets]
    end_tets = final[tets]
    start_det = np.linalg.det(start_tets[:, 1:] - start_tets[:, :1])
    end_det = np.linalg.det(end_tets[:, 1:] - end_tets[:, :1])
    j = end_det / start_det
    assert np.all(start_det > 0.) and np.all(j > 0.)
    accepted = []
    with stdout.open() as stream:
        for line in stream:
            if line.startswith("human_standing_progress=accepted "):
                accepted.append(int(re.search(r"\bstep=(\d+)", line)[1]))

    norms = []
    for line in (run / "stderr.log").read_text().splitlines():
        if line.startswith("matter_solver_domain_norms="):
            norms.append(json.loads(line.split("=", 1)[1]))
    assert accepted == list(range(1, len(accepted)+1))
    assert sorted({int(row["root"]) for row in norms}) == list(range(len(accepted)))
    mechanical_overruns = [row for row in norms if row["nonvascular_norm"] > row["mechanical_threshold"]]
    vascular_overruns = [row for row in norms if row["vascular_norm"] > row["vascular_threshold"]]
    integrated = event(stdout, "resting_integrated_body=completed")
    terminal = event(stdout, "resting_fascia_fem_terminal_audit=passed")
    bed = event(stdout, "resting_fem_skin_bed_terminal_history")
    coupled = table(run / "native/resting-coupled.csv")
    pressure = [float(r["pleural_pa"]) for r in coupled]
    time = [float(r["time_s"]) for r in coupled]
    assert np.isclose(float(integrated["simulated_s"]), time[-1], rtol=1.e-10, atol=0.)
    duration = time[-1]
    assert len(accepted) == int(terminal["accepted_steps"]) and integrated["root_assistance"] == "false"
    report = {
        "input_sha256": {name: digest(run / name) for name in
                         ("stdout.log", "stderr.log", "launch.json", "native/resting-coupled.csv",
                          "shell-mass.csv.nodes.csv", "shell-mass.csv.tets.csv")},
        "steps": len(accepted), "simulated_seconds": duration, "recorded_pressure_range_pa": [min(pressure), max(pressure)],
        "pressure_samples": len(pressure), "positive_terminal_cells": len(tets),
        "minimum_terminal_j": float(j.min()), "maximum_terminal_j": float(j.max()),
        "maximum_shell_node_motion_m": float(np.linalg.norm(final - current, axis=1).max()),
        "mechanical_norm_overruns": mechanical_overruns, "vascular_norm_overruns": vascular_overruns,
        "bed_rows": int(bed["loaded_rows"]), "bed_normal_reaction_n": float(bed["normal_impulse_sum_Ns"]) * len(accepted) / duration,
        "ligament_rows": int(terminal["registered_ligament_force_rows"]),
        "tissue_body_contact_force_sum_n": float(terminal["body_contact_normal_force_N"]),
        "root_assistance": False, "body_wall_seconds": float(integrated["wall_s"]),
        "body_realtime_factor": float(integrated["real_time_factor"]),
        "process_wall_seconds": execution["wall_seconds"], "process_realtime_factor": duration / execution["wall_seconds"],
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: val for key, val in report.items() if key != "input_sha256"}))


if __name__ == "__main__":
    main()
