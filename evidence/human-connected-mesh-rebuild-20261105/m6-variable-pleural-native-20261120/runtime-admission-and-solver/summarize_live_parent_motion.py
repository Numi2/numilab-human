"""Record compact first-step evidence without promoting a rejected candidate."""
import csv
from collections import defaultdict
import json
from pathlib import Path
import statistics
import struct
import sys

def collect(lines, prefix):
    prefix += "="
    return [json.loads(line.removeprefix(prefix))
            for line in lines
            if line.startswith(prefix)]

for name in sys.argv[1:]:
    run = Path(name)
    stderr = (run / "stderr.log").read_text().splitlines()
    stdout = (run / "stdout.log").read_text().splitlines()
    execution = run / "execution.json"
    failure = run / "failure.json"
    trace = run / "coupled.csv"
    if not trace.is_file():
        trace = run / "native/resting-coupled.csv"
    accepted = []
    if trace.is_file():
        with trace.open(newline="") as stream:
            accepted = [row for row in csv.DictReader(stream) if float(row["time_s"]) > 0]
    times = [float(row["time_s"]) for row in accepted]
    if any(next_time <= time for time, next_time in zip(times, times[1:])):
        raise ValueError(f"{run}: the accepted trace is not strictly ordered")
    accepted_time = times[-1] if times else 0.0
    run_proofs = {
        "native_initialization": [
            line for line in stdout + stderr if line.startswith((
                "whole_body_skin_fat_registered_volume_install=", "registered_ligament_source_geometry=",
                "resting_vascular_external_pressure=", "resting_fem_skin_bed_compiled=",
                "resting_FEM_bed_release_initialization=", "human_contact_row_cache ",
                "human_compact_immutable_contacts ", "human_immutable_contact_storage=",
                "matter_parallel_constitutive_backtracking="))
        ],
        "solver_work": collect(stderr, "matter_fgmres_active_work"),
        "domain_norms": collect(stderr, "matter_solver_domain_norms"),
        "contact_work": collect(stderr, "matter_contact_work"),
        "deformable_limit_minima": collect(stderr, "fem_deformable_limit_minimum"),
        "accepted_time_s": accepted_time,
        "accepted_rows": len(accepted),
        "native_terminal": [
            line for line in stdout if line.startswith((
                "resting_integrated_body=", "resting_integrated_throughput ",
                "resting_fascia_fem_terminal_audit=", "resting_fem_skin_bed_terminal_history ",
                "resting_fem_skin_bed_accepted_root_reaction "))
        ],
    }
    if accepted:
        for field in ("pleural_pa", "body_mass_residual_kg",
                      "root_assistance_n", "root_assistance_nm"):
            if field in accepted[0]:
                values = [float(row[field]) for row in accepted]
                run_proofs.setdefault("accepted_sample_ranges", {})[field] = {
                    "minimum": min(values), "maximum": max(values),
                }
    if execution.is_file():
        run_proofs["execution"] = json.loads(execution.read_text())
        run_proofs["verified_end_to_end_simulated_per_wall_second"] = (
            accepted_time / run_proofs["execution"]["wall_seconds"])
    if failure.is_file():
        run_proofs["failure"] = json.loads(failure.read_text())
    line_search = []
    for item in collect(stderr, "vascular_iterate"):
        arenas = item["arenas"]
        alpha, = struct.unpack("<f", bytes.fromhex(arenas["alpha"]))
        object_bytes = bytes.fromhex(arenas["fem_line_search"])
        objects = list(struct.iter_unpack("<4f", object_bytes))
        limiting_index, limiting = min(enumerate(objects), key=lambda entry: entry[1][0])
        observation = {
            "iteration": item["iteration"], "stage": item["stage"],
            "common_alpha": alpha, "minimum_object_alpha": limiting[0],
        }
        if item["stage"] == "after_fem_backtracking" and limiting[0] < 1:
            observation.update(
                limiting_constitutive_object=limiting_index,
                limiting_constitutive_volume_ratio=limiting[1],
            )
        line_search.append(observation)
    if line_search:
        run_proofs["line_search"] = line_search
    backtracking_trials = collect(stderr, "fem_backtracking_trials")
    if backtracking_trials:
        run_proofs["constitutive_witness"] = []
        for entry in backtracking_trials:
            trials = entry["trials"]
            failure = next((trial for trial in reversed(trials) if trial["invalid_tiles"]), None)
            run_proofs["constitutive_witness"].append({
                **{key: entry[key] for key in ("root", "iteration", "object", "selected_alpha")},
                "closest_rejected_trial": failure,
                "smallest_tested_trial": trials[-1] if trials else None,
            })
    boundary = sorted((item for item in collect(stderr, "gpu_progress")
                       if item["phase"] == "matter_pre"), key=lambda item: item["sequence"])
    groups = defaultdict(list)
    for first, second in zip(boundary, boundary[1:]):
        if first["sequence"] + 1 == second["sequence"]:
            groups[first["boundary"]].append(second["wall_elapsed_s"] - first["wall_elapsed_s"])
    run_proofs["gpu_boundary_cpu_arrival_intervals_s"] = {
        stage: {"median": statistics.median(values), "minimum": min(values),
                "maximum": max(values), "count": len(values)}
        for stage, values in groups.items()
    }
    (run / "compact-motion-evidence.json").write_text(json.dumps(run_proofs, indent=2) + "\n")
    print(f"{run}: accepted_time_s={accepted_time} accepted_rows={len(accepted)}")
