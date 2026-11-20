"""Bind accepted time and the narrow geometry audit to the native trial."""
import csv
import json
import math
from pathlib import Path

from build_hand_trial import HERE, digest


def prepare_case(case):
    selected = {"matter_solver_domain_norms", "fem_backtracking_trials",
                "fem_newton_terminal_certificate", "matter_fgmres_active_work"}
    stdout = (case / "stdout.log").read_text().splitlines()
    stderr = case / "stderr.log"
    logged = {}
    for line in stderr.open():
        kind, _, payload = line.partition("=")
        if kind in selected:
            row = json.loads(payload)
            logged.setdefault(kind, []).append(row)
    with (case / "native/resting-coupled.csv").open() as f:
        trace = list(csv.DictReader(f))
    complete = json.loads((case / "execution.json").read_text())
    return {
        "launch_sha256": digest(case / "launch.json"),
        "stdout_sha256": digest(case / "stdout.log"),
        "stderr_sha256": digest(stderr),
        "accepted_physical_steps": len(trace),
        "time_s": [float(row["time_s"]) for row in trace],
        "pleural_pa": [float(row["pleural_pa"]) for row in trace],
        "support_impulse_ns": [float(row["normal_impulse_ns"]) for row in trace],
        "root_assistance_n": [float(row["root_assistance_n"]) for row in trace],
        "body_mass_residual_kg": [float(row["body_mass_residual_kg"]) for row in trace],
        "rest_preparation": next(line for line in stdout if line.startswith(
            "resting_initialization=dynamic_release ")),
        "tissue_mass_reference": next(line for line in stdout if line.startswith(
            "whole_body_tissue_mass_reference=")),
        "runtime_binding": next(line for line in stdout if line.startswith(
            "resting_tissue_shared_runtime=bound ")),
        "registration_acceptance": next(line for line in stdout if line.startswith(
            "resting_skin_fat_whole_body_preparation=accepted ")),
        "execution": complete,
        "process_realtime_ratio": float(trace[-1]["time_s"]) / complete["wall_seconds"],
        "domain_norms": logged["matter_solver_domain_norms"],
        "rejecting_step_backtracking": [
            row for row in logged.get("fem_backtracking_trials", []) if row["root"] == len(trace)
        ],
        "rejecting_step_skin_certificate": next(
            row["objects"][15] for row in logged["fem_newton_terminal_certificate"]
            if row["root"] == len(trace)),
        "rejecting_step_active_work": next(
            row for row in logged["matter_fgmres_active_work"] if row["root"] == len(trace)),
    }


def main():
    primary = HERE / "hand-clearance-native-replay"
    repeat = HERE / "hand-clearance-ninth-root"
    original = HERE / "connected-interface-replay-solid-witness/hand-solid.csv"
    changed = primary / "hand-solid.csv"
    with original.open() as f:
        old = {tuple(row[key] for key in ("node","body","bone")):row for row in csv.DictReader(f)}
    with changed.open() as f:
        new = {tuple(row[key] for key in ("node","body","bone")):row for row in csv.DictReader(f)}
    assert set(old) == set(new)
    assert len(new) == 16
    clear = {key for key,row in old.items() if row["inside_rays"] == "3"}
    assert len(clear) == 4
    assert all(row["outside_rays"] == "3" and row["ambiguous"] == "0" for row in new.values())
    record = {
        "source_clearance_sha256": digest(original),
        "tested_clearance_sha256": digest(changed),
        "preparation_witness_minimum_clearance_m": min(
            float(row["closest_distance_m"]) for row in new.values()),
        "resolved_witnesses": [dict(source=old[key], candidate=new[key]) for key in sorted(clear)],
        "original": prepare_case(primary),
        "repeat": prepare_case(repeat),
    }
    a, b = record["original"], record["repeat"]
    assert a["time_s"] == b["time_s"]
    assert a["accepted_physical_steps"] == 8
    for replay in (a,b):
        assert replay["execution"]["exit"] == 1
        assert replay["domain_norms"][-1]["root"] == 8
        assert all(v == 0. for v in replay["support_impulse_ns"] + replay["root_assistance_n"] +
                   replay["body_mass_residual_kg"])
        assert math.isfinite(replay["domain_norms"][-1]["nonvascular_norm"])
    print(json.dumps(record, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
