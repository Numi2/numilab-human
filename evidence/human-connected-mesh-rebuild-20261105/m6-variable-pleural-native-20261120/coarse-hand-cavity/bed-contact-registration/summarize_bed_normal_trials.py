"""Certify retained accepted roots, norms, and support for the paired bed fit."""
import csv
import json
from pathlib import Path

from build_hand_trial import digest
from compare_bank_trials import endpoint_checks, witnesses


def summarize(directory, expected):
    stdout = (directory / "stdout.log").read_text().splitlines()
    stderr = (directory / "stderr.log").read_text().splitlines()
    trace_path = directory / "native/resting-coupled.csv"
    with trace_path.open(newline="") as stream:
        trace = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in trace] == list(range(1, expected + 1))
    norms = [json.loads(line.partition("=")[2]) for line in stderr
             if line.startswith("matter_solver_domain_norms=")]
    assert [record["root"] for record in norms] == list(range(expected))
    assert all(record["nonvascular_norm"] <= record["mechanical_threshold"]
               and record["vascular_norm"] <= record["vascular_threshold"]
               for record in norms)
    volume = endpoint_checks(directory)
    assert volume["terminal_nonpositive_skin_fat_cells"] == 0
    accepted = [line for line in stdout if line.startswith("human_standing_progress=accepted ")]
    assert [int(line.split()[1].partition("=")[2]) for line in accepted] == list(range(1, expected + 1))
    support, = [line for line in stdout if line.startswith("resting_fem_skin_bed_accepted_root_reaction ")]
    reaction = dict(item.split("=", 1) for item in support.split()[1:])
    pressure = [float(row["pleural_pa"]) for row in trace]
    launch_path = directory / "launch.json"
    launch = json.loads(launch_path.read_text())
    assert launch["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] == "1"
    assert all(float(row["root_assistance_n"]) == 0 and float(row["root_assistance_nm"]) == 0
               for row in trace)
    complete = [line for line in stdout if line.startswith("resting_integrated_body=completed ")]
    path = directory / "execution.json"
    receipt = json.loads(path.read_text()) if path.is_file() else None
    return {
        "trace_sha256": digest(trace_path),
        "launch_sha256": digest(launch_path),
        "binary_sha256": launch["native_binary_sha256"],
        "accepted_roots": expected,
        "simulated_s": float(trace[-1]["time_s"]),
        "elapsed_process": receipt,
        "terminal_status_line": complete[0] if complete else None,
        "peak_mechanical_residual": max(record["nonvascular_norm"] for record in norms),
        "peak_vascular_residual": max(record["vascular_norm"] for record in norms),
        "pleural_pa_range": [min(pressure), max(pressure)],
        "initial_point_bank_penetrations": len(witnesses(directory)),
        "support_contact_rows": int(reaction["valid_contact_sample_count"]),
        "terminal_step_mean_support_force_N": float(reaction["mean_normal_force_N"]),
        "shell_and_ligament_audit": volume,
    }


if __name__ == "__main__":
    runs = [("bed-normal-fixed-deep-native", 4), ("bed-normal-fixed-deep-prefix", 64)]
    result = {name: summarize(Path(name), count) for name, count in runs}
    with Path("bed-normal-fixed-deep-native-summary.json").open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    for name, record in result.items():
        audit = record["shell_and_ligament_audit"]
        print(name, record["accepted_roots"], record["simulated_s"], record["pleural_pa_range"],
              record["terminal_step_mean_support_force_N"], audit["terminal_skin_fat_min_j"],
              audit["active_krylov_columns"])
