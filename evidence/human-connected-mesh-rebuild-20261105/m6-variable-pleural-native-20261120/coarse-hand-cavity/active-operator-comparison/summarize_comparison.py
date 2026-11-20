"""Summarize captured startup benchmarks and pre-existing regional gates."""
import json
from pathlib import Path
import re

BODY = re.compile(r"^resting_integrated_body=completed simulated_s=(\S+) wall_s=(\S+) real_time_factor=(\S+)")
WORK = "matter_fgmres_active_work="


def summary(run, comparison):
    files = Path(run)
    launch = json.loads((files / "launch.json").read_text())
    execution = json.loads((files / "execution.json").read_text())
    assert execution["exit"] == 0, execution
    with (files / "stdout.log").open() as log:
        matches = [match for line in log if (match := BODY.match(line))]
    [match] = matches
    seconds, body_wall, body_factor = map(float, match.groups())
    assert abs(seconds / body_wall - body_factor) <= 1.e-15
    with (files / "stderr.log").open() as log:
        work = [json.loads(line[len(WORK):]) for line in log if line.startswith(WORK)]
    records = {
        "binary_sha256": launch["native_binary_sha256"],
        "simulated_s": seconds, "body_wall_s": body_wall, "body_realtime_factor": body_factor,
        "process_wall_s": execution["wall_seconds"],
        "process_realtime_factor": seconds / execution["wall_seconds"],
        "roots": len(work),
        "max_active_newton_starts": max(row["active_newton_begins"] for row in work),
        **{key: sum(row[key] for row in work) for key in (
            "encoded_cycles", "cycles_with_work", "used_columns_sum", "budget_exhausted_solves")},
        "newton_budget": int(launch["environment"]["NUMI_HUMAN_PASSIVE_FEM_NEWTON_ITERATIONS"]),
        "variable_external_pleural_enabled": launch["environment"]["NM_VASCULAR_STEP_EXTERNAL_PRESSURE"] == "1",
    }
    if comparison is None:
        return records
    field = json.loads(Path(f"{comparison}-fields.json").read_text())
    trace = json.loads(Path(f"{comparison}-traces.json").read_text())
    records["comparison"] = {
        "report_prefix": comparison,
        "native_segment_speed_ratio": trace["native_segment_speed_ratio"],
        "peak_accepted_position_difference_m": field["max_accepted_position_difference_m"],
        "exact_nodal_mass": field["nodal_masses_match"],
        "exact_reference_positions": field["reference_positions_match"],
        "force_and_strain_differences": {key: len(field[key]) for key in (
            "force_regions_over_1pct", "strain_metrics_over_1pct", "strain_metrics_over_5pct")},
        "terminal_positive_skin_fat": trace["candidate"]["terminal_checks"]["terminal_nonpositive_skin_fat_cells"] == 0,
        "terminal_min_skin_fat_J": trace["candidate"]["terminal_checks"]["terminal_skin_fat_min_j"],
        "pressure_range_Pa": [trace["coupled_trace"]["pleural_pa"][key]
                              for key in ("candidate_min","candidate_max")],
    }
    return records


def main():
    runs = ("control", "assembled", "borrowed", "graph", "six", "six-long", "control-long")
    comparison = {"control": None, "control-long": "baseline-repeat",
                  "six-long": "matched-six-long"}
    results = {name: summary(name, comparison.get(name, name)) for name in runs}
    results["retained_control"] = summary("../bed-normal-remaining-banks-prefix", None)
    results["six_vs_retained_control"] = summary("six-long", "six-long")["comparison"]
    assert len({row["binary_sha256"] for row in results.values() if "binary_sha256" in row}) == 1
    with Path("summary.json").open("x") as report:
        json.dump(results, report, indent=2, allow_nan=False)
        report.write("\n")


if __name__ == "__main__":
    main()
