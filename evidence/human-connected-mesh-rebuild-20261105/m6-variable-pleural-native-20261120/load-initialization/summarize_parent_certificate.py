"""Extract compact third-root certificates from the native solver trace."""
from argparse import ArgumentParser
import csv
import json
from pathlib import Path
import struct

from run_recruitment import digest


def measure(case):
    result = dict(launch_sha256=digest(case / "launch.json"),
                  stderr_sha256=digest(case / "stderr.log"),
                  execution=json.loads((case / "execution.json").read_text()))
    with (case / "native/resting-coupled.csv").open() as trace:
        rows = list(csv.DictReader(trace))
        result["accepted_roots"] = len(rows)
        result["accepted_seconds"] = float(rows[-1]["time_s"]) if rows else 0.0
    phases = []
    with (case / "stderr.log").open() as lines:
        for line in lines:
            kind, _, raw = line.partition("=")
            if line.startswith("human_joint_failure "):
                result["failure_receipt"] = line.strip()
            if kind not in ("matter_solver_domain_norms", "fem_backtracking_trials",
                            "vascular_iterate", "fem_deformable_limit_trace"):
                continue
            row = json.loads(raw)
            if row["root"] != 2:
                continue
            if kind == "matter_solver_domain_norms":
                result["third_root_residuals"] = row
            elif kind == "fem_backtracking_trials" and row["iteration"] == 0:
                result["third_root_skin_backtracking"] = row
            elif kind == "fem_deformable_limit_trace" and row["iteration"] == 0:
                result["third_root_initial_contact_limit"] = dict(
                    contact_count=row["active_counts"],
                    line_search_alpha=row["line_search_alpha"],
                    witnesses=[{k: c[k] for k in ("contact_slot", "identity", "source", "nodes",
                                                  "diagnostic_reason", "lane_bound_after")}
                               for c in row["contacts"]])
            elif kind == "vascular_iterate" and row["stage"] in (
                    "after_fem_backtracking", "before_vascular_line_search", "line_search"):
                arenas = row["arenas"]
                alpha, = struct.unpack("<f", bytes.fromhex(arenas["alpha"]))
                phase = {"iteration": row["iteration"], "stage": row["stage"], "allocated_alpha": alpha}
                if row["stage"] == "after_fem_backtracking":
                    skin = struct.unpack_from("<f", bytes.fromhex(arenas["fem_line_search"]), 15 * 16)
                    phase["skin_alpha"] = skin[0]
                phases.append(phase)
    if phases:
        result["third_root_alpha_trace"] = phases
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, nargs="+")
    args = parser.parse_args()
    result = {case.name: measure(case) for case in args.cases}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
