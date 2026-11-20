"""Correlate the accepted line search and projected/physical Krylov residuals."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def events(path, names):
    with path.open() as stream:
        for line in stream:
            key, sep, value = line.partition("=")
            if sep and key in names:
                yield key, json.loads(value)

def insert(table, key, value):
    if key in table and table[key] != value:
        raise ValueError(f"Conflicting duplicate event: {key}")
    table[key] = value

def main():
    first, physical, control = map(Path, sys.argv[1:])
    files = sorted(path.name for path in control.joinpath("native").glob("*.csv"))
    traces = {file: sha(control / "native" / file) for file in files}
    for run in (first, physical):
        assert all(sha(run / "native" / file) == digest
                   for file, digest in traces.items()), run
    columns, stages, certified = {}, {}, {}
    observed_work = {}
    physical_skips = []
    keys = ("fem_krylov_column", "fem_newton_iterate", "fem_true_residual_cycle",
            "matter_fgmres_active_work", "fem_residual_armijo_trace_skipped")
    for key, row in events(physical / "stderr.log", keys):
        if key == "fem_krylov_column" and row["column_executed"]:
            ix = (row["newton"], row["restart"], row["column"])
            assert row["root"] == 1 and row["microtick"] == 0
            insert(columns, ix, row)
        elif key == "fem_newton_iterate":
            assert row["root"] == 1 and row["microtick"] == 0
            insert(stages, (row["iteration"], row["stage"]), row)
        elif key == "fem_true_residual_cycle":
            assert row["root"] == 1 and len(row["environments"]) == 1
            insert(certified, (row["newton_iteration"], row["restart_cycle"]),
                   row["environments"][0])
        elif key == "matter_fgmres_active_work":
            insert(observed_work, row["root"], row)
        elif key == "fem_residual_armijo_trace_skipped":
            physical_skips.append(row["reason"])
    assert 0 < len(columns) == observed_work[1]["used_columns_sum"]
    groups = defaultdict(list)
    for (i, restart, col), row in sorted(columns.items()):
        groups[i, restart].append(row)
    assert len(groups) == observed_work[1]["cycles_with_work"]
    newton = []
    last_active = max(i for i, _ in groups)
    for i in range(last_active + 1):
        cycles = []
        for (iteration, restart), row in sorted(certified.items()):
            if i != iteration:
                continue
            trace = groups[iteration, restart]
            used_columns = int(row["diagnostics"][3])
            assert len(trace) == used_columns
            cycles.append({
                "restart": restart,
                "columns": used_columns,
                "true_l2": row["true_l2"],
                "true_max_abs": row["true_max_abs"],
                "cycle_certified": row["diagnostics"][2] > .5,
                **({
                    "arnoldi_trace": [c["arnoldi_residual_estimate"] for c in trace],
                    "scaled_reference": trace[0]["row_scaled_reference_norm"],
                } if trace else {}),
            })
        row = stages[i, "final_line_search"]
        next_row = stages[i + 1, "final_line_search"]
        newton.append({
            "iteration": i,
            "physical_rhs_norm": row["fgmres"][0]["diagnostics"][1],
            "newton_residual_after_correction": next_row["fgmres"][0]["nonlinear"][1],
            "alpha_by_stage": {stage: v["alpha"][0]
                               for (k, stage), v in stages.items() if k == i},
            "restarts": cycles,
        })
    print(json.dumps({
        "exact_native_trace_sha256": traces,
        "measured_root": 1,
        "active_columns": len(columns),
        "restarters_with_columns": len([v for v in groups.values() if v]),
        "terminal_latched": stages[last_active + 1, "final_line_search"]["fgmres"][0]["nonlinear"][3],
        "armijo_observer_skip_reasons": sorted(set(physical_skips)),
        "newton": newton,
    }, indent=2))

if __name__ == "__main__":
    main()
