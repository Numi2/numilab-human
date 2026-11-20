"""Screen a changed solver metric without weakening the native certificates."""
from argparse import ArgumentParser
from collections import defaultdict
import csv
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "fgmres-endpoint-window-020/root-1-physical-trace"
THRESHOLD = Decimal(".01")

def load(run):
    launch = json.loads((run / "launch.json").read_text().replace(str(run), "<run>"))
    eof = (run / "execution.json")
    execution = json.loads(eof.read_text()) if eof.exists() else {"exit": None}
    err = (run / "stderr.log").read_text()
    out = (run / "stdout.log").read_text()
    with (run / "native/resting-coupled.csv").open(newline="") as stream:
        trace = list(csv.DictReader(stream))
    works = {}
    defects = defaultdict(dict)
    for line in err.splitlines():
        kind, _, event = line.partition("=")
        if kind == "matter_fgmres_active_work":
            row = json.loads(event)
            works[row["root"]] = {name: row[name] for name in
                ("used_columns_sum", "cycles_with_work", "budget_exhausted_solves")}
        elif kind == "fem_true_residual_cycle":
            row = json.loads(event)
            if row["root"] == 1:
                defects[row["newton_iteration"]][row["restart_cycle"]] = row["environments"][0]
    stage = sum(int(x) for x in re.findall(
        r"(?m)^matter_gpu_stage=matter_newton_linear step=1 elapsed_ns=(\d+)$", err)) / 1e6
    events = [line for line in out.splitlines() if line.startswith((
        "resting_integrated_body=", "resting_FEM_terminal_audit=",
        "resting_fem_skin_bed_accepted_root_reaction=",
        "resting_fem_skin_bed_accepted_root_reaction "))]
    return launch, execution, trace, works, defects, stage, events, err

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    reference = (args.reference or REFERENCE).resolve()
    ref_plan, ref_run, ref_rows, ref_work, ref_defects, ref_stage, ref_events, _ = load(reference)
    run = args.candidate.resolve()
    plan, execution, rows, work, defects, stage, events, err = load(run)
    assert plan["baseline_launch_sha256"] == ref_plan["baseline_launch_sha256"]
    assert "matter_fgmres_row_scale_power=0.5 physical_certificate=unscaled" in err
    old, new = ref_plan["environment"], plan["environment"]
    differences = {k for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    if args.reference:
        assert differences == {"NM_FGMRES_ROW_SCALE_SQRT"}, differences
        assert plan["argv"] == ref_plan["argv"]
        assert plan["pins"] == ref_plan["pins"]
    else:
        assert differences == {
            "DYLD_LIBRARY_PATH", "NUMI_MATTER_METALLIB_OVERRIDE",
            "NM_FGMRES_ROW_TRANSFORM_WORK_DOMAIN", "NM_FGMRES_ROW_SCALE_SQRT",
        }, differences
        assert plan["argv"][1:] == ref_plan["argv"][1:]
    assert execution["exit"] == ref_run["exit"] == 0
    expected_steps = int(plan["argv"][plan["argv"].index("--muscle-step-count") + 1])
    assert len(rows) == len(ref_rows) == expected_steps
    assert sorted(work) == sorted(ref_work) == list(range(expected_steps))
    assert ref_stage > 0 and stage > 0
    mismatches = []
    changed_fields = []
    exact_fields = 0
    for previous, current in zip(ref_rows, rows):
        assert previous.keys() == current.keys()
        for name in current:
            if previous[name] == current[name]:
                exact_fields += 1
                continue
            x = previous[name]
            y = current[name]
            try:
                x, y = Decimal(x), Decimal(y)
            except InvalidOperation:
                mismatches.append(dict(step=current.get("step"), field=name,
                                       different_nonnumeric_value=[x,y]))
                continue
            if not x.is_finite() or not y.is_finite():
                mismatches.append(dict(step=current.get("step"), field=name,
                                       nonfinite=[str(x), str(y)]))
                continue
            relative = abs((y-x)/x) if x else None
            change = dict(step=current.get("step"), field=name, reference=str(x), candidate=str(y),
                          absolute_delta=str(abs(y-x)),
                          relative_delta=None if relative is None else float(relative))
            changed_fields.append(change)
            if relative is None or relative > THRESHOLD:
                mismatches.append(change)
    print(json.dumps({
        "scope": "Native short-prefix screen. No regional force or strain qualification.",
        "same_binary": args.reference is not None,
        "accepted_steps": [len(ref_rows), len(rows)],
        "reference_prefix_result": ref_run,
        "candidate_prefix_result": execution,
        "exact_accepted_fields": exact_fields,
        "all_numeric_trace_changes": changed_fields,
        "changes_outside_1_percent_comparison": mismatches,
        "reference_work": ref_work,
        "candidate_work": work,
        "reference_root1_true_residual": ref_defects,
        "candidate_root1_true_residual": defects,
        "profiled_linear_gpu_ms": [ref_stage, stage],
        "reference_endpoint": ref_events,
        "candidate_endpoint": events,
        "candidate_promoted": False,
    },indent=2,allow_nan=False))

if __name__ == "__main__":
    main()
