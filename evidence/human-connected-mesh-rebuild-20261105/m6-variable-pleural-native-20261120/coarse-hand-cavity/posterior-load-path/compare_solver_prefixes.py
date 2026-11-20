"""Compare shared accepted roots and preserve a rejected native root."""
from argparse import ArgumentParser
import csv
import gzip
import json
from pathlib import Path
import re


def read_run(path):
    with (path / "native/resting-coupled.csv").open() as stream:
        trace = list(csv.DictReader(stream))
    work = {}
    rejection = None
    with (path / "stderr.log").open() as stream:
        for line in stream:
            if line.startswith("matter_fgmres_active_work="):
                row = json.loads(line.split("=", 1)[1])
                work[row["root"]] = row
            elif line.startswith("resting_common_failure "):
                rejection = dict(re.findall(r"(\w+)=(\S+)", line))
    final_body_seconds = None
    with gzip.open(path / "stdout.log.gz", "rt") as stream:
        for line in stream:
            if line.startswith("resting_integrated_body=completed "):
                final_body_seconds = float(re.search(r"\bwall_s=(\S+)", line)[1])
    return dict(
        execution=json.loads((path / "execution.json").read_text()),
        accepted_steps=len(trace),
        simulated_seconds=float(trace[-1]["time_s"]) if "time_s" in trace[-1] else None,
        final_body_seconds=final_body_seconds,
        pleural_range_pa=[aggregate(float(row["pleural_pa"]) for row in trace)
                          for aggregate in (min, max)],
        zero_root_assistance=all(float(row["root_assistance_n"]) == 0.0 for row in trace),
    ), trace, work, rejection


def compare(a, b):
    summary_a, trace_a, work_a, rejected_a = read_run(a)
    summary_b, trace_b, work_b, rejected_b = read_run(b)
    shared = min(len(trace_a), len(trace_b))
    first_difference = None
    max_difference = {}
    for step, (row_a, row_b) in enumerate(zip(trace_a, trace_b), 1):
        difference = {name: (reference, row_b[name]) for name, reference in row_a.items()
                      if reference != row_b[name]}
        if difference and first_difference is None:
            first_difference = {"accepted_step": step, "fields": difference}
        for name, (reference, candidate) in difference.items():
            max_difference[name] = max(max_difference.get(name, 0.0),
                                       abs(float(reference) - float(candidate)))
    rejections = []
    for rejected in (rejected_a, rejected_b):
        if rejected:
            root = int(rejected["segment_start"])
            rejections.append(dict(root=root, stand_code=int(rejected["stand_code"]),
                                   matter_code=int(rejected["matter_code"]),
                                   matter_diagnostics=json.loads(rejected["matter_diagnostics"]),
                                   reference=work_a.get(root), candidate=work_b.get(root)))
    return dict(reference=summary_a, candidate=summary_b,
                shared_accepted_steps=shared,
                first_accepted_difference=first_difference,
                max_accepted_difference=max_difference, rejections=rejections)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("reference", "candidate", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(compare(args.reference, args.candidate), stream, indent=2, allow_nan=False)
        stream.write("\n")
