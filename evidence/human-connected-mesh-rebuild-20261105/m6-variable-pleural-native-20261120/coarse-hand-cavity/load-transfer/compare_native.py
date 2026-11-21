"""Compare stable geometric witnesses and support in two matched native runs."""
from argparse import ArgumentParser
from collections import Counter
import csv
import json
from pathlib import Path
import re

from audit import digest


def witness_keys(endpoint):
    return {(kind, tuple(sorted(nodes)), body, bone, face)
            for kind, _, nodes, body, bone, face in endpoint["witnesses"]}


def execution(run):
    fields = {}
    with (run / "stdout.log").open() as stream:
        for line in stream:
            key = line.split(" ", 1)[0].strip()
            if key.startswith("resting_integrated_body="):
                fields["body"] = line.rstrip()
            elif key == "resting_fem_skin_bed_accepted_root_reaction":
                fields["bed"] = line.rstrip()
    values = {label: dict(re.findall(r"(\w+)=([^ ]+)", row))
              for label, row in fields.items()}
    assert values["body"]["resting_integrated_body"] == "completed"
    assert values["body"]["root_assistance"] == "false"
    assert values["bed"]["articulated_support_rows"] == "0"
    path = run / "native/resting-coupled.csv"
    with path.open() as stream:
        trace = list(csv.DictReader(stream))
    assert [int(row["step"]) for row in trace] == list(range(1, 129))
    pleural = [float(row["pleural_pa"]) for row in trace]
    assert min(pleural) < max(pleural)
    return {
        "last_accepted_bed_force_N": float(values["bed"]["mean_normal_force_N"]),
        "contact_samples": int(values["bed"]["valid_contact_sample_count"]),
        "simulated_s": float(values["body"]["simulated_s"]),
        "body_wall_s": float(values["body"]["wall_s"]),
        "body_real_time_factor": float(values["body"]["real_time_factor"]),
        "pleural_pa_range": [min(pleural), max(pleural)],
        "process": json.loads((run / "execution.json").read_text()),
        "launch_sha256": digest(run / "launch.json"), "trace_sha256": digest(path),
    }


def compare(control, candidate, output):
    a = json.loads((control / "bone-endpoints.json").read_text())
    b = json.loads((candidate / "bone-endpoints.json").read_text())
    assert a["shell_faces"] == b["shell_faces"]
    assert a["accepted_endpoint_step"] == b["accepted_endpoint_step"] == 128
    assert a["input_hashes"]["physical-bone-initial.bin"] == b["input_hashes"]["physical-bone-initial.bin"]
    report = {"control": execution(control), "candidate": execution(candidate), "crossings": {}}
    for condition in ("initial", "terminal"):
        ref, fit = witness_keys(a[condition]), witness_keys(b[condition])
        assert len(ref) == a[condition]["pairs"] and len(fit) == b[condition]["pairs"]
        new, gone = fit - ref, ref - fit
        report["crossings"][condition] = {
            "control_pairs": len(ref), "candidate_pairs": len(fit),
            "removed_by_role": dict(Counter(row[0] for row in sorted(gone))),
            "added_by_role": dict(Counter(row[0] for row in sorted(new))),
            "first_added_ids": sorted(new)[:10], "first_removed_ids": sorted(gone)[:10],
        }
    with output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    for k, v in report["crossings"].items():
        print(k, {m: n for m, n in v.items() if not m.startswith("first_")})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("control", "candidate", "output"):
        parser.add_argument(name, type=Path)
    compare(**vars(parser.parse_args()))
