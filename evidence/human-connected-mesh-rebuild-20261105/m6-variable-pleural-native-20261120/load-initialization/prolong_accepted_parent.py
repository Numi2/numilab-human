"""Use the native endpoint observer to extract exactly the pinned parent skin/fat rows."""
from argparse import ArgumentParser
import csv
import json
import math
from pathlib import Path

import numpy as np

from prolong_skin_trial import run
from run_same_source_parent import SOURCE
from run_recruitment import digest

AXES = "xyz"


def fields(line):
    return dict(item.split("=", 1) for item in line.rstrip().split(",")[1:])


def extract_rows(native_log, coarse):
    result = {}
    object_name = None
    endpoint = None
    footer = None
    for line in native_log.open():
        if line.startswith("resting_FEM_endpoint_capture=begin "):
            assert endpoint is None
            endpoint = dict(item.split("=", 1) for item in line.split() if "=" in item)
        if line.startswith("resting_integrated_body=completed "):
            footer = dict(item.split("=", 1) for item in line.split() if "=" in item)
        if not line.startswith("resting_FEM_node,"):
            continue
        node = fields(line)
        index = int(node["global"])
        if index not in coarse:
            continue
        reference = coarse[index]
        if object_name is None:
            object_name = node["object"]
        assert node["object"] == object_name
        assert index not in result
        assert int(node["local"]) == int(reference["parent_node_index"])
        assert all(float(np.float32(node[key])) == float(reference[value]) for key, value in (
            ("initial_mass_kg", "compiled_lumped_mass_kg"),
            ("mass_kg", "compiled_lumped_mass_kg"),
            *((f"initial_{axis}", f"current_{axis}_m") for axis in AXES),
        )), node
        mode = int(node["reference_mode"])
        assert mode in (0, 1) and mode == int(node["initial_reference_mode"])
        current = dict(reference)
        for axis in AXES:
            position_key = f"current_{axis}_m"
            if mode == 1:
                # Apply the accepted displacement increment to the exact
                # pinned source geometry without subtracting projected worlds.
                delta = math.fsum(float(node[k]) * sign for k, sign in (
                    (f"reference_u_hi_{axis}", 1.0), (f"initial_reference_u_hi_{axis}", -1.0),
                    (f"reference_u_lo_{axis}", 1.0), (f"initial_reference_u_lo_{axis}", -1.0),
                ))
                value = float(reference[position_key]) + delta
                assert abs(float(np.float32(value)) - float(node[f"accepted_{axis}"])) <= 1.e-7
            else:
                value = float(np.float32(node[f"accepted_{axis}"]))
            assert math.isfinite(value)
            current[position_key] = format(value, ".17g")
        result[index] = current
    assert endpoint and footer, "missing successful native endpoint transaction"
    assert coarse.keys() == result.keys(), "the endpoint schema disagrees with the compiled parent"
    return result, endpoint, footer, object_name


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    args = parser.parse_args()
    execution = json.loads((args.case / "execution.json").read_text())
    assert execution["exit"] == 0 and execution["stop"] == "exit"
    launch = json.loads((args.case / "launch.json").read_text())
    source = SOURCE / "compiled-parent-cell-mass.csv.nodes.csv"
    assert digest(source) == launch["parent_table_sha256"]
    with source.open(newline="") as inp:
        reader = csv.DictReader(inp)
        fieldnames = reader.fieldnames
        reference = {int(row["compiled_global_node"]): row for row in reader}
    native_log = args.case / "stdout.log"
    rows, endpoint, footer, object_name = extract_rows(native_log, reference)
    terminal_steps = int(endpoint["steps"])
    assert int(launch["argv"][launch["argv"].index("--muscle-step-count") + 1]) == terminal_steps
    assert footer["physiology_body_clock"] == "matched" and footer["root_assistance"] == "false"
    table = args.case / "accepted-parent.csv"
    with table.open("x", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[index] for index in reference)
    record = {
        "accepted_steps": terminal_steps,
        "simulated_seconds": float(footer["simulated_s"]),
        "object_name": object_name,
        "node_count": len(rows),
        "native_log_sha256": digest(native_log),
        "node_table_sha256": digest(table),
        "launch_sha256": digest(args.case / "launch.json"),
    }
    with (args.case / "parent-endpoint.json").open("x") as out:
        json.dump(record, out, indent=2)
        out.write("\n")
    run(table, record["node_table_sha256"], args.case / "prolong-accepted.json",
        publish_nodes=False)


if __name__ == "__main__":
    main()
