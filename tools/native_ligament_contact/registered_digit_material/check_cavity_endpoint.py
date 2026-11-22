"""Verify the accepted frame and the installed material cells from the same full native transaction."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from run_cavity_motion import digest


def verify(trial, tools, implementation, predicate):
    receipt = json.loads((trial / "terminal.receipt.json").read_text())
    launch = json.loads((trial / "launch.json").read_text())
    state = json.loads((trial / "admission.json").read_text())
    if receipt["accepted_step"] != state["accepted_steps"] or launch["environment"][
        "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"
    ] != str(receipt["accepted_step"]):
        raise ValueError("Terminal frame does not match capture settings")
    env = dict(
        os.environ,
        PYTHONPATH=str(predicate.parent)
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    )

    def checked(script, log, *args):
        with (trial / log).open("w") as output:
            subprocess.run(
                [sys.executable, str(tools / script), *map(str, args)],
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                check=True,
            )

    checked(
        "audit_installed_layer_boundaries.py",
        "accepted-layers.log",
        trial / "launch.json",
        predicate,
        trial / "accepted-layers.json",
        "--accepted-nodes",
        trial / "physical-reference.node",
        trial / "physical-accepted.node",
    )
    checked(
        "audit_registered_initial_envelope.py",
        "envelope.log",
        "--terminal",
        implementation,
        trial / "accepted-terminal.mrvpack.zst",
        trial / "terminal.receipt.json",
        trial / "admission.json",
        trial / "launch.json",
        trial / "envelope",
    )
    checked(
        "classify_registered_envelope.py",
        "classify.log",
        trial / "envelope/accepted-envelope.json",
        trial / "stdout.log",
        trial / "envelope/physical-banks.json",
        "--window-receipt",
        trial / "native-traces/resting-ocular-sclera-window.json",
    )

    def read(p):
        return json.loads((trial / p).read_text())

    layers = read("accepted-layers.json")
    banks = read("envelope/physical-banks.json")
    envelope = read("envelope/accepted-envelope.json")
    rows = [
        dict(x.split("=", 1) for x in line.split()[1:])
        for line in (trial / "stdout.log").read_text().splitlines()
        if line.startswith("registered_cell_shape ")
    ]
    if sorted((r["phase"], int(r["material"])) for r in rows) != [
        ("accepted", 15),
        ("accepted", 16),
        ("reference", 15),
        ("reference", 16),
    ] or any(
        int(r["step"]) != (receipt["accepted_step"] if r["phase"] == "accepted" else 0)
        for r in rows
    ):
        raise ValueError(
            "Native material ledger does not contain both phases and materials"
        )
    pressure = []
    with (trial / "native-traces/resting-coupled.csv").open() as f:
        for row in csv.DictReader(f):
            if "pleural_pa" in row:
                pressure.append(float(row["pleural_pa"]))
    result = dict(
        accepted_steps=receipt["accepted_step"],
        simulated_s=receipt["accepted_time_s"],
        frame_sha256=receipt["pack_file_sha256"],
        physical_nodes_sha256=digest(trial / "physical-accepted.node"),
        complete_wall_s=state["wall_seconds"],
        end_to_end_factor=receipt["accepted_time_s"] / state["wall_seconds"],
        material_audits=rows,
        layer_crossing_pairs=sum(map(sum, layers["pair_counts"])),
        checked_banks=banks["checked_banks"],
        skin_self_intersections=envelope["skin_self_intersections"],
        physical_skin_tissue_pairs=banks["physical_nonocular_pairs"],
        ocular_source_pairs=banks["ocular_source_pairs"],
        maximum_skin_motion_m=envelope["maximum_skin_motion_m"],
        pleural_pressure_pa=pressure,
    )
    (trial / "verified-endpoint.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    verify(*(Path(s).resolve() for s in sys.argv[1:]))
