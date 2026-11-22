"""Measure dynamic displacement and bed load independently from reference-shape differences."""

import json
import re
import sys
from pathlib import Path
import numpy as np
from run_cavity_motion import digest


def compute(trial, output):
    evidence = json.loads((trial / "verified-endpoint.json").read_text())
    layers = json.loads((trial / "accepted-layers.json").read_text())
    launch = json.loads((trial / "launch.json").read_text())
    start = Path(launch["environment"]["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    end = trial / "physical-accepted.node"
    if (
        layers["inputs"]["NATIVE_LOADED_START_NODE_SHA256"] != digest(start)
        or layers["inputs"]["NATIVE_ACCEPTED_NODE_SHA256"] != digest(end)
        or evidence["physical_nodes_sha256"] != digest(end)
    ):
        raise ValueError("Mismatched loaded or accepted material")
    arrays = [np.loadtxt(f, skiprows=1) for f in (start, end)]
    if arrays[0].shape != arrays[1].shape or not np.array_equal(
        arrays[0][:, 0], arrays[1][:, 0]
    ):
        raise ValueError("Material node identity changed")
    d = np.linalg.norm(arrays[1][:, 1:4] - arrays[0][:, 1:4], axis=1)
    text = (trial / "stdout.log").read_text()

    def last(pattern):
        found = re.findall(pattern, text, re.MULTILINE)
        if not found:
            raise ValueError("Missing native metric " + pattern)
        return float(found[-1])

    reaction = last(
        r"^resting_fem_skin_bed_accepted_root_reaction .*?mean_normal_force_N=([0-9.eE+-]+)"
    )
    horizon = (
        last(r"^human_execution_stage=native_horizon_end wall_elapsed_ms=([0-9.eE+-]+)")
        / 1000.0
    )
    weight = (
        float(launch["argv"][launch["argv"].index("--resting-reference-mass-kg") + 1])
        * 9.81
    )
    env = json.loads((trial / "envelope/accepted-envelope.json").read_text())
    tissue = Path(launch["argv"][launch["argv"].index("--soft-tissue-payload") + 1])
    provenance = json.loads(tissue.with_suffix(".manifest.json").read_text())
    if digest(tissue) != provenance["payload"]["sha256"]:
        raise ValueError("Registered anatomy payload changed")
    names = {
        r["stable_id"]: r.get("label", r.get("member_id"))
        for r in provenance["source"]["surfaces"]
        if r["layer"] == "muscle"
    }
    worst = [
        dict(stable_id=stable, label=names.get(stable), pairs=count)
        for sem, stable, _, count in sorted(
            env["affected_banks"], key=lambda row: -row[-1]
        )
        if sem == 51005
    ][:6]
    data = dict(
        physical_simulated_s=evidence["simulated_s"],
        maximum_loaded_to_accepted_displacement_m=float(d.max()),
        rms_loaded_to_accepted_displacement_m=float(np.sqrt(np.mean(d * d))),
        max_displacement_node=int(d.argmax()),
        registered_outer_surface_reference_to_accepted_max_m=env[
            "maximum_skin_motion_m"
        ],
        native_horizon_s=horizon,
        native_horizon_factor=evidence["simulated_s"] / horizon,
        bed_normal_reaction_n=reaction,
        weight_n=weight,
        bed_reaction_to_weight=reaction / weight,
        major_muscle_crossings=worst,
        launch_sha256=digest(trial / "launch.json"),
    )
    output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    compute(*(Path(s).resolve() for s in sys.argv[1:]))
