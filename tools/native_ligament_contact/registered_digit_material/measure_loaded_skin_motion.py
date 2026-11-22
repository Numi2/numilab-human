"""Measure release displacement from the installed loaded pose in registered world coordinates."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from material_context import sha


def measure(trial):
    layers = json.loads((trial / "accepted-layers.json").read_text())
    result = json.loads((trial / "verified-endpoint.json").read_text())
    env = json.loads((trial / "launch.json").read_text())["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"
    first, last = [trial / f"physical-{s}.node" for s in ("reference", "accepted")]
    mapping = Path(env[prefix + "COMBINED_NODE_MAP_CSV"])
    registered = Path(env[prefix + "SURFACE_O_NODES_CSV"])
    for path, key in [
        (first, "NATIVE_REFERENCE_NODE_SHA256"),
        (last, "NATIVE_ACCEPTED_NODE_SHA256"),
        (mapping, "COMBINED_NODE_MAP_CSV"),
        (registered, "SURFACE_O_NODES_CSV"),
    ]:
        if sha(path) != layers["inputs"][key]:
            raise ValueError(f"Unmatched capture {key}")
    if (
        layers["inputs"]["NATIVE_ACCEPTED_NODE_SHA256"]
        != result["physical_nodes_sha256"]
    ):
        raise ValueError("Different material endpoint")
    outer = (
        np.loadtxt(registered, delimiter=",", skiprows=1, usecols=(2, 3, 4))
        .astype("f4")
        .astype(float)
    )
    node = np.full(len(outer), -1, dtype=int)
    donor = np.full(len(outer), -1, dtype=int)
    with mapping.open() as stream:
        for row in csv.DictReader(stream):
            if row["registered_surface_layer"] == "O":
                idx = int(row["registered_surface_vertex_row"])
                if idx < 0 or idx >= len(outer) or node[idx] != -1:
                    raise ValueError("Invalid surface index")
                node[idx] = int(row["joint_node_id"]) - 1
                donor[idx] = int(row["donor_body"])
    if np.any(node < 0) or len(np.unique(node)) != len(node):
        raise ValueError("Incomplete outer-skin correspondence")
    start, end = [
        np.loadtxt(path, skiprows=1, usecols=(1, 2, 3))[node] for path in (first, last)
    ]

    def displacement(a, b):
        length = np.linalg.norm(a - b, axis=1)
        if not np.all(np.isfinite(length)):
            raise ValueError("Nonfinite physical motion")
        i = np.argmax(length)
        return {
            "rms_m": float(np.sqrt(np.mean(length**2))),
            "maximum_m": float(length[i]),
            "p99_m": float(np.quantile(length, 0.99)),
            "peak_joint_node_one_based": int(node[i] + 1),
            "peak_donor": int(donor[i]),
            "nodes_over_10_um": int(np.sum(length > 1e-5)),
            "nodes_over_100_um": int(np.sum(length > 1e-4)),
        }

    return {
        "accepted_steps": result["accepted_steps"],
        "simulated_s": result["simulated_s"],
        "physical_node_sha256": result["physical_nodes_sha256"],
        "registered_outer_nodes": len(node),
        "reference_to_loaded": displacement(outer, start),
        "loaded_to_accepted": displacement(start, end),
        "reference_to_accepted": displacement(outer, end),
    }


if __name__ == "__main__":
    print(json.dumps(measure(Path(sys.argv[1]).resolve()), indent=2, allow_nan=False))
