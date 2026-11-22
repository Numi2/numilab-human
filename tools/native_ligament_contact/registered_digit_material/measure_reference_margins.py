"""Count near-threshold cells in the actual installed mesh at matched physical states."""

import json
import sys
from pathlib import Path

import numpy as np
from fit_dorsal_material import inputs
from material_context import sha
from supported_material_admission import installed_topology
from torso_cell_geometry import geom


def count(control, trial):
    x, _, t, roles, pins = inputs(control)
    launch = json.loads((control / "launch.json").read_text())
    t, roles = installed_topology(t, roles, launch["environment"])
    out = {}
    bounds = (1.0, 1.0001, 1.001, 1.01, 1.1)
    states = {"material_reference": x.astype(float)}
    for p in (control, trial):
        report = json.loads((p / "verified-endpoint.json").read_text())
        if sha(p / "physical-accepted.node") != report["physical_nodes_sha256"]:
            raise ValueError("Changed physical pose")
        env = json.loads((p / "launch.json").read_text())["environment"]
        for key in (
            "NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256",
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE_SHA256",
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_ELEMENTS_SHA256",
        ):
            if env[key] != launch["environment"][key]:
                raise ValueError("Topology or reference changed")
        states[p.name] = np.loadtxt(
            p / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3)
        )
    for name, coords in states.items():
        v, q = np.empty(len(t)), np.empty(len(t))
        for start in range(0, len(t), 32768):
            rows = slice(start, start + 32768)
            v[rows], q[rows] = geom(coords, t[rows])
        out[name] = {}
        for role, material in ((15, "dermis"), (16, "adipose")):
            mask = roles == role
            out[name][material] = {
                "cells": int(mask.sum()),
                "minimum_volume_m3": float(v[mask].min()),
                "minimum_shape": float(q[mask].min()),
                "volume_below": [int(np.sum(v[mask] <= 1e-18 * b)) for b in bounds],
                "shape_below": [int(np.sum(q[mask] < 1e-5 * b)) for b in bounds],
            }
    return {"source": pins, "threshold_factors": bounds, "counts": out}


if __name__ == "__main__":
    print(json.dumps(count(*(Path(v).resolve() for v in sys.argv[1:])), indent=2))
