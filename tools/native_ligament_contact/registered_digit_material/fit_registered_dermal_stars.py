"""Recondition witnessed interior dermal cells at both material and physical positions."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from cavity_material_witness import accepted_node_pin
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha
from registered_density import density_bounds
from registered_material_stars import run_patch


def run(case, witness, owner, material, out):
    x, y, t, tag, source = inputs(case)
    d = json.loads(witness.read_text())
    env = json.loads((case / "launch.json").read_text())["environment"]
    if (
        d["source"] != source
        or sha(case / "physical-accepted.node") != d["accepted"]
        or d["accepted"] != accepted_node_pin(case)
    ):
        raise ValueError("Different material capture")
    z = np.loadtxt(case / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    flow = material / "fine-mass-routed/current-density-closed-flow.npz"
    if (
        sha(flow)
        != json.loads((flow.parent / "mass-transfer.json").read_text())["flow_sha256"]
    ):
        raise ValueError("Mass map changed")
    with np.load(flow) as f:
        mass = f["element_mass_kg"]
        groups = f["group_for_cell"]
    mesh = Path(env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"])
    if sha(mesh) != env["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256"]:
        raise ValueError("Unknown connectivity")
    with np.load(mesh) as f:
        keep = f["retained_input_cell_rows"]
        retired = np.setdiff1d(np.arange(len(t)), keep)
        protected = np.unique(
            np.r_[t[retired].ravel(), f["replacement_tetrahedra"].ravel()]
        )
    nodefile = Path(
        env[
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_CSV"
        ]
    )
    if (
        sha(nodefile)
        != env[
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_COMBINED_NODE_MAP_SHA256"
        ]
    ):
        raise ValueError("Unknown source layers")
    movable = np.zeros(len(x), bool)
    with nodefile.open() as f:
        for r in csv.DictReader(f):
            if r["source_partition"] == "skin" and not r["registered_surface_layer"]:
                movable[int(r["joint_node_id"]) - 1] = True
    movable[protected] = False
    props, pins = materials(owner)
    bounds = density_bounds(owner, 15)
    results = []
    for w in d["material_witnesses"]:
        row = w["parents"][0]
        seed = np.array(w["local_nodes"])
        if not np.array_equal(seed, t[row]) or w["material"] != 15:
            raise ValueError("Wrong source witness")
        free0 = seed[movable[seed]]
        ring = np.unique(t[np.isin(t, free0).any(1)])
        free = ring[movable[ring]]
        cells = np.flatnonzero(np.isin(t, free).any(1))
        if not np.isin(cells, keep).all() or not np.all(tag[cells] == 15):
            raise ValueError("Unregistered material")
        core = cells.copy()
        allowed = (
            (tag == 15)
            & np.isin(groups, np.unique(groups[core]))
            & np.isin(np.arange(len(t)), keep)
        )
        attempts = []
        for depth in range(3):
            if depth:
                # Keep the additional source-group mass carriers physically in place.
                boundary = np.unique(t[cells])
                extra = np.flatnonzero(
                    allowed
                    & np.isin(t, boundary).any(1)
                    & ~np.isin(np.arange(len(t)), cells)
                )
                if not len(extra):
                    break
                center = x[seed].mean(0)
                order = np.argsort(
                    np.linalg.norm(x[t[extra]].mean(1) - center, axis=1), kind="stable"
                )
                cells = np.union1d(cells, extra[order[:40]])
            nodes = np.unique(t[cells])
            tt = np.searchsorted(nodes, t[cells])
            vv = np.searchsorted(nodes, free)
            states = [a[nodes].astype(float) for a in (x, y, z)]
            ri = int(np.flatnonzero(cells == row)[0])
            tries = []
            for axis in range(3):
                for reserve in (1.02e-5, 1.005e-5):
                    try:
                        v = run_patch(
                            states,
                            tt,
                            ri,
                            vv,
                            mass[cells],
                            groups[cells],
                            axis,
                            True,
                            reserve,
                            *props[0],
                            density_limits=bounds,
                        )
                    except ValueError as exc:
                        if str(exc) != "Fixed condition is invalid":
                            raise
                        v = {"status": "fixed neighbor outside construction constraint"}
                    v.update(axis=axis, target=reserve)
                    tries.append(v)
                    print(
                        row,
                        depth,
                        len(cells),
                        axis,
                        reserve,
                        v["status"],
                        v.get("failures"),
                        flush=True,
                    )
                    if v["status"] == "passed":
                        break
            attempts.append({"cells": cells.tolist(), "trials": tries})
            if any(v["status"] == "passed" for v in tries):
                break
        results.append({"source_row": row, "free": free.tolist(), "attempts": attempts})
    out.write_text(
        json.dumps(
            {
                "source": source,
                "witness": sha(witness),
                "material": pins,
                "limit": bounds,
                "mass": sha(flow),
                "results": results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*(Path(v).resolve() for v in sys.argv[1:]))
