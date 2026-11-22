"""Prepare source-owned skin/fat cells and their exact force residual near loaded surfaces."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_cavity_response import elastic
from material_context import sha
from supported_material_admission import stress_limits

from support_patch import prepare


def run(run, material, transfer, owner, out, witness=None, *, bonded_interface=None):
    d = prepare(run, material, transfer, owner)
    t = d["t"]
    x = d["ref"]
    y = d["current"]
    end = d["end"]
    layer = d["layer"]
    # Include all outer nodes active at either end of the matched release.
    patches = np.flatnonzero(
        (layer == 1) & (np.minimum(y[:, 2], end[:, 2]) < d["width"])
    )
    movable = np.isin(layer, [0, 2]) & (~np.isin(np.arange(len(layer)), d["anchors"]))
    # Also leave every replacement-cavity mass moment exactly unchanged.
    movable[np.unique(d["dm_tets"])] = False
    if bonded_interface is not None and witness is not None:
        raise ValueError("Select a bed-release witness or a bonded material interface")
    if bonded_interface is not None:
        event = json.loads(bonded_interface.read_text())
        if event["material_sources"] != d["source"]:
            raise ValueError("Unmatched material source")
        witnesses = event["material_witnesses"]
        repair_cells = np.array(
            [r["composed_material_row"] for r in witnesses], dtype=int
        )
        if (
            not len(repair_cells)
            or np.min(repair_cells) < 0
            or np.max(repair_cells) >= len(t)
        ):
            raise ValueError("Invalid material witnesses")
        if any(
            t[i].tolist() != r["nodes"]
            for i, r in zip(repair_cells, witnesses, strict=True)
        ):
            raise ValueError("Witness differs from the installed material")
        seed = np.unique(t[repair_cells])
        if not np.all(layer[seed] == 2):
            raise ValueError("Expected a bonded material interface")
        neighbors = np.unique(t[np.isin(t, seed).any(1)])
        free = neighbors[movable[neighbors]]
        varied = np.union1d(np.flatnonzero(np.isin(t, free).any(1)), repair_cells)
        patches = np.intersect1d(patches, np.unique(t[varied]))
    else:
        ids = np.flatnonzero(np.isin(t, patches).any(1))
        candidates = np.unique(t[ids])
        # The inner dermis/fat interface has a single shared physical coordinate.
        # Both exposed surfaces, the contact footprint and skeletal sites stay exact.
        free = candidates[movable[candidates]]
        # Complete local stars for donor compensation and load redistribution.
        neighbor = np.unique(t[np.isin(t, free).any(1)])
        free = neighbor[movable[neighbor]]
        repair_cells = np.array([], dtype=int)
        if witness is not None:
            record = json.loads(witness.read_text())
            if (
                record["source"]["reference_sha256"] != d["source"]["reference_sha256"]
                or record["source"]["elements_sha256"] != d["source"]["elements_sha256"]
            ):
                raise ValueError("Witness from a different physical mesh")
            repair_cells = np.array(
                sorted({r["native_material_row"] for r in record["material_witnesses"]})
            )
            expected = {
                r["native_material_row"]: r["local_nodes"]
                for r in record["material_witnesses"]
            }
            if not all(t[row].tolist() == expected[row] for row in repair_cells):
                raise ValueError("Witness does not map to the active material")
            interior = np.unique(t[repair_cells])
            interior = interior[(layer[interior] == 0) & movable[interior]]
            neighbors = np.unique(t[np.isin(t, interior).any(1)])
            extra = neighbors[(layer[neighbors] == 0) & movable[neighbors]]
            if not len(extra):
                raise ValueError("No free interior at quality witness")
            free = np.union1d(free, extra)
        varied = np.flatnonzero(np.isin(t, free).any(1))
    rows = np.unique(np.r_[t[varied].ravel(), patches])
    complete = np.flatnonzero(np.isin(t, rows).any(1))
    full = np.unique(t[complete])
    baseline = np.zeros((len(full), 3))
    for tag in [15, 16]:
        baseline += elastic(
            x, y, t[complete[d["tags"][complete] == tag]], full, *d["props"][tag - 15]
        )[1]
    baseline = baseline[np.searchsorted(full, rows)]
    role = d["tags"][varied]
    tlocal = np.searchsorted(rows, t[varied])
    free_local = np.searchsorted(rows, free)
    # Ancestral mass ledgers group source cells by material and anatomical donor.
    launch = json.loads((run / "launch.json").read_text())
    with np.load(launch["environment"]["NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ"]) as r:
        keep = r["retained_input_cell_rows"]
    if varied.max() >= len(keep):
        raise ValueError("Replacement carrier touched")
    labels = material / "mesh/current-child.labels.csv"
    if (
        sha(labels)
        != json.loads(labels.with_suffix(".json").read_text())["labels_sha256"]
    ):
        raise ValueError("Source cell ancestry changed")
    label = np.loadtxt(
        labels, delimiter=",", skiprows=1, usecols=(0, 1, 2), dtype="i4"
    )[keep[varied]]
    if not np.array_equal(label[:, 0], keep[varied]) or not np.array_equal(
        label[:, 1], role
    ):
        raise ValueError("Invalid source cell mapping")
    element_mass = d["cell_mass"][varied]
    constraints = []

    def lump(selected):
        a = np.zeros(len(rows))
        for j in range(4):
            np.add.at(a, tlocal[selected, j], element_mass[selected] / 4)
        return a[free_local]

    for m in (15, 16):
        for donor in np.unique(label[role == m, 2]):
            constraints.append(lump((role == m) & (label[:, 2] == donor)))
        # Preserve the nodal-owner mass moments used in articulation as well.
        l = lump(role == m)
        for donor in np.unique(d["donor"][free]):
            constraints.append(l * (d["donor"][free] == donor))
    weights = np.array(constraints)
    weights = weights[np.max(weights, axis=1) > 0]
    out.mkdir(exist_ok=False)
    np.savez_compressed(
        out / "material.npz",
        x=x[rows],
        y=y[rows],
        endpoint=end[rows],
        tet=tlocal,
        roles=role,
        full_rows=rows,
        free=free_local,
        probes=np.searchsorted(rows, patches),
        base_elastic=baseline,
        source_cells=keep[varied],
        W=weights,
        guard_cells=np.searchsorted(varied, repair_cells),
        **(
            {"critical_nodes": np.searchsorted(rows, seed)}
            if bonded_interface is not None
            else {}
        ),
        mass=d["mass"][rows],
        donor=d["donor"][rows],
        layer=layer[rows],
        properties=d["props"],
        limits=stress_limits(d["materials"]),
        dt=d["dt"],
        width=d["width"],
    )
    info = {
        "geometry": d["source"],
        "source_launch_sha256": sha(run / "launch.json"),
        "reference_comparison": sha(run / "physical-accepted.node"),
        "labels_sha256": sha(labels),
        "materials": d["materials"],
        "touched_cells": len(tlocal),
        "complete_force_rows": len(rows),
        "free": len(free),
        "probes": len(patches),
        "source_mass_groups": len(weights),
        "prepared_sha256": sha(out / "material.npz"),
    }
    if bonded_interface is not None:
        info["bound_witness"] = sha(bonded_interface)
    (out / "source.json").write_text(json.dumps(info, indent=2) + "\n")
    print(
        {
            k: info[k]
            for k in [
                "touched_cells",
                "complete_force_rows",
                "free",
                "probes",
                "source_mass_groups",
            ]
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trial", "material", "transfer", "owner", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("witness", type=Path, nargs="?")
    parser.add_argument("--bonded-interface", type=Path)
    args = parser.parse_args()
    run(
        *(
            getattr(args, k).resolve()
            for k in ("trial", "material", "transfer", "owner", "out")
        ),
        args.witness.resolve() if args.witness else None,
        bonded_interface=args.bonded_interface.resolve()
        if args.bonded_interface
        else None,
    )
