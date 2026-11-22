"""Recover all loaded skin contact faces, their exact cell stars, and source mass constraints."""

import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.linalg import qr
from material_context import sha
from support_patch import prepare
from conserved_cavity_rows import inventories
from fit_native_bed_barrier_load import force


def prepare_fit(base, out):
    run = base / "native-prefix"
    mat = base.parent / "paired-reference"
    owner = base / "owner"
    d = prepare(run, mat, owner / "transfer.json", owner / "native-owner")
    labels, _ = inventories(run, mat)
    env = json.loads((run / "launch.json").read_text())["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"

    def asset(k):
        p = Path(env[prefix + k + "_CSV"])
        if sha(p) != env[prefix + k + "_SHA256"]:
            raise ValueError(k)
        return p

    with asset("COMBINED_NODE_MAP").open() as f:
        rows = sorted(
            (int(r["registered_surface_vertex_row"]), int(r["joint_node_id"]) - 1)
            for r in csv.DictReader(f)
            if r["registered_surface_layer"] == "O"
        )
    sequential, outer = np.asarray(rows).T
    if not np.array_equal(sequential, np.arange(len(outer))) or len(
        np.unique(outer)
    ) != len(outer):
        raise ValueError("Outer ownership")
    faces = outer[
        np.loadtxt(
            asset("SURFACE_O_FACES"),
            delimiter=",",
            skiprows=1,
            usecols=(1, 2, 3),
            dtype=int,
        )
    ]
    x, y = d["ref"], d["current"]
    w = float(d["width"])
    dt = float(d["dt"])
    z = y[:, 2]
    p = x[faces]
    area = np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1) * 0.5
    dual = np.bincount(faces.ravel(), weights=np.repeat(area / 3, 3), minlength=len(z))
    footprint = faces[z[faces].min(1) < w + 3e-6]
    protected = np.unique(np.r_[d["anchors"], d["dm_tets"].ravel()])
    # Include the neighboring unloaded material as a source-conserving deformation basis.
    roots = np.unique(footprint)
    near_t = d["t"][np.isin(d["t"], roots).any(1)]
    adjustable = np.setdiff1d(np.unique(near_t), protected)
    adjustable = adjustable[
        (np.isin(d["layer"][adjustable], [0, 1, 2])) & (z[adjustable] < 0.002)
    ]
    cell = np.flatnonzero(np.isin(d["t"], adjustable).any(1))
    face = np.flatnonzero(np.isin(faces, adjustable).any(1) | (z[faces].min(1) < w))
    nodes = np.unique(np.r_[faces[face].ravel(), d["t"][cell].ravel()])
    t = np.searchsorted(nodes, d["t"][cell])
    f = np.searchsorted(nodes, faces[face])
    free = np.searchsorted(nodes, adjustable)
    mass = d["mass"][nodes]
    ol = d["layer"][nodes] == 1
    density = np.zeros(len(nodes))
    density[ol] = (
        force(0.85 * w, mass[ol], dt, w) / force(0.85 * w, 1.0, dt, w) / dual[nodes[ol]]
    )
    W = []
    for g in np.unique(labels[cell, 1:], axis=0):
        sel = (labels[cell, 1:] == g).all(1)
        row = np.bincount(
            t[sel].ravel(),
            weights=np.repeat(d["cell_mass"][cell[sel]] / 4, 4),
            minlength=len(nodes),
        )
        W.append(row[free])
    raw = np.asarray(W)
    norm = np.linalg.norm(raw, axis=1)
    W = raw[norm > 0] / norm[norm > 0, None]
    _, rr, piv = qr(W.T, mode="economic", pivoting=True)
    W = W[piv[: np.linalg.matrix_rank(rr)]]
    # The remaining outer faces stay outside the band even under the displacement bound.
    gap = z[faces[np.setdiff1d(np.arange(len(faces)), face)]].min()
    if gap < w or not np.isfinite(density).all() or np.any(area <= 0):
        raise ValueError("Incomplete material support")
    out.mkdir(exist_ok=True)
    p = out / "support.npz"
    np.savez_compressed(
        p,
        x=x[nodes],
        y=y[nodes],
        nodes=nodes,
        t=t,
        f=f,
        free=free,
        W=W,
        raw=raw,
        area=area[face],
        density=density,
        props=d["props"],
        tags=d["tags"][cell],
        mass=mass,
        dt=dt,
        width=w,
        excluded_min=gap,
    )
    j = dict(
        source=d["source"],
        data=sha(p),
        mesh=sha(asset("SURFACE_O_FACES")),
        mass=sha(mat / "fine-mass-routed/current-density-closed-flow.npz"),
        free=len(free),
        cells=len(t),
        faces=len(f),
        source_mass_constraints=len(W),
        bed_plane_height_m=0.0,
    )
    (out / "input.json").write_text(json.dumps(j, indent=2) + "\n")
    print(j, flush=True)


if __name__ == "__main__":
    prepare_fit(*(Path(x).resolve() for x in sys.argv[1:]))
