"""Count tetrahedra made of vertices from a single registered anatomical layer wall."""

from pathlib import Path
import json
import sys
import numpy as np
from compile_connected_joint import digest


def census(case, out):
    cfg = json.loads((case / "launch.json").read_bytes())
    folder = Path(
        cfg["environment"][
            "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE"
        ]
    ).parent
    x = np.loadtxt(folder / "mesh.node", skiprows=1, usecols=(1, 2, 3))
    cells = np.loadtxt(
        folder / "mesh.ele", skiprows=1, usecols=(1, 2, 3, 4, 5), dtype="i4"
    )
    tag = np.zeros(len(x), dtype="u1")
    with np.load(folder / "material-walls.npz") as walls:
        for i, key in enumerate(["O", "I", "D", "C"]):
            tag[np.unique(walls[key])] |= 1 << i
    rows = {}
    record = []
    for k in range(0, len(cells), 16000):
        block = cells[k : k + 16000]
        t = block[:, :4] - 1
        role = block[:, -1]
        p = x[t]
        common = np.bitwise_and.reduce(tag[t], axis=1) != 0
        a, b, c = (p[:, i] - p[:, 0] for i in [1, 2, 3])
        d = np.einsum("ij,ij->i", a, np.cross(b, c))
        if (d <= 0).any():
            raise ValueError("Unexpected reference inversion")
        areas = np.max(
            [
                np.linalg.norm(np.cross(u, v), axis=1)
                for u, v in [(a, b), (a, c), (b, c), (b - a, c - a)]
            ],
            axis=0,
        )
        h = d / areas
        maxedge = np.max(
            [np.linalg.norm(u, axis=1) for u in [a, b, c, b - a, c - a, c - b]], axis=0
        )
        for n, mask in [
            ("all", np.ones(len(h), dtype=bool)),
            ("altitude_below_1um", h < 1e-6),
            ("aspect_above_1000", maxedge / h > 1000),
            ("aspect_above_10000", maxedge / h > 10000),
        ]:
            for m in set(role):
                q = mask & (role == m)
                r = rows.setdefault(n, {}).setdefault(str(m), [0, 0, 0.0, 0.0])
                r[0] += int(q.sum())
                r[1] += int((q & common).sum())
                r[2] += d[q].sum() / 6
                r[3] += d[q & common].sum() / 6
        i = h.argmin()
        record.append((float(h[i]), int(k + i), float(maxedge[i])))
    out.write_text(
        json.dumps(
            dict(
                launch=digest(case / "launch.json"),
                inputs={
                    k: digest(folder / k)
                    for k in ["mesh.node", "mesh.ele", "material-walls.npz"]
                },
                roles={"15": "skin", "16": "fat"},
                columns=[
                    "cells",
                    "same_wall_cells",
                    "volume_m3",
                    "same_wall_volume_m3",
                ],
                categories=rows,
                minimum_altitude_and_row_and_edge_m=min(record),
                scope="A same-wall cell has four vertices on at least one identical registered O/I/D/C wall. The scale and aspect counts describe reference geometry. They are not material failure thresholds.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    census(*(Path(p).resolve() for p in sys.argv[1:]))
