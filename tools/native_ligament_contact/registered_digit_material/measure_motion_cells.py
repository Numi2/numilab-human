"""Locate motion quality failures with source rows and matched before/after states."""

import json
import sys
from pathlib import Path
import numpy as np
from bind_posterior_equilibrium import coordinates
from fit_dorsal_material import inputs
from torso_cell_geometry import geom
from material_context import sha


def analyze(ctrl, run, case, out):
    r, y, t, role, ids = inputs(run)
    x0, y0, t0, m0, old_ids = inputs(ctrl)
    # Only the installed coordinates are eligible to differ.
    if not (
        np.array_equal(r, x0) and np.array_equal(t, t0) and np.array_equal(role, m0)
    ):
        raise ValueError("Source topology changed")
    g = coordinates(run / "physical-accepted.node")
    old = coordinates(ctrl / "physical-accepted.node")
    area = np.load(case / "field.npz")["nodes"]
    counts = {15: 0, 16: 0}
    evidence = []
    for off in range(0, len(t), 32768):
        s = t[off : off + 32768]
        vol, q = geom(g, s)
        bad = np.flatnonzero((vol <= 1e-18) | (q < 1e-5))
        for ii in bad:
            z = s[[ii]]
            row = int(off + ii)
            faces = {}
            for label, xyz in [
                ("reference", r),
                ("control_initial", y0),
                ("control_accepted", old),
                ("candidate_initial", y),
                ("candidate_accepted", g),
            ]:
                v, k = geom(xyz[z][0].astype(float), np.arange(4)[None])
                faces[label] = dict(volume_m3=float(v[0]), normalized_shape=float(k[0]))
            counts[int(role[row])] += 1
            evidence.append(
                dict(
                    source_row=row,
                    material=int(role[row]),
                    vertices=z[0].tolist(),
                    changed_at_initialization=np.intersect1d(z, area).tolist(),
                    centroid_m=r[z[0]].mean(0).tolist(),
                    quality=faces,
                )
            )
    status = json.loads((run / "native-outcome.json").read_text())
    if (
        status["accepted_steps"] != 1
        or status["exit_code"]
        or not status["initial_install"]
    ):
        raise ValueError("Missing first-step receipt")
    for row in status["material_endpoints"]:
        if row["phase"] == "accepted" and (
            counts[int(row["material"])] != int(row["shape_lt_1e_5"])
            or int(row["volume_le_1e_18"]) != 0
        ):
            raise ValueError("Native totals differ")
    out.write_text(
        json.dumps(
            dict(
                source=ids,
                pose=sha(case / "current.node"),
                accepted=sha(run / "physical-accepted.node"),
                counts=counts,
                rows=evidence,
            ),
            indent=2,
        )
        + "\n"
    )
    print([(k["source_row"], k["changed_at_initialization"]) for k in evidence])


if __name__ == "__main__":
    analyze(*(Path(p).resolve() for p in sys.argv[1:]))
