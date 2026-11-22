"""Compute feasible provisional starts on the matched one-step response segment."""

from pathlib import Path
import json, sys
import numpy as np
from bind_posterior_equilibrium import coordinates
from fit_dorsal_material import inputs
from torso_cell_geometry import geom
from material_context import sha


def screen(ctrl, run, case, witness, out):
    start, end, t, role, ids = inputs(run)
    old = coordinates(ctrl / "physical-accepted.node")
    new = coordinates(run / "physical-accepted.node")
    f = np.load(case / "field.npz")
    rec = json.loads(witness.read_text())
    if sha(run / "physical-accepted.node") != rec["accepted"]:
        raise ValueError("Unmatched captured motion")
    frac = [0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0625]
    result = []
    for alpha in frac:
        p = old + (new - old) * alpha
        n = 0
        low = np.inf
        vh = np.inf
        for i in range(0, len(t), 32768):
            v, q = geom(p, t[i : i + 32768])
            n += int(np.sum((v <= 1e-18) | (q < 1e-5)))
            low = min(low, float(q.min()))
            vh = min(vh, float(v.min()))
        result.append(
            dict(
                fraction=alpha, extrapolated_failures=n, min_shape=low, min_volume_m3=vh
            )
        )
    out.write_text(
        json.dumps(
            dict(
                control_accepted_sha256=sha(ctrl / "physical-accepted.node"),
                fit=ids,
                changed_nodes=len(f["nodes"]),
                results=result,
                scope="Offline interpolation between two independently accepted endpoints. Only a fresh native run can verify any intervening initialization.",
            ),
            indent=2,
        )
        + "\n"
    )
    print([(r["fraction"], r["extrapolated_failures"]) for r in result])


if __name__ == "__main__":
    screen(*(Path(p).resolve() for p in sys.argv[1:]))
