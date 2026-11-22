"""Compose a tested local reference correction with the registered whole-body source patch."""

import json, sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from material_context import sha


def write_coordinates(src, dst, rows, points):
    changed = dict(zip(rows.tolist(), points.tolist()))
    with src.open() as f, dst.open("w") as out:
        out.write(next(f))
        for i, line in enumerate(f):
            if i in changed:
                p = changed[i]
                out.write(f"{i + 1} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
            else:
                out.write(line)


def run(native, parent, case):
    patch = np.load(case / "field.npz")
    r = json.loads((case / "quality.json").read_text())
    g = json.loads((case / "interfaces.json").read_text())
    if (
        g["new"]
        or g["cleared"] == 0
        or any(v["bad"] for phase in r["quality"].values() for v in phase.values())
    ):
        raise ValueError("Unqualified candidate")
    x, y, t, role, pins = inputs(native)
    assert r["inputs"] == pins
    j = json.load(open(native / "launch.json"))
    e = j["environment"]
    ref = Path(e["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE"])
    current = Path(e["NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE"])
    ids = patch["nodes"]
    for pts, f, out in ((x, ref, "reference.node"), (y, current, "current.node")):
        pts[ids, 0] = (pts[ids, 0].astype(float) + patch["delta_x_m"]).astype("f4")
        write_coordinates(f, case / out, ids, pts[ids])
    original = np.load(parent)
    changed_cells = np.flatnonzero(np.any(np.isin(t, ids), axis=1))
    cells = np.union1d(original["cell"], changed_cells)
    rows = np.union1d(original["nodes"], np.unique(t[cells]))
    np.savez_compressed(
        case / "proposal.npz",
        nodes=rows,
        cell=cells,
        x=x[rows],
        t=np.searchsorted(rows, t[cells]).astype("i4"),
        role=role[cells],
    )
    (case / "material-identity.json").write_text(
        json.dumps(
            dict(
                reference=sha(case / "reference.node"),
                current=sha(case / "current.node"),
                proposal=sha(case / "proposal.npz"),
                native_source=pins,
                parent_proposal=sha(parent),
                new_physical_rows=len(ids),
                touched_existing_cells=len(changed_cells),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*(Path(v).resolve() for v in sys.argv[1:]))
