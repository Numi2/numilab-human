"""Compose disjoint physical authoring fields and recertify their complete cell stars on both poses."""

import json, sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs, evaluate
from material_context import sha


def main(run, previous, out):
    a = np.load(previous / "field.npz")
    b = np.load(out / "local-field.npz")
    local = json.loads((out / "pelvic-fit.json").read_text())
    if (
        not local["selected"]
        or local["selected"]["new_pairs"]
        or np.intersect1d(a["nodes"], b["nodes"]).size
    ):
        raise ValueError("Ambiguous or intersecting local repair")
    ids = np.concatenate((a["nodes"], b["nodes"]))
    dx = np.concatenate((a["delta_x_m"], b["delta_x_m"]))
    order = np.argsort(ids)
    ids = ids[order]
    dx = dx[order]
    x, y, t, tag, pins = inputs(run)
    mask = np.isin(t, ids).any(1)
    cells = t[mask]
    role = tag[mask]
    if pins != json.loads((previous / "quality.json").read_text())["inputs"]:
        raise ValueError("Changed parent mesh")
    quality = {}
    for name, state in [("reference", x), ("loaded", y)]:
        q = state.copy()
        target = q[ids, 0].astype(float) + dx
        q[ids, 0] = target
        if not np.array_equal(q[ids, 0], target):
            raise ValueError("Patch is not exactly representable")
        quality[name] = {int(r): evaluate(q, cells[role == r]) for r in np.unique(role)}
        if any(row["bad"] for row in quality[name].values()):
            raise ValueError("Cell failed")
    np.savez_compressed(out / "field.npz", nodes=ids, delta_x_m=dx)
    (out / "quality.json").write_text(
        json.dumps(
            dict(
                quality=quality,
                inputs=pins,
                changed_nodes=len(ids),
                affected_cells=len(cells),
                parent_field_sha256=sha(previous / "field.npz"),
                local_field_sha256=sha(out / "local-field.npz"),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main(*(Path(p).resolve() for p in sys.argv[1:]))
