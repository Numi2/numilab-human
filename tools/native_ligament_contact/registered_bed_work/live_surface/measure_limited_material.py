"""Resolve the failed bed row to its registered material star and independently integrate its reaction."""

import json, re, sys
from pathlib import Path
import numpy as np
from material_context import sha
from fit_refined_preload import mechanical
from compare_bed_sampling import integrate


def measure(root):
    case = root / "limit-witness/one"
    (w,) = [
        json.loads(s.split("=", 1)[1])
        for s in (case / "stderr.log").read_text().splitlines()
        if s.startswith("registered_plane_limit_witness=")
    ]
    (first,) = re.findall(
        r"\bfirst_global_outer_fem_node=(\d+)\b", (case / "stdout.log").read_text()
    )
    d = dict(np.load(root / "load/support.npz"))
    local = w["node"] - int(first)
    (i,) = np.flatnonzero(d["nodes"] == local)
    y = d["y"].copy()
    y[d["free"]] = np.load(root / "wrench-fit/preload.npz")["coordinates"]
    sel = (d["f"] == i).any(1)
    faces = d["f"][sel]
    dt = float(d["dt"])
    if (
        i not in d["free"]
        or w["pair"] != local
        or np.linalg.norm(y[i, :2] - w["point_xyz_m"][:2]) > 2e-5
    ):
        raise ValueError("Unmatched material feature")
    _, elastic = mechanical((d["x"], y), d["t"], d["tags"], d["props"])

    def loads(y, order):
        return integrate(
            y, faces, d["area"][sel], d["density"], dt, float(d["width"]), order
        )[0][i]

    initial = float(loads(y, 4))
    state = y.copy()
    state[i, 2] = w["gap_m"]
    orders = [4, 8, 16, 32, 64, 128, 256]
    q = {k: float(loads(state, k)) for k in orders}
    if abs(q[256] - q[128]) > 1e-6 or not (
        w["gap_m"] > w["floor_m"] and w["closing_displacement_m"] > w["gap_m"]
    ):
        raise ValueError("Unresolved integration or unrelated bound")
    out = dict(
        native=w,
        native_normal_N=w["normal_impulse_Ns"] / dt,
        registered_node=int(local),
        mass_kg=float(d["mass"][i]),
        reference_xyz_m=d["x"][i].tolist(),
        loaded_xyz_m=y[i].tolist(),
        initial_skin_fat_elastic_force_N=elastic[i].tolist(),
        initial_bed_normal_N=initial,
        initial_adjacent_gaps_m=y[faces, 2].tolist(),
        loaded_cell_count=int(np.sum((d["t"] == i).any(1))),
        fixed_neighbours_at_witness_gap_force_by_order_N=q,
        source=sha(case / "stderr.log"),
        displacement_scope="Quadrature sensitivity varies only this vertex height from the loaded pose. Other positions at the failed Newton iterate were not reconstructed.",
    )
    (root / "limiting-material.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    measure(Path(sys.argv[1]).resolve())
