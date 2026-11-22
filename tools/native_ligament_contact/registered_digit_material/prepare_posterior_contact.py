"""Extract complete physical material stars under the load-bearing posterior skin."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
from fit_native_bed_barrier_load import force
from material_context import sha
from support_patch import prepare


def prepare_patch(run, material, transfer, owner, peaks, out):
    bed = json.loads(peaks.read_text())
    if bed["source_launch"] != sha(run / "launch.json") or bed["receipt"] != sha(
        run / "terminal.receipt.json"
    ):
        raise ValueError("Unmatched native load")
    data = prepare(run, material, transfer, owner)
    launch = json.loads((run / "launch.json").read_text())
    env = launch["environment"]
    prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_"

    def source(key):
        p = Path(env[prefix + key + "_CSV"])
        if sha(p) != env[prefix + key + "_SHA256"]:
            raise ValueError("Unmatched source " + key)
        return p

    table = source("COMBINED_NODE_MAP")
    with table.open() as f:
        pairs = sorted(
            (int(r["registered_surface_vertex_row"]), int(r["joint_node_id"]) - 1)
            for r in csv.DictReader(f)
            if r["registered_surface_layer"] == "O"
        )
    rows, outer = np.array(pairs).T
    if not np.array_equal(rows, np.arange(len(rows))) or not np.array_equal(
        np.sort(outer), np.flatnonzero(data["layer"] == 1)
    ):
        raise ValueError("Different outer boundary")
    fpath = source("SURFACE_O_FACES")
    triangles = outer[
        np.loadtxt(fpath, delimiter=",", skiprows=1, usecols=(1, 2, 3), dtype=int)
    ]
    bed_z = np.minimum(data["current"][:, 2], data["end"][:, 2])
    selected = bed_z[triangles].min(1) < data["width"]
    patch_nodes = np.unique(triangles[selected])
    selected_cells = np.flatnonzero(np.isin(data["t"], patch_nodes).any(1))
    face_ids = np.flatnonzero(np.isin(triangles, patch_nodes).any(1))
    nodes = np.unique(
        np.r_[data["t"][selected_cells].ravel(), triangles[face_ids].ravel()]
    )
    t = np.searchsorted(nodes, data["t"][selected_cells])
    f = np.searchsorted(nodes, triangles[face_ids])
    outside = (data["layer"] == 1) & (~np.isin(np.arange(len(data["layer"])), nodes))
    if np.any(outside & (bed_z < data["width"])):
        raise ValueError("Incomplete support patch")
    base = []
    for phase, key in [("current", "initial_barrier_N"), ("end", "endpoint_barrier_N")]:
        x = data[phase]
        a = (data["layer"] == 1) & (x[:, 2] < data["width"])
        loads = force(x[a, 2], data["mass"][a], data["dt"], data["width"])
        if not np.isclose(loads.sum(), bed[key], rtol=1e-11):
            raise ValueError("Native stationary reference changed")
        base.append([float(loads.sum()), *(loads @ x[a, :2]).tolist()])
    out.mkdir(exist_ok=False)
    np.savez_compressed(
        out / "patch.npz",
        nodes=nodes,
        cell_ids=selected_cells,
        face_ids=face_ids,
        t=t,
        f=f,
        marked=selected[face_ids],
        x=data["ref"][nodes],
        y=data["current"][nodes],
        end=data["end"][nodes],
        cell_mass=data["cell_mass"][selected_cells],
        mass=data["mass"][nodes],
        tags=data["tags"][selected_cells],
        layer=data["layer"][nodes],
        donor=data["donor"][nodes],
        props=data["props"],
        dt=data["dt"],
        width=data["width"],
        native_wrench=np.array(base),
    )
    result = {
        "registered": data["source"],
        "registered_tets": len(data["t"]),
        "registered_nodes": len(data["ref"]),
        "registered_surface": str(fpath),
        "surface_sha256": sha(fpath),
        "topology_patch_sha256": sha(out / "patch.npz"),
        "matched_receipt": bed["receipt"],
        "source_mass": json.loads(transfer.read_text())["mass"],
        "loaded_faces": int(selected.sum()),
        "complete_cell_stars": len(t),
        "complete_face_stars": len(f),
        "local_nodes": len(nodes),
    }
    (out / "source.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(
        json.dumps(prepare_patch(*(Path(s).resolve() for s in sys.argv[1:])), indent=2)
    )
