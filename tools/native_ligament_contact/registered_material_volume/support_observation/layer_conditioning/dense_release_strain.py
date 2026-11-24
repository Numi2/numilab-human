"""Map an actual dense line-search failure to its registered material and release configuration."""

from pathlib import Path
import json
import sys
import numpy as np
from compile_connected_joint import digest


def read(p):
    return json.loads(p.read_bytes())


def run(case, limit, out):
    c = read(case / "launch.json")
    w = read(limit)
    if w["launch"] != digest(case / "launch.json") or w["native_log"] != digest(
        case / "stderr.log"
    ):
        raise ValueError("Different rejected step")
    e = c["environment"]
    p = Path(
        e["NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE"]
    ).parent
    data = read(p / "native-import.json")
    files = [
        p / n
        for n in ["mesh.node", "current.node", "mesh.ele", "current-child.labels.csv"]
    ]
    if any(
        Path(e[k]).resolve() != f.resolve()
        for k, f in zip(
            [
                "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_NODE",
                "NUMI_HUMAN_CURRENT_MATERIAL_TETGEN_NODE",
                "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_ELEMENTS",
            ],
            files[:3],
        )
    ):
        raise ValueError("Release files do not match the native input paths")
    ref, current = [np.loadtxt(f, skiprows=1, usecols=(1, 2, 3)) for f in files[:2]]
    t = np.loadtxt(files[2], skiprows=1, usecols=(0, 1, 2, 3, 4), dtype="i4")
    label = np.loadtxt(files[3], delimiter=",", skiprows=1, dtype="i8")
    if (
        len(ref) != data["nodes"]
        or len(t) != data["cells"]
        or len(label) != len(t)
        or not np.array_equal(t[:, 0] - 1, label[:, 0])
        or not np.array_equal(label[:, 0], np.arange(len(t)))
    ):
        raise ValueError("Incorrect child ancestry")
    v = t[:, 1:] - 1
    little = np.empty(len(t))
    ids = np.array(w["cell_backtracking"][-1]["unit_trial_witnesses"])
    source = w["cell_backtracking"][-1]
    off = source["global_cell"] - source["registered_cell"]
    ids -= off
    for i in range(0, len(v), 8000):
        k = v[i : i + 8000]
        x = ref[k]
        y = current[k]
        native = np.linalg.det(x[:, 1:] - x[:, :1])
        new = np.linalg.det(y[:, 1:] - y[:, :1])
        if np.any(native <= 0):
            raise ValueError("Nonpositive reference tets")
        little[i : i + len(k)] = new / native
    details = []
    for i in ids:
        x = ref[v[i]]
        y = current[v[i]]
        F = np.linalg.solve(x[1:] - x[0], y[1:] - y[0])
        sv = np.linalg.svd(F, compute_uv=False)
        edges = x[:, None] - x
        length = np.linalg.norm(edges, axis=2)
        face = np.array(
            [
                np.linalg.norm(np.cross(a - c, b - c)) / 2
                for a, b, c in [np.delete(x, k, axis=0) for k in range(4)]
            ]
        )
        volume = np.linalg.det(x[1:] - x[0]) / 6
        details.append(
            dict(
                global_tet=int(i + off),
                registered_row=int(i),
                material=int(label[i, 1]),
                donor=int(label[i, 2]),
                region=int(label[i, 3]),
                source_face=int(label[i, 4]),
                node_indices_zero_based=v[i].tolist(),
                release_centroid_m=y.mean(0).tolist(),
                release_J=little[i],
                release_stretches=sv.tolist(),
                reference_minimum_altitude_mm=3000 * volume / face.max(),
                reference_max_edge_mm=1000 * length.max(),
            )
        )
    low = little < 0.40
    regs, counts = np.unique(label[low, 3], return_counts=True)
    stats = dict(
        min=float(little.min()),
        negative=int(np.sum(little <= 0)),
        below_035=int(np.sum(little < 0.35)),
        below_040=int(low.sum()),
        below_050=int(np.sum(little < 0.50)),
        region_counts_below_040=dict(zip(map(str, regs), map(int, counts))),
    )
    out.write_text(
        json.dumps(
            dict(
                launch=w["launch"],
                inputs={f.name: digest(f) for f in files},
                native_import=data,
                release=stats,
                trial=source,
                witnesses=details,
                scope="Jacobian from stored reference to stored current coordinates supplied to the dense compiler, before any physical step. The native Newton certificate describes the later root. Reference shape quality, stored release distortion and trial rejection are different measurements.",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run(*(Path(x).resolve() for x in sys.argv[1:]))
