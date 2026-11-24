"""Trace connectivity from the native bed impulse to the actual registered cortical contact nodes."""

import csv
import json
import re
import runpy
import sys
from pathlib import Path

import igl
import numpy as np
from compile_connected_joint import digest
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from audit_native_shoulder_contacts import load


def inspect(w, case, out):
    cfg = json.loads((case / "launch.json").read_bytes())
    target = Path(cfg["argv"][0]).parent / "NumiHumanCorticalSkinBinding.inc"
    s = target.read_text()
    v = w / "material-import/volume"
    cells = v / "mesh.ele"
    wall = v / "material-walls.npz"
    stamps = dict(re.findall(r'constexpr char (\w+)\[\]="([0-9a-f]{64})"', s))
    if (
        digest(cells) != stamps["elements"]
        or digest(Path(cfg["argv"][3])) != stamps["bone"]
    ):
        raise ValueError("Incompatible compiled skin and bone")
    samples = np.array(
        [
            list(map(float, m))
            for m in re.findall(
                r"^\{(\d+),(\d+),(\d+),(\d+),([\d.eE+\-]+),([\d.eE+\-]+)\},", s, re.M
            )
        ]
    )
    if not len(samples):
        raise ValueError("Missing native cortical samples")
    t = np.loadtxt(cells, skiprows=1, usecols=(1, 2, 3, 4), dtype="i4") - 1
    p = np.loadtxt(case / "physical-accepted.node", skiprows=1, usecols=(1, 2, 3))
    impulse = case / "physical-posterior-bed-history.csv"
    with impulse.open() as h:
        rows = list(csv.DictReader(h))
    bed = np.array([int(r["registered_skin_node"]) for r in rows])
    forces = np.array([float(r["terminal_microstep_normal_impulse_Ns"]) for r in rows])
    if not np.isin(bed, np.unique(np.load(wall)["O"])).all() or (forces <= 0).any():
        raise ValueError("Invalid bed material")
    graph = coo_matrix(
        (np.ones(3 * len(t), dtype="i1"), (np.repeat(t[:, 0], 3), t[:, 1:].ravel())),
        shape=(len(p),) * 2,
    ).tocsr()
    _, region = connected_components(graph, directed=False)
    source = bed[np.argmax(forces)]
    sid = samples[:, 0].astype(int)
    banks = samples[:, 2].astype(int)
    info = {}
    section = np.linalg.norm(p - p[source], axis=1)
    for radius in (0.12, 0.24, 0.48):
        # Every path shorter than radius stays inside the Euclidean ball. Include
        # edges with both ends inside even when another corner of that cell lies outside.
        keep = section < radius
        candidate = t[keep[t].any(1)]
        keys = []
        for a, b in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
            ok = keep[candidate[:, a]] & keep[candidate[:, b]]
            u, h = candidate[ok, a], candidate[ok, b]
            keys.append(np.minimum(u, h).astype("int64") * len(p) + np.maximum(u, h))
        keys = np.unique(np.concatenate(keys))
        a, b = keys // len(p), keys % len(p)
        costs = np.linalg.norm(p[a] - p[b], axis=1)
        G = csr_matrix(
            (np.r_[costs, costs], (np.r_[a, b], np.r_[b, a])), shape=(len(p),) * 2
        )
        dist, prev = dijkstra(
            G, indices=int(source), return_predecessors=True, limit=radius
        )
        for bank in sorted(set(banks)):
            nodes = np.unique(sid[banks == bank])
            reachable = nodes[region[nodes] == region[source]]
            hit = reachable[np.isfinite(dist[reachable])]
            q = dict(
                samples=len(nodes),
                in_source_component=len(reachable),
                nearest_euclidean_m=float(section[nodes].min()),
            )
            if len(hit):
                k = int(hit[np.argmin(dist[hit])])
                path = [k]
                while path[-1] != source:
                    path.append(int(prev[path[-1]]))
                q.update(nearest_sample=k, edge_path_m=float(dist[k]), path=path[::-1])
            info[int(bank)] = q
        if "path" in info[10] or not info[10]["in_source_component"]:
            break
    api = runpy.run_path((w / "primitive-registry.path").read_text().strip())
    shapes, cap = load(case, api)
    for bank, r in info.items():
        if "path" not in r:
            continue
        pt = p[r["nearest_sample"]][None, :]
        bone, faces = shapes[51004, bank]
        d, fac, nearest = igl.point_mesh_squared_distance(
            pt, np.ascontiguousarray(bone), np.ascontiguousarray(faces, dtype="i8")
        )
        r.update(
            bone_distance_m=float(np.sqrt(d[0])),
            bone_face=int(fac[0]),
            point_on_bone_m=nearest[0].tolist(),
        )
    rec = dict(
        launch=digest(case / "launch.json"),
        compiled_samples=digest(target),
        material=digest(case / "physical-accepted.node"),
        skin_fat_elements=stamps["elements"],
        pack=cap["pack_file_sha256"],
        source_node=int(source),
        source_impulse_fraction=float(forces.max() / forces.sum()),
        source_component_nodes=int(np.count_nonzero(region == region[source])),
        total_nodes=len(p),
        bed_nodes_in_component=int(np.count_nonzero(region[bed] == region[source])),
        cortex=info,
        scope="Native skin/fat connectivity, edge-path length and geometric bone distance. A path proves topological connectivity only. Contact forces require an active unilateral row and constitutive stress; subtracting contact thickness or imposing a constraint is not part of this observation.",
    )
    out.write_text(json.dumps(rec, indent=2, allow_nan=False) + "\n")
    print("scapula", info[10], flush=True)


if __name__ == "__main__":
    inspect(*(Path(x).resolve() for x in sys.argv[1:]))
