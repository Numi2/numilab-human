"""Quantify local kinematic conditioning on the exact admitted skin/fat volume."""
from itertools import combinations
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

launch = json.loads(Path(sys.argv[1]).read_text())
prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_"
paths = {key: Path(launch["environment"][prefix+key]) for key in ("NODE","ELEMENTS")}
digests = {}
for key, path in paths.items():
    with path.open("rb") as stream:
        digests[key] = hashlib.file_digest(stream,"sha256").hexdigest()
    assert digests[key] == launch["environment"][prefix+key+"_SHA256"], path

points = np.loadtxt(paths["NODE"], skiprows=1, usecols=(0,1,2,3))
assert np.array_equal(points[:, 0], np.arange(1, len(points)+1)), "Node ordering changed"
points = points[:, 1:]
cells = np.loadtxt(paths["ELEMENTS"], skiprows=1, usecols=(0,1,2,3,4,5), dtype=np.int32)
assert np.array_equal(cells[:, 0], np.arange(1, len(cells)+1)), "Cell ordering changed"
nodes, role = cells[:, 1:5] - 1, cells[:, 5]
assert (len(points), len(nodes)) == (311020, 1448804)
assert np.array_equal(points, points.astype(np.float32))
assert nodes.min() >= 0 and nodes.max() < len(points) and set(role) == {15,16}

shared_nodes = np.zeros(len(points), np.uint8)
for material, bit in ((15, 1), (16, 2)):
    shared_nodes[nodes[role == material]] |= bit
touches_interface = np.any(shared_nodes[nodes] == 3, axis=1)
condition = np.empty(len(nodes))
measure = np.empty(len(nodes))
volume = np.empty(len(nodes))
quality = np.empty(len(nodes))
edges = tuple(combinations(range(4), 2))
for start in range(0, len(nodes), 32768):
    current = slice(start, start+32768)
    p = points[nodes[current]]
    basis = p[:, 1:] - p[:, :1]
    singular = np.linalg.svd(basis, compute_uv=False)
    determinant = np.linalg.det(basis)
    assert np.all(singular[:, -1] > 0) and np.all(determinant > 0)
    condition[current] = singular[:, 0] / singular[:, -1]
    diameter = np.sqrt(np.maximum.reduce(
        [np.einsum("ni,ni->n", p[:,a]-p[:,b], p[:,a]-p[:,b]) for a,b in edges]))
    volume[current] = determinant/6
    quality[current] = determinant / diameter**3
    gradients = np.linalg.inv(basis)
    measure[current] = volume[current] * (
        np.sum(gradients**2, axis=(1,2)) + np.sum(gradients.sum(axis=2)**2,axis=1))

def summarize(select):
    ids = np.flatnonzero(select)
    if not len(ids):
        return dict(cells=0)
    by_condition = ids[np.argsort(-condition[ids], kind="stable")]
    by_proxy = ids[np.argsort(-measure[ids], kind="stable")]
    def witness(i):
        return dict(row_zero_based=int(i), node_rows_zero_based=nodes[i].tolist(),
                    condition=float(condition[i]), volume_m3=float(volume[i]),
                    normalized_determinant=float(quality[i]),
                    gradient_energy_proxy_m=float(measure[i]),
                    center_m=points[nodes[i]].mean(axis=0).tolist())
    return dict(cells=len(ids), volume_m3=float(volume[ids].sum()),
                min_volume_m3=float(volume[ids].min()),
                min_normalized_determinant=float(quality[ids].min()),
                condition_p50_p95_p99_p999_max=np.quantile(condition[ids],
                                                           [0.5,0.95,0.99,0.999,1]).tolist(),
                count_by_condition_floor={str(floor):int(np.count_nonzero(condition[ids]>=floor))
                                          for floor in (10,100,1000,10000)},
                highest_condition=[witness(i) for i in by_condition[:5]],
                highest_gradient_energy_proxy=[witness(i) for i in by_proxy[:5]],
                top_1000_proxy_share=float(measure[by_proxy[:1000]].sum() / measure[ids].sum()))

result = dict(source_sha256=digests, shared_interface_nodes=int((shared_nodes==3).sum()),
              definition="Largest over smallest singular value of the reference edge basis. Volume times the sum of squared linear shape-function gradients is a geometry-only stiffness proxy. Neither is a coupled operator condition number.")
for material in (15, 16):
    result[str(material)] = summarize(role == material)
    result[str(material)+"_touches_interface"] = summarize((role == material) & touches_interface)
    result[str(material)+"_away_from_interface"] = summarize((role == material) & ~touches_interface)
print(json.dumps(result, indent=2))
