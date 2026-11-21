"""Independently certify exact Float32 cavity coverage, volumes and ownership."""
from collections import Counter
from fractions import Fraction
from functools import cache
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

run_path, proposal_path = map(Path, sys.argv[1:])
launch = json.loads(run_path.read_text())
proposal = json.loads(proposal_path.read_text())
prefix = "NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_TETGEN_"
paths = [Path(launch["environment"][prefix+key]) for key in ("NODE","ELEMENTS")]
for key, path in zip(("NODE","ELEMENTS"),paths):
    with path.open("rb") as stream:
        observed = hashlib.file_digest(stream, "sha256").hexdigest()
    assert observed == proposal["input_sha256"][key] == launch["environment"][prefix+key+"_SHA256"]
source = np.loadtxt(paths[0], skiprows=1, usecols=(0,1,2,3))
points = source[:,1:]
assert np.array_equal(source[:,0],np.arange(1,len(points)+1))
assert np.array_equal(points, points.astype(np.float32))
con = np.loadtxt(paths[1], skiprows=1, dtype=np.int32)
assert np.array_equal(con[:,0],np.arange(1,len(con)+1))
cells, material = con[:,1:5]-1, con[:,5]

@cache
def rational_point(row):
    return [Fraction(float(x)) for x in points[row]]

def subtract(x,y):
    return [left-right for left,right in zip(x,y)]

def determinant(points):
    o,a,b,c = points
    u,v,w = (subtract(x,o) for x in (a,b,c))
    return sum(u[i] * (v[(i+1)%3]*w[(i+2)%3] - v[(i+2)%3]*w[(i+1)%3]) for i in range(3))

def signed_chain(tetrahedra):
    net = Counter()
    for cell in tetrahedra:
        for omitted in range(4):
            face = [cell[i] for i in range(4) if i != omitted]
            parity = omitted + sum(i > j for i,j in combinations(face,2))
            net[tuple(sorted(face))] += 1 if parity % 2 == 0 else -1
    return {face:value for face,value in net.items() if value}

nodes_seen = set()
retired_seen = set()
by_mode = Counter()
by_material = Counter()
min_volume = None
min_shape_floor_margin = None
for entry in proposal["replacements"]:
    retired, new = entry["retired"], entry["new"]
    assert retired and not retired_seen.intersection(retired)
    assert all(0 <= row < len(cells) for row in retired)
    assert set(material[retired]) == {entry["material_id"]}
    old = cells[retired].tolist()
    old_nodes = set(node for t in old for node in t)
    assert old_nodes == set(node for t in new for node in t)
    assert not nodes_seen.intersection(old_nodes)
    assert signed_chain(old) == signed_chain(new)
    assert len(new) == len({tuple(sorted(t)) for t in new})
    old_total = sum(determinant(list(map(rational_point,tet))) for tet in old)
    new_total = 0
    for tet in new:
        p = list(map(rational_point,tet))
        det = determinant(p)
        local_volume = det / 6
        # Float32 dyadics make both floor checks exact. Squaring avoids an
        # approximate square root in det / maximal_edge_length cubed.
        max_length_squared = max(sum(value**2 for value in subtract(a,b))
                                 for a,b in combinations(p,2))
        assert local_volume > Fraction(1,10**18)
        assert det**2 >= max_length_squared**3 / 10**10
        margin = float(det**2 / max_length_squared**3 * 10**10)**.5
        min_volume = min(local_volume, min_volume) if min_volume is not None else local_volume
        min_shape_floor_margin = (min(margin,min_shape_floor_margin)
                                  if min_shape_floor_margin is not None else margin)
        new_total += det
    assert old_total == new_total
    nodes_seen.update(old_nodes)
    retired_seen.update(retired)
    by_mode[entry["kind"]] += 1
    by_material[entry["material_id"]] += 1

assert len(retired_seen) == proposal["affected_baseline_cells"]
print(json.dumps(dict(
    validity="exact_binary32_oriented_cavity_boundary_positive_volume",
    proposal_sha256=hashlib.sha256(proposal_path.read_bytes()).hexdigest(),
    selected_operations=len(proposal["replacements"]),
    modes=dict(by_mode),
    by_material=dict(by_material),
    retired_cells=len(retired_seen),
    added_cells=proposal["new_cells"],
    final_cell_count=len(cells)-len(retired_seen)+proposal["new_cells"],
    minimum_new_cell_volume_m3=float(min_volume) if min_volume is not None else None,
    minimum_shape_floor_multiple=min_shape_floor_margin,
    moved_nodes=0,
    transfer_admitted=False), indent=2))
