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

# Pack three sorted indices in 57 bits. Pair each shared face exactly once,
# retaining its owner from each side when identifying a material interface.
bits = (len(points)-1).bit_length()
assert 3*bits <= 64
mask = (1 << bits)-1
face_keys = np.empty(4*len(nodes), dtype=np.uint64)
for face_index, corners in enumerate(combinations(range(4), 3)):
    face = np.sort(nodes[:, corners], axis=1).astype(np.uint64)
    face_keys[face_index*len(nodes):(face_index+1)*len(nodes)] = (
        face[:,0] << (2*bits) | face[:,1] << bits | face[:,2])
order = np.argsort(face_keys, kind="stable")
sides = face_keys[order]
starts = np.r_[0,np.flatnonzero(sides[1:] != sides[:-1])+1]
counts = np.diff(starts, append=len(sides))
assert counts.max() == 2, "Nonmanifold full-volume reference face"
exterior_keys = sides[starts[counts==1]]
pair_starts = starts[counts==2]
is_interface = (role[order[pair_starts] % len(nodes)] !=
                role[order[pair_starts+1] % len(nodes)])
interface_keys = sides[pair_starts[is_interface]]
fixed = np.zeros(len(points), dtype=np.bool_)
for keys in (exterior_keys,interface_keys):
    for shift in (0,bits,2*bits):
        fixed[(keys >> shift) & mask] = True
fixed_count = np.count_nonzero(fixed[nodes], axis=1)
script=Path.home()/'Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/bounded625-local-cavity-repair'
sys.path.insert(0,str(script))
from local_fixed_boundary_cavity_repair_625 import FACE_PATTERN, _ring

from collections import Counter
import time
def volumes(points, tetrahedra):
    p = points[tetrahedra]
    d = p[:,1:] - p[:,:1]
    det = np.linalg.det(d)
    diameter = np.maximum.reduce([np.linalg.norm(p[:,a]-p[:,b],axis=1) for a,b in edges])
    return det / 6, det / diameter**3

tets, material, geometry = nodes, role, points
corners=tets.ravel()
index=np.argsort(corners,kind='stable')//4
offsets=np.r_[0,np.cumsum(np.bincount(corners,minlength=len(geometry)))]

def hits(nodes):
    corner=np.asarray(nodes,dtype=int)
    lists=[index[offsets[node]:offsets[node+1]] for node in corner]
    first=min(lists,key=len)
    return set(int(cell) for cell in first if all(np.any(lst==cell) for lst in lists))

def faces(tet):
    return tet[FACE_PATTERN]

def chain(tetrahedra):
    oriented=Counter()
    for tet in tetrahedra:
        for a,b,c in faces(tet):
            orientation=1 if sum((a>b,a>c,b>c))%2==0 else -1
            oriented[tuple(sorted((int(a),int(b),int(c))))]+=orientation
    return {f:v for f,v in oriented.items() if v}

def make_proposals(row):
    cell=tets[row]
    for face in faces(cell):
        guard=hits(face)
        if len(guard)!=2:
            continue
        a,b,c=sorted(map(int,face))
        d,e=[int(v) for r in sorted(guard) for v in tets[r] if v not in face]
        if d==e or hits((d,e)):
            continue
        yield '2-3',guard,np.asarray(((d,e,a,b),(d,e,b,c),(d,e,c,a)),int)
    for edge in combinations(cell,2):
        guard=hits(edge)
        if len(guard) not in (3,4):
            continue
        edge=tuple(map(int,edge))
        ring=_ring(tets[sorted(guard)],edge)
        if not ring:
            continue
        a,b,c,*ending=ring
        d,e=edge
        if not ending:
            if hits((a,b,c))-guard:
                continue
            yield '3-2',guard,np.asarray(((a,b,c,d),(a,b,c,e)),int)
            continue
        x=ending[0]
        for diagonal,separator in (((a,c),((a,b,c),(a,c,x))),
                                   ((b,x),((b,c,x),(b,x,a)))):
            if hits(diagonal)-guard or any(hits(f)-guard for f in separator):
                continue
            yield '4-4',guard,np.asarray([(*f,node) for f in separator for node in (d,e)],int)


# Active physics, mass and ancestry are not touched in this diagnostic.
# Candidate energy is geometric. Centered singular values make its condition
# comparison independent of node order and the selected corner origin.
def metrics(local):
    rows = geometry[local]
    singular = np.linalg.svd(rows - rows.mean(axis=1,keepdims=True), compute_uv=False)
    basis = rows[:,1:] - rows[:,:1]
    inverse = np.linalg.inv(basis)
    local_mass = np.abs(np.linalg.det(basis)) / 6
    proxy = local_mass * (np.sum(inverse**2,axis=(1,2)) +
                          np.sum(inverse.sum(axis=2)**2,axis=1))
    return float(np.max(singular[:,0]/singular[:,-1])), float(proxy.sum())

begin = time.monotonic()
limit = int(sys.argv[2])
fixed = np.flatnonzero((fixed_count == 4) & (condition >= 1000))
fixed = fixed[np.argsort(-measure[fixed],kind="stable")][:limit]
selected, used, vertices_used = [], set(), set()
tested, rejections = Counter(), Counter()
for row in fixed:
    if row in used or any(v in vertices_used for v in tets[row]):
        continue
    best = None
    for kind, retired, replacement in make_proposals(row):
        tested[kind] += 1
        ri = sorted(retired)
        old = tets[ri]
        if np.any(material[ri] != material[row]) or any(v in vertices_used for v in old.ravel()):
            rejections["ownership_or_overlap"] += 1
            continue
        v, q = volumes(geometry, replacement)
        if np.any(np.abs(v) <= 1e-18) or np.any(np.abs(q) < 1e-5):
            rejections["absolute_or_shape_floor"] += 1
            continue
        flip = v < 0
        replacement[flip] = replacement[flip][:,[1,0,2,3]]
        if chain(old) != chain(replacement):
            rejections["oriented_cavity_boundary"] += 1
            continue
        if any(hits(t)-retired for t in replacement):
            rejections["duplicate_cell"] += 1
            continue
        old_volume = volume[ri].sum()
        if not np.isclose(old_volume, np.abs(v).sum(), rtol=1e-11, atol=1e-22):
            rejections["volume"] += 1
            continue
        before, old_proxy = metrics(old)
        after, new_proxy = metrics(replacement)
        if not (after <= .80*before and new_proxy <= .90*old_proxy):
            rejections["comparative_conditioning"] += 1
            continue
        score = old_proxy - new_proxy
        if best is None or score > best[0]:
            best = (score, ri, old, dict(kind=kind, retired=ri, material_id=int(material[row]),
                        new=replacement.tolist(), old_centered_condition=before,
                        new_centered_condition=after, old_gradient_energy_proxy=old_proxy,
                        new_gradient_energy_proxy=new_proxy, cavity_volume_m3=float(old_volume),
                        parent_union_from_retired_current_rows=ri))
    if best:
        score, ri, old, entry = best
        selected.append(entry)
        used.update(ri)
        vertices_used.update(old.ravel())

result = dict(input_sha256=digests, scope="Disjoint same-material cavity proposal. Mass ancestry, global intersections and native dynamics are not qualified.",
             target_count=len(fixed), tested=dict(tested), rejections=dict(rejections),
             selected=len(selected), affected_baseline_cells=len(used),
             new_cells=sum(len(e["new"]) for e in selected),
             rejected_or_unused_target_count=len(set(fixed)-used),
             whole_body_proxy_fraction_removed=float(sum(e["old_gradient_energy_proxy"]-e["new_gradient_energy_proxy"]
                                                        for e in selected)/measure.sum()),
             wall_s=time.monotonic()-begin, replacements=selected)
print(json.dumps(result, indent=2))
