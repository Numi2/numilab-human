"""Repair the exact Float32 determinant failures in separated local cavities."""
from collections import Counter
import json
from itertools import combinations
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, hstack, vstack

ROOT = Path.cwd().parent
PRIOR = ROOT / "resting-clearance-and-atrium-010"
COMPILER = ROOT / "captured-fat-sliver-495/proposal/compiler"
FLOOR = 1e-5
EDGES = np.asarray(list(combinations(range(4), 2)))
AXES = np.arange(3)

with np.load(PRIOR / "fine-transfer-screen-map.npz", allow_pickle=False) as data:
    original = data["old_world_m"].astype(np.float32)
    start = (data["old_world_m"] + data["fine_node_delta_m"]).astype(np.float32)
    roles = data["roles"]
topology = np.loadtxt(COMPILER / "volume-export/volume/mesh.ele", skiprows=1, dtype=np.int32)
flips = json.loads((PRIOR / "fine-fixed-boundary-flips.json").read_text())["replacements"]
retained = np.ones(len(topology), dtype=bool)
for operation in flips:
    retained[operation["retired"]] = False
cells = np.vstack((topology[retained, 1:5] - 1, [t for r in flips for t in r["new"]]))
material = np.r_[topology[retained, 5], [r["material_id"] for r in flips for _ in r["new"]]]
del topology
more_flips=json.loads(Path("fine-relaxed-flips.json").read_text())["replacements"]
keep=np.ones(len(cells),dtype=bool)
for operation in more_flips:
    keep[operation["retired"]]=False
cells=np.vstack((cells[keep], [t for operation in more_flips for t in operation["new"]]))
material=np.r_[material[keep], [operation["material_id"] for operation in more_flips for _ in operation["new"]]]
position = start.copy()
with np.load("fine-local-relaxation.npz",allow_pickle=False) as data:
    position[data["changed_rows"]]=data["positions_f32"]

flat = cells.ravel()
star_cells = np.argsort(flat, kind="stable") // 4
star_offset = np.r_[0, np.cumsum(np.bincount(flat, minlength=len(start)))]

def star(nodes):
    return np.unique(np.concatenate([star_cells[star_offset[n]:star_offset[n+1]] for n in nodes]))

def measurements(point, tet):
    p = point[tet].astype(np.float64)
    e = p[:, 1:] - p[:, :1]
    edge = p[:, EDGES[:, 0]] - p[:, EDGES[:, 1]]
    scale = np.max(np.sum(edge * edge, axis=2), axis=1) ** 1.5
    cross = np.cross(e[:, [1, 2, 0]], e[:, [2, 0, 1]])
    det = np.einsum("ni,ni->n", e[:, 0], cross[:, 0])
    jacobian = np.concatenate((-cross.sum(axis=1)[:, None], cross), axis=1)
    return det, scale, jacobian

def qualities(point, tet):
    det, scale, _ = measurements(point, tet)
    return det / scale

def measure_all():
    quality = np.empty(len(cells))
    for first in range(0, len(cells), 60000):
        block = slice(first, first+60000)
        quality[block] = qualities(position, cells[block])
    return quality

def objective(q):
    return (int(np.count_nonzero(q <= 0)), int(np.count_nonzero(q < FLOOR)),
            float(np.maximum(0., FLOOR-q).sum()))

def fit(free, affected, bound_mm, multiplier, exterior_cap_mm):
    tets = cells[affected]
    det, scale, jacobian = measurements(position, tets)
    current_q = det/scale
    relocal = np.searchsorted(free, tets)
    valid = free[np.minimum(relocal, len(free)-1)] == tets
    row, corner = np.nonzero(valid)
    n = len(free)*3
    points=position[tets].astype(float)
    edges=points[:,EDGES[:,0]]-points[:,EDGES[:,1]]
    longest=np.argmax(np.sum(edges*edges,axis=2),axis=1)
    domain=np.arange(len(points))
    direction=edges[domain,longest]
    gradient=3*direction/np.sum(direction*direction,axis=1)[:,None]
    scale_gradient=np.zeros_like(jacobian)
    scale_gradient[domain,EDGES[longest,0]]=gradient
    scale_gradient[domain,EDGES[longest,1]]=-gradient
    jacobian=jacobian/scale[:,None,None]-current_q[:,None,None]*scale_gradient
    values = jacobian[row,corner] * .001
    partial = coo_matrix((values.ravel(), (np.repeat(row,3),
                          (relocal[row,corner,None]*3+AXES).ravel())), shape=(len(tets),n)).tocsr()

    # Require a margin around the Float32 gate on bad cells. The exact post-fit
    # census below also guards every formerly valid cell touched by the fit.
    mask = current_q < FLOOR
    goal = np.where(mask, FLOOR*multiplier, FLOOR)
    scores = np.flatnonzero(mask)
    slack = coo_matrix((-np.ones(len(scores)), (scores, np.arange(len(scores)))),
                       shape=(len(tets),len(scores))).tocsr()
    identity = eye(n, format="csr")
    absolute_value = vstack((hstack((identity,-identity)), hstack((-identity,-identity))))
    lhs = vstack((hstack((-partial,coo_matrix(partial.shape),slack)),
                  hstack((absolute_value,coo_matrix((2*n,len(scores))))))).tocsr()
    if multiplier>1.5:
        # Bound the first-order adverse effect of rounding an affected vertex.
        ulp=np.abs(np.spacing(position[tets])).astype(float)
        ulp[~valid]=0.
        padding=.55*np.sum(np.abs(jacobian)*ulp,axis=(1,2))
        goal+=np.where(mask,padding,np.minimum(padding,.5*np.maximum(0.,current_q-FLOOR)))
    rhs = np.r_[current_q-goal,np.zeros(2*n)]
    # Bound cumulative excursions on the exterior and penalize them in the fit.
    exterior = roles[free] == "O"
    difference = (position[free].astype(float) - start[free]) * 1000
    lower = np.maximum(-bound_mm, -exterior_cap_mm - difference)
    upper = np.minimum(bound_mm, exterior_cap_mm - difference)
    lower[~exterior] = -bound_mm
    upper[~exterior] = bound_mm
    if np.any(lower > upper):
        return None
    penalty = np.repeat(np.where(exterior, 16., 1.),3)
    lp = linprog(np.r_[np.zeros(n),penalty,np.full(len(scores), 2e5)], A_ub=lhs,b_ub=rhs,
                 bounds=list(zip(lower.ravel(),upper.ravel()))+[(0,bound_mm)]*n+[(0,None)]*len(scores),
                 method="highs",options={"primal_feasibility_tolerance":1e-9})
    return lp.x[:n].reshape(-1,3)*.001 if lp.success else None

def groups(failed):
    keys = {int(r) for r in failed}
    remaining = set(keys)
    while remaining:
        root = min(remaining)
        seen = {root}
        queue = [root]
        while queue:
            row = queue.pop()
            for neighbor in star(cells[row]):
                if neighbor in remaining and neighbor not in seen:
                    seen.add(int(neighbor))
                    queue.append(int(neighbor))
        remaining.difference_update(seen)
        yield np.asarray(sorted(seen),dtype=int)

initial = measure_all()
assert objective(initial)[:2] == (8, 19)
history = []
started = time.monotonic()
for cycle, (exterior, bound, grow, multiplier) in enumerate((
    (.3, .08, 1, 2.), (.3, .08, 2, 2.), (.32, .18, 2, 2.),
    (.37, .24, 3, 2.), (.45, .24, 3, 2.), (.45, .3, 3, 3.))):
    current = measure_all()
    broken = np.flatnonzero(current < FLOOR)
    if not len(broken):
        break
    accepted = Counter()
    print("cycle",cycle,"begin",objective(current)[:2],flush=True)
    for group in groups(broken):
        available = np.unique(cells[group])
        for _ in range(grow):
            if len(available) > 1400:
                break
            available = np.unique(cells[star(available)])
        free = available[(roles[available] != "O") | (exterior > 0)]
        if not len(free):
            continue
        affected = star(free)
        existing = qualities(position, cells[affected])
        old_score = objective(existing)
        delta = fit(free, affected, bound, multiplier, exterior)
        if delta is None:
            accepted["infeasible"] += 1
            continue
        before = position[free].copy()
        best = None
        for a in (1., .875, .75, .625, .5, .375, .25, .125, .0625):
            position[free] = before.astype(float) + a*delta
            q = qualities(position,cells[affected])
            if np.any((existing >= FLOOR) & (q < FLOOR)):
                continue
            score = objective(q)
            if score < old_score and (best is None or score < best[0]):
                best = score, position[free].copy()
        position[free] = before if best is None else best[1]
        accepted["rejected" if best is None else "improved"] += 1
    q = measure_all()
    row = dict(cycle=cycle, exterior_component_cap_mm=exterior, quality=objective(q),
               iterations=dict(accepted), seconds=time.monotonic()-started)
    print(json.dumps(row),flush=True)
    history.append(row)
    indices = np.flatnonzero(np.any(position != start,axis=1))
    np.savez_compressed("fine-determinant-polish.npz", changed_rows=indices, positions_f32=position[indices])

q = measure_all()
changed = np.linalg.norm(position.astype(float)-start,axis=1)
report = dict(initial_inverted=objective(initial)[0],initial_positive_below_floor=11,
              final_inverted=objective(q)[0], final_positive_below_floor=int(np.count_nonzero((q>0)&(q<FLOOR))),
              minimum_determinant=float(q.min()), remaining_rows=np.flatnonzero(q<FLOOR).tolist(),
              changed_by_role={str(r):int(np.count_nonzero((roles==r)&(changed>0))) for r in np.unique(roles)},
              max_motion_by_role_m={str(r):float(np.max(changed[roles==r])) for r in np.unique(roles)},
              float32_no_regressions=bool(np.all(q[initial>=FLOOR]>=FLOOR)), history=history,
              native_admitted=False, crossing_certified=False)
Path("fine-determinant-polish.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report),flush=True)
