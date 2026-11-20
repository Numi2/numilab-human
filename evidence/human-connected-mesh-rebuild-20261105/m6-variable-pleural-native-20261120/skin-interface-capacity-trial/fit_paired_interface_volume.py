"""Fit a local volume correction without altering the exposed skin or deep layer."""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, eye, hstack, vstack

from export_repaired_tetgen import array_hash
from screen_paired_mass_routes import HERE, PREVIOUS, load, sha

EDGES = np.asarray(list(combinations(range(4),2)))
FLOOR = 1e-5
MAX_AXIS_M = 0.00014


def measurements(point, tets):
    p = point[tets].astype(np.float64)
    e = p[:,1:] - p[:,:1]
    directed = p[:,EDGES[:,0]] - p[:,EDGES[:,1]]
    squared = np.einsum("nea,nea->ne",directed,directed)
    longest = np.argmax(squared,axis=1)
    rows = np.arange(len(p))
    scale = squared[rows,longest] ** 1.5
    cross = np.cross(e[:,[1,2,0]],e[:,[2,0,1]])
    det = np.einsum("na,na->n",e[:,0],cross[:,0])
    ddet = np.concatenate((-cross.sum(axis=1)[:,None],cross),axis=1)
    slope = 3*directed[rows,longest] / squared[rows,longest,None]
    scale_rel = np.zeros_like(ddet)
    scale_rel[rows,EDGES[longest,0]] = slope
    scale_rel[rows,EDGES[longest,1]] = -slope
    quality = det / scale
    return det / 6., quality, ddet / 6., ddet/scale[:,None,None] - quality[:,None,None]*scale_rel


def sparse_gradient(deriv, tets, free):
    loc = np.searchsorted(free,tets)
    match = free[np.minimum(loc,len(free)-1)] == tets
    row, corner = np.nonzero(match)
    col = (loc[row,corner,None]*3 + np.arange(3)).ravel()
    return coo_matrix((deriv[row,corner].ravel(),(np.repeat(row,3),col)),
                      shape=(len(tets),3*len(free))).tocsr()


audit = json.loads((HERE/"full-paired-route-audit.json").read_text())
proof = json.loads((HERE/"paired-region-density-capacity-bound.json").read_text())["Skin"]["analytic_deficit_cut"]
_,points,cells,material,_,_,roles,_,_ = load()
with np.load(PREVIOUS/"fine-mass-local-position-delta.npz",allow_pickle=False) as p:
    points[p["changed_rows"]] = p["positions_f32"]
assert array_hash(points,"<f4") == audit["geometry_sha256"]
assert array_hash(cells,"<i4") == audit["tetrahedron_sha256"]
receipt = audit["role_reports"]["Skin"]
input_mask = HERE/receipt["region_bounds_file"]
assert sha(input_mask) == receipt["region_bounds_sha256"]
with np.load(input_mask,allow_pickle=False) as info:
    bits, ids = info["group_and_regions"][:,1:],info["region_ids"]
    outside = np.zeros(bits.shape[1],dtype=np.uint64)
    for index in np.flatnonzero(~np.isin(ids,proof["region_ids"])):
        outside[index//64] |= np.uint64(1) << np.uint64(index%64)
    match = ~np.any(bits & outside, axis=1)
    confined = info["child_rows"][match[info["child_mask_index"]]]
assert len(confined) == proof["constrained_cells"]
patch = np.unique(cells[confined])
free = patch[roles[patch] == "I"]
assert len(free) > 0
visible = np.isin(cells,free).any(axis=1)
rows = np.flatnonzero(visible)
affected = cells[rows]
confined_local = np.isin(rows,confined)
skin = material[rows] == 15
volume,quality,dvolume,dquality = measurements(points,affected)
assert np.min(quality) >= FLOOR
n = 3*len(free)
vmat = sparse_gradient(dvolume,affected,free)
qmat = sparse_gradient(dquality,affected,free)
# Express movement in millimetres and volume in cubic millimetres so
# HiGHS sees both the determinant and the mass-limited volume constraints.
mm3_per_m3 = 1e9
mm = .001
cderiv = vmat[confined_local].sum(axis=0) * mm * mm3_per_m3
sderiv = vmat[skin].sum(axis=0) * mm * mm3_per_m3
goal = proof["shortage_kg"] / 900.
ordinary = eye(n,format="csr")
abs_values = vstack((hstack((ordinary,-ordinary)),hstack((-ordinary,-ordinary)))).tocsr()
inequality = vstack((hstack((-qmat*mm,coo_matrix(qmat.shape))),
                     abs_values,
                     hstack((coo_matrix(cderiv),coo_matrix((1,n)))))).tocsr()
equality = hstack((coo_matrix(sderiv),coo_matrix((1,n)))).tocsr()
outcome = {"cut_source_regions":proof["region_ids"],"free_interface_nodes":len(free),
           "incident_cells":len(rows),"required_volume_decrease_m3":goal,
           "fixed_exposed_and_deep_surfaces":True,"attempts":[]}
solution = linprog(np.r_[np.zeros(n),np.ones(n)],A_ub=inequality,
    b_ub=np.r_[quality-FLOOR,np.zeros(2*n),-1.05*goal*mm3_per_m3],
    A_eq=equality,b_eq=np.zeros(1),bounds=[(-MAX_AXIS_M/mm,MAX_AXIS_M/mm)]*n+[(0,MAX_AXIS_M/mm)]*n,
    method="highs",options={"time_limit":60.,"primal_feasibility_tolerance":1e-9})
outcome["linear_solver"] = solution.message
if solution.success:
    direction = solution.x[:n].reshape(-1,3)*mm
    after = points.copy()
    after[free] = points[free].astype(np.float64) + direction
    candidate_v, status, _, _ = measurements(after,affected)
    patch_cut = float(np.sum(volume[confined_local]-candidate_v[confined_local]))
    bounded = np.max(np.linalg.norm(after[free].astype(np.float64)-points[free],axis=1))
    actual = {"confined_volume_decrease_m3":patch_cut,
              "skin_volume_change_m3":float(np.sum(candidate_v[skin]-volume[skin])),
              "fat_volume_change_m3":float(np.sum(candidate_v[~skin]-volume[~skin])),
              "failed_quality_cells":int(np.count_nonzero(status<FLOOR)),
              "inverted_cells":int(np.count_nonzero(status<=0.)),
              "minimum_normalized_volume":float(np.min(status)),
              "max_displacement_m":float(bounded)}
    outcome["attempts"].append(actual)
    if (patch_cut >= goal and bounded <= .00025 and np.min(status) >= FLOOR and
            abs(actual["skin_volume_change_m3"]) <= .01*goal):
        modified = free[np.any(after[free] != points[free],axis=1)]
        candidate = HERE/"paired-interface-volume-trial.npz"
        with candidate.open("xb") as stream:
            np.savez_compressed(stream,changed_rows=modified, positions_f32=after[modified],affected_rows=rows)
        outcome["candidate_sha256"] = sha(candidate)
        outcome["candidate_geometry_sha256"] = array_hash(after,"<f4")
        outcome["stage"] = "Float32 geometry candidate; registration and crossing checks required"
(HERE/"paired-interface-volume-trial.json").write_text(json.dumps(outcome,indent=2)+"\n")
print(json.dumps(outcome,indent=2))
