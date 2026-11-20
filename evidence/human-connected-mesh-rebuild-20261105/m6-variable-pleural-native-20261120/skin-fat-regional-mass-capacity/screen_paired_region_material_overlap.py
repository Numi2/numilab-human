"""Screen whether verified material overlap could close the regional capacity cut."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from paired_route_projection import PairedRouteProjection
from reflow_paired_mass import sha

EDGES = np.array([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])


def transform(xyz):
    return np.linalg.inv((xyz[1:]-xyz[:1]).T)


def weights(points, tet, inverse):
    b = (points-tet[:1]) @ inverse.T
    return np.column_stack((1.-b.sum(axis=1),b))


def clip_segments(points, beta):
    intersections = []
    for a, b in EDGES:
        lo, hi = 0., 1.
        for initial, final in zip(beta[a], beta[b]):
            if initial < 0 and final < 0:
                lo, hi = 1., 0.
                break
            if initial < 0:
                lo = max(lo, -initial / (final-initial))
            elif final < 0:
                hi = min(hi, initial / (initial-final))
        if lo <= hi:
            direction = points[b]-points[a]
            intersections.extend((points[a]+lo*direction,points[a]+hi*direction))
    return intersections


def overlap(first, second, inv_first, inv_second):
    corners = clip_segments(first,weights(first,second,inv_second))
    corners += clip_segments(second,weights(second,first,inv_first))
    if len(corners) < 4:
        return 0.
    points = np.asarray(corners)-first[0]
    if np.linalg.matrix_rank(points[1:] - points[:1]) < 3:
        return 0.
    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        # Near-degenerate geometry gives no trustworthy positive witness.
        return 0.


root = Path.cwd()
audit = json.loads((root/"full-paired-route-audit.json").read_text())
certificate = json.loads((root/"paired-region-density-capacity-bound.json").read_text())["Skin"]
cut = certificate["analytic_deficit_cut"]
path = root.parent/"resting-clearance-and-atrium-014/fine-current-paired-semantic-cycle-mass.npz"
assert sha(path) == audit["source_mass_sha256"]
with np.load(path) as f:
    prior = {name:f[name] for name in f.files}
domains = np.zeros(len(prior["group_for_cell"]),np.int32)
mask_path = root/audit["role_reports"]["Skin"]["region_bounds_file"]
assert sha(mask_path) == audit["role_reports"]["Skin"]["region_bounds_sha256"]
with np.load(mask_path) as mask:
    rows, gids, bits, rids = (mask[k] for k in ("child_rows","child_mask_index","group_and_regions","region_ids"))
    outside = np.zeros(bits.shape[1]-1,np.uint64)
    for index in np.flatnonzero(~np.isin(rids,cut["region_ids"])):
        outside[index//64] |= np.uint64(1) << np.uint64(index%64)
    chosen = rows[np.all((bits[gids,1:] & outside) == 0,axis=1)]
assert len(chosen) == cut["constrained_cells"]
projection = PairedRouteProjection(prior,audit,domains)
child = projection.points[projection.cells[chosen]].astype(np.float64)
assert np.all(projection.material[chosen] == 15)
plo, phi = child.min(axis=1), child.max(axis=1)
parent = projection.parent
outside_rows = np.flatnonzero((parent["materials"] == 15) & ~np.isin(parent["regions"],cut["region_ids"]))
tet = parent["points"][parent["tets"][outside_rows]]
tlo, thi = tet.min(axis=1), tet.max(axis=1)
neighbor = np.all((thi > plo.min(axis=0)) & (tlo < phi.max(axis=0)),axis=1)
outside_rows, tet, tlo, thi = outside_rows[neighbor],tet[neighbor],tlo[neighbor],thi[neighbor]
inverse = [transform(value) for value in tet]
pairs = broad_cells = 0
child_volume = 0.
total_overlap = 0.
positive = []
tolerance = 2e-15
for row, P, lo, hi in zip(chosen,child,plo,phi):
    neighbors = np.flatnonzero(np.all((thi > lo) & (tlo < hi),axis=1))
    if not len(neighbors):
        continue
    pairs += len(neighbors)
    broad_cells += 1
    inv = transform(P)
    V = abs(np.linalg.det((P[1:]-P[:1]).T)) / 6.
    child_volume += V
    cell_overlap = 0.
    for source in neighbors:
        value = overlap(P,tet[source],inv,inverse[source])
        if value <= tolerance:
            continue
        parent_row = int(outside_rows[source])
        positive.append((int(row),parent_row,int(parent["regions"][parent_row]),value))
        cell_overlap += value
    total_overlap += min(V,cell_overlap)
record = {
    "verified_cut_cell_count":len(chosen),"nearby_outside_same_material_parent_cells":len(outside_rows),
    "broad_phase_pairs":pairs,"broad_phase_child_cells":broad_cells,
    "broad_phase_child_volume_m3":child_volume,"required_volume_at_min_skin_density_m3":cut["shortage_kg"]/900.,
    "tet_overlap_cell_count":len({r[0] for r in positive}),
    "tet_overlap_pairs":len(positive),
    "tet_overlap_volume_m3_capped_per_child":total_overlap,
    "retained_largest_overlaps":sorted(positive,key=lambda x:-x[-1])[:24],
    "scope":"Geometric overlap screen only. Current source routes and admission are unchanged.",
}
(root/"paired-region-material-overlap.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))
