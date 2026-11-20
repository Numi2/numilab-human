"""Check necessary regional capacity without the carrier-distance restriction."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from paired_mass_density_reserve import density_bounds
from paired_route_projection import PairedRouteProjection
from reflow_paired_mass import sha


def region_partitions(words, region_ids):
    index = np.arange(len(region_ids), dtype=np.uint64)
    return ((words[:, index//64] >> (index%64)) & np.uint64(1)) != 0


root = Path.cwd()
audit = json.loads((root/"full-paired-route-audit.json").read_text())
path = root.parent/"resting-clearance-and-atrium-014/fine-current-paired-semantic-cycle-mass.npz"
assert sha(path) == audit["source_mass_sha256"]
with np.load(path) as f:
    prior = {key: f[key] for key in f.files}
roles = {"Skin":15,"Fat":16}
boxes = {}
group_for_cell = np.full(len(prior["group_for_cell"]),-1,dtype=np.int32)
start = 0
for name, data in audit["role_reports"].items():
    bounds = root/data["region_bounds_file"]
    assert sha(bounds) == data["region_bounds_sha256"]
    with np.load(bounds) as p:
        rows, domains, bits, region_ids = (p[k] for k in
            ("child_rows","child_mask_index","group_and_regions","region_ids"))
    masks, inv = np.unique(bits[:, 1:],axis=0,return_inverse=True)
    group_id = inv[domains]
    boxes[name] = rows, group_id, masks, region_ids
    group_for_cell[rows] = start + group_id
    start += len(masks)
assert np.all(group_for_cell >= 0)
projection = PairedRouteProjection(prior,audit,group_for_cell)
minimum, maximum = density_bounds(projection.points,projection.cells,projection.material)
source = np.bincount(prior["carrier_row"],weights=prior["mass_flow_kg"],
                     minlength=len(projection.tuples))
result = {}
for name,(rows,group_id,masks,region_ids) in boxes.items():
    role = roles[name]
    lo = np.bincount(group_id,weights=minimum[rows])
    hi = np.bincount(group_id,weights=maximum[rows])
    assert np.all(projection.material[rows] == role)
    selected = projection.tuples[:,0] == role
    source_digit = np.searchsorted(region_ids,projection.tuples[selected,2])
    quotas = np.bincount(source_digit,weights=source[selected],minlength=len(region_ids))
    allowed = region_partitions(masks,region_ids)
    assert np.all(allowed.sum(axis=1)>0)
    group, region = np.where(allowed)
    size = len(group)
    eq = coo_matrix((np.ones(size),(region,np.arange(size))),shape=(len(quotas),size)).tocsr()
    cap = coo_matrix((np.ones(size),(group,np.arange(size))),shape=(len(lo),size)).tocsr()
    unit = 1e6
    model = linprog(np.zeros(size),A_ub=vstack((cap,-cap)),b_ub=np.r_[hi,-lo]*unit,
                    A_eq=eq,b_eq=quotas*unit,bounds=(0.,None),method="highs")
    row = {
        "cell_count":len(rows),"domain_count":len(masks),"legal_region_edges":size,
        "strict_group_capacities_feasible":bool(model.success),"solver_message":model.message,
        "source_mass_kg":float(quotas.sum()),
    }
    if not model.success:
        # No plausible quantitative lower bound without an independently
        # checked feasible solution to the minimum unmet-density LP.
        from scipy.sparse import eye, hstack
        slack = eye(len(lo),format="csr")
        zero_slack = slack * 0
        min_deficit = linprog(np.r_[np.zeros(size),np.ones(len(lo))],
            A_ub=vstack((hstack((cap,zero_slack)),hstack((-cap,-slack)))),b_ub=np.r_[hi,-lo]*unit,
            A_eq=hstack((eq,coo_matrix((len(quotas),len(lo))))),b_eq=quotas*unit,
            bounds=(0.,None),method="highs")
        if min_deficit.success:
            missing = min_deficit.x[size:] / unit
            excess = min_deficit.x[:size] / unit
            assert np.max(np.abs(eq@excess-quotas)) < 1e-11
            assert np.max(np.maximum(0.,cap@excess-hi)) < 1e-11
            assert np.max(np.maximum(0.,lo-cap@excess-missing)) < 1e-11
            row["relaxed_lower_mass_deficit_kg"] = float(missing.sum())
            row["deficit_domains"] = [{"mass_underflow_kg":float(missing[d]),
                                       "region_ids":region_ids[allowed[d]].tolist()}
                                      for d in np.flatnonzero(missing>1e-11)]
            # The dual is a separate lower-bound witness for the omitted
            # distance graph, up to the solver feasibility tolerance.
            dual = min_deficit.ineqlin.marginals
            enforce = min_deficit.eqlin.marginals
            dual_gap = float((dual@np.r_[hi,-lo]) + enforce@quotas)
            assert abs(dual_gap-row["relaxed_lower_mass_deficit_kg"]) < 1e-10
            row["relaxed_dual_lower_bound_kg"] = dual_gap
            cut_ids = sorted({r for witness in row["deficit_domains"] for r in witness["region_ids"]})
            cut = np.isin(region_ids, cut_ids)
            constrained = ~np.any(allowed[:, ~cut], axis=1)
            subset_rows = rows[constrained[group_id]]
            demand = float(lo[constrained].sum())
            supply = float(quotas[cut].sum())
            assert abs(demand - supply - dual_gap) < 1e-10
            subset_points = projection.points[np.unique(projection.cells[subset_rows])]
            row["analytic_deficit_cut"] = {
                "region_ids": cut_ids, "minimum_required_kg": demand, "maximum_available_kg": supply,
                "shortage_kg": demand - supply, "constrained_cells": len(subset_rows),
                "region_quota_by_id_kg": dict(zip(map(str, region_ids[cut]), quotas[cut].tolist())),
                "min_world_position_m": subset_points.min(axis=0).tolist(),
                "max_world_position_m": subset_points.max(axis=0).tolist(),
            }
    else:
        allocation = model.x / unit
        row["maximum_carrier_region_residual_kg"] = float(np.max(np.abs(eq@allocation-quotas)))
        row["maximum_lower_mass_violation_kg"] = float(np.maximum(0.,lo-cap@allocation).max())
        row["maximum_upper_mass_violation_kg"] = float(np.maximum(0.,cap@allocation-hi).max())
    result[name] = row
(root/"paired-region-density-capacity-bound.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
