"""Route typed source mass while minimizing actual skeletal donor changes."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, hstack, vstack
from scipy.spatial import cKDTree

from audit import digest, rows
from extension import determinant
from map_mass import regional_context


def node_mass(cells, cell_mass, count):
    return np.bincount(cells.ravel(), weights=np.repeat(.25 * cell_mass, 4), minlength=count)


def map_balanced(run, shared, fitted, out):
    compiler, _, old_points, cells = regional_context(run, shared)
    with np.load(fitted / "posterior-interface-fit.npz") as candidate:
        points = candidate["nodes_reference"]
        assert np.array_equal(candidate["cells"], cells)
    with np.load(fitted / "mass.npz") as known:
        supported = np.c_[known["parent_cell"], known["target_cell"]]
    donors = np.array([int(r["node_frame_owner_body"])
                       for r in rows(run / "shell-mass.csv.nodes.csv")])
    groups = compiler.group_for_parent
    target = points[cells].mean(axis=1)
    changed = np.any(np.any(points != old_points, axis=1)[cells], axis=1)
    radius = .075
    touched_donors = np.unique(donors[cells[changed]])
    adjacent = ((cKDTree(target[changed]).query(target)[0] <= radius) &
                np.isin(compiler.material, compiler.material[changed]) &
                np.isin(donors[cells], touched_donors).any(axis=1))
    edit = np.isin(groups, np.unique(groups[changed])) | adjacent
    active = np.flatnonzero(edit)
    frozen = np.flatnonzero(~edit)
    selected = np.unique(groups[active])
    volume = determinant(points, cells) / 6.
    source_mass = compiler.source_kg
    lo, hi = compiler.density_limits.T * volume
    donor_rows = donors[cells[active]]
    relevant_donors = np.unique(donor_rows)
    node_indices = np.searchsorted(relevant_donors, donor_rows)
    k = len(active)
    node_map = coo_matrix((np.full(4*k, .25),
        (node_indices.ravel(), np.repeat(np.arange(k), 4))),
        shape=(len(relevant_donors), k)).tocsr()
    old_debits = node_map @ source_mass[active]
    sizes = [8, 24]
    solved = None
    for width in sizes:
        first = []
        second = []
        for g in selected:
            members = active[groups[active] == g]
            x, y = compiler.source_centroid[members], target[members]
            xs, ys = cKDTree(x), cKDTree(y)
            neighbors = np.arange(1, min(width, len(members)) + 1)
            edge1 = ys.query(x, k=neighbors)[1].reshape(len(members), -1)
            edge2 = xs.query(y, k=neighbors)[1].reshape(len(members), -1)
            first.extend((np.repeat(members, len(neighbors)), members[edge2.ravel()]))
            second.extend((members[edge1.ravel()], np.repeat(members, len(neighbors))))
        support = np.vstack((np.c_[np.concatenate(first), np.concatenate(second)],
                             supported[np.isin(supported[:, 0], active)]))
        a, b = np.unique(support, axis=0).T
        distance = np.linalg.norm(compiler.source_centroid[a] - target[b], axis=1)
        ok = distance <= radius
        a, b, distance = a[ok], b[ok], distance[ok]
        assert np.array_equal(groups[a], groups[b])
        columns = np.arange(len(a))
        ones = np.ones(len(a))
        src = coo_matrix((ones, (np.searchsorted(active, a), columns)), shape=(k, len(a))).tocsr()
        dst = coo_matrix((ones, (np.searchsorted(active, b), columns)), shape=(k, len(a))).tocsr()
        lower, upper = 1000.*lo[active], 1000.*hi[active]
        margin = np.minimum(.001*(upper-lower), 1e-8)
        # First minimize the maximum skeletal donor change, then find the
        # shortest feasible transport for that optimum. The nominal geometry
        # can impose a nonzero minimum even with exactly conserved cell pools.
        debit = node_map @ dst
        z_dst = coo_matrix((k, 1))
        z_debit = coo_matrix(-np.ones((len(old_debits), 1)))
        bound = vstack((hstack((dst, z_dst)), hstack((-dst, z_dst)),
                        hstack((debit, z_debit)), hstack((-debit, z_debit))), format="csr")
        rhs = np.r_[upper-margin, -lower-margin, 1000.*old_debits, -1000.*old_debits]
        equality = hstack((src, z_dst), format="csr")
        eq = 1000.*source_mass[active]
        options = {"time_limit": 12., "primal_feasibility_tolerance": 1e-9}
        first_pass = linprog(np.r_[np.zeros(len(a)), 1.], A_eq=equality, b_eq=eq, A_ub=bound,
                             b_ub=rhs, bounds=(0., None), method="highs", options=options)
        if not first_pass.success:
            print(first_pass.message, flush=True)
            continue
        max_debit_grams = float(first_pass.x[-1])
        solved = linprog(np.r_[distance/radius, 0.], A_eq=equality, b_eq=eq, A_ub=bound, b_ub=rhs,
                         bounds=[(0., None)]*len(a)+[(0., max_debit_grams+1e-8)],
                         method="highs", options=options)
        print(json.dumps({"neighbors": width, "edges": len(a), "status": int(solved.status),
                          "minimum_max_donor_change_kg": max_debit_grams/1000.,
                          "message": solved.message}), flush=True)
        if solved.success:
            break
    if solved is None or not solved.success:
        raise ValueError("No joint density- and route-feasible transport")
    keep = solved.x[:-1] > 0.
    a, b = np.r_[a[keep], frozen], np.r_[b[keep], frozen]
    weight = np.r_[solved.x[:-1][keep]/1000., source_mass[frozen]]
    cell_mass = np.bincount(b, weights=weight, minlength=len(cells))
    new_nodes = node_mass(cells, cell_mass, len(points))
    old_nodes = node_mass(cells, source_mass, len(points))
    delta = np.bincount(donors, weights=new_nodes-old_nodes)
    src_err = np.abs(np.bincount(a, weights=weight, minlength=len(cells))-source_mass).max()
    radius_max = np.linalg.norm(compiler.source_centroid[a] - target[b], axis=1).max()
    assert np.abs(delta).max() <= max_debit_grams/1000. + 3e-11 and src_err < 1e-12
    assert np.all(cell_mass / volume >= lo / volume-1e-5)
    assert np.all(cell_mass / volume <= hi / volume+1e-5)
    assert radius_max <= radius + 1e-14
    assert np.array_equal(groups[a], groups[b])
    with out.open("xb") as f:
        np.savez_compressed(f, parent_cell=a, target_cell=b, mass_kg=weight,
                           cell_mass_kg=cell_mass, node_mass_kg=new_nodes)
    print(json.dumps({"success": True, "routes": len(weight), "regional_pools": len(np.unique(groups)),
                      "source_error_kg": float(src_err), "total_mass_kg": float(cell_mass.sum()),
                      "maximum_nodal_donor_mass_error_kg": float(np.abs(delta).max()),
                      "maximum_route_m": float(radius_max), "output_sha256": digest(out)}))


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for name in ("run", "shared", "fitted", "out"):
        p.add_argument(name, type=Path)
    map_balanced(**vars(p.parse_args()))
