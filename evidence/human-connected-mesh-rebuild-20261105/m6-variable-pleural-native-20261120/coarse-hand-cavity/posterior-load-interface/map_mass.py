"""Conserve the existing typed source mass using the existing bounded transporter."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack
from scipy.sparse.csgraph import maximum_flow
from scipy.spatial import cKDTree

from audit import digest, rows
from extension import determinant


def regional_context(run, shared_tools):
    sys.path[:0] = [str(shared_tools), str(shared_tools / "thickness-rebuild-034")]
    from regional_mass import RegionalMass
    reference = rows(run / "shell-mass.csv.nodes.csv")
    points = np.array([[float(v[f"rest_{a}_m"]) for a in "xyz"] for v in reference])
    tet_rows = rows(run / "shell-mass.csv.tets.csv")
    labels = rows(run / "shell-mass.csv.labels.csv")
    cells = np.array([[int(v[f"parent_node{i}"]) for i in range(4)] for v in tet_rows])
    regions = np.array([int(r["anatomical_region_id"]) for r in labels])
    build = shared_tools / "exact-shell-bone-audit-032/build-commands.json"
    compiler = RegionalMass.from_original(points, cells, tet_rows, build)
    assert np.array_equal(compiler.material, [int(r["material_index"]) for r in labels])
    assert np.array_equal(compiler.donor, [int(r["geometric_donor_body"]) for r in labels])
    typed, compiler.group_for_parent = np.unique(
        np.c_[compiler.material, compiler.donor, regions], axis=0, return_inverse=True)
    return compiler, typed, points, cells


def feasible_graph_seed(source, target, source_grams, lower_grams, upper_grams, radius):
    """Use integer flow to find a small support for the continuous mass solve."""
    n = len(source)
    origin, compulsory, discretionary, sink = 2*n, 2*n+1, 2*n+2, 2*n+3
    scale = 1e9 / source_grams.sum()
    supply = np.rint(source_grams * scale).astype(np.int64)
    minimum = np.floor(lower_grams * scale).astype(np.int64)
    maximum = np.ceil(upper_grams * scale).astype(np.int64)
    total = int(supply.sum())
    required = int(minimum.sum())
    graph = cKDTree(source).sparse_distance_matrix(cKDTree(target), radius, output_type="ndarray")
    i, j = graph["i"], graph["j"]
    ids = np.arange(n)
    u = np.r_[np.full(n, origin), i, ids+n, ids+n, compulsory, discretionary]
    v = np.r_[ids, j+n, np.full(n, compulsory), np.full(n, discretionary), sink, sink]
    w = np.r_[supply, supply[i], minimum, maximum-minimum,
              required, max(total-required, 0)]
    network = coo_matrix((w, (u, v)), shape=(2*n+4, 2*n+4), dtype=np.int64).tocsr()
    flow = maximum_flow(network, origin, sink)
    block = flow.flow[:n, n:2*n].tocoo()
    keep = block.data > 0
    return block.row[keep] * n + block.col[keep], {
        "full_radius_edges": len(i), "routed_seed_fraction": float(flow.flow_value / total),
        "seed_used_edges": int(keep.sum()),
    }


def flow_by_group(compiler, points, cells, determinants, changed, radius):
    """Find a local minimum-distance transport on bounded support graphs."""
    groups = compiler.group_for_parent
    active_groups = np.unique(groups[changed])
    unchanged = np.flatnonzero(~np.isin(groups, active_groups))
    src, dst, amount = [unchanged], [unchanged], [compiler.source_kg[unchanged]]
    centers = points[cells].mean(axis=1)
    volumes = determinants / 6.
    receipts = []
    for g in active_groups:
        ids = np.flatnonzero(groups == g)
        count = len(ids)
        source = compiler.source_centroid[ids]
        target = centers[ids]
        old_grams = 1e3 * compiler.source_kg[ids]
        low, high = (compiler.density_limits[ids] * 1e3 * volumes[ids, None]).T
        margins = np.minimum((high - low) * .001, 1e-8)
        src_tree, dst_tree = cKDTree(source), cKDTree(target)
        seed, seed_report = feasible_graph_seed(source, target, old_grams, low, high, radius)
        attempts = []
        for width in sorted({min(count, 8), min(count, 24)}):
            keys = np.unique(np.r_[seed, np.arange(count) * (count + 1)])
            count_ids = np.arange(count)
            neighbors = np.arange(1, width + 1)
            forward = dst_tree.query(source, k=neighbors)[1].reshape(count, width)
            reverse = src_tree.query(target, k=neighbors)[1].reshape(count, width)
            keys = np.unique(np.r_[keys, np.repeat(count_ids, width) * count + forward.ravel(),
                                   reverse.ravel() * count + np.repeat(count_ids, width)])
            i, j = np.divmod(keys, count)
            dist = np.linalg.norm(source[i] - target[j], axis=1)
            keep = dist <= radius
            i, j, dist = i[keep], j[keep], dist[keep]
            cols = np.arange(len(i))
            source_rows = coo_matrix((np.ones(len(cols)), (i, cols)), shape=(count, len(cols))).tocsr()
            target_rows = coo_matrix((np.ones(len(cols)), (j, cols)), shape=(count, len(cols))).tocsr()
            solved = linprog(dist / radius, A_eq=source_rows, b_eq=old_grams,
                             A_ub=vstack((target_rows, -target_rows), format="csr"),
                             b_ub=np.r_[high - margins, -low - margins], method="highs",
                             options={"time_limit": 12., "primal_feasibility_tolerance": 1e-9})
            attempts.append({"neighbors": width, "route_count": len(i), "status": int(solved.status),
                             "message": solved.message})
            if solved.success:
                break
        receipts.append({"pool": int(g), "cell_count": count, "seed": seed_report, "attempts": attempts})
        if not solved.success:
            return None, None, {"success": False, "pools": receipts}
        mask = solved.x > 0.
        src.append(ids[i[mask]])
        dst.append(ids[j[mask]])
        amount.append(solved.x[mask] / 1e3)
    a, b, w = map(np.concatenate, (src, dst, amount))
    mass = np.bincount(b, weights=w, minlength=len(volumes))
    residual = np.bincount(a, weights=w, minlength=len(volumes)) - compiler.source_kg
    density = mass / volumes
    routes = np.linalg.norm(compiler.source_centroid[a] - centers[b], axis=1)
    low, high = compiler.density_limits.T
    assert np.max(np.abs(residual)) < 1e-12
    assert np.all(density >= low - 1e-5) and np.all(density <= high + 1e-5)
    assert np.max(routes) <= radius + 1e-14
    assert np.array_equal(groups[a], groups[b])
    return mass, (a, b, w), {
        "success": True, "pools": receipts, "positive_routes": len(w),
        "maximum_source_mass_error_kg": float(np.max(np.abs(residual))),
        "maximum_source_target_distance_m": float(routes.max()),
        "rerouted_mass_kg": float(w[a != b].sum()),
        "total_mass_distance_kg_m": float(w @ routes),
        "density_bounds_by_material_kg_m3": {
            str(m): [float(density[compiler.material == m].min()), float(density[compiler.material == m].max())]
            for m in np.unique(compiler.material)
        },
        "reference_mass_centroid_shift_m": (
            (mass @ centers - compiler.source_kg @ compiler.source_centroid) / mass.sum()).tolist(),
    }


def check(run, shared_tools, candidate, emitted=None, exact=False):
    compiler, typed, before, cells = regional_context(run, shared_tools)
    with np.load(candidate) as data:
        points = data["nodes_reference"]
        assert np.array_equal(data["cells"], cells)
    changed = np.any(points != before, axis=1)[cells].any(axis=1)
    assert changed.any()
    d = determinant(points, cells)
    assert d.min() > 0.
    lo, hi = compiler.determinant_bounds()
    group = compiler.group_for_parent
    pool = np.bincount(group, weights=d)
    source_pool = np.bincount(group, weights=compiler.source_kg)
    deficit = source_pool * (pool / hi - 1.)
    surplus = source_pool * (1. - pool / lo)
    invalid = (deficit > 1e-9) | (surplus > 1e-9)
    report = {
        "scope": "Same material, anatomical region and rigid donor, dosed by original native source quotas and existing density bounds. Source masses are unchanged.",
        "launch_sha256": digest(run / "launch.json"), "candidate_sha256": digest(candidate),
        "affected_typed_pools": int(len(np.unique(group[changed]))), "all_pools": len(typed),
        "minimum_mass_deficit_kg": float(np.maximum(deficit, 0.).sum()),
        "minimum_mass_surplus_kg": float(np.maximum(surplus, 0.).sum()),
        "witnesses": [{"material_donor_region": typed[i].tolist(),
                       "shortage_kg": float(max(0., deficit[i])),
                       "excess_kg": float(max(0., surplus[i]))}
                      for i in np.flatnonzero(invalid)],
    }
    if not invalid.any():
        if exact:
            from regional_mass import RADIUS_M
            mass, flow, closure = flow_by_group(compiler, points, cells, d, changed, RADIUS_M)
        else:
            mass, flow, closure = compiler.allocate(points, cells, np.arange(len(d)), d, changed)
        report["transport"] = closure
        if flow is not None and emitted is not None:
            a, b, amount = flow
            nodal = np.bincount(cells.ravel(), weights=np.repeat(mass / 4., 4), minlength=len(points))
            assert abs(nodal.sum() - compiler.source_kg.sum()) < 1e-10
            with emitted.open("xb") as f:
                np.savez_compressed(f, parent_cell=a, target_cell=b, mass_kg=amount,
                                    cell_mass_kg=mass, node_mass_kg=nodal)
            report["mapped_mass_sha256"] = digest(emitted)
            report["total_mapped_mass_kg"] = float(nodal.sum())
    return report


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("shared_tools", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mass", type=Path)
    parser.add_argument("--solve-transport", action="store_true")
    a = parser.parse_args()
    result = check(a.run, a.shared_tools, a.candidate, a.mass, a.solve_transport)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "witnesses"}, indent=2))


if __name__ == "__main__":
    main()
