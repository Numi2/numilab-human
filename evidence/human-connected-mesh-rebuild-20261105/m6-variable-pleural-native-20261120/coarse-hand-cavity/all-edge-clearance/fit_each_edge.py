"""Fit spatially independent contact groups without changing tissue ownership."""
from argparse import ArgumentParser
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, vstack

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sparse_fit_feasibility
import sparse_fit_screened

from fit_hand_cavity import EDGES, PARENT, determinant, quality, table, digest
from fit_hand_interface import coordinates
from build_hand_trial import SOURCE

HERE = Path(__file__).resolve().parents[1]
ROLE_COUNT = 2411
MM = .001
ANCESTRY = SOURCE / "NumiHumanElbowBaselineProlongation065.hpp"


def refinement_edges():
    source = ANCESTRY.read_text()
    edges = np.array([tuple(map(int,row)) for row in
                      re.findall(r"\{\{(\d+)u,(\d+)u,(\d+)u\}\}",source)])
    assert edges.shape==(79,3)
    assert np.array_equal(edges[:,0],np.arange(2332,ROLE_COUNT))
    assert np.all(edges[:,1:]<2332)
    return edges


def fit_group(starting, topology, cell_material, constraints, columns, edges, fixed,
              polish=False, sparse=False, point_targets=None, warm_start=None, screen_linear_box=False):
    point_targets = point_targets or {}
    moved = np.sort(np.concatenate([columns + role * ROLE_COUNT for role in range(3)]))
    incident = np.flatnonzero(np.isin(topology, moved).any(axis=1))
    tets = topology[incident]
    seeds = np.array([local for local, *_ in constraints], dtype=int)
    directions = np.array([normal for _, normal, *_ in constraints]).reshape(-1, 3)
    distances = np.array([distance for _, _, distance, _ in constraints])
    assert sparse, "the all-edge constraints require the sparse solver"
    max_nodes, max_cells = 660, 3600
    if len(moved) > max_nodes or len(incident) > max_cells:
        return None, {"result":"requires local sparseness or a new volume patch",
                      "moved_columns":len(columns),"cells":len(incident)}
    row, corner, moving_row = np.nonzero(tets[:, :, None] == moved)
    shell_role_rows = np.searchsorted(moved, np.vstack([columns + i * ROLE_COUNT for i in range(3)]))
    a = np.searchsorted(moved, seeds)
    d0 = determinant(starting, tets)
    assert np.all(d0 > 0.)
    materials = cell_material[incident]
    weights = np.stack([np.where(materials == k, d0, 0.) for k in (15, 16)])
    layer_ref_volume_m3 = weights.sum(axis=1) / 6.
    weights /= weights.sum(axis=1, keepdims=True)
    span = len(moved) * 3
    ncols = len(columns)
    point_nodes = np.array(sorted(point_targets), dtype=int)
    point_indices = np.searchsorted(moved, point_nodes)
    assert np.array_equal(moved[point_indices], point_nodes)
    point_columns = (3 * point_indices[:, None] + np.arange(3)).ravel()
    point_jacobian = np.zeros((len(point_columns), span))
    point_jacobian[np.arange(len(point_columns)), point_columns] = 1.
    point_target_mm = np.array([point_targets[i] for i in point_nodes]).ravel() / MM

    # Keep every explicit child vertex in the same reference-affine edge
    # partition of its parent. No remapping or barycentric tolerance change.
    included_edges = edges[np.isin(edges[:,0],columns)]
    linear = np.zeros((len(included_edges)*3, len(moved)))
    for role in range(3):
        for edge_i, edge in enumerate(included_edges):
            for coefficient, column in zip((1.,-.5,-.5),edge,strict=True):
                local = role * ROLE_COUNT + column
                index = np.searchsorted(moved,local)
                if index < len(moved) and moved[index]==local:
                    linear[role*len(included_edges)+edge_i,index]=coefficient
    match_partition = np.kron(linear,np.eye(3))
    depth_limits = np.repeat([[.50001, .50001], [2.49999, 9.99999]], ncols, axis=1)
    clearance_jac = np.zeros((len(seeds), len(moved), 3))
    clearance_jac[np.arange(len(seeds)), a] = directions
    clearance_jac = clearance_jac.reshape(len(seeds), span)
    changed_base = starting[tets]
    ea, eb = np.array(EDGES).T
    base_diameter = np.linalg.norm(changed_base[:, ea] - changed_base[:, eb], axis=2).max(axis=1)
    edge_incidence = np.eye(4)[ea] - np.eye(4)[eb]
    quality_design = np.full(len(tets), .89)
    base_emitted_quality = quality(starting.astype(np.float32).astype(np.float64), tets)

    def emitted_quality(x):
        candidate = starting.copy()
        candidate[moved] += x.reshape(-1,3) * MM
        emitted = candidate.astype(np.float32).astype(np.float64)
        return quality(emitted,tets) / base_emitted_quality


    axes = np.arange(3)
    gradient_rows = np.repeat(row, 3)
    gradient_cols = (3 * moving_row[:, None] + axes).ravel()
    depth_rows = np.repeat(np.arange(2 * ncols), 3)
    depth_outer = (3 * shell_role_rows[:-1].ravel()[:, None] + axes).ravel()
    depth_inner = (3 * shell_role_rows[1:].ravel()[:, None] + axes).ravel()
    depth_coords = (np.tile(depth_rows, 2), np.r_[depth_outer, depth_inner])
    shape_constraints = csr_matrix(np.vstack((match_partition, point_jacobian)))
    weights_sparse = csr_matrix(weights)
    clearance_sparse = csr_matrix(clearance_jac)

    def geometry(x, gradient=False):
        p = changed_base.copy()
        p[row, corner] += x.reshape(-1, 3)[moving_row] * MM
        u, v, w = (p[:, 1:] - p[:, :1]).transpose(1, 0, 2)
        u_cross = np.cross(v, w)
        normalized_det = np.einsum("ij,ij->i", u, u_cross) / d0
        edge_vectors = p[:, ea] - p[:, eb]
        lengths = np.linalg.norm(edge_vectors, axis=2)
        edge_shape = normalized_det[:, None] - quality_design[:, None] * (lengths / base_diameter[:, None])**3
        if not gradient:
            return normalized_det, edge_shape.T.ravel()

        v_cross, w_cross = np.cross(w, u), np.cross(u, v)
        derivatives = np.stack((-u_cross-v_cross-w_cross, u_cross, v_cross, w_cross), axis=1)
        det_grad = derivatives[row, corner] * (MM / d0[row, None])
        shape_scale = 3. * MM * quality_design / base_diameter**3
        edge_grads = (shape_scale[:, None] * lengths)[:, :, None] * edge_vectors
        edge_grads = (edge_grads[row] * edge_incidence.T[corner, :, None]).transpose(1, 0, 2)
        shape_values = det_grad[None, :, :] - edge_grads
        # Each edge gives an inequality algebraically equivalent to the old
        # max-edge criterion when intersected with the other five. Keeping all
        # six gradients gives the LP the correct branches at edge-length ties.
        block_rows = np.arange(len(ea))[:, None] * len(tets) + gradient_rows
        shape_grad = csr_matrix((shape_values.ravel(),
                                 (block_rows.ravel(), np.tile(gradient_cols, len(ea)))),
                                shape=(len(tets) * len(ea), span))
        det_grad = csr_matrix((det_grad.ravel(), (gradient_rows, gradient_cols)),
                              shape=(len(tets), span))
        return det_grad, shape_grad

    def depth(x, gradient=False):
        p = starting[moved] / MM + x.reshape(-1, 3)
        differences = (p[shell_role_rows[:-1]] - p[shell_role_rows[1:]]).reshape(-1, 3)
        lengths = np.linalg.norm(differences, axis=1)
        if not gradient:
            return lengths
        direction = (differences / np.maximum(lengths[:, None], 1.e-8)).ravel()
        return csr_matrix((np.r_[direction, -direction], depth_coords), shape=(2*ncols, span))

    def inequalities(x, target, gradient=False):
        jac, shapes = geometry(x, gradient)
        thick = depth(x, gradient)
        if gradient:
            return vstack((jac, thick, -thick, clearance_sparse, shapes), format="csr")
        return np.concatenate((jac - .92, thick - depth_limits[0],
                               depth_limits[1] - thick, clearance_jac @ x - distances * np.where(prior_full_clearance, 1., target), shapes))

    def volume(x, target, gradient=False):
        jac, _ = geometry(x, gradient)
        if gradient:
            return vstack((weights_sparse @ jac, shape_constraints), format="csr")
        return np.concatenate((weights @ (jac - 1.), match_partition @ x,
                               x[point_columns] - point_target_mm * target))

    x = (np.zeros(span) if warm_start is None else
         ((warm_start[moved] - starting[moved]) / MM).ravel())
    prior_full_clearance = clearance_jac @ x >= distances - 1.e-7

    # Check the actual assembled sparse derivatives against central
    # differences on the pinned mesh before submitting the first LP.
    rng = np.random.default_rng(4027)
    probe, push = rng.normal(0., .025, (2, span))
    eps = 1.e-4
    for values, jacobian in (
        (lambda q: inequalities(q, 1.), inequalities(probe, 1., True)),
        (lambda q: volume(q, 1.), volume(probe, 1., True)),
    ):
        numer = (values(probe + eps * push) - values(probe - eps * push)) / (2 * eps)
        analyt = jacobian @ push
        assert np.allclose(numer, analyt, atol=1.e-7, rtol=2.e-6), np.abs(numer - analyt).max()

    pinned = set(fixed)
    limit = max(3., np.abs(point_target_mm).max(initial=0.) + 0.25)
    # Include the shortest axis-aligned box that can reach each
    # measured tangent-plane witness. A fixed 3 mm box made several
    # exact witnesses infeasible before the tissue constraints applied.
    required_reach_mm = distances / np.abs(directions).sum(axis=1)
    limit = max(limit, float(required_reach_mm.max(initial=0.)) + .25)
    bounds = [(0., 0.) if row in pinned else (-limit, limit) for row in moved for _ in range(3)]
    assert np.all(np.abs(x) <= limit)
    assert all(np.all(x[3*i:3*i+3] == 0.) for i, node in enumerate(moved) if node in pinned)
    steps = []
    if point_targets:
        count = max(4, int(np.ceil(np.abs(point_target_mm).max()/0.27)))
        targets = np.linspace(1./count, 1., count)
    else:
        targets = (.25, .5, .75, .825, .9, .95, 1.)
    solver = sparse_fit_screened if screen_linear_box else sparse_fit_feasibility
    for target in targets:
        conditions = [
            {"type": "ineq", "fun": lambda x: inequalities(x, target),
             "jac": lambda x: inequalities(x, target, True)},
            {"type": "eq", "fun": lambda x: volume(x, target), "jac": lambda x: volume(x, target, True)},
        ]

        def solve(center, initial, maxiter):
            return solver.solve(initial,conditions,bounds,maxiter)

        def check(result, mode):
            margins = inequalities(result.x, target)
            lengths = (len(tets), 2*ncols, 2*ncols, len(seeds), len(tets)*len(ea))
            families = ("determinant", "minimum_thickness", "maximum_thickness",
                        "bone_clearance", "shape")
            split = np.split(margins, np.cumsum(lengths)[:-1])
            active = [(float(block.min()), family, int(block.argmin()))
                      for family, block in zip(families, split, strict=True) if block.size]
            minimum, family, local = min(active, key=lambda row: row[0])
            if family in ("determinant", "shape"):
                witness = {"parent_cell_row": int(incident[local % len(tets)])}
                if family == "shape":
                    witness["edge_corners"] = [int(ea[local // len(tets)]), int(eb[local // len(tets)])]
            elif family == "bone_clearance":
                seed, _, _, bank = constraints[local]
                witness = {"node": int(seed), "bank": [int(owner) for owner in bank]}
            else:
                witness = {"interface": "skin" if local < ncols else "fat",
                           "column": int(columns[local % ncols])}
            volume_error = volume(result.x, target)
            accepted = (minimum >= -2e-7 and np.max(np.abs(volume_error), initial=0.) <= 2e-8
                        and np.all(np.abs(volume_error[:2] * layer_ref_volume_m3) <= 5.e-13))
            print(f"seeds={seeds.tolist()} clearance={target} mode={mode} "
                  f"feasible={accepted} minimum={minimum:.4g} {result.message}", flush=True)
            steps.append({"fraction":target, "mode":mode, "minimum_constraint":minimum,
                          "minimum_constraint_family": family,
                          "minimum_constraint_witness": witness,
                          "volume_rel_error":float(np.max(np.abs(volume_error))),
                          "message":str(result.message)})
            return accepted, minimum, float(np.max(np.abs(volume_error)))

        result = solve(np.zeros(span), x, 60)
        accepted, minimum, volume_error = check(result, "minimum_shift")
        # A nearly feasible proposal can be repaired with a small correction
        # instead of continuing to minimize the original displacement.
        if polish and target == 1. and not accepted and minimum >= -1.e-3 and volume_error <= 1.e-5:
            result = solve(result.x.copy(), result.x, 50)
            accepted, _, _ = check(result, "feasibility_correction")
        if not accepted:
            break
        if polish and target == 1.:
            exact = emitted_quality(result.x)
            failing = exact < .88
            if failing.any() and exact.min() >= .85:
                # Strengthen only the Float32 witnesses. Continuous shape
                # estimates can overstate quality for nearly coplanar cells.
                quality_design[failing] += .88 - exact[failing] + .015
                tightened = solve(result.x.copy(), result.x, 50)
                improved, _, _ = check(tightened, "emission_correction")
                steps[-1]["design_witness_cells"] = incident[failing].tolist()
                steps[-1]["min_emitted_relative_quality"] = float(emitted_quality(tightened.x).min())
                if improved:
                    result = tightened
        x = result.x

    candidate = starting.copy()
    candidate[moved] += x.reshape(-1, 3) * MM
    changed = determinant(candidate, tets) - d0
    emitted = candidate.astype(np.float32).astype(np.float64)
    exact_q = quality(emitted, tets) / base_emitted_quality
    assert len(set(moved)) == len(moved)
    achieved = clearance_jac @ ((candidate[moved] - starting[moved]).ravel() / MM) / distances
    assert np.allclose(achieved, clearance_jac @ x / distances, atol=1.e-9, rtol=0.)
    point_error_mm = float(np.abs(x[point_columns] - point_target_mm).max(initial=0.))
    certified = (float(achieved.min(initial=1.)) >= .999999
                 and float(min(exact_q)) >= .88 and point_error_mm <= 2.e-8)
    report = {
        "solver_stages": steps, "moved_columns":len(columns), "complete_star_cells":len(incident),
        "min_emitted_relative_quality":float(min(exact_q)),
        "minimum_emitted_signed_determinant_m3":float(min(determinant(emitted,tets))),
        "minimum_authored_det_ratio":float(min(determinant(candidate,tets)/d0)),
        "layer_reference_volume_change_m3":[float(changed[materials==k].sum()/6) for k in (15,16)],
        "modeled_clearance":[{"row":int(k),"body":identity[0],"bone":identity[1],"fraction":float(f)}
                              for (k,_,_,identity),f in zip(constraints,achieved,strict=True)],
        "maximum_shift_mm":float(np.linalg.norm(x.reshape(-1,3),axis=1).max()),
        "fit_component_bound_mm":limit,
        "witnesses_outside_3mm_box":int((required_reach_mm>3.).sum()),
        "previously_cleared_modeled_witnesses":int(prior_full_clearance.sum()),
        **({"point_target_error_mm": point_error_mm} if point_targets else {}),
    }
    return (candidate if certified else None), report


def groups(topology, constraints, edges, extra_ring=False):
    # Merge stencils if their moving vertices share a cell. That makes
    # volume and quality checks for different solves independent.
    groups = []
    topology_columns = topology % ROLE_COUNT
    for constraint in constraints:
        node = constraint[0]
        columns = np.unique(topology[np.any(topology == node,axis=1)] % ROLE_COUNT)
        # A direct contact witness on a refinement node requires at least
        # one movable ancestor. Ancestor motion carries every descendant.
        edges_on_seeds = edges[edges[:,0]==node % ROLE_COUNT]
        columns = np.union1d(columns,edges_on_seeds[:,1:].ravel())
        descendants = edges[np.isin(edges[:,1:],columns).any(axis=1),0]
        columns = np.union1d(columns,descendants)
        if extra_ring:
            # Let the next connected shell ring distribute strain from the
            # obstructed one. Include every refinement midpoint whose parent
            # moves, then group all cells affected by that larger stencil.
            touched = np.isin(topology_columns,columns).any(axis=1)
            columns = np.unique(topology_columns[touched])
            descendants = edges[np.isin(edges[:,1:],columns).any(axis=1),0]
            columns = np.union1d(columns,descendants)
        cells = np.flatnonzero(np.isin(topology_columns,columns).any(axis=1))
        touched = set(cells)
        merged = [constraint]
        for index in range(len(groups)-1,-1,-1):
            other_cols, other_cells, other_constraints = groups[index]
            if touched.isdisjoint(other_cells):
                continue
            columns = np.union1d(columns,other_cols)
            touched.update(other_cells)
            merged.extend(other_constraints)
            del groups[index]
        groups.append((columns,touched,merged))
    return groups


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("witness",type=Path)
    parser.add_argument("starting_fit",type=Path)
    parser.add_argument("out",type=Path)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument("--bodies",type=int,nargs="+")
    selection.add_argument("--banks",nargs="+",help="Typed body/bone pair, e.g. 59/95")
    parser.add_argument("--pin",type=int,nargs="+",default=[],
                        help="Exact local volume nodes to retain")
    parser.add_argument("--reserve-root-rows",action="store_true",
                        help="Keep selected points in the old accepted finger repair")
    parser.add_argument("--extra-ring",action="store_true",
                        help="Move one additional connected ring at blocked patches")
    parser.add_argument("--polish",action="store_true",
                        help="Repair a bounded, near-feasible full-clearance proposal")
    parser.add_argument("--sparse-feasibility",action="store_true",
                        help="Use bounded sparse feasibility updates for one connected patch")
    parser.add_argument("--warm-start", type=Path,
                        help="Use a qualified offline fit as initial coordinates")
    parser.add_argument("--screen-linear-box", action="store_true")
    args=parser.parse_args()
    banks={tuple(map(int,bank.split("/"))) for bank in args.banks or ()}
    assert all(len(bank)==2 for bank in banks)
    node_path=PARENT.with_name(PARENT.name+".nodes.csv")
    cell_path=PARENT.with_name(PARENT.name+".tets.csv")
    ref=coordinates(table(node_path),"rest")
    cells=table(cell_path)
    topology=np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in cells])
    materials=np.array([int(row["material_index"]) for row in cells])
    candidate=ref.copy()
    predecessor=json.loads(args.starting_fit.read_text())
    for row in predecessor["candidate_nodes"]:
        candidate[row["local_node"]]+=row["world_delta_m"]
    warm = None
    if args.warm_start:
        warm = ref.copy()
        warm_report = json.loads(args.warm_start.read_text())
        assert warm_report["tangent_plane_fit_pass"]
        assert warm_report["previous_fit_sha256"] == digest(args.starting_fit)
        assert warm_report["closed_solid_witness_sha256"] == digest(args.witness)
        for row in warm_report["candidate_nodes"]:
            warm[row["local_node"]] += row["world_delta_m"]
    constraints=[]
    with args.witness.open(newline="") as src:
        for row in csv.DictReader(src):
            if row["inside_rays"]!="3" or row["ambiguous"]!="0":
                continue
            local=int(row["layer"])*ROLE_COUNT+int(row["row"])
            body,bone=int(row["body"]),int(row["bone"])
            if banks:
                if (body,bone) not in banks:
                    continue
            elif args.bodies is not None:
                if body not in args.bodies:
                    continue
            else:
                x,y,_=candidate[local]
                if not (abs(x)>.20 and -1.10<y<-.83):
                    continue
            normal=np.array([float(row[f"normal_{axis}"]) for axis in "xyz"])
            length=np.linalg.norm(normal)
            assert abs(length-1.)<1.e-5
            normal/=length
            constraints.append((local,normal,float(row["closest_distance_m"])/MM+.05,(body,bone)))
    assert constraints, "no matching exact witnesses"
    reports=[]
    assert all(0<=row<3*ROLE_COUNT for row in args.pin)
    fixed=set(args.pin)
    if args.reserve_root_rows:
        old=json.loads((HERE/"hand-clearance-fit.json").read_text())
        seeds={row[0] for row in constraints}
        fixed.update(row["local_node"] for row in old["candidate_nodes"]
                     if row["local_node"] not in seeds)
    assert not fixed.intersection(row[0] for row in constraints)
    edges=refinement_edges()
    for columns,_,these in groups(topology,constraints,edges,args.extra_ring):
        trial,report=fit_group(candidate,topology,materials,these,columns,edges,fixed,
                                 args.polish,args.sparse_feasibility,warm_start=warm,
                                 screen_linear_box=args.screen_linear_box)
        report["accepted_fit"]=trial is not None
        reports.append(report)
        if trial is not None:
            candidate=trial
    edge_delta = (candidate-ref).reshape(3,ROLE_COUNT,3)[:,edges]
    assert np.max(np.abs(edge_delta[:,:,0]-.5*(edge_delta[:,:,1]+edge_delta[:,:,2]))) < 2.e-11
    changed=np.flatnonzero(np.any(candidate!=ref,axis=1))
    report={
        "registered_parent_nodes_sha256":digest(node_path),
        "registered_parent_cells_sha256":digest(cell_path),
        "closed_solid_witness_sha256":digest(args.witness),
        "previous_fit_sha256":digest(args.starting_fit),
        **({"warm_fit_sha256":digest(args.warm_start)} if args.warm_start else {}),
        "refinement_ancestor_edges_sha256":digest(ANCESTRY),
        "selected_pair_count":len(constraints),
        "linear_box_screening":args.screen_linear_box,
        **({"extra_halo_ring":1} if args.extra_ring else {}),
        **({"pinned_rows":sorted(fixed)} if fixed else {}),
        "group_reports":reports,
        "tangent_plane_fit_pass":bool(any(r["accepted_fit"] for r in reports)),
        "scope":"Connected local fits constrained by exact bone point witnesses. Validate under current native Float32 geometry.",
        "candidate_nodes":[{"local_node":int(k),"source_world_m":ref[k].tolist(),
                            "world_delta_m":(candidate[k]-ref[k]).tolist()} for k in changed],
    }
    with args.out.open("x") as handle:
        json.dump(report,handle,indent=2,allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
