"""Fit spatially independent contact groups without changing tissue ownership."""
from argparse import ArgumentParser
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from fit_hand_cavity import EDGES, PARENT, determinant, quality, table, digest
from fit_hand_interface import coordinates
from build_hand_trial import SOURCE

HERE = Path(__file__).resolve().parent
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


def fit_group(starting, topology, cell_material, constraints, columns, edges):
    moved = np.sort(np.concatenate([columns + role * ROLE_COUNT for role in range(3)]))
    incident = np.flatnonzero(np.isin(topology, moved).any(axis=1))
    tets = topology[incident]
    seeds = np.array([local for local, *_ in constraints])
    directions = np.array([normal for _, normal, *_ in constraints])
    distances = np.array([distance for _, _, distance, _ in constraints])
    if len(moved) > 240 or len(incident) > 2400:
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

    def shape(x, grad=False):
        p = changed_base.copy()
        p[row, corner] += x.reshape(-1, 3)[moving_row] * MM
        edges = p[:, ea] - p[:, eb]
        size = np.linalg.norm(edges, axis=2)
        worst = np.argmax(size,axis=1)
        active = np.arange(len(tets))
        ratio = size[active,worst] / base_diameter
        if not grad:
            return evaluate(x) - .89 * ratio**3
        slope = (3. * MM / base_diameter**3) * size[active,worst]
        deriv = slope[:,None] * edges[active,worst]
        jac = np.zeros((len(tets), len(moved), 3))
        jac[row, moving_row] = edge_incidence[worst[row],corner,None] * deriv[row]
        return evaluate(x,True) - .89 * jac.reshape(len(tets),span)

    def evaluate(x, grad=False):
        delta = x.reshape(-1, 3)
        p = changed_base.copy()
        p[row, corner] += delta[moving_row] * MM
        u, v, w = (p[:, 1:] - p[:, :1]).transpose(1, 0, 2)
        u_cross = np.cross(v, w)
        if not grad:
            return np.einsum("ij,ij->i", u, u_cross) / d0
        v_cross, w_cross = np.cross(w, u), np.cross(u, v)
        derivatives = np.stack((-u_cross-v_cross-w_cross, u_cross, v_cross, w_cross), axis=1)
        gradient = np.zeros((len(tets), len(moved), 3))
        gradient[row, moving_row] = derivatives[row, corner] * (MM / d0[row, None])
        return gradient.reshape(len(tets), span)

    def depth(x, grad=False):
        p = starting[moved] / MM + x.reshape(-1, 3)
        differences = (p[shell_role_rows[:-1]] - p[shell_role_rows[1:]]).reshape(-1, 3)
        lengths = np.linalg.norm(differences, axis=1)
        if not grad:
            return lengths
        direction = differences / np.maximum(lengths[:, None], 1.e-8)
        jac = np.zeros((2*ncols, len(moved), 3))
        r = np.arange(2*ncols)
        jac[r, shell_role_rows[:-1].ravel()] = direction
        jac[r, shell_role_rows[1:].ravel()] = -direction
        return jac.reshape(2*ncols, span)

    def inequalities(x, target, grad=False):
        jacs = evaluate(x, grad)
        thick = depth(x, grad)
        if grad:
            return np.concatenate((jacs, thick, -thick, clearance_jac, shape(x, True)))
        return np.concatenate((jacs - .92, thick - depth_limits[0],
                               depth_limits[1] - thick, clearance_jac @ x - distances * target, shape(x)))

    def volume(x, grad=False):
        if grad:
            return np.concatenate((weights @ evaluate(x, True),match_partition))
        return np.concatenate((weights @ (evaluate(x) - 1.),match_partition @ x))

    x = np.zeros(span)
    steps = []
    for target in (.25, .5, .75, 1.):
        result = minimize(
            lambda x: .5 * (x @ x), x, jac=lambda x: x,
            constraints=[
                {"type": "ineq", "fun": lambda x: inequalities(x, target),
                 "jac": lambda x: inequalities(x, target, True)},
                {"type": "eq", "fun": volume, "jac": lambda x: volume(x, True)},
            ],
            bounds=[(-3., 3.)] * span, method="SLSQP",
            options={"maxiter": 60, "ftol": 1e-11},
        )
        minimum = float(inequalities(result.x, target).min())
        print(f"seeds={seeds.tolist()} clearance={target} feasible={minimum>=-2e-7} minimum={minimum:.4g} {result.message}",
              flush=True)
        steps.append({"fraction":target,"minimum_constraint":minimum,"volume_rel_error":float(np.max(np.abs(volume(result.x)))),"message":result.message})
        if minimum < -2e-7 or np.max(np.abs(volume(result.x)),initial=0.)>2.e-8 or np.any(np.abs(volume(result.x)[:2] * layer_ref_volume_m3) > 5.e-13):
            break
        x = result.x

    candidate = starting.copy()
    candidate[moved] += x.reshape(-1, 3) * MM
    changed = determinant(candidate, tets) - d0
    emitted = candidate.astype(np.float32).astype(np.float64)
    before_f32 = starting.astype(np.float32).astype(np.float64)
    exact_q = quality(emitted, tets) / quality(before_f32, tets)
    assert len(set(moved)) == len(moved)
    achieved = clearance_jac @ ((candidate[moved] - starting[moved]).ravel() / MM) / distances
    assert np.allclose(achieved, clearance_jac @ x / distances, atol=1.e-9, rtol=0.)
    certified = float(min(achieved)) >= .999999 and float(min(exact_q)) >= .88
    report = {
        "solver_stages": steps, "moved_columns":len(columns), "complete_star_cells":len(incident),
        "min_emitted_relative_quality":float(min(exact_q)),
        "minimum_emitted_signed_determinant_m3":float(min(determinant(emitted,tets))),
        "minimum_authored_det_ratio":float(min(determinant(candidate,tets)/d0)),
        "layer_reference_volume_change_m3":[float(changed[materials==k].sum()/6) for k in (15,16)],
        "modeled_clearance":[{"row":int(k),"body":identity[0],"bone":identity[1],"fraction":float(f)}
                              for (k,_,_,identity),f in zip(constraints,achieved,strict=True)],
        "maximum_shift_mm":float(np.linalg.norm(x.reshape(-1,3),axis=1).max()),
    }
    return (candidate if certified else None), report


def groups(topology, constraints, edges):
    # Merge stencils if their moving vertices share a cell. That makes
    # volume and quality checks for different solves independent.
    groups = []
    for constraint in constraints:
        node = constraint[0]
        columns = np.unique(topology[np.any(topology == node,axis=1)] % ROLE_COUNT)
        # A direct contact witness on a refinement node requires at least
        # one movable ancestor. Ancestor motion carries every descendant.
        edges_on_seeds = edges[edges[:,0]==node % ROLE_COUNT]
        columns = np.union1d(columns,edges_on_seeds[:,1:].ravel())
        descendants = edges[np.isin(edges[:,1:],columns).any(axis=1),0]
        columns = np.union1d(columns,descendants)
        cells = np.flatnonzero(np.isin(topology % ROLE_COUNT,columns).any(axis=1))
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
    parser.add_argument("--bodies",type=int,nargs="+")
    args=parser.parse_args()
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
    constraints=[]
    with args.witness.open(newline="") as src:
        for row in csv.DictReader(src):
            if row["inside_rays"]!="3" or row["ambiguous"]!="0":
                continue
            local=int(row["layer"])*ROLE_COUNT+int(row["row"])
            body,bone=int(row["body"]),int(row["bone"])
            x,y,_=candidate[local]
            if args.bodies is not None:
                if body not in args.bodies:
                    continue
            elif not (abs(x)>.20 and -1.10<y<-.83):
                continue
            normal=np.array([float(row[f"normal_{axis}"]) for axis in "xyz"])
            length=np.linalg.norm(normal)
            assert abs(length-1.)<1.e-5
            normal/=length
            constraints.append((local,normal,float(row["closest_distance_m"])/MM+.05,(body,bone)))
    assert constraints, "no matching exact witnesses"
    reports=[]
    edges=refinement_edges()
    for columns,_,these in groups(topology,constraints,edges):
        trial,report=fit_group(candidate,topology,materials,these,columns,edges)
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
        "refinement_ancestor_edges_sha256":digest(ANCESTRY),
        "selected_pair_count":len(constraints),
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
