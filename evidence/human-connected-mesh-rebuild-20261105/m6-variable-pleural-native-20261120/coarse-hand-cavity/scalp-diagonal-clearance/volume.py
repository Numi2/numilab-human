"""Fit conservative tetrahedral stencils below a coherent surface diagonal change."""
from argparse import ArgumentParser
from collections import Counter
from itertools import combinations
from pathlib import Path
import json

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack

from audit import digest, rows
from audit_bed_motion_shell import terminal_positions
from cavity import LABEL, cell_shape, determinant, oriented_boundary
from map_cavity import overlap


def parity(triangle):
    a, b, c = map(int, triangle)
    return (-1) ** (int(a > b) + int(a > c) + int(b > c))


def replace_caps(old, old_caps, new_caps):
    border = oriented_boundary(old)
    for base, target in zip(old_caps, new_caps, strict=True):
        signs = [border.pop(tuple(sorted(face))) * parity(face) for face in base]
        if signs[0] != signs[1]:
            raise ValueError("inconsistent reference cap orientation")
        for face in target:
            key = tuple(sorted(face))
            if key in border:
                raise ValueError("alternative cap face already occupies cavity boundary")
            border[key] = signs[0] * parity(face)
    return border


def retetrahedralize(frames, cells, target, reserved):
    vertices = np.unique(cells).tolist()
    options = np.array([tet for tet in combinations(vertices, 4) if tet not in reserved])
    if not len(options):
        return None, "no_available_cells"
    d = np.stack([determinant(x, options) for x in frames])
    flip = d[0] < 0.
    options[flip] = options[flip][:, [0,2,1,3]]
    d[:,flip] *= -1.
    floor = np.array([cell_shape(x,cells,determinant(x,cells)).min() for x in frames])
    shape = np.stack([cell_shape(x,options,det) for x,det in zip(frames,d,strict=True)])
    valid = (d > 0.).all(axis=0) & (shape >= floor[:,None]).all(axis=0)
    options, shape = options[valid], shape[:,valid]
    n = len(options)
    if n == 0:
        return None, "all_choices_fail_shape"
    face_index = {f:i for i,f in enumerate(target)}
    row, col, val = [], [], []
    for i,tet in enumerate(options):
        for face, sign in oriented_boundary([tet]).items():
            row.append(face_index.setdefault(face,len(face_index)))
            col.append(i)
            val.append(sign)
    signed = coo_matrix((val,(row,col)), shape=(len(face_index),n)).tocsr()
    rhs = np.zeros(len(face_index))
    rhs[:len(target)] = list(target.values())
    limit = np.full(len(face_index),2.)
    limit[:len(target)] = 1.
    if np.any(np.diff(signed.indptr)[:len(target)] == 0):
        return None, "missing_admissible_boundary_face"
    signed_count = np.ones((1,n))
    constraint = LinearConstraint(vstack((signed,abs(signed),signed_count)),
        np.r_[rhs,np.zeros_like(rhs),len(cells)], np.r_[rhs,limit,len(cells)])
    result = milp(np.mean(floor[:,None]/shape,axis=0), integrality=np.ones(n),
                  bounds=Bounds(0.,1.), constraints=constraint, options={"time_limit":2.})
    if result.x is None:
        return None, "infeasible_or_timeout:" + str(result.status)
    replacement = options[result.x > .5]
    assert len(replacement) == len(cells) and oriented_boundary(replacement) == target
    old_total = np.array([determinant(x,cells).sum() for x in frames])
    new_total = np.array([determinant(x,replacement).sum() for x in frames])
    if np.any(np.abs(new_total/old_total-1.) > .02):
        return None, "cavity_volume_change"
    return replacement, None


def transfer(rest, old, new, previous_mass, density, allow_transport=False):
    used = rest[np.unique(old)]
    local = (rest-used.mean(axis=0))/np.ptp(used,axis=0).max()
    old_volume = determinant(local,old)/6.
    new_volume = determinant(local,new)/6.
    matrix = np.array([[overlap(local[a],local[b]) for b in new] for a in old])
    a, b = matrix.sum(axis=1)/old_volume, matrix.sum(axis=0)/new_volume
    coverage = float(min(a.min(),b.min()))
    assert max(a.max(),b.max()) <= 1.+2.e-8
    if allow_transport:
        local_distance = np.linalg.norm(local[old].mean(axis=1)[:, None] - local[new].mean(axis=1), axis=2)
        if np.max(local_distance) * np.ptp(used,axis=0).max() > .075:
            return None
        # Minimize transfer away from known geometric overlap in a six-cell
        # stencil. Feasibility is the same bounded density/source-mass ledger.
        source, destination = np.repeat(np.eye(len(old)),len(new),axis=1), np.tile(np.eye(len(new)),len(old))
        allocated = previous_mass * 1.e3
        limits = np.asarray(density)[:,None] * determinant(rest,new)[None,:] / 6. * 1.e3
        cost = local_distance + (matrix == 0.)
        solved = linprog(cost.ravel(), A_eq=source, b_eq=allocated,
                         A_ub=np.r_[destination,-destination], b_ub=np.r_[limits[1],-limits[0]],
                         method="highs", options={"primal_feasibility_tolerance":1.e-10})
        if not solved.success:
            return None
        flow = solved.x.reshape(len(old),len(new))
        np.testing.assert_allclose(flow.sum(axis=1),allocated,rtol=2.e-10,atol=1.e-13)
        return flow/allocated[:,None], coverage
    if coverage < .93:
        return None
    return matrix/matrix.sum(axis=1)[:,None], coverage


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("run","screen","mass","output"):
        p.add_argument(name,type=Path)
    p.add_argument("--bounded-mass-transport", action="store_true")
    a = p.parse_args()
    nodes = a.run / "shell-mass.csv.nodes.csv"
    tets = a.run / "shell-mass.csv.tets.csv"
    labels = a.run / "shell-mass.csv.labels.csv"
    nodedata = rows(nodes)
    rest = np.array([[r[f"rest_{k}_m"] for k in "xyz"] for r in nodedata],float)
    x = np.array([[r[f"current_{k}_m"] for k in "xyz"] for r in nodedata],np.float32)
    end = terminal_positions(a.run/"stdout.log",x)
    frames = (rest,x.astype(float),end.astype(float))
    cells = np.array([[int(r[f"parent_node{k}"]) for k in range(4)] for r in rows(tets)])
    typed = [tuple(r[k] for k in LABEL) for r in rows(labels)]
    owners = np.array([int(r[0]) for r in typed])
    n = len(x)//3
    inventory = {tuple(sorted(row)) for row in cells}
    screen = json.loads(a.screen.read_text())
    assert screen["inputs_sha256"][tets.name] == digest(tets)
    with np.load(a.mass) as data:
        src, dst, w, cell_mass = (data[k] for k in ("parent_cell","target_cell","mass_kg","cell_mass_kg"))
    # Use the installed material files' existing bounds (read below).
    build = Path(json.loads((a.run/"launch.json").read_text())["argv"][0]).parent
    recipe = json.loads((build/"build-commands.json").read_text())[0]
    density = {}
    for material, flag in ((15,"SKIN_DERMIS"),(16,"SUBCUTANEOUS_ADIPOSE")):
        source = Path(next(line.split("=",1)[1].strip('"') for line in recipe
                           if line.startswith("-DNUMI_HUMAN_"+flag+"_MATERIAL=")))
        import re
        match = re.search(r"parameter\s+density\b[^\n]*?in\s*\[\s*([.\d]+)\s*,\s*([.\d]+)\s*\]",source.read_text())
        if match is None:
            raise ValueError(f"unsupported native density declaration: {source}")
        density[material] = tuple(map(float,match.groups()))
    variants = []
    reasons = Counter()
    for option in screen["all_surface_candidates"]:
        before, after = (np.asarray(option[k]).reshape(3,2,3) for k in ("before","after"))
        column = np.asarray(option["columns"])
        selected = np.all(np.isin(cells%n,column),axis=1)
        patches = []
        for side, material in enumerate((15,16)):
            retired = np.flatnonzero(selected & (owners == material))
            if len(retired) != 6 or len({typed[r] for r in retired}) != 1:
                reasons["mixed_bank_or_refined_cell_footprint"] += 1
                break
            old = cells[retired]
            target = replace_caps(old,before[side:side+2],after[side:side+2])
            new, reason = retetrahedralize(frames,old,target,inventory-{tuple(sorted(r)) for r in old})
            if reason:
                reasons[reason] += 1
                break
            route = transfer(rest,old,new,cell_mass[retired],density[material],a.bounded_mass_transport)
            if route is None:
                reasons["transport_infeasible" if a.bounded_mass_transport else "insufficient_original_volume_overlap"] += 1
                break
            route, fraction = route
            mass = cell_mass[retired] @ route
            d = mass*6./determinant(rest,new)
            lo,hi = density[material]
            if np.any((d<lo-1.e-5)|(d>hi+1.e-5)):
                reasons["density_range"] += 1
                break
            patches.append({"retired": retired.tolist(),"original":old.tolist(),
                "replacement":new.tolist(),"typed_bank":list(typed[retired[0]]),
                "overlap_transfer":route.tolist(),"minimum_covered_volume_fraction":fraction,
                "receiving_density_kg_m3":d.tolist()})
        if len(patches) == 2:
            variants.append({**option,"cavities":patches})
    picked, used = [],set()
    for option in variants:
        if used.isdisjoint(option["columns"]):
            used.update(option["columns"])
            picked.append(option)
    candidate = cells.copy()
    selected_rows = []
    mass_s, mass_d, mass_w = [],[],[]
    for option in picked:
        for cavity in option["cavities"]:
            retired = np.array(cavity["retired"])
            selected_rows.extend(retired.tolist())
            candidate[retired] = cavity["replacement"]
            choose = np.isin(dst, retired)
            fractions = np.array(cavity["overlap_transfer"])[np.searchsorted(retired,dst[choose])]
            moved = w[choose,None]*fractions
            nz, col = np.nonzero(moved)
            mass_s.extend(src[choose][nz].tolist())
            mass_d.extend(retired[col].tolist())
            mass_w.extend(moved[nz,col].tolist())
    preserve = ~np.isin(dst, selected_rows)
    s, t, weight = np.r_[src[preserve],np.array(mass_s,int)], np.r_[dst[preserve],np.array(mass_d,int)], np.r_[w[preserve],mass_w]
    per_source = np.bincount(s,weights=weight)
    original = np.bincount(src,weights=w)
    np.testing.assert_allclose(per_source,original,rtol=2.e-12,atol=1.e-15)
    mapped = np.bincount(t,weights=weight,minlength=len(cells))
    assert len({tuple(sorted(cell)) for cell in candidate}) == len(cells)
    new_floor = [float(cell_shape(f,candidate,determinant(f,candidate)).min()) for f in frames]
    assert min(new_floor) > 0.
    a.output.mkdir()
    np.savez_compressed(a.output/"candidate.npz", cells=candidate, original_cells=cells, changed_rows=sorted(selected_rows),
                        parent_cell=s, target_cell=t, mass_kg=weight, cell_mass_kg=mapped)
    report = {"source_sha256": {path.name:digest(path) for path in (nodes,tets,labels,a.screen,a.mass)},
              "candidates_with_both_volume_cavities":len(variants), "rejections":dict(reasons),
              "selected":picked,"minimum_shape_by_frame":new_floor, "total_mapped_mass_kg":float(mapped.sum()),
              "maximum_source_mass_error_kg":float(np.abs(per_source-original).max()),
              "bounded_mass_transport": a.bounded_mass_transport, "selected_pair_gain":np.asarray([k["gains"] for k in picked],dtype=int).sum(axis=0).tolist()}
    with (a.output/"geometry.json").open("x") as f:
        json.dump(report,f,indent=2,allow_nan=False); f.write("\n")
    print({k:v for k,v in report.items() if k!="selected"}, "patches",len(picked))


if __name__ == "__main__":
    main()
