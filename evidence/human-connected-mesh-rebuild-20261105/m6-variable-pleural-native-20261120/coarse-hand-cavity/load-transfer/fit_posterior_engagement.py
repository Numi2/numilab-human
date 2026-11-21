"""Fit deep fat toward registered posterior interfaces with unchanged outer skin."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from audit import BONE, SHELL, accepted_pose, digest, fem_boundaries, muscle_geometry, records, rows
from audit_bed_layer_surfaces import audit, role_surfaces
from balance_transfer import node_mass
from check_candidate import targets
from extension import determinant, quality, signed_height_gradients
from fit_interface import Constraints, separation_rows
from fit_muscle_interface import exact_pairs, prepare_target
from map_mass import regional_context


def fit_constraints(base, ref, cells, near, free, compiler, fixed_cell_mass):
    c = Constraints(free)
    for frame in (base, ref):
        d0 = determinant(frame, cells[near])
        gradient = signed_height_gradients(frame, cells[near]) / (1000. * d0[:, None])
        for tet, slope in zip(cells[near], gradient, strict=True):
            c.add(tet, -slope, .55)
    if fixed_cell_mass:
        d = determinant(ref, cells[near])
        slopes = signed_height_gradients(ref, cells[near]) / (1000. * d[:, None])
        density_min, density_max = compiler.density_limits[near].T
        six_mass = 6. * compiler.source_kg[near]
        lower = six_mass / (density_max * d)
        upper = six_mass / (density_min * d)
        assert lower.max() < 1. + 1.e-8 and upper.min() > 1. - 1.e-8
        for ids, slope, lo, hi in zip(cells[near], slopes, lower, upper, strict=True):
            c.add(ids, slope, max(0., hi - 1. - 5.e-8))
            c.add(ids, -slope, max(0., 1. - lo - 5.e-8))
        return c
    pool = compiler.group_for_parent
    volume = np.bincount(pool, weights=determinant(ref, cells))
    lo, hi = compiler.determinant_bounds()
    g = signed_height_gradients(ref, cells[near])
    for group in np.unique(pool[near]):
        use = pool[near] == group
        d = volume[group]
        delta = g[use].ravel() / (1000. * d)
        ids = cells[near][use].ravel()
        c.add(ids, delta, max(0., (hi[group] - d) / d - 1.e-7))
        c.add(ids, -delta, max(0., (d - lo[group]) / d - 1.e-7))
    return c


def prevent_new_crossings(c, mesh, faces, domain, travel):
    total = 0
    for face in faces:
        triangle = mesh[face]
        lower = triangle.min(axis=0) - [0., 0., travel]
        upper = triangle.max(axis=0) + [0., 0., travel]
        for points, tri_lo, tri_hi in domain.values():
            visible = np.flatnonzero(np.all((tri_lo <= upper) & (tri_hi >= lower), axis=1))
            for j in visible:
                interval = separation_rows(triangle, points[j])
                if interval is None:
                    continue
                weights, gaps = interval
                if np.min(gaps) <= 0. <= np.max(gaps):
                    continue  # Witness exists at baseline and is audited independently.
                sign = 1. if gaps[0] > 0. else -1.
                for w, g in zip(weights, gaps, strict=True):
                    margin = min(.0001, .5 * abs(g))
                    c.add(face, w * sign, 1000. * (abs(g) - margin))
                    total += 1
    return total


def optimize(c, free, chosen, travel):
    closure_count = len(chosen)
    max_area = max(r["projected_area_m2"] for r in chosen)
    for i, r in enumerate(chosen):
        c.add(r["fat_nodes"], [-1. / 3.] * 3, 0., i)
        # Constraints.add subtracts a relaxation variable. Here it is the
        # guaranteed closure, so move it to the other side of the inequality.
        assert c.cols[-1] == len(free) + i
        c.values[-1] = 1.
        c.add(r["fat_nodes"], [1. / 3.] * 3, 1000. * (r["fat_muscle_gap_m"] - .0001))
    for i, node in enumerate(free):
        c.add([node], [1.], 0., closure_count + i)
        c.add([node], [-1.], 0., closure_count + i)
    cost = np.r_[np.zeros(len(free)),
                 [-r["projected_area_m2"] / max_area for r in chosen],
                 np.full(len(free), .003)]
    bounds = ([(-1000. * travel, 1000. * travel)] * len(free)
              + [(0., 1000. * (r["fat_muscle_gap_m"] - .0001)) for r in chosen]
              + [(0., None)] * len(free))
    matrix = coo_matrix((c.values, (c.rows, c.cols)),
                        shape=(len(c.rhs), len(cost))).tocsr()
    fit = linprog(cost, A_ub=matrix, b_ub=c.rhs, bounds=bounds, method="highs",
                  options={"time_limit": 12.})
    assert fit.success, fit.message
    return fit.x[:len(free)] * .001


def main():
    p = ArgumentParser(description=__doc__)
    for name in ("run", "routes", "bone", "partition", "reader", "shared", "output"):
        p.add_argument(name, type=Path)
    p.add_argument("--also-audit-installed-muscles", action="store_true")
    p.add_argument("--preserve-cell-masses", action="store_true")
    a = p.parse_args()
    a.output.mkdir()
    install_for_float32()
    data = json.loads(a.routes.read_text())
    available = [r for r in data["routes"] if r["muscle_bone_gap_m"] <= .001]
    chosen = [r for r in available if .0001 < r["fat_muscle_gap_m"] <= .006]
    assert chosen
    compiler, _, ref, cells = regional_context(a.run, a.shared)
    role = role_surfaces(rows(a.run / "shell-mass.csv.tets.csv"))
    log = a.run / "stdout.log"
    start, anchors, objects, evidence = records(log)
    registration = json.loads(evidence["resting_initial_anatomical_registration"].partition("=")[2])
    base = start[np.unique(objects[SHELL])]
    assert len(base) == len(ref)
    n = len(base) // 3
    moves = np.unique([r["fat_nodes"] for r in chosen])
    active = np.array([f for group in role.values() for f in group])
    one_ring = np.unique(active[np.isin(active, moves).any(axis=1)])
    anchor_nodes = set()
    with log.open() as stream:
        for line in stream:
            if line.startswith("whole_body_skin_attachment "):
                anchor_nodes.add(2 * n + int(line.partition(" local_vertex=")[2].split()[0]))
    assert len(anchor_nodes) == 14
    free = np.array([i for i in one_ring if i >= 2 * n and i not in anchor_nodes])
    affected_faces = active[np.isin(active, free).any(axis=1)]
    near = np.flatnonzero(np.isin(cells, free).any(axis=1))

    bones = np.memmap(a.bone, dtype=BONE, mode="r")
    pose, _ = accepted_pose(a.run / "native/resting-body-motion-partition.csv", registration)
    muscles, _, source_hash, _, starts = muscle_geometry(
        json.loads((a.run / "launch.json").read_text()), pose, a.reader)
    offsets = sorted(starts.values()) + [len(muscles)]
    others = np.ones(len(muscles), dtype=bool)
    additions = {}
    for bank in (29, 30):
        path = a.partition / f"gluteal-{bank}.npz"
        assert digest(path) == data["gluteal_mesh_sha256"][str(bank)]
        begin = starts[bank]
        others[begin:offsets[offsets.index(begin) + 1]] = False
        with np.load(path) as m:
            additions[f"gluteus-{bank}"] = m["source_nodes"][m["source_faces"]].astype(float)
    fem, _ = fem_boundaries(start, objects, anchors)
    scene = dict(bones=bones["vertices"], muscles=muscles[others], other_fem=fem, **additions)
    if a.also_audit_installed_muscles:
        scene["installed-gluteus"] = muscles[~others]
    travel = .004
    pad = np.array([1.e-7, 1.e-7, travel + 1.e-7])
    lo = base[affected_faces].min(axis=(0, 1)) - pad
    hi = base[affected_faces].max(axis=(0, 1)) + pad
    prepared = {name: (ids, prepare_target(points, ids)) for name, points in scene.items()
                if len(ids := targets(points, (lo, hi)))}
    local_domain = {}
    for name, (ids, _) in prepared.items():
        listed = scene[name][ids]
        local_domain[name] = listed, listed.min(axis=1), listed.max(axis=1)
    start_pairs = exact_pairs(base.astype(np.float32), affected_faces, prepared)
    touched = np.isin(np.arange(len(base)), free)
    start_shell = audit(base, role, touched)[0]
    assert not start_shell
    constraints = fit_constraints(
        base, ref, cells, near, free, compiler, a.preserve_cell_masses)
    barriers = prevent_new_crossings(constraints, base, affected_faces, local_domain, travel)
    delta = optimize(constraints, free, chosen, travel)
    reports = []
    for step in (1., .75, .5, .25, .125, .0625):
        candidate, reference = base.copy(), ref.copy()
        candidate[free, 2] += step * delta
        reference[free, 2] += step * delta
        candidate = candidate.astype(np.float32)
        det = determinant(candidate.astype(float), cells)
        rdet = determinant(reference, cells)
        q = min(quality(candidate.astype(float), cells, det).min(),
                quality(reference, cells, rdet).min())
        pairs = exact_pairs(candidate, affected_faces, prepared)
        density = 6. * compiler.source_kg / rdet
        density_ok = (density >= compiler.density_limits[:, 0] - 1.e-5).all() and (
                      density <= compiler.density_limits[:, 1] + 1.e-5).all()
        introduced = pairs - start_pairs
        shells = audit(candidate.astype(float), role, touched)[0]
        advance = np.mean((candidate.astype(float) - base)[[r["fat_nodes"] for r in chosen], 2], axis=1)
        gaps = np.array([r["clearance_m"] for r in chosen])
        adjusted = gaps - advance
        report = dict(scale=step, minimum_quality=float(q),
                      maximum_delta_m=float(step * np.max(np.abs(delta))),
                      introduced_crossings=dict(Counter(group for group, _, _ in introduced)),
                      shell_crossings=sum(map(len, shells.values())),
                      original_cell_mass_within_density_bounds=bool(density_ok),
                      median_route_gap_reduction_m=float(np.median(advance)),
                      initial_engagement_under_2_mm=int((gaps < .002).sum()),
                      candidate_engagement_under_2_mm=int((adjusted < .002).sum()))
        reports.append(report)
        print(json.dumps(report), flush=True)
        if (not introduced and not shells and q >= 1e-5 and
                (density_ok or not a.preserve_cell_masses) and
                report["median_route_gap_reduction_m"] > 1.e-6):
            np.savez_compressed(a.output / "posterior-interface-fit.npz",
                                nodes_current=candidate, nodes_reference=reference,
                                cells=cells.astype(np.uint32))
            if a.preserve_cell_masses:
                old = np.arange(len(cells))
                source_mass = compiler.source_kg
                np.savez_compressed(a.output / "balanced-mass.npz",
                                    parent_cell=old, target_cell=old, mass_kg=source_mass,
                                    cell_mass_kg=source_mass,
                                    node_mass_kg=node_mass(cells, source_mass, len(base)))
            break
    result = dict(routes_sha256=digest(a.routes), bone_sha256=digest(a.bone),
                  nhtiss4_sha256=source_hash, candidate_routes=len(chosen),
                  eligible_distal_contacts=len(available), free_nodes=len(free),
                  frozen_elastic_sites=len(anchor_nodes), separation_constraints=barriers,
                  changed_face_scope=len(affected_faces),
                  installed_gluteal_audit=a.also_audit_installed_muscles,
                  preserved_original_cell_mass=a.preserve_cell_masses, trials=reports)
    with (a.output / "assessment.json").open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
