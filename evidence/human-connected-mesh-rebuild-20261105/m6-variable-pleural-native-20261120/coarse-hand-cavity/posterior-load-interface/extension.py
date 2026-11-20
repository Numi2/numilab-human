"""Build a posterior fat reference from typed muscle witnesses and screen feasibility."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from audit import SHELL, accepted_pose, contacts, digest, muscle_geometry, records, rows


def determinant(points, cells):
    corners = points[cells]
    return np.linalg.det(corners[:, 1:] - corners[:, :1])


def quality(points, cells, det):
    corners = points[cells]
    distance2 = np.sum((corners[:, :, None] - corners[:, None, :])**2, axis=-1)
    return det / np.max(distance2, axis=(1, 2))**1.5


def smooth_window(gap_m):
    phase = np.clip((gap_m - .005) / (.020 - .005), 0., 1.)
    return 1. - phase**2 * (3. - 2. * phase)


def signed_height_gradients(points, cells):
    """Exact derivative of a tetrahedron determinant with respect to z."""
    u, v, w = np.moveaxis(points[cells[:, 1:], :2] - points[cells[:, :1], :2], 1, 0)
    basis = np.stack((wedge(v, w), wedge(w, u), wedge(u, v)), axis=1)
    return np.c_[-basis.sum(axis=1), basis]


def wedge(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def overlap_xy(subject, clip):
    """The vertices of the common projected polygon."""
    orient = np.sign(wedge(clip[1] - clip[0], clip[2] - clip[0]))
    if orient == 0.:
        return []
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        next_polygon = []
        for u, v in zip(subject, np.roll(subject, -1, axis=0)):
            f = orient * wedge(b - a, u - a)
            g = orient * wedge(b - a, v - a)
            if (f < 0.) != (g < 0.):
                next_polygon.append(u + f / (f - g) * (v - u))
            if g >= 0.:
                next_polygon.append(v)
        if not next_polygon:
            return []
        subject = np.array(next_polygon)
    return subject


def barycentric(points, tri):
    uv = np.linalg.solve((tri[1:] - tri[0]).T, (points - tri[0]).T).T
    return np.c_[1. - uv.sum(axis=1), uv]


def clearance_cut(base, target):
    """Return increment limits that preserve vertical ordering on the overlap."""
    for tri in (base, target):
        a, b = tri[1:, :2] - tri[0, :2]
        if abs(wedge(a, b)) < 1e-14:
            return np.eye(3), np.zeros(3)
    overlap = overlap_xy(base[:, :2], target[:, :2])
    if len(overlap) == 0:
        return np.eye(3), np.zeros(3)
    weights = barycentric(overlap, base[:, :2])
    z0 = weights @ base[:, 2]
    z1 = barycentric(overlap, target[:, :2]) @ target[:, 2]
    gap = z1 - z0
    if np.all(gap <= -1e-9):
        return np.empty((0, 3)), np.empty(0)
    if np.min(gap) < 1e-9:
        return np.eye(3), np.zeros(3)
    # Preserve at least a small fraction of each proven separation and leave
    # 0.15 mm where the original gap allows it. Zero is always feasible.
    return np.maximum(weights, 0.), gap - np.minimum(.00015, .5 * gap)


def collision_rows(world, limit, id_table, bounds, cuts):
    data, rows, cols, right = [], [], [], []
    for face, target in cuts:
        ids = id_table[face]
        if not np.any(ids >= 0):
            continue
        coefficient, gaps = clearance_cut(world[face], target)
        for weight, cap in zip(coefficient, gaps, strict=True):
            row_number = len(right)
            for index, value in zip(ids, weight, strict=True):
                if index >= 0 and value > 0.:
                    data.append(value * bounds[index] / limit)
                    cols.append(index)
                    rows.append(row_number)
            right.append(cap / limit)
    return coo_matrix((data, (rows, cols)), shape=(len(right), len(bounds))), right


def fit_heights(world, reference, cells, local, moves, cuts=(), mass_compiler=None):
    nodes = np.flatnonzero(moves[:, 2] > 0.)
    id_table = np.full(len(world), -1, dtype=int)
    id_table[nodes] = np.arange(len(nodes))
    near = cells[local]
    ids = id_table[near]
    active = ids >= 0
    bounds = moves[nodes, 2]
    matrix = []
    minimum_ratio = .45
    for points in (world, reference):
        d0 = determinant(points, near)
        gradient = signed_height_gradients(points, near)
        active_ids = ids[active]
        normalized = (gradient / d0[:, None])[active] * bounds[active_ids]
        row = np.broadcast_to(np.arange(len(near))[:, None], active.shape)[active]
        matrix.append(coo_matrix((-normalized, (row, active_ids)), shape=(len(near), len(nodes))))
    # Use a dimensionless coordinate per node so the LP feasibility tolerance
    # is well above the Float32 rounding margin at open interfaces.
    upper = np.full(2 * len(near), 1. - minimum_ratio)
    if mass_compiler is not None:
        groups = mass_compiler.group_for_parent
        by_pool = np.unique(groups[local])
        rows = np.searchsorted(by_pool, groups[local])
        old = np.bincount(groups, weights=determinant(reference, cells))[by_pool]
        low, high = (x[by_pool] for x in mass_compiler.determinant_bounds())
        # Keep the existing source masses. This bound is necessary for any
        # later conservative transport at the compiled material density limits.
        # Reserve roundoff headroom inside nonempty admissible intervals.
        eps = 1e-7 * old
        low = np.minimum(low + eps, old)
        high = np.maximum(high - eps, old)
        grad = signed_height_gradients(reference, near) / old[rows, None]
        pool_rows = np.broadcast_to(rows[:, None], active.shape)[active]
        cols = ids[active]
        change = coo_matrix((grad[active] * bounds[cols], (pool_rows, cols)),
                            shape=(len(by_pool), len(bounds))).tocsr()
        matrix.extend((change, -change))
        upper = np.r_[upper, (high - old) / old, (old - low) / old]
    if cuts:
        system, limits = collision_rows(world, bounds.max(), id_table, bounds, cuts)
        matrix.append(system)
        upper = np.r_[upper, limits]
    solved = linprog(-bounds / bounds.max(), A_ub=vstack(matrix, format="csr"),
                    b_ub=upper,
                    bounds=(0., 1.), method="highs")
    if not solved.success:
        raise RuntimeError(solved.message)
    fitted = np.zeros_like(moves)
    fitted[nodes, 2] = bounds * solved.x
    trial = (world + fitted).astype(np.float32).astype(float)
    ratios = determinant(trial, near) / determinant(world, near)
    ref_ratios = determinant(reference + fitted, near) / determinant(reference, near)
    assert min(ratios.min(), ref_ratios.min()) >= minimum_ratio - .0001
    return fitted


def fit(run, screen, reader, emitted=None, exclusions=(), mass_tools=None):
    """Test registered deep-node displacements in both native geometry frames."""
    report = json.loads(screen.read_text())
    assert report["inputs_sha256"]["launch.json"] == digest(run / "launch.json")
    assert report["inputs_sha256"]["stdout.log"] == digest(run / "stdout.log")
    reference_rows = rows(run / "shell-mass.csv.nodes.csv")
    reference = np.array([[float(v[f"rest_{a}_m"]) for a in "xyz"] for v in reference_rows])
    world = np.array([[float(v[f"current_{a}_m"]) for a in "xyz"] for v in reference_rows])
    n, remainder = divmod(len(world), 3)
    assert not remainder and np.all(np.isfinite(world)) and np.all(np.isfinite(reference))
    log_points, _, native_cells, evidence = records(run / "stdout.log")
    assert np.array_equal(world, log_points[[int(v["compiled_global_node"]) for v in reference_rows]])
    tet_rows = rows(run / "shell-mass.csv.tets.csv")
    cells = np.array([[int(v[f"parent_node{i}"]) for i in range(4)] for v in tet_rows])
    assert len(native_cells[SHELL]) == len(cells)
    material = np.array([int(v["material_index"]) for v in tet_rows])
    mass = np.array([float(v["parent_mass_kg"]) for v in tet_rows])
    donor = np.array([int(v["donor_body"]) for v in tet_rows])
    skin_det, ref_det = (determinant(p, cells) for p in (world, reference))
    assert min(skin_det.min(), ref_det.min()) > 0

    reg = json.loads(evidence["resting_initial_anatomical_registration"].partition("=")[2])
    pose, _ = accepted_pose(run / "native/resting-body-motion-partition.csv", reg)
    launch = json.loads((run / "launch.json").read_text())
    muscles, identity, source_hash, _, banks = muscle_geometry(launch, pose, reader)
    assert source_hash == report["posed_muscle_source_sha256"]

    outer = world[:n]
    skin_inner = world[n:2*n]
    deep = world[2*n:]
    window = smooth_window(outer[:, 2] - report["bed_plane_offset_m"])
    known_gluteal = {tuple(v["outer_face_nodes"]) for v in report["samples"]
                     if (h := v["posed_muscle"]["first_above_deep"]) and
                     h["label"] in ("right gluteus maximus", "left gluteus maximus") and
                     not v["posed_muscle"]["deep_inside_welded_source_ids"]}
    nodes = np.unique([i for face in known_gluteal for i in face])
    nodes = nodes[(window[nodes] > 0) & (deep[nodes, 2] > outer[nodes, 2])]
    face_hits = contacts(deep[nodes, :2], muscles)
    z_displacement = np.zeros(n)
    anchors = []
    for node, hits in zip(nodes, face_hits, strict=True):
        below = [i for i, z in hits if outer[node, 2] <= z < deep[node, 2]]
        nearest = next(((i, z) for i, z in hits if z >= deep[node, 2]), None)
        if below or nearest is None:
            continue
        i, z = nearest
        source = identity(i)
        if source["label"] not in ("right gluteus maximus", "left gluteus maximus"):
            continue
        gap = z - deep[node, 2]
        if not .002 <= gap <= .035:
            continue
        z_displacement[node] = gap * window[node]
        anchors.append([int(node), source["stable_id"], source["source_bank_face"], gap])
    assert anchors
    moving = z_displacement != 0.
    moves = np.zeros_like(world)
    moves[2*n:, 2] = z_displacement
    assert not np.any(moves[:2*n])
    current_thickness = np.linalg.norm(deep - skin_inner, axis=1)
    layer_present = np.any((cells >= 2*n) & (moves[cells, 2] != 0.), axis=1)
    assert layer_present.any() and np.all(material[layer_present] == 16)

    clearance = .0003
    limited = np.maximum(moves - clearance * np.sign(moves), 0.)
    cuts = {}
    for receipt in exclusions:
        previous = json.loads(receipt.read_text())
        assert previous["muscle_sha256"] == source_hash
        assert previous["shell_startup_inputs_sha256"] == report["inputs_sha256"]["launch.json"]
        for hit in previous["cross_structure"]["muscle_source"]["added_pair_witnesses"]:
            face = tuple(hit["skin_fat_face"])
            source = hit["target"]
            target_index = banks[source["stable_id"]] + source["source_bank_face"]
            assert identity(target_index) == source
            cuts[face, target_index] = (np.array(face), muscles[target_index])
    compiler = None
    if mass_tools is not None:
        from map_mass import regional_context
        compiler, _, prior, prior_cells = regional_context(run, mass_tools)
        assert np.array_equal(prior, reference) and np.array_equal(prior_cells, cells)
    fitted = fit_heights(world, reference, cells, layer_present, limited,
                         [cuts[key] for key in sorted(cuts)], compiler)
    trial_receipts = []
    cases = [("direct", f, f * moves) for f in (1., .75, .5, .25, .125)]
    method = "determinant_and_surface_constrained" if cuts else "determinant_constrained"
    if compiler is not None:
        method += "_with_typed_mass_bounds"
    cases.append((method, 1., fitted))
    for method, fraction, increment in cases:
        active_world = (world + increment).astype(np.float32).astype(float)
        active_ref = reference + increment
        det = determinant(active_world, cells)
        ref = determinant(active_ref, cells)
        scaled_mass = mass * ref / ref_det
        donor_mass = np.bincount(donor, scaled_mass - mass)
        thickness = np.linalg.norm(active_world[2*n:] - active_world[n:2*n], axis=1)
        bad = np.flatnonzero((det <= 0.) | (ref <= 0.))
        emitted_ratio = det / skin_det
        qualities = quality(active_world, cells, det)
        lowest = np.argsort(emitted_ratio, kind="stable")[:8]
        trial_receipts.append({
            "method": method, "fraction": fraction,
            "worst_emitted_det_ratio": float(emitted_ratio.min()),
            "worst_reference_det_ratio": float(np.min(ref / ref_det)),
            "minimum_emitted_shape": float(qualities.min()),
            "inverted_or_degenerate_count": len(bad), "first_invalid_cells": bad[:8].tolist(),
            "expected_mass_change_at_unchanged_density_kg": float((scaled_mass - mass).sum()),
            "same_density_mass_change_by_donor_kg": {
                str(i): float(delta) for i, delta in enumerate(donor_mass) if abs(delta) > 1e-8
            },
            "max_tissue_column_thickness_m": float(thickness[moving].max()),
            "source_stage_residual_m": float(max(
                gap - increment[2*n + node, 2] for node, _, _, gap in anchors)),
            "near_bed_nodes_within_half_mm_target": int(sum(
                outer[node, 2] - report["bed_plane_offset_m"] <= .005
                and gap - increment[2*n + node, 2] <= .0005
                for node, _, _, gap in anchors)),
            "near_bed_target_nodes": int(sum(
                outer[node, 2] - report["bed_plane_offset_m"] <= .005
                for node, _, _, _ in anchors)),
            "worst_emitted_cells": [[int(i), float(emitted_ratio[i])] for i in lowest],
        })
    if emitted is not None:
        # This is an unadmitted authoring candidate. Preserve the integrator's
        # exact topology, unchanged outer geometry and both coordinate frames.
        with emitted.open("xb") as f:
            np.savez_compressed(f, nodes_current=(world + fitted).astype(np.float32),
                               nodes_reference=reference + fitted,
                               cells=cells.astype(np.uint32))
    return {
        "candidate_sha256": digest(emitted) if emitted else None,
        "model_scope": "Offline posterior geometric feasibility. Outer and dermal interface frozen, typed gluteal source only. No mechanical, material or solver changes. Density and mass feasibility are reported separately from shape.",
        "launch_sha256": report["inputs_sha256"]["launch.json"],
        "screen_sha256": digest(screen), "rest_nodes_sha256": digest(run / "shell-mass.csv.nodes.csv"),
        "anchors_columns": ["deep_column", "stable_muscle_id", "face", "rest_gap_m"],
        "modified_nodes": int(moving.sum()), "adjacent_cells": int(layer_present.sum()),
        "registered_crossing_constraints": len(cuts),
        "by_muscle": dict(Counter(source for _, source, _, _ in anchors)),
        "original_maximum_fat_thickness_m": float(current_thickness.max()),
        "original_minimum_shape": float(quality(world, cells, skin_det).min()),
        "anchors": anchors, "tests": trial_receipts,
    }


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("screen", type=Path)
    p.add_argument("reader", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--exclude-crossings-from", type=Path, action="append", default=[])
    p.add_argument("--preserve-typed-mass-with", type=Path)
    a = p.parse_args()
    result = fit(a.run, a.screen, a.reader, a.candidate, a.exclude_crossings_from,
                 a.preserve_typed_mass_with)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"modified_nodes": result["modified_nodes"], "tests": result["tests"]}, indent=2))


if __name__ == "__main__":
    main()
