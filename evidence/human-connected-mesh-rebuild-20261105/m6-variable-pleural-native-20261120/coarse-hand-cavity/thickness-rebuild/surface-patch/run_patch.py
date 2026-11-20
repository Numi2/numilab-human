"""Rebuild the measured connected scalp footprint with a matched control."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
from time import monotonic

import numpy as np

from rebuild import (AUDIT, HERE, PREVIOUS, ROLE_COUNT, TRIANGLE_ROW, RegionalMass,
                     SkullContact, SolidSkull, all_role_pair_counts, digest, exact_bone_pairs,
                     fit_layers, install_for_float32, load_registered_points, load_skin,
                     owner, prepare_surface_search, ray_hit, ray_hits, role_surfaces, table,
                     determinant, quality)
from paired_surface import refine
from verify_rebuild import signed_boundary


def locked_columns(data):
    paired = len(data["edge_parent"])
    return np.r_[np.arange(3 * ROLE_COUNT).reshape(3, ROLE_COUNT).T,
                 np.arange(3 * ROLE_COUNT, 3 * ROLE_COUNT + paired).reshape(-1, 3)]


def target_column(triple, x, skin_tri, skin_search, bone_tri, bone_search):
    o, i, d = x[triple]
    direction = o - d
    extent = np.linalg.norm(direction)
    direction /= extent
    skin_rows, skin_tree, skin_radius = skin_search
    bone_rows, bone_tree, bone_radius = bone_search
    near_skin = skin_rows[skin_tree.query_ball_point(o, skin_radius + .020)]
    near_bone = bone_rows[bone_tree.query_ball_point(o, bone_radius + .020)]
    match = ray_hit(o, direction, skin_tri[near_skin])
    intersections, _ = ray_hits(o, direction, bone_tri[near_bone], front_only=False)
    if match is None:
        return None
    delta, _ = match
    intercept = intersections.max() if len(intersections) else None
    skin_width = (o - i) @ direction
    move = np.full(3, delta)
    gap = None
    if intercept is not None:
        deep_target = intercept + .00005
        gap = delta - deep_target
        if .001 <= gap < extent:
            width = np.clip(skin_width, .0005, gap - .0005)
            move[1] += skin_width - width
            move[2] += extent - gap
    return direction, move, gap


def unchanged_boundaries(cells, parent_cells, parent_row, seed):
    for row in np.unique(parent_row[len(cells) - 3 * (len(cells) - len(parent_cells)) // 2:]):
        incident = cells[parent_row == row]
        if len(incident) == 1:
            continue
        assert len(incident) == 3
        prior = parent_cells[row]
        center = np.setdiff1d(incident, prior)
        assert len(center) == 1
        split = np.intersect1d(incident[0], incident[1])
        split = np.intersect1d(split, incident[2])
        assert center[0] in split and len(split) == 2
        apex = split[split != center[0]][0]
        base = [int(node) for node in prior if node != apex]
        expected = signed_boundary(prior[None])
        face = tuple(sorted(base))
        sign = expected.pop(face)
        for i, a in enumerate(face):
            for b in face[i+1:]:
                opposite = next(j for j, node in enumerate(face) if node != a and node != b)
                expected[(a, b, int(center[0]))] = sign * (-1) ** opposite
        # Refined topology has exactly one oriented subdivision of the
        # old triangular boundary, with the other three faces unchanged.
        assert signed_boundary(incident) == expected
        for points in seed:
            np.testing.assert_allclose(determinant(points, incident).sum(),
                                       determinant(points, prior[None])[0],
                                       rtol=8.e-4, atol=2.e-14)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--lp", choices=("highs", "highs-ipm"), default="highs-ipm")
    args = parser.parse_args()
    args.output.mkdir()
    start = monotonic()
    install_for_float32()
    checkpoint = HERE / "cavity-batch-037"
    candidate = checkpoint / "local-deep-restoration-03/rebuild.npz"
    footprint_doc = json.loads((checkpoint / "deep-face-components.json").read_text())
    source_receipt = json.loads((candidate.parent / "receipt.json").read_text())
    assert digest(candidate) == footprint_doc["sha256"] == source_receipt["candidate_sha256"]
    with np.load(candidate, allow_pickle=False) as saved:
        data = dict(saved)
    points, rest, cells, parent, surface, ancestry = (data[name] for name in
        ("world", "rest", "elements", "parent", "surface", "surface_parent"))
    points = points.astype(float)
    source_columns = locked_columns(data)

    reference = HERE / "compiled-reference"
    tets = reference / "shell-mass.csv.tets.csv"
    nodes = reference / "shell-mass.csv.nodes.csv"
    reference_receipt = reference / "execution.json"
    assert digest(reference_receipt) == source_receipt["compiled_reference_receipt_sha256"]
    reference_doc = json.loads(reference_receipt.read_text())
    for path in (tets, nodes):
        assert digest(path) == reference_doc[path.name]["sha256"]
    rows = table(tets)
    original_reference = np.array([[float(row[f"rest_{axis}_m"]) for axis in "xyz"]
                                   for row in table(nodes)])
    original_cells = np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in rows])
    regional = RegionalMass.from_original(original_reference, original_cells, rows, AUDIT / "build-commands.json")
    root_roles = [role for role, faces in role_surfaces(rows).items() for _ in faces]
    roles = np.asarray(root_roles)[ancestry]

    body = source_receipt["body"]
    bone_path = HERE / "inputs/physical-bone-initial.bin"
    baseline_path = AUDIT / "exact-startup-crossings.json"
    assert digest(baseline_path) == source_receipt["exact_baseline_sha256"]
    baseline = json.loads(baseline_path.read_text())
    assert digest(bone_path) == baseline["source"][bone_path.name]
    bone = np.memmap(bone_path, dtype=TRIANGLE_ROW, mode="r")
    bank = bone[bone["body"] == body]

    source_contact = SkullContact(surface, ancestry, roles, bank)
    old_contact = source_contact.measure(points)
    assert old_contact.counts == {r: source_receipt["candidate_skull_pairs"].get(r, 0)
                                  for r in old_contact.counts}
    old_deep = old_contact.face_rows[roles[old_contact.face_rows] == "deep"]
    deep_component = footprint_doc["connected_components"][0]
    footprint = np.unique(old_deep[np.isin(ancestry[old_deep], deep_component["faces"])])
    active = np.flatnonzero(np.any(np.isin(source_columns, surface[footprint].ravel()), axis=1))
    for _ in range(2):
        touched = np.any(np.isin(surface, source_columns[active].ravel()), axis=1) & (roles == "deep")
        active = np.flatnonzero(np.any(np.isin(source_columns, surface[touched].ravel()), axis=1))
    old_total, old_raw, _ = exact_bone_pairs(points, surface, ancestry, bone)
    assert len(old_total) == source_receipt["new_pair_count"]
    report = {"source_sha256": digest(candidate), "skull_bank": body,
              "source_document_sha256": digest(candidate.parent / "receipt.json"),
              "footprint_deep_parents": deep_component["faces"], "footprint_crossed_child_faces": len(footprint),
              "initial_columns": int(len(active)), "lp_method": args.lp, "source_skull_pairs": old_contact.counts,
              "source_raw_pairs": old_raw, "source_whole_body_pairs": len(old_total)}
    seed, reference_seed, split_cells = points, rest, cells
    new = {}
    solid = SolidSkull(bank, np.unique(surface), points)
    if args.refine:
        mesh, splits, rejected = refine(source_columns, cells, points, rest, surface, roles, footprint, solid)
        if not splits:
            raise RuntimeError("no footprint splits survived geometric admission")
        seed, reference_seed, split_cells = (mesh[key] for key in ("world", "rest", "elements"))
        unchanged_boundaries(split_cells, cells, mesh["split_seed_cells"], (seed, reference_seed))
        parent = parent[mesh["split_seed_cells"]]
        surface = mesh["surface"]
        ancestry = ancestry[mesh["split_seed_faces"]]
        roles = roles[mesh["split_seed_faces"]]
        columns = np.vstack((source_columns[active], np.arange(len(points), len(seed)).reshape(-1, 3)))
        new = {key: value for key, value in mesh.items() if key.startswith("split_seed_")}
        report.update(splits=splits, unsplit=rejected)
    else:
        columns = source_columns[active]

    split_total, split_raw, seed_triangles = exact_bone_pairs(seed, surface, ancestry, bone)
    report.update(seed_raw_pairs=split_raw, seed_pairs_cleared=len(old_total - split_total),
                  seed_new_pair_count=len(split_total - old_total), seed_nodes=len(seed),
                  seed_cells=len(split_cells))
    if split_total != old_total or args.screen_only:
        # Any new Float32 identities at the unchanged split are recorded and
        # rejected before fitting. Face-count reduction is not repair evidence.
        report.update(seconds=monotonic() - start, seed_new_pairs=sorted(split_total - old_total))
        (args.output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        print({key: value for key, value in report.items() if key not in ("splits", "seed_new_pairs", "footprint_deep_parents")}, flush=True)
        return

    original = HERE / "inputs"
    skin_csv = original / "registered-skin-native.csv"
    detail = json.loads((PREVIOUS / "source-clearance.json").read_text())
    assert digest(skin_csv) == detail["source"][skin_csv.name] == source_receipt["registered_source_sha256"]
    launch = json.loads((original / "launch.json").read_text())["argv"]
    skin = Path(launch[launch.index("--skin-payload") + 1])
    assert digest(skin) == detail["source"]["NHSKIN_sha256"]
    _, _, _, _, faces, _, _ = load_skin(skin)
    detailed = load_registered_points(skin_csv, detail["source_vertex_rows"])[faces].astype(float)
    bank_tri = bank["vertices"].astype(float)
    lo, hi = seed[columns].min(axis=(0, 1)) - .025, seed[columns].max(axis=(0, 1)) + .025
    skin_search = prepare_surface_search(detailed, lo, hi)
    bone_search = prepare_surface_search(bank_tri, lo, hi)
    proposals, kept, gaps = [], [], []
    support = np.flatnonzero(points[:ROLE_COUNT, 2] <= 1.e-5)
    for triple in columns:
        if np.intersect1d(triple, support).size:
            continue
        targets = target_column(triple, seed, detailed, skin_search, bank_tri, bone_search)
        if targets is None:
            continue
        direction, move, gap = targets
        kept.append(triple)
        proposals.extend((node, shift, direction, weight) for node, shift, weight in
                         zip(triple, move, (1., 1., 2.), strict=True))
        if gap is not None:
            gaps.append(gap)
    report.update(fitted_columns=len(kept), measured_ray_gaps=len(gaps),
                  ray_gap_quantiles_mm=list(1000. * np.quantile(gaps, (0., .1, .5, .9, 1.))),
                  ray_gaps_below_1_mm=int(np.count_nonzero(np.asarray(gaps) < .001)))
    contact = SkullContact(surface, ancestry, roles, bank)
    solid.nodes = np.unique(surface)
    assert not solid.newly_unsafe(seed, solid.nodes)
    print({key: value for key, value in report.items() if key not in ("splits", "footprint_deep_parents")}, flush=True)
    final, final_rest, history = fit_layers(seed, reference_seed, split_cells, parent, regional,
                                            proposals, kept, all_edges=True, contact=contact,
                                            restore_local=True, solid=solid, tangent=True, restore_deep=True, lp_method=args.lp)
    np.testing.assert_array_equal(final[support], points[support])
    (args.output / "line-search.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    rest_det = determinant(final_rest, split_cells)
    moved = np.any(final_rest != reference_seed, axis=1)
    native_changed = np.any(final_rest[:len(original_reference)] != original_reference, axis=1)
    moved[:len(native_changed)] |= native_changed
    active_parent = parent[np.any(moved[split_cells], axis=1)]
    a, b = data["route_source"], data["route_receiver"]
    active_parent = np.unique(np.r_[active_parent, a[a != b], b[a != b]])
    mass, routing, ledger = regional.allocate(final_rest, split_cells, parent, rest_det, active_parent)
    exact, raw, final_triangles = exact_bone_pairs(final, surface, ancestry, bone)
    touched = {i for i, face in enumerate(surface) if np.any(seed[face] != final[face])}
    prior_self = set(map(tuple, owner._audit_self_pairs_touching_faces(seed_triangles, touched)["triangle_pairs"]))
    final_self = set(map(tuple, owner._audit_self_pairs_touching_faces(final_triangles, touched)["triangle_pairs"]))
    old_solid = solid.locations(points, np.unique(data["surface"]))
    final_solid = solid.locations(final, np.unique(surface))
    novel_solid = {key: value for key, value in final_solid.items()
                   if key not in old_solid or (value != old_solid[key] and value != "outside")}
    report.update(skull_pairs=all_role_pair_counts(exact, root_roles, body),
                  whole_body_pairs=len(exact), raw_body_pairs=raw,
                  new_other_bank_pairs=sorted(pair for pair in exact - old_total if pair[1] != body),
                  new_self_pairs=sorted(final_self - prior_self), old_solid_count=len(old_solid),
                  final_solid_count=len(final_solid), new_solid_pairs=[(*k, v) for k, v in novel_solid.items()],
                  cell_mass_route=ledger, total_mass_kg=None if mass is None else float(mass.sum()),
                  rest_min_det=float(rest_det.min()), world_min_det=float(determinant(final, split_cells).min()),
                  seconds=monotonic() - start)
    if routing is not None:
        new.update(cell_mass_kg=mass, route_source=routing[0], route_receiver=routing[1], route_mass_kg=routing[2])
    archive = args.output / "rebuild.npz"
    np.savez_compressed(archive, world=final.astype(np.float32), rest=final_rest,
                        elements=split_cells, parent=parent, surface=surface, surface_parent=ancestry,
                        material=regional.material[parent], donor=regional.donor[parent], **new)
    report["candidate_sha256"] = digest(archive)
    (args.output / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print({key: value for key, value in report.items() if key not in ("splits", "footprint_deep_parents")}, flush=True)


if __name__ == "__main__":
    main()
