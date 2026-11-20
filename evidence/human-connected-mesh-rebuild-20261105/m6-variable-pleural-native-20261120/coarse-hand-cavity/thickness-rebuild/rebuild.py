"""Rebuild a matched skull tissue patch and certify its conservative transport."""
from argparse import ArgumentParser
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREVIOUS = ROOT / "source-coarse-clearance-033"
AUDIT = ROOT / "exact-shell-bone-audit-032"
sys.path.insert(0, str(PREVIOUS))
from audit_source import TRIANGLE_ROW, digest, install_for_float32, load_registered_points, load_skin, owner
from audit_bed_layer_surfaces import role_surfaces
from compare_bank_trials import emitted
from fit_bank_group_partitions import ROLE_COUNT
from fit_hand_cavity import determinant, quality, table
from refine_clear_banks import child_surfaces, confined_bisect, exact_bone_pairs, ray_hit, ray_hits

from fit_layers import fit_layers
from regional_mass import RegionalMass


def prepare_surface_search(triangles, lower, upper):
    rows = np.flatnonzero(np.all(triangles.min(axis=1) <= upper, axis=1)
                          & np.all(triangles.max(axis=1) >= lower, axis=1))
    centers = triangles[rows].mean(axis=1)
    radius = np.linalg.norm(triangles[rows] - centers[:, None], axis=2).max()
    return rows, cKDTree(centers), radius


def add_midpoint_columns(edges, world, material, cells):
    origins = []
    columns = []
    parents = np.arange(len(cells))
    new_world, new_reference = world.tolist(), material.tolist()
    for a, b in sorted(edges):
        role_edges = [(a + layer * ROLE_COUNT, b + layer * ROLE_COUNT) for layer in range(3)]
        if not all(np.any(np.any(cells == p, axis=1) & np.any(cells == q, axis=1))
                   for p, q in role_edges):
            continue
        indices = []
        for p, q in role_edges:
            vertex = len(new_world)
            cells, parents = confined_bisect(cells, parents, p, q, vertex)
            new_world.append((.5 * (world[p] + world[q])).tolist())
            new_reference.append((.5 * (material[p] + material[q])).tolist())
            origins.append({p, q})
            indices.append(vertex)
        columns.append(indices)
    return np.array(new_world, np.float32).astype(np.float64), np.array(new_reference), cells, parents, origins, columns


def all_role_pair_counts(pairs, roles, body):
    return dict(Counter(roles[face] for face, bank, *_ in pairs if bank == body))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--whole-skull", action="store_true")
    parser.add_argument("--all-edges", action="store_true")
    parser.add_argument("--hold-parent", type=int, action="append", default=[],
                        help="keep all three layers of columns touching a measured pivot witness fixed")
    args = parser.parse_args()
    start = monotonic()
    args.output.mkdir()
    install_for_float32()
    body = 23
    enum_base = AUDIT / "exact-startup-crossings.json"
    enum_source = PREVIOUS / "source-clearance.json"
    base, source = (json.loads(p.read_text()) for p in (enum_base, enum_source))
    assert all(bank["source_pairs"] == 0 for bank in source["banks"] if bank["body"] == body)
    geometry = HERE / "inputs"
    source_path, bone_path = (geometry / name for name in ("registered-skin-native.csv", "physical-bone-initial.bin"))
    assert digest(source_path) == source["source"][source_path.name]
    assert digest(bone_path) == base["source"][bone_path.name]
    assert digest(geometry / "initial-skin-fem.csv") == base["source"]["initial-skin-fem.csv"]
    launch = json.loads((geometry / "launch.json").read_text())["argv"]
    skin = Path(launch[launch.index("--skin-payload") + 1])
    assert digest(skin) == source["source"]["NHSKIN_sha256"]
    _, _, _, _, source_faces, _, _ = load_skin(skin)
    source_world = load_registered_points(source_path, source["source_vertex_rows"])
    skin_tri = source_world[source_faces].astype(np.float64)
    bone = np.memmap(bone_path, dtype=TRIANGLE_ROW, mode="r")
    bank = bone["vertices"][bone["body"] == body].astype(np.float64)
    world, _ = emitted(geometry)

    reference = HERE / "compiled-reference"
    reference_identity = json.loads((reference / "execution.json").read_text())
    assert reference_identity["exit"] == 0
    assert digest(reference / "initial-skin-fem.csv") == digest(geometry / "initial-skin-fem.csv")
    tets = reference / "shell-mass.csv.tets.csv"
    points = reference / "shell-mass.csv.nodes.csv"
    for path in (tets, points):
        assert digest(path) == reference_identity[path.name]["sha256"]
    rows = table(tets)
    cells = np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in rows], dtype=np.int32)
    reference_nodes = table(points)
    assert np.array_equal(world, [[float(row[f"current_{axis}_m"]) for axis in "xyz"] for row in reference_nodes])
    material = np.array([[float(row[f"rest_{axis}_m"]) for axis in "xyz"] for row in reference_nodes])
    signed_ref = determinant(material, cells)
    ledger_volume = np.array([float(row["parent_reference_volume_m3"]) for row in rows])
    assert np.all(signed_ref > 0.)
    np.testing.assert_allclose(signed_ref / 6., ledger_volume, atol=1.e-20, rtol=5.e-10)
    regional = RegionalMass.from_original(material, cells, rows, AUDIT / "build-commands.json")
    support = set(np.flatnonzero(world[:ROLE_COUNT, 2] <= 1.e-5))
    assert len(world) == 3 * ROLE_COUNT
    surface = role_surfaces(rows)
    root_faces = [face for faces in surface.values() for face in faces]
    roles = [key for key, faces in surface.items() for _ in faces]
    target_faces = {face for role, face, _, bank_id, *_ in base["witnesses"]
                    if bank_id == body and role != "side" and (role == "outer" or args.whole_skull)}
    rim = {node % ROLE_COUNT for face in target_faces for node in root_faces[face]}
    ring = {face for face in surface["outer"] if rim.intersection(face)}
    fixed = {row for face in surface["outer"] if face not in ring for row in face} | support
    free = sorted(set().union(*ring) - fixed)
    edges = {edge for face in target_faces
             for edge in combinations(sorted({p % ROLE_COUNT for p in root_faces[face]}), 2)
             if not support.intersection(edge)}
    new_world, new_ref, cells, parents, origins, edge_columns = add_midpoint_columns(edges, world, material, cells)
    column_groups = edge_columns + [[node + role * ROLE_COUNT for role in range(3)] for node in free]
    if any(not 0 <= row < len(rows) for row in args.hold_parent):
        parser.error("--hold-parent must name an original compiled tetrahedron")
    held_nodes = np.unique(cells[np.isin(parents, args.hold_parent)])
    if len(held_nodes):
        held = set(held_nodes)
        column_groups = [triple for triple in column_groups if not held.intersection(triple)]
    min_corner = new_world[np.unique(column_groups)].min(axis=0) - .025
    max_corner = new_world[np.unique(column_groups)].max(axis=0) + .025
    skin_rows, skin_tree, skin_radius = prepare_surface_search(skin_tri, min_corner, max_corner)
    bone_rows, bone_tree, bone_radius = prepare_surface_search(bank, min_corner, max_corner)
    proposals, columns = [], []
    measured_gap = []
    for triple in column_groups:
        o, i, d = new_world[triple]
        axis = o - d
        extent = np.linalg.norm(axis)
        axis /= extent
        near = skin_rows[skin_tree.query_ball_point(o, skin_radius + .02)]
        match = ray_hit(o, axis, skin_tri[near])
        if match is None:
            continue
        delta, _ = match
        nearby_bank = bone_rows[bone_tree.query_ball_point(o, bone_radius + .02)]
        intercept, _ = ray_hits(o, axis, bank[nearby_bank], front_only=False)
        skin_width = float((o - i) @ axis)
        shift = [delta] * 3
        if len(intercept):
            gap = delta - intercept.max() - .00005
            measured_gap.append([int(triple[0]), float(gap), float(extent)])
            if .001 <= gap < extent:
                target_skin = np.clip(skin_width, .0005, gap - .0005)
                shift[1] += skin_width - target_skin
                shift[2] += extent - gap
        columns.append(triple)
        proposals.extend((node, value, axis, weight) for node, value, weight in
                         zip(triple, shift, (1., 1., 2.), strict=True))
    if not columns:
        raise RuntimeError("the target patch contains no aligned source/bone columns")
    print(f"fitting {len(columns)} columns across {len(target_faces)} parent faces", flush=True)
    candidate, rest, history = fit_layers(new_world, new_ref, cells, parents, regional, proposals, columns,
                                               all_edges=args.all_edges)
    assert np.array_equal(candidate[list(support)], world[list(support)])
    assert np.array_equal(candidate[list(fixed)], world[list(fixed)])
    assert np.array_equal(candidate[held_nodes], new_world[held_nodes])
    assert np.array_equal(rest[held_nodes], new_ref[held_nodes])
    print(f"ran {len(history)} linearizations", flush=True)
    volume = determinant(rest, cells)
    changed = np.any(rest != new_ref, axis=1)
    active_parent = np.unique(parents[np.any(changed[cells], axis=1)])
    mass, routing, ledger = regional.allocate(rest, cells, parents, volume, active_parent)
    print(f"mass route status {ledger}", flush=True)

    faces, face_ancestry, _ = child_surfaces(cells, regional.material[parents], surface, len(world), origins)
    before = {(face, bank_id, asset, tri)
              for _, face, _, bank_id, asset, tri in base["witnesses"]}
    bisected, bisected_raw, original_records = exact_bone_pairs(new_world, faces, face_ancestry, bone)
    assert bisected == before, "unchanged bisection altered the reference crossing identities"
    after, after_raw, candidate_records = exact_bone_pairs(candidate, faces, face_ancestry, bone)
    check = {i for i, face in enumerate(faces) if np.any(candidate[face] != new_world[face])}
    old_self = set(map(tuple, owner._audit_self_pairs_touching_faces(original_records, check)["triangle_pairs"]))
    new_self = set(map(tuple, owner._audit_self_pairs_touching_faces(candidate_records, check)["triangle_pairs"]))
    archive = args.output / "rebuild.npz"
    arrays = dict(world=candidate.astype(np.float32), rest=rest, elements=cells, parent=parents,
                  material=regional.material[parents], donor=regional.donor[parents],
                  surface=faces, surface_parent=np.array(face_ancestry),
                  edge_parent=np.asarray([sorted(origin) for origin in origins]))
    if routing is not None:
        arrays.update(cell_mass_kg=mass, route_source=routing[0], route_receiver=routing[1],
                      route_mass_kg=routing[2])
    np.savez_compressed(archive, **arrays)
    receipt = {
        "scope": "Offline shell rebuild using matched current and compiled material reference geometry",
        "candidate_sha256": digest(archive),
        "exact_baseline_sha256": digest(enum_base),
        "compiled_reference_receipt_sha256": digest(reference / "execution.json"),
        "registered_source_sha256": source["source"][source_path.name],
        "body": body, "whole_skull_footprint": args.whole_skull, "all_shape_edges": args.all_edges,
        "held_parent_rows": args.hold_parent, "held_nodes": held_nodes.tolist(),
        "parent_faces": len(target_faces), "active_columns": len(columns), "split_columns": len(edge_columns),
        "world_and_rest_nodes": len(candidate), "tetrahedra": len(cells),
        "fit": history, "source_gap_columns": measured_gap,
        "mass_route": ledger, "base_mass_kg": float(regional.source_kg.sum()),
        "candidate_mass_kg": float(mass.sum()) if mass is not None else None,
        "baseline_skull_pairs": all_role_pair_counts(before, roles, body),
        "candidate_skull_pairs": all_role_pair_counts(after, roles, body),
        "old_pair_count": len(before), "new_pair_count": len(after),
        "raw_bisected_pair_count": bisected_raw, "raw_candidate_pair_count": after_raw,
        "new_outside_pairs": sorted(pair for pair in after - before if pair[1] != body),
        "new_skull_pairs": sorted(pair for pair in after - before if pair[1] == body),
        "removed_pair_count": len(before - after),
        "new_self_pairs": sorted(new_self - old_self),
        "minimum_candidate_world_det_m3": float(determinant(candidate, cells).min()),
        "minimum_candidate_reference_det_m3": float(volume.min()),
        "minimum_candidate_world_shape": float(quality(candidate, cells).min()),
        "minimum_candidate_reference_shape": float(quality(rest, cells).min()),
        "seconds": monotonic() - start,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in receipt.items()
                      if key not in ("source_gap_columns", "new_outside_pairs", "new_skull_pairs", "new_self_pairs")}),
          flush=True)


if __name__ == "__main__":
    main()
