"""Conforming source-guided skin/fat refinement against matched native bone faces."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
from time import monotonic

import numpy as np
from scipy.spatial import cKDTree

from audit_source import ROOT, TRIANGLE_ROW, digest, install_for_float32, load_registered_points, load_skin, owner, records
from audit_bed_layer_surfaces import role_surfaces
from compare_bank_trials import emitted
from fit_bank_group_partitions import ROLE_COUNT
from fit_refined_patch import fit_patch
from fit_hand_cavity import FACES, PARENT, determinant, quality, table


def ray_hits(origin, unit_direction, triangles, front_only=True):
    a, b, c = triangles.transpose(1, 0, 2)
    normal = np.cross(b - a, c - a)
    normal_length = np.linalg.norm(normal, axis=1)
    incidence = normal @ unit_direction
    # Match the outer surface, excluding back-facing and grazing intersections.
    if not front_only:
        alignment = np.abs(incidence)
    else:
        alignment = incidence
    usable = (normal_length > 0.) & (alignment > 0.2 * normal_length)
    distance = np.full(len(a), np.inf)
    distance[usable] = np.einsum("ij,ij->i", normal[usable], a[usable] - origin) / incidence[usable]
    selected = np.flatnonzero(np.abs(distance) < 0.02)
    relative = origin + distance[selected, None] * unit_direction - a[selected]
    ab, ac = (b - a)[selected], (c - a)[selected]
    u = np.einsum("ij,ij->i", np.cross(relative, ac), normal[selected]) / normal_length[selected]**2
    v = np.einsum("ij,ij->i", np.cross(ab, relative), normal[selected]) / normal_length[selected]**2
    inside = (u >= -1.e-9) & (v >= -1.e-9) & (u + v <= 1. + 1.e-9)
    return distance[selected[inside]], selected[inside]


def ray_hit(origin, unit_direction, triangles):
    distances, faces = ray_hits(origin, unit_direction, triangles)
    if len(faces) == 0:
        return None
    nearest = np.argmin(np.abs(distances))
    return float(distances[nearest]), faces[nearest]


def confined_bisect(cells, parents, a, b, midpoint):
    selected = np.any(cells == a, axis=1) & np.any(cells == b, axis=1)
    split = cells[selected]
    if not len(split):
        return cells, parents
    first, second = split.copy(), split.copy()
    first[first == a] = midpoint
    second[second == b] = midpoint
    return (np.concatenate((cells[~selected], first, second)),
            np.concatenate((parents[~selected], parents[selected], parents[selected])))


def child_surfaces(cells, material, parents, count, origins):
    base_lookup = {}
    for role, faces in parents.items():
        for face in faces:
            base_lookup[face] = (len(base_lookup), role)
    owners = defaultdict(list)
    for nodes, typed in zip(cells.tolist(), material.tolist(), strict=True):
        for face in combinations(sorted(nodes), len(FACES[0])):
            owners[face].append(typed)
    faces, ancestry, roles = [], [], []
    for face, typed in sorted(owners.items()):
        assert len(typed) in (1, 2)
        if len(typed) == 2 and typed[0] == typed[1]:
            continue
        original = set.union(*(origins[node - count] if node >= count else {node} for node in face))
        identity, role = base_lookup[tuple(sorted(original))]
        faces.append(face)
        ancestry.append(identity)
        roles.append(role)
    return np.array(faces), ancestry, roles


def exact_bone_pairs(points, faces, ancestry, bone):
    triangles, _ = owner.float32_surface_records(points, faces)
    shell = owner._prepare_surface_aabb(triangles)
    bodies, starts, counts = np.unique(bone["body"], return_index=True, return_counts=True)
    pairs = set()
    raw = 0
    for body, begin, count in zip(bodies, starts, counts, strict=True):
        bank = bone[begin:begin + count]
        result = owner._audit_pair_prepared_first(shell, records(bank["vertices"]))
        raw += result["count"]
        for face, local in result["triangle_pairs"]:
            row = bank[local]
            pairs.add((ancestry[face], int(body), int(row["bone"]), int(row["triangle"])))
    return pairs, raw, triangles


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("output", type=Path)
    targets = parser.add_mutually_exclusive_group()
    targets.add_argument("--bone-envelope", action="store_true")
    targets.add_argument("--fit-layer-thickness", action="store_true")
    parser.add_argument("--connected", action="store_true")
    parser.add_argument("--body", type=int, default=23)
    args = parser.parse_args()
    start = monotonic()
    install_for_float32()
    args.output.mkdir()
    base_path = ROOT / "exact-shell-bone-audit-032/exact-startup-crossings.json"
    source_path = Path(__file__).with_name("source-clearance.json")
    base = json.loads(base_path.read_text())
    source = json.loads(source_path.read_text())
    body = args.body
    assert all(bank["source_pairs"] == 0 for bank in source["banks"] if bank["body"] == body)
    assert digest(args.native / "physical-bone-initial.bin") == base["source"]["physical-bone-initial.bin"]
    assert digest(args.native / "registered-skin-native.csv") == source["source"]["registered-skin-native.csv"]
    launch = json.loads((args.native / "launch.json").read_text())["argv"]
    skin = Path(launch[launch.index("--skin-payload") + 1])
    assert digest(skin) == source["source"]["NHSKIN_sha256"]
    _, _, _, _, source_faces, _, _ = load_skin(skin)
    source_points = load_registered_points(args.native / "registered-skin-native.csv",
                                           source["source_vertex_rows"])
    if (args.bone_envelope or args.fit_layer_thickness) and not args.connected:
        parser.error("Layer envelope options require --connected")
    bone = np.memmap(args.native / "physical-bone-initial.bin", dtype=TRIANGLE_ROW, mode="r")
    original, original_nodal_mass = emitted(args.native)
    assert original.shape == (3 * ROLE_COUNT, 3)
    original = original.astype(np.float32).astype(np.float64)
    original_count = len(original)
    rows = table(PARENT.with_name(PARENT.name + ".tets.csv"))
    cells = np.array([[int(row[f"parent_node{i}"]) for i in range(4)] for row in rows], dtype=np.int32)
    typed = np.array([int(row["material_index"]) for row in rows])
    parent_mass = np.array([float(row["parent_mass_kg"]) for row in rows])
    support = set(np.flatnonzero(original[:ROLE_COUNT, 2] <= 1.e-5).tolist())
    old_surfaces = role_surfaces(rows)
    root_faces = [face for bank in old_surfaces.values() for face in bank]
    roles = [role for role, faces in old_surfaces.items() for _ in faces]
    before = {(face, bank, asset, tri) for _, face, _, bank, asset, tri in base["witnesses"]}
    desired = {face for role, face, _, bank, *_ in base["witnesses"] if role == "outer" and bank == body}
    edges = {edge for face in desired for edge in combinations(root_faces[face], 2)}
    edges = sorted(edges - {e for e in edges if support.intersection(e)})
    assert edges and all(0 <= a < b < ROLE_COUNT for a, b in edges)

    # Only inspect detailed triangles in the local search envelope. Bounding
    # spheres keep the query conservative even for non-uniform source faces.
    low = original[np.unique(edges)].min(axis=0) - 0.025
    high = original[np.unique(edges)].max(axis=0) + 0.025
    triangles = source_points[source_faces].astype(np.float64)
    local = np.flatnonzero(np.all(triangles.min(axis=1) <= high, axis=1) &
                           np.all(triangles.max(axis=1) >= low, axis=1))
    centers = triangles[local].mean(axis=1)
    radius = np.linalg.norm(triangles[local] - centers[:, None], axis=2).max()
    tree = cKDTree(centers)
    points = original.tolist()
    parents = np.arange(len(cells))
    origins = []
    proposals = []
    bank_triangles = bone["vertices"][bone["body"] == body].astype(np.float64)
    bank_centers = bank_triangles.mean(axis=1)
    bank_radius = np.linalg.norm(bank_triangles - bank_centers[:, None], axis=2).max()
    bank_tree = cKDTree(bank_centers)
    layer_clearance = []
    independent = []

    def add_proposal(nodes, outer, inner, direction):
        near = local[tree.query_ball_point(outer, radius + 0.02)]
        match = ray_hit(outer, direction, triangles[near])
        if match is None:
            return
        distance, index = match
        extent = float(np.dot(outer - inner, direction))
        nearby_bones = bank_tree.query_ball_point(outer, bank_radius + 0.02)
        crossings, _ = ray_hits(outer, direction, bank_triangles[nearby_bones], front_only=False)
        if len(crossings):
            bone_outward = float(crossings.max())
            # The registered outer surface and the bone bound the physical
            # room available to preserve both existing layer thicknesses.
            layer_clearance.append([int(nodes[0]), int(near[index]), distance, extent,
                                    bone_outward, distance - bone_outward])
            if args.bone_envelope:
                distance = max(distance, bone_outward + extent + 0.00005)
        source_face = int(near[index])
        proposals.append((nodes, distance, direction, source_face))
        if args.fit_layer_thickness:
            original_interface = np.asarray(points[nodes[1]])
            skin = float(np.dot(outer - original_interface, direction))
            interface = distance
            deep = distance
            if len(crossings):
                usable_space = distance - bone_outward - 0.00005
                if usable_space >= 0.001 and usable_space < extent:
                    tapered_skin = np.clip(usable_space * skin / extent, 0.0005, usable_space - 0.0005)
                    interface += skin - tapered_skin
                    deep += extent - usable_space
            independent.extend(([node], target, direction, source_face)
                               for node, target in zip(nodes, (distance, interface, deep), strict=True))

    for a, b in edges:
        new = []
        for layer in range(3):
            p, q = a + layer * ROLE_COUNT, b + layer * ROLE_COUNT
            index = len(points)
            if not np.any(np.any(cells == p, axis=1) & np.any(cells == q, axis=1)):
                # A material may not share the diagonal of the adjacent layer.
                continue
            midpoint = 0.5 * (original[p] + original[q])
            cells, parents = confined_bisect(cells, parents, p, q, index)
            points.append(midpoint.tolist())
            origins.append({p, q})
            new.append(index)
        assert new
        if len(new) != 3:
            continue
        outer, _, inner = (np.array(points[index]) for index in new)
        direction = outer - inner
        direction /= np.linalg.norm(direction)
        add_proposal(new, outer, inner, direction)

    edge_proposals = len(proposals)
    if args.connected:
        # Free only existing columns strictly inside the crossed patch and
        # one adjacent face ring. The rest of the original skin stays fixed.
        boundary = set(np.unique([root_faces[face] for face in desired])) - support
        ring = [face for face in old_surfaces["outer"] if boundary.intersection(face)]
        ring_lookup = set(ring)
        perimeter = {node for face in old_surfaces["outer"] if face not in ring_lookup for node in face}
        free = sorted(set(np.unique(ring)) - perimeter - support)
        for node in free:
            column = [node + i * ROLE_COUNT for i in range(3)]
            outer, _, inner = original[column]
            direction = outer - inner
            direction /= np.linalg.norm(direction)
            add_proposal(column, outer, inner, direction)
    reference = np.array(points, dtype=np.float32).astype(np.float64)
    points = reference.copy()
    volumes = determinant(reference, cells)
    shapes = quality(reference, cells)
    assert np.min(volumes) > 0.
    assert np.min(shapes) > 0.
    faces, face_ancestry, _ = child_surfaces(cells, typed[parents], old_surfaces, original_count, origins)
    cell_neighbors = defaultdict(set)
    for i, tet in enumerate(cells):
        for node in tet:
            if node >= original_count:
                cell_neighbors[node].add(i)
    moves = []
    # Keep conservative element and mass gates on each individual operation.
    # Re-evaluate overlapping incident tets against the same fixed split mesh.
    if not args.connected:
        for new, distance, direction, source_face in proposals:
            incident = np.array(sorted(set.union(*(cell_neighbors[node] for node in new))))
            amount = min(abs(distance), 0.003)
            direction = np.copysign(1., distance) * direction
            if amount < 1.e-7:
                continue
            for fraction in (1., 0.5, 0.25, 0.125, 0.0625):
                points[new] = (reference[new] + amount * fraction * direction).astype(np.float32)
                moved_volume = determinant(points, cells[incident])
                moved_shape = quality(points, cells[incident])
                if np.all(moved_volume > 0.88 * volumes[incident]) and np.all(moved_shape >= .88 * shapes[incident]):
                    moves.append([*new, source_face, distance, amount * fraction])
                    break
            else:
                points[new] = reference[new]
    iterations = []
    if args.connected:
        if args.fit_layer_thickness:
            groups = [nodes for nodes, *_ in proposals]
            points, iterations = fit_patch(reference, cells, parents, independent, groups)
        else:
            points, iterations = fit_patch(reference, cells, parents, proposals)
        moves = [row for row in proposals if np.any(points[row[0]] != reference[row[0]])]
    else:
        assert np.array_equal(points[:original_count], original)
    assert np.array_equal(points[list(support)], original[list(support)])
    candidate_volume = determinant(points, cells)
    candidate_shape = quality(points, cells)
    assert np.all(candidate_volume >= (.75 if args.connected else .88) * volumes)
    assert np.all(candidate_shape >= .88 * shapes)
    bisected, bisected_raw, before_records = exact_bone_pairs(reference, faces, face_ancestry, bone)
    after, after_raw, after_records = exact_bone_pairs(points, faces, face_ancestry, bone)
    selected = {i for i, face in enumerate(faces) if np.any(points[face] != reference[face])}
    old_self = set(map(tuple, owner._audit_self_pairs_touching_faces(before_records, selected)["triangle_pairs"]))
    new_self = set(map(tuple, owner._audit_self_pairs_touching_faces(after_records, selected)["triangle_pairs"]))
    mass_volume = np.bincount(parents, weights=candidate_volume, minlength=len(rows))
    mapped = parent_mass[parents] * candidate_volume / mass_volume[parents]
    nodal_mass = np.bincount(cells.ravel(), weights=np.repeat(mapped / 4., 4), minlength=len(points))
    ledger = {
        str(key): {"parent_kg": float(parent_mass[typed == key].sum()),
                   "child_kg": float(mapped[typed[parents] == key].sum())}
        for key in np.unique(typed)
    }
    assert np.max(np.abs(np.bincount(parents, weights=mapped) - parent_mass)) < 1.e-14
    candidate = args.output / "refined.npz"
    np.savez_compressed(candidate, points=points.astype(np.float32), tets=cells, parent=parents,
                        roles=typed[parents], surface=faces, surface_parent=np.array(face_ancestry),
                        cell_mass_kg=mapped, nodal_mass_kg=nodal_mass,
                        edge_parent=np.array([sorted(edge) for edge in origins]))
    def targeted(pairs):
        return sum(roles[face] == "outer" and bank == body for face, bank, *_ in pairs)
    summary = {
        "scope": "Offline exact skin/fat screen in the native initial pose. Counts use original shell-face and bone-face ancestry.",
        "base_audit_sha256": digest(base_path),
        "source_audit_sha256": digest(source_path),
        "proposal_sha256": digest(candidate),
        "targeted_bone_body": body,
        "original_vertices": original_count,
        "refined_vertices": len(points),
        "original_tets": len(rows),
        "refined_tets": len(cells),
        "target_faces": len(desired),
        "target_edges": len(edges),
        "matched_edges": edge_proposals,
        "connected_iterations": iterations,
        "bone_envelope_target": args.bone_envelope,
        "fitted_separate_layer_depths": args.fit_layer_thickness,
        "layer_clearance_columns": ["outer_node", "source_face", "source_distance_m",
                                    "existing_total_thickness_m", "outermost_sampled_bone_m",
                                    "space_for_both_layers_m"],
        "layer_clearance": layer_clearance,
        "columns_too_thick_for_source_envelope": sum(row[3] + 0.00005 > row[5] for row in layer_clearance),
        "matched_original_columns": len(proposals) - edge_proposals,
        "moved_original_nodes": int(np.any(points[:original_count] != original, axis=1).sum()),
        "moved_edge_columns": len(moves),
        "ray_distance_m_range": [min(p[1] for p in proposals), max(p[1] for p in proposals)] if proposals else [],
        "target_pairs": {"original": targeted(before), "bisected": targeted(bisected), "guided": targeted(after)},
        "whole_body_parent_pairs": {"original": len(before), "bisected": len(bisected), "guided": len(after)},
        "whole_body_raw_pairs": {"original": len(before), "bisected": bisected_raw, "guided": after_raw},
        "new_bone_pairs": sorted(after - before),
        "removed_bone_pairs": sorted(before - after),
        "added_unchanged_bisection_bone_pairs": sorted(bisected - before),
        "removed_unchanged_bisection_bone_pairs": sorted(before - bisected),
        "new_local_shell_pairs": sorted(new_self - old_self),
        "baseline_local_shell_pairs": len(old_self),
        "candidate_local_shell_pairs": len(new_self),
        "minimum_det_m3": float(candidate_volume.min()),
        "minimum_quality": float(candidate_shape.min()),
        "minimum_quality_ratio_to_bisection": float((candidate_shape / shapes).min()),
        "original_compiler_parent_mass_kg": float(parent_mass.sum()),
        "original_native_nodal_mass_kg": float(original_nodal_mass.sum()),
        "candidate_nodal_mass_kg": float(nodal_mass.sum()),
        "candidate_mass_by_material": ledger,
        "wall_seconds": monotonic() - start,
    }
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "layer_clearance" and (not key.endswith("_pairs") or isinstance(value, dict))}))


if __name__ == "__main__":
    main()
