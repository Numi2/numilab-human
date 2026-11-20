"""Refine the matched layer faces and all incident tetrahedra together."""
from collections import defaultdict
from itertools import combinations

import numpy as np

from fit_hand_cavity import determinant, quality


FACES = np.array(list(combinations(range(4), 3)))
BARYCENTRIC = ((1. / 3.,) * 3, (.5, .25, .25), (.25, .5, .25), (.25, .25, .5))


def paired_triples(columns, faces, roles):
    lookup = {}
    keys = defaultdict(dict)
    for index, triple in enumerate(columns):
        for layer, node in enumerate(triple):
            lookup[int(node)] = (index, layer)
    for index, (face, role) in enumerate(zip(faces, roles, strict=True)):
        if role == "side":
            continue
        column_ids, layers = zip(*(lookup[node] for node in face), strict=True)
        assert len(set(layers)) == 1
        key = tuple(sorted(column_ids))
        assert layers[0] not in keys[key]
        keys[key][layers[0]] = index
    return lookup, keys


def stellar_children(tet, face, center):
    """Replace each face vertex in turn, retaining the tetrahedron orientation."""
    slots = np.flatnonzero(np.isin(tet, face))
    assert len(slots) == 3
    children = np.repeat(tet[None], 3, axis=0)
    children[np.arange(3), slots] = center
    return children


def refine(columns, cells, points, rest, faces, roles, footprint, solid):
    """Accept the connected patch splits that preserve both frame floors."""
    columns = np.asarray(columns)
    lookup, keys = paired_triples(columns, faces, roles)
    incidence = defaultdict(list)
    for row, cell in enumerate(cells):
        for local in FACES:
            incidence[tuple(sorted(cell[local]))].append(row)
    floors = [(float(determinant(p, cells).min()), float(quality(p, cells).min()))
              for p in (points, rest)]
    rejected = defaultdict(int)
    replacement = {}
    face_replacement = {}
    world_add, rest_add, seeds, weights = [], [], [], []
    retained = []
    for face_row in footprint:
        face = faces[face_row]
        key = tuple(sorted(lookup[node][0] for node in face))
        match = keys[key]
        if len(match) != 3:
            rejected["unmatched_layer_faces"] += 1
            continue
        corners = columns[list(key)].T
        rows = [match[layer] for layer in range(3)]
        stars = [incidence[tuple(sorted(tri))] for tri in corners]
        assert list(map(len, stars)) == [1, 2, 1]
        child_begin = len(points) + len(world_add)
        node_ids = np.arange(child_begin, child_begin + 3)
        prepared = {row: stellar_children(cells[row], tri, vertex)
                    for tri, vertex, star in zip(corners, node_ids, stars, strict=True)
                    for row in star}
        assert len(prepared) == sum(map(len, stars))
        assert replacement.keys().isdisjoint(prepared)
        local = np.r_[np.unique(cells[list(prepared)]), node_ids]
        variant = np.searchsorted(local, np.vstack(list(prepared.values())))
        best = None
        exceeds_floor = False
        for scalar in BARYCENTRIC:
            barycentric = np.asarray(scalar)
            trial_world = (points[corners] * barycentric[None, :, None]).sum(1).astype(np.float32)
            trial_rest = (rest[corners] * barycentric[None, :, None]).sum(1)
            metrics = []
            for p, inserted, (det_min, q_min) in zip((points, rest), (trial_world, trial_rest),
                                                      floors, strict=True):
                values = np.vstack((p[local[:-3]], inserted))
                metrics.extend((determinant(values, variant).min() / det_min,
                                quality(values, variant).min() / q_min))
            score = min(metrics)
            if score < 1.:
                continue
            exceeds_floor = True
            if solid.locations(trial_world, np.arange(3)):
                continue
            if best is None or best[0] < score:
                best = (score, trial_world, trial_rest, barycentric, metrics)
        if best is None:
            reason = "new_bone_containment" if exceeds_floor else "child_frame_floor"
            rejected[reason] += 1
            continue
        _, trial_world, trial_rest, barycentric, metrics = best
        replacement.update(prepared)
        for row, tri, vertex in zip(rows, corners, node_ids, strict=True):
            triangles = np.c_[tri, np.roll(tri, -1), np.full(3, vertex)]
            face_replacement[row] = np.sort(triangles, axis=1)
        world_add.extend(trial_world)
        rest_add.extend(trial_rest)
        seeds.extend(corners)
        weights.append(barycentric)
        retained.append({"source_deep_child": int(face_row), "split_faces": rows,
                         "tets": list(prepared), "child_floor_ratios": list(map(float, metrics))})
    unchanged = np.setdiff1d(np.arange(len(cells)), np.fromiter(replacement, dtype=np.intp))
    child = np.concatenate((cells[unchanged], *replacement.values()))
    child_rows = np.r_[unchanged, np.repeat(list(replacement), 3)].astype(np.intp)
    faces_kept = np.setdiff1d(np.arange(len(faces)), np.fromiter(face_replacement, dtype=np.intp))
    surface = np.concatenate((faces[faces_kept], *face_replacement.values()))
    surface_rows = np.r_[faces_kept, np.repeat(list(face_replacement), 3)].astype(np.intp)
    mesh = {
        "world": np.vstack((points, np.asarray(world_add).reshape(-1, 3))),
        "rest": np.vstack((rest, np.asarray(rest_add).reshape(-1, 3))),
        "elements": child,
        "split_seed_nodes": np.asarray(seeds).reshape(-1, 3),
        "split_seed_barycentric": np.asarray(weights).reshape(-1, 3),
        "split_seed_cells": child_rows,
        "surface": surface,
        "split_seed_faces": surface_rows,
    }
    return mesh, retained, dict(rejected)
