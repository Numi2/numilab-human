"""Remove numerical slivers without moving respiratory source vertices.

Only almost-planar diagonal flips inside one affine respiratory field cell are
eligible. Every exact reciprocal copy must participate in the same operation.
This is asset preparation, not a time-dependent tissue solver.
"""
from __future__ import annotations

import numpy as np


def improve_sliver_faces(prepared, *, minimum_altitude_m=2.5e-7,
                         maximum_nonplanarity_m=1e-7, spacing_m=1/64):
    if (minimum_altitude_m != 2.5e-7 or maximum_nonplanarity_m != 1e-7 or
            spacing_m != 1/64):
        raise ValueError('unsupported respiratory mesh quality bounds')
    points, lookup, local_to_global, global_to_local = [], {}, {}, {}
    faces, lineage = {}, {}
    for sid in sorted(prepared):
        vertices, source_faces, parents, _ = prepared[sid]
        local = []
        for p in vertices[:, :3]:
            key = tuple(map(float, p))
            if key not in lookup:
                lookup[key] = len(points)
                points.append(key)
            local.append(lookup[key])
        local_to_global[sid] = np.asarray(local, dtype=np.int64)
        if len(set(local)) != len(local):
            raise ValueError('quality improvement requires welded source coordinates')
        global_to_local[sid] = {v: i for i, v in enumerate(local)}
        faces[sid] = local_to_global[sid][source_faces].copy()
        lineage[sid] = [set() for _ in source_faces]
        for parent, children in parents.items():
            for child in children:
                lineage[sid][child].add(parent)
    xyz = np.asarray(points, dtype=float)
    owners, edges = {}, {}

    def key(tri):
        return tuple(sorted(map(int, tri)))

    def tri_edges(tri):
        return [tuple(sorted((int(tri[a]), int(tri[b]))))
                for a, b in ((0, 1), (1, 2), (2, 0))]

    def add(sid, index):
        tri = faces[sid][index]
        owners.setdefault(key(tri), set()).add((sid, index))
        for edge in tri_edges(tri):
            edges.setdefault((sid, edge), set()).add(index)

    def remove(sid, index):
        tri = faces[sid][index]
        owner_key = key(tri)
        owners[owner_key].remove((sid, index))
        if not owners[owner_key]:
            del owners[owner_key]
        for edge in tri_edges(tri):
            edge_key = (sid, edge)
            edges[edge_key].remove(index)
            if not edges[edge_key]:
                del edges[edge_key]

    for sid, f in faces.items():
        for index in range(len(f)):
            add(sid, index)

    def geometry(f):
        p = xyz[f]
        normal = np.cross(p[..., 1, :] - p[..., 0, :], p[..., 2, :] - p[..., 0, :])
        lengths = np.stack([np.linalg.norm(p[..., 1, :] - p[..., 2, :], axis=-1),
                            np.linalg.norm(p[..., 2, :] - p[..., 0, :], axis=-1),
                            np.linalg.norm(p[..., 0, :] - p[..., 1, :], axis=-1)], axis=-1)
        return normal, lengths, np.linalg.norm(normal, axis=-1) / lengths.max(axis=-1)

    planes = np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1),
                         (1, -1, 0), (1, 0, -1), (0, 1, -1)))
    initial = []
    for sid, f in faces.items():
        _, _, alt = geometry(f)
        initial.extend((float(alt[i]), sid, int(i)) for i in np.flatnonzero(alt < minimum_altitude_m))
    changes, skips = [], {}

    def skip(reason):
        skips[reason] = skips.get(reason, 0) + 1

    for _, sid, index in sorted(initial):
        tri = faces[sid][index]
        normal, lengths, altitude = geometry(tri)
        if altitude >= minimum_altitude_m:
            continue
        opposite = int(np.argmax(lengths))
        c, a, b = (int(tri[opposite]), int(tri[(opposite+1) % 3]), int(tri[(opposite+2) % 3]))
        edge = tuple(sorted((a, b)))
        adjacent = edges[(sid, edge)]
        if len(adjacent) != 2:
            skip('non-two-face edge'); continue
        neighbor = next(i for i in adjacent if i != index)
        other = faces[sid][neighbor]
        d = int(next(i for i in other if i not in edge))
        copies1, copies2 = owners[key(tri)].copy(), owners[key(other)].copy()
        owner_ids = {s for s, _ in copies1}
        if (owner_ids != {s for s, _ in copies2} or
                len(copies1) != len(owner_ids) or len(copies2) != len(owner_ids)):
            skip('anatomical interface ownership boundary'); continue
        if len({a, b, c, d}) != 4 or any((s, tuple(sorted((c, d)))) in edges for s in owner_ids):
            skip('existing diagonal'); continue
        quad = xyz[[a, b, c, d]]
        coordinates = quad @ planes.T / spacing_m
        if not np.all(np.ceil(coordinates.max(axis=0)-1) <= np.floor(coordinates.min(axis=0))):
            skip('different affine field cells'); continue
        neighbor_normal = np.cross(xyz[b]-xyz[a], xyz[d]-xyz[a])
        neighbor_length = np.linalg.norm(neighbor_normal)
        if not neighbor_length:
            skip('degenerate neighbor'); continue
        nonplanarity = abs(float(np.dot(xyz[c]-xyz[a], neighbor_normal))) / neighbor_length
        if nonplanarity > maximum_nonplanarity_m:
            skip('nonplanar source patch'); continue
        proposed = np.asarray(((c, a, d), (c, d, b)), dtype=np.int64)
        new_normal, _, new_alt = geometry(proposed)
        old_normal, _, _ = geometry(np.asarray((tri, other)))
        mean = old_normal.sum(axis=0)
        if np.dot(new_normal[0], mean) < 0:
            proposed = proposed[:, (0, 2, 1)]
            new_normal = -new_normal
        if not np.all(new_normal @ mean > 0) or new_alt.min() < minimum_altitude_m:
            skip('orientation or quality not improved'); continue
        updates = []
        for owner in sorted(owner_ids):
            i = next(i for s, i in copies1 if s == owner)
            j = next(i for s, i in copies2 if s == owner)
            old_normals, _, _ = geometry(faces[owner][[i, j]])
            oriented = proposed.copy()
            if np.dot(new_normal[0], old_normals.sum(axis=0)) < 0:
                oriented = oriented[:, (0, 2, 1)]
            updates.append((owner, i, j, oriented))
        for owner, i, j, oriented in updates:
            remove(owner, i); remove(owner, j)
            faces[owner][[i, j]] = oriented
            combined_lineage = lineage[owner][i] | lineage[owner][j]
            lineage[owner][i] = combined_lineage.copy()
            lineage[owner][j] = combined_lineage.copy()
            add(owner, i); add(owner, j)
        changes.append({'source_surface': sid, 'source_face': index, 'neighbor_face': neighbor,
                        'shared_surface_owners': sorted(owner_ids),
                        'old_altitude_m': float(altitude), 'new_minimum_altitude_m': float(new_alt.min()),
                        'maximum_patch_nonplanarity_m': nonplanarity,
                        'modified_faces': {str(s): [i, j] for s, i, j, _ in updates}})
    remaining = {}
    for sid in sorted(prepared):
        vertices, old_faces, parents, detail = prepared[sid]
        f = np.asarray([[global_to_local[sid][int(i)] for i in tri] for tri in faces[sid]], dtype=np.int64)
        if len(f) != len(old_faces):
            raise AssertionError('diagonal flips changed face count')
        remapped = {parent: [] for parent in parents}
        for i, original_parents in enumerate(lineage[sid]):
            for parent in original_parents:
                remapped[parent].append(i)
        out = vertices.copy()
        p = out[f, :3].astype(float)
        normals = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
        summed = np.zeros((len(out), 3), dtype=float)
        for k in range(3):
            np.add.at(summed, f[:, k], normals)
        lengths = np.linalg.norm(summed, axis=1)
        if np.any(lengths == 0):
            raise ValueError('quality improvement left a zero vertex normal')
        out[:, 3:6] = summed / lengths[:, None]
        if not np.array_equal(out[:, :3], vertices[:, :3]):
            raise AssertionError('diagonal flips moved source vertices')
        _, _, alt = geometry(faces[sid])
        remaining[str(sid)] = {'minimum_altitude_m': float(alt.min()),
                               'below_bound_face_count': int(np.count_nonzero(alt < minimum_altitude_m))}
        detail['diagonal_flip_count'] = sum(sid in x['shared_surface_owners'] for x in changes)
        prepared[sid] = out, f, remapped, detail
    return {'method': 'shared_reciprocal_diagonal_flip_within_one_affine_field_cell',
            'minimum_altitude_target_m': minimum_altitude_m,
            'maximum_nonplanarity_m': maximum_nonplanarity_m,
            'source_vertex_displacement_m': 0, 'flip_count': len(changes),
            'changes': changes, 'skipped_reasons': skips, 'remaining': remaining,
            'qualification': 'local mesh conditioning only; remaining thin faces and native cycle validation must be checked'}
