"""Contract short interface edges while protecting retained source vertices."""
from argparse import ArgumentParser
import heapq
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from audit import closed_surface, digest
from partition_gluteal import surface_volume


def contract(points, facets, source, maximum):
    p = points.astype(float)
    f = facets.copy()
    vertex_faces = [set() for _ in p]
    face_keys = {}
    radius = np.zeros(len(p))
    live = np.ones(len(f), dtype=bool)
    for i, face in enumerate(f):
        for v in face:
            vertex_faces[v].add(i)
        face_keys[tuple(sorted(face))] = i
    queue, attempted = [], set()

    def enqueue(a, b):
        if source[a] >= 0 and source[b] >= 0:
            return
        a, b = sorted((int(a), int(b)))
        d = float(np.linalg.norm(p[a] - p[b]))
        if d <= maximum:
            heapq.heappush(queue, (d, a, b))

    for a, b in np.unique(np.sort(f[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1), axis=0):
        enqueue(a, b)

    count = 0
    while queue:
        d, a, b = heapq.heappop(queue)
        if source[b] >= 0:
            a, b = b, a
        common = vertex_faces[a] & vertex_faces[b]
        signature = (a, b, tuple(sorted(vertex_faces[a])), tuple(sorted(vertex_faces[b])))
        if len(common) != 2 or signature in attempted or radius[b] + d > maximum:
            continue
        attempted.add(signature)
        local = vertex_faces[a] | vertex_faces[b]
        one_ring_a = {int(v) for row in vertex_faces[a] for v in f[row]} - {a, b}
        one_ring_b = {int(v) for row in vertex_faces[b] for v in f[row]} - {a, b}
        opposite = {int(v) for row in common for v in f[row]} - {a, b}
        if len(opposite) != 2 or one_ring_a & one_ring_b != opposite:
            continue
        keep = np.array(sorted(vertex_faces[b] - common), dtype=int)
        previous = p[f[keep]]
        updated = np.where(f[keep] == b, a, f[keep])
        candidate = p[updated]
        normal = np.cross(previous[:, 1] - previous[:, 0], previous[:, 2] - previous[:, 0])
        new_normal = np.cross(candidate[:, 1] - candidate[:, 0], candidate[:, 2] - candidate[:, 0])
        if np.any(np.einsum("ij,ij->i", normal, new_normal) <= 0.):
            continue
        keys = [tuple(sorted(face)) for face in updated]
        if len(set(keys)) != len(keys) or any(face_keys.get(k, -1) not in local and
                                             k in face_keys for k in keys):
            continue
        if any(face_keys.get(k, -1) in vertex_faces[a] - common for k in keys):
            continue
        for row in sorted(vertex_faces[b]):
            face_keys.pop(tuple(sorted(f[row])))
            for v in f[row]:
                vertex_faces[v].remove(row)
        for row, face, key in zip(keep, updated, keys, strict=True):
            f[row] = face
            face_keys[key] = row
            for v in face:
                vertex_faces[v].add(row)
        live[sorted(common)] = False
        radius[a] = max(radius[a], radius[b] + d)
        count += 1
        for v in sorted({int(v) for row in vertex_faces[a] for v in f[row]} - {a}):
            enqueue(a, v)

    nodes, mapping = np.unique(f[live], return_inverse=True)
    emitted, surface = points[nodes], mapping.reshape(-1, 3)
    assert np.array_equal(np.sort(source[nodes][source[nodes] >= 0]), np.sort(source[source >= 0]))
    assert closed_surface(emitted, surface)
    return emitted, surface, nodes, live, count, radius.max()


def omit_detached_source_fragments(points, faces, vertex_parents, face_parents, original):
    with np.load(original) as base:
        original_points, originals = base["source_nodes"], base["source_faces"]
    edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    _, labels = connected_components(
        coo_matrix((np.ones(len(edges)), edges.T), shape=(len(points), len(points))),
        directed=False)
    by_face = labels[faces[:, 0]]
    selected = np.ones(len(faces), dtype=bool)
    removed = []
    for group in np.unique(by_face):
        use = np.flatnonzero(by_face == group)
        parents = face_parents[use]
        if not np.all(parents >= 2346):
            continue
        ids = faces[use]
        assert len(use) in (4, 6, 8) and len(np.unique(parents)) == len(use)
        for new, previous in zip(ids, originals[parents], strict=True):
            assert set(map(tuple, points[new])) == set(map(tuple, original_points[previous]))
        length = float(np.linalg.norm(np.ptp(points[np.unique(ids)].astype(float), axis=0)))
        volume = surface_volume(points.astype(float), ids)
        assert 0 < length < .0006 and 0 < volume < 2.e-12
        removed.append(dict(source_faces=sorted(map(int, parents)), diagonal_m=length,
                            volume_m3=volume))
        selected[use] = False
    kept, indices = np.unique(faces[selected], return_inverse=True)
    return points[kept], indices.reshape(-1, 3), vertex_parents[kept], face_parents[selected], removed


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("maximum_m", type=float)
    p.add_argument("--original-source", type=Path)
    a = p.parse_args()
    a.output.mkdir()
    result = {}
    for side in (29, 30):
        filename = f"gluteal-{side}.npz"
        origin, target = a.input / filename, a.output / filename
        with np.load(origin) as m:
            x, f = m["source_nodes"], m["source_faces"]
            old_vertex, old_face = m["source_vertex_ids"], m["source_face_parents"]
        removed = []
        if a.original_source is not None:
            x, f, old_vertex, old_face, removed = omit_detached_source_fragments(
                x, f, old_vertex, old_face, a.original_source / filename)
            assert len(removed) == 3
        y, g, nodes, kept, count, bound = contract(x, f, old_vertex, a.maximum_m)
        np.savez_compressed(target, source_nodes=y, source_faces=g,
                            source_vertex_ids=old_vertex[nodes], source_face_parents=old_face[kept])
        result[str(side)] = dict(input_sha256=digest(origin), output_sha256=digest(target),
                                 omitted_source_fragments=removed, contracted_edges=count, faces=len(g), max_coordinate_shift_bound_m=float(bound),
                                 relative_volume_change=surface_volume(y.astype(float), g) /
                                                        surface_volume(x.astype(float), f) - 1.)
    (a.output / "contraction.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
