"""Try bounded tetrahedron flips without moving the authored surface."""
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

from extension import determinant, quality


def faces_of(tet):
    return combinations(sorted(tet), 3)


def edges_of(tet):
    return combinations(sorted(tet), 2)


def rim(tets):
    return {k for k, v in Counter(face for tet in tets for face in faces_of(tet)).items() if v == 1}


def prepare_replacement(points, current, proposal, floor):
    if rim(current) != rim(proposal):
        return None
    fresh = np.asarray(proposal)
    det = determinant(points, fresh)
    flipped = det < 0.
    fresh[flipped] = fresh[flipped][:, [0, 2, 1, 3]]
    det = np.abs(det)
    before_det = determinant(points, current)
    volume = before_det.sum()
    if det.min() <= 0. or not np.isclose(det.sum(), volume, rtol=1e-11, atol=0.):
        return None
    before = quality(points, current, before_det)
    after = quality(points, fresh, det)
    before_count = int((before < floor).sum())
    after_count = int((after < floor).sum())
    if after_count >= before_count or after.min() < before.min():
        return None
    return (before_count - after_count, float(after.min())), fresh


def improve(points, cells, floor, max_passes=3):
    points = points.astype(float)
    flips = Counter()
    initial_rim = rim(cells)
    counts = []
    for _ in range(max_passes):
        tet = {i: tuple(row) for i, row in enumerate(cells)}
        faces, edges = defaultdict(set), defaultdict(set)

        def attach(i):
            for face in faces_of(tet[i]):
                faces[face].add(i)
            for edge in edges_of(tet[i]):
                edges[edge].add(i)

        def detach(i):
            for face in faces_of(tet[i]):
                faces[face].remove(i)
            for edge in edges_of(tet[i]):
                edges[edge].remove(i)
            del tet[i]

        for i in tet:
            attach(i)
        bad = np.flatnonzero(quality(points, cells, determinant(points, cells)) < floor)
        counts.append(len(bad))
        next_id = len(tet)
        for index in bad:
            if index not in tet:
                continue
            old = tet[index]
            proposals = []
            for face in faces_of(old):
                ids = sorted(faces[face])
                if len(ids) != 2:
                    continue
                a, b = sorted(set(tet[ids[0]] + tet[ids[1]]) - set(face))
                if edges[a, b]:
                    continue
                proposals.append(("two_to_three", ids,
                                  [(a, b, u, v) for u, v in combinations(face, 2)]))
            for edge in edges_of(old):
                ids = sorted(edges[edge])
                if len(ids) != 3:
                    continue
                ring = sorted({v for i in ids for v in tet[i]} - set(edge))
                if len(ring) != 3 or faces[tuple(ring)]:
                    continue
                proposals.append(("three_to_two", ids, [(*ring, apex) for apex in edge]))
            best = None
            for kind, ids, proposal in proposals:
                current = np.asarray([tet[i] for i in ids])
                result = prepare_replacement(points, current, proposal, floor)
                if result is not None and (best is None or result[0] > best[0]):
                    best = result[0], kind, ids, result[1]
            if best is None:
                continue
            _, kind, ids, replacement = best
            for i in ids:
                detach(i)
            for row in replacement:
                tet[next_id] = tuple(row)
                attach(next_id)
                next_id += 1
            flips[kind] += 1
        if next_id == len(cells):
            break
        cells = np.asarray(list(tet.values()), dtype=cells.dtype)
    assert rim(cells) == initial_rim
    counts.append(int((quality(points, cells, determinant(points, cells)) < floor).sum()))
    return cells, {"performed_flips": dict(flips), "under_quality_by_pass": counts}
