"""Local triangulation repair inside one exact, identified source facet.

The operation changes two index triples, preserves every source coordinate,
and joins both faces' ancestry. It does not qualify native geometry, material
mechanics, interfaces, or a complete anatomical surface.
"""
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class SourceFacetDiagonalFlip:
    face_indices: tuple[int, int]
    original_faces: tuple[tuple[int, int, int], tuple[int, int, int]]
    replacement_faces: tuple[tuple[int, int, int], tuple[int, int, int]]
    source_ancestor_faces: tuple[int, ...]


def _integer(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError('source mesh indices must be integers')
    return int(value)


def _point(value):
    if len(value) != 3:
        raise ValueError('exact source points require three coordinates')
    result = []
    for coordinate in value:
        if isinstance(coordinate, Fraction):
            result.append(coordinate)
        else:
            result.append(Fraction(_integer(coordinate)))
    return tuple(result)


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2],
            a[0]*b[1]-a[1]*b[0])


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def flip_same_source_facet_diagonal(
        vertices_exact, faces, source_facet_labels, edge, *,
        face_lineage, protected_faces=()):
    """Return a repaired face array and explicit multi-parent provenance.

    Labels must identify original source facets, not inferred geometric
    proximity. Coordinates must be original rational values; rounded native
    points cannot authenticate their source facet. Only a strictly convex,
    coplanar quadrilateral with opposite edge winding is accepted. Callers
    must protect interface faces and independently validate the final mesh.
    No input is mutated, including on failure.
    """
    source = np.asarray(faces)
    if source.ndim != 2 or source.shape[1] != 3 or source.dtype.kind not in 'iu':
        raise ValueError('source faces must be an integer triangle array')
    if (source.size and (source.min() < 0 or source.max() >= len(vertices_exact))):
        raise ValueError('source face index is out of bounds')
    if len(source_facet_labels) != len(source) or len(face_lineage) != len(source):
        raise ValueError('every source face needs facet identity and ancestry')
    if len(edge) != 2:
        raise ValueError('a diagonal requires two endpoints')
    left, right = map(_integer, edge)
    if left == right or min(left, right) < 0 or max(left, right) >= len(vertices_exact):
        raise ValueError('source diagonal endpoints are invalid')
    protected = {_integer(i) for i in protected_faces}
    if any(i < 0 or i >= len(source) for i in protected):
        raise ValueError('protected source face index is out of bounds')
    owners = []
    for index, face in enumerate(source):
        if left in face and right in face:
            if len(set(map(int, face))) != 3:
                raise ValueError('source diagonal touches a degenerate face')
            for corner in range(3):
                u, v, opposite = map(int, (face[corner], face[(corner+1) % 3],
                                           face[(corner+2) % 3]))
                if {u, v} == {left, right}:
                    owners.append((index, u, v, opposite))
                    break
    if len(owners) != 2:
        raise ValueError('source diagonal must have exactly two incident faces')
    (first, u, v, a), (second, vv, uu, b) = owners
    if first in protected or second in protected:
        raise ValueError('source diagonal touches a protected interface face')
    if (u, v) != (uu, vv) or len({u, v, a, b}) != 4:
        raise ValueError('source diagonal winding or quadrilateral is invalid')
    label = source_facet_labels[first]
    if label is None or label != source_facet_labels[second]:
        raise ValueError('source diagonal crosses original facet identities')
    if any(a in face and b in face for face in source):
        raise ValueError('replacement source diagonal already exists')
    points = {i: _point(vertices_exact[i]) for i in (u, v, a, b)}
    reference = _cross(_sub(points[v], points[u]), _sub(points[a], points[u]))
    neighbor = _cross(_sub(points[u], points[v]), _sub(points[b], points[v]))
    if not any(reference) or _dot(reference, neighbor) <= 0:
        raise ValueError('source patch is degenerate or oppositely oriented')
    if _dot(reference, _sub(points[b], points[u])) != 0:
        raise ValueError('source quadrilateral is not exactly coplanar')
    replacements = ((a, u, b), (a, b, v))
    normals = [_cross(_sub(points[q[1]], points[q[0]]),
                      _sub(points[q[2]], points[q[0]])) for q in replacements]
    if any(_dot(normal, reference) <= 0 for normal in normals):
        raise ValueError('replacement leaves the convex source patch')
    if tuple(x+y for x, y in zip(*normals)) != tuple(
            x+y for x, y in zip(reference, neighbor)):
        raise ValueError('replacement does not conserve the exact source patch')
    ancestors = set()
    for index in (first, second):
        original = tuple(_integer(i) for i in face_lineage[index])
        if not original or any(i < 0 for i in original):
            raise ValueError('changed source face lacks valid ancestry')
        ancestors.update(original)
    result = source.copy()
    result[[first, second]] = replacements
    provenance = SourceFacetDiagonalFlip(
        (first, second), tuple(tuple(map(int, source[i])) for i in (first, second)),
        replacements, tuple(sorted(ancestors)))
    return result, provenance
