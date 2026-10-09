"""Exact embeddedness and disjointness audit of published cavity geometry.

The executed geometry is the quotient's Float64 metre coordinates. Conversion
with Fraction.from_float preserves every binary coordinate exactly; a common
integer denominator then permits exact orientation and intersection predicates.
No tolerance, capping, vertex movement, physical stepping or overlap repair is
performed. Shared topological faces are never exempted wholesale: intersections
must be confined to their common vertex or edge. Nested closed solids are tested
by exact ray parity after the complete surface intersection test.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import math
import struct

from .cardiac_cavity_geometry import analyze_topology
from .model import ImportError as HumanImportError
from .physiology import canonical

SCHEMA = "HumanPack.cardiac-cavity-intersections.v1"
ALGORITHM = "exact_binary64_integer_triangle_predicates_and_rational_ray_parity_v1"


def require(ok, message):
    if not ok:
        raise HumanImportError("cardiac cavity intersections: " + message)


def _sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _orient2(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def _axes(normal):
    return tuple(k for k in range(3) if k != max(range(3), key=lambda j: abs(normal[j])))


def _signs(numerator, denominator, triangle, normal):
    x, y = _axes(normal)
    return [(b[x]-a[x])*(numerator[y]-denominator*a[y]) -
            (b[y]-a[y])*(numerator[x]-denominator*a[x])
            for a, b in zip(triangle, triangle[1:]+triangle[:1])]


def _inside(signs):
    return min(signs) >= 0 or max(signs) <= 0


_FLOAT32_LATTICE_DENOMINATOR = 1 << 149


def float32_point_lattice_key(point):
    """Return the exact integer-lattice key for a packed Float32 xyz point.

    This is for interfaces exported from Float32 mesh payloads. Converting each
    packed coordinate through ``float`` preserves its value exactly, and the
    common 2**149 denominator is divisible by every finite Float32 denominator.
    Reject binary64-only coordinates so callers cannot accidentally compare a
    different representation from the one the native loader consumes.
    """
    require(len(point) == 3, "Float32 point must have three coordinates")
    result = []
    for value in point:
        coordinate = float(value)
        require(math.isfinite(coordinate), "Float32 point must be finite")
        try:
            packed = struct.pack("<f", coordinate)
        except (OverflowError, struct.error):
            require(False, "point coordinate is outside Float32 range")
        require(struct.unpack("<f", packed)[0] == coordinate,
                "point coordinate is not an exact Float32 value")
        numerator, denominator = coordinate.as_integer_ratio()
        require(_FLOAT32_LATTICE_DENOMINATOR % denominator == 0,
                "Float32 coordinate is not on the common integer lattice")
        result.append(numerator * (_FLOAT32_LATTICE_DENOMINATOR // denominator))
    return tuple(result)


def float32_triangle_lattice_key(vertices, face):
    """Return a winding-independent exact key for a packed Float32 triangle."""
    require(len(face) == 3, "Float32 face must have three vertex indices")
    points = tuple(float32_point_lattice_key(vertices[int(index)]) for index in face)
    require(len(set(points)) == 3, "Float32 face has repeated coordinates")
    return tuple(sorted(points))


def float32_lattice_point_on_triangle(point_numerator, denominator, triangle):
    """Test a rational point against an exact triangle on the Float32 lattice.

    ``point_numerator / denominator`` and ``triangle`` use the integer lattice
    above. The scalar denominator is shared by all three coordinates, matching
    the exact predicate owner used by intersection and source audits.
    """
    require(len(point_numerator) == 3 and len(triangle) == 3,
            "point and triangle must be xyz triples")
    require(isinstance(denominator, int) and denominator > 0,
            "point denominator must be a positive integer scalar")
    normal = _cross(_sub(triangle[1], triangle[0]), _sub(triangle[2], triangle[0]))
    require(any(normal), "Float32 triangle is exactly degenerate")
    if sum(normal[k] * (triangle[0][k] * denominator - point_numerator[k])
           for k in range(3)) != 0:
        return False
    return _inside(_signs(point_numerator, denominator, triangle, normal))


def _coplanar_points(first, second, normal):
    axes = _axes(normal)
    a, b = [tuple(v[k] for k in axes) for v in first], [tuple(v[k] for k in axes) for v in second]
    def inside2(p, tri):
        return _inside([_orient2(tri[i], tri[(i+1) % 3], p) for i in range(3)])
    points = [first[i] for i, p in enumerate(a) if inside2(p, b)]
    points.extend(second[i] for i, p in enumerate(b) if inside2(p, a))
    # Collinear contact endpoints are included above. Strict edge crossings
    # supply every remaining vertex of the convex intersection polygon.
    for i in range(3):
        x, y = a[i], a[(i+1) % 3]
        for j in range(3):
            c, d = b[j], b[(j+1) % 3]
            u, v = _orient2(c, d, x), _orient2(c, d, y)
            r, s = _orient2(x, y, c), _orient2(x, y, d)
            if u*v < 0 and r*s < 0:
                t = Fraction(u, u-v)
                points.append(tuple(first[i][k]+t*(first[(i+1) % 3][k]-first[i][k]) for k in range(3)))
    return points


def _segment_triangle(p, q, triangle, normal):
    dp, dq = _dot(normal, _sub(p, triangle[0])), _dot(normal, _sub(q, triangle[0]))
    if (dp > 0 and dq > 0) or (dp < 0 and dq < 0) or dp == dq:
        return []
    denominator = dp-dq
    numerator = tuple(dp*q[k]-dq*p[k] for k in range(3))
    if denominator < 0:
        denominator, numerator = -denominator, tuple(-x for x in numerator)
    if not _inside(_signs(numerator, denominator, triangle, normal)):
        return []
    return [tuple(Fraction(x, denominator) for x in numerator)]


def triangle_intersection_points(first, second):
    """Convex-intersection vertices for two nondegenerate integer triangles."""
    na = _cross(_sub(first[1], first[0]), _sub(first[2], first[0]))
    nb = _cross(_sub(second[1], second[0]), _sub(second[2], second[0]))
    require(any(na) and any(nb), "degenerate triangle")
    sa = [_dot(na, _sub(p, first[0])) for p in second]
    sb = [_dot(nb, _sub(p, second[0])) for p in first]
    if min(sa) > 0 or max(sa) < 0 or min(sb) > 0 or max(sb) < 0:
        return []
    if not any(sa):
        return _coplanar_points(first, second, na)
    points = []
    for i in range(3):
        points.extend(_segment_triangle(first[i], first[(i+1) % 3], second, nb))
        points.extend(_segment_triangle(second[i], second[(i+1) % 3], first, na))
    return points


def _records(vertices, faces):
    records = []
    for index, ids in enumerate(faces):
        tri = tuple(vertices[k] for k in ids)
        require(any(_cross(_sub(tri[1], tri[0]), _sub(tri[2], tri[0]))), "exactly degenerate triangle")
        records.append((tri, tuple(min(v[k] for v in tri) for k in range(3)),
                        tuple(max(v[k] for v in tri) for k in range(3)), index, tuple(ids)))
    return records


def _allowed_shared_point(point, common):
    if point in common:
        return True
    if len(common) == 2:
        a, b = tuple(common)
        return _cross(_sub(point, a), _sub(b, a)) == (0, 0, 0) and all(
            min(a[k], b[k]) <= point[k] <= max(a[k], b[k]) for k in range(3))
    return False


@dataclass(frozen=True)
class _PreparedSurfaceAABB:
    records: tuple
    root: tuple | None


def _prepare_surface_aabb(records):
    """Snapshot exact immutable records and build the existing closed-box tree."""
    records = tuple(records)

    def tree(positions):
        lo = tuple(min(records[i][1][k] for i in positions) for k in range(3))
        hi = tuple(max(records[i][2][k] for i in positions) for k in range(3))
        if len(positions) <= 8:
            return lo, hi, tuple(positions), None, None
        axis = max(range(3), key=lambda k: hi[k]-lo[k])
        positions.sort(key=lambda i: records[i][1][axis]+records[i][2][axis])
        middle = len(positions)//2
        return lo, hi, None, tree(positions[:middle]), tree(positions[middle:])

    return _PreparedSurfaceAABB(
        records, tree(list(range(len(records)))) if records else None)


def _query_surface_aabb(queries, index):
    if index.root is None:
        return
    for row in queries:
        lo, hi = row[1:3]
        stack = [index.root]
        while stack:
            lower, upper, positions, left, right = stack.pop()
            if any(hi[k] < lower[k] or upper[k] < lo[k] for k in range(3)):
                continue
            if positions is None:
                stack.extend((right, left))
                continue
            for position in positions:
                other = index.records[position]
                if all(hi[k] >= other[1][k] and other[2][k] >= lo[k] for k in range(3)):
                    yield row, other


def _aabb_candidate_pairs(first, second, *, same_surface):
    """Enumerate closed-box overlaps with exact bounds and source face IDs."""
    if not first or not second:
        return
    for row, other in _query_surface_aabb(first, _prepare_surface_aabb(second)):
        if not same_surface or other[3] > row[3]:
            yield row, other


def _audit_pair_prepared_first(first_index, second, *, same_surface=False):
    """Reuse a first-surface index across targets without changing predicates.

    Querying the smaller target avoids repeatedly walking every skin face.
    Restore the original first/second argument order before the exact narrow
    phase, including face IDs and shared-index classification. The index owns
    a tuple snapshot, so replacing entries in a caller's list cannot stale it.
    """
    candidates = (
        (first, other)
        for other, first in _query_surface_aabb(second, first_index)
        if not same_surface or other[3] > first[3]
    )
    return _audit_record_pairs(candidates, same_surface=same_surface)


def _audit_pair(first, second, *, same_surface):
    return _audit_record_pairs(
        _aabb_candidate_pairs(first, second, same_surface=same_surface),
        same_surface=same_surface)


def _audit_record_pairs(candidate_pairs, *, same_surface):
    candidates, allowed, pairs = 0, 0, []
    for (tri, lo, hi, i, ids), (other, lower, upper, j, other_ids) in candidate_pairs:
        candidates += 1
        points = triangle_intersection_points(tri, other)
        if not points:
            continue
        # Use shared *indices*, not coincident but topologically unrelated
        # coordinates. Duplicate triangles are defects even when coincident.
        shared_ids = set(ids) & set(other_ids) if same_surface else set()
        common = {tri[ids.index(index)] for index in shared_ids}
        if same_surface and len(shared_ids) in (1, 2) and all(_allowed_shared_point(p, common) for p in points):
            allowed += 1
        else:
            pairs.append([i, j])
    return {"triangle_pairs": sorted(pairs), "count": len(pairs), "aabb_candidate_pairs": candidates,
            "allowed_shared_vertex_or_edge_pairs": allowed}


def point_location(point, records):
    """Exact parity for a closed embedded surface; unresolved rays fail closed."""
    normals = [_cross(_sub(row[0][1], row[0][0]), _sub(row[0][2], row[0][0])) for row in records]
    for row, normal in zip(records, normals):
        tri = row[0]
        if _dot(normal, _sub(point, tri[0])) == 0 and _inside(_signs(point, 1, tri, normal)):
            return {"location": "boundary", "ray": None, "crossings": None}
    for attempt in range(1, 65):
        direction = (1, attempt, attempt*attempt)
        crossings, ambiguous = 0, False
        for row, normal in zip(records, normals):
            tri = row[0]
            numerator, denominator = _dot(normal, _sub(tri[0], point)), _dot(normal, direction)
            if denominator == 0:
                if numerator == 0:
                    ambiguous = True
                    break
                continue
            if numerator*denominator <= 0:
                continue
            hit = tuple(point[k]*denominator+numerator*direction[k] for k in range(3))
            if denominator < 0:
                denominator, hit = -denominator, tuple(-v for v in hit)
            signs = _signs(hit, denominator, tri, normal)
            if _inside(signs):
                if 0 in signs:
                    ambiguous = True
                    break
                crossings += 1
        if not ambiguous:
            return {"location": "inside" if crossings % 2 else "outside", "ray": list(direction),
                    "crossings": crossings, "attempts": attempt}
    return {"location": "indeterminate", "ray": None, "crossings": None, "attempts": 64}


def audit_cavity_intersections(cavities: dict) -> dict:
    """Audit quotient surfaces without trusting cached source topology claims."""
    require(isinstance(cavities, dict) and isinstance(cavities.get("chambers"), list) and
            1 <= len(cavities["chambers"]) <= 64, "invalid cavity collection")
    meshes, denominator = {}, 1
    for chamber in cavities["chambers"]:
        require(isinstance(chamber, dict) and isinstance(chamber.get("source_id"), str) and
                chamber["source_id"] and chamber["source_id"] not in meshes, "missing or duplicate surface ID")
        q = chamber.get("exact_coordinate_quotient")
        require(isinstance(q, dict), "missing exact-coordinate quotient")
        v, f = q.get("vertices_m"), q.get("triangles")
        require(isinstance(v, list) and 3 <= len(v) <= 20000 and isinstance(f, list) and 1 <= len(f) <= 40000,
                "empty or oversized mesh")
        require(all(isinstance(p, (list, tuple)) and len(p) == 3 and all(type(x) in (int, float) and
                    abs(x) <= 1e6 and math.isfinite(x) for x in p) for p in v), "invalid Float64 metre coordinates")
        topology = analyze_topology(v, f)
        rational = [tuple(Fraction.from_float(float(x)) for x in p) for p in v]
        for p in rational:
            for x in p:
                denominator = max(denominator, x.denominator)  # All denominators are powers of two.
        meshes[chamber["source_id"]] = (v, f, rational, topology)
    integer, records, surfaces = {}, {}, {}
    for name in sorted(meshes):
        v, f, rational, topology = meshes[name]
        points = [tuple(x.numerator*(denominator//x.denominator) for x in p) for p in rational]
        integer[name], records[name] = points, _records(points, f)
        report = _audit_pair(records[name], records[name], same_surface=True)
        # Containment samples vertex zero, so every published vertex must be
        # part of the connected surface. An unused point is not a domain witness.
        connected_closed = (topology["closed_oriented_manifold_candidate"]
                            and not topology["unused_vertex_ids"]
                            and topology["face_component_count"] == 1)
        report.update(closed_connected_oriented_manifold=connected_closed,
                      embedded_closed_surface=connected_closed and report["count"] == 0,
                      geometry_sha256=hashlib.sha256(canonical({"vertices_m": v, "triangles": f})).hexdigest())
        surfaces[name] = report
    pairs = []
    for first, second in itertools.combinations(sorted(meshes), 2):
        report = _audit_pair(records[first], records[second], same_surface=False)
        if not surfaces[first]["embedded_closed_surface"] or not surfaces[second]["embedded_closed_surface"]:
            containment = {"status": "not_checked_invalid_surface"}
        elif report["count"]:
            containment = {"status": "not_checked_intersecting_surfaces"}
        else:
            a = point_location(integer[first][0], records[second])
            b = point_location(integer[second][0], records[first])
            containment = {"status": "checked", "first_in_second": a, "second_in_first": b,
                           "representative_vertex_index": 0}
        disjoint = containment.get("status") == "checked" and all(
            containment[k]["location"] == "outside" for k in ("first_in_second", "second_in_first"))
        report.update(first=first, second=second, containment=containment, disjoint_closed_domains=disjoint)
        pairs.append(report)
    return {"schema": SCHEMA, "algorithm": ALGORITHM,
            "coordinate_semantics": "exact_rational_value_of_published_binary64_metres",
            "source_coordinates_modified": False, "added_faces": 0, "overlap_repair": False,
            "per_surface": surfaces, "per_pair": pairs,
            "all_surfaces_embedded": all(r["embedded_closed_surface"] for r in surfaces.values()),
            "all_domains_disjoint": all(r["embedded_closed_surface"] for r in surfaces.values()) and
                                    all(r["disjoint_closed_domains"] for r in pairs),
            "qualification": "geometric_intersection_audit_only"}
