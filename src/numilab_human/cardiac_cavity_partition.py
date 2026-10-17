"""Conservative, offline Boolean ownership candidates for source cardiac cavities.

The source binary64 metre geometry is interpreted exactly. A common rational
surface arrangement preserves the source union and each exclusive region; only
the shared region receives alternative ownership. This is geometric authoring,
not an anatomical valve selection, mass attribution, or physical stepping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from . import cardiac_cavity_intersections as predicates
from . import cardiac_partition_certificate as certificate
from .cardiac_cavity_geometry import extract_cavity_surfaces
from .cardiac_face_arrangement import subdivide_triangle
from .model import ImportError as HumanImportError
from .physiology import canonical

NAMES = ("right_atrium", "right_ventricle")
ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "HumanPack.cardiac-cavity-partition-candidates.v1"


def require(ok, message):
    if not ok:
        raise HumanImportError("cardiac cavity partition: " + message)


def orient_triangle_corner_payload(vertices, corner_payload, *, reverse=False):
    """Reorder a triangle and every per-corner payload as one operation.

    ``corner_payload[i]`` must describe ``vertices[i]``. Reversing a triangle
    changes its winding with permutation ``(0, 2, 1)``; applying that same
    permutation to barycentrics, normals, or other corner-owned records keeps
    their source correspondence intact, including when the triangle was copied
    directly from an existing mesh.
    """
    require(len(vertices) == 3 and len(corner_payload) == 3,
            "triangle and corner payload must both have three entries")
    require(type(reverse) is bool, "triangle orientation flag must be boolean")
    order = (0, 2, 1) if reverse else (0, 1, 2)
    return (tuple(vertices[index] for index in order),
            tuple(corner_payload[index] for index in order))


def rational_source_surfaces(cavities):
    """Take exact rational values of the published, verified quotient vertices."""
    selected = {row["source_id"]: row for row in cavities["chambers"] if row["source_id"] in NAMES}
    require(set(selected) == set(NAMES), "missing right cardiac cavity source")
    return {name: {"vertices": [tuple(Fraction.from_float(float(x)) for x in p)
                               for p in selected[name]["exact_coordinate_quotient"]["vertices_m"]],
                   "triangles": selected[name]["exact_coordinate_quotient"]["triangles"],
                   "source_sha256": selected[name]["source"]["sha256"]}
            for name in NAMES}


class IntegerRayClassifier:
    """Exact source point classification through the normalized predicate owner."""

    def __init__(self, records):
        self.prepared = predicates.prepare_point_location(records)

    def location(self, point):
        result = predicates.point_location_prepared_indexed(point, self.prepared)
        require(result["location"] != "indeterminate",
                "indeterminate exact ray classification")
        return result["location"]



def _on_edge(point, a, b):
    return (predicates._cross(predicates._sub(point, a), predicates._sub(b, a)) == (0, 0, 0)
            and all(min(a[k], b[k]) <= point[k] <= max(a[k], b[k]) for k in range(3)))


def _validate_declared_feature_pair(first, second, kind):
    """Validate that a caller-declared shared feature is the entire exact contact."""
    common = set(first) & set(second)
    points = set(predicates.triangle_intersection_points(first, second))
    if kind == "shared_vertex":
        require(len(common) == 1 and points == common,
                "declared shared vertex has additional or missing exact contact")
        return
    if kind == "shared_edge":
        require(len(common) == 2 and points == common,
                "declared shared edge has additional or missing exact contact")
        return
    if kind == "opposed_full_face":
        require(len(common) == 3 and set(first) == set(second),
                "declared opposed full face is not an identical full triangle")
        n_first, n_second = _exact_triangle_normal(first), _exact_triangle_normal(second)
        require(not any(predicates._cross(n_first, n_second))
                and predicates._dot(n_first, n_second) < 0,
                "declared full-face pair is not oppositely oriented")
        require(points == common,
                "declared opposed full face has additional or missing exact contact")
        return
    require(False, "unsupported declared shared feature kind")


def _coplanar_overlap_polygon(first, second, normal):
    """Return the exact convex boundary of a positive-area coplanar overlap."""
    points = set(predicates._coplanar_points(first, second, normal))
    if len(points) < 3:
        return ()
    dropped = max(range(3), key=lambda axis: abs(normal[axis]))
    axes = tuple(axis for axis in range(3) if axis != dropped)
    by_projection = {tuple(point[axis] for axis in axes): tuple(point) for point in points}
    projected = sorted(by_projection)

    def orient(a, b, c):
        return ((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))

    lower = []
    for point in projected:
        while len(lower) >= 2 and orient(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(projected):
        while len(upper) >= 2 and orient(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    corners = lower[:-1] + upper[:-1]
    if len(corners) < 3:
        return ()

    # Preserve any exact collinear boundary vertices. They are constraints in
    # the source mesh and must not disappear from the local subdivision.
    polygon = []
    for start, end in zip(corners, corners[1:] + corners[:1]):
        axis = 0 if start[0] != end[0] else 1
        on_edge = [point for point in projected
                   if orient(start, end, point) == 0
                   and all(min(a, b) <= value <= max(a, b)
                           for value, a, b in zip(point, start, end))]
        on_edge.sort(key=lambda point: (point[axis]-start[axis])/(end[axis]-start[axis]))
        polygon.extend(on_edge[:-1])
    area2 = sum(a[0]*b[1]-a[1]*b[0]
                for a, b in zip(polygon, polygon[1:]+polygon[:1]))
    if area2 == 0:
        return ()
    return tuple(by_projection[point] for point in polygon)


def _point_in_convex_polygon(point, polygon, normal):
    dropped = max(range(3), key=lambda axis: abs(normal[axis]))
    axes = tuple(axis for axis in range(3) if axis != dropped)
    p = tuple(point[axis] for axis in axes)
    projected = [tuple(vertex[axis] for axis in axes) for vertex in polygon]
    signs = [((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]))
             for a, b in zip(projected, projected[1:]+projected[:1])]
    return min(signs) >= 0 or max(signs) <= 0


def construct_arrangement(source_surfaces, *, source_names=NAMES, declared_feature_pairs=None,
                          coplanar_priority_pairs=()):
    """Return source-parented exact subtriangles and intersection provenance.

    This bounded authoring path supports transverse intersections with segment
    endpoints, including point pairs supported by nonzero incident-star cuts.
    Callers may provide exact shared-feature face pairs; each declaration is
    checked against the full exact contact before it is excluded from cutting.
    Explicit ``coplanar_priority_pairs`` opt a face pair into an exact local
    subdivision of its positive-area, oppositely oriented overlap. Both copies
    of that overlap are classified as the shared inside interface; all other
    undeclared coplanar or tangent contacts still fail closed.
    Opposed full faces are retained unchanged as outside boundary faces on both
    sources. Any undeclared coplanar patch, isolated tangency, or unsupported
    planar cell topology still fails closed. Input surfaces must subsequently
    pass the independent certificate.
    """
    source_names = tuple(source_names)
    require(len(source_names) == 2 and all(isinstance(name, str) and name for name in source_names)
            and len(set(source_names)) == 2 and isinstance(source_surfaces, dict)
            and set(source_surfaces) == set(source_names), "expected two distinct named source surfaces")
    denominator = math.lcm(*(Fraction(x).denominator for surface in source_surfaces.values()
                            for point in surface["vertices"] for x in point))
    vertices = {name: [tuple(Fraction(x).numerator*(denominator//Fraction(x).denominator) for x in point)
                       for point in source_surfaces[name]["vertices"]] for name in source_names}
    records = {name: predicates._records(vertices[name], source_surfaces[name]["triangles"]) for name in source_names}
    if declared_feature_pairs is None:
        declared_feature_pairs = {}
    require(isinstance(declared_feature_pairs, dict),
            "declared shared features must be a face-pair mapping")
    normalized_declarations = {}
    for raw_pair, kind in declared_feature_pairs.items():
        require(isinstance(raw_pair, tuple) and len(raw_pair) == 2
                and all(isinstance(face, int) and not isinstance(face, bool) and face >= 0
                        for face in raw_pair),
                "declared shared feature has an invalid face pair")
        require(raw_pair not in normalized_declarations
                and kind in ("shared_vertex", "shared_edge", "opposed_full_face"),
                "declared shared feature has a duplicate pair or invalid kind")
        normalized_declarations[raw_pair] = kind
    normalized_coplanar = set()
    require(isinstance(coplanar_priority_pairs, (tuple, list, set, frozenset)),
            "coplanar priority faces must be a finite face-pair collection")
    for raw_pair in coplanar_priority_pairs:
        require(isinstance(raw_pair, (tuple, list)) and len(raw_pair) == 2
                and all(isinstance(face, int) and not isinstance(face, bool) and face >= 0
                        for face in raw_pair),
                "coplanar priority face pair is invalid")
        pair = tuple(raw_pair)
        require(pair not in normalized_coplanar and pair not in normalized_declarations,
                "coplanar priority pair is duplicated or already declared as a shared feature")
        normalized_coplanar.add(pair)
    pair_audit = predicates._audit_pair(records[source_names[0]], records[source_names[1]], same_surface=False)
    segments = {name: defaultdict(list) for name in source_names}
    boundary_points = {name: defaultdict(set) for name in source_names}
    edge_faces = {name: defaultdict(list) for name in source_names}
    vertex_faces = {name: defaultdict(set) for name in source_names}
    point_contacts = []
    declared_seen = set()
    coplanar_seen = set()
    coplanar_patches = {name: {} for name in source_names}
    declared_full_faces = {name: set() for name in source_names}
    declared_rows = []
    for name in source_names:
        for face_id, face in enumerate(source_surfaces[name]["triangles"]):
            for vertex in face:
                vertex_faces[name][vertices[name][vertex]].add(face_id)
            for i in range(3):
                edge_faces[name][tuple(sorted((face[i], face[(i+1) % 3])))].append(face_id)
    intersection_segments = []
    for pair in pair_audit["triangle_pairs"]:
        pair_key = tuple(map(int, pair))
        first, second = (records[name][face_id][0] for name, face_id in zip(source_names, pair))
        declared_kind = normalized_declarations.get(pair_key)
        if declared_kind is not None:
            _validate_declared_feature_pair(first, second, declared_kind)
            declared_seen.add(pair_key)
            declared_rows.append({"source_faces": list(pair_key), "feature": declared_kind})
            if declared_kind == "opposed_full_face":
                for name, face_id in zip(source_names, pair_key):
                    require(face_id not in declared_full_faces[name],
                            "a source full face is declared against multiple shared faces")
                    declared_full_faces[name].add(face_id)
            continue
        normals = [predicates._cross(predicates._sub(tri[1], tri[0]), predicates._sub(tri[2], tri[0]))
                   for tri in (first, second)]
        if not any(predicates._cross(*normals)):
            require(pair_key in normalized_coplanar,
                    "coplanar or tangent face intersection is unsupported")
            require(predicates._dot(normals[0], normals[1]) < 0,
                    "coplanar priority overlap faces must have opposite winding")
            polygon = _coplanar_overlap_polygon(first, second, normals[0])
            require(len(polygon) >= 3,
                    "coplanar priority pair has no positive-area overlap polygon")
            for name, face_id in zip(source_names, pair_key):
                require(face_id not in coplanar_patches[name],
                        "a source face participates in multiple coplanar priority patches")
                coplanar_patches[name][face_id] = (polygon, normals[0])
                face = source_surfaces[name]["triangles"][face_id]
                for a, b in zip(polygon, polygon[1:]+polygon[:1]):
                    segments[name][face_id].append((a, b))
                for point in polygon:
                    for i in range(3):
                        a, b = face[i], face[(i+1) % 3]
                        if _on_edge(point, vertices[name][a], vertices[name][b]):
                            for neighbor in edge_faces[name][tuple(sorted((a, b)))]:
                                boundary_points[name][neighbor].add(point)
            for a, b in zip(polygon, polygon[1:]+polygon[:1]):
                intersection_segments.append({"source_faces": list(pair), "endpoints": (a, b)})
            coplanar_seen.add(pair_key)
            continue
        require(pair_key not in normalized_coplanar,
                "declared coplanar priority pair is not coplanar")
        points = sorted(set(predicates.triangle_intersection_points(first, second)))
        if len(points) == 1:
            point_contacts.append((tuple(pair), points[0]))
            continue
        require(len(points) >= 2, "empty face contact is unsupported")
        a, b = points[0], points[-1]
        require(a != b and all(_on_edge(p, a, b) for p in points), "intersection is not one exact segment")
        intersection_segments.append({"source_faces": list(pair), "endpoints": (a, b)})
        for name, face_id in zip(source_names, pair):
            segments[name][face_id].append((a, b))
            face = source_surfaces[name]["triangles"][face_id]
            for i in range(3):
                edge = tuple(sorted((face[i], face[(i+1) % 3])))
                for point in points:
                    if _on_edge(point, vertices[name][edge[0]], vertices[name][edge[1]]):
                        for neighbor in edge_faces[name][edge]:
                            boundary_points[name][neighbor].add(point)
    require(declared_seen == set(normalized_declarations),
            "a declared shared feature is absent from the exact intersurface contact set")
    require(coplanar_seen == normalized_coplanar,
            "a declared coplanar priority pair is absent from the exact intersurface contact set")
    # Refinement can split a transverse cut exactly at a source vertex.
    # A pair meeting only at that vertex needs no zero-length cut. It must,
    # however, be supported by a nonzero segment in each incident source star.
    # An isolated point/tangent never supplies this evidence.
    for pair, point in point_contacts:
        for name, face_id in zip(source_names, pair):
            triangle = records[name][face_id][0]
            star = {face_id}
            on_boundary = False
            face = source_surfaces[name]["triangles"][face_id]
            for i in range(3):
                a, b = face[i], face[(i+1) % 3]
                if _on_edge(point, vertices[name][a], vertices[name][b]):
                    on_boundary = True
                    star.update(edge_faces[name][tuple(sorted((a, b)))])
            if point in vertex_faces[name]:
                star.update(vertex_faces[name][point])
            own_cut = any(_on_edge(point, a, b)
                          for a, b in segments[name][face_id])
            supported = own_cut or (on_boundary and any(
                _on_edge(point, a, b) for neighbor in star
                for a, b in segments[name][neighbor]))
            require(supported,
                    f"point-only face contact is unsupported: {pair} {name}:{face_id}")
            # Propagate any edge point to both incident faces exactly.
            for i in range(3):
                a, b = face[i], face[(i+1) % 3]
                if _on_edge(point, vertices[name][a], vertices[name][b]):
                    for neighbor in edge_faces[name][tuple(sorted((a, b)))]:
                        boundary_points[name][neighbor].add(point)
    classifiers = {name: IntegerRayClassifier(records[name]) for name in source_names}
    result = []
    for name, other in (source_names, tuple(reversed(source_names))):
        for face_id, row in enumerate(records[name]):
            if face_id in declared_full_faces[name]:
                result.append({"vertices": tuple(tuple(Fraction(x, denominator) for x in point)
                                                  for point in row[0]),
                               "source": name, "source_face": face_id,
                               "other_location": "outside"})
                continue
            children = subdivide_triangle(row[0], segments[name][face_id], boundary_points[name][face_id])
            for triangle in children:
                centroid = tuple(sum(p[k] for p in triangle)/3 for k in range(3))
                patch = coplanar_patches[name].get(face_id)
                if patch is not None and _point_in_convex_polygon(centroid, patch[0], patch[1]):
                    location = "inside"
                else:
                    location = classifiers[other].location(centroid)
                require(location in ("inside", "outside"), f"boundary child interior {name}:{face_id}")
                result.append({"vertices": tuple(tuple(Fraction(x, denominator) for x in p) for p in triangle),
                               "source": name, "source_face": face_id, "other_location": location})
    declared_rows.sort(key=lambda row: (row["source_faces"], row["feature"]))
    declaration_bytes = json.dumps(declared_rows, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8")
    declared_counts = Counter(row["feature"] for row in declared_rows)
    return result, {"source_intersecting_triangle_pairs": pair_audit["triangle_pairs"],
                    "source_intersecting_triangle_pair_count": pair_audit["count"],
                    "declared_boundary_feature_pairs": declared_rows,
                    "declared_boundary_feature_pair_count": len(declared_rows),
                    "declared_boundary_feature_counts": dict(sorted(declared_counts.items())),
                    "declared_boundary_feature_pairs_sha256": hashlib.sha256(declaration_bytes).hexdigest(),
                    "coplanar_priority_overlap_pairs": [{"source_faces": list(pair),
                        "overlap_polygon_m": [tuple(Fraction(x, denominator) for x in point)
                                              for point in coplanar_patches[source_names[0]][pair[0]][0]]}
                        for pair in sorted(coplanar_seen)],
                    "coplanar_priority_overlap_pair_count": len(coplanar_seen),
                    "supported_point_contact_count": len(point_contacts),
                    "common_integer_scale_per_metre": denominator,
                    "intersection_segments_m": [{"source_faces": row["source_faces"],
                        "endpoints": tuple(tuple(Fraction(x, denominator) for x in point) for point in row["endpoints"])}
                        for row in intersection_segments],
                    "source_face_count": sum(len(source_surfaces[n]["triangles"]) for n in source_names),
                    "subtriangle_count": len(result)}


def _exact_triangle_normal(triangle):
    return predicates._cross(predicates._sub(triangle[1], triangle[0]),
                             predicates._sub(triangle[2], triangle[0]))


class _NormalProbeInSourcePlane(HumanImportError):
    """An otherwise valid normal line needs a different face-interior origin."""


def _bounded_normal_probe(point, normal, records):
    """Choose a rational normal offset before the next nonzero source-plane crossing."""
    nearest = None
    for triangle, _lower, _upper, _face_id, _ids in records:
        other_normal = _exact_triangle_normal(triangle)
        denominator = predicates._dot(other_normal, normal)
        numerator = predicates._dot(other_normal, predicates._sub(triangle[0], point))
        if denominator == 0:
            if numerator == 0:
                raise _NormalProbeInSourcePlane(
                    "cardiac cavity partition: normal probe line lies in another source face plane")
            continue
        crossing = Fraction(numerator, denominator)
        if crossing == 0:
            continue
        distance = abs(crossing)
        nearest = distance if nearest is None else min(nearest, distance)
    require(nearest is not None and nearest > 0,
            "no bounded nonzero source-plane crossing for normal probes")
    return nearest / 2


def _triangle_normal_probe(triangle, records):
    """Choose an exact interior origin whose normal line avoids source planes.

    A child-face centroid can lie on an unrelated perpendicular source plane.
    Positive barycentric weights (1,t,t*t) give a rational interior curve. Each
    forbidden plane excludes at most two integer parameters: otherwise that
    plane contains this nondegenerate triangle, contradicting perpendicular
    normals. Thus 2*N+1 candidates suffice; no geometric tolerance is changed.
    """
    normal = _exact_triangle_normal(triangle)
    require(any(normal), "normal probe requires a nondegenerate triangle")
    for parameter in range(1, 2 * len(records) + 2):
        weights = (1, parameter, parameter * parameter)
        total = sum(weights)
        point = tuple(sum((weight * vertex[axis]
                           for weight, vertex in zip(weights, triangle)), Fraction(0)) / total
                      for axis in range(3))
        try:
            step = _bounded_normal_probe(point, normal, records)
        except _NormalProbeInSourcePlane:
            continue
        return point, normal, step, parameter
    raise HumanImportError(
        "cardiac cavity partition: no source-plane-safe rational interior probe")


def _point_on_segment(point, segment):
    return _on_edge(point, segment[0], segment[1])


def _oriented_surface_topology(records):
    """Collect exact child half-edges before distinguishing arrangement cuts."""
    edge_uses = defaultdict(list)
    for child_id, (triangle, _lower, _upper, _face_id, ids) in enumerate(records):
        for a, b in zip(ids, ids[1:] + ids[:1]):
            edge_uses[tuple(sorted((a, b)))].append((child_id, a, b))
    return edge_uses


def construct_nonzero_winding_self_union(source_surface, *, allow_coplanar_overlap=False):
    """Extract the exact boundary of the nonzero signed-winding set of one shell.

    The strict default supports transverse segment-valued self-intersections.
    The opt-in coplanar overlap option uses the joint arrangement and complete
    parent support described by construct_positive_winding_self_union, while
    retaining the nonzero material rule. The arrangement splits every incident
    source face, classifies each connected
    ordinary-face patch using exact probes before the next source-plane
    crossing, and retains patches separating zero from nonzero winding.
    Unsupported point-only, unresolved, or nonmanifold arrangements fail closed.
    This is a rational source-space result; Float32 conversion requires a
    separate audit and is not implied here.
    """
    return _construct_winding_self_union(source_surface, positive_only=False,
                                         allow_coplanar_overlap=allow_coplanar_overlap)


def construct_positive_winding_self_union(source_surface, *, point_contact_cover_face_indices=None,
                                          allow_coplanar_overlap=False):
    """Extract the boundary of the strictly positive signed-winding material.

    This is an explicit inference for an outward-oriented reference shell with
    inverted exterior folds, not a substitute for the nonzero material rule.
    Negative exterior lobes are excluded; opposite-oriented cavities inside
    positive material remain cavities. A globally reversed shell is rejected.
    With allow_coplanar_overlap=True, coincident source cells are arranged in
    the same pass as transverse cuts and classified against the original closed
    source. Their exact signed sheet multiplicity determines the winding jump;
    the emitted boundary retains every supporting source face. Boundary-only
    point contacts may be subdivided, but singular output contacts still fail.
    Exact arrangement, manifold, self-intersection and side-probe checks remain
    mandatory. Attribute ancestry and Float32 conversion need separate audits.

    The optional cover names one closed, outward, self-clear input component.
    A point-only source contact may be internal to this explicitly inferred
    repair solid. Exact strict containment is required at every such point;
    boundary/external contacts still fail. This only enables construction of
    the regularized material boundary: the emitted surface must still pass all
    exact self, manifold and winding checks. Default behavior remains strict.
    """
    return _construct_winding_self_union(
        source_surface, positive_only=True,
        point_contact_cover_face_indices=point_contact_cover_face_indices,
        allow_coplanar_overlap=allow_coplanar_overlap)


def _construct_winding_self_union(source_surface, *, positive_only,
                                  point_contact_cover_face_indices=None,
                                  allow_coplanar_overlap=False):
    require(type(allow_coplanar_overlap) is bool, "coplanar overlap option must be boolean")
    require(isinstance(source_surface, dict), "expected one named source surface")
    vertices = source_surface.get("vertices")
    faces = source_surface.get("triangles")
    source_sha = source_surface.get("source_sha256")
    require(isinstance(vertices, (list, tuple)) and vertices and
            isinstance(faces, list) and faces and
            isinstance(source_sha, str) and len(source_sha) == 64 and
            all(c in "0123456789abcdef" for c in source_sha),
            "invalid source surface identity or geometry")
    exact_vertices = []
    for point in vertices:
        require(isinstance(point, (list, tuple)) and len(point) == 3 and
                all(type(value) is int or isinstance(value, Fraction) for value in point),
                "source coordinates must be exact integer or Fraction triples")
        exact_vertices.append(tuple(Fraction(value) for value in point))
    require(all(isinstance(face, (list, tuple)) and len(face) == 3 and
                all(type(index) is int and 0 <= index < len(exact_vertices) for index in face)
                for face in faces), "invalid source triangle indices")

    records = predicates._records(exact_vertices, faces)
    prepared = predicates.prepare_signed_winding(records)
    point_cover = None
    covered_point_contacts = []
    point_contact_arrangements = []
    if point_contact_cover_face_indices is not None:
        selected = point_contact_cover_face_indices
        require(isinstance(selected, (list, tuple)) and bool(selected) and
                all(type(index) is int and 0 <= index < len(faces) for index in selected)
                and len(set(selected)) == len(selected),
                "point-contact cover requires distinct valid input face indices")
        cover_records = [records[index] for index in selected]
        point_cover = predicates.prepare_signed_winding(cover_records)
        require(point_cover.face_component_count == 1,
                "point-contact cover must be one closed component")
        require(predicates._audit_pair(cover_records, cover_records, same_surface=True)["count"] == 0,
                "point-contact cover must be self-clear")
        cover_volume6 = sum(predicates._dot(triangle[0],
                               predicates._cross(triangle[1], triangle[2]))
                            for triangle, *_ in cover_records)
        require(cover_volume6 > 0, "point-contact cover must be outward oriented")
    intersection_audit = predicates._audit_pair(records, records, same_surface=True)
    segments_by_face = defaultdict(set)
    boundary_points_by_face = defaultdict(set)
    edge_faces = defaultdict(list)
    for face_id, face in enumerate(faces):
        for a, b in zip(face, face[1:] + face[:1]):
            edge_faces[tuple(sorted((a, b)))].append(face_id)

    segments = []
    for first_id, second_id in intersection_audit["triangle_pairs"]:
        first, second = records[first_id][0], records[second_id][0]
        first_normal, second_normal = (_exact_triangle_normal(first),
                                       _exact_triangle_normal(second))
        points = sorted(set(predicates.triangle_intersection_points(first, second)))
        if len(points) == 1 and point_cover is not None:
            location = predicates.signed_winding_number(points[0], point_cover)
            require(location["status"] == "resolved" and location["winding_number"] == 1,
                    "point-only contact is not strictly inside the declared cover")
            covered_point_contacts.append({"face_pair": [first_id, second_id],
                                           "point": points[0],
                                           "cover_winding_number": 1})
            # No cut is required inside a positive open neighborhood of fill
            # material. Its transverse boundary cuts still partition the source
            # patches; all emitted-boundary predicates below remain mandatory.
            continue
        if len(points) == 1 and allow_coplanar_overlap:
            point = points[0]
            # Conform point contacts in the joint arrangement. A valid exact
            # triangle intersection may be interior to one face and on an edge
            # of the other; the boundary-side point must propagate to adjacent
            # source faces, while the interior face is split by three spokes.
            # This is arrangement support only: retained singular contacts
            # still fail the unchanged emitted self-intersection/topology gates.
            boundary_faces = []
            interior_faces = []
            for face_id, triangle in ((first_id, first), (second_id, second)):
                edges = tuple(zip(triangle, triangle[1:] + triangle[:1]))
                if any(_point_on_segment(point, edge) for edge in edges):
                    boundary_faces.append(face_id)
                    boundary_points_by_face[face_id].add(point)
                    face = faces[face_id]
                    for a, b in zip(face, face[1:] + face[:1]):
                        if _point_on_segment(point, (exact_vertices[a], exact_vertices[b])):
                            for neighbor in edge_faces[tuple(sorted((a, b)))]:
                                boundary_points_by_face[neighbor].add(point)
                else:
                    normal = _exact_triangle_normal(triangle)
                    offset = sum(normal[axis] * (point[axis] - triangle[0][axis])
                                 for axis in range(3))
                    require(any(normal) and offset == 0,
                            "point-only contact is not on an exact nondegenerate face")
                    interior_faces.append((face_id, triangle))
            require(not interior_faces or (len(interior_faces) == 1 and len(boundary_faces) == 1),
                    "point-only contact must meet one face interior and one source boundary")
            contact_record = {
                "face_pair": [first_id, second_id],
                "point": point,
                "boundary_face_ids": sorted(boundary_faces),
                "interior_face_ids": sorted(face_id for face_id, _triangle in interior_faces),
            }
            if interior_faces:
                face_id, triangle = interior_faces[0]
                spokes = []
                for vertex in triangle:
                    require(point != vertex, "interior point-contact spoke has zero length")
                    spoke = (point, vertex)
                    segments_by_face[face_id].add(spoke)
                    spokes.append(spoke)
                contact_record["arrangement"] = "three_interior_face_spokes_plus_boundary_propagation"
                contact_record["interior_spokes"] = spokes
            else:
                contact_record["arrangement"] = "existing_boundary_point_propagation"
                contact_record["interior_spokes"] = []
            point_contact_arrangements.append(contact_record)
            continue
        require(len(points) >= 2,
                "point-only self-contact is unsupported")
        transverse = any(predicates._cross(first_normal, second_normal))
        if transverse:
            start, end = points[0], points[-1]
            require(start != end and all(_point_on_segment(point, (start, end)) for point in points),
                    "self-intersection is not one exact transverse segment")
            cuts = [(start, end)]
        else:
            require(allow_coplanar_overlap, "coplanar self-intersection is unsupported")
            polygon = _coplanar_overlap_polygon(first, second, first_normal)
            # Edge-only contacts still need conforming subdivisions.
            cuts = list(zip(polygon, polygon[1:] + polygon[:1])) if polygon else [(points[0], points[-1])]
        for start, end in cuts:
            require(start != end, "degenerate self-intersection constraint")
            segments.append({"face_pair": [first_id, second_id], "endpoints": (start, end)})
            segments_by_face[first_id].add((start, end))
            segments_by_face[second_id].add((start, end))
            for face_id in (first_id, second_id):
                face = faces[face_id]
                for a, b in zip(face, face[1:] + face[:1]):
                    for neighbor in edge_faces[tuple(sorted((a, b)))]:
                        for point in (start, end):
                            if _point_on_segment(point, (exact_vertices[a], exact_vertices[b])):
                                boundary_points_by_face[neighbor].add(point)

    child_triangles = []
    child_parent = []
    for face_id, face in enumerate(faces):
        triangle = tuple(exact_vertices[index] for index in face)
        children = subdivide_triangle(
            triangle,
            sorted(segments_by_face[face_id]),
            sorted(boundary_points_by_face[face_id]),
        )
        child_triangles.extend(children)
        child_parent.extend([face_id] * len(children))

    child_vertices = sorted({point for triangle in child_triangles for point in triangle})
    child_vertex_id = {point: index for index, point in enumerate(child_vertices)}
    child_faces = [tuple(child_vertex_id[point] for point in triangle)
                   for triangle in child_triangles]
    child_records = predicates._records(child_vertices, child_faces)
    edge_uses = _oriented_surface_topology(child_records)

    cut_edges = set()
    for child_id, triangle in enumerate(child_triangles):
        parent_id = child_parent[child_id]
        parent_segments = segments_by_face[parent_id]
        for a, b in zip(triangle, triangle[1:] + triangle[:1]):
            if any(_point_on_segment(a, segment) and _point_on_segment(b, segment)
                   for segment in parent_segments):
                cut_edges.add(tuple(sorted((child_vertex_id[a], child_vertex_id[b]))))

    if allow_coplanar_overlap:
        # Coincident sheet boundaries can have any balanced even incidence.
        # Do not demand a closed deduplicated intermediate surface: transverse
        # cuts and winding classification must be resolved in this same pass.
        cut_edges.update(edge for edge, uses in edge_uses.items() if len(uses) != 2)
    for edge, uses in edge_uses.items():
        if edge in cut_edges:
            forward = sum(first < second for _child, first, second in uses)
            reverse = len(uses) - forward
            require(len(uses) >= (2 if allow_coplanar_overlap else 4) and len(uses) % 2 == 0 and forward == reverse,
                    "intersection arrangement edge has unbalanced oriented incidence")
        else:
            require(len(uses) == 2, "subdivision did not produce a closed two-face ordinary edge")
            first, second = uses
            require(first[1] == second[2] and first[2] == second[1],
                    "subdivision produced inconsistent ordinary edge orientation")

    adjacency = [set() for _ in child_triangles]
    for edge, uses in edge_uses.items():
        if edge in cut_edges:
            continue
        first, second = uses
        adjacency[first[0]].add(second[0])
        adjacency[second[0]].add(first[0])

    coincident_children = defaultdict(list)
    if allow_coplanar_overlap:
        for child_id, triangle in enumerate(child_triangles):
            coincident_children[tuple(sorted(triangle))].append(child_id)
    patches = []
    unseen = set(range(len(child_triangles)))
    while unseen:
        seed = min(unseen)
        component = []
        pending = [seed]
        while pending:
            child_id = pending.pop()
            if child_id not in unseen:
                continue
            unseen.remove(child_id)
            component.append(child_id)
            pending.extend(adjacency[child_id] & unseen)
        representative_id = min(component)
        triangle = child_triangles[representative_id]
        point, normal, step, probe_parameter = _triangle_normal_probe(triangle, records)
        require(not any(_point_on_segment(point, segment["endpoints"]) for segment in segments),
                "arrangement patch representative lies on an intersection segment")
        minus_point = tuple(point[axis] - step * normal[axis] for axis in range(3))
        plus_point = tuple(point[axis] + step * normal[axis] for axis in range(3))
        minus = predicates.signed_winding_number(minus_point, prepared)
        plus = predicates.signed_winding_number(plus_point, prepared)
        require(minus["status"] == plus["status"] == "resolved",
                "self-union side winding is boundary or indeterminate")
        winding_minus, winding_plus = minus["winding_number"], plus["winding_number"]
        multiplicity = 1
        if allow_coplanar_overlap:
            multiplicity = sum(
                1 if predicates._dot(normal, _exact_triangle_normal(child_triangles[index])) > 0 else -1
                for index in coincident_children[tuple(sorted(triangle))])
            for child_id in component:
                child_normal = _exact_triangle_normal(child_triangles[child_id])
                require(sum(
                    1 if predicates._dot(child_normal, _exact_triangle_normal(child_triangles[index])) > 0 else -1
                    for index in coincident_children[tuple(sorted(child_triangles[child_id]))]) == multiplicity,
                    "ordinary patch crosses a signed sheet multiplicity change")
        require(winding_minus == winding_plus + multiplicity,
                "oriented patch does not separate winding levels by its exact sheet multiplicity")
        inside_minus = winding_minus > 0 if positive_only else winding_minus != 0
        inside_plus = winding_plus > 0 if positive_only else winding_plus != 0
        if inside_minus and not inside_plus:
            keep, reverse = True, False
        elif not inside_minus and inside_plus:
            keep, reverse = True, True
        else:
            keep, reverse = False, False
        patches.append({
            "patch_id": len(patches),
            "child_face_ids": sorted(component),
            "source_face_ids": sorted({child_parent[index] for index in component}),
            "representative_child_face": representative_id,
            "probe_step": step,
            "probe_point": point,
            "probe_candidate_parameter": probe_parameter,
            "winding_minus": winding_minus,
            "winding_plus": winding_plus,
            **({"sheet_multiplicity": multiplicity} if allow_coplanar_overlap else {}),
            "kept": keep,
            "reversed": reverse,
        })

    output_triangles = []
    output_ancestry = []
    retained_patch_probes = []
    for patch in patches:
        if not patch["kept"]:
            continue
        for child_id in patch["child_face_ids"]:
            triangle = child_triangles[child_id]
            output_triangles.append((triangle[0], triangle[2], triangle[1])
                                    if patch["reversed"] else triangle)
            output_ancestry.append({
                "source_face": child_parent[child_id],
                "patch_id": patch["patch_id"],
                "reversed_from_source": patch["reversed"],
            })
        triangle = child_triangles[patch["representative_child_face"]]
        point = tuple(sum(vertex[axis] for vertex in triangle) / 3 for axis in range(3))
        normal = _exact_triangle_normal(triangle)
        if patch["reversed"]:
            normal = tuple(-value for value in normal)
        step = patch["probe_step"]
        retained_patch_probes.append((
            tuple(point[axis] - step * normal[axis] for axis in range(3)),
            tuple(point[axis] + step * normal[axis] for axis in range(3)),
        ))

    if allow_coplanar_overlap:
        grouped_output = {}
        for triangle, ancestry in zip(output_triangles, output_ancestry, strict=True):
            key = tuple(sorted(triangle))
            if key in grouped_output:
                existing, parents = grouped_output[key]
                require(predicates._dot(_exact_triangle_normal(existing),
                                        _exact_triangle_normal(triangle)) > 0,
                        "selected coincident output cells disagree in orientation")
                parents.append(ancestry)
            else:
                grouped_output[key] = (triangle, [ancestry])
        output_triangles, output_ancestry = [], []
        for triangle, origins in grouped_output.values():
            source_faces = sorted({origin["source_face"] for origin in origins})
            ancestry = dict(origins[0])
            ancestry["source_faces"] = source_faces
            ancestry["reversed_from_sources"] = [
                predicates._dot(_exact_triangle_normal(triangle),
                                _exact_triangle_normal(records[index][0])) < 0
                for index in source_faces]
            output_triangles.append(triangle)
            output_ancestry.append(ancestry)
    require(output_triangles, "selected winding material has no boundary")
    output_mesh = indexed_mesh(output_triangles, convert=lambda value: value)
    output_records = predicates._records(output_mesh["vertices_m"], output_mesh["triangles"])
    output_winding = predicates.prepare_signed_winding(output_records)
    output_self_audit = predicates._audit_pair(output_records, output_records, same_surface=True)
    require(output_self_audit["count"] == 0,
            "selected winding boundary still has exact self-intersections")
    output_topology = _validate_output_topology(output_records)
    for inside_point, outside_point in retained_patch_probes:
        inside = predicates.signed_winding_number(inside_point, output_winding)
        outside = predicates.signed_winding_number(outside_point, output_winding)
        require(inside["status"] == outside["status"] == "resolved" and
                inside["winding_number"] == 1 and outside["winding_number"] == 0,
                "emitted patch does not bound exactly the selected winding material")

    result = {
        "source_sha256": source_sha,
        "material_rule": "strictly_positive" if positive_only else "nonzero",
        "input_face_count": len(faces),
        "input_component_count": prepared.face_component_count,
        "input_self_intersection_pairs": intersection_audit["triangle_pairs"],
        "input_self_intersection_count": intersection_audit["count"],
        "exact_intersection_segments": segments,
        "point_contact_arrangements": point_contact_arrangements,
        "subdivided_face_count": len(child_triangles),
        "arrangement_patch_count": len(patches),
        "retained_patch_count": sum(patch["kept"] for patch in patches),
        "reversed_patch_count": sum(patch["reversed"] for patch in patches),
        "patches": patches,
        "output_mesh": output_mesh,
        "output_face_ancestry": output_ancestry,
        "output_topology": output_topology,
        "output_self_intersection_audit": output_self_audit,
        "float32_conversion_audited": False,
    }
    if point_cover is not None:
        result["point_contact_cover_face_indices"] = list(point_contact_cover_face_indices)
        result["covered_point_contacts"] = covered_point_contacts
    return result


def _validate_output_topology(records):
    """Validate output topology and report exact component/edge counts."""
    component_count = predicates._validate_closed_oriented_record_topology(records)
    vertices = {vertex_id for _triangle, _lo, _hi, _face_id, ids in records for vertex_id in ids}
    edges = {tuple(sorted((a, b))) for _triangle, _lo, _hi, _face_id, ids in records
             for a, b in zip(ids, ids[1:] + ids[:1])}
    return {"closed_oriented_2_manifold": True,
            "component_count": component_count,
            "vertex_count": len(vertices), "edge_count": len(edges), "face_count": len(records)}



def indexed_mesh(triangle_points, *, convert=float):
    """Canonical common-coordinate conversion, without coordinate welding."""
    points = sorted({tuple(p) for triangle in triangle_points for p in triangle})
    lookup = {point: i for i, point in enumerate(points)}
    converted = [tuple(convert(x) for x in point) for point in points]
    require(len(set(converted)) == len(points), "coordinate conversion collapsed distinct rational vertices")
    if convert is float:
        require(all(math.isfinite(x) for p in converted for x in p), "nonfinite emitted geometry")
    return {"vertices_m": converted, "triangles": [tuple(lookup[tuple(p)] for p in tri) for tri in triangle_points]}


def encode_rational(value):
    """Lossless hexadecimal rational JSON representation, recursively."""
    if isinstance(value, Fraction):
        return certificate.encode_rational(value)
    if isinstance(value, dict):
        return {key: encode_rational(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_rational(item) for item in value]
    return value


def _sha(value):
    return hashlib.sha256(canonical(value)+b"\n").hexdigest()


def _triangle_points(mesh):
    return [tuple(tuple(mesh["vertices_m"][i]) for i in face) for face in mesh["triangles"]]


def _decode_fraction(value):
    return Fraction(int(value[0], 16), int(value[1], 16))


def _moment_difference(emitted, exact):
    def difference(a, b):
        if len(a) == 2 and all(isinstance(x, str) for x in a):
            return certificate.encode_rational(_decode_fraction(a)-_decode_fraction(b))
        return [difference(x, y) for x, y in zip(a, b, strict=True)]
    return {key: difference(emitted[key], exact[key]) for key in
            ("volume_m3", "first_volume_moment_m4", "second_volume_moment_m5")}


def compile_partitions(*, sources=ROOT / "Sources", source_lock=ROOT / "sources.lock.json", progress=None):
    """Compile and independently audit both explicit geometric conventions.

    No candidate is selected as an anatomical boundary and no native payload is
    changed. Exact rational source-set evidence and emitted binary64 geometry
    admission are distinct parts of the resulting artifact.
    """
    def stage(label):
        if progress is not None:
            progress(label)

    stage("verify pinned source cavities")
    cavities = extract_cavity_surfaces(sources=Path(sources), source_lock=Path(source_lock))
    original_audit = predicates.audit_cavity_intersections(cavities)
    require(original_audit["all_surfaces_embedded"], "source cavity is not embedded")
    require(all(pair["disjoint_closed_domains"] for pair in original_audit["per_pair"]
                if {pair["first"], pair["second"]} != set(NAMES)), "unexpected source cavity overlap")
    source_surfaces = rational_source_surfaces(cavities)
    expected = {name: certificate.source_geometry_sha256(surface) for name, surface in source_surfaces.items()}
    stage("construct exact common source-face arrangement")
    arrangement, construction = construct_arrangement(source_surfaces)
    stage("independently certify source coverage, membership and additive moments")
    proof = certificate.certify_partition(source_surfaces, arrangement, expected_geometry_sha256=expected)
    triangle_sets = certificate.build_triangle_sets(arrangement)
    authored_points = {p for row in arrangement for p in row["vertices"]}
    converted_points = {p: tuple(float(x) for x in p) for p in authored_points}
    require(len(set(converted_points.values())) == len(authored_points), "global Float64 vertex map is not injective")
    maximum_error = max(abs(Fraction.from_float(q)-x) for p, rounded in converted_points.items() for x, q in zip(p, rounded))
    originals = {p for surface in source_surfaces.values() for p in surface["vertices"]}
    require(originals <= authored_points and all(tuple(Fraction.from_float(x) for x in converted_points[p]) == p
                                               for p in originals), "source vertex coordinate changed")
    unchanged = [row for row in cavities["chambers"] if row["source_id"] not in NAMES]
    candidates = {}
    for name, exact in triangle_sets["partitions"].items():
        stage("audit emitted Float64 geometry: " + name)
        regions = {region: indexed_mesh(triangles) for region, triangles in exact["regions"].items()}
        interface = indexed_mesh(exact["shared_interface"])
        emitted_audit = certificate.audit_shared_interface_partition(
            {region: _triangle_points(mesh) for region, mesh in regions.items()}, _triangle_points(interface))
        # The two unchanged left cavities remain disjoint from each emitted
        # right region. Historical strict no-contact admission is appropriate
        # for these pairs; only the declared right shared interface is exempt.
        composition = predicates.audit_cavity_intersections({"chambers": [
            {"source_id": region, "exact_coordinate_quotient": mesh} for region, mesh in regions.items()] + unchanged})
        other_pairs = [pair for pair in composition["per_pair"] if {pair["first"], pair["second"]} != set(NAMES)]
        require(composition["all_surfaces_embedded"] and all(pair["disjoint_closed_domains"] for pair in other_pairs),
                "emitted composition intersects another cavity")
        candidates[name] = {"priority_region": name.removesuffix("_priority"),
            "ownership_convention": "priority retains its source domain; other retains the closure of its source-exclusive region",
            "regions": regions, "shared_interface": interface, "emitted_float64_audit": emitted_audit,
            "other_cavity_pairs": other_pairs, "four_cavity_interiors_disjoint": True,
            "geometry_sha256": _sha({"regions": regions, "shared_interface": interface}),
            "emitted_minus_rational_moments": {region: _moment_difference(
                emitted_audit["per_region"][region]["moments"], proof["partitions"][name]["regions"][region]["moments"])
                for region in NAMES},
            "emitted_minus_rational_union_moments": _moment_difference(emitted_audit["union_moments"], proof["union"]["moments"]),
            "biological_selection": False, "mechanical_mass_assigned": False}
    overlap = _decode_fraction(proof["intersection"]["moments"]["volume_m3"])
    require(overlap > 0, "source has no positive overlap to partition")
    return {"schema": SCHEMA,
        "source_geometry": cavities, "source_geometry_sha256": _sha(cavities),
        "source_intersection_audit": original_audit,
        "construction": encode_rational(construction), "arrangement": encode_rational(arrangement),
        "exact_certificate": proof, "candidates": candidates,
        "shared_source_volume_m3": float(overlap), "shared_source_volume_ml": float(overlap*10**6),
        "rounding": {"method": "one_binary64_conversion_per_unique_rational_point",
            "original_source_vertices_exactly_preserved": True,
            "added_intersection_vertex_count": len(authored_points-originals),
            "maximum_coordinate_error_m_exact": certificate.encode_rational(maximum_error),
            "maximum_coordinate_error_m": float(maximum_error),
            "exact_source_set_proof_applies_to": "rational_arrangement",
            "emitted_geometry_qualified_separately": True},
        "selection": None, "physical_stepping": False, "native_payload_changed": False,
        "hydraulic_parameters_changed": False, "added_mechanical_mass_kg": 0.0,
        "qualification": {"source_union_preserved_exactly_in_rational_construction": True,
            "source_exclusive_regions_preserved_exactly_in_rational_construction": True,
            "both_emitted_candidates_have_disjoint_interiors": True,
            "anatomical_valve_interface_selected": False, "cardiac_phase_registered": False,
            "subject_or_body_frame_registered": False, "blood_tissue_mass_partition": False,
            "physiological_calibration": False, "standing_walking": False},
        "boundary": "Two explicit geometric ownership candidates resolve duplicated source volume. Neither selects a biological atrioventricular interface. Subject, phase, density, body registration, donor blood inclusion and conservative mechanical mass/momentum transfer remain unresolved."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "Sources")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "sources.lock.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compile_partitions(sources=args.sources, source_lock=args.source_lock,
                                    progress=lambda text: print(text, file=sys.stderr, flush=True))
        payload = canonical(result)+b"\n"
        require(not args.output.is_symlink() and (not args.output.exists() or args.output.read_bytes() == payload),
                "output is immutable; choose a new path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not args.output.exists():
            with args.output.open("xb") as stream:
                stream.write(payload)
        print(json.dumps({"status": "compiled_geometric_partition_candidates", "output": str(args.output),
            "sha256": hashlib.sha256(payload).hexdigest(), "candidates": list(result["candidates"]),
            "shared_source_volume_ml": result["shared_source_volume_ml"],
            "emitted_interiors_disjoint": True, "anatomical_interface_selected": False,
            "mechanical_mass_changed": False}))
        return 0
    except (HumanImportError, OSError, KeyError, TypeError, ValueError, OverflowError) as error:
        print(json.dumps({"status": "rejected", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
