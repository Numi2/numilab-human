"""Conservative, offline Boolean ownership candidates for source cardiac cavities.

The source binary64 metre geometry is interpreted exactly. A common rational
surface arrangement preserves the source union and each exclusive region; only
the shared region receives alternative ownership. This is geometric authoring,
not an anatomical valve selection, mass attribution, or physical stepping.
"""
from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

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
    """Exact parity with cached source normals and integer query numerators.

    This authoring classifier is separately checked by the partition certificate.
    Positive ray directions allow a conservative AABB rejection. Boundary rays
    retry deterministically; unresolved queries fail closed.
    """
    def __init__(self, records):
        self.rows = [(tri, lower, upper, predicates._cross(predicates._sub(tri[1], tri[0]),
                                                          predicates._sub(tri[2], tri[0])))
                     for tri, lower, upper, _, _ in records]
        self.lower = tuple(min(row[1][k] for row in self.rows) for k in range(3))
        self.upper = tuple(max(row[2][k] for row in self.rows) for k in range(3))

    def location(self, point):
        denominator = math.lcm(*(Fraction(x).denominator for x in point))
        numerator = tuple(Fraction(x).numerator*(denominator//Fraction(x).denominator) for x in point)
        if any(numerator[k] < self.lower[k]*denominator or numerator[k] > self.upper[k]*denominator for k in range(3)):
            return "outside"
        for tri, lower, upper, normal in self.rows:
            if any(numerator[k] < lower[k]*denominator or numerator[k] > upper[k]*denominator for k in range(3)):
                continue
            if sum(normal[k]*(tri[0][k]*denominator-numerator[k]) for k in range(3)) == 0 and predicates._inside(
                    predicates._signs(numerator, denominator, tri, normal)):
                return "boundary"
        candidates = [row for row in self.rows if all(row[2][k]*denominator >= numerator[k] for k in range(3))]
        for attempt in range(1, 65):
            direction = (1, attempt, attempt*attempt)
            crossings, ambiguous = 0, False
            for tri, _, _, normal in candidates:
                plane = sum(normal[k]*(tri[0][k]*denominator-numerator[k]) for k in range(3))
                divisor = sum(normal[k]*direction[k] for k in range(3))
                if divisor == 0:
                    if plane == 0:
                        ambiguous = True
                        break
                    continue
                if plane*divisor <= 0:
                    continue
                hit = tuple(numerator[k]*divisor+plane*direction[k] for k in range(3))
                hit_denominator = denominator*divisor
                if hit_denominator < 0:
                    hit_denominator, hit = -hit_denominator, tuple(-x for x in hit)
                signs = predicates._signs(hit, hit_denominator, tri, normal)
                if predicates._inside(signs):
                    if 0 in signs:
                        ambiguous = True
                        break
                    crossings += 1
            if not ambiguous:
                return "inside" if crossings % 2 else "outside"
        require(False, "indeterminate exact ray classification")


def _on_edge(point, a, b):
    return (predicates._cross(predicates._sub(point, a), predicates._sub(b, a)) == (0, 0, 0)
            and all(min(a[k], b[k]) <= point[k] <= max(a[k], b[k]) for k in range(3)))


def construct_arrangement(source_surfaces, *, source_names=NAMES):
    """Return source-parented exact subtriangles and intersection provenance.

    This bounded authoring path supports transverse intersections with segment
    endpoints. Coplanar patches, tangencies and unsupported planar cell topology
    are rejected, not repaired by a tolerance or a guessed anatomical plane.
    Input surfaces must subsequently pass the independent certificate.
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
    pair_audit = predicates._audit_pair(records[source_names[0]], records[source_names[1]], same_surface=False)
    segments = {name: defaultdict(list) for name in source_names}
    boundary_points = {name: defaultdict(set) for name in source_names}
    edge_faces = {name: defaultdict(list) for name in source_names}
    for name in source_names:
        for face_id, face in enumerate(source_surfaces[name]["triangles"]):
            for i in range(3):
                edge_faces[name][tuple(sorted((face[i], face[(i+1) % 3])))].append(face_id)
    intersection_segments = []
    for pair in pair_audit["triangle_pairs"]:
        first, second = (records[name][face_id][0] for name, face_id in zip(source_names, pair))
        normals = [predicates._cross(predicates._sub(tri[1], tri[0]), predicates._sub(tri[2], tri[0]))
                   for tri in (first, second)]
        require(any(predicates._cross(*normals)), "coplanar or tangent face intersection is unsupported")
        points = sorted(set(predicates.triangle_intersection_points(first, second)))
        require(len(points) >= 2, "point-only face contact is unsupported")
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
    classifiers = {name: IntegerRayClassifier(records[name]) for name in source_names}
    result = []
    for name, other in (source_names, tuple(reversed(source_names))):
        for face_id, row in enumerate(records[name]):
            children = subdivide_triangle(row[0], segments[name][face_id], boundary_points[name][face_id])
            for triangle in children:
                centroid = tuple(sum(p[k] for p in triangle)/3 for k in range(3))
                location = classifiers[other].location(centroid)
                require(location in ("inside", "outside"), f"boundary child interior {name}:{face_id}")
                result.append({"vertices": tuple(tuple(x/denominator for x in p) for p in triangle),
                               "source": name, "source_face": face_id, "other_location": location})
    return result, {"source_intersecting_triangle_pairs": pair_audit["triangle_pairs"],
                    "source_intersecting_triangle_pair_count": pair_audit["count"],
                    "common_integer_scale_per_metre": denominator,
                    "intersection_segments_m": [{"source_faces": row["source_faces"],
                        "endpoints": tuple(tuple(x/denominator for x in point) for point in row["endpoints"])}
                        for row in intersection_segments],
                    "source_face_count": sum(len(source_surfaces[n]["triangles"]) for n in source_names),
                    "subtriangle_count": len(result)}


def _exact_triangle_normal(triangle):
    return predicates._cross(predicates._sub(triangle[1], triangle[0]),
                             predicates._sub(triangle[2], triangle[0]))


def _bounded_normal_probe(point, normal, records):
    """Choose a rational normal offset before the next nonzero source-plane crossing."""
    nearest = None
    for triangle, _lower, _upper, _face_id, _ids in records:
        other_normal = _exact_triangle_normal(triangle)
        denominator = predicates._dot(other_normal, normal)
        numerator = predicates._dot(other_normal, predicates._sub(triangle[0], point))
        if denominator == 0:
            require(numerator != 0,
                    "normal probe line lies in another source face plane")
            continue
        crossing = Fraction(numerator, denominator)
        if crossing == 0:
            continue
        distance = abs(crossing)
        nearest = distance if nearest is None else min(nearest, distance)
    require(nearest is not None and nearest > 0,
            "no bounded nonzero source-plane crossing for normal probes")
    return nearest / 2


def _point_on_segment(point, segment):
    return _on_edge(point, segment[0], segment[1])


def _oriented_surface_topology(records):
    """Collect exact child half-edges before distinguishing arrangement cuts."""
    edge_uses = defaultdict(list)
    for child_id, (triangle, _lower, _upper, _face_id, ids) in enumerate(records):
        for a, b in zip(ids, ids[1:] + ids[:1]):
            edge_uses[tuple(sorted((a, b)))].append((child_id, a, b))
    return edge_uses


def construct_nonzero_winding_self_union(source_surface):
    """Extract the exact boundary of the nonzero signed-winding set of one shell.

    Only transverse, segment-valued self-intersections are supported. The
    arrangement splits every incident source face, classifies each connected
    ordinary-face patch using exact probes before the next source-plane
    crossing, and retains patches separating zero from nonzero winding.
    Coplanar, point-only, unresolved, or nonmanifold arrangements fail closed.
    This is a rational source-space result; Float32 conversion requires a
    separate audit and is not implied here.
    """
    return _construct_winding_self_union(source_surface, positive_only=False)


def construct_positive_winding_self_union(source_surface):
    """Extract the boundary of the strictly positive signed-winding material.

    This is an explicit inference for an outward-oriented reference shell with
    inverted exterior folds, not a substitute for the nonzero material rule.
    Negative exterior lobes are excluded; opposite-oriented cavities inside
    positive material remain cavities. A globally reversed shell is rejected.
    Exact arrangement, manifold, self-intersection and side-probe checks remain
    mandatory. Attribute ancestry and Float32 conversion need separate audits.
    """
    return _construct_winding_self_union(source_surface, positive_only=True)


def _construct_winding_self_union(source_surface, *, positive_only):
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
        require(len(points) >= 2,
                "point-only self-contact is unsupported")
        require(any(predicates._cross(first_normal, second_normal)),
                "coplanar self-intersection is unsupported")
        start, end = points[0], points[-1]
        require(start != end and all(_point_on_segment(point, (start, end)) for point in points),
                "self-intersection is not one exact transverse segment")
        segments.append({"face_pair": [first_id, second_id], "endpoints": (start, end)})
        segments_by_face[first_id].add((start, end))
        segments_by_face[second_id].add((start, end))
        for face_id in (first_id, second_id):
            face = faces[face_id]
            for a, b in zip(face, face[1:] + face[:1]):
                if any(_point_on_segment(point, (exact_vertices[a], exact_vertices[b]))
                       for point in (start, end)):
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

    for edge, uses in edge_uses.items():
        if edge in cut_edges:
            forward = sum(first < second for _child, first, second in uses)
            reverse = len(uses) - forward
            require(len(uses) >= 4 and len(uses) % 2 == 0 and forward == reverse,
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
        point = tuple(sum(vertex[axis] for vertex in triangle) / 3 for axis in range(3))
        require(not any(_point_on_segment(point, segment["endpoints"]) for segment in segments),
                "arrangement patch representative lies on an intersection segment")
        normal = _exact_triangle_normal(triangle)
        step = _bounded_normal_probe(point, normal, records)
        minus_point = tuple(point[axis] - step * normal[axis] for axis in range(3))
        plus_point = tuple(point[axis] + step * normal[axis] for axis in range(3))
        minus = predicates.signed_winding_number(minus_point, prepared)
        plus = predicates.signed_winding_number(plus_point, prepared)
        require(minus["status"] == plus["status"] == "resolved",
                "self-union side winding is boundary or indeterminate")
        winding_minus, winding_plus = minus["winding_number"], plus["winding_number"]
        require(winding_minus == winding_plus + 1,
                "oriented patch does not separate adjacent winding levels by +1")
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
            "winding_minus": winding_minus,
            "winding_plus": winding_plus,
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

    return {
        "source_sha256": source_sha,
        "material_rule": "strictly_positive" if positive_only else "nonzero",
        "input_face_count": len(faces),
        "input_component_count": prepared.face_component_count,
        "input_self_intersection_pairs": intersection_audit["triangle_pairs"],
        "input_self_intersection_count": intersection_audit["count"],
        "exact_intersection_segments": segments,
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
