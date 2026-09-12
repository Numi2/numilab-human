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


def construct_arrangement(source_surfaces):
    """Return source-parented exact subtriangles and intersection provenance.

    This bounded authoring path supports transverse intersections with segment
    endpoints. Coplanar patches, tangencies and unsupported planar cell topology
    are rejected, not repaired by a tolerance or a guessed anatomical plane.
    Input surfaces must subsequently pass the independent certificate.
    """
    require(set(source_surfaces) == set(NAMES), "expected the two right source cavities")
    denominator = math.lcm(*(Fraction(x).denominator for surface in source_surfaces.values()
                            for point in surface["vertices"] for x in point))
    vertices = {name: [tuple(Fraction(x).numerator*(denominator//Fraction(x).denominator) for x in point)
                       for point in source_surfaces[name]["vertices"]] for name in NAMES}
    records = {name: predicates._records(vertices[name], source_surfaces[name]["triangles"]) for name in NAMES}
    pair_audit = predicates._audit_pair(records[NAMES[0]], records[NAMES[1]], same_surface=False)
    segments = {name: defaultdict(list) for name in NAMES}
    boundary_points = {name: defaultdict(set) for name in NAMES}
    edge_faces = {name: defaultdict(list) for name in NAMES}
    for name in NAMES:
        for face_id, face in enumerate(source_surfaces[name]["triangles"]):
            for i in range(3):
                edge_faces[name][tuple(sorted((face[i], face[(i+1) % 3])))].append(face_id)
    intersection_segments = []
    for pair in pair_audit["triangle_pairs"]:
        first, second = (records[name][face_id][0] for name, face_id in zip(NAMES, pair))
        normals = [predicates._cross(predicates._sub(tri[1], tri[0]), predicates._sub(tri[2], tri[0]))
                   for tri in (first, second)]
        require(any(predicates._cross(*normals)), "coplanar or tangent face intersection is unsupported")
        points = sorted(set(predicates.triangle_intersection_points(first, second)))
        require(len(points) >= 2, "point-only face contact is unsupported")
        a, b = points[0], points[-1]
        require(a != b and all(_on_edge(p, a, b) for p in points), "intersection is not one exact segment")
        intersection_segments.append({"source_faces": list(pair), "endpoints": (a, b)})
        for name, face_id in zip(NAMES, pair):
            segments[name][face_id].append((a, b))
            face = source_surfaces[name]["triangles"][face_id]
            for i in range(3):
                edge = tuple(sorted((face[i], face[(i+1) % 3])))
                for point in points:
                    if _on_edge(point, vertices[name][edge[0]], vertices[name][edge[1]]):
                        for neighbor in edge_faces[name][edge]:
                            boundary_points[name][neighbor].add(point)
    classifiers = {name: IntegerRayClassifier(records[name]) for name in NAMES}
    result = []
    for name, other in (NAMES, tuple(reversed(NAMES))):
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
                    "source_face_count": sum(len(source_surfaces[n]["triangles"]) for n in NAMES),
                    "subtriangle_count": len(result)}


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
