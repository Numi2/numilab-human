"""Exact source-face-owned partition for overlapping left basal airway proxies.

The input meshes are BodyParts3D solid proxies, not measured airway lumen surfaces.
This routine preserves their exact union and source-face lineage while assigning
each exposed subtriangle to its original source member. It does not infer a
lumen, cap a port, or change registration or physics.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math
from typing import Any

from . import cardiac_cavity_partition as arrangement
from . import cardiac_partition_certificate as certificate
from .model import ImportError as HumanImportError


SOURCE_MEMBERS = ("FJ2445", "FJ2446")
SOURCE_IDENTITY = {
    "FJ2445": {
        "concept_id": "FMA68232",
        "label": "left lateral basal segmental bronchial tree",
        "member_sha256": "6e6c5a231e65b7bef20a7e53d8502c2f6ac6130274ddd5ec1bf14ac7505d46ea",
    },
    "FJ2446": {
        "concept_id": "FMA68233",
        "label": "left posterior basal segmental bronchial tree",
        "member_sha256": "7512a2599ad515e6da4bee78fcf6c733bfde4051e751dc7b244b22dabd62c0cb",
    },
}
MM_TO_M = Fraction.from_float(0.001)
_EXACT_PARTITION_CACHE: dict[tuple, tuple[list, dict, dict]] = {}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HumanImportError("airway sibling partition: " + message)


def _exact_source_surface(member: str, surface: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    identity = SOURCE_IDENTITY[member]
    _require(surface.get("source_sha256") == identity["member_sha256"],
             f"{member} source member hash differs from the audited source")
    vertices_mm = surface.get("vertices_mm")
    triangles_raw = surface.get("triangles")
    _require(isinstance(vertices_mm, list) and vertices_mm and
             isinstance(triangles_raw, list) and triangles_raw,
             f"{member} source geometry is empty")
    exact_vertices: list[tuple[Fraction, Fraction, Fraction]] = []
    source_mm_by_exact: dict[tuple[Fraction, Fraction, Fraction], tuple[float, float, float]] = {}
    exact_index: dict[tuple[Fraction, Fraction, Fraction], int] = {}
    raw_to_exact: list[int] = []
    for raw in vertices_mm:
        _require(isinstance(raw, (tuple, list)) and len(raw) == 3 and
                 all(math.isfinite(float(value)) for value in raw),
                 f"{member} has a malformed source vertex")
        mm = tuple(float(value) for value in raw)
        metres = tuple(Fraction.from_float(float(value * 0.001)) for value in mm)
        index = exact_index.get(metres)
        if index is None:
            index = len(exact_vertices)
            exact_index[metres] = index
            exact_vertices.append(metres)
            source_mm_by_exact[metres] = mm
        raw_to_exact.append(index)
    triangles: list[tuple[int, int, int]] = []
    for row in triangles_raw:
        _require(isinstance(row, (tuple, list)) and len(row) == 3 and
                 all(type(index) is int and 0 <= index < len(raw_to_exact) for index in row),
                 f"{member} has a malformed source face")
        face = tuple(raw_to_exact[index] for index in row)
        _require(len(set(face)) == 3, f"{member} has a collapsed source face")
        triangles.append(face)
    return {
        "vertices": exact_vertices,
        "triangles": triangles,
        "source_sha256": surface["source_sha256"],
    }, source_mm_by_exact


def partition_airway_sibling_overlap(
    surfaces: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build and independently certify the source-face-owned exterior union.

    Each surfaces row has raw OBJ vertices_mm, triangles and the SHA-256 of
    its unmodified OBJ member bytes. Returned triangles retain source winding
    and each emitted triangle's original OBJ face index.
    """
    _require(isinstance(surfaces, dict) and set(surfaces) == set(SOURCE_MEMBERS),
             "both audited source members are required")
    exact_surfaces: dict[str, dict[str, Any]] = {}
    source_mm_by_exact: dict[str, dict] = {}
    for member in SOURCE_MEMBERS:
        exact_surfaces[member], source_mm_by_exact[member] = _exact_source_surface(member, surfaces[member])

    expected_geometry = {
        member: certificate.source_geometry_sha256(exact_surfaces[member])
        for member in SOURCE_MEMBERS
    }
    cache_key = tuple(
        (member, exact_surfaces[member]["source_sha256"], expected_geometry[member])
        for member in SOURCE_MEMBERS
    )
    cached = _EXACT_PARTITION_CACHE.get(cache_key)
    if cached is None:
        rows, arrangement_report = arrangement.construct_arrangement(
            exact_surfaces, source_names=SOURCE_MEMBERS,
        )
        proof = certificate.certify_partition(
            exact_surfaces, rows, expected_geometry_sha256=expected_geometry,
            source_names=SOURCE_MEMBERS,
        )
        cached = (rows, arrangement_report, proof)
        _EXACT_PARTITION_CACHE[cache_key] = cached
    rows, arrangement_report, proof = cached
    _require(proof["source_face_coverage_exact"] and proof["child_interiors_uncut"] and
             proof["independent_classification_exact"] and proof["source_union_preserved"] and
             proof["source_exclusive_regions_preserved"],
             "exact source-face certificate did not close")
    _require(proof["union"]["topology"]["closed"] and
             proof["union"]["topology"]["face_components"] == 1,
             "exposed source patches do not form one closed connected solid proxy")
    _require(arrangement_report["source_intersecting_triangle_pair_count"] > 0,
             "expected sibling source overlap was not found")

    output: dict[str, dict[str, Any]] = {}
    maximum_emission_error_m = Fraction(0)
    for member in SOURCE_MEMBERS:
        source_rows = [
            row for row in rows
            if row["source"] == member and row["other_location"] == "outside"
        ]
        exact_points = sorted({
            tuple(point)
            for row in source_rows
            for point in row["vertices"]
        })
        _require(exact_points, f"{member} has no source-owned exterior surface")
        point_to_index = {point: index for index, point in enumerate(exact_points)}
        vertices_mm: list[tuple[float, float, float]] = []
        for point in exact_points:
            original_mm = source_mm_by_exact[member].get(point)
            if original_mm is not None:
                emitted_mm = original_mm
            else:
                emitted_mm = tuple(float(value / MM_TO_M) for value in point)
            _require(all(math.isfinite(value) for value in emitted_mm),
                     f"{member} partition emitted a nonfinite coordinate")
            emitted_metres = tuple(Fraction.from_float(float(value * 0.001)) for value in emitted_mm)
            error = max(abs(emitted_metres[axis] - point[axis]) for axis in range(3))
            maximum_emission_error_m = max(maximum_emission_error_m, error)
            vertices_mm.append(tuple(emitted_mm))
        emitted_metres = [
            tuple(Fraction.from_float(float(value * 0.001)) for value in point)
            for point in vertices_mm
        ]
        _require(len(set(emitted_metres)) == len(emitted_metres),
                 f"{member} binary64 millimetre conversion collapsed exact partition vertices")
        triangles = [
            tuple(point_to_index[tuple(point)] for point in row["vertices"])
            for row in source_rows
        ]
        _require(all(len(set(face)) == 3 for face in triangles),
                 f"{member} binary64 millimetre conversion collapsed an emitted face")
        output[member] = {
            "vertices_mm": vertices_mm,
            "triangles": triangles,
            "source_face_indices": [int(row["source_face"]) for row in source_rows],
        }

    lineage = {}
    for member in SOURCE_MEMBERS:
        data = output[member]
        encoded = b"".join(int(index).to_bytes(4, "little") for index in data["source_face_indices"])
        lineage[member] = {
            "source_member_sha256": SOURCE_IDENTITY[member]["member_sha256"],
            "source_face_count": len(exact_surfaces[member]["triangles"]),
            "emitted_face_count": len(data["triangles"]),
            "emitted_source_face_indices": data["source_face_indices"],
            "emitted_source_face_indices_sha256": hashlib.sha256(encoded).hexdigest(),
            "emitted_faces_preserve_source_winding": True,
        }

    return {
        "meshes": output,
        "proof": {
            "method": "exact rational transverse arrangement and independent source-face certificate",
            "source_identity": SOURCE_IDENTITY,
            "source_geometry_identity": expected_geometry,
            "arrangement": arrangement.encode_rational(arrangement_report),
            "certificate": proof,
            "face_lineage": lineage,
            "geometry_interpretation": (
                "Inferred source-solid overlap partition of sibling segmental-bronchial-tree "
                "proxies; it preserves the source union and source-exclusive exterior patches. "
                "The source meshes are not lumen measurements. This result does not establish "
                "a shared airway lumen, port, physiological volume, or mechanics."
            ),
            "emission_coordinate_semantics": (
                "Partition intersections are computed exactly from the published binary64 "
                "source-metre values. New rational points are rounded once to source millimetres "
                "for the existing visual transform."
            ),
            "maximum_rational_to_binary64_source_metre_error_m": (
                f"{maximum_emission_error_m.numerator}/{maximum_emission_error_m.denominator}"
            ),
            "source_vertices_moved": False,
            "new_vertices_added_by_subdivision": True,
            "source_faces_added": False,
            "source_face_lineage_preserved": True,
            "exact_rational_source_union_and_source_exclusive_regions_certified": True,
            "emitted_coordinates_round_rational_intersections": True,
            "physiological_lumen_assigned": False,
            "physical_stepping": False,
        },
    }
