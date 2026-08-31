"""Compile a nonvisual BodyParts3D abductor-digiti-minimi mechanics witness.

BodyParts3D supplies the muscle and bone surfaces, but not fibre directions or
force parameters.  This module therefore keeps exact source geometry separate
from two explicit inferences: surface-nearest enthesis clusters and a bounded
specific-tension sensitivity analysis.  It does not silently promote either
inference to subject-specific anatomy.
"""

from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path
from typing import Any, Iterable

from .model import (
    ImportError,
    _bodyparts_obj_geometry,
    _bodyparts_obj_member,
    read_json,
    sha256,
    write_json,
)


SCHEMA = "numi.human.bodyparts3d-abductor-digiti-minimi-inference.v1"
PAYLOAD_MAGIC = b"NHADM1\0\0"
PAYLOAD_ABI = 1
PAYLOAD_RECORD_BYTES = 64
EXPECTED_ARCHIVE_SHA256 = (
    "40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e"
)
MEMBERS = {
    "right": {
        "muscle": "FJ1466",
        "origin_bone": "FJ3382",
        "insertion_bone": "FJ3323",
    },
    "left": {
        "muscle": "FJ1466M",
        "origin_bone": "FJ3276",
        "insertion_bone": "FJ3314",
    },
}


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _scale(a: tuple[float, float, float], value: float) -> tuple[float, float, float]:
    return a[0] * value, a[1] * value, a[2] * value


def _length(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def _closest_point_on_triangle(
    point: tuple[float, float, float],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the exact Euclidean closest point on one nondegenerate triangle."""
    ab = _sub(b, a)
    ac = _sub(c, a)
    ap = _sub(point, a)
    d1, d2 = _dot(ab, ap), _dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a
    bp = _sub(point, b)
    d3, d4 = _dot(ab, bp), _dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return _add(a, _scale(ab, d1 / (d1 - d3)))
    cp = _sub(point, c)
    d5, d6 = _dot(ab, cp), _dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return _add(a, _scale(ac, d2 / (d2 - d6)))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = _sub(c, b)
        return _add(b, _scale(bc, (d4 - d3) / ((d4 - d3) + (d5 - d6))))
    denominator = va + vb + vc
    if abs(denominator) <= 1.0e-18:
        # Source meshes should not contain collapsed faces.  Treating one as a
        # segment would conceal source drift, so fail closed instead.
        raise ImportError("BodyParts3D enthesis inference encountered a degenerate triangle")
    inverse = 1.0 / denominator
    return _add(a, _add(_scale(ab, vb * inverse), _scale(ac, vc * inverse)))


def _nearest_surface_point(
    point: tuple[float, float, float],
    vertices: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
) -> tuple[float, tuple[float, float, float]]:
    best_distance = math.inf
    best_point: tuple[float, float, float] | None = None
    for ia, ib, ic in triangles:
        candidate = _closest_point_on_triangle(point, vertices[ia], vertices[ib], vertices[ic])
        distance = _length(_sub(point, candidate))
        if distance < best_distance:
            best_distance, best_point = distance, candidate
    if best_point is None:
        raise ImportError("BodyParts3D enthesis bone surface has no triangles")
    return best_distance, best_point


def _centroid(points: Iterable[tuple[float, float, float]]) -> tuple[float, float, float]:
    points = list(points)
    if not points:
        raise ImportError("BodyParts3D enthesis cluster is empty")
    inverse = 1.0 / len(points)
    return tuple(sum(point[axis] for point in points) * inverse for axis in range(3))  # type: ignore[return-value]


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ImportError("cannot calculate a percentile of an empty enthesis cluster")
    location = probability * (len(ordered) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _surface_cluster(
    muscle_vertices: list[tuple[float, float, float]],
    bone_vertices: list[tuple[float, float, float]],
    bone_triangles: list[tuple[int, int, int]],
) -> dict[str, Any]:
    candidates = []
    for index, vertex in enumerate(muscle_vertices):
        distance, surface_point = _nearest_surface_point(vertex, bone_vertices, bone_triangles)
        candidates.append((distance, index, vertex, surface_point))
    # A fixed fraction makes the result resolution-independent while a floor
    # of eight vertices prevents a single noisy tip from defining an enthesis.
    count = min(len(candidates), max(8, math.ceil(0.025 * len(candidates))))
    selected = sorted(candidates, key=lambda item: (item[0], item[1]))[:count]
    distances = [item[0] for item in selected]
    return {
        "selection_method": "closest_2_5_percent_muscle_vertices_to_exact_bone_triangle_surface_minimum_8",
        "muscle_vertex_count": len(muscle_vertices),
        "selected_vertex_count": count,
        "selected_vertex_indices": [item[1] for item in selected],
        "muscle_cluster_centroid_source_mm": list(_centroid(item[2] for item in selected)),
        "bone_surface_centroid_source_mm": list(_centroid(item[3] for item in selected)),
        "surface_distance_mm": {
            "minimum": min(distances),
            "median": _percentile(distances, 0.5),
            "maximum": max(distances),
        },
    }


def _anchor_records(value: Any, member_id: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, dict):
        source = value.get("source")
        if isinstance(source, dict) and source.get("member_id") == member_id:
            records.append(value)
        for child in value.values():
            records.extend(_anchor_records(child, member_id))
    elif isinstance(value, list):
        for child in value:
            records.extend(_anchor_records(child, member_id))
    return records


def _registered_anchor(receipt: dict[str, Any], member_id: str) -> dict[str, Any]:
    records = _anchor_records(receipt, member_id)
    if len(records) != 1:
        raise ImportError(
            f"registration receipt has {len(records)} records for BodyParts3D {member_id}; expected one"
        )
    record = records[0]
    registration = record.get("registration")
    source = record.get("source")
    target = record.get("target")
    if not all(isinstance(item, dict) for item in (registration, source, target)):
        raise ImportError(f"registration record for {member_id} is incomplete")
    matrix = registration.get("source_obj_mm_to_core_inertial_body_m")
    if (
        not isinstance(matrix, list) or len(matrix) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in matrix)
        or any(not math.isfinite(float(value)) for row in matrix for value in row)
    ):
        raise ImportError(f"registration matrix for {member_id} is invalid")
    return record


def _transform(matrix: list[list[float]], point: list[float]) -> list[float]:
    result = [
        sum(float(matrix[row][column]) * point[column] for column in range(3))
        + float(matrix[row][3])
        for row in range(3)
    ]
    if not all(math.isfinite(value) for value in result):
        raise ImportError("registered ADM endpoint is non-finite")
    return result


def _member_geometry(sources: Path, member_id: str) -> tuple[dict[str, Any], list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    _, member, obj = _bodyparts_obj_member(sources, "is_a", member_id)
    vertices, triangles, _ = _bodyparts_obj_geometry(obj, member)
    if not vertices or not triangles:
        raise ImportError(f"BodyParts3D member {member_id} has no surface geometry")
    return {
        "member_id": member_id,
        "member": member,
        "member_sha256": hashlib.sha256(obj).hexdigest(),
        "vertex_count": len(vertices),
        "triangle_count": len(triangles),
    }, vertices, triangles


def _endpoint(
    cluster: dict[str, Any], anchor: dict[str, Any], bone_metadata: dict[str, Any]
) -> dict[str, Any]:
    source = anchor["source"]
    registration = anchor["registration"]
    if source.get("member_sha256") != bone_metadata["member_sha256"]:
        raise ImportError(f"registration receipt member hash drifted for {bone_metadata['member_id']}")
    attachment = registration.get("attachment_surface_refinement")
    return {
        "bone": bone_metadata,
        "target_core_body_index": anchor["target"].get("core_body_index"),
        "target_core_body_name": anchor["target"].get("name"),
        "bone_surface_point_source_mm": cluster["bone_surface_centroid_source_mm"],
        "point_core_inertial_body_m": _transform(
            registration["source_obj_mm_to_core_inertial_body_m"],
            cluster["bone_surface_centroid_source_mm"],
        ),
        "cluster": cluster,
        "registration_status": registration.get("status"),
        "registration_attachment_surface_refinement": attachment,
        "registration_has_chain_fallback": bool(
            isinstance(registration.get("upper_limb_chain_translation_fallback"), dict)
            and registration["upper_limb_chain_translation_fallback"].get("applied")
        ),
    }


def compile_adm_inference(sources: Path, registration_path: Path) -> dict[str, Any]:
    sources = sources.resolve()
    archive = sources / "isa_BP3D_4.0_obj_99.zip"
    if not archive.is_file():
        raise ImportError(f"BodyParts3D is-a archive is absent: {archive}")
    archive_hash = sha256(archive)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise ImportError("BodyParts3D 4.0 is-a archive SHA-256 does not match the pinned source")
    receipt = read_json(registration_path.resolve())
    hands: list[dict[str, Any]] = []
    for side, members in MEMBERS.items():
        muscle_metadata, muscle_vertices, _ = _member_geometry(sources, members["muscle"])
        origin_metadata, origin_vertices, origin_triangles = _member_geometry(
            sources, members["origin_bone"]
        )
        insertion_metadata, insertion_vertices, insertion_triangles = _member_geometry(
            sources, members["insertion_bone"]
        )
        origin_cluster = _surface_cluster(muscle_vertices, origin_vertices, origin_triangles)
        insertion_cluster = _surface_cluster(muscle_vertices, insertion_vertices, insertion_triangles)
        origin_indices = set(origin_cluster["selected_vertex_indices"])
        insertion_indices = set(insertion_cluster["selected_vertex_indices"])
        overlap = len(origin_indices & insertion_indices)
        origin_centroid = tuple(origin_cluster["muscle_cluster_centroid_source_mm"])
        insertion_centroid = tuple(insertion_cluster["muscle_cluster_centroid_source_mm"])
        separation = _length(_sub(insertion_centroid, origin_centroid))
        if overlap or separation < 20.0:
            raise ImportError(
                f"{side} ADM endpoint clusters are not anatomically separable "
                f"(overlap={overlap}, separation={separation:.6g} mm)"
            )
        origin_anchor = _registered_anchor(receipt, members["origin_bone"])
        insertion_anchor = _registered_anchor(receipt, members["insertion_bone"])
        hands.append({
            "side": side,
            "muscle": muscle_metadata,
            "origin": _endpoint(origin_cluster, origin_anchor, origin_metadata),
            "insertion": _endpoint(insertion_cluster, insertion_anchor, insertion_metadata),
            "source_endpoint_separation_mm": separation,
            "endpoint_cluster_overlap_count": overlap,
        })
    right, left = hands
    parity_residuals = {}
    for endpoint in ("origin", "insertion"):
        right_point = right[endpoint]["cluster"]["bone_surface_centroid_source_mm"]
        left_point = left[endpoint]["cluster"]["bone_surface_centroid_source_mm"]
        parity_residuals[endpoint] = [
            abs(right_point[0] + left_point[0]),
            abs(right_point[1] - left_point[1]),
            abs(right_point[2] - left_point[2]),
        ]
    maximum_parity_residual = max(value for residual in parity_residuals.values() for value in residual)
    if maximum_parity_residual > 5.0:
        raise ImportError(
            f"bilateral ADM source endpoint parity drifted by {maximum_parity_residual:.6g} mm"
        )
    pcsa_mm2 = 111.0
    specific_tension_n_cm2 = [20.0, 26.8, 55.0]
    force_capacity = [value * (pcsa_mm2 / 100.0) for value in specific_tension_n_cm2]
    return {
        "schema": SCHEMA,
        "source": {
            "name": "BodyParts3D",
            "version": "4.0",
            "license": "CC BY 4.0",
            "archive": archive.name,
            "archive_sha256": archive_hash,
        },
        "hands": hands,
        "bilateral_validation": {
            "method": "source_rest_frame_x_reflection_endpoint_surface_parity",
            "endpoint_residual_xyz_mm": parity_residuals,
            "maximum_residual_mm": maximum_parity_residual,
            "maximum_allowed_mm": 5.0,
            "passed": True,
        },
        "literature_parameters": {
            "physiological_cross_sectional_area_mm2": pcsa_mm2,
            "optimal_fibre_length_mm": 96.0,
            "muscle_mass_g": 11.0,
            "belly_length_mm": 84.0,
            "pennation_angle_degrees": 0.0,
            "external_insertion_tendon_length_mm": 11.0,
            "source": "https://doi.org/10.1111/joa.12877",
            "specimen_boundary": "one fresh-frozen 60-year-old male hand; not population calibration",
        },
        "force_capacity_sensitivity": {
            "specific_tension_n_cm2": {
                "low": specific_tension_n_cm2[0],
                "recommended_systematic_review": specific_tension_n_cm2[1],
                "high_in_vivo_study": specific_tension_n_cm2[2],
            },
            "maximum_isometric_force_n": {
                "low": force_capacity[0],
                "nominal": force_capacity[1],
                "high": force_capacity[2],
            },
            "sources": [
                "https://doi.org/10.1152/japplphysiol.00296.2024",
                "https://doi.org/10.1113/expphysiol.2009.048967",
            ],
            "boundary": "sensitivity analysis only; ADM-specific maximum force was not measured",
        },
        "insertion_branch_sensitivity": {
            "default_v1_bone_fraction": 1.0,
            "optional_extensor_expansion_fraction": 0.31,
            "observed_extensor_expansion_fraction_range": [0.0, 0.5],
            "source": "https://pubmed.ncbi.nlm.nih.gov/11901393/",
            "boundary": "population variation; this BodyParts3D individual is not assigned an unobserved branch",
        },
        "evidence_boundary": (
            "Exact pinned BodyParts3D surfaces and registered bone identities; endpoint clusters, "
            "fibre direction, and force capacity are explicit inference/sensitivity. Pisiform "
            "registrations retain their documented upper-limb-chain fallback. This artifact "
            "does not prove live Hill-type actuation, neutral-pose equilibrium, or subject-specific anatomy."
        ),
    }


def compile_adm_payload(artifact: dict[str, Any]) -> bytes:
    if artifact.get("schema") != SCHEMA or len(artifact.get("hands", [])) != 2:
        raise ImportError("ADM payload requires one validated bilateral inference artifact")
    source_hash = artifact.get("source", {}).get("archive_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ImportError("ADM inference artifact has no source archive hash")
    records = bytearray()
    force = artifact["force_capacity_sensitivity"]["maximum_isometric_force_n"]
    literature = artifact["literature_parameters"]
    for side_index, hand in enumerate(artifact["hands"]):
        if hand.get("side") != ("right" if side_index == 0 else "left"):
            raise ImportError("ADM inference hands are not in canonical right/left order")
        flags = 1
        if hand["origin"].get("registration_has_chain_fallback"):
            flags |= 2
        record = struct.pack(
            "<4I12f",
            side_index,
            int(hand["origin"]["target_core_body_index"]),
            int(hand["insertion"]["target_core_body_index"]),
            flags,
            *map(float, hand["origin"]["point_core_inertial_body_m"]),
            *map(float, hand["insertion"]["point_core_inertial_body_m"]),
            float(literature["optimal_fibre_length_mm"]) * 0.001,
            float(literature["external_insertion_tendon_length_mm"]) * 0.001,
            float(literature["physiological_cross_sectional_area_mm2"]),
            float(force["low"]),
            float(force["nominal"]),
            float(force["high"]),
        )
        if len(record) != PAYLOAD_RECORD_BYTES:
            raise AssertionError("NHADM1 record layout drifted")
        records.extend(record)
    return struct.pack(
        "<8s3I32s", PAYLOAD_MAGIC, PAYLOAD_ABI, 2, PAYLOAD_RECORD_BYTES,
        bytes.fromhex(source_hash),
    ) + records


def build(
    sources: Path, registration: Path, output: Path,
    payload_output: Path | None = None,
) -> dict[str, Any]:
    artifact = compile_adm_inference(sources, registration)
    if payload_output is not None:
        payload = compile_adm_payload(artifact)
        payload_output = payload_output.resolve()
        payload_output.parent.mkdir(parents=True, exist_ok=True)
        payload_output.write_bytes(payload)
        artifact["runtime_payload"] = {
            "abi": "NHADM1",
            "file": payload_output.name,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    write_json(output.resolve(), artifact)
    return artifact
