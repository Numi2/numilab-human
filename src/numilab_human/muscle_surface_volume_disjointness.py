"""Exact pairwise disjointness audit for closed compiled muscle surfaces.

This audits candidate muscle domains only. It does not assign physical-volume,
mass, material, force-transfer, or whole-body owners.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any

from . import model as human
from .cardiac_cavity_intersections import _audit_pair, _records, point_location
from .compiled_quotient_embeddedness import coordinate_quotient
from .physiology import canonical


SCHEMA = "numi.human.compiled-muscle-volume-disjointness.v1"
PREDICATE_FILES = (
    "muscle_surface_volume_disjointness.py",
    "muscle_surface_embeddedness.py",
    "compiled_quotient_embeddedness.py",
    "whole_body_embeddedness.py",
    "surface_topology_audit.py",
    "cardiac_cavity_intersections.py",
    "cardiac_cavity_geometry.py",
)


class MuscleDisjointnessError(human.ImportError):
    """A compiled muscle volume disjointness audit cannot be admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MuscleDisjointnessError("muscle volume disjointness: " + message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable(path: Path, value: dict[str, Any]) -> str:
    raw = canonical(value) + b"\n"
    _require(not path.is_symlink(), "output path is a symlink")
    if path.exists():
        _require(path.read_bytes() == raw, "output is immutable; choose a new path")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def _surface_pair_status(
    first: dict[str, Any], second: dict[str, Any]
) -> dict[str, Any]:
    """Classify two exact integer-coordinate closed surfaces."""
    first_vertices = first["vertices"]
    second_vertices = second["vertices"]
    first_lo = tuple(min(point[axis] for point in first_vertices) for axis in range(3))
    first_hi = tuple(max(point[axis] for point in first_vertices) for axis in range(3))
    second_lo = tuple(
        min(point[axis] for point in second_vertices) for axis in range(3)
    )
    second_hi = tuple(
        max(point[axis] for point in second_vertices) for axis in range(3)
    )
    if any(
        first_hi[axis] < second_lo[axis] or second_hi[axis] < first_lo[axis]
        for axis in range(3)
    ):
        return {
            "status": "strictly_disjoint_aabbs",
            "aabb_candidate_pairs": 0,
            "intersection_pairs": 0,
            "containment": "impossible_for_strictly_disjoint_aabbs",
        }

    checked = _audit_pair(first["records"], second["records"], same_surface=False)
    pair_digest = hashlib.sha256(
        json.dumps(checked["triangle_pairs"], separators=(",", ":")).encode()
    ).hexdigest()
    if checked["count"]:
        return {
            "status": "surface_intersection",
            "aabb_candidate_pairs": checked["aabb_candidate_pairs"],
            "intersection_pairs": checked["count"],
            "intersection_pair_list_sha256": pair_digest,
            "first_intersecting_face_pairs": checked["triangle_pairs"][:16],
            "containment": "not_checked_after_surface_intersection",
        }

    first_in_second = point_location(first_vertices[0], second["records"])
    second_in_first = point_location(second_vertices[0], first["records"])
    locations = (first_in_second["location"], second_in_first["location"])
    if "indeterminate" in locations or "boundary" in locations:
        status = "indeterminate_containment"
    elif locations == ("outside", "outside"):
        status = "separate_closed_domains"
    else:
        status = "nested_closed_domains"
    return {
        "status": status,
        "aabb_candidate_pairs": checked["aabb_candidate_pairs"],
        "intersection_pairs": 0,
        "containment": {
            "first_representative_vertex_in_second": first_in_second,
            "second_representative_vertex_in_first": second_in_first,
        },
    }


def _exact_integer_vertices(
    vertices: list[list[float]], scale: int
) -> list[tuple[int, int, int]]:
    converted: list[tuple[int, int, int]] = []
    for point in vertices:
        result = []
        for value in point:
            _require(
                type(value) in (int, float) and math.isfinite(float(value)),
                "compiled coordinate is nonfinite",
            )
            numerator, denominator = float(value).as_integer_ratio()
            _require(
                scale % denominator == 0, "common exact coordinate scale is invalid"
            )
            result.append(numerator * (scale // denominator))
        converted.append(tuple(result))
    return converted


def audit(
    payload: Path, manifest_path: Path, embeddedness_path: Path, output: Path
) -> dict[str, Any]:
    import numpy as np

    payload, manifest_path, embeddedness_path = (
        Path(path).resolve() for path in (payload, manifest_path, embeddedness_path)
    )
    # Preserve the final path component so _write_immutable can reject a symlink.
    output = Path(output).absolute()
    for path, label in (
        (payload, "payload"),
        (manifest_path, "manifest"),
        (embeddedness_path, "embeddedness receipt"),
    ):
        _require(
            path.is_file() and not path.is_symlink(),
            f"{label} is missing or redirected",
        )
    manifest = human.read_json(manifest_path)
    embeddedness = human.read_json(embeddedness_path)
    manifest_sha = _sha256_file(manifest_path)
    embeddedness_sha = _sha256_file(embeddedness_path)
    payload_sha = _sha256_file(payload)
    declared = manifest.get("payload")
    _require(
        manifest.get("schema")
        == "numi.human.bodyparts3d-myosim-fullbody-muscle-surface-visual-payload.v1"
        and isinstance(declared, dict)
        and declared.get("file") == payload.name
        and declared.get("sha256") == payload_sha
        and declared.get("bytes") == payload.stat().st_size,
        "manifest does not bind this NHTISS4 payload",
    )
    _require(
        embeddedness.get("schema")
        == "numi.human.compiled-muscle-surface-embeddedness.v1"
        and embeddedness.get("source", {}).get("payload_sha256") == payload_sha
        and embeddedness.get("source", {}).get("manifest_sha256") == manifest_sha,
        "self-embeddedness receipt does not bind this exact payload and manifest",
    )
    live_predicates = {
        name: _sha256_file(Path(__file__).with_name(name))
        for name in PREDICATE_FILES[1:]
    }
    _require(
        embeddedness["source"].get("predicate_sha256") == live_predicates,
        "self-embeddedness predicate source changed since its receipt",
    )

    raw = payload.read_bytes()
    _require(len(raw) >= 64, "truncated NHTISS4 payload")
    (
        magic,
        abi,
        record_count,
        binding_count,
        vertex_count,
        index_count,
        fingerprint,
        source_sha,
    ) = struct.unpack_from("<8s6I32s", raw)
    _require(
        magic == b"NHTISS4\0"
        and abi == 5
        and record_count == 150
        and len(raw)
        == 64
        + 32 * record_count
        + 36 * binding_count
        + 56 * vertex_count
        + 4 * index_count,
        "unsupported NHTISS4 ABI or payload byte ranges",
    )
    _require(
        declared.get("surface_count") == record_count
        and declared.get("binding_count") == binding_count
        and declared.get("vertex_count") == vertex_count
        and declared.get("index_count") == index_count
        and declared.get("registration_fingerprint32") == f"{fingerprint:08x}"
        and manifest.get("source", {}).get("myosim_source_archive_sha256")
        == source_sha.hex(),
        "manifest semantics differ from the compiled payload",
    )

    records = np.frombuffer(
        raw, dtype="<u4", count=record_count * 8, offset=64
    ).reshape(-1, 8)
    vertex_offset = 64 + 32 * record_count + 36 * binding_count
    positions = np.ndarray(
        (vertex_count, 3),
        dtype="<f4",
        buffer=raw,
        offset=vertex_offset,
        strides=(56, 4),
    )
    index_offset = vertex_offset + 56 * vertex_count
    indices = np.frombuffer(raw, dtype="<u4", count=index_count, offset=index_offset)
    _require(bool(np.isfinite(positions).all()), "payload contains nonfinite vertices")
    manifest_rows = manifest.get("source", {}).get("surfaces")
    evidence_rows = embeddedness.get("surfaces")
    _require(
        isinstance(manifest_rows, list)
        and isinstance(evidence_rows, list)
        and len(manifest_rows) == len(evidence_rows) == record_count,
        "complete surface source identity is required",
    )
    evidence_by_id = {
        row.get("stable_id"): row for row in evidence_rows if isinstance(row, dict)
    }
    _require(
        len(evidence_by_id) == record_count,
        "embeddedness surface identities are malformed",
    )

    surface_rows: list[dict[str, Any]] = []
    exact_meshes: list[dict[str, Any]] = []
    quotient_vertices_for_scale: list[list[list[float]]] = []
    for source, record in zip(manifest_rows, records, strict=True):
        (
            _,
            _,
            first_vertex,
            local_vertex_count,
            first_index,
            local_index_count,
            stable_id,
            layer_code,
        ) = (int(value) for value in record)
        member_id = source.get("member_id")
        evidence = evidence_by_id.get(stable_id)
        expected_layer = {1: "muscle", 2: "tendon"}.get(layer_code)
        _require(
            isinstance(member_id, str)
            and evidence is not None
            and source.get("stable_id") == stable_id
            and source.get("layer") == expected_layer
            and evidence.get("member_id") == member_id
            and evidence.get("layer") == expected_layer,
            "payload, manifest and self-embeddedness identities disagree",
        )
        _require(
            local_index_count > 0
            and local_index_count % 3 == 0
            and first_vertex + local_vertex_count <= vertex_count
            and first_index + local_index_count <= index_count,
            f"invalid geometry ranges for {member_id}",
        )
        global_faces = indices[first_index : first_index + local_index_count]
        _require(
            bool(
                (
                    (global_faces >= first_vertex)
                    & (global_faces < first_vertex + local_vertex_count)
                ).all()
            ),
            f"surface indices escape {member_id}",
        )
        vertex_bytes = raw[
            vertex_offset + 56 * first_vertex : vertex_offset
            + 56 * (first_vertex + local_vertex_count)
        ]
        index_bytes = raw[
            index_offset + 4 * first_index : index_offset
            + 4 * (first_index + local_index_count)
        ]
        geometry_sha = hashlib.sha256(vertex_bytes + index_bytes).hexdigest()
        _require(
            evidence.get("compiled_geometry_sha256") == geometry_sha,
            f"self-embeddedness geometry hash differs for {member_id}",
        )
        is_candidate = (
            expected_layer == "muscle"
            and evidence.get("status") == "closed_embedded_source_candidate"
            and evidence.get("single_embedded_surface_candidate") is True
        )
        surface = {
            "stable_id": stable_id,
            "member_id": member_id,
            "layer": expected_layer,
            "compiled_geometry_sha256": geometry_sha,
            "triangle_count": local_index_count // 3,
            "closed_embedded_candidate": is_candidate,
        }
        surface_rows.append(surface)
        if not is_candidate:
            continue
        source_vertices = (
            positions[first_vertex : first_vertex + local_vertex_count, :3]
            .astype(float)
            .tolist()
        )
        local_faces = (global_faces.reshape(-1, 3) - first_vertex).tolist()
        quotient_vertices, quotient_faces, _ = coordinate_quotient(
            source_vertices, local_faces
        )
        surface["quotient_vertex_count"] = len(quotient_vertices)
        surface["quotient_triangle_count"] = len(quotient_faces)
        quotient_vertices_for_scale.append(quotient_vertices)
        exact_meshes.append(
            {
                "surface": surface,
                "vertices_float": quotient_vertices,
                "faces": quotient_faces,
            }
        )

    _require(
        len(exact_meshes) > 0,
        "self-embeddedness receipt has no closed muscle candidates",
    )
    common_denominator = 1
    for vertices in quotient_vertices_for_scale:
        for point in vertices:
            for value in point:
                denominator = float(value).as_integer_ratio()[1]
                _require(
                    denominator & (denominator - 1) == 0,
                    "compiled coordinate denominator is not a power of two",
                )
                common_denominator = max(common_denominator, denominator)

    for mesh in exact_meshes:
        integer_vertices = _exact_integer_vertices(
            mesh.pop("vertices_float"), common_denominator
        )
        mesh["vertices"] = integer_vertices
        mesh["records"] = _records(integer_vertices, mesh["faces"])

    pair_rows: list[dict[str, Any]] = []
    for first_index, first in enumerate(exact_meshes):
        for second in exact_meshes[first_index + 1 :]:
            pair = _surface_pair_status(first, second)
            pair_rows.append(
                {
                    "first_member_id": first["surface"]["member_id"],
                    "first_stable_id": first["surface"]["stable_id"],
                    "second_member_id": second["surface"]["member_id"],
                    "second_stable_id": second["surface"]["stable_id"],
                    **pair,
                }
            )
            if len(pair_rows) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "pairs_audited": len(pair_rows),
                            "pair_status_counts": dict(
                                Counter(row["status"] for row in pair_rows)
                            ),
                        }
                    ),
                    flush=True,
                )

    pair_counts = Counter(row["status"] for row in pair_rows)
    candidate_count = len(exact_meshes)
    expected_pair_count = candidate_count * (candidate_count - 1) // 2
    _require(
        len(pair_rows) == expected_pair_count, "pairwise muscle audit is incomplete"
    )
    pairwise_domains_disjoint = all(
        row["status"] in {"strictly_disjoint_aabbs", "separate_closed_domains"}
        for row in pair_rows
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "pairwise_muscle_volume_domains_disjoint"
        if pairwise_domains_disjoint
        else "pairwise_muscle_volume_domains_not_disjoint",
        "source": {
            "payload": str(payload.relative_to(human.REPOSITORY_ROOT))
            if payload.is_relative_to(human.REPOSITORY_ROOT)
            else str(payload),
            "payload_sha256": payload_sha,
            "manifest": str(manifest_path.relative_to(human.REPOSITORY_ROOT))
            if manifest_path.is_relative_to(human.REPOSITORY_ROOT)
            else str(manifest_path),
            "manifest_sha256": manifest_sha,
            "self_embeddedness_receipt": str(
                embeddedness_path.relative_to(human.REPOSITORY_ROOT)
            )
            if embeddedness_path.is_relative_to(human.REPOSITORY_ROOT)
            else str(embeddedness_path),
            "self_embeddedness_receipt_sha256": embeddedness_sha,
            "common_exact_coordinate_scale_power_of_two": common_denominator.bit_length()
            - 1,
            "predicate_sha256": {
                name: _sha256_file(Path(__file__).with_name(name))
                for name in PREDICATE_FILES
            },
        },
        "coverage": {
            "source_surface_count": len(surface_rows),
            "source_muscle_surface_count": sum(
                row["layer"] == "muscle" for row in surface_rows
            ),
            "source_tendon_surface_count": sum(
                row["layer"] == "tendon" for row in surface_rows
            ),
            "closed_embedded_muscle_candidate_count": candidate_count,
            "candidate_pair_count": len(pair_rows),
            "expected_candidate_pair_count": expected_pair_count,
            "pair_status_counts": dict(sorted(pair_counts.items())),
            "pairwise_domains_disjoint": pairwise_domains_disjoint,
        },
        "surfaces": surface_rows,
        "pairs": pair_rows,
        "qualification": {
            "pairwise_embedded_muscle_candidate_domains_disjoint": pairwise_domains_disjoint,
            "all_source_muscle_surfaces_admitted": candidate_count == 148,
            "cross_layer_disjointness": False,
            "whole_body_disjointness": False,
            "physical_volume_owner": False,
            "mechanical_mass_owner": False,
            "muscle_mechanics": False,
        },
        "boundary": (
            "Exact pairwise intersection and containment checks for the closed, "
            "self-embedded compiled muscle-surface candidates only. Invalid/open "
            "muscle surfaces, tendons, skin, bone, organs and other tissue layers "
            "are excluded from this volume-domain pair set. This audit does not "
            "establish a disjoint whole-body partition or assign any physical "
            "volume, mass, material, force, or mechanics owner."
        ),
    }
    _write_immutable(output, result)
    return result


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--embeddedness",
        type=Path,
        required=True,
        help="exact compiled muscle self-embeddedness receipt for this payload",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.set_defaults(handler=run)


def run(arguments: argparse.Namespace) -> int:
    result = audit(
        arguments.payload, arguments.manifest, arguments.embeddedness, arguments.output
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_muscles": result["coverage"][
                    "closed_embedded_muscle_candidate_count"
                ],
                "candidate_pairs": result["coverage"]["candidate_pair_count"],
                "pair_status_counts": result["coverage"]["pair_status_counts"],
                "physical_volume_owner": result["qualification"][
                    "physical_volume_owner"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return run(parser.parse_args(argv))
    except (
        human.ImportError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        struct.error,
    ) as error:
        parser.exit(2, f"muscle-surface-volume-disjointness: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
