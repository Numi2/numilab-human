#!/usr/bin/env python3
"""Prepare an experimental NHSKIN ABI 5 owner-partitioned skin candidate.

This splits the registered rest-world surface by an explicit body-side weight
scalar, caps only the newly created seam, preserves exact declared pre-existing
boundary edges, and emits one same-ABI mesh plus ancestry/report sidecars.
It is visual geometry preparation; it does not create tissue fracture physics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from numilab_human.resting_scene import (
    NHCNT_HEADER, NHCNT_MAGIC, NHCNT_RECORD,
    load_rigid, load_skin, source_shell_world,
)
from numilab_human.waist_skin_partition import (
    _uses,
    decode_nhskin_abi5,
    emit_nhskin_abi5_half,
    split_surface_by_signed_field,
)

HEADER = struct.Struct("<8s5I32s")
VERTEX = np.dtype([
    ("p", "<f4", (3,)), ("n", "<f4", (3,)),
    ("bi", "<u4", (4,)), ("bw", "<f4", (4,)),
])


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_partition(path: Path):
    doc = json.loads(path.read_text())
    part = doc.get("partition", doc)
    parent = set(map(int, part["parent_side_body_indices"]))
    child = set(map(int, part["child_side_body_indices"]))
    if not parent or not child or parent & child:
        raise ValueError("partition owner sides must be nonempty and disjoint")
    return parent, child


def _load_existing_boundaries(path: Path, source_sha: str):
    doc = json.loads(path.read_text())
    if doc.get("input_sha256") != source_sha:
        raise ValueError("existing-boundary declaration is not pinned to this NHSKIN payload")
    edges = {tuple(sorted(map(int, edge))) for edge in doc["allowed_existing_boundary_edges"]}
    if not edges:
        raise ValueError("existing-boundary declaration must be explicit and nonempty")
    return doc, edges


def _read_half(payload: bytes, source_binding_indices: list[int]):
    decoded = decode_nhskin_abi5(payload)
    if len(source_binding_indices) != decoded["binding_count"]:
        raise ValueError("emitted half binding map does not match NHSKIN header")
    rows = np.frombuffer(
        payload, dtype=VERTEX, count=decoded["vertex_count"],
        offset=HEADER.size + 36 * decoded["binding_count"],
    ).copy()
    # The C++ loader validates all four diagnostic IDs, including zero-weight lanes.
    if np.any(rows["bi"] >= decoded["binding_count"]):
        raise ValueError("emitted half has an out-of-range diagnostic binding index")
    rows["bi"] = np.asarray(source_binding_indices, dtype=np.uint32)[rows["bi"]]
    return decoded, rows


def _ancestry(half):
    source_vertex = np.full(len(half.vertex_provenance), -1, dtype=np.int64)
    edge_a = np.full(len(source_vertex), -1, dtype=np.int64)
    edge_b = np.full(len(source_vertex), -1, dtype=np.int64)
    t = np.full(len(source_vertex), np.nan, dtype=np.float64)
    for i, row in enumerate(half.vertex_provenance):
        if row["kind"] == "source_vertex":
            source_vertex[i] = int(row["source_vertex"])
        elif row["kind"] == "source_edge":
            edge_a[i], edge_b[i] = map(int, row["source_edge"])
            t[i] = float(row["t_from_first"])
        else:
            raise ValueError(f"unsupported vertex ancestry record {row!r}")
    return source_vertex, edge_a, edge_b, t


def _remap_nhcnt_source_geometry(source_path, output_path, receipt_path, *,
                                 rigid, source_vertex_count, source_to_output,
                                 candidate_skin_sha):
    """Remap NHCNT1's 1-based NHSKIN source vertex IDs through exact ancestry."""
    source = source_path.read_bytes()
    if len(source) < NHCNT_HEADER.size:
        raise ValueError("NHCNT source payload is truncated")
    (magic, abi, body_count, count, reserved, rigid_source_sha,
     *header_values) = NHCNT_HEADER.unpack_from(source)
    if (magic != NHCNT_MAGIC or abi != 1 or reserved != 0
            or body_count != rigid["body_count"] or count <= 0
            or len(source) != NHCNT_HEADER.size + count * NHCNT_RECORD.size
            or rigid_source_sha.hex() != rigid["source_sha256"]):
        raise ValueError("NHCNT1 header does not match the pinned rigid source")
    output = bytearray(source)
    mappings = []
    changed_fields = set()
    for i in range(count):
        offset = NHCNT_HEADER.size + i * NHCNT_RECORD.size
        values = NHCNT_RECORD.unpack_from(source, offset)
        body_index, source_geometry_id = map(int, values[:2])
        source_vertex = source_geometry_id - 1
        if source_vertex < 0 or source_vertex >= source_vertex_count:
            raise ValueError(f"NHCNT row {i} source geometry ID is outside pinned NHSKIN")
        if source_vertex not in source_to_output:
            raise ValueError(f"NHCNT row {i} references a source vertex absent from the cut output")
        target_geometry_id = int(source_to_output[source_vertex]) + 1
        struct.pack_into("<I", output, offset + 4, target_geometry_id)
        if target_geometry_id != source_geometry_id:
            changed_fields.update(range(offset + 4, offset + 8))
        mappings.append({
            "row": i, "body_index": body_index,
            "source_geometry_id_1based": source_geometry_id,
            "source_vertex_index_0based": source_vertex,
            "candidate_geometry_id_1based": target_geometry_id,
            "candidate_vertex_index_0based": target_geometry_id - 1,
        })
    for offset, (left, right) in enumerate(zip(source, output, strict=True)):
        if left != right and offset not in changed_fields:
            raise ValueError("NHCNT remap changed bytes outside sourceGeometryIndex fields")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output)
    remap_report = {
        "schema": "numi.human.nhcnt-source-geometry-ancestry-remap.v1",
        "status": "exact_source_vertex_ancestry_remap",
        "source_payload": {"path": str(source_path),
                           "sha256": hashlib.sha256(source).hexdigest(),
                           "bytes": len(source)},
        "candidate_payload": {"path": str(output_path),
                              "sha256": hashlib.sha256(output).hexdigest(),
                              "bytes": len(output)},
        "candidate_skin_sha256": candidate_skin_sha,
        "record_count": count,
        "record_size_bytes": NHCNT_RECORD.size,
        "source_geometry_index_semantics": "1-based NHSKIN source vertex index",
        "source_geometry_ids_remapped": sum(
            row["source_geometry_id_1based"] != row["candidate_geometry_id_1based"]
            for row in mappings
        ),
        "all_non_geometry_id_bytes_identical": True,
        "records": mappings,
        "evidence_boundary": "Preserves the existing NHCNT contact seed identities through NHSKIN vertex compaction; does not qualify contact coverage or mechanical fracture.",
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(remap_report, indent=2) + "\n")
    return remap_report


def prepare(args):
    skin_path, rigid_path = args.skin, args.rigid
    skin_sha, rigid_sha = _sha(skin_path), _sha(rigid_path)
    if skin_sha != args.expected_skin_sha256.lower():
        raise ValueError(f"NHSKIN SHA-256 mismatch: {skin_sha}")
    if rigid_sha != args.expected_rigid_sha256.lower():
        raise ValueError(f"NHRIGID SHA-256 mismatch: {rigid_sha}")

    raw = skin_path.read_bytes()
    decoded = decode_nhskin_abi5(raw)
    rigid = load_rigid(rigid_path)
    skin = load_skin(skin_path, rigid)
    world, _, _ = source_shell_world(rigid, skin)
    if world.shape != (decoded["vertex_count"], 3):
        raise ValueError("registered rest-world vertices differ from NHSKIN vertex count")

    parent_body_ids, child_body_ids = _load_partition(args.partition)
    boundary_doc, declared_boundary = _load_existing_boundaries(
        args.existing_boundaries, skin_sha
    )
    binding_bodies = np.asarray([int(row[0]) for row in skin["bindings"]], dtype=np.int64)
    parent_columns = np.flatnonzero(np.isin(binding_bodies, list(parent_body_ids)))
    child_columns = np.flatnonzero(np.isin(binding_bodies, list(child_body_ids)))
    if len(parent_columns) + len(child_columns) != len(binding_bodies):
        raise ValueError("partition does not assign every NHSKIN binding row exactly once")
    if set(parent_columns) & set(child_columns):
        raise ValueError("partition assigns a binding row to both sides")

    parent_support = skin["weights"][:, parent_columns].sum(axis=1)
    child_support = skin["weights"][:, child_columns].sum(axis=1)
    owner_field = child_support - parent_support
    positive, negative = split_surface_by_signed_field(
        world, decoded["triangles"], owner_field,
        attributes={
            "full_binding_weights": skin["weights"],
            "registered_rest_normals": decoded["vertex_rows"][:, 3:6],
        },
        allowed_existing_boundary_edges=declared_boundary,
        max_projection_residual=args.max_cap_projection_residual_m,
    )
    seam_rows = [row for row in positive.vertex_provenance if row["kind"] == "source_edge"]
    if not seam_rows:
        raise ValueError("owner scalar produced no interpolated seam")
    seam_vertices = np.asarray([
        positive.vertices[i] for i, row in enumerate(positive.vertex_provenance)
        if row["kind"] == "source_edge"
    ])
    anchor = np.asarray(args.anchor, dtype=np.float64)
    if anchor.shape != (3,) or not np.isfinite(anchor).all():
        raise ValueError("anchor must be three finite world coordinates")
    max_seam_radius = float(np.linalg.norm(seam_vertices - anchor, axis=1).max())
    if max_seam_radius > args.max_seam_radius_m:
        raise ValueError(f"owner interface exceeds declared waist radius ({max_seam_radius:.6g} m)")

    child_payload, child_info = emit_nhskin_abi5_half(raw, positive, rigid, skin, child_body_ids)
    parent_payload, parent_info = emit_nhskin_abi5_half(raw, negative, rigid, skin, parent_body_ids)
    child, child_rows = _read_half(child_payload, child_info["binding_indices_from_source"])
    parent, parent_rows = _read_half(parent_payload, parent_info["binding_indices_from_source"])
    if child["registration_fingerprint"] != parent["registration_fingerprint"] or child["source_sha256"] != parent["source_sha256"]:
        raise ValueError("half registration/source metadata mismatch")

    nv = child["vertex_count"] + parent["vertex_count"]
    nb = decoded["binding_count"]
    triangles = np.concatenate((
        child["triangles"],
        parent["triangles"] + child["vertex_count"],
    ))
    vertices = np.concatenate((child_rows, parent_rows))
    full_weights = np.zeros((nv, nb), dtype="<f4")
    full_weights[:child["vertex_count"], child_info["binding_indices_from_source"]] = child["weights"]
    full_weights[child["vertex_count"]:, parent_info["binding_indices_from_source"]] = parent["weights"]
    if np.max(np.abs(full_weights.astype(np.float64).sum(axis=1) - 1.0)) > 3e-5:
        raise ValueError("merged NHSKIN full-weight rows do not sum to one")
    bind_start = HEADER.size
    binding_bytes = raw[bind_start:bind_start + 36 * nb]
    payload = (
        HEADER.pack(b"NHSKIN1\0", 5, nb, nv, int(triangles.size),
                    decoded["registration_fingerprint"],
                    bytes.fromhex(decoded["source_sha256"]))
        + binding_bytes
        + vertices.tobytes()
        + np.asarray(triangles, dtype="<u4").reshape(-1).tobytes()
        + full_weights.tobytes()
    )
    if len(payload) != HEADER.size + 36*nb + 56*nv + 4*triangles.size + 4*nv*nb:
        raise ValueError("merged NHSKIN byte extent is inconsistent")

    child_sv, child_ea, child_eb, child_t = _ancestry(positive)
    parent_sv, parent_ea, parent_eb, parent_t = _ancestry(negative)
    source_face = np.concatenate((positive.triangle_source_face, negative.triangle_source_face))
    triangle_kind = np.concatenate((positive.triangle_kind, negative.triangle_kind))
    cap_loop = np.concatenate((
        positive.triangle_cap_loop,
        np.where(negative.triangle_cap_loop >= 0, negative.triangle_cap_loop + 1, -1),
    ))
    if not np.array_equal(
        np.unique(source_face[source_face >= 0]), np.arange(len(decoded["triangles"]))
    ):
        raise ValueError("source-face ancestry does not cover every source triangle")
    if not np.array_equal(triangle_kind == 1, source_face == -1):
        raise ValueError("cap/source triangle ancestry labels disagree")

    uses = _uses(triangles.astype(np.int64))
    if any(len(rows) > 2 for rows in uses.values()):
        raise ValueError("merged output contains a nonmanifold edge")
    for rows in uses.values():
        if len(rows) == 2 and (rows[0][1], rows[0][2]) != (rows[1][2], rows[1][1]):
            raise ValueError("merged output has inconsistent edge winding")
    actual_boundary = {edge for edge, rows in uses.items() if len(rows) == 1}
    adjacency = [set() for _ in range(len(triangles))]
    for rows in uses.values():
        if len(rows) == 2:
            left, right = rows[0][0], rows[1][0]
            adjacency[left].add(right)
            adjacency[right].add(left)
    unseen = set(range(len(triangles)))
    component_faces = []
    while unseen:
        stack = [unseen.pop()]
        count = 0
        while stack:
            face = stack.pop()
            count += 1
            next_faces = adjacency[face] & unseen
            unseen.difference_update(next_faces)
            stack.extend(next_faces)
        component_faces.append(count)
    if len(component_faces) != 2:
        raise ValueError(f"expected exactly two detached surface components, found {len(component_faces)}")
    child_source_to_vertex = {}
    for i, source_vertex in enumerate(child_sv):
        if source_vertex >= 0:
            if int(source_vertex) in child_source_to_vertex:
                raise ValueError("duplicate source vertex on child half")
            child_source_to_vertex[int(source_vertex)] = i
    expected_boundary = set()
    for a, b in declared_boundary:
        if a not in child_source_to_vertex or b not in child_source_to_vertex:
            raise ValueError("declared retained boundary did not remain on child half")
        expected_boundary.add(tuple(sorted((child_source_to_vertex[a],
                                            child_source_to_vertex[b]))))
    if actual_boundary != expected_boundary:
        raise ValueError("merged boundary differs from exact declared existing openings")

    source_vertex = np.concatenate((child_sv, parent_sv))
    edge_a = np.concatenate((child_ea, parent_ea))
    edge_b = np.concatenate((child_eb, parent_eb))
    edge_t = np.concatenate((child_t, parent_t))
    side = np.concatenate((
        np.ones(child["vertex_count"], dtype=np.int8),
        -np.ones(parent["vertex_count"], dtype=np.int8),
    ))
    face_side = np.concatenate((
        np.ones(child["index_count"] // 3, dtype=np.int8),
        -np.ones(parent["index_count"] // 3, dtype=np.int8),
    ))
    expected_seam_vertices = len(seam_rows)
    if np.count_nonzero(edge_a >= 0) != expected_seam_vertices * 2:
        raise ValueError("both halves do not retain every seam vertex ancestry record")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    tmp.write_bytes(payload)
    tmp.replace(args.output)
    args.lineage.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.lineage,
        vertex_source_index=source_vertex, vertex_source_edge_a=edge_a,
        vertex_source_edge_b=edge_b, vertex_t_from_edge_a=edge_t,
        vertex_side=side, triangle_source_face=source_face,
        triangle_kind=triangle_kind, triangle_cap_loop=cap_loop,
        triangle_side=face_side,
    )

    output_skin = load_skin(args.output, rigid)
    output_world, _, _ = source_shell_world(rigid, output_skin)
    expected_world = np.concatenate((positive.vertices, negative.vertices))
    rest_error = float(np.linalg.norm(output_world - expected_world, axis=1).max())
    if rest_error > 1e-4:
        raise ValueError(f"merged NHSKIN rest-world error exceeds 0.1mm: {rest_error:.6g} m")
    referenced = np.unique(decoded["triangles"])
    if not np.array_equal(np.unique(source_vertex[source_vertex >= 0]), referenced):
        raise ValueError("source vertex ancestry differs from source-referenced vertices")
    source_to_output = {}
    for output_index, source_index in enumerate(source_vertex):
        if source_index >= 0:
            if int(source_index) in source_to_output:
                raise ValueError("a source vertex was duplicated across the owner partition")
            source_to_output[int(source_index)] = output_index

    support_remap = None
    if args.support_contact_source is not None:
        if args.support_contact_output is None or args.support_contact_receipt is None:
            raise ValueError("NHCNT remapping requires both output and receipt paths")
        support_remap = _remap_nhcnt_source_geometry(
            args.support_contact_source, args.support_contact_output,
            args.support_contact_receipt, rigid=rigid,
            source_vertex_count=decoded["vertex_count"],
            source_to_output=source_to_output,
            candidate_skin_sha=hashlib.sha256(payload).hexdigest(),
        )

    report = {
        "schema": "numi.human.waist-owner-field-visual-skin-candidate.v1",
        "status": "experimental_same_abi_visual_skin_candidate_prepared",
        "inputs": {
            "skin_path": str(skin_path), "skin_sha256": skin_sha,
            "rigid_path": str(rigid_path), "rigid_sha256": rigid_sha,
            "partition_path": str(args.partition),
            "existing_boundary_path": str(args.existing_boundaries),
        },
        "partition": {
            "parent_body_count": len(parent_body_ids), "child_body_count": len(child_body_ids),
            "parent_binding_count": len(parent_info["binding_indices_from_source"]),
            "child_binding_count": len(child_info["binding_indices_from_source"]),
            "scalar": "sum(child-side full NHSKIN weights) - sum(parent-side full NHSKIN weights)",
        },
        "interface": {
            "anchor_world_m": anchor.tolist(), "max_seam_radius_m": max_seam_radius,
            "seam_vertex_count": expected_seam_vertices,
            "cap_loop_projection_residuals_m": positive.cap_loop_projection_residuals.tolist(),
            "max_cap_projection_residual_m": float(positive.cap_loop_projection_residuals.max()),
            "cap_validation": "deterministic best-fit projection; simple loops, nonoverlapping projected caps, consistent winding, nonmanifold and retained-boundary checks",
        },
        "output": {
            "path": str(args.output), "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload), "abi": 5, "binding_count": nb,
            "vertex_count": nv, "triangle_count": int(triangles.size // 3),
            "cap_triangle_count": int(np.count_nonzero(triangle_kind == 1)),
            "connected_component_count": len(component_faces),
            "component_triangle_counts": component_faces,
            "retained_boundary_edge_count": len(actual_boundary),
            "source_face_descendant_triangle_count": int(np.count_nonzero(source_face >= 0)),
            "source_face_ancestry_unique_count": int(len(np.unique(source_face[source_face >= 0]))),
            "rest_world_max_error_m": rest_error,
            "full_weight_row_sum_max_error": float(np.max(np.abs(full_weights.astype(np.float64).sum(axis=1)-1))),
            "binding_table_byte_identical": binding_bytes == raw[bind_start:bind_start + 36 * nb],
            "lineage_path": str(args.lineage),
            "lineage_sha256": _sha(args.lineage),
        },
        "support_contact_remap": support_remap,
        "preserved_existing_boundaries": {
            "edge_count": len(declared_boundary),
            "loop_edge_counts": [len(loop["edge_pairs"]) for loop in boundary_doc.get("loops", [])],
            "exact_output_edge_set": True,
        },
        "native_validation": "Not run by this preparation command; validate the exact output skin and remapped NHCNT hashes in the native owner before making runtime claims.",
        "evidence_boundary": "Visual mesh preparation and contact-seed ID remapping only; neither establishes physical tissue fracture nor anatomical qualification.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skin", type=Path, required=True)
    parser.add_argument("--rigid", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True,
                        help="JSON with parent_side_body_indices and child_side_body_indices")
    parser.add_argument("--existing-boundaries", type=Path, required=True,
                        help="exact source edge declaration pinned by input_sha256")
    parser.add_argument("--expected-skin-sha256", required=True)
    parser.add_argument("--expected-rigid-sha256", required=True)
    parser.add_argument("--anchor", type=float, nargs=3, required=True,
                        metavar=("X", "Y", "Z"), help="waist anchor in registered rest-world metres")
    parser.add_argument("--max-seam-radius-m", type=float, default=0.25)
    parser.add_argument("--max-cap-projection-residual-m", type=float, default=0.03)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--support-contact-source", type=Path,
                        help="optional NHCNT1 whose 1-based source vertex IDs must be remapped")
    parser.add_argument("--support-contact-output", type=Path)
    parser.add_argument("--support-contact-receipt", type=Path)
    args = parser.parse_args()
    report = prepare(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
