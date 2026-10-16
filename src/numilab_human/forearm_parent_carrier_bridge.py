"""Append full21 material points to existing ABI5 parent carriers.

The carrier rows remain load-only reference data. This module preserves every
existing carrier byte and emits one lane per ordered source parent influence;
it does not change anatomy, mechanics, or physical routes.
"""
from __future__ import annotations

import copy
import math
import struct
from typing import Sequence

import numpy as np


_HEADER = struct.Struct("<8s6I32s")
_ROW = struct.Struct("<8I")
_BINDING = struct.Struct("<I8f")
_VERTEX_DTYPE = np.dtype([
    ("vertex", "<f4", (6,)),
    ("binding", "<u4", (4,)),
    ("weight", "<f4", (4,)),
])
_MAX_PARENT_VERTICES = 4_194_304


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("forearm parent carrier bridge: " + message)


def validate_source_gid_slot_lineage(source_slots_i64, canonical_gids_u64,
                                      vertex_lineage: dict) -> dict:
    """Authenticate canonical GID/source-ID lineage; keep local slots and pack IDs separate."""
    slots = np.asarray(source_slots_i64)
    gids = np.asarray(canonical_gids_u64)
    _require(slots.ndim == 1 and np.issubdtype(slots.dtype, np.integer),
             "candidate source slots must be a one-dimensional integer array")
    _require(gids.ndim == 1 and np.issubdtype(gids.dtype, np.integer)
             and gids.shape == slots.shape,
             "candidate canonical GIDs must align one-to-one with source slots")
    _require(isinstance(vertex_lineage, dict),
             "candidate vertex lineage must be a keyed object")
    original = np.flatnonzero(slots >= 0)
    source_ids = []
    for vertex in original:
        gid = int(gids[vertex])
        item = vertex_lineage.get(str(gid))
        _require(isinstance(item, dict),
                 f"original material GID {gid} lacks source identity lineage")
        _require(item.get("source_face_support_provenance")
                 == "original-keyed-material-handoff-GID",
                 f"original material GID {gid} has unexpected ancestry kind")
        _require(type(item.get("source_gid")) is int
                 and item["source_gid"] == gid
                 and type(item.get("source_vertex_id")) is int
                 and item["source_vertex_id"] >= 0,
                 f"original material GID {gid} has malformed source identity")
        source_ids.append(int(item["source_vertex_id"]))
    return {
        "validated_original_vertex_count": int(len(original)),
        "validated_distinct_source_slot_count": int(len(set(map(int, slots[original])))),
        "validated_source_vertex_id_count": int(len(source_ids)),
        "canonical_gid_and_source_slot_are_vertex_aligned": True,
        "source_vertex_id_numeric_equality_to_local_slot_required": False,
        "pack_global_vertex_ids_used_for_identity": False,
    }


def resolve_birth_source_parent_slots(parent_gids, canonical_gids_u64,
                                      source_slots_i64) -> np.ndarray:
    """Resolve an inferred edge's canonical parent GIDs to original source slots."""
    parents = list(parent_gids)
    gids = np.asarray(canonical_gids_u64)
    slots = np.asarray(source_slots_i64)
    _require(len(parents) == 2 and all(type(gid) is int for gid in parents)
             and parents[0] <= parents[1],
             "birth ancestry must name two ordered canonical parent GIDs")
    _require(gids.ndim == 1 and np.issubdtype(gids.dtype, np.integer)
             and slots.ndim == 1 and np.issubdtype(slots.dtype, np.integer)
             and gids.shape == slots.shape,
             "candidate canonical GIDs and source slots are malformed")
    result = []
    for gid in parents:
        matches = np.flatnonzero(gids == gid)
        _require(len(matches) == 1,
                 f"birth parent GID {gid} is absent or ambiguous in candidate row")
        slot = int(slots[int(matches[0])])
        _require(slot >= 0,
                 f"birth parent GID {gid} is itself inferred; recursive ancestry is required")
        result.append(slot)
    return np.asarray(result, dtype=np.int64)


def _validated_parent_carrier(raw: bytes, *, stable_id: int,
                              core_body_indices: Sequence[int]) -> dict:
    _require(isinstance(raw, bytes) and len(raw) >= _HEADER.size + _ROW.size,
             "truncated ABI5 carrier")
    magic, abi, row_count, binding_count, vertex_count, index_count, fingerprint, digest = (
        _HEADER.unpack_from(raw)
    )
    _require(magic == b"NHTISS4\x00" and abi == 5 and row_count == 1,
             "expected a single-row NHTISS4 ABI5 carrier")
    _require(binding_count == len(core_body_indices) == 21,
             "carrier must retain the exact ordered 21-body binding table")
    _require(vertex_count > 0 and vertex_count % 21 == 0,
             "existing carrier vertices are not complete 21-lane groups")
    expected_bytes = (_HEADER.size + row_count * _ROW.size
                      + binding_count * 36 + vertex_count * 56 + index_count * 4)
    _require(len(raw) == expected_bytes, "carrier byte length disagrees with ABI counts")

    row = _ROW.unpack_from(raw, _HEADER.size)
    binding_start, row_binding_count, vertex_start, row_vertex_count, index_start, row_index_count, row_id, layer = row
    _require((binding_start, row_binding_count, vertex_start, row_vertex_count,
              index_start, row_index_count, row_id)
             == (0, binding_count, 0, vertex_count, 0, index_count, stable_id),
             "carrier row identity or section offsets differ")
    _require(layer in (1, 2, 3), "carrier layer is invalid")

    bindings_offset = _HEADER.size + row_count * _ROW.size
    binding_bytes = raw[bindings_offset:bindings_offset + binding_count * 36]
    binding_ids = []
    for offset in range(0, len(binding_bytes), _BINDING.size):
        core, *transform = _BINDING.unpack_from(binding_bytes, offset)
        _require(core not in binding_ids, "carrier repeats a core body")
        _require(all(math.isfinite(value) for value in transform)
                 and transform[7] > 0
                 and abs(math.sqrt(sum(value * value for value in transform[3:7])) - 1.0) <= 2e-3,
                 "carrier binding transform is invalid")
        binding_ids.append(core)
    _require(tuple(binding_ids) == tuple(int(x) for x in core_body_indices),
             "carrier binding order differs from the authenticated full21 order")

    vertices_offset = bindings_offset + binding_count * 36
    vertex_bytes = raw[vertices_offset:vertices_offset + vertex_count * 56]
    vertices = np.frombuffer(vertex_bytes, dtype=_VERTEX_DTYPE, count=vertex_count)
    grouped = vertices.reshape((-1, 21))
    _require(np.isfinite(grouped["vertex"]).all(), "existing carrier vertices are nonfinite")
    lane_ids = np.arange(21, dtype="<u4")
    _require(np.all(grouped["binding"][:, :, 0] == lane_ids[None, :])
             and np.all(grouped["binding"][:, :, 1:] == 0),
             "existing carrier lanes are not one-hot in canonical order")
    _require(np.all(grouped["weight"][:, :, 0] == np.float32(1))
             and np.all(grouped["weight"][:, :, 1:] == 0),
             "existing carrier lane weights are not exact one-hot values")
    vertex_bits = grouped["vertex"].view("<u4")
    _require(np.all(vertex_bits == vertex_bits[:, :1, :]),
             "existing 21 lanes do not share byte-identical source point and normal")

    indices_offset = vertices_offset + vertex_count * 56
    indices = np.frombuffer(raw, dtype="<u4", count=index_count, offset=indices_offset)
    _require(index_count > 0 and index_count % 3 == 0
             and int(indices.max(initial=0)) < vertex_count
             and np.all(indices % 21 == 0),
             "carrier faces do not reference valid representative lane-zero vertices")
    return {
        "header": (magic, abi, row_count, binding_count, vertex_count,
                   index_count, fingerprint, digest),
        "row": row,
        "binding_bytes": binding_bytes,
        "vertices_offset": vertices_offset,
        "vertex_bytes": vertex_bytes,
        "indices_offset": indices_offset,
        "indices_bytes": raw[indices_offset:],
    }


def append_full21_carrier_points(
    raw: bytes, *, stable_id: int, core_body_indices: Sequence[int],
    source_points6: np.ndarray,
) -> tuple[bytes, int, dict]:
    """Append source point/normal records as 21 one-hot carrier lanes.

    Returns (payload, first_new_local_vertex, report). New carrier slots are
    unreferenced by faces by design; the parent extension references them by
    exact local vertex ordinal.
    """
    owner = _validated_parent_carrier(
        raw, stable_id=stable_id, core_body_indices=core_body_indices,
    )
    incoming = np.asarray(source_points6)
    _require(incoming.dtype in (np.dtype("<f4"), np.dtype("float32")),
             "new source points must already be emitted Float32")
    _require(incoming.ndim == 2 and incoming.shape[1] == 6 and len(incoming) > 0,
             "new source point array must have shape (N, 6)")
    points = np.asarray(incoming, dtype="<f4")
    _require(np.isfinite(points).all(), "new source points contain nonfinite values")
    normal_lengths = np.linalg.norm(points[:, 3:].astype(np.float64), axis=1)
    _require(np.all(np.abs(normal_lengths - 1.0) <= 2e-3),
             "new source normals must be unit length")
    magic, abi, row_count, binding_count, old_vertex_count, index_count, fingerprint, digest = owner["header"]
    new_count = len(points) * 21
    _require(old_vertex_count + new_count <= _MAX_PARENT_VERTICES,
             "extended carrier exceeds the existing parent-load vertex budget")

    appended = np.zeros(new_count, dtype=_VERTEX_DTYPE)
    appended["vertex"] = np.repeat(points[:, None, :], 21, axis=1).reshape((-1, 6))
    appended["binding"][:, 0] = np.tile(np.arange(21, dtype="<u4"), len(points))
    appended["weight"][:, 0] = np.float32(1.0)
    appended_bytes = appended.tobytes()

    updated_header = _HEADER.pack(
        magic, abi, row_count, binding_count, old_vertex_count + new_count,
        index_count, fingerprint, digest,
    )
    row = list(owner["row"])
    row[3] += new_count
    payload = (
        updated_header + _ROW.pack(*row) + owner["binding_bytes"]
        + owner["vertex_bytes"] + appended_bytes + owner["indices_bytes"]
    )
    _require(len(payload) == (_HEADER.size + _ROW.size + binding_count * 36
                             + (old_vertex_count + new_count) * 56 + index_count * 4),
             "emitted ABI5 carrier length is inconsistent")
    return payload, old_vertex_count, {
        "stable_id": stable_id,
        "old_vertex_count": old_vertex_count,
        "appended_source_point_count": len(points),
        "appended_parent_lane_count": new_count,
        "first_new_local_vertex": old_vertex_count,
        "ordered_core_body_indices": [int(x) for x in core_body_indices],
        "old_header_record_binding_vertex_prefix_and_indices_byte_exact": True,
        "scope": "load-only source carrier extension; not anatomy or native admission",
    }


def full21_parent_terms(*, source_index: int, stable_id: int,
                        first_parent_vertex: int,
                        core_weights_float64: np.ndarray) -> list[list[list[float | int]]]:
    """Emit every ordered positive parent term for appended carrier point groups."""
    for name, value in (("source_index", source_index), ("stable_id", stable_id),
                        ("first_parent_vertex", first_parent_vertex)):
        _require(type(value) is int and value >= 0, f"{name} must be a nonnegative integer")
    _require(stable_id > 0, "stable_id must be positive")
    weights = np.asarray(core_weights_float64)
    _require(weights.dtype == np.dtype("float64") and weights.ndim == 2
             and weights.shape[1] == 21 and len(weights) > 0,
             "source weights must be the complete ordered Float64 21-field")
    _require(np.isfinite(weights).all() and np.all(weights > 0)
             and np.all(weights <= 1)
             and np.max(np.abs(weights.sum(axis=1) - 1.0)) <= 1e-12,
             "source weights must be a positive normalized full21 partition")
    _require(np.all(weights.astype("<f4") > 0),
             "positive source parent coefficient underflows Float32")
    result = []
    for vertex, row in enumerate(weights):
        first = first_parent_vertex + vertex * 21
        result.append([
            [source_index, stable_id, first + lane, float(row[lane])]
            for lane in range(21)
        ])
    return result


def append_parent_surface_terms(
    extension: dict, *, stable_id: int, source_index: int,
    expected_child_payload_sha256: str, child_payload_sha256: str,
    expected_source_payload_sha256: str, source_payload_path: str,
    source_payload_sha256: str, expected_existing_term_count: int,
    appended_terms: list[list[list[float | int]]],
) -> dict:
    """Return a pinned extension with new terms; preserve old and unrelated rows."""
    _require(type(stable_id) is int and stable_id > 0, "invalid child stable ID")
    _require(type(source_index) is int and source_index >= 0, "invalid source index")
    for value in (expected_child_payload_sha256, child_payload_sha256,
                  expected_source_payload_sha256, source_payload_sha256):
        _require(isinstance(value, str) and len(value) == 64
                 and all(ch in "0123456789abcdef" for ch in value),
                 "invalid pinned SHA-256")
    _require(isinstance(source_payload_path, str) and source_payload_path,
             "source payload path is missing")
    _require(isinstance(appended_terms, list) and len(appended_terms) > 0,
             "new source terms are missing")
    _require(type(expected_existing_term_count) is int and expected_existing_term_count >= 0,
             "invalid expected child vertex count")

    _require(extension.get("child_payload_sha256") == expected_child_payload_sha256,
             "extension is not bound to the exact expected child payload")
    sources = extension.get("source_payloads")
    surfaces = extension.get("surfaces")
    _require(isinstance(sources, list) and source_index < len(sources)
             and isinstance(surfaces, list),
             "parent source/surface inventories are malformed")
    old_source = sources[source_index]
    _require(isinstance(old_source, dict)
             and old_source.get("sha256") == expected_source_payload_sha256,
             "parent source payload identity differs")
    matching = [row for row in surfaces
                if isinstance(row, dict) and row.get("stable_id") == stable_id]
    _require(len(matching) == 1, "child stable ID is absent or duplicated in parent extension")
    row_terms = matching[0].get("vertices")
    _require(isinstance(row_terms, list)
             and len(row_terms) == expected_existing_term_count,
             "child terms do not match the pinned original vertex count")

    normalized = []
    for vertex_index, terms in enumerate(appended_terms):
        _require(isinstance(terms, list) and len(terms) == 21,
                 f"new child vertex {vertex_index} does not retain all 21 parent terms")
        parsed = []
        for lane, term in enumerate(terms):
            _require(isinstance(term, list) and len(term) == 4,
                     f"new child vertex {vertex_index} term {lane} is malformed")
            src, parent_id, parent_vertex, coefficient = term
            _require(type(src) is int and src == source_index
                     and type(parent_id) is int and parent_id == stable_id
                     and type(parent_vertex) is int and parent_vertex >= 0
                     and type(coefficient) in (int, float)
                     and math.isfinite(float(coefficient)) and float(coefficient) > 0,
                     f"new child vertex {vertex_index} term {lane} has invalid identity/value")
            parsed.append([src, parent_id, parent_vertex, float(coefficient)])
        _require(all(parsed[lane][2] + 1 == parsed[lane + 1][2] for lane in range(20)),
                 f"new child vertex {vertex_index} parent lane order is not contiguous")
        _require(abs(sum(float(term[3]) for term in parsed) - 1.0) <= 1e-12,
                 f"new child vertex {vertex_index} coefficients are not normalized")
        normalized.append(parsed)

    result = copy.deepcopy(extension)
    result["child_payload_sha256"] = child_payload_sha256
    result["source_payloads"][source_index]["path"] = source_payload_path
    result["source_payloads"][source_index]["sha256"] = source_payload_sha256
    target = next(surface for surface in result["surfaces"]
                  if surface.get("stable_id") == stable_id)
    target["vertices"].extend(normalized)
    return result




def accepted_poses_from_receipt(receipt_path):
    """Load only accepted dynamic body poses from the pinned step receipt."""
    import hashlib
    import json
    from pathlib import Path

    path = Path(receipt_path).resolve()
    raw = path.read_bytes()
    receipt = json.loads(raw)
    _require(receipt.get("schema") == "numi.human.accepted-render-geometry.v1"
             and receipt.get("accepted_step") == 0,
             "expected an accepted step-zero render-geometry receipt")
    records = receipt.get("accepted_body_poses")
    _require(isinstance(records, list)
             and len(records) == receipt.get("accepted_body_pose_count"),
             "accepted dynamic body pose inventory is incomplete")
    poses = {}
    for record in records:
        _require(isinstance(record, dict)
                 and type(record.get("body_index")) is int
                 and record["body_index"] not in poses,
                 "accepted dynamic body pose identity is invalid or duplicated")
        position = np.asarray(record.get("position_m"), dtype="<f4")
        quaternion = np.asarray(record.get("quaternion_xyzw"), dtype="<f4")
        _require(position.shape == (3,) and quaternion.shape == (4,)
                 and np.isfinite(position).all() and np.isfinite(quaternion).all()
                 and abs(float(np.linalg.norm(quaternion.astype(np.float64))) - 1.0) <= 2e-3,
                 "accepted dynamic body pose values are malformed")
        poses[record["body_index"]] = np.concatenate((position, quaternion)).astype("<f4")
    _require(len(poses) == 157, "accepted dynamic body pose count differs from current ABI")
    return poses, {
        "accepted_receipt_path": str(path),
        "accepted_receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "accepted_step": 0,
        "accepted_body_pose_count": len(poses),
        "pose_owner": "accepted_body_poses from exact accepted step receipt; not NHRIGID static poses",
    }


def full21_parent_world_positions(row_data, poses, source_points_f32,
                                  core_weights_float64):
    """Replay the full21 load-only carrier map in native owner Float32 order."""
    from . import passive_attachment_composition as pc

    weights = np.asarray(core_weights_float64)
    points = np.asarray(source_points_f32)
    _require(weights.dtype == np.dtype("float64") and weights.ndim == 2
             and weights.shape[1] == 21 and len(weights) > 0
             and np.isfinite(weights).all() and np.all(weights > 0)
             and np.max(np.abs(weights.sum(axis=1) - 1.0)) <= 1e-12,
             "full21 source weights are not a positive normalized Float64 field")
    _require(points.dtype in (np.dtype("<f4"), np.dtype("float32"))
             and points.shape == (len(weights), 3) and np.isfinite(points).all(),
             "source points must be Float32 and align with full21 weights")
    _require(row_data["vertex_count"] >= 21 and row_data["vertex_count"] % 21 == 0,
             "parent row does not contain complete 21-lane groups")
    indices = np.asarray(row_data["binding_indices"][:21, 0], dtype=np.int64)
    lane_weights = np.asarray(row_data["weights"][:21, 0], dtype="<f4")
    _require(np.array_equal(indices, np.arange(21))
             and np.all(lane_weights == np.float32(1))
             and np.all(row_data["binding_indices"][:21, 1:] == 0)
             and np.all(row_data["weights"][:21, 1:] == 0),
             "parent row first source group is not canonical one-hot lane order")
    _require(len(row_data["bindings"]) == 21,
             "parent row binding inventory does not match full21")
    world_by_binding = pc._binding_world_positions(row_data, points, poses)
    result = np.zeros((len(points), 3), dtype="<f4")
    compiled_coefficients = weights.astype("<f4")
    for lane in range(21):
        _require(int(indices[lane]) < world_by_binding.shape[1],
                 "full21 lane references an absent registered binding")
        contribution = np.asarray(
            world_by_binding[:, indices[lane], :]
            * compiled_coefficients[:, lane, None],
            dtype="<f4",
        )
        result = np.asarray(result + contribution, dtype="<f4")
    return result


def solve_full21_parent_source_points(
    row_data, poses, target_world_f32, core_weights_float64, *,
    max_replay_residual_m: float = 2e-7,
) -> tuple[np.ndarray, dict]:
    """Invert an accepted-pose full21 field and verify its existing CPU owner replay.

    The affine solve is an authoring inverse. The candidate is forward-emitted
    through the existing parent binding owner and rounded to Float32; the
    returned report keeps the replay explicitly separate from native evidence.
    """
    from itertools import product
    from . import passive_attachment_composition as pc

    weights = np.asarray(core_weights_float64)
    target = np.asarray(target_world_f32)
    _require(target.dtype in (np.dtype("<f4"), np.dtype("float32"))
             and target.ndim == 2 and target.shape == (len(weights), 3)
             and len(target) > 0 and np.isfinite(target).all(),
             "target must be accepted-frame Float32 positions aligned with full21 weights")
    _require(math.isfinite(max_replay_residual_m) and max_replay_residual_m > 0,
             "invalid source replay residual bound")
    # Validate the material field and the carrier's 21 ordered lanes first.
    full21_parent_world_positions(
        row_data, poses, np.asarray(target[:1], dtype="<f4"),
        np.asarray(weights[:1], dtype=np.float64),
    )
    affine_by_lane, bias_by_lane = pc._child_rest_affine(row_data, poses)
    lane_affine = affine_by_lane[:21]
    lane_bias = bias_by_lane[:21]
    affine = np.sum(weights[:, :, None, None] * lane_affine[None, :, :, :], axis=1)
    bias = np.sum(weights[:, :, None] * lane_bias[None, :, :], axis=1)
    singular = np.linalg.svd(affine, compute_uv=False)
    _require(np.isfinite(singular).all() and np.all(singular[:, 0] > 0)
             and np.all(singular[:, -1] > 0),
             "full21 affine is singular")
    condition = singular[:, 0] / singular[:, -1]
    _require(np.isfinite(condition).all(), "full21 affine condition is nonfinite")
    solved = np.linalg.solve(
        affine, (target.astype(np.float64) - bias)[..., None],
    )[..., 0].astype("<f4")

    def replay(points):
        return full21_parent_world_positions(row_data, poses, points, weights)

    best = solved.copy()
    best_world = replay(best)
    best_error = np.max(
        np.abs(best_world.astype(np.float64) - target.astype(np.float64)), axis=1,
    )
    best_distance = np.zeros(len(best), dtype=np.float64)
    negative = np.float32(-np.inf)
    positive = np.float32(np.inf)
    # Check the immediate 3x3x3 Float32 neighborhood of the affine inverse.
    # This is a bounded representability search, not a physics or anatomy solver.
    for directions in product((-1, 0, 1), repeat=3):
        if directions == (0, 0, 0):
            continue
        candidate = solved.copy()
        for axis, direction in enumerate(directions):
            if direction:
                candidate[:, axis] = np.nextafter(
                    candidate[:, axis],
                    negative if direction < 0 else positive,
                    dtype=np.float32,
                )
        emitted = replay(candidate)
        error = np.max(
            np.abs(emitted.astype(np.float64) - target.astype(np.float64)), axis=1,
        )
        distance = np.sum(
            (candidate.astype(np.float64) - solved.astype(np.float64)) ** 2, axis=1,
        )
        improve = (error < best_error) | ((error == best_error) & (distance < best_distance))
        best[improve] = candidate[improve]
        best_world[improve] = emitted[improve]
        best_error[improve] = error[improve]
        best_distance[improve] = distance[improve]

    worst = float(best_error.max(initial=0.0))
    _require(worst <= max_replay_residual_m,
             f"full21 Float32 source replay residual {worst:.9g} m exceeds bound")
    return best, {
        "point_count": len(best),
        "ordered_parent_lanes": 21,
        "max_affine_condition_number": float(condition.max(initial=0.0)),
        "max_affine_inverse_replay_residual_m": float(
            np.max(np.abs(replay(solved).astype(np.float64) - target.astype(np.float64)))
        ),
        "max_float32_neighborhood_replay_residual_m": worst,
        "float32_neighbor_offsets_per_axis": 1,
        "exact_replay_output_matches_target": best_world.tobytes() == target.tobytes(),
        "source_owner": "passive_attachment_composition._child_rest_affine and _binding_world_positions",
        "evidence_boundary": "CPU current-accepted-pose source inverse/replay only; native/anatomical admission remains separate",
    }




def remap_full21_terms_by_source_slot(
    *, source_index: int, stable_id: int, source_slots_i64: np.ndarray,
    core_weights_float64: np.ndarray,
    original_terms: list[list[list[float | int]]],
    first_new_parent_vertex: int,
) -> tuple[list[list[list[float | int]]], list[int], dict]:
    """Reorder original exact terms by authenticated source slot and emit births.

    Nonnegative slots reuse their exact existing 21-term records after checking
    the candidate field against them. A -1 slot receives one newly appended
    21-lane carrier group; no GID-only or pack-global index substitution occurs.
    """
    slots = np.asarray(source_slots_i64)
    weights = np.asarray(core_weights_float64)
    _require(slots.dtype == np.dtype("int64") and slots.ndim == 1
             and len(slots) > 0 and np.all(slots >= -1),
             "source-slot crosswalk must be a nonnegative-or-minus-one Int64 vector")
    _require(weights.dtype == np.dtype("float64")
             and weights.shape == (len(slots), 21)
             and np.isfinite(weights).all() and np.all(weights > 0)
             and np.all(weights <= 1)
             and np.max(np.abs(weights.sum(axis=1) - 1.0)) <= 1e-12,
             "candidate full21 weights are not a positive normalized Float64 field")
    _require(type(source_index) is int and source_index >= 0
             and type(stable_id) is int and stable_id > 0,
             "invalid source or stable-row identity")
    _require(type(first_new_parent_vertex) is int
             and first_new_parent_vertex == len(original_terms) * 21,
             "new carrier does not start after the complete original 21-lane table")
    for slot, terms in enumerate(original_terms):
        _require(isinstance(terms, list) and len(terms) == 21,
                 f"original source slot {slot} lacks its full21 parent terms")
        coefficients = []
        for lane, term in enumerate(terms):
            _require(isinstance(term, list) and len(term) == 4
                     and type(term[0]) is int and term[0] == source_index
                     and type(term[1]) is int and term[1] == stable_id
                     and type(term[2]) is int and term[2] == slot * 21 + lane
                     and type(term[3]) in (int, float)
                     and math.isfinite(float(term[3])) and float(term[3]) > 0,
                     f"original source slot {slot} term {lane} has different ownership")
            coefficients.append(float(term[3]))
        _require(abs(sum(coefficients) - 1.0) <= 1e-12,
                 f"original source slot {slot} terms are not normalized")

    births = [int(index) for index in np.flatnonzero(slots == -1)]
    birth_terms = full21_parent_terms(
        source_index=source_index, stable_id=stable_id,
        first_parent_vertex=first_new_parent_vertex,
        core_weights_float64=weights[np.asarray(births, dtype=np.int64)],
    ) if births else []
    birth_cursor = 0
    result = []
    for candidate_vertex, raw_slot in enumerate(slots):
        slot = int(raw_slot)
        if slot >= 0:
            _require(slot < len(original_terms),
                     f"candidate vertex {candidate_vertex} source slot is out of range")
            old_weights = np.asarray(
                [float(term[3]) for term in original_terms[slot]], dtype=np.float64,
            )
            _require(old_weights.tobytes() == weights[candidate_vertex].tobytes(),
                     f"candidate vertex {candidate_vertex} changed its source full21 field")
            result.append(copy.deepcopy(original_terms[slot]))
        else:
            result.append(birth_terms[birth_cursor])
            birth_cursor += 1
    return result, births, {
        "candidate_vertex_count": len(slots),
        "reused_original_source_slot_count": int(np.count_nonzero(slots >= 0)),
        "new_source_birth_count": len(births),
        "new_parent_lane_count": 21 * len(births),
        "all_original_full21_coefficients_byte_exact": True,
        "all_new_lanes_ordered_and_positive": True,
        "source_slot_crosswalk_used": True,
    }


def replace_parent_surface_terms(
    extension: dict, *, stable_id: int, source_index: int,
    expected_child_payload_sha256: str, child_payload_sha256: str,
    expected_source_payload_sha256: str, source_payload_path: str,
    source_payload_sha256: str, expected_existing_term_count: int,
    replacement_terms: list[list[list[float | int]]],
) -> dict:
    """Rebind one exact child row to a complete remapped full21 term list."""
    _require(type(stable_id) is int and stable_id > 0
             and type(source_index) is int and source_index >= 0,
             "invalid child/source identity")
    for value in (expected_child_payload_sha256, child_payload_sha256,
                  expected_source_payload_sha256, source_payload_sha256):
        _require(isinstance(value, str) and len(value) == 64
                 and all(ch in "0123456789abcdef" for ch in value),
                 "invalid pinned SHA-256")
    _require(isinstance(source_payload_path, str) and source_payload_path
             and type(expected_existing_term_count) is int
             and expected_existing_term_count >= 0,
             "invalid source path or existing row count")
    _require(extension.get("child_payload_sha256") == expected_child_payload_sha256,
             "extension is not bound to the exact expected child payload")
    sources, surfaces = extension.get("source_payloads"), extension.get("surfaces")
    _require(isinstance(sources, list) and source_index < len(sources)
             and isinstance(surfaces, list), "parent extension inventories are malformed")
    _require(isinstance(sources[source_index], dict)
             and sources[source_index].get("sha256") == expected_source_payload_sha256,
             "source carrier identity differs from the pinned input")
    rows = [row for row in surfaces
            if isinstance(row, dict) and row.get("stable_id") == stable_id]
    _require(len(rows) == 1 and isinstance(rows[0].get("vertices"), list)
             and len(rows[0]["vertices"]) == expected_existing_term_count,
             "target child row or original term count differs")
    _require(isinstance(replacement_terms, list) and replacement_terms,
             "complete replacement term table is empty")
    for vertex, terms in enumerate(replacement_terms):
        _require(isinstance(terms, list) and len(terms) == 21,
                 f"replacement vertex {vertex} does not retain all 21 terms")
        _require(all(
            isinstance(term, list) and len(term) == 4
            and type(term[0]) is int and term[0] == source_index
            and type(term[1]) is int and term[1] == stable_id
            and type(term[2]) is int and term[2] >= 0
            and type(term[3]) in (int, float) and math.isfinite(float(term[3]))
            and float(term[3]) > 0
            for term in terms
        ), f"replacement vertex {vertex} has malformed parent term identity")
        _require(all(terms[lane][2] + 1 == terms[lane + 1][2] for lane in range(20))
                 and abs(sum(float(term[3]) for term in terms) - 1.0) <= 1e-12,
                 f"replacement vertex {vertex} has invalid parent lane order/partition")
    result = copy.deepcopy(extension)
    result["child_payload_sha256"] = child_payload_sha256
    result["source_payloads"][source_index]["path"] = source_payload_path
    result["source_payloads"][source_index]["sha256"] = source_payload_sha256
    target = next(row for row in result["surfaces"] if row["stable_id"] == stable_id)
    target["vertices"] = copy.deepcopy(replacement_terms)
    return result



def _binding_transform_identity(binding):
    core = binding.get("core")
    _require(type(core) is int and core >= 0, "route binding has invalid typed Core identity")
    translation = np.asarray(binding.get("translation"), dtype="<f4")
    quaternion = np.asarray(binding.get("quaternion"), dtype="<f4")
    scale = np.asarray([binding.get("scale")], dtype="<f4")
    _require(translation.shape == (3,) and quaternion.shape == (4,)
             and scale.shape == (1,)
             and np.isfinite(translation).all() and np.isfinite(quaternion).all()
             and np.isfinite(scale).all() and scale[0] > 0,
             "route binding transform is malformed")
    return core, translation.tobytes(), quaternion.tobytes(), scale.tobytes()


def remap_child_four_lane_route_by_binding_identity(
    source_bindings, target_bindings, binding_indices_u32, weights_f32,
):
    """Map an authenticated legacy route by exact Core and registered transform identity."""
    indices = np.asarray(binding_indices_u32)
    weights = np.asarray(weights_f32)
    _require(indices.shape == (4,) and np.issubdtype(indices.dtype, np.integer),
             "source auxiliary route must contain exactly four binding indices")
    _require(weights.dtype in (np.dtype("<f4"), np.dtype("float32"))
             and weights.shape == (4,) and np.isfinite(weights).all()
             and np.all(weights >= 0) and np.count_nonzero(weights > 0) > 0
             and abs(float(weights.astype(np.float64).sum()) - 1.0) <= 2e-7,
             "source auxiliary route must be a positive normalized Float32 four-lane field")
    source_weight_bytes = weights.tobytes()
    source = list(source_bindings)
    target = list(target_bindings)
    _require(source and target, "source or target child binding inventory is empty")
    identities = [_binding_transform_identity(value) for value in target]
    mapped = np.zeros(4, dtype="<u4")
    lane_report = []
    for lane in range(4):
        if weights[lane] == 0:
            lane_report.append({"lane": lane, "active": False, "target_binding_index": 0})
            continue
        source_index = int(indices[lane])
        _require(0 <= source_index < len(source),
                 f"active source route lane {lane} references an absent binding")
        identity = _binding_transform_identity(source[source_index])
        if source_index < len(target) and identities[source_index] == identity:
            target_index = source_index
        else:
            matches = [i for i, value in enumerate(identities) if value == identity]
            _require(len(matches) == 1,
                     f"active source route lane {lane} has {len(matches)} exact typed transform matches")
            target_index = matches[0]
        mapped[lane] = target_index
        lane_report.append({
            "lane": lane, "active": True, "source_binding_index": source_index,
            "target_binding_index": int(target_index), "typed_core": identity[0],
            "registered_transform_identity_exact": True,
        })
    return mapped, weights.astype("<f4", copy=True), {
        "active_route_binding_identity_remap": lane_report,
        "source_route_weights_bytes": source_weight_bytes.hex(),
        "active_weights_byte_exact": weights.astype("<f4").tobytes() == source_weight_bytes,
        "inactive_lane_weights_byte_exact": True,
        "route_support_truncated": False,
    }


def emit_child_with_inferred_four_lane_route(
    row_data, poses, binding_indices_u32, weights_f32, target_world_f32,
    initial_source_point_f32, normal_f32, *, max_replay_residual_m=2e-6,
):
    """Inverse-solve a child source point while retaining one ancestor's exact 4-lane route."""
    from . import passive_attachment_composition as pc

    indices = np.asarray(binding_indices_u32, dtype="<u4")
    weights = np.asarray(weights_f32, dtype="<f4")
    target = np.asarray(target_world_f32, dtype="<f4")
    initial = np.asarray(initial_source_point_f32, dtype="<f4")
    normal = np.asarray(normal_f32, dtype="<f4")
    _require(indices.shape == (4,) and weights.shape == (4,)
             and np.isfinite(weights).all() and np.all(weights >= 0)
             and np.count_nonzero(weights > 0) > 0
             and abs(float(weights.astype(np.float64).sum()) - 1.0) <= 2e-7,
             "inferred child route is not an exact normalized four-lane field")
    _require(target.shape == (3,) and initial.shape == (3,)
             and normal.shape == (3,) and np.isfinite(target).all()
             and np.isfinite(initial).all() and np.isfinite(normal).all(),
             "inferred child rest join has malformed point or normal")
    _require(np.all(indices[weights > 0] < len(row_data["bindings"])),
             "inferred child route references an absent target binding")
    normal_length = float(np.linalg.norm(normal.astype(np.float64)))
    _require(math.isfinite(normal_length) and normal_length > 1e-12,
             "inferred child normal is degenerate")
    normal = np.asarray(normal.astype(np.float64) / normal_length, dtype="<f4")

    temp = dict(row_data)
    temp["binding_indices"] = indices[None, :].copy()
    temp["weights"] = weights[None, :].copy()
    temp["vertex_count"] = 1
    point, condition = pc._solve_sparse_rest_positions(
        temp, target[None, :], initial[None, :], poses,
        vertex_indices=np.asarray([0], dtype=np.int64),
    )
    replay = pc._native_blended_rest_positions(temp, point, poses)
    residual = float(np.max(np.abs(replay.astype(np.float64) - target.astype(np.float64))))
    _require(math.isfinite(residual) and residual <= max_replay_residual_m,
             "inferred legacy route fails accepted-pose rest join")
    return np.concatenate((point[0], normal)).astype("<f4"), indices.copy(), weights.copy(), {
        "max_four_lane_replay_residual_m": residual,
        "max_affine_condition_number": float(np.max(condition)),
        "actual_target_world_f32": target.astype(float).tolist(),
        "four_lane_reemitted_world_f32": replay[0].astype(float).tolist(),
        "route_weights_byte_exact": True,
        "route_support_truncated": False,
        "child_source_point_is_inferred": True,
    }


def derive_full21_birth_source_point(
    row_data, poses, endpoint_source_points_f32, endpoint_weights_float64,
    edge_fraction: float, *, max_replay_residual_m: float = 2e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Pull an inferred edge birth through the actual accepted-pose full21 owner.

    Both endpoint positions are replayed from their pinned source points and
    complete ordered 21-parent terms. The accepted-frame target is the
    Float32 edge interpolation of those owner-emitted endpoints. The new full21
    coefficients are interpolated in Float64 and the source point is obtained
    by the existing bounded owner inverse/re-emission helper. Static candidate
    world coordinates are never an input.
    """
    edge = float(edge_fraction)
    _require(math.isfinite(edge) and 0.0 <= edge <= 1.0,
             "birth edge fraction is outside [0, 1]")
    points = np.asarray(endpoint_source_points_f32)
    weights = np.asarray(endpoint_weights_float64)
    _require(points.dtype in (np.dtype("<f4"), np.dtype("float32"))
             and points.shape == (2, 3) and np.isfinite(points).all(),
             "birth endpoints must be two finite Float32 source points")
    _require(weights.dtype == np.dtype("float64")
             and weights.shape == (2, 21) and np.isfinite(weights).all()
             and np.all(weights > 0) and np.all(weights <= 1)
             and np.max(np.abs(weights.sum(axis=1) - 1.0)) <= 1e-12,
             "birth endpoints must carry complete positive normalized full21 fields")
    endpoint_world = full21_parent_world_positions(row_data, poses, points, weights)
    fraction64 = np.float64(edge)
    target_world = np.asarray(
        endpoint_world[0].astype(np.float64) * (1.0 - fraction64)
        + endpoint_world[1].astype(np.float64) * fraction64,
        dtype="<f4",
    )[None, :]
    birth_weights = np.asarray(
        weights[0] * (1.0 - fraction64) + weights[1] * fraction64,
        dtype=np.float64,
    )[None, :]
    # The two normalized endpoint rows make this normalization an identity
    # apart from Float64 accumulation; keep the original positive 21 lanes.
    total = float(birth_weights.sum())
    _require(math.isfinite(total) and total > 0,
             "interpolated full21 birth field has invalid total")
    birth_weights = np.asarray(birth_weights / total, dtype=np.float64)
    source_point, inverse_report = solve_full21_parent_source_points(
        row_data, poses, target_world, birth_weights,
        max_replay_residual_m=max_replay_residual_m,
    )
    replay = full21_parent_world_positions(
        row_data, poses, source_point, birth_weights,
    )
    edge_source_point = np.asarray(
        points[0].astype(np.float64) * (1.0 - fraction64)
        + points[1].astype(np.float64) * fraction64,
        dtype="<f4",
    )
    residual = float(np.max(np.abs(
        replay.astype(np.float64) - target_world.astype(np.float64),
    )))
    _require(residual <= max_replay_residual_m,
             "inferred birth fails actual accepted-pose full21 forward replay")
    return source_point[0], birth_weights[0], target_world[0], {
        "edge_fraction_from_lower_canonical_gid": edge,
        "endpoint_world_positions_f32": endpoint_world.astype(float).tolist(),
        "target_world_position_f32": target_world[0].astype(float).tolist(),
        "edge_interpolated_source_point_f32": edge_source_point.astype(float).tolist(),
        "full21_parent_source_point_f32": source_point[0].astype(float).tolist(),
        "source_point_pullback_delta_m": float(np.linalg.norm(
            source_point[0].astype(np.float64) - edge_source_point.astype(np.float64),
        )),
        "full21_weights_float64": birth_weights[0].tolist(),
        "full21_weight_lane_count": 21,
        "full21_weight_min": float(birth_weights.min()),
        "full21_weight_sum": float(birth_weights.sum()),
        "actual_accepted_pose_full21_replay_f32": replay[0].astype(float).tolist(),
        "max_full21_forward_replay_residual_m": residual,
        "inverse": inverse_report,
        "source_owner": "accepted-pose full21 carrier replay and solve_full21_parent_source_points",
        "static_candidate_positions_world_used": False,
    }


def derive_child_four_lane_birth(
    row_data, poses, endpoint_source_slots, target_world_f32, edge_fraction: float,
    *, max_replay_residual_m: float = 2e-7,
):
    """Emit one child ABI5 vertex from an actual accepted-frame full21 target.

    The four legacy child lanes are formed only by exact endpoint support
    interpolation. If that union exceeds four, callers must change construction;
    this helper never truncates a material route. It then inverts and forward
    replays the current child binding owner at the supplied actual target.
    """
    from . import passive_attachment_composition as pc

    edge = float(edge_fraction)
    _require(math.isfinite(edge) and 0.0 <= edge <= 1.0,
             "child birth edge fraction is outside [0, 1]")
    slots = np.asarray(endpoint_source_slots)
    _require(slots.shape == (2,) and np.issubdtype(slots.dtype, np.integer)
             and np.all(slots >= 0) and np.all(slots < row_data["vertex_count"]),
             "child birth endpoints are not valid original source slots")
    target = np.asarray(target_world_f32)
    _require(target.dtype in (np.dtype("<f4"), np.dtype("float32"))
             and target.shape == (3,) and np.isfinite(target).all(),
             "child birth target must be an actual-frame Float32 point")
    _require(math.isfinite(max_replay_residual_m) and max_replay_residual_m > 0,
             "child birth replay tolerance is invalid")

    acc = {}
    for endpoint, scale in zip(map(int, slots), (1.0 - edge, edge)):
        for lane in range(4):
            weight = float(row_data["weights"][endpoint, lane]) * scale
            if weight > 0.0:
                binding = int(row_data["binding_indices"][endpoint, lane])
                _require(binding < len(row_data["bindings"]),
                         "child endpoint uses an absent local binding")
                acc[binding] = acc.get(binding, 0.0) + weight
    support = sorted((binding, weight) for binding, weight in acc.items()
                     if weight > 0.0)
    _require(0 < len(support) <= 4,
             f"child birth requires {len(support)} positive route lanes; refusing truncation")
    binding_ids = np.zeros(4, dtype="<u4")
    lane_weights = np.zeros(4, dtype="<f4")
    raw = np.asarray([weight for _, weight in support], dtype=np.float64)
    raw /= float(raw.sum())
    for lane, (binding, _weight) in enumerate(support):
        binding_ids[lane] = binding
        lane_weights[lane] = np.float32(raw[lane])
    # Correct only Float32 partition rounding in the largest existing support.
    largest = int(np.argmax(lane_weights[:len(support)]))
    correction = np.float32(1.0 - float(lane_weights.astype(np.float64).sum()))
    lane_weights[largest] = np.float32(lane_weights[largest] + correction)
    _require(np.all(lane_weights[:len(support)] > 0)
             and np.all(lane_weights[len(support):] == 0)
             and abs(float(lane_weights.astype(np.float64).sum()) - 1.0) <= 2e-7,
             "child birth route partition lost positive lane support")

    temp = dict(row_data)
    temp["binding_indices"] = binding_ids[None, :]
    temp["weights"] = lane_weights[None, :]
    temp["vertex_count"] = 1
    endpoint = np.asarray(row_data["positions"][slots], dtype="<f4")
    initial = np.asarray(
        endpoint[0].astype(np.float64) * (1.0 - edge)
        + endpoint[1].astype(np.float64) * edge, dtype="<f4",
    )[None, :]
    point, condition = pc._solve_sparse_rest_positions(
        temp, target[None, :], initial, poses, vertex_indices=np.asarray([0]),
    )
    replay = pc._native_blended_rest_positions(temp, point, poses)
    residual = float(np.max(np.abs(
        replay.astype(np.float64) - target.astype(np.float64),
    )))
    _require(math.isfinite(residual) and residual <= max_replay_residual_m,
             "child birth fails actual accepted-pose four-lane forward replay")

    normal = np.asarray(
        row_data["normals"][slots[0]].astype(np.float64) * (1.0 - edge)
        + row_data["normals"][slots[1]].astype(np.float64) * edge,
        dtype="<f4",
    )
    norm = float(np.linalg.norm(normal.astype(np.float64)))
    _require(math.isfinite(norm) and norm > 1e-12,
             "child birth endpoint normals cancel")
    normal = np.asarray(normal.astype(np.float64) / norm, dtype="<f4")
    return (
        np.concatenate((point[0], normal)).astype("<f4"),
        binding_ids,
        lane_weights,
        {
            "endpoint_source_slots": slots.astype(int).tolist(),
            "edge_fraction_from_lower_canonical_gid": edge,
            "positive_route_binding_indices": [int(x) for x, _ in support],
            "positive_route_binding_count": len(support),
            "four_lane_support_truncated": False,
            "actual_target_world_f32": target.astype(float).tolist(),
            "four_lane_reemitted_world_f32": replay[0].astype(float).tolist(),
            "max_four_lane_replay_residual_m": residual,
            "max_affine_condition_number": float(np.max(condition)),
            "static_candidate_positions_world_used": False,
        },
    )
