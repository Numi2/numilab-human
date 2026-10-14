"""Source-bound dual-body reference carriers for connected axillary vessels.

The inferred source-mesh field uses the existing passive parent ABI. It is not
measured physiology, anatomy admission, or native-pose qualification.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import struct
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

N27_RECEIPT_SHA256 = "944bc88d257137f2bedd5898404c45cefd2901d719707519771ae5f4cfce3752"
N27_PAYLOAD_SHA256 = "cde8705a813fe56686776c9cc038cbf06f413e1de76099d2a019d5e1b748a8d4"
PORT_REPORT_SHA256 = "e638c6dfeb1030ab6aca113f8d658acf02820ffec883d7e19ebf7f9a8aa7c249"
RIGID_SHA256 = "2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4"
CARRIER_477_SHA256 = "4c9a443d8a77101847e8513a7ef102c322b9af5e317298bedb87c75b85e80cd9"
ISA_ARCHIVE_SHA256 = "40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e"
FJ2217_OBJ_SHA256 = "7b448d1f2a071e88d1a2ebe54e7e5d417719663a5b01d1226f492b71ebdc19cd"


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("vessel owner blend: " + message)


def _mesh_graph(vertices: np.ndarray, faces: np.ndarray):
    xyz = np.asarray(vertices, dtype="<f4")
    tri = np.asarray(faces, dtype=np.int64)
    _require(
        xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all(),
        "invalid source coordinates",
    )
    _require(
        tri.ndim == 2
        and tri.shape[1] == 3
        and len(tri)
        and tri.min() >= 0
        and tri.max() < len(xyz),
        "invalid source faces",
    )
    graph: list[dict[int, float]] = [{} for _ in range(len(xyz))]
    for a, b, c in tri:
        for i, j in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            if j not in graph[i]:
                length = float(
                    np.linalg.norm(
                        xyz[i].astype(np.float64) - xyz[j].astype(np.float64)
                    )
                )
                _require(
                    math.isfinite(length) and length > 0, "invalid or zero source edge"
                )
                graph[i][j] = graph[j][i] = length
    return graph, tri


def _mesh_components(graph):
    unseen = set(range(len(graph)))
    components = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        stack = [root]
        component = {root}
        while stack:
            i = stack.pop()
            for j in graph[i]:
                if j in unseen:
                    unseen.remove(j)
                    component.add(j)
                    stack.append(j)
        components.append(component)
    return components


def _geodesic_distance(graph, seeds: set[int], component: set[int]) -> np.ndarray:
    _require(
        seeds and seeds.issubset(component),
        "collar seed leaves its connected source component",
    )
    dist = np.full(len(graph), np.inf, dtype=np.float64)
    queue = []
    for sid in sorted(seeds):
        dist[sid] = 0.0
        heapq.heappush(queue, (0.0, sid))
    while queue:
        value, i = heapq.heappop(queue)
        if value != dist[i]:
            continue
        for j, length in graph[i].items():
            candidate = value + length
            if candidate < dist[j]:
                dist[j] = candidate
                heapq.heappush(queue, (candidate, j))
    _require(
        np.isfinite(dist[np.asarray(sorted(component), dtype=np.int64)]).all(),
        "collar seeds do not reach the connected source component",
    )
    return dist


def _exact_coordinate_quotient_graph(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[list[dict[int, float]], np.ndarray, np.ndarray, list[list[int]]]:
    """Build the existing edge graph after exact F32 XYZ alias quotienting."""
    xyz = np.ascontiguousarray(vertices, dtype="<f4")
    source_tri = np.ascontiguousarray(faces, dtype=np.int64)
    _require(
        xyz.ndim == 2 and xyz.shape[1] == 3 and np.isfinite(xyz).all(),
        "invalid source coordinates",
    )
    _require(
        source_tri.ndim == 2
        and source_tri.shape[1] == 3
        and len(source_tri)
        and source_tri.min() >= 0
        and source_tri.max() < len(xyz),
        "invalid source faces",
    )
    key_to_class: dict[bytes, int] = {}
    class_ids = np.empty(len(xyz), dtype=np.int64)
    classes: list[list[int]] = []
    class_xyz: list[np.ndarray] = []
    for local_id, point in enumerate(xyz):
        key = point.tobytes()
        class_id = key_to_class.get(key)
        if class_id is None:
            class_id = len(classes)
            key_to_class[key] = class_id
            classes.append([])
            class_xyz.append(point.copy())
        classes[class_id].append(local_id)
        class_ids[local_id] = class_id
    quotient_tri = class_ids[source_tri]
    _require(
        np.all(
            (quotient_tri[:, 0] != quotient_tri[:, 1])
            & (quotient_tri[:, 1] != quotient_tri[:, 2])
            & (quotient_tri[:, 2] != quotient_tri[:, 0])
        ),
        "exact coordinate quotient collapses a source face",
    )
    graph, tri = _mesh_graph(
        np.asarray(class_xyz, dtype="<f4"),
        quotient_tri,
    )
    return graph, tri, class_ids, classes


def geodesic_owner_partition(
    vertices: np.ndarray,
    faces: np.ndarray,
    proximal_vertices: set[int],
    distal_vertices: set[int],
    *,
    isolated_distal_remainder: set[int] | None = None,
    exact_coordinate_quotient: bool = False,
):
    """Return exact-collar [Body20 proximal, Body91 distal] F32 weights.

    One explicit collar-free one-face remainder may retain its prior distal
    Body91 parent. The caller must validate its exact old owner terms first.
    """
    prox, distal = set(map(int, proximal_vertices)), set(map(int, distal_vertices))
    _require(
        prox and distal and prox.isdisjoint(distal),
        "collar sets are empty or overlapping",
    )
    source_tri = np.ascontiguousarray(faces, dtype=np.int64)
    if exact_coordinate_quotient:
        graph, _tri, class_ids, classes = _exact_coordinate_quotient_graph(
            vertices, source_tri
        )
        prox_graph = {int(class_ids[v]) for v in prox}
        distal_graph = {int(class_ids[v]) for v in distal}
        _require(
            prox_graph.isdisjoint(distal_graph),
            "exact coordinate alias merges proximal and distal collar constraints",
        )
    else:
        graph, _tri = _mesh_graph(vertices, source_tri)
        class_ids = np.arange(len(vertices), dtype=np.int64)
        classes = [[i] for i in range(len(vertices))]
        prox_graph, distal_graph = prox, distal
    components = _mesh_components(graph)
    seed_components = [
        c for c in components if c.intersection(prox_graph | distal_graph)
    ]
    _require(
        len(seed_components) == 1
        and (prox_graph | distal_graph).issubset(seed_components[0]),
        "proximal/distal collars do not share one connected source component",
    )
    connected = seed_components[0]
    remainder_components = [c for c in components if c is not connected]
    remainder = set().union(*remainder_components) if remainder_components else set()
    allowed_source = set(map(int, isolated_distal_remainder or ()))
    _require(
        allowed_source.issubset(set(range(len(class_ids)))),
        "declared distal remainder leaves source vertex slots",
    )
    if allowed_source:
        _require(
            len(allowed_source) == 3
            and sum(
                all(int(v) in allowed_source for v in face) for face in source_tri
            )
            == 1,
            "only the declared isolated one-face remainder may retain distal ownership",
        )
    allowed = {int(class_ids[v]) for v in allowed_source}
    quotient_joined_remainder = bool(
        exact_coordinate_quotient and allowed and allowed.issubset(connected)
    )
    expected_remainder = set() if quotient_joined_remainder else allowed
    _require(
        remainder == expected_remainder,
        "unmapped disconnected source component is not explicitly retained",
    )
    if allowed and not quotient_joined_remainder:
        _require(
            len(remainder_components) == 1,
            "only the declared isolated one-face remainder may retain distal ownership",
        )
    dp = _geodesic_distance(graph, prox_graph, connected)
    dd = _geodesic_distance(graph, distal_graph, connected)
    ids = np.asarray(sorted(connected), dtype=np.int64)
    denom = dp + dd
    _require(
        np.isfinite(denom[ids]).all() and np.all(denom[ids] > 0),
        "invalid source route extent",
    )
    t = np.clip(dp[ids] / denom[ids], 0, 1)
    q = t * t * t * (t * (t * 6 - 15) + 10)
    interior_graph = np.asarray(
        sorted(connected - prox_graph - distal_graph), dtype=np.int64
    )
    saturation_corrections = []
    if exact_coordinate_quotient:
        q_f32 = q.astype("<f4")
        lower = np.nextafter(np.float32(0.0), np.float32(1.0), dtype=np.float32)
        upper = np.nextafter(np.float32(1.0), np.float32(0.0), dtype=np.float32)
        for graph_vertex in interior_graph:
            position = int(np.searchsorted(ids, graph_vertex))
            original = np.float32(q_f32[position])
            corrected = lower if original == 0.0 else upper if original == 1.0 else original
            if corrected != original:
                saturation_corrections.append(
                    {
                        "quotient_vertex_id": int(graph_vertex),
                        "original_rounded_distal_f32": float(original),
                        "corrected_distal_f32": float(corrected),
                        "correction_abs": float(abs(float(corrected) - float(q[position]))),
                    }
                )
                q_f32[position] = corrected
        distal_weight = np.ones(len(graph), dtype="<f4")
        distal_weight[ids] = q_f32
        distal_weight[np.asarray(sorted(prox_graph), dtype=np.int64)] = np.float32(0.0)
        distal_weight[np.asarray(sorted(distal_graph), dtype=np.int64)] = np.float32(1.0)
        one = np.ones(len(graph), dtype="<f4")
        weights_graph = np.column_stack((one - distal_weight, distal_weight)).astype("<f4")
    else:
        distal_weight = np.ones(len(graph), dtype=np.float64)
        distal_weight[ids] = q
        distal_weight[np.asarray(sorted(prox_graph), dtype=np.int64)] = 0.0
        distal_weight[np.asarray(sorted(distal_graph), dtype=np.int64)] = 1.0
        weights_graph = np.column_stack((1 - distal_weight, distal_weight)).astype("<f4")
    weights = weights_graph[class_ids]
    _require(
        np.isfinite(weights).all()
        and np.all(weights >= 0)
        and np.max(np.abs(weights.astype(np.float64).sum(axis=1) - 1)) <= 1e-7,
        "invalid owner partition",
    )
    pids = np.asarray(sorted(prox), dtype=np.int64)
    dids = np.asarray(sorted(distal), dtype=np.int64)
    _require(
        np.all(weights[pids] == [1, 0]) and np.all(weights[dids] == [0, 1]),
        "collar weights changed",
    )
    _require(
        np.all(weights_graph[interior_graph] > 0),
        "interior owner coefficient underflows F32 after minimal saturation correction",
    )
    return weights, {
        "method": "quintic smoothstep of source-mesh edge-geodesic distance ratio",
        "proximal_owner_core_body": 20,
        "distal_owner_core_body": 91,
        "proximal_vertex_count": len(prox),
        "distal_vertex_count": len(distal),
        "source_geodesic_extent_m": float(denom[ids].max()),
        "minimum_positive_float32_coefficient": float(weights[weights > 0].min()),
        "interior_vertex_count": len(interior_graph),
        "interior_float32_saturation_correction_count": len(saturation_corrections),
        "interior_float32_saturation_corrections": saturation_corrections,
        "graph_method": (
            "exact Float32 XYZ coordinate quotient before edge-geodesic solve"
            if exact_coordinate_quotient
            else "indexed source-vertex edge graph"
        ),
        "exact_coordinate_quotient_class_count": len(classes) if exact_coordinate_quotient else 0,
        "exact_coordinate_alias_class_count": (
            sum(len(group) > 1 for group in classes)
            if exact_coordinate_quotient
            else 0
        ),
        "exact_coordinate_alias_classes": (
            [group for group in classes if len(group) > 1]
            if exact_coordinate_quotient
            else []
        ),
        "all_exact_coordinate_alias_members_share_f32_owner_weights": (
            all(
                np.all(weights[group] == weights[group[0]])
                for group in classes
                if len(group) > 1
            )
            if exact_coordinate_quotient
            else False
        ),
        "declared_distal_remainder_joined_by_exact_coordinate_quotient": quotient_joined_remainder,
        "isolated_distal_remainder_vertex_count": (
            0 if quotient_joined_remainder else len(allowed_source)
        ),
        "isolated_distal_remainder_face_count": (
            0 if quotient_joined_remainder else (1 if allowed_source else 0)
        ),
        "isolated_distal_remainder_preserves_existing_body91_owner": bool(
            allowed_source and not quotient_joined_remainder
        ),
        "weights_are_inferred_not_measured": True,
    }


def _transfer_row(vertices6: np.ndarray, from_pose: np.ndarray, to_pose: np.ndarray):
    from .resting_scene import _rotation_xyzw

    v = np.asarray(vertices6, dtype="<f4")
    rf, rt = _rotation_xyzw(from_pose[3:]), _rotation_xyzw(to_pose[3:])
    world = v[:, :3].astype(np.float64) @ rf.T + from_pose[:3]
    world_n = v[:, 3:].astype(np.float64) @ rf.T
    local = (world - to_pose[:3]) @ rt
    normal = world_n @ rt
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    out = np.column_stack((local, normal)).astype("<f4")
    pe = float(
        np.max(np.abs(out[:, :3].astype(np.float64) @ rt.T + to_pose[:3] - world))
    )
    ne = float(np.max(np.abs(out[:, 3:].astype(np.float64) @ rt.T - world_n)))
    _require(
        np.isfinite(out).all() and pe <= 2e-6 and ne <= 2e-6,
        "Body20 rest carrier differs from Body91 points",
    )
    return out, pe, ne


def _collar(report: dict, parent: int, child: int):
    found = [
        r
        for r in report.get("rows", [])
        if int(r.get("parent_stable_id", -1)) == parent
        and int(r.get("child_stable_id", -1)) == child
    ]
    _require(len(found) == 1, f"missing frozen collar {parent}->{child}")
    p = set(map(int, found[0].get("unique_parent_intersection_vertex_local_ids", [])))
    c = set(map(int, found[0].get("unique_child_intersection_vertex_local_ids", [])))
    _require(p and c, f"collar {parent}->{child} has no exact vertex ancestry")
    return p, c


def _terms(weights: np.ndarray, source_index: int, surface_ids=(1, 2)):
    _require(len(surface_ids) == weights.shape[1], "owner surface IDs differ")
    return [
        [
            [source_index, surface_id, i, float(value)]
            for surface_id, value in zip(surface_ids, row)
            if value > 0
        ]
        for i, row in enumerate(weights)
    ]


def _verify_encoded_vessel_owner_maps(
    carrier_bytes: bytes,
    carrier_surfaces: list[dict[str, Any]],
    owner_updates: dict[int, list],
    expected_rows: dict[int, dict[str, Any]],
    aliases: list[list[int]],
) -> dict[str, Any]:
    """Verify emitted ABI5 carrier points and parent terms for rows 475/477."""
    from .passive_attachment_composition import (
        _biceps_row_arrays,
        _read_nhtiss4,
        _row_slices,
    )

    with tempfile.TemporaryDirectory(prefix="numi-vessel-owner-proof-") as directory:
        path = Path(directory) / "carrier.nhtissue"
        path.write_bytes(carrier_bytes)
        data = _read_nhtiss4(path)
        records = {int(row[6]): row for row in data["records"]}
        surface_by_id = {int(row["stable_id"]): row for row in carrier_surfaces}
        expected_specs = {
            475: ((1, 2), (20, 91)),
            477: ((3, 4), (20, 91)),
        }
        decoded_rows = {}
        for source_id, (surface_ids, core_ids) in expected_specs.items():
            source = expected_rows[source_id]
            _require(source_id in owner_updates, f"missing owner update for {source_id}")
            for surface_id, core_id in zip(surface_ids, core_ids):
                _require(surface_id in records and surface_id in surface_by_id,
                         f"emitted carrier row {surface_id} is missing")
                row = _row_slices(data, records[surface_id])
                arrays = _biceps_row_arrays(row)
                expected_vertices = np.ascontiguousarray(
                    source["carrier_vertices6"][core_id], dtype="<f4"
                )
                expected_faces = np.ascontiguousarray(source["faces"], dtype="<u4")
                _require(arrays["vertices6"].tobytes() == expected_vertices.tobytes()
                         and arrays["faces"].tobytes() == expected_faces.tobytes(),
                         f"carrier {surface_id} points/faces differ from source owner {source_id}")
                bindings = row["binding_bytes"]
                _require(len(bindings) == 72,
                         f"carrier {surface_id} binding table is not the exact body20/body91 pair")
                for local_id in range(len(expected_vertices)):
                    binding_slot = int(arrays["binding_indices"][local_id, 0])
                    _require(binding_slot in (0, 1)
                             and struct.unpack_from("<I", bindings, binding_slot * 36)[0] == core_id
                             and np.array_equal(arrays["weights"][local_id],
                                                np.asarray([1, 0, 0, 0], dtype="<f4")),
                             f"carrier {surface_id} vertex {local_id} does not resolve to body {core_id}")
                emitted_terms = surface_by_id[surface_id].get("vertices", [])
                _require(len(emitted_terms) == len(expected_vertices)
                         and all(term == [[0, surface_id, local_id, 1.0]]
                                 for local_id, term in enumerate(emitted_terms)),
                         f"carrier {surface_id} source-vertex ordering changed")
                decoded_rows[surface_id] = arrays["vertices6"]
        for source_id, (surface_ids, _core_ids) in expected_specs.items():
            source_terms = owner_updates[source_id]
            _require(len(source_terms) == len(expected_rows[source_id]["weights"]),
                     f"source owner-map length changed for {source_id}")
            weights = np.ascontiguousarray(expected_rows[source_id]["weights"], dtype="<f4")
            for local_id, terms in enumerate(source_terms):
                expected = [
                    [5, surface_id, local_id, float(weights[local_id, column])]
                    for column, surface_id in enumerate(surface_ids)
                    if weights[local_id, column] > 0
                ]
                _require(terms == expected,
                         f"source owner terms differ at row {source_id} vertex {local_id}")
        alias_rows = []
        weights477 = np.ascontiguousarray(expected_rows[477]["weights"], dtype="<f4")
        terms477 = owner_updates[477]
        for group in aliases:
            members = sorted(map(int, group))
            _require(len(members) > 1 and all(i < len(weights477) for i in members),
                     "exact-coordinate alias class is malformed")
            first = members[0]
            _require(all(weights477[i].tobytes() == weights477[first].tobytes()
                         and [term[0:2] + term[3:] for term in terms477[i]]
                         == [term[0:2] + term[3:] for term in terms477[first]]
                         for i in members),
                     "exact-coordinate aliases have different compiled owner coefficients")
            owner_records = []
            for column, surface_id, core_id in ((0, 3, 20), (1, 4, 91)):
                if weights477[first, column] == 0:
                    _require(all(not any(int(term[1]) == surface_id for term in terms477[i])
                                 for i in members),
                             "zero-weight body unexpectedly appears in alias parent terms")
                    continue
                points = decoded_rows[surface_id][np.asarray(members, dtype=np.int64), :3]
                _require(all(point.tobytes() == points[0].tobytes() for point in points[1:]),
                         f"body {core_id} carrier points differ across exact aliases")
                expected_coeff = float(weights477[first, column])
                _require(all(any(int(term[1]) == surface_id
                                 and int(term[2]) == i
                                 and struct.pack("<f", float(term[3]))
                                 == struct.pack("<f", expected_coeff)
                                 for term in terms477[i]) for i in members),
                         f"body {core_id} parent terms differ across exact aliases")
                owner_records.append({
                    "core_body_index": core_id,
                    "carrier_surface_id": surface_id,
                    "coefficient_f32_hex": weights477[first, column].tobytes().hex(),
                    "carrier_xyz_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
                })
            alias_rows.append({"source_local_vertex_ids": members,
                               "owner_coefficients_f32_hex": weights477[first].tobytes().hex(),
                               "resolved_carrier_owners": owner_records,
                               "per_slot_normals_retained_without_alias_normal_equality": True})
    return {
        "carrier_payload_sha256": hashlib.sha256(carrier_bytes).hexdigest(),
        "all_four_existing_carrier_rows_redecoded_and_matched_exact_source_points": True,
        "carrier_rows_resolve_to_core_bodies_20_and_91": True,
        "source_rows_475_and_477_parent_maps_match_emitted_carriers": True,
        "row477_exact_coordinate_alias_class_count": len(alias_rows),
        "all_row477_aliases_have_equal_parent_coefficients_and_same_body_local_points": True,
        "alias_classes": alias_rows,
        "normals_are_preserved_per_slot_and_not_used_as_an_alias_continuity_test": True,
        "scope": "Source-owner algebra proof only; native coupled-pose and anatomy gates remain required.",
    }


def _retire_isolated_source_face(
    row: dict[str, Any], face_id: int, source_face: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove one exact source-ancestral detached face without compacting vertices."""
    faces = np.asarray(row["faces"], dtype=np.uint32)
    expected = np.asarray(source_face, dtype=np.uint32)
    _require(
        0 <= face_id < len(faces) and np.array_equal(faces[face_id], expected),
        "scoped FJ2217 face retirement lacks exact local/source face ancestry",
    )
    kept = np.concatenate((faces[:face_id], faces[face_id + 1 :]), axis=0)
    record = tuple(row["record"])
    _require(
        len(record) == 8
        and int(record[5]) == 477
        and int(record[2]) == len(row["vertices6"]),
        "row477 identity/vertex allocation changed before face retirement",
    )
    replacement = {
        **row,
        "faces": kept,
        "record": (*record[:4], int(kept.size), *record[5:]),
    }
    _require(
        np.array_equal(replacement["vertices6"], row["vertices6"])
        and len(kept) == len(faces) - 1,
        "face retirement moved/compacted source vertices",
    )
    return replacement, {
        "stable_id": 477,
        "source_member": "FJ2217",
        "source_face_id_zero_based": face_id,
        "removed_face_count": 1,
        "input_face_count": len(faces),
        "output_face_count": len(kept),
        "input_vertex_count": len(row["vertices6"]),
        "output_vertex_count": len(replacement["vertices6"]),
        "unchanged_source_vertex_prefix_byte_exact": True,
        "unused_original_vertex_ids_after_retirement": sorted(
            set(map(int, np.asarray(source_face).tolist()))
        ),
        "source_face_bytes_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "physical_mass_or_state_added": False,
        "native_confirmation_required": True,
        "anatomical_admission": False,
    }


def validate_local_source_face_retirement(
    base_row: dict[str, Any],
    candidate_row: dict[str, Any],
    retirement: dict[str, Any],
    reemit_proof: dict[str, Any],
    expected_pre_retirement_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct the one pinned FJ2217 face retirement without compacting slots."""
    lineage = retirement.get("lineage", {})
    face_id = int(retirement.get("source_face_id_zero_based", -1))
    base_faces = np.ascontiguousarray(base_row.get("faces"), dtype="<u4")
    candidate_faces = np.ascontiguousarray(candidate_row.get("faces"), dtype="<u4")
    expected_row = base_row if expected_pre_retirement_row is None else expected_pre_retirement_row
    vertices = np.ascontiguousarray(expected_row.get("vertices6"), dtype="<f4")
    candidate_vertices = np.ascontiguousarray(candidate_row.get("vertices6"), dtype="<f4")
    _require(
        retirement.get("stable_id") == 477
        and retirement.get("removed_face_count") == 1
        and retirement.get("source_archive_sha256") == ISA_ARCHIVE_SHA256
        and retirement.get("source_member_sha256") == FJ2217_OBJ_SHA256
        and retirement.get("source_face_and_nha_ancestry_exact") is True
        and retirement.get("all_other_source_faces_retained_in_original_order") is True
        and retirement.get("unchanged_source_vertex_prefix_byte_exact") is True
        and retirement.get("input_vertex_count") == len(vertices)
        and retirement.get("output_vertex_count") == len(vertices)
        and retirement.get("retained_unused_vertex_slots") == 3,
        "FJ2217 retirement declaration is not the exact pinned one-face operation",
    )
    _require(
        lineage.get("classification")
        == "unexplained_isolated_raw_source_face; scoped_reference_retirement"
        and lineage.get("anatomical_interface_proven") is False
        and lineage.get("local_retirement_proposed") is True
        and lineage.get("retirement_operation_changes_no_mesh_coordinates") is True
        and lineage.get("source_archive_sha256") == ISA_ARCHIVE_SHA256
        and lineage.get("source_member_sha256") == FJ2217_OBJ_SHA256
        and lineage.get("source_face_and_nha_face_arrays_exact") is True
        and lineage.get("source_face_id_zero_based") == face_id
        and lineage.get("nha_local_face_id_zero_based") == face_id,
        "FJ2217 source-face lineage is absent or changed",
    )
    _require(
        base_faces.ndim == 2 and base_faces.shape[1] == 3
        and 0 <= face_id < len(base_faces)
        and len(candidate_faces) == len(base_faces) - 1,
        "FJ2217 retirement face index or dimensions changed",
    )
    source_face = base_faces[face_id].copy()
    source_face_digest = hashlib.sha256(source_face.tobytes()).hexdigest()
    _require(
        retirement.get("source_face_bytes_sha256") == source_face_digest
        and lineage.get("ordered_source_vertex_ids_zero_based") == source_face.astype(int).tolist()
        and lineage.get("nha_local_vertex_ids_zero_based") == source_face.astype(int).tolist(),
        "FJ2217 retired face no longer matches exact source vertex ancestry",
    )
    expected_pre_faces = np.ascontiguousarray(expected_row.get("faces"), dtype="<u4")
    _require(
        expected_pre_faces.tobytes() == base_faces.tobytes(),
        "route110 changed FJ2217 source-face order before scoped retirement",
    )
    expected_faces = np.concatenate((base_faces[:face_id], base_faces[face_id + 1 :]))
    _require(
        candidate_faces.tobytes() == expected_faces.tobytes()
        and candidate_vertices.shape == vertices.shape
        and candidate_vertices.tobytes() == vertices.tobytes(),
        "FJ2217 retirement changed another face, vertex, or vertex order",
    )
    original_record = tuple(map(int, base_row.get("record", ())))
    candidate_record = tuple(map(int, candidate_row.get("record", ())))
    _require(
        len(original_record) == len(candidate_record) == 8
        and candidate_record[0] == original_record[0] == 91
        and candidate_record[2] == original_record[2] == len(vertices)
        and candidate_record[4] == len(candidate_faces) * 3
        and candidate_record[5:] == original_record[5:] == (477, 2, 0),
        "FJ2217 ABI5 body, allocation, identity, or face-count metadata changed",
    )
    retired_vertices = set(map(int, source_face))
    _require(
        all(not retired_vertices.intersection(map(int, face)) for face in candidate_faces),
        "retired FJ2217 source slots remain referenced by another face",
    )
    _require(
        477 in reemit_proof.get("replaced_stable_ids", [])
        and reemit_proof.get("stable_id_order_preserved") is True
        and reemit_proof.get("global_offsets_rebased_by_existing_abi5_encoder") is True
        and reemit_proof.get("registration_and_source_header_bytes_preserved") is True,
        "FJ2217 retirement did not use the exact existing ABI5 row-replacement owner",
    )
    return {
        "stable_id": 477,
        "source_face_id_zero_based": face_id,
        "removed_face_count": 1,
        "unchanged_source_vertex_prefix_byte_exact": True,
        "all_other_faces_exact_and_ordered": True,
        "retired_slots_unreferenced": sorted(retired_vertices),
        "existing_abi5_reemit_proof": True,
        "anatomical_admission": False,
    }



FJ2217_CAP_SOURCE_FACE_ID = 1858
FJ2217_CAP_SOURCE_VERTEX_IDS = (930, 1066, 1063)
FJ2217_CAP_ALIAS_VERTEX_IDS = (1068, 1065, 926)


def reinstate_source_face_through_exact_aliases(
    row: dict[str, Any],
    face_id: int,
    source_face: np.ndarray,
    *,
    source_row: dict[str, Any] | None = None,
    alias_vertex_ids: tuple[int, int, int] = FJ2217_CAP_ALIAS_VERTEX_IDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore one source face using existing Float32-equal row vertices.

    The only supported use is the pinned FJ2217 face 1858 cap. It either
    replaces that face's original corner IDs or inserts it into the exact
    source-order hole left by the historical retirement.
    """
    faces = np.ascontiguousarray(row.get("faces"), dtype="<u4")
    source = np.ascontiguousarray(source_face, dtype="<u4").reshape(-1)
    vertices6 = np.ascontiguousarray(row.get("vertices6"), dtype="<f4")
    expected = row if source_row is None else source_row
    expected_faces = np.ascontiguousarray(expected.get("faces"), dtype="<u4")
    expected_vertices6 = np.ascontiguousarray(expected.get("vertices6"), dtype="<f4")
    aliases = tuple(map(int, alias_vertex_ids))
    _require(
        int(row.get("record", (0, 0, 0, 0, 0, -1, 0, 0))[5]) == 477
        and int(face_id) == FJ2217_CAP_SOURCE_FACE_ID
        and source.tolist() == list(FJ2217_CAP_SOURCE_VERTEX_IDS)
        and aliases == FJ2217_CAP_ALIAS_VERTEX_IDS,
        "only the pinned FJ2217 source cap and exact alias triple may be reinstated",
    )
    _require(
        faces.ndim == 2 and faces.shape[1] == 3
        and expected_faces.ndim == 2 and expected_faces.shape[1] == 3
        and vertices6.ndim == 2 and vertices6.shape[1] == 6
        and expected_vertices6.shape == vertices6.shape
        and 0 <= face_id < len(expected_faces)
        and np.array_equal(expected_faces[face_id], source)
        and expected_vertices6.tobytes() == vertices6.tobytes(),
        "FJ2217 source face/order or unchanged vertex/normal bytes are not pinned",
    )
    _require(
        max((*source.tolist(), *aliases)) < len(vertices6)
        and all(alias in set(map(int, faces.reshape(-1))) for alias in aliases),
        "FJ2217 cap aliases are not existing referenced source slots",
    )
    for source_vertex, alias_vertex in zip(source.tolist(), aliases):
        _require(
            vertices6[source_vertex, :3].astype("<f4").tobytes()
            == vertices6[alias_vertex, :3].astype("<f4").tobytes(),
            f"FJ2217 cap alias {alias_vertex} is not Float32-coordinate exact",
        )
    if len(faces) == len(expected_faces):
        _require(
            faces.tobytes() == expected_faces.tobytes(),
            "FJ2217 source row differs outside its exact cap replacement",
        )
        output_faces = faces.copy()
        output_faces[face_id] = np.asarray(aliases, dtype="<u4")
        operation = "replace_original_face_corners_by_exact_coordinate_aliases"
    elif len(faces) + 1 == len(expected_faces):
        _require(
            np.array_equal(faces, np.concatenate((expected_faces[:face_id], expected_faces[face_id + 1 :]))),
            "FJ2217 retired input is not the exact source-order face hole",
        )
        output_faces = np.concatenate(
            (faces[:face_id], np.asarray([aliases], dtype="<u4"), faces[face_id:]),
            axis=0,
        )
        operation = "reinsert_retired_source_face_at_original_order_position"
    else:
        raise ValueError("vessel owner blend: FJ2217 cap input face count is unsupported")
    original_record = tuple(map(int, row["record"]))
    replacement = {
        **row,
        "faces": output_faces,
        "record": (*original_record[:4], int(output_faces.size), *original_record[5:]),
    }
    _require(
        replacement["vertices6"].tobytes() == vertices6.tobytes()
        and len(output_faces) == len(expected_faces)
        and output_faces[face_id].tolist() == list(aliases),
        "FJ2217 cap reinsertion changed coordinates, count, or source face order",
    )
    source_digest = hashlib.sha256(source.tobytes()).hexdigest()
    declaration = {
        "schema": "numi.human.local-source-face-exact-alias-reinstatement.v1",
        "stable_id": 477,
        "source_member": "FJ2217",
        "source_archive_sha256": ISA_ARCHIVE_SHA256,
        "source_member_sha256": FJ2217_OBJ_SHA256,
        "source_face_id_zero_based": int(face_id),
        "source_vertex_ids_zero_based": list(FJ2217_CAP_SOURCE_VERTEX_IDS),
        "alias_vertex_ids_zero_based": list(FJ2217_CAP_ALIAS_VERTEX_IDS),
        "source_face_bytes_sha256": source_digest,
        "operation": operation,
        "input_face_count": len(faces),
        "output_face_count": len(output_faces),
        "input_vertex_count": len(vertices6),
        "output_vertex_count": len(vertices6),
        "coordinate_equivalence": "bitwise Float32 xyz equality for each corresponding source/alias vertex",
        "coordinate_aliases_float32_bit_exact": True,
        "source_face_ancestry_exact": True,
        "all_other_faces_byte_exact_and_ordered": True,
        "all_vertex_and_normal_bytes_unchanged": True,
        "unused_original_vertex_slots_preserved": True,
        "physical_mass_or_blood_state_added": False,
        "native_confirmation_required": True,
        "anatomical_admission": False,
    }
    return replacement, declaration


def validate_local_source_face_reinstatement(
    base_row: dict[str, Any],
    candidate_row: dict[str, Any],
    reinstatement: dict[str, Any],
    reemit_proof: dict[str, Any],
    expected_source_row: dict[str, Any],
) -> dict[str, Any]:
    """Verify the exact ABI5 FJ2217 cap successor and its retained source ancestry."""
    face_id = int(reinstatement.get("source_face_id_zero_based", -1))
    aliases = tuple(map(int, reinstatement.get("alias_vertex_ids_zero_based", ())))
    source_ids = tuple(map(int, reinstatement.get("source_vertex_ids_zero_based", ())))
    expected_faces = np.ascontiguousarray(expected_source_row.get("faces"), dtype="<u4")
    expected_vertices = np.ascontiguousarray(expected_source_row.get("vertices6"), dtype="<f4")
    candidate_faces = np.ascontiguousarray(candidate_row.get("faces"), dtype="<u4")
    candidate_vertices = np.ascontiguousarray(candidate_row.get("vertices6"), dtype="<f4")
    base_faces = np.ascontiguousarray(base_row.get("faces"), dtype="<u4")
    base_vertices = np.ascontiguousarray(base_row.get("vertices6"), dtype="<f4")
    _require(
        reinstatement.get("schema") == "numi.human.local-source-face-exact-alias-reinstatement.v1"
        and reinstatement.get("stable_id") == 477
        and reinstatement.get("source_member") == "FJ2217"
        and reinstatement.get("source_archive_sha256") == ISA_ARCHIVE_SHA256
        and reinstatement.get("source_member_sha256") == FJ2217_OBJ_SHA256
        and face_id == FJ2217_CAP_SOURCE_FACE_ID
        and source_ids == FJ2217_CAP_SOURCE_VERTEX_IDS
        and aliases == FJ2217_CAP_ALIAS_VERTEX_IDS
        and reinstatement.get("coordinate_aliases_float32_bit_exact") is True
        and reinstatement.get("source_face_ancestry_exact") is True
        and reinstatement.get("all_other_faces_byte_exact_and_ordered") is True
        and reinstatement.get("all_vertex_and_normal_bytes_unchanged") is True
        and reinstatement.get("physical_mass_or_blood_state_added") is False,
        "FJ2217 cap reinstatement declaration is not the exact pinned operation",
    )
    _require(
        expected_faces.ndim == 2 and expected_faces.shape[1] == 3
        and 0 <= face_id < len(expected_faces)
        and expected_faces[face_id].tolist() == list(source_ids)
        and candidate_faces.shape == expected_faces.shape
        and candidate_faces.tobytes()
        == np.concatenate(
            (expected_faces[:face_id], np.asarray([aliases], dtype="<u4"), expected_faces[face_id + 1 :]),
            axis=0,
        ).tobytes(),
        "FJ2217 output face ancestry or order differs from the exact source cap",
    )
    _require(
        candidate_vertices.shape == expected_vertices.shape
        and candidate_vertices.tobytes() == expected_vertices.tobytes()
        and base_vertices.shape == expected_vertices.shape
        and base_vertices.tobytes() == expected_vertices.tobytes(),
        "FJ2217 cap repair changed position, normal, or vertex-prefix bytes",
    )
    for source_vertex, alias_vertex in zip(source_ids, aliases):
        _require(
            expected_vertices[source_vertex, :3].astype("<f4").tobytes()
            == expected_vertices[alias_vertex, :3].astype("<f4").tobytes(),
            f"FJ2217 alias {alias_vertex} is not Float32-coordinate exact",
        )
    _require(
        len(base_faces) in (len(expected_faces), len(expected_faces) - 1)
        and reinstatement.get("input_face_count") == len(base_faces)
        and reinstatement.get("output_face_count") == len(expected_faces)
        and reinstatement.get("input_vertex_count") == len(base_vertices)
        and reinstatement.get("output_vertex_count") == len(candidate_vertices),
        "FJ2217 cap input/output dimensions differ from its receipt",
    )
    if len(base_faces) == len(expected_faces):
        _require(
            np.array_equal(base_faces, expected_faces),
            "FJ2217 base face namespace differs outside the cap",
        )
    else:
        _require(
            np.array_equal(base_faces, np.concatenate((expected_faces[:face_id], expected_faces[face_id + 1 :]))),
            "FJ2217 base is not the exact historical one-face retirement",
        )
    original_record = tuple(map(int, expected_source_row["record"]))
    candidate_record = tuple(map(int, candidate_row["record"]))
    _require(
        len(candidate_record) == 8
        and candidate_record[0] == original_record[0] == 91
        and candidate_record[2] == original_record[2] == len(candidate_vertices)
        and candidate_record[4] == candidate_faces.size
        and candidate_record[5:] == original_record[5:] == (477, 2, 0)
        and candidate_record[3] == original_record[3],
        "FJ2217 ABI5 identity, bounds, or face-count metadata changed",
    )
    _require(
        reinstatement.get("source_face_bytes_sha256")
        == hashlib.sha256(expected_faces[face_id].tobytes()).hexdigest()
        and 477 in reemit_proof.get("replaced_stable_ids", [])
        and reemit_proof.get("stable_id_order_preserved") is True
        and reemit_proof.get("global_offsets_rebased_by_existing_abi5_encoder") is True
        and reemit_proof.get("registration_and_source_header_bytes_preserved") is True,
        "FJ2217 source ancestry or existing ABI5 row-replacement proof is incomplete",
    )
    return {
        "stable_id": 477,
        "source_face_id_zero_based": face_id,
        "restored_face_count": 1,
        "source_vertex_ids_zero_based": list(source_ids),
        "alias_vertex_ids_zero_based": list(aliases),
        "alias_coordinates_float32_bit_exact": True,
        "all_other_faces_exact_and_ordered": True,
        "all_vertices_and_normals_unchanged": True,
        "existing_abi5_reemit_proof": True,
        "physical_mass_or_blood_state_added": False,
        "anatomical_admission": False,
    }



def validate_fj2217_route_face_operation(
    base_row: dict[str, Any],
    candidate_row: dict[str, Any],
    expected_route_row: dict[str, Any],
    *,
    reinstatement: dict[str, Any] | None,
    active_retirement: dict[str, Any] | None,
    historical_retirement: dict[str, Any] | None,
    route_reemit_proof: dict[str, Any],
    historical_retirement_reemit_proof: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate a current reinstatement or replay one exact legacy retirement receipt."""
    if reinstatement is not None:
        _require(
            active_retirement is None,
            "FJ2217 reinstatement receipt must not retain an active retirement",
        )
        operation = reinstatement.get("operation")
        if operation == "replace_original_face_corners_by_exact_coordinate_aliases":
            _require(
                historical_retirement is None
                and historical_retirement_reemit_proof is None,
                "fresh FJ2217 replacement must not fabricate historical retirement",
            )
        elif operation == "reinsert_retired_source_face_at_original_order_position":
            _require(
                isinstance(historical_retirement, dict)
                and isinstance(historical_retirement_reemit_proof, dict),
                "FJ2217 reinsertion must retain its exact historical retirement receipt",
            )
            source_face = np.ascontiguousarray(
                expected_route_row["faces"][FJ2217_CAP_SOURCE_FACE_ID], dtype="<u4"
            )
            historical_row, _ = _retire_isolated_source_face(
                expected_route_row, FJ2217_CAP_SOURCE_FACE_ID, source_face
            )
            validate_local_source_face_retirement(
                expected_route_row,
                historical_row,
                historical_retirement,
                historical_retirement_reemit_proof,
                expected_pre_retirement_row=expected_route_row,
            )
        else:
            raise ValueError("FJ2217 reinstatement operation is not a pinned operation")
        proof = validate_local_source_face_reinstatement(
            base_row,
            candidate_row,
            reinstatement,
            route_reemit_proof,
            expected_source_row=expected_route_row,
        )
        return {**proof, "receipt_operation": operation}

    _require(
        isinstance(active_retirement, dict)
        and historical_retirement is None
        and historical_retirement_reemit_proof is None,
        "legacy FJ2217 receipt lacks its active scoped retirement proof",
    )
    proof = validate_local_source_face_retirement(
        base_row,
        candidate_row,
        active_retirement,
        route_reemit_proof,
        expected_pre_retirement_row=expected_route_row,
    )
    return {**proof, "receipt_operation": "legacy_scoped_retirement_replay"}

def validate_fj2217_reinstatement_parent_maps_unchanged(
    base_parent_bindings: dict[str, Any],
    candidate_parent_bindings: dict[str, Any],
) -> dict[str, Any]:
    """Prove the direct cap successor leaves all existing parent maps intact."""
    _require(
        isinstance(base_parent_bindings, dict)
        and isinstance(candidate_parent_bindings, dict),
        "FJ2217 parent-binding tables must be objects",
    )
    for key in ("child_payload_sha256", "source_payloads", "surfaces", "anatomy_surfaces"):
        _require(
            base_parent_bindings.get(key) == candidate_parent_bindings.get(key),
            f"FJ2217 cap successor changed passive parent field {key}",
        )
    base_rows = {
        int(item["stable_id"]): item
        for item in base_parent_bindings.get("anatomy_surfaces", [])
    }
    candidate_rows = {
        int(item["stable_id"]): item
        for item in candidate_parent_bindings.get("anatomy_surfaces", [])
    }
    _require(
        477 in base_rows and base_rows == candidate_rows,
        "FJ2217 cap successor changed ordered passive parent maps",
    )
    encoded = json.dumps(
        base_parent_bindings.get("anatomy_surfaces"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "stable_id": 477,
        "anatomy_surface_map_sha256": hashlib.sha256(encoded).hexdigest(),
        "parent_source_payload_sha256s": [
            item["sha256"] for item in base_parent_bindings["source_payloads"]
        ],
        "all_parent_surface_maps_unchanged": True,
        "ordered_parent_terms_unchanged": True,
        "source_payload_order_unchanged": True,
    }


ROUTE110_DIR = Path(
    "/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/"
    "full-anatomy-delivery-audit-1378/route-candidate-110-nonlinear-closed-volume-runner-004"
)
ROUTE110_PINS = {
    "npz": "2f30c92319f24f8163fd9f8e6cbffec45c1171838fb23908d78c29f74672125b",
    "report": "d313b78d7d2a0a2cf64878d76c15c03f0ea5a64a4f5767c172828197d90c332c",
    "runner": "f1c2c0613851a6faacf084ead04d98da98ada5b735705237d05a3cdcd5de6941",
    "checkpoint": "6f980184b96f5af917a0ef5db7c66b3514acb586f5cfc3b5e09323e7adcd0909",
}
ROUTE110_ROWS = (7, 10, 469, 471, 473, 475, 477, 479, 481, 483, 485, 487, 489)



def apply_parent_surface_updates(
    parent_bindings: dict[str, Any],
    preparation: dict[str, Any],
    anatomy_payload_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Append one carrier and replace only the two pinned axillary-vessel maps."""
    updated = deepcopy(parent_bindings)
    sources = updated.get("source_payloads")
    _require(
        isinstance(sources, list) and len(sources) == 5,
        "expected the exact five-source passive-parent table",
    )
    source_entry = {
        "path": preparation["carrier_source_path"],
        "sha256": preparation["carrier_payload_sha256"],
    }
    _require(
        source_entry["path"] == "assets/route110-dual-vessel-reference-carriers.nhtissue",
        "new carrier path changed",
    )
    _require(
        isinstance(anatomy_payload_sha256, str) and len(anatomy_payload_sha256) == 64,
        "invalid anatomy payload identity",
    )
    maps = updated.get("anatomy_surfaces")
    _require(isinstance(maps, list), "passive anatomy surface maps are missing")
    by_id = {int(row["stable_id"]): deepcopy(row) for row in maps}
    _require(
        len(by_id) == len(maps) and 475 not in by_id and 477 in by_id,
        "pinned 475/477 passive map identities changed",
    )
    updates = preparation["parent_surface_updates"]
    _require(
        set(map(int, updates)) == {475, 477}, "passive owner update stable IDs changed"
    )
    for sid in (475, 477):
        terms = updates[sid]
        _require(
            len(terms) > 0 and all(len(vertex_terms) > 0 for vertex_terms in terms),
            f"row {sid} has a vertex with no positive owner",
        )
        by_id[sid] = (
            {"stable_id": sid, "vertices": terms}
            if sid == 475
            else {**by_id[sid], "vertices": terms}
        )
    updated["source_payloads"] = [*sources, source_entry]
    updated["anatomy_surfaces"] = [by_id[sid] for sid in sorted(by_id)]
    updated["anatomy_payload_sha256"] = anatomy_payload_sha256
    owner_map_sha256 = hashlib.sha256(
        json.dumps(
            {str(sid): updates[sid] for sid in (475, 477)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return updated, owner_map_sha256


def prepare_vessel_owner_blend(
    *,
    base_receipt_path: Path,
    n27_receipt_path: Path,
    port_report_path: Path,
    rigid_path: Path,
    sources_path: Path,
    route_npz_path: Path = ROUTE110_DIR / "candidate-110-source-rows.npz",
    route_report_path: Path = ROUTE110_DIR / "report.json",
    route_script_path: Path = ROUTE110_DIR / "run_route110.py",
    route_checkpoint_path: Path = ROUTE110_DIR / "target-prep-checkpoint.json",
) -> dict[str, Any]:
    """Prepare NHTISS carrier bytes and anatomy-parent terms for rows 475/477.

    This is source/hash/rest-frame preparation only. Geometry admission and
    accepted-pose/native gates remain separate.
    """
    from . import model as human
    from .passive_attachment_composition import (
        _biceps_row_arrays,
        _read_nhtiss4,
        _row_slices,
    )
    from .passive_tissue_parent_carriers import CarrierRow, encode_carriers
    from .resting_passive_viscera import _parse_payload, _record_content_bytes
    from .resting_scene import load_rigid

    for path, digest, label in (
        (n27_receipt_path, N27_RECEIPT_SHA256, "N27 receipt"),
        (port_report_path, PORT_REPORT_SHA256, "086 collar report"),
        (rigid_path, RIGID_SHA256, "NHRIGID"),
    ):
        _require(
            Path(path).is_file() and _sha(Path(path)) == digest, label + " hash changed"
        )
    n27_path, base_path = (
        Path(n27_receipt_path).resolve(),
        Path(base_receipt_path).resolve(),
    )
    n27, base = json.loads(n27_path.read_text()), json.loads(base_path.read_text())
    n27_payload, base_payload = (
        Path(n27["payload"]["path"]).resolve(),
        Path(base["payload"]["path"]).resolve(),
    )
    _require(
        _sha(n27_payload) == N27_PAYLOAD_SHA256 == n27["payload"]["sha256"],
        "N27 payload changed",
    )
    _require(
        _sha(base_payload) == base["payload"]["sha256"]
        and base["functional_bindings"]["anatomy_payload_sha256"]
        == base["payload"]["sha256"],
        "base receipt does not bind NHA",
    )
    _, _, nr = _parse_payload(n27_payload.read_bytes())
    _, _, br = _parse_payload(base_payload.read_bytes())
    route = load_route110_reference_rows(
        n27_receipt_path=n27_path,
        base_rows=nr,
        route_npz_path=route_npz_path,
        route_report_path=route_report_path,
        route_script_path=route_script_path,
        route_checkpoint_path=route_checkpoint_path,
    )
    route_rows = route["rows"]
    for sid, member, compartment in ((475, "FJ2216", 3), (477, "FJ2217", 4)):
        _require(
            _record_content_bytes(nr[sid]) == _record_content_bytes(br[sid]),
            f"row {sid} differs from pinned N27",
        )
        for rec in (n27, base):
            meta = rec["provenance"]["source_id_map"][str(sid)]
            _require(
                meta.get("source_member") == member
                and meta.get("body_index") == 91
                and meta.get("layer") == 2
                and meta.get("vascular_compartment_id") == compartment,
                f"row {sid} source identity/body/layer/CVSim binding changed",
            )
    source_dir = Path(sources_path).resolve()
    archive_entry = next(
        (
            x
            for x in n27["provenance"].get("bodyparts3d_archives", [])
            if x.get("hierarchy") == "is_a"
            and x.get("file") == "isa_BP3D_4.0_obj_99.zip"
        ),
        None,
    )
    _require(
        archive_entry is not None and archive_entry.get("sha256") == ISA_ARCHIVE_SHA256,
        "pinned is_a archive entry changed",
    )
    source_archive = source_dir / archive_entry["file"]
    _require(
        source_archive.is_file() and _sha(source_archive) == ISA_ARCHIVE_SHA256,
        "pinned FJ2217 source archive changed",
    )
    _, source_member_path, source_obj = human._bodyparts_obj_member(
        source_dir, "is_a", "FJ2217"
    )
    _require(
        source_member_path == "isa_BP3D_4.0_obj_99/FJ2217.obj"
        and hashlib.sha256(source_obj).hexdigest() == FJ2217_OBJ_SHA256,
        "raw FJ2217 source member identity changed",
    )
    source_xyz, source_faces = human._bodyparts_obj_triangles(
        source_obj, source_member_path
    )
    source_faces = np.asarray(source_faces, dtype=np.int64)
    _require(
        source_faces.shape == nr[477]["faces"].shape
        and np.array_equal(source_faces, nr[477]["faces"]),
        "FJ2217 source face order or face ancestry differs from N27 row477",
    )
    _require(
        len(source_xyz) == len(nr[477]["vertices6"]),
        "FJ2217 source vertex order/count differs from N27 row477",
    )
    report = json.loads(Path(port_report_path).read_text())
    _require(
        report.get("source_nha_sha256") == N27_PAYLOAD_SHA256,
        "086 report does not bind N27",
    )
    p471, c475 = _collar(report, 471, 475)
    p475, _c479 = _collar(report, 475, 479)
    _p473, c477 = _collar(report, 473, 477)
    p477, _c481 = _collar(report, 477, 481)
    collars = {475: (c475, p475), 477: (c477, p477)}
    for sid, (prox, distal) in collars.items():
        _require(
            prox.isdisjoint(distal) and max(prox | distal) < len(nr[sid]["vertices6"]),
            f"row {sid} collar IDs invalid",
        )
    common = n27["provenance"]["cardiac_geometry_binding"]["common_field"]
    ranges = common["passive_attachment_ranges"]
    _require(
        not any(int(x.get("stable_id", -1)) in (473, 475, 477) for x in ranges),
        "unexpected common-field range on 473/475/477",
    )
    rows471 = [x for x in ranges if int(x.get("stable_id", -1)) == 471]
    _require(len(rows471) == 1, "471 common range missing or repeated")
    map_info = common["map"]
    map_path = Path(map_info["path"])
    if not map_path.is_absolute():
        map_path = n27_path.parent / map_path
    _require(
        _sha(map_path)
        == map_info["sha256"]
        == "c0b0811fe2e2fb45dc37201205e36af39cf092739cfb52c56fdbad9a67dbb858",
        "common map bytes changed",
    )
    _require(
        map_path.stat().st_size == 620718 * 112
        and map_info["record_count"] == 620718
        and map_info["record_stride_bytes"] == 112,
        "common map dimensions changed",
    )
    rr = rows471[0]
    start, count = int(rr["first_vertex"]), int(rr["vertex_count"])
    _require(
        count == len(nr[471]["vertices6"]),
        "471 common range differs from its source row",
    )
    fv = (
        np.fromfile(map_path, dtype="<f4")
        .reshape(-1, 7, 4)[start + np.asarray(sorted(p471), dtype=np.int64)]
        .view("<u4")
    )
    _require(
        np.all(fv == 0), "471 matched collar has nonzero common-field coefficients"
    )
    nb = n27["passive_tissue_parent_bindings"]
    cb = base.get("passive_tissue_parent_bindings")
    _require(
        isinstance(cb, dict)
        and cb.get("child_payload_sha256") == nb["child_payload_sha256"]
        and [x["sha256"] for x in cb.get("source_payloads", [])]
        == [x["sha256"] for x in nb["source_payloads"]],
        "existing passive parent source identity changed",
    )
    base_surface_maps = {
        int(x["stable_id"]): x["vertices"] for x in cb.get("anatomy_surfaces", [])
    }
    n27_surface_maps = {
        int(x["stable_id"]): x["vertices"] for x in nb.get("anatomy_surfaces", [])
    }
    _require(
        475 not in base_surface_maps
        and 477 in base_surface_maps
        and base_surface_maps[477] == n27_surface_maps.get(477),
        "base 475/477 passive map is not the pinned N27 owner map",
    )
    carrier_path = (n27_path.parent / nb["source_payloads"][4]["path"]).resolve()
    _require(_sha(carrier_path) == CARRIER_477_SHA256, "477 carrier asset changed")
    cd = _read_nhtiss4(carrier_path)
    cr = {int(x[6]): x for x in cd["records"]}
    old = {sid: _row_slices(cd, cr[sid]) for sid in (1, 2)}
    ca = {sid: _biceps_row_arrays(old[sid]) for sid in (1, 2)}
    _require(
        np.array_equal(ca[2]["vertices6"], nr[477]["vertices6"])
        and all(np.array_equal(ca[sid]["faces"], nr[477]["faces"]) for sid in (1, 2)),
        "477 carrier ancestry/order changed",
    )
    owner475 = route_rows[475]
    owner477 = route_rows[477]
    w475, q475 = geodesic_owner_partition(
        owner475["vertices6"][:, :3],
        owner475["faces"],
        *collars[475],
        exact_coordinate_quotient=True,
    )
    baseline_477 = next(
        (
            x
            for x in n27["passive_tissue_parent_bindings"].get("anatomy_surfaces", [])
            if int(x.get("stable_id", -1)) == 477
        ),
        None,
    )
    _require(
        baseline_477 is not None
        and len(baseline_477.get("vertices", [])) == len(nr[477]["vertices6"]),
        "pinned 477 prior owner map is missing",
    )
    graph477, tri477 = _mesh_graph(owner477["vertices6"][:, :3], owner477["faces"])
    components477 = _mesh_components(graph477)
    main477 = next(
        c for c in components477 if c.intersection(collars[477][0] | collars[477][1])
    )
    remainder_components = [c for c in components477 if c is not main477]
    isolated477 = set().union(*remainder_components) if remainder_components else set()
    if isolated477:
        _require(
            len(remainder_components) == 1
            and len(isolated477) == 3
            and sum(all(int(v) in isolated477 for v in face) for face in tri477) == 1,
            "477 has an unapproved disconnected source remainder",
        )
        for vertex in isolated477:
            _require(
                baseline_477["vertices"][vertex] == [[4, 2, vertex, 1.0]],
                "477 isolated source face is not already Body91-only",
            )
    w477, q477 = geodesic_owner_partition(
        owner477["vertices6"][:, :3],
        owner477["faces"],
        *collars[477],
        isolated_distal_remainder=isolated477,
        exact_coordinate_quotient=True,
    )
    isolated_face_ids = [
        int(face_id)
        for face_id, face in enumerate(tri477)
        if isolated477 and all(int(vertex) in isolated477 for vertex in face)
    ]
    isolated_face_lineage = None
    if isolated477:
        _require(
            len(isolated_face_ids) == 1
            and isolated_face_ids[0] < len(source_faces)
            and np.array_equal(
                tri477[isolated_face_ids[0]], source_faces[isolated_face_ids[0]]
            ),
            "FJ2217 isolated source face ancestry is not one exact source face",
        )
        source_face_id = isolated_face_ids[0]
        face_vertex_ids = [int(x) for x in source_faces[source_face_id]]
        points = np.asarray(owner477["vertices6"][face_vertex_ids, :3], dtype=np.float64)
        face_area_m2 = float(
            np.linalg.norm(np.cross(points[1] - points[0], points[2] - points[0])) / 2.0
        )
        main_ids = np.asarray(sorted(main477), dtype=np.int64)
        main_points = owner477["vertices6"][main_ids, :3]
        coincident_main_ids = {}
        for local_vertex in sorted(isolated477):
            raw_position = owner477["vertices6"][local_vertex, :3]
            coincident = [
                int(main_ids[i])
                for i, point in enumerate(main_points)
                if point.tobytes() == raw_position.tobytes()
            ]
            coincident_main_ids[str(local_vertex)] = coincident
        main_face_ids = [
            int(face_id)
            for face_id, face in enumerate(tri477)
            if not any(int(vertex) in isolated477 for vertex in face)
        ]
        geometric_edge_face_ancestry = {}
        for edge_slot in range(3):
            va = face_vertex_ids[edge_slot]
            vb = face_vertex_ids[(edge_slot + 1) % 3]
            ka = owner477["vertices6"][va, :3].tobytes()
            kb = owner477["vertices6"][vb, :3].tobytes()
            geometric_edge_face_ancestry[f"{va}-{vb}"] = [
                face_id
                for face_id in main_face_ids
                if ka
                in [
                    owner477["vertices6"][int(vertex), :3].tobytes()
                    for vertex in tri477[face_id]
                ]
                and kb
                in [
                    owner477["vertices6"][int(vertex), :3].tobytes()
                    for vertex in tri477[face_id]
                ]
            ]
        exact_coordinate_face_matches = [
            face_id
            for face_id in main_face_ids
            if sorted(
                owner477["vertices6"][int(vertex), :3].tobytes()
                for vertex in tri477[face_id]
            )
            == sorted(points.astype("<f4")[i].tobytes() for i in range(3))
        ]
        _require(
            np.isfinite(points).all() and face_area_m2 > 0,
            "isolated FJ2217 source face has invalid source geometry",
        )
        isolated_face_lineage = {
            "classification": "exact-source-ancestral-cap-reinstatement-candidate",
            "anatomical_interface_proven": False,
            "local_exact_alias_reinstatement_proposed": True,
            "source_archive_sha256": ISA_ARCHIVE_SHA256,
            "source_member_path": source_member_path,
            "source_member_sha256": FJ2217_OBJ_SHA256,
            "source_face_id_zero_based": source_face_id,
            "nha_local_face_id_zero_based": isolated_face_ids[0],
            "ordered_source_vertex_ids_zero_based": face_vertex_ids,
            "nha_local_vertex_ids_zero_based": face_vertex_ids,
            "source_face_and_nha_face_arrays_exact": True,
            "source_vertex_count": len(source_xyz),
            "source_face_count": len(source_faces),
            "nha_local_bounds_m": {
                "min": points.min(axis=0).tolist(),
                "max": points.max(axis=0).tolist(),
                "extent": (points.max(axis=0) - points.min(axis=0)).tolist(),
            },
            "nha_local_triangle_area_mm2": face_area_m2 * 1e6,
            "coincident_main_component_vertex_ids": coincident_main_ids,
            "geometric_edge_matching_main_face_ids": geometric_edge_face_ancestry,
            "exact_coordinate_main_face_matches": exact_coordinate_face_matches,
            "source_face_retained_with_existing_body91_only_parent_terms": False,
            "source_face_owner_terms_reassigned_by_exact_coordinate_quotient": True,
            "source_face_parent_assignment_is_not_anatomical_admission": True,
            "route110_source_transform_applied_before_retirement": True,
            "reinstatement_scope": "restore only this source face at original index 1858 with the pinned existing Float32-exact alias IDs; preserve all vertex/normal bytes and every other source face",
            "reinstatement_operation_changes_no_mesh_coordinates": True,
        }
    row477_replacement = None
    reinstatement = None
    if isolated_face_lineage is not None:
        source_face_id = int(isolated_face_lineage["source_face_id_zero_based"])
        _require(
            np.array_equal(br[477]["faces"], nr[477]["faces"])
            and np.array_equal(br[477]["vertices6"], nr[477]["vertices6"]),
            "FJ2217 source parent row differs from pinned N27",
        )
        row477_replacement, reinstatement = reinstate_source_face_through_exact_aliases(
            owner477,
            source_face_id,
            source_faces[source_face_id],
            source_row=owner477,
        )
        final_faces = np.asarray(row477_replacement["faces"], dtype="<u4")
        used_vertices = set(map(int, np.unique(final_faces.reshape(-1))))
        exact_alias_groups = q477["exact_coordinate_alias_classes"]
        cap_owner_classes = []
        cap_ids = set(map(int, FJ2217_CAP_ALIAS_VERTEX_IDS))
        source_cap_ids = set(map(int, source_faces[source_face_id]))
        for group in exact_alias_groups:
            members = set(map(int, group))
            if members.intersection(cap_ids | source_cap_ids):
                _require(
                    np.all(w477[np.asarray(sorted(members))] == w477[min(members)]),
                    "FJ2217 cap source/active coordinate aliases have different F32 owner weights",
                )
                cap_owner_classes.append(
                    {
                        "local_vertex_ids": sorted(members),
                        "active_reinstated_face_vertex_ids": sorted(members & cap_ids),
                        "active_referenced_members": sorted(members & used_vertices),
                        "historical_source_cap_vertex_ids": sorted(members & source_cap_ids),
                        "owner_terms": w477[min(members)].astype("<f4").tolist(),
                    }
                )
        _require(
            len(cap_owner_classes) == 3
            and all(
                item["active_reinstated_face_vertex_ids"]
                and item["active_referenced_members"]
                for item in cap_owner_classes
            ),
            "FJ2217 cap does not have three referenced exact-coordinate owner classes",
        )
        isolated_face_lineage["exact_coordinate_quotient_owner_field_repair"] = {
            "method": q477["graph_method"],
            "scope": "rows475+477 prepared jointly by existing 471-475 and 473-477 collar seeds",
            "cap_coordinate_owner_classes": cap_owner_classes,
            "all_29_exact_coordinate_alias_classes_share_owner_weights": q477[
                "all_exact_coordinate_alias_members_share_f32_owner_weights"
            ],
            "active_source_and_alias_owner_terms_are_identical": True,
            "per_slot_normals_and_vertex_allocation_preserved": True,
            "no_mesh_coordinates_or_faces_changed_by_owner_repair": True,
            "remaining_motion_claim": "identical source points, per-body carrier points, coefficients, and body IDs make each exact alias class share the existing deterministic body-owner point map; no native run or physiological claim is included",
        }
        reinstatement.update(
            {
                "source_member_path": source_member_path,
                "source_face_and_nha_ancestry_exact": True,
                "original_unused_vertex_slots_retained": len(isolated477),
                "lineage": isolated_face_lineage,
            }
        )
    rigid = load_rigid(Path(rigid_path))
    _require(rigid["sha256"] == RIGID_SHA256, "NHRIGID bytes changed")
    row20, errpos, errnormal = _transfer_row(
        owner475["vertices6"], rigid["core_poses"][91], rigid["core_poses"][20]
    )
    row20_477, errpos_477, errnormal_477 = _transfer_row(
        owner477["vertices6"], rigid["core_poses"][91], rigid["core_poses"][20]
    )
    def binding(row, core):
        raw = row["binding_bytes"]
        _require(len(raw) == 36, "existing owner carrier binding count changed")
        got, *values = struct.unpack("<I8f", raw)
        _require(
            got == core and values == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            "owner carrier transform is not canonical identity",
        )
        return raw

    binds = binding(old[1], 20) + binding(old[2], 91)
    def owner_weights(weights, column):
        result = np.zeros((len(weights), 2), dtype="<f4")
        result[:, column] = 1.0
        return result

    faces477 = row477_replacement["faces"] if row477_replacement is not None else owner477["faces"]
    carrier, carrier_surface_rows, encode_proof = encode_carriers(
        [
            CarrierRow(1, row20, owner475["faces"], binds, owner_weights(w475, 0)),
            CarrierRow(2, owner475["vertices6"], owner475["faces"], binds, owner_weights(w475, 1)),
            CarrierRow(3, row20_477, faces477, binds, owner_weights(w477, 0)),
            CarrierRow(4, owner477["vertices6"], faces477, binds, owner_weights(w477, 1)),
        ],
        registration_fingerprint=int(cd["fingerprint"]),
        source_digest=bytes(cd["source_digest"]),
    )
    owner_updates = {
        475: _terms(w475, 5, (1, 2)),
        477: _terms(w477, 5, (3, 4)),
    }
    emitted_owner_proof = _verify_encoded_vessel_owner_maps(
        carrier,
        carrier_surface_rows,
        owner_updates,
        {
            475: {"weights": w475, "faces": owner475["faces"],
                  "carrier_vertices6": {20: row20, 91: owner475["vertices6"]}},
            477: {"weights": w477, "faces": faces477,
                  "carrier_vertices6": {20: row20_477, 91: owner477["vertices6"]}},
        },
        q477["exact_coordinate_alias_classes"],
    )
    return {
        "carrier_payload": carrier,
        "carrier_payload_sha256": hashlib.sha256(carrier).hexdigest(),
        "carrier_source_index": 5,
        "carrier_source_path": "assets/route110-dual-vessel-reference-carriers.nhtissue",
        "carrier_encoding_proof": encode_proof,
        "parent_surface_updates": owner_updates,
        "emitted_owner_map_proof": emitted_owner_proof,
        "route110": route,
        "route_row_replacements": {
            sid: row for sid, row in route_rows.items() if sid != 477
        },
        "row_replacements": {477: row477_replacement}
        if row477_replacement is not None
        else {},
        "local_source_face_reinstatement": reinstatement,
        "rows": {
            "475": {
                **q475,
                "stable_id": 475,
                "source_member": "FJ2216",
                "proximal_neighbor_stable_id": 471,
                "distal_neighbor_stable_id": 479,
                "body20_carrier_rest_world_max_error_m": errpos,
                "body20_carrier_rest_normal_max_error": errnormal,
                "body91_carrier_vertices_byte_exact_to_nha": True,
            },
            "477": {
                **q477,
                "stable_id": 477,
                "source_member": "FJ2217",
                "proximal_neighbor_stable_id": 473,
                "distal_neighbor_stable_id": 481,
                "baseline_carrier_payload_sha256": CARRIER_477_SHA256,
                "carrier_payload_source_index": 5,
                "carrier_rows": [3, 4],
                "body20_carrier_rest_world_max_error_m": errpos_477,
                "body20_carrier_rest_normal_max_error": errnormal_477,
                "body91_carrier_vertices_byte_exact_to_nha": True,
                "source_cap_ancestry": isolated_face_lineage,
            },
        },
        "input_pins": {
            "n27_receipt_sha256": N27_RECEIPT_SHA256,
            "n27_payload_sha256": N27_PAYLOAD_SHA256,
            "collar_report_sha256": PORT_REPORT_SHA256,
            "nhrigid_sha256": RIGID_SHA256,
            "existing_477_carrier_sha256": CARRIER_477_SHA256,
            "isa_archive_sha256": ISA_ARCHIVE_SHA256,
            "fj2217_source_member_sha256": FJ2217_OBJ_SHA256,
            "fj2217_source_face_order_matches_nha_row477": True,
            "common_map_sha256": map_info["sha256"],
            "471_exact_collar_common_coefficients_zero": True,
        },
        "scope": "Inferred source-mesh owner weights only; no anatomy, collision, poses, native, mass, force, or CVSim-state admission.",
    }



def load_route110_reference_rows(
    *,
    n27_receipt_path: Path,
    base_rows: dict[int, dict[str, Any]],
    route_npz_path: Path = ROUTE110_DIR / "candidate-110-source-rows.npz",
    route_report_path: Path = ROUTE110_DIR / "report.json",
    route_script_path: Path = ROUTE110_DIR / "run_route110.py",
    route_checkpoint_path: Path = ROUTE110_DIR / "target-prep-checkpoint.json",
) -> dict[str, Any]:
    """Load exact route110 row arrays; source preparation only, never admission."""
    receipt_path = Path(n27_receipt_path).resolve()
    _require(_sha(receipt_path) == N27_RECEIPT_SHA256, "N27 receipt hash changed")
    receipt = json.loads(receipt_path.read_text())
    payload = Path(receipt["payload"]["path"])
    _require(receipt["payload"]["sha256"] == N27_PAYLOAD_SHA256
             and _sha(payload) == N27_PAYLOAD_SHA256, "route110 input is not exact N27")
    paths = {"npz": Path(route_npz_path).resolve(),
             "report": Path(route_report_path).resolve(),
             "runner": Path(route_script_path).resolve(),
             "checkpoint": Path(route_checkpoint_path).resolve()}
    for key, path in paths.items():
        _require(path.is_file() and _sha(path) == ROUTE110_PINS[key],
                 f"route110 {key} pin changed")
    report = json.loads(paths["report"].read_text())
    checkpoint = json.loads(paths["checkpoint"].read_text())
    _require(report.get("candidate_sha256") == ROUTE110_PINS["npz"]
             and report.get("script_sha256") == ROUTE110_PINS["runner"]
             and report.get("target_preparation_checkpoint_sha256") == ROUTE110_PINS["checkpoint"]
             and report.get("status") == "ERROR_EXACT_P6_AUDIT"
             and report.get("exact_candidate_checks") == 0
             and report.get("native_or_gpu_run") is False,
             "route110 report provenance/status changed")
    _require(checkpoint.get("runner_sha256") == ROUTE110_PINS["runner"]
             and checkpoint.get("capture_pack_sha256") ==
             "a70cbdc2e42b97a38fd6afa46598f023ae705bbd36686dcebf1840399295c1fd"
             and checkpoint.get("capture_receipt_sha256") ==
             "b7847f3970694f770d4e36d1c72dd8f8bdea77840df352b9cffa84bd025ae7f2",
             "route110 checkpoint ancestry changed")
    want = {f"stable_{sid}_{part}" for sid in ROUTE110_ROWS
            for part in ("before_local_m", "after_local_m", "faces",
                         "normals_before", "normals_after")}
    want |= {f"FJ3482_{part}" for part in
             ("before_local_m", "after_local_m", "faces", "normals_after")}
    want |= {"ambient_hotspot_centers_world_m", "ambient_hotspot_displacements_m"}
    rows, records = {}, []
    with np.load(paths["npz"], allow_pickle=False) as z:
        _require(set(z.files) == want, "route110 NPZ key inventory changed")
        for sid in ROUTE110_ROWS:
            key, base = f"stable_{sid}", base_rows.get(sid)
            _require(base is not None, f"route110 source row {sid} missing")
            before = np.ascontiguousarray(z[f"{key}_before_local_m"], dtype="<f4")
            after = np.ascontiguousarray(z[f"{key}_after_local_m"], dtype="<f4")
            faces = np.ascontiguousarray(z[f"{key}_faces"], dtype="<u4")
            nb = np.ascontiguousarray(z[f"{key}_normals_before"], dtype="<f4")
            na = np.ascontiguousarray(z[f"{key}_normals_after"], dtype="<f4")
            src, srcf = np.ascontiguousarray(base["vertices6"], dtype="<f4"), np.ascontiguousarray(base["faces"], dtype="<u4")
            _require(before.shape == after.shape == nb.shape == na.shape == src[:, :3].shape
                     and faces.shape == srcf.shape and before.tobytes() == src[:, :3].tobytes()
                     and nb.tobytes() == src[:, 3:].tobytes() and faces.tobytes() == srcf.tobytes()
                     and np.isfinite(after).all() and np.isfinite(na).all()
                     and (not faces.size or (faces.min() >= 0 and faces.max() < len(after))),
                     f"route110 row {sid} does not match exact N27 F32 ancestry")
            norms = np.linalg.norm(na.astype(np.float64), axis=1)
            _require(np.isfinite(norms).all() and np.max(np.abs(norms - 1)) <= 2e-5,
                     f"route110 row {sid} normals are invalid")
            rows[sid] = {**base, "vertices6": np.column_stack((after, na)).astype("<f4"), "faces": faces}
            records.append({"stable_id": sid,
                            "source_member": receipt["provenance"]["source_id_map"][str(sid)]["source_member"],
                            "before_xyz_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
                            "after_xyz_sha256": hashlib.sha256(after.tobytes()).hexdigest(),
                            "before_normals_sha256": hashlib.sha256(nb.tobytes()).hexdigest(),
                            "after_normals_sha256": hashlib.sha256(na.tobytes()).hexdigest(),
                            "faces_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
                            "xyz_changed": before.tobytes() != after.tobytes(),
                            "normals_changed": nb.tobytes() != na.tobytes()})
        bridge = {k: np.ascontiguousarray(z[f"FJ3482_{k}"], dtype="<u4" if k == "faces" else "<f4")
                  for k in ("before_local_m", "after_local_m", "faces", "normals_after")}
    _require(bridge["before_local_m"].shape == bridge["after_local_m"].shape == bridge["normals_after"].shape
             and bridge["faces"].ndim == 2 and bridge["faces"].shape[1] == 3
             and np.isfinite(bridge["after_local_m"]).all() and np.isfinite(bridge["normals_after"]).all()
             and (not bridge["faces"].size or bridge["faces"].max() < len(bridge["after_local_m"])),
             "route110 FJ3482 arrays are invalid")
    return {"rows": rows, "bridge_rows": {525: bridge}, "records": records,
            "input_paths": {key: str(value) for key, value in paths.items()},
            "candidate_npz_sha256": ROUTE110_PINS["npz"],
            "candidate_report_sha256": ROUTE110_PINS["report"],
            "candidate_runner_sha256": ROUTE110_PINS["runner"],
            "target_preparation_checkpoint_sha256": ROUTE110_PINS["checkpoint"],
            "retained_status": report["status"], "exact_candidate_checks": 0,
            "native_or_gpu_run": False,
            "scope": "Exact per-row transforms from an offline, unadmitted route110 source candidate."}


def apply_route110_bridge_row(compiled_row: dict[str, Any], bridge: dict[str, Any]) -> dict[str, Any]:
    """Apply route110 FJ3482 XYZ/normals after exact source-compiler input match."""
    before = np.ascontiguousarray(bridge["before_local_m"], dtype="<f4")
    after = np.ascontiguousarray(bridge["after_local_m"], dtype="<f4")
    normals = np.ascontiguousarray(bridge["normals_after"], dtype="<f4")
    faces = np.ascontiguousarray(bridge["faces"], dtype="<u4")
    old = np.ascontiguousarray(compiled_row["vertices6"], dtype="<f4")
    oldf = np.ascontiguousarray(compiled_row["faces"], dtype="<u4")
    _require(old.shape == (len(before), 6) and old[:, :3].tobytes() == before.tobytes()
             and oldf.tobytes() == faces.tobytes() and after.shape == normals.shape == before.shape,
             "route110 FJ3482 differs from existing ABI5 compiler output")
    return {**compiled_row, "vertices6": np.column_stack((after, normals)).astype("<f4"), "faces": faces}
