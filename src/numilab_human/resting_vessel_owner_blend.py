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


def geodesic_owner_partition(
    vertices: np.ndarray,
    faces: np.ndarray,
    proximal_vertices: set[int],
    distal_vertices: set[int],
    *,
    isolated_distal_remainder: set[int] | None = None,
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
    graph, tri = _mesh_graph(vertices, faces)
    components = _mesh_components(graph)
    seed_components = [c for c in components if c.intersection(prox | distal)]
    _require(
        len(seed_components) == 1 and (prox | distal).issubset(seed_components[0]),
        "proximal/distal collars do not share one connected source component",
    )
    connected = seed_components[0]
    remainder_components = [c for c in components if c is not connected]
    remainder = set().union(*remainder_components) if remainder_components else set()
    allowed = set(isolated_distal_remainder or ())
    _require(
        remainder == allowed,
        "unmapped disconnected source component is not explicitly retained",
    )
    if allowed:
        _require(
            len(remainder_components) == 1
            and len(allowed) == 3
            and sum(all(int(v) in allowed for v in face) for face in tri) == 1,
            "only the declared isolated one-face remainder may retain distal ownership",
        )
    dp = _geodesic_distance(graph, prox, connected)
    dd = _geodesic_distance(graph, distal, connected)
    ids = np.asarray(sorted(connected), dtype=np.int64)
    denom = dp + dd
    _require(
        np.isfinite(denom[ids]).all() and np.all(denom[ids] > 0),
        "invalid source route extent",
    )
    t = np.clip(dp[ids] / denom[ids], 0, 1)
    q = t * t * t * (t * (t * 6 - 15) + 10)
    distal_weight = np.ones(len(graph), dtype=np.float64)
    distal_weight[ids] = q
    distal_weight[np.asarray(sorted(prox), dtype=np.int64)] = 0.0
    distal_weight[np.asarray(sorted(distal), dtype=np.int64)] = 1.0
    weights = np.column_stack((1 - distal_weight, distal_weight)).astype("<f4")
    _require(
        np.isfinite(weights).all()
        and np.all(weights >= 0)
        and np.max(np.abs(weights.astype(np.float64).sum(axis=1) - 1)) <= 1e-7,
        "invalid owner partition",
    )
    pids, dids = np.asarray(sorted(prox)), np.asarray(sorted(distal))
    _require(
        np.all(weights[pids] == [1, 0]) and np.all(weights[dids] == [0, 1]),
        "collar weights changed",
    )
    interior = np.asarray(sorted(connected - prox - distal), dtype=np.int64)
    _require(np.all(weights[interior] > 0), "interior owner coefficient underflows F32")
    return weights, {
        "method": "quintic smoothstep of source-mesh edge-geodesic distance ratio",
        "proximal_owner_core_body": 20,
        "distal_owner_core_body": 91,
        "proximal_vertex_count": len(prox),
        "distal_vertex_count": len(distal),
        "source_geodesic_extent_m": float(denom[ids].max()),
        "minimum_positive_float32_coefficient": float(weights[weights > 0].min()),
        "interior_vertex_count": len(interior),
        "isolated_distal_remainder_vertex_count": len(allowed),
        "isolated_distal_remainder_face_count": 1 if allowed else 0,
        "isolated_distal_remainder_preserves_existing_body91_owner": bool(allowed),
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
        owner475["vertices6"][:, :3], owner475["faces"], *collars[475]
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
            "classification": "unexplained_isolated_raw_source_face; scoped_reference_retirement",
            "anatomical_interface_proven": False,
            "local_retirement_proposed": True,
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
            "source_face_retained_with_existing_body91_only_parent_terms": True,
            "route110_source_transform_applied_before_retirement": True,
            "retirement_scope": "remove only this isolated source face from the NHANAT reference-row face list; preserve all 1089 original vertex slots and all other source faces byte-exact",
            "retirement_operation_changes_no_mesh_coordinates": True,
        }
    row477_replacement = None
    retirement = None
    if isolated_face_lineage is not None:
        source_face_id = int(isolated_face_lineage["source_face_id_zero_based"])
        _require(
            np.array_equal(br[477]["faces"], nr[477]["faces"])
            and np.array_equal(br[477]["vertices6"], nr[477]["vertices6"]),
            "FJ2217 retirement parent row differs from pinned N27 source",
        )
        row477_replacement, retirement = _retire_isolated_source_face(
            owner477, source_face_id, source_faces[source_face_id]
        )
        retirement.update(
            {
                "source_member_sha256": FJ2217_OBJ_SHA256,
                "source_archive_sha256": ISA_ARCHIVE_SHA256,
                "source_face_and_nha_ancestry_exact": True,
                "retained_unused_vertex_slots": len(isolated477),
                "all_other_source_faces_retained_in_original_order": True,
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
    carrier, _, encode_proof = encode_carriers(
        [
            CarrierRow(1, row20, owner475["faces"], binds, owner_weights(w475, 0)),
            CarrierRow(2, owner475["vertices6"], owner475["faces"], binds, owner_weights(w475, 1)),
            CarrierRow(3, row20_477, faces477, binds, owner_weights(w477, 0)),
            CarrierRow(4, owner477["vertices6"], faces477, binds, owner_weights(w477, 1)),
        ],
        registration_fingerprint=int(cd["fingerprint"]),
        source_digest=bytes(cd["source_digest"]),
    )
    return {
        "carrier_payload": carrier,
        "carrier_payload_sha256": hashlib.sha256(carrier).hexdigest(),
        "carrier_source_index": 5,
        "carrier_source_path": "assets/route110-dual-vessel-reference-carriers.nhtissue",
        "carrier_encoding_proof": encode_proof,
        "parent_surface_updates": {
            475: _terms(w475, 5, (1, 2)),
            477: _terms(w477, 5, (3, 4)),
        },
        "route110": route,
        "route_row_replacements": {
            sid: row for sid, row in route_rows.items() if sid != 477
        },
        "row_replacements": {477: row477_replacement}
        if row477_replacement is not None
        else {},
        "local_source_face_retirement": retirement,
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
                "isolated_source_face_ancestry": isolated_face_lineage,
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
