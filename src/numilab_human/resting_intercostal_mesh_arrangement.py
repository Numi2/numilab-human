"""Local, source-bound ABI5 composition for retained intercostal arrangement patches.

This is a narrow source owner for stable IDs 313, 315, and 317. It keeps the
original NHA vertex slots and unchanged source faces, uses only exact positive
parent maps for appended vertices, and emits the existing ABI5 row layout.
It does not qualify the resulting anatomy or its motion.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np

N27_PAYLOAD_SHA256 = "cde8705a813fe56686776c9cc038cbf06f413e1de76099d2a019d5e1b748a8d4"
N27_RECEIPT_SHA256 = "944bc88d257137f2bedd5898404c45cefd2901d719707519771ae5f4cfce3752"
REPAIR_PARENT_RECEIPT_SHA256 = "92a2215222230e4f1ad02a06583240e5c569c57807c2b9558c289c495c62fa2e"
REPAIR_PARENT_PAYLOAD_SHA256 = "50ab8248d2e0f55dd6cb8b24afc34c8844061d5841ecdf89d7daf03288816bfd"
REPAIR_REPORT_SHA256 = "9327e129cc630c24d2abe1e316fe0077568729f2600dee40359da50e910f1bd6"
FJ2217_CAP_PARENT_RECEIPT_SHA256 = "a7795cc7b6bde0eb1fade7fca0d792eb24cc9ac73a29fc3fe4b001b91ff692e5"
FJ2217_CAP_PARENT_PAYLOAD_SHA256 = "3a3693b8de2264127e9190a151dca460002ae3b8a0a09edea312bd4eed73ea42"
FJ2217_CAP_CANDIDATE_PAYLOAD_SHA256 = "f83a541b794fe9bae699e6784fe091e147d5bad284847773c803c5309027d361"
SOURCE004_INDEX_PARENT_RECEIPT_SHA256 = "98fc9e82233c4a4a6bc6af38c23e774d3a3373afec94d7cef9122108a76a238d"
SOURCE004_INDEX_PARENT_PAYLOAD_SHA256 = "9fb3ffb10e458589b2c69779bd1336bfc4dc8e704079a1e68e4b22a2b5b7ba09"
SOURCE004_INDEX_CHILD_PAYLOAD_SHA256 = "5613eb15de08a7dfe38f583f24c67c8cd4848f7f809c7e4783d27b3ccf326104"
SOURCE004_INDEX_REPORT_SHA256 = "c10a12145733fa73f20211bea75fb2c442ea7883ab275b9ef30c294c24999a1d"
SOURCE004_INDEX_DESCRIPTORS_SHA256 = "f52417e691a28c2a3419916845599fbf2fcb13db615d377a92794e9912fbe498"
N27_CAPTURE_MANIFEST_SHA256 = "3df61c8bf202e343a9124588fa94be17e9876db17d1a6ef5dd5fa615aaa7b815"
ARRANGEMENT_INPUTS_045_SHA256 = "288c31c4333333a0a186adf8dd07b76a65c176fd80ec75db84280bba1ceaab87"
GENERIC_NULL_RETIREMENT_SOURCE = Path(
    "/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/"
    "full-anatomy-delivery-audit-1378/generic-arrangement-positive-ray-bounded-004/"
    "repair_intersection_arrangement_generic_004.py"
)
GENERIC_NULL_RETIREMENT_SHA256 = "5e1801bec091668e7a068095b42c9a639e18d49e766b632a0d9162978d83fa58"
NULL_RETIREMENT_LEDGER_SCHEMA = "numi.human.intercostal-null-face-retirements.v1"
REPAIRED_IDS = frozenset((313, 315, 317))

# Pinned local replay inputs; each exception is limited to the named source operation.
INTERCOSTAL_AUDIT_ROOT = Path(
    "/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/"
    "full-anatomy-delivery-audit-1378"
)
CONTRACTION_313_EVIDENCE = {
    "captured_arrangement_lineage_path": INTERCOSTAL_AUDIT_ROOT / "intercostal-arrangement-source-repair-036-001/source-face-lineage.json",
    "captured_arrangement_lineage_sha256": "57cd651ac3efb92d77a756f0c82ac8b43aca594867714019c770bb53c25661d2",
    "pre_candidate_path": INTERCOSTAL_AUDIT_ROOT / "intercostal-arrangement-source-repair-036-001/captured-arrangement-candidate.json",
    "pre_candidate_sha256": "486584697bd60ce9d836cf497257614db76aa86f160bf65cab53cfd37c69e31e",
    "executed_candidate_path": INTERCOSTAL_AUDIT_ROOT / "intercostal-local-regularization-037-001/captured-arrangement-candidate.json",
    "executed_candidate_sha256": "6029a748052bcd62f221ba9640eb220bd61d4d69248629dbe3ab5a7f4e30a9e4",
    "executed_local_lineage_path": INTERCOSTAL_AUDIT_ROOT / "intercostal-local-regularization-037-001/local-repair-lineage.json",
    "executed_local_lineage_sha256": "b3fb4c79e7f390015aa8191cd3895bfb5cc0c09d64cb1e1cbe9f7cd44cda9c2b",
    "executed_geometry_report_path": INTERCOSTAL_AUDIT_ROOT / "intercostal-local-regularization-audit-038-001/report.json",
    "executed_geometry_report_sha256": "236ee269e89ad5db0e900829b7a6d730e6e3f413d4067e123d147c99ccebef03",
}
TIP_RETESS_317_EVIDENCE = {
    "captured_arrangement_lineage_path": INTERCOSTAL_AUDIT_ROOT / "intercostal-inner-arrangement-repair-039-001/row-317/source-face-lineage.json",
    "captured_arrangement_lineage_sha256": "bfa9fbacf68939bf31044e4642ba1e34d261503866d98d10a20965e09dbb7ce2",
    "baseline_candidate_path": INTERCOSTAL_AUDIT_ROOT / "intercostal-inner-adaptive-fan-041-001/row-317/captured-arrangement-candidate.json",
    "baseline_candidate_sha256": "db50bc5844eec45a66cacd8ab72e850f5bddbb13c1381984950159af60c4f032",
    "executed_candidate_path": INTERCOSTAL_AUDIT_ROOT / "intercostal-local-tip-retessellation-044-001/captured-arrangement-candidate.json",
    "executed_candidate_sha256": "213e39223b5c86958476c016cdbbc547330fc5e209556aba7ef99473ac339384",
    "executed_local_lineage_path": INTERCOSTAL_AUDIT_ROOT / "intercostal-local-tip-retessellation-044-001/local-repair-lineage.json",
    "executed_local_lineage_sha256": "b1a9aa067aad6df78f0eb9e70b47661d9befee1e203975153fb86651ebd1a4fc",
    "executed_geometry_report_path": INTERCOSTAL_AUDIT_ROOT / "intercostal-local-tip-retessellation-044-001/report.json",
    "executed_geometry_report_sha256": "fba64f5524ee611bdb2620c4f4dccd47cdcc72a14f3bb715bc0692cd5f409de8",
}
PRIOR_FAN_317_EVIDENCE = {
    "candidate_sha256": "db50bc5844eec45a66cacd8ab72e850f5bddbb13c1381984950159af60c4f032",
    "lineage_sha256": "d41bc2e2bec3f8d056db4600b8e441a185f672b255f76eba9f620b00181ad0e0",
    "report_path": INTERCOSTAL_AUDIT_ROOT / "intercostal-inner-adaptive-fan-041-001/row-317/report.json",
    "report_sha256": "df316413892d5d03d61e18263f4fd563529e66603d1313d2cac120ceabbb3209",
    "status": "Retained six-pair full self audit; all six pairs involve explicitly retired tip faces, and final delta audit is zero.",
}

CONTRACTION_313_SLOT = 98694
CONTRACTION_313_PATCH_SLOTS = (98694, 98695, 98696, 98697)
CONTRACTION_313_PATCH_SOURCE_FACES = (4114, 4122, 4124)
TIP_RETESS_317_PATCH_SIGNATURES = (
    (87371, (3295, 43555, 5156), 43559, (10349, 87170, 87173),
     (10346, 10348, 10349), ((5156, 3295), (3295, 43555), (43555, 5156))),
    (87372, (32479, 43499, 34170), 43569, (68539, 87314, 87339),
     (66923, 68534, 68539), ((34170, 32479), (32479, 43499), (43499, 34170))),
)


def _load_json_with_sha(path: Path, expected_sha256: str, label: str) -> dict:
    path = Path(path).resolve()
    _require(path.is_file() and sha_file(path) == expected_sha256,
             label + " bytes differ from the pinned source")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(label + " is not valid pinned JSON") from error
    _require(isinstance(value, dict), label + " root is not an object")
    return value


def _candidate_row(candidate: dict, stable_id: int) -> dict:
    row_key = str(stable_id)
    try:
        return {
            "faces": candidate["faces"][row_key],
            "face_ancestry": candidate["face_ancestry"][row_key],
            "points_exact": candidate["points_exact"][row_key],
            "parent_maps": candidate["parent_maps"][row_key],
        }
    except (KeyError, TypeError) as error:
        raise ValueError(f"pinned candidate lacks row {stable_id}") from error


def _validate_contraction_face_replay(
    *, pre_face, post_face, current_face, pre_ancestry, post_ancestry,
    current_ancestry, operation, sidecar_entry, pre_row, post_row, current_row,
) -> dict:
    """Prove only the pinned 313 face-slot substitution from local execution."""
    expected_operation = {
        "contracted_vertices": [49441, 49442],
        "max_displacement_m": 0.00019755848423334504,
        "target_vertex": 2059,
    }
    expected_ancestry = {
        "input_index": 4114, "origin_kind": 1, "parent_face_index": 4114,
        "parent_stable_id": 313, "partition_role": 0,
    }
    expected_sidecar = {
        "new_face_slot": CONTRACTION_313_SLOT,
        "paired_witness_faces": [4122, 4124],
        "source_ancestry": {
            "input_index": 4114, "origin_kind": 0, "parent_face_index": 4114,
            "parent_stable_id": 313, "partition_role": 0,
        },
        "source_face_slot": 4114,
    }
    _require(operation == expected_operation,
             "313 contraction differs from the pinned local operation")
    _require(sidecar_entry == expected_sidecar,
             "313 contraction slot lost its frozen source-face/witness ancestry")
    _require(pre_face == [2052, 49441, 2056]
             and post_face == [2052, 2059, 2056]
             and current_face == post_face,
             "313 contraction face does not replay the one pinned vertex substitution")
    _require(pre_ancestry == post_ancestry == current_ancestry == expected_ancestry,
             "313 contraction face ancestry differs from the pinned source facet")
    vertices = (2052, 49441, 49442, 2056, 2059)
    for vertex in vertices:
        _require(pre_row["points_exact"][vertex] == post_row["points_exact"][vertex]
                 == current_row["points_exact"][vertex],
                 f"313 contraction replay changed exact point {vertex}")
        _require(pre_row["parent_maps"][vertex] == post_row["parent_maps"][vertex]
                 == current_row["parent_maps"][vertex],
                 f"313 contraction replay changed exact parent terms at {vertex}")
    return {
        "kind": "pinned_313_local_contraction_face_replay",
        "face_slot": CONTRACTION_313_SLOT,
        "source_face_slot": 4114,
        "paired_witness_faces": [4122, 4124],
        "pre_face": pre_face,
        "post_face": post_face,
        "contracted_vertex": 49441,
        "target_vertex": 2059,
        "touched_point_terms_sha256": _canonical_json_sha256({
            str(vertex): {
                "point": current_row["points_exact"][vertex],
                "parent_terms": current_row["parent_maps"][vertex],
            } for vertex in vertices
        }),
        "executed_local_contraction_only_for_this_support_exception": True,
    }






def _load_source_face_sidecar_by_slot(lineage: dict, stable_id: int) -> dict[int, dict]:
    """Load the captured local-arrangement sidecar through its recorded SHA pin."""
    path_value = lineage.get("captured_arrangement_lineage_path")
    digest = lineage.get("captured_arrangement_lineage_sha256")
    if path_value is None and digest is None:
        return {}
    _require(isinstance(path_value, str) and isinstance(digest, str),
             f"{stable_id} captured source-face sidecar path/hash pair is incomplete")
    captured = _load_json_with_sha(Path(path_value), digest,
                                   f"{stable_id} captured source-face sidecar")
    sidecar = captured.get("sidecar")
    _require(isinstance(sidecar, list),
             f"{stable_id} captured source-face sidecar is missing its rows")
    by_slot = {}
    for entry in sidecar:
        _require(
            isinstance(entry, dict)
            and type(entry.get("new_face_slot")) is int
            and type(entry.get("source_face_slot")) is int
            and entry["new_face_slot"] >= 0
            and entry["source_face_slot"] >= 0
            and isinstance(entry.get("paired_witness_faces"), list)
            and len(entry["paired_witness_faces"]) > 0
            and all(type(value) is int and value >= 0
                    for value in entry["paired_witness_faces"])
            and len(set(entry["paired_witness_faces"])) == len(entry["paired_witness_faces"])
            and entry["source_face_slot"] not in entry["paired_witness_faces"]
            and isinstance(entry.get("source_ancestry"), dict),
            f"{stable_id} captured source-face sidecar has a malformed entry",
        )
        slot = int(entry["new_face_slot"])
        _require(slot not in by_slot,
                 f"{stable_id} captured source-face sidecar repeats output slot {slot}")
        by_slot[slot] = entry
    return by_slot


def _sidecar_declared_source_patch_faces(
    *, stable_id: int, face_slot: int, origin: dict, sidecar_entry: dict,
    source_face_count: int,
) -> tuple[int, ...]:
    expected_source_ancestry = {
        key: int(origin[key])
        for key in ("input_index", "parent_face_index", "parent_stable_id", "partition_role")
    }
    expected_source_ancestry["origin_kind"] = 0
    source_face_slot = int(origin["parent_face_index"])
    witnesses = list(map(int, sidecar_entry.get("paired_witness_faces", [])))
    _require(
        stable_id == int(origin["parent_stable_id"])
        and sidecar_entry.get("new_face_slot") == face_slot
        and sidecar_entry.get("source_face_slot") == source_face_slot
        and sidecar_entry.get("source_ancestry") == expected_source_ancestry
        and bool(witnesses)
        and len(set(witnesses)) == len(witnesses)
        and source_face_slot not in witnesses
        and all(0 <= value < source_face_count for value in (source_face_slot, *witnesses)),
        f"{stable_id} face {face_slot} sidecar does not bind its exact source ancestry",
    )
    return (source_face_slot, *witnesses)


def _validate_sidecar_source_patch_support(
    *, stable_id: int, face_slot: int, origin: dict,
    sidecar_entry: dict, source_vertices6: np.ndarray, source_faces: np.ndarray,
    support_vertex_ids: set[int],
) -> dict:
    """Bind a remeshed face's positive source support to its exact sidecar face patch."""
    expected_source_ancestry = {
        key: int(origin[key])
        for key in ("input_index", "parent_face_index", "parent_stable_id", "partition_role")
    }
    expected_source_ancestry["origin_kind"] = 0
    source_face_slot = int(origin["parent_face_index"])
    witnesses = list(map(int, sidecar_entry["paired_witness_faces"]))
    _require(
        stable_id == int(origin["parent_stable_id"])
        and sidecar_entry.get("new_face_slot") == face_slot
        and sidecar_entry.get("source_face_slot") == source_face_slot
        and sidecar_entry.get("source_ancestry") == expected_source_ancestry
        and bool(witnesses)
        and len(set(witnesses)) == len(witnesses)
        and source_face_slot not in witnesses,
        f"{stable_id} face {face_slot} sidecar does not bind its exact source ancestry",
    )
    faces = np.ascontiguousarray(source_faces, dtype="<u4")
    patch_face_ids = (source_face_slot, *witnesses)
    _require(all(0 <= value < len(faces) for value in patch_face_ids),
             f"{stable_id} face {face_slot} sidecar patch is outside the pinned source row")
    patch_corner_ids = {
        int(vertex) for patch_face_id in patch_face_ids for vertex in faces[patch_face_id]
    }
    patch_support = _validate_within_declared_patch_support(
        stable_id=stable_id,
        face_slot=face_slot,
        expected_parent_face_index=source_face_slot,
        source_vertices6=source_vertices6,
        source_faces=faces,
        source_patch_face_ids=patch_face_ids,
        support_vertex_ids=support_vertex_ids,
        operation_kind="pinned_source_face_sidecar",
        operation_sha256=_canonical_json_sha256(sidecar_entry),
    )
    aliases = patch_support["support_source_patch_corner_aliases"]
    return {
        "kind": "pinned_source_face_sidecar_patch_support",
        "stable_id": stable_id,
        "face_slot": face_slot,
        "source_face_slot_028_namespace": source_face_slot,
        "paired_witness_faces_028_namespace": witnesses,
        "source_patch_face_indices_028_namespace": list(patch_face_ids),
        "source_patch_corner_ids": sorted(patch_corner_ids),
        "support_vertex_ids": sorted(support_vertex_ids),
        "support_source_patch_corner_aliases": aliases,
        "sidecar_entry_sha256": _canonical_json_sha256(sidecar_entry),
        "source_patch_face_arrays_sha256": sha_bytes(
            np.ascontiguousarray(faces[list(patch_face_ids)], dtype="<u4").tobytes()),
        "support_is_exact_float32_coordinate_quotient_of_declared_sidecar_patch": patch_support[
            "support_is_exact_float32_coordinate_quotient_of_declared_patch"],
    }



def _exact_source_face_coordinate_aliases(
    *, source_vertices6: np.ndarray, source_vertex_ids: set[int],
    source_face_corner_ids: set[int],
) -> dict | None:
    """Return an exact Float32 coordinate quotient proof, or None when support escapes."""
    vertices = np.ascontiguousarray(source_vertices6, dtype="<f4")
    if (vertices.ndim != 2 or vertices.shape[1] != 6
            or any(not 0 <= index < len(vertices) for index in source_vertex_ids)
            or any(not 0 <= index < len(vertices) for index in source_face_corner_ids)):
        return None
    corners_by_xyz: dict[bytes, list[int]] = {}
    for corner in sorted(source_face_corner_ids):
        corners_by_xyz.setdefault(vertices[corner, :3].tobytes(), []).append(corner)
    aliases = {}
    for source_vertex in sorted(source_vertex_ids):
        matches = corners_by_xyz.get(vertices[source_vertex, :3].tobytes(), [])
        if not matches:
            return None
        aliases[str(source_vertex)] = matches
    return {
        "support_vertex_ids": sorted(source_vertex_ids),
        "source_face_corner_coordinate_aliases": aliases,
        "support_is_exact_float32_coordinate_quotient_of_parent_face": True,
        "source_support_coordinate_sha256": sha_bytes(
            np.ascontiguousarray(vertices[sorted(source_vertex_ids), :3], dtype="<f4").tobytes()),
    }



def _validate_within_declared_patch_support(
    *, stable_id: int, face_slot: int, expected_parent_face_index: int,
    source_vertices6: np.ndarray, source_faces: np.ndarray,
    source_patch_face_ids, support_vertex_ids: set[int],
    operation_kind: str, operation_sha256: str,
) -> dict:
    """Require exact source ancestry and positive parents inside one declared patch."""
    faces = np.ascontiguousarray(source_faces, dtype="<u4")
    patch_ids = [int(value) for value in source_patch_face_ids]
    _require(
        bool(patch_ids)
        and len(patch_ids) == len(set(patch_ids))
        and type(expected_parent_face_index) is int
        and expected_parent_face_index in patch_ids
        and all(0 <= value < len(faces) for value in patch_ids)
        and isinstance(operation_kind, str) and operation_kind
        and isinstance(operation_sha256, str) and len(operation_sha256) == 64,
        f"{stable_id} face {face_slot} declared source patch is malformed",
    )
    patch_corners = {
        int(vertex) for patch_face_id in patch_ids for vertex in faces[patch_face_id]
    }
    aliases = _exact_source_face_coordinate_aliases(
        source_vertices6=source_vertices6,
        source_vertex_ids=set(map(int, support_vertex_ids)),
        source_face_corner_ids=patch_corners,
    )
    escape_message = (
        "positive parent support escapes its exact sidecar source patch"
        if operation_kind == "pinned_source_face_sidecar"
        else "positive parent support escapes its declared source patch"
    )
    _require(aliases is not None,
             f"{stable_id} face {face_slot} {escape_message}")
    return {
        "kind": "source_bound_declared_patch_support",
        "operation_kind": operation_kind,
        "operation_sha256": operation_sha256,
        "stable_id": stable_id,
        "face_slot": face_slot,
        "expected_parent_face_index_028_namespace": expected_parent_face_index,
        "source_patch_face_indices_028_namespace": patch_ids,
        "source_patch_face_arrays_sha256": sha_bytes(
            np.ascontiguousarray(faces[patch_ids], dtype="<u4").tobytes()),
        "source_patch_corner_ids": sorted(patch_corners),
        "support_vertex_ids": sorted(map(int, support_vertex_ids)),
        "support_source_patch_corner_aliases": aliases[
            "source_face_corner_coordinate_aliases"],
        "support_is_exact_float32_coordinate_quotient_of_declared_patch": True,
    }


def _validate_vertex_fan_split_face_support(
    *, stable_id: int, face_slot: int, face_qvertex_ids, origin: dict,
    operation: dict, operation_sha256: str, source_vertices6: np.ndarray,
    source_faces: np.ndarray, source_patch_face_ids, support_vertex_ids: set[int],
    raw_parent_maps: list,
) -> dict:
    """Bind one fan-modified face to its exact operation, source ancestry, and local patch."""
    face_ids = [int(value) for value in face_qvertex_ids]
    operation_faces = operation.get("faces")
    center = operation.get("new_vertex")
    neighbors = operation.get("positive_blend_neighbors")
    patch_ids = list(map(int, source_patch_face_ids))
    _require(
        isinstance(operation_faces, list)
        and all(type(value) is int for value in operation_faces)
        and len(operation_faces) == len(set(operation_faces))
        and face_slot in operation_faces
        and type(center) is int
        and isinstance(neighbors, list)
        and all(type(value) is int for value in neighbors)
        and len(neighbors) == len(set(neighbors))
        and center not in neighbors
        and 0 <= center < len(raw_parent_maps)
        and all(0 <= value < len(raw_parent_maps) for value in neighbors)
        and len(face_ids) == 3 and len(set(face_ids)) == 3
        and set(face_ids).issubset({center, *neighbors})
        and raw_parent_maps[center] == operation.get("positive_parent_terms"),
        f"{stable_id} face {face_slot} does not replay its exact fan-split operation",
    )
    parent_face = origin.get("parent_face_index")
    _require(
        origin.get("parent_stable_id") == stable_id
        and origin.get("partition_role") == 0
        and type(parent_face) is int
        and parent_face in patch_ids,
        f"{stable_id} face {face_slot} ancestry is outside its declared fan patch",
    )
    source_parent_faces = origin.get("parent_source_face_indices", [parent_face])
    _require(
        isinstance(source_parent_faces, list) and source_parent_faces
        and all(type(value) is int and value in patch_ids for value in source_parent_faces),
        f"{stable_id} face {face_slot} source ancestry escapes its declared fan patch",
    )
    patch_proof = _validate_within_declared_patch_support(
        stable_id=stable_id, face_slot=face_slot,
        expected_parent_face_index=parent_face,
        source_vertices6=source_vertices6, source_faces=source_faces,
        source_patch_face_ids=patch_ids,
        support_vertex_ids=support_vertex_ids,
        operation_kind="pinned_vertex_fan_split",
        operation_sha256=operation_sha256,
    )
    return {
        **patch_proof,
        "kind": "pinned_vertex_fan_split_declared_patch_support",
        "new_vertex": center,
        "positive_blend_neighbors": neighbors,
        "face_qvertex_ids": face_ids,
        "positive_parent_terms_sha256": _canonical_json_sha256(
            operation.get("positive_parent_terms")),
        "fan_operation_face_slots_sha256": sha_bytes(
            np.asarray(operation_faces, dtype="<u8").tobytes()),
        "source_face_ancestry_is_inside_declared_patch": True,
    }


def _validate_313_contraction_source_patch_support(
    *, source_vertices6: np.ndarray, source_faces: np.ndarray,
    support_vertex_ids: set[int], proof: dict,
) -> dict:
    """Bind 313 positive parent support to the exact source faces in the frozen contraction sidecar."""
    _require(proof.get("kind") == "pinned_313_local_contraction_patch_replay"
             and proof.get("source_patch_face_indices_028_namespace")
             == list(CONTRACTION_313_PATCH_SOURCE_FACES),
             "313 contraction support lacks its exact declared 028 source patch")
    faces = np.ascontiguousarray(source_faces, dtype="<u4")
    vertices = np.ascontiguousarray(source_vertices6, dtype="<f4")
    patch_ids = tuple(map(int, proof["source_patch_face_indices_028_namespace"]))
    _require(vertices.ndim == 2 and vertices.shape[1] == 6
             and faces.ndim == 2 and faces.shape[1] == 3
             and all(0 <= face_id < len(faces) for face_id in patch_ids),
             "313 contraction source patch is outside the exact 028 row")
    corner_ids = sorted({int(vertex) for face_id in patch_ids for vertex in faces[face_id]})
    corner_keys = {}
    for vertex in corner_ids:
        corner_keys.setdefault(vertices[vertex, :3].tobytes(), []).append(vertex)
    aliases = {}
    for vertex in sorted(support_vertex_ids):
        _require(0 <= vertex < len(vertices),
                 "313 contraction positive parent support is outside the exact 028 row")
        key = vertices[vertex, :3].tobytes()
        _require(key in corner_keys,
                 "313 contraction positive parent support escapes its exact sidecar source patch")
        aliases[str(vertex)] = corner_keys[key]
    return {
        "support_vertex_ids": sorted(support_vertex_ids),
        "support_source_patch_corner_aliases": aliases,
        "source_patch_vertex_ids": corner_ids,
        "source_patch_face_arrays_sha256": sha_bytes(
            np.ascontiguousarray(faces[list(patch_ids)], dtype="<u4").tobytes()),
        "source_patch_exact_float32_coordinate_quotient_sha256": sha_bytes(
            np.ascontiguousarray(vertices[corner_ids, :3], dtype="<f4").tobytes()),
        "support_is_exact_float32_coordinate_quotient_of_declared_source_patch": True,
    }



def _validate_contraction_face_group_replay(
    *, pre_row: dict, post_row: dict, current_row: dict,
    pre_ancestry, post_ancestry, current_ancestry,
    operation: dict, sidecar_entries: list[dict], local_lineage: dict,
) -> dict[int, dict]:
    """Replay the complete four-slot 313 contraction patch, including retired twins."""
    expected_operation = {
        "contracted_vertices": [49441, 49442],
        "max_displacement_m": 0.00019755848423334504,
        "target_vertex": 2059,
    }
    generated_ancestry = {
        "input_index": 4114, "origin_kind": 1, "parent_face_index": 4114,
        "parent_stable_id": 313, "partition_role": 0,
    }
    source_ancestry = {**generated_ancestry, "origin_kind": 0}
    specs = (
        (98694, [2052, 49441, 2056], [2052, 2059, 2056]),
        (98695, [2052, 49442, 49441], None),
        (98696, [2055, 49442, 2052], [2055, 2059, 2052]),
        (98697, [2055, 49441, 49442], None),
    )
    expected_sidecar = [
        {
            "new_face_slot": slot,
            "paired_witness_faces": [4122, 4124],
            "source_ancestry": source_ancestry,
            "source_face_slot": 4114,
        }
        for slot, _pre_face, _post_face in specs
    ]
    _require(operation == expected_operation,
             "313 contraction patch differs from the pinned local operation")
    _require(sidecar_entries == expected_sidecar,
             "313 contraction patch slots or witness ancestry differ from the frozen source lineage")
    for slot, pre_face, post_face in specs:
        _require(pre_row["faces"][slot] == pre_face
                 and post_row["faces"][slot] == post_face
                 and current_row["faces"][slot] == post_face,
                 f"313 contraction patch face slot {slot} differs from its frozen execution")
        _require(pre_ancestry[slot] == post_ancestry[slot] == current_ancestry[slot]
                 == generated_ancestry,
                 f"313 contraction patch ancestry differs at face slot {slot}")
    retired = {
        int(entry["slot"]): entry
        for entry in local_lineage.get("retired", [])
        if isinstance(entry, dict) and type(entry.get("slot")) is int
        and entry["slot"] in (98695, 98697)
    }
    expected_retired = {
        98695: {"slot": 98695, "face": [2052, 2059, 2059], "kind": "exact_zero_area",
                "ancestry": generated_ancestry},
        98697: {"slot": 98697, "face": [2055, 2059, 2059], "kind": "exact_zero_area",
                "ancestry": generated_ancestry},
    }
    _require(retired == expected_retired,
             "313 contraction patch retired slots differ from the exact executed zero-area faces")

    touched_vertices = (2052, 2055, 2056, 2059, 49441, 49442)
    for vertex in touched_vertices:
        _require(pre_row["points_exact"][vertex]
                 == post_row["points_exact"][vertex]
                 == current_row["points_exact"][vertex]
                 and pre_row["parent_maps"][vertex]
                 == post_row["parent_maps"][vertex]
                 == current_row["parent_maps"][vertex],
                 f"313 contraction patch changed exact point or positive parent terms at {vertex}")
    map_hash = _canonical_json_sha256({
        str(vertex): {
            "point": current_row["points_exact"][vertex],
            "parent_terms": current_row["parent_maps"][vertex],
        }
        for vertex in touched_vertices
    })
    group_proof = {
        "kind": "pinned_313_local_contraction_patch_replay",
        "source_face_slot": 4114,
        "source_patch_face_indices_028_namespace": list(CONTRACTION_313_PATCH_SOURCE_FACES),
        "paired_witness_faces": [4122, 4124],
        "patch_face_slots": list(CONTRACTION_313_PATCH_SLOTS),
        "live_face_slots": [98694, 98696],
        "retired_zero_area_face_slots": [98695, 98697],
        "touched_point_terms_sha256": map_hash,
        "sidecar_sha256": _canonical_json_sha256(sidecar_entries),
        "executed_local_contraction_only_for_this_support_exception": True,
    }
    return {
        slot: {
            **group_proof,
            "face_slot": slot,
            "post_face": post_face,
        }
        for slot, _pre_face, post_face in specs
        if post_face is not None
    }



def _validate_current_candidate_replay(*, executed_row: dict, current_row: dict,
                                       lineage: dict, stable_id: int,
                                       fan_split_operations: list[dict] | None = None) -> dict:
    """Bind composed full_candidate geometry to execution, allowing only pinned ancestry reclassification."""
    _require(current_row["faces"] == executed_row["faces"],
             f"{stable_id} composer candidate faces differ from executed geometry")
    _require(current_row["points_exact"] == executed_row["points_exact"]
             and current_row["parent_maps"] == executed_row["parent_maps"],
             f"{stable_id} composer candidate changed executed point/map geometry")
    expected_ancestry = [dict(value) for value in executed_row["face_ancestry"]]
    slots = lineage.get("modified_original_facet_slots_reclassified_as_generated_source_children", [])
    _require(isinstance(slots, list) and all(type(slot) is int for slot in slots)
             and len(set(slots)) == len(slots),
             f"{stable_id} source reclassification slots are malformed")
    repair_operations = lineage.get("repair_operations", {})
    fan_splits = (
        repair_operations.get("vertex_fan_splits", [])
        if fan_split_operations is None else fan_split_operations
    )
    _require(isinstance(fan_splits, list),
             f"{stable_id} fan-split ancestry is malformed")
    fan_face_slots = set()
    for operation in fan_splits:
        faces = operation.get("faces") if isinstance(operation, dict) else None
        _require(isinstance(faces, list) and all(type(face) is int for face in faces),
                 f"{stable_id} fan-split operation has malformed face slots")
        fan_face_slots.update(faces)
    _require(set(slots).issubset(fan_face_slots),
             f"{stable_id} reclassified facets are not covered by pinned fan-split operations")
    for slot in slots:
        _require(0 <= slot < len(expected_ancestry)
                 and expected_ancestry[slot].get("origin_kind") == 0,
                 f"{stable_id} reclassified slot {slot} is not an executed original source facet")
        parent_face = expected_ancestry[slot].get("parent_face_index")
        _require(type(parent_face) is int and parent_face >= 0,
                 f"{stable_id} reclassified slot {slot} lacks one exact source parent")
        expected_ancestry[slot].update({
            "origin_kind": 1,
            "local_vertex_fan_split": True,
            "parent_source_face_indices": [parent_face],
        })
    _require(current_row["face_ancestry"] == expected_ancestry,
             f"{stable_id} composer ancestry changes outside the pinned source-child reclassification")
    return {
        "typed_identity": [51010, stable_id],
        "executed_geometry_arrays_match": True,
        "ancestry_reclassified_source_child_slots": len(slots),
        "ancestry_reclassification_slots_sha256": sha_bytes(
            np.asarray(slots, dtype="<u8").tobytes()),
        "no_point_or_parent_map_geometry_changed": True,
    }


def _directed_edges(face) -> list[tuple[int, int]]:
    return [(int(face[index]), int(face[(index + 1) % 3])) for index in range(3)]


def _same_oriented_cycle(first, second) -> bool:
    first, second = list(first), list(second)
    return len(first) == len(second) and any(
        first == second[offset:] + second[:offset] for offset in range(len(second))
    )


def _validate_tip_retessellation_transition(
    *, baseline_faces, baseline_ancestry, executed_faces, executed_ancestry,
    baseline_points, executed_points, baseline_maps, executed_maps,
    operations: dict, expected_signatures=TIP_RETESS_317_PATCH_SIGNATURES,
) -> dict[int, dict]:
    """Replay the declared three-face tip caps with a fixed, unchanged outer ring."""
    patches = operations.get("patches")
    retired_records = operations.get("retired")
    _require(isinstance(patches, list) and len(patches) == len(expected_signatures)
             and isinstance(retired_records, list),
             "tip-retessellation patch/retirement ledger is incomplete")
    _require(baseline_points == executed_points and baseline_maps == executed_maps,
             "317 local tip retessellation changed candidate coordinates or positive parent maps")
    expected_faces = list(baseline_faces)
    expected_ancestry = [dict(item) for item in baseline_ancestry]
    _require(len(executed_faces) == len(baseline_faces) + len(patches)
             and len(executed_ancestry) == len(executed_faces),
             "317 cap faces are not appended after the pinned baseline")
    retired_by_slot = {int(item["slot"]): item for item in retired_records}
    expected_retired_slots = set()
    proofs = {}
    for index, (patch, signature) in enumerate(zip(patches, expected_signatures)):
        new_slot, new_face, old_tip, retired_slots, parent_source_faces, boundary_edges = signature
        _require(
            (patch.get("new_slot"), tuple(patch.get("new_face", ())), patch.get("old_tip"),
             tuple(patch.get("retired_slots", ())),
             tuple(patch.get("source_face_ancestry", {}).get("parent_source_face_indices", ())),
             tuple(tuple(edge) for edge in patch.get("unchanged_oriented_boundary_edges", ())))
            == (new_slot, new_face, old_tip, retired_slots, parent_source_faces, boundary_edges),
            f"317 tip-retessellation operation {index} differs from its frozen local patch",
        )
        _require(new_slot == len(baseline_faces) + index,
                 "317 tip cap is not appended at its pinned output face slot")
        ring = set(new_face)
        _require(len(ring) == 3 and old_tip not in ring
                 and tuple(patch["source_face_ancestry"].get(
                     "parent_source_face_indices", ())) == parent_source_faces,
                 "317 cap ring or source-face ancestry is malformed")
        operation_retired = set(map(int, patch["retired_slots"]))
        _require(not expected_retired_slots.intersection(operation_retired),
                 "317 tip patches repeat a retired source face slot")
        expected_retired_slots.update(operation_retired)
        retired_faces = []
        for slot in retired_slots:
            _require(slot in retired_by_slot,
                     f"317 cap retired slot {slot} lacks its executed face witness")
            retired = retired_by_slot[slot]
            _require(
                0 <= slot < len(baseline_faces)
                and baseline_faces[slot] == retired.get("face")
                and baseline_ancestry[slot] == retired.get("ancestry"),
                f"317 cap retired face {slot} differs from baseline source ancestry",
            )
            expected_faces[slot] = None
            retired_faces.append(retired["face"])
        # The three removed oriented triangles must expose exactly the declared
        # unchanged ring; internal edges cancel only as opposite directed pairs.
        edge_counts: dict[tuple[int, int], int] = {}
        for face in retired_faces:
            _require(len(face) == 3 and old_tip in face,
                     "317 retired fan does not share its declared local tip")
            for edge in _directed_edges(face):
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        boundary = []
        visited = set()
        for edge in edge_counts:
            undirected = tuple(sorted(edge))
            if undirected in visited:
                continue
            visited.add(undirected)
            forward = edge_counts.get(edge, 0)
            reverse = edge_counts.get((edge[1], edge[0]), 0)
            _require(forward <= 1 and reverse <= 1
                     and (forward == 1 or reverse == 1),
                     "317 retired fan has duplicate or malformed directed edges")
            if forward != reverse:
                boundary.append(edge if forward else (edge[1], edge[0]))
        _require(set(boundary) == set(boundary_edges)
                 and _same_oriented_cycle(boundary, boundary_edges),
                 "317 retired source fan does not bound the exact oriented replacement cap")
        _require(tuple(executed_faces[new_slot]) == new_face
                 and executed_ancestry[new_slot] == patch["source_face_ancestry"],
                 "317 emitted cap face or ancestry differs from the pinned operation")
        expected_faces.append(list(new_face))
        expected_ancestry.append(dict(patch["source_face_ancestry"]))
        ring_hash = _canonical_json_sha256({
            str(vertex): {
                "point": baseline_points[vertex],
                "parent_terms": baseline_maps[vertex],
            } for vertex in sorted(ring)
        })
        proofs[new_slot] = {
            "kind": "pinned_317_three_face_tip_retessellation",
            "new_face_slot": new_slot,
            "new_face": list(new_face),
            "retired_face_slots": list(retired_slots),
            "parent_source_face_indices": list(parent_source_faces),
            "fixed_ring_vertex_ids": sorted(ring),
            "fixed_ring_point_and_parent_terms_sha256": ring_hash,
            "unchanged_oriented_boundary_edges": [list(edge) for edge in boundary_edges],
            "signed_geometric_volume_change_m3": patch.get("signed_geometric_volume_change_m3"),
            "source_point_arrays_unchanged": True,
        }
    _require(expected_retired_slots == set(retired_by_slot),
             "317 tip source retirement ledger contains undeclared or missing slots")
    _require(expected_faces == executed_faces and expected_ancestry == executed_ancestry,
             "317 executed faces/ancestry differ outside the two declared tip retessellations")
    return proofs


def _validate_tip_retessellation_support(
    *, patch: dict, raw_slot: int, face, origin: dict, parent_maps,
    raw_points: list, raw_parent_maps: list, source_faces: np.ndarray,
    source_vertex_count: int, replay_proof: dict, stable_id: int,
) -> dict:
    """Bind cap corner maps to the exact frozen fixed-ring retessellation replay."""
    face_ids = list(map(int, face))
    _require(stable_id == 317 and raw_slot == patch["new_slot"]
             and face_ids == patch["new_face"],
             "tip-retessellation support exception is outside its pinned cap face")
    _require(origin == patch["source_face_ancestry"],
             "tip-retessellation cap ancestry differs from its executed operation")
    source_face_ids = patch["source_face_ancestry"]["parent_source_face_indices"]
    _require(len(source_face_ids) == 3 and len(set(source_face_ids)) == 3
             and all(0 <= face_id < len(source_faces) for face_id in source_face_ids),
             "tip-retessellation source-face support is malformed")
    _require(
        isinstance(replay_proof, dict)
        and replay_proof.get("kind") == "pinned_317_three_face_tip_retessellation"
        and replay_proof.get("new_face_slot") == raw_slot
        and replay_proof.get("new_face") == face_ids
        and replay_proof.get("parent_source_face_indices") == source_face_ids
        and replay_proof.get("retired_face_slots") == patch["retired_slots"]
        and replay_proof.get("fixed_ring_vertex_ids") == sorted(face_ids)
        and replay_proof.get("source_point_arrays_unchanged") is True,
        "317 cap lacks its exact frozen output/retirement/fixed-ring replay proof",
    )
    _require(
        len(raw_points) == len(raw_parent_maps) == len(parent_maps)
        and all(0 <= qvertex < len(raw_points) for qvertex in face_ids),
        "317 cap ring point/map arrays are incomplete",
    )
    ring_payload = {
        str(qvertex): {
            "point": raw_points[qvertex],
            "parent_terms": raw_parent_maps[qvertex],
        }
        for qvertex in sorted(face_ids)
    }
    ring_hash = _canonical_json_sha256(ring_payload)
    _require(
        ring_hash == replay_proof.get("fixed_ring_point_and_parent_terms_sha256"),
        "317 cap ring point or positive-parent terms differ from the frozen retessellation",
    )
    support = set()
    parent_term_payload = {}
    for qvertex in sorted(face_ids):
        terms = parent_maps[qvertex]
        _require(bool(terms), f"317 cap ring vertex {qvertex} has no positive parent terms")
        weight_sum = Fraction(0)
        for source_vertex, weight in terms:
            _require(
                type(source_vertex) is int and 0 <= source_vertex < source_vertex_count
                and isinstance(weight, Fraction) and weight > 0,
                f"317 cap ring vertex {qvertex} has an invalid positive parent term",
            )
            support.add(source_vertex)
            weight_sum += weight
        _require(weight_sum == 1, f"317 cap ring vertex {qvertex} parent weights do not sum to one")
        parent_term_payload[str(qvertex)] = [
            [source_vertex, str(weight.numerator), str(weight.denominator)]
            for source_vertex, weight in terms
        ]
    source_patch_corners = {
        int(vertex) for face_id in source_face_ids for vertex in source_faces[face_id]
    }
    return {
        "source_face_indices_028_namespace": list(source_face_ids),
        "source_face_arrays_sha256": sha_bytes(np.ascontiguousarray(
            source_faces[list(source_face_ids)], dtype="<u4").tobytes()),
        "declared_source_patch_corner_ids": sorted(source_patch_corners),
        "fixed_ring_point_and_parent_terms_sha256": ring_hash,
        "fixed_ring_parent_term_payload_sha256": _canonical_json_sha256(parent_term_payload),
        "positive_parent_support_ids": sorted(support),
        "positive_parent_support_sha256": sha_bytes(
            np.asarray(sorted(support), dtype="<u4").tobytes()),
        "positive_parent_support_is_exact_frozen_retessellation_ring": True,
        "positive_parent_support_need_not_be_single_triangle_corner_subset": True,
        "retessellation_replay_proof_kind": replay_proof["kind"],
    }


def _load_313_contraction_replay(arrangement: dict, current_candidate: dict) -> dict[int, dict]:
    lineage = arrangement.get("lineage", {})
    for key in (
        "captured_arrangement_lineage_path", "captured_arrangement_lineage_sha256",
        "executed_candidate_path", "executed_candidate_sha256",
        "executed_local_lineage_path", "executed_local_lineage_sha256",
        "executed_geometry_report_path", "executed_geometry_report_sha256",
    ):
        expected = CONTRACTION_313_EVIDENCE[key]
        if key.endswith("_path"):
            _require(Path(lineage.get(key, "")).resolve() == Path(expected).resolve(),
                     "313 contraction source path differs from the frozen execution")
        else:
            _require(lineage.get(key) == expected,
                     "313 contraction source hash differs from the frozen execution")
    pre = _load_json_with_sha(
        CONTRACTION_313_EVIDENCE["pre_candidate_path"],
        CONTRACTION_313_EVIDENCE["pre_candidate_sha256"], "313 pre-contraction candidate")
    post = _load_json_with_sha(
        CONTRACTION_313_EVIDENCE["executed_candidate_path"],
        CONTRACTION_313_EVIDENCE["executed_candidate_sha256"], "313 executed candidate")
    captured_lineage = _load_json_with_sha(
        CONTRACTION_313_EVIDENCE["captured_arrangement_lineage_path"],
        CONTRACTION_313_EVIDENCE["captured_arrangement_lineage_sha256"],
        "313 source-face lineage")
    local_lineage = _load_json_with_sha(
        CONTRACTION_313_EVIDENCE["executed_local_lineage_path"],
        CONTRACTION_313_EVIDENCE["executed_local_lineage_sha256"],
        "313 executed contraction lineage")
    _load_json_with_sha(
        CONTRACTION_313_EVIDENCE["executed_geometry_report_path"],
        CONTRACTION_313_EVIDENCE["executed_geometry_report_sha256"],
        "313 executed local geometry report")
    _require(lineage.get("repair_operations") == local_lineage
             and local_lineage.get("status") == "DOCUMENTED_LOCAL_REFERENCE_MESH_REPAIR"
             and local_lineage.get("adds_physical_mass_or_state") is False,
             "313 contraction is not bound to its executed local operation")
    sidecar = captured_lineage.get("sidecar")
    _require(isinstance(sidecar, list), "313 source-face lineage has no exact face-slot sidecar")
    slots = set(CONTRACTION_313_PATCH_SLOTS)
    matches = [entry for entry in sidecar
               if isinstance(entry, dict) and entry.get("new_face_slot") in slots]
    _require(len(matches) == len(slots),
             "313 contraction has incomplete frozen four-face source lineage")
    pre_row = _candidate_row(pre, 313)
    post_row = _candidate_row(post, 313)
    current_row = _candidate_row(current_candidate, 313)
    _validate_current_candidate_replay(
        executed_row=post_row, current_row=current_row,
        lineage=lineage, stable_id=313)
    return _validate_contraction_face_group_replay(
        pre_row=pre_row, post_row=post_row, current_row=current_row,
        pre_ancestry=pre_row["face_ancestry"],
        post_ancestry=post_row["face_ancestry"],
        current_ancestry=current_row["face_ancestry"],
        operation=local_lineage.get("contraction"),
        sidecar_entries=matches, local_lineage=local_lineage)


def _load_vertex_fan_split_operations(
    *, arrangement: dict, current_candidate: dict, stable_id: int,
) -> tuple[list[dict], dict]:
    """Load fan operations only from the hash-pinned execution that produced this row."""
    lineage = arrangement.get("lineage", {})
    if stable_id == 317:
        prior = lineage.get("prior_fan_repair")
        _require(
            isinstance(prior, dict)
            and prior.get("candidate_sha256") == PRIOR_FAN_317_EVIDENCE["candidate_sha256"]
            and prior.get("lineage_sha256") == PRIOR_FAN_317_EVIDENCE["lineage_sha256"]
            and isinstance(prior.get("report_path"), str)
            and Path(prior["report_path"]).resolve()
                == PRIOR_FAN_317_EVIDENCE["report_path"].resolve()
            and prior.get("report_sha256") == PRIOR_FAN_317_EVIDENCE["report_sha256"]
            and prior.get("status") == PRIOR_FAN_317_EVIDENCE["status"],
            "317 prior fan repair differs from its frozen source evidence",
        )
        report_path = PRIOR_FAN_317_EVIDENCE["report_path"].resolve()
        candidate_path = report_path.parent / "captured-arrangement-candidate.json"
        local_lineage_path = report_path.parent / "local-repair-lineage.json"
        _require(candidate_path == TIP_RETESS_317_EVIDENCE["baseline_candidate_path"].resolve()
                 and sha_file(candidate_path) == PRIOR_FAN_317_EVIDENCE["candidate_sha256"],
                 "317 prior fan candidate differs from the exact 041 baseline")
        local_lineage = _load_json_with_sha(
            local_lineage_path, PRIOR_FAN_317_EVIDENCE["lineage_sha256"],
            "317 prior executed fan lineage")
        report = _load_json_with_sha(
            report_path, PRIOR_FAN_317_EVIDENCE["report_sha256"],
            "317 prior fan report")
        _require(
            report.get("typed_identity") == [51010, 317]
            and report.get("status") == "STOP_EXACT_SELF_FAILURE"
            and report.get("candidate_sha256") == PRIOR_FAN_317_EVIDENCE["candidate_sha256"]
            and report.get("lineage_sha256") == PRIOR_FAN_317_EVIDENCE["lineage_sha256"]
            and report.get("candidate_admission") is False
            and report.get("native_run") is False
            and local_lineage.get("typed_identity") == [51010, 317]
            and local_lineage.get("status") == "DOCUMENTED_LOCAL_REFERENCE_MESH_REPAIR"
            and local_lineage.get("adds_physical_mass_or_state") is False
            and local_lineage.get("baseline_lineage_sha256")
                == TIP_RETESS_317_EVIDENCE["captured_arrangement_lineage_sha256"],
            "317 prior fan operation/report provenance differs from its retained failure",
        )
        operations = local_lineage.get("vertex_fan_splits")
        _require(isinstance(operations, list) and bool(operations),
                 "317 prior fan lineage has no executed split operations")
        return operations, {
            "kind": "retained_317_prior_fan_operation",
            "candidate_sha256": PRIOR_FAN_317_EVIDENCE["candidate_sha256"],
            "lineage_path": str(local_lineage_path.resolve()),
            "lineage_sha256": PRIOR_FAN_317_EVIDENCE["lineage_sha256"],
            "report_path": str(report_path),
            "report_sha256": PRIOR_FAN_317_EVIDENCE["report_sha256"],
            "retained_report_status": report["status"],
            "candidate_admission": False,
            "native_run": False,
        }

    _require(stable_id in (313, 315),
             f"{stable_id} has no frozen fan-operation owner")
    for path_key, hash_key in (
        ("executed_candidate_path", "executed_candidate_sha256"),
        ("executed_local_lineage_path", "executed_local_lineage_sha256"),
        ("executed_geometry_report_path", "executed_geometry_report_sha256"),
    ):
        _require(isinstance(lineage.get(path_key), str)
                 and isinstance(lineage.get(hash_key), str),
                 f"{stable_id} executed fan evidence lacks {path_key}/{hash_key}")
    executed_candidate = _load_json_with_sha(
        Path(lineage["executed_candidate_path"]),
        lineage["executed_candidate_sha256"],
        f"{stable_id} executed fan candidate")
    local_lineage = _load_json_with_sha(
        Path(lineage["executed_local_lineage_path"]),
        lineage["executed_local_lineage_sha256"],
        f"{stable_id} executed fan lineage")
    report = _load_json_with_sha(
        Path(lineage["executed_geometry_report_path"]),
        lineage["executed_geometry_report_sha256"],
        f"{stable_id} executed fan report")
    _require(
        lineage.get("repair_operations") == local_lineage
        and lineage.get("typed_identity") == [51010, stable_id]
        and local_lineage.get("typed_identity") in (None, [51010, stable_id])
        and local_lineage.get("status") == "DOCUMENTED_LOCAL_REFERENCE_MESH_REPAIR"
        and local_lineage.get("adds_physical_mass_or_state") is False
        and report.get("typed_identity") == [51010, stable_id]
        and str(report.get("status", "")).startswith("PASS")
        and report.get("candidate_sha256") == lineage["executed_candidate_sha256"]
        and report.get("lineage_sha256") == lineage["executed_local_lineage_sha256"]
        and report.get("candidate_admission") is False
        and report.get("native_run") is False,
        f"{stable_id} fan operation is not bound to its pinned successful source execution",
    )
    operations = local_lineage.get("vertex_fan_splits")
    _require(isinstance(operations, list) and bool(operations),
             f"{stable_id} executed lineage has no fan split operations")
    _validate_current_candidate_replay(
        executed_row=_candidate_row(executed_candidate, stable_id),
        current_row=_candidate_row(current_candidate, stable_id),
        lineage=lineage, stable_id=stable_id,
        fan_split_operations=operations,
    )
    return operations, {
        "kind": "executed_local_fan_operation",
        "executed_candidate_path": str(Path(lineage["executed_candidate_path"]).resolve()),
        "executed_candidate_sha256": lineage["executed_candidate_sha256"],
        "lineage_path": str(Path(lineage["executed_local_lineage_path"]).resolve()),
        "lineage_sha256": lineage["executed_local_lineage_sha256"],
        "report_path": str(Path(lineage["executed_geometry_report_path"]).resolve()),
        "report_sha256": lineage["executed_geometry_report_sha256"],
        "report_status": report["status"],
        "candidate_admission": False,
        "native_run": False,
    }


def _fan_operation_source_patch_face_ids(
    *, stable_id: int, operation: dict, live_raw_slots: list[int],
    qfaces: np.ndarray, live_ancestry: list[dict],
    sidecar_by_slot: dict[int, dict], source_face_count: int,
) -> list[int]:
    center = operation.get("new_vertex")
    neighbors = operation.get("positive_blend_neighbors")
    _require(type(center) is int and isinstance(neighbors, list)
             and all(type(value) is int for value in neighbors),
             f"{stable_id} fan operation center/neighborhood is malformed")
    neighborhood = {center, *neighbors}
    patch_ids: set[int] = set()
    incident_slot_count = 0
    for raw_slot, face, origin in zip(live_raw_slots, qfaces, live_ancestry):
        if not neighborhood.intersection(map(int, face)):
            continue
        incident_slot_count += 1
        parent_face = origin.get("parent_face_index")
        _require(
            origin.get("parent_stable_id") == stable_id
            and origin.get("partition_role") == 0
            and type(parent_face) is int
            and 0 <= parent_face < source_face_count,
            f"{stable_id} fan-neighborhood face {raw_slot} has invalid source ancestry",
        )
        ancestry_faces = origin.get("parent_source_face_indices", [parent_face])
        _require(
            isinstance(ancestry_faces, list) and ancestry_faces
            and all(type(value) is int and 0 <= value < source_face_count
                    for value in ancestry_faces),
            f"{stable_id} fan-neighborhood face {raw_slot} has invalid parent-source ancestry",
        )
        patch_ids.update([parent_face, *ancestry_faces])
        sidecar_entry = sidecar_by_slot.get(raw_slot)
        if sidecar_entry is not None:
            patch_ids.update(_sidecar_declared_source_patch_faces(
                stable_id=stable_id, face_slot=raw_slot, origin=origin,
                sidecar_entry=sidecar_entry, source_face_count=source_face_count))
    _require(incident_slot_count > 0 and bool(patch_ids),
             f"{stable_id} fan operation has no live source-face neighborhood")
    return sorted(patch_ids)


def _load_317_tip_retessellation_replay(
    arrangement: dict, current_candidate: dict,
    fan_split_operations: list[dict] | None = None,
) -> dict[int, dict]:
    lineage = arrangement.get("lineage", {})
    for key, expected in TIP_RETESS_317_EVIDENCE.items():
        if key.endswith("_path"):
            if key == "baseline_candidate_path":
                continue
            _require(Path(lineage.get(key, "")).resolve() == Path(expected).resolve(),
                     "317 tip-retessellation source path differs from its frozen execution")
        else:
            if key == "baseline_candidate_sha256":
                continue
            _require(lineage.get(key) == expected,
                     "317 tip-retessellation source hash differs from its frozen execution")
    baseline = _load_json_with_sha(
        TIP_RETESS_317_EVIDENCE["baseline_candidate_path"],
        TIP_RETESS_317_EVIDENCE["baseline_candidate_sha256"],
        "317 pre-retessellation baseline candidate")
    executed = _load_json_with_sha(
        TIP_RETESS_317_EVIDENCE["executed_candidate_path"],
        TIP_RETESS_317_EVIDENCE["executed_candidate_sha256"],
        "317 executed tip-retessellation candidate")
    local_lineage = _load_json_with_sha(
        TIP_RETESS_317_EVIDENCE["executed_local_lineage_path"],
        TIP_RETESS_317_EVIDENCE["executed_local_lineage_sha256"],
        "317 executed tip-retessellation lineage")
    _load_json_with_sha(
        TIP_RETESS_317_EVIDENCE["executed_geometry_report_path"],
        TIP_RETESS_317_EVIDENCE["executed_geometry_report_sha256"],
        "317 executed tip-retessellation report")
    _require(lineage.get("repair_operations") == local_lineage
             and local_lineage.get("typed_identity") == [51010, 317]
             and local_lineage.get("candidate_admission") is False
             and local_lineage.get("native_run") is False,
             "317 tip retessellation is not bound to its exact local source operation")
    _require(local_lineage.get("baseline_candidate_sha256")
             == TIP_RETESS_317_EVIDENCE["baseline_candidate_sha256"],
             "317 local operation names a different baseline candidate")
    baseline_row = _candidate_row(baseline, 317)
    executed_row = _candidate_row(executed, 317)
    current_row = _candidate_row(current_candidate, 317)
    proofs = _validate_tip_retessellation_transition(
        baseline_faces=baseline_row["faces"],
        baseline_ancestry=baseline_row["face_ancestry"],
        executed_faces=executed_row["faces"],
        executed_ancestry=executed_row["face_ancestry"],
        baseline_points=baseline_row["points_exact"],
        executed_points=executed_row["points_exact"],
        baseline_maps=baseline_row["parent_maps"],
        executed_maps=executed_row["parent_maps"],
        operations=local_lineage)
    _validate_current_candidate_replay(
        executed_row=executed_row, current_row=current_row,
        lineage=lineage, stable_id=317,
        fan_split_operations=fan_split_operations)
    return proofs


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("local intercostal arrangement: " + message)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_fraction_pair(value, label: str) -> Fraction:
    _require(
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(part, str) for part in value),
        label + " must be a two-string exact rational",
    )
    n_text, d_text = value
    try:
        numerator, denominator = int(n_text), int(d_text)
    except ValueError as error:
        raise ValueError(label + " contains a noninteger rational") from error
    _require(
        str(numerator) == n_text and str(denominator) == d_text and denominator > 0,
        label + " is not a canonical integer pair with positive denominator",
    )
    return Fraction(numerator, denominator)


def _canonical_json_sha256(value) -> str:
    return sha_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8"))


def _retirement_operation_evidence(lineage: dict) -> dict:
    keys = (
        "captured_arrangement_lineage_path", "captured_arrangement_lineage_sha256",
        "executed_candidate_path", "executed_candidate_sha256",
        "executed_geometry_report_path", "executed_geometry_report_sha256",
        "executed_local_lineage_path", "executed_local_lineage_sha256",
    )
    evidence = {key: lineage.get(key) for key in keys if key in lineage}
    for path_key, hash_key in (
        ("captured_arrangement_lineage_path", "captured_arrangement_lineage_sha256"),
        ("executed_candidate_path", "executed_candidate_sha256"),
        ("executed_geometry_report_path", "executed_geometry_report_sha256"),
        ("executed_local_lineage_path", "executed_local_lineage_sha256"),
    ):
        if path_key not in evidence and hash_key not in evidence:
            continue
        _require(path_key in evidence and hash_key in evidence
                 and isinstance(evidence[path_key], str)
                 and isinstance(evidence[hash_key], str),
                 "lineage operation evidence has an incomplete path/hash pair")
        path = Path(evidence[path_key]).resolve()
        _require(path.is_file() and sha_file(path) == evidence[hash_key],
                 "lineage operation evidence source hash differs at " + str(path))
    for key in ("repair_operations", "prior_fan_repair"):
        if key in lineage:
            # These frozen fields include partial/retired child operations. Keep
            # their exact executed values; do not infer any missing geometry.
            evidence[key] = lineage[key]
    return evidence


def build_null_face_retirement_ledger(
    *, stable_id: int, candidate: dict, candidate_sha256: str,
    lineage: dict, lineage_sha256: str, report_sha256: str,
) -> dict:
    """Describe frozen null face slots without changing candidate geometry."""
    row_key = str(stable_id)
    try:
        faces = candidate["faces"][row_key]
        ancestry = candidate["face_ancestry"][row_key]
    except (KeyError, TypeError) as error:
        raise ValueError("null-retirement candidate lacks the requested row") from error
    _require(len(faces) == len(ancestry), "null-retirement face and ancestry counts differ")
    active_children: dict[int, list[int]] = {}
    for slot, (face, origin) in enumerate(zip(faces, ancestry)):
        if face is not None and isinstance(origin, dict) and origin.get("origin_kind") == 1:
            active_children.setdefault(int(origin["parent_face_index"]), []).append(slot)
    retirements = []
    for slot, (face, origin) in enumerate(zip(faces, ancestry)):
        if face is not None:
            continue
        _require(isinstance(origin, dict), f"null slot {slot} has no source ancestry")
        parent_face = int(origin["parent_face_index"])
        child_slots = sorted(active_children.get(parent_face, []))
        effective = dict(origin)
        effective["origin_kind"] = 1
        retirements.append({
            "face_slot": slot,
            "source_ancestry": dict(origin),
            "effective_retired_ancestry": effective,
            "classification": (
                "superseded_by_active_generated_child" if child_slots
                else "fully_retired_local_arrangement"
            ),
            "active_generated_child_slots_same_parent": child_slots,
        })
    operation_evidence = _retirement_operation_evidence(lineage)
    return {
        "schema": NULL_RETIREMENT_LEDGER_SCHEMA,
        "status": "FROZEN_NULL_SLOT_RETIREMENT_METADATA_NOT_ANATOMY_ADMISSION",
        "stable_id": stable_id,
        "source_payload_sha256": REPAIR_PARENT_PAYLOAD_SHA256,
        "arrangement_inputs_045_sha256": ARRANGEMENT_INPUTS_045_SHA256,
        "candidate_sha256": candidate_sha256,
        "lineage_sha256": lineage_sha256,
        "report_sha256": report_sha256,
        "generic_retirement_implementation": {
            "path": str(GENERIC_NULL_RETIREMENT_SOURCE),
            "sha256": GENERIC_NULL_RETIREMENT_SHA256,
            "operation_lines": [608, 616, 633, 634, 654],
        },
        "operation_evidence": operation_evidence,
        "candidate_face_slot_count": len(faces),
        "null_face_slots": [entry["face_slot"] for entry in retirements],
        "null_face_slots_sha256": _canonical_json_sha256(retirements),
        "retirements": retirements,
        "candidate_geometry_changed": False,
        "candidate_admission": False,
        "native_or_pose_predicates_run": False,
    }


def validate_null_face_retirement_ledger(
    *, stable_id: int, candidate: dict, candidate_sha256: str,
    lineage: dict, lineage_sha256: str, report_sha256: str, ledger: dict,
) -> dict[int, dict]:
    """Fail closed unless every candidate null has exact source-slot lineage."""
    expected = build_null_face_retirement_ledger(
        stable_id=stable_id, candidate=candidate, candidate_sha256=candidate_sha256,
        lineage=lineage, lineage_sha256=lineage_sha256, report_sha256=report_sha256)
    _require(isinstance(ledger, dict), "candidate null faces lack a retirement ledger")
    _require(ledger == expected,
             "null-face ledger is incomplete, ancestry-mismatched, or not bound to frozen operations")
    _require(GENERIC_NULL_RETIREMENT_SOURCE.is_file()
             and sha_file(GENERIC_NULL_RETIREMENT_SOURCE) == GENERIC_NULL_RETIREMENT_SHA256,
             "frozen generic null-retirement implementation hash differs")
    return {entry["face_slot"]: entry for entry in ledger["retirements"]}


def load_pinned_arrangement(
    *,
    candidate_path: Path,
    candidate_sha256: str,
    lineage_path: Path,
    lineage_sha256: str,
    report_path: Path,
    report_sha256: str,
    source_payload_sha256: str,
    source_payload_index: int,
    stable_id: int | None = None,
    null_face_retirement_ledger_path: Path | None = None,
    null_face_retirement_ledger_sha256: str | None = None,
) -> dict:
    """Load existing candidate inputs and their explicit null-slot retirement ledger."""
    paths = {
        "candidate": (Path(candidate_path), candidate_sha256),
        "lineage": (Path(lineage_path), lineage_sha256),
        "report": (Path(report_path), report_sha256),
    }
    loaded = {}
    for label, (path, expected) in paths.items():
        path = path.resolve()
        _require(path.is_file() and sha_file(path) == expected, label + " file hash differs")
        loaded[label + "_path"] = str(path)
        loaded[label + "_sha256"] = expected
        loaded[label + "_raw"] = path.read_bytes()
    _require(
        source_payload_sha256 == REPAIR_PARENT_PAYLOAD_SHA256,
        "full_candidate parent is not the pinned 028 retained-face NHA",
    )
    _require(
        type(source_payload_index) is int and source_payload_index == 1,
        "full_candidate parent payload index must be the existing source slot 1",
    )
    try:
        candidate = json.loads(loaded["candidate_raw"])
        lineage = json.loads(loaded["lineage_raw"])
        report = json.loads(loaded["report_raw"])
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("candidate, lineage, or report is not valid JSON") from error
    _require(
        isinstance(candidate, dict)
        and {"faces", "face_ancestry", "parent_maps", "points_exact"}.issubset(candidate),
        "full_candidate is missing existing face ancestry or parent map fields",
    )
    _require(
        isinstance(lineage, dict)
        and lineage.get("status") == "DOCUMENTED_LOCAL_REFERENCE_MESH_REPAIR"
        and lineage.get("adds_physical_mass_or_state") is False,
        "local-repair lineage is missing or changes a physical owner",
    )
    _require(
        isinstance(report, dict)
        and str(report.get("status", "")).startswith("PASS_")
        and report.get("candidate_admission") is False
        and report.get("native_run") is False
        and report.get("candidate_sha256") == candidate_sha256
        and report.get("lineage_sha256") == lineage_sha256,
        "source preparation report does not pass and bind this candidate/lineage",
    )
    _require((null_face_retirement_ledger_path is None)
             == (null_face_retirement_ledger_sha256 is None),
             "null-face retirement ledger requires both path and hash")
    retirement_ledger = None
    if null_face_retirement_ledger_path is not None:
        ledger_path = Path(null_face_retirement_ledger_path).resolve()
        _require(ledger_path.is_file()
                 and sha_file(ledger_path) == null_face_retirement_ledger_sha256,
                 "null-face retirement ledger path/hash differs")
        try:
            retirement_ledger = json.loads(ledger_path.read_bytes())
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError("null-face retirement ledger is not valid JSON") from error
        _require(type(stable_id) is int and stable_id in REPAIRED_IDS,
                 "null-face retirement ledger requires an explicit repaired stable ID")
        validate_null_face_retirement_ledger(
            stable_id=stable_id, candidate=candidate, candidate_sha256=candidate_sha256,
            lineage=lineage, lineage_sha256=lineage_sha256, report_sha256=report_sha256,
            ledger=retirement_ledger)
    else:
        _require(type(stable_id) is int and stable_id in REPAIRED_IDS,
                 "arrangement loader requires an explicit repaired stable ID")
        _require(not any(face is None for face in candidate["faces"].get(str(stable_id), [])),
                 "full_candidate null face slots lack a pinned retirement ledger")
    return {
        **loaded,
        "source_payload_sha256": source_payload_sha256,
        "source_payload_index": source_payload_index,
        "candidate": candidate,
        "lineage": lineage,
        "report": report,
        "null_face_retirement_ledger_path": (
            None if null_face_retirement_ledger_path is None
            else str(Path(null_face_retirement_ledger_path).resolve())),
        "null_face_retirement_ledger_sha256": null_face_retirement_ledger_sha256,
        "null_face_retirement_ledger": retirement_ledger,
    }



def load_pinned_fragment_repair(
    *,
    base_receipt_path: Path,
    base_payload_sha256: str,
    repair_parent_receipt_path: Path,
    repair_parent_receipt_sha256: str,
    repair_parent_payload_path: Path,
    repair_report_path: Path,
    repair_report_sha256: str,
) -> dict:
    """Verify the existing 028 face-deletion parent and its N27 ancestry."""
    from .resting_pleura_proxy import _parse_payload

    paths = {
        "base_receipt": (Path(base_receipt_path), N27_RECEIPT_SHA256),
        "repair_parent_receipt": (Path(repair_parent_receipt_path), REPAIR_PARENT_RECEIPT_SHA256),
        "repair_parent_payload": (Path(repair_parent_payload_path), REPAIR_PARENT_PAYLOAD_SHA256),
        "repair_report": (Path(repair_report_path), REPAIR_REPORT_SHA256),
    }
    loaded = {}
    for label, (path, pinned) in paths.items():
        path = path.resolve()
        _require(path.is_file() and sha_file(path) == pinned, label + " hash differs from pinned 028/N27 input")
        loaded[label + "_path"] = str(path)
        loaded[label + "_sha256"] = pinned
        loaded[label + "_raw"] = path.read_bytes()
    _require(base_payload_sha256 == N27_PAYLOAD_SHA256, "append base payload is not pinned N27")
    _require(repair_parent_receipt_sha256 == REPAIR_PARENT_RECEIPT_SHA256,
             "028 parent receipt hash differs")
    _require(repair_report_sha256 == REPAIR_REPORT_SHA256, "028 report hash differs")
    base_receipt = json.loads(loaded["base_receipt_raw"])
    parent_receipt = json.loads(loaded["repair_parent_receipt_raw"])
    report = json.loads(loaded["repair_report_raw"])
    base_payload_path = Path(base_receipt.get("payload", {}).get("path", "")).resolve()
    parent_payload = parent_receipt.get("payload", {})
    _require(base_receipt.get("payload", {}).get("sha256") == N27_PAYLOAD_SHA256
             and sha_file(base_payload_path) == N27_PAYLOAD_SHA256,
             "028 source receipt does not resolve to pinned N27 payload")
    _require(parent_payload.get("sha256") == REPAIR_PARENT_PAYLOAD_SHA256
             and Path(parent_payload.get("path", "")).resolve()
             == Path(loaded["repair_parent_payload_path"]),
             "028 repaired payload receipt/path differs")
    declaration = parent_receipt.get("provenance", {}).get(
        "local_intercostal_source_fragment_repair", {})
    _require(declaration.get("source_payload_sha256") == N27_PAYLOAD_SHA256
             and declaration.get("report_sha256") == REPAIR_REPORT_SHA256
             and Path(declaration.get("report_path", "")).resolve()
             == Path(loaded["repair_report_path"]),
             "028 receipt does not bind the retained fragment-repair report")
    _require(report.get("status") == "PASS_SCOPED_SOURCE_PREPARATION_REPAIR_NOT_ANATOMY_ADMISSION"
             and report.get("candidate_admission") is False
             and report.get("all_vertex_normal_bytes_owners_and_unrelated_faces_unchanged") is True
             and report.get("physical_mass_and_circulation_state_added") is False
             and report.get("source_payload", {}).get("sha256") == N27_PAYLOAD_SHA256
             and report.get("source_receipt", {}).get("sha256") == N27_RECEIPT_SHA256
             and report.get("output_payload", {}).get("sha256") == REPAIR_PARENT_PAYLOAD_SHA256,
             "028 report is not the pinned source-only face deletion")
    _require(set(report.get("surfaces", {})) == {str(value) for value in REPAIRED_IDS},
             "028 report does not cover exactly intercostal IDs 313/315/317")
    _, _, base_rows = _parse_payload(base_payload_path.read_bytes())
    _, _, parent_rows = _parse_payload(Path(loaded["repair_parent_payload_path"]).read_bytes())
    _require(set(base_rows) == set(parent_rows), "028 repair changed the stable-ID inventory")
    base_source_map = base_receipt.get("provenance", {}).get("source_id_map", {})
    parent_source_map = parent_receipt.get("provenance", {}).get("source_id_map", {})
    _require(set(base_source_map) == set(parent_source_map)
             == {str(stable_id) for stable_id in base_rows},
             "028 repair changed the source identity inventory")
    for stable_id in base_rows:
        if stable_id in REPAIRED_IDS:
            source, parent = base_rows[stable_id], parent_rows[stable_id]
            _require(_row_identity(source, stable_id) == _row_identity(parent, stable_id)
                     and _same_bytes(source["vertices6"], parent["vertices6"], "<f4"),
                     f"028 repair changed intercostal identity or vertices for {stable_id}")
            _repair_face_map(stable_id=stable_id, source_faces=source["faces"],
                             parent_faces=parent["faces"], repair_report=report)
            source_identity = dict(base_source_map[str(stable_id)])
            parent_identity = dict(parent_source_map[str(stable_id)])
            source_identity.pop("repair", None)
            parent_repair = parent_identity.pop("repair", None)
            _require(source_identity == parent_identity
                     and isinstance(parent_repair, dict)
                     and "isolated_opposed_fragments" in parent_repair,
                     f"028 source identity/isolated-fragment provenance changed for {stable_id}")
        else:
            _require(_normalized_row_bytes(base_rows[stable_id])
                     == _normalized_row_bytes(parent_rows[stable_id]),
                     f"028 repair changed unrelated N27 row {stable_id}")
            _require(base_source_map[str(stable_id)] == parent_source_map[str(stable_id)],
                     f"028 repair changed unrelated source identity {stable_id}")
    return {
        **{key: value for key, value in loaded.items() if not key.endswith("_raw")},
        "repair_report": report,
        "base_rows": base_rows,
        "repair_parent_rows": parent_rows,
        "base_source_id_map": base_source_map,
        "repair_parent_source_id_map": parent_source_map,
        "fragment_repair_declaration": parent_receipt["provenance"][
            "local_intercostal_source_fragment_repair"],
        "unrelated_n27_rows_byte_exact": True,
    }


def _round_fraction_to_f32(value: Fraction) -> np.float32:
    """Round an exact rational directly to IEEE-754 binary32, ties to even."""
    value = Fraction(value)
    if value == 0:
        word = 0
    else:
        sign = 0x80000000 if value < 0 else 0
        magnitude = abs(value)
        numerator, denominator = magnitude.numerator, magnitude.denominator
        exponent = numerator.bit_length() - denominator.bit_length()
        if exponent >= 0:
            if numerator < (denominator << exponent):
                exponent -= 1
        elif (numerator << (-exponent)) < denominator:
            exponent -= 1

        def rounded_ratio(n: int, d: int) -> int:
            quotient, remainder = divmod(n, d)
            twice = 2 * remainder
            if twice > d or (twice == d and quotient & 1):
                quotient += 1
            return quotient

        if exponent >= -126:
            shift = 23 - exponent
            if shift >= 0:
                significand = rounded_ratio(numerator << shift, denominator)
            else:
                significand = rounded_ratio(numerator, denominator << (-shift))
            if significand == (1 << 24):
                exponent += 1
                significand = 1 << 23
            _require(exponent <= 127, "exact parent interpolation overflows Float32")
            exponent_bits = exponent + 127
            fraction_bits = significand - (1 << 23)
        else:
            significand = rounded_ratio(numerator << 149, denominator)
            if significand >= (1 << 23):
                exponent_bits, fraction_bits = 1, 0
            else:
                exponent_bits, fraction_bits = 0, significand
        word = sign | (exponent_bits << 23) | fraction_bits
    return np.asarray([word], dtype="<u4").view("<f4")[0]


def load_pinned_step0_capture_parent_xyz(*, capture_manifest_path: Path,
                                        capture_manifest_sha256: str,
                                        source_rows: dict[int, dict]) -> tuple[dict[int, np.ndarray], dict]:
    """Bind source parents to exact P6 step-0 world coordinates by indexed face ancestry."""
    manifest_path = Path(capture_manifest_path).resolve()
    _require(capture_manifest_sha256 == N27_CAPTURE_MANIFEST_SHA256
             and manifest_path.is_file()
             and sha_file(manifest_path) == N27_CAPTURE_MANIFEST_SHA256,
             "accepted capture manifest is not the pinned N27 input")
    try:
        manifest = json.loads(manifest_path.read_bytes())
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("accepted capture manifest is not valid JSON") from error
    capture_rows = manifest.get("capture_rows")
    _require(isinstance(capture_rows, list), "accepted capture manifest has no capture rows")
    step_zero = [row for row in capture_rows if isinstance(row, dict) and row.get("step") == 0]
    _require(len(step_zero) == 1, "accepted capture manifest must contain one step-0 row")
    capture = step_zero[0]
    _require(capture.get("accepted_time_s") == 0, "step-0 capture time is not exactly zero")
    pack_path = Path(capture.get("pack_path", "")).resolve()
    receipt_path = Path(capture.get("receipt_path", "")).resolve()
    pack_sha256, receipt_sha256 = capture.get("recorded_pack_sha256"), capture.get("receipt_sha256")
    _require(pack_path.is_file() and isinstance(pack_sha256, str)
             and sha_file(pack_path) == pack_sha256
             and receipt_path.is_file() and isinstance(receipt_sha256, str)
             and sha_file(receipt_path) == receipt_sha256,
             "step-0 pack or receipt differs from pinned manifest bytes")
    from .resting_anatomy_verify import load_capture
    surfaces, capture_meta = load_capture(pack_path, receipt_path)
    _require(capture_meta.get("pack_sha256") == pack_sha256
             and capture_meta.get("capture_receipt_sha256") == receipt_sha256
             and capture_meta.get("accepted_step") == 0
             and capture_meta.get("accepted_time_s") == 0,
             "P6 capture loader returned a different accepted input identity")
    capture_receipt = json.loads(receipt_path.read_bytes())
    body20 = [row for row in capture_receipt.get("accepted_body_poses", [])
              if isinstance(row, dict) and row.get("body_index") == 20]
    _require(len(body20) == 1, "step-0 receipt has no unique accepted body-20 pose")
    captured_parent_xyz, row_proofs = {}, {}
    for stable_id in sorted(REPAIRED_IDS):
        _require(stable_id in source_rows, f"N27 source row {stable_id} is absent")
        source_row = source_rows[stable_id]
        identity = _row_identity(source_row, stable_id)
        _require(identity[0] == 20, f"N27 source row {stable_id} is not body-20 owned")
        surface_key = (51010, stable_id)
        _require(surface_key in surfaces, f"typed step-0 capture row {surface_key} is absent")
        positions, faces, _normals, _global_ids = surfaces[surface_key]
        source_faces = np.ascontiguousarray(source_row["faces"], dtype="<u4")
        capture_faces = np.ascontiguousarray(faces, dtype="<u4")
        _require(_same_bytes(source_faces, capture_faces, "<u4"),
                 f"P6 face order differs for stable ID {stable_id}")
        source_count = len(source_row["vertices6"])
        source_to_capture = np.full(source_count, -1, dtype=np.int64)
        capture_to_source = np.full(len(positions), -1, dtype=np.int64)
        for face_index, (source_face, capture_face) in enumerate(zip(source_faces, capture_faces)):
            for source_vertex, capture_vertex in zip(source_face, capture_face):
                source_vertex, capture_vertex = int(source_vertex), int(capture_vertex)
                prior_capture, prior_source = int(source_to_capture[source_vertex]), int(capture_to_source[capture_vertex])
                _require(prior_capture in (-1, capture_vertex) and prior_source in (-1, source_vertex),
                         f"P6 indexed vertex correspondence conflicts at face {face_index}")
                source_to_capture[source_vertex], capture_to_source[capture_vertex] = capture_vertex, source_vertex
        _require(np.all(source_to_capture >= 0) and np.all(capture_to_source >= 0)
                 and len(np.unique(source_to_capture)) == source_count and source_count == len(positions),
                 f"P6 vertex correspondence is not bijective for stable ID {stable_id}")
        world_xyz = np.ascontiguousarray(np.asarray(positions, dtype="<f4")[source_to_capture], dtype="<f4")
        _require(np.isfinite(world_xyz).all(), f"P6 world positions are nonfinite for {stable_id}")
        captured_parent_xyz[stable_id] = world_xyz
        row_proofs[str(stable_id)] = {
            "typed_identity": [51010, stable_id], "source_body_index": identity[0],
            "source_vertex_count": source_count, "captured_vertex_count": len(positions),
            "source_face_count": len(source_faces),
            "source_face_bytes_sha256": sha_bytes(source_faces.tobytes()),
            "capture_face_bytes_sha256": sha_bytes(capture_faces.tobytes()),
            "face_order_and_corner_correspondence_exact": True,
            "source_to_capture_vertex_map_sha256": sha_bytes(source_to_capture.astype("<u4").tobytes()),
            "captured_world_parent_xyz_f32_sha256": sha_bytes(world_xyz.tobytes()),
            "all_source_vertex_slots_have_capture_ancestry": True,
        }
    verifier_path = Path(__file__).with_name("resting_anatomy_verify.py")
    report = {
        "method": "existing P6 load_capture typed-surface decode; exact indexed face ancestry, no proximity fallback",
        "capture_manifest_path": str(manifest_path), "capture_manifest_sha256": N27_CAPTURE_MANIFEST_SHA256,
        "accepted_step": 0, "accepted_time_s": 0,
        "pack_path": str(pack_path), "pack_sha256": pack_sha256,
        "capture_receipt_path": str(receipt_path), "capture_receipt_sha256": receipt_sha256,
        "accepted_body20_pose": body20[0],
        "capture_loader_source_path": str(verifier_path.resolve()),
        "capture_loader_source_sha256": sha_file(verifier_path),
        "rows": row_proofs, "native_or_pose_predicates_run": False, "anatomical_admission": False,
    }
    return captured_parent_xyz, report


def compose_pinned_arrangement_set(
    *, base_rows: dict, repair_context: dict, arrangement_descriptors: dict[int, dict],
    capture_manifest_path: Path | None = None, capture_manifest_sha256: str | None = None,
) -> tuple[dict, dict]:
    """Rebuild the three local arrangement rows from only their pinned parents."""
    _require(set(map(int, base_rows)) >= REPAIRED_IDS,
             "pinned N27 base lacks one of the three intercostal rows")
    _require(set(map(int, arrangement_descriptors)) == REPAIRED_IDS,
             "arrangement descriptors must cover exactly IDs 313/315/317")
    _require((capture_manifest_path is None) == (capture_manifest_sha256 is None),
             "capture reference requires both a manifest path and hash")
    captured_parent_xyz_by_id, capture_report = None, None
    if capture_manifest_path is not None:
        captured_parent_xyz_by_id, capture_report = load_pinned_step0_capture_parent_xyz(
            capture_manifest_path=capture_manifest_path,
            capture_manifest_sha256=capture_manifest_sha256, source_rows=base_rows)
    composed, summaries = {}, {}
    for stable_id in sorted(REPAIRED_IDS):
        descriptor = arrangement_descriptors[stable_id]
        _require(isinstance(descriptor, dict), f"arrangement descriptor {stable_id} is malformed")
        arrangement = load_pinned_arrangement(
            candidate_path=Path(descriptor["candidate_path"]), candidate_sha256=descriptor["candidate_sha256"],
            lineage_path=Path(descriptor["lineage_path"]), lineage_sha256=descriptor["lineage_sha256"],
            report_path=Path(descriptor["report_path"]), report_sha256=descriptor["report_sha256"],
            source_payload_sha256=descriptor["source_payload_sha256"],
            source_payload_index=descriptor["source_payload_index"],
            stable_id=stable_id,
            null_face_retirement_ledger_path=Path(
                descriptor["null_face_retirement_ledger_path"]),
            null_face_retirement_ledger_sha256=descriptor[
                "null_face_retirement_ledger_sha256"])
        row, summary = compose_intercostal_arrangement_row(
            stable_id=stable_id, source_row=base_rows[stable_id],
            parent_row=repair_context["repair_parent_rows"][stable_id], arrangement=arrangement,
            repair_report=repair_context["repair_report"],
            captured_source_xyz=None if captured_parent_xyz_by_id is None else captured_parent_xyz_by_id[stable_id])
        composed[stable_id] = row
        if capture_report is not None:
            summary["captured_reference_shared"] = {k: v for k, v in capture_report.items() if k != "rows"}
            summary["captured_reference_row"] = capture_report["rows"][str(stable_id)]
        summaries[str(stable_id)] = {k: v for k, v in summary.items()
                                     if k not in {"candidate_vertex_parent_maps", "candidate_faces"}}
    return composed, summaries

def _row_identity(row: dict, stable_id: int) -> tuple[int, int, int, int]:
    record = tuple(map(int, row["record"]))
    result = (
        int(row["body_index"]),
        int(row["layer"]),
        int(row["flags"]),
        int(record[5]),
    )
    _require(result[3] == stable_id, "row stable ID differs from the requested identity")
    return result


def _same_bytes(first, second, dtype) -> bool:
    return (
        np.ascontiguousarray(first, dtype=dtype).tobytes()
        == np.ascontiguousarray(second, dtype=dtype).tobytes()
    )


def _normalized_row_bytes(row: dict) -> bytes:
    body, layer, flags, stable_id = (
        int(row["body_index"]),
        int(row["layer"]),
        int(row["flags"]),
        int(row["record"][5]),
    )
    vertices = np.ascontiguousarray(row["vertices6"], dtype="<f4")
    faces = np.ascontiguousarray(row["faces"], dtype="<u4")
    record = np.asarray([body, len(vertices), len(faces) * 3, stable_id, layer, flags], dtype="<u4")
    return record.tobytes() + vertices.tobytes() + faces.tobytes()


def _repair_face_map(
    *,
    stable_id: int,
    source_faces: np.ndarray,
    parent_faces: np.ndarray,
    repair_report: dict,
) -> np.ndarray:
    report_row = repair_report.get("surfaces", {}).get(str(stable_id))
    _require(isinstance(report_row, dict), f"028 repair report omits stable ID {stable_id}")
    _require(
        int(report_row.get("source_face_count", -1)) == len(source_faces)
        and int(report_row.get("retained_face_count", -1)) == len(parent_faces)
        and report_row.get("source_coordinates_modified") is False
        and report_row.get("new_vertices") == 0
        and report_row.get("new_faces") == 0,
        f"028 repair report counts or scope differ for stable ID {stable_id}",
    )
    removed = np.asarray(report_row.get("removed_source_face_ids", []), dtype=np.int64)
    _require(
        removed.ndim == 1
        and np.all((0 <= removed) & (removed < len(source_faces)))
        and len(np.unique(removed)) == len(removed)
        and np.all(np.diff(removed) > 0),
        f"028 removed source-face IDs are malformed for stable ID {stable_id}",
    )
    keep = np.ones(len(source_faces), dtype=bool)
    keep[removed] = False
    retained = np.flatnonzero(keep).astype(np.int64)
    _require(
        _same_bytes(source_faces[retained], parent_faces, "<u4"),
        f"028 retained face order differs from exact N27 source order for stable ID {stable_id}",
    )
    return retained


def _read_row_maps(
    *, stable_id: int, source_vertices6: np.ndarray, points_exact, parent_maps,
    source_payload_index: int, captured_source_xyz: np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[tuple[tuple[int, Fraction], ...], ...]]:
    source_xyz = np.ascontiguousarray(source_vertices6[:, :3], dtype="<f4")
    _require(source_vertices6.ndim == 2 and source_vertices6.shape[1] == 6
             and np.isfinite(source_vertices6).all(), "source ABI5 row vertices are invalid")
    _require(len(points_exact) == len(parent_maps), "full_candidate point and parent-map vertex counts differ")
    captured_xyz = source_xyz if captured_source_xyz is None else np.ascontiguousarray(captured_source_xyz, dtype="<f4")
    _require(captured_xyz.shape == source_xyz.shape and np.isfinite(captured_xyz).all(),
             "captured parent coordinates do not cover the pinned N27 source row")
    points, parsed_maps = np.empty((len(points_exact), 3), dtype="<f4"), []
    for vertex_index, (point_row, map_row) in enumerate(zip(points_exact, parent_maps)):
        _require(isinstance(point_row, (list, tuple)) and len(point_row) == 3
                 and isinstance(map_row, list) and bool(map_row),
                 f"full_candidate vertex {vertex_index} lacks exact XYZ or parent terms")
        exact_point = tuple(parse_fraction_pair(value, f"full_candidate point {vertex_index}") for value in point_row)
        local_xyz, captured_world_xyz = [Fraction(0)] * 3, [Fraction(0)] * 3
        parsed_terms, seen, weight_sum = [], set(), Fraction(0)
        for term in map_row:
            _require(isinstance(term, (list, tuple)) and len(term) == 5,
                     f"parent term {vertex_index} has the wrong schema")
            payload_index, parent_id, parent_vertex = term[:3]
            _require(all(type(value) is int for value in (payload_index, parent_id, parent_vertex))
                     and payload_index == source_payload_index and parent_id == stable_id
                     and 0 <= parent_vertex < len(source_xyz),
                     f"parent term {vertex_index} escapes its pinned N27 source row")
            key = (payload_index, parent_id, parent_vertex)
            _require(key not in seen, "parent map repeats a source vertex")
            seen.add(key)
            weight = parse_fraction_pair(term[3:5], f"parent weight {vertex_index}")
            _require(weight > 0, "parent weights must be strictly positive")
            weight_sum += weight
            parsed_terms.append((parent_vertex, weight))
            for axis in range(3):
                local_xyz[axis] += Fraction.from_float(float(source_xyz[parent_vertex, axis])) * weight
                captured_world_xyz[axis] += Fraction.from_float(float(captured_xyz[parent_vertex, axis])) * weight
        _require(weight_sum == 1, f"parent weights do not sum exactly to one at vertex {vertex_index}")
        candidate_f32 = tuple(_round_fraction_to_f32(value) for value in exact_point)
        mapped_world_f32 = tuple(_round_fraction_to_f32(value) for value in captured_world_xyz)
        _require(all(candidate_f32[i].tobytes() == mapped_world_f32[i].tobytes() for i in range(3)),
                 f"full_candidate point {vertex_index} is not canonical F32 of its exact positive map in captured frame")
        points[vertex_index] = [_round_fraction_to_f32(value) for value in local_xyz]
        parsed_maps.append(tuple(parsed_terms))
    return points, tuple(parsed_maps)

def compose_intercostal_arrangement_row(
    *,
    stable_id: int,
    source_row: dict,
    parent_row: dict,
    arrangement: dict,
    repair_report: dict,
    captured_source_xyz: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """Compose a 028 arrangement patch while preserving original NHA slots/faces."""
    _require(stable_id in REPAIRED_IDS, "only stable IDs 313, 315, and 317 are supported")
    _require(
        arrangement.get("source_payload_sha256") == REPAIR_PARENT_PAYLOAD_SHA256
        and arrangement.get("source_payload_index") == 1,
        "arrangement is not bound to the pinned 028 parent payload",
    )
    _require(
        _row_identity(source_row, stable_id) == _row_identity(parent_row, stable_id),
        "source and 028 row owner/stable identity differs",
    )
    source_vertices6 = np.ascontiguousarray(source_row["vertices6"], dtype="<f4")
    parent_vertices6 = np.ascontiguousarray(parent_row["vertices6"], dtype="<f4")
    source_faces = np.ascontiguousarray(source_row["faces"], dtype="<u4")
    parent_faces = np.ascontiguousarray(parent_row["faces"], dtype="<u4")
    _require(
        _same_bytes(source_vertices6, parent_vertices6, "<f4"),
        "028 changed original vertices6 prefix or normals",
    )
    retained_n27_face_ids = _repair_face_map(
        stable_id=stable_id,
        source_faces=source_faces,
        parent_faces=parent_faces,
        repair_report=repair_report,
    )
    raw = arrangement.get("candidate_raw")
    _require(
        isinstance(raw, bytes)
        and sha_bytes(raw) == arrangement.get("candidate_sha256"),
        "full_candidate bytes changed after hash verification",
    )
    candidate = json.loads(raw)
    row_key = str(stable_id)
    try:
        raw_faces = candidate["faces"][row_key]
        ancestry = candidate["face_ancestry"][row_key]
        raw_parent_maps = candidate["parent_maps"][row_key]
        raw_points = candidate["points_exact"][row_key]
    except (KeyError, TypeError) as error:
        raise ValueError("full_candidate lacks row " + row_key) from error
    _require(
        len(raw_faces) == len(ancestry)
        and len(raw_parent_maps) == len(raw_points)
        and bool(raw_faces),
        "full_candidate face/ancestry/point/map counts differ",
    )
    points, parent_maps = _read_row_maps(
        stable_id=stable_id,
        source_vertices6=source_vertices6,
        points_exact=raw_points,
        parent_maps=raw_parent_maps,
        source_payload_index=1,
        captured_source_xyz=captured_source_xyz,
    )
    null_slots = [slot for slot, face in enumerate(raw_faces) if face is None]
    retirement_ledger = arrangement.get("null_face_retirement_ledger")
    retired_by_slot = {}
    if null_slots:
        _require(isinstance(retirement_ledger, dict),
                 "full_candidate null face slots lack a validated retirement ledger")
        retired_by_slot = validate_null_face_retirement_ledger(
            stable_id=stable_id, candidate=candidate,
            candidate_sha256=arrangement["candidate_sha256"],
            lineage=arrangement["lineage"], lineage_sha256=arrangement["lineage_sha256"],
            report_sha256=arrangement["report_sha256"], ledger=retirement_ledger)
        _require(set(retired_by_slot) == set(null_slots),
                 "retirement ledger does not cover exactly the null face slots")
    live_faces = []
    live_ancestry = []
    live_raw_slots = []
    for slot, (face, raw_origin) in enumerate(zip(raw_faces, ancestry)):
        if face is None:
            _require(slot in retired_by_slot,
                     f"full_candidate null face {slot} is not explicitly retired")
            origin = retired_by_slot[slot]["effective_retired_ancestry"]
            _require(origin["origin_kind"] == 1,
                     f"full_candidate retired face ancestry {slot} is not reclassified")
            continue
        _require(
            isinstance(face, (list, tuple))
            and len(face) == 3
            and all(type(index) is int for index in face)
            and min(face) >= 0
            and max(face) < len(points),
            f"full_candidate face {slot} has invalid local vertex indices",
        )
        origin = raw_origin
        _require(
            isinstance(origin, dict)
            and all(type(origin.get(key)) is int for key in (
                "input_index", "origin_kind", "parent_face_index",
                "parent_stable_id", "partition_role"))
            and origin["parent_stable_id"] == stable_id
            and origin["input_index"] == origin["parent_face_index"]
            and 0 <= origin["parent_face_index"] < len(parent_faces)
            and origin["partition_role"] == 0
            and origin["origin_kind"] in (0, 1),
            f"full_candidate face ancestry {slot} is outside the declared source row",
        )
        live_faces.append(tuple(face))
        live_ancestry.append(origin)
        live_raw_slots.append(slot)
    qfaces = np.ascontiguousarray(np.asarray(live_faces, dtype="<u4"))
    original_slots = [
        slot for slot, origin in enumerate(live_ancestry)
        if origin["origin_kind"] == 0
    ]
    original_parent_faces = [
        live_ancestry[slot]["parent_face_index"] for slot in original_slots
    ]
    active_generated_by_parent: dict[int, list[int]] = {}
    for slot, origin in enumerate(live_ancestry):
        if origin["origin_kind"] == 1:
            active_generated_by_parent.setdefault(origin["parent_face_index"], []).append(slot)
    fully_retired_source_parents: dict[int, int] = {}
    if retirement_ledger is not None:
        for entry in retirement_ledger["retirements"]:
            if (entry["source_ancestry"]["origin_kind"] == 0
                    and entry["classification"] == "fully_retired_local_arrangement"):
                parent_face = entry["source_ancestry"]["parent_face_index"]
                fully_retired_source_parents[parent_face] = (
                    fully_retired_source_parents.get(parent_face, 0) + 1)
    _require(
        original_slots == sorted(original_slots)
        and original_parent_faces == sorted(original_parent_faces)
        and len(set(original_parent_faces)) == len(original_parent_faces),
        "retained origin_kind=0 source faces are not unique and in original order",
    )
    retained_parent_set = set(original_parent_faces)
    for parent_face in range(len(parent_faces)):
        retained_count = int(parent_face in retained_parent_set)
        child_count = int(bool(active_generated_by_parent.get(parent_face)))
        retired_count = fully_retired_source_parents.get(parent_face, 0)
        _require(
            (retained_count == 1 and retired_count == 0)
            or (retained_count == 0 and child_count > 0 and retired_count == 0)
            or (retained_count == 0 and child_count == 0 and retired_count == 1),
            f"028 parent face {parent_face} is missing or ambiguously retained/replaced/retired",
        )
    for slot in original_slots:
        face_id = live_ancestry[slot]["parent_face_index"]
        qxyz = points[qfaces[slot]]
        source_xyz = source_vertices6[parent_faces[face_id], :3]
        _require(
            _same_bytes(qxyz, source_xyz, "<f4"),
            f"origin_kind=0 face {slot} changed source coordinates or winding",
        )

    patch_slots = [
        slot for slot, origin in enumerate(live_ancestry)
        if origin["origin_kind"] == 1
    ]
    used_qvertices = {int(vertex) for face in qfaces for vertex in face}
    unreferenced_qvertices = sorted(set(range(len(points))) - used_qvertices)
    explicit_operation_retired_vertices = set()
    operation_lineage = arrangement.get("lineage", {})
    repair_operations = operation_lineage.get("repair_operations", {})
    source_face_sidecar_by_slot = _load_source_face_sidecar_by_slot(
        operation_lineage, stable_id)
    has_frozen_fan_owner = (
        stable_id == 317 and isinstance(operation_lineage.get("prior_fan_repair"), dict)
    ) or (
        stable_id in (313, 315)
        and isinstance(operation_lineage.get("executed_local_lineage_path"), str)
    )
    if has_frozen_fan_owner:
        fan_split_operations, fan_operation_provenance = _load_vertex_fan_split_operations(
            arrangement=arrangement, current_candidate=candidate, stable_id=stable_id)
    else:
        # Small synthetic composer callers have no declared fan edit. Any actual
        # out-of-triangle support still fails below unless a pinned owner exists.
        fan_split_operations, fan_operation_provenance = [], None
    contraction = repair_operations.get("contraction")
    contraction_face_replays = {}
    if stable_id == 313 and "executed_candidate_path" in operation_lineage:
        contraction_face_replays = _load_313_contraction_replay(
            arrangement, candidate)
    tip_retessellation_replays = {}
    tip_retessellation_patches = {}
    if stable_id == 317 and "executed_candidate_path" in operation_lineage:
        tip_retessellation_replays = _load_317_tip_retessellation_replay(
            arrangement, candidate, fan_split_operations=fan_split_operations)
        tip_retessellation_patches = {
            int(operation["new_slot"]): operation
            for operation in repair_operations.get("patches", [])
            if isinstance(operation, dict) and type(operation.get("new_slot")) is int
        }
    if isinstance(contraction, dict):
        explicit_operation_retired_vertices.update(
            int(value) for value in contraction.get("contracted_vertices", []))
    for operation in fan_split_operations:
        if isinstance(operation, dict) and type(operation.get("new_vertex")) is int:
            explicit_operation_retired_vertices.add(int(operation["new_vertex"]))
    for operation in repair_operations.get("retired", []):
        if isinstance(operation, dict):
            explicit_operation_retired_vertices.update(
                int(value) for value in operation.get("face", []))
    unused_source_prefix_count = 0
    unused_documented_retirement_count = 0
    for qvertex in unreferenced_qvertices:
        terms = parent_maps[qvertex]
        if (len(terms) == 1 and terms[0][1] == 1
                and _same_bytes(points[qvertex], source_vertices6[[terms[0][0]], :3], "<f4")):
            unused_source_prefix_count += 1
        else:
            _require(
                qvertex in explicit_operation_retired_vertices,
                f"unreferenced generated vertex {qvertex} lacks an executed local-retirement operation",
            )
            unused_documented_retirement_count += 1
    patch_parent_faces = []
    patch_support = {}
    local_support_replay_proofs = []
    exact_coordinate_alias_proofs = []
    fan_split_operations_by_face: dict[int, list[dict]] = {}
    fan_operation_patches: dict[str, list[int]] = {}
    for operation in fan_split_operations:
        if isinstance(operation, dict) and type(operation.get("new_vertex")) is int:
            for operation_face in operation.get("faces", []):
                fan_split_operations_by_face.setdefault(int(operation_face), []).append(operation)
    contraction_support_proofs: dict[int, dict] = {}
    recorded_contraction_support_faces: set[int] = set()
    sidecar_support_proofs: dict[int, dict] = {}
    recorded_sidecar_support_faces: set[int] = set()
    for slot in patch_slots:
        origin = live_ancestry[slot]
        face_id = origin["parent_face_index"]
        parent_corner_ids = set(map(int, parent_faces[face_id]))
        raw_slot = live_raw_slots[slot]
        for qvertex_value in qfaces[slot]:
            qvertex = int(qvertex_value)
            terms = parent_maps[qvertex]
            support = {source_vertex for source_vertex, _weight in terms}
            exact_alias_proof = None
            if not support.issubset(parent_corner_ids):
                exact_alias_proof = _exact_source_face_coordinate_aliases(
                    source_vertices6=source_vertices6,
                    source_vertex_ids=support,
                    source_face_corner_ids=parent_corner_ids,
                )
            if not support.issubset(parent_corner_ids) and exact_alias_proof is None:
                face_qvertices = set(map(int, qfaces[slot]))
                matching_fan = fan_split_operations_by_face.get(raw_slot, [])
                replay_proof = None
                valid_fan_proofs = []
                rejected_fan_proofs = []
                for fan in matching_fan:
                    operation_sha = _canonical_json_sha256(fan)
                    if operation_sha not in fan_operation_patches:
                        fan_operation_patches[operation_sha] = (
                            _fan_operation_source_patch_face_ids(
                                stable_id=stable_id, operation=fan,
                                live_raw_slots=live_raw_slots, qfaces=qfaces,
                                live_ancestry=live_ancestry,
                                sidecar_by_slot=source_face_sidecar_by_slot,
                                source_face_count=len(parent_faces)))
                    try:
                        fan_proof = _validate_vertex_fan_split_face_support(
                            stable_id=stable_id, face_slot=raw_slot,
                            face_qvertex_ids=face_qvertices, origin=origin,
                            operation=fan, operation_sha256=operation_sha,
                            source_vertices6=source_vertices6,
                            source_faces=parent_faces,
                            source_patch_face_ids=fan_operation_patches[operation_sha],
                            support_vertex_ids=support,
                            raw_parent_maps=raw_parent_maps,
                        )
                    except ValueError as error:
                        rejected_fan_proofs.append(str(error))
                        continue
                    fan_proof["operation_source_provenance"] = fan_operation_provenance
                    valid_fan_proofs.append(fan_proof)
                if matching_fan:
                    _require(bool(valid_fan_proofs),
                             f"{stable_id} face {raw_slot} fan lineage has no valid declared-patch proof: "
                             + "; ".join(rejected_fan_proofs[:3]))
                    valid_fan_proofs.sort(key=lambda proof: proof["operation_sha256"])
                    replay_proof = valid_fan_proofs[0]
                    replay_proof["other_independently_valid_operation_sha256"] = [
                        proof["operation_sha256"] for proof in valid_fan_proofs[1:]
                    ]
                elif raw_slot in contraction_face_replays:
                    if raw_slot not in contraction_support_proofs:
                        patch_proof = dict(contraction_face_replays[raw_slot])
                        _require(list(map(int, qfaces[slot])) == patch_proof["post_face"]
                                 and face_id == patch_proof["source_face_slot"],
                                 "313 contraction support proof does not bind this emitted face")
                        face_support = set().union(*(
                            {source_vertex for source_vertex, _weight in parent_maps[qvertex]}
                            for qvertex in face_qvertices
                        ))
                        patch_proof.update(
                            _validate_313_contraction_source_patch_support(
                                source_vertices6=source_vertices6,
                                source_faces=parent_faces,
                                support_vertex_ids=face_support,
                                proof=patch_proof,
                            )
                        )
                        contraction_support_proofs[raw_slot] = patch_proof
                    replay_proof = contraction_support_proofs[raw_slot]
                elif raw_slot in tip_retessellation_replays:
                    patch = tip_retessellation_patches.get(raw_slot)
                    _require(isinstance(patch, dict),
                             "317 tip-retessellation face lacks its exact local operation")
                    replay_proof = {
                        **tip_retessellation_replays[raw_slot],
                        **_validate_tip_retessellation_support(
                            patch=patch, raw_slot=raw_slot,
                            face=list(map(int, qfaces[slot])), origin=origin,
                            parent_maps=parent_maps, raw_points=raw_points,
                            raw_parent_maps=raw_parent_maps, source_faces=parent_faces,
                            source_vertex_count=len(source_vertices6),
                            replay_proof=tip_retessellation_replays[raw_slot],
                            stable_id=stable_id),
                    }
                elif raw_slot in source_face_sidecar_by_slot:
                    if raw_slot not in sidecar_support_proofs:
                        face_support = set().union(*(
                            {source_vertex for source_vertex, _weight in parent_maps[qvertex]}
                            for qvertex in face_qvertices
                        ))
                        sidecar_support_proofs[raw_slot] = (
                            _validate_sidecar_source_patch_support(
                                stable_id=stable_id,
                                face_slot=raw_slot,
                                origin=origin,
                                sidecar_entry=source_face_sidecar_by_slot[raw_slot],
                                source_vertices6=source_vertices6,
                                source_faces=parent_faces,
                                support_vertex_ids=face_support,
                            )
                        )
                    replay_proof = sidecar_support_proofs[raw_slot]
                _require(
                    replay_proof is not None,
                    f"patch face {raw_slot} has parent support outside its source triangle without its exact executed local operation",
                )
                if (replay_proof.get("kind") == "pinned_313_local_contraction_patch_replay"
                        and raw_slot in contraction_support_proofs):
                    if raw_slot not in recorded_contraction_support_faces:
                        local_support_replay_proofs.append(replay_proof)
                        recorded_contraction_support_faces.add(raw_slot)
                elif replay_proof.get("kind") == "pinned_source_face_sidecar_patch_support":
                    if raw_slot not in recorded_sidecar_support_faces:
                        local_support_replay_proofs.append(replay_proof)
                        recorded_sidecar_support_faces.add(raw_slot)
                else:
                    local_support_replay_proofs.append(replay_proof)
            if exact_alias_proof is not None:
                exact_coordinate_alias_proofs.append({
                    "face_slot": raw_slot,
                    "quotient_vertex": qvertex,
                    **exact_alias_proof,
                })
            patch_support.setdefault(qvertex, set()).update(support)
        patch_parent_faces.append(int(retained_n27_face_ids[face_id]))

    output_vertices = [source_vertices6]
    q_to_output: dict[int, int] = {}
    appended_parent_maps = []
    base_vertex_count = len(source_vertices6)
    for qvertex in sorted(patch_support):
        terms = parent_maps[qvertex]
        if len(terms) == 1 and terms[0][1] == 1:
            source_vertex = terms[0][0]
            _require(
                _same_bytes(points[qvertex], source_vertices6[[source_vertex], :3], "<f4"),
                f"direct parent at quotient vertex {qvertex} has a different source coordinate",
            )
            q_to_output[qvertex] = source_vertex
            continue
        weights = np.asarray([float(weight) for _source_vertex, weight in terms], dtype=np.float64)
        source_slots = [source_vertex for source_vertex, _weight in terms]
        point = points[qvertex]
        normal = np.einsum(
            "i,ij->j",
            weights,
            source_vertices6[source_slots, 3:6].astype(np.float64),
        )
        normal_length = float(np.linalg.norm(normal))
        _require(
            np.isfinite(point).all()
            and np.isfinite(normal).all()
            and normal_length > 1e-12,
            f"positive parent interpolation produced invalid appended vertex {qvertex}",
        )
        appended = np.concatenate(
            (point.astype("<f4"), (normal / normal_length).astype("<f4"))
        )
        q_to_output[qvertex] = base_vertex_count + len(appended_parent_maps)
        appended_parent_maps.append(terms)
        output_vertices.append(appended.reshape(1, 6))
    mapped_patch_faces = np.asarray(
        [
            [q_to_output[int(vertex)] for vertex in qfaces[slot]]
            for slot in patch_slots
        ],
        dtype="<u4",
    ).reshape(-1, 3)
    _require(
        not len(mapped_patch_faces)
        or (
            np.all(mapped_patch_faces[:, 0] != mapped_patch_faces[:, 1])
            and np.all(mapped_patch_faces[:, 1] != mapped_patch_faces[:, 2])
            and np.all(mapped_patch_faces[:, 2] != mapped_patch_faces[:, 0])
        ),
        "parent-map resolution created a repeated-index patch face",
    )
    vertices6 = np.ascontiguousarray(np.concatenate(output_vertices, axis=0), dtype="<f4")
    retained_parent_face_ids_028 = np.asarray(original_parent_faces, dtype=np.int64)
    retained_parent_faces = np.ascontiguousarray(
        parent_faces[retained_parent_face_ids_028], dtype="<u4"
    )
    omitted_parent_face_ids_028 = [
        parent_face for parent_face in range(len(parent_faces))
        if parent_face not in retained_parent_set
    ]
    faces = np.ascontiguousarray(
        np.concatenate((retained_parent_faces, mapped_patch_faces), axis=0), dtype="<u4"
    )
    _require(
        _same_bytes(faces[:len(retained_parent_faces)], retained_parent_faces, "<u4")
        and len(faces) == len(retained_parent_faces) + len(mapped_patch_faces)
        and np.isfinite(vertices6).all()
        and int(faces.max(initial=0)) < len(vertices6)
        and len(vertices6) > len(source_vertices6),
        "arrangement did not emit the exact retained source faces and positive generated children",
    )
    candidate_row = {
        **source_row,
        "vertices6": vertices6,
        "faces": faces,
        "record": (
            int(source_row["record"][0]),
            int(source_row["record"][1]),
            len(vertices6),
            int(source_row["record"][3]),
            len(faces) * 3,
            stable_id,
            int(source_row["record"][6]),
            int(source_row["record"][7]),
        ),
    }
    candidate_row_maps = (
        tuple(((vertex, Fraction(1)),) for vertex in range(base_vertex_count))
        + tuple(appended_parent_maps)
    )
    summary = {
        "typed_identity": [51010, stable_id],
        "source_n27_payload_sha256": N27_PAYLOAD_SHA256,
        "source_n27_row_sha256": sha_bytes(_normalized_row_bytes(source_row)),
        "repair_parent_payload_sha256": REPAIR_PARENT_PAYLOAD_SHA256,
        "repair_parent_face_count": len(parent_faces),
        "retained_n27_face_map_sha256": sha_bytes(
            np.asarray([retained_n27_face_ids[parent_face]
                        for parent_face in original_parent_faces], dtype="<u8").tobytes()),
        "retained_n27_source_face_count": len(original_parent_faces),
        "028_parent_source_face_count": len(parent_faces),
        "retained_028_parent_face_ids": original_parent_faces,
        "omitted_replaced_or_fully_retired_028_parent_face_ids": omitted_parent_face_ids_028,
        "omitted_replaced_or_fully_retired_n27_source_face_ids": [
            int(retained_n27_face_ids[parent_face])
            for parent_face in omitted_parent_face_ids_028
        ],
        "emitted_028_parent_face_count": len(retained_parent_faces),
        "omitted_028_parent_face_count": len(omitted_parent_face_ids_028),
        "retained_parent_faces_emitted_byte_exact_in_original_order": True,
        "active_generated_parent_face_count": len(active_generated_by_parent),
        "fully_retired_028_source_face_count": len(fully_retired_source_parents),
        "full_candidate_sha256": arrangement["candidate_sha256"],
        "lineage_sha256": arrangement["lineage_sha256"],
        "report_sha256": arrangement["report_sha256"],
        "compiled_row_sha256": sha_bytes(_normalized_row_bytes(candidate_row)),
        "compiled_candidate_vertices6_prefix_matches_entire_N27_row": True,
        "all_028_faces_preserved_byte_exact_in_original_order": (
            len(original_slots) == len(parent_faces)
            and not fully_retired_source_parents
            and not active_generated_by_parent),
        "all_retained_origin_kind0_faces_byte_exact_in_original_order": True,
        "explicit_null_retirements_complete": (
            len(retired_by_slot) == len(null_slots)),
        "full_candidate_origin_kind0_face_count": len(original_slots),
        "full_candidate_patch_face_count": len(patch_slots),
        "full_candidate_null_face_slot_count": len(null_slots),
        "unreferenced_candidate_vertex_count": len(unreferenced_qvertices),
        "unreferenced_candidate_vertex_ids_sha256": sha_bytes(
            np.asarray(unreferenced_qvertices, dtype="<u8").tobytes()),
        "unreferenced_source_prefix_vertex_count": unused_source_prefix_count,
        "unreferenced_documented_operation_retirement_count": unused_documented_retirement_count,
        "unreferenced_vertices_bound_to_source_or_executed_local_operation": True,
        "null_face_retirement_ledger_sha256": arrangement.get(
            "null_face_retirement_ledger_sha256"),
        "null_face_retirement_classification_counts": {
            classification: sum(
                entry["classification"] == classification
                for entry in retirement_ledger["retirements"])
            for classification in (
                "superseded_by_active_generated_child",
                "fully_retired_local_arrangement")
        } if retirement_ledger is not None else {},
        "patch_parent_face_ids_in_N27_namespace": patch_parent_faces,
        "local_source_support_replay_proofs": local_support_replay_proofs,
        "exact_float32_coordinate_alias_support_proofs": exact_coordinate_alias_proofs,
        "new_appended_vertex_count": len(appended_parent_maps),
        "positive_exact_parent_map_for_each_appended_vertex": True,
        "new_child_normals": "Float64 weighted interpolation of exact positive N27 parent normals, unit-normalized and emitted as Float32",
        "point_coordinate_contract": {
            "full_candidate_points_exact_frame": "captured_step0_world_F32",
            "abi5_emitted_child_positions_frame": "NHA_source_rest_local_F32",
            "parent_weights": "exact positive rationals",
            "rational_to_float32": "direct IEEE-754 round-to-nearest-ties-to-even; no intermediate Float64",
            "captured_reference_validation": captured_source_xyz is not None,
        },
        "physical_mass_or_state_added": False,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
        "native_confirmation_required": True,
        "candidate_vertex_parent_maps": candidate_row_maps,
        "candidate_faces": faces,
    }
    return candidate_row, summary




def encode_payload_with_row_replacements(raw: bytes, replacements: dict[int, dict]) -> tuple[bytes, dict]:
    """Re-emit ABI5 with only selected local rows replaced and offsets rebased."""
    from .resting_anatomy_interface_patch import encode_payload
    from .resting_pleura_proxy import _parse_payload

    header, _records, original_rows = _parse_payload(raw)
    _require(set(map(int, replacements)).issubset(original_rows),
             "row replacement targets a stable ID absent from the ABI5 parent")
    updated_rows = dict(original_rows)
    for stable_id, row in replacements.items():
        _require(_row_identity(row, int(stable_id)) == _row_identity(
            original_rows[int(stable_id)], int(stable_id)),
            f"replacement changed body/layer/flags/stable ID for {stable_id}")
        updated_rows[int(stable_id)] = row
    output = encode_payload(header, updated_rows)
    decoded_header, _new_records, decoded_rows = _parse_payload(output)
    _require(tuple(decoded_header[5:]) == tuple(header[5:])
             and len(decoded_rows) == len(original_rows)
             and list(decoded_rows) == list(original_rows),
             "ABI5 re-emit changed source header identity or stable-ID order")
    for stable_id, row in original_rows.items():
        expected = replacements.get(stable_id, row)
        _require(_normalized_row_bytes(expected)
                 == _normalized_row_bytes(decoded_rows[stable_id]),
                 f"ABI5 re-emit changed row content for stable ID {stable_id}")
    return output, {
        "stable_id_order_preserved": True,
        "row_count_preserved": len(original_rows),
        "replaced_stable_ids": sorted(map(int, replacements)),
        "unmodified_rows_byte_exact": len(original_rows) - len(replacements),
        "global_offsets_rebased_by_existing_abi5_encoder": True,
        "registration_and_source_header_bytes_preserved": True,
    }



def _validate_route110_source_declaration(route_decl: dict, route_data: dict, pins: dict) -> None:
    """Bind the append receipt to the exact offline source-transfer inputs."""
    _require(
        route_decl.get("candidate_npz_sha256") == pins["npz"]
        and route_decl.get("candidate_report_sha256") == pins["report"]
        and route_decl.get("candidate_runner_sha256") == pins["runner"]
        and route_decl.get("target_preparation_checkpoint_sha256") == pins["checkpoint"]
        and route_decl.get("retained_candidate_status") == route_data["retained_status"]
        and route_decl.get("exact_candidate_checks") == 0
        and route_decl.get("native_or_gpu_run") is False
        and route_decl.get("records") == route_data["records"],
        "route110 source/report/code pins or per-row ancestry differ",
    )


_CAPTURE_DECODER_CONTEXT_KEYS = frozenset({
    "capture_loader_source_path",
    "capture_loader_source_sha256",
})


def _capture_reference_identity(reference: dict) -> dict:
    """Return the frozen capture/geometry identity without decoder code context."""
    _require(isinstance(reference, dict), "captured reference is not an object")
    return {key: value for key, value in reference.items()
            if key not in _CAPTURE_DECODER_CONTEXT_KEYS}


def _historical_capture_loader_commit(path: str, expected_sha256: str) -> str:
    """Prove historical decoder bytes by owner-relative Git identity.

    Historical receipts record an absolute checkout path. That path is provenance,
    not a requirement that the old checkout still exist or share the current
    checkout's location. Bind the suffix to the one permitted decoder owner and
    find its exact bytes in this repository's immutable Git objects.
    """
    repo_root = Path(__file__).resolve().parents[2]
    normalized = PurePosixPath(str(path).replace("\\", "/"))
    owner_parts = ("src", "numilab_human", "resting_anatomy_verify.py")
    if ".." in normalized.parts or normalized.parts[-3:] != owner_parts:
        raise ValueError(
            "local intercostal arrangement: historical capture decoder path has the wrong repository owner"
        )
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)):
        raise ValueError(
            "local intercostal arrangement: historical capture decoder SHA is malformed"
        )
    relative = "/".join(owner_parts)
    commits = subprocess.run(
        ["git", "-C", str(repo_root), "log", "--all", "--format=%H", "--", relative],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    for commit in commits:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative}"],
            check=False, capture_output=True,
        )
        if result.returncode == 0 and sha_bytes(result.stdout) == expected_sha256:
            return commit
    raise ValueError(
        "local intercostal arrangement: historical capture decoder SHA is not present in Git history"
    )

def _validate_route110_reemit_proofs(
    route_decl: dict,
    vessel_blend: dict,
    expected_route_replaced: list[int],
    expected_row_count: int,
) -> dict:
    """Validate the route-wide ABI5 replay separately from a scoped FJ2217 edit."""
    route_reemit = route_decl.get("abi5_reemit_proof")
    aggregate_reemit = vessel_blend.get("abi5_retirement_reemit_proof")
    reinstatement = vessel_blend.get("local_source_face_reinstatement")
    scoped_reemit = vessel_blend.get("abi5_reinstatement_reemit_proof")
    _require(
        isinstance(route_reemit, dict)
        and isinstance(aggregate_reemit, dict)
        and route_decl.get("replaced_stable_ids") == expected_route_replaced
        and route_reemit.get("replaced_stable_ids") == expected_route_replaced
        and route_reemit.get("stable_id_order_preserved") is True
        and route_reemit.get("global_offsets_rebased_by_existing_abi5_encoder") is True
        and route_reemit.get("registration_and_source_header_bytes_preserved") is True
        and route_reemit.get("row_count_preserved") == expected_row_count
        and route_reemit.get("unmodified_rows_byte_exact")
        == expected_row_count - len(expected_route_replaced)
        and aggregate_reemit == route_reemit,
        "route110 ABI5 aggregate replacement set or owner proof differs",
    )
    if reinstatement is not None:
        _require(
            isinstance(scoped_reemit, dict)
            and scoped_reemit.get("replaced_stable_ids") == [477]
            and scoped_reemit.get("row_count_preserved") == expected_row_count
            and scoped_reemit.get("unmodified_rows_byte_exact") == expected_row_count - 1
            and scoped_reemit.get("stable_id_order_preserved") is True
            and scoped_reemit.get("global_offsets_rebased_by_existing_abi5_encoder") is True
            and scoped_reemit.get("registration_and_source_header_bytes_preserved") is True,
            "FJ2217 reinstatement proof is not scoped to its one exact row",
        )
    else:
        _require(
            scoped_reemit is None,
            "FJ2217 reinstatement reemit proof has no corresponding operation",
        )
    return route_reemit


def _load_source004_index_only_parent(
    *, receipt: dict, candidate_rows: dict, aggregate: dict
) -> dict:
    """Verify the one pinned source004 child that changes only intercostal faces."""
    from .resting_pleura_proxy import _parse_payload

    declaration = aggregate.get("source004_index_only_reemit")
    _require(
        isinstance(declaration, dict)
        and declaration.get("schema") == "numi.human.intercostal.source004-index-only-reemit.v1"
        and declaration.get("source004_receipt_sha256") == SOURCE004_INDEX_PARENT_RECEIPT_SHA256
        and declaration.get("source004_payload_sha256") == SOURCE004_INDEX_PARENT_PAYLOAD_SHA256
        and declaration.get("candidate_payload_sha256") == SOURCE004_INDEX_CHILD_PAYLOAD_SHA256
        and declaration.get("candidate_payload_source_sha256")
        == SOURCE004_INDEX_CHILD_PAYLOAD_SHA256
        and declaration.get("composition_report_sha256") == SOURCE004_INDEX_REPORT_SHA256
        and declaration.get("arrangement_descriptors_sha256") == SOURCE004_INDEX_DESCRIPTORS_SHA256
        and declaration.get("changed_stable_ids") == sorted(REPAIRED_IDS),
        "source004 index-only reemit declaration differs from pinned inputs",
    )
    parent_receipt_path = Path(str(declaration.get("source004_receipt_path", ""))).resolve()
    parent_payload_path = Path(str(declaration.get("source004_payload_path", ""))).resolve()
    candidate_payload_path = Path(str(declaration.get("candidate_payload_path", ""))).resolve()
    candidate_payload_source_path = Path(
        str(declaration.get("candidate_payload_source_path", ""))
    ).resolve()
    report_path = Path(str(declaration.get("composition_report_path", ""))).resolve()
    descriptor_path = Path(str(declaration.get("arrangement_descriptors_path", ""))).resolve()
    _require(
        parent_receipt_path.is_file()
        and sha_file(parent_receipt_path) == SOURCE004_INDEX_PARENT_RECEIPT_SHA256
        and parent_payload_path.is_file()
        and sha_file(parent_payload_path) == SOURCE004_INDEX_PARENT_PAYLOAD_SHA256
        and report_path.is_file()
        and sha_file(report_path) == SOURCE004_INDEX_REPORT_SHA256
        and descriptor_path.is_file()
        and sha_file(descriptor_path) == SOURCE004_INDEX_DESCRIPTORS_SHA256
        and candidate_payload_path.is_file()
        and sha_file(candidate_payload_path) == SOURCE004_INDEX_CHILD_PAYLOAD_SHA256
        and candidate_payload_source_path.is_file()
        and sha_file(candidate_payload_source_path) == SOURCE004_INDEX_CHILD_PAYLOAD_SHA256
        and Path(receipt.get("payload", {}).get("path", "")).resolve() == candidate_payload_path
        and receipt.get("payload", {}).get("sha256") == SOURCE004_INDEX_CHILD_PAYLOAD_SHA256,
        "source004 index-only parent, candidate, or preparation bytes changed",
    )
    try:
        parent_receipt = json.loads(parent_receipt_path.read_bytes())
        report = json.loads(report_path.read_bytes())
        descriptors = json.loads(descriptor_path.read_bytes())
        parent_header, _, parent_rows = _parse_payload(parent_payload_path.read_bytes())
        child_header, _, child_rows = _parse_payload(candidate_payload_path.read_bytes())
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("source004 index-only parent/candidate inputs are unreadable") from error
    parent_payload = parent_receipt.get("payload", {})
    _require(
        descriptors.get("schema") == "numi.human.local-intercostal-arrangement-inputs.v1"
        and descriptors.get("status")
        == "IMMUTABLE_046_DESCRIPTOR_SUCCESSOR_METADATA_ONLY_NOT_ADMISSION"
        and descriptors.get("candidate_geometry_bytes_unchanged") is True
        and descriptors.get("source_payload_index") == 1
        and descriptors.get("source_payload_sha256") == REPAIR_PARENT_PAYLOAD_SHA256
        and set(descriptors.get("rows", {})) == {str(stable_id) for stable_id in REPAIRED_IDS}
        and report.get("row_summaries", {}).keys() == descriptors.get("rows", {}).keys(),
        "source004 046 arrangement descriptor identity differs",
    )
    for stable_id in REPAIRED_IDS:
        key = str(stable_id)
        descriptor = descriptors["rows"][key]
        summary = report["row_summaries"][key]
        _require(
            descriptor.get("source_payload_index") == descriptors["source_payload_index"]
            and descriptor.get("source_payload_sha256") == descriptors["source_payload_sha256"]
            and descriptor.get("candidate_sha256") == summary.get("full_candidate_sha256")
            and descriptor.get("lineage_sha256") == summary.get("lineage_sha256")
            and descriptor.get("report_sha256") == summary.get("report_sha256")
            and descriptor.get("null_face_retirement_ledger_sha256")
            == summary.get("null_face_retirement_ledger_sha256")
            and all(
                isinstance(descriptor.get(path_key), str)
                and isinstance(descriptor.get(sha_key), str)
                and Path(descriptor[path_key]).is_file()
                and sha_file(Path(descriptor[path_key]).resolve()) == descriptor[sha_key]
                for path_key, sha_key in (
                    ("candidate_path", "candidate_sha256"),
                    ("lineage_path", "lineage_sha256"),
                    ("report_path", "report_sha256"),
                    ("null_face_retirement_ledger_path", "null_face_retirement_ledger_sha256"),
                )
            ),
            f"source004 row {stable_id} descriptor lineage or source file differs",
        )
    parent_payload = parent_receipt.get("payload", {})
    _require(
        parent_payload.get("abi") == 5
        and parent_payload.get("sha256") == SOURCE004_INDEX_PARENT_PAYLOAD_SHA256
        and len(parent_rows) == len(child_rows) == 525
        and list(parent_rows) == list(child_rows) == list(candidate_rows)
        and parent_header[0:4] == child_header[0:4]
        and parent_header[5:7] == child_header[5:7]
        and child_header[4] == sum(len(row["faces"]) * 3 for row in child_rows.values())
        and report.get("changed_row_ids") == sorted(REPAIRED_IDS)
        and report.get("geometry_predicates_run") is False
        and report.get("native_or_gpu_run") is False
        and report.get("candidate_payload", {}).get("sha256") == SOURCE004_INDEX_CHILD_PAYLOAD_SHA256
        and Path(report.get("candidate_payload", {}).get("path", "")).resolve()
        == candidate_payload_source_path
        and report.get("inputs", {}).get("source004_receipt", {}).get("sha256")
        == SOURCE004_INDEX_PARENT_RECEIPT_SHA256
        and report.get("inputs", {}).get("source004_payload", {}).get("sha256")
        == SOURCE004_INDEX_PARENT_PAYLOAD_SHA256
        and report.get("inputs", {}).get("arrangement_descriptors_046", {}).get("sha256")
        == SOURCE004_INDEX_DESCRIPTORS_SHA256,
        "source004 index-only ABI5 row/header/report identity differs",
    )
    reemit = declaration.get("abi5_reemit_proof")
    _require(
        isinstance(reemit, dict)
        and reemit.get("replaced_stable_ids") == sorted(REPAIRED_IDS)
        and reemit.get("row_count_preserved") == len(parent_rows)
        and reemit.get("unmodified_rows_byte_exact") == len(parent_rows) - len(REPAIRED_IDS)
        and reemit.get("stable_id_order_preserved") is True
        and reemit.get("global_offsets_rebased_by_existing_abi5_encoder") is True
        and reemit.get("registration_and_source_header_bytes_preserved") is True
        and reemit == report.get("abi5_reemit_proof"),
        "source004 index-only ABI5 row-replacement proof is incomplete",
    )
    for stable_id in parent_rows:
        before, after = parent_rows[stable_id], child_rows[stable_id]
        _require(
            _row_identity(before, stable_id) == _row_identity(after, stable_id)
            and _same_bytes(before["vertices6"], after["vertices6"], "<f4"),
            f"source004 index-only child changed row {stable_id} identity or vertex/normal bytes",
        )
        if stable_id in REPAIRED_IDS:
            _require(
                not _same_bytes(before["faces"], after["faces"], "<u4"),
                f"source004 declared index repair has no face-index delta at {stable_id}",
            )
        else:
            _require(
                _normalized_row_bytes(before) == _normalized_row_bytes(after),
                f"source004 index-only child changed unrelated row {stable_id}",
            )
        _require(
            _normalized_row_bytes(after) == _normalized_row_bytes(candidate_rows[stable_id]),
            f"source004 parsed child differs from supplied row {stable_id}",
        )
    parent_source_map = parent_receipt.get("provenance", {}).get("source_id_map", {})
    candidate_source_map = receipt.get("provenance", {}).get("source_id_map", {})
    _require(
        set(parent_source_map) == set(candidate_source_map)
        and set(parent_source_map) == {str(stable_id) for stable_id in child_rows},
        "source004 index-only source-ID map inventory differs",
    )
    for stable_id in child_rows:
        key = str(stable_id)
        if stable_id in REPAIRED_IDS:
            before = deepcopy(parent_source_map[key])
            after = deepcopy(candidate_source_map[key])
            before_repair = before.get("repair", {}) or {}
            after_repair = after.get("repair", {}) or {}
            before_local = before_repair.pop("local_intersection_remesh", None)
            after_local = after_repair.pop("local_intersection_remesh", None)
            _require(
                before == after
                and before_repair == after_repair
                and isinstance(before_local, dict)
                and isinstance(after_local, dict),
                f"source004 index-only source identity changed at stable ID {stable_id}",
            )
        else:
            _require(
                parent_source_map[key] == candidate_source_map[key],
                f"source004 index-only source-ID map changed at stable ID {stable_id}",
            )
    return {
        "rows": parent_rows,
        "source_id_map": parent_source_map,
        "proof": reemit,
        "audit": {
            "method": "pinned_source004_ABI5_child_with_only_313_315_317_face_index_replacements",
            "source004_receipt_sha256": SOURCE004_INDEX_PARENT_RECEIPT_SHA256,
            "source004_payload_sha256": SOURCE004_INDEX_PARENT_PAYLOAD_SHA256,
            "candidate_payload_sha256": SOURCE004_INDEX_CHILD_PAYLOAD_SHA256,
            "candidate_payload_source_path": str(candidate_payload_source_path),
            "candidate_payload_source_sha256": SOURCE004_INDEX_CHILD_PAYLOAD_SHA256,
            "composition_report_sha256": SOURCE004_INDEX_REPORT_SHA256,
            "arrangement_descriptors_sha256": SOURCE004_INDEX_DESCRIPTORS_SHA256,
            "all_other_source004_rows_byte_exact": True,
            "repaired_row_vertex_and_normal_bytes_exact": True,
            "source_id_map_unchanged_outside_repaired_row_pointers": True,
        },
    }


def _resolve_historical_fj2217_retirement_reemit_proof(
    *,
    receipt_provenance: dict,
    vessel_blend: dict,
    historical_retirement: dict,
    reinstatement: dict,
    route_reemit_proof: dict,
    expected_route_row: dict,
    candidate_rows: dict,
) -> tuple[dict, dict, dict, dict]:
    """Resolve the omitted cap-history proof from its exact ABI5 parent receipt."""
    from .resting_pleura_proxy import _parse_payload
    from .resting_vessel_owner_blend import (
        FJ2217_CAP_SOURCE_FACE_ID,
        _retire_isolated_source_face,
        validate_local_source_face_retirement,
    )

    cap = receipt_provenance.get("axillary_477_cap_reinstatement_successor")
    _require(
        isinstance(cap, dict)
        and cap.get("schema") == "numi.human.axillary-477-cap-reinstatement-successor.v1",
        "FJ2217 reinsertion lacks its pinned cap-successor provenance",
    )
    parent_receipt_path = Path(str(cap.get("base_receipt_path", ""))).resolve()
    parent_receipt_sha = cap.get("base_receipt_sha256")
    parent_payload_sha = cap.get("base_payload_sha256")
    _require(
        parent_receipt_sha == FJ2217_CAP_PARENT_RECEIPT_SHA256
        and parent_payload_sha == FJ2217_CAP_PARENT_PAYLOAD_SHA256
        and parent_receipt_path.is_file()
        and sha_file(parent_receipt_path) == FJ2217_CAP_PARENT_RECEIPT_SHA256,
        "FJ2217 cap parent receipt is not the exact pinned a779 source",
    )
    try:
        parent_receipt = json.loads(parent_receipt_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("FJ2217 pinned cap parent receipt is unreadable") from error
    parent_payload = parent_receipt.get("payload", {})
    parent_payload_path = Path(str(parent_payload.get("path", ""))).resolve()
    _require(
        parent_payload.get("abi") == 5
        and parent_payload.get("sha256") == FJ2217_CAP_PARENT_PAYLOAD_SHA256
        and parent_payload_path.is_file()
        and sha_file(parent_payload_path) == FJ2217_CAP_PARENT_PAYLOAD_SHA256,
        "FJ2217 cap parent payload does not match its pinned ABI5 receipt",
    )
    try:
        parent_header, _, parent_rows = _parse_payload(parent_payload_path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError("FJ2217 cap parent payload is not a valid ABI5 source") from error

    parent_blend = (
        parent_receipt.get("provenance", {})
        .get("passive_brachiocephalic_vein_extension", {})
        .get("passive_tissue_owner_blend", {})
    )
    parent_retirement = parent_blend.get("local_source_face_retirement")
    parent_reemit = parent_blend.get("abi5_retirement_reemit_proof")
    _require(
        parent_retirement == historical_retirement
        and isinstance(parent_reemit, dict)
        and parent_reemit == route_reemit_proof
        and parent_reemit == vessel_blend.get("abi5_retirement_reemit_proof")
        and parent_reemit.get("row_count_preserved") == len(parent_rows)
        and parent_payload.get("surface_count") == len(parent_rows)
        and 477 in parent_rows,
        "FJ2217 parent retirement declaration or aggregate ABI5 proof differs",
    )

    source_faces = np.ascontiguousarray(expected_route_row.get("faces"), dtype="<u4")
    _require(
        0 <= FJ2217_CAP_SOURCE_FACE_ID < len(source_faces),
        "FJ2217 route source row lacks its pinned cap face",
    )
    reconstructed_parent_row, _ = _retire_isolated_source_face(
        expected_route_row,
        FJ2217_CAP_SOURCE_FACE_ID,
        source_faces[FJ2217_CAP_SOURCE_FACE_ID],
    )
    validate_local_source_face_retirement(
        expected_route_row,
        reconstructed_parent_row,
        historical_retirement,
        parent_reemit,
        expected_pre_retirement_row=expected_route_row,
    )
    _require(
        _normalized_row_bytes(parent_rows[477])
        == _normalized_row_bytes(reconstructed_parent_row),
        "pinned a779 row 477 differs from exact source-face retirement replay",
    )

    candidate_payload_path = Path(str(cap.get("candidate_payload_path", ""))).resolve()
    _require(
        cap.get("candidate_payload_sha256") == FJ2217_CAP_CANDIDATE_PAYLOAD_SHA256
        and candidate_payload_path.is_file()
        and sha_file(candidate_payload_path) == FJ2217_CAP_CANDIDATE_PAYLOAD_SHA256,
        "FJ2217 cap candidate is not the exact pinned reinstatement output",
    )
    try:
        candidate_header, _, reinstated_rows = _parse_payload(candidate_payload_path.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError("FJ2217 reinstatement candidate is not a valid ABI5 source") from error
    scoped_reemit = vessel_blend.get("abi5_reinstatement_reemit_proof")
    parent_record = tuple(map(int, parent_rows[477]["record"]))
    cap_record = tuple(map(int, reinstated_rows[477]["record"]))
    _require(
        cap.get("reinstatement") == reinstatement
        and cap.get("abi5_reemit_proof") == scoped_reemit
        and isinstance(scoped_reemit, dict)
        and scoped_reemit.get("replaced_stable_ids") == [477]
        and candidate_header[0:4] == parent_header[0:4]
        and candidate_header[4] == parent_header[4] + 3
        and candidate_header[5:7] == parent_header[5:7]
        and len(reinstated_rows) == len(parent_rows)
        and cap_record[:4] == parent_record[:4]
        and cap_record[4] == parent_record[4] + 3
        and cap_record[5:] == parent_record[5:] == (477, 2, 0)
        and 477 in candidate_rows
        and _normalized_row_bytes(reinstated_rows[477])
        == _normalized_row_bytes(candidate_rows[477]),
        "pinned FJ2217 cap payload, scoped proof, or current row differs",
    )
    proof_sha256 = sha_bytes(
        json.dumps(parent_reemit, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )
    return parent_reemit, {
        "method": "resolve_exact_abi5_retirement_proof_from_cap_pinned_parent_receipt",
        "parent_receipt_sha256": FJ2217_CAP_PARENT_RECEIPT_SHA256,
        "parent_payload_sha256": FJ2217_CAP_PARENT_PAYLOAD_SHA256,
        "parent_row_477_matches_exact_retirement_reconstruction": True,
        "parent_aggregate_proof_sha256": proof_sha256,
        "cap_candidate_payload_sha256": FJ2217_CAP_CANDIDATE_PAYLOAD_SHA256,
        "cap_candidate_row_477_matches_current_candidate": True,
    }, parent_rows[477], reinstated_rows[477]


def validate_receipt_composition(
    *,
    base_receipt_path: Path,
    base_payload_sha256: str,
    receipt: dict,
    candidate_rows: dict,
) -> dict:
    """Validate one existing ABI5 append receipt with only the three local row repairs."""
    from .resting_pleura_proxy import _parse_payload

    provenance = receipt.get("provenance", {})
    aggregate = provenance.get("local_intercostal_mesh_repair")
    _require(isinstance(aggregate, dict)
             and aggregate.get("schema") == "numi.human.local_intercostal_mesh_repair.v1",
             "NHA receipt lacks the scoped local intercostal repair declaration")
    _require(Path(aggregate.get("base_receipt_path", "")).resolve()
             == Path(base_receipt_path).resolve()
             and aggregate.get("base_receipt_sha256") == N27_RECEIPT_SHA256
             and aggregate.get("base_payload_sha256") == base_payload_sha256 == N27_PAYLOAD_SHA256,
             "local intercostal repair is not bound to the exact direct N27 parent")
    context = load_pinned_fragment_repair(
        base_receipt_path=base_receipt_path,
        base_payload_sha256=base_payload_sha256,
        repair_parent_receipt_path=Path(aggregate.get("repair_parent_receipt_path", "")),
        repair_parent_receipt_sha256=aggregate.get("repair_parent_receipt_sha256"),
        repair_parent_payload_path=Path(aggregate.get("repair_parent_payload_path", "")),
        repair_report_path=Path(aggregate.get("repair_report_path", "")),
        repair_report_sha256=aggregate.get("repair_report_sha256"),
    )
    _require(provenance.get("local_intercostal_source_fragment_repair")
             == context["fragment_repair_declaration"],
             "append receipt dropped or changed the earlier 028 fragment-repair declaration")
    captured_reference = aggregate.get("captured_reference", {})
    _require(isinstance(captured_reference, dict)
             and captured_reference.get("capture_manifest_sha256") == N27_CAPTURE_MANIFEST_SHA256,
             "append receipt lacks the pinned step-0 captured-world reference")
    reemit = aggregate.get("abi5_payload_reemit_proof", {})
    _require(aggregate.get("unrelated_n27_rows_byte_exact") is True
             and aggregate.get("unchanged_original_vertex_slots_preserved") is True
             and aggregate.get("no_physical_mass_or_state_added") is True
             and reemit.get("stable_id_order_preserved") is True
             and reemit.get("global_offsets_rebased_by_existing_abi5_encoder") is True
             and reemit.get("registration_and_source_header_bytes_preserved") is True
             and reemit.get("replaced_stable_ids") == sorted(REPAIRED_IDS)
             and reemit.get("unmodified_rows_byte_exact") == 522,
             "aggregate repair declaration omits its scoped ABI5 preservation proof")
    source_map = provenance.get("source_id_map", {})
    source004_index_only = None
    if "source004_index_only_reemit" in aggregate:
        source004_index_only = _load_source004_index_only_parent(
            receipt=receipt, candidate_rows=candidate_rows, aggregate=aggregate
        )
    base_receipt = json.loads(Path(base_receipt_path).read_text())
    base_map = base_receipt.get("provenance", {}).get("source_id_map", {})
    _, _, base_rows = _parse_payload(Path(base_receipt["payload"]["path"]).read_bytes())
    _require(set(base_rows).issubset(candidate_rows)
             and set(candidate_rows) == set(base_rows) | {525, 526},
             "append candidate stable-ID inventory differs from N27 plus bridge rows")
    records = aggregate.get("rows", [])
    _require(isinstance(records, list)
             and [int(record.get("stable_id", -1)) for record in records] == sorted(REPAIRED_IDS),
             "aggregate repair rows do not bind exactly stable IDs 313/315/317")
    descriptors = {}
    for record in records:
        stable_id = int(record["stable_id"])
        descriptor = record.get("full_candidate", {})
        _require(isinstance(descriptor, dict), f"missing full_candidate descriptor for {stable_id}")
        descriptors[stable_id] = {
            "candidate_path": descriptor.get("candidate_path"),
            "candidate_sha256": descriptor.get("candidate_sha256"),
            "lineage_path": descriptor.get("lineage_path"),
            "lineage_sha256": descriptor.get("lineage_sha256"),
            "report_path": descriptor.get("report_path"),
            "report_sha256": descriptor.get("report_sha256"),
            "null_face_retirement_ledger_path": descriptor.get(
                "null_face_retirement_ledger_path"),
            "null_face_retirement_ledger_sha256": descriptor.get(
                "null_face_retirement_ledger_sha256"),
            "source_payload_sha256": descriptor.get("source_payload_sha256"),
            "source_payload_index": descriptor.get("source_payload_index"),
        }
    composed, summaries = compose_pinned_arrangement_set(
        base_rows=base_rows, repair_context=context, arrangement_descriptors=descriptors,
        capture_manifest_path=Path(captured_reference["capture_manifest_path"]),
        capture_manifest_sha256=captured_reference["capture_manifest_sha256"])
    reconstructed_capture_reference = {
        **summaries["313"]["captured_reference_shared"],
        "rows": {str(stable_id): summaries[str(stable_id)]["captured_reference_row"]
                 for stable_id in sorted(REPAIRED_IDS)},
    }
    historical_loader_path = captured_reference.get("capture_loader_source_path")
    historical_loader_sha = captured_reference.get("capture_loader_source_sha256")
    _require(isinstance(historical_loader_path, str)
             and isinstance(historical_loader_sha, str)
             and len(historical_loader_sha) == 64,
             "append receipt lacks the historical capture decoder identity")
    historical_loader_commit = _historical_capture_loader_commit(
        historical_loader_path, historical_loader_sha)
    current_loader_path = Path(__file__).with_name("resting_anatomy_verify.py").resolve()
    current_loader_sha = sha_file(current_loader_path)
    _require(reconstructed_capture_reference.get("capture_loader_source_path")
             == str(current_loader_path)
             and reconstructed_capture_reference.get("capture_loader_source_sha256")
             == current_loader_sha,
             "current capture decoder execution identity is malformed")
    _require(_capture_reference_identity(reconstructed_capture_reference)
             == _capture_reference_identity(captured_reference),
             "captured-world source rows, pack identity, or pose differ from the append receipt")
    capture_decoder_provenance = {
        "historical_decoder_path": historical_loader_path,
        "historical_decoder_sha256": historical_loader_sha,
        "historical_decoder_git_commit": historical_loader_commit,
        "current_execution_decoder_path": str(current_loader_path),
        "current_execution_decoder_sha256": current_loader_sha,
        "captured_world_pack_pose_and_row_identity_exact": True,
    }
    extension = provenance.get("passive_brachiocephalic_vein_extension", {})
    vessel_blend = extension.get("passive_tissue_owner_blend", {})
    route_decl = extension.get("route110_reference_field")
    _require(isinstance(route_decl, dict)
             and route_decl.get("schema") == "numi.human.route110.registered-reference-row-transfer.v1",
             "append receipt lacks the exact route110 source-row transfer declaration")
    from .resting_vessel_owner_blend import (
        ROUTE110_PINS,
        load_route110_reference_rows,
        validate_fj2217_route_face_operation,
    )
    route_paths = route_decl.get("input_paths", {})
    route_data = load_route110_reference_rows(
        n27_receipt_path=base_receipt_path,
        base_rows=base_rows,
        route_npz_path=Path(route_paths.get("npz", "")),
        route_report_path=Path(route_paths.get("report", "")),
        route_script_path=Path(route_paths.get("runner", "")),
        route_checkpoint_path=Path(route_paths.get("checkpoint", "")),
    )
    _validate_route110_source_declaration(route_decl, route_data, ROUTE110_PINS)
    route_changed_ids = {
        int(record["stable_id"])
        for record in route_data["records"]
        if record["xyz_changed"] or record["normals_changed"]
    }
    expected_route_replaced = sorted(route_changed_ids | {525})
    reinstatement = vessel_blend.get("local_source_face_reinstatement")
    active_retirement = vessel_blend.get("local_source_face_retirement")
    historical_retirement = vessel_blend.get("historical_local_source_face_retirement")
    historical_retirement_reemit = vessel_blend.get(
        "historical_abi5_retirement_reemit_proof"
    )
    historical_retirement_proof_resolution = None
    reinstatement_parent_row = None
    reinstatement_candidate_row = None
    route_reemit = _validate_route110_reemit_proofs(
        route_decl, vessel_blend, expected_route_replaced, len(candidate_rows)
    )
    if (
        isinstance(reinstatement, dict)
        and reinstatement.get("operation")
        == "reinsert_retired_source_face_at_original_order_position"
    ):
        (
            resolved_proof,
            historical_retirement_proof_resolution,
            reinstatement_parent_row,
            reinstatement_candidate_row,
        ) = _resolve_historical_fj2217_retirement_reemit_proof(
                receipt_provenance=provenance,
                vessel_blend=vessel_blend,
                historical_retirement=historical_retirement,
                reinstatement=reinstatement,
                route_reemit_proof=route_reemit,
                expected_route_row=route_data["rows"][477],
                candidate_rows=candidate_rows,
            )
        _require(
            historical_retirement_reemit is None
            or historical_retirement_reemit == resolved_proof,
            "inline FJ2217 historical proof differs from its pinned parent proof",
        )
        historical_retirement_reemit = resolved_proof
    bridge = route_data["bridge_rows"][525]
    _require(
        candidate_rows[525]["vertices6"][:, :3].astype("<f4").tobytes()
        == np.ascontiguousarray(bridge["after_local_m"], dtype="<f4").tobytes()
        and candidate_rows[525]["vertices6"][:, 3:].astype("<f4").tobytes()
        == np.ascontiguousarray(bridge["normals_after"], dtype="<f4").tobytes()
        and candidate_rows[525]["faces"].astype("<u4").tobytes()
        == np.ascontiguousarray(bridge["faces"], dtype="<u4").tobytes(),
        "appended FJ3482 output differs from pinned route110 source row",
    )
    compiler_binding = route_decl.get("bridge_compiler_binding", {})
    _require(
        compiler_binding.get("stable_id") == 525
        and compiler_binding.get("source_member") == "FJ3482"
        and compiler_binding.get("compiler_output_matches_route110_before_exact") is True
        and compiler_binding.get("existing_compiler_before_xyz_sha256")
        == hashlib.sha256(np.ascontiguousarray(bridge["before_local_m"], dtype="<f4").tobytes()).hexdigest()
        and compiler_binding.get("existing_compiler_faces_sha256")
        == hashlib.sha256(np.ascontiguousarray(bridge["faces"], dtype="<u4").tobytes()).hexdigest(),
        "route110 bridge compiler input binding is incomplete",
    )
    expected_route_rows = route_data["rows"]
    for stable_id in sorted(route_changed_ids - {477}):
        _require(
            _normalized_row_bytes(candidate_rows[stable_id])
            == _normalized_row_bytes(expected_route_rows[stable_id]),
            f"route110 source-row transfer differs at stable ID {stable_id}",
        )
    cap_operation_proof = None
    legacy_retirement_rows = {}
    if reinstatement is not None or active_retirement is not None:
        _require(
            477 in route_changed_ids,
            "FJ2217 cap face operation lacks route110 row transfer",
        )
        cap_expected_route_row = expected_route_rows[477]
        if reinstatement_candidate_row is not None:
            cap_expected_route_row = {
                **expected_route_rows[477],
                "record": reinstatement_candidate_row["record"],
            }
        cap_operation_proof = validate_fj2217_route_face_operation(
            reinstatement_parent_row if reinstatement_parent_row is not None else base_rows[477],
            reinstatement_candidate_row if reinstatement_candidate_row is not None else candidate_rows[477],
            cap_expected_route_row,
            reinstatement=reinstatement,
            active_retirement=active_retirement,
            historical_retirement=historical_retirement,
            route_reemit_proof=route_reemit,
            historical_retirement_reemit_proof=historical_retirement_reemit,
        )
        if cap_operation_proof["receipt_operation"] == "legacy_scoped_retirement_replay":
            legacy_retirement_rows[477] = cap_operation_proof
    else:
        _require(
            477 not in route_changed_ids
            and _normalized_row_bytes(base_rows[477]) == _normalized_row_bytes(candidate_rows[477]),
            "FJ2217 changed without its pinned route transfer/cap operation",
        )

    def expected_route_pointer(stable_id: int) -> dict:
        pointer = {
            "method": "route110 offline registered local reference transform",
            "stable_id": stable_id,
            "candidate_npz_sha256": ROUTE110_PINS["npz"],
            "candidate_report_sha256": ROUTE110_PINS["report"],
            "candidate_runner_sha256": ROUTE110_PINS["runner"],
            "target_preparation_checkpoint_sha256": ROUTE110_PINS["checkpoint"],
        }
        record = next((x for x in route_data["records"] if int(x["stable_id"]) == stable_id), None)
        if record is not None:
            pointer.update(record)
        else:
            pointer.update({
                "source_member": "FJ3482",
                "before_xyz_sha256": hashlib.sha256(
                    np.ascontiguousarray(bridge["before_local_m"], dtype="<f4").tobytes()
                ).hexdigest(),
                "after_xyz_sha256": hashlib.sha256(
                    np.ascontiguousarray(bridge["after_local_m"], dtype="<f4").tobytes()
                ).hexdigest(),
                "faces_sha256": hashlib.sha256(
                    np.ascontiguousarray(bridge["faces"], dtype="<u4").tobytes()
                ).hexdigest(),
                "xyz_changed": True,
            })
        return pointer

    for stable_id in route_changed_ids:
        actual = json.loads(json.dumps(source_map.get(str(stable_id), {})))
        baseline = json.loads(json.dumps(base_map.get(str(stable_id), {})))
        actual_repair = actual.pop("repair", {}) or {}
        baseline_repair = baseline.pop("repair", {}) or {}
        expected_repair = deepcopy(baseline_repair)
        expected_repair["route110_reference_field"] = expected_route_pointer(stable_id)
        if stable_id == 477:
            if reinstatement is not None:
                if historical_retirement is not None:
                    expected_repair["historical_local_source_face_retirement"] = historical_retirement
                expected_repair["local_source_face_reinstatement"] = reinstatement
            elif active_retirement is not None:
                expected_repair["local_source_face_retirement"] = active_retirement
        expected_identity = deepcopy(baseline)
        if stable_id in (475, 477):
            expected_identity["passive_tissue_owner_binding"] = {
                "method": "inferred geodesic quintic smoothstep between exact source collar ancestry",
                "anatomy_surface_map_sha256": vessel_blend.get("source_owner_map_sha256"),
                "source_payload_index": 5,
                "carrier_surface_ids": [1, 2] if stable_id == 475 else [3, 4],
                "physical_state_or_mass_added": False,
                "anatomical_admission": False,
            }
        _require(actual == expected_identity and actual_repair == expected_repair,
                 f"route110 source identity or repair pointer differs at stable ID {stable_id}")
    new_bridge_map = source_map.get("525", {})
    _require(
        new_bridge_map.get("source_member") == "FJ3482"
        and new_bridge_map.get("body_index") == 20
        and new_bridge_map.get("layer") == 2
        and new_bridge_map.get("vascular_compartment_id") == 4
        and (new_bridge_map.get("repair") or {}).get("route110_reference_field")
        == expected_route_pointer(525),
        "appended FJ3482 route transfer identity/provenance differs",
    )
    changed_original_ids = set(REPAIRED_IDS) | route_changed_ids
    for stable_id in base_rows:
        candidate = candidate_rows[stable_id]
        if stable_id in REPAIRED_IDS:
            row_record = next(record for record in records if int(record["stable_id"]) == stable_id)
            summary = summaries[str(stable_id)]
            _require(_normalized_row_bytes(composed[stable_id])
                     == _normalized_row_bytes(candidate),
                     f"compiled ABI5 row differs from exact source composition for {stable_id}")
            _require(row_record.get("captured_reference_row")
                     == summary["captured_reference_row"]
                     and row_record.get("captured_reference_manifest_sha256")
                     == captured_reference["capture_manifest_sha256"]
                     and row_record.get("compiled_row_sha256") == summary["compiled_row_sha256"]
                     and row_record.get("new_appended_vertex_count")
                     == summary["new_appended_vertex_count"]
                     and row_record.get("retained_n27_face_map_sha256")
                     == summary["retained_n27_face_map_sha256"],
                     f"receipt row repair proof differs from reconstructed row {stable_id}")
            actual_id = json.loads(json.dumps(source_map.get(str(stable_id), {})))
            expected_id = json.loads(json.dumps(base_map.get(str(stable_id), {})))
            expected_repair = context["repair_parent_source_id_map"][str(stable_id)].get(
                "repair", {})
            actual_repair = actual_id.get("repair")
            _require(isinstance(actual_repair, dict) and isinstance(expected_repair, dict),
                     f"source map repair entry is malformed for {stable_id}")
            row_binding = actual_repair.pop("local_intersection_remesh", None)
            expected_id.pop("repair", None)
            actual_id.pop("repair", None)
            _require(actual_id == expected_id
                     and actual_repair == expected_repair
                     and row_binding == row_record,
                     f"source identity, prior 028 repair, or row repair pointer differs for {stable_id}")
        elif stable_id in route_changed_ids:
            if stable_id == 477 and cap_operation_proof is not None:
                _require(
                    cap_operation_proof.get("stable_id") == 477,
                    "FJ2217 route row cap-operation proof was not reconstructed",
                )
            else:
                _require(
                    _normalized_row_bytes(candidate)
                    == _normalized_row_bytes(expected_route_rows[stable_id]),
                    f"route110 source-row transfer differs at stable ID {stable_id}",
                )
        elif source004_index_only is not None:
            _require(
                _normalized_row_bytes(source004_index_only["rows"][stable_id])
                == _normalized_row_bytes(candidate),
                f"source004 child changed unrelated original row {stable_id}",
            )
            _require(
                source_map.get(str(stable_id))
                == source004_index_only["source_id_map"].get(str(stable_id)),
                f"source004 child changed unrelated source identity {stable_id}",
            )
        else:
            _require(_normalized_row_bytes(base_rows[stable_id])
                     == _normalized_row_bytes(candidate),
                     f"unrelated original N27 row changed: {stable_id}")
            _require(source_map.get(str(stable_id)) == base_map.get(str(stable_id)),
                     f"unrelated original source identity changed: {stable_id}")
    unchanged_count = (
        len(source004_index_only["rows"]) - len(REPAIRED_IDS)
        if source004_index_only is not None
        else len(base_rows) - len(changed_original_ids)
    )
    _require(unchanged_count >= 0,
             "original N27 changed-row accounting is negative")
    return {
        "base_row_count": len(base_rows),
        "unchanged_original_row_count": unchanged_count,
        "repaired_original_row_count": len(REPAIRED_IDS),
        "route_transferred_original_row_count": len(route_changed_ids),
        "route_transferred_stable_ids": sorted(route_changed_ids),
        "bridge_appended_row_count": 2,
        "all_unmodified_original_rows_byte_exact": True,
        "all_512_unmodified_rows_byte_exact": unchanged_count == 512,
        "source004_index_only_reemit": (
            source004_index_only["audit"] if source004_index_only is not None else None
        ),
        "all_source004_unmodified_rows_byte_exact": (
            unchanged_count == len(source004_index_only["rows"]) - len(REPAIRED_IDS)
            if source004_index_only is not None else None
        ),
        "fj2217_cap_operation": cap_operation_proof,
        "legacy_retirement_replayed": bool(legacy_retirement_rows),
        # Compatibility for the historical source-007 validator. This alias is
        # true only when its exact old retirement operation was reconstructed;
        # a fresh reinstatement never fabricates a retirement proof.
        "exact_local_vessel_source_face_retirement_reconstructed": bool(
            legacy_retirement_rows
        ),
        "reinstatement_reconstructed": bool(
            cap_operation_proof
            and cap_operation_proof["receipt_operation"]
            in {
                "replace_original_face_corners_by_exact_coordinate_aliases",
                "reinsert_retired_source_face_at_original_order_position",
            }
        ),
        "historical_failed_source_face_retirement_preserved": isinstance(
            historical_retirement, dict
        ) or bool(legacy_retirement_rows),
        "historical_retirement_proof_resolution": historical_retirement_proof_resolution,
        "all_three_repaired_rows_reconstructed_from_pinned_positive_maps": True,
        "captured_step0_world_points_and_local_abi5_emission_bound": True,
        "capture_decoder_provenance": capture_decoder_provenance,
        "all_source_id_map_repair_pointers_match_aggregate": True,
        "physical_mass_or_state_added": False,
        "row_summaries": summaries,
    }

def verify_compiled_candidate_row(
    *,
    stable_id: int,
    source_row: dict,
    parent_row: dict,
    candidate_row: dict,
    arrangement: dict,
    repair_report: dict,
    captured_source_xyz: np.ndarray | None = None,
) -> dict:
    """Reconstruct the only permitted row composition and compare exact ABI5 bytes."""
    expected, summary = compose_intercostal_arrangement_row(
        stable_id=stable_id,
        source_row=source_row,
        parent_row=parent_row,
        arrangement=arrangement,
        repair_report=repair_report,
        captured_source_xyz=captured_source_xyz,
    )
    _require(
        _normalized_row_bytes(expected) == _normalized_row_bytes(candidate_row),
        f"compiled ABI5 row differs from exact source-prefix/face/parent-map composition for {stable_id}",
    )
    summary["compiled_row_sha256"] = sha_bytes(_normalized_row_bytes(candidate_row))
    return summary
