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
LOCAL_315_RETAINED_CROSSWALK_NPZ_SHA256 = "ee6ab8f67dd992a3c613fa758a7a9e2bc3d39d7086cf34f6276c3278365912cd"
LOCAL_315_RETAINED_CROSSWALK_REPORT_SHA256 = "e3da03b16959ec633499d711b64d26af70f748ea3e6db40be65c4f8f4970accb"
LOCAL_315_RETAINED_CANDIDATE_NPZ_SHA256 = "a9035fe15ec4d02b11a21d4366147d93df31bd3533dfadc19126fd8a88fbcd16"
LOCAL_315_SOURCE_CAPTURE_NPZ_SHA256 = "40fd61102da9200a9a1998dc9bc963d5e8c328a0f1ff0086d2d61196f3062411"
LOCAL_315_SOURCE_CAPTURE_REPORT_SHA256 = "5ddf7f6d15475b98d670b17585c3845e75715f8a446cdb135311b37e593f1014"
LOCAL_315_SOURCE_PAYLOAD_SHA256 = "3f1d5a19a03dc156ebfd63f9e5ab5769007fde6ea78251339ce0c81b99171afc"
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





def compose_local_reconstruction_component_patch(
    parent: dict,
    patch: dict,
    *,
    stable_id: int,
    replaced_parent_face_ids,
    patch_parent_face_ids,
) -> tuple[dict, dict]:
    """Insert a local repair into a complete row without losing its other components.

    This only composes exact Float32 arrays and explicit parent-face ancestry.
    The caller must still authenticate the source, rest transform, attachments,
    closed solids, nearby structures, and native motion through existing gates.
    Original allocation slots remain present, including unused source slots.
    """
    _require(type(stable_id) is int and stable_id in (313, 315, 317),
             "component patch has an unsupported typed intercostal identity")

    def geometry(value, label):
        _require(isinstance(value, dict), f"{label} geometry is not an object")
        p = np.asarray(value.get("positions_source_local_f32"))
        w = np.asarray(value.get("positions_world_f32"))
        f = np.asarray(value.get("faces"))
        _require(
            p.dtype == w.dtype == np.dtype("<f4")
            and p.ndim == 2 and p.shape[1] == 3 and p.shape == w.shape
            and len(p) > 0 and np.isfinite(p).all() and np.isfinite(w).all(),
            f"{label} positions are not finite paired Float32 arrays",
        )
        _require(
            f.dtype == np.dtype("<u4") and f.ndim == 2 and f.shape[1] == 3
            and len(f) > 0 and int(f.max()) < len(p),
            f"{label} faces are not valid uint32 triangles",
        )
        for positions in (p, w):
            triangles = positions[f].astype(np.float64)
            cross = np.cross(triangles[:, 1] - triangles[:, 0],
                             triangles[:, 2] - triangles[:, 0])
            _require(np.any(cross != 0, axis=1).all(),
                     f"{label} has a degenerate emitted triangle")
        return (np.ascontiguousarray(p), np.ascontiguousarray(w),
                np.ascontiguousarray(f))

    p, w, faces = geometry(parent, "complete parent")
    pp, pw, pf = geometry(patch, "local patch")
    raw_ids = list(replaced_parent_face_ids)
    _require(raw_ids and all(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        and 0 <= int(value) < len(faces) for value in raw_ids),
        "replaced parent face slots are malformed or out of range")
    selected = {int(value) for value in raw_ids}
    _require(len(selected) == len(raw_ids), "replaced parent face slots repeat")
    raw_parents = list(patch_parent_face_ids)
    _require(len(raw_parents) == len(pf), "local patch ancestry does not cover every face")
    patch_parents = []
    for row in raw_parents:
        values = list(row)
        _require(values and all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            and int(value) in selected for value in values),
            "local patch ancestry leaves the replaced parent region")
        patch_parents.append(sorted({int(value) for value in values}))

    lookup = {}
    for slot, (local, world) in enumerate(zip(p, w, strict=True)):
        key = local.tobytes()
        if key in lookup:
            _require(w[lookup[key]].tobytes() == world.tobytes(),
                     "parent aliases disagree in the world frame")
        else:
            lookup[key] = slot
    out_p = [value.copy() for value in p]
    out_w = [value.copy() for value in w]
    patch_slots = []
    for local, world in zip(pp, pw, strict=True):
        key = local.tobytes()
        slot = lookup.get(key)
        if slot is None:
            slot = len(out_p)
            lookup[key] = slot
            out_p.append(local.copy())
            out_w.append(world.copy())
        else:
            _require(out_w[slot].tobytes() == world.tobytes(),
                     "patch source-coordinate alias disagrees in the world frame")
        patch_slots.append(slot)
    _require(len(out_p) <= np.iinfo(np.uint32).max,
             "composed component allocation exceeds ABI5 uint32 capacity")
    patch_slots = np.asarray(patch_slots, dtype="<u4")
    retained = np.asarray(
        [i for i in range(len(faces)) if i not in selected], dtype="<u4")
    emitted_faces = np.concatenate((faces[retained], patch_slots[pf]), axis=0)
    parents = [[int(i)] for i in retained] + patch_parents
    offsets = np.zeros(len(parents) + 1, dtype="<u8")
    for i, row in enumerate(parents):
        offsets[i + 1] = offsets[i] + len(row)
    flat = np.asarray([i for row in parents for i in row], dtype="<u4")
    result = {
        "positions_source_local_f32": np.asarray(out_p, dtype="<f4"),
        "positions_world_f32": np.asarray(out_w, dtype="<f4"),
        "faces": np.ascontiguousarray(emitted_faces, dtype="<u4"),
        "face_parent_offsets_u64": offsets,
        "face_parent_ids_u32": flat,
        "retained_parent_face_ids_u32": retained,
        "replaced_parent_face_ids_u32": np.asarray(sorted(selected), dtype="<u4"),
        "patch_to_output_vertex_slots_u32": patch_slots,
        "inferred_or_reconstructed_face_mask": np.concatenate((
            np.zeros(len(retained), dtype=bool), np.ones(len(pf), dtype=bool))),
    }
    geometry(result, "composed row")
    return result, {
        "stable_id": stable_id,
        "status": "COMPOSED_COMPLETE_ROW_UNVERIFIED_GEOMETRY",
        "parent_allocated_vertex_count": len(p),
        "candidate_allocated_vertex_count": len(result["positions_source_local_f32"]),
        "parent_face_count": len(faces),
        "retained_parent_face_count": len(retained),
        "replaced_parent_face_count": len(selected),
        "patch_face_count": len(pf),
        "candidate_face_count": len(emitted_faces),
        "original_allocations_preserved": True,
        "untouched_faces_preserved_in_parent_order": True,
        "anatomical_admission": False,
        "native_admission": False,
        "attachment_map_admission": False,
    }

LOCAL_RECONSTRUCTION_ANCESTRY_SCHEMA = (
    "numi.human.intercostal-local-reconstruction-parent-face-owners.v1"
)
LOCAL_BODY20_TRANSFORM_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/predict_brachiocephalic_captured_svc_poses.py"
)
LOCAL_BODY20_TRANSFORM_SHA256 = (
    "017a90ed6c9eadbdfda9c4d4b31a4966ca10c8dd1c79577686635a296145f7d2"
)
LOCAL_RECONSTRUCTION_RIB_IDS = tuple(range(48, 60))
LOCAL_RECONSTRUCTION_SOURCE_BONE_IDS = (26, *LOCAL_RECONSTRUCTION_RIB_IDS, 178)
LOCAL_1091_SOURCE_ASSIGNMENT_SCRIPT_SHA256 = (
    "0a7b4bb8fdda1c4cbee1166c7acf53d9758c1caf390c5f770f870461b3d02656"
)
LOCAL_1091_SOURCE_ASSIGNMENT_INPUT_PINS_SHA256 = (
    "e38c500be9164da02d3abb3a0db529565ecb6a50664a80e4becacc470a3edbac"
)
LOCAL_RECONSTRUCTION_SURFACE_SOURCE_ID_BASE = 51010000
LOCAL_RECONSTRUCTION_EO68_SOURCE_ID = 900020068
LOCAL_RECONSTRUCTION_973_RECEIPT_SHA256 = (
    "d6a6154985d54f5eeb49a5aef54928d9fd03fc53aae3003ecfa1e91ad417174e"
)
LOCAL_RECONSTRUCTION_973_PAYLOAD_SHA256 = (
    "7386d90713612938bf839b69eb04de1b7537e404bafc943b2ee0142c46c1a029"
)


def _local_reconstruction_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _local_reconstruction_hex_sha(value, label: str) -> str:
    _require(
        isinstance(value, str) and len(value) == 64
        and all(char in "0123456789abcdef" for char in value),
        f"{label} is not a lowercase SHA-256",
    )
    return value


def _load_local_body20_transform():
    """Load the pinned production Float32 transform used for captured Body20 rows."""
    import importlib.util

    _require(
        LOCAL_BODY20_TRANSFORM_PATH.is_file()
        and sha_file(LOCAL_BODY20_TRANSFORM_PATH) == LOCAL_BODY20_TRANSFORM_SHA256,
        "captured Body20 Float32 transform bytes differ from the pinned source",
    )
    spec = importlib.util.spec_from_file_location(
        "_numi_pinned_local_body20_transform",
        LOCAL_BODY20_TRANSFORM_PATH,
    )
    _require(spec is not None and spec.loader is not None,
             "pinned captured Body20 transform cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require(callable(getattr(module, "transform_f32", None)),
             "pinned captured Body20 transform has no transform_f32 owner")
    return module.transform_f32


def _body20_rotation64(q_xyzw) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype="<f4")
    _require(q.shape == (4,) and np.isfinite(q).all(),
             "captured Body20 quaternion is malformed")
    x, y, z, w = map(float, q)
    matrix = np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)
    _require(np.isfinite(matrix).all() and abs(float(np.linalg.det(matrix))) > 1e-12,
             "captured Body20 quaternion produces a singular inverse")
    return matrix


def _surface_normals_allow_unused_slots(vertices, faces):
    """Use the existing surface-normal owner while retaining unreferenced slots."""
    positions = np.asarray(vertices)
    triangles = np.asarray(faces)
    _require(
        positions.ndim == 2 and positions.shape[1] == 3
        and np.isfinite(positions).all()
        and triangles.ndim == 2 and triangles.shape[1] == 3
        and np.issubdtype(triangles.dtype, np.integer)
        and np.all(triangles >= 0)
        and int(triangles.max(initial=0)) < len(positions)
        and len(triangles) > 0,
        "local reconstruction normal inputs are malformed",
    )
    triangles_i64 = np.asarray(triangles, dtype=np.int64)
    referenced = np.unique(triangles_i64)
    _require(len(referenced) > 0, "local reconstruction has no referenced vertices")
    compact_faces = np.ascontiguousarray(
        np.searchsorted(referenced, triangles_i64), dtype="<u4",
    )
    from .resting_anatomy_interface_patch import normals as surface_normals
    compact_normals = surface_normals(
        np.ascontiguousarray(positions[referenced]),
        compact_faces,
    )
    normals = np.zeros(positions.shape, dtype=np.float64)
    normals[referenced] = compact_normals
    return normals



def replace_row_source_positions(
    row: dict, positions_f32, *, changed_source_vertex_slots,
) -> tuple[dict, dict]:
    """Repair source positions while restricting normal updates to incident faces.

    This is an asset-time operation. The caller still owns ancestry, attachment,
    complete self/neighbor audits, and native motion admission.
    """
    original = np.asarray(row["vertices6"])
    positions = np.asarray(positions_f32)
    faces = np.asarray(row["faces"])
    _require(
        original.dtype == np.dtype("<f4") and original.ndim == 2
        and original.shape[1] == 6 and np.isfinite(original).all()
        and positions.dtype == np.dtype("<f4")
        and positions.shape == original[:, :3].shape
        and np.isfinite(positions).all()
        and np.issubdtype(faces.dtype, np.integer) and faces.ndim == 2
        and faces.shape[1] == 3 and len(faces) > 0
        and int(faces.min()) >= 0 and int(faces.max()) < len(positions),
        "source-position repair requires finite exact Float32 rows and valid integer faces",
    )
    declared = list(changed_source_vertex_slots)
    _require(
        all(isinstance(slot, (int, np.integer)) and not isinstance(slot, (bool, np.bool_))
            for slot in declared)
        and len(declared) == len(set(map(int, declared)))
        and all(0 <= int(slot) < len(positions) for slot in declared),
        "source-position repair changed-slot declaration is invalid",
    )
    changed = np.flatnonzero(np.any(
        np.ascontiguousarray(positions).view("<u4")
        != np.ascontiguousarray(original[:, :3]).view("<u4"), axis=1,
    ))
    _require(
        len(changed) > 0 and sorted(map(int, declared)) == changed.tolist(),
        "source-position repair must declare exactly every changed source slot",
    )
    referenced = np.unique(faces)
    _require(
        np.isin(changed, referenced).all(),
        "source-position repair cannot move unreferenced allocations",
    )
    # Original seams may have separate normal/material slots at one exact point.
    # A local repair must preserve that point identity for every alias.
    _, aliases = np.unique(original[:, :3], axis=0, return_inverse=True)
    for group in np.unique(aliases[changed]):
        slots = np.flatnonzero(aliases == group)
        _require(
            np.all(positions[slots] == positions[slots[0]]),
            "source-position repair split an exact source-coordinate seam",
        )
    incident_faces = np.flatnonzero(np.isin(faces, changed).any(axis=1))
    normal_slots = np.unique(faces[incident_faces])
    triangles = positions[faces[incident_faces]].astype(np.float64)
    _require(
        np.all(np.linalg.norm(np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ), axis=1) > 0),
        "source-position repair creates a degenerate incident face",
    )
    regenerated = np.asarray(
        _surface_normals_allow_unused_slots(positions, faces), dtype="<f4",
    )
    _require(
        np.isfinite(regenerated[normal_slots]).all()
        and np.all(np.linalg.norm(regenerated[normal_slots], axis=1) > .99),
        "source-position repair has invalid local normals",
    )
    vertices6 = np.array(original, copy=True)
    vertices6[:, :3] = positions
    vertices6[normal_slots, 3:6] = regenerated[normal_slots]
    updated = dict(row)
    updated["vertices6"] = vertices6
    outside = np.setdiff1d(np.arange(len(positions)), normal_slots)
    _require(
        vertices6[outside].tobytes() == original[outside].tobytes(),
        "source-position repair changed an unrelated vertex record",
    )
    return updated, {
        "changed_source_vertex_slots": changed.tolist(),
        "incident_source_face_ids": incident_faces.tolist(),
        "regenerated_normal_source_slots": normal_slots.tolist(),
        "source_faces_unchanged": True,
        "exact_coordinate_seams_preserved": True,
        "unaffected_vertex_records_byte_exact": True,
        "complete_geometry_and_native_motion_admission_required": True,
    }


def _localize_body20_normals_f32(
    world_normals_f32, faces, quaternion_xyzw, *, unused_normal_fallback_f32=None,
):
    """Localize referenced normals and keep unused allocations loader-valid."""
    normals = np.ascontiguousarray(world_normals_f32, dtype="<f4")
    triangles = np.asarray(faces)
    _require(
        normals.ndim == 2 and normals.shape[1] == 3 and len(normals) > 0
        and np.isfinite(normals).all()
        and triangles.ndim == 2 and triangles.shape[1] == 3
        and np.issubdtype(triangles.dtype, np.integer)
        and np.all(triangles >= 0) and int(triangles.max(initial=0)) < len(normals),
        "Body20 normal localization inputs are malformed",
    )
    referenced = np.zeros(len(normals), dtype=np.bool_)
    referenced[np.asarray(triangles, dtype=np.int64).reshape(-1)] = True
    rotation = _body20_rotation64(quaternion_xyzw)
    local = normals.astype(np.float64) @ rotation
    lengths = np.linalg.norm(local, axis=1)
    _require(np.isfinite(lengths).all() and np.all(lengths[referenced] > 1e-20),
             "referenced Body20 normal is zero or invalid")
    nonzero = lengths > 1e-20
    local[nonzero] /= lengths[nonzero, None]
    _require(not np.any(referenced & ~nonzero),
             "referenced Body20 normal is zero")

    unused = np.flatnonzero(~referenced)
    if len(unused):
        # The native loader validates every allocated vertex, including slots
        # absent from the face index buffer. Keep an authenticated unit source
        # normal byte-exact where the local slot namespace is proven; new or
        # invalid unused slots receive a deterministic unit placeholder.
        local[unused] = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if unused_normal_fallback_f32 is not None:
            fallback = np.asarray(unused_normal_fallback_f32)
            _require(
                fallback.ndim == 2 and fallback.shape[1] == 3
                and len(fallback) <= len(normals)
                and fallback.dtype.kind == "f" and fallback.dtype.itemsize == 4
                and np.isfinite(fallback).all(),
                "unused Body20 normal fallback is malformed",
            )
            fallback64 = fallback.astype(np.float64)
            fallback_lengths = np.linalg.norm(fallback64, axis=1)
            valid_fallback = np.isfinite(fallback_lengths) & (
                np.abs(fallback_lengths - 1.0) <= 2.0e-3)
            fallback_slots = unused[unused < len(fallback)]
            valid_slots = fallback_slots[valid_fallback[fallback_slots]]
            if len(valid_slots):
                # Preserve authenticated source Float32 bits exactly.
                local[valid_slots] = fallback[valid_slots].astype(np.float64)
    localized = np.ascontiguousarray(local.astype("<f4"), dtype="<f4")
    all_lengths = np.linalg.norm(localized.astype(np.float64), axis=1)
    _require(np.isfinite(localized).all()
             and np.all(np.abs(all_lengths - 1.0) <= 2.0e-3),
             "localized Body20 normals violate the all-allocation unit-normal loader invariant")
    return localized


def _inverse_body20_positions_f32(world_positions_f32, quaternion_xyzw, position_m):
    """Invert the pinned Body20 transform, then report its actual Float32 replay.

    The inverse is a Float64 solve rounded to the local ABI5 Float32 format. A
    Float32 forward transform is many-to-one, so exact recovery of every input
    world bit is not assumed. The returned replay and its delta are the inputs
    to the subsequent geometry audit and native one-step confirmation.
    """
    world = np.ascontiguousarray(world_positions_f32, dtype="<f4")
    q = np.asarray(quaternion_xyzw, dtype="<f4")
    t = np.asarray(position_m, dtype="<f4")
    _require(
        world.ndim == 2 and world.shape[1] == 3 and len(world) > 0
        and np.isfinite(world).all() and q.shape == (4,) and t.shape == (3,)
        and np.isfinite(q).all() and np.isfinite(t).all(),
        "captured Body20 inverse inputs are malformed",
    )
    rotation = _body20_rotation64(q)
    local = np.linalg.solve(rotation, (world.astype(np.float64) - t.astype(np.float64)).T).T
    base_f32 = np.ascontiguousarray(local, dtype="<f4")
    transform_f32 = _load_local_body20_transform()
    base_replay = np.ascontiguousarray(transform_f32(q, t, base_f32), dtype="<f4")
    world_bits = world.view("<u4")
    base_exact = np.all(base_replay.view("<u4") == world_bits, axis=1)
    base_delta = base_replay.astype(np.float64) - world.astype(np.float64)
    best_local = base_f32.copy()
    best_replay = base_replay.copy()
    best_exact = base_exact.copy()
    best_distance2 = np.einsum("ij,ij->i", base_delta, base_delta)
    best_offsets = np.zeros((len(world), 3), dtype=np.int8)
    below = np.nextafter(base_f32, np.full_like(base_f32, -np.inf))
    above = np.nextafter(base_f32, np.full_like(base_f32, np.inf))
    offsets = [(0, 0, 0)]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if (dx, dy, dz) != (0, 0, 0):
                    offsets.append((dx, dy, dz))
    for offset in offsets[1:]:
        candidate = base_f32.copy()
        for axis, step in enumerate(offset):
            if step < 0:
                candidate[:, axis] = below[:, axis]
            elif step > 0:
                candidate[:, axis] = above[:, axis]
        candidate_replay = np.ascontiguousarray(
            transform_f32(q, t, candidate), dtype="<f4",
        )
        candidate_exact = np.all(candidate_replay.view("<u4") == world_bits, axis=1)
        candidate_delta = candidate_replay.astype(np.float64) - world.astype(np.float64)
        candidate_distance2 = np.einsum("ij,ij->i", candidate_delta, candidate_delta)
        take_exact = candidate_exact & ~best_exact
        take_closer = (
            ~candidate_exact & ~best_exact & (candidate_distance2 < best_distance2)
        )
        take = take_exact | take_closer
        if np.any(take):
            best_local[take] = candidate[take]
            best_replay[take] = candidate_replay[take]
            best_exact[take] = candidate_exact[take]
            best_distance2[take] = candidate_distance2[take]
            best_offsets[take] = offset
    delta = best_replay.astype(np.float64) - world.astype(np.float64)
    offset_counts = {
        ",".join(map(str, offset)): int(np.count_nonzero(np.all(best_offsets == offset, axis=1)))
        for offset in offsets
        if np.any(np.all(best_offsets == offset, axis=1))
    }
    changed_bits = np.any(best_replay.view("<u4") != world_bits, axis=1)
    report = {
        "input_world_positions_sha256": _local_reconstruction_sha256(world.tobytes()),
        "local_inverse_base_f32_sha256": _local_reconstruction_sha256(base_f32.tobytes()),
        "local_positions_f32_sha256": _local_reconstruction_sha256(best_local.tobytes()),
        "replayed_world_positions_sha256": _local_reconstruction_sha256(best_replay.tobytes()),
        "world_roundtrip_exact": best_replay.tobytes() == world.tobytes(),
        "world_roundtrip_changed_vertex_count": int(np.count_nonzero(changed_bits)),
        "world_roundtrip_max_abs_delta_m": float(np.max(np.abs(delta), initial=0.0)),
        "world_roundtrip_rms_delta_m": float(np.sqrt(np.mean(delta * delta))),
        "neighbor_search": {
            "method": "vectorized per-coordinate adjacent Float32 ULP offsets",
            "offsets_per_vertex": len(offsets),
            "exact_replay_vertex_count_from_base": int(np.count_nonzero(base_exact)),
            "exact_replay_vertex_count_after_search": int(np.count_nonzero(best_exact)),
            "selected_nonzero_ulp_offset_vertex_count": int(np.count_nonzero(
                np.any(best_offsets != 0, axis=1))),
            "selected_offset_vertex_counts": offset_counts,
            "selection": "prefer exact world Float32 bits; otherwise minimize squared world-space replay error; ties retain first candidate",
        },
        "forward_transform_owner_sha256": LOCAL_BODY20_TRANSFORM_SHA256,
        "pose_replay_requires_native_one_step_confirmation": True,
    }
    return best_local, best_replay, report


def _require_sha_list(values, label: str) -> list[str]:
    _require(isinstance(values, (list, tuple)) and values,
             f"{label} are missing")
    result = [_local_reconstruction_hex_sha(value, label) for value in values]
    _require(len(result) == len(set(result)), f"{label} contain duplicate hashes")
    return result


def _parse_local_reconstruction_source_reports(raw_reports):
    """Read the retained faces/source_rows ancestry and typed tool.json registry."""
    parsed = []
    for raw in raw_reports:
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "local intercostal reconstruction: authenticated source report is invalid JSON"
            ) from error
        _require(isinstance(value, dict), "authenticated source report root is not an object")
        parsed.append(value)

    tools: dict[int, dict] = {}
    source_tool_report_by_id: dict[int, str] = {}
    for raw, report in zip(raw_reports, parsed, strict=True):
        rows = report.get("rows")
        if not isinstance(rows, list):
            continue
        report_sha = _local_reconstruction_sha256(raw)
        for row in rows:
            if not isinstance(row, dict) or "source_bone" not in row:
                continue
            source_bone = row.get("source_bone")
            _require(
                isinstance(source_bone, list) and len(source_bone) == 2
                and type(source_bone[0]) is int and source_bone[0] == 51004
                and type(source_bone[1]) is int
                and source_bone[1] in LOCAL_RECONSTRUCTION_SOURCE_BONE_IDS,
                "typed source-bone tool registry identity is malformed or unsupported",
            )
            bone_id = source_bone[1]
            derived_id = row.get("derived_source_id")
            faces = row.get("faces")
            tool_sha = row.get("tool_sha256")
            _require(
                type(derived_id) is int and derived_id == 900000000 + bone_id
                and type(faces) is int and faces > 0
                and isinstance(tool_sha, str) and len(tool_sha) == 64
                and all(char in "0123456789abcdef" for char in tool_sha),
                "typed source-bone tool registry record is malformed",
            )
            _require(
                derived_id not in source_tool_report_by_id,
                "derived source ID is ambiguous across authenticated operation reports",
            )
            source_tool_report_by_id[derived_id] = report_sha
            if bone_id not in LOCAL_RECONSTRUCTION_RIB_IDS:
                # Sternum and costal-cartilage tools are valid typed sources,
                # but they do not identify a runtime rib channel.
                continue
            tools[derived_id] = {
                "rib_stable_id": bone_id,
                "face_count": faces,
                "tool_sha256": tool_sha,
                "source_report_sha256": report_sha,
            }

    face_claims: dict[str, dict[int, list[tuple[int, int, str, bool]]]] = {}
    for raw, report in zip(raw_reports, parsed, strict=True):
        faces = report.get("faces")
        if not isinstance(faces, list):
            continue
        report_sha = _local_reconstruction_sha256(raw)
        claims: dict[int, list[tuple[int, int, str, bool]]] = {}
        seen_output_faces: set[int] = set()
        for face in faces:
            _require(
                isinstance(face, dict)
                and type(face.get("output_face")) is int
                and face["output_face"] >= 0,
                "source-face ancestry contains an invalid output face ID",
            )
            _require(face["output_face"] not in seen_output_faces,
                     "source-face ancestry repeats an output face ID")
            seen_output_faces.add(face["output_face"])
            source_rows = face.get("source_rows")
            _require(isinstance(source_rows, list),
                     "source-face ancestry row lacks source_rows")
            exact = face.get("exact_source_face_claim") is True
            for pair in source_rows:
                _require(
                    isinstance(pair, (list, tuple)) and len(pair) == 2
                    and type(pair[0]) is int and pair[0] >= 0
                    and type(pair[1]) is int and pair[1] >= 0,
                    "source-face ancestry contains a malformed source_rows pair",
                )
                claims.setdefault(pair[1], []).append(
                    (face["output_face"], pair[0], str(face.get("classification", "")), exact)
                )
        face_claims[report_sha] = claims
    return tools, face_claims


def derive_local_reconstruction_973_runtime_face_owners(
    receipt_raw: bytes,
    expected_receipt_sha256: str,
    source_payload_raw: bytes,
    expected_source_payload_sha256: str,
    *,
    stable_id: int,
) -> tuple[dict[int, int | None], dict]:
    """Project the retained 973 runtime vertex maps onto exact NHA face slots.

    A face receives a rib only when all three of its exact source vertices are
    present in exactly one authenticated runtime map. Boundary, shared, and
    unmapped faces remain unowned; no geometric proximity is used.
    """
    from .composed_bone_body_frame import _RUNTIME_ATTACHMENT_RIBS
    from .resting_pleura_proxy import _parse_payload

    _require(type(stable_id) is int and stable_id == 313,
             "973 runtime maps are only pinned for typed intercostal row 313")
    receipt_sha = _local_reconstruction_hex_sha(
        expected_receipt_sha256, "973 runtime receipt pin")
    payload_sha = _local_reconstruction_hex_sha(
        expected_source_payload_sha256, "973 runtime source payload pin")
    _require(receipt_sha == LOCAL_RECONSTRUCTION_973_RECEIPT_SHA256
             and payload_sha == LOCAL_RECONSTRUCTION_973_PAYLOAD_SHA256,
             "973 runtime receipt and payload are not the accepted pinned source pair")
    _require(isinstance(receipt_raw, bytes) and receipt_raw
             and _local_reconstruction_sha256(receipt_raw) == receipt_sha,
             "973 runtime receipt bytes differ from their retained pin")
    _require(isinstance(source_payload_raw, bytes) and source_payload_raw
             and _local_reconstruction_sha256(source_payload_raw) == payload_sha,
             "973 runtime source payload bytes differ from their retained pin")
    try:
        receipt = json.loads(receipt_raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("local intercostal reconstruction: 973 receipt is invalid JSON") from error
    _require(isinstance(receipt, dict)
             and receipt.get("schema") == "numi.human.resting-anatomy-receipt.v1"
             and receipt.get("payload", {}).get("sha256") == payload_sha,
             "973 receipt does not bind the exact source anatomy payload")
    _header, _records, rows = _parse_payload(source_payload_raw)
    row = rows.get(stable_id)
    _require(isinstance(row, dict)
             and row.get("body_index") == 20 and row.get("layer") == 1
             and row.get("flags") == 0,
             "973 source payload lacks the exact typed Body20 intercostal row")
    vertices = np.asarray(row.get("vertices6"))
    faces = np.asarray(row.get("faces"))
    _require(
        vertices.ndim == 2 and vertices.shape[1] == 6 and np.isfinite(vertices).all()
        and faces.ndim == 2 and faces.shape[1] == 3
        and np.issubdtype(faces.dtype, np.integer)
        and np.all(faces >= 0) and np.all(faces < len(vertices)),
        "973 source row geometry is malformed",
    )
    bindings = receipt.get("functional_bindings")
    _require(isinstance(bindings, dict),
             "973 receipt lacks functional attachment bindings")
    prefix = f"intercostal_{stable_id}_rib"
    vertex_ribs: dict[int, set[int]] = {}
    sidecar_shas: dict[int, str] = {}
    for key, sidecar in bindings.items():
        if not isinstance(key, str) or not key.startswith(prefix) or not key.endswith("_motion_attachment"):
            continue
        token = key[len(prefix):-len("_motion_attachment")]
        _require(token.isdecimal(),
                 "973 runtime attachment key has a malformed typed rib ID")
        rib_id = int(token)
        _require(rib_id in _RUNTIME_ATTACHMENT_RIBS,
                 "973 runtime attachment key names a non-runtime rib ID")
        _require(isinstance(sidecar, dict)
                 and sidecar.get("schema")
                     == f"numi.human.intercostal{stable_id}-rib{rib_id}-interface-motion-attachment.v1"
                 and sidecar.get("anatomy_payload_sha256") == payload_sha,
                 "973 runtime map sidecar identity or source payload pin differs")
        surface = sidecar.get("surface")
        channel = sidecar.get("existing_rib_channel")
        _require(isinstance(surface, dict) and isinstance(channel, dict)
                 and surface.get("semantic") == 51010
                 and surface.get("stable_id") == stable_id
                 and surface.get("body_index") == 20
                 and surface.get("layer") == 1
                 and surface.get("local_vertex_count") == len(vertices),
                 "973 runtime sidecar does not bind the exact Body20 source surface")
        member, member_sha, channel_index, _source_range_count = _RUNTIME_ATTACHMENT_RIBS[rib_id]
        _require(channel.get("stable_id") == rib_id
                 and channel.get("semantic") == 51004
                 and channel.get("body_index") == 20
                 and channel.get("channel_kind") == 5
                 and channel.get("channel_index") == channel_index
                 and channel.get("source_member_id") == member
                 and channel.get("source_member_sha256") == member_sha,
                 "973 runtime sidecar differs from the existing typed rib-channel registry")
        raw_ids = surface.get("interface_local_vertex_ids")
        _require(isinstance(raw_ids, list) and raw_ids
                 and all(type(value) is int and 0 <= value < len(vertices) for value in raw_ids)
                 and len(raw_ids) == len(set(raw_ids)),
                 "973 runtime sidecar has invalid source vertex slots")
        ids = np.ascontiguousarray(raw_ids, dtype="<u4")
        ids_sha = _local_reconstruction_hex_sha(
            surface.get("interface_local_vertex_ids_sha256"),
            "973 runtime source vertex map hash")
        _require(_local_reconstruction_sha256(ids.tobytes()) == ids_sha,
                 "973 runtime source vertex map differs from its declared hash")
        sidecar_shas[rib_id] = _local_reconstruction_sha256(
            json.dumps(sidecar, sort_keys=True, separators=(",", ":")).encode())
        for vertex_id in ids.tolist():
            vertex_ribs.setdefault(int(vertex_id), set()).add(rib_id)
    _require(bool(sidecar_shas),
             "973 receipt contains no supported row-313 runtime rib maps")
    owners: dict[int, int | None] = {}
    counts = {rib_id: 0 for rib_id in sidecar_shas}
    ambiguous = []
    for face_id, triangle in enumerate(faces):
        memberships = [
            vertex_ribs.get(int(vertex_id), set())
            for vertex_id in triangle
        ]
        common = set.intersection(*(set(value) for value in memberships)) if memberships else set()
        if len(common) == 1:
            rib_id = next(iter(common))
            owners[face_id] = rib_id
            counts[rib_id] += 1
        else:
            owners[face_id] = None
            if any(memberships) and len(ambiguous) < 32:
                ambiguous.append({
                    "source_face_id": face_id,
                    "vertex_rib_ids": [sorted(value) for value in memberships],
                    "common_rib_ids": sorted(common),
                })
    return owners, {
        "source_receipt_sha256": receipt_sha,
        "source_payload_sha256": payload_sha,
        "stable_id": stable_id,
        "source_face_count": len(faces),
        "source_vertex_count": len(vertices),
        "authenticated_runtime_map_rib_ids": sorted(sidecar_shas),
        "runtime_map_sidecar_sha256s": {
            str(rib_id): sidecar_shas[rib_id] for rib_id in sorted(sidecar_shas)
        },
        "source_face_owner_counts": {
            str(rib_id): counts[rib_id] for rib_id in sorted(counts)
        },
        "unowned_source_face_count": sum(owner is None for owner in owners.values()),
        "ambiguous_source_face_witnesses": ambiguous,
        "ambiguous_source_face_witnesses_truncated": (
            sum(owner is None for owner in owners.values()) > len(ambiguous)),
        "assignment_method": "exact all-three-vertices membership in one pinned runtime map",
        "geometric_proximity_used_for_ownership": False,
        "anatomical_admission": False,
        "native_admission": False,
    }


def _local_reconstruction_npz_faces(raw: bytes, label: str) -> np.ndarray:
    """Read one pinned ABI face array from a retained candidate/tool NPZ."""
    import io

    _require(isinstance(raw, bytes) and raw,
             f"{label} candidate NPZ bytes are missing")
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            faces = np.asarray(archive["faces"])
    except (KeyError, ValueError, OSError) as error:
        raise ValueError(f"local intercostal reconstruction: {label} NPZ is invalid") from error
    _require(
        faces.ndim == 2 and faces.shape[1] == 3
        and np.issubdtype(faces.dtype, np.integer)
        and np.all(faces >= 0)
        and np.all(faces <= np.iinfo(np.uint32).max),
        f"{label} NPZ faces are malformed or exceed ABI5 uint32",
    )
    return np.ascontiguousarray(faces, dtype="<u4")


def _authenticated_1091_source_operation(
    *,
    stable_id: int,
    operation_raw: bytes,
    source_parent_candidate_raw: bytes,
    assignment_script_raw: bytes,
    assignment_input_pins_raw: bytes,
    eo68_tool_raw: bytes,
) -> tuple[dict, int]:
    """Authenticate 1091's typed surface/EO source IDs and frozen tool."""
    _require(isinstance(operation_raw, bytes) and operation_raw,
             "1091 source operation bytes are missing")
    operation_sha = _local_reconstruction_sha256(operation_raw)
    try:
        operation = json.loads(operation_raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("local intercostal reconstruction: 1091 operation record is invalid JSON") from error
    _require(
        isinstance(operation, dict)
        and operation.get("identity") == [51010, stable_id]
        and operation.get("parent_candidate_sha256")
            == _local_reconstruction_sha256(source_parent_candidate_raw)
        and type(operation.get("derived_original_id")) is int
        and operation["derived_original_id"] == LOCAL_RECONSTRUCTION_EO68_SOURCE_ID
        and operation.get("target_identity") == [51005, 68],
        "1091 operation identity, parent pin, or non-rib target differs",
    )
    _local_reconstruction_hex_sha(
        operation.get("target_source_sha256"), "1091 EO68 source pin")
    tool_sha = _local_reconstruction_hex_sha(
        operation.get("derived_tool_sha256"), "1091 EO68 tool pin")
    _require(
        _local_reconstruction_sha256(eo68_tool_raw) == tool_sha,
        "1091 EO68 tool bytes differ from the operation pin",
    )
    script_sha = _local_reconstruction_sha256(assignment_script_raw)
    pins_sha = _local_reconstruction_sha256(assignment_input_pins_raw)
    _require(script_sha == LOCAL_1091_SOURCE_ASSIGNMENT_SCRIPT_SHA256,
             "1091 source-ID assignment owner differs from its retained code pin")
    _require(pins_sha == LOCAL_1091_SOURCE_ASSIGNMENT_INPUT_PINS_SHA256,
             "1091 source-ID assignment input pins differ from their retained pin")
    try:
        input_pins = json.loads(assignment_input_pins_raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("local intercostal reconstruction: 1091 input pins are invalid JSON") from error
    _require(isinstance(input_pins, dict)
             and input_pins.get(
                 "/Users/n/numi-human-retained-delivery-20261009/"
                 "anatomy-completion-1276/full-anatomy-delivery-audit-1378/"
                 "construct_intercostal_all_three_external_oblique68_gap_1091.py"
             ) == script_sha
             and _local_reconstruction_sha256(source_parent_candidate_raw)
                 in input_pins.values()
             and operation["target_source_sha256"] in input_pins.values(),
             "1091 script and source rows are not bound by the retained input pins")
    _require(
        b"run_original_id=np.asarray([oid]" in assignment_script_raw
        and b"face_id=np.arange(len(f)" in assignment_script_raw
        and b"51010000+sid" in assignment_script_raw,
        "1091 source-ID assignment code no longer declares typed row/face lineage",
    )
    eo_faces = _local_reconstruction_npz_faces(eo68_tool_raw, "1091 EO68 tool")
    _require(len(eo_faces) > 0, "1091 EO68 tool has no faces")
    return {
        "operation_sha256": operation_sha,
        "source_id_assignment_script_sha256": script_sha,
        "source_id_assignment_input_pins_sha256": pins_sha,
        "typed_surface_source_id": LOCAL_RECONSTRUCTION_SURFACE_SOURCE_ID_BASE + stable_id,
        "derived_nonrib_source_id": LOCAL_RECONSTRUCTION_EO68_SOURCE_ID,
        "derived_nonrib_tool_sha256": tool_sha,
        "derived_nonrib_tool_face_count": len(eo_faces),
        "target_identity": [51005, 68],
        "target_source_sha256": operation["target_source_sha256"],
    }, len(eo_faces)


def derive_local_reconstruction_source_face_lineage(
    source_face_ancestry_raw: bytes,
    expected_source_face_ancestry_sha256: str,
    *,
    stable_id: int,
    candidate_npz_raw: bytes,
    expected_candidate_npz_sha256: str,
    current_operation_ancestry_raw: bytes,
    current_parent_candidate_raw: bytes,
    source_face_ancestry_parent_candidate_raw: bytes,
    source_operation_ancestry_raw: bytes | None = None,
    source_operation_parent_candidate_raw: bytes | None = None,
    source_assignment_script_raw: bytes | None = None,
    source_assignment_input_pins_raw: bytes | None = None,
    derived_nonrib_tool_raw: bytes | None = None,
    authenticated_source_report_raws=(),
    authenticated_source_report_sha256s=(),
) -> tuple[list[list[int]], list[int | None], dict]:
    """Resolve source-face lineage without trusting NPZ process-local IDs.

    The 1091 path is admitted only after validating its pinned source-ID
    assignment owner, input pins, row identity, parent candidate, and actual
    EO68 tool bytes. It maps source rows 51010000+stable_id to exact parent
    face slots and recognizes 900020068 as non-rib support. Other typed rib
    tool rows can own a face only through the authenticated typed source-face
    reports. Multi-parent, mixed-source, and unresolved faces remain unowned.

    Slot-preserving repairs such as 1095 instead inherit the same face slot
    from their exact parent candidate, after the operation pins the source
    ancestry and both candidate face arrays are byte-identical.
    """
    _require(type(stable_id) is int and stable_id in (313, 315, 317),
             "source-face lineage stable ID is not a typed intercostal row")
    source_face_sha = _local_reconstruction_hex_sha(
        expected_source_face_ancestry_sha256, "source-face ancestry pin")
    _require(
        isinstance(source_face_ancestry_raw, bytes)
        and _local_reconstruction_sha256(source_face_ancestry_raw) == source_face_sha,
        "source-face ancestry bytes differ from their independent pin",
    )
    candidate_sha = _local_reconstruction_hex_sha(
        expected_candidate_npz_sha256, "candidate NPZ pin")
    _require(
        isinstance(candidate_npz_raw, bytes)
        and _local_reconstruction_sha256(candidate_npz_raw) == candidate_sha,
        "candidate NPZ bytes differ from their independent pin",
    )
    candidate_faces = _local_reconstruction_npz_faces(candidate_npz_raw, "candidate")
    current_parent_sha = _local_reconstruction_sha256(current_parent_candidate_raw)
    current_parent_faces = _local_reconstruction_npz_faces(
        current_parent_candidate_raw, "current parent candidate")
    try:
        ancestry = json.loads(source_face_ancestry_raw)
        operation = json.loads(current_operation_ancestry_raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("local intercostal reconstruction: face or operation ancestry is invalid JSON") from error
    _require(isinstance(ancestry, dict) and isinstance(ancestry.get("faces"), list),
             "source-face ancestry has no typed faces inventory")
    _require(isinstance(operation, dict),
             "current operation ancestry root is not an object")
    output_face_ids = [
        face.get("output_face") if isinstance(face, dict) else None
        for face in ancestry["faces"]
    ]
    _require(
        len(output_face_ids) == len(candidate_faces)
        and all(type(value) is int for value in output_face_ids)
        and sorted(output_face_ids) == list(range(len(candidate_faces))),
        "source-face ancestry does not cover the exact candidate face slots",
    )

    # Position-only local fan/alias repairs keep every face slot. Their own
    # operation report links the unchanged ancestry and exact parent candidate.
    slot_preserving = (
        (
            operation.get("faces_retain_parent_slots") is True
            or operation.get("source_face_ancestry_unchanged") is True
            or operation.get("faces_and_allocation_ids_byte_exact") is True
        )
        and isinstance(operation.get("parent_face_ancestry_sha256"), str)
    )
    if slot_preserving:
        _require(
            operation.get("parent_face_ancestry_sha256") == source_face_sha
            and operation.get("parent_candidate_sha256") == current_parent_sha
            and operation.get("candidate_sha256") == candidate_sha,
            "slot-preserving operation does not bind the exact candidate and face ancestry",
        )
        _require(
            len(candidate_faces) == len(current_parent_faces)
            and candidate_faces.tobytes() == current_parent_faces.tobytes(),
            "slot-preserving operation changed the parent face slots",
        )
        return (
            [[face_id] for face_id in range(len(candidate_faces))],
            [None] * len(candidate_faces),
            {
                "mode": "authenticated_parent_face_slots_preserved",
                "source_face_ancestry_sha256": source_face_sha,
                "candidate_npz_sha256": candidate_sha,
                "current_parent_candidate_sha256": current_parent_sha,
                "preserved_face_slot_count": len(candidate_faces),
                "unowned_face_count": 0,
                "ambiguous_face_count": 0,
                "anatomical_admission": False,
                "native_admission": False,
            },
        )

    # The 1091 source operation is the only supported producer of the EO68
    # namespace used by this local reconstruction family.
    source_operation_raw = (
        current_operation_ancestry_raw
        if source_operation_ancestry_raw is None
        else source_operation_ancestry_raw
    )
    source_parent_raw = (
        current_parent_candidate_raw
        if source_operation_parent_candidate_raw is None
        else source_operation_parent_candidate_raw
    )
    _require(
        source_assignment_script_raw is not None
        and source_assignment_input_pins_raw is not None
        and derived_nonrib_tool_raw is not None,
        "1091 source assignment code, pins, or EO68 tool bytes are missing",
    )
    source_operation, eo68_face_count = _authenticated_1091_source_operation(
        stable_id=stable_id,
        operation_raw=source_operation_raw,
        source_parent_candidate_raw=source_parent_raw,
        assignment_script_raw=source_assignment_script_raw,
        assignment_input_pins_raw=source_assignment_input_pins_raw,
        eo68_tool_raw=derived_nonrib_tool_raw,
    )
    source_parent_faces = _local_reconstruction_npz_faces(
        source_parent_raw, "1091 typed surface parent")
    _require(
        ancestry.get("parent_candidate_sha256")
            == _local_reconstruction_sha256(source_face_ancestry_parent_candidate_raw),
        "source-face ancestry does not bind the exact 1091 floating candidate",
    )
    floating_faces = _local_reconstruction_npz_faces(
        source_face_ancestry_parent_candidate_raw, "1091 floating source")
    _require(len(floating_faces) > 0,
             "1091 floating source candidate has no faces")
    _require(
        source_operation["typed_surface_source_id"]
            == LOCAL_RECONSTRUCTION_SURFACE_SOURCE_ID_BASE + stable_id,
        "1091 typed row source ID differs from the requested surface",
    )

    reports = list(authenticated_source_report_raws)
    report_pins = list(authenticated_source_report_sha256s)
    if source_face_ancestry_raw not in reports:
        reports.append(source_face_ancestry_raw)
        report_pins.append(source_face_sha)
    typed_tools, source_face_claims = _parse_local_reconstruction_source_reports(reports)
    pinned_report_shas = _require_sha_list(
        report_pins, "independently pinned source-face lineage reports")
    actual_report_shas = [_local_reconstruction_sha256(raw) for raw in reports]
    _require(actual_report_shas == pinned_report_shas,
             "source-face lineage report bytes differ from their independent pins")
    own_report_sha = source_face_sha
    surface_source_id = source_operation["typed_surface_source_id"]
    nonrib_source_id = source_operation["derived_nonrib_source_id"]

    face_parent_ids: list[list[int]] = [[] for _ in candidate_faces]
    direct_rib_owners: list[int | None] = [None] * len(candidate_faces)
    ambiguous_faces: list[dict] = []
    unowned_reasons: dict[str, int] = {}
    direct_owner_count = 0
    inherited_parent_count = 0
    for face in ancestry["faces"]:
        output_face = int(face["output_face"])
        source_rows = face.get("source_rows")
        _require(isinstance(source_rows, list) and source_rows,
                 f"source-face lineage output face {output_face} has no source rows")
        unique_pairs: set[tuple[int, int]] = set()
        for pair in source_rows:
            _require(
                isinstance(pair, (list, tuple)) and len(pair) == 2
                and type(pair[0]) is int and pair[0] >= 0
                and type(pair[1]) is int and pair[1] >= 0,
                f"source-face lineage output face {output_face} has malformed source rows",
            )
            unique_pairs.add((int(pair[0]), int(pair[1])))
        surface_pairs = sorted(
            (source_id, source_face_id) for source_id, source_face_id in unique_pairs
            if source_id == surface_source_id
        )
        nonrib_pairs = sorted(
            (source_id, source_face_id) for source_id, source_face_id in unique_pairs
            if source_id == nonrib_source_id
        )
        typed_rib_pairs = sorted(
            (source_id, source_face_id) for source_id, source_face_id in unique_pairs
            if source_id in typed_tools
        )
        unresolved_pairs = sorted(
            pair for pair in unique_pairs
            if pair[0] not in {surface_source_id, nonrib_source_id}
            and pair[0] not in typed_tools
        )
        _require(
            all(source_face_id < len(source_parent_faces)
                for _source_id, source_face_id in surface_pairs),
            f"source-face lineage output face {output_face} exceeds its pinned parent face array",
        )
        _require(
            all(source_face_id < eo68_face_count
                for _source_id, source_face_id in nonrib_pairs),
            f"source-face lineage output face {output_face} exceeds its pinned EO68 tool",
        )
        rib_ids = set()
        for source_id, source_face_id in typed_rib_pairs:
            tool = typed_tools[source_id]
            _require(source_face_id < tool["face_count"],
                     f"source-face lineage output face {output_face} exceeds a typed rib tool")
            _require(
                _source_face_evidence_matches_typed_rib(
                    source_report_sha=own_report_sha,
                    parent_face_id=output_face,
                    source_face_ids=[source_face_id],
                    rib_stable_id=tool["rib_stable_id"],
                    method="direct_typed_rib_source",
                    source_face_claims=source_face_claims,
                    typed_tool_rows=typed_tools,
                ),
                f"source-face lineage output face {output_face} has an unauthenticated typed rib claim",
            )
            rib_ids.add(tool["rib_stable_id"])
        # Inherit only from one unique source row/face with no competing
        # derived or unresolved input. A multi-parent cut face stays unowned.
        if (
            len(unique_pairs) == 1 and len(surface_pairs) == 1
            and not nonrib_pairs and not typed_rib_pairs and not unresolved_pairs
        ):
            face_parent_ids[output_face] = [surface_pairs[0][1]]
            inherited_parent_count += 1
        elif (
            len(unique_pairs) > 1 or len(rib_ids) > 1
            or (surface_pairs and (nonrib_pairs or typed_rib_pairs or unresolved_pairs))
            or (nonrib_pairs and (surface_pairs or typed_rib_pairs or unresolved_pairs))
            or unresolved_pairs
        ):
            ambiguous_faces.append({
                "output_face": output_face,
                "source_rows": [list(pair) for pair in sorted(unique_pairs)],
                "reason": "multi_parent_or_mixed_source_lineage",
            })
            unowned_reasons["multi_parent_or_mixed_source_lineage"] = (
                unowned_reasons.get("multi_parent_or_mixed_source_lineage", 0) + 1)
        elif nonrib_pairs:
            unowned_reasons["authenticated_nonrib_eo68_source"] = (
                unowned_reasons.get("authenticated_nonrib_eo68_source", 0) + 1)
        elif typed_rib_pairs and len(rib_ids) == 1:
            direct_rib_owners[output_face] = next(iter(rib_ids))
            direct_owner_count += 1
        else:
            unowned_reasons["no_unique_authenticated_rib_parent"] = (
                unowned_reasons.get("no_unique_authenticated_rib_parent", 0) + 1)

    return face_parent_ids, direct_rib_owners, {
        "mode": "authenticated_1091_source_face_lineage",
        "source_face_ancestry_sha256": source_face_sha,
        "source_face_ancestry_parent_candidate_sha256": _local_reconstruction_sha256(
            source_face_ancestry_parent_candidate_raw),
        "candidate_npz_sha256": candidate_sha,
        "source_operation_sha256": source_operation["operation_sha256"],
        "source_operation_parent_candidate_sha256": _local_reconstruction_sha256(
            source_parent_raw),
        "typed_surface_source_id": surface_source_id,
        "derived_nonrib_source_id": nonrib_source_id,
        "derived_nonrib_tool_sha256": source_operation["derived_nonrib_tool_sha256"],
        "derived_nonrib_tool_face_count": eo68_face_count,
        "typed_rib_tool_report_sha256s": sorted(
            _local_reconstruction_sha256(raw)
            for raw in reports if raw != source_face_ancestry_raw),
        "output_face_count": len(candidate_faces),
        "single_parent_face_count": inherited_parent_count,
        "direct_typed_rib_face_count": direct_owner_count,
        "unowned_face_count": len(candidate_faces) - inherited_parent_count - direct_owner_count,
        "unowned_reason_counts": unowned_reasons,
        "ambiguous_face_witnesses": ambiguous_faces[:32],
        "ambiguous_face_witnesses_truncated": len(ambiguous_faces) > 32,
        "anatomical_admission": False,
        "native_admission": False,
    }


def _source_face_evidence_matches_typed_rib(
    *,
    source_report_sha: str,
    parent_face_id: int,
    source_face_ids: list[int],
    rib_stable_id: int,
    method: str,
    source_face_claims: dict[str, dict[int, list[tuple[int, int, str, bool]]]],
    typed_tool_rows: dict[int, dict],
) -> bool:
    claims = source_face_claims.get(source_report_sha)
    if not claims:
        return False
    for source_face_id in source_face_ids:
        matching = [
            item for item in claims.get(source_face_id, [])
            if item[0] == parent_face_id
        ]
        if not matching:
            return False
        valid = False
        for _output_face, source_row_id, classification, exact in matching:
            tool = typed_tool_rows.get(source_row_id)
            if (
                tool is not None
                and tool["rib_stable_id"] == rib_stable_id
                and source_face_id < tool["face_count"]
                and exact
                and (method != "inherited_authenticated_parent_patch"
                     or classification == "unchanged_parent_face")
            ):
                valid = True
                break
        if not valid:
            return False
    return True



def _exact_geometry_parent_patch_evidence(
    *, parent_payload_raw, parent_faces, parent_face_id, item,
    source_report_raw, source_parent_npz_raws_by_sha, cache,
):
    """Prove reindexed ownership from the original typed floating triangle.

    No proximity tolerance, synthetic source report, or inferred face support
    can establish this relation. Only an unchanged single-source claim whose
    exact emitted oriented triangle is still present may inherit the channel.
    """
    import io

    # The normalizer has already joined these immutable bytes by their
    # independently checked hash; avoid hashing a full report for every face.
    report_sha = item["source_report_sha256"]
    if report_sha not in cache:
        report = json.loads(source_report_raw)
        parent_sha = report.get("parent_candidate_sha256")
        _require(parent_sha in source_parent_npz_raws_by_sha,
                 "exact geometry evidence lacks its pinned source parent NPZ")
        try:
            with np.load(io.BytesIO(source_parent_npz_raws_by_sha[parent_sha]),
                         allow_pickle=False) as z:
                arrays = {k: z[k].copy() for k in (
                    "positions_world_f64", "faces", "source_original_id", "source_face_id")}
        except (KeyError, ValueError, OSError) as error:
            raise ValueError("exact geometry source parent NPZ is invalid") from error
        points, faces = arrays["positions_world_f64"], arrays["faces"]
        _require(points.dtype == np.dtype("<f8") and points.ndim == 2
                 and points.shape[1] == 3 and np.isfinite(points).all()
                 and faces.ndim == 2 and faces.shape[1] == 3
                 and np.issubdtype(faces.dtype, np.integer)
                 and np.all(faces >= 0) and np.all(faces < len(points))
                 and arrays["source_original_id"].shape == (len(faces),)
                 and arrays["source_face_id"].shape == (len(faces),)
                 and np.issubdtype(arrays["source_original_id"].dtype, np.integer)
                 and np.issubdtype(arrays["source_face_id"].dtype, np.integer)
                 and np.all(arrays["source_original_id"] >= 0)
                 and np.all(arrays["source_face_id"] >= 0),
                 "exact geometry source parent arrays are malformed")
        by_id = {row["output_face"]: row for row in report["faces"]}
        _require(len(by_id) == len(report["faces"]),
                 "exact geometry source face inventory repeats an output face")
        cache[report_sha] = (arrays, by_id)
    arrays, by_id = cache[report_sha]
    if "current_parent_world" not in cache:
        try:
            with np.load(io.BytesIO(parent_payload_raw), allow_pickle=False) as z:
                current_points = z["positions_world_f32"].copy()
                current_faces = z["faces"].copy()
        except (KeyError, ValueError, OSError) as error:
            raise ValueError("exact geometry current parent NPZ is invalid") from error
        _require(current_points.dtype == np.dtype("<f4")
                 and current_points.ndim == 2 and current_points.shape[1] == 3
                 and np.isfinite(current_points).all()
                 and current_faces.shape == parent_faces.shape
                 and np.array_equal(current_faces, parent_faces)
                 and np.all(parent_faces < len(current_points)),
                 "exact geometry current parent arrays differ from pinned faces")
        cache["current_parent_world"] = current_points
    source_output_id = item.get("source_output_face_id")
    _require(type(source_output_id) is int and source_output_id in by_id,
             "exact geometry evidence has an invalid historical output face")
    row = by_id[source_output_id]
    support = row.get("parent_output_face_ids")
    _require(row.get("classification") == "unchanged_parent_face"
             and row.get("exact_source_face_claim") is True
             and isinstance(support, list) and len(support) == 1
             and type(support[0]) is int
             and 0 <= support[0] < len(arrays["faces"]),
             "exact geometry evidence is not one unchanged source triangle")
    source_id = support[0]
    pair = [int(arrays["source_original_id"][source_id]),
            int(arrays["source_face_id"][source_id])]
    _require(row.get("source_rows") == [pair]
             and item.get("source_face_ids") == [pair[1]],
             "exact geometry source row differs from original floating ancestry")
    historical = np.ascontiguousarray(
        arrays["positions_world_f64"][arrays["faces"][source_id]], dtype="<f4")
    current = cache["current_parent_world"][parent_faces[parent_face_id]]
    _require(np.isfinite(historical).all(),
             "exact geometry source triangle is not finite after Float32 emission")
    def oriented_key(triangle):
        rows = [tuple(row.view("<u4").tolist()) for row in triangle]
        return min(tuple(rows[k:] + rows[:k]) for k in range(3))
    _require(oriented_key(current) == oriented_key(historical),
             "exact geometry parent triangle differs from its historical typed source")
    _require(np.any(np.cross(current[1].astype(np.float64)-current[0],
                             current[2].astype(np.float64)-current[0]) != 0),
             "exact geometry parent triangle is degenerate")
    return source_output_id


def _normalize_local_reconstruction_parent_face_owners(
    owner_report_raw: bytes,
    expected_owner_report_sha256: str,
    *,
    stable_id: int,
    parent_faces,
    parent_payload_raw: bytes,
    operation_ancestry_raw: bytes,
    authenticated_source_report_raws,
    authenticated_source_report_sha256s,
    authenticated_source_parent_npz_raws=(),
    authenticated_source_parent_npz_sha256s=(),
) -> tuple[dict[int, int | None], dict]:
    """Authenticate explicit parent-face rib ownership, including inherited caps."""
    expected_sha = _local_reconstruction_hex_sha(
        expected_owner_report_sha256, "parent face-owner report pin")
    _require(_local_reconstruction_sha256(owner_report_raw) == expected_sha,
             "parent face-owner report bytes differ from their declared pin")
    try:
        report = json.loads(owner_report_raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("local intercostal reconstruction: parent face-owner report is invalid JSON") from error
    _require(
        isinstance(report, dict)
        and report.get("schema") == LOCAL_RECONSTRUCTION_ANCESTRY_SCHEMA
        and type(report.get("stable_id")) is int and report["stable_id"] == stable_id,
        "parent face-owner report schema or typed surface identity differs",
    )
    _require(isinstance(parent_payload_raw, bytes) and parent_payload_raw,
             "pinned local-repair parent payload bytes are missing")
    _require(isinstance(operation_ancestry_raw, bytes) and operation_ancestry_raw,
             "pinned local-repair ancestry bytes are missing")
    _require(
        isinstance(authenticated_source_report_raws, (list, tuple))
        and authenticated_source_report_raws
        and all(isinstance(raw, bytes) and raw for raw in authenticated_source_report_raws),
        "authenticated source report bytes are missing",
    )
    expected_source_report_shas = _require_sha_list(
        authenticated_source_report_sha256s,
        "independently pinned source report hashes",
    )
    parent_payload_sha = _local_reconstruction_sha256(parent_payload_raw)
    operation_chain_sha = _local_reconstruction_sha256(operation_ancestry_raw)
    authenticated_source_report_shas = [
        _local_reconstruction_sha256(raw) for raw in authenticated_source_report_raws
    ]
    _require(
        len(authenticated_source_report_shas) == len(expected_source_report_shas)
        and authenticated_source_report_shas == expected_source_report_shas,
        "authenticated source report bytes differ from their independent pins",
    )
    typed_tool_rows, source_face_claims = _parse_local_reconstruction_source_reports(
        authenticated_source_report_raws,
    )
    source_parent_raws = list(authenticated_source_parent_npz_raws)
    source_parent_shas = list(authenticated_source_parent_npz_sha256s)
    _require(len(source_parent_raws) == len(source_parent_shas)
             and len(source_parent_shas) == len(set(source_parent_shas))
             and all(isinstance(raw, bytes) and raw for raw in source_parent_raws)
             and all(_local_reconstruction_sha256(raw) == expected
                     for raw, expected in zip(source_parent_raws, source_parent_shas)),
             "exact geometry source parent NPZs differ from their independent pins")
    source_parent_npz_raws_by_sha = dict(zip(source_parent_shas, source_parent_raws))
    source_raws_by_sha = dict(zip(authenticated_source_report_shas,
                                 authenticated_source_report_raws))
    exact_geometry_cache = {}
    _require(report.get("parent_payload_sha256") == parent_payload_sha,
             "owner report parent payload hash does not authenticate supplied bytes")
    _require(report.get("operation_ancestry_sha256") == operation_chain_sha,
             "owner report operation ancestry hash does not authenticate supplied bytes")
    raw_faces = np.asarray(parent_faces)
    _require(raw_faces.ndim == 2 and raw_faces.shape[1] == 3
             and np.issubdtype(raw_faces.dtype, np.integer)
             and np.all(raw_faces >= 0)
             and np.all(raw_faces <= np.iinfo(np.uint32).max),
             "authenticated parent face indices exceed the ABI5 uint32 range")
    faces = np.ascontiguousarray(raw_faces, dtype="<u4")
    _require(len(faces) > 0, "authenticated parent faces are empty")
    faces_sha = _local_reconstruction_sha256(faces.tobytes())
    _require(
        report.get("parent_faces_sha256") == faces_sha
        and type(report.get("parent_face_count")) is int
        and report["parent_face_count"] == len(faces),
        "parent face-owner report is not bound to the exact parent face array",
    )
    entries = report.get("parent_face_owners")
    _require(isinstance(entries, list), "parent face-owner inventory is missing")
    owners: dict[int, int | None] = {}
    methods: dict[int, str] = {}
    for entry in entries:
        _require(isinstance(entry, dict)
                 and type(entry.get("parent_face_id")) is int
                 and 0 <= entry["parent_face_id"] < len(faces),
                 "parent face-owner row has an invalid face ID")
        face_id = int(entry["parent_face_id"])
        _require(face_id not in owners, "parent face-owner inventory repeats a face ID")
        rib_ids = entry.get("rib_stable_ids")
        _require(
            isinstance(rib_ids, list)
            and all(type(rib_id) is int and rib_id in LOCAL_RECONSTRUCTION_RIB_IDS
                    for rib_id in rib_ids)
            and len(rib_ids) == len(set(rib_ids)),
            "parent face-owner row has malformed typed rib IDs",
        )
        evidence = entry.get("evidence")
        _require(isinstance(evidence, list), "parent face-owner row lacks explicit evidence")
        evidenced_ribs = set()
        for item in evidence:
            _require(
                isinstance(item, dict)
                and type(item.get("rib_stable_id")) is int
                and item["rib_stable_id"] in LOCAL_RECONSTRUCTION_RIB_IDS
                and item.get("method") in (
                    "direct_typed_rib_source",
                    "inherited_authenticated_parent_patch",
                    "exact_geometry_authenticated_parent_patch",
                ),
                "parent face-owner evidence has an invalid typed rib or method",
            )
            source_report_sha = _local_reconstruction_hex_sha(
                item.get("source_report_sha256"), "parent ownership evidence report hash")
            _require(
                source_report_sha in authenticated_source_report_shas,
                "parent ownership evidence report hash does not match supplied source report bytes",
            )
            source_faces = item.get("source_face_ids")
            _require(
                isinstance(source_faces, list) and source_faces
                and all(type(value) is int and value >= 0 for value in source_faces)
                and len(source_faces) == len(set(source_faces)),
                "parent ownership evidence lacks source face lineage",
            )
            rib_id = int(item["rib_stable_id"])
            _require(rib_id not in evidenced_ribs,
                     "parent face-owner evidence repeats a typed rib")
            evidence_face_id = face_id
            evidence_method = item["method"]
            if evidence_method == "exact_geometry_authenticated_parent_patch":
                evidence_face_id = _exact_geometry_parent_patch_evidence(
                    parent_payload_raw=parent_payload_raw,
                    parent_faces=faces, parent_face_id=face_id, item=item,
                    source_report_raw=source_raws_by_sha[source_report_sha],
                    source_parent_npz_raws_by_sha=source_parent_npz_raws_by_sha,
                    cache=exact_geometry_cache,
                )
                evidence_method = "inherited_authenticated_parent_patch"
            _require(
                _source_face_evidence_matches_typed_rib(
                    source_report_sha=source_report_sha,
                    parent_face_id=evidence_face_id,
                    source_face_ids=source_faces,
                    rib_stable_id=rib_id,
                    method=evidence_method,
                    source_face_claims=source_face_claims,
                    typed_tool_rows=typed_tool_rows,
                ),
                "parent rib label is not proven by the pinned typed source-face ancestry and tool registry",
            )
            evidenced_ribs.add(rib_id)
            methods[face_id] = item["method"]
        _require(
            evidenced_ribs == set(rib_ids)
            and len(rib_ids) <= 1,
            "parent face has missing or ambiguous rib attribution",
        )
        owners[face_id] = rib_ids[0] if rib_ids else None
    _require(len(owners) == len(faces),
             "parent face-owner report does not cover every parent face")
    return owners, {
        "parent_owner_report_sha256": expected_sha,
        "parent_payload_sha256": parent_payload_sha,
        "operation_ancestry_sha256": operation_chain_sha,
        "authenticated_source_report_sha256s": sorted(authenticated_source_report_shas),
        "parent_faces_sha256": faces_sha,
        "parent_face_count": len(faces),
        "explicit_parent_face_owner_count": len(owners),
        "inherited_parent_face_owner_count": sum(
            methods.get(face_id) == "inherited_authenticated_parent_patch"
            for face_id in methods),
        "direct_typed_source_face_owner_count": sum(
            methods.get(face_id) == "direct_typed_rib_source"
            for face_id in methods),
        "exact_geometry_inherited_parent_face_owner_count": sum(
            method == "exact_geometry_authenticated_parent_patch"
            for method in methods.values()),
        "authenticated_source_parent_npz_sha256s": source_parent_shas,
        "unowned_parent_face_count": sum(owner is None for owner in owners.values()),
    }


def derive_local_reconstruction_face_rib_owners(
    face_parent_ids,
    parent_face_owners: dict[int, int | None],
    *,
    parent_face_count: int,
    return_diagnostics: bool = False,
) -> list[int | None] | tuple[list[int | None], dict]:
    """Resolve child owners from authenticated parents, including unique rib consensus."""
    _require(isinstance(parent_face_owners, dict) and parent_face_owners,
             "authenticated parent face-owner map is missing")
    _require(type(parent_face_count) is int and parent_face_count > 0,
             "parent face count is invalid")
    _require(type(return_diagnostics) is bool,
             "face-owner diagnostics mode is malformed")
    resolved: list[int | None] = []
    direct_count = 0
    consensus_count = 0
    unowned_count = 0
    consensus_face_ids: list[int] = []
    for child_face_id, parent_ids in enumerate(face_parent_ids):
        if type(parent_ids) is int:
            parent_ids = [parent_ids]
        _require(
            isinstance(parent_ids, (list, tuple))
            and all(type(value) is int and 0 <= value < parent_face_count for value in parent_ids)
            and len(parent_ids) == len(set(parent_ids)),
            f"child face {child_face_id} has missing or invalid parent ancestry",
        )
        if not parent_ids:
            # A derived non-rib tool face or an unresolved mixed-source face is
            # retained geometrically and explicitly has no runtime rib owner.
            resolved.append(None)
            unowned_count += 1
            continue
        _require(
            all(parent_id in parent_face_owners for parent_id in parent_ids),
            f"child face {child_face_id} references an unauthenticated parent face",
        )
        parent_ribs = [parent_face_owners[parent_id] for parent_id in parent_ids]
        _require(
            all(rib_id is None or (type(rib_id) is int and rib_id in LOCAL_RECONSTRUCTION_RIB_IDS)
                for rib_id in parent_ribs),
            f"child face {child_face_id} has a non-typed authenticated parent rib owner",
        )
        if len(parent_ids) == 1:
            owner = parent_ribs[0]
            resolved.append(owner)
            if owner is None:
                unowned_count += 1
            else:
                direct_count += 1
            continue
        unique_parent_ribs = set(parent_ribs)
        if len(unique_parent_ribs) == 1 and None not in unique_parent_ribs:
            # This is a narrow consensus inheritance: every authenticated
            # parent names the same existing typed channel. Mixed or missing
            # ancestry remains unmapped; geometry alone never assigns an owner.
            resolved.append(parent_ribs[0])
            consensus_count += 1
            consensus_face_ids.append(child_face_id)
        else:
            resolved.append(None)
            unowned_count += 1
    if not return_diagnostics:
        return resolved
    consensus_ids = np.asarray(consensus_face_ids, dtype="<u8")
    return resolved, {
        "single_authenticated_parent_face_count": direct_count,
        "same_typed_rib_parent_consensus_face_count": consensus_count,
        "unowned_or_ambiguous_parent_face_count": unowned_count,
        "same_typed_rib_parent_consensus_child_face_ids_sha256": (
            _local_reconstruction_sha256(consensus_ids.tobytes())),
        "same_typed_rib_parent_consensus_is_inferred": consensus_count > 0,
    }


def derive_local_reconstruction_attachment_maps(
    faces,
    face_rib_owners: list[int | None],
    interface_face_mask,
    *,
    vertex_count: int,
    required_rib_ids=None,
    return_diagnostics: bool = False,
) -> dict[int, list[int]] | tuple[dict[int, list[int]], dict]:
    """Build fresh maps; leave vertices with conflicting interface owners unmapped."""
    raw_triangles = np.asarray(faces)
    uint32_max = int(np.iinfo(np.uint32).max)
    _require(
        raw_triangles.ndim == 2 and raw_triangles.shape[1] == 3
        and np.issubdtype(raw_triangles.dtype, np.integer)
        and np.all(raw_triangles >= 0)
        and np.all(raw_triangles <= uint32_max),
        "interface face indices exceed the ABI5 uint32 range",
    )
    _require(type(vertex_count) is int and 0 < vertex_count <= uint32_max + 1
             and np.all(raw_triangles < vertex_count),
             "interface face indices exceed the local vertex allocation")
    triangles = np.ascontiguousarray(raw_triangles, dtype="<u4")
    raw_interface = np.asarray(interface_face_mask)
    _require(raw_interface.dtype == np.dtype(np.bool_) and raw_interface.ndim == 1,
             "interface face mask is not an exact one-dimensional boolean array")
    interface = raw_interface
    _require(len(triangles) == len(face_rib_owners) == len(interface),
             "interface faces or local vertex allocation are malformed")
    required = set(required_rib_ids or ())
    _require(all(type(rib_id) is int and rib_id in LOCAL_RECONSTRUCTION_RIB_IDS
                 for rib_id in required),
             "required attachment inventory contains a non-typed rib")
    vertex_ribs: dict[int, set[int]] = {}
    observed_rib_ids: set[int] = set()
    for face_id, (triangle, rib_id, is_interface) in enumerate(
        zip(triangles, face_rib_owners, interface, strict=True)
    ):
        if not is_interface or rib_id is None:
            continue
        _require(type(rib_id) is int and rib_id in LOCAL_RECONSTRUCTION_RIB_IDS,
                 f"interface face {face_id} has a non-typed rib owner")
        observed_rib_ids.add(rib_id)
        vertices = {int(value) for value in triangle}
        _require(len(vertices) == 3, f"interface face {face_id} is degenerate")
        for vertex_id in vertices:
            vertex_ribs.setdefault(vertex_id, set()).add(rib_id)
    _require(bool(observed_rib_ids),
             "no authenticated rib-channel owners are observed on interface faces")
    ambiguous = {
        vertex_id: sorted(rib_ids)
        for vertex_id, rib_ids in sorted(vertex_ribs.items())
        if len(rib_ids) > 1
    }
    local_by_rib = {rib_id: set() for rib_id in observed_rib_ids}
    for vertex_id, rib_ids in vertex_ribs.items():
        if len(rib_ids) == 1:
            local_by_rib[next(iter(rib_ids))].add(vertex_id)
    _require(
        all(local_by_rib[rib_id] for rib_id in observed_rib_ids),
        "an observed rib has no unambiguous interface vertices",
    )
    _require(required == observed_rib_ids,
             "required attachment inventory differs from observed authenticated rib owners")
    maps = {
        rib_id: sorted(vertex_ids)
        for rib_id, vertex_ids in sorted(local_by_rib.items())
    }
    diagnostics = {
        "ambiguous_shared_interface_vertex_count": len(ambiguous),
        "ambiguous_shared_interface_vertices": [
            {"local_vertex_id": vertex_id, "rib_stable_ids": rib_ids}
            for vertex_id, rib_ids in ambiguous.items()
        ],
    }
    if return_diagnostics:
        return maps, diagnostics
    return maps


def _authenticated_local_reconstruction_body20_pose(
    accepted_receipt_raw: bytes,
    *,
    expected_source_payload_sha256: str,
    accepted_receipt_path: str | Path,
    accepted_capture_pack_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Read Body20 only from the exact accepted step-0 receipt and pack."""
    expected_source_payload_sha256 = _local_reconstruction_hex_sha(
        expected_source_payload_sha256, "source ABI5 payload pin")
    _require(isinstance(accepted_receipt_raw, bytes) and accepted_receipt_raw,
             "accepted capture receipt bytes are missing")
    receipt_path, pack_path = Path(accepted_receipt_path), Path(accepted_capture_pack_path)
    _require(receipt_path.is_file() and receipt_path.read_bytes() == accepted_receipt_raw,
             "accepted receipt bytes differ from the pinned receipt path")
    _require(pack_path.is_file(), "accepted capture pack path is missing")
    pack_sha256 = sha_file(pack_path)
    _local_reconstruction_hex_sha(pack_sha256, "accepted capture pack hash")
    try:
        receipt = json.loads(accepted_receipt_raw)
        from .resting_anatomy_verify import validate_capture_receipt
        validate_capture_receipt(receipt, pack_sha256)
    except (TypeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            "local intercostal reconstruction: accepted receipt does not authenticate the pinned capture"
        ) from error
    _require(type(receipt.get("accepted_step")) is int and receipt["accepted_step"] == 0,
             "local intercostal reconstruction requires the exact accepted step-0 receipt")
    _require(receipt.get("common_field_source_anatomy_payload_sha256")
             == expected_source_payload_sha256,
             "accepted step-0 receipt does not bind the exact source anatomy payload")
    accepted_rows = [row for row in receipt["accepted_body_poses"]
                     if row.get("body_index") == 20]
    registered_rows = [row for row in receipt.get("accepted_registered_body_poses", [])
                       if row.get("body_index") == 20]
    _require(len(accepted_rows) == 1 and len(registered_rows) == 1
             and accepted_rows[0].get("position_m") == registered_rows[0].get("position_m")
             and accepted_rows[0].get("quaternion_xyzw") == registered_rows[0].get("quaternion_xyzw"),
             "accepted and registered receipt do not bind one exact Body20 pose")
    registered = registered_rows[0]
    q = np.asarray(registered["quaternion_xyzw"], dtype="<f4")
    t = np.asarray(registered["position_m"], dtype="<f4")
    _require(q.shape == (4,) and t.shape == (3,) and np.isfinite(q).all()
             and np.isfinite(t).all(), "accepted registered Body20 pose is malformed")
    return q, t, _local_reconstruction_sha256(accepted_receipt_raw), pack_sha256


def _local_reconstruction_runtime_attachment_map_sidecars(
    *,
    nha_stable_id: int,
    interface_maps: dict[int, list[int]],
    vertex_count: int,
    face_rib_owners: list[int | None],
    interface_face_mask,
    source_payload_sha256: str,
    compiled_payload_sha256: str,
    candidate_npz_sha256: str,
    candidate_faces_sha256: str,
    face_parent_lineage_sha256: str,
    accepted_receipt_sha256: str,
    accepted_capture_pack_sha256: str,
    parent_owner_report: dict,
) -> dict[str, dict]:
    """Emit current maps as the runtime receipt ABI, pinned to this source owner."""
    from .composed_bone_body_frame import _RUNTIME_ATTACHMENT_RIBS

    mask = np.asarray(interface_face_mask)
    _require(
        type(nha_stable_id) is int and nha_stable_id in (313, 315, 317)
        and mask.dtype == np.dtype(np.bool_)
        and mask.ndim == 1 and len(mask) == len(face_rib_owners)
        and len(interface_maps) > 0,
        "local reconstruction runtime map sidecar identity or interface mask is malformed",
    )
    output: dict[str, dict] = {}
    uint32_max = int(np.iinfo(np.uint32).max)
    for rib_id, raw_ids in sorted(interface_maps.items()):
        _require(
            type(rib_id) is int and rib_id in _RUNTIME_ATTACHMENT_RIBS
            and isinstance(raw_ids, list) and raw_ids
            and all(type(value) is int and 0 <= value <= uint32_max
                    and value < vertex_count for value in raw_ids)
            and len(raw_ids) == len(set(raw_ids)),
            "local reconstruction runtime map sidecar has invalid typed rib slots",
        )
        ids = sorted(raw_ids)
        ids_sha = _local_reconstruction_sha256(
            np.ascontiguousarray(ids, dtype="<u4").tobytes())
        interface_face_ids = [
            index for index, (owner, marked) in enumerate(
                zip(face_rib_owners, mask.tolist(), strict=True))
            if marked and owner == rib_id
        ]
        _require(interface_face_ids,
                 f"rib{rib_id} attachment map has no authenticated interface faces")
        face_ids_sha = _local_reconstruction_sha256(
            np.ascontiguousarray(interface_face_ids, dtype="<u4").tobytes())
        member, member_sha, channel, _source_range_count = _RUNTIME_ATTACHMENT_RIBS[rib_id]
        source_partition = {
            "schema": "numi.human.intercostal-local-reconstruction-interface-source.v1",
            "actual_capture_pack_sha256": accepted_capture_pack_sha256,
            "actual_capture_receipt_sha256": accepted_receipt_sha256,
            "base_anatomy_payload_sha256": source_payload_sha256,
            "candidate_npz_sha256": candidate_npz_sha256,
            "candidate_faces_sha256": candidate_faces_sha256,
            "face_parent_lineage_sha256": face_parent_lineage_sha256,
            "parent_owner_report_sha256": parent_owner_report["parent_owner_report_sha256"],
            "parent_payload_sha256": parent_owner_report["parent_payload_sha256"],
            "operation_ancestry_sha256": parent_owner_report["operation_ancestry_sha256"],
            "authenticated_source_report_sha256s": parent_owner_report[
                "authenticated_source_report_sha256s"],
            "typed_source_rib_member": {
                "stable_id": rib_id,
                "member_id": member,
                "member_sha256": member_sha,
                "channel_index": channel,
            },
            "interface_face_ids_sha256": face_ids_sha,
            "interface_face_count": len(interface_face_ids),
            "interface_vertex_count": len(ids),
            "anatomical_admission": False,
            "native_admission": False,
            "retained_motion_admission": False,
        }
        output[str(rib_id)] = {
            "schema": f"numi.human.intercostal{nha_stable_id}-rib{rib_id}-interface-motion-attachment.v1",
            "status": "inferred_passive_visual_attachment_source_preparation_only",
            "runtime_claim_boundary": (
                "The existing typed rib-kind deformation is reused only for this exact "
                "source-owned interface vertex subset; no force, mass, tissue model, or "
                "collision claim is added."
            ),
            "anatomy_payload_sha256": compiled_payload_sha256,
            "source_partition": source_partition,
            "surface": {
                "semantic": 51010,
                "stable_id": nha_stable_id,
                "body_index": 20,
                "layer": 1,
                "interface_local_vertex_ids": ids,
                "interface_local_vertex_ids_sha256": ids_sha,
                "local_vertex_count": vertex_count,
            },
            "existing_rib_channel": {
                "stable_id": rib_id,
                "semantic": 51004,
                "body_index": 20,
                "channel_kind": 5,
                "channel_index": channel,
                "source_member_id": member,
                "source_member_sha256": member_sha,
            },
        }
    return output


def _authenticated_local_reconstruction_partial_source_positions(
    world_positions_f32, source_row, *, stable_id: int,
    expected_source_payload_sha256: str, candidate_npz_sha256: str,
    q_f32, t_f32, spec: dict,
):
    """Retain only exact source-slot correspondences and invert every other point.

    The crosswalk report binds a candidate NPZ, a native source-allocation map,
    and the parent ABI5 source. Its zero placeholders are never read for emitted
    positions: only retained-mask entries are copied; all remaining points go
    through the existing Body20 inverse owner.
    """
    import io

    _require(stable_id == 315 and isinstance(spec, dict),
             "partial source-position retention is currently scoped to typed row 315")
    trusted_pins = {
        "crosswalk_npz_sha256": LOCAL_315_RETAINED_CROSSWALK_NPZ_SHA256,
        "crosswalk_report_sha256": LOCAL_315_RETAINED_CROSSWALK_REPORT_SHA256,
        "reference_candidate_npz_sha256": LOCAL_315_RETAINED_CANDIDATE_NPZ_SHA256,
        "source_capture_npz_sha256": LOCAL_315_SOURCE_CAPTURE_NPZ_SHA256,
        "source_capture_report_sha256": LOCAL_315_SOURCE_CAPTURE_REPORT_SHA256,
    }

    def pinned_bytes(field: str, digest_field: str, label: str) -> tuple[bytes, str]:
        raw = spec.get(field)
        digest = _local_reconstruction_hex_sha(spec.get(digest_field), label + " SHA-256")
        _require(isinstance(raw, bytes) and _local_reconstruction_sha256(raw) == digest,
                 label + " bytes differ from their independent pin")
        return raw, digest

    crosswalk_raw, crosswalk_sha = pinned_bytes(
        "crosswalk_npz_raw", "crosswalk_npz_sha256", "source-position crosswalk")
    report_raw, report_sha = pinned_bytes(
        "crosswalk_report_raw", "crosswalk_report_sha256", "source-position crosswalk report")
    reference_candidate_raw, reference_candidate_sha = pinned_bytes(
        "reference_candidate_npz_raw", "reference_candidate_npz_sha256",
        "crosswalk reference candidate NPZ")
    source_capture_raw, source_capture_sha = pinned_bytes(
        "source_capture_npz_raw", "source_capture_npz_sha256", "native source-allocation map NPZ")
    source_capture_report_raw, source_capture_report_sha = pinned_bytes(
        "source_capture_report_raw", "source_capture_report_sha256",
        "native source-allocation extraction report")
    _require(
        {"crosswalk_npz_sha256": crosswalk_sha,
         "crosswalk_report_sha256": report_sha,
         "reference_candidate_npz_sha256": reference_candidate_sha,
         "source_capture_npz_sha256": source_capture_sha,
         "source_capture_report_sha256": source_capture_report_sha} == trusted_pins
        and expected_source_payload_sha256 == LOCAL_315_SOURCE_PAYLOAD_SHA256,
        "partial source-position evidence is not the pinned 1176/005 owner bundle")
    try:
        crosswalk_report = json.loads(report_raw)
        source_capture_report = json.loads(source_capture_report_raw)
        with np.load(io.BytesIO(crosswalk_raw), allow_pickle=False) as archive:
            crosswalk = {key: archive[key].copy() for key in archive.files}
        with np.load(io.BytesIO(reference_candidate_raw), allow_pickle=False) as archive:
            reference_candidate = {key: archive[key].copy() for key in archive.files}
        with np.load(io.BytesIO(source_capture_raw), allow_pickle=False) as archive:
            source_capture = {key: archive[key].copy() for key in archive.files}
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        raise ValueError("partial source-position retention evidence is malformed") from error

    _require(set(crosswalk) == {
        "candidate_to_source_vertex_i64", "retained_source_local_positions_f32", "retained_mask",
    }, "source-position crosswalk NPZ keys changed")
    reference_candidate_keys = set(reference_candidate)
    reference_candidate_required = {"positions_world_f32", "faces"}
    reference_candidate_allowed = reference_candidate_required | {
        "source_original_id", "source_face_id",
    }
    _require(
        reference_candidate_required <= reference_candidate_keys
        and reference_candidate_keys <= reference_candidate_allowed,
        "crosswalk reference candidate NPZ keys changed",
    )
    # These two arrays are operation-local hints embedded by the candidate
    # writer. Their values are deliberately not used for source ownership;
    # only the hash-pinned crosswalk/report establishes retained slots.
    _require(set(source_capture) == {
        "positions_world_f32", "faces_u32", "normals_f32", "captured_vertex_records_f32",
        "capture_global_vertex_ids_u32", "source_to_capture_vertex_map_i64",
        "capture_faces_global_u32", "capture_faces_compact_u32", "primitive_record_u32",
        "source_record_u32",
    }, "native source-allocation map NPZ keys changed")

    world = np.asarray(world_positions_f32)
    candidate_positions = np.asarray(reference_candidate["positions_world_f32"])
    candidate_faces = np.asarray(reference_candidate["faces"])
    source_indices = np.asarray(crosswalk["candidate_to_source_vertex_i64"])
    retained_local = np.asarray(crosswalk["retained_source_local_positions_f32"])
    retained_mask = np.asarray(crosswalk["retained_mask"])
    _require(
        world.dtype == np.dtype("<f4") and world.ndim == 2 and world.shape[1] == 3
        and np.isfinite(world).all()
        and candidate_positions.dtype == np.dtype("<f4")
        and candidate_positions.shape == world.shape
        and np.isfinite(candidate_positions).all()
        and candidate_faces.dtype.kind in "iu" and candidate_faces.ndim == 2
        and candidate_faces.shape[1] == 3 and candidate_faces.size > 0
        and np.array_equal(np.ascontiguousarray(candidate_positions, dtype="<f4").view("<u4"),
                           np.ascontiguousarray(world, dtype="<f4").view("<u4")),
        "crosswalk reference candidate positions differ from the current pinned candidate")
    _require(candidate_npz_sha256 == reference_candidate_sha,
             "partial source-position crosswalk is bound to a different candidate NPZ")
    _require(
        source_indices.dtype == np.dtype("<i8") and source_indices.shape == (len(world),)
        and retained_local.dtype == np.dtype("<f4") and retained_local.shape == world.shape
        and retained_mask.dtype == np.dtype(np.bool_) and retained_mask.shape == (len(world),)
        and np.array_equal(retained_mask, source_indices >= 0)
        and np.all(source_indices >= -1),
        "source-position crosswalk mask, sentinels, or allocation changed")

    inputs = crosswalk_report.get("inputs")
    _require(
        crosswalk_report.get("status") == "PASS_EXACT_RETAINED_SOURCE_POSITION_CROSSWALK_ONLY",
        "source-position crosswalk report is not the retained-position owner result")
    _require(
        isinstance(inputs, dict)
        and crosswalk_report.get("output_sha256") == crosswalk_sha
        and crosswalk_report.get("candidate_vertex_count") == len(world)
        and crosswalk_report.get("anatomical_admission") is False
        and crosswalk_report.get("native_admission") is False
        and crosswalk_report.get("ambiguous_local_alias_count") == 0,
        "source-position crosswalk report identity or admission state is invalid")

    def report_input_sha256(basename: str) -> str:
        matches = [value for path, value in inputs.items() if Path(path).name == basename]
        _require(len(matches) == 1 and isinstance(matches[0], str) and len(matches[0]) == 64,
                 "source-position crosswalk report lacks a unique pinned " + basename)
        return matches[0]

    crosswalk_candidate_sha = report_input_sha256("row-315-candidate.npz")
    crosswalk_source_payload_sha = report_input_sha256("compiled-emission-attempt.nhanatomy")
    crosswalk_capture_map_sha = report_input_sha256("row-315-native-source-step0.npz")
    _require(
        crosswalk_candidate_sha == reference_candidate_sha
        and crosswalk_source_payload_sha == expected_source_payload_sha256
        and crosswalk_capture_map_sha == source_capture_sha,
        "crosswalk input hashes do not bind the supplied candidate, source payload, and native map")
    _require(
        crosswalk_report.get("output_path", "").endswith("retained-source-vertex-crosswalk.npz")
        and crosswalk_report.get("retained_exact_world_and_unambiguous_local_count")
            == int(np.count_nonzero(retained_mask))
        and crosswalk_report.get("generated_or_changed_point_count")
            == int(np.count_nonzero(~retained_mask))
        and crosswalk_report.get("ambiguous_local_alias_count") == 0
        and crosswalk_report.get("retained_duplicate_world_alias_count") == 0,
        "source-position crosswalk report counts differ from its pinned map")

    source_report_row = source_capture_report.get("rows", {}).get("315", {})
    source_local = np.asarray(source_row["vertices6"][:, :3])
    source_local_normals = np.asarray(source_row["vertices6"][:, 3:])
    source_local_sha = _local_reconstruction_hex_sha(
        spec.get("source_row_local_positions_sha256"), "source-row local positions SHA-256")
    _require(source_local.dtype == np.dtype("<f4")
             and _local_reconstruction_sha256(
                 np.ascontiguousarray(source_local, dtype="<f4").tobytes()) == source_local_sha,
             "source-row local positions differ from their independent pin")
    source_world = np.asarray(source_capture.get("positions_world_f32"))
    source_faces = np.asarray(source_capture.get("faces_u32"))
    source_normals = np.asarray(source_capture.get("normals_f32"))
    source_records = np.asarray(source_capture.get("captured_vertex_records_f32"))
    global_ids = np.asarray(source_capture.get("capture_global_vertex_ids_u32"))
    source_map = np.asarray(source_capture.get("source_to_capture_vertex_map_i64"))
    capture_faces_global = np.asarray(source_capture.get("capture_faces_global_u32"))
    capture_faces_compact = np.asarray(source_capture.get("capture_faces_compact_u32"))
    source_record = np.asarray(source_capture.get("source_record_u32"))
    primitive_record = np.asarray(source_capture.get("primitive_record_u32"))
    source_crosswalk = source_report_row.get("ordered_face_corner_crosswalk", {})
    source_base = source_report_row.get("native_source_allocation_base")
    _require(
        source_capture_report.get("status") == "PASS_NATIVE_REJECTED_FRAME_ARRAYS_EXTRACTED_DIAGNOSTIC_ONLY"
        and source_capture_report.get("source_capture_kind") == "rejected_surface_audit_step0_not_accepted_capture"
        and source_capture_report.get("anatomical_admission") is False
        and source_capture_report.get("native_scene_admission") is False
        and source_capture_report.get("motion_admission") is False
        and source_capture_report.get("source_payload_sha256") == expected_source_payload_sha256
        and source_report_row.get("typed_identity") == [51010, 315]
        and source_report_row.get("body_index") == 20
        and source_report_row.get("layer") == 1
        and source_report_row.get("flags") == 0
        and source_report_row.get("source_vertex_count") == len(source_local)
        and source_report_row.get("index_count") == len(source_faces.reshape(-1))
        and source_report_row.get("source_payload_sha256") == expected_source_payload_sha256
        and isinstance(source_base, int)
        and [value for path, value in source_capture_report.get("input_sha256", {}).items()
             if Path(path).name == "compiled-emission-attempt.nhanatomy"]
            == [expected_source_payload_sha256]
        and source_report_row.get("captured_positions_world_f32_file_sha256") == source_capture_sha
        and source_capture_report_sha == _local_reconstruction_hex_sha(
            spec.get("source_capture_report_sha256"), "native source-map report SHA-256"),
        "native source-position evidence is not pinned to the current Body20/layer1 ABI5 source row")
    _require(
        isinstance(source_crosswalk, dict)
        and source_crosswalk.get("status") == "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED"
        and source_crosswalk.get("source_face_count") == len(source_faces)
        and source_crosswalk.get("capture_face_count") == len(capture_faces_global)
        and source_crosswalk.get("source_faces_u32_sha256")
            == source_report_row.get("source_faces_u32_sha256"),
        "native source map lacks the exact ordered face-corner identity")
    _require(
        source_local.dtype == np.dtype("<f4") and source_local_normals.dtype == np.dtype("<f4")
        and source_local.shape == source_world.shape
        and source_world.dtype == np.dtype("<f4")
        and source_normals.dtype == np.dtype("<f4")
        and source_records.dtype == np.dtype("<f4")
        and source_faces.dtype == np.dtype("<u4")
        and global_ids.dtype == np.dtype("<u4")
        and source_map.dtype == np.dtype("<i8")
        and capture_faces_global.dtype == np.dtype("<u4")
        and capture_faces_compact.dtype == np.dtype("<u4")
        and source_record.dtype == np.dtype("<u4")
        and primitive_record.dtype == np.dtype("<u4")
        and source_world.shape == source_local.shape
        and source_normals.shape == source_local.shape
        and source_records.shape == (len(source_local), 20)
        and global_ids.shape == source_map.shape == (len(source_local),)
        and source_faces.shape == capture_faces_global.shape == capture_faces_compact.shape
        and np.array_equal(global_ids, source_base + np.arange(len(source_local), dtype=np.uint32))
        and np.array_equal(source_world, source_records[:, :3])
        and np.array_equal(source_normals, source_records[:, 4:7])
        and np.array_equal(source_record, np.asarray(source_row["record"], dtype=np.uint32))
        and np.array_equal(source_record, np.asarray(source_report_row["source_record"], dtype=np.uint32))
        and np.array_equal(source_faces, np.asarray(source_row["faces"], dtype=np.uint32))
        and np.array_equal(capture_faces_global, source_base + source_faces)
        and np.array_equal(primitive_record,
                           np.asarray(source_report_row["captured_primitive_record"], dtype=np.uint32))
        and np.array_equal(capture_faces_compact,
                           np.unique(capture_faces_global.reshape(-1), return_inverse=True)[1]
                           .reshape(capture_faces_compact.shape)),
        "native source-allocation map arrays differ from the exact ABI5 row or typed extraction report")
    used_global = np.unique(capture_faces_global.reshape(-1))
    source_referenced = np.zeros(len(source_local), dtype=np.bool_)
    source_referenced[source_faces.reshape(-1)] = True
    _require(
        source_map.shape == source_referenced.shape
        and np.all(source_map[~source_referenced] == -1)
        and np.all(source_map[source_referenced] >= 0)
        and np.all(source_map[source_referenced] < len(used_global))
        and np.array_equal(used_global[source_map[source_referenced]],
                           source_base + np.flatnonzero(source_referenced).astype(np.uint32))
        and _local_reconstruction_sha256(source_world.tobytes())
            == source_report_row.get("captured_positions_world_f32_sha256")
        and _local_reconstruction_sha256(source_normals.tobytes())
            == source_report_row.get("captured_normals_f32_sha256")
        and _local_reconstruction_sha256(source_records.tobytes())
            == source_report_row.get("captured_full_vertex_records_f32_sha256")
        and _local_reconstruction_sha256(source_faces.tobytes())
            == source_report_row.get("source_faces_u32_sha256")
        and _local_reconstruction_sha256(capture_faces_global.tobytes())
            == source_report_row.get("captured_faces_global_u32_sha256")
        and _local_reconstruction_sha256(source_capture_raw)
            == source_report_row.get("captured_positions_world_f32_file_sha256"),
        "native source-allocation index/crosswalk hashes do not authenticate source slots")

    candidate_source_slots = source_indices[retained_mask]
    _require(
        np.all(candidate_source_slots < len(source_local))
        and np.array_equal(
            np.ascontiguousarray(world[retained_mask], dtype="<f4").view("<u4"),
            np.ascontiguousarray(source_world[candidate_source_slots], dtype="<f4").view("<u4"))
        and np.array_equal(
            np.ascontiguousarray(retained_local[retained_mask], dtype="<f4").view("<u4"),
            np.ascontiguousarray(source_local[candidate_source_slots], dtype="<f4").view("<u4")),
        "retained candidate vertices are not exact world/source-local matches to authenticated source slots")

    retained_count = int(np.count_nonzero(retained_mask))
    inverse_mask = ~retained_mask
    _require(retained_count > 0 and np.any(inverse_mask),
             "partial source-position mode must retain and invert nonempty vertex subsets")
    local_positions = np.empty(world.shape, dtype="<f4")
    local_positions[retained_mask] = retained_local[retained_mask]
    inverse_local, inverse_replay, inverse_report = _inverse_body20_positions_f32(
        world[inverse_mask], q_f32, t_f32,
    )
    local_positions[inverse_mask] = inverse_local
    _require(np.isfinite(local_positions).all(),
             "partial source-position binding emitted non-finite local coordinates")
    unused_normal_fallback = np.zeros(world.shape, dtype="<f4")
    unused_normal_fallback[retained_mask] = source_local_normals[candidate_source_slots]
    binding_report = {
        "mode": "authenticated_partial_source_row_positions_retained_and_changed_points_inverted",
        "source_position_inverse_performed": True,
        "source_position_inverse_applied_vertex_count": int(np.count_nonzero(inverse_mask)),
        "source_position_retained_vertex_count": retained_count,
        "candidate_to_source_vertex_map_sha256": _local_reconstruction_sha256(source_indices.tobytes()),
        "retained_mask_sha256": _local_reconstruction_sha256(retained_mask.tobytes()),
        "retained_source_local_positions_sha256": _local_reconstruction_sha256(
            np.ascontiguousarray(retained_local[retained_mask], dtype="<f4").tobytes()),
        "source_row_payload_sha256": expected_source_payload_sha256,
        "source_row_local_positions_sha256": _local_reconstruction_sha256(
            np.ascontiguousarray(source_local, dtype="<f4").tobytes()),
        "candidate_npz_sha256": candidate_npz_sha256,
        "crosswalk_npz_sha256": crosswalk_sha,
        "crosswalk_report_sha256": report_sha,
        "crosswalk_reference_candidate_npz_sha256": reference_candidate_sha,
        "source_capture_npz_sha256": source_capture_sha,
        "source_capture_report_sha256": source_capture_report_sha,
        "source_capture_kind": source_capture_report["source_capture_kind"],
        "retained_world_positions_exact_source_capture": True,
        "retained_local_positions_exact_source_row": True,
        "generated_position_placeholders_consumed": False,
        "changed_points_body20_inverse": inverse_report,
        "native_replay_required_for_emitted_world_geometry": True,
        "anatomical_admission": False,
        "native_admission": False,
    }
    return local_positions, binding_report, unused_normal_fallback


def _authenticated_candidate_source_local_positions(
    candidate_npz_raw, world_positions_f32, *, q_f32, t_f32, spec,
):
    """Retain pinned repair coordinates only after exact existing-owner replay.

    These coordinates are inferred repairs, not measurements. Candidate hashing
    and Float32 replay establish emission identity; geometry and attachments
    still require their existing independent gates.
    """
    import io

    key = spec.get("candidate_source_local_positions_key")
    _require(isinstance(key, str) and bool(key),
             "candidate source-local position key is malformed")
    candidate_sha = _local_reconstruction_hex_sha(
        spec.get("candidate_npz_sha256"), "candidate NPZ pin")
    _require(isinstance(candidate_npz_raw, bytes)
             and _local_reconstruction_sha256(candidate_npz_raw) == candidate_sha,
             "candidate source-local NPZ differs from its independent pin")
    local_sha = _local_reconstruction_hex_sha(
        spec.get("candidate_source_local_positions_sha256"),
        "candidate source-local position pin")
    try:
        with np.load(io.BytesIO(candidate_npz_raw), allow_pickle=False) as archive:
            local = archive[key].copy()
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise ValueError("candidate source-local positions are missing or malformed") from error
    world = np.asarray(world_positions_f32)
    _require(local.dtype == np.dtype("<f4")
             and world.dtype == np.dtype("<f4")
             and local.ndim == 2 and local.shape[1] == 3
             and len(local) > 0 and local.shape == world.shape
             and np.isfinite(local).all() and np.isfinite(world).all(),
             "candidate source-local positions must match finite Float32 world allocation")
    local = np.ascontiguousarray(local, dtype="<f4")
    world = np.ascontiguousarray(world, dtype="<f4")
    _require(_local_reconstruction_sha256(local.tobytes()) == local_sha,
             "candidate source-local position bytes differ from their independent pin")
    replay = np.asarray(_load_local_body20_transform()(q_f32, t_f32, local))
    _require(replay.dtype == np.dtype("<f4")
             and replay.shape == world.shape
             and np.ascontiguousarray(replay).tobytes() == world.tobytes(),
             "candidate source-local positions do not exactly replay through authenticated Body20")
    return local, {
        "mode": "pinned_repair_source_local_positions_exact_forward_replay",
        "source_position_inverse_performed": False,
        "candidate_npz_sha256": candidate_sha,
        "candidate_source_local_positions_key": key,
        "source_row_local_positions_sha256": local_sha,
        "candidate_world_positions_f32_sha256": _local_reconstruction_sha256(world.tobytes()),
        "replayed_world_positions_f32_sha256": _local_reconstruction_sha256(replay.tobytes()),
        "source_local_positions_byte_exact": True,
        "source_measurement_claim": False,
        "anatomical_admission": False,
        "native_replay_required_for_emitted_world_geometry": True,
    }


def compile_local_reconstruction_intercostal_payload(
    raw_payload: bytes,
    replacements: dict[int, dict],
    *,
    expected_source_payload_sha256: str,
    accepted_receipt_raw: bytes,
    accepted_receipt_path: str | Path,
    accepted_capture_pack_path: str | Path,
) -> tuple[bytes, dict]:
    """Re-emit selected Body20 rows from local reconstruction inputs via ABI5.

    Each replacement supplies pinned captured-world Float32 positions, child
    faces, explicit face-parent lineage, an interface-face mask, the exact
    parent face array, and its hash-pinned owner report. By default, source-local
    positions are obtained with the authenticated Body20 inverse. A caller may
    explicitly request preserve_source_row_local_positions when faces/maps are
    being rebound to a source row whose entire local position allocation is
    authenticated. Row 315 may instead carry an authenticated partial source
    position crosswalk: exact source slots are retained and only generated or
    changed points pass through the existing Body20 inverse. A pinned candidate
    source-local array may instead be preserved when its forward Float32 replay
    through that same authenticated Body20 pose exactly equals the candidate
    world array. This does not confer anatomical or attachment admission.
    Normals are regenerated from the candidate geometry.
    Rib maps are limited to typed owners observed in authenticated parent
    lineage; unobserved channels remain absent. Admission stays false.
    """
    _require(isinstance(replacements, dict) and replacements
             and set(replacements) <= {313, 315, 317},
             "local reconstruction replacement IDs are outside typed intercostal rows")
    expected_source_payload_sha256 = _local_reconstruction_hex_sha(
        expected_source_payload_sha256, "source ABI5 payload pin")
    _require(isinstance(raw_payload, bytes)
             and _local_reconstruction_sha256(raw_payload) == expected_source_payload_sha256,
             "source ABI5 payload bytes differ from their independent pin")
    q, t, accepted_receipt_sha256, accepted_capture_pack_sha256 = (
        _authenticated_local_reconstruction_body20_pose(
            accepted_receipt_raw,
            expected_source_payload_sha256=expected_source_payload_sha256,
            accepted_receipt_path=accepted_receipt_path,
            accepted_capture_pack_path=accepted_capture_pack_path,
        )
    )
    pose_sha = accepted_receipt_sha256
    from .resting_pleura_proxy import _parse_payload

    _header, _records, source_rows = _parse_payload(raw_payload)
    composed_rows: dict[int, dict] = {}
    row_reports: dict[str, dict] = {}
    attachment_maps: dict[str, dict[int, list[int]]] = {}
    attachment_sidecar_inputs: dict[int, dict] = {}
    for stable_id, spec in sorted(replacements.items()):
        _require(type(stable_id) is int and isinstance(spec, dict),
                 "replacement stable ID or specification is malformed")
        source_row = source_rows.get(stable_id)
        _require(
            isinstance(source_row, dict)
            and source_row.get("body_index") == 20
            and source_row.get("layer") == 1 and source_row.get("flags") == 0,
            f"stable ID {stable_id} is not the typed Body20/layer1/flags0 row",
        )
        preserve_source_row_local_positions = spec.get(
            "preserve_source_row_local_positions", False)
        _require(type(preserve_source_row_local_positions) is bool,
                 f"stable ID {stable_id} source-local-position mode is malformed")
        partial_source_position_spec = spec.get("partial_source_position_retention")
        _require(partial_source_position_spec is None or isinstance(partial_source_position_spec, dict),
                 f"stable ID {stable_id} partial source-position crosswalk is malformed")
        candidate_source_local_mode = spec.get("candidate_source_local_positions_key") is not None
        _require(sum((preserve_source_row_local_positions,
                      partial_source_position_spec is not None,
                      candidate_source_local_mode)) <= 1,
                 f"stable ID {stable_id} cannot combine source-position retention modes")
        _require(
            not (preserve_source_row_local_positions
                 and spec.get("source_emission_diagnostic_only") is True),
            f"stable ID {stable_id} diagnostic emission cannot use source-local map mode",
        )
        if spec.get("source_emission_diagnostic_only") is True:
            import io

            candidate_npz_raw = spec.get("candidate_npz_raw")
            candidate_npz_sha256 = _local_reconstruction_hex_sha(
                spec.get("candidate_npz_sha256"),
                f"stable ID {stable_id} candidate NPZ hash")
            _require(isinstance(candidate_npz_raw, bytes)
                     and _local_reconstruction_sha256(candidate_npz_raw) == candidate_npz_sha256,
                     f"stable ID {stable_id} diagnostic candidate NPZ differs from its pin")
            try:
                with np.load(io.BytesIO(candidate_npz_raw), allow_pickle=False) as candidate_npz:
                    candidate_positions_raw = np.asarray(candidate_npz["positions_world_f32"])
                    candidate_faces_raw = np.asarray(candidate_npz["faces"])
            except (KeyError, ValueError, OSError) as error:
                raise ValueError(
                    f"local intercostal reconstruction: stable ID {stable_id} diagnostic NPZ is invalid"
                ) from error
            _require(
                candidate_positions_raw.dtype.kind == "f"
                and candidate_positions_raw.dtype.itemsize == 4
                and candidate_positions_raw.ndim == 2
                and candidate_positions_raw.shape[1] == 3
                and len(candidate_positions_raw) > 0
                and np.isfinite(candidate_positions_raw).all()
                and np.issubdtype(candidate_faces_raw.dtype, np.integer)
                and np.all(candidate_faces_raw >= 0)
                and np.all(candidate_faces_raw <= np.iinfo(np.uint32).max),
                f"stable ID {stable_id} diagnostic NPZ geometry is not typed finite ABI5 input",
            )
            world_positions = np.ascontiguousarray(candidate_positions_raw, dtype="<f4")
            faces = np.ascontiguousarray(candidate_faces_raw, dtype="<u4")
            supplied_positions = np.asarray(spec.get("world_positions_f32"))
            supplied_faces = np.asarray(spec.get("faces"))
            _require(
                supplied_positions.dtype.kind == "f" and supplied_positions.dtype.itemsize == 4
                and supplied_positions.shape == world_positions.shape
                and np.isfinite(supplied_positions).all()
                and supplied_faces.dtype.kind in "iu"
                and supplied_faces.shape == faces.shape
                and np.all(supplied_faces >= 0)
                and np.all(supplied_faces <= np.iinfo(np.uint32).max)
                and np.ascontiguousarray(supplied_positions, dtype="<f4").tobytes()
                    == world_positions.tobytes()
                and np.ascontiguousarray(supplied_faces, dtype="<u4").tobytes()
                    == faces.tobytes(),
                f"stable ID {stable_id} diagnostic arrays differ from the exact NPZ",
            )
            _require(
                faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
                and int(faces.max(initial=0)) < len(world_positions)
                and np.all(faces[:, 0] != faces[:, 1])
                and np.all(faces[:, 1] != faces[:, 2])
                and np.all(faces[:, 2] != faces[:, 0]),
                f"stable ID {stable_id} diagnostic faces are malformed",
            )
            world_normals_f32 = np.ascontiguousarray(
                _surface_normals_allow_unused_slots(world_positions, faces), dtype="<f4",
            )
            world_vertices6 = np.ascontiguousarray(
                np.concatenate((world_positions, world_normals_f32), axis=1), dtype="<f4")
            partial_unused_normal_fallback = None
            if candidate_source_local_mode:
                local_positions, inverse_report = _authenticated_candidate_source_local_positions(
                    candidate_npz_raw, world_positions, q_f32=q, t_f32=t, spec=spec)
            elif partial_source_position_spec is not None:
                local_positions, inverse_report, partial_unused_normal_fallback = (
                    _authenticated_local_reconstruction_partial_source_positions(
                        world_positions, source_row, stable_id=stable_id,
                        expected_source_payload_sha256=expected_source_payload_sha256,
                        candidate_npz_sha256=candidate_npz_sha256,
                        q_f32=q, t_f32=t, spec=partial_source_position_spec,
                    )
                )
            else:
                local_positions, _replayed_world, inverse_report = _inverse_body20_positions_f32(
                    world_positions, q, t,
                )
            local_normals = _localize_body20_normals_f32(
                world_normals_f32, faces, q,
                unused_normal_fallback_f32=partial_unused_normal_fallback,
            )
            vertices6 = np.ascontiguousarray(
                np.concatenate((local_positions, local_normals), axis=1), dtype="<f4")
            _require(np.isfinite(vertices6).all(),
                     f"stable ID {stable_id} diagnostic emitted non-finite ABI5 vertices")
            row = {
                **source_row,
                "vertices6": vertices6,
                "faces": np.ascontiguousarray(faces, dtype=np.int64),
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
            composed_rows[stable_id] = row
            attachment_maps[str(stable_id)] = {}
            row_reports[str(stable_id)] = {
                "stable_id": stable_id,
                "source_parent_row_sha256": _local_reconstruction_sha256(
                    _normalized_row_bytes(source_row)),
                "compiled_row_sha256": _local_reconstruction_sha256(
                    _normalized_row_bytes(row)),
                "candidate_npz_sha256": candidate_npz_sha256,
                "candidate_world_positions_f32_sha256": _local_reconstruction_sha256(
                    world_positions.tobytes()),
                "derived_world_normals_f32_sha256": _local_reconstruction_sha256(
                    world_normals_f32.tobytes()),
                "candidate_faces_sha256": _local_reconstruction_sha256(faces.tobytes()),
                "body20_inverse_and_replay": inverse_report,
                "body20_source_position_binding": inverse_report,
                "candidate_source_local_positions_preserved": candidate_source_local_mode,
                "source_row_local_positions_partially_preserved": (
                    partial_source_position_spec is not None),
                "source_row_local_positions_sha256": (
                    inverse_report.get("source_row_local_positions_sha256")
                    if partial_source_position_spec is not None or candidate_source_local_mode else None),
                "native_replay_required_for_emitted_world_geometry": True,
                "attachment_ownership_status": "unresolved_and_unmapped",
                "unowned_child_face_count": len(faces),
                "unowned_interface_face_count": None,
                "interface_map_vertex_counts": {},
                "source_emission_diagnostic_only": True,
                "anatomical_admission": False,
                "native_admission": False,
            }
            continue
        _require(
            "source_emission_diagnostic_only" not in spec
            or type(spec["source_emission_diagnostic_only"]) is bool,
            f"stable ID {stable_id} source-emission mode flag is malformed",
        )
        candidate_npz_raw = spec.get("candidate_npz_raw")
        candidate_npz_sha256 = _local_reconstruction_hex_sha(
            spec.get("candidate_npz_sha256"), f"stable ID {stable_id} candidate NPZ hash")
        _require(isinstance(candidate_npz_raw, bytes)
                 and _local_reconstruction_sha256(candidate_npz_raw) == candidate_npz_sha256,
                 f"stable ID {stable_id} candidate NPZ bytes differ from their pin")
        import io
        try:
            with np.load(io.BytesIO(candidate_npz_raw), allow_pickle=False) as candidate_npz:
                candidate_positions_raw = np.asarray(candidate_npz["positions_world_f32"])
                candidate_faces_raw = np.asarray(candidate_npz["faces"])
        except (KeyError, ValueError, OSError) as error:
            raise ValueError(f"local intercostal reconstruction: stable ID {stable_id} candidate NPZ is invalid") from error
        _require(
            candidate_positions_raw.dtype.kind == "f"
            and candidate_positions_raw.dtype.itemsize == 4
            and candidate_positions_raw.ndim == 2
            and candidate_positions_raw.shape[1] == 3
            and len(candidate_positions_raw) > 0
            and np.isfinite(candidate_positions_raw).all(),
            f"stable ID {stable_id} candidate NPZ positions must be finite typed Float32",
        )
        candidate_positions = np.ascontiguousarray(candidate_positions_raw, dtype="<f4")
        _require(
            np.issubdtype(candidate_faces_raw.dtype, np.integer)
            and np.all(candidate_faces_raw >= 0)
            and np.all(candidate_faces_raw <= np.iinfo(np.uint32).max),
            f"stable ID {stable_id} candidate NPZ face indices exceed the ABI5 uint32 range",
        )
        candidate_faces = np.ascontiguousarray(candidate_faces_raw, dtype="<u4")
        raw_vertices = np.asarray(spec.get("world_positions_f32"))
        raw_faces = np.asarray(spec.get("faces"))
        raw_parent_faces = np.asarray(spec.get("parent_faces"))
        _require(
            raw_vertices.dtype.kind == "f" and raw_vertices.dtype.itemsize == 4
            and np.issubdtype(raw_faces.dtype, np.integer)
            and np.all(raw_faces >= 0)
            and np.all(raw_faces <= np.iinfo(np.uint32).max)
            and np.issubdtype(raw_parent_faces.dtype, np.integer)
            and np.all(raw_parent_faces >= 0)
            and np.all(raw_parent_faces <= np.iinfo(np.uint32).max),
            f"stable ID {stable_id} candidate arrays do not preserve typed Float32/face inputs",
        )
        world_positions = np.ascontiguousarray(raw_vertices, dtype="<f4")
        faces = np.ascontiguousarray(raw_faces, dtype="<u4")
        _require(world_positions.tobytes() == candidate_positions.tobytes()
                 and faces.tobytes() == candidate_faces.tobytes(),
                 f"stable ID {stable_id} emitted geometry differs from the pinned candidate NPZ")
        world_normals_f32 = np.ascontiguousarray(
            _surface_normals_allow_unused_slots(world_positions, faces), dtype="<f4",
        )
        world_vertices6 = np.ascontiguousarray(
            np.concatenate((world_positions, world_normals_f32), axis=1), dtype="<f4")
        parent_faces = np.ascontiguousarray(raw_parent_faces, dtype="<u4")
        face_parent_ids = spec.get("face_parent_ids")
        interface_mask = spec.get("interface_face_mask")
        raw_interface = np.asarray(interface_mask)
        _require(raw_interface.dtype == np.dtype(np.bool_),
                 f"stable ID {stable_id} interface face mask is not boolean")
        _require(
            world_positions.ndim == 2 and world_positions.shape[1] == 3
            and len(world_positions) > 0 and np.isfinite(world_vertices6).all()
            and faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
            and int(faces.max(initial=0)) < len(world_vertices6)
            and np.all(faces[:, 0] != faces[:, 1])
            and np.all(faces[:, 1] != faces[:, 2])
            and np.all(faces[:, 2] != faces[:, 0])
            and isinstance(face_parent_ids, list) and len(face_parent_ids) == len(faces),
            f"stable ID {stable_id} candidate arrays or face lineage are malformed",
        )
        owner_report_raw = spec.get("parent_face_owner_report_raw")
        _require(isinstance(owner_report_raw, bytes),
                 f"stable ID {stable_id} authenticated parent owner report is missing")
        parent_payload_raw = spec.get("parent_payload_raw")
        operation_ancestry_raw = spec.get("operation_ancestry_raw")
        _require(isinstance(parent_payload_raw, bytes),
                 f"stable ID {stable_id} parent candidate bytes are missing")
        try:
            with np.load(io.BytesIO(parent_payload_raw), allow_pickle=False) as parent_npz:
                parent_key = spec.get("parent_faces_key", "faces")
                pinned_parent_faces_raw = np.asarray(parent_npz[parent_key])
        except (KeyError, ValueError, OSError) as error:
            raise ValueError(f"local intercostal reconstruction: stable ID {stable_id} parent NPZ is invalid") from error
        _require(
            np.issubdtype(pinned_parent_faces_raw.dtype, np.integer)
            and np.all(pinned_parent_faces_raw >= 0)
            and np.all(pinned_parent_faces_raw <= np.iinfo(np.uint32).max),
            f"stable ID {stable_id} parent NPZ face indices exceed the ABI5 uint32 range",
        )
        pinned_parent_faces = np.ascontiguousarray(
            pinned_parent_faces_raw, dtype="<u4")
        _require(pinned_parent_faces.tobytes() == parent_faces.tobytes(),
                 f"stable ID {stable_id} parent face array differs from pinned parent NPZ")
        parent_owners, owner_report = _normalize_local_reconstruction_parent_face_owners(
            owner_report_raw,
            spec.get("parent_face_owner_report_sha256"),
            stable_id=stable_id,
            parent_faces=parent_faces,
            parent_payload_raw=parent_payload_raw,
            operation_ancestry_raw=operation_ancestry_raw,
            authenticated_source_report_raws=spec.get("authenticated_source_report_raws"),
            authenticated_source_report_sha256s=spec.get(
                "authenticated_source_report_sha256s"),
            authenticated_source_parent_npz_raws=spec.get(
                "authenticated_source_parent_npz_raws", ()),
            authenticated_source_parent_npz_sha256s=spec.get(
                "authenticated_source_parent_npz_sha256s", ()),
        )
        face_rib_owners, child_owner_resolution = derive_local_reconstruction_face_rib_owners(
            face_parent_ids, parent_owners, parent_face_count=len(parent_faces),
            return_diagnostics=True,
        )
        required_rib_ids = spec.get("required_rib_ids", ())
        _require(
            isinstance(required_rib_ids, (list, tuple, set))
            and all(type(rib_id) is int and rib_id in LOCAL_RECONSTRUCTION_RIB_IDS
                    for rib_id in required_rib_ids),
            f"stable ID {stable_id} required attachment IDs are not typed existing ribs",
        )
        interface_maps, attachment_diagnostics = derive_local_reconstruction_attachment_maps(
            faces, face_rib_owners, interface_mask,
            vertex_count=len(world_vertices6),
            required_rib_ids=required_rib_ids,
            return_diagnostics=True,
        )
        partial_unused_normal_fallback = None
        if preserve_source_row_local_positions:
            source_local_positions_raw = np.asarray(source_row["vertices6"][:, :3])
            source_local_positions_sha256 = _local_reconstruction_hex_sha(
                spec.get("source_row_local_positions_sha256"),
                f"stable ID {stable_id} source-row local position hash")
            _require(
                source_local_positions_raw.dtype == np.dtype("<f4")
                and source_local_positions_raw.ndim == 2
                and source_local_positions_raw.shape == world_positions.shape
                and np.isfinite(source_local_positions_raw).all()
                and _local_reconstruction_sha256(
                    np.ascontiguousarray(source_local_positions_raw, dtype="<f4").tobytes())
                    == source_local_positions_sha256,
                f"stable ID {stable_id} source-row local positions differ from independent pin or allocation",
            )
            local_positions = np.ascontiguousarray(
                source_local_positions_raw, dtype="<f4")
            inverse_report = {
                "mode": "authenticated_source_row_local_positions_retained",
                "source_position_inverse_performed": False,
                "source_row_payload_sha256": expected_source_payload_sha256,
                "source_row_local_positions_sha256": source_local_positions_sha256,
                "candidate_world_positions_f32_sha256": (
                    _local_reconstruction_sha256(world_positions.tobytes())),
                "source_local_positions_byte_exact": True,
                "native_replay_required_for_emitted_world_geometry": True,
            }
        elif candidate_source_local_mode:
            local_positions, inverse_report = _authenticated_candidate_source_local_positions(
                candidate_npz_raw, world_positions, q_f32=q, t_f32=t, spec=spec)
        elif partial_source_position_spec is not None:
            local_positions, inverse_report, partial_unused_normal_fallback = (
                _authenticated_local_reconstruction_partial_source_positions(
                    world_vertices6[:, :3], source_row, stable_id=stable_id,
                    expected_source_payload_sha256=expected_source_payload_sha256,
                    candidate_npz_sha256=candidate_npz_sha256,
                    q_f32=q, t_f32=t, spec=partial_source_position_spec,
                )
            )
        else:
            local_positions, _replayed_world, inverse_report = _inverse_body20_positions_f32(
                world_vertices6[:, :3], q, t,
            )
        local_normals = _localize_body20_normals_f32(
            world_normals_f32, faces, q,
            unused_normal_fallback_f32=(
                source_row["vertices6"][:, 3:]
                if preserve_source_row_local_positions else partial_unused_normal_fallback
            ),
        )
        vertices6 = np.ascontiguousarray(
            np.concatenate((local_positions, local_normals), axis=1), dtype="<f4")
        _require(np.isfinite(vertices6).all(), f"stable ID {stable_id} emitted non-finite ABI5 vertices")
        row = {
            **source_row,
            "vertices6": vertices6,
            "faces": np.ascontiguousarray(faces, dtype=np.int64),
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
        composed_rows[stable_id] = row
        attachment_maps[str(stable_id)] = interface_maps
        attachment_sidecar_inputs[stable_id] = {
            "face_rib_owners": face_rib_owners,
            "interface_face_mask": raw_interface,
            "owner_report": owner_report,
            "candidate_npz_sha256": candidate_npz_sha256,
            "candidate_faces_sha256": _local_reconstruction_sha256(faces.tobytes()),
            "face_parent_lineage_sha256": _local_reconstruction_sha256(
                json.dumps(face_parent_ids, separators=(",", ":")).encode()),
        }
        row_reports[str(stable_id)] = {
            "stable_id": stable_id,
            "source_parent_row_sha256": _local_reconstruction_sha256(
                _normalized_row_bytes(source_row)),
            "compiled_row_sha256": _local_reconstruction_sha256(
                _normalized_row_bytes(row)),
            "candidate_npz_sha256": candidate_npz_sha256,
            "candidate_world_positions_f32_sha256": _local_reconstruction_sha256(
                world_positions.tobytes()),
            "derived_world_normals_f32_sha256": _local_reconstruction_sha256(
                world_normals_f32.tobytes()),
            "observed_rib_channel_ids": sorted(interface_maps),
            "unobserved_existing_rib_channel_ids": sorted(
                set(LOCAL_RECONSTRUCTION_RIB_IDS) - set(interface_maps)),
            "candidate_faces_sha256": _local_reconstruction_sha256(
                faces.tobytes()),
            "face_parent_lineage_sha256": _local_reconstruction_sha256(
                json.dumps(face_parent_ids, separators=(",", ":")).encode()),
            "face_rib_owner_counts": {
                str(rib_id): face_rib_owners.count(rib_id)
                for rib_id in LOCAL_RECONSTRUCTION_RIB_IDS
                if rib_id in face_rib_owners
            },
            "child_face_rib_owner_resolution": child_owner_resolution,
            "unowned_child_face_count": face_rib_owners.count(None),
            "unowned_interface_face_count": sum(
                bool(is_interface) and rib_id is None
                for is_interface, rib_id in zip(raw_interface, face_rib_owners, strict=True)),
            "interface_map_vertex_counts": {
                str(rib_id): len(vertex_ids) for rib_id, vertex_ids in interface_maps.items()
            },
            "ambiguous_shared_interface_vertex_count": (
                attachment_diagnostics["ambiguous_shared_interface_vertex_count"]),
            "ambiguous_shared_interface_vertices": (
                attachment_diagnostics["ambiguous_shared_interface_vertices"]),
            "parent_owner_report": owner_report,
            "body20_inverse_and_replay": (
                None if preserve_source_row_local_positions else inverse_report),
            "body20_source_position_binding": inverse_report,
            "source_row_local_positions_preserved": preserve_source_row_local_positions,
            "candidate_source_local_positions_preserved": candidate_source_local_mode,
            "source_row_local_positions_partially_preserved": (
                partial_source_position_spec is not None),
            "source_row_local_positions_sha256": (
                inverse_report.get("source_row_local_positions_sha256")
                if preserve_source_row_local_positions or partial_source_position_spec is not None
                or candidate_source_local_mode else None),
            "authenticated_parent_patch_inference_used": (
                owner_report["inherited_parent_face_owner_count"] > 0),
            "actual_source_bone_proximity_crosscheck_required": True,
            "runtime_attachment_owner_rebind_required": True,
            "native_step0_confirmation_required": True,
            "anatomical_admission": False,
            "native_admission": False,
            "mass_or_physical_owner_changed": False,
        }
    output, abi_report = encode_payload_with_row_replacements(raw_payload, composed_rows)
    compiled_payload_sha256 = _local_reconstruction_sha256(output)
    runtime_attachment_map_sidecars: dict[str, dict[str, dict]] = {}
    for stable_id, maps in sorted(
        (int(key), value) for key, value in attachment_maps.items()
    ):
        if not maps:
            runtime_attachment_map_sidecars[str(stable_id)] = {}
            continue
        sidecar_inputs = attachment_sidecar_inputs[stable_id]
        runtime_attachment_map_sidecars[str(stable_id)] = (
            _local_reconstruction_runtime_attachment_map_sidecars(
                nha_stable_id=stable_id,
                interface_maps=maps,
                vertex_count=len(composed_rows[stable_id]["vertices6"]),
                face_rib_owners=sidecar_inputs["face_rib_owners"],
                interface_face_mask=sidecar_inputs["interface_face_mask"],
                source_payload_sha256=expected_source_payload_sha256,
                compiled_payload_sha256=compiled_payload_sha256,
                candidate_npz_sha256=sidecar_inputs["candidate_npz_sha256"],
                candidate_faces_sha256=sidecar_inputs["candidate_faces_sha256"],
                face_parent_lineage_sha256=sidecar_inputs["face_parent_lineage_sha256"],
                accepted_receipt_sha256=accepted_receipt_sha256,
                accepted_capture_pack_sha256=accepted_capture_pack_sha256,
                parent_owner_report=sidecar_inputs["owner_report"],
            )
        )
    report = {
        "schema": "numi.human.intercostal-local-reconstruction-source-compile.v1",
        "status": (
            "PASS_ABI5_SOURCE_EMISSION_DIAGNOSTIC_ONLY"
            if any(row.get("source_emission_diagnostic_only") is True
                   for row in row_reports.values())
            else "PASS_ABI5_LOCAL_RECONSTRUCTION_ROWS_COMPILED"
        ),
        "source_payload_sha256": expected_source_payload_sha256,
        "compiled_payload_sha256": compiled_payload_sha256,
        "accepted_receipt_sha256": pose_sha,
        "accepted_capture_pack_sha256": accepted_capture_pack_sha256,
        "accepted_body20_pose": {
            "quaternion_xyzw_f32_sha256": _local_reconstruction_sha256(q.tobytes()),
            "position_m_f32_sha256": _local_reconstruction_sha256(t.tobytes()),
            "transform_owner_sha256": LOCAL_BODY20_TRANSFORM_SHA256,
        },
        "abi5_owner_report": abi_report,
        "row_reports": row_reports,
        "runtime_attachment_maps": attachment_maps,
        "runtime_attachment_map_sidecars": runtime_attachment_map_sidecars,
        "source_local_positions_preserved_stable_ids": sorted(
            int(stable_id) for stable_id, row in row_reports.items()
            if row.get("source_row_local_positions_preserved") is True),
        "source_local_positions_partially_preserved_stable_ids": sorted(
            int(stable_id) for stable_id, row in row_reports.items()
            if row.get("source_row_local_positions_partially_preserved") is True),
        "candidate_source_local_positions_preserved_stable_ids": sorted(
            int(stable_id) for stable_id, row in row_reports.items()
            if row.get("candidate_source_local_positions_preserved") is True),
        "attachment_ownership_unresolved_stable_ids": sorted(
            int(stable_id) for stable_id, row in row_reports.items()
            if row.get("source_emission_diagnostic_only") is True),
        "source_emission_diagnostic_only": any(
            row.get("source_emission_diagnostic_only") is True
            for row in row_reports.values()),
        "unmodified_rows_byte_exact": abi_report["unmodified_rows_byte_exact"],
        "mass_or_physical_owners_changed": False,
        "native_or_gpu_work_performed": False,
        "native_step0_confirmation_required": True,
        "anatomical_admission": False,
        "native_admission": False,
    }
    return output, report




def bind_local_reconstruction_runtime_attachment_receipt(
    receipt: dict,
    compiled_payload: bytes,
    interface_maps: dict[int, list[int]],
    *,
    nha_stable_id: int,
    bones_payload_sha256: str,
    runtime_attachment_map_sidecars: dict[str, dict] | None = None,
    attachment_source_payload_raw: bytes | None = None,
    expected_attachment_source_payload_sha256: str | None = None,
) -> dict:
    """Rebind and revalidate maps through the retained runtime attachment owner.

    A caller composing regional repairs into a different full payload may
    supply the independently pinned source payload used for these sidecars.
    Only byte-identical typed row geometry may cross that composition boundary;
    other row changes still require their own existing deformation owners.
    The default path requires sidecars already bound to the final payload.
    """
    from .composed_bone_body_frame import (
        bind_intercostal_runtime_attachment_metadata,
        validate_intercostal_runtime_attachment_metadata,
    )
    from .resting_pleura_proxy import _parse_payload

    payload_sha = _local_reconstruction_sha256(compiled_payload)
    _header, _records, rows = _parse_payload(compiled_payload)
    row = rows.get(nha_stable_id)
    _require(isinstance(row, dict), "compiled ABI5 row is missing for attachment rebind")
    composition_proof = None
    if attachment_source_payload_raw is None:
        _require(expected_attachment_source_payload_sha256 is None,
                 "attachment source payload pin supplied without source bytes")
    else:
        expected_source_sha = _local_reconstruction_hex_sha(
            expected_attachment_source_payload_sha256, "attachment source payload pin")
        _require(isinstance(attachment_source_payload_raw, bytes)
                 and _local_reconstruction_sha256(attachment_source_payload_raw)
                     == expected_source_sha,
                 "attachment source payload bytes differ from their independent pin")
        _require(isinstance(runtime_attachment_map_sidecars, dict),
                 "row-preserving attachment composition requires source sidecars")
        source_header, _source_records, source_rows = _parse_payload(
            attachment_source_payload_raw)
        _require(tuple(source_header[:3]) == tuple(_header[:3])
                 and tuple(source_header[5:]) == tuple(_header[5:])
                 and list(source_rows) == list(rows),
                 "attachment composition changed registration or ordered identities")
        source_row = source_rows.get(nha_stable_id)
        _require(isinstance(source_row, dict)
                 and _normalized_row_bytes(source_row) == _normalized_row_bytes(row),
                 "attachment composition changed its typed row geometry")
        composition_proof = {
            "source_payload_sha256": expected_source_sha,
            "final_payload_sha256": payload_sha,
            "stable_id": nha_stable_id,
            "normalized_typed_row_sha256": _local_reconstruction_sha256(
                _normalized_row_bytes(row)),
            "typed_row_geometry_byte_exact": True,
            "physical_state_mass_or_force_added": False,
            "anatomical_admission": False,
            "native_admission": False,
        }
        rebound = deepcopy(runtime_attachment_map_sidecars)
        for sidecar in rebound.values():
            _require(isinstance(sidecar, dict)
                     and sidecar.get("anatomy_payload_sha256") == expected_source_sha
                     and sidecar.get("surface", {}).get("stable_id") == nha_stable_id
                     and isinstance(sidecar.get("source_partition"), dict),
                     "source attachment sidecar differs from its pinned payload or row")
            history = sidecar["source_partition"].setdefault(
                "row_preserving_payload_rebindings", [])
            _require(isinstance(history, list),
                     "attachment payload composition history is malformed")
            history.append(deepcopy(composition_proof))
            sidecar["anatomy_payload_sha256"] = payload_sha
        runtime_attachment_map_sidecars = rebound
    if runtime_attachment_map_sidecars is not None:
        _require(
            isinstance(runtime_attachment_map_sidecars, dict)
            and set(runtime_attachment_map_sidecars) == {str(rib_id) for rib_id in interface_maps},
            "runtime attachment map sidecars differ from exact observed maps",
        )
        bindings = receipt.get("functional_bindings") if isinstance(receipt, dict) else None
        _require(isinstance(bindings, dict),
                 "runtime attachment receipt lacks functional_bindings")
        for rib_id, ids in sorted(interface_maps.items()):
            sidecar = runtime_attachment_map_sidecars[str(rib_id)]
            _require(isinstance(sidecar, dict)
                     and sidecar.get("anatomy_payload_sha256") == payload_sha
                     and sidecar.get("surface", {}).get("stable_id") == nha_stable_id,
                     "runtime map sidecar is not bound to this compiled NHA row")
            key = f"intercostal_{nha_stable_id}_rib{rib_id}_motion_attachment"
            previous = bindings.get(key)
            if isinstance(previous, dict) and isinstance(previous.get("source_partition"), dict):
                sidecar = deepcopy(sidecar)
                sidecar["source_partition"]["retained_prior_source_partition"] = deepcopy(
                    previous["source_partition"])
            bindings[key] = deepcopy(sidecar)
    bind_intercostal_runtime_attachment_metadata(
        receipt,
        row,
        interface_maps,
        nha_stable_id=nha_stable_id,
        nha_payload_sha256=payload_sha,
        bones_payload_sha256=bones_payload_sha256,
    )
    validate_intercostal_runtime_attachment_metadata(
        receipt,
        row,
        interface_maps,
        nha_stable_id=nha_stable_id,
        nha_payload_sha256=payload_sha,
        bones_payload_sha256=bones_payload_sha256,
    )
    receipt["functional_bindings"]["anatomy_payload_sha256"] = payload_sha
    return {
        "nha_stable_id": nha_stable_id,
        "compiled_payload_sha256": payload_sha,
        "bones_payload_sha256": bones_payload_sha256,
        "interface_maps_validated_by_existing_runtime_owner": True,
        **({"row_preserving_payload_rebind": composition_proof}
           if composition_proof is not None else {}),
        "anatomical_admission": False,
        "native_admission": False,
    }


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
