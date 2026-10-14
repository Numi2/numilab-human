"""Source-only preparation for the current-capture brachiocephalic network gate.

This module pins the actual accepted step-0 capture, its invoked NHA receipt,
registered rigid-source baselines, all-target AABB inventory, and frozen
source-collar ancestry. It deliberately runs no geometry predicates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import resting_anatomy_verify as shared_verifier
from . import resting_brachiocephalic_current_capture_gate as capture_gate
from . import resting_brachiocephalic_source007 as source007

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = capture_gate.DATA_ROOT
METADATA_PREFLIGHT = DATA_ROOT / (
    "vein008-forearm474-back158-capture-metadata-preflight-001/"
    "vascular-capture-metadata-preflight.json"
)
METADATA_PREFLIGHT_SHA256 = (
    "cad1587870260c310ed34f8a1b3281bbde804cfb40efe8b9df7a76a35ddfbcbf"
)
ROUTE100_REPORT = DATA_ROOT / "route-candidate-100-network-pair-comparison-001/report.json"
ROUTE100_REPORT_SHA256 = (
    "cb4e79b32598aa48f069408977940e985cae95a9c09c228b486b839f35cb4318"
)
PORT086 = DATA_ROOT / "arterial-venous-route-field-086/proximal-source-port-witnesses-086.json"
PORT086_SHA256 = (
    "e638c6dfeb1030ab6aca113f8d658acf02820ffec883d7e19ebf7f9a8aa7c249"
)
PORT087 = DATA_ROOT / "arterial-venous-route-field-087/proximal-source-port-witnesses-087.json"
PORT087_SHA256 = (
    "5b1ae310b27207560f5bd8d3c70f109cb43b52a3c63be8aad061268fa4dade1d"
)
PORT088 = DATA_ROOT / "arterial-venous-route-field-086/../arterial-venous-route-field-088/joint-port-station-mapping-088.json"
PORT088_SHA256 = (
    "d45f1fef2bcc8949d7dca5316fca30abdb0e12c7e7792bdfa75438293a717f30"
)
PORT089 = DATA_ROOT / "arterial-venous-route-field-089/proximal-source-port-witnesses-089.json"
PORT089_SHA256 = (
    "2593e0510f8ff1c97788b118725cfe0037c78de879fd9700d3a2562077ac6d7a"
)
ROUTE073_REPORT = DATA_ROOT / "arterial-venous-route-field-073/network-port-station-audit-003.json"
ROUTE073_REPORT_SHA256 = (
    "4a2d56f1ffc0f365f3c841a1ebf883197e68ce8ea7696cad37b6dd5b2c027748"
)
BRIDGE_ANCESTRY_REPORT = source007.COLLAR_ANCESTRY
BRIDGE_ANCESTRY_SHA256 = source007.COLLAR_SHA["ancestry"]
BRIDGE_ATTRIBUTION_REPORT = source007.COLLAR_ATTRIBUTION
BRIDGE_ATTRIBUTION_SHA256 = source007.COLLAR_SHA["attribution"]
BRIDGE_REPLAY_REPORT = source007.COLLAR_REPLAY
BRIDGE_REPLAY_SHA256 = source007.COLLAR_SHA["replay"]
BRIDGE_STATIONARITY_REPORT = source007.COLLAR_STATIONARITY
BRIDGE_STATIONARITY_SHA256 = source007.COLLAR_SHA["stationarity"]
TRANSFORM_OWNER = REPO / "scripts/predict_brachiocephalic_captured_svc_poses.py"
TRANSFORM_OWNER_SHA256 = "017a90ed6c9eadbdfda9c4d4b31a4966ca10c8dd1c79577686635a296145f7d2"

# This ordered list is pinned independently in the source-and-capture metadata
# preflight. The two SVC bridge ports are deliberately outside this 13-edge list.
NETWORK_EDGES = (
    ((51011, 7), (51011, 471)),
    ((51011, 471), (51011, 475)),
    ((51011, 475), (51011, 479)),
    ((51011, 479), (51011, 483)),
    ((51011, 479), (51011, 487)),
    ((51011, 525), (51011, 469)),
    ((51011, 525), (51011, 473)),
    ((51011, 473), (51011, 477)),
    ((51011, 477), (51011, 481)),
    ((51011, 481), (51011, 485)),
    ((51011, 481), (51011, 489)),
    ((51011, 526), (51011, 468)),
    ((51011, 526), (51011, 472)),
)
SVC_PORTS = (
    ((51011, 525), (51011, 10)),
    ((51011, 526), (51011, 10)),
)
POTENTIAL_NONNETWORK_OVERLAP = ((51011, 469), (51011, 473))
NETWORK_STABLE_IDS = tuple(sorted({
    key[1] for edge in (*NETWORK_EDGES, *SVC_PORTS) for key in edge
} | {POTENTIAL_NONNETWORK_OVERLAP[0][1], POTENTIAL_NONNETWORK_OVERLAP[1][1]}))
OUTPUT_DIR = DATA_ROOT / "current-capture-network-gate-005-seamowner"
PREFLIGHT_PATH = OUTPUT_DIR / "source0-seamowner-preflight-003.json"
OWNER_COMPOSITION_DIR = DATA_ROOT / "brachiocephalic-seam-owner-composition-001"
OWNER_COMPOSITION_RECEIPT = OWNER_COMPOSITION_DIR / "resting-anatomy-receipt.json"
OWNER_COMPOSITION_REPORT = OWNER_COMPOSITION_DIR / "source-only-composition-report.json"
OWNER_COMPOSITION_PAYLOAD_PATH = OWNER_COMPOSITION_DIR / "resting-thorax.nhanatomy"
OWNER_COMPOSITION_PAYLOAD_SHA256 = "f83a541b794fe9bae699e6784fe091e147d5bad284847773c803c5309027d361"
OWNER_COMPOSITION_CARRIER = (
    OWNER_COMPOSITION_DIR / "assets/brachiocephalic-seam-owner-carriers.nhtissue"
)
OWNER_COMPOSITION_RECEIPT_SHA256 = (
    "5ed44e1cfa912f8bf291347aff19c16cf2c55c4b9a36c3ab04fb6aaf17b70474"
)
OWNER_COMPOSITION_REPORT_SHA256 = (
    "b0469defa346664211d8d9af68c4596f63adc7d9b124c8404bfcf5d844516f65"
)
OWNER_COMPOSITION_PAYLOAD_SHA256 = (
    "f83a541b794fe9bae699e6784fe091e147d5bad284847773c803c5309027d361"
)
OWNER_COMPOSITION_CARRIER_SHA256 = (
    "f4f68f0439d78284d6db578119198ae89250facd88b005d467cda6df5352a687"
)
OWNER_PREPARATION_DIR = DATA_ROOT / "brachiocephalic-exact-coordinate-owner-preparation-002"
OWNER_REPORT_28FC = OWNER_PREPARATION_DIR / "owner-preparation-report-original-28fc-recovered.json"
OWNER_REPORT_C37 = OWNER_PREPARATION_DIR / "owner-preparation-report-current-c37-successor.json"
OWNER_ERRATUM = OWNER_PREPARATION_DIR / "owner-preparation-report-erratum-001.json"
OWNER_REPORT_28FC_SHA256 = (
    "28fc23175356a7a53cc13e4ff79c626fe1c62a244c697c337cc3d082902a86ff"
)
OWNER_REPORT_C37_SHA256 = (
    "c37ee901ebd617943a8e30885710053d6026348559b4c1c565ab9f45ed8e9aa3"
)
OWNER_ERRATUM_SHA256 = (
    "210a0f6b88df17ca163e3b2e30e329d6b9b8ef66e252a1ae0790a809653ad839"
)
OWNER_MODULE = REPO / "src/numilab_human/resting_vessel_owner_blend.py"
OWNER_MODULE_SHA256 = (
    "336f1199d6ea44b1c3ddf2b3559b0b08213a169802aa97b34fbaa8b2aa2248ce"
)
OWNER_COMPOSER = REPO / "scripts/compose_brachiocephalic_owner_field_cap006.py"
OWNER_COMPOSER_SHA256 = (
    "c653f9b3440eec8a0e7271c2ae6733bdfe417160dc079ff1a21759c508ec3e66"
)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError("current-capture network preparation: " + message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resolved_pin(path: Path, expected: str) -> dict:
    resolved = Path(path).resolve(strict=True)
    actual = _sha_file(resolved)
    _require(actual == expected,
             f"frozen input changed at {resolved}: expected {expected}, got {actual}")
    return {"path": str(resolved), "sha256": actual, "size_bytes": resolved.stat().st_size}


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def _git_state() -> dict:
    head = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status_lines": status}


def _load_transform():
    _resolved_pin(TRANSFORM_OWNER, TRANSFORM_OWNER_SHA256)
    spec = importlib.util.spec_from_file_location(
        "_numi_current_capture_registered_rigid_transform", TRANSFORM_OWNER)
    _require(spec is not None and spec.loader is not None,
             "registered rigid transform owner cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transform = getattr(module, "transform_f32", None)
    _require(callable(transform), "existing transform_f32 implementation is missing")
    return transform


def _registered_rigid_source_surfaces(
    context: dict, source_rows: dict | None = None, source_map: dict | None = None
) -> tuple[dict, dict]:
    transform_f32 = _load_transform()
    capture_receipt = context["capture_receipt"]
    accepted_poses = capture_receipt.get("accepted_body_poses")
    _require(isinstance(accepted_poses, list) and accepted_poses,
             "actual capture receipt has no accepted rigid body poses")
    poses = {}
    for pose in accepted_poses:
        body_index = int(pose["body_index"])
        _require(body_index not in poses, f"duplicate accepted body pose {body_index}")
        q = np.ascontiguousarray(pose["quaternion_xyzw"], dtype="<f4")
        t = np.ascontiguousarray(pose["position_m"], dtype="<f4")
        _require(q.shape == (4,) and t.shape == (3,)
                 and np.isfinite(q).all() and np.isfinite(t).all(),
                 f"invalid accepted rigid pose for body {body_index}")
        poses[body_index] = (pose, q, t)

    source_surfaces = {}
    records = {}
    source_rows = context["rows"] if source_rows is None else source_rows
    source_map = context["source_map"] if source_map is None else source_map
    for stable_id in NETWORK_STABLE_IDS:
        row = source_rows.get(stable_id)
        _require(isinstance(row, dict), f"actual ABI5 payload lacks row {stable_id}")
        body_index = int(row["body_index"])
        _require(body_index in poses,
                 f"accepted capture lacks rigid pose for source row {stable_id}/body {body_index}")
        source_map_row = source_map.get(str(stable_id), {})
        _require(int(source_map_row.get("body_index", -1)) == body_index,
                 f"receipt owner differs from decoded ABI5 body for row {stable_id}")
        local_xyz = np.ascontiguousarray(row["vertices6"][:, :3], dtype="<f4")
        faces = np.ascontiguousarray(row["faces"], dtype="<u4")
        pose, q, t = poses[body_index]
        world = np.ascontiguousarray(transform_f32(q, t, local_xyz), dtype="<f4")
        _require(world.shape == local_xyz.shape and np.isfinite(world).all(),
                 f"registered rigid transform output invalid for row {stable_id}")
        key = (51011, stable_id)
        captured = context["surfaces"].get(key)
        _require(captured is not None, f"actual accepted capture lacks typed row {key}")
        capture_faces = np.ascontiguousarray(captured[1], dtype="<u4")
        same_faces = capture_faces.shape == faces.shape and capture_faces.tobytes() == faces.tobytes()
        source_surfaces[key] = (world, faces, None, np.arange(len(world), dtype=np.uint64))
        ident = capture_gate._source_identity(source_map, stable_id)
        records[str(stable_id)] = {
            "typed_identity": list(key),
            "source_member": ident.get("source_member"),
            "source_member_sha256": ident.get("source_member_sha256"),
            "source_label": ident.get("source_owner_metadata", {}).get("label"),
            "source_concept_id": ident.get("source_owner_metadata", {}).get("concept_id"),
            "receipt_body_index": body_index,
            "receipt_layer": int(row["layer"]),
            "recorded_source_compartment_id": ident.get("vascular_compartment_id"),
            "body_pose_source": "actual accepted capture receipt, rigid-only; common cardiac field excluded",
            "body_pose_quaternion_xyzw_f32": [float(value) for value in q],
            "body_pose_position_m_f32": [float(value) for value in t],
            "body_pose_q_sha256": _sha_bytes(q.tobytes()),
            "body_pose_t_sha256": _sha_bytes(t.tobytes()),
            "nha_local_vertices6_xyz_f32_sha256": _sha_bytes(local_xyz.tobytes()),
            "rigid_source_world_xyz_f32_sha256": _sha_bytes(world.tobytes()),
            "nha_faces_u32_sha256": _sha_bytes(faces.tobytes()),
            "captured_faces_u32_sha256": _sha_bytes(capture_faces.tobytes()),
            "capture_face_order_equals_nha": same_faces,
            "source_vertex_count": len(local_xyz),
            "source_face_count": len(faces),
            "source_nha_row_sha256": capture_gate._source_row_record(
                source_rows, stable_id)["vertices6_sha256"],
            "captured_positions_world_f32_sha256": _sha_bytes(
                np.ascontiguousarray(captured[0], dtype="<f4").tobytes()),
            "captured_global_vertex_ids_sha256": _sha_bytes(
                np.ascontiguousarray(captured[3], dtype="<u8").tobytes()),
        }
    return source_surfaces, records


def _source_witness_pairs(path: Path, expected_sha: str, parent: int, child: int) -> dict:
    _resolved_pin(path, expected_sha)
    data = json.loads(path.read_text())
    matches = [row for row in data.get("rows", [])
               if int(row.get("parent_stable_id", -1)) == parent
               and int(row.get("child_stable_id", -1)) == child]
    _require(len(matches) == 1, f"{path.name} lacks unique source edge {parent}->{child}")
    row = matches[0]
    witnesses = row.get("witnesses")
    _require(isinstance(witnesses, list)
             and len(witnesses) == int(row.get("exact_intersection_pair_count", -1)),
             f"{path.name} source witness rows/count differ for {parent}->{child}")
    pairs = np.asarray([
        (int(item["parent_face_local"]), int(item["child_face_local"]))
        for item in witnesses
    ], dtype="<i8")
    return {
        "path": str(path.resolve()),
        "sha256": expected_sha,
        "source_nha_sha256": data.get("source_nha_sha256"),
        "parent_stable_id": parent,
        "child_stable_id": child,
        "pair_count": len(pairs),
        "pairs_sha256": _sha_bytes(pairs.tobytes()),
        "pairs": pairs,
        "evidence_status": data.get("status"),
        "evidence_class": "frozen source-coordinate exact pair ancestry; historical, not current-capture admission",
    }


def _route100_witness_pairs(parent: int, child: int) -> dict:
    _resolved_pin(ROUTE100_REPORT, ROUTE100_REPORT_SHA256)
    data = json.loads(ROUTE100_REPORT.read_text())
    poses = [pose for pose in data.get("poses", []) if int(pose.get("step", -1)) == 0]
    _require(len(poses) == 1, "route100 has no unique step-0 comparison row")
    key = f"{parent}->{child}"
    raw = poses[0].get("network_pairs", {}).get(key)
    _require(isinstance(raw, dict), f"route100 lacks source face-pair ancestry seed {key}")
    pairs = np.asarray(raw.get("candidate_face_pairs", []), dtype="<i8")
    _require(pairs.ndim == 2 and pairs.shape[1] == 2 and len(pairs) == int(raw.get("candidate_count", -1)),
             f"route100 face pairs/count differ for {key}")
    return {
        "path": str(ROUTE100_REPORT.resolve()),
        "sha256": ROUTE100_REPORT_SHA256,
        "parent_stable_id": parent,
        "child_stable_id": child,
        "pair_count": len(pairs),
        "pairs_sha256": _sha_bytes(pairs.tobytes()),
        "pairs": pairs,
        "evidence_status": data.get("status"),
        "evidence_class": "retained step-0 exact face-pair list; source/capture ancestry seed, not a current geometry predicate",
    }


def _bridge_witness_pairs(context: dict, bridge_id: int, target_id: int) -> dict:
    key = ((51011, bridge_id), (51011, target_id))
    pairs = np.ascontiguousarray(context["port_pairs"][key], dtype="<i8")
    return {
        "path": str(BRIDGE_ANCESTRY_REPORT.resolve()),
        "sha256": BRIDGE_ANCESTRY_SHA256,
        "bridge_stable_id": bridge_id,
        "target_stable_id": target_id,
        "pair_count": len(pairs),
        "pairs_sha256": _sha_bytes(pairs.tobytes()),
        "pairs": pairs,
        "evidence_status": "FROZEN_SOURCE_ANCESTRY_ONLY_NOT_ADMISSION",
        "evidence_class": "frozen exact bridge face-patch ancestry; historical source, not current-capture admission",
    }


def _apply_receipt_face_retirements(item: dict, parent: int, child: int,
                                    context: dict) -> dict:
    pairs = item.get("pairs")
    if pairs is None:
        return item
    pairs = np.ascontiguousarray(pairs, dtype="<i8").copy()
    remaps = []
    identities = []
    for stable_id, column, side in ((parent, 0, "parent"), (child, 1, "child")):
        repair = context["source_map"][str(stable_id)].get("repair") or {}
        if not isinstance(repair, dict):
            continue
        reinstated = repair.get("local_source_face_reinstatement")
        if isinstance(reinstated, dict):
            row = context["rows"][stable_id]
            faces = np.ascontiguousarray(row["faces"], dtype="<u4")
            vertices6 = np.ascontiguousarray(row["vertices6"], dtype="<f4")
            source_ids = (930, 1066, 1063)
            aliases = (1068, 1065, 926)
            face_id = 1858
            _require(
                stable_id == 477
                and reinstated.get("schema")
                == "numi.human.local-source-face-exact-alias-reinstatement.v1"
                and reinstated.get("stable_id") == 477
                and reinstated.get("source_member") == "FJ2217"
                and reinstated.get("source_face_id_zero_based") == face_id
                and tuple(reinstated.get("source_vertex_ids_zero_based", ())) == source_ids
                and tuple(reinstated.get("alias_vertex_ids_zero_based", ())) == aliases
                and reinstated.get("coordinate_aliases_float32_bit_exact") is True
                and reinstated.get("source_face_ancestry_exact") is True
                and reinstated.get("all_other_faces_byte_exact_and_ordered") is True
                and reinstated.get("all_vertex_and_normal_bytes_unchanged") is True
                and reinstated.get("physical_mass_or_blood_state_added") is False
                and reinstated.get("output_face_count") == len(faces)
                and len(faces) == 2110
                and faces[face_id].tolist() == list(aliases),
                "FJ2217 face identity reinstatement is not the pinned ordered cap",
            )
            for source_vertex, alias_vertex in zip(source_ids, aliases):
                _require(
                    vertices6[source_vertex, :3].astype("<f4").tobytes()
                    == vertices6[alias_vertex, :3].astype("<f4").tobytes(),
                    f"FJ2217 source/alias coordinate bits differ at {source_vertex}/{alias_vertex}",
                )
            values = pairs[:, column]
            _require(
                (values >= 0).all() and (values < len(faces)).all(),
                f"current {side} face ancestry is outside reinstated row477",
            )
            identities.append({
                "typed_identity": [51011, stable_id],
                "side": side,
                "source_input_face_count": reinstated.get("input_face_count"),
                "current_output_face_count": len(faces),
                "reinstated_source_face_id_zero_based": face_id,
                "mapping": "identity; exact source face order restored at its original index",
                "matched_alias_vertex_ids_zero_based": list(aliases),
                "pair_face_1858_is_valid": True,
                "historical_retirement_is_not_active": (
                    "local_source_face_retirement" not in repair
                    and isinstance(repair.get("historical_local_source_face_retirement"), dict)
                ),
            })
            continue
        retired = repair.get("local_source_face_retirement")
        if not isinstance(retired, dict):
            continue
        lineage = retired.get("lineage")
        _require(isinstance(lineage, dict),
                 f"row {stable_id} retirement lacks its source-face lineage record")
        retired_face = int(lineage["nha_local_face_id_zero_based"])
        source_count = int(retired["input_face_count"])
        output_count = int(retired["output_face_count"])
        row_count = len(context["rows"][stable_id]["faces"])
        _require(
            retired.get("all_other_source_faces_retained_in_original_order") is True
            and retired.get("source_face_and_nha_ancestry_exact") is True
            and lineage.get("retirement_operation_changes_no_mesh_coordinates") is True
            and output_count == row_count == source_count - 1
            and 0 <= retired_face < source_count,
            f"row {stable_id} retirement provenance does not describe one order-preserving face deletion",
        )
        values = pairs[:, column]
        _require((values >= 0).all() and (values < source_count).all(),
                 f"historical {side} face ancestry is outside the retired source namespace")
        retired_references = int(np.count_nonzero(values == retired_face))
        _require(retired_references == 0,
                 f"historical {side} collar uses explicitly retired source face {retired_face}")
        pairs[:, column] = np.where(values > retired_face, values - 1, values)
        _require((pairs[:, column] < output_count).all(),
                 f"mapped {side} source face ancestry is outside the current NHA row")
        remaps.append({
            "typed_identity": [51011, stable_id],
            "side": side,
            "source_input_face_count": source_count,
            "current_output_face_count": output_count,
            "retired_source_face_id_zero_based": retired_face,
            "retired_face_pair_references": retired_references,
            "mapping": "old face IDs below retirement unchanged; IDs above retirement decrement by one",
            "receipt_bound_all_other_faces_original_order": True,
        })
    if remaps:
        item["historical_pair_ids_sha256"] = item["pairs_sha256"]
        item["pairs"] = pairs
        item["pairs_sha256"] = _sha_bytes(pairs.tobytes())
        item["source_face_index_remap"] = remaps
        item["source_face_index_remap_provenance"] = (
            "historical A779 receipt source_id_map repair.local_source_face_retirement"
        )
    if identities:
        item["source_face_index_identity"] = identities
        item["source_face_index_identity_provenance"] = (
            "successor receipt repair.local_source_face_reinstatement; no face-ID decrement"
        )
    return item


def _source_edge_occurrences(data: dict, parent: int, child: int) -> int:
    count = 0
    for row in data.get("rows", []):
        if (int(row.get("parent_stable_id", -1)) == parent
                and int(row.get("child_stable_id", -1)) == child):
            count += 1
    for pose in data.get("poses", []):
        if int(pose.get("step", -1)) != 0:
            continue
        if f"{parent}->{child}" in pose.get("network_pairs", {}):
            count += 1
    return count


def _missing_source_collar_search(pending_edges: list[str]) -> dict:
    inputs = (
        (PORT086, PORT086_SHA256, "exact_source_port_rows"),
        (PORT087, PORT087_SHA256, "exact_source_port_rows"),
        (PORT089, PORT089_SHA256, "exact_source_port_rows"),
        (ROUTE100_REPORT, ROUTE100_REPORT_SHA256, "step_zero_candidate_face_pairs"),
    )
    searched = []
    documents = {}
    for path, expected_sha, evidence_kind in inputs:
        _resolved_pin(path, expected_sha)
        data = json.loads(path.read_text())
        searched.append({
            "path": str(path.resolve()),
            "sha256": expected_sha,
            "evidence_kind": evidence_kind,
        })
        documents[path] = data
    results = {}
    pending_pairs = {"479->487": (479, 487), "481->489": (481, 489)}
    _require(set(pending_edges) == set(pending_pairs),
             "pending source collar edge set differs from the two known missing branches")
    for edge in pending_edges:
        parent, child = pending_pairs[edge]
        occurrences = {
            str(path.resolve()): _source_edge_occurrences(data, parent, child)
            for path, data in documents.items()
        }
        _require(not any(occurrences.values()),
                 f"new frozen source ancestry now exists for {edge}; update the explicit edge owner")
        results[edge] = {
            "source_edge_witness_occurrences": occurrences,
            "searched_frozen_inputs": searched,
            "status": "NO_FROZEN_EXACT_SOURCE_FACE_ANCESTRY_IN_PINNED_INPUTS",
            "geometry_predicates_run": False,
        }
    return results


def _edge_ancestry(context: dict) -> tuple[dict, dict]:
    evidence = {}
    source_report_specs = {
        (7, 471): (PORT087, PORT087_SHA256),
        (471, 475): (PORT086, PORT086_SHA256),
        (475, 479): (PORT086, PORT086_SHA256),
        (473, 477): (PORT086, PORT086_SHA256),
        (477, 481): (PORT086, PORT086_SHA256),
    }
    route100_specs = {(479, 483), (481, 485)}
    bridge_specs = {(525, 469), (525, 473), (526, 468), (526, 472)}
    pending = []
    for first, second in NETWORK_EDGES:
        parent, child = first[1], second[1]
        pair = (first, second)
        if pair in context["port_pairs"]:
            item = _bridge_witness_pairs(context, parent, child)
        elif (parent, child) in source_report_specs:
            path, sha = source_report_specs[(parent, child)]
            item = _source_witness_pairs(path, sha, parent, child)
        elif (parent, child) in route100_specs:
            item = _route100_witness_pairs(parent, child)
        elif (parent, child) in bridge_specs:
            raise ValueError(f"bridge source ancestry API lacks expected pair {parent}->{child}")
        else:
            item = {
                "parent_stable_id": parent,
                "child_stable_id": child,
                "pair_count": None,
                "pairs_sha256": None,
                "evidence_status": "SOURCE_BOUND_COLLAR_DERIVATION_REQUIRED",
                "evidence_class": "not inferred from graph adjacency; derive exact registered source collar before capture predicates",
            }
            pending.append(f"{parent}->{child}")
        if item.get("pairs") is not None:
            item = _apply_receipt_face_retirements(item, parent, child, context)
            pairs = np.asarray(item["pairs"], dtype="<i8")
            source_parent = context["rows"][parent]
            source_child = context["rows"][child]
            _require(pairs.ndim == 2 and pairs.shape[1] == 2 and len(pairs) > 0,
                     f"empty or malformed source ancestry for {parent}->{child}")
            _require((pairs[:, 0] >= 0).all() and (pairs[:, 0] < len(source_parent["faces"])).all()
                     and (pairs[:, 1] >= 0).all() and (pairs[:, 1] < len(source_child["faces"])).all(),
                     f"source face-pair ancestry is out of bounds for {parent}->{child}")
            item["source_parent_faces_sha256"] = _sha_bytes(
                np.ascontiguousarray(source_parent["faces"], dtype="<u4").tobytes())
            item["source_child_faces_sha256"] = _sha_bytes(
                np.ascontiguousarray(source_child["faces"], dtype="<u4").tobytes())
        evidence[f"{parent}->{child}"] = item

    svc = {
        f"{bridge_id}->10": _bridge_witness_pairs(context, bridge_id, 10)
        for bridge_id in (525, 526)
    }
    # This known overlap is deliberately kept outside the route graph.
    potential = _source_witness_pairs(PORT086, PORT086_SHA256, 469, 473)
    potential["classification"] = "separate_potential_overlap_not_a_declared_circulation_edge"
    return evidence, {
        "svc_bridge_ports": svc,
        "potential_469_473_source_overlap": potential,
        "pending_source_collar_edges": pending,
    }


def _all_nearby_aabb(surfaces: dict) -> dict:
    result = {}
    keys = sorted(surfaces)
    for stable_id in NETWORK_STABLE_IDS:
        key = (51011, stable_id)
        positions = np.ascontiguousarray(surfaces[key][0], dtype="<f4")
        lo, hi = positions.min(axis=0), positions.max(axis=0)
        nearby = []
        for other in keys:
            if other == key:
                continue
            candidate = np.ascontiguousarray(surfaces[other][0], dtype="<f4")
            other_lo, other_hi = candidate.min(axis=0), candidate.max(axis=0)
            if capture_gate._aabb_overlaps(lo, hi, other_lo, other_hi):
                nearby.append(list(other))
        result[f"{key[0]}:{key[1]}"] = nearby
    return result


def _stable_owner_rows(bindings: dict) -> dict[int, dict]:
    rows = bindings.get("anatomy_surfaces")
    _require(isinstance(rows, list), "owner binding anatomy_surfaces is not a list")
    result = {}
    for row in rows:
        _require(isinstance(row, dict) and type(row.get("stable_id")) is int,
                 "owner surface stable IDs must be non-boolean JSON integers")
        stable_id = row["stable_id"]
        _require(stable_id not in result and stable_id >= 0,
                 "owner surface stable IDs are duplicated or negative")
        result[stable_id] = row
    return result


def _validate_owner_report_erratum() -> dict:
    erratum = json.loads(OWNER_ERRATUM.read_text())
    old = erratum.get("recovered_prior_report", {})
    current = erratum.get("canonical_current_report", {})
    diffs = erratum.get("field_differences", [])
    _require(old.get("sha256") == OWNER_REPORT_28FC_SHA256
             and current.get("sha256") == OWNER_REPORT_C37_SHA256
             and erratum.get("versioned_current_copy", {}).get("sha256")
             == OWNER_REPORT_C37_SHA256,
             "owner report erratum does not bind both exact frozen report versions")
    expected_diffs = [
        {"kind": "added", "new": 5,
         "path": "owner_field.carrier_source_index_standalone"},
        {"kind": "added", "new": "assets/route110-dual-vessel-reference-carriers.nhtissue",
         "path": "owner_field.carrier_source_path_standalone"},
    ]
    _require(diffs == expected_diffs
             and "without guessing" in erratum.get("why_fields_added", ""),
             "owner report erratum field list or reason changed")
    return erratum


def _validate_reinstated_cap_face_transition(
    actual_faces: np.ndarray,
    current_faces: np.ndarray,
    actual_vertices6: np.ndarray,
    current_vertices6: np.ndarray,
) -> dict:
    old_faces = np.ascontiguousarray(actual_faces, dtype="<u4")
    new_faces = np.ascontiguousarray(current_faces, dtype="<u4")
    old_vertices = np.ascontiguousarray(actual_vertices6, dtype="<f4")
    new_vertices = np.ascontiguousarray(current_vertices6, dtype="<f4")
    _require(old_faces.shape == (2109, 3) and new_faces.shape == (2110, 3),
             "FJ2217 face transition is not the one-face 2109-to-2110 reinstatement")
    _require(old_vertices.shape == new_vertices.shape == (1089, 6)
             and old_vertices.tobytes() == new_vertices.tobytes(),
             "FJ2217 cap transition changed vertex position or normal bytes")
    face_id = 1858
    aliases = np.asarray([1068, 1065, 926], dtype="<u4")
    source_vertices = (930, 1066, 1063)
    alias_vertices = (1068, 1065, 926)
    _require(np.array_equal(old_faces[:face_id], new_faces[:face_id])
             and np.array_equal(old_faces[face_id:], new_faces[face_id + 1:])
             and np.array_equal(new_faces[face_id], aliases),
             "FJ2217 reinstatement does not restore only original ordered face 1858")
    for source_vertex, alias_vertex in zip(source_vertices, alias_vertices):
        _require(old_vertices[source_vertex, :3].tobytes()
                 == old_vertices[alias_vertex, :3].tobytes(),
                 f"FJ2217 source/alias Float32 coordinates differ at "
                 f"{source_vertex}/{alias_vertex}")
    return {
        "typed_identity": [51011, 477],
        "operation": "reinsert_retired_source_face_at_original_order_position",
        "source_member": "FJ2217",
        "source_face_id_zero_based": face_id,
        "source_vertex_ids_zero_based": list(source_vertices),
        "exact_coordinate_alias_vertex_ids_zero_based": list(alias_vertices),
        "source_face_ordered_vertices_after_alias_resolution": [930, 1066, 1063],
        "input_face_count": len(old_faces),
        "output_face_count": len(new_faces),
        "all_other_face_bytes_and_order_exact": True,
        "all_vertex_position_and_normal_bytes_unchanged": True,
        "coordinate_aliases_exact_xyz_only": True,
        "physical_mass_or_blood_state_added": False,
    }


def _compare_payload_rows(
    actual_rows: dict,
    current_rows: dict,
    actual_records: list,
    current_records: list,
) -> dict:
    _require(set(actual_rows) == set(current_rows),
             "current composition changed the ABI5 stable-ID row inventory")
    actual_ids = [int(tuple(record)[5]) for record in actual_records]
    current_ids = [int(tuple(record)[5]) for record in current_records]
    _require(actual_ids == current_ids and actual_ids.count(477) == 1,
             "current composition changed ABI5 row order or lacks the unique cap row")
    cap_position = actual_ids.index(477)
    cap_delta = None
    for position, stable_id in enumerate(actual_ids):
        old_row, new_row = actual_rows[stable_id], current_rows[stable_id]
        old_vertices = np.ascontiguousarray(old_row["vertices6"], dtype="<f4")
        new_vertices = np.ascontiguousarray(new_row["vertices6"], dtype="<f4")
        _require(old_vertices.shape == new_vertices.shape
                 and old_vertices.tobytes() == new_vertices.tobytes(),
                 f"ABI5 row {stable_id} position/normal bytes changed")
        old_faces = np.ascontiguousarray(old_row["faces"], dtype="<u4")
        new_faces = np.ascontiguousarray(new_row["faces"], dtype="<u4")
        if stable_id == 477:
            cap_delta = _validate_reinstated_cap_face_transition(
                old_faces, new_faces, old_vertices, new_vertices)
        else:
            _require(old_faces.shape == new_faces.shape
                     and old_faces.tobytes() == new_faces.tobytes(),
                     f"ABI5 row {stable_id} face bytes/order changed outside the declared cap")
        old_record = tuple(map(int, old_row["record"]))
        new_record = tuple(map(int, new_row["record"]))
        expected_record = list(old_record)
        if position == cap_position:
            expected_record[4] += 3
        elif position > cap_position:
            expected_record[3] += 3
        _require(new_record == tuple(expected_record),
                 f"ABI5 row {stable_id} record differs from the exact cap index-count/offset shift")
    _require(cap_delta is not None, "ABI5 source comparison omitted the FJ2217 cap row")
    return {
        "status": "ALL_ABI5_ROWS_BYTE_EXACT_EXCEPT_DECLARED_FJ2217_CAP_FACE",
        "row_count": len(actual_rows),
        "position_and_normal_rows_exact": len(actual_rows),
        "face_rows_exact_outside_cap": len(actual_rows) - 1,
        "record_order_preserved": True,
        "cap_face": cap_delta,
        "record_offset_effect": {
            "row_477_index_count_increment": 3,
            "rows_after_477_index_start_increment": 3,
            "other_record_fields_unchanged": True,
        },
    }


def _validate_source_map_cap_reinstatement(
    actual_source_map: dict, current_source_map: dict
) -> dict:
    _require(set(actual_source_map) == set(current_source_map),
             "current composition changed source_id_map stable-ID inventory")
    for key, value in actual_source_map.items():
        if key == "477":
            continue
        _require(value == current_source_map[key],
                 f"source_id_map entry {key} changed outside the declared FJ2217 cap")
    old = actual_source_map.get("477")
    new = current_source_map.get("477")
    _require(isinstance(old, dict) and isinstance(new, dict),
             "source_id_map lacks the FJ2217 cap row")
    old_identity = {key: value for key, value in old.items() if key != "repair"}
    new_identity = {key: value for key, value in new.items() if key != "repair"}
    _require(old_identity == new_identity,
             "FJ2217 source identity changed while reinstating its source face")
    old_repair = old.get("repair")
    new_repair = new.get("repair")
    _require(isinstance(old_repair, dict) and isinstance(new_repair, dict),
             "FJ2217 cap repair provenance is missing")
    old_retirement = old_repair.get("local_source_face_retirement")
    _require(isinstance(old_retirement, dict)
             and "local_source_face_retirement" not in new_repair
             and new_repair.get("historical_local_source_face_retirement") == old_retirement,
             "historical FJ2217 retirement is not retained separately from active state")
    old_other = {key: value for key, value in old_repair.items()
                 if key != "local_source_face_retirement"}
    new_other = {key: value for key, value in new_repair.items()
                 if key not in {
                     "historical_local_source_face_retirement",
                     "local_source_face_reinstatement",
                 }}
    _require(old_other == new_other,
             "FJ2217 repair metadata changed beyond preserving history and reinstating the cap")
    reinstatement = new_repair.get("local_source_face_reinstatement")
    _require(isinstance(reinstatement, dict)
             and reinstatement.get("schema")
             == "numi.human.local-source-face-exact-alias-reinstatement.v1"
             and reinstatement.get("stable_id") == 477
             and reinstatement.get("source_member") == "FJ2217"
             and reinstatement.get("source_face_id_zero_based") == 1858
             and reinstatement.get("source_vertex_ids_zero_based") == [930, 1066, 1063]
             and reinstatement.get("alias_vertex_ids_zero_based") == [1068, 1065, 926]
             and reinstatement.get("input_face_count") == 2109
             and reinstatement.get("output_face_count") == 2110
             and reinstatement.get("coordinate_aliases_float32_bit_exact") is True
             and reinstatement.get("source_face_ancestry_exact") is True
             and reinstatement.get("all_other_faces_byte_exact_and_ordered") is True
             and reinstatement.get("all_vertex_and_normal_bytes_unchanged") is True
             and reinstatement.get("physical_mass_or_blood_state_added") is False,
             "FJ2217 source-map reinstatement does not bind the exact cap operation")
    return {
        "status": "SOURCE_IDENTITIES_EQUAL_EXCEPT_DECLARED_FJ2217_REPAIR_SUCCESSOR",
        "all_other_source_map_entries_exact": True,
        "historical_retirement_exactly_preserved": True,
        "active_retirement_absent": True,
        "source_face_reinstatement_exact": True,
    }


def _current_payload_row_transition(context: dict, current_receipt: dict) -> tuple[dict, dict]:
    payload = current_receipt.get("payload", {})
    path = Path(payload.get("path", "")).resolve(strict=True)
    _require(payload.get("sha256") == OWNER_COMPOSITION_PAYLOAD_SHA256
             and _sha_file(path) == OWNER_COMPOSITION_PAYLOAD_SHA256,
             "current source-only NHA payload does not match its frozen composition pin")
    actual_payload = context["anatomy_receipt"].get("payload", {})
    _require(actual_payload.get("sha256") == capture_gate.ANATOMY_PAYLOAD_SHA256,
             "actual accepted capture receipt no longer binds the pinned A779 payload")
    header, records, rows = capture_gate.nhanat_codec._parse_payload(path.read_bytes())
    _require(int(header[1]) == 5 and len(records) == int(payload.get("surface_count", -1))
             and set(rows) == set(context["rows"]),
             "source-only composition is not the exact ABI5 row inventory")
    row_delta = _compare_payload_rows(
        context["rows"], rows, context["records"], records)
    source_map_delta = _validate_source_map_cap_reinstatement(
        context["anatomy_receipt"]["provenance"]["source_id_map"],
        current_receipt["provenance"]["source_id_map"])
    return ({
        "actual_capture_payload_sha256": actual_payload["sha256"],
        "current_source_only_payload_path": str(path),
        "current_source_only_payload_sha256": payload["sha256"],
        "full_payload_hashes_differ_as_expected": (
            actual_payload["sha256"] != payload["sha256"]),
        "rowwise_geometry_and_abi_delta": row_delta,
        "source_id_map_delta": source_map_delta,
        "scope": (
            "Exact all-row NHA comparison: every vertices6 position/normal and every face "
            "is byte-identical except FJ2217 source face 1858 is restored at its original "
            "order position; following index offsets advance by three."
        ),
        "not_native_emitted_or_geometry_admitted": True,
    }, rows)


def _validate_common_field_descriptor_transition(actual: dict, current: dict) -> dict:
    _require(actual.get("anatomy_payload_sha256")
             == capture_gate.ANATOMY_PAYLOAD_SHA256
             and current.get("anatomy_payload_sha256")
             == OWNER_COMPOSITION_PAYLOAD_SHA256,
             "common-field descriptor does not rebind only the expected NHA payload hashes")
    old_other = {key: value for key, value in actual.items()
                 if key != "anatomy_payload_sha256"}
    new_other = {key: value for key, value in current.items()
                 if key != "anatomy_payload_sha256"}
    _require(old_other == new_other,
             "cap composition changed common map, polynomials, domains, volumes, or range contract")
    return {
        "all_common_field_semantics_and_assets_exact": True,
        "only_rebound_field": "anatomy_payload_sha256",
        "actual_capture_payload_sha256": actual["anatomy_payload_sha256"],
        "current_source_only_payload_sha256": current["anatomy_payload_sha256"],
    }


def _owner_composition_delta(actual_receipt: dict, current_receipt: dict,
                             payload_transition: dict | None = None) -> dict:
    actual_payload = actual_receipt.get("payload", {})
    current_payload = current_receipt.get("payload", {})
    _require(actual_payload.get("sha256") == capture_gate.ANATOMY_PAYLOAD_SHA256
             and current_payload.get("sha256") == OWNER_COMPOSITION_PAYLOAD_SHA256,
             "actual capture or source-only successor payload hash changed")
    if payload_transition is not None:
        _require(payload_transition.get("full_payload_hashes_differ_as_expected") is True
                 and payload_transition.get("rowwise_geometry_and_abi_delta", {}).get("status")
                 == "ALL_ABI5_ROWS_BYTE_EXACT_EXCEPT_DECLARED_FJ2217_CAP_FACE"
                 and payload_transition.get("source_id_map_delta", {}).get("status")
                 == "SOURCE_IDENTITIES_EQUAL_EXCEPT_DECLARED_FJ2217_REPAIR_SUCCESSOR",
                 "rowwise NHA source transition did not pass its exact cap-only comparison")
    actual_prov = actual_receipt.get("provenance", {})
    current_prov = current_receipt.get("provenance", {})
    actual_source_map = actual_prov.get("source_id_map", {})
    current_source_map = current_prov.get("source_id_map", {})
    compared_ids = sorted(set(NETWORK_STABLE_IDS) | {313, 315, 317})
    source_identity_equal = {}
    for stable_id in compared_ids:
        key = str(stable_id)
        actual_entry = actual_source_map.get(key)
        current_entry = current_source_map.get(key)
        if stable_id == 477 and payload_transition is not None:
            actual_entry = {name: value for name, value in actual_entry.items()
                            if name != "repair"}
            current_entry = {name: value for name, value in current_entry.items()
                             if name != "repair"}
        source_identity_equal[key] = actual_entry == current_entry
    _require(all(source_identity_equal.values()),
             "source-only owner composition changed a network/intercostal source identity")

    actual_bindings = actual_receipt.get("passive_tissue_parent_bindings", {})
    current_bindings = current_receipt.get("passive_tissue_parent_bindings", {})
    actual_sources = actual_bindings.get("source_payloads")
    current_sources = current_bindings.get("source_payloads")
    _require(isinstance(actual_sources, list) and len(actual_sources) == 7
             and isinstance(current_sources, list) and len(current_sources) == 8
             and current_sources[:7] == actual_sources
             and current_sources[7].get("sha256") == OWNER_COMPOSITION_CARRIER_SHA256
             and Path(current_sources[7].get("path", "")).resolve()
             == OWNER_COMPOSITION_CARRIER.resolve()
             and _sha_file(OWNER_COMPOSITION_CARRIER)
             == OWNER_COMPOSITION_CARRIER_SHA256,
             "composition does not retain the exact seven-source prefix and append pinned source 7")
    actual_maps = _stable_owner_rows(actual_bindings)
    current_maps = _stable_owner_rows(current_bindings)
    _require(set(actual_maps) == set(current_maps),
             "owner composition changed the anatomy-surface ID inventory")
    changed_ids = sorted(stable_id for stable_id in actual_maps
                         if actual_maps[stable_id] != current_maps[stable_id])
    _require(changed_ids == [475, 477],
             "owner composition changed owner rows beyond the declared 475/477 update")
    changed_rows = {}
    for stable_id in changed_ids:
        old_row, new_row = actual_maps[stable_id], current_maps[stable_id]
        old_meta = {key: value for key, value in old_row.items() if key != "vertices"}
        new_meta = {key: value for key, value in new_row.items() if key != "vertices"}
        _require(old_meta == new_meta,
                 f"row {stable_id} owner metadata changed outside its vertex terms")
        old_terms, new_terms = old_row.get("vertices"), new_row.get("vertices")
        _require(isinstance(old_terms, list) and isinstance(new_terms, list)
                 and len(old_terms) == len(new_terms),
                 f"row {stable_id} owner term array shape changed")
        changed_vertices = [index for index, (before, after)
                            in enumerate(zip(old_terms, new_terms)) if before != after]
        changed_rows[str(stable_id)] = {
            "vertex_count": len(old_terms),
            "changed_vertex_count": len(changed_vertices),
            "changed_vertex_ids_sha256": _sha_bytes(
                np.asarray(changed_vertices, dtype="<u8").tobytes()),
            "old_terms_sha256": _sha_bytes(json.dumps(
                old_terms, sort_keys=True, separators=(",", ":")).encode()),
            "current_terms_sha256": _sha_bytes(json.dumps(
                new_terms, sort_keys=True, separators=(",", ":")).encode()),
        }

    old_common = actual_prov.get("cardiac_geometry_binding", {}).get("common_field")
    new_common = current_prov.get("cardiac_geometry_binding", {}).get("common_field")
    common_delta = _validate_common_field_descriptor_transition(old_common, new_common)
    composition = current_prov.get("passive_brachiocephalic_exact_coordinate_owner_composition", {})
    _require(composition.get("final_source_payload_count") == 8
             and composition.get("existing_source_payload_prefix_semantics_preserved") is True
             and composition.get("additional_physical_mass_kg") == 0
             and composition.get("additional_force_or_state") is False
             and composition.get("anatomical_admission") is False,
             "composition does not preserve the 8-source/state/mass contract")
    return {
        "schema": "numi.human.brachiocephalic.current-seamowner-composition-delta.v1",
        "status": "SOURCE_ONLY_OWNER_DELTA_PINNED_NOT_NATIVE_EMITTED",
        "actual_capture_receipt_sha256": capture_gate.ANATOMY_RECEIPT_SHA256,
        "current_owner_composition_receipt_sha256": OWNER_COMPOSITION_RECEIPT_SHA256,
        "actual_capture_payload_sha256": actual_payload["sha256"],
        "nha_payload_sha256": current_payload["sha256"],
        "payload_row_transition": payload_transition,
        "network_and_intercostal_source_identity_equal": source_identity_equal,
        "actual_parent_payload_count": len(actual_sources),
        "current_parent_payload_count": len(current_sources),
        "prefix_source_payloads_exactly_preserved": True,
        "appended_source_index": 7,
        "appended_source_sha256": current_sources[7]["sha256"],
        "owner_rows_changed_only": changed_ids,
        "changed_owner_rows": changed_rows,
        "common_field_descriptor_transition": common_delta,
        "existing_common_field_semantics_unchanged": True,
        "additional_physical_mass_kg": 0,
        "additional_force_or_state": False,
        "current_composition_is_final_integrated_receipt": False,
        "actual_capture_emitted_using_current_owner_receipt": False,
        "current_capture_candidate_ready": False,
        "reason_capture_not_current": (
            "The available accepted step-0 pack is bound to the seven-payload A779 "
            "receipt. This source-only successor appends payload 7, updates the 475/477 "
            "owner terms, rebinds the common-field descriptor to the new NHA hash, and "
            "restores only FJ2217 source face 1858. Although the all-row source comparison "
            "is exact under that declared cap operation, no native step-0 pack has emitted "
            "this receipt; regenerate the capture and target inventory before predicates."
        ),
    }


def _source_identity_summary(source_map: dict, stable_id: int) -> dict:
    row = source_map.get(str(stable_id), {})
    owner = row.get("source_owner_metadata", {})
    return {
        "typed_identity": [51011, stable_id],
        "name": row.get("name"),
        "concept_id": owner.get("concept_id"),
        "source_member": row.get("source_member"),
        "body_index": row.get("body_index"),
        "layer": row.get("layer"),
        "vascular_compartment_id": row.get("vascular_compartment_id"),
    }


def _source_identity_crosscheck(source_map: dict) -> dict:
    expected = {
        487: ("left ulnar artery", "FMA22798", "FJ2258", 92, 3),
        489: ("left ulnar vein", "FMA22952", "FJ2260", 92, 4),
    }
    result = {}
    for stable_id, (name, concept, member, body, compartment) in expected.items():
        row = source_map.get(str(stable_id), {})
        owner = row.get("source_owner_metadata", {})
        got = (row.get("name"), owner.get("concept_id"), row.get("source_member"),
               row.get("body_index"), row.get("vascular_compartment_id"))
        _require(got == (name, concept, member, body, compartment),
                 f"branch identity metadata changed for stable ID {stable_id}")
        result[str(stable_id)] = {
            "typed_identity": [51011, stable_id],
            "name": name,
            "concept_id": concept,
            "source_member": member,
            "body_index": body,
            "vascular_compartment_id": compartment,
            "identity_source": "current NHA receipt source_id_map/source_owner_metadata",
        }
    return result


def _source0_preflight_status(pending_edges, mismatched_faces, owner_delta: dict) -> str:
    if pending_edges or mismatched_faces or not owner_delta.get("current_capture_candidate_ready"):
        return "SOURCE0_SEAMOWNER_PREPARED_PENDING_NATIVE_EMISSION_AND_TWO_SOURCE_COLLARS"
    return "SOURCE0_SEAMOWNER_CAPTURE_READY_NO_GEOMETRY_PREDICATE"


def build_preflight() -> dict:
    metadata_pin = _resolved_pin(METADATA_PREFLIGHT, METADATA_PREFLIGHT_SHA256)
    metadata = json.loads(METADATA_PREFLIGHT.read_text())
    links = metadata.get("candidate_network_links", {})
    frozen_edges = tuple(
        (tuple(map(int, edge[0])), tuple(map(int, edge[1])))
        for edge in links.get("branch_edges_13_excluding_two_svc_bridge_ports", [])
    )
    frozen_svc = tuple(
        (tuple(map(int, edge[0])), tuple(map(int, edge[1])))
        for edge in links.get("svc_bridge_ports", [])
    )
    _require(frozen_edges == NETWORK_EDGES and frozen_svc == SVC_PORTS
             and links.get("total_link_count") == 15
             and links.get("links_are_topology_inventory_only") is True,
             "actual frozen network topology inventory differs from the declared 13+2 graph")
    context = capture_gate._load_current_inputs()
    _resolved_pin(OWNER_COMPOSITION_RECEIPT, OWNER_COMPOSITION_RECEIPT_SHA256)
    _resolved_pin(OWNER_COMPOSITION_REPORT, OWNER_COMPOSITION_REPORT_SHA256)
    _resolved_pin(OWNER_REPORT_28FC, OWNER_REPORT_28FC_SHA256)
    _resolved_pin(OWNER_REPORT_C37, OWNER_REPORT_C37_SHA256)
    _resolved_pin(OWNER_ERRATUM, OWNER_ERRATUM_SHA256)
    _resolved_pin(OWNER_MODULE, OWNER_MODULE_SHA256)
    _resolved_pin(OWNER_COMPOSER, OWNER_COMPOSER_SHA256)
    erratum = _validate_owner_report_erratum()
    current_receipt = json.loads(OWNER_COMPOSITION_RECEIPT.read_text())
    payload_transition, current_source_rows = _current_payload_row_transition(
        context, current_receipt)
    owner_delta = _owner_composition_delta(
        context["anatomy_receipt"], current_receipt, payload_transition)
    source_baselines, baseline_records = _registered_rigid_source_surfaces(
        context, current_source_rows, current_receipt["provenance"]["source_id_map"])
    _require(set(source_baselines) == {
        (51011, stable_id) for stable_id in NETWORK_STABLE_IDS
    }, "source-rigid baseline does not cover the exact 17 typed network rows")
    for key, surface in source_baselines.items():
        _require(
            _sha_bytes(np.ascontiguousarray(surface[0], dtype="<f4").tobytes())
            == baseline_records[str(key[1])]["rigid_source_world_xyz_f32_sha256"],
            f"source-rigid position bytes differ from the recorded baseline for {key}",
        )
    edge_evidence, separate_evidence = _edge_ancestry(context)
    pending_source_collar_search = _missing_source_collar_search(
        separate_evidence["pending_source_collar_edges"])
    code_pins = capture_gate._current_package_pins()
    code_pins[str(Path(__file__).resolve())] = _sha_file(Path(__file__).resolve())
    code_pins[str(TRANSFORM_OWNER.resolve())] = TRANSFORM_OWNER_SHA256
    code_pins[str(Path(shared_verifier.__file__).resolve())] = _sha_file(
        Path(shared_verifier.__file__).resolve())
    input_paths = (
        capture_gate.CAPTURE_PACK, capture_gate.CAPTURE_RECEIPT,
        capture_gate.ANATOMY_RECEIPT, capture_gate.PREPARED_SUCCESSOR_RECEIPT,
        OWNER_COMPOSITION_RECEIPT, OWNER_COMPOSITION_REPORT, OWNER_COMPOSITION_CARRIER,
        OWNER_COMPOSITION_PAYLOAD_PATH,
        OWNER_REPORT_28FC, OWNER_REPORT_C37, OWNER_ERRATUM,
        OWNER_MODULE, OWNER_COMPOSER,
        context["payload_path"], METADATA_PREFLIGHT, ROUTE100_REPORT, PORT086,
        PORT087, PORT088, PORT089, ROUTE073_REPORT, BRIDGE_ANCESTRY_REPORT,
        BRIDGE_ATTRIBUTION_REPORT, BRIDGE_REPLAY_REPORT, BRIDGE_STATIONARITY_REPORT,
        source007.SOURCE_ARCHIVE, source007.REGISTRATION, source007.RIGID,
        source007.MANIFOLD_SO, source007.MANIFOLD_RECORD,
        source007.MANIFOLD_METADATA, source007.MANIFOLD_WHEEL,
    )
    input_pins = {
        str(Path(path).resolve(strict=True)): _sha_file(Path(path).resolve(strict=True))
        for path in input_paths
    }
    expected_inputs = {
        str(OWNER_COMPOSITION_RECEIPT.resolve()): OWNER_COMPOSITION_RECEIPT_SHA256,
        str(OWNER_COMPOSITION_REPORT.resolve()): OWNER_COMPOSITION_REPORT_SHA256,
        str(OWNER_COMPOSITION_CARRIER.resolve()): OWNER_COMPOSITION_CARRIER_SHA256,
        str(OWNER_COMPOSITION_PAYLOAD_PATH.resolve()): OWNER_COMPOSITION_PAYLOAD_SHA256,
        str(OWNER_REPORT_28FC.resolve()): OWNER_REPORT_28FC_SHA256,
        str(OWNER_REPORT_C37.resolve()): OWNER_REPORT_C37_SHA256,
        str(OWNER_ERRATUM.resolve()): OWNER_ERRATUM_SHA256,
        str(OWNER_MODULE.resolve()): OWNER_MODULE_SHA256,
        str(OWNER_COMPOSER.resolve()): OWNER_COMPOSER_SHA256,
        str(METADATA_PREFLIGHT.resolve()): METADATA_PREFLIGHT_SHA256,
        str(ROUTE100_REPORT.resolve()): ROUTE100_REPORT_SHA256,
        str(PORT086.resolve()): PORT086_SHA256,
        str(PORT087.resolve()): PORT087_SHA256,
        str(PORT088.resolve()): PORT088_SHA256,
        str(PORT089.resolve()): PORT089_SHA256,
        str(ROUTE073_REPORT.resolve()): ROUTE073_REPORT_SHA256,
        str(BRIDGE_ANCESTRY_REPORT.resolve()): BRIDGE_ANCESTRY_SHA256,
        str(BRIDGE_ATTRIBUTION_REPORT.resolve()): BRIDGE_ATTRIBUTION_SHA256,
        str(BRIDGE_REPLAY_REPORT.resolve()): BRIDGE_REPLAY_SHA256,
        str(BRIDGE_STATIONARITY_REPORT.resolve()): BRIDGE_STATIONARITY_SHA256,
        str(source007.SOURCE_ARCHIVE.resolve()): source007.SOURCE_ARCHIVE_SHA,
        str(source007.REGISTRATION.resolve()): source007.REGISTRATION_SHA,
        str(source007.RIGID.resolve()): source007.RIGID_SHA,
    }
    for path, expected in expected_inputs.items():
        _require(input_pins.get(path) == expected,
                 f"pinned source/capture witness input changed: {path}")
    capture = context["capture_identity"]
    _require(capture.get("pack_sha256") == capture_gate.CAPTURE_PACK_SHA256
             and capture.get("capture_receipt_sha256") == capture_gate.CAPTURE_RECEIPT_SHA256
             and capture.get("accepted_step") == 0,
             "actual current capture identity changed")
    mismatched_faces = [
        sid for sid, row in baseline_records.items()
        if row["capture_face_order_equals_nha"] is not True
    ]
    all_known_edges_bound = all(
        edge_evidence[f"{first[1]}->{second[1]}"].get("pairs") is not None
        for first, second in NETWORK_EDGES)
    return {
        "schema": "numi.human.brachiocephalic.current-capture-network-preflight.v2",
        "status": _source0_preflight_status(
            separate_evidence["pending_source_collar_edges"], mismatched_faces,
            owner_delta),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "source_repository": _git_state(),
        "input_sha256": input_pins,
        "current_package_code_sha256": code_pins,
        "actual_accepted_capture": {
            "pack_path": str(capture_gate.CAPTURE_PACK.resolve()),
            "pack_sha256": capture.get("pack_sha256"),
            "receipt_path": str(capture_gate.CAPTURE_RECEIPT.resolve()),
            "receipt_sha256": capture.get("capture_receipt_sha256"),
            "accepted_step": capture.get("accepted_step"),
            "accepted_time_s": capture.get("accepted_time_s"),
            "typed_surface_count": len(context["surfaces"]),
            "capture_receipt_sha256": capture_gate.CAPTURE_RECEIPT_SHA256,
            "body_pose_count": len(context["capture_receipt"]["accepted_body_poses"]),
            "common_field_source_anatomy_payload_sha256":
                context["capture_receipt"].get("common_field_source_anatomy_payload_sha256"),
            "invoked_anatomy_receipt_path": str(capture_gate.ANATOMY_RECEIPT.resolve()),
            "invoked_anatomy_receipt_sha256": capture_gate.ANATOMY_RECEIPT_SHA256,
            "prepared_86a_receipt_is_not_capture_source": True,
        },
        "source_only_current_seamowner_composition": {
            **owner_delta,
            "report_version_erratum": erratum,
            "source_only_composition_report_path": str(OWNER_COMPOSITION_REPORT.resolve()),
            "source_only_composition_report_sha256": OWNER_COMPOSITION_REPORT_SHA256,
            "erratum_path": str(OWNER_ERRATUM.resolve()),
            "erratum_sha256": OWNER_ERRATUM_SHA256,
            "recovered_28fc_report_sha256": OWNER_REPORT_28FC_SHA256,
            "current_c37_report_sha256": OWNER_REPORT_C37_SHA256,
        },
        "source_bound_rigid_rest_baseline": {
            "source_nha_payload_sha256": OWNER_COMPOSITION_PAYLOAD_SHA256,
            "source_nha_payload_path": str(OWNER_COMPOSITION_PAYLOAD_PATH.resolve()),
            "payload_is_current_capture_source": False,
            "method": (
                "For each required network row, apply the existing pinned transform_f32 "
                "to exact NHA vertices6 XYZ using that row's body_index and the actual "
                "accepted rigid pose from the invoked capture receipt. Faces and vertex "
                "ordinals remain the exact ABI5 source ancestry. The cardiac common-field "
                "displacement is excluded from this baseline."
            ),
            "transform_owner_path": str(TRANSFORM_OWNER.resolve()),
            "transform_owner_sha256": TRANSFORM_OWNER_SHA256,
            "rows": baseline_records,
            "rows_with_exact_capture_face_order": len(baseline_records) - len(mismatched_faces),
            "rows_with_face_order_mismatch": mismatched_faces,
            "meaningful_calibre_baseline_ready": all_known_edges_bound and not mismatched_faces,
            "station_transport_for_execution": (
                "At execution, derive each candidate collar material station from exact "
                "interface witnesses/source face ancestry, then transport that station "
                "to this rigid-source baseline by the existing geometry-gate material-vertex ancestry."
            ),
            "same_capture_comparison_is_not_preservation_evidence": True,
            "clinical_calibre_claim": False,
        },
        "network_topology": {
            "metadata_preflight_path": str(METADATA_PREFLIGHT.resolve()),
            "metadata_preflight_sha256": metadata_pin["sha256"],
            "metadata_status": metadata.get("status"),
            "topology_only": True,
            "branch_edge_count": len(NETWORK_EDGES),
            "branch_edges_13_excluding_two_svc_bridge_ports": [
                [list(first), list(second)] for first, second in NETWORK_EDGES],
            "svc_bridge_port_count": len(SVC_PORTS),
            "svc_bridge_ports": [[list(first), list(second)] for first, second in SVC_PORTS],
            "total_link_count": len(NETWORK_EDGES) + len(SVC_PORTS),
            "edge_source_ancestry": {
                edge: {key: value for key, value in item.items() if key != "pairs"}
                for edge, item in edge_evidence.items()
            },
            "svc_source_ancestry": {
                edge: {key: value for key, value in item.items() if key != "pairs"}
                for edge, item in separate_evidence["svc_bridge_ports"].items()
            },
            "potential_469_473_source_overlap_not_network_edge": {
                key: value for key, value in separate_evidence[
                    "potential_469_473_source_overlap"].items() if key != "pairs"
            },
            "pending_source_collar_edges": separate_evidence["pending_source_collar_edges"],
            "pending_source_collar_search": pending_source_collar_search,
            "pending_collar_identity": {
                "479->487": {
                    "parent": _source_identity_summary(
                        current_receipt["provenance"]["source_id_map"], 479),
                    "child": _source_identity_summary(
                        current_receipt["provenance"]["source_id_map"], 487),
                    "reason": "no frozen exact source-face collar ancestry was found",
                },
                "481->489": {
                    "parent": _source_identity_summary(
                        current_receipt["provenance"]["source_id_map"], 481),
                    "child": _source_identity_summary(
                        current_receipt["provenance"]["source_id_map"], 489),
                    "reason": "no frozen exact source-face collar ancestry was found",
                },
            },
            "geometry_predicates_run": False,
            "source_junction_anatomy_admitted": False,
        },
        "branch_identity_crosscheck": {
            **_source_identity_crosscheck(current_receipt["provenance"]["source_id_map"]),
            **{str(sid): {
                "name": context["source_map"][str(sid)].get("name"),
                "concept_id": context["source_map"][str(sid)].get(
                    "source_owner_metadata", {}).get("concept_id"),
                "source_member": context["source_map"][str(sid)].get("source_member"),
            }
            for sid in (483, 485)
        }},
        "all_nearby_aabb_targets_by_network_row": _all_nearby_aabb(context["surfaces"]),
        "aabb_inventory_scope": (
            "Historical A779 capture broadphase only: each of the 17 typed vessel/network "
            "rows compared against every surface decoded from the actual accepted pack. "
            "The pack predates source payload 7 and changed 475/477 parent terms, so "
            "regenerate this exact inventory from the final merged native step-0 pack "
            "before any geometry predicate."
        ),
        "current_verifier": {
            "path": str(Path(shared_verifier.__file__).resolve()),
            "sha256": code_pins[str(Path(shared_verifier.__file__).resolve())],
            "historical_source007_P6_imported": False,
        },
        "planned_solid_and_collar_gate": {
            "gate_owner": "current shared verifier plus existing brachiocephalic 3D port-continuity owner",
            "shared_verifier_path": str(Path(shared_verifier.__file__).resolve()),
            "shared_verifier_sha256": _sha_file(Path(shared_verifier.__file__).resolve()),
            "execution_prerequisites": [
                "final merged receipt retains the exact 8-source bound and is the receipt actually invoked by a fresh native step-0 capture",
                "all 15 graph links have source-registered exact face-patch ancestry, including 479->487 and 481->489",
                "candidate and source-rigid baseline rows bind exact ABI5 source face/vertex ancestry",
                "the all-target inventory is regenerated from that exact fresh capture",
            ],
            "edge_count": 15,
            "source_edge_pairs_missing": separate_evidence["pending_source_collar_edges"],
            "all_target_pair_rule": (
                "Use the shared verifier for every source vessel against all nearby captured "
                "targets; only the exact frozen collar face-patch pairs on the 15 named edges "
                "enter the existing 3D port proof. The 469/473 potential overlap is not a "
                "network edge or implicit allowlist."
            ),
            "frozen_interface_rule": (
                "Observed pose face pairs must stay within the exact source collar "
                "face-patch cross-product; all intersections outside that declared patch reject."
            ),
            "required_per_edge": [
                "closed, oriented, embedded, single-component original surfaces",
                "positive volumetric overlap and exact rational point inside both original Float32 surfaces",
                "source-transported own-material collar area ratio in [0.8,1.25]",
                "source-transported own-material min-Feret ratio in [0.9,1.1]",
                "separate numerical Mesh64 CSG diagnostics, never a substitute for exact inside-both",
                "all off-collar pairs rejected; no generic partial-overlap allowance",
            ],
            "connectivity_inference": {
                "statement": (
                    "If each original surface is verified as a closed connected embedded oriented "
                    "2-manifold, it bounds a solid region; an exact point strictly inside both "
                    "shows positive interior intersection, so the two regions' union is connected."
                ),
                "topology_reference": (
                    "Hatcher, Algebraic Topology, Section 3.3, Alexander duality "
                    "Corollary 3.45: https://pi.math.cornell.edu/~hatcher/AT/ATch3.pdf"
                ),
                "numerical_boolean_component_count_is_not_the_exact_connectivity_proof": True,
            },
            "scope_limit": "one accepted step-0 capture; no all-pose/native admission",
        },
        "aabb_target_count_by_network_row": {
            key: len(value) for key, value in _all_nearby_aabb(
                context["surfaces"]).items()
        },
        "all_target_geometry_predicates_run": False,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
    }


def run_preflight() -> dict:
    _require(not OUTPUT_DIR.exists(),
             "preserve existing outputs; use a fresh named output directory")
    started = time.perf_counter()
    preflight = build_preflight()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    _write_json(PREFLIGHT_PATH, preflight)
    return {
        "status": preflight["status"],
        "preflight_path": str(PREFLIGHT_PATH.resolve()),
        "preflight_sha256": _sha_file(PREFLIGHT_PATH),
        "elapsed_seconds": time.perf_counter() - started,
        "geometry_predicates_run": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.parse_args(argv)
    print(json.dumps(run_preflight(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
