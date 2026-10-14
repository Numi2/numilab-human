"""Source-only preparation for the current-capture brachiocephalic network gate.

This module pins either the historical accepted step-0 inputs or an explicitly
supplied merged-capture bundle, then records its invoked NHA receipt,
registered rigid-source baselines, all-target AABB inventory, and frozen
source-collar ancestry. It deliberately runs no geometry predicates.
"""
from __future__ import annotations

import argparse
import csv
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
from . import resting_brachiocephalic_geometry_gate as geometry_gate
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

ULNAR_SOURCE_PROOF = DATA_ROOT / (
    "ulnar-source-junction-preparation-001/source-rigid-ulnar-junction-proof.json")
ULNAR_SOURCE_PROOF_SHA256 = (
    "46f2a149aa8f330c90f4335becf31a09517c2c6c4893c120923bcabd5532aef4")
ULNAR_SOURCE_PREFLIGHT = DATA_ROOT / (
    "ulnar-source-junction-preparation-001/source-rigid-ulnar-junction-preflight.json")
ULNAR_SOURCE_PREFLIGHT_SHA256 = (
    "411a4956164ef812c68f64b535185b322b706b73af957a5deebac81f2563d8fa")
ULNAR_PROOF_BASE_DIR = DATA_ROOT / "brachiocephalic-seam-owner-composition-001"
ULNAR_PROOF_BASE_RECEIPT = ULNAR_PROOF_BASE_DIR / "resting-anatomy-receipt.json"
ULNAR_PROOF_BASE_RECEIPT_SHA256 = (
    "5ed44e1cfa912f8bf291347aff19c16cf2c55c4b9a36c3ab04fb6aaf17b70474")
ULNAR_PROOF_BASE_PAYLOAD = ULNAR_PROOF_BASE_DIR / "resting-thorax.nhanatomy"
ULNAR_PROOF_BASE_PAYLOAD_SHA256 = (
    "f83a541b794fe9bae699e6784fe091e147d5bad284847773c803c5309027d361")
ULNAR_SOURCE_LINKS = {
    (479, 487): "left-brachial-artery-to-left-ulnar-artery",
    (481, 489): "left-medial-brachial-vein-to-left-ulnar-vein",
}

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

# Independent source-parent calibration baseline for the two vessels whose
# native motion is driven by existing NHTISS parent influences. These inputs
# predate the seam-owner replacement and remain separately hash-pinned.
PARENT_POSE_BASELINE_RECEIPT = DATA_ROOT / (
    "forearm492-native-source-preparation-001/resting-anatomy-receipt.json")
PARENT_POSE_BASELINE_RECEIPT_SHA256 = (
    "1b32f01e90b821f8b04dd15b2951c6a28613db0b90385f9b0510ea1090e942f4")
PARENT_POSE_BASELINE_PAYLOAD = DATA_ROOT.parent / (
    "brachiocephalic-source-candidate-007-006/resting-thorax.nhanatomy")
PARENT_POSE_BASELINE_PAYLOAD_SHA256 = (
    "3a3693b8de2264127e9190a151dca460002ae3b8a0a09edea312bd4eed73ea42")
PARENT_POSE_BASELINE_CARRIER = DATA_ROOT / (
    "brachiocephalic-joint-parent-composition-008-008/joint/assets/"
    "route110-dual-vessel-reference-carriers.nhtissue")
PARENT_POSE_BASELINE_CARRIER_SHA256 = (
    "f77cf7d6da5677178e858f68abe92d7272925f5a783eb8739eba01ad14027393")
PARENT_POSE_FORWARD_OWNER = DATA_ROOT.parent / (
    "neck-back-retained-pose-audit-002/forward_all_positive_weights.py")
PARENT_POSE_FORWARD_OWNER_SHA256 = (
    "ee755eabac267563862c95552524c715bc8dd9897a2bcac7e29e483f5b4a57c5")
PARENT_POSE_SOURCE_INDEX = 6
PARENT_POSE_ROWS = {475: (1, 2), 477: (3, 4)}
PARENT_POSE_BODY_BY_SOURCE_ROW = {1: 20, 2: 91, 3: 20, 4: 91}
COMMON_FIELD_BASELINE_ROWS = {7: 381, 10: 318, 471: 362, 525: 376, 526: 275}
COMMON_FIELD_COORDINATE_COLUMNS = (
    "common_coordinate_RA", "common_coordinate_RV",
    "common_coordinate_LA", "common_coordinate_LV",
    "common_coordinate_RA_material",
    "common_coordinate_ventricular_material",
    "common_coordinate_LA_material",
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
        vertex_map, ancestry = _source_to_capture_vertex_map(
            faces, capture_faces, len(local_xyz), len(captured[0]))
        same_faces = vertex_map is not None
        if same_faces:
            aligned_world = np.empty((len(captured[0]), 3), dtype="<f4")
            for source_vertex, capture_vertex in enumerate(vertex_map):
                if capture_vertex >= 0:
                    aligned_world[int(capture_vertex)] = world[source_vertex]
            source_positions = aligned_world
            source_faces = capture_faces
            source_vertex_ids = np.ascontiguousarray(captured[3], dtype="<u8")
        else:
            source_positions = world
            source_faces = faces
            source_vertex_ids = np.arange(len(world), dtype="<u8")
        source_surfaces[key] = (
            source_positions, source_faces, None, source_vertex_ids)
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
            "rigid_source_world_xyz_f32_sha256": _sha_bytes(source_positions.tobytes()),
            "rigid_source_world_xyz_f32_raw_order_sha256": _sha_bytes(world.tobytes()),
            "nha_faces_u32_sha256": _sha_bytes(faces.tobytes()),
            "captured_faces_u32_sha256": _sha_bytes(capture_faces.tobytes()),
            "capture_face_order_equals_nha": same_faces,
            "capture_face_vertex_ancestry": ancestry,
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




def _validated_ulnar_source_proof_links(data: dict) -> dict[tuple[int, int], dict]:
    _require(data.get("schema") == "numi.human.brachiocephalic-ulnar-source-proof.v1"
             and data.get("status") == "SOURCE_RIGID_ULNAR_JUNCTIONS_PASSED_UNADMITTED"
             and data.get("passed") is True
             and data.get("requested_link_count") == 2
             and data.get("evaluated_link_count") == 2
             and data.get("first_failed_link") is None
             and data.get("source_candidate_admitted") is False
             and data.get("native_or_gpu_work_performed") is False
             and data.get("source_rigid_calibre_is_self_reference") is True
             and data.get("current_capture_or_motion_admission") is False,
             "ulnar source proof is not the exact passed, unadmitted 2-link source-rest report")
    _require(data.get("preflight_sha256") == ULNAR_SOURCE_PREFLIGHT_SHA256
             and Path(data.get("preflight_path", "")).resolve()
             == ULNAR_SOURCE_PREFLIGHT.resolve(),
             "ulnar source proof was not run against its pinned preflight")
    raw_links = data.get("links")
    _require(isinstance(raw_links, list) and len(raw_links) == 2,
             "ulnar source proof does not contain exactly two branch links")
    links = {}
    expected_ids = set(ULNAR_SOURCE_LINKS.values())
    for row in raw_links:
        _require(isinstance(row, dict), "ulnar source proof link is not an object")
        parent, child = row.get("parent"), row.get("child")
        _require(isinstance(parent, list) and len(parent) == 2
                 and isinstance(child, list) and len(child) == 2
                 and parent[0] == child[0] == 51011
                 and type(parent[1]) is int and type(child[1]) is int,
                 "ulnar source proof link identities are malformed")
        edge = (parent[1], child[1])
        _require(edge in ULNAR_SOURCE_LINKS and edge not in links
                 and row.get("link_id") == ULNAR_SOURCE_LINKS[edge]
                 and row.get("passed") is True
                 and row.get("status") == "SOURCE_RIGID_JUNCTION_CONTINUITY_SCREEN_PASSED_UNADMITTED"
                 and row.get("current_capture_used") is False
                 and row.get("source_geometry_frame") == "registered NHRIGID source rest",
                 f"ulnar source proof has an unrecognized or failed edge {edge}")
        derivation = row.get("source_interface_derivation")
        _require(isinstance(derivation, dict)
                 and derivation.get("connected_contact_component_count") == 1
                 and derivation.get("single_connected_source_contact_required") is True,
                 f"ulnar source proof lacks one connected exact source interface for {edge}")
        solid = row.get("port_solid_and_calibre_proof")
        _require(isinstance(solid, dict) and solid.get("passed") is True
                 and solid.get("source_geometry_mutated") is False
                 and solid.get("calibre_gate", {}).get("passed") is True,
                 f"ulnar source proof lacks its existing solid/calibre owner result for {edge}")
        links[edge] = row
    _require(set(links) == set(ULNAR_SOURCE_LINKS)
             and {row["link_id"] for row in links.values()} == expected_ids,
             "ulnar source proof branch coverage differs from the two missing route edges")
    return links


def _source_identity_for_witness(source_map: dict, stable_id: int) -> dict:
    row = source_map.get(str(stable_id))
    _require(isinstance(row, dict), f"source owner map lacks ulnar proof row {stable_id}")
    return {key: row.get(key) for key in (
        "source_member", "source_sha256", "name", "provider", "body_index",
        "layer", "vascular_compartment_id", "source_owner_metadata")}


def _ulnar_source_witness_pairs(
    data: dict, parent: int, child: int, proof_rows: dict, proof_source_map: dict,
    current_rows: dict, current_source_map: dict,
) -> dict:
    edge = (parent, child)
    links = _validated_ulnar_source_proof_links(data)
    _require(edge in links, f"ulnar source proof lacks edge {parent}->{child}")
    link = links[edge]
    derivation = link["source_interface_derivation"]
    raw_pairs = derivation.get("source_face_pairs")
    _require(isinstance(raw_pairs, list) and bool(raw_pairs),
             f"ulnar source proof has no exact face pairs for {edge}")
    pairs = np.asarray(raw_pairs, dtype="<i8")
    _require(pairs.ndim == 2 and pairs.shape[1] == 2
             and len(pairs) == len(raw_pairs)
             and np.array_equal(pairs, np.asarray(sorted(set(map(tuple, pairs))), dtype="<i8"))
             and _sha_bytes(pairs.tobytes()) == derivation.get("source_face_pairs_sha256"),
             f"ulnar source proof face-pair bytes/count are invalid for {edge}")
    source_audit = link.get("unexempted_source_pair_audit", {})
    audit_pairs = np.asarray(source_audit.get("unexpected_triangle_pairs", []), dtype="<i8")
    _require(source_audit.get("triangle_pair_count") == len(pairs)
             and audit_pairs.shape == pairs.shape
             and np.array_equal(audit_pairs, pairs),
             f"ulnar source pairs differ from their exact unexempted source audit for {edge}")

    source_hashes = {}
    for item in link["existing_per_port_owner_trace"].get("manifold_inputs", []):
        diagnostic = item.get("input_diagnostics", {})
        identity = diagnostic.get("identity", item.get("surface_identity"))
        _require(isinstance(identity, list) and len(identity) == 2
                 and identity[0] == 51011 and type(identity[1]) is int,
                 f"ulnar source owner trace identity is malformed for {edge}")
        sid = identity[1]
        _require(sid in edge, f"ulnar source owner trace names unrelated row {sid}")
        observed = (diagnostic.get("source_face_sha256"),
                    diagnostic.get("source_position_sha256"))
        _require(all(isinstance(value, str) and len(value) == 64 for value in observed),
                 f"ulnar source owner trace lacks source geometry hashes for {sid}")
        if sid in source_hashes:
            _require(source_hashes[sid] == observed,
                     f"ulnar source owner trace changes geometry hash between stages for {sid}")
        source_hashes[sid] = observed
    _require(set(source_hashes) == set(edge),
             f"ulnar source owner trace does not bind both rows for {edge}")

    row_bindings = {}
    for sid in edge:
        proof_row, current_row = proof_rows.get(sid), current_rows.get(sid)
        _require(isinstance(proof_row, dict) and isinstance(current_row, dict),
                 f"ulnar proof/current source payload lacks row {sid}")
        proof_xyz = np.ascontiguousarray(proof_row["vertices6"][:, :3], dtype="<f4")
        current_xyz = np.ascontiguousarray(current_row["vertices6"][:, :3], dtype="<f4")
        proof_faces = np.ascontiguousarray(proof_row["faces"], dtype="<u4")
        current_faces = np.ascontiguousarray(current_row["faces"], dtype="<u4")
        _require(proof_xyz.shape == current_xyz.shape
                 and proof_xyz.tobytes() == current_xyz.tobytes()
                 and proof_faces.shape == current_faces.shape
                 and proof_faces.tobytes() == current_faces.tobytes(),
                 f"current source row {sid} differs from the ulnar proof's exact local XYZ/face ancestry")
        _require(_sha_bytes(proof_faces.tobytes()) == source_hashes[sid][0],
                 f"ulnar proof source face hash differs from ABI5 input row {sid}")
        _require(_source_identity_for_witness(proof_source_map, sid)
                 == _source_identity_for_witness(current_source_map, sid),
                 f"current source owner identity differs from ulnar proof row {sid}")
        _require(int(proof_row["body_index"]) == int(current_row["body_index"])
                 == int(current_source_map[str(sid)]["body_index"]),
                 f"current ABI5 rigid owner differs from ulnar proof row {sid}")
        row_bindings[str(sid)] = {
            "typed_identity": [51011, sid],
            "source_face_sha256": _sha_bytes(current_faces.tobytes()),
            "source_local_xyz_sha256": _sha_bytes(current_xyz.tobytes()),
            "same_exact_local_xyz_and_ordered_faces_as_proof_input": True,
            "source_owner_identity_equal_to_proof_input": True,
            "registered_rest_position_sha256_from_proof": source_hashes[sid][1],
        }
    _require((pairs[:, 0] >= 0).all()
             and (pairs[:, 0] < len(current_rows[parent]["faces"])).all()
             and (pairs[:, 1] >= 0).all()
             and (pairs[:, 1] < len(current_rows[child]["faces"])).all(),
             f"ulnar source pair face IDs are outside current source rows for {edge}")
    return {
        "path": str(ULNAR_SOURCE_PROOF.resolve()),
        "sha256": ULNAR_SOURCE_PROOF_SHA256,
        "preflight_path": str(ULNAR_SOURCE_PREFLIGHT.resolve()),
        "preflight_sha256": ULNAR_SOURCE_PREFLIGHT_SHA256,
        "parent_stable_id": parent,
        "child_stable_id": child,
        "pair_count": len(pairs),
        "pairs_sha256": _sha_bytes(pairs.tobytes()),
        "pairs": pairs,
        "source_rows": row_bindings,
        "evidence_status": link["status"],
        "evidence_class": (
            "exact connected source-rest interface ancestry; current capture, "
            "motion, and anatomy admission remain separate"),
        "source_rigid_calibre_is_self_reference": True,
    }


def _validate_ulnar_proof_pin_namespaces(
    raw_data_pins: dict, raw_code_pins: dict,
    expected_data_pins: dict, expected_code_pins: dict,
) -> tuple[dict, dict]:
    """Keep immutable proof inputs separate from its executable dependencies."""
    _require(isinstance(raw_data_pins, dict) and isinstance(raw_code_pins, dict),
             "ulnar source proof lacks separate data/code pin maps")

    def normalize(pins, label):
        result = {}
        for raw_path, digest in pins.items():
            _require(isinstance(raw_path, str) and Path(raw_path).is_absolute()
                     and isinstance(digest, str) and len(digest) == 64
                     and all(ch in "0123456789abcdef" for ch in digest),
                     f"ulnar proof {label} pin is malformed")
            path = str(Path(raw_path).resolve())
            _require(path not in result,
                     f"ulnar proof {label} pins repeat a resolved path")
            result[path] = digest
        return result

    data_pins = normalize(raw_data_pins, "data")
    code_pins = normalize(raw_code_pins, "code")
    expected_data = normalize(expected_data_pins, "expected data")
    expected_code = normalize(expected_code_pins, "expected code")
    _require(all(data_pins.get(path) == digest
                 for path, digest in expected_data.items()),
             "ulnar source proof historical data-input identity changed")
    _require(all(code_pins.get(path) == digest
                 for path, digest in expected_code.items()),
             "ulnar source proof executable dependency identity changed")
    return data_pins, code_pins


def _load_ulnar_source_witnesses(
    current_rows: dict, current_source_map: dict,
) -> tuple[dict[tuple[int, int], dict], dict]:
    _resolved_pin(ULNAR_SOURCE_PROOF, ULNAR_SOURCE_PROOF_SHA256)
    _resolved_pin(ULNAR_SOURCE_PREFLIGHT, ULNAR_SOURCE_PREFLIGHT_SHA256)
    data = json.loads(ULNAR_SOURCE_PROOF.read_text())
    raw_inputs = data.get("source_input_sha256")
    _require(isinstance(raw_inputs, dict), "ulnar source proof lacks input hashes")
    source_input_pins = {
        str(ULNAR_PROOF_BASE_RECEIPT.resolve()): ULNAR_PROOF_BASE_RECEIPT_SHA256,
        str(ULNAR_PROOF_BASE_PAYLOAD.resolve()): ULNAR_PROOF_BASE_PAYLOAD_SHA256,
        str(source007.SOURCE_ARCHIVE.resolve()): source007.SOURCE_ARCHIVE_SHA,
        str(source007.QUOTIENT_REPORT.resolve()): source007.QUOTIENT_REPORT_SHA,
        str(source007.QUOTIENT_NPZ.resolve()): source007.QUOTIENT_NPZ_SHA,
        str(source007.REGISTRATION.resolve()): source007.REGISTRATION_SHA,
        str(source007.RIGID.resolve()): source007.RIGID_SHA,
        **{str(Path(path).resolve()): digest
           for path, digest in source007.ROUTE110_SHA.items()},
    }
    raw_code_pins = data.get("current_shared_geometry_owner_code_sha256", {})
    normalized_inputs, normalized_code_pins = (
        _validate_ulnar_proof_pin_namespaces(
            raw_inputs, raw_code_pins, source_input_pins, source007.MANIFOLD_SHA))
    manifold_code_pins = {
        str(Path(path).resolve()): digest
        for path, digest in source007.MANIFOLD_SHA.items()
    }
    code_pins = normalized_code_pins
    for path, digest in manifold_code_pins.items():
        _resolved_pin(Path(path), digest)
    transform_pins = [
        (Path(path).resolve(), digest) for path, digest in code_pins.items()
        if Path(path).name == TRANSFORM_OWNER.name
    ] if isinstance(code_pins, dict) else []
    _require(len(transform_pins) == 1
             and transform_pins[0][1] == TRANSFORM_OWNER_SHA256
             and transform_pins[0][0].is_file()
             and _sha_file(transform_pins[0][0]) == TRANSFORM_OWNER_SHA256,
             "ulnar source proof does not pin the registered rigid transform owner")
    _resolved_pin(ULNAR_PROOF_BASE_RECEIPT, ULNAR_PROOF_BASE_RECEIPT_SHA256)
    _resolved_pin(ULNAR_PROOF_BASE_PAYLOAD, ULNAR_PROOF_BASE_PAYLOAD_SHA256)
    _, _, proof_rows = capture_gate.nhanat_codec._parse_payload(
        ULNAR_PROOF_BASE_PAYLOAD.read_bytes())
    proof_receipt = json.loads(ULNAR_PROOF_BASE_RECEIPT.read_text())
    proof_source_map = proof_receipt.get("provenance", {}).get("source_id_map", {})
    current_pairs = {}
    _validated_ulnar_source_proof_links(data)
    for parent, child in ULNAR_SOURCE_LINKS:
        current_pairs[(parent, child)] = _ulnar_source_witness_pairs(
            data, parent, child, proof_rows, proof_source_map,
            current_rows, current_source_map)
    summary = {
        "path": str(ULNAR_SOURCE_PROOF.resolve()),
        "sha256": ULNAR_SOURCE_PROOF_SHA256,
        "preflight_path": str(ULNAR_SOURCE_PREFLIGHT.resolve()),
        "preflight_sha256": ULNAR_SOURCE_PREFLIGHT_SHA256,
        "status": data["status"],
        "passed_unadmitted": True,
        "source_rigid_calibre_is_self_reference": True,
        "current_capture_or_motion_admission": False,
        "covered_edges": [
            {"parent": parent, "child": child,
             "pair_count": current_pairs[(parent, child)]["pair_count"],
             "pairs_sha256": current_pairs[(parent, child)]["pairs_sha256"]}
            for parent, child in ULNAR_SOURCE_LINKS
        ],
        "source_data_input_sha256": source_input_pins,
        "proof_recorded_data_input_sha256": normalized_inputs,
        "manifold_dependency_code_sha256": manifold_code_pins,
        "proof_code_pin_count": len(code_pins),
        "proof_source_rows_match_current_candidate_exactly": True,
    }
    return current_pairs, summary

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
    _require(set(pending_edges) <= set(pending_pairs),
             "pending source collar edge set contains an undeclared vascular edge")
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
    ulnar_pairs, ulnar_proof = _load_ulnar_source_witnesses(
        context["rows"], context["source_map"])
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
        elif (parent, child) in ulnar_pairs:
            item = ulnar_pairs[(parent, child)]
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
        "ulnar_source_proof": ulnar_proof,
        "pending_source_collar_edges": pending,
    }


def _runtime_network_source_pairs(edge_evidence: dict, separate_evidence: dict) -> dict:
    """Resolve exactly the 13 graph edges plus the two SVC interfaces."""
    _require(isinstance(edge_evidence, dict) and isinstance(separate_evidence, dict),
             "runtime source ancestry inputs must be mappings")
    expected_edge_names = {f"{first[1]}->{second[1]}" for first, second in NETWORK_EDGES}
    _require(set(edge_evidence) == expected_edge_names,
             "runtime route ancestry does not cover the exact 13 declared links")
    pairs = {}
    for first, second in NETWORK_EDGES:
        edge = f"{first[1]}->{second[1]}"
        item = edge_evidence[edge]
        raw = item.get("pairs") if isinstance(item, dict) else None
        _require(item.get("evidence_status") not in {
            "SOURCE_BOUND_COLLAR_DERIVATION_REQUIRED", None
        } and raw is not None and len(raw) > 0,
            f"runtime source collar ancestry is missing for {edge}")
        pairs[(first, second)] = raw

    svc_rows = separate_evidence.get("svc_bridge_ports", {})
    expected_svc_names = {f"{first[1]}->{second[1]}" for first, second in SVC_PORTS}
    _require(isinstance(svc_rows, dict) and set(svc_rows) == expected_svc_names,
             "runtime ancestry does not cover both explicit SVC interfaces")
    for first, second in SVC_PORTS:
        edge = f"{first[1]}->{second[1]}"
        item = svc_rows[edge]
        raw = item.get("pairs") if isinstance(item, dict) else None
        _require(raw is not None and len(raw) > 0,
                 f"runtime source collar ancestry is missing for {edge}")
        pairs[(first, second)] = raw
    _require(len(pairs) == 15,
             "runtime source ancestry must contain exactly 15 connected-network links")
    _require(POTENTIAL_NONNETWORK_OVERLAP not in pairs
             and (POTENTIAL_NONNETWORK_OVERLAP[1],
                  POTENTIAL_NONNETWORK_OVERLAP[0]) not in pairs,
             "the separate 469/473 overlap must not become an implicit graph edge")
    return pairs


def run_all_target_network_screen(
    context: dict, edge_evidence: dict, separate_evidence: dict,
) -> dict:
    """Run the existing central collision oracle against the full captured pack."""
    source_pairs = _runtime_network_source_pairs(edge_evidence, separate_evidence)
    return geometry_gate.audit_network_rows_against_all_targets(
        shared_verifier,
        context["surfaces"],
        (*NETWORK_EDGES, *SVC_PORTS),
        source_pairs,
    )


def _source_to_capture_vertex_map(
    source_faces: np.ndarray, capture_faces: np.ndarray,
    source_vertex_count: int, capture_vertex_count: int,
) -> tuple[np.ndarray | None, dict]:
    """Match raw NHA corners to capture-local vertices by exact ordered face ancestry."""
    source = np.ascontiguousarray(source_faces, dtype="<u4")
    captured = np.ascontiguousarray(capture_faces, dtype="<u4")
    if (source.ndim != 2 or source.shape[1] != 3
            or captured.ndim != 2 or captured.shape[1] != 3
            or source.shape != captured.shape):
        return None, {
            "status": "CAPTURE_TO_SOURCE_FACE_ANCESTRY_NOT_ALIGNED",
            "source_face_count": len(source),
            "capture_face_count": len(captured),
            "face_count_equal": source.shape == captured.shape,
        }
    if (int(source.max(initial=0)) >= source_vertex_count
            or int(captured.max(initial=0)) >= capture_vertex_count):
        raise ValueError("capture/source face corner index is out of range")
    source_to_capture = np.full(source_vertex_count, -1, dtype="<i8")
    capture_to_source = np.full(capture_vertex_count, -1, dtype="<i8")
    for face_id, (source_face, capture_face) in enumerate(zip(source, captured, strict=True)):
        for source_vertex, capture_vertex in zip(source_face, capture_face, strict=True):
            source_vertex, capture_vertex = int(source_vertex), int(capture_vertex)
            if source_to_capture[source_vertex] not in (-1, capture_vertex):
                raise ValueError(
                    f"source vertex {source_vertex} maps to multiple capture vertices")
            if capture_to_source[capture_vertex] not in (-1, source_vertex):
                raise ValueError(
                    f"capture vertex {capture_vertex} merges distinct source vertices")
            source_to_capture[source_vertex] = capture_vertex
            capture_to_source[capture_vertex] = source_vertex
    used_source = set(map(int, np.unique(source)))
    used_capture = set(map(int, np.unique(captured)))
    if (set(map(int, np.flatnonzero(source_to_capture >= 0))) != used_source
            or set(map(int, np.flatnonzero(capture_to_source >= 0))) != used_capture
            or len(used_capture) != capture_vertex_count):
        return None, {
            "status": "CAPTURE_TO_SOURCE_VERTEX_ANCESTRY_INCOMPLETE",
            "source_used_vertex_count": len(used_source),
            "capture_used_vertex_count": len(used_capture),
            "source_unreferenced_vertex_count": source_vertex_count - len(used_source),
            "capture_unreferenced_vertex_count": capture_vertex_count - len(used_capture),
        }
    return source_to_capture, {
        "status": "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED",
        "source_face_count": len(source),
        "capture_face_count": len(captured),
        "source_used_vertex_count": len(used_source),
        "capture_used_vertex_count": len(used_capture),
        "source_unreferenced_vertex_count": source_vertex_count - len(used_source),
        "capture_unreferenced_vertex_count": capture_vertex_count - len(used_capture),
        "source_faces_u32_sha256": _sha_bytes(source.tobytes()),
        "capture_faces_u32_sha256": _sha_bytes(captured.tobytes()),
        "map_sha256": _sha_bytes(source_to_capture.tobytes()),
    }


def _blend_parent_pose_vertices(
    terms_by_vertex: list, parent_world_positions: dict[int, np.ndarray],
    *, expected_source_index: int, stable_id: int,
) -> tuple[np.ndarray, dict]:
    """Reconstruct mapped material vertices in Float32 owner-term order."""
    _require(isinstance(terms_by_vertex, list) and bool(terms_by_vertex),
             f"parent baseline map for {stable_id} is empty")
    arrays = {
        source_id: np.ascontiguousarray(values, dtype="<f4")
        for source_id, values in parent_world_positions.items()
    }
    _require(all(values.ndim == 2 and values.shape[1] == 3
                 and np.isfinite(values).all()
                 for values in arrays.values()),
             "parent baseline source positions are not finite XYZ arrays")
    output = np.zeros((len(terms_by_vertex), 3), dtype="<f4")
    used_rows = set()
    coefficients = []
    total_terms = 0
    for vertex_id, terms in enumerate(terms_by_vertex):
        _require(isinstance(terms, list) and bool(terms),
                 f"parent baseline map has an empty row {stable_id}/{vertex_id}")
        coefficient_sum = 0.0
        accum = np.zeros(3, dtype="<f4")
        for term in terms:
            _require(
                isinstance(term, list) and len(term) == 4
                and type(term[0]) is int and term[0] == expected_source_index
                and type(term[1]) is int and term[1] in arrays
                and type(term[2]) is int and 0 <= term[2] < len(arrays[term[1]])
                and type(term[3]) in (int, float) and not isinstance(term[3], bool)
                and np.isfinite(term[3]) and 0.0 < float(term[3]) <= 1.0,
                f"invalid parent baseline term for {stable_id}/{vertex_id}",
            )
            coefficient = np.float32(term[3])
            source_point = arrays[term[1]][term[2]]
            product = np.asarray(source_point * coefficient, dtype="<f4")
            accum = np.asarray(accum + product, dtype="<f4")
            coefficient_sum += float(term[3])
            coefficients.append(float(term[3]))
            used_rows.add(term[1])
            total_terms += 1
        _require(abs(coefficient_sum - 1.0) <= 1e-12,
                 f"parent baseline coefficients do not partition vertex {stable_id}/{vertex_id}")
        output[vertex_id] = accum
    return output, {
        "vertex_count": len(terms_by_vertex),
        "parent_term_count": total_terms,
        "used_parent_source_rows": sorted(used_rows),
        "coefficient_min_f64": min(coefficients),
        "coefficient_max_f64": max(coefficients),
        "maximum_partition_residual_f64": max(
            abs(sum(float(term[3]) for term in terms) - 1.0)
            for terms in terms_by_vertex),
        "source_term_order_diagnostic": True,
        "bit_exact_native_accumulation_claim": False,
        "output_xyz_f32_sha256": _sha_bytes(output.tobytes()),
    }


def _common_field_displace_vertices(local_xyz, displacement_basis, coordinates):
    """Apply the existing seven-coordinate material map in explicit Float32 order."""
    local = np.ascontiguousarray(local_xyz, dtype="<f4").copy()
    basis = np.ascontiguousarray(displacement_basis, dtype="<f4")
    q = np.ascontiguousarray(coordinates, dtype="<f4")
    _require(local.ndim == 2 and local.shape[1] == 3
             and basis.shape == (len(local), 7, 4) and q.shape == (7,)
             and np.isfinite(local).all() and np.isfinite(basis).all()
             and np.isfinite(q).all(),
             "common-field source displacement inputs are malformed")
    for coordinate_index in range(7):
        product = np.ascontiguousarray(
            basis[:, coordinate_index, :3] * q[coordinate_index], dtype="<f4")
        local = np.ascontiguousarray(local + product, dtype="<f4")
    return local


def _common_field_source_surfaces(
    context: dict, source_rows: dict, source_map: dict,
    current_receipt: dict,
) -> tuple[dict, dict, dict]:
    """Reconstruct the five mapped torso vessel rows from the actually invoked common owner.

    This is a source-coordinate reconstruction using the accepted native
    coordinate vector and existing map. It is an offline arithmetic model;
    Metal FMA/order and native output are not claimed bit-exact.
    """
    pack_path = context.get("capture_source_paths", {}).get(
        "capture_pack_path", capture_gate.CAPTURE_PACK)
    native_run = Path(pack_path).resolve().parents[1]
    identity_path = native_run / "common-field-map-identity.json"
    _require(identity_path.is_file(),
             "accepted run lacks the existing common-field identity receipt")
    identity_bytes = identity_path.read_bytes()
    identity_sha = _sha_bytes(identity_bytes)
    identity = json.loads(identity_bytes)
    capture_receipt = context["capture_receipt"]
    source_payload_sha = current_receipt.get("payload", {}).get("sha256")
    _require(identity_sha == capture_receipt.get(
                 "common_field_identity_receipt_sha256")
             and identity.get("source_anatomy_payload_sha256") == source_payload_sha
             and capture_receipt.get("common_field_source_anatomy_payload_sha256")
             == source_payload_sha
             and identity.get("geometry_mode") == "common_seven_coordinate_v1"
             and identity.get("coordinate_order") == [
                 "RA", "RV", "LA", "LV", "RA-material",
                 "ventricular-material", "LA-material"],
             "accepted common-field receipt is not bound to the invoked NHA/map ABI")
    _require(identity.get("map_record_layout")
             == "seven float4 displacements, xyz metres per unit dimensionless coordinate, W exact +0",
             "common-field map record layout changed")
    components = {
        "identity": identity_path,
        "map": native_run / identity["map"]["path"],
        "parameters": native_run / identity["parameters"]["path"],
        "polynomials": native_run / identity["polynomials"]["path"],
        "domain_boxes": native_run / identity["domain_boxes"]["path"],
        "surface_audit": native_run / "resting-surface-audit.csv",
    }
    for name, record_name in (
        ("map", "common_field_map_sha256"),
        ("parameters", "common_field_parameters_sha256"),
        ("polynomials", "common_field_polynomial_sha256"),
        ("domain_boxes", "common_field_domain_boxes_sha256"),
    ):
        record = identity[{
            "map": "map", "parameters": "parameters",
            "polynomials": "polynomials", "domain_boxes": "domain_boxes",
        }[name]]
        _require(record.get("sha256") == capture_receipt.get(record_name),
                 f"accepted capture common-field {name} hash differs from identity receipt")
    expected_component_sizes = {
        "map": int(identity["map"]["record_count"])
                   * int(identity["map"]["record_stride_bytes"]),
        "parameters": int(identity["parameters"]["record_bytes"]),
        "polynomials": int(identity["polynomials"]["term_count"])
                        * int(identity["polynomials"]["volume_count"]) * 4,
        "domain_boxes": int(identity["domain_boxes"]["count"])
                        * int(identity["domain_boxes"]["record_stride_bytes"]),
    }
    component_pins = {}
    component_bytes = {}
    for name, path in components.items():
        _require(path.is_file(), f"accepted common-field {name} input is missing")
        raw = path.read_bytes()
        digest = _sha_bytes(raw)
        if name in expected_component_sizes:
            _require(len(raw) == expected_component_sizes[name],
                     f"common-field {name} byte count differs from the identity receipt")
            record_name = {
                "map": "map", "parameters": "parameters",
                "polynomials": "polynomials", "domain_boxes": "domain_boxes",
            }[name]
            _require(digest == identity[record_name]["sha256"],
                     f"common-field {name} digest differs from the identity receipt")
        component_pins[str(path.resolve())] = digest
        component_bytes[name] = raw
    map_record_count = int(identity["map"]["record_count"])
    map_stride = int(identity["map"]["record_stride_bytes"])
    _require(map_stride == 112
             and len(component_bytes["map"]) == map_record_count * map_stride,
             "common-field map is not the existing 112-byte ABI")
    all_records = np.frombuffer(component_bytes["map"], dtype="<f4").reshape(
        map_record_count, 7, 4)
    coordinate_bits = np.ascontiguousarray(all_records[:, :, 3]).view("<u4")
    _require(np.all(coordinate_bits == 0),
             "common-field map W lanes are not canonical positive zero")
    csv_text = component_bytes["surface_audit"].decode("utf-8")
    parsed = list(csv.DictReader(csv_text.splitlines()))
    zero_rows = [row for row in parsed
                 if row.get("step") == "0" and row.get("time_s") == "0"]
    _require(bool(zero_rows)
             and all(row.get("geometry_mode") == "common_seven_coordinate_v1"
                     and row.get("common_coordinate_solver_status") == "0"
                     for row in zero_rows),
             "accepted surface audit lacks a valid step-zero common coordinate record")
    coordinate_rows = [np.asarray([
        np.float32(float(row[column])) for column in COMMON_FIELD_COORDINATE_COLUMNS
    ], dtype="<f4") for row in zero_rows]
    _require(all(np.isfinite(row).all() for row in coordinate_rows)
             and all(row.tobytes() == coordinate_rows[0].tobytes()
                     for row in coordinate_rows[1:]),
             "duplicate step-zero audit records disagree on solved common coordinates")
    coordinates = coordinate_rows[0]
    pose_rows = [
        row for row in capture_receipt.get("accepted_body_poses", [])
        if row.get("body_index") == 20
    ]
    _require(len(pose_rows) == 1,
             "accepted capture lacks a unique Body20 pose for passive vessel rows")
    pose = pose_rows[0]
    q = np.ascontiguousarray(pose.get("quaternion_xyzw"), dtype="<f4")
    t = np.ascontiguousarray(pose.get("position_m"), dtype="<f4")
    _require(q.shape == (4,) and t.shape == (3,)
             and np.isfinite(q).all() and np.isfinite(t).all(),
             "accepted Body20 pose is invalid")
    transform_f32 = _load_transform()
    ranges = identity.get("passive_attachment_ranges", [])
    mapped_network_ids = {
        item.get("stable_id") for item in ranges
        if item.get("semantic") == 51011
        and item.get("stable_id") in NETWORK_STABLE_IDS
    }
    _require(mapped_network_ids == set(COMMON_FIELD_BASELINE_ROWS),
             "existing common-field map covers an undeclared network-row set")
    surfaces = {}
    row_reports = {}
    for stable_id, expected_count in COMMON_FIELD_BASELINE_ROWS.items():
        row = source_rows.get(stable_id)
        source_identity = source_map.get(str(stable_id))
        _require(isinstance(row, dict) and isinstance(source_identity, dict),
                 f"common-field baseline row {stable_id} is absent")
        _require(int(row["body_index"]) == 20
                 and int(row["record"][2]) == expected_count
                 and int(row["record"][5]) == stable_id
                 and source_identity.get("body_index") == 20,
                 f"common-field row {stable_id} lost its ABI5 Body20 source identity")
        matches = [
            item for item in ranges
            if item.get("semantic") == 51011
            and item.get("stable_id") == stable_id
        ]
        _require(len(matches) == 1,
                 f"common-field map lacks a unique stable-ID range for row {stable_id}")
        region = matches[0]
        first = region.get("first_vertex")
        count = region.get("vertex_count")
        _require(type(region.get("body_index")) is int
                 and region["body_index"] == 20
                 and type(first) is int and type(count) is int
                 and count == expected_count and 0 <= first <= map_record_count - count,
                 f"common-field range is invalid for row {stable_id}")
        basis = all_records[first:first + count]
        local_xyz = np.ascontiguousarray(row["vertices6"][:, :3], dtype="<f4")
        deformed_local = _common_field_displace_vertices(
            local_xyz, basis, coordinates)
        world = np.ascontiguousarray(
            transform_f32(q, t, deformed_local), dtype="<f4")
        faces = np.ascontiguousarray(row["faces"], dtype="<u4")
        captured = context["surfaces"].get((51011, stable_id))
        _require(captured is not None,
                 f"accepted capture lacks common-field row {(51011, stable_id)}")
        capture_faces = np.ascontiguousarray(captured[1], dtype="<u4")
        vertex_map, ancestry = _source_to_capture_vertex_map(
            faces, capture_faces, len(world), len(captured[0]))
        _require(vertex_map is not None
                 and np.all(vertex_map >= 0)
                 and len(world) == len(captured[0]),
                 f"common-field row {stable_id} cannot align to exact capture ancestry")
        aligned = np.empty_like(np.asarray(captured[0], dtype="<f4"))
        for source_vertex, capture_vertex in enumerate(vertex_map):
            aligned[int(capture_vertex)] = world[source_vertex]
        actual = np.ascontiguousarray(captured[0], dtype="<f4")
        residual = np.abs(aligned.astype("<f8") - actual.astype("<f8"))
        surfaces[(51011, stable_id)] = (
            aligned, capture_faces, None,
            np.ascontiguousarray(captured[3], dtype="<u8"))
        row_reports[str(stable_id)] = {
            "typed_identity": [51011, stable_id],
            "body_index": 20,
            "source_vertex_count": len(local_xyz),
            "source_face_count": len(faces),
            "common_map_range": {"first_vertex": first, "vertex_count": count},
            "source_local_xyz_f32_sha256": _sha_bytes(local_xyz.tobytes()),
            "source_faces_u32_sha256": _sha_bytes(faces.tobytes()),
            "map_range_f32_sha256": _sha_bytes(
                np.ascontiguousarray(basis, dtype="<f4").tobytes()),
            "source_deformed_local_xyz_f32_sha256": _sha_bytes(
                deformed_local.tobytes()),
            "source_owner_world_xyz_f32_sha256": _sha_bytes(world.tobytes()),
            "capture_vertex_ancestry": ancestry,
            "aligned_owner_world_xyz_f32_sha256": _sha_bytes(aligned.tobytes()),
            "actual_capture_xyz_f32_sha256": _sha_bytes(actual.tobytes()),
            "owner_reconstruction_vs_capture_max_abs_m": float(residual.max()),
            "owner_reconstruction_vs_capture_rms_m": float(
                np.sqrt(np.mean(residual * residual))),
            "exact_position_match_count": int(np.count_nonzero(
                np.all(aligned.view("<u4") == actual.view("<u4"), axis=1))),
            "comparison_scope": (
                "diagnostic comparison of an offline Float32 source/map reconstruction "
                "with actual captured positions; does not establish native arithmetic equivalence"),
        }
    method = (
        "Read the actual captured step-zero solved seven-coordinate vector, then apply "
        "the existing bound common-field map range to each source-local vertex using "
        "sequential Float32 product/add arithmetic and the existing Body20 transform. "
        "This is a source-owner reconstruction baseline; Metal FMA and exact native "
        "arithmetic are not claimed."
    )
    report = {
        "status": "COMMON_FIELD_OWNER_SOURCE_RECONSTRUCTION_READY_NOT_ADMISSION",
        "geometry_predicates_run": False,
        "native_arithmetic_bit_exact_claim": False,
        "common_field_identity_receipt_sha256": identity_sha,
        "source_anatomy_payload_sha256": source_payload_sha,
        "map_record_count": map_record_count,
        "map_record_stride_bytes": map_stride,
        "coordinate_order": identity["coordinate_order"],
        "accepted_step_zero_coordinates_f32": [float(value) for value in coordinates],
        "identical_step_zero_audit_record_count": len(zero_rows),
        "accepted_body20_pose": {
            "position_m_f32": [float(value) for value in t],
            "quaternion_xyzw_f32": [float(value) for value in q],
            "body_pose_source": "actual accepted capture receipt",
        },
        "input_sha256": component_pins,
        "rows": row_reports,
        "method": method,
        "historical_map_prefix_and_physical_state_changed": False,
        "clinical_or_physiological_claim": False,
    }
    return surfaces, report, component_pins


def _load_parent_pose_forward_owner():
    _resolved_pin(PARENT_POSE_FORWARD_OWNER, PARENT_POSE_FORWARD_OWNER_SHA256)
    spec = importlib.util.spec_from_file_location(
        "_numi_retained_parent_pose_forward_owner", PARENT_POSE_FORWARD_OWNER)
    _require(spec is not None and spec.loader is not None,
             "retained parent-pose forward owner cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("load_tissue", "row_data", "forward"):
        _require(callable(getattr(module, name, None)),
                 f"retained parent-pose owner lacks {name}")
    return module


def _parent_pose_source_surfaces(
    context: dict, current_source_rows: dict, current_receipt: dict,
) -> tuple[dict, dict]:
    """Build separate 475/477 source baselines from their registered tissue parents."""
    for path, digest in (
        (PARENT_POSE_BASELINE_RECEIPT, PARENT_POSE_BASELINE_RECEIPT_SHA256),
        (PARENT_POSE_BASELINE_PAYLOAD, PARENT_POSE_BASELINE_PAYLOAD_SHA256),
        (PARENT_POSE_BASELINE_CARRIER, PARENT_POSE_BASELINE_CARRIER_SHA256),
        (PARENT_POSE_FORWARD_OWNER, PARENT_POSE_FORWARD_OWNER_SHA256),
    ):
        _resolved_pin(path, digest)
    baseline_receipt = json.loads(PARENT_POSE_BASELINE_RECEIPT.read_text())
    baseline_payload = baseline_receipt.get("payload", {})
    _require(Path(baseline_payload.get("path", "")).resolve()
             == PARENT_POSE_BASELINE_PAYLOAD.resolve()
             and baseline_payload.get("sha256") == PARENT_POSE_BASELINE_PAYLOAD_SHA256
             and baseline_payload.get("abi") == 5,
             "retained parent-pose baseline receipt does not bind its exact ABI5 source")
    _, _, baseline_rows = capture_gate.nhanat_codec._parse_payload(
        PARENT_POSE_BASELINE_PAYLOAD.read_bytes())
    _require(set(PARENT_POSE_ROWS).issubset(baseline_rows)
             and set(PARENT_POSE_ROWS).issubset(current_source_rows),
             "parent-pose baseline is missing source vessel rows")
    for stable_id in PARENT_POSE_ROWS:
        old_map = baseline_receipt.get("provenance", {}).get(
            "source_id_map", {}).get(str(stable_id), {})
        new_map = current_receipt.get("provenance", {}).get(
            "source_id_map", {}).get(str(stable_id), {})
        identity_fields = ("name", "body_index", "layer", "source_member",
                           "source_member_sha256", "vascular_compartment_id")
        _require(all(old_map.get(field) == new_map.get(field)
                     for field in identity_fields),
                 f"registered source identity changed for parent-mapped row {stable_id}")
        old_owner = old_map.get("source_owner_metadata", {})
        new_owner = new_map.get("source_owner_metadata", {})
        _require(all(old_owner.get(field) == new_owner.get(field)
                     for field in ("label", "concept_id")),
                 f"anatomical identity changed for parent-mapped row {stable_id}")
        old_vertices = np.ascontiguousarray(
            baseline_rows[stable_id]["vertices6"], dtype="<f4")
        new_vertices = np.ascontiguousarray(
            current_source_rows[stable_id]["vertices6"], dtype="<f4")
        old_faces = np.ascontiguousarray(
            baseline_rows[stable_id]["faces"], dtype="<u4")
        new_faces = np.ascontiguousarray(
            current_source_rows[stable_id]["faces"], dtype="<u4")
        _require(old_vertices.shape == new_vertices.shape
                 and old_vertices.tobytes() == new_vertices.tobytes(),
                 f"source material vertices changed for parent-mapped row {stable_id}")
        if stable_id == 475:
            _require(old_faces.shape == new_faces.shape
                     and old_faces.tobytes() == new_faces.tobytes(),
                     "row 475 source face ancestry changed")
        elif old_faces.tobytes() != new_faces.tobytes():
            cap = _validate_reinstated_cap_face_transition(
                old_faces, new_faces, old_vertices, new_vertices)
            _require(cap["source_face_id_zero_based"] == 1858,
                     "row 477 face delta is not the frozen source cap reinstatement")

    bindings = baseline_receipt.get("passive_tissue_parent_bindings", {})
    parent_rows = _stable_owner_rows(bindings)
    payloads = bindings.get("source_payloads", [])
    _require(PARENT_POSE_SOURCE_INDEX < len(payloads),
             "historical parent map does not contain source payload index 6")
    source_payload = payloads[PARENT_POSE_SOURCE_INDEX]
    _require(source_payload.get("sha256") == PARENT_POSE_BASELINE_CARRIER_SHA256
             and Path(source_payload.get("path", "")).resolve()
             == PARENT_POSE_BASELINE_CARRIER.resolve(),
             "historical parent map source index 6 is not the pinned route110 carrier")
    parent_terms = {}
    for stable_id, source_ids in PARENT_POSE_ROWS.items():
        owner = parent_rows.get(stable_id)
        _require(isinstance(owner, dict)
                 and len(owner.get("vertices", []))
                 == len(current_source_rows[stable_id]["vertices6"]),
                 f"historical parent map has the wrong vertex count for {stable_id}")
        parent_terms[stable_id] = owner["vertices"]
        _require(all(
            isinstance(terms, list) and bool(terms)
            and all(isinstance(term, list) and len(term) == 4
                    and type(term[1]) is int for term in terms)
            for terms in owner["vertices"]),
            f"historical parent map terms are malformed for {stable_id}")
        used = {term[1] for terms in owner["vertices"] for term in terms}
        _require(used.issubset(set(source_ids)) and used,
                 f"historical parent map uses an unexpected source row for {stable_id}")

    helper = _load_parent_pose_forward_owner()
    tissue = helper.load_tissue(PARENT_POSE_BASELINE_CARRIER)
    _require(tissue.get("raw") is not None
             and hashlib.sha256(tissue["raw"]).hexdigest()
             == PARENT_POSE_BASELINE_CARRIER_SHA256,
             "parent-pose carrier decoder did not return the exact pinned bytes")
    accepted = context["capture_receipt"].get("accepted_body_poses")
    _require(isinstance(accepted, list) and accepted,
             "accepted current capture has no body-pose list")
    poses = {}
    pose_digests = {}
    for pose in accepted:
        _require(isinstance(pose, dict) and type(pose.get("body_index")) is int,
                 "accepted body pose identity is not a JSON integer")
        body = pose["body_index"]
        _require(body not in poses,
                 f"accepted body pose list has duplicate body {body}")
        position = np.ascontiguousarray(pose.get("position_m"), dtype="<f4")
        quaternion = np.ascontiguousarray(pose.get("quaternion_xyzw"), dtype="<f4")
        _require(position.shape == (3,) and quaternion.shape == (4,)
                 and np.isfinite(position).all() and np.isfinite(quaternion).all(),
                 f"accepted body pose {body} is invalid")
        poses[body] = (position, quaternion)
        pose_digests[str(body)] = {
            "position_m_f32_sha256": _sha_bytes(position.tobytes()),
            "quaternion_xyzw_f32_sha256": _sha_bytes(quaternion.tobytes()),
        }

    parent_world = {}
    parent_reports = {}
    for source_row, expected_body in PARENT_POSE_BODY_BY_SOURCE_ROW.items():
        row = helper.row_data(tissue, source_row)
        _require(row["vc"] == (902 if source_row in (1, 2) else 1089)
                 and expected_body in poses,
                 f"registered carrier row {source_row} count or accepted pose changed")
        active_counts = (row["weights"] > 0).sum(axis=1)
        row_bodies = set()
        for vertex_id, active in enumerate(active_counts):
            _require(int(active) == 1,
                     f"carrier row {source_row}/{vertex_id} is not one-hot")
            slot = int(np.flatnonzero(row["weights"][vertex_id] > 0)[0])
            local_binding = int(row["local"][vertex_id, slot])
            binding = tissue["bindings"][row["fb"] + local_binding]
            row_bodies.add(int(binding["core"]))
            _require(float(row["weights"][vertex_id, slot]) == 1.0,
                     f"carrier row {source_row}/{vertex_id} weight is not exactly one")
        _require(row_bodies == {expected_body},
                 f"carrier row {source_row} is not owned only by body {expected_body}")
        positions = np.ascontiguousarray(
            helper.forward(row["positions"], row["local"], row["weights"],
                           row, tissue, poses), dtype="<f4")
        _require(positions.shape == (row["vc"], 3)
                 and np.isfinite(positions).all(),
                 f"parent-pose forward output is invalid for carrier row {source_row}")
        parent_world[source_row] = positions
        parent_reports[str(source_row)] = {
            "vertex_count": row["vc"],
            "triangle_count": row["ic"] // 3,
            "resolved_core_body_index": expected_body,
            "binding_cardinality": 1,
            "all_positive_weights_exactly_one": True,
            "source_vertex_bytes_sha256": _sha_bytes(row["positions"].tobytes()),
            "source_binding_bytes_sha256": _sha_bytes(row["binding_bytes"]),
            "source_local_binding_and_weight_sha256": _sha_bytes(
                row["local"].tobytes() + row["weights"].tobytes()),
            "accepted_body_pose": pose_digests[str(expected_body)],
            "forward_xyz_f32_sha256": _sha_bytes(positions.tobytes()),
        }

    surfaces = {}
    vessel_reports = {}
    for stable_id, source_ids in PARENT_POSE_ROWS.items():
        positions, blend_report = _blend_parent_pose_vertices(
            parent_terms[stable_id], parent_world,
            expected_source_index=PARENT_POSE_SOURCE_INDEX,
            stable_id=stable_id,
        )
        current = current_source_rows[stable_id]
        faces = np.ascontiguousarray(current["faces"], dtype="<u4")
        capture_surface = context["surfaces"].get((51011, stable_id))
        _require(capture_surface is not None
                 and len(capture_surface[3]) == len(capture_surface[0]),
                 f"accepted capture lacks typed material identities for {stable_id}")
        capture_faces = np.ascontiguousarray(capture_surface[1], dtype="<u4")
        vertex_map, alignment = _source_to_capture_vertex_map(
            faces, capture_faces, len(positions), len(capture_surface[0]))
        captured = np.ascontiguousarray(capture_surface[0], dtype="<f4")
        delta_report = None
        if vertex_map is None:
            surface_positions = positions
            surface_faces = faces
            surface_ids = np.arange(len(positions), dtype="<u8")
        else:
            surface_positions = np.empty_like(captured, dtype="<f4")
            for source_vertex, capture_vertex in enumerate(vertex_map):
                if capture_vertex >= 0:
                    surface_positions[int(capture_vertex)] = positions[source_vertex]
            surface_faces = capture_faces
            surface_ids = np.ascontiguousarray(capture_surface[3], dtype="<u8")
            delta = np.linalg.norm(
                captured.astype(np.float64) - surface_positions.astype(np.float64), axis=1)
            delta_report = {
                "max": float(delta.max()),
                "mean": float(delta.mean()),
                "p95": float(np.quantile(delta, 0.95)),
            }
        surfaces[(51011, stable_id)] = (
            surface_positions, surface_faces, None, surface_ids)
        vessel_reports[str(stable_id)] = {
            "typed_identity": [51011, stable_id],
            "baseline_source_rows": list(source_ids),
            "baseline_parent_source_payload_index": PARENT_POSE_SOURCE_INDEX,
            "baseline_parent_source_payload_sha256": PARENT_POSE_BASELINE_CARRIER_SHA256,
            "baseline_parent_source_map_sha256": _sha_bytes(json.dumps(
                parent_terms[stable_id], sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode()),
            "baseline_face_source": (
                "candidate NHA row after exact source-identity/face-transition checks"),
            "capture_vertex_ancestry": alignment,
            "capture_aligned_current_surface": vertex_map is not None,
            "face_count": len(surface_faces),
            "material_vertex_count": len(surface_positions),
            "source_baseline_vertex_count": len(positions),
            "baseline_xyz_f32_sha256": _sha_bytes(positions.tobytes()),
            "capture_aligned_baseline_xyz_f32_sha256": (
                _sha_bytes(surface_positions.tobytes())
                if vertex_map is not None else None),
            "accepted_capture_xyz_f32_sha256": _sha_bytes(captured.tobytes()),
            "same_capture_position_delta_diagnostic_m": delta_report,
            "same_capture_delta_is_not_a_physiology_or_native_error_bound": True,
            **blend_report,
        }
    return surfaces, {
        "status": "PARENT_POSE_SOURCE_BASELINES_RECONSTRUCTED_OFFLINE",
        "source_baseline_receipt_path": str(PARENT_POSE_BASELINE_RECEIPT.resolve()),
        "source_baseline_receipt_sha256": PARENT_POSE_BASELINE_RECEIPT_SHA256,
        "source_baseline_payload_path": str(PARENT_POSE_BASELINE_PAYLOAD.resolve()),
        "source_baseline_payload_sha256": PARENT_POSE_BASELINE_PAYLOAD_SHA256,
        "source_parent_carrier_path": str(PARENT_POSE_BASELINE_CARRIER.resolve()),
        "source_parent_carrier_sha256": PARENT_POSE_BASELINE_CARRIER_SHA256,
        "forward_owner_path": str(PARENT_POSE_FORWARD_OWNER.resolve()),
        "forward_owner_sha256": PARENT_POSE_FORWARD_OWNER_SHA256,
        "accepted_body_pose_count": len(accepted),
        "parent_source_rows": parent_reports,
        "vessels": vessel_reports,
        "anatomical_admission": False,
        "native_or_gpu_work_performed": False,
        "method_limit": (
            "Existing NHTISS parent rows are forwarded through the accepted rigid body poses, "
            "then blended with the predecessor receipt's exact Float64 parent-map terms using "
            "ordered Float32 owner arithmetic. This supplies an independent source-parent "
            "calibre reference only for rows 475 and 477. It does not reconstruct cardiac "
            "common-field motion or prove native emission."),
    }


def _parent_motion_owner_inventory(receipt: dict, network_ids: set[int]) -> dict:
    bindings = receipt.get("passive_tissue_parent_bindings", {})
    sources = bindings.get("source_payloads", [])
    rows = _stable_owner_rows(bindings)
    source_hashes = []
    for index, source in enumerate(sources):
        digest = source.get("sha256") if isinstance(source, dict) else None
        _require(isinstance(digest, str) and len(digest) == 64
                 and all(character in "0123456789abcdef" for character in digest),
                 f"parent source payload {index} has an invalid SHA-256 identity")
        source_hashes.append(digest)
    result = {}
    for stable_id in sorted(network_ids):
        row = rows.get(stable_id)
        if row is None:
            result[str(stable_id)] = {
                "mode": "existing_native_rigid_body_binding",
                "parent_map_present": False,
            }
            continue
        terms_by_vertex = row.get("vertices")
        _require(isinstance(terms_by_vertex, list) and bool(terms_by_vertex),
                 f"parent motion map for vessel {stable_id} has no vertices")
        used_sources = set()
        for vertex_id, terms in enumerate(terms_by_vertex):
            _require(isinstance(terms, list) and bool(terms),
                     f"parent motion map has an empty vertex term row {stable_id}/{vertex_id}")
            coeff_sum = 0.0
            for term in terms:
                _require(isinstance(term, list) and len(term) == 4
                         and type(term[0]) is int and 0 <= term[0] < len(source_hashes)
                         and type(term[1]) is int and term[1] >= 0
                         and type(term[2]) is int and term[2] >= 0
                         and type(term[3]) in (int, float)
                         and np.isfinite(term[3]) and 0.0 < float(term[3]) <= 1.0,
                         f"invalid parent motion term for vessel {stable_id}/{vertex_id}")
                used_sources.add(term[0])
                coeff_sum += float(term[3])
            _require(abs(coeff_sum - 1.0) <= 1e-12,
                     f"parent motion coefficients do not sum to one for {stable_id}/{vertex_id}")
        canonical = json.dumps(
            terms_by_vertex, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        result[str(stable_id)] = {
            "mode": "existing_native_passive_anatomy_parent_influences",
            "parent_map_present": True,
            "vertex_count": len(terms_by_vertex),
            "parent_map_terms_sha256": _sha_bytes(canonical),
            "used_source_payload_indices": sorted(used_sources),
            "used_source_payload_sha256": [
                source_hashes[index] for index in sorted(used_sources)],
            "motion_values_are_from_existing_native_owner": True,
            "independent_parent_pose_reconstruction_performed": False,
        }
    return {
        "source_payload_count": len(source_hashes),
        "source_payload_sha256": source_hashes,
        "rows": result,
        "all_parent_rows_bound_to_existing_native_map": True,
        "parent_pose_reconstruction_performed": False,
        "scope_limit": (
            "Records exact parent-map ownership used by the invoked native receipt. "
            "It does not reconstruct multi-body parent-pose positions for independent "
            "calibre comparison."),
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


def _compare_payload_rows_exact(
    first_rows: dict, second_rows: dict, first_records: list, second_records: list,
) -> dict:
    _require(set(first_rows) == set(second_rows),
             "emitted candidate changed the ABI5 stable-ID row inventory")
    _require([tuple(map(int, row)) for row in first_records]
             == [tuple(map(int, row)) for row in second_records],
             "emitted candidate ABI5 record order or allocation differs from its receipt")
    for stable_id in sorted(first_rows):
        first, second = first_rows[stable_id], second_rows[stable_id]
        first_vertices = np.ascontiguousarray(first["vertices6"], dtype="<f4")
        second_vertices = np.ascontiguousarray(second["vertices6"], dtype="<f4")
        first_faces = np.ascontiguousarray(first["faces"], dtype="<u4")
        second_faces = np.ascontiguousarray(second["faces"], dtype="<u4")
        _require(first_vertices.shape == second_vertices.shape
                 and first_vertices.tobytes() == second_vertices.tobytes()
                 and first_faces.shape == second_faces.shape
                 and first_faces.tobytes() == second_faces.tobytes()
                 and tuple(map(int, first["record"])) == tuple(map(int, second["record"])),
                 f"emitted candidate ABI5 row {stable_id} differs byte-for-byte")
    return {
        "status": "ALL_ABI5_ROWS_BYTE_EXACT_CAPTURED_SOURCE_CANDIDATE",
        "row_count": len(first_rows),
        "position_and_normal_rows_exact": len(first_rows),
        "face_rows_exact": len(first_rows),
        "record_order_and_allocation_exact": True,
    }


def _captured_candidate_owner_binding_delta(actual_receipt: dict,
                                            current_receipt: dict) -> dict:
    actual_provenance = actual_receipt.get("provenance", {})
    current_provenance = current_receipt.get("provenance", {})
    _require(actual_provenance.get("source_id_map")
             == current_provenance.get("source_id_map"),
             "invoked candidate capture source identities differ from the source receipt")
    actual_bindings = actual_receipt.get("passive_tissue_parent_bindings", {})
    current_bindings = current_receipt.get("passive_tissue_parent_bindings", {})
    actual_rows, actual_hashes = _semantic_parent_owner_rows(
        actual_bindings, "invoked candidate capture")
    current_rows, current_hashes = _semantic_parent_owner_rows(
        current_bindings, "candidate source receipt")
    _require(actual_hashes == current_hashes and actual_rows == current_rows,
             "invoked candidate capture parent source order or semantic owner maps differ")
    extension = current_provenance.get(
        "passive_brachiocephalic_exact_coordinate_owner_composition", {})
    _require(extension.get("final_source_payload_count") == len(current_hashes)
             and len(current_hashes) <= 8
             and extension.get("additional_physical_mass_kg") == 0
             and extension.get("additional_force_or_state") is False
             and extension.get("anatomical_admission") is False,
             "captured candidate receipt exceeds existing owner/state/mass bounds")
    return {
        "schema": "numi.human.brachiocephalic.captured-candidate-owner-binding.v1",
        "status": "INVOKED_CAPTURE_AND_SOURCE_RECEIPT_OWNER_MAPS_EXACT",
        "source_payload_count": len(current_hashes),
        "source_payload_sha256_in_native_order": current_hashes,
        "parent_map_surface_count": len(current_rows),
        "source_id_map_exact": True,
        "parent_owner_terms_exact_after_payload_identity_resolution": True,
        "additional_physical_mass_kg": 0,
        "additional_force_or_state": False,
        "current_capture_candidate_ready": True,
        "current_capture_emitted_this_candidate_receipt": True,
    }


def _current_payload_row_transition(context: dict, current_receipt: dict) -> tuple[dict, dict]:
    payload = current_receipt.get("payload", {})
    path = Path(payload.get("path", "")).resolve(strict=True)
    candidate_hash = payload.get("sha256")
    _require(isinstance(candidate_hash, str) and len(candidate_hash) == 64
             and all(character in "0123456789abcdef" for character in candidate_hash)
             and _sha_file(path) == candidate_hash,
             "current source receipt does not bind its exact NHA payload")
    actual_provenance = context["anatomy_receipt"].get("provenance", {})
    actual_payload = context["anatomy_receipt"].get("payload", {})
    actual_payload_hash = context.get("capture_source_sha256", {}).get(
        "anatomy_payload", capture_gate.ANATOMY_PAYLOAD_SHA256)
    _require(actual_payload.get("sha256") == actual_payload_hash,
             "actual accepted capture receipt no longer binds its pinned anatomy payload")
    header, records, rows = capture_gate.nhanat_codec._parse_payload(path.read_bytes())
    actual_payload_path = Path(actual_payload.get("path", "")).resolve(strict=True)
    if actual_payload.get("sha256") == candidate_hash:
        actual_header, actual_records, actual_rows = capture_gate.nhanat_codec._parse_payload(
            actual_payload_path.read_bytes())
        _require(int(actual_header[1]) == 5,
                 "invoked current candidate capture is not NHANAT ABI5")
        exact_rows = _compare_payload_rows_exact(
            actual_rows, rows, actual_records, records)
        _require(actual_provenance.get("source_id_map")
                 == current_receipt.get("provenance", {}).get("source_id_map"),
                 "invoked capture source map differs from its candidate receipt")
        return ({
            "actual_capture_payload_sha256": actual_payload["sha256"],
            "current_source_only_payload_path": str(path),
            "current_source_only_payload_sha256": payload["sha256"],
            "full_payload_hashes_differ_as_expected": False,
            "rowwise_geometry_and_abi_delta": exact_rows,
            "source_id_map_delta": {
                "status": "SOURCE_IDENTITIES_BYTE_EXACT_CAPTURED_CANDIDATE",
                "all_source_id_map_entries_exact": True,
            },
            "scope": (
                "The accepted native pack's invoked NHA and the candidate receipt bind "
                "the same ABI5 payload; every decoded row and record was compared byte-exactly."),
            "not_native_emitted_or_geometry_admitted": False,
            "native_emitted_current_candidate": True,
        }, rows)
    _require(candidate_hash == OWNER_COMPOSITION_PAYLOAD_SHA256
             and int(header[1]) == 5
             and len(records) == int(payload.get("surface_count", -1))
             and set(rows) == set(context["rows"]),
             "unemitted source-only successor is not the pinned ABI5 source candidate")
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


def _semantic_parent_owner_rows(bindings: dict, label: str) -> tuple[dict[int, dict], list[str]]:
    sources = bindings.get("source_payloads")
    _require(isinstance(sources, list) and 0 < len(sources) <= 8,
             f"{label} parent payload inventory is outside the native bound")
    hashes = []
    for index, source in enumerate(sources):
        digest = source.get("sha256") if isinstance(source, dict) else None
        _require(isinstance(digest, str) and len(digest) == 64
                 and all(character in "0123456789abcdef" for character in digest),
                 f"{label} source payload {index} lacks a canonical SHA-256")
        hashes.append(digest)
    _require(len(set(hashes)) == len(hashes),
             f"{label} repeats a source payload identity and makes indices ambiguous")
    raw_rows = _stable_owner_rows(bindings)
    semantic = {}
    for stable_id, row in raw_rows.items():
        copied = {key: value for key, value in row.items() if key != "vertices"}
        vertices = row.get("vertices")
        _require(isinstance(vertices, list),
                 f"{label} parent owner row {stable_id} has no vertex terms")
        semantic_vertices = []
        for vertex_id, terms in enumerate(vertices):
            _require(isinstance(terms, list) and bool(terms),
                     f"{label} parent owner row {stable_id}/{vertex_id} is empty")
            semantic_terms = []
            coefficient_sum = 0.0
            for term in terms:
                _require(isinstance(term, list) and len(term) == 4
                         and type(term[0]) is int and 0 <= term[0] < len(hashes)
                         and type(term[1]) is int and term[1] >= 0
                         and type(term[2]) is int and term[2] >= 0
                         and type(term[3]) in (int, float)
                         and np.isfinite(term[3]) and 0.0 < float(term[3]) <= 1.0,
                         f"{label} parent owner term is invalid at {stable_id}/{vertex_id}")
                coefficient = float(term[3])
                coefficient_sum += coefficient
                semantic_terms.append([hashes[term[0]], term[1], term[2], coefficient])
            _require(abs(coefficient_sum - 1.0) <= 1e-12,
                     f"{label} parent owner terms do not form a partition at "
                     f"{stable_id}/{vertex_id}")
            semantic_vertices.append(semantic_terms)
        copied["vertices"] = semantic_vertices
        semantic[stable_id] = copied
    return semantic, hashes


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
    actual_maps, actual_source_hashes = _semantic_parent_owner_rows(
        actual_bindings, "invoked capture")
    current_maps, current_source_hashes = _semantic_parent_owner_rows(
        current_bindings, "current composition")
    _require(isinstance(actual_sources, list) and len(actual_sources) == 7
             and isinstance(current_sources, list) and len(current_sources) == 8
             and set(actual_source_hashes).issubset(set(current_source_hashes))
             and set(current_source_hashes) - set(actual_source_hashes)
             == {OWNER_COMPOSITION_CARRIER_SHA256}
             and _sha_file(OWNER_COMPOSITION_CARRIER)
             == OWNER_COMPOSITION_CARRIER_SHA256,
             "composition must retain the seven source identities and add only its pinned carrier")
    carrier_index = current_source_hashes.index(OWNER_COMPOSITION_CARRIER_SHA256)
    carrier_entry = current_sources[carrier_index]
    _require(Path(carrier_entry.get("path", "")).resolve()
             == OWNER_COMPOSITION_CARRIER.resolve(),
             "the added carrier digest resolves to a different registered path")
    source_index_remap = {
        str(index): current_source_hashes.index(digest)
        for index, digest in enumerate(actual_source_hashes)
    }
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
        "prefix_source_payloads_exactly_preserved": (
            current_source_hashes[:len(actual_source_hashes)] == actual_source_hashes),
        "existing_source_payload_identity_set_preserved": True,
        "source_payload_identity_set_preserved_plus_one_carrier": True,
        "source_payload_index_remap_by_sha256": source_index_remap,
        "source_payload_sha256_in_current_order": current_source_hashes,
        "appended_source_index": carrier_index,
        "added_carrier_source_index": carrier_index,
        "appended_source_sha256": current_source_hashes[carrier_index],
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
            "The available accepted step-0 pack is bound to the prior A779 receipt. "
            "This source-only successor adds the pinned route carrier, updates the 475/477 "
            "owner terms by source-payload identity, rebinds the common-field descriptor, "
            "and restores FJ2217 source face 1858. The full ABI5 comparison is exact "
            "under those declared operations, but no native step-0 pack has emitted this "
            "receipt; regenerate the capture and all-target inventory before predicates."
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
    if pending_edges:
        return "SOURCE0_SEAMOWNER_PREPARED_PENDING_SOURCE_COLLARS_AND_NATIVE_EMISSION"
    if mismatched_faces:
        return "SOURCE0_SEAMOWNER_PREPARED_PENDING_SOURCE_FACE_ORDER_AND_NATIVE_EMISSION"
    if not owner_delta.get("current_capture_candidate_ready"):
        return "SOURCE0_SEAMOWNER_PREPARED_PENDING_FINAL_NATIVE_EMISSION"
    return "SOURCE0_SEAMOWNER_CAPTURE_READY_NO_GEOMETRY_PREDICATE"


def _validated_capture_input_bundle(value: dict) -> dict:
    """Validate an explicit current capture/source receipt bundle.

    The default entrypoint retains the historical A779 preflight unchanged.
    New native captures can supply their own exact pack, pack receipt, invoked
    anatomy receipt, and payload hash; no old capture path is silently rebound.
    """
    expected = {
        "capture_pack_path", "capture_pack_sha256",
        "capture_receipt_path", "capture_receipt_sha256",
        "anatomy_receipt_path", "anatomy_receipt_sha256",
        "anatomy_payload_sha256",
    }
    _require(isinstance(value, dict) and set(value) == expected,
             "explicit current-capture input bundle must contain exactly seven pinned fields")
    normalized = {}
    for name in ("capture_pack_path", "capture_receipt_path", "anatomy_receipt_path"):
        raw = value.get(name)
        _require(isinstance(raw, str) and Path(raw).is_absolute(),
                 f"explicit capture input {name} must be an absolute path")
        path = Path(raw).resolve(strict=True)
        _require(path.is_file(), f"explicit capture input is not a file: {name}")
        normalized[name] = path
    for name in ("capture_pack_sha256", "capture_receipt_sha256", "anatomy_receipt_sha256",
                 "anatomy_payload_sha256"):
        digest = value.get(name)
        _require(isinstance(digest, str) and len(digest) == 64
                 and all(character in "0123456789abcdef" for character in digest),
                 f"explicit capture input {name} must be lowercase SHA-256")
        normalized[name] = digest
    for path_key, hash_key in (
        ("capture_pack_path", "capture_pack_sha256"),
        ("capture_receipt_path", "capture_receipt_sha256"),
        ("anatomy_receipt_path", "anatomy_receipt_sha256"),
    ):
        _require(_sha_file(normalized[path_key]) == normalized[hash_key],
                 f"explicit current-capture input digest mismatch: {path_key}")
    anatomy_receipt = json.loads(normalized["anatomy_receipt_path"].read_text())
    payload = anatomy_receipt.get("payload", {})
    payload_path = Path(payload.get("path", "")).resolve(strict=True)
    _require(payload.get("sha256") == normalized["anatomy_payload_sha256"]
             and _sha_file(payload_path) == normalized["anatomy_payload_sha256"],
             "explicit current-capture source receipt does not bind the supplied NHA payload")
    return {**normalized, "payload_path": payload_path}


def _load_current_capture_context(inputs: dict) -> dict:
    """Reuse the existing capture gate validator with a temporary explicit pin set."""
    inputs = _validated_capture_input_bundle(inputs)
    names = {
        "CAPTURE_PACK": inputs["capture_pack_path"],
        "CAPTURE_PACK_SHA256": inputs["capture_pack_sha256"],
        "CAPTURE_RECEIPT": inputs["capture_receipt_path"],
        "CAPTURE_RECEIPT_SHA256": inputs["capture_receipt_sha256"],
        "ANATOMY_RECEIPT": inputs["anatomy_receipt_path"],
        "ANATOMY_RECEIPT_SHA256": inputs["anatomy_receipt_sha256"],
        # In this mode the only valid successor is the source receipt that the
        # fresh native capture actually invoked.
        "PREPARED_SUCCESSOR_RECEIPT": inputs["anatomy_receipt_path"],
        "PREPARED_SUCCESSOR_RECEIPT_SHA256": inputs["anatomy_receipt_sha256"],
        "ANATOMY_PAYLOAD_SHA256": inputs["anatomy_payload_sha256"],
    }
    old = {name: getattr(capture_gate, name) for name in names}
    try:
        for name, value in names.items():
            setattr(capture_gate, name, value)
        context = capture_gate._load_current_inputs()
    finally:
        for name, value in old.items():
            setattr(capture_gate, name, value)
    context["explicit_capture_input_bundle"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in inputs.items()
    }
    context["capture_source_paths"] = {
        key: inputs[key] for key in (
            "capture_pack_path", "capture_receipt_path", "anatomy_receipt_path",
            "payload_path")
    }
    context["capture_source_sha256"] = {
        "capture_pack": inputs["capture_pack_sha256"],
        "capture_receipt": inputs["capture_receipt_sha256"],
        "anatomy_receipt": inputs["anatomy_receipt_sha256"],
        "anatomy_payload": inputs["anatomy_payload_sha256"],
    }
    return context


def build_preflight(capture_inputs: dict | None = None) -> dict:
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
    explicit_capture = capture_inputs is not None
    if explicit_capture:
        context = _load_current_capture_context(capture_inputs)
        current_receipt = context["anatomy_receipt"]
        erratum = {
            "status": "HISTORICAL_SOURCE_COMPOSITION_ERRATUM_NOT_USED",
            "reason": "current exact-candidate mode consumes only the receipt actually invoked by this native capture",
        }
    else:
        context = capture_gate._load_current_inputs()
        for path, digest in (
            (OWNER_COMPOSITION_RECEIPT, OWNER_COMPOSITION_RECEIPT_SHA256),
            (OWNER_COMPOSITION_REPORT, OWNER_COMPOSITION_REPORT_SHA256),
            (OWNER_REPORT_28FC, OWNER_REPORT_28FC_SHA256),
            (OWNER_REPORT_C37, OWNER_REPORT_C37_SHA256),
            (OWNER_ERRATUM, OWNER_ERRATUM_SHA256),
            (OWNER_MODULE, OWNER_MODULE_SHA256),
            (OWNER_COMPOSER, OWNER_COMPOSER_SHA256),
        ):
            _resolved_pin(path, digest)
        erratum = _validate_owner_report_erratum()
        current_receipt = json.loads(OWNER_COMPOSITION_RECEIPT.read_text())
    payload_transition, current_source_rows = _current_payload_row_transition(
        context, current_receipt)
    if payload_transition.get("native_emitted_current_candidate") is True:
        owner_delta = _captured_candidate_owner_binding_delta(
            context["anatomy_receipt"], current_receipt)
    else:
        _require(not explicit_capture,
                 "explicit source candidate must be the exact source receipt invoked by the accepted capture")
        owner_delta = _owner_composition_delta(
            context["anatomy_receipt"], current_receipt, payload_transition)
    source_baselines, baseline_records = _registered_rigid_source_surfaces(
        context, current_source_rows, current_receipt["provenance"]["source_id_map"])
    parent_motion_inventory = _parent_motion_owner_inventory(
        current_receipt, set(NETWORK_STABLE_IDS))
    parent_pose_surfaces, parent_pose_report = _parent_pose_source_surfaces(
        context, current_source_rows, current_receipt)
    _require(set(parent_pose_surfaces) == {
        (51011, stable_id) for stable_id in PARENT_POSE_ROWS
    }, "independent parent-pose baseline does not cover rows 475 and 477")
    common_field_surfaces, common_field_report, common_field_input_pins = (
        _common_field_source_surfaces(
            context, current_source_rows,
            current_receipt["provenance"]["source_id_map"], current_receipt))
    _require(set(common_field_surfaces) == {
        (51011, stable_id) for stable_id in COMMON_FIELD_BASELINE_ROWS
    }, "existing common-field source baseline does not cover all five mapped network rows")
    _require(set(source_baselines) == {
        (51011, stable_id) for stable_id in NETWORK_STABLE_IDS
    }, "source-rigid baseline does not cover the exact 17 typed network rows")
    for key, surface in source_baselines.items():
        _require(
            _sha_bytes(np.ascontiguousarray(surface[0], dtype="<f4").tobytes())
            == baseline_records[str(key[1])]["rigid_source_world_xyz_f32_sha256"],
            f"source-rigid position bytes differ from the recorded baseline for {key}",
        )
    ancestry_context = {
        **context,
        "rows": current_source_rows,
        "source_map": current_receipt["provenance"]["source_id_map"],
    }
    edge_evidence, separate_evidence = _edge_ancestry(ancestry_context)
    pending_edges = separate_evidence["pending_source_collar_edges"]
    pending_source_collar_search = (
        _missing_source_collar_search(pending_edges) if pending_edges else {
            "status": "ALL_NETWORK_SOURCE_COLLARS_HAVE_PINNED_ANCESTRY",
            "searched": False,
            "geometry_predicates_run": False,
        })
    code_pins = capture_gate._current_package_pins()
    code_pins[str(Path(__file__).resolve())] = _sha_file(Path(__file__).resolve())
    code_pins[str(TRANSFORM_OWNER.resolve())] = TRANSFORM_OWNER_SHA256
    code_pins[str(PARENT_POSE_FORWARD_OWNER.resolve())] = (
        PARENT_POSE_FORWARD_OWNER_SHA256)
    code_pins[str(Path(shared_verifier.__file__).resolve())] = _sha_file(
        Path(shared_verifier.__file__).resolve())
    ulnar_proof = separate_evidence["ulnar_source_proof"]
    source_history_paths = (
        ULNAR_SOURCE_PROOF, ULNAR_SOURCE_PREFLIGHT,
        ULNAR_PROOF_BASE_RECEIPT, ULNAR_PROOF_BASE_PAYLOAD,
        METADATA_PREFLIGHT, ROUTE100_REPORT, PORT086, PORT087, PORT088, PORT089,
        ROUTE073_REPORT, BRIDGE_ANCESTRY_REPORT, BRIDGE_ATTRIBUTION_REPORT,
        BRIDGE_REPLAY_REPORT, BRIDGE_STATIONARITY_REPORT,
        source007.SOURCE_ARCHIVE, source007.REGISTRATION, source007.RIGID,
        source007.MANIFOLD_SO, source007.MANIFOLD_RECORD,
        source007.MANIFOLD_METADATA, source007.MANIFOLD_WHEEL,
        PARENT_POSE_BASELINE_RECEIPT, PARENT_POSE_BASELINE_PAYLOAD,
        PARENT_POSE_BASELINE_CARRIER, PARENT_POSE_FORWARD_OWNER,
    )
    source_history_pins = {
        str(METADATA_PREFLIGHT.resolve()): METADATA_PREFLIGHT_SHA256,
        str(ULNAR_SOURCE_PROOF.resolve()): ULNAR_SOURCE_PROOF_SHA256,
        str(ULNAR_SOURCE_PREFLIGHT.resolve()): ULNAR_SOURCE_PREFLIGHT_SHA256,
        str(ULNAR_PROOF_BASE_RECEIPT.resolve()): ULNAR_PROOF_BASE_RECEIPT_SHA256,
        str(ULNAR_PROOF_BASE_PAYLOAD.resolve()): ULNAR_PROOF_BASE_PAYLOAD_SHA256,
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
        str(PARENT_POSE_BASELINE_RECEIPT.resolve()):
            PARENT_POSE_BASELINE_RECEIPT_SHA256,
        str(PARENT_POSE_BASELINE_PAYLOAD.resolve()):
            PARENT_POSE_BASELINE_PAYLOAD_SHA256,
        str(PARENT_POSE_BASELINE_CARRIER.resolve()):
            PARENT_POSE_BASELINE_CARRIER_SHA256,
        str(PARENT_POSE_FORWARD_OWNER.resolve()):
            PARENT_POSE_FORWARD_OWNER_SHA256,
    }
    if explicit_capture:
        source_paths = context["capture_source_paths"]
        source_hashes = context["capture_source_sha256"]
        input_paths = (
            source_paths["capture_pack_path"], source_paths["capture_receipt_path"],
            source_paths["anatomy_receipt_path"], source_paths["payload_path"],
            *source_history_paths,
            *(Path(path) for path in ulnar_proof["proof_recorded_data_input_sha256"]),
        )
        expected_inputs = {
            str(source_paths["capture_pack_path"].resolve()): source_hashes["capture_pack"],
            str(source_paths["capture_receipt_path"].resolve()): source_hashes["capture_receipt"],
            str(source_paths["anatomy_receipt_path"].resolve()): source_hashes["anatomy_receipt"],
            str(source_paths["payload_path"].resolve()): source_hashes["anatomy_payload"],
            **source_history_pins,
            **ulnar_proof["proof_recorded_data_input_sha256"],
        }
    else:
        input_paths = (
            capture_gate.CAPTURE_PACK, capture_gate.CAPTURE_RECEIPT,
            *source_history_paths,
            *(Path(path) for path in ulnar_proof["proof_recorded_data_input_sha256"]),
            capture_gate.ANATOMY_RECEIPT, capture_gate.PREPARED_SUCCESSOR_RECEIPT,
            OWNER_COMPOSITION_RECEIPT, OWNER_COMPOSITION_REPORT, OWNER_COMPOSITION_CARRIER,
            OWNER_COMPOSITION_PAYLOAD_PATH, OWNER_REPORT_28FC, OWNER_REPORT_C37,
            OWNER_ERRATUM, OWNER_MODULE, OWNER_COMPOSER, context["payload_path"],
        )
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
            **source_history_pins,
            **ulnar_proof["proof_recorded_data_input_sha256"],
        }
    input_pins = {
        str(Path(path).resolve(strict=True)): _sha_file(Path(path).resolve(strict=True))
        for path in input_paths
    }
    for path, expected in common_field_input_pins.items():
        _require(input_pins.get(path, expected) == expected,
                 f"common-field source input changed after reconstruction: {path}")
        input_pins[path] = expected
        expected_inputs[path] = expected
    for path, expected in expected_inputs.items():
        _require(input_pins.get(path) == expected,
                 f"pinned source/capture witness input changed: {path}")
    capture = context["capture_identity"]
    expected_pack_sha = (context["capture_source_sha256"]["capture_pack"]
                         if explicit_capture else capture_gate.CAPTURE_PACK_SHA256)
    expected_capture_receipt_sha = (
        context["capture_source_sha256"]["capture_receipt"]
        if explicit_capture else capture_gate.CAPTURE_RECEIPT_SHA256)
    capture_paths = context.get("capture_source_paths", {})
    actual_pack_path = (capture_paths.get("capture_pack_path", capture_gate.CAPTURE_PACK)
                        if explicit_capture else capture_gate.CAPTURE_PACK)
    actual_capture_receipt_path = (
        capture_paths.get("capture_receipt_path", capture_gate.CAPTURE_RECEIPT)
        if explicit_capture else capture_gate.CAPTURE_RECEIPT)
    actual_anatomy_receipt_path = (
        capture_paths.get("anatomy_receipt_path", capture_gate.ANATOMY_RECEIPT)
        if explicit_capture else capture_gate.ANATOMY_RECEIPT)
    actual_anatomy_receipt_sha = (
        context["capture_source_sha256"]["anatomy_receipt"]
        if explicit_capture else capture_gate.ANATOMY_RECEIPT_SHA256)
    current_payload_hash = current_receipt["payload"]["sha256"]
    current_payload_path = context["payload_path"]
    _require(capture.get("pack_sha256") == expected_pack_sha
             and capture.get("capture_receipt_sha256") == expected_capture_receipt_sha
             and capture.get("accepted_step") == 0,
             "actual current capture identity changed")
    mismatched_faces = [
        sid for sid, row in baseline_records.items()
        if row["capture_face_order_equals_nha"] is not True
    ]
    all_known_edges_bound = all(
        edge_evidence[f"{first[1]}->{second[1]}"].get("pairs") is not None
        for first, second in NETWORK_EDGES)
    parent_mapped_ids = {
        int(stable_id) for stable_id, row
        in parent_motion_inventory["rows"].items()
        if row["parent_map_present"]
    }
    parent_pose_ids = set(map(int, parent_pose_report["vessels"]))
    parent_pose_capture_aligned = (
        parent_pose_ids == parent_mapped_ids
        and all(row["capture_aligned_current_surface"]
                for row in parent_pose_report["vessels"].values()))
    common_field_capture_aligned = all(
        row["capture_vertex_ancestry"]["status"]
        == "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED"
        and row["source_vertex_count"] == row["common_map_range"]["vertex_count"]
        for row in common_field_report["rows"].values())
    rigid_only_network_rows = sorted(
        set(NETWORK_STABLE_IDS)
        - parent_mapped_ids
        - set(COMMON_FIELD_BASELINE_ROWS))
    all_network_owner_baselines_ready = (
        set(map(int, parent_pose_report["vessels"])) == parent_mapped_ids
        and set(common_field_report["rows"]) == set(map(
            str, COMMON_FIELD_BASELINE_ROWS))
        and parent_pose_capture_aligned
        and common_field_capture_aligned
        and len(source_baselines) == len(NETWORK_STABLE_IDS))
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
            "pack_path": str(actual_pack_path.resolve()),
            "pack_sha256": capture.get("pack_sha256"),
            "receipt_path": str(actual_capture_receipt_path.resolve()),
            "receipt_sha256": capture.get("capture_receipt_sha256"),
            "accepted_step": capture.get("accepted_step"),
            "accepted_time_s": capture.get("accepted_time_s"),
            "typed_surface_count": len(context["surfaces"]),
            "capture_receipt_sha256": expected_capture_receipt_sha,
            "body_pose_count": len(context["capture_receipt"]["accepted_body_poses"]),
            "common_field_source_anatomy_payload_sha256":
                context["capture_receipt"].get("common_field_source_anatomy_payload_sha256"),
            "invoked_anatomy_receipt_path": str(actual_anatomy_receipt_path.resolve()),
            "invoked_anatomy_receipt_sha256": actual_anatomy_receipt_sha,
            "explicit_merged_capture_input_bundle": context.get(
                "explicit_capture_input_bundle"),
            "historical_a779_capture_is_current": not explicit_capture,
        },
        ("captured_candidate_source_receipt" if explicit_capture
         else "source_only_current_seamowner_composition"): {
            **owner_delta,
            "report_version_erratum": erratum,
            "source_only_composition_report_path": (
                None if explicit_capture else str(OWNER_COMPOSITION_REPORT.resolve())),
            "source_only_composition_report_sha256": (
                None if explicit_capture else OWNER_COMPOSITION_REPORT_SHA256),
            "erratum_path": None if explicit_capture else str(OWNER_ERRATUM.resolve()),
            "erratum_sha256": None if explicit_capture else OWNER_ERRATUM_SHA256,
            "recovered_28fc_report_sha256": (
                None if explicit_capture else OWNER_REPORT_28FC_SHA256),
            "current_c37_report_sha256": (
                None if explicit_capture else OWNER_REPORT_C37_SHA256),
        },
        "source_bound_rigid_rest_baseline": {
            "source_nha_payload_sha256": current_payload_hash,
            "source_nha_payload_path": str(current_payload_path.resolve()),
            "payload_is_current_capture_source": (
                current_payload_hash
                == context["capture_receipt"].get(
                    "common_field_source_anatomy_payload_sha256")),
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
            "independent_parent_pose_source_surfaces": {
                str(stable_id): {
                    "typed_identity": [51011, stable_id],
                    "positions_world_f32_sha256": _sha_bytes(
                        np.ascontiguousarray(
                            parent_pose_surfaces[(51011, stable_id)][0],
                            dtype="<f4").tobytes()),
                    "faces_u32_sha256": _sha_bytes(
                        np.ascontiguousarray(
                            parent_pose_surfaces[(51011, stable_id)][1],
                            dtype="<u4").tobytes()),
                }
                for stable_id in PARENT_POSE_ROWS
            },
            "independent_parent_pose_calibre_baseline": parent_pose_report,
            "parent_mapped_row_calibre_baseline_ready":
                parent_pose_capture_aligned,
            "existing_common_field_owner_source_baseline": common_field_report,
            "common_field_owner_source_baseline_ready": common_field_capture_aligned,
            "common_field_rows_still_pending_independent_baseline": [],
            "rigid_only_network_row_ids": rigid_only_network_rows,
            "rows_with_exact_capture_face_order": len(baseline_records) - len(mismatched_faces),
            "rows_with_face_order_mismatch": mismatched_faces,
            "meaningful_calibre_baseline_ready": (
                all_known_edges_bound and not mismatched_faces
                and all_network_owner_baselines_ready),
            "network_owner_baseline_coverage_complete":
                all_network_owner_baselines_ready,
            "network_calibre_gate_passed": False,
            "parent_motion_owner_inventory": parent_motion_inventory,
            "parent_mapped_rows_requiring_a_parent_pose_calibre_baseline": [
                int(stable_id) for stable_id, row
                in parent_motion_inventory["rows"].items()
                if row["parent_map_present"]
            ],
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
            "ulnar_source_junction_proof": separate_evidence["ulnar_source_proof"],
            "pending_source_collar_search": pending_source_collar_search,
            "pending_collar_identity": {
                edge: {
                    "parent": _source_identity_summary(
                        current_receipt["provenance"]["source_id_map"], int(edge.split("->")[0])),
                    "child": _source_identity_summary(
                        current_receipt["provenance"]["source_id_map"], int(edge.split("->")[1])),
                    "reason": "no pinned exact source-face collar ancestry is available",
                }
                for edge in separate_evidence["pending_source_collar_edges"]
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
            "Explicit merged-capture broadphase: each of the 17 typed vessel/network "
            "rows is compared against every surface decoded from the exact accepted "
            "pack and invoked receipt in the supplied input bundle."
            if explicit_capture else
            "Historical A779 capture broadphase only: each of the 17 typed vessel/network "
            "rows is compared against every surface decoded from that accepted pack. "
            "The pack predates source payload 7 and changed 475/477 parent terms, so "
            "regenerate this exact inventory from a fresh merged native step-0 pack "
            "before any geometry predicate."
        ),
        "current_verifier": {
            "path": str(Path(shared_verifier.__file__).resolve()),
            "sha256": code_pins[str(Path(shared_verifier.__file__).resolve())],
            "historical_source007_P6_imported": False,
        },
        "source_and_parent_motion_ownership": {
            "status": (
                "PARENT_POSE_SOURCE_BASELINES_COMPUTED_CAPTURE_ALIGNMENT_PENDING"
                if not parent_pose_capture_aligned
                else "SOURCE_OWNER_BASELINES_READY_FOR_GEOMETRY_REVIEW"
                if all_network_owner_baselines_ready
                else "SOURCE_OWNER_BASELINE_CAPTURE_ALIGNMENT_PENDING"),
            "owner_inventory": parent_motion_inventory,
            "independent_parent_pose_baseline": parent_pose_report,
            "parent_pose_source_surfaces_available": sorted(
                [list(key) for key in parent_pose_surfaces]),
            "parent_mapped_rows_with_independent_baselines": sorted(
                int(value) for value in parent_pose_report["vessels"]),
            "existing_common_field_owner_source_baseline": common_field_report,
            "common_field_rows_still_pending_independent_baseline": [],
            "source_rigid_baseline_is_independent_of_same_capture_geometry": True,
            "parent_pose_matched_calibre_baseline_ready":
                parent_pose_capture_aligned,
            "common_field_source_baseline_ready": common_field_capture_aligned,
            "rigid_only_network_rows": rigid_only_network_rows,
            "whole_network_calibre_baseline_ready":
                all_network_owner_baselines_ready,
            "whole_network_calibre_gate_passed": False,
            "no_unverified_multi_body_parent_motion_is_claimed": True,
        },
        "planned_solid_and_collar_gate": {
            "gate_owner": "current shared verifier plus existing brachiocephalic 3D port-continuity owner",
            "shared_verifier_path": str(Path(shared_verifier.__file__).resolve()),
            "shared_verifier_sha256": _sha_file(Path(shared_verifier.__file__).resolve()),
            "execution_prerequisites": [
                "final merged receipt retains the exact 8-source bound and is the receipt actually invoked by a fresh native step-0 capture",
                "all 15 graph links have source-registered exact face-patch ancestry, including 479->487 and 481->489",
                "candidate and source-rigid baseline rows bind exact ABI5 source face/vertex ancestry",
                "every parent-mapped vessel has a parent-pose-matched independent source station baseline",
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


def run_preflight(capture_inputs: dict | None = None,
                  output_dir: Path | None = None) -> dict:
    if capture_inputs is not None:
        _require(output_dir is not None,
                 "explicit current-capture inputs require a fresh output directory")
    destination = OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    _require(not destination.exists(),
             "preserve existing outputs; use a fresh named output directory")
    started = time.perf_counter()
    preflight = build_preflight(capture_inputs)
    destination.mkdir(parents=True, exist_ok=False)
    preflight_path = destination / "source0-seamowner-preflight.json"
    _write_json(preflight_path, preflight)
    return {
        "status": preflight["status"],
        "preflight_path": str(preflight_path.resolve()),
        "preflight_sha256": _sha_file(preflight_path),
        "elapsed_seconds": time.perf_counter() - started,
        "geometry_predicates_run": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--capture-inputs-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    capture_inputs = (None if args.capture_inputs_json is None
                      else json.loads(args.capture_inputs_json.read_text()))
    print(json.dumps(run_preflight(capture_inputs, args.output_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
