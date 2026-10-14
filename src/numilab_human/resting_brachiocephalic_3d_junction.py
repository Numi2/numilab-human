"""Source-bound 3D continuity proof for the right brachiocephalic junction.

This offline geometry screen consumes retained source007 ABI5 rows and the
current shared Human verifier from this package. Historical HP98 source007
receipts remain provenance only. It does not modify meshes or simulate blood
state.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import resting_anatomy_verify as current_shared_verifier
from . import (
    resting_brachiocephalic_current_capture_network_preparation as network_preparation,
)
from . import resting_brachiocephalic_geometry_gate as gate
from . import resting_brachiocephalic_port_diagnostic as portdiag
from . import resting_brachiocephalic_source007 as source007
from . import resting_pleura_proxy as nhanat_codec

A = source007.A
GATE_DIR = source007.GATE_DIR
SOURCE004_REPORT = portdiag.SIX_DIAGNOSTIC_DIR / "six-port-section-diagnostic.json"
SOURCE004_PREFLIGHT = portdiag.SIX_PREFLIGHT_PATH
SOURCE004_REPORT_SHA256 = "354876d040f774d8acf18071779289ee9cfc2ce50137650087ecf2f099399700"
SOURCE004_PREFLIGHT_SHA256 = "9d32d5a707ea4ba6444fd680e4fc9256473c486c84c3f9a17def56b392e70d5e"
FIRST_EXECUTION_DIR = GATE_DIR / "port-526-472-3d-junction-diagnostic-001"
FIRST_EXECUTION_REPORT = FIRST_EXECUTION_DIR / "3d-junction-continuity-report.json"
FIRST_EXECUTION_REPORT_SHA256 = "25fbacbf5546a944007287ee7ada8e5ae34dd9bbd7e1a45b7313a894ffb404ab"
FIRST_EXECUTION_PREFLIGHT = GATE_DIR / "port-526-472-3d-junction-preflight-002.json"
FIRST_EXECUTION_PREFLIGHT_SHA256 = "5a7b2af5735a7c1d8808a78455ee5362844466e7079e669bd1d5d922ef3a1669"
OUTPUT_DIR = GATE_DIR / "port-526-472-3d-junction-diagnostic-003"
INITIALIZATION_DRYRUN_REPORT = GATE_DIR / "port-526-472-3d-junction-initialization-dryrun-002.json"
PREVIOUS_PREFLIGHT_PATH = GATE_DIR / "port-526-472-3d-junction-preflight-001.json"
PREVIOUS_PREFLIGHT_SHA256 = "48d5b21121eab978f0f3e8165d2afd4197c17bd4d8c5ba317b523dff8dc39179"
PREFLIGHT_PATH = GATE_DIR / "port-526-472-3d-junction-preflight-004.json"
SOURCE_INTEGRITY_DIR = GATE_DIR / "port-526-472-source-integrity-preparation-003"
SOURCE_INTEGRITY_REPORT = SOURCE_INTEGRITY_DIR / "source-integrity-preparation.json"
SOURCE_INTEGRITY_SCRIPT = SOURCE_INTEGRITY_DIR / "prepare-source-integrity.py"
SOURCE_INTEGRITY_REPORT_SHA256 = "34e540ec56ec8fd09b1fe94ecc42f6a5edced055f0e007874000973442bdcdda"
SOURCE_INTEGRITY_SCRIPT_SHA256 = "f5361a110afa93f55750a46e6b8fb8497c39e1f56afedffa90b8734067c2de78"
REPORT_PATH = OUTPUT_DIR / "3d-junction-continuity-report.json"
PROGRESS_PATH = OUTPUT_DIR / "progress.json"
BRIDGE_KEY = (51011, 526)
TARGET_KEY = (51011, 472)
PAIR = (BRIDGE_KEY, TARGET_KEY)
# Legacy entry point remains restricted to the frozen FJ3583/right-subclavian
# test. The network entry point below consumes the canonical source-preparation
# owner's exact 13 branch edges plus two SVC ports.
PARTIAL_OVERLAP_PORT = PAIR
VOLUME_BALANCE_RELATIVE_TOLERANCE = 1e-6


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("brachiocephalic 3D junction: " + message)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def declared_network_links() -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Return the canonical 13 route edges and two SVC ports from their owner."""
    links = (*network_preparation.NETWORK_EDGES,
             *network_preparation.SVC_PORTS)
    _require(len(links) == 15 and len(set(links)) == 15,
             "canonical source graph must contain exactly 15 unique directed links")
    _require(all(
        isinstance(edge, (tuple, list)) and len(edge) == 2
        and all(isinstance(identity, (tuple, list)) and len(identity) == 2
                and type(identity[0]) is int and type(identity[1]) is int
                and identity[0] == 51011 and identity[1] > 0
                for identity in edge)
        and edge[0] != edge[1]
        for edge in links),
        "canonical source graph contains malformed typed surface identities")
    _require(((51011, 469), (51011, 473)) not in links
             and ((51011, 473), (51011, 469)) not in links,
             "separate 469/473 overlap must not be promoted to a route edge")
    return links


def _strict_source_link_ids(evidence: dict) -> tuple[int, int]:
    if "parent_stable_id" in evidence or "child_stable_id" in evidence:
        parent, child = evidence.get("parent_stable_id"), evidence.get("child_stable_id")
    else:
        parent, child = evidence.get("bridge_stable_id"), evidence.get("target_stable_id")
    _require(type(parent) is int and type(child) is int,
             "source collar ancestry must carry exact integer parent/child IDs")
    return parent, child


def _source_face_array(source_rows: Mapping, stable_id: int) -> np.ndarray:
    row = source_rows.get(stable_id, source_rows.get(str(stable_id)))
    _require(isinstance(row, dict) and "faces" in row,
             f"source row {stable_id} has no ABI5 face array")
    faces = np.asarray(row["faces"])
    _require(faces.ndim == 2 and faces.shape[1] == 3
             and faces.dtype.kind in "iu" and len(faces) > 0
             and int(faces.min()) >= 0,
             f"source row {stable_id} has malformed indexed faces")
    return np.ascontiguousarray(faces, dtype="<u4")


def bind_network_link_source_ancestry(
    bridge_key, target_key, *, source_link_evidence: dict,
    source_rows: Mapping, source_map: Mapping,
) -> dict:
    """Bind one declared edge to exact typed rows and frozen source face ancestry."""
    bridge_key, target_key = gate._identity(bridge_key), gate._identity(target_key)
    edge = (bridge_key, target_key)
    _require(edge in set(declared_network_links()),
             f"undeclared vascular link {bridge_key[1]}->{target_key[1]}")
    _require(isinstance(source_link_evidence, dict)
             and isinstance(source_rows, Mapping)
             and isinstance(source_map, Mapping),
             "source link evidence, ABI5 rows, and receipt map must be mappings")
    evidence_status = source_link_evidence.get("evidence_status")
    _require(isinstance(evidence_status, str) and bool(evidence_status)
             and evidence_status != "SOURCE_BOUND_COLLAR_DERIVATION_REQUIRED",
             "source link ancestry is pending or has no frozen evidence status")
    parent, child = _strict_source_link_ids(source_link_evidence)
    _require(edge == ((51011, parent), (51011, child)),
             "typed link IDs disagree with the frozen source-face ancestry IDs")
    path = source_link_evidence.get("path")
    report_sha = source_link_evidence.get("sha256")
    _require(isinstance(path, str) and path
             and isinstance(report_sha, str) and len(report_sha) == 64
             and all(character in "0123456789abcdef" for character in report_sha),
             "source ancestry must name its frozen report path and SHA-256")
    raw_pairs = np.asarray(source_link_evidence.get("pairs"))
    _require(raw_pairs.ndim == 2 and raw_pairs.shape[1] == 2
             and raw_pairs.dtype.kind in "iu" and len(raw_pairs) > 0
             and int(raw_pairs.min()) >= 0,
             "source ancestry must contain nonempty exact integer face pairs")
    pairs = np.ascontiguousarray(raw_pairs, dtype="<i8")
    _require(len(set(map(tuple, pairs.tolist()))) == len(pairs),
             "source face-pair ancestry contains duplicate pairs")
    pair_sha = _sha_bytes(pairs.tobytes())
    _require(source_link_evidence.get("pairs_sha256") == pair_sha,
             "frozen source face-pair array SHA-256 does not match its ancestry row")

    parent_faces = _source_face_array(source_rows, parent)
    child_faces = _source_face_array(source_rows, child)
    parent_faces_sha = _sha_bytes(parent_faces.tobytes())
    child_faces_sha = _sha_bytes(child_faces.tobytes())
    _require(source_link_evidence.get("source_parent_faces_sha256")
             == parent_faces_sha
             and source_link_evidence.get("source_child_faces_sha256")
             == child_faces_sha,
             "frozen source ancestry is not bound to the current ABI5 parent/child face arrays")
    _require(int(pairs[:, 0].max()) < len(parent_faces)
             and int(pairs[:, 1].max()) < len(child_faces),
             "frozen source face-pair ancestry is outside its typed source rows")

    identities = {}
    for stable_id in (parent, child):
        identity = source_map.get(str(stable_id))
        _require(isinstance(identity, dict)
                 and isinstance(identity.get("source_member"), str)
                 and bool(identity["source_member"])
                 and type(identity.get("body_index")) is int
                 and identity["body_index"] >= 0
                 and type(identity.get("layer")) is int
                 and identity["layer"] > 0,
                 f"receipt source identity for stable ID {stable_id} is incomplete")
        compartment = identity.get("vascular_compartment_id")
        _require(compartment is None or (type(compartment) is int and compartment > 0),
                 f"receipt vascular compartment for stable ID {stable_id} is malformed")
        owner_metadata = identity.get("source_owner_metadata")
        identities[str(stable_id)] = {
            "typed_identity": [51011, stable_id],
            "source_member": identity["source_member"],
            "source_name": identity.get("name"),
            "source_concept_id": (
                owner_metadata.get("concept_id")
                if isinstance(owner_metadata, dict) else None),
            "body_index": identity["body_index"],
            "layer": identity["layer"],
            "vascular_compartment_id": compartment,
        }
    return {
        "typed_link": [list(bridge_key), list(target_key)],
        "link_id": f"{parent}->{child}",
        "source_evidence_path": path,
        "source_evidence_sha256": report_sha,
        "source_evidence_status": source_link_evidence.get("evidence_status"),
        "source_parent_faces_sha256": parent_faces_sha,
        "source_child_faces_sha256": child_faces_sha,
        "source_face_pair_ancestry_sha256": pair_sha,
        "source_face_pair_count": len(pairs),
        "source_face_pair_ancestry": pairs.tolist(),
        "source_identities": identities,
        "source_local_face_indices_bound": True,
        "clinical_or_anatomical_admission": False,
    }


def _network_edge_name(edge) -> str:
    return f"{edge[0][1]}->{edge[1][1]}"


def prepare_network_link_audit_inputs(
    edge_evidence: Mapping, separate_evidence: Mapping, *,
    source_rows: Mapping, source_map: Mapping,
    candidate_surfaces: Mapping, baseline_surfaces: Mapping,
) -> tuple[dict, ...]:
    """Bind existing source/capture owners into the exact 15-link 3D input list."""
    links = declared_network_links()
    expected_edges = {
        _network_edge_name(edge): edge
        for edge in network_preparation.NETWORK_EDGES
    }
    _require(isinstance(edge_evidence, Mapping)
             and set(edge_evidence) == set(expected_edges),
             "source edge evidence must cover exactly the 13 declared route edges")
    svc_evidence = separate_evidence.get("svc_bridge_ports")
    expected_svc = {
        _network_edge_name(edge): edge
        for edge in network_preparation.SVC_PORTS
    }
    _require(isinstance(svc_evidence, Mapping)
             and set(svc_evidence) == set(expected_svc),
             "source interface evidence must cover exactly the two declared SVC ports")
    required_surface_keys = {
        key for edge in links for key in edge
    }
    _require(isinstance(candidate_surfaces, Mapping)
             and required_surface_keys.issubset(candidate_surfaces)
             and isinstance(baseline_surfaces, Mapping)
             and required_surface_keys.issubset(baseline_surfaces),
             "candidate or independent source-baseline surface mapping lacks a network row")
    output = []
    for edge in links:
        name = _network_edge_name(edge)
        evidence = (edge_evidence[name] if name in expected_edges
                    else svc_evidence[name])
        bind_network_link_source_ancestry(
            edge[0], edge[1], source_link_evidence=evidence,
            source_rows=source_rows, source_map=source_map)
        output.append({
            "bridge_key": edge[0],
            "target_key": edge[1],
            "bridge_surface": candidate_surfaces[edge[0]],
            "target_surface": candidate_surfaces[edge[1]],
            "source_link_evidence": evidence,
            "source_rows": source_rows,
            "source_map": source_map,
            "baseline_bridge_surface": baseline_surfaces[edge[0]],
            "baseline_target_surface": baseline_surfaces[edge[1]],
        })
    _require(len(output) == 15
             and tuple((item["bridge_key"], item["target_key"]) for item in output)
             == links,
             "prepared source network inputs differ from the canonical 15-link order")
    return tuple(output)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state() -> dict:
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True, capture_output=True, text=True).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status_lines": status}


def _jsonable(value):
    return portdiag._jsonable(value)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                     ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _source004_plane_failure(report: dict) -> dict:
    """Summarize the retained 004 failure without re-running its plane cuts."""
    calls = {
        int(item["call_index"]): item
        for item in report.get("trace", {}).get("section_calls", [])
        if item.get("port_identity") == {
            "bridge": list(BRIDGE_KEY), "target": list(TARGET_KEY)}
    }
    candidates = []
    for ordinal, (fraction, base) in enumerate(
            zip((0.2, 0.4, 0.6, 0.8), (8, 14, 20, 26), strict=True), start=1):
        bridge = calls.get(base)
        target = calls.get(base + 1)
        union = calls.get(base + 4)
        _require(bridge is not None and target is not None and union is not None,
                 "pinned source004 section witness set is incomplete")
        bridge_metrics = bridge["section_metrics"]
        target_metrics = target["section_metrics"]
        union_metrics = union["section_metrics"]
        _require(bridge["positions_dtype"] == "float32"
                 and target["positions_dtype"] == "float32",
                 "source004 direct-source plane witness is not native F32 geometry")
        bridge_loops = bridge_metrics.get("loop_metrics", [])
        target_loops = target_metrics.get("loop_metrics", [])
        candidates.append({
            "candidate_ordinal": ordinal,
            "candidate_fraction_from_pinned_004_policy": fraction,
            "section_plane_signature_sha256": bridge["section_plane_signature_sha256"],
            "center_world_m": bridge["center_world_m"],
            "tangent_world_unit": bridge["tangent_world_unit"],
            "bridge526_source_section": {
                "call_index": int(bridge["call_index"]),
                "coordinate_dtype": bridge["positions_dtype"],
                "surface_positions_sha256": bridge["positions_sha256"],
                "source_faces_sha256": bridge["faces_sha256"],
                "loop_count": int(bridge_metrics.get("loop_count", -1)),
                "segment_count": int(bridge_metrics.get("segment_count", -1)),
                "node_count": int(bridge_metrics.get("section_node_count", -1)),
                "degree_histogram": bridge_metrics.get("section_degree_histogram", {}),
                "degree_not_two_node_count": sum(
                    int(count) for degree, count in
                    bridge_metrics.get("section_degree_histogram", {}).items()
                    if int(degree) != 2),
                "loop_face_ancestry": [{
                    "owner_ids": loop.get("owner_ids", []),
                    "source_face_ancestor_ids": loop.get("source_face_ancestor_ids", []),
                    "source_face_ancestor_ids_sha256": loop.get(
                        "source_face_ancestor_ids_sha256"),
                    "area_mm2": loop.get("area_mm2"),
                    "minimum_feret_width_mm": loop.get("minimum_feret_width_mm"),
                } for loop in bridge_loops],
                "edge_ids_retained_in_source004": False,
            },
            "target472_source_section": {
                "call_index": int(target["call_index"]),
                "coordinate_dtype": target["positions_dtype"],
                "surface_positions_sha256": target["positions_sha256"],
                "source_faces_sha256": target["faces_sha256"],
                "loop_count": int(target_metrics.get("loop_count", -1)),
                "segment_count": int(target_metrics.get("segment_count", -1)),
                "node_count": int(target_metrics.get("section_node_count", -1)),
                "degree_histogram": target_metrics.get("section_degree_histogram", {}),
                "loop_face_ancestry": [{
                    "owner_ids": loop.get("owner_ids", []),
                    "source_face_ancestor_ids": loop.get("source_face_ancestor_ids", []),
                    "source_face_ancestor_ids_sha256": loop.get(
                        "source_face_ancestor_ids_sha256"),
                } for loop in target_loops],
            },
            "floating_union_section": {
                "call_index": int(union["call_index"]),
                "coordinate_dtype": union["positions_dtype"],
                "mesh_face_sha256": union["faces_sha256"],
                "loop_count": int(union_metrics.get("loop_count", -1)),
                "degree_histogram": union_metrics.get("section_degree_histogram", {}),
                "loop_face_ancestry": [{
                    "owner_ids": loop.get("owner_ids", []),
                    "source_face_ancestor_ids": loop.get("source_face_ancestor_ids", []),
                    "source_face_ancestor_ids_sha256": loop.get(
                        "source_face_ancestor_ids_sha256"),
                } for loop in union_metrics.get("loop_metrics", [])],
                "precision_limitation": "floating derived CSG section, not exact topology proof",
            },
        })
    _require([item["bridge526_source_section"]["loop_count"] for item in candidates]
             == [2, 0, 0, 0],
             "pinned source004 526 section failure pattern changed")
    _require(all(item["target472_source_section"]["loop_count"] == 1
                 for item in candidates),
             "pinned source004 target 472 section failure pattern changed")
    _require(candidates[0]["bridge526_source_section"]["degree_not_two_node_count"] == 0,
             "source004's first source-526 multi-loop witness has unexpected branching")
    return {
        "report_path": str(SOURCE004_REPORT),
        "report_sha256": SOURCE004_REPORT_SHA256,
        "diagnosis": "target_tangent_plane_multi_loop_then_bridge_absent",
        "candidate_count": len(candidates),
        "candidate_planes": candidates,
        "source_section_graph_branching_proven": False,
        "reason_shared_planar_section_uniqueness_is_not_a_valid_construction_premise": (
            "The first target-tangent plane produces two disjoint closed 526 loops; "
            "the other three candidate planes produce no 526 section at all while "
            "472 has one. Retained degree histograms show no degree-not-two node "
            "for the first multi-loop witness. This is a plane/station mismatch at "
            "a curved junction, not evidence of a broken closed 3D source vessel."),
        "source_edge_witness_limitation": (
            "004 retained segment/node counts and exact source-face ancestry, but "
            "not the individual section edge IDs."),
    }


def _verify_source007_compiler_code_pins(code_and_dependency_pins: dict) -> dict:
    """Verify the historical owner sources that emitted source007, not today's tree."""
    source_repo = code_and_dependency_pins.get("source_repository", {})
    revision = source_repo.get("head")
    _require(isinstance(revision, str) and len(revision) == 40
             and source_repo.get("dirty") is False,
             "source007 compiler source revision is not a clean frozen commit")
    source_hashes = code_and_dependency_pins.get("source_code_sha256", {})
    _require(isinstance(source_hashes, dict) and source_hashes,
             "source007 compiler owner code hash set is missing")
    repo = Path(source007.__file__).resolve().parents[2]
    verified = {}
    for raw_path, expected in sorted(source_hashes.items()):
        path = Path(raw_path)
        actual_current = sha_file(path) if path.is_file() else None
        if actual_current == expected:
            method = "current_file_bytes"
            verified_hash = actual_current
        else:
            # Emission pins retain original absolute source paths. Integration
            # can run from another checkout; resolve the original file's Git
            # owner rather than pretending it belongs to this runtime checkout.
            historical_repo = repo
            try:
                relative = path.resolve().relative_to(repo.resolve())
            except ValueError:
                owner = subprocess.run(
                    ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
                    check=True, capture_output=True, text=True).stdout.strip()
                historical_repo = Path(owner).resolve()
                relative = path.resolve().relative_to(historical_repo)
            historical = subprocess.run(
                ["git", "-C", str(historical_repo), "show",
                 f"{revision}:{relative.as_posix()}"],
                check=True, capture_output=True).stdout
            verified_hash = _sha_bytes(historical)
            _require(verified_hash == expected,
                     "source007 historical compiler code hash mismatch at " + str(path))
            method = "frozen_source_commit_git_object"
        verified[str(path)] = {
            "expected_sha256": expected,
            "current_sha256": actual_current,
            "verified_sha256": verified_hash,
            "verification_method": method,
        }
    current = source007._code_and_dependency_pins()
    expected_wheel = code_and_dependency_pins.get("manifold3d_retained_wheel", {})
    _require(current.get("manifold3d_retained_wheel") == expected_wheel,
             "source007 retained Manifold dependency hashes changed")
    return {
        "frozen_source_repository": source_repo,
        "source_code_file_count": len(verified),
        "verified_source_code": verified,
        "manifold3d_retained_wheel_sha256": expected_wheel,
        "passed": True,
    }


def _execution_verifier():
    """Return the package's current shared verifier for this new gate."""
    verifier = current_shared_verifier
    required = ("load_capture", "prepare_surface", "audit_surface_pair")
    if not all(callable(getattr(verifier, name, None)) for name in required):
        raise ValueError(
            "brachiocephalic 3D junction: current shared Human verifier API is incomplete")
    return verifier


def _runtime_code_pins() -> dict:
    repo = Path(__file__).resolve().parents[2]
    source = repo / "src/numilab_human"
    paths = {
        "diagnostic": Path(__file__).resolve(),
        "geometry_gate": source / "resting_brachiocephalic_geometry_gate.py",
        "source007": source / "resting_brachiocephalic_source007.py",
        "legacy_port_diagnostic": source / "resting_brachiocephalic_port_diagnostic.py",
        "nhanat_codec": source / "resting_pleura_proxy.py",
        "pose_predictor": source / "resting_brachiocephalic_pose_predictor.py",
        "current_shared_verifier": source / "resting_anatomy_verify.py",
        "current_capture_network_preparation": source /
            "resting_brachiocephalic_current_capture_network_preparation.py",
    }
    return {name: {"path": str(path.resolve()), "sha256": sha_file(path)}
            for name, path in paths.items()}


def build_preflight() -> dict:
    """Pin execution inputs without decoding geometry arrays or running predicates."""
    _require(not OUTPUT_DIR.exists(), "retain an existing diagnostic output; do not overwrite")
    _require(not PREFLIGHT_PATH.exists(), "retain an existing preflight; do not overwrite")
    _require(FIRST_EXECUTION_REPORT.is_file()
             and sha_file(FIRST_EXECUTION_REPORT) == FIRST_EXECUTION_REPORT_SHA256
             and FIRST_EXECUTION_PREFLIGHT.is_file()
             and sha_file(FIRST_EXECUTION_PREFLIGHT) == FIRST_EXECUTION_PREFLIGHT_SHA256,
             "preserved diagnostic-001 report or its source preflight changed")
    failed_execution = json.loads(FIRST_EXECUTION_REPORT.read_text())
    _require(failed_execution.get("error") == {
                 "type": "AttributeError",
                 "message": "module 'numilab_human.resting_brachiocephalic_port_diagnostic' has no attribute '_parse_payload'"}
             and failed_execution.get("native_or_gpu_work_performed") is False
             and failed_execution.get("source_compiler_invoked") is False
             and failed_execution.get("source_geometry_mutated") is False,
             "diagnostic-001 is not the retained pre-geometry codec wiring failure")
    _require(PREVIOUS_PREFLIGHT_PATH.is_file()
             and sha_file(PREVIOUS_PREFLIGHT_PATH) == PREVIOUS_PREFLIGHT_SHA256,
             "the first 3D preflight predecessor changed")
    _require(SOURCE_INTEGRITY_REPORT.is_file()
             and sha_file(SOURCE_INTEGRITY_REPORT) == SOURCE_INTEGRITY_REPORT_SHA256
             and SOURCE_INTEGRITY_SCRIPT.is_file()
             and sha_file(SOURCE_INTEGRITY_SCRIPT) == SOURCE_INTEGRITY_SCRIPT_SHA256,
             "source integrity precheck report or program changed")
    source_integrity = json.loads(SOURCE_INTEGRITY_REPORT.read_text())
    _require(source_integrity.get("status")
             == "SOURCE_INTEGRITY_PRECHECK_PASS_SELF_AUDIT_AND_3D_GEOMETRY_DEFERRED"
             and source_integrity.get("passed") is True
             and source_integrity.get("scope", {}).get("exact_self_audit_invoked") is False
             and source_integrity.get("scope", {}).get("numerical_or_exact_boolean_invoked") is False,
             "source identity/finite/index/closure precheck is not the pinned narrow pass")
    prior_preflight = json.loads(PREVIOUS_PREFLIGHT_PATH.read_text())
    _require(source_integrity.get("program", {}).get(
                 "repository_runner_sha256") == prior_preflight.get("runtime_code_sha256"),
             "source integrity precheck did not use the exact predecessor runner")
    source_bridge_checks = source_integrity.get(
        "source007_bridge_exact_self_checks", {}).get("bridges", {})
    _require(set(source_bridge_checks) == {"525", "526"}
             and all(item.get("pinned_source007_exact_self_clear") is True
                     and item.get("matches_pinned_source007_geometry") is True
                     for item in source_bridge_checks.values()),
             "source007 exact self-clear evidence for appended bridge rows is incomplete")
    target_integrity = source_integrity.get("surfaces", {}).get("472", {})
    _require(target_integrity.get("closed_after_exact_coordinate_seam_quotient") is True
             and target_integrity.get("one_referenced_face_component") is True
             and target_integrity.get("exact_self_audit_performed") is False,
             "source target 472 integrity precheck is not a closure-only pass")
    _require(SOURCE004_REPORT.is_file()
             and sha_file(SOURCE004_REPORT) == SOURCE004_REPORT_SHA256,
             "source007 004 rejection report changed")
    _require(SOURCE004_PREFLIGHT.is_file()
             and sha_file(SOURCE004_PREFLIGHT) == SOURCE004_PREFLIGHT_SHA256,
             "source007 004 preflight changed")
    previous = json.loads(SOURCE004_REPORT.read_text())
    previous_preflight = json.loads(SOURCE004_PREFLIGHT.read_text())
    _require(previous.get("status") == "SIX_PORT_RIGID_ONLY_SOURCE_SCREEN_REJECTED"
             and previous.get("first_failed_port") == {
                 "bridge": list(BRIDGE_KEY), "target": list(TARGET_KEY)},
             "source007 004 is not the pinned 526-to-472 rejection")
    prior_ports = previous.get("helper_result", {}).get("ports", [])
    _require(len(prior_ports) == 6, "source007 004 did not evaluate all six frozen ports")
    search = prior_ports[-1].get("target_station_candidate_search", {})
    _require(search.get("candidate_count_requested") == 4
             and search.get("candidate_count_evaluated") == 4
             and search.get("passed") is False
             and search.get("mesh_coordinates_modified") is False,
             "source007 004 did not exhaust exactly four immutable station candidates")

    _require(portdiag.RUN_REPORT.is_file()
             and sha_file(portdiag.RUN_REPORT) == portdiag.RUN_REPORT_SHA256,
             "pinned source007 geometry report changed")
    _require(portdiag.RUN_RECEIPT.is_file()
             and sha_file(portdiag.RUN_RECEIPT) == portdiag.RUN_RECEIPT_SHA256,
             "pinned source007 execution receipt changed")
    _require(source007.PREFLIGHT_PATH.is_file(),
             "source007 source preflight is missing")
    source_preflight = json.loads(source007.PREFLIGHT_PATH.read_text())
    _require(source_preflight.get("status") == "PREFLIGHT_PINNED_NO_GEOMETRY_DECODED",
             "source007 source preflight status changed")
    source_input_pins = source007._frozen_input_pins()
    _require(source_input_pins == source_preflight.get("input_sha256"),
             "source007 compiler input hashes differ from its retained preflight")
    source_owner_code_verification = _verify_source007_compiler_code_pins(
        source_preflight.get("code_and_dependency_pins", {}))
    source_owner_code_pins = source_preflight["code_and_dependency_pins"]
    plane_failure = _source004_plane_failure(previous)
    candidate_files = portdiag._actual_file_hashes(source007.CANDIDATE_DIR)
    preparation_files = portdiag._actual_file_hashes(source007.PREPARATION_DIR)
    _require(candidate_files == previous_preflight.get("candidate007_files_sha256"),
             "source007 candidate files differ from the completed 004 replay")
    _require(preparation_files == previous_preflight.get("prep007_files_sha256"),
             "source007 common-field preparation files differ from 004")
    _require(candidate_files.get("resting-thorax.nhanatomy")
             == previous.get("candidate_payload_sha256"),
             "source007 NHANAT payload differs from completed 004")
    prep_report_sha = previous.get("preparation_report_sha256")
    _require(prep_report_sha == preparation_files.get(
        "brachiocephalic-common-field-preparation.json"),
        "source007 preparation report differs from completed 004")

    capture_manifest = json.loads(source007.CAPTURE_MANIFEST.read_text())
    rows = capture_manifest.get("capture_rows", [])
    source0 = [row for row in rows if int(row.get("step", -1)) == 0]
    _require(len(source0) == 1
             and source0[0].get("recorded_pack_sha256") == source007.STEP0_PACK_SHA
             and source0[0].get("receipt_sha256") == source007.STEP0_RECEIPT_SHA,
             "accepted source0 capture pins changed")
    pack_path = Path(source0[0]["pack_path"])
    receipt_path = Path(source0[0]["receipt_path"])
    _require(sha_file(pack_path) == source007.STEP0_PACK_SHA
             and sha_file(receipt_path) == source007.STEP0_RECEIPT_SHA,
             "accepted source0 pack or receipt bytes changed")

    pins = {
        "previous_3d_preflight": {"path": str(PREVIOUS_PREFLIGHT_PATH),
                                  "sha256": PREVIOUS_PREFLIGHT_SHA256},
        "superseded_failed_run_preflight": {
            "path": str(FIRST_EXECUTION_PREFLIGHT),
            "sha256": FIRST_EXECUTION_PREFLIGHT_SHA256},
        "superseded_failed_run_report": {
            "path": str(FIRST_EXECUTION_REPORT),
            "sha256": FIRST_EXECUTION_REPORT_SHA256},
        "source_integrity_preparation_report": {
            "path": str(SOURCE_INTEGRITY_REPORT),
            "sha256": SOURCE_INTEGRITY_REPORT_SHA256},
        "source_integrity_preparation_script": {
            "path": str(SOURCE_INTEGRITY_SCRIPT),
            "sha256": SOURCE_INTEGRITY_SCRIPT_SHA256},
        "source007_report": {"path": str(portdiag.RUN_REPORT),
                             "sha256": portdiag.RUN_REPORT_SHA256},
        "source007_receipt": {"path": str(portdiag.RUN_RECEIPT),
                              "sha256": portdiag.RUN_RECEIPT_SHA256},
        "source007_preflight": {"path": str(source007.PREFLIGHT_PATH),
                                "sha256": sha_file(source007.PREFLIGHT_PATH)},
        "source007_input_sha256": source_input_pins,
        "source007_compiler_source_code_sha256": source_owner_code_pins[
            "source_code_sha256"],
        "source007_compiler_repository": source_owner_code_pins[
            "source_repository"],
        "source007_compiler_source_code_verification": source_owner_code_verification,
        "candidate007_files": candidate_files,
        "preparation007_files": preparation_files,
        "source004_report": {"path": str(SOURCE004_REPORT),
                             "sha256": SOURCE004_REPORT_SHA256},
        "source004_preflight": {"path": str(SOURCE004_PREFLIGHT),
                                "sha256": SOURCE004_PREFLIGHT_SHA256},
        "source0_pack": {"path": str(pack_path), "sha256": source007.STEP0_PACK_SHA},
        "source0_receipt": {"path": str(receipt_path),
                            "sha256": source007.STEP0_RECEIPT_SHA},
        "capture_manifest": {"path": str(source007.CAPTURE_MANIFEST),
                             "sha256": source007.CAPTURE_MANIFEST_SHA},
    }
    code_pins = _runtime_code_pins()
    wheel = dict(source007.MANIFOLD_SHA)
    for raw_path, expected in wheel.items():
        path = Path(raw_path)
        _require(path.is_file() and sha_file(path) == expected,
                 "retained Manifold wheel pin changed: " + raw_path)
    state = _git_state()
    _require(not state["dirty"], "preflight requires a clean committed source checkout")
    return {
        "schema": "numi.human.brachiocephalic-3d-junction-preflight.v3",
        "status": "PREFLIGHT_PINNED_NO_GEOMETRY_DECODED",
        "source_repository": state,
        "scope": {
            "bridge": list(BRIDGE_KEY),
            "target": list(TARGET_KEY),
            "port_count": 1,
            "source007_compiler_invoked": False,
            "all_six_ports_replayed": False,
            "all_nearby_targets_evaluated": False,
            "all_pose_checks": False,
            "native_or_gpu_work": False,
            "mesh_coordinates_modified": False,
            "material_branch": {
                "bridge": {"stable_id": 526, "source_member": "FJ3583",
                           "material": "right brachiocephalic vein"},
                "selected_target": {"stable_id": 472, "source_member": "FJ3587",
                                    "material": "right subclavian vein"},
                "selected_frozen_port": [[51011, 526], [51011, 472]],
                "expected_frozen_port_count_for_both_appended_bridges": 6,
                "coverage_limit": "one of three FJ3583 ports; not full tree admission",
            },
            "method": "independent closed-solid 3D overlap and two separately ancestry-bound source collar sections",
            "shared_planar_junction_section_required": False,
            "plane_candidate_sweep": False,
            "requirements": [
                "exact-coordinate-quotiented source closure and zero exact self intersections for candidate and baseline bridge and target",
                "one connected source face component per vessel after exact seam quotient",
                "frozen collar face-patch ancestry with exact captured source interface points for candidate and baseline",
                "positive volumetric intersection and one exact P6 inside-both witness for candidate and baseline",
                "one connected closed source component per vessel and one connected numerical union component",
                "numerical union-volume accounting equals source volumes minus overlap within declared relative tolerance 1e-6",
                "one positive closed source section per vessel at its own collar-ancestry material station",
                "candidate-to-baseline section area ratio [0.8,1.25] and minimum-Feret ratio [0.9,1.1]",
            ],
            "volume_balance_relative_tolerance": VOLUME_BALANCE_RELATIVE_TOLERANCE,
            "existing_cvsim_compartment_id": 4,
            "added_physical_mass_kg": 0,
            "added_blood_state_or_force": False,
        },
        "inputs": pins,
        "source004_plane_failure": plane_failure,
        "source_integrity_preparation": source_integrity,
        "runtime_code_sha256": code_pins,
        "manifold3d_retained_wheel_sha256": wheel,
        "current_shared_verifier_sha256": code_pins[
            "current_shared_verifier"]["sha256"],
        "historical_source007_p6_evidence_only": {
            "path": str((source007.P6_PACKAGE / "resting_anatomy_verify.py").resolve()),
            "sha256": source007.P6_VERIFIER_SHA,
            "loaded_by_this_gate": False,
        },
        "geometry_arrays_loaded": False,
        "geometry_predicates_run": False,
    }


def write_preflight() -> dict:
    report = build_preflight()
    _write_json(PREFLIGHT_PATH, report)
    return {"report_path": str(PREFLIGHT_PATH),
            "report_sha256": sha_file(PREFLIGHT_PATH),
            "status": report["status"]}


def _source_surface(verifier, key, value, label: str):
    detail = {"role": label, "identity": list(key)}
    try:
        _require(isinstance(value, (tuple, list)) and len(value) >= 2,
                 label + " surface must contain positions and faces")
        positions = gate._float32_xyz(value[0], name=label)
        faces = gate._faces(value[1], name=label, vertex_count=len(positions))
        normals = value[2] if len(value) > 2 else None
        source_ids = (value[3] if len(value) > 3
                      else np.arange(len(positions), dtype=np.uint64))
        prepared = verifier.prepare_surface(
            key, positions, faces, normals=normals, source_vertex_ids=source_ids)
        self_audit = prepared.self_audit()
        topology = prepared.quotient_topology
        component_count = int(topology.get("face_component_count", -1))
        signed_volume = gate._signed_volume(prepared.positions, prepared.faces)
        detail.update({
            "prepared_surface": prepared.diagnostics(),
            "source_vertex_ids_sha256": _sha_bytes(
                np.asarray(prepared.source_vertex_ids, dtype="<u8").tobytes()),
            "referenced_face_component_count": component_count,
            "exact_self_audit_count": int(self_audit.get("count", -1)),
            "exact_self_audit_pairs": _jsonable(self_audit.get("pairs", [])),
            "signed_volume_m3": signed_volume,
            "absolute_source_volume_m3": abs(signed_volume),
            "closed_self_free_single_component": bool(
                prepared.closed and int(self_audit.get("count", -1)) == 0
                and component_count == 1 and np.isfinite(signed_volume)
                and abs(signed_volume) > 0.0),
        })
        return prepared, detail
    except Exception as error:  # noqa: BLE001 - retain exact source-preparation failures
        detail["error"] = {"type": type(error).__name__, "message": str(error)}
        return None, detail


def _own_collar_station(surface, witness_points, collar_face_ids, *, role: str):
    """Derive one section from this vessel's own frozen collar and source interior."""
    positions = surface.positions.astype(np.float64)
    faces = surface.faces
    collar_faces = np.asarray(sorted(set(map(int, collar_face_ids))), dtype=np.int64)
    _require(collar_faces.ndim == 1 and len(collar_faces) > 0
             and int(collar_faces.min()) >= 0 and int(collar_faces.max()) < len(faces),
             role + " has no valid frozen collar face ancestry")
    collar_vertices = np.unique(faces[collar_faces].reshape(-1))
    referenced_vertices = np.unique(faces.reshape(-1))
    interior_vertices = np.setdiff1d(referenced_vertices, collar_vertices)
    _require(len(collar_vertices) >= 3 and len(interior_vertices) >= 2,
             role + " collar or non-collar source ancestry is insufficient")
    collar_center = positions[collar_vertices].mean(axis=0)
    interior_center = positions[interior_vertices].mean(axis=0)
    direction = interior_center - collar_center
    direction_norm = float(np.linalg.norm(direction))
    _require(np.isfinite(direction_norm) and direction_norm > 1e-15,
             role + " source has no collar-to-interior direction")
    direction /= direction_norm

    projection = positions[referenced_vertices] @ direction
    support = referenced_vertices[[int(np.argmin(projection)), int(np.argmax(projection))]]
    tangent = positions[int(support[1])] - positions[int(support[0])]
    tangent_norm = float(np.linalg.norm(tangent))
    _require(np.isfinite(tangent_norm) and tangent_norm > 1e-15,
             role + " source tangent support collapsed")
    tangent /= tangent_norm
    if float(np.dot(tangent, direction)) < 0.0:
        support = support[::-1]
        tangent = -tangent

    weights = np.full(len(collar_vertices), 1.0 / len(collar_vertices), dtype=np.float64)
    ancestry = {
        "material_anchor_local_vertex_ids": collar_vertices.astype(int).tolist(),
        "material_anchor_source_vertex_ids": [
            int(surface.source_vertex_ids[int(index)]) for index in collar_vertices],
        "material_anchor_weights": weights.tolist(),
        "tangent_support_local_vertex_ids": support.astype(int).tolist(),
        "tangent_support_source_vertex_ids": [
            int(surface.source_vertex_ids[int(index)]) for index in support],
        "frozen_collar_face_ids": collar_faces.astype(int).tolist(),
    }
    anchor = np.sum(positions[collar_vertices] * weights[:, None], axis=0)
    collar_points = positions[faces[collar_faces]].reshape(-1, 3)
    raw_positive_extent = float(np.max((collar_points - anchor) @ tangent))
    margin = max(float(np.max(np.abs(np.spacing(surface.positions))) * 8.0),
                 float(np.finfo(np.float32).tiny))
    _require(np.isfinite(raw_positive_extent) and raw_positive_extent >= -margin,
             role + " frozen collar extent is inconsistent with its material anchor")
    positive_extent = max(0.0, raw_positive_extent)
    station = gate._transport_material_station(surface, {
        **ancestry,
        "tangent_world_unit": tangent.tolist(),
        "provenance": (
            "single vessel-local collar-to-interior direction and referenced "
            "source-vertex extreme tangent; no shared junction plane"),
        "inferred": True,
        "station_offset_from_anchor_m": positive_extent + margin,
    })
    points = np.asarray(witness_points, dtype=np.float64)
    _require(points.ndim == 2 and points.shape[1] == 3 and len(points) > 0
             and np.isfinite(points).all(),
             role + " needs finite exact frozen-interface witness points")
    junction = points.mean(axis=0)
    station["own_collar_station_policy"] = "OWN_COLLAR_OUTSIDE_PATCH_V1"
    station["source_role"] = role
    station["collar_to_interior_direction_world_unit"] = direction.tolist()
    station["collar_center_world_m"] = collar_center.tolist()
    station["noncollar_referenced_centroid_world_m"] = interior_center.tolist()
    station["referenced_tangent_support_local_vertex_ids"] = support.astype(int).tolist()
    station["referenced_tangent_support_source_vertex_ids"] = ancestry[
        "tangent_support_source_vertex_ids"]
    station["raw_positive_collar_projection_extent_m"] = raw_positive_extent
    station["positive_collar_projection_extent_m"] = positive_extent
    station["exact_interface_witness_points_sha256"] = _sha_bytes(
        np.ascontiguousarray(points, dtype="<f8").tobytes())
    station["exact_interface_witness_centroid_world_m"] = junction.tolist()
    station["junction_witness_signed_distance_to_station_m"] = float(
        np.dot(junction - np.asarray(station["section_center_world_m"]), tangent))
    station["mesh_coordinates_modified"] = False
    return station


def _station_calibre(surface, baseline_surface, witness_points, collar_faces, *,
                     role: str, owner_id: int) -> dict:
    result = {"role": role, "owner_id": owner_id}
    try:
        station = _own_collar_station(
            surface, witness_points, collar_faces, role=role)
        baseline_station = gate._transport_material_station(baseline_surface, station)
        candidate_section = gate._source_surface_section(surface, station)
        baseline_section = gate._source_surface_section(
            baseline_surface, baseline_station)
        candidate_metric, candidate_error = gate._section_metric_for_owner(
            candidate_section, owner_id, label=f"{role} candidate source section")
        baseline_metric, baseline_error = gate._section_metric_for_owner(
            baseline_section, owner_id, label=f"{role} baseline source section")
        comparison = None
        comparison_error = None
        if candidate_metric is not None and baseline_metric is not None:
            try:
                comparison = gate._section_comparison(candidate_metric, baseline_metric)
            except Exception as error:  # noqa: BLE001 - retain the actual comparison failure
                comparison_error = {"type": type(error).__name__, "message": str(error)}
        result.update({
            "passed": bool(comparison and comparison.get("passed")),
            "station": station,
            "baseline_station_transported_by_source_ancestry": baseline_station,
            "candidate_section": candidate_section,
            "baseline_section": baseline_section,
            "candidate_metric_error": candidate_error,
            "baseline_metric_error": baseline_error,
            "candidate_to_baseline_comparison": comparison,
            "comparison_error": comparison_error,
            "declared_area_ratio": [0.8, 1.25],
            "declared_minimum_feret_ratio": [0.9, 1.1],
            "clinical_calibre_claim": False,
        })
    except Exception as error:  # noqa: BLE001 - preserve failed station derivation/evaluation
        result.update({
            "passed": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        })
    return result


def _interface_gate(
    verifier, bridge, target, frozen_pairs, *, declared_vessel_ports=None,
) -> dict:
    typed_port = (tuple(map(int, bridge.key)), tuple(map(int, target.key)))
    declared_ports = (gate.DECLARED_VESSEL_PORTS if declared_vessel_ports is None
                      else frozenset(declared_vessel_ports))
    port_scope_passed = typed_port in declared_ports
    pairs = gate._pair_rows(
        frozen_pairs, name=(
            f"frozen source collar ancestry {typed_port[0][1]}->{typed_port[1][1]}"),
        bridge_face_count=len(bridge.faces), target_face_count=len(target.faces))
    bridge_patch = sorted({first for first, _ in pairs})
    target_patch = sorted({second for _, second in pairs})
    allowed = {(first, second) for first in bridge_patch for second in target_patch}
    audit = verifier.audit_surface_pair(
        bridge, target, intended_face_pairs=tuple(sorted(allowed)))
    observed = tuple(tuple(map(int, pair)) for pair in
                     audit.get("declared_interface_triangle_pairs", []))
    unexpected = tuple(tuple(map(int, pair)) for pair in
                       audit.get("unexpected_triangle_pairs", []))
    exact_points = gate._exact_interface_points(verifier, bridge, target, observed)
    within_patch = all(pair in allowed for pair in observed)
    containment = audit.get("containment", [])
    partial_overlap_boundary = gate._declared_partial_vessel_port_overlap(
        bridge.key, target.key, audit, observed, unexpected, exact_points, allowed,
        declared_vessel_ports=declared_ports)
    exact_partial_overlap = bool(
        port_scope_passed and partial_overlap_boundary.get("passed"))
    pair_patch_and_collar_passed = bool(
        port_scope_passed and not unexpected and within_patch and bool(exact_points)
        and (bool(audit.get("passed")) or exact_partial_overlap)
    )
    return {
        "passed": pair_patch_and_collar_passed,
        "typed_port": [list(typed_port[0]), list(typed_port[1])],
        "documented_vessel_junction_scope": {
            "passed": port_scope_passed,
            "typed_port": [list(typed_port[0]), list(typed_port[1])],
            "allowed_typed_port": (
                [list(typed_port[0]), list(typed_port[1])]
                if port_scope_passed else None),
            "scope_kind": ("declared_source_network_link"
                           if declared_vessel_ports is not None
                           else "legacy_single_port_entrypoint"),
            "frozen_matched_face_patch_required": True,
            "downstream_closed_solid_confirmation_required": True,
            "scope": (
                "FJ3583 right brachiocephalic vein to FJ3587 right subclavian vein"
                if typed_port == PARTIAL_OVERLAP_PORT else
                "the exact typed link named by its source ancestry"),
        },
        "p6_audit_passed": bool(audit.get("passed")),
        "exact_partial_overlap_witnesses": {
            "passed": exact_partial_overlap,
            "requires_separate_closed_solid_confirmation": exact_partial_overlap,
            "classification": (
                "symmetric_exact_partial_volumetric_overlap"
                if exact_partial_overlap else "not_classified_as_expected_partial_overlap"),
            "boundary_predicate": partial_overlap_boundary,
            "containment": containment,
        },
        "observed_pair_count": len(observed),
        "observed_face_pairs": [list(pair) for pair in observed],
        "unexpected_face_pairs": [list(pair) for pair in unexpected],
        "observed_pairs_within_frozen_face_patch_cross_product": within_patch,
        "frozen_face_pairs": [list(pair) for pair in pairs],
        "frozen_face_pairs_sha256": _sha_bytes(
            np.ascontiguousarray(pairs, dtype="<i8").tobytes()),
        "bridge_frozen_collar_faces": bridge_patch,
        "bridge_frozen_collar_faces_sha256": _sha_bytes(
            np.ascontiguousarray(bridge_patch, dtype="<i8").tobytes()),
        "target_frozen_collar_faces": target_patch,
        "target_frozen_collar_faces_sha256": _sha_bytes(
            np.ascontiguousarray(target_patch, dtype="<i8").tobytes()),
        "exact_interface_point_count": len(exact_points),
        "exact_interface_points_world_m": [list(point) for point in exact_points],
        "exact_interface_points_sha256": _sha_bytes(
            np.ascontiguousarray(exact_points, dtype="<f8").tobytes()),
        "audit_result": _jsonable(audit),
    }


def _volume_accounting(first_volume: float, second_volume: float,
                       intersection_volume: float, union_volume: float) -> dict:
    expected_union = first_volume + second_volume - intersection_volume
    scale = max(first_volume + second_volume, np.finfo(np.float64).tiny)
    residual = union_volume - expected_union
    relative_error = abs(residual) / max(abs(expected_union), np.finfo(np.float64).tiny)
    tolerance = VOLUME_BALANCE_RELATIVE_TOLERANCE
    no_input_volume_loss = union_volume + scale * tolerance >= max(
        first_volume, second_volume)
    return {
        "first_source_volume_m3": first_volume,
        "second_source_volume_m3": second_volume,
        "intersection_volume_m3": intersection_volume,
        "expected_union_volume_m3": expected_union,
        "observed_union_volume_m3": union_volume,
        "signed_volume_balance_residual_m3": residual,
        "relative_volume_balance_error": relative_error,
        "declared_relative_tolerance": tolerance,
        "union_volume_not_less_than_either_source": no_input_volume_loss,
        "passed": bool(expected_union > 0.0 and relative_error <= tolerance
                       and no_input_volume_loss),
        "identity": "V(union)=V(first)+V(second)-V(intersection)",
        "scope": "floating Manifold volume accounting, not physical blood volume",
    }


def _material_branch_scope(source_map: dict, pairs_by_port: dict) -> dict:
    """Bind the selected right bridge/subclavian branch without overclaiming the tree."""
    expected_ports = set(source007.PORT_KEYS)
    _require(set(pairs_by_port) == expected_ports,
             "all six expected frozen collar-port ancestry arrays are required")
    bridge = source_map.get("526", {})
    target = source_map.get("472", {})
    _require(bridge.get("source_member") == "FJ3583"
             and int(bridge.get("body_index", -1)) == 20
             and int(bridge.get("layer", -1)) == 2
             and int(bridge.get("vascular_compartment_id", -1)) == 4,
             "stable ID 526 is not the pinned FJ3583 right brachiocephalic branch")
    _require(target.get("source_member") == "FJ3587"
             and target.get("name") == "right subclavian vein"
             and int(target.get("body_index", -1)) == 20
             and int(target.get("layer", -1)) == 2
             and int(target.get("vascular_compartment_id", -1)) == 4,
             "stable ID 472 is not the pinned FJ3587 right subclavian target")
    ports = []
    for pair in source007.PORT_KEYS:
        pairs = pairs_by_port[pair]
        ports.append({
            "bridge": list(pair[0]),
            "target": list(pair[1]),
            "face_pair_count": len(pairs),
            "ordered_face_pairs_json_sha256": portdiag._json_hash(pairs),
        })
    selected = next(item for item in ports
                    if item["bridge"] == [51011, 526]
                    and item["target"] == [51011, 472])
    _require(selected["face_pair_count"] == 63,
             "selected FJ3583-to-FJ3587 frozen interface pair count changed")
    return {
        "bridge_identity": bridge,
        "target_identity": target,
        "selected_material_branch": "FJ3583 right brachiocephalic vein to FJ3587 right subclavian vein",
        "selected_frozen_interface": selected,
        "all_six_expected_ports_present": True,
        "all_frozen_ports": ports,
        "coverage_limit": (
            "This screen proves only the 526-to-472 interface among FJ3583's "
            "three frozen ports; it does not establish the complete right venous tree."),
        "existing_cvsim_compartment_id": 4,
        "added_physical_mass_kg": 0,
        "added_blood_state_or_force": False,
    }


def _source007_bridge_self_check_reuse(source007_report: dict,
                                        bridge_surfaces: dict) -> dict:
    """Reuse pinned exact source-0 self checks for both appended bridge IDs."""
    checks = {
        int(item.get("identity", [-1, -1])[1]): item
        for item in source007_report.get("source_closed_self_checks", [])
        if isinstance(item.get("identity"), list) and len(item["identity"]) == 2
    }
    output = {
        "source007_report_path": str(portdiag.RUN_REPORT),
        "source007_report_sha256": portdiag.RUN_REPORT_SHA256,
        "reused_existing_exact_p6_self_audits": True,
        "bridges": {},
    }
    for stable_id in (525, 526):
        key = (51011, stable_id)
        pinned = checks.get(stable_id, {})
        surface = bridge_surfaces.get(key)
        if surface is None:
            output["bridges"][str(stable_id)] = {
                "passed": False,
                "error": "source007 bridge surface is missing",
            }
            continue
        positions_sha = _sha_bytes(
            np.ascontiguousarray(surface[0], dtype="<f4").tobytes())
        faces_sha = _sha_bytes(np.ascontiguousarray(surface[1]).tobytes())
        matched = bool(
            pinned.get("closed_oriented_source") is True
            and pinned.get("exact_self_clear") is True
            and int(pinned.get("exact_self_intersection_count", -1)) == 0
            and pinned.get("positions_sha256") == positions_sha
            and pinned.get("faces_sha256") == faces_sha)
        output["bridges"][str(stable_id)] = {
            "positions_sha256": positions_sha,
            "faces_sha256": faces_sha,
            "vertex_count": len(surface[0]),
            "triangle_count": len(surface[1]),
            "pinned_source007_closed_oriented_source": pinned.get(
                "closed_oriented_source"),
            "pinned_source007_exact_self_intersection_count": pinned.get(
                "exact_self_intersection_count"),
            "pinned_source007_exact_self_clear": pinned.get("exact_self_clear"),
            "matches_pinned_geometry_bytes": bool(
                pinned.get("positions_sha256") == positions_sha
                and pinned.get("faces_sha256") == faces_sha),
            "passed": matched,
        }
    output["passed"] = (
        set(checks) == {525, 526}
        and all(item.get("passed") is True for item in output["bridges"].values()))
    return output


def _decode_candidate_payload(raw: bytes, source_map: dict):
    """Decode ABI5 through the existing NHANAT codec owner and bind row IDs."""
    parser = getattr(nhanat_codec, "_parse_payload", None)
    _require(callable(parser), "existing resting_pleura_proxy NHANAT decoder is missing")
    decoded = parser(raw)
    _require(isinstance(decoded, tuple) and len(decoded) == 3,
             "existing ABI5 decoder must return header, records, and row map")
    header, records, rows = decoded
    _require(isinstance(rows, dict)
             and set(rows) == set(map(int, source_map)),
             "decoded NHANAT stable IDs differ from the bound source_id_map")
    _require({525, 526, 472}.issubset(rows),
             "decoded NHANAT payload lacks the selected bridge and target IDs")
    return header, records, rows


def _prepare_source007_inputs(preflight: dict) -> dict:
    """Run source/capture initialization only, stopping before geometry predicates."""
    required_helpers = {
        "nhanat_codec._parse_payload": getattr(nhanat_codec, "_parse_payload", None),
        "source007._validate_source_candidate": getattr(
            source007, "_validate_source_candidate", None),
        "current_shared_verifier.audit_surface_pair": getattr(
            current_shared_verifier, "audit_surface_pair", None),
        "source007._step0_source": getattr(source007, "_step0_source", None),
        "source007._load_predictor_transform": getattr(
            source007, "_load_predictor_transform", None),
        "source007._port_target_surfaces": getattr(
            source007, "_port_target_surfaces", None),
        "source007._bridge_surfaces": getattr(source007, "_bridge_surfaces", None),
        "source007._read_port_pairs": getattr(source007, "_read_port_pairs", None),
        "source007._load_retained_manifold": getattr(
            source007, "_load_retained_manifold", None),
    }
    _require(all(callable(helper) for helper in required_helpers.values()),
             "one or more pinned source/capture/dependency owner helpers are missing")

    candidate_payload = source007.CANDIDATE_DIR / "resting-thorax.nhanatomy"
    candidate_receipt = source007.PREPARATION_DIR / "resting-anatomy-receipt.json"
    prep_report = source007.PREPARATION_DIR / "brachiocephalic-common-field-preparation.json"
    validation = source007._validate_source_candidate(candidate_receipt, prep_report)
    expected_payload_sha = preflight["inputs"]["candidate007_files"][
        "resting-thorax.nhanatomy"]
    _require(sha_file(candidate_payload) == expected_payload_sha,
             "candidate NHA payload differs from the current preflight")
    header, records, rows = _decode_candidate_payload(
        candidate_payload.read_bytes(), validation["source_map"])

    verifier = _execution_verifier()
    manifest = json.loads(source007.CAPTURE_MANIFEST.read_text())
    captured, identity, capture_meta = source007._step0_source(manifest, verifier)
    body20_pose = capture_meta.get("body20", {})
    _require(isinstance(identity, dict) and identity.get("accepted_step") == 0
             and isinstance(body20_pose, dict)
             and np.asarray(body20_pose.get("quaternion_xyzw")).shape == (4,)
             and np.asarray(body20_pose.get("position_m")).shape == (3,),
             "accepted source-0 pack/receipt/body20 initialization contract failed")
    transform = source007._load_predictor_transform()
    _require(callable(transform), "pinned source-0 Float32 rigid transform is not callable")
    target_surfaces = source007._port_target_surfaces(captured, rows)
    bridge_surfaces = source007._bridge_surfaces(
        rows, capture_meta["body20"], transform)
    _require(set(target_surfaces) == set(source007.REQUIRED_SOURCE_TARGETS)
             and TARGET_KEY in target_surfaces,
             "accepted source-0 target surface identity contract failed")
    _require(set(bridge_surfaces) == set(source007.BRIDGE_KEYS)
             and BRIDGE_KEY in bridge_surfaces,
             "source-0 rigid bridge surface identity contract failed")

    source007_report = json.loads(portdiag.RUN_REPORT.read_text())
    bridge_self_check_reuse = _source007_bridge_self_check_reuse(
        source007_report, bridge_surfaces)
    pairs_by_port, attribution, stationarity = source007._read_port_pairs()
    material_branch = _material_branch_scope(validation["source_map"], pairs_by_port)
    _require(PAIR in pairs_by_port and len(pairs_by_port[PAIR]) == 63,
             "frozen selected 526-to-472 face-pair interface is missing or changed")
    _require(stationarity.get("status") == "PASS_SOURCE_PORT_SUPPORT_FIXED",
             "frozen source collar support stationarity evidence changed")

    return {
        "candidate_payload": candidate_payload,
        "candidate_receipt": candidate_receipt,
        "preparation_report": prep_report,
        "validation": validation,
        "header": header,
        "record_count": len(records),
        "rows": rows,
        "verifier": verifier,
        "captured": captured,
        "identity": identity,
        "capture_meta": capture_meta,
        "target_surfaces": target_surfaces,
        "bridge_surfaces": bridge_surfaces,
        "bridge_self_check_reuse": bridge_self_check_reuse,
        "pairs_by_port": pairs_by_port,
        "attribution": attribution,
        "stationarity": stationarity,
        "material_branch": material_branch,
        "required_helper_symbols": sorted(required_helpers),
    }


def run_initialization_dryrun() -> dict:
    """Exercise the real source/capture/dependency setup before exact predicates."""
    _require(PREFLIGHT_PATH.is_file(), "run the no-geometry 3D preflight first")
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    _assert_current_pins(preflight)
    _require(not INITIALIZATION_DRYRUN_REPORT.exists(),
             "retain an existing initialization dry-run report; do not overwrite")
    started = time.perf_counter()
    report = {
        "schema": "numi.human.brachiocephalic-3d-junction-initialization.v1",
        "status": "RUNNING_CURRENT_SHARED_VERIFIER_INITIALIZATION_DRYRUN",
        "preflight_path": str(PREFLIGHT_PATH),
        "preflight_sha256": sha_file(PREFLIGHT_PATH),
        "recorded_start_utc": datetime.now(UTC).isoformat(),
        "geometry_arrays_loaded": False,
        "exact_geometry_predicates_invoked": False,
        "numerical_boolean_invoked": False,
        "source_compiler_invoked": False,
        "source_geometry_mutated": False,
        "native_or_gpu_work_performed": False,
    }
    try:
        prepared = _prepare_source007_inputs(preflight)
        manifold3d = source007._load_retained_manifold()
        _require(callable(getattr(manifold3d.Manifold, "to_mesh64", None)),
                 "retained Manifold Mesh64 setup contract failed")
        bridge = prepared["bridge_surfaces"][BRIDGE_KEY]
        target = prepared["target_surfaces"][TARGET_KEY]
        report.update({
            "status": "INITIALIZATION_AND_DECODE_PASS_PRE_EXACT_PREDICATES",
            "current_shared_verifier_path": str(
                Path(current_shared_verifier.__file__).resolve()),
            "current_shared_verifier_sha256": sha_file(
                Path(current_shared_verifier.__file__).resolve()),
            "geometry_arrays_loaded": True,
            "candidate_payload_sha256": sha_file(prepared["candidate_payload"]),
            "source0_pack_sha256": prepared["identity"]["pack_sha256"],
            "source0_receipt_sha256": prepared["identity"][
                "capture_receipt_sha256"],
            "decoded_surface_count": prepared["record_count"],
            "decoded_stable_id_count": len(prepared["rows"]),
            "decoded_max_stable_id": max(prepared["rows"]),
            "decoded_ids_include_472_525_526": all(
                stable_id in prepared["rows"] for stable_id in (472, 525, 526)),
            "source0_target_surface_count": len(prepared["target_surfaces"]),
            "source0_bridge_surface_count": len(prepared["bridge_surfaces"]),
            "selected_bridge_vertex_count": len(bridge[0]),
            "selected_bridge_triangle_count": len(bridge[1]),
            "selected_target_vertex_count": len(target[0]),
            "selected_target_triangle_count": len(target[1]),
            "selected_frozen_face_pair_count": len(
                prepared["pairs_by_port"][PAIR]),
            "material_branch_scope": prepared["material_branch"],
            "source007_bridge_self_check_reuse": prepared[
                "bridge_self_check_reuse"],
            "stationarity_status": prepared["stationarity"]["status"],
            "attribution_candidate107": prepared["attribution"].get(
                "candidate107_payload", {}),
            "required_helper_symbols_called_or_loaded": prepared[
                "required_helper_symbols"],
            "retained_manifold_module_path": str(Path(manifold3d.__file__).resolve()),
            "retained_manifold_mesh64_callable": True,
            "first_not_invoked_exact_geometry_stage": "audit_brachiocephalic_3d_junction",
            "limitations": [
                "This is source/capture/dependency initialization only.",
                "No exact closure/self, pair, inside, overlap, section, or CSG predicate was invoked.",
                "No anatomical or junction admission is claimed.",
            ],
        })
    except Exception as error:  # noqa: BLE001 - preserve the exact setup failure
        report["status"] = "INITIALIZATION_DRYRUN_ABORTED_WITH_ERROR"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["recorded_finish_utc"] = datetime.now(UTC).isoformat()
    report["elapsed_seconds"] = time.perf_counter() - started
    _write_json(INITIALIZATION_DRYRUN_REPORT, report)
    return {
        "status": report["status"],
        "report_path": str(INITIALIZATION_DRYRUN_REPORT),
        "report_sha256": sha_file(INITIALIZATION_DRYRUN_REPORT),
        "elapsed_seconds": report["elapsed_seconds"],
    }


def _solid_gate(verifier, manifold3d, bridge, target, *, role: str) -> dict:
    output = {"role": role, "passed": False, "floating_boolean_exact_topology_proven": False}
    try:
        first, first_info = gate._make_manifold(bridge, manifold3d)
        second, second_info = gate._make_manifold(target, manifold3d)
        first_status, second_status = str(first.status()), str(second.status())
        first_components = len(first.decompose())
        second_components = len(second.decompose())
        inputs_passed = (first_status.endswith("NoError")
                         and second_status.endswith("NoError")
                         and first_components == 1 and second_components == 1)
        bridge_volume = abs(gate._signed_volume(bridge.positions, bridge.faces))
        target_volume = abs(gate._signed_volume(target.positions, target.faces))
        result = {
            "bridge_manifold_input": _jsonable(first_info),
            "target_manifold_input": _jsonable(second_info),
            "bridge_status": first_status,
            "target_status": second_status,
            "bridge_component_count": first_components,
            "target_component_count": second_components,
            "bridge_source_volume_m3": bridge_volume,
            "target_source_volume_m3": target_volume,
            "manifold_inputs_passed": inputs_passed,
        }
        if not inputs_passed:
            output.update(result)
            output["rejection"] = "Manifold input status/component check failed"
            return output

        intersection = manifold3d.Manifold.batch_boolean(
            [first, second], manifold3d.OpType.Intersect)
        union = manifold3d.Manifold.batch_boolean(
            [first, second], manifold3d.OpType.Add)
        intersection_status, union_status = str(intersection.status()), str(union.status())
        intersection_volume = float(intersection.volume())
        union_volume = float(union.volume())
        intersection_components = len(intersection.decompose())
        union_components = len(union.decompose())
        intersection_witness = (
            gate._exact_inside_witness(verifier, bridge, target, intersection)
            if intersection_volume > 0.0 else None)
        union_positions, union_faces, union_mesh = gate._manifold_mesh(union)
        lineage = gate._mesh_face_lineage(
            union_mesh, {
                bridge.key[1]: len(bridge.quotient_faces),
                target.key[1]: len(target.quotient_faces),
            })
        lineage_report = lineage["report"]
        lineage_passed = bool(
            lineage_report.get("generated_face_count") == 0
            and lineage_report.get("unmapped_face_count") == 0
            and set(lineage_report.get("run_original_ids", []))
            == {bridge.key[1], target.key[1]})
        accounting = _volume_accounting(
            bridge_volume, target_volume, intersection_volume, union_volume)
        status_passed = (
            intersection_status.endswith("NoError")
            and union_status.endswith("NoError"))
        output.update(result)
        output.update({
            "intersection_status": intersection_status,
            "intersection_volume_m3": intersection_volume,
            "intersection_component_count": intersection_components,
            "positive_intersection_volume": intersection_volume > 0.0,
            "exact_inside_both_witness": intersection_witness,
            "union_status": union_status,
            "union_volume_m3": union_volume,
            "union_component_count": union_components,
            "union_connected_numerical_solid": bool(
                union_status.endswith("NoError") and union_components == 1),
            "union_triangle_count": len(union_faces),
            "union_vertex_count": len(union_positions),
            "union_float64_mesh64_lineage": lineage_report,
            "union_lineage_has_both_sources_without_generated_or_unmapped_faces":
                lineage_passed,
            "volume_accounting": accounting,
            "csg_analysis_precision": "Manifold Mesh64 Float64; floating Boolean only",
            "floating_boolean_exact_topology_proven": False,
        })
        output["passed"] = bool(
            status_passed and intersection_volume > 0.0
            and intersection_witness is not None
            and union_status.endswith("NoError") and union_components == 1
            and lineage_passed and accounting["passed"])
        if not output["passed"]:
            output["rejection"] = "solid overlap, exact witness, connected union, source lineage, or volume accounting failed"
    except Exception as error:  # noqa: BLE001 - retain partial numerical solid evidence
        output["error"] = {"type": type(error).__name__, "message": str(error)}
    return output


def audit_brachiocephalic_3d_junction(
    verifier, manifold3d, *, bridge_key, bridge_surface, target_key, target_surface,
    frozen_face_pairs, baseline_bridge_surface, baseline_target_surface,
    _source_link_scope=None,
) -> dict:
    """Prove one source-bound vessel link without a shared planar junction cut."""
    bridge_key = gate._identity(bridge_key)
    target_key = gate._identity(target_key)
    typed_link = (bridge_key, target_key)
    if _source_link_scope is None:
        _require(typed_link == PAIR,
                 "legacy 3D junction entry point is restricted to stable IDs 526 and 472")
    else:
        _require(typed_link in set(declared_network_links())
                 and _source_link_scope.get("typed_link")
                 == [list(bridge_key), list(target_key)],
                 "3D source-link scope does not match a declared network edge")

    surfaces = {}
    surface_reports = {}
    for label, key, value in (
        ("candidate_bridge", bridge_key, bridge_surface),
        ("candidate_target", target_key, target_surface),
        ("baseline_bridge", bridge_key, baseline_bridge_surface),
        ("baseline_target", target_key, baseline_target_surface),
    ):
        prepared, evidence = _source_surface(verifier, key, value, label)
        surfaces[label] = prepared
        surface_reports[label] = evidence

    topology_passed = all(
        surfaces[name] is not None
        and surface_reports[name].get("closed_self_free_single_component") is True
        for name in ("candidate_bridge", "candidate_target",
                     "baseline_bridge", "baseline_target"))
    ancestry = {}
    ancestry_issues = []
    if topology_passed:
        for role in ("bridge", "target"):
            candidate = surfaces["candidate_" + role]
            baseline = surfaces["baseline_" + role]
            same_faces = (candidate.faces.shape == baseline.faces.shape
                          and np.array_equal(candidate.faces, baseline.faces))
            same_ids = (candidate.source_vertex_ids.shape == baseline.source_vertex_ids.shape
                        and np.array_equal(candidate.source_vertex_ids,
                                           baseline.source_vertex_ids))
            candidate_volume = gate._signed_volume(candidate.positions, candidate.faces)
            baseline_volume = gate._signed_volume(baseline.positions, baseline.faces)
            orientation = candidate_volume * baseline_volume > 0.0
            ratio = (abs(candidate_volume) / abs(baseline_volume)
                     if baseline_volume != 0.0 else None)
            positive_volumes = bool(
                np.isfinite(candidate_volume) and np.isfinite(baseline_volume)
                and abs(candidate_volume) > 0.0 and abs(baseline_volume) > 0.0)
            evidence = {
                "candidate_face_ancestry_sha256": _sha_bytes(candidate.faces.tobytes()),
                "baseline_face_ancestry_sha256": _sha_bytes(baseline.faces.tobytes()),
                "candidate_source_vertex_ids_sha256": _sha_bytes(
                    np.asarray(candidate.source_vertex_ids, dtype="<u8").tobytes()),
                "baseline_source_vertex_ids_sha256": _sha_bytes(
                    np.asarray(baseline.source_vertex_ids, dtype="<u8").tobytes()),
                "same_ordered_source_face_ancestry": same_faces,
                "same_source_vertex_ancestry": same_ids,
                "candidate_signed_volume_m3": candidate_volume,
                "baseline_signed_volume_m3": baseline_volume,
                "candidate_to_baseline_source_volume_ratio": ratio,
                "source_volumes_finite_and_positive": positive_volumes,
                "source_volume_ratio_is_diagnostic_only": True,
                "source_orientation_preserved": orientation,
                "passed": bool(same_faces and same_ids and orientation
                               and positive_volumes),
            }
            ancestry[role] = evidence
            if not evidence["passed"]:
                ancestry_issues.append(
                    role + " candidate/baseline source identity, orientation, or volume changed")
    ancestry_passed = topology_passed and not ancestry_issues

    result = {
        "schema": "numi.human.brachiocephalic-3d-junction-continuity.v1",
        "bridge": list(bridge_key),
        "target": list(target_key),
        "status": "RUNNING",
        "passed": False,
        "source_topology_gate": {
            "passed": topology_passed,
            "require_one_referenced_source_component_per_vessel": True,
            "surfaces": surface_reports,
        },
        "candidate_baseline_ancestry_gate": {
            "passed": ancestry_passed,
            "vessels": ancestry,
            "rejections": ancestry_issues,
        },
        "port_interfaces": {},
        "solid_gates": {},
        "independent_collar_calibre": {},
        "limitations": {
            "shared_planar_junction_section_required": False,
            "union_section_loop_count_used_as_continuity_premise": False,
            "plane_candidate_sweep": False,
            "candidate_source_faces_or_coordinates_modified": False,
            "union_boolean_exact_topology_proven": False,
            "clinical_calibre": "not_established",
            "native_gpu_confirmation": "required",
            "accepted_pose_scope": "source-0 rigid-source screen only",
            "all_pose_qualification": "not_performed",
            "all_nearby_collision_checks": "not_performed",
            "anatomical_admission": False,
            "existing_cvsim_compartment_id": 4,
            "added_physical_mass_kg": 0,
            "added_blood_state_or_force": False,
            "existing_blood_state_owner_mutated": False,
        },
        "source004_plane_failure": (
            _source004_plane_failure(json.loads(SOURCE004_REPORT.read_text()))
            if typed_link == PAIR else {
                "applies": False,
                "historical_report_scope": "526->472 only",
            }),
    }
    if _source_link_scope is not None:
        result["declared_network_link_source_ancestry"] = _source_link_scope
    if not topology_passed:
        result["status"] = "REJECT_SOURCE_CLOSED_SELF_FREE_COMPONENT_GATE"
        result["rejections"] = [
            name + " is not exact-coordinate-quotiented closed, exact-self-free, and single-component"
            for name in ("candidate_bridge", "candidate_target",
                         "baseline_bridge", "baseline_target")
            if surface_reports[name].get("closed_self_free_single_component") is not True]
        return result
    if not ancestry_passed:
        result["status"] = "REJECT_CANDIDATE_BASELINE_SOURCE_ANCESTRY_OR_VOLUME"
        result["rejections"] = ancestry_issues
        return result

    pairs = gate._pair_rows(
        frozen_face_pairs, name="frozen 526/472 source collar ancestry",
        bridge_face_count=len(surfaces["candidate_bridge"].faces),
        target_face_count=len(surfaces["candidate_target"].faces))
    for name, bridge_name, target_name in (
        ("candidate", "candidate_bridge", "candidate_target"),
        ("baseline", "baseline_bridge", "baseline_target"),
    ):
        result["port_interfaces"][name] = _interface_gate(
            verifier, surfaces[bridge_name], surfaces[target_name], pairs,
            declared_vessel_ports=(declared_network_links()
                                   if _source_link_scope is not None else None))

    if not all(item.get("passed") for item in result["port_interfaces"].values()):
        result["status"] = "REJECT_FROZEN_COLLAR_EXACT_INTERFACE_GATE"
        result["rejections"] = [
            name + " does not have exact interface points within its frozen collar face ancestry"
            for name, item in result["port_interfaces"].items()
            if item.get("passed") is not True]
        return result

    candidate_points = result["port_interfaces"]["candidate"][
        "exact_interface_points_world_m"]
    baseline_points = result["port_interfaces"]["baseline"][
        "exact_interface_points_world_m"]
    for role, candidate_name, baseline_name, owner, collar_name in (
        ("bridge", "candidate_bridge", "baseline_bridge", bridge_key[1],
         "bridge_frozen_collar_faces"),
        ("target", "candidate_target", "baseline_target", target_key[1],
         "target_frozen_collar_faces"),
    ):
        collar_faces = result["port_interfaces"]["candidate"][collar_name]
        result["independent_collar_calibre"][role] = _station_calibre(
            surfaces[candidate_name], surfaces[baseline_name],
            candidate_points, collar_faces, role=role, owner_id=owner)
        result["independent_collar_calibre"][role]["baseline_interface_point_sha256"] = (
            result["port_interfaces"]["baseline"]["exact_interface_points_sha256"])
        result["independent_collar_calibre"][role]["baseline_station_witness_point_count"] = (
            len(baseline_points))

    result["solid_gates"]["candidate"] = _solid_gate(
        verifier, manifold3d, surfaces["candidate_bridge"],
        surfaces["candidate_target"], role="candidate")
    result["solid_gates"]["baseline"] = _solid_gate(
        verifier, manifold3d, surfaces["baseline_bridge"],
        surfaces["baseline_target"], role="baseline")
    interface_passed = all(item.get("passed") for item in result["port_interfaces"].values())
    calibre_passed = all(item.get("passed") for item in
                         result["independent_collar_calibre"].values())
    solid_passed = all(item.get("passed") for item in result["solid_gates"].values())
    result["passed"] = bool(topology_passed and ancestry_passed
                            and interface_passed and calibre_passed and solid_passed)
    result["status"] = (
        "PASS_SOURCE_BOUND_3D_JUNCTION_CONTINUITY_SCREEN"
        if result["passed"] else "REJECT_SOURCE_BOUND_3D_JUNCTION_CONTINUITY_SCREEN")
    result["requirements"] = {
        "exact_source_closure_and_self_free_after_exact_seam_quotient": topology_passed,
        "one_connected_referenced_face_component_per_source_vessel": topology_passed,
        "candidate_baseline_source_ancestry_and_volume": ancestry_passed,
        "matched_frozen_collar_ancestry_and_exact_interface_points": interface_passed,
        "positive_overlap_exact_inside_both_and_connected_union": solid_passed,
        "each_vessel_positive_one_loop_at_its_own_collar_station": all(
            item.get("passed") for item in result["independent_collar_calibre"].values()),
    }
    result["rejections"] = [
        f"independent {role} collar section or calibre comparison failed"
        for role, item in result["independent_collar_calibre"].items()
        if item.get("passed") is not True]
    result["rejections"].extend(
        f"{role} candidate/baseline 3D overlap or union proof failed"
        for role, item in result["solid_gates"].items()
        if item.get("passed") is not True)
    return result


def _network_continuity_metric(result: dict, source_scope: dict) -> dict:
    solid = result.get("solid_gates", {})
    interfaces = result.get("port_interfaces", {})
    calibre = result.get("independent_collar_calibre", {})
    return {
        "link_id": source_scope["link_id"],
        "collision_pair_counts_are_not_continuity_metrics": True,
        "source_face_pair_count_is_patch_ancestry_only":
            source_scope["source_face_pair_count"],
        "source_face_pair_ancestry_sha256":
            source_scope["source_face_pair_ancestry_sha256"],
        "observed_exact_interface_points_by_reference": {
            role: value.get("exact_interface_point_count")
            for role, value in interfaces.items()},
        "source_solid_connectivity_inference": {
            "premises": (
                "each original input is closed, exact-self-free, and one "
                "connected surface component; the exact predicate locates "
                "a strict interior point in both bounded solids"),
            "conclusion": (
                "the bounded source solids have nonempty interior intersection "
                "and their set-theoretic union is connected"),
            "based_on_original_surfaces_and_exact_inside_both_witness": True,
            "floating_boolean_union_is_separate_diagnostic": True,
        },
        "candidate_and_independent_baseline_solid_metrics": {
            role: {
                "passed": value.get("passed"),
                "intersection_volume_m3": value.get("intersection_volume_m3"),
                "positive_intersection_volume": value.get(
                    "positive_intersection_volume"),
                "exact_inside_both_witness": value.get(
                    "exact_inside_both_witness"),
                "floating_union_component_count": value.get(
                    "union_component_count"),
                "floating_union_volume_accounting": value.get(
                    "volume_accounting"),
                "floating_boolean_exact_topology_proven": False,
            }
            for role, value in solid.items()},
        "independent_material_station_calibre": {
            role: {
                "passed": value.get("passed"),
                "candidate_to_baseline_ratios": (
                    value.get("candidate_to_baseline_comparison", {})
                    .get("candidate_to_baseline_ratios")),
                "declared_area_ratio": value.get("declared_area_ratio"),
                "declared_minimum_feret_ratio": value.get(
                    "declared_minimum_feret_ratio"),
                "candidate_station_offset_from_own_collar_m": (
                    value.get("station", {}).get(
                        "station_offset_from_anchor_m")),
                "baseline_station_offset_from_own_collar_m": (
                    value.get("baseline_station_transported_by_source_ancestry", {})
                    .get("station_offset_from_anchor_m")),
                "candidate_interface_signed_distance_to_station_m": (
                    value.get("station", {}).get(
                        "junction_witness_signed_distance_to_station_m")),
            }
            for role, value in calibre.items()},
        "calibre_bounds_are_source_geometry_consistency_not_clinical": True,
    }


def audit_brachiocephalic_network_link_3d(
    verifier, manifold3d, *, bridge_key, bridge_surface, target_key, target_surface,
    source_link_evidence: dict, source_rows: Mapping, source_map: Mapping,
    baseline_bridge_surface, baseline_target_surface,
) -> dict:
    """Apply the existing closed-solid and independent station owner to one graph edge."""
    scope = bind_network_link_source_ancestry(
        bridge_key, target_key, source_link_evidence=source_link_evidence,
        source_rows=source_rows, source_map=source_map)
    parent, child = gate._identity(bridge_key), gate._identity(target_key)
    for label, surface, stable_id, expected_sha in (
        ("candidate parent", bridge_surface, parent[1],
         scope["source_parent_faces_sha256"]),
        ("baseline parent", baseline_bridge_surface, parent[1],
         scope["source_parent_faces_sha256"]),
        ("candidate child", target_surface, child[1],
         scope["source_child_faces_sha256"]),
        ("baseline child", baseline_target_surface, child[1],
         scope["source_child_faces_sha256"]),
    ):
        _require(isinstance(surface, (tuple, list)) and len(surface) >= 2,
                 f"{label} has no indexed source faces")
        faces = np.asarray(surface[1])
        _require(faces.ndim == 2 and faces.shape[1] == 3
                 and faces.dtype.kind in "iu"
                 and _sha_bytes(np.ascontiguousarray(faces, dtype="<u4").tobytes())
                 == expected_sha,
                 f"{label} face ancestry differs from source ABI5 row {stable_id}")
    result = audit_brachiocephalic_3d_junction(
        verifier, manifold3d,
        bridge_key=parent, bridge_surface=bridge_surface,
        target_key=child, target_surface=target_surface,
        frozen_face_pairs=scope["source_face_pair_ancestry"],
        baseline_bridge_surface=baseline_bridge_surface,
        baseline_target_surface=baseline_target_surface,
        _source_link_scope=scope)
    result["continuity_metric"] = _network_continuity_metric(result, scope)
    result["continuity_metric"]["passed"] = bool(result.get("passed"))
    return result


def audit_brachiocephalic_network_links_3d(
    verifier, manifold3d, link_inputs,
) -> dict:
    """Fail-first continuity audit over the exact 13 source edges plus two SVC ports."""
    links = declared_network_links()
    _require(isinstance(link_inputs, (tuple, list)) and len(link_inputs) == len(links),
             "network 3D audit requires exactly 15 declared per-link inputs")
    normalized = {}
    for item in link_inputs:
        _require(isinstance(item, dict), "network link input must be an object")
        bridge = gate._identity(item.get("bridge_key"))
        target = gate._identity(item.get("target_key"))
        edge = (bridge, target)
        _require(edge in set(links) and edge not in normalized,
                 f"network 3D audit contains an undeclared or duplicate edge {edge}")
        normalized[edge] = item
    _require(set(normalized) == set(links),
             "network 3D audit input does not cover the exact 13 route edges and two SVC ports")
    results = []
    for edge in links:
        item = normalized[edge]
        result = audit_brachiocephalic_network_link_3d(
            verifier, manifold3d,
            bridge_key=edge[0], target_key=edge[1],
            bridge_surface=item.get("bridge_surface"),
            target_surface=item.get("target_surface"),
            source_link_evidence=item.get("source_link_evidence"),
            source_rows=item.get("source_rows"),
            source_map=item.get("source_map"),
            baseline_bridge_surface=item.get("baseline_bridge_surface"),
            baseline_target_surface=item.get("baseline_target_surface"),
        )
        results.append(result)
        if result.get("passed") is not True:
            return {
                "status": "REJECT_SOURCE_NETWORK_LINK_CONTINUITY",
                "passed": False,
                "requested_link_count": len(links),
                "evaluated_link_count": len(results),
                "first_failed_link": [list(edge[0]), list(edge[1])],
                "links": results,
                "all_target_collision_gate_is_separate": True,
                "anatomical_admission": False,
            }
    return {
        "status": "PASS_SOURCE_NETWORK_LINK_CONTINUITY_UNADMITTED",
        "passed": True,
        "requested_link_count": len(links),
        "evaluated_link_count": len(results),
        "first_failed_link": None,
        "links": results,
        "all_target_collision_gate_is_separate": True,
        "anatomical_admission": False,
        "native_gpu_confirmation_required": True,
    }


def _assert_current_pins(preflight: dict) -> None:
    _require(preflight.get("status") == "PREFLIGHT_PINNED_NO_GEOMETRY_DECODED",
             "preflight has not been source-bound")
    _require(preflight.get("source_repository") == _git_state(),
             "source checkout revision or cleanliness differs from preflight")
    current = _runtime_code_pins()
    _require(current == preflight.get("runtime_code_sha256"),
             "source or runtime code differs from preflight")
    pins = preflight["inputs"]
    for key in ("previous_3d_preflight", "superseded_failed_run_preflight",
                "superseded_failed_run_report", "source_integrity_preparation_report",
                "source_integrity_preparation_script"):
        entry = pins[key]
        path = Path(entry["path"])
        _require(path.is_file() and sha_file(path) == entry["sha256"],
                 "pinned source preparation input changed: " + str(path))
    _require(source007._frozen_input_pins() == pins["source007_input_sha256"],
             "source007 compiler input hash set changed after preflight")
    compiler_preflight = json.loads(Path(
        pins["source007_preflight"]["path"]).read_text())
    current_owner_code_verification = _verify_source007_compiler_code_pins(
        compiler_preflight.get("code_and_dependency_pins", {}))
    _require(current_owner_code_verification
             == pins["source007_compiler_source_code_verification"]
             and compiler_preflight.get("code_and_dependency_pins", {}).get(
                 "source_code_sha256")
             == pins["source007_compiler_source_code_sha256"]
             and compiler_preflight.get("code_and_dependency_pins", {}).get(
                 "source_repository")
             == pins["source007_compiler_repository"],
             "source007 historical compiler source/P6 owner code changed after preflight")
    for key in ("source007_report", "source007_receipt", "source007_preflight",
                "source004_report", "source004_preflight", "source0_pack",
                "source0_receipt", "capture_manifest"):
        entry = pins[key]
        path = Path(entry["path"])
        _require(path.is_file() and sha_file(path) == entry["sha256"],
                 "pinned input changed: " + str(path))
    _require(portdiag._actual_file_hashes(source007.CANDIDATE_DIR)
             == pins["candidate007_files"]
             and portdiag._actual_file_hashes(source007.PREPARATION_DIR)
             == pins["preparation007_files"],
             "source007 emitted candidate/preparation bytes changed")
    for raw_path, expected in preflight["manifold3d_retained_wheel_sha256"].items():
        path = Path(raw_path)
        _require(path.is_file() and sha_file(path) == expected,
                 "retained Manifold wheel changed: " + raw_path)


def run() -> dict:
    _require(PREFLIGHT_PATH.is_file(), "run the no-geometry 3D preflight first")
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    _assert_current_pins(preflight)
    _require(not OUTPUT_DIR.exists(), "retain existing diagnostic output; do not overwrite")
    OUTPUT_DIR.mkdir(parents=True)
    started = time.perf_counter()
    report = {
        "schema": "numi.human.brachiocephalic-3d-junction-execution.v1",
        "status": "RUNNING_CURRENT_SHARED_VERIFIER_3D_JUNCTION_SCREEN",
        "current_shared_verifier_path": str(
            Path(current_shared_verifier.__file__).resolve()),
        "current_shared_verifier_sha256": sha_file(
            Path(current_shared_verifier.__file__).resolve()),
        "recorded_start_utc": datetime.now(UTC).isoformat(),
        "elapsed_seconds": None,
        "scope": preflight["scope"],
        "preflight_path": str(PREFLIGHT_PATH),
        "preflight_sha256": sha_file(PREFLIGHT_PATH),
        "candidate_payload_sha256": preflight["inputs"]["candidate007_files"][
            "resting-thorax.nhanatomy"],
        "preparation_report_sha256": preflight["inputs"]["preparation007_files"][
            "brachiocephalic-common-field-preparation.json"],
        "source004_rejection_sha256": SOURCE004_REPORT_SHA256,
        "source004_plane_failure": preflight["source004_plane_failure"],
        "source_integrity_preparation_report_sha256": SOURCE_INTEGRITY_REPORT_SHA256,
        "source_geometry_mutated": False,
        "native_or_gpu_work_performed": False,
        "source_compiler_invoked": False,
    }
    _write_json(PROGRESS_PATH, report)
    try:
        prepared = _prepare_source007_inputs(preflight)
        candidate_receipt = prepared["candidate_receipt"]
        prep_report = prepared["preparation_report"]
        validation = prepared["validation"]
        verifier = prepared["verifier"]
        identity = prepared["identity"]
        capture_meta = prepared["capture_meta"]
        bridge_surfaces = prepared["bridge_surfaces"]
        target_surfaces = prepared["target_surfaces"]
        bridge_self_check_reuse = prepared["bridge_self_check_reuse"]
        report["source007_bridge_exact_self_check_reuse"] = bridge_self_check_reuse
        pairs_by_port = prepared["pairs_by_port"]
        attribution = prepared["attribution"]
        stationarity = prepared["stationarity"]
        material_branch = prepared["material_branch"]
        report["material_branch_scope"] = material_branch
        candidate_bridge = bridge_surfaces[BRIDGE_KEY]
        candidate_target = target_surfaces[TARGET_KEY]
        baseline_bridge = candidate_bridge
        baseline_target = candidate_target
        report["source007_identity"] = {
            "candidate_receipt_sha256": sha_file(candidate_receipt),
            "preparation_report_sha256": sha_file(prep_report),
            "source_map_525": validation["source_map"].get("525"),
            "source_map_526": validation["source_map"].get("526"),
            "source_map_472": validation["source_map"].get("472"),
            "candidate_baseline_same_source0_geometry": True,
            "body20_pose": _jsonable(capture_meta["body20"]),
            "source0_identity": _jsonable(identity),
            "target_capture_pack_sha256": identity["pack_sha256"],
            "target_capture_receipt_sha256": identity["capture_receipt_sha256"],
            "common_field_map_record_count": validation["common_field"]["map"]["record_count"],
            "existing_cvsim_compartment_id": 4,
            "added_physical_mass_kg": 0,
            "added_blood_state_or_force": False,
        }
        report["input_geometry"] = {
            "bridge_positions_sha256": _sha_bytes(candidate_bridge[0].tobytes()),
            "bridge_faces_sha256": _sha_bytes(candidate_bridge[1].tobytes()),
            "bridge_source_vertex_count": len(candidate_bridge[0]),
            "bridge_source_face_count": len(candidate_bridge[1]),
            "target_positions_sha256": _sha_bytes(candidate_target[0].tobytes()),
            "target_faces_sha256": _sha_bytes(candidate_target[1].tobytes()),
            "target_source_vertex_count": len(candidate_target[0]),
            "target_source_face_count": len(candidate_target[1]),
            "frozen_face_pairs_sha256": _sha_bytes(
                np.ascontiguousarray(pairs_by_port[PAIR], dtype="<i8").tobytes()),
            "frozen_face_pair_count": len(pairs_by_port[PAIR]),
            "candidate_bridge_matches_source007_rigid_only_step0": True,
            "baseline_is_same_source0_geometry_for_this_screen": True,
            "candidate_payload_unchanged": True,
        }
        _write_json(PROGRESS_PATH, report)
        manifold = source007._load_retained_manifold()
        proof = audit_brachiocephalic_3d_junction(
            verifier, manifold,
            bridge_key=BRIDGE_KEY, bridge_surface=candidate_bridge,
            target_key=TARGET_KEY, target_surface=candidate_target,
            frozen_face_pairs=pairs_by_port[PAIR],
            baseline_bridge_surface=baseline_bridge,
            baseline_target_surface=baseline_target)
        proof["material_branch_scope"] = material_branch
        proof["source007_bridge_exact_self_check_reuse"] = bridge_self_check_reuse
        proof["limitations"]["candidate_baseline_same_source0_geometry"] = True
        proof["limitations"]["candidate_baseline_ratios_are_source0_self_comparisons"] = True
        if not bridge_self_check_reuse.get("passed"):
            proof["passed"] = False
            proof["status"] = "REJECT_SOURCE007_APPENDED_BRIDGE_SELF_CHECK_REUSE"
            proof.setdefault("rejections", []).append(
                "pinned source007 exact closure/self-clear checks do not match current source 525/526 geometry")
        report["helper_result"] = _jsonable(proof)
        report["status"] = proof["status"]
        report["source007_frozen_collar_attribution"] = {
            "candidate107_contains_FJ3583": attribution.get(
                "candidate107_payload", {}).get("contains_FJ3583"),
            "stationarity_status": stationarity.get("status"),
            "right_bridge_source_id": 526,
            "target_source_id": 472,
            "frozen_pair_count": len(pairs_by_port[PAIR]),
        }
    except Exception as error:  # noqa: BLE001 - keep every bounded failure on disk
        report["status"] = "DIAGNOSTIC_ABORTED_OR_REJECTED_WITH_ERROR"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["recorded_finish_utc"] = datetime.now(UTC).isoformat()
    report["elapsed_seconds"] = time.perf_counter() - started
    report["source_geometry_mutated"] = False
    report["native_or_gpu_work_performed"] = False
    report["source_compiler_invoked"] = False
    report["candidate_payload_sha256_after"] = sha_file(
        source007.CANDIDATE_DIR / "resting-thorax.nhanatomy")
    report["preparation_report_sha256_after"] = sha_file(
        source007.PREPARATION_DIR / "brachiocephalic-common-field-preparation.json")
    _write_json(REPORT_PATH, report)
    _write_json(PROGRESS_PATH, report)
    return {
        "status": report["status"],
        "report": str(REPORT_PATH),
        "report_sha256": sha_file(REPORT_PATH),
        "elapsed_seconds": report["elapsed_seconds"],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--initialize-only", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        result = write_preflight()
    elif args.initialize_only:
        result = run_initialization_dryrun()
    else:
        result = run()
    print(json.dumps(result, sort_keys=True))
    status = result.get("status", "")
    return 0 if status.startswith((
        "PREFLIGHT_PINNED", "INITIALIZATION_AND_DECODE_PASS_PRE_EXACT_PREDICATES",
        "PASS_SOURCE_BOUND", "REJECT_SOURCE_BOUND", "REJECT_SOURCE_CLOSED",
        "REJECT_CANDIDATE", "REJECT_FROZEN")) else 2


if __name__ == "__main__":
    raise SystemExit(main())

