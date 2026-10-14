"""Source-rigid junction proof preparation for the two left ulnar branches.

This narrowly binds the missing 479→487 and 481→489 material interfaces to the
exact composed ABI5 source and NHRIGID body rests. The preparation phase only
loads, validates, rigid-transforms, and hashes source rows. The separately
guarded source-proof phase derives exact P6 face-pair ancestry and invokes the
existing per-port solid/calibre owner; it is not current-capture admission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import resting_anatomy_verify as shared_verifier
from . import (
    resting_brachiocephalic_current_capture_network_preparation as network_preparation,
)
from . import resting_brachiocephalic_geometry_gate as geometry_gate
from . import resting_brachiocephalic_port_diagnostic as port_diagnostic
from . import resting_brachiocephalic_source007 as source007
from . import resting_pleura_proxy as nhanat_codec
from . import resting_scene

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = source007.ROUTE_DIR
SOURCE_RECEIPT = DATA_ROOT / "brachiocephalic-seam-owner-composition-001/resting-anatomy-receipt.json"
SOURCE_REPORT = DATA_ROOT / "brachiocephalic-seam-owner-composition-001/source-only-composition-report.json"
SOURCE_PAYLOAD = DATA_ROOT / "brachiocephalic-seam-owner-composition-001/resting-thorax.nhanatomy"
SOURCE_CARRIER = DATA_ROOT / "brachiocephalic-seam-owner-composition-001/assets/brachiocephalic-seam-owner-carriers.nhtissue"
SOURCE_RECEIPT_SHA256 = "5ed44e1cfa912f8bf291347aff19c16cf2c55c4b9a36c3ab04fb6aaf17b70474"
SOURCE_REPORT_SHA256 = "b0469defa346664211d8d9af68c4596f63adc7d9b124c8404bfcf5d844516f65"
SOURCE_PAYLOAD_SHA256 = "f83a541b794fe9bae699e6784fe091e147d5bad284847773c803c5309027d361"
SOURCE_CARRIER_SHA256 = "f4f68f0439d78284d6db578119198ae89250facd88b005d467cda6df5352a687"
SOURCE_ROW_COUNT = 525
TRANSFORM_OWNER = REPO / "scripts/predict_brachiocephalic_captured_svc_poses.py"
TRANSFORM_OWNER_SHA256 = "017a90ed6c9eadbdfda9c4d4b31a4966ca10c8dd1c79577686635a296145f7d2"
ISA_TABLE = source007.A / "input-retention-1380/verified-inputs/recovered-isa_element_parts.txt"
ISA_TABLE_SHA256 = "a3de74423f943b0d724ae8f59b3a817f87c423a544f8db98113b1980817cbeaf"
PREPARATION_DIR = DATA_ROOT / "ulnar-source-junction-preparation-001"
PREFLIGHT_PATH = PREPARATION_DIR / "source-rigid-ulnar-junction-preflight.json"
PROGRESS_PATH = PREPARATION_DIR / "source-rigid-ulnar-junction-progress.json"
REPORT_PATH = PREPARATION_DIR / "source-rigid-ulnar-junction-proof.json"

# These are the two named branch links already present in the vascular route
# inventory. Source labels select the exact rows; face ancestry is derived by
# the shared exact oracle at execution time and is never borrowed from another
# radial or brachiocephalic interface.
BRANCH_LINKS = (
    {
        "link_id": "left-brachial-artery-to-left-ulnar-artery",
        "parent_stable_id": 479,
        "child_stable_id": 487,
        "parent_member": "FJ2219",
        "parent_fma": "FMA22692",
        "parent_name": "left brachial artery",
        "parent_body_index": 91,
        "parent_myo_body": "humerus_l",
        "parent_compartment_id": 3,
        "child_member": "FJ2258",
        "child_fma": "FMA22798",
        "child_name": "left ulnar artery",
        "child_body_index": 92,
        "child_myo_body": "ulna_l",
        "child_compartment_id": 3,
    },
    {
        "link_id": "left-medial-brachial-vein-to-left-ulnar-vein",
        "parent_stable_id": 481,
        "child_stable_id": 489,
        "parent_member": "FJ2313",
        "parent_fma": "FMA22936",
        "parent_name": "left medial brachial vein",
        "parent_body_index": 91,
        "parent_myo_body": "humerus_l",
        "parent_compartment_id": 4,
        "child_member": "FJ2260",
        "child_fma": "FMA22952",
        "child_name": "left ulnar vein",
        "child_body_index": 92,
        "child_myo_body": "ulna_l",
        "child_compartment_id": 4,
    },
)

EXPECTED_MEMBER_SHA256 = {
    "FJ2219": "e3c8e2be45d0f25f4e2efa9a046bc13b66c4e43e724a2c2b0424d230719cc6db",
    "FJ2258": "4ddf684432c015ef59cd0bc8cb72b8890632ebde5f2d92e3c1002209ac482b8a",
    "FJ2313": "3707bf67b5959e140fea0eca643bc31db00e97620e63afbe48ed10357af44ec8",
    "FJ2260": "e58b787a03ebd643c3a9651c697f027fa16c415e9e01d190b572c477df7c926f",
}
EXPECTED_COUNTS = {
    479: (1737, 3470),
    487: (3562, 7114),
    481: (1454, 2852),
    489: (2962, 5918),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("ulnar source junction proof: " + message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _required_source_files() -> tuple[Path, ...]:
    return (
        source007.BASE_RECEIPT, source007.BASE_PAYLOAD,
        source007.SOURCE_ARCHIVE, source007.REGISTRATION, source007.RIGID,
        source007.QUOTIENT_REPORT, source007.QUOTIENT_NPZ,
        source007.ROUTE110_NPZ, source007.ROUTE110_REPORT,
        source007.ROUTE110_RUNNER, source007.ROUTE110_CHECKPOINT,
        SOURCE_RECEIPT, SOURCE_REPORT, SOURCE_PAYLOAD, SOURCE_CARRIER,
        ISA_TABLE,
    )


def _source_file_pins() -> dict[str, str]:
    pins = {}
    for path in _required_source_files():
        resolved = path.resolve(strict=True)
        pins[str(resolved)] = _sha_file(resolved)
    expected = {
        **source007.INPUT_SHA,
        str(SOURCE_RECEIPT): SOURCE_RECEIPT_SHA256,
        str(SOURCE_REPORT): SOURCE_REPORT_SHA256,
        str(SOURCE_PAYLOAD): SOURCE_PAYLOAD_SHA256,
        str(SOURCE_CARRIER): SOURCE_CARRIER_SHA256,
        str(ISA_TABLE): ISA_TABLE_SHA256,
    }
    expected = {str(Path(path).resolve()): digest for path, digest in expected.items()
                if Path(path).resolve() in {Path(key) for key in pins}}
    _require(pins == expected, "one or more source, NHRIGID, registration, or compiler input pins changed")
    return pins


def _code_pins() -> dict[str, str]:
    package_root = Path(shared_verifier.__file__).resolve().parent
    paths = set(package_root.glob("*.py"))
    paths.update({
        Path(__file__).resolve(),
        Path(geometry_gate.__file__).resolve(),
        Path(source007.__file__).resolve(),
        Path(nhanat_codec.__file__).resolve(),
        Path(resting_scene.__file__).resolve(),
        source007.MANIFOLD_SO.resolve(),
        source007.MANIFOLD_RECORD.resolve(),
        source007.MANIFOLD_METADATA.resolve(),
        source007.MANIFOLD_WHEEL.resolve(),
        TRANSFORM_OWNER.resolve(),
    })
    return {str(path.resolve(strict=True)): _sha_file(path.resolve())
            for path in sorted(paths, key=lambda item: str(item.resolve()))}


def _expected_isa_rows() -> set[tuple[str, str, str]]:
    return {
        (link[f"{role}_fma"], link[f"{role}_name"], link[f"{role}_member"])
        for link in BRANCH_LINKS for role in ("parent", "child")
    }


def validate_source_branch_identities(source_map: dict, rows: dict) -> dict:
    """Return strict exact row identities for only the two declared branch links."""
    selected = {}
    for link in BRANCH_LINKS:
        for role in ("parent", "child"):
            stable_id = link[f"{role}_stable_id"]
            member = link[f"{role}_member"]
            identity = source_map.get(str(stable_id))
            row = rows.get(stable_id)
            _require(isinstance(identity, dict) and isinstance(row, dict),
                     f"source map or ABI5 table lacks stable row {stable_id}")
            owner = identity.get("source_owner_metadata")
            expected = {
                "source_member": member,
                "source_sha256": EXPECTED_MEMBER_SHA256[member],
                "name": link[f"{role}_name"],
                "body_index": link[f"{role}_body_index"],
                "layer": 2,
                "vascular_compartment_id": link[f"{role}_compartment_id"],
                "owner_concept_id": link[f"{role}_fma"],
                "owner_myo_body": link[f"{role}_myo_body"],
                "owner_core_body_index": link[f"{role}_body_index"],
            }
            actual = {
                "source_member": identity.get("source_member"),
                "source_sha256": identity.get("source_sha256"),
                "name": identity.get("name"),
                "body_index": identity.get("body_index"),
                "layer": identity.get("layer"),
                "vascular_compartment_id": identity.get("vascular_compartment_id"),
                "owner_concept_id": (owner or {}).get("concept_id"),
                "owner_myo_body": (owner or {}).get("myosim_body"),
                "owner_core_body_index": (owner or {}).get("core_body_index"),
            }
            _require(actual == expected,
                     f"typed owner/member/FMA/compartment identity changed for row {stable_id}")
            record = tuple(map(int, row.get("record", ())))
            vertices6 = np.ascontiguousarray(row.get("vertices6"), dtype="<f4")
            faces = np.ascontiguousarray(row.get("faces"), dtype="<u4")
            expected_vertex_count, expected_face_count = EXPECTED_COUNTS[stable_id]
            _require(
                len(record) == 8 and record[0] == link[f"{role}_body_index"]
                and record[2] == expected_vertex_count
                and record[4] == expected_face_count * 3
                and record[5] == stable_id and record[6] == 2 and record[7] == 0
                and vertices6.shape == (expected_vertex_count, 6)
                and faces.shape == (expected_face_count, 3)
                and vertices6.dtype == np.dtype("<f4")
                and faces.dtype == np.dtype("<u4")
                and np.isfinite(vertices6).all()
                and int(faces.min()) >= 0 and int(faces.max()) < len(vertices6),
                f"ABI5 counts, owner, layer, flags, or arrays changed for row {stable_id}",
            )
            selected[str(stable_id)] = {
                "typed_identity": [51011, stable_id],
                "source_member": member,
                "source_member_sha256": identity["source_sha256"],
                "source_fma_id": link[f"{role}_fma"],
                "source_name": identity["name"],
                "core_body_index": int(row["body_index"]),
                "layer": int(row["layer"]),
                "vascular_compartment_id": int(identity["vascular_compartment_id"]),
                "abi5_record_fields": list(record),
                "vertex_count": len(vertices6),
                "face_count": len(faces),
                "local_vertices6_sha256": _sha_bytes(vertices6.tobytes()),
                "local_xyz_sha256": _sha_bytes(vertices6[:, :3].tobytes()),
                "local_normals_sha256": _sha_bytes(vertices6[:, 3:6].tobytes()),
                "faces_sha256": _sha_bytes(faces.tobytes()),
            }
    return selected


def _load_source_rows(*, pins: dict[str, str] | None = None) -> tuple[dict, dict, dict, dict]:
    pins = _source_file_pins() if pins is None else pins
    receipt = json.loads(SOURCE_RECEIPT.read_text())
    composition = json.loads(SOURCE_REPORT.read_text())
    payload = receipt.get("payload", {})
    _require(
        Path(payload.get("path", "")).resolve() == SOURCE_PAYLOAD.resolve()
        and payload.get("sha256") == SOURCE_PAYLOAD_SHA256
        and payload.get("surface_count") == SOURCE_ROW_COUNT,
        "f83 source receipt no longer binds the exact 525-row ABI5 payload",
    )
    provenance = receipt.get("provenance", {})
    _require(provenance.get("rigid_payload_sha256") == source007.RIGID_SHA
             and provenance.get("bodyparts_registration_sha256") == source007.REGISTRATION_SHA,
             "f83 source receipt changed the pinned NHRIGID or body registration lineage")
    expected_outputs = {
        str(SOURCE_RECEIPT.resolve()): SOURCE_RECEIPT_SHA256,
        str(SOURCE_PAYLOAD.resolve()): SOURCE_PAYLOAD_SHA256,
        str(SOURCE_CARRIER.resolve()): SOURCE_CARRIER_SHA256,
    }
    reported_outputs = {
        str(Path(path).resolve()): digest
        for path, digest in composition.get("outputs", {}).items()
    }
    _require(
        composition.get("status") == "SOURCE_ONLY_NOT_LAUNCH_RECEIPT_NOT_ANATOMICAL_ADMISSION"
        and reported_outputs == {
            str(Path(path).resolve()): digest
            for path, digest in expected_outputs.items()
        },
        "f83 source-only composition report does not bind its exact receipt/payload/carrier outputs",
    )
    expected_route_links = {
        (link["parent_stable_id"], link["child_stable_id"])
        for link in BRANCH_LINKS
    }
    actual_route_links = {
        (first[1], second[1])
        for first, second in network_preparation.NETWORK_EDGES
    }
    _require(expected_route_links.issubset(actual_route_links),
             "selected ulnar branch links no longer match the existing vascular route inventory")
    header, records, rows = nhanat_codec._parse_payload(SOURCE_PAYLOAD.read_bytes())
    source_map = provenance.get("source_id_map", {})
    _require(int(header[1]) == 5 and int(header[2]) == SOURCE_ROW_COUNT
             and len(records) == SOURCE_ROW_COUNT
             and set(rows) == {int(value) for value in source_map},
             "f83 payload decode disagrees with its ABI5 source identity table")
    selected = validate_source_branch_identities(source_map, rows)
    isa_lines = {
        tuple(line.split("\t"))
        for line in ISA_TABLE.read_text().splitlines()
        if line and line.count("\t") == 2
    }
    _require(_expected_isa_rows().issubset(isa_lines),
             "recovered BP3D ISA table no longer confirms selected FJ/FMA/name rows")
    return receipt, composition, rows, {
        "source_map": source_map,
        "payload_abi": int(header[1]),
        "payload_surface_count": int(header[2]),
        "selected_rows": selected,
        "pins": pins,
    }


def _registered_rest_surfaces(rows: dict) -> tuple[dict, dict]:
    rigid = resting_scene.load_rigid(source007.RIGID)
    _require(rigid.get("sha256") == source007.RIGID_SHA
             and all(body in rigid.get("core_poses", {}) for body in (91, 92)),
             "exact NHRIGID does not expose both selected registered source bodies")
    _require(TRANSFORM_OWNER.is_file()
             and _sha_file(TRANSFORM_OWNER) == TRANSFORM_OWNER_SHA256,
             "current repository registered-rest transform owner changed")
    spec = importlib.util.spec_from_file_location(
        "_numi_ulnar_source_registered_rest_transform", TRANSFORM_OWNER)
    _require(spec is not None and spec.loader is not None,
             "registered-rest transform owner cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    transform_f32 = getattr(module, "transform_f32", None)
    _require(callable(transform_f32),
             "current repository registered-rest transform owner is missing transform_f32")
    zero = np.zeros(3, dtype="<f4")
    surfaces = {}
    poses = {}
    for link in BRANCH_LINKS:
        for role in ("parent", "child"):
            stable_id = link[f"{role}_stable_id"]
            body = link[f"{role}_body_index"]
            pose = np.asarray(rigid["core_poses"][body], dtype=np.float64)
            _require(pose.shape == (7,) and np.isfinite(pose).all(),
                     f"NHRIGID body {body} rest pose is malformed")
            translation = np.ascontiguousarray(pose[:3], dtype="<f4")
            quaternion = np.ascontiguousarray(pose[3:7], dtype="<f4")
            _require(abs(float(np.linalg.norm(quaternion.astype(np.float64))) - 1.0) <= 2e-5,
                     f"NHRIGID body {body} rest quaternion is not unit length")
            row = rows[stable_id]
            local_xyz = np.ascontiguousarray(row["vertices6"][:, :3], dtype="<f4")
            local_normals = np.ascontiguousarray(row["vertices6"][:, 3:6], dtype="<f4")
            world_xyz = np.ascontiguousarray(
                transform_f32(quaternion, translation, local_xyz), dtype="<f4")
            world_normals = np.ascontiguousarray(
                transform_f32(quaternion, zero, local_normals), dtype="<f4")
            _require(world_xyz.shape == local_xyz.shape
                     and world_normals.shape == local_normals.shape
                     and np.isfinite(world_xyz).all() and np.isfinite(world_normals).all(),
                     f"registered Float32 transform failed for selected row {stable_id}")
            key = (51011, stable_id)
            surfaces[key] = (
                world_xyz,
                np.ascontiguousarray(row["faces"], dtype="<u4"),
                world_normals,
                np.arange(len(world_xyz), dtype="<u8"),
            )
            poses[str(body)] = {
                "core_body_index": body,
                "pose_xyz_quaternion_xyzw_f64": pose.tolist(),
                "position_f32_sha256": _sha_bytes(translation.tobytes()),
                "quaternion_f32_sha256": _sha_bytes(quaternion.tobytes()),
                "registered_world_positions_f32_sha256": _sha_bytes(world_xyz.tobytes()),
                "registered_world_normals_f32_sha256": _sha_bytes(world_normals.tobytes()),
                "source_geometry_only": True,
                "capture_pose_used": False,
            }
    return surfaces, poses


def _preflight_code_and_asset_pins() -> tuple[dict, dict]:
    source_pins = _source_file_pins()
    code_pins = _code_pins()
    return source_pins, code_pins


def build_source_preflight() -> dict:
    _require(not _git_state()["dirty"],
             "commit source proof code before creating an immutable preflight")
    _require(not PREPARATION_DIR.exists(),
             "preparation output exists; preserve it and choose a new sequence")
    source_pins, code_pins = _preflight_code_and_asset_pins()
    _receipt, composition, rows, selected = _load_source_rows(pins=source_pins)
    surfaces, poses = _registered_rest_surfaces(rows)
    source_rows = selected["selected_rows"]
    links = []
    for link in BRANCH_LINKS:
        parent = source_rows[str(link["parent_stable_id"])]
        child = source_rows[str(link["child_stable_id"])]
        links.append({
            "link_id": link["link_id"],
            "parent": parent,
            "child": child,
            "edge_selection": {
                "typed_parent": [51011, link["parent_stable_id"]],
                "typed_child": [51011, link["child_stable_id"]],
                "vascular_route_inventory_binding": "existing 13-edge source route list",
                "source_identity_binding": "FJ member, FMA concept, ABI5 stable ID, core body, layer, and CVSim compartment checked exactly",
                "face_ancestry": "not borrowed; derive from exact registered-source triangle intersections under the existing shared verifier",
                "branch_relation_source": "named arterial or venous branch identities and existing route edge; geometric source contact remains independently unproven until source-proof execution",
            },
        })
    source_poses = {
        str(key): {
            "positions_world_f32_sha256": _sha_bytes(surface[0].tobytes()),
            "faces_u32_sha256": _sha_bytes(surface[1].tobytes()),
            "normals_world_f32_sha256": _sha_bytes(surface[2].tobytes()),
            "vertex_count": len(surface[0]),
            "face_count": len(surface[1]),
        }
        for key, surface in sorted(surfaces.items())
    }
    return {
        "schema": "numi.human.vascular-ulnar-source-junction-preflight.v1",
        "status": "SOURCE_RIGID_REFERENCE_PINNED_NO_GEOMETRY_PREDICATE",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "source_repository": _git_state(),
        "current_shared_geometry_owner_code_sha256": code_pins,
        "source_input_sha256": source_pins,
        "source_payload": {
            "receipt_path": str(SOURCE_RECEIPT.resolve()),
            "receipt_sha256": SOURCE_RECEIPT_SHA256,
            "composition_report_path": str(SOURCE_REPORT.resolve()),
            "composition_report_sha256": SOURCE_REPORT_SHA256,
            "payload_path": str(SOURCE_PAYLOAD.resolve()),
            "payload_sha256": SOURCE_PAYLOAD_SHA256,
            "carrier_path": str(SOURCE_CARRIER.resolve()),
            "carrier_sha256": SOURCE_CARRIER_SHA256,
            "abi": selected["payload_abi"],
            "surface_count": selected["payload_surface_count"],
            "decoded_stable_id_max": max(rows),
            "geometry_owner": "f83 composed source, exact NHRIGID core body transforms 91/92",
            "native_capture_or_pose_used": False,
        },
        "registered_nhrigid_rest_poses": poses,
        "selected_source_row_bindings": source_rows,
        "source_rigid_rows": source_poses,
        "branch_links_in_order": links,
        "source_selection_policy": {
            "arterial_link": "FJ2219/FMA22692 left brachial artery stable 479 to FJ2258/FMA22798 left ulnar artery stable 487",
            "venous_link": "FJ2313/FMA22936 left medial brachial vein stable 481 to FJ2260/FMA22952 left ulnar vein stable 489",
            "no_radial_witness_reuse": True,
            "no_existing_face_pairs_borrowed": True,
            "target_pair_sequence": ["479->487", "481->489"],
            "stop_after_first_failed_link": True,
        },
        "future_source_proof_method": {
            "source_surfaces": "exact f83 ABI5 mesh rows transformed by pinned Float32 owner with NHRIGID core rest poses; candidate and source-rigid baseline are identical for source-only continuity diagnostics",
            "contact_face_pairs": "derive all exact P6 intersecting triangle pairs from the named two source surfaces, then require one connected component under shared exact intersection-point coordinates",
            "continuity_and_calibre": "call existing audit_brachiocephalic_port_pair_continuity once per named link; it independently checks closed self-free inputs, exact interface, positive intersection, exact inside-both point, one numerical union component, and source-station sections",
            "station_source": "existing gate inferred PCA material station from each branch source and exact contact patch; inferred source geometry probe, not a measured centerline",
            "calibre_limit": "area ratio 0.8-1.25 and minimum-Feret ratio 0.9-1.1 against identical source-rigid rows; diagnostic self-reference only, not preservation evidence",
            "csg_limit": "Manifold Mesh64 is a floating Boolean diagnostic, not exact Boolean topology proof",
            "native_capture_or_motion_admission": False,
            "clinical_or_physiological_claim": False,
        },
        "source_only_composition_status": composition.get("status"),
        "source_only_receipt_sha256": _sha_file(SOURCE_RECEIPT),
        "source_only_geometry_predicates_run": False,
        "native_or_gpu_work_performed": False,
        "source_candidate_admitted": False,
        "future_proof_output_dir": str(PREPARATION_DIR.resolve()),
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=False) if not path.parent.exists() else None
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def write_source_preflight() -> dict:
    report = build_source_preflight()
    PREPARATION_DIR.mkdir(parents=True, exist_ok=False)
    _write_json(PREFLIGHT_PATH, report)
    report["preflight_path"] = str(PREFLIGHT_PATH.resolve())
    report["preflight_sha256"] = _sha_file(PREFLIGHT_PATH)
    return report


def _assert_preflight(path: Path) -> dict:
    preflight = json.loads(path.read_text())
    _require(preflight.get("schema") == "numi.human.vascular-ulnar-source-junction-preflight.v1"
             and preflight.get("status")
             == "SOURCE_RIGID_REFERENCE_PINNED_NO_GEOMETRY_PREDICATE",
             "source-rigid preflight schema/status is not executable")
    _require(preflight.get("source_repository") == _git_state()
             and not _git_state()["dirty"],
             "repository revision or source proof code changed after preflight")
    for group in ("current_shared_geometry_owner_code_sha256", "source_input_sha256"):
        for raw_path, expected in preflight.get(group, {}).items():
            pinned = Path(raw_path)
            _require(pinned.is_file() and _sha_file(pinned) == expected,
                     f"pinned source/code input changed: {raw_path}")
    return preflight


def _pair_exact_points(verifier, first, second, pair: tuple[int, int]):
    first_face, second_face = pair
    first_triangle = [
        [Fraction.from_float(float(component)) for component in point]
        for point in first.quotient_positions[
            first.quotient_faces[first_face]].tolist()
    ]
    second_triangle = [
        [Fraction.from_float(float(component)) for component in point]
        for point in second.quotient_positions[
            second.quotient_faces[second_face]].tolist()
    ]
    return tuple(verifier.ci.triangle_intersection_points(first_triangle, second_triangle))


def _contact_components(pair_points: dict[tuple[int, int], tuple]) -> list[list[tuple[int, int]]]:
    """Group exact face-pair contacts by shared exact rational intersection points."""
    pairs = sorted(pair_points)
    parent = {pair: pair for pair in pairs}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parent[max(root_first, root_second)] = min(root_first, root_second)

    by_point = {}
    for pair, points in pair_points.items():
        for point in points:
            by_point.setdefault(tuple(point), []).append(pair)
    for point_pairs in by_point.values():
        for item in point_pairs[1:]:
            union(point_pairs[0], item)
    groups = {}
    for pair in pairs:
        groups.setdefault(find(pair), []).append(pair)
    return [sorted(group) for _root, group in sorted(groups.items())]


def _exact_points_digest(pair_points: dict[tuple[int, int], tuple]) -> tuple[int, str]:
    records = []
    for pair in sorted(pair_points):
        encoded = []
        for point in sorted(set(pair_points[pair])):
            encoded.append([[int(component.numerator), int(component.denominator)]
                            for component in point])
        records.append([list(pair), encoded])
    raw = json.dumps(records, separators=(",", ":")).encode()
    return sum(len(points) for points in pair_points.values()), _sha_bytes(raw)


def run_source_proof(preflight_path: Path) -> dict:
    preflight = _assert_preflight(preflight_path)
    _require(not REPORT_PATH.exists() and not PROGRESS_PATH.exists(),
             "preserve previous proof output and choose a new immutable sequence")
    _source_receipt, _composition, rows, _selected = _load_source_rows(
        pins=preflight["source_input_sha256"])
    surfaces, _poses = _registered_rest_surfaces(rows)
    manifold = source007._load_retained_manifold()
    progress = {
        "schema": "numi.human.brachiocephalic-ulnar-source-proof.v1",
        "status": "RUNNING_SOURCE_RIGID_479_TO_487_THEN_481_TO_489",
        "preflight_path": str(preflight_path.resolve()),
        "preflight_sha256": _sha_file(preflight_path),
        "source_input_sha256": preflight["source_input_sha256"],
        "current_shared_geometry_owner_code_sha256": (
            preflight["current_shared_geometry_owner_code_sha256"]),
        "source_candidate_admitted": False,
        "native_or_gpu_work_performed": False,
        "links": [],
    }
    _write_json(PROGRESS_PATH, progress)
    for link in BRANCH_LINKS:
        parent_key = (51011, link["parent_stable_id"])
        child_key = (51011, link["child_stable_id"])
        item = {
            "link_id": link["link_id"],
            "parent": list(parent_key),
            "child": list(child_key),
            "status": "RUNNING_EXACT_SOURCE_INTERFACE_DERIVATION",
            "source_geometry_frame": "registered NHRIGID source rest",
            "current_capture_used": False,
        }
        progress["links"].append(item)
        _write_json(PROGRESS_PATH, progress)
        try:
            first = geometry_gate._prepare_vessel(
                shared_verifier, parent_key, surfaces[parent_key])
            second = geometry_gate._prepare_vessel(
                shared_verifier, child_key, surfaces[child_key])
            raw_audit = shared_verifier.audit_surface_pair(first, second)
            item["unexempted_source_pair_audit"] = raw_audit
            _write_json(PROGRESS_PATH, progress)
            pairs = tuple(sorted(tuple(map(int, pair))
                                 for pair in raw_audit.get("unexpected_triangle_pairs", [])))
            _require(bool(pairs) and raw_audit.get("triangle_pair_count") == len(pairs),
                     f"source surfaces {parent_key}/{child_key} have no exact contact-pair witness")
            pair_points = {pair: _pair_exact_points(
                shared_verifier, first, second, pair) for pair in pairs}
            _require(all(pair_points.values()),
                     f"one or more exact contact face pairs have no rational intersection point for {link['link_id']}")
            components = _contact_components(pair_points)
            point_count, point_digest = _exact_points_digest(pair_points)
            item["source_interface_derivation"] = {
                "exact_pair_source": "shared verifier audit_surface_pair with no predeclared face-pair exemptions",
                "source_face_pairs": [list(pair) for pair in pairs],
                "source_face_pairs_sha256": _sha_bytes(
                    np.asarray(pairs, dtype="<i8").tobytes()),
                "exact_rational_contact_point_count_with_multiplicity": point_count,
                "exact_rational_contact_point_map_sha256": point_digest,
                "connected_contact_component_count": len(components),
                "connected_contact_components": [
                    {
                        "face_pair_count": len(group),
                        "face_pairs_sha256": _sha_bytes(
                            np.asarray(group, dtype="<i8").tobytes()),
                    } for group in components
                ],
                "single_connected_source_contact_required": True,
                "other_existing_port_face_ids_reused": False,
                "raw_unexempted_pair_audit": raw_audit,
            }
            _write_json(PROGRESS_PATH, progress)
            _require(len(components) == 1,
                     f"source interface for {link['link_id']} has multiple disconnected contact components")
            recorder = port_diagnostic.Recorder(PROGRESS_PATH)
            recorder.bind(progress)
            observed_manifold = port_diagnostic._recording_manifold_proxy(
                manifold, recorder)
            restore = port_diagnostic.install_recording_wrappers(
                geometry_gate, shared_verifier, recorder)
            try:
                result = geometry_gate.audit_brachiocephalic_port_pair_continuity(
                    shared_verifier, observed_manifold,
                    bridge_key=parent_key, bridge_surface=surfaces[parent_key],
                    target_key=child_key, target_surface=surfaces[child_key],
                    frozen_face_pairs=[list(pair) for pair in pairs],
                    baseline_bridge_surface=surfaces[parent_key],
                    baseline_target_surface=surfaces[child_key],
                )
            finally:
                restore()
                item["existing_per_port_owner_trace"] = recorder.data
            item["port_solid_and_calibre_proof"] = result
            item["passed"] = result.get("passed") is True
            item["status"] = (
                "SOURCE_RIGID_JUNCTION_CONTINUITY_SCREEN_PASSED_UNADMITTED"
                if item["passed"] else "SOURCE_RIGID_JUNCTION_CONTINUITY_SCREEN_REJECTED")
            if not item["passed"]:
                item["failure_stage"] = "existing per-port solid/calibre owner"
        except Exception as error:  # noqa: BLE001 - preserve exact source-stage failure
            item["passed"] = False
            item["status"] = "SOURCE_RIGID_JUNCTION_CONTINUITY_SCREEN_REJECTED"
            item["failure_stage"] = "exact interface derivation or existing per-port owner"
            item["failure"] = {"type": type(error).__name__, "message": str(error)}
        _write_json(PROGRESS_PATH, progress)
        if item.get("passed") is not True:
            break
    progress["evaluated_link_count"] = len(progress["links"])
    progress["requested_link_count"] = len(BRANCH_LINKS)
    progress["first_failed_link"] = next(
        ({"link_id": row["link_id"], "parent": row["parent"], "child": row["child"]}
         for row in progress["links"] if row.get("passed") is not True), None)
    progress["passed"] = (
        len(progress["links"]) == len(BRANCH_LINKS)
        and all(row.get("passed") is True for row in progress["links"]))
    progress["status"] = (
        "SOURCE_RIGID_ULNAR_JUNCTIONS_PASSED_UNADMITTED"
        if progress["passed"] else "SOURCE_RIGID_ULNAR_JUNCTIONS_REJECTED")
    progress["source_rigid_calibre_is_self_reference"] = True
    progress["current_capture_or_motion_admission"] = False
    progress["recorded_finish_utc"] = datetime.now(UTC).isoformat()
    _write_json(REPORT_PATH, progress)
    progress["report_path"] = str(REPORT_PATH.resolve())
    progress["report_sha256"] = _sha_file(REPORT_PATH)
    _write_json(PROGRESS_PATH, progress)
    return progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--run-source-proof", action="store_true")
    parser.add_argument("--preflight", type=Path, default=PREFLIGHT_PATH)
    args = parser.parse_args(argv)
    try:
        if args.prepare:
            report = write_source_preflight()
            output = {
                "status": report["status"],
                "preflight_path": str(PREFLIGHT_PATH.resolve()),
                "preflight_sha256": _sha_file(PREFLIGHT_PATH),
                "source_input_sha256": report["source_input_sha256"],
                "source_candidate_admitted": False,
            }
            print(json.dumps(output, sort_keys=True))
            return 0
        report = run_source_proof(args.preflight)
        print(json.dumps({
            "status": report["status"],
            "report_path": report.get("report_path"),
            "report_sha256": report.get("report_sha256"),
            "evaluated_link_count": report.get("evaluated_link_count"),
            "first_failed_link": report.get("first_failed_link"),
            "passed": report.get("passed"),
            "current_capture_or_motion_admission": False,
        }, sort_keys=True))
        return 0 if report.get("passed") else 2
    except Exception as error:  # noqa: BLE001 - retain command-level preparation failures
        print(json.dumps({"status": "PREPARATION_OR_PROOF_FAILED",
                          "error_type": type(error).__name__,
                          "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
