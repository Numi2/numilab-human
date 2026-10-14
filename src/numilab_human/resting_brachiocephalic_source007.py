"""Bounded source-007 brachiocephalic geometry gate.

The command prepares IDs 525/526 through the existing ABI5 append and common
cardiac-map owners, checks the six source ports in frozen order, then delegates
all-AABB source-0 collision checks to the central HP98/P6 verifier. It creates
no GPU asset or new physical state and does not admit anatomy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import struct
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path

import numpy as np

A = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276")
P6_ROOT = Path("/Users/n/numi-human-resting-anatomy-gates-publish-1398")
P6_SOURCE = P6_ROOT / "src"
P6_PACKAGE = P6_SOURCE / "numilab_human"
PREDICTOR_PATH = P6_ROOT / "scripts/predict_brachiocephalic_captured_svc_poses.py"
BASE_RECEIPT = A / "liver-island-native-027-002/resting-anatomy-receipt.json"
BASE_PAYLOAD = A / "liver-island-retirement-candidate-1384-002/resting-thorax.nhanatomy"
SOURCE_ARCHIVE = Path("/Users/n/numi-human-left-pq-publication-1344/Sources/partof_BP3D_4.0_obj_99.zip")
SOURCE_DIR = SOURCE_ARCHIVE.parent
REGISTRATION = A / "input-retention-1380/verified-inputs/053-registration-source-b1b410ad.json"
RIGID = A / "input-retention-1380/verified-inputs/060-myosim-fullbody-core-reference.nhrigid"
QUOTIENT_DIR = A / "brachiocephalic-source-quotient-audit-060-001"
QUOTIENT_REPORT = QUOTIENT_DIR / "report.json"
QUOTIENT_NPZ = QUOTIENT_DIR / "source-rows.npz"
ROUTE_DIR = A / "full-anatomy-delivery-audit-1378"
COLLAR_ANCESTRY = ROUTE_DIR / "bridge-collar-ancestry-107-001/report.json"
COLLAR_ATTRIBUTION = ROUTE_DIR / "bridge-collar-ancestry-107-002/report.json"
COLLAR_REPLAY = ROUTE_DIR / "bridge-collar-ancestry-107-003/verification.json"
COLLAR_STATIONARITY = ROUTE_DIR / "bridge-collar-ancestry-107-004/report.json"
COLLAR_DISTANCES = ROUTE_DIR / "bridge-collar-ancestry-107-001/bridge-port-geodesic-distances-mm.npz"
CAPTURE_MANIFEST = A / "accepted-capture-inputs-1398-001.json"
CANDIDATE_DIR = A / "brachiocephalic-source-candidate-007-001"
PREPARATION_DIR = A / "brachiocephalic-common-field-preparation-007-001"
GATE_DIR = A / "brachiocephalic-source-007-six-port-geometry-001"
PREFLIGHT_PATH = A / "brachiocephalic-source-007-preflight-002.json"
MANIFOLD_ROOT = A / "retained-cpu-reference-deps-1384/manifold3d-3.5.4"
MANIFOLD_SO = MANIFOLD_ROOT / "manifold3d.cpython-313-darwin.so"
MANIFOLD_RECORD = MANIFOLD_ROOT / "manifold3d-3.5.4.dist-info/RECORD"
MANIFOLD_METADATA = MANIFOLD_ROOT / "manifold3d-3.5.4.dist-info/METADATA"
MANIFOLD_WHEEL = MANIFOLD_ROOT / "manifold3d-3.5.4.dist-info/WHEEL"

N27_RECEIPT_SHA = "944bc88d257137f2bedd5898404c45cefd2901d719707519771ae5f4cfce3752"
N27_PAYLOAD_SHA = "cde8705a813fe56686776c9cc038cbf06f413e1de76099d2a019d5e1b748a8d4"
SOURCE_ARCHIVE_SHA = "9fbc713fffeee924a5a657d9813d84d7eb957bded63adb854931dd5e3eb61c97"
REGISTRATION_SHA = "b1b410ad6d4ac8c0c95fd0c3e10f655b24c890d5767a5c377cf78e28ef598f8e"
RIGID_SHA = "2c78cb4150b97cea6e8169dad9e8f5dd59af667b247e07b56e48e857435560e4"
QUOTIENT_REPORT_SHA = "f189f18efcbfa2953374caf5c3b11af90b135f972764713953de3360bbe140da"
QUOTIENT_NPZ_SHA = "02df2dba7d698d8e22ca7b6493d1b0b97d1afa00b06150b39d1e0a7990e3e385"
COLLAR_SHA = {
    "ancestry": "e72864fb11923322a39e6a924481fc3928e060fe029d7baa68ca1740c82a544a",
    "attribution": "72e3bddc19a9ca404678db3a2a274031b82a0d01981fa9addbcc01776d37e94d",
    "replay": "e0120f894560890b1c3ddf373ff417a23f1ecc1139fd5546fa25516869a0be1f",
    "stationarity": "014f7d41e6e4e1079bc390c365e9af13d1be9f5410deb499821df7847dd457e7",
    "distances": "5f0a714e88691f3c4f230e7e214d0ed34e6014b222ce58a943635ae2d0e38806",
}
CAPTURE_MANIFEST_SHA = "3df61c8bf202e343a9124588fa94be17e9876db17d1a6ef5dd5fa615aaa7b815"
STEP0_PACK_SHA = "a70cbdc2e42b97a38fd6afa46598f023ae705bbd36686dcebf1840399295c1fd"
STEP0_RECEIPT_SHA = "b7847f3970694f770d4e36d1c72dd8f8bdea77840df352b9cffa84bd025ae7f2"
P6_VERIFIER_SHA = "87d21c53c4ac003fa8df7410cc681d91a90465af808217d8af4561155d5094bb"
PREDICTOR_SHA = "017a90ed6c9eadbdfda9c4d4b31a4966ca10c8dd1c79577686635a296145f7d2"
MANIFOLD_SHA = {
    str(MANIFOLD_SO): "e76ec87bcff071b06890ae5c2e6ba4906d46e208153c0fddb9d896ae731b3255",
    str(MANIFOLD_RECORD): "c267afae79ef8cf617be33e2323bbd47a6b7eaca2bff2c31d90a4c818aafe8a8",
    str(MANIFOLD_METADATA): "9437700ed7e21b85cc42dc5d2afeb1db0941f648eb9b86a86bbe38d0d5571635",
    str(MANIFOLD_WHEEL): "28da2d8db1df847adb203b5d00c77612dce84174458902bb3064630a0ef337c8",
}
CAPTURE_STEPS = (0, 3807, 4767, 5023, 5599, 6207, 7423, 8000)
BRIDGE_ROWS = {
    525: {"member": "FJ3482", "ports": (10, 469, 473)},
    526: {"member": "FJ3583", "ports": (10, 468, 472)},
}
BRIDGE_KEYS = ((51011, 525), (51011, 526))
PORT_KEYS = tuple(
    (bridge_key, (51011, stable_id))
    for bridge_key in BRIDGE_KEYS
    for stable_id in BRIDGE_ROWS[bridge_key[1]]["ports"]
)
REQUIRED_SOURCE_TARGETS = ((51011, 7), (51011, 10), (51011, 468),
                           (51011, 469), (51011, 472), (51011, 473))
REQUIRED_SKIN_KEY = (51007, 1)

INPUT_SHA = {
    str(BASE_RECEIPT): N27_RECEIPT_SHA,
    str(BASE_PAYLOAD): N27_PAYLOAD_SHA,
    str(SOURCE_ARCHIVE): SOURCE_ARCHIVE_SHA,
    str(REGISTRATION): REGISTRATION_SHA,
    str(RIGID): RIGID_SHA,
    str(QUOTIENT_REPORT): QUOTIENT_REPORT_SHA,
    str(QUOTIENT_NPZ): QUOTIENT_NPZ_SHA,
    str(COLLAR_ANCESTRY): COLLAR_SHA["ancestry"],
    str(COLLAR_ATTRIBUTION): COLLAR_SHA["attribution"],
    str(COLLAR_REPLAY): COLLAR_SHA["replay"],
    str(COLLAR_STATIONARITY): COLLAR_SHA["stationarity"],
    str(COLLAR_DISTANCES): COLLAR_SHA["distances"],
    str(CAPTURE_MANIFEST): CAPTURE_MANIFEST_SHA,
    str(PREDICTOR_PATH): PREDICTOR_SHA,
    str(P6_PACKAGE / "resting_anatomy_verify.py"): P6_VERIFIER_SHA,
    **MANIFOLD_SHA,
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("source-007 brachiocephalic gate: " + message)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                     ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _code_paths() -> list[Path]:
    root = Path(__file__).resolve().parents[2]
    source = root / "src/numilab_human"
    paths = [
        Path(__file__).resolve(),
        source / "resting_brachiocephalic_geometry_gate.py",
        source / "resting_brachiocephalic_common_field.py",
        source / "resting_brachiocephalic_pose_predictor.py",
        source / "resting_major_vessels.py",
        source / "resting_passive_viscera.py",
        source / "organ_family_geometry.py",
        source / "resting_pleura_proxy.py",
        source / "resting_anatomy_interface_patch.py",
        PREDICTOR_PATH,
    ]
    paths.extend(sorted(P6_PACKAGE.glob("*.py")))
    return sorted({path.resolve() for path in paths})


def _git_revision() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True).stdout.splitlines()
        return {"head": revision, "dirty": bool(status), "status_lines": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"head": None, "dirty": None, "error": str(error)}


def _frozen_input_pins() -> dict:
    pins = {}
    for raw_path, expected in INPUT_SHA.items():
        path = Path(raw_path)
        _require(path.is_file(), "missing pinned input " + str(path))
        actual = sha_file(path)
        _require(actual == expected,
                 f"pinned input hash changed at {path}: expected {expected}, got {actual}")
        pins[str(path)] = actual
    return pins


def _code_and_dependency_pins() -> dict:
    code = {}
    for path in _code_paths():
        _require(path.is_file(), "source owner or P6 verifier code is missing: " + str(path))
        code[str(path)] = sha_file(path)
    return {
        "source_repository": _git_revision(),
        "source_code_sha256": code,
        "manifold3d_retained_wheel": dict(MANIFOLD_SHA),
        "manifold3d_version": "3.5.4",
        "manifold3d_analysis_precision": "Mesh64 Float64 for derived CSG sections; floating analysis only",
    }


def build_source007_preflight() -> dict:
    """Pin the bounded run inputs without decoding any geometry arrays."""
    input_pins = _frozen_input_pins()
    code_pins = _code_and_dependency_pins()
    _require(not CANDIDATE_DIR.exists() and not PREPARATION_DIR.exists()
             and not GATE_DIR.exists(),
             "candidate007/prep007/gate output already exists")
    _require(not PREFLIGHT_PATH.exists(),
             "retain an existing source-007 preflight; do not overwrite")
    receipt = json.loads(BASE_RECEIPT.read_text())
    _require(receipt.get("payload", {}).get("sha256") == N27_PAYLOAD_SHA
             and Path(receipt["payload"]["path"]).resolve() == BASE_PAYLOAD.resolve(),
             "pinned direct N27 receipt/payload binding changed")
    source_ids = receipt.get("provenance", {}).get("source_id_map", {})
    _require(len(source_ids) == 523 and max(map(int, source_ids)) == 524,
             "N27 receipt source IDs do not establish max+1 IDs 525/526")
    with BASE_PAYLOAD.open("rb") as stream:
        header_bytes = stream.read(struct.calcsize("<8sIIIIII"))
    _require(len(header_bytes) == struct.calcsize("<8sIIIIII"),
             "pinned N27 NHANAT header is truncated")
    magic, abi, surface_count, *_ = struct.unpack("<8sIIIIII", header_bytes)
    _require(magic == b"NHANAT1\0" and abi == 5 and surface_count == 523,
             "pinned N27 payload header is not NHANAT ABI5 with 523 rows")
    manifest = json.loads(CAPTURE_MANIFEST.read_text())
    capture_rows = manifest.get("capture_rows", [])
    _require(tuple(int(row.get("step", -1)) for row in capture_rows) == CAPTURE_STEPS
             and capture_rows[0].get("recorded_pack_sha256") == STEP0_PACK_SHA
             and capture_rows[0].get("receipt_sha256") == STEP0_RECEIPT_SHA,
             "accepted capture input manifest or source-0 pins changed")
    source0_pack = Path(capture_rows[0]["pack_path"])
    source0_receipt = Path(capture_rows[0]["receipt_path"])
    _require(source0_pack.is_file() and sha_file(source0_pack) == STEP0_PACK_SHA
             and source0_receipt.is_file() and sha_file(source0_receipt) == STEP0_RECEIPT_SHA,
             "accepted source-0 pack or receipt bytes differ from their frozen hashes")
    return {
        "schema": "numi.human.brachiocephalic-source-007-preflight.v1",
        "status": "PREFLIGHT_PINNED_NO_GEOMETRY_DECODED",
        "source_repository": code_pins["source_repository"],
        "input_sha256": input_pins,
        "code_and_dependency_pins": code_pins,
        "n27_header": {
            "magic": magic.rstrip(b"\0").decode("ascii"),
            "abi": int(abi),
            "surface_count": int(surface_count),
            "max_stable_id_from_receipt_source_map": 524,
            "next_stable_ids": [525, 526],
        },
        "candidate007_path": str(CANDIDATE_DIR),
        "prep007_path": str(PREPARATION_DIR),
        "geometry_gate_path": str(GATE_DIR),
        "outputs_absent": True,
        "accepted_source0_pack_path": str(source0_pack),
        "accepted_source0_pack_sha256": sha_file(source0_pack),
        "accepted_source0_receipt_path": str(source0_receipt),
        "accepted_source0_receipt_sha256": sha_file(source0_receipt),
        "captured_pack_bytes_hashed_but_not_decoded": True,
        "nhanat_vertex_or_face_arrays_loaded": False,
        "geometry_predicates_run": False,
        "anatomical_admission": False,
    }


def write_source007_preflight() -> dict:
    report = build_source007_preflight()
    _write_json(PREFLIGHT_PATH, report)
    report["report_path"] = str(PREFLIGHT_PATH)
    report["report_sha256"] = sha_file(PREFLIGHT_PATH)
    return report


def _load_central_p6():
    _require(P6_PACKAGE.is_dir(), "pinned HP98/P6 verifier package is missing")
    package_name = "_numi_human_hp98_p6"
    package = types.ModuleType(package_name)
    package.__path__ = [str(P6_PACKAGE)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    return importlib.import_module(package_name + ".resting_anatomy_verify")


def _load_predictor_transform():
    spec = importlib.util.spec_from_file_location("_numi_human_source007_pose", PREDICTOR_PATH)
    _require(spec is not None and spec.loader is not None,
             "pinned source transform owner cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _require(callable(getattr(module, "transform_f32", None)),
             "pinned transform_f32 owner is missing")
    return module.transform_f32


def _load_retained_manifold():
    _require(MANIFOLD_ROOT.is_dir(), "retained Manifold3D 3.5.4 package is missing")
    if str(MANIFOLD_ROOT) not in sys.path:
        sys.path.insert(0, str(MANIFOLD_ROOT))
    importlib.invalidate_caches()
    module = importlib.import_module("manifold3d")
    _require(Path(module.__file__).resolve() == MANIFOLD_SO.resolve(),
             "Manifold3D did not load from the pinned retained wheel")
    _require(callable(getattr(module.Manifold, "to_mesh64", None)),
             "retained Manifold3D lacks Mesh64 analysis API")
    metadata = MANIFOLD_METADATA.read_text()
    _require("\nVersion: 3.5.4\n" in "\n" + metadata + "\n",
             "retained Manifold3D metadata version differs from 3.5.4")
    return module


def _read_port_pairs() -> tuple[dict, dict, dict]:
    ancestry = json.loads(COLLAR_ANCESTRY.read_text())
    attribution = json.loads(COLLAR_ATTRIBUTION.read_text())
    replay = json.loads(COLLAR_REPLAY.read_text())
    stationarity = json.loads(COLLAR_STATIONARITY.read_text())
    _require(ancestry.get("current_nha_sha256") == N27_PAYLOAD_SHA,
             "collar ancestry is not tied to exact N27")
    _require(replay.get("status") == "PASS",
             "independent frozen collar replay did not pass")
    _require(stationarity.get("status") == "PASS_SOURCE_PORT_SUPPORT_FIXED",
             "frozen source port stationarity report did not pass")
    candidate = attribution.get("candidate107_payload", {})
    _require(candidate.get("contains_FJ3482") is True
             and candidate.get("contains_FJ3583") is False,
             "candidate107 left/right bridge attribution changed")
    member_reports = {row.get("member"): row for row in ancestry.get("members", [])}
    _require(set(member_reports) == {"FJ3482", "FJ3583"},
             "frozen collar report does not bind both source bridge rows")
    expected = {}
    for stable_id, item in BRIDGE_ROWS.items():
        member = item["member"]
        report = member_reports[member]
        _require(int(report.get("proposed_stable_id", -1)) == stable_id,
                 "frozen collar stable ID changed for " + member)
        _require(report.get("current_attachment_frame") ==
                 "Body20 local; exact retained NHRIGID default body20 pose",
                 "frozen collar body frame changed for " + member)
        ports = report.get("ports", {})
        _require(set(ports) == {str(value) for value in item["ports"]},
                 "frozen collar port set changed for " + member)
        for target_id in item["ports"]:
            port = ports[str(target_id)]
            pairs = port.get("exact_face_pairs")
            _require(isinstance(pairs, list) and len(pairs) > 0
                     and len(pairs) == int(port.get("pair_count", -1)),
                     f"frozen exact face-pair witness is malformed for {member}/{target_id}")
            _require(port.get("source_sha256") is not None
                     and port.get("target_registered_default_world_xyz_float32_sha256"),
                     f"frozen pair provenance is incomplete for {member}/{target_id}")
            expected[((51011, stable_id), (51011, target_id))] = pairs
    _require(set(expected) == set(PORT_KEYS),
             "frozen collar report does not provide exactly six source interfaces")
    return expected, attribution, stationarity


def _step0_source(manifest: dict, av) -> tuple[dict, dict, dict]:
    rows = manifest.get("capture_rows", [])
    _require(tuple(int(row.get("step", -1)) for row in rows) == CAPTURE_STEPS,
             "accepted source manifest is not the exact eight-capture set")
    selected = [row for row in rows if int(row["step"]) == 0]
    _require(len(selected) == 1, "source-0 accepted capture is missing or duplicated")
    capture = selected[0]
    pack_path, receipt_path = Path(capture["pack_path"]), Path(capture["receipt_path"])
    _require(capture.get("recorded_pack_sha256") == STEP0_PACK_SHA
             and capture.get("receipt_sha256") == STEP0_RECEIPT_SHA
             and sha_file(pack_path) == STEP0_PACK_SHA
             and sha_file(receipt_path) == STEP0_RECEIPT_SHA,
             "source-0 accepted pack or receipt differs from the frozen capture input")
    surfaces, identity = av.load_capture(pack_path, receipt_path)
    _require(identity.get("accepted_step") == 0
             and abs(float(identity.get("accepted_time_s", -1)) - 0.0) <= 1e-12,
             "loaded source-0 capture does not identify accepted step/time zero")
    _require(identity.get("pack_sha256") == STEP0_PACK_SHA
             and identity.get("capture_receipt_sha256") == STEP0_RECEIPT_SHA,
             "central P6 load_capture identity differs from pinned source-0 inputs")
    receipt = json.loads(receipt_path.read_text())
    poses = {int(row["body_index"]): row for row in receipt.get("accepted_body_poses", [])}
    _require(20 in poses, "source-0 accepted receipt has no body20 pose")
    return surfaces, identity, {"capture": capture, "receipt": receipt, "body20": poses[20]}


def _validate_source_candidate(receipt_path: Path, prep_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    preparation = json.loads(prep_path.read_text())
    payload_path = Path(receipt["payload"]["path"])
    _require(payload_path.is_file() and sha_file(payload_path) == receipt["payload"]["sha256"],
             "source candidate receipt does not bind its NHA payload")
    _require(receipt["payload"]["sha256"] == preparation.get("input_candidate_payload_sha256")
             and preparation.get("anatomy_payload_modified") is False,
             "common-field prep changed the NHA source payload")
    _require(preparation.get("appended_record_count") == 651
             and preparation.get("map_prefix_byte_exact") is True
             and preparation.get("volume_polynomials_and_domain_boxes_byte_preserved") is True
             and preparation.get("input_common_map_record_count") == 620718,
             "prepared common field differs from exact two-row append")
    _require(preparation.get("existing_cvsim_compartment_id") == 4
             and preparation.get("added_physical_mass_kg") == 0
             and preparation.get("additional_physical_state_or_force") is False,
             "source rows added vascular state, mass, or force")
    parent = preparation.get("original_523_row_byte_preservation", {})
    _require(parent.get("all_523_existing_rows_byte_exact") is True
             and parent.get("receipt_sha256") == N27_RECEIPT_SHA
             and parent.get("payload_sha256") == N27_PAYLOAD_SHA,
             "candidate does not retain exact N27 parent geometry")
    ranges = preparation.get("ranges", [])
    _require([int(row.get("stable_id", -1)) for row in ranges] == [525, 526],
             "common-field appended stable IDs differ from max(N27)+1")
    extension = receipt.get("provenance", {}).get(
        "passive_brachiocephalic_vein_extension", {})
    _require(extension.get("base_receipt_sha256") == N27_RECEIPT_SHA
             and extension.get("base_payload_sha256") == N27_PAYLOAD_SHA
             and receipt.get("provenance", {}).get("rigid_payload_sha256") == RIGID_SHA
             and receipt.get("provenance", {}).get("bodyparts_registration_sha256") == REGISTRATION_SHA,
             "appended NHA candidate is not directly source-bound to exact N27/rest inputs")
    source_map = receipt.get("provenance", {}).get("source_id_map", {})
    for stable_id, spec in BRIDGE_ROWS.items():
        identity = source_map.get(str(stable_id), {})
        _require(identity.get("source_member") == spec["member"]
                 and int(identity.get("body_index", -1)) == 20
                 and int(identity.get("vascular_compartment_id", -1)) == 4
                 and int(identity.get("layer", -1)) == 2,
                 "source ID/body/layer/CVSim binding changed for " + spec["member"])
    common = receipt.get("provenance", {}).get("cardiac_geometry_binding", {}).get("common_field", {})
    _require(common.get("anatomy_payload_sha256") == receipt["payload"]["sha256"]
             and int(common.get("map", {}).get("record_count", -1)) == 621369
             and int(common.get("map", {}).get("record_stride_bytes", -1)) == 112,
             "final NHA receipt lacks appended 621369-record common-field map")
    prep_owner = receipt.get("provenance", {}).get(
        "passive_brachiocephalic_common_field_preparation", {})
    _require(prep_owner.get("appended_range_stable_ids") == [525, 526]
             and prep_owner.get("additional_physical_mass_kg") == 0
             and prep_owner.get("additional_physical_state_or_force") is False,
             "common-field receipt ownership or passive-state declaration changed")
    return {"receipt": receipt, "preparation": preparation,
            "payload_path": payload_path, "source_map": source_map,
            "common_field": common}


def _port_target_surfaces(surfaces: dict, rows: dict) -> dict:
    missing = sorted(set(REQUIRED_SOURCE_TARGETS) - set(surfaces))
    _require(not missing, "source-0 pack lacks mandatory actual target identities: " + str(missing))
    for semantic, stable_id in REQUIRED_SOURCE_TARGETS:
        target_positions, target_faces = surfaces[(semantic, stable_id)][:2]
        row = rows[stable_id]
        _require(np.array_equal(np.asarray(target_faces, dtype="<u4"),
                                np.asarray(row["faces"], dtype="<u4")),
                 f"accepted source-0 face order differs from exact N27 row {stable_id}")
        _require(np.asarray(target_positions).dtype == np.dtype("<f4"),
                 f"source-0 target {stable_id} is not packed Float32 XYZ")
    _require(REQUIRED_SKIN_KEY in surfaces,
             "source-0 geometry lacks the exact skin identity needed for declared enclosure policy")
    return {key: surfaces[key] for key in REQUIRED_SOURCE_TARGETS}


def _bridge_surfaces(rows: dict, body20_pose: dict, transform_f32) -> dict:
    q = np.asarray(body20_pose.get("quaternion_xyzw"), dtype="<f4")
    t = np.asarray(body20_pose.get("position_m"), dtype="<f4")
    _require(q.shape == (4,) and t.shape == (3,)
             and np.isfinite(q).all() and np.isfinite(t).all(),
             "source-0 accepted body20 rigid pose is malformed")
    output = {}
    for stable_id in BRIDGE_ROWS:
        row = rows[stable_id]
        _require(int(row["body_index"]) == 20 and int(row["layer"]) == 2,
                 "source-0 bridge row has wrong NHA owner")
        local_xyz = np.ascontiguousarray(row["vertices6"][:, :3], dtype="<f4")
        faces = np.ascontiguousarray(row["faces"], dtype="<u4")
        world = transform_f32(q, t, local_xyz)
        _require(world.shape == local_xyz.shape and np.isfinite(world).all(),
                 "source-0 body20 transform produced invalid bridge coordinates")
        output[(51011, stable_id)] = (
            world, faces, None, np.arange(len(world), dtype=np.uint64))
    return output


def _source_closed_checks(av, bridges: dict) -> list[dict]:
    output = []
    for key in BRIDGE_KEYS:
        try:
            positions, faces = bridges[key][:2]
            prepared = av.prepare_surface(
                key, positions, faces, source_vertex_ids=bridges[key][3])
            self_result = prepared.self_audit()
            row = {
                "identity": list(key),
                "positions_sha256": hashlib.sha256(prepared.positions.tobytes()).hexdigest(),
                "faces_sha256": hashlib.sha256(prepared.faces.tobytes()).hexdigest(),
                "topology": prepared.diagnostics(),
                "closed_oriented_source": bool(prepared.closed),
                "exact_self_intersection_count": int(self_result["count"]),
                "exact_self_clear": int(self_result["count"]) == 0,
            }
        except Exception as error:  # noqa: BLE001 - preserve per-row gate failure evidence
            row = {
                "identity": list(key),
                "passed": False,
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        output.append(row)
        if not row.get("closed_oriented_source", False) or not row.get("exact_self_clear", False):
            break
    return output


def _predict_step0_bridges(
    rows: dict, rigid_bridges: dict, captured: dict, body20_pose: dict,
    preparation: dict, transform_f32,
) -> tuple[dict, dict]:
    """Predict step-0 bridge poses from the accepted capture's SVC10 motion."""
    from .resting_brachiocephalic_pose_predictor import (
        predict_brachiocephalic_world_positions,
    )

    svc_key = (51011, 10)
    _require(svc_key in captured, "accepted source-0 capture lacks SVC10")
    svc_native = np.ascontiguousarray(captured[svc_key][0], dtype="<f4")
    svc_faces = np.ascontiguousarray(captured[svc_key][1], dtype="<u4")
    svc_row = rows[10]
    _require(np.array_equal(svc_faces, np.ascontiguousarray(svc_row["faces"], dtype="<u4")),
             "captured SVC10 face order differs from source-map ancestry")
    q = np.asarray(body20_pose.get("quaternion_xyzw"), dtype="<f4")
    t = np.asarray(body20_pose.get("position_m"), dtype="<f4")
    svc_rigid = transform_f32(
        q, t, np.ascontiguousarray(svc_row["vertices6"][:, :3], dtype="<f4"))
    _require(svc_rigid.shape == svc_native.shape,
             "same-pose rigid-only and captured SVC10 vertex arrays differ")
    predicted = {}
    details = {}
    bridge_records = preparation.get("bridges", {})
    for stable_id, spec in BRIDGE_ROWS.items():
        key = (51011, stable_id)
        _require(key in rigid_bridges, "rigid-only source bridge is missing " + spec["member"])
        bridge_record = bridge_records.get(spec["member"], {})
        ancestry = bridge_record.get("vertex_ancestry")
        _require(isinstance(ancestry, list) and len(ancestry) == len(rigid_bridges[key][0]),
                 "compiled common-map vertex ancestry is missing for " + spec["member"])
        result = predict_brachiocephalic_world_positions(
            rigid_bridges[key][0], svc_native, svc_rigid, ancestry)
        positions = np.ascontiguousarray(result["positions_world_m"], dtype="<f4")
        _require(positions.shape == rigid_bridges[key][0].shape
                 and np.isfinite(positions).all(),
                 "captured-SVC predictor returned invalid bridge coordinates")
        predicted[key] = (
            positions, rigid_bridges[key][1], None, rigid_bridges[key][3])
        details[spec["member"]] = {
            "stable_id": stable_id,
            "vertex_count": len(positions),
            "vertex_ancestry_count": len(ancestry),
            "positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
            "svc_native_world_positions_sha256": hashlib.sha256(svc_native.tobytes()).hexdigest(),
            "svc_rigid_only_world_positions_sha256": hashlib.sha256(svc_rigid.tobytes()).hexdigest(),
            "captured_minus_rigid_svc_displacement_sha256": hashlib.sha256(
                result["captured_svc_displacement_world_m"].tobytes()).hexdigest(),
            "svc_faces_sha256": hashlib.sha256(svc_faces.tobytes()).hexdigest(),
            "projected_displacement_sha256": hashlib.sha256(
                result["projected_bridge_displacement_world_m"].tobytes()).hexdigest(),
            "prediction_status": result["status"],
            "native_field_error_bound": result["native_field_error_bound"],
            "float32_roundoff_bound_max_m":
                result["float32_interpolation_roundoff_bound_max_m"],
            "float32_roundoff_observed_max_m":
                result["float32_interpolation_roundoff_observed_max_m"],
            "float32_roundoff_bound_scope": result["float32_bound_scope"],
            "native_confirmation": result["native_confirmation"],
            "source_geometry_mutated": False,
        }
    return predicted, details


def _decorate_same_source_calibre(result: dict) -> dict:
    result["calibre_interpretation"] = (
        "same-source geometry sanity only: source-0 candidate and comparison meshes "
        "share the exact source faces and positions; ratio 1 is not independent "
        "physical calibre preservation, calibration, or clinical evidence")
    for port in result.get("ports", []):
        port["calibre_interpretation"] = result["calibre_interpretation"]
        gate = port.get("calibre_gate", {})
        gate["same_source_geometry_sanity_only"] = True
        gate["independent_physical_calibration"] = False
    return result


def run_source007() -> dict:
    report_path = GATE_DIR / "source-007-geometry-report.json"
    _require(not GATE_DIR.exists(), "retain any existing source-007 gate output; do not overwrite")
    report = {
        "schema": "numi.human.brachiocephalic-source-007-geometry-gate.v1",
        "status": "PREFLIGHT",
        "source_candidate": str(CANDIDATE_DIR),
        "common_field_preparation": str(PREPARATION_DIR),
        "source_geometry_gate": str(GATE_DIR),
        "anatomical_admission": False,
        "all_pose_admission": False,
        "native_admission": False,
        "native_confirmation": "required",
        "limits": {
            "source0_scope": "rigid-only source closure and six-port checks are separate from the step-0 captured-SVC displacement prediction; the all-nearby P6 target screen uses predicted coupled bridge coordinates",
            "later_pose_scope": "the other seven accepted-pose predictions and exact P6 target checks remain pending",
            "physical_calibre": "not established; source-identical baseline ratios are geometry sanity only",
            "floating_union": "Manifold3D CSG and Mesh64 sections are numerical analysis, not exact union topology or emitted assets",
            "captured_field_prediction": "step-0 bridge displacement reuses actual captured SVC10 minus same-pose rigid-only NHA SVC10 through prepared barycentric ancestry; its roundoff bound covers interpolation only and does not bound native map/kernel error",
            "native_common_field": "not evaluated; native 16-pose confirmation remains required",
            "overall_anatomy": "rigid-source closure/six-port plus step-0 predicted all-target checks are bounded regional screens, not Human completion",
        },
    }
    _write_json(GATE_DIR / "progress.json", report)
    stage = "pinned-input-preflight"
    try:
        report["input_sha256"] = _frozen_input_pins()
        report["code_and_dependency_pins"] = _code_and_dependency_pins()
        _require(not CANDIDATE_DIR.exists() and not PREPARATION_DIR.exists(),
                 "candidate007 or prep007 already exists; retain it and choose a reviewed successor")
        base_receipt = json.loads(BASE_RECEIPT.read_text())
        _require(base_receipt.get("payload", {}).get("sha256") == N27_PAYLOAD_SHA
                 and Path(base_receipt["payload"]["path"]).resolve() == BASE_PAYLOAD.resolve(),
                 "pinned direct N27 receipt/payload binding changed")
        _write_json(GATE_DIR / "progress.json", report)

        stage = "direct-abi5-source-append"
        report["status"] = "RUNNING_DIRECT_ABI5_SOURCE_APPEND"
        _write_json(GATE_DIR / "progress.json", report)
        from .resting_major_vessels import append_brachiocephalic_veins
        appended = append_brachiocephalic_veins(
            base_receipt=BASE_RECEIPT,
            output=CANDIDATE_DIR,
            sources_path=SOURCE_DIR,
            registration_path=REGISTRATION,
            rigid_path=RIGID,
            quotient_npz_path=QUOTIENT_NPZ,
            quotient_report_path=QUOTIENT_REPORT,
        )
        candidate_receipt_path = CANDIDATE_DIR / "resting-anatomy-receipt.json"
        _require(candidate_receipt_path.is_file()
                 and appended.get("receipt", {}).get("path") == str(candidate_receipt_path),
                 "existing ABI5 append owner did not emit candidate007 receipt")
        report["candidate007_files_sha256"] = {
            str(path.relative_to(CANDIDATE_DIR)): sha_file(path)
            for path in sorted(CANDIDATE_DIR.rglob("*")) if path.is_file()
        }
        report["status"] = "SOURCE_APPEND_COMPLETE"
        _write_json(GATE_DIR / "progress.json", report)

        stage = "common-cardiac-map-source-preparation"
        report["status"] = "RUNNING_SOURCE_BOUND_COMMON_MAP_PREPARATION"
        _write_json(GATE_DIR / "progress.json", report)
        from .resting_brachiocephalic_common_field import (
            prepare_brachiocephalic_common_field,
        )
        prepared = prepare_brachiocephalic_common_field(
            candidate_receipt_path=candidate_receipt_path,
            output_dir=PREPARATION_DIR,
            collar_ancestry_path=COLLAR_ANCESTRY,
            collar_attribution_path=COLLAR_ATTRIBUTION,
            collar_replay_path=COLLAR_REPLAY,
            collar_stationarity_path=COLLAR_STATIONARITY,
            port_distances_path=COLLAR_DISTANCES,
        )
        prep_receipt_path = PREPARATION_DIR / "resting-anatomy-receipt.json"
        prep_report_path = PREPARATION_DIR / "brachiocephalic-common-field-preparation.json"
        _require(prep_receipt_path.is_file() and prep_report_path.is_file()
                 and prepared.get("status") == "SOURCE_PREPARATION_ONLY_NOT_ADMISSION",
                 "existing common-field owner did not emit source-prep007")
        report["candidate007_files_sha256"] = {
            str(path.relative_to(CANDIDATE_DIR)): sha_file(path)
            for path in sorted(CANDIDATE_DIR.rglob("*")) if path.is_file()
        }
        report["prep007_files_sha256"] = {
            str(path.relative_to(PREPARATION_DIR)): sha_file(path)
            for path in sorted(PREPARATION_DIR.rglob("*")) if path.is_file()
        }
        report["preparation_result"] = prepared
        report["status"] = "SOURCE_AND_COMMON_MAP_PREPARATION_COMPLETE"
        _write_json(GATE_DIR / "progress.json", report)

        stage = "source-and-map-identity-validation"
        validated = _validate_source_candidate(prep_receipt_path, prep_report_path)
        source_payload_path = validated["payload_path"]
        from .resting_pleura_proxy import _parse_payload
        _, _, rows = _parse_payload(source_payload_path.read_bytes())
        _require(set(rows) == set(map(int, validated["source_map"])),
                 "decoded source-007 rows differ from exact source_id_map")
        _require(max(rows) == 526 and {525, 526}.issubset(rows),
                 "decoded source-007 stable IDs are not max(N27)+1=525/526")
        bridge_source_records = {}
        for stable_id, spec in BRIDGE_ROWS.items():
            row = rows[stable_id]
            prepared_row = validated["preparation"]["bridges"][spec["member"]]
            xyz_hash = hashlib.sha256(
                np.ascontiguousarray(row["vertices6"][:, :3], dtype="<f4").tobytes()).hexdigest()
            face_hash = hashlib.sha256(
                np.ascontiguousarray(row["faces"], dtype="<i4").tobytes()).hexdigest()
            _require(xyz_hash == prepared_row.get("source_local_xyz_sha256")
                     and face_hash == prepared_row.get("source_faces_sha256"),
                     "decoded bridge row differs from source-060/common-map record " + spec["member"])
            source_identity = validated["source_map"][str(stable_id)]
            bridge_source_records[spec["member"]] = {
                "stable_id": stable_id,
                "raw_source_zip_sha256": SOURCE_ARCHIVE_SHA,
                "raw_source_member_sha256": source_identity.get("source_sha256"),
                "source_local_xyz_sha256": xyz_hash,
                "source_faces_sha256": face_hash,
                "vertex_count": len(row["vertices6"]),
                "triangle_count": len(row["faces"]),
                "body_index": int(row["body_index"]),
                "vascular_compartment_id": 4,
            }

        stage = "source-0-central-capture-load"
        report["status"] = "LOADING_PINNED_SOURCE0_WITH_HP98_P6"
        _write_json(GATE_DIR / "progress.json", report)
        av = _load_central_p6()
        manifest = json.loads(CAPTURE_MANIFEST.read_text())
        captured, capture_identity, capture_meta = _step0_source(manifest, av)
        _port_target_surfaces(captured, rows)
        skin_container_keys = tuple(sorted(key for key in captured if key[0] == 51007))
        _require(REQUIRED_SKIN_KEY in skin_container_keys,
                 "source-0 pack does not contain required skin identity (51007,1)")
        transform_f32 = _load_predictor_transform()
        bridges = _bridge_surfaces(rows, capture_meta["body20"], transform_f32)
        report["source0"] = {
            "capture_manifest_path": str(CAPTURE_MANIFEST),
            "capture_manifest_sha256": CAPTURE_MANIFEST_SHA,
            "accepted_step": int(capture_identity["accepted_step"]),
            "accepted_time_s": float(capture_identity["accepted_time_s"]),
            "pack_path": capture_meta["capture"]["pack_path"],
            "pack_sha256": capture_identity["pack_sha256"],
            "receipt_path": capture_meta["capture"]["receipt_path"],
            "receipt_sha256": capture_identity["capture_receipt_sha256"],
            "body20_position_m": [float(x) for x in capture_meta["body20"]["position_m"]],
            "body20_quaternion_xyzw": [float(x) for x in capture_meta["body20"]["quaternion_xyzw"]],
            "native_surface_count": len(captured),
            "native_semantic_counts": {
                str(key): int(value) for key, value in sorted(Counter(k[0] for k in captured).items())
            },
            "target_universe": "every typed anatomical surface returned by pinned HP98/P6 load_capture for this exact accepted step-0 pack",
            "mandatory_source_targets": [list(key) for key in REQUIRED_SOURCE_TARGETS],
            "skin_enclosure_identities": [list(key) for key in skin_container_keys],
            "source_geometry_screen": "rigid-only body20 transformed source bridge coordinates; not a coupled pose gate",
            "bridge_rigid_source_positions_sha256": {
                str(sid): hashlib.sha256(bridges[(51011, sid)][0].tobytes()).hexdigest()
                for sid in sorted(BRIDGE_ROWS)
            },
            "same_pose_policy": "captured target rows and source-bridge body20 transforms use the same accepted step-0 receipt",
        }
        report["source_candidate"] = {
            "receipt_path": str(prep_receipt_path),
            "receipt_sha256": sha_file(prep_receipt_path),
            "payload_path": str(source_payload_path),
            "payload_sha256": sha_file(source_payload_path),
            "prep_report_path": str(prep_report_path),
            "prep_report_sha256": sha_file(prep_report_path),
            "common_map_sha256": validated["common_field"]["map"]["sha256"],
            "stable_ids": [525, 526],
            "existing_cvsim_compartment_id": 4,
            "added_physical_mass_kg": 0,
            "additional_physical_state_or_force": False,
            "original_523_rows_byte_exact": True,
            "bridge_source_records": bridge_source_records,
        }
        _write_json(GATE_DIR / "progress.json", report)

        stage = "p6-source-closure-and-exact-self"
        report["status"] = "CHECKING_SOURCE_BRIDGE_CLOSURE_AND_EXACT_SELF"
        _write_json(GATE_DIR / "progress.json", report)
        report["source_closed_self_checks"] = _source_closed_checks(av, bridges)
        _write_json(GATE_DIR / "progress.json", report)
        _require(len(report["source_closed_self_checks"]) == 2
                 and all(row.get("closed_oriented_source") is True
                         and row.get("exact_self_clear") is True
                         for row in report["source_closed_self_checks"]),
                 "source closure/self gate stopped at the first failing bridge")

        stage = "six-port-overlap-and-sections"
        report["status"] = "CHECKING_SIX_SOURCE_PORTS_STOP_ON_FIRST_FAILURE"
        _write_json(GATE_DIR / "progress.json", report)
        port_pairs, attribution, stationarity = _read_port_pairs()
        report["source0"]["frozen_collar_evidence"] = {
            "independent_replay_status": "PASS",
            "candidate107_contains_FJ3482": bool(
                attribution.get("candidate107_payload", {}).get("contains_FJ3482")),
            "candidate107_contains_FJ3583": bool(
                attribution.get("candidate107_payload", {}).get("contains_FJ3583")),
            "source_port_support_status": stationarity.get("status"),
            "source_interface_count": len(port_pairs),
            "scope": "source ancestry and port-support evidence only; not pose or continuity admission",
        }
        _write_json(GATE_DIR / "progress.json", report)
        port_target_keys = sorted({pair[1] for pair in PORT_KEYS})
        candidate_ports = {key: captured[key] for key in port_target_keys}
        candidate_ports.update(bridges)
        baseline_ports = {key: captured[key] for key in port_target_keys}
        baseline_ports.update(bridges)
        from .resting_brachiocephalic_geometry_gate import (
            audit_brachiocephalic_six_port_continuity,
        )
        manifold = _load_retained_manifold()
        six_port = audit_brachiocephalic_six_port_continuity(
            av, manifold, candidate_ports, baseline_ports, port_pairs)
        _decorate_same_source_calibre(six_port)
        report["six_port_source_geometry"] = six_port
        _write_json(GATE_DIR / "progress.json", report)
        _require(six_port.get("passed") is True
                 and six_port.get("requested_port_count") == 6
                 and six_port.get("evaluated_port_count") == 6,
                 "source-0 six-port continuity/calibre screen stopped or failed")
        report["source0"]["rigid_only_six_port_screen"] = {
            "status": "PASSED_SOURCE_GEOMETRY_ONLY",
            "scope": "closed source bridge against frozen source interfaces; same-source calibre ratios are geometry sanity only",
            "candidate107_left_bridge_present": bool(
                attribution.get("candidate107_payload", {}).get("contains_FJ3482")),
            "right_bridge_scope": "frozen source-060 baseline ancestry; candidate107 does not contain FJ3583",
            "station_method": "P6 exact port witnesses plus deterministic PCA axis oriented toward each source-segment interior; inferred source station, not measured centerline",
            "physical_calibre_preservation_claim": False,
        }
        report["status"] = "RIGID_ONLY_SIX_PORT_SOURCE_SCREEN_COMPLETE"
        _write_json(GATE_DIR / "progress.json", report)

        stage = "step0-captured-svc-displacement-prediction"
        report["status"] = "PREDICTING_STEP0_BRIDGES_FROM_CAPTURED_SVC_DISPLACEMENT"
        _write_json(GATE_DIR / "progress.json", report)
        predicted_bridges, prediction_details = _predict_step0_bridges(
            rows, bridges, captured, capture_meta["body20"], validated["preparation"],
            transform_f32)
        prediction_path = GATE_DIR / "predicted-step0-bridge-positions.npz"
        with prediction_path.open("wb") as stream:
            np.savez_compressed(stream, **{
                f"stable_{stable_id}_xyz_f32": predicted_bridges[(51011, stable_id)][0]
                for stable_id in sorted(BRIDGE_ROWS)
            })
        report["source0"]["captured_svc_displacement_prediction"] = {
            "status": "OFFLINE_CAPTURED_SVC_DISPLACEMENT_PREDICTION_ONLY",
            "method": "same accepted source-0 receipt: captured packed SVC10 world coordinates minus same-pose rigid-only NHA SVC10, interpolated through each compiled bridge vertex ancestry and quintic fade",
            "bridges": prediction_details,
            "prediction_arrays_path": str(prediction_path),
            "prediction_arrays_sha256": sha_file(prediction_path),
            "native_field_error_bound": "not_available",
            "float32_roundoff_scope": "interpolation of supplied Float32 captured coordinates and pinned ancestry only; no bound on native common-map reconstruction, polynomial evaluation, or GPU kernel arithmetic",
            "native_confirmation": "required",
            "source_geometry_mutated": False,
        }
        _write_json(GATE_DIR / "progress.json", report)

        stage = "all-aabb-source0-predicted-collisions"
        report["status"] = "CHECKING_PREDICTED_BRIDGES_AGAINST_ALL_DECODED_SOURCE0_TARGETS_WITH_HP98_P6"
        _write_json(GATE_DIR / "progress.json", report)
        from .resting_brachiocephalic_geometry_gate import (
            audit_captured_pose_surface_intersections,
        )
        bridge_positions = {key: value[0] for key, value in predicted_bridges.items()}
        bridge_faces = {key: value[1] for key, value in predicted_bridges.items()}
        collision = audit_captured_pose_surface_intersections(
            av, captured, bridge_positions, bridge_faces, port_pairs)
        report["all_aabb_source0_collision_gate"] = collision
        verifier_rows = collision.get("verifier_result", {}).get("rows", [])
        nearby = [row for row in verifier_rows if row.get("stage") == "nearby_pair"]
        report["all_aabb_source0_pair_audit"] = {
            "passed": bool(collision.get("surface_intersection_stage_passed")),
            "central_verifier": "HP98/P6 audit_changed_surfaces; no duplicate triangle-intersection or enclosure predicate",
            "aabb_pair_selection": "all decoded targets were supplied to P6; its existing source-0 AABB routing selected pair predicates",
            "bridge_coordinates": "step-0 captured-SVC offline prediction using the existing predictor and compiled per-vertex common-map ancestry; not native coordinates",
            "decoded_target_count": len(captured),
            "nearby_pair_count": len(nearby),
            "nearby_pairs": [
                {"identities": row.get("identities"),
                 "triangle_pair_count": int(row.get("triangle_pair_count", 0)),
                 "passed": bool(row.get("passed"))}
                for row in nearby
            ],
            "semantic_counts": report["source0"]["native_semantic_counts"],
            "skin_enclosure_policy": {
                "declared_container_identities": [list(key) for key in skin_container_keys],
                "declared_for_each_bridge": True,
                "owner": "HP98/P6 intended_enclosures role-aware contract",
                "containment_allowed": True,
                "boundary_crossings_still_rejected": True,
                "same_accepted_source0_pose": True,
                "bridge_pose_source": "captured SVC10 displacement predictor; native common-map error is not bounded",
            },
        }
        _require(collision.get("surface_intersection_stage_passed") is True,
                 "source-0 all-AABB target collision screen rejected a bridge")
        report["status"] = "SOURCE0_RIGID_SOURCE_AND_PREDICTED_ALL_AABB_SCREENS_PASSED_UNADMITTED"
        report["anatomical_admission"] = False
        report["all_pose_admission"] = False
        report["native_admission"] = False
        report["acceptance_gates_remaining"] = [
            "the other seven accepted-pose offline common-field projections and exact P6 checks",
            "native common-map and bridge deformation confirmation",
            "remaining route-wide artery, muscle, and other Human integration gates",
        ]
        report["candidate007_files_sha256"] = {
            str(path.relative_to(CANDIDATE_DIR)): sha_file(path)
            for path in sorted(CANDIDATE_DIR.rglob("*")) if path.is_file()
        }
        report["prep007_files_sha256"] = {
            str(path.relative_to(PREPARATION_DIR)): sha_file(path)
            for path in sorted(PREPARATION_DIR.rglob("*")) if path.is_file()
        }
        report["geometry_gate_code_sha256"] = sha_file(Path(__file__).resolve())
        report["status_detail"] = (
            "Source-0 bridge and target geometry gates passed only; no later-pose, "
            "native common-field, physiological, or whole-Human admission is asserted.")
        _write_json(report_path, report)
        return report
    except Exception as error:  # noqa: BLE001 - write durable stage failure before returning
        report["status"] = "SOURCE007_REJECTED_OR_INCOMPLETE_AT_" + stage.upper().replace("-", "_")
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
        report["failed_stage"] = stage
        report["anatomical_admission"] = False
        report["all_pose_admission"] = False
        report["native_admission"] = False
        _write_json(report_path, report)
        _write_json(GATE_DIR / "progress.json", report)
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true",
                      help="hash pins and read NHA/capture headers only; do not load geometry")
    mode.add_argument("--run-source007", action="store_true",
                      help="execute append, source prep and bounded source-0 geometry gates")
    args = parser.parse_args(argv)
    if args.preflight_only:
        result = write_source007_preflight()
        print(json.dumps({
            "status": result.get("status"),
            "report": result.get("report_path"),
            "report_sha256": result.get("report_sha256"),
            "geometry_predicates_run": result.get("geometry_predicates_run"),
        }, sort_keys=True))
        return 0
    result = run_source007()
    print(json.dumps({
        "status": result.get("status"),
        "report": str(GATE_DIR / "source-007-geometry-report.json"),
        "failed_stage": result.get("failed_stage"),
        "failure": result.get("failure"),
    }, sort_keys=True))
    return 0 if result.get("status") == "SOURCE0_RIGID_SOURCE_AND_PREDICTED_ALL_AABB_SCREENS_PASSED_UNADMITTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
