"""Offline all-capture transfer gate for appended brachiocephalic source rows.

Consumes the existing ABI5 append compiler, accepted packs, common cardiac map
predictor, and shared P6 geometry owners. This module does not compile meshes,
modify source coordinates, launch native/GPU work, or claim anatomy admission.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import struct
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import resting_anatomy_verify as shared_verifier
from . import resting_brachiocephalic_geometry_gate as gate
from . import resting_brachiocephalic_port_diagnostic as portdiag
from . import resting_brachiocephalic_source007 as source007
from . import resting_pleura_proxy as nhanat_codec

A = source007.A
GATE_DIR = source007.GATE_DIR
JUNCTION_OWNER_PATH = Path(__file__).with_name("resting_brachiocephalic_3d_junction.py")
RUN_DIR = GATE_DIR / "multipose-transfer-gate-001"
PREFLIGHT_PATH = GATE_DIR / "multipose-transfer-gate-preflight-003.json"
PROGRESS_PATH = RUN_DIR / "progress.json"
REPORT_PATH = RUN_DIR / "multipose-transfer-gate-report.json"
INITIALIZATION_PATH = GATE_DIR / "multipose-transfer-initialization-003.json"

# The accepted N27 packs refer to this source payload. It is intentionally
# separate from the current append compiler input, which can include the
# source-fragment repair while retaining this exact ancestry parent.
N27_CAPTURE_SOURCE = (
    A / "liver-island-retirement-candidate-1384-002/resting-thorax.nhanatomy"
)
N27_CAPTURE_RECEIPT = N27_CAPTURE_SOURCE.with_name("resting-anatomy-receipt.json")
N27_CAPTURE_SOURCE_SHA256 = (
    "cde8705a813fe56686776c9cc038cbf06f413e1de76099d2a019d5e1b748a8d4"
)

LATE_DIR = (
    A / "full-anatomy-delivery-audit-1378/intercostal-late-settled-preparation-028-002"
)
LATE_INPUT = LATE_DIR / "late-capture-input.json"
LATE_PREPARATION_REPORT = LATE_DIR / "report.json"
LATE_SOURCE_BINDING = LATE_DIR / "source-binding-preparation.json"
LATE_COVERAGE_ADDENDUM = LATE_DIR / "coverage-addendum.json"
REPAIR_REPORT = (
    A / "intercostal-source-fragment-repair-028-003/source-fragment-repair-report.json"
)

SOURCE0_PORT_REPORT = (
    GATE_DIR
    / "port-525-526-target-collar-diagnostic-004/six-port-section-diagnostic.json"
)
SOURCE0_PORT_REPORT_SHA256 = (
    "354876d040f774d8acf18071779289ee9cfc2ce50137650087ecf2f099399700"
)
SOURCE0_LAST_PORT_REPORT = (
    GATE_DIR
    / "port-526-472-3d-junction-diagnostic-002/3d-junction-continuity-report.json"
)
SOURCE0_LAST_PORT_REPORT_SHA256 = (
    "02f7305e53b6aadf8bd06d48fdb0c1a83165b98a09616c059f42368b7fb668ef"
)

CAPTURE_STEPS = source007.CAPTURE_STEPS
TRANSFER_STEP = 14975
TRANSFER_TIME_S = 29.950001422548667
DEFORMABLE_OWNER = 0xFFFFFFFF
ROUTE_TARGET_IDS = frozenset((7, 10, 468, 469, 472, 473))
REPAIRED_IDS = frozenset((313, 315, 317))
SOURCE_BOUND_TYPED_KEYS = frozenset(
    {(51011, stable_id) for stable_id in ROUTE_TARGET_IDS}
    | {(51010, stable_id) for stable_id in REPAIRED_IDS}
)
SOURCE_IDENTITY_FIELDS = (
    "source_member",
    "source_sha256",
    "body_index",
    "layer",
    "name",
    "provider",
    "source_owner_metadata",
    "vascular_compartment_id",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("brachiocephalic transfer gate: " + message)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)


def _git_state() -> dict:
    repo = Path(__file__).resolve().parents[2]
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"head": head, "dirty": bool(status), "status_lines": status}


def _code_paths(verifier=None, retention_owner=None) -> list[Path]:
    paths = [
        Path(__file__).resolve(),
        JUNCTION_OWNER_PATH.resolve(),
        Path(source007.__file__).resolve(),
        Path(gate.__file__).resolve(),
        Path(portdiag.__file__).resolve(),
        Path(shared_verifier.__file__).resolve(),
        Path(nhanat_codec.__file__).resolve(),
        Path(source007.PREDICTOR_PATH).resolve(),
        Path(__file__).with_name("resting_intercostal_mesh_arrangement.py").resolve(),
    ]
    if verifier is not None:
        paths.append(Path(verifier.__file__).resolve())
    if retention_owner is not None:
        paths.append(Path(retention_owner.__file__).resolve())
    return sorted(set(paths), key=str)


def _file_pins(paths) -> dict[str, str]:
    return {
        str(Path(path).resolve()): sha_file(Path(path))
        for path in sorted(set(map(Path, paths)), key=str)
    }


def _capture_rows(manifest_path: Path) -> list[dict]:
    manifest = json.loads(manifest_path.read_text())
    rows = manifest.get("capture_rows", [])
    _require(
        tuple(int(row.get("step", -1)) for row in rows) == CAPTURE_STEPS,
        "the eight pinned N27 capture rows changed",
    )
    return rows


def _late_capture_row() -> tuple[dict, dict]:
    manifest = json.loads(LATE_INPUT.read_text())
    source = manifest.get("source", {})
    rows = manifest.get("capture_rows", [])
    _require(
        len(rows) == 1 and int(rows[0].get("step", -1)) == TRANSFER_STEP,
        "late N22 transfer input must contain only accepted step 14975",
    )
    _require(
        abs(float(manifest.get("accepted_time_s", TRANSFER_TIME_S)) - TRANSFER_TIME_S)
        <= 1e-9
        or "accepted_time_s" not in manifest,
        "late N22 transfer time changed",
    )
    _require(
        source.get("path") and source.get("sha256"),
        "late N22 source payload identity is missing",
    )
    return source, rows[0]


def _source_candidate_paths() -> tuple[Path, Path, Path]:
    return (
        source007.PREPARATION_DIR / "resting-anatomy-receipt.json",
        source007.PREPARATION_DIR / "brachiocephalic-common-field-preparation.json",
        source007.CANDIDATE_DIR / "resting-thorax.nhanatomy",
    )


def _all_input_paths() -> list[Path]:
    rows = _capture_rows(source007.CAPTURE_MANIFEST)
    late_source, late_capture = _late_capture_row()
    receipt_path, prep_path, candidate_path = _source_candidate_paths()
    paths = [
        source007.CAPTURE_MANIFEST,
        source007.BASE_RECEIPT,
        source007.BASE_PAYLOAD,
        N27_CAPTURE_SOURCE,
        N27_CAPTURE_RECEIPT,
        receipt_path,
        prep_path,
        candidate_path,
        source007.SOURCE_ARCHIVE,
        source007.REGISTRATION,
        source007.RIGID,
        source007.QUOTIENT_REPORT,
        source007.QUOTIENT_NPZ,
        source007.COLLAR_ANCESTRY,
        source007.COLLAR_ATTRIBUTION,
        source007.COLLAR_REPLAY,
        source007.COLLAR_STATIONARITY,
        source007.COLLAR_DISTANCES,
        LATE_INPUT,
        LATE_PREPARATION_REPORT,
        LATE_SOURCE_BINDING,
        LATE_COVERAGE_ADDENDUM,
        REPAIR_REPORT,
        SOURCE0_PORT_REPORT,
        SOURCE0_LAST_PORT_REPORT,
    ]
    paths.extend(
        Path(row[key]) for row in rows for key in ("pack_path", "receipt_path")
    )
    paths.extend(
        (
            Path(late_source["path"]),
            Path(late_capture["pack_path"]),
            Path(late_capture["receipt_path"]),
        )
    )
    paths.extend(Path(path) for path in source007.MANIFOLD_SHA)
    paths.extend(Path(path) for path in source007.INPUT_SHA)
    return sorted({path.resolve() for path in paths}, key=str)


def build_preflight(arrangement_candidates: dict[int, dict] | None = None) -> dict:
    """Hash all inputs and code without decoding geometry or invoking predicates."""
    _require(
        not RUN_DIR.exists() and not PREFLIGHT_PATH.exists(),
        "retain an existing multipose attempt; do not overwrite it",
    )
    _require(
        JUNCTION_OWNER_PATH.is_file(),
        "the pinned closed-solid 3D junction owner is not integrated yet",
    )
    verifier = source007._load_central_p6()
    required = ("load_capture", "audit_changed_surfaces", "prepare_surface")
    missing = [name for name in required if not callable(getattr(verifier, name, None))]
    _require(
        not missing,
        "pinned external HP98 P6 verifier lacks required owners: " + ", ".join(missing),
    )
    _require(
        callable(getattr(shared_verifier, "retain_bound_source_faces", None))
        and callable(getattr(shared_verifier, "_pack_sections", None)),
        "pinned local shared verifier lacks retained-face or pack-section owner",
    )
    candidate_receipt, prep_path, candidate_path = _source_candidate_paths()
    validation = source007._validate_source_candidate(candidate_receipt, prep_path)
    _require(
        validation["payload_path"].resolve() == candidate_path.resolve(),
        "source candidate receipt does not resolve to its pinned output",
    )
    capture_rows = _capture_rows(source007.CAPTURE_MANIFEST)
    late_source, late_row = _late_capture_row()
    late_source_path = Path(late_source["path"])
    _require(
        sha_file(late_source_path) == late_source["sha256"],
        "late N22 source payload differs from its transfer manifest",
    )
    _require(
        sha_file(N27_CAPTURE_SOURCE) == N27_CAPTURE_SOURCE_SHA256,
        "original N27 capture-source payload changed",
    )
    _require(
        sha_file(SOURCE0_PORT_REPORT) == SOURCE0_PORT_REPORT_SHA256
        and sha_file(SOURCE0_LAST_PORT_REPORT) == SOURCE0_LAST_PORT_REPORT_SHA256,
        "frozen source-0 port continuity evidence changed",
    )
    late_report = json.loads(LATE_PREPARATION_REPORT.read_text())
    late_addendum = json.loads(LATE_COVERAGE_ADDENDUM.read_text())
    binding = json.loads(LATE_SOURCE_BINDING.read_text())
    _require(
        late_report.get("candidate_admission") is False
        and late_addendum.get("actual_capture_steps") == [TRANSFER_STEP]
        and binding.get("n22_source_sha256") == late_source["sha256"],
        "late preparation scope or N22 source binding changed",
    )
    aggregate = validation["receipt"].get("provenance", {}).get(
        "local_intercostal_mesh_repair", {})
    repair_rows = aggregate.get("rows", [])
    _require(
        [int(row.get("stable_id", -1)) for row in repair_rows] == sorted(REPAIRED_IDS),
        "source candidate receipt lacks the exact three-row arrangement provenance",
    )
    arrangement_preflight = {}
    arrangement_paths = [
        Path(aggregate["repair_parent_receipt_path"]),
        Path(aggregate["repair_parent_payload_path"]),
        Path(aggregate["repair_report_path"]),
    ]
    if arrangement_candidates:
        _require(
            set(arrangement_candidates) == REPAIRED_IDS,
            "explicit arrangement preflight must cover exactly rows 313/315/317",
        )
    for record in repair_rows:
        stable_id = int(record["stable_id"])
        descriptor = record.get("full_candidate", {})
        _require(
            isinstance(descriptor, dict)
            and descriptor.get("source_payload_sha256")
            == "50ab8248d2e0f55dd6cb8b24afc34c8844061d5841ecdf89d7daf03288816bfd"
            and descriptor.get("source_payload_index") == 1,
            f"candidate row {stable_id} has the wrong full_candidate parent namespace",
        )
        normalized = {
            key: descriptor.get(key) for key in (
                "candidate_path", "candidate_sha256", "lineage_path", "lineage_sha256",
                "report_path", "report_sha256", "source_payload_sha256",
                "source_payload_index",
            )
        }
        if arrangement_candidates:
            supplied = arrangement_candidates[stable_id]
            for key, value in normalized.items():
                _require(supplied.get(key) == value,
                         f"explicit arrangement preflight descriptor differs for {stable_id}/{key}")
        for path_key, sha_key in (
            ("candidate_path", "candidate_sha256"),
            ("lineage_path", "lineage_sha256"),
            ("report_path", "report_sha256"),
        ):
            path = Path(normalized[path_key]).resolve()
            expected = normalized[sha_key]
            _require(path.is_file() and sha_file(path) == expected,
                     f"retained arrangement {path_key} path/hash differs for {stable_id}")
            arrangement_paths.append(path)
        arrangement_preflight[str(stable_id)] = normalized
    input_paths = _all_input_paths() + arrangement_paths
    code_paths = sorted(
        set(_code_paths(verifier, shared_verifier) + source007._code_paths()), key=str
    )
    source_map = validation["source_map"]
    _require(
        len(source_map)
        == len(validation["receipt"].get("provenance", {}).get("source_id_map", {})),
        "validated candidate source identity map is incomplete",
    )
    return {
        "schema": "numi.human.brachiocephalic-multipose-transfer-preflight.v1",
        "status": "PREFLIGHT_PINNED_NO_GEOMETRY_DECODED",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_repository": _git_state(),
        "runtime_code_sha256": _file_pins(code_paths),
        "input_sha256": _file_pins(input_paths),
        "capture_manifest": {
            "path": str(source007.CAPTURE_MANIFEST),
            "sha256": sha_file(source007.CAPTURE_MANIFEST),
            "steps": list(CAPTURE_STEPS),
            "row_count": len(capture_rows),
        },
        "late_transfer": {
            "input_path": str(LATE_INPUT),
            "input_sha256": sha_file(LATE_INPUT),
            "source_path": str(late_source_path),
            "source_sha256": late_source["sha256"],
            "pack_path": str(late_row["pack_path"]),
            "pack_sha256": sha_file(Path(late_row["pack_path"])),
            "receipt_path": str(late_row["receipt_path"]),
            "receipt_sha256": sha_file(Path(late_row["receipt_path"])),
            "accepted_step": TRANSFER_STEP,
            "accepted_time_s": TRANSFER_TIME_S,
            "separate_transfer_capture": True,
            "N27_continuation_claim": False,
        },
        "source_candidate": {
            "receipt_path": str(candidate_receipt),
            "receipt_sha256": sha_file(candidate_receipt),
            "preparation_path": str(prep_path),
            "preparation_sha256": sha_file(prep_path),
            "payload_path": str(candidate_path),
            "payload_sha256": sha_file(candidate_path),
            "stable_ids": [525, 526],
            "validated_by_existing_abi5_append_owner": True,
            "candidate_directory_files_sha256": portdiag._actual_file_hashes(
                source007.CANDIDATE_DIR
            ),
            "preparation_directory_files_sha256": portdiag._actual_file_hashes(
                source007.PREPARATION_DIR
            ),
        },
        "source0_reuse": {
            "five_port_report_path": str(SOURCE0_PORT_REPORT),
            "five_port_report_sha256": SOURCE0_PORT_REPORT_SHA256,
            "last_port_report_path": str(SOURCE0_LAST_PORT_REPORT),
            "last_port_report_sha256": SOURCE0_LAST_PORT_REPORT_SHA256,
            "whole_candidate_payload_hash_is_not_row_gate": True,
        },
        "n27_capture_source": {
            "payload_path": str(N27_CAPTURE_SOURCE),
            "payload_sha256": N27_CAPTURE_SOURCE_SHA256,
            "receipt_path": str(N27_CAPTURE_RECEIPT),
            "receipt_sha256": sha_file(N27_CAPTURE_RECEIPT),
        },
        "repair_ancestry": {
            "report_path": str(REPAIR_REPORT),
            "report_sha256": sha_file(REPAIR_REPORT),
            "stable_ids": sorted(REPAIRED_IDS),
        },
        "arrangement_candidates": arrangement_preflight,
        "decoded_geometry_arrays_loaded": False,
        "geometry_predicates_invoked": False,
        "source_compiler_invoked": False,
        "native_or_gpu_work_performed": False,
        "anatomical_admission": False,
    }


def write_preflight(arrangement_candidates: dict[int, dict] | None = None) -> dict:
    report = build_preflight(arrangement_candidates)
    _write_json(PREFLIGHT_PATH, report)
    return {
        "status": report["status"],
        "preflight_path": str(PREFLIGHT_PATH),
        "preflight_sha256": sha_file(PREFLIGHT_PATH),
    }


def _normalized_pin_map(pins: dict) -> dict:
    return {str(Path(path).resolve()): digest for path, digest in pins.items()}


def _assert_preflight(preflight: dict) -> None:
    _require(
        preflight.get("status") == "PREFLIGHT_PINNED_NO_GEOMETRY_DECODED",
        "multipose preflight is not pinned",
    )
    _require(
        preflight.get("source_repository") == _git_state(),
        "source checkout revision or cleanliness differs from preflight",
    )
    for raw_path, expected in preflight.get("runtime_code_sha256", {}).items():
        path = Path(raw_path)
        _require(
            path.is_file() and sha_file(path) == expected,
            "source/verifier code changed after preflight: " + raw_path,
        )
    for raw_path, expected in preflight.get("input_sha256", {}).items():
        path = Path(raw_path)
        _require(
            path.is_file() and sha_file(path) == expected,
            "multipose input changed after preflight: " + raw_path,
        )
    _require(
        _normalized_pin_map(source007._frozen_input_pins())
        == _normalized_pin_map(source007.INPUT_SHA),
        "source007 frozen source/compiler pins changed",
    )
    for stable_id, descriptor in preflight.get("arrangement_candidates", {}).items():
        _require(
            str(int(stable_id)) == stable_id
            and int(stable_id) in REPAIRED_IDS
            and isinstance(descriptor, dict),
            "preflight full_candidate binding targets an unsupported stable ID",
        )
        candidate_path = str(Path(descriptor.get("candidate_path", "")).resolve())
        _require(
            preflight.get("input_sha256", {}).get(candidate_path)
            == descriptor.get("candidate_sha256")
            and descriptor.get("source_payload_sha256")
            == "50ab8248d2e0f55dd6cb8b24afc34c8844061d5841ecdf89d7daf03288816bfd"
            and descriptor.get("source_payload_index") == 1,
            "preflight full_candidate hash or N27 parent binding is missing",
        )
    expected_candidate = preflight["source_candidate"]
    _require(
        portdiag._actual_file_hashes(source007.CANDIDATE_DIR)
        == expected_candidate["candidate_directory_files_sha256"],
        "candidate output inventory changed",
    )
    _require(
        portdiag._actual_file_hashes(source007.PREPARATION_DIR)
        == expected_candidate["preparation_directory_files_sha256"],
        "preparation output inventory changed",
    )
    _require(
        sha_file(Path(expected_candidate["payload_path"]))
        == expected_candidate["payload_sha256"],
        "current append output differs from multipose preflight",
    )
    _require(
        sha_file(Path(expected_candidate["receipt_path"]))
        == expected_candidate["receipt_sha256"]
        and sha_file(Path(expected_candidate["preparation_path"]))
        == expected_candidate["preparation_sha256"],
        "current append receipt or preparation differs from multipose preflight",
    )


def _canonical_row_bytes(row: dict) -> bytes:
    """Normalize only global buffer offsets; preserve exact per-row ABI5 content."""
    record = tuple(map(int, row["record"]))
    (
        body,
        _vertex_start,
        vertex_count,
        _index_start,
        index_count,
        stable_id,
        layer,
        flags,
    ) = record
    vertices = np.ascontiguousarray(row["vertices6"], dtype="<f4")
    faces = np.ascontiguousarray(row["faces"], dtype="<u4")
    _require(
        vertices.ndim == 2
        and vertices.shape == (vertex_count, 6)
        and faces.ndim == 2
        and faces.shape == (index_count // 3, 3)
        and index_count % 3 == 0,
        "decoded ABI5 row does not agree with its record descriptor",
    )
    normalized_record = struct.pack(
        "<8I", body, 0, vertex_count, 0, index_count, stable_id, layer, flags
    )
    return normalized_record + vertices.tobytes() + faces.tobytes()


def _row_identity_signature(identity: dict) -> dict:
    _require(isinstance(identity, dict), "source ID map entry is not an object")
    return {key: identity.get(key) for key in SOURCE_IDENTITY_FIELDS}


def _exact_array_bytes(first, second, dtype) -> bool:
    return (
        np.ascontiguousarray(first, dtype=dtype).tobytes()
        == np.ascontiguousarray(second, dtype=dtype).tobytes()
    )



def load_full_candidate_arrangement_binding(
    *,
    candidate_path: Path,
    candidate_sha256: str,
    source_payload_sha256: str,
    source_payload_index: int,
    lineage_path: Path,
    lineage_sha256: str,
    report_path: Path,
    report_sha256: str,
) -> dict:
    """Load the exact full_candidate triplet against its pinned 028 source parent."""
    from .resting_intercostal_mesh_arrangement import load_pinned_arrangement

    _require(
        source_payload_sha256
        == "50ab8248d2e0f55dd6cb8b24afc34c8844061d5841ecdf89d7daf03288816bfd"
        and source_payload_index == 1,
        "full_candidate must retain the pinned 028 source payload/index namespace",
    )
    return load_pinned_arrangement(
        candidate_path=candidate_path,
        candidate_sha256=candidate_sha256,
        lineage_path=lineage_path,
        lineage_sha256=lineage_sha256,
        report_path=report_path,
        report_sha256=report_sha256,
        source_payload_sha256=source_payload_sha256,
        source_payload_index=source_payload_index,
    )


def prepare_full_candidate_arrangement_binding(
    *,
    binding: dict,
    stable_id: int,
    source_row: dict,
    candidate_row: dict,
    repair_parent_row: dict,
    repair_report: dict,
) -> dict:
    """Compose the compiled candidate from original N27 plus the pinned 028 lineage."""
    from .resting_intercostal_mesh_arrangement import (
        compose_intercostal_arrangement_row,
        verify_compiled_candidate_row,
    )

    expected_row, summary = compose_intercostal_arrangement_row(
        stable_id=stable_id,
        source_row=source_row,
        parent_row=repair_parent_row,
        arrangement=binding,
        repair_report=repair_report,
    )
    verified = verify_compiled_candidate_row(
        stable_id=stable_id,
        source_row=source_row,
        parent_row=repair_parent_row,
        candidate_row=candidate_row,
        arrangement=binding,
        repair_report=repair_report,
    )
    _require(
        _canonical_row_bytes(expected_row) == _canonical_row_bytes(candidate_row),
        "ABI5 candidate row differs from the source-bound N27/028 composition",
    )
    return {
        "typed_identity": [51010, int(stable_id)],
        "candidate_path": binding["candidate_path"],
        "candidate_sha256": binding["candidate_sha256"],
        "source_payload_sha256": N27_CAPTURE_SOURCE_SHA256,
        "full_candidate_parent_payload_sha256": binding["source_payload_sha256"],
        "source_payload_index": binding["source_payload_index"],
        "full_candidate_raw": binding["candidate_raw"],
        "full_candidate_file_hash_verified": True,
        "source_row_sha256": _sha_bytes(_canonical_row_bytes(source_row)),
        "compiled_candidate_row_sha256": _sha_bytes(_canonical_row_bytes(candidate_row)),
        "full_candidate_parent_lineage_sha256": binding["lineage_sha256"],
        "full_candidate_report_sha256": binding["report_sha256"],
        "repair_parent_row_sha256": _sha_bytes(_canonical_row_bytes(repair_parent_row)),
        "repair_parent_face_count": len(repair_parent_row["faces"]),
        "candidate_faces": summary["candidate_faces"],
        "parent_map_terms": summary["candidate_vertex_parent_maps"],
        "retained_n27_face_map_sha256": verified["retained_n27_face_map_sha256"],
        "patch_parent_face_ids_in_N27_namespace": verified[
            "patch_parent_face_ids_in_N27_namespace"
        ],
        "new_appended_vertex_count": verified["new_appended_vertex_count"],
        "all_028_faces_preserved_byte_exact_in_original_order": verified[
            "all_028_faces_preserved_byte_exact_in_original_order"
        ],
        "positive_exact_parent_map_for_each_appended_vertex": verified[
            "positive_exact_parent_map_for_each_appended_vertex"
        ],
        "physical_mass_or_state_added": False,
        "native_confirmation": "required",
    }


def _project_arrangement_capture_surface(
    *,
    key: tuple[int, int],
    capture_surface: tuple,
    source_row: dict,
    prepared_binding: dict,
    verifier,
) -> tuple[tuple, dict]:
    """Project an arrangement through the exact original capture vertex namespace."""
    retain = getattr(verifier, "retain_bound_source_faces", None)
    _require(
        callable(retain),
        "central P6 retained-face owner is required to bind arrangement parents",
    )
    source_faces = np.ascontiguousarray(source_row["faces"], dtype="<u4")
    positions, capture_faces, normals, gids = capture_surface
    gids = np.asarray(gids)
    capture_faces = np.asarray(capture_faces, dtype="<u4")
    _require(
        len(source_faces) > 0 and capture_faces.size > 0 and gids.ndim == 1
        and capture_faces.shape == source_faces.shape,
        "original parent source surface is empty or has a different face inventory",
    )
    first_capture_local = int(capture_faces[0, 0])
    first_source_local = int(source_faces[0, 0])
    _require(
        0 <= first_capture_local < len(gids),
        "first source face corner is absent from captured global IDs",
    )
    global_base = int(gids[first_capture_local]) - first_source_local
    _projected_parent, parent_proof = retain(
        key,
        capture_surface,
        source_faces,
        global_base,
        np.arange(len(source_faces), dtype=np.int64),
    )
    _require(
        parent_proof.get("all_ordered_source_facets_match") is True
        and parent_proof.get("captured_positions_normals_and_vertex_ids_unchanged") is True
        and int(parent_proof.get("retained_source_face_count", -1)) == len(source_faces),
        "central P6 did not bind the complete original source face inventory",
    )
    local_by_source = {}
    for capture_slot, gid in enumerate(gids):
        source_vertex = int(gid) - global_base
        if 0 <= source_vertex < len(source_row["vertices6"]):
            _require(
                source_vertex not in local_by_source,
                "capture vertex namespace maps one source vertex more than once",
            )
            local_by_source[source_vertex] = capture_slot
    needed = {
        source_vertex
        for face in prepared_binding["candidate_faces"]
        for candidate_vertex in face
        for source_vertex, _weight in prepared_binding["parent_map_terms"][int(candidate_vertex)]
    }
    _require(
        needed.issubset(local_by_source),
        "captured source row omits vertices required by full_candidate parent maps",
    )
    candidate_count = len(prepared_binding["parent_map_terms"])
    projected_positions = np.empty((candidate_count, 3), dtype="<f4")
    projected_normals = np.empty((candidate_count, 3), dtype="<f4")
    source_positions = np.asarray(positions, dtype="<f4")
    source_normals = np.asarray(normals, dtype="<f4")
    for candidate_vertex, terms in enumerate(prepared_binding["parent_map_terms"]):
        slots = [local_by_source[source_vertex] for source_vertex, _weight in terms]
        if len(terms) == 1 and terms[0][1] == 1:
            projected_positions[candidate_vertex] = source_positions[slots[0]]
            projected_normals[candidate_vertex] = source_normals[slots[0]]
            continue
        weights = np.asarray([float(weight) for _source_vertex, weight in terms], dtype=np.float64)
        point = np.einsum("i,ij->j", weights, source_positions[slots].astype(np.float64))
        normal = np.einsum("i,ij->j", weights, source_normals[slots].astype(np.float64))
        normal_length = float(np.linalg.norm(normal))
        _require(
            np.isfinite(point).all() and np.isfinite(normal).all()
            and normal_length > 1e-12,
            "captured parent-map interpolation produced invalid coordinates or normals",
        )
        projected_positions[candidate_vertex] = point.astype("<f4")
        projected_normals[candidate_vertex] = (normal / normal_length).astype("<f4")
    _require(
        np.isfinite(projected_positions).all() and np.isfinite(projected_normals).all(),
        "captured arrangement projection is nonfinite",
    )
    surface = (
        projected_positions,
        prepared_binding["candidate_faces"].copy(),
        projected_normals,
        np.arange(candidate_count, dtype=np.uint64),
    )
    return surface, {
        "typed_identity": list(key),
        "full_candidate_sha256": prepared_binding["candidate_sha256"],
        "source_payload_sha256": prepared_binding["source_payload_sha256"],
        "source_payload_index": prepared_binding["source_payload_index"],
        "source_capture_pack_positions_sha256": _sha_bytes(
            np.asarray(positions, dtype="<f4").tobytes()
        ),
        "source_capture_pack_faces_sha256": _sha_bytes(
            np.asarray(capture_faces, dtype="<u4").tobytes()
        ),
        "source_global_vertex_base": global_base,
        "central_p6_complete_ordered_parent_face_proof": parent_proof,
        "projected_candidate_positions_sha256": _sha_bytes(projected_positions.tobytes()),
        "projected_candidate_faces_sha256": _sha_bytes(
            prepared_binding["candidate_faces"].tobytes()
        ),
        "projected_candidate_normals_sha256": _sha_bytes(projected_normals.tobytes()),
        "parent_map_terms": sum(
            len(terms) for terms in prepared_binding["parent_map_terms"]
        ),
        "projection": (
            "full_candidate positive exact rational parent maps over actual accepted-capture "
            "parent vertices; weighted Float64 interpolation lowered once to Float32, "
            "unit normals renormalized"
        ),
        "source_vertex_ids": (
            "candidate-local map ordinals for offline P6 preparation; source lineage "
            "remains in full_candidate parent maps"
        ),
        "native_confirmation": "required",
    }


def _retained_face_ids(
    source_row: dict, candidate_row: dict, repair_row: dict | None
) -> np.ndarray:
    source_faces = np.asarray(source_row["faces"], dtype="<u4")
    candidate_faces = np.asarray(candidate_row["faces"], dtype="<u4")
    _require(
        _exact_array_bytes(source_row["vertices6"], candidate_row["vertices6"], "<f4"),
        "candidate row changed source vertices/normals for stable ID "
        + str(source_row["record"][5]),
    )
    for label in ("body_index", "layer", "flags"):
        _require(
            int(source_row[label]) == int(candidate_row[label]),
            f"candidate row changed {label} for stable ID {source_row['record'][5]}",
        )
    if repair_row is None:
        retained = np.arange(len(source_faces), dtype=np.int64)
    else:
        removed = np.asarray(
            repair_row.get("removed_source_face_ids", []), dtype=np.int64
        )
        _require(
            removed.ndim == 1
            and (
                len(removed) == 0
                or (
                    removed.min() >= 0
                    and removed.max() < len(source_faces)
                    and np.all(np.diff(removed) > 0)
                )
            ),
            "repair report removed-face ancestry is invalid",
        )
        _require(
            int(repair_row.get("source_face_count", -1)) == len(source_faces),
            "repair report source-face count differs from N27 capture parent",
        )
        retained_mask = np.ones(len(source_faces), dtype=bool)
        retained_mask[removed] = False
        retained = np.flatnonzero(retained_mask).astype(np.int64)
        _require(
            int(repair_row.get("retained_face_count", -1)) == len(retained),
            "repair report retained-face count is inconsistent",
        )
    _require(
        _exact_array_bytes(candidate_faces, source_faces[retained], "<u4"),
        "candidate row is not the exact retained N27 source-face ancestry for stable ID "
        + str(source_row["record"][5]),
    )
    return retained


def _capture_owner_map(verifier, pack_path: Path, surfaces: dict) -> dict:
    """Read only existing P6 pack primitive identity/owner fields."""
    _require(
        callable(getattr(verifier, "_pack_sections", None)),
        "central P6 pack section parser is unavailable for owner verification",
    )
    sections = verifier._pack_sections(pack_path)
    raw, count, stride = sections[4]
    _require(
        stride == 64 and len(raw) == count * stride,
        "accepted pack primitive section is not the central P6 64-byte layout",
    )
    primitives = np.frombuffer(raw, dtype="<u4").reshape(-1, 16)
    owners: dict[tuple[int, int], int] = {}
    for primitive in primitives:
        semantic, stable_id = int(primitive[4]), int(primitive[7])
        if semantic not in verifier.ANATOMY_SEMANTICS:
            continue
        key = (semantic, stable_id)
        owner = int(primitive[6])
        _require(
            key in surfaces,
            "central load_capture and primitive owner identity inventories differ",
        )
        if key in owners:
            _require(
                owners[key] == owner,
                "one captured typed surface has inconsistent primitive body owners",
            )
        owners[key] = owner
    _require(
        set(owners) == set(surfaces),
        "captured typed surfaces and primitive owner map are not identical",
    )
    return owners


def _unbound_semantic_counts(keys) -> dict:
    counts = {}
    for semantic, _stable_id in keys:
        key = str(int(semantic))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _audit_capture_row_bindings(
    *,
    surfaces: dict,
    captured_owners: dict,
    transfer_rows: dict,
    baseline_rows: dict,
    candidate_rows: dict,
    baseline_source_map: dict,
    candidate_source_map: dict,
    repair_surfaces: dict,
    verifier,
    transfer_label: str,
    source_bound_keys=None,
    arrangement_candidates: dict[int, dict] | None = None,
) -> tuple[dict, dict]:
    """Bind explicitly mapped ABI5 surfaces and preserve all capture targets."""
    surface_keys = set(surfaces)
    bound_keys = surface_keys if source_bound_keys is None else set(source_bound_keys)
    _require(
        bound_keys.issubset(surface_keys),
        "an explicitly source-bound typed surface is absent from the accepted capture",
    )
    selected_faces: dict[str, np.ndarray] = {}
    arrangement_surfaces = {}
    arrangement_projections = {}
    arrangement_candidates = arrangement_candidates or {}
    _require(
        set(arrangement_candidates).issubset(bound_keys and {int(key[1]) for key in bound_keys}),
        "arrangement binding is supplied for an unrequested capture target",
    )
    rows = []
    failures = []
    for key in sorted(bound_keys):
        stable_id = int(key[1])
        transfer = transfer_rows.get(stable_id)
        baseline = baseline_rows.get(stable_id)
        candidate = candidate_rows.get(stable_id)
        row_failure = []
        if transfer is None or baseline is None or candidate is None:
            row_failure.append(
                "source row missing from transfer, N27, or candidate ABI5 payload"
            )
        else:
            transfer_bytes = _canonical_row_bytes(transfer)
            baseline_bytes = _canonical_row_bytes(baseline)
            if transfer_bytes != baseline_bytes:
                row_failure.append(
                    "transfer source row bytes differ from original N27 capture source"
                )
            base_id = baseline_source_map.get(str(stable_id))
            candidate_id = candidate_source_map.get(str(stable_id))
            if base_id is None or candidate_id is None:
                row_failure.append(
                    "N27/candidate source ID identity map entry is missing"
                )
            else:
                if _row_identity_signature(base_id) != _row_identity_signature(
                    candidate_id
                ):
                    row_failure.append(
                        "candidate source ID, provenance, or owner identity differs from N27"
                    )
                if int(base_id.get("body_index", -1)) != int(baseline["body_index"]):
                    row_failure.append(
                        "N27 receipt body owner differs from its ABI5 row"
                    )
                if int(base_id.get("layer", -1)) != int(baseline["layer"]):
                    row_failure.append("N27 receipt layer differs from its ABI5 row")
                if int(candidate_id.get("body_index", -1)) != int(
                    candidate["body_index"]
                ):
                    row_failure.append(
                        "candidate receipt body owner differs from its ABI5 row"
                    )
                if int(candidate_id.get("layer", -1)) != int(candidate["layer"]):
                    row_failure.append(
                        "candidate receipt layer differs from its ABI5 row"
                    )
            repair = repair_surfaces.get(str(stable_id))
            try:
                arrangement = arrangement_candidates.get(stable_id)
                if arrangement is None:
                    retained = _retained_face_ids(baseline, candidate, repair)
                    selected_faces[str(stable_id)] = retained
                else:
                    _require(
                        arrangement.get("typed_identity") == [int(key[0]), stable_id]
                        and arrangement.get("source_payload_sha256")
                        == N27_CAPTURE_SOURCE_SHA256
                        and arrangement.get("candidate_sha256"),
                        "prepared full_candidate binding does not match this typed N27 row",
                    )
                    _require(
                        arrangement.get("compiled_candidate_row_sha256")
                        == _sha_bytes(_canonical_row_bytes(candidate)),
                        "prepared full_candidate binding differs from the compiled ABI5 row",
                    )
                    projected_surface, projection = _project_arrangement_capture_surface(
                        key=key,
                        capture_surface=surfaces[key],
                        source_row=baseline,
                        prepared_binding=arrangement,
                        verifier=verifier,
                    )
                    arrangement_surfaces[key] = projected_surface
                    arrangement_projections[str(stable_id)] = {
                        "full_candidate_binding": {
                            field: value for field, value in arrangement.items()
                            if field not in {
                                "parent_map_terms", "source_parent_faces", "candidate_faces"
                            }
                        },
                        "capture_projection": projection,
                    }
            except (ValueError, KeyError) as error:
                row_failure.append(str(error))
        owner = captured_owners.get(key)
        if owner is None:
            row_failure.append("captured primitive body owner is missing")
        elif candidate is not None:
            source_body = int(candidate["body_index"])
            if owner != DEFORMABLE_OWNER and owner != source_body:
                row_failure.append(
                    f"captured owner {owner} differs from NHA source body owner {source_body}"
                )
            if stable_id in ROUTE_TARGET_IDS and owner != source_body:
                row_failure.append(
                    f"vascular route target owner {owner} is not its NHA rigid source body {source_body}"
                )
        rows.append(
            {
                "typed_identity": list(key),
                "stable_id": stable_id,
                "transfer_label": transfer_label,
                "captured_primitive_owner": owner,
                "captured_owner_is_deformable_sentinel": owner == DEFORMABLE_OWNER,
                "candidate_source_body_index": None
                if candidate is None
                else int(candidate["body_index"]),
                "candidate_source_layer": None
                if candidate is None
                else int(candidate["layer"]),
                "transfer_source_row_sha256": None
                if transfer is None
                else _sha_bytes(_canonical_row_bytes(transfer)),
                "N27_capture_source_row_sha256": None
                if baseline is None
                else _sha_bytes(_canonical_row_bytes(baseline)),
                "candidate_source_row_sha256": None
                if candidate is None
                else _sha_bytes(_canonical_row_bytes(candidate)),
                "transfer_row_matches_N27_by_exact_normalized_ABI5_bytes": bool(
                    transfer is not None
                    and baseline is not None
                    and _canonical_row_bytes(transfer) == _canonical_row_bytes(baseline)
                ),
                "candidate_row_is_exact_N27_or_reported_retained_face_projection": bool(
                    transfer is not None
                    and baseline is not None
                    and candidate is not None
                    and str(stable_id) in selected_faces
                ),
                "candidate_row_has_verified_full_candidate_parent_maps": (
                    stable_id in arrangement_candidates
                    and str(stable_id) in arrangement_projections
                ),
                "source_identity_matches_N27": bool(
                    baseline_source_map.get(str(stable_id)) is not None
                    and candidate_source_map.get(str(stable_id)) is not None
                    and _row_identity_signature(baseline_source_map[str(stable_id)])
                    == _row_identity_signature(candidate_source_map[str(stable_id)])
                ),
                "transfer_source_identity_resolved_through_exact_N27_row_match": bool(
                    transfer is not None
                    and baseline is not None
                    and _canonical_row_bytes(transfer) == _canonical_row_bytes(baseline)
                    and baseline_source_map.get(str(stable_id)) is not None
                ),
                "transfer_identity_authority": (
                    "pinned N27 source_id_map inherited only through exact normalized "
                    "ABI5 row-byte equality; mismatches fail closed"
                ),
                "passed": not row_failure,
                "failures": row_failure,
            }
        )
        if row_failure:
            failures.extend(
                {"typed_identity": list(key), "reason": reason}
                for reason in row_failure
            )
    projected = dict(surfaces)
    if failures:
        return {
            "passed": False,
            "transfer_label": transfer_label,
            "whole_payload_hash_mismatch_is_not_used_as_row_failure": True,
            "captured_target_count": len(rows),
            "capture_surface_count_total": len(surface_keys),
            "source_bound_typed_identities": [list(key) for key in sorted(bound_keys)],
            "unbound_capture_semantic_counts": _unbound_semantic_counts(
                surface_keys - bound_keys
            ),
            "per_target_rows": rows,
            "retained_face_projection": [],
            "full_candidate_parent_map_projection": arrangement_projections,
            "all_requested_source_bound_targets_bound": False,
            "all_captured_targets_source_bound": len(bound_keys) == len(surface_keys),
            "failures": failures,
        }, projected
    # The only source-face edits permitted here are the three independently
    # reported intercostal fragment removals. Use HP97/P6's shared retention
    # owner before strict target preparation; coordinates/normals/vertex IDs
    # remain the actual captured arrays.
    retain = getattr(verifier, "retain_bound_source_faces", None)
    repaired_keys = [
        key for key in sorted(bound_keys)
        if key[1] in REPAIRED_IDS and key[1] not in arrangement_candidates
    ]
    _require(
        all(str(key[1]) in selected_faces for key in repaired_keys),
        "repaired target row has no source-face ancestry selection",
    )
    if repaired_keys:
        _require(
            callable(retain),
            "central P6 retain_bound_source_faces helper is required for repaired target rows",
        )
    ancestry = []
    for key in repaired_keys:
        row = transfer_rows[int(key[1])]
        retained = selected_faces[str(key[1])]
        value, proof = retain(
            key, surfaces[key], row["faces"], int(row["record"][1]), retained
        )
        _require(
            proof.get("all_ordered_source_facets_match") is True
            and proof.get("captured_positions_normals_and_vertex_ids_unchanged")
            is True,
            "central P6 did not prove complete retained source-face ancestry",
        )
        _require(
            len(value[1]) == len(candidate_rows[int(key[1])]["faces"]),
            "retained capture faces differ from compiled candidate row count",
        )
        projected[key] = value
        ancestry.append(
            {
                "typed_identity": list(key),
                "source_face_count": int(proof["source_face_count"]),
                "retained_source_face_count": int(proof["retained_source_face_count"]),
                "source_vertex_namespace_offset": int(
                    proof["source_capture_vertex_namespace_offset"]
                ),
                "source_face_ancestry_sha256": _sha_bytes(
                    np.asarray(retained, dtype="<u8").tobytes()
                ),
                "source_face_ancestry_ranges": _index_ranges(retained),
                "captured_positions_normals_and_vertex_ids_unchanged": True,
                "owner": "central HP97/P6 retain_bound_source_faces",
            }
        )
    projected.update(arrangement_surfaces)
    return {
        "passed": True,
        "transfer_label": transfer_label,
        "whole_payload_hash_mismatch_is_not_used_as_row_failure": True,
        "captured_target_count": len(rows),
        "capture_surface_count_total": len(surface_keys),
        "source_bound_typed_identities": [list(key) for key in sorted(bound_keys)],
        "unbound_capture_semantic_counts": _unbound_semantic_counts(
            surface_keys - bound_keys
        ),
        "per_target_rows": rows,
        "retained_face_projection": ancestry,
        "full_candidate_parent_map_projection": arrangement_projections,
        "all_requested_source_bound_targets_bound": True,
        "all_captured_targets_source_bound": len(bound_keys) == len(surface_keys),
        "N27_continuation_claim": transfer_label == "N27 accepted capture",
    }, projected


def _index_ranges(values) -> list[list[int]]:
    values = np.asarray(values, dtype=np.int64)
    if not len(values):
        return []
    breaks = np.flatnonzero(np.diff(values) != 1) + 1
    groups = np.split(values, breaks)
    return [[int(group[0]), int(group[-1])] for group in groups]


def _validate_appended_bridge_owner_inputs(
    candidate_rows: dict, source_map: dict, preparation: dict
) -> dict:
    result = {}
    bridge_records = preparation.get("bridges", {})
    for stable_id, spec in source007.BRIDGE_ROWS.items():
        key = str(stable_id)
        row = candidate_rows.get(stable_id)
        identity = source_map.get(key)
        member = spec["member"]
        _require(
            row is not None and identity is not None,
            "appended bridge source row or source identity is missing: " + member,
        )
        _require(
            identity.get("source_member") == member
            and int(identity.get("body_index", -1)) == 20
            and int(identity.get("layer", -1)) == 2
            and int(identity.get("vascular_compartment_id", -1)) == 4,
            "appended bridge source/member/body/layer/CVSim owner changed: " + member,
        )
        _require(
            isinstance(identity.get("source_sha256"), str)
            and len(identity["source_sha256"]) == 64,
            "appended bridge source hash is missing: " + member,
        )
        owner = identity.get("source_owner_metadata", {})
        _require(
            owner.get("member_id") == member
            and int(owner.get("core_body_index", -1)) == 20
            and owner.get("layer") == "vessel",
            "appended bridge source owner metadata changed: " + member,
        )
        bridge_prep = bridge_records.get(member, {})
        ancestry = bridge_prep.get("vertex_ancestry")
        _require(
            isinstance(ancestry, list) and len(ancestry) == len(row["vertices6"]),
            "compiled common-map vertex ancestry is missing: " + member,
        )
        _require(
            int(row["record"][5]) == stable_id
            and int(row["record"][0]) == 20
            and int(row["record"][6]) == 2,
            "appended bridge ABI5 descriptor owner changed: " + member,
        )
        xyz_hash = _sha_bytes(
            np.ascontiguousarray(row["vertices6"][:, :3], dtype="<f4").tobytes()
        )
        faces_hash = _sha_bytes(
            np.ascontiguousarray(row["faces"], dtype="<i4").tobytes()
        )
        _require(
            xyz_hash == bridge_prep.get("source_local_xyz_sha256")
            and faces_hash == bridge_prep.get("source_faces_sha256"),
            "appended bridge decoded row differs from compiler's source-060 row pins: "
            + member,
        )
        result[key] = {
            "source_member": member,
            "stable_id": stable_id,
            "source_member_sha256": identity["source_sha256"],
            "source_owner_body_index": int(row["body_index"]),
            "source_owner_layer": int(row["layer"]),
            "vascular_compartment_id": int(identity["vascular_compartment_id"]),
            "vertex_count": len(row["vertices6"]),
            "face_count": len(row["faces"]),
            "normalized_row_sha256": _sha_bytes(_canonical_row_bytes(row)),
            "source_local_xyz_sha256": xyz_hash,
            "source_faces_sha256": faces_hash,
            "compiler_source060_xyz_sha256": bridge_prep["source_local_xyz_sha256"],
            "compiler_source060_faces_sha256": bridge_prep["source_faces_sha256"],
            "common_map_vertex_ancestry_count": len(ancestry),
            "additional_physical_mass_kg": 0,
            "additional_physical_state_or_force": False,
        }
    return {
        "passed": True,
        "rows": result,
        "owner_contract": "existing NHA ABI5 append compiler, body20/layer2, existing CVSim compartment4",
        "N22_transfer_source_rows_for_appended_bridges": "not applicable; bridges are appended source rows",
    }


def _read_pose(receipt: dict) -> dict:
    matches = [
        row
        for row in receipt.get("accepted_body_poses", [])
        if int(row.get("body_index", -1)) == 20
    ]
    _require(
        len(matches) == 1,
        "accepted capture receipt must contain exactly one body20 pose",
    )
    pose = matches[0]
    q = np.asarray(pose.get("quaternion_xyzw"), dtype="<f4")
    t = np.asarray(pose.get("position_m"), dtype="<f4")
    _require(
        q.shape == (4,)
        and t.shape == (3,)
        and np.isfinite(q).all()
        and np.isfinite(t).all(),
        "accepted body20 pose is malformed",
    )
    return pose


def _verify_capture_common_field(
    receipt: dict,
    *,
    expected_source_sha: str,
    preparation: dict,
    label: str,
) -> dict:
    _require(
        receipt.get("common_field_source_anatomy_payload_sha256")
        == expected_source_sha,
        label + " accepted receipt has unexpected source anatomy hash",
    )
    _require(
        receipt.get("cardiac_geometry_mode") == "common_seven_coordinate_v1",
        label + " is not using the existing common seven-coordinate field",
    )
    comparisons = {
        "map": ("common_field_map_sha256", "input_common_map_sha256"),
        "polynomials": (
            "common_field_polynomial_sha256",
            "input_common_polynomials_sha256",
        ),
        "domain_boxes": (
            "common_field_domain_boxes_sha256",
            "input_common_domain_boxes_sha256",
        ),
    }
    checked = {}
    for label_name, (receipt_key, prep_key) in comparisons.items():
        observed, expected = receipt.get(receipt_key), preparation.get(prep_key)
        _require(
            isinstance(observed, str) and observed == expected,
            f"{label} {label_name} hash differs from the pinned existing-map prefix",
        )
        checked[label_name] = observed
    return {
        "capture_source_anatomy_payload_sha256": expected_source_sha,
        "whole_payload_matches_current_N27": expected_source_sha
        == source007.N27_PAYLOAD_SHA,
        "existing_common_field_prefix_hashes": checked,
        "new_bridge_map_bytes_in_late_capture": False,
        "late_capture_scope": "transfer capture; existing common map/polynomial/domain match only",
    }


def _verify_source0_reuse(
    source0_report: dict,
    last_port_report: dict,
    source0_identity: dict,
    bridges: dict,
    captured: dict,
) -> dict:
    helper = source0_report.get("helper_result", {})
    ports = helper.get("ports", [])
    expected = source007.PORT_KEYS
    _require(
        len(ports) == 6
        and helper.get("requested_port_count") == 6
        and helper.get("evaluated_port_count") == 6,
        "source-0 frozen six-port report is incomplete",
    )
    for ordinal, pair in enumerate(expected[:5]):
        row = ports[ordinal]
        _require(
            (tuple(row.get("bridge", ())), tuple(row.get("target", ()))) == pair
            and row.get("passed") is True,
            "one of the first five frozen source-0 ports is not a pass",
        )
        _require(
            row.get("solid_gate", {}).get("passed") is True
            and row.get("calibre_gate", {}).get("passed") is True
            and row.get("port_pair_gate", {}).get("passed") is True,
            "a first-five source-0 solid/interface/calibre proof is incomplete",
        )
    failed_port = helper.get("first_failed_port", {})
    _require(
        tuple(failed_port.get("bridge", ())) == expected[5][0]
        and tuple(failed_port.get("target", ())) == expected[5][1],
        "source-0 section diagnostic did not stop at the expected final port",
    )
    _require(
        ports[5].get("passed") is False, "source-0 final-port rejection record changed"
    )

    last = last_port_report.get("helper_result", {})
    _require(
        last.get("passed") is True
        and last.get("bridge") == list(expected[5][0])
        and last.get("target") == list(expected[5][1]),
        "frozen source-0 3D proof for the last port no longer passes",
    )
    _require(
        last.get("solid_gates", {}).get("candidate", {}).get("passed") is True
        and last.get("independent_collar_calibre", {}).get("bridge", {}).get("passed")
        is True
        and last.get("independent_collar_calibre", {}).get("target", {}).get("passed")
        is True,
        "source-0 3D last-port solid/calibre proof is incomplete",
    )

    expected_source0 = source0_report.get("input_geometry", {})
    _require(
        expected_source0.get("target_capture_pack_sha256")
        == source0_identity.get("pack_sha256")
        and expected_source0.get("target_capture_receipt_sha256")
        == source0_identity.get("capture_receipt_sha256"),
        "reusable source-0 port report is not bound to the current accepted pack",
    )
    bridge_rows = expected_source0.get("bridge_surfaces", {})
    target_rows = expected_source0.get("target_surfaces", {})
    for stable_id in (525, 526):
        key = (51011, stable_id)
        _require(
            key in bridges and str(stable_id) in bridge_rows,
            "source-0 appended bridge row is absent from retained port evidence",
        )
        positions, faces = bridges[key][:2]
        frozen = bridge_rows[str(stable_id)]
        _require(
            _sha_bytes(np.asarray(positions, dtype="<f4").tobytes())
            == frozen.get("positions_sha256")
            and _sha_bytes(np.asarray(faces, dtype="<u4").tobytes())
            == frozen.get("faces_sha256"),
            "current appended bridge row differs from per-row source-0 frozen bytes",
        )
    for stable_id in sorted({target_id for _, target_id in source007.PORT_KEYS}):
        key = (51011, stable_id)
        _require(
            key in captured and str(stable_id) in target_rows,
            "source-0 captured port target is absent from retained report",
        )
        positions, faces = captured[key][:2]
        frozen = target_rows[str(stable_id)]
        _require(
            _sha_bytes(np.asarray(positions, dtype="<f4").tobytes())
            == frozen.get("positions_sha256")
            and _sha_bytes(np.asarray(faces, dtype="<u4").tobytes())
            == frozen.get("faces_sha256"),
            "current source-0 target surface differs from exact retained capture bytes",
        )
    return {
        "status": "REUSED_FIRST_FIVE_DIAGNOSTIC_004_AND_FINAL_3D_DIAGNOSTIC_002",
        "first_five_solid_interface_calibre_passes_reused": 5,
        "last_port_3d_pass_reused": True,
        "source0_candidate_rows_rechecked_by_exact_transformed_geometry_hash": [
            525,
            526,
        ],
        "source0_target_rows_rechecked_by_exact_capture_geometry_hash": sorted(
            {target_id for _, target_id in source007.PORT_KEYS}
        ),
        "whole_candidate_payload_hash_used_as_row_gate": False,
        "source0_calibre_scope": "same-source source-0 self-comparison only; not physical or clinical calibre evidence",
        "source0_3d_report_scope": "one frozen 526-to-472 rigid-source port only",
    }


def _load_capture(
    verifier, row: dict, *, expected_step: int
) -> tuple[dict, dict, dict]:
    pack, receipt_path = Path(row["pack_path"]), Path(row["receipt_path"])
    expected_pack = row.get("recorded_pack_sha256")
    expected_receipt = row.get("receipt_sha256")
    _require(
        pack.is_file() and sha_file(pack) == expected_pack,
        f"accepted step {expected_step} pack differs from its capture manifest",
    )
    _require(
        receipt_path.is_file() and sha_file(receipt_path) == expected_receipt,
        f"accepted step {expected_step} receipt differs from its capture manifest",
    )
    surfaces, identity = verifier.load_capture(pack, receipt_path)
    receipt = json.loads(receipt_path.read_text())
    _require(
        int(identity.get("accepted_step", -1)) == expected_step
        and int(receipt.get("accepted_step", -1)) == expected_step,
        f"accepted step identity mismatch at {expected_step}",
    )
    _require(
        abs(
            float(identity.get("accepted_time_s", -1))
            - float(receipt.get("accepted_time_s", -2))
        )
        <= 1e-12,
        f"accepted time identity mismatch at {expected_step}",
    )
    capture_meta = {
        "pack_path": str(pack),
        "pack_sha256": identity["pack_sha256"],
        "receipt_path": str(receipt_path),
        "receipt_sha256": identity["capture_receipt_sha256"],
        "accepted_step": expected_step,
        "accepted_time_s": float(identity["accepted_time_s"]),
        "body20": _read_pose(receipt),
        "receipt": receipt,
    }
    return surfaces, identity, capture_meta


def _load_nha_rows(path: Path, expected_sha: str) -> tuple[tuple, dict]:
    _require(
        path.is_file() and sha_file(path) == expected_sha,
        "pinned source NHA payload hash changed: " + str(path),
    )
    header, _records, rows = nhanat_codec._parse_payload(path.read_bytes())
    return header, rows


def _row_map_from_receipt(receipt_path: Path) -> dict:
    receipt = json.loads(receipt_path.read_text())
    source_map = receipt.get("provenance", {}).get("source_id_map", {})
    _require(
        isinstance(source_map, dict) and source_map,
        "NHA source receipt has no source ID identity map",
    )
    return source_map


def _runtime_input_context(preflight: dict) -> dict:
    verifier = source007._load_central_p6()
    retention_owner = shared_verifier
    _require(
        callable(getattr(retention_owner, "retain_bound_source_faces", None))
        and callable(getattr(retention_owner, "_pack_sections", None)),
        "central HP97/P6 retained-source-face or pack-section helper is unavailable",
    )
    candidate_receipt_path, candidate_prep_path, candidate_payload_path = (
        _source_candidate_paths()
    )
    validation = source007._validate_source_candidate(
        candidate_receipt_path, candidate_prep_path
    )
    _require(
        validation["payload_path"].resolve() == candidate_payload_path.resolve(),
        "current ABI5 candidate path differs from existing compiler receipt",
    )
    candidate_header, candidate_rows = _load_nha_rows(
        candidate_payload_path, preflight["source_candidate"]["payload_sha256"]
    )
    baseline_header, baseline_rows = _load_nha_rows(
        N27_CAPTURE_SOURCE, N27_CAPTURE_SOURCE_SHA256
    )
    late_manifest = json.loads(LATE_INPUT.read_text())
    late_source = late_manifest["source"]
    transfer_header, transfer_rows = _load_nha_rows(
        Path(late_source["path"]), late_source["sha256"]
    )
    candidate_map = validation["source_map"]
    baseline_map = _row_map_from_receipt(N27_CAPTURE_RECEIPT)
    prep = validation["preparation"]
    repair = json.loads(REPAIR_REPORT.read_text())
    _require(
        repair.get("status")
        == "PASS_SCOPED_SOURCE_PREPARATION_REPAIR_NOT_ANATOMY_ADMISSION"
        and repair.get("source_coordinates_modified") is not True,
        "pinned fragment repair report does not permit source-face-only ancestry",
    )
    repair_surfaces = repair.get("surfaces", {})
    _require(
        set(repair_surfaces) == {str(value) for value in REPAIRED_IDS},
        "fragment repair report does not cover exactly the retained three IDs",
    )
    from .resting_intercostal_mesh_arrangement import load_pinned_fragment_repair
    repair_binding = validation["receipt"]["provenance"]["local_intercostal_mesh_repair"]
    repair_context = load_pinned_fragment_repair(
        base_receipt_path=source007.BASE_RECEIPT,
        base_payload_sha256=N27_CAPTURE_SOURCE_SHA256,
        repair_parent_receipt_path=Path(repair_binding["repair_parent_receipt_path"]),
        repair_parent_receipt_sha256=repair_binding["repair_parent_receipt_sha256"],
        repair_parent_payload_path=Path(repair_binding["repair_parent_payload_path"]),
        repair_report_path=Path(repair_binding["repair_report_path"]),
        repair_report_sha256=repair_binding["repair_report_sha256"],
    )
    arrangement_candidates = {}
    for record in repair_binding["rows"]:
        stable_id = int(record["stable_id"])
        descriptor = record["full_candidate"]
        binding = load_full_candidate_arrangement_binding(
            candidate_path=Path(descriptor["candidate_path"]),
            candidate_sha256=descriptor["candidate_sha256"],
            source_payload_sha256=descriptor["source_payload_sha256"],
            source_payload_index=descriptor["source_payload_index"],
            lineage_path=Path(descriptor["lineage_path"]),
            lineage_sha256=descriptor["lineage_sha256"],
            report_path=Path(descriptor["report_path"]),
            report_sha256=descriptor["report_sha256"],
        )
        arrangement_candidates[stable_id] = prepare_full_candidate_arrangement_binding(
            binding=binding,
            stable_id=stable_id,
            source_row=baseline_rows[stable_id],
            repair_parent_row=repair_context["repair_parent_rows"][stable_id],
            repair_report=repair_context["repair_report"],
            candidate_row=candidate_rows[stable_id],
        )
    append_owner = _validate_appended_bridge_owner_inputs(
        candidate_rows, candidate_map, prep
    )
    _require(
        int(candidate_header[5]) == int(baseline_header[5])
        and int(transfer_header[5]) == int(baseline_header[5]),
        "NHA registration fingerprint differs across source transfer",
    )
    return {
        "verifier": verifier,
        "retention_owner": retention_owner,
        "validation": validation,
        "candidate_rows": candidate_rows,
        "candidate_header": candidate_header,
        "baseline_rows": baseline_rows,
        "baseline_header": baseline_header,
        "transfer_rows": transfer_rows,
        "transfer_header": transfer_header,
        "candidate_map": candidate_map,
        "baseline_map": baseline_map,
        "preparation": prep,
        "repair_surfaces": repair_surfaces,
        "arrangement_candidates": arrangement_candidates,
        "append_owner": append_owner,
        "transform": source007._load_predictor_transform(),
        "manifold": source007._load_retained_manifold(),
    }


def _load_3d_junction_owner():
    try:
        return importlib.import_module(
            __package__ + ".resting_brachiocephalic_3d_junction"
        )
    except ImportError as error:
        raise ValueError(
            "the source-bound closed-solid 3D junction owner is unavailable"
        ) from error


def _audit_pose_port_continuity(
    verifier, manifold3d, candidate_surfaces, baseline_surfaces, port_face_pairs
) -> dict:
    """Route five interfaces through the existing gate and the curved Y port through its 3D owner."""
    expected = tuple(source007.PORT_KEYS)
    _require(
        set(port_face_pairs) == set(expected),
        "multipose continuity requires the exact six frozen collar witnesses",
    )
    junction_pair = ((51011, 526), (51011, 472))
    _require(
        junction_pair in expected,
        "frozen six-port list lacks the declared 526-to-472 Y-junction",
    )
    junction_owner = _load_3d_junction_owner()
    results = []
    for bridge_key, target_key in expected:
        pair = (bridge_key, target_key)
        started = time.perf_counter()
        try:
            if pair == junction_pair:
                result = junction_owner.audit_brachiocephalic_3d_junction(
                    verifier,
                    manifold3d,
                    bridge_key=bridge_key,
                    bridge_surface=candidate_surfaces[bridge_key],
                    target_key=target_key,
                    target_surface=candidate_surfaces[target_key],
                    frozen_face_pairs=port_face_pairs[pair],
                    baseline_bridge_surface=baseline_surfaces[bridge_key],
                    baseline_target_surface=baseline_surfaces[target_key],
                )
                method = "source-bound independent-collar 3D junction owner"
            else:
                result = gate.audit_brachiocephalic_port_pair_continuity(
                    verifier,
                    manifold3d,
                    bridge_key=bridge_key,
                    bridge_surface=candidate_surfaces[bridge_key],
                    target_key=target_key,
                    target_surface=candidate_surfaces[target_key],
                    frozen_face_pairs=port_face_pairs[pair],
                    baseline_bridge_surface=baseline_surfaces[bridge_key],
                    baseline_target_surface=baseline_surfaces[target_key],
                )
                method = "existing single-port solid and calibre gate"
        except Exception as error:  # noqa: BLE001 - retain earliest rejected port
            result = {
                "schema": "numi.human.brachiocephalic-port-replay.v1",
                "bridge": list(bridge_key),
                "target": list(target_key),
                "passed": False,
                "failure": {"type": type(error).__name__, "message": str(error)},
                "source_geometry_mutated": False,
            }
            method = (
                "source-bound independent-collar 3D junction owner"
                if pair == junction_pair
                else "existing single-port solid and calibre gate"
            )
        results.append(
            {
                "method": method,
                "elapsed_seconds": time.perf_counter() - started,
                **result,
            }
        )
        if result.get("passed") is not True:
            break
    passed = len(results) == len(expected) and all(
        row.get("passed") is True for row in results
    )
    first_failed = next((row for row in results if row.get("passed") is not True), None)
    return {
        "schema": "numi.human.brachiocephalic-multipose-port-continuity.v1",
        "status": (
            "SIX_PORT_SOLID_AND_LOCAL_CALIBRE_SCREEN_PASSED"
            if passed
            else "SIX_PORT_SOLID_AND_LOCAL_CALIBRE_SCREEN_REJECTED_OR_INCOMPLETE"
        ),
        "passed": passed,
        "requested_port_count": len(expected),
        "evaluated_port_count": len(results),
        "first_failed_port": (
            None
            if first_failed is None
            else {
                "bridge": first_failed.get("bridge"),
                "target": first_failed.get("target"),
            }
        ),
        "ports": results,
        "source_geometry_mutated": False,
        "anatomical_admission": False,
        "native_or_gpu_work_performed": False,
    }


def _verify_n27_capture_rows(manifest_rows, context, report) -> None:
    """Run all eight accepted N27 captures, stop-first, retaining stage outputs."""
    verifier = context["verifier"]
    retention_owner = context["retention_owner"]
    prep = context["preparation"]
    candidate_rows = context["candidate_rows"]
    baseline_rows = context["baseline_rows"]
    baseline_map = context["baseline_map"]
    candidate_map = context["candidate_map"]
    repair_surfaces = context["repair_surfaces"]
    transform = context["transform"]
    pairs, _attribution, _stationarity = source007._read_port_pairs()
    _require(
        set(pairs) == set(source007.PORT_KEYS), "frozen six-port ancestry keys changed"
    )
    results = report.setdefault("N27_accepted_captures", [])
    for capture_row in manifest_rows:
        step = int(capture_row["step"])
        started = time.perf_counter()
        entry = {
            "capture_kind": "N27 accepted capture",
            "accepted_step": step,
            "accepted_time_s": None,
            "stage": "load-capture",
            "status": "RUNNING",
            "source_compiler_invoked": False,
            "native_or_gpu_work_performed": False,
            "source_geometry_mutated": False,
        }
        results.append(entry)
        _write_json(PROGRESS_PATH, report)
        try:
            captured, identity, capture_meta = _load_capture(
                verifier, capture_row, expected_step=step
            )
            receipt = capture_meta["receipt"]
            entry["accepted_time_s"] = capture_meta["accepted_time_s"]
            entry["capture_identity"] = {
                "pack_sha256": identity["pack_sha256"],
                "receipt_sha256": identity["capture_receipt_sha256"],
                "accepted_step": step,
                "accepted_time_s": capture_meta["accepted_time_s"],
                "body20_pose": capture_meta["body20"],
            }
            entry["common_field_source"] = _verify_capture_common_field(
                receipt,
                expected_source_sha=N27_CAPTURE_SOURCE_SHA256,
                preparation=prep,
                label=f"N27 step {step}",
            )
            owners = _capture_owner_map(
                verifier, Path(capture_row["pack_path"]), captured
            )
            binding, strict_targets = _audit_capture_row_bindings(
                surfaces=captured,
                captured_owners=owners,
                transfer_rows=baseline_rows,
                baseline_rows=baseline_rows,
                candidate_rows=candidate_rows,
                baseline_source_map=baseline_map,
                candidate_source_map=candidate_map,
                repair_surfaces=repair_surfaces,
                verifier=retention_owner,
                transfer_label="N27 accepted capture",
                source_bound_keys=SOURCE_BOUND_TYPED_KEYS,
                arrangement_candidates=context.get("arrangement_candidates"),
            )
            entry["source_row_binding"] = binding
            _require(
                binding.get("passed") is True,
                "per-target exact source row, source identity, or owner binding failed",
            )
            entry["compiled_body20_row_emission"] = (
                source007._body20_source_to_owner_f32_emission(
                    candidate_rows,
                    capture_meta["body20"],
                    None,
                    candidate_map,
                    transform,
                    comparison_surfaces=strict_targets,
                    captured_owners=owners,
                )
            )
            rigid = source007._bridge_surfaces(
                candidate_rows, capture_meta["body20"], transform
            )
            predicted, prediction = source007._predict_step0_bridges(
                candidate_rows, rigid, captured, capture_meta["body20"], prep, transform
            )
            entry["bridge_prediction"] = {
                "status": "OFFLINE_CAPTURED_SVC_COMMON_FIELD_PREDICTION",
                "bridges": prediction,
                "native_common_field_error_bound": "not_available",
                "prediction_roundoff_bound_limit": "Interpolation-only F32 bound; no native map/polynomial/kernel error bound.",
                "native_confirmation_required": True,
            }

            if step == 0:
                prior = json.loads(SOURCE0_PORT_REPORT.read_text())
                prior_last = json.loads(SOURCE0_LAST_PORT_REPORT.read_text())
                entry["reused_source0_six_port_evidence"] = _verify_source0_reuse(
                    prior, prior_last, identity, rigid, captured
                )

            entry["stage"] = "P6-all-decoded-target-collision-screen"
            _write_json(PROGRESS_PATH, report)
            collision = gate.audit_captured_pose_surface_intersections(
                verifier,
                strict_targets,
                {key: value[0] for key, value in predicted.items()},
                {key: value[1] for key, value in predicted.items()},
                pairs,
            )
            entry["all_target_collision"] = collision
            _require(
                collision.get("surface_intersection_stage_passed") is True,
                "central P6 all-target collision screen rejected predicted bridges",
            )
            entry["all_target_collision_passed"] = True

            if step != 0:
                entry["stage"] = "six-port-3D-continuity-and-calibre"
                _write_json(PROGRESS_PATH, report)
                candidate_surfaces = dict(strict_targets)
                candidate_surfaces.update(predicted)
                baseline_surfaces = dict(strict_targets)
                baseline_surfaces.update(rigid)
                continuity = _audit_pose_port_continuity(
                    verifier,
                    context["manifold"],
                    candidate_surfaces,
                    baseline_surfaces,
                    pairs,
                )
                entry["six_port_continuity"] = continuity
                _require(
                    continuity.get("passed") is True
                    and continuity.get("evaluated_port_count") == 6
                    and continuity.get("requested_port_count") == 6,
                    "source-bound six-port 3D continuity stopped or rejected",
                )
            else:
                entry["six_port_continuity"] = {
                    "status": "REUSED_FROZEN_SOURCE0_PORT_PROOFS",
                    "first_five": SOURCE0_PORT_REPORT_SHA256,
                    "526_to_472": SOURCE0_LAST_PORT_REPORT_SHA256,
                    "source0_self_calibre_limitation": True,
                }
            entry["status"] = "N27_CAPTURE_PREDICTED_GEOMETRY_SCREEN_PASSED_UNADMITTED"
            entry["stage"] = "complete"
        except Exception as error:
            entry["status"] = "N27_CAPTURE_REJECTED_OR_INCOMPLETE"
            entry["failure"] = {"type": type(error).__name__, "message": str(error)}
            report["status"] = "STOPPED_AT_FIRST_N27_CAPTURE_FAILURE"
            report["first_failure"] = {
                "capture_kind": "N27 accepted capture",
                "accepted_step": step,
                "stage": entry["stage"],
                **entry["failure"],
            }
            entry["elapsed_seconds"] = time.perf_counter() - started
            _write_json(PROGRESS_PATH, report)
            raise
        entry["elapsed_seconds"] = time.perf_counter() - started
        _write_json(PROGRESS_PATH, report)


def _verify_late_transfer(context, report) -> None:
    """Evaluate the N22 pose as a separate transfer capture, never N27 continuation."""
    verifier = context["verifier"]
    retention_owner = context["retention_owner"]
    prep = context["preparation"]
    candidate_rows = context["candidate_rows"]
    baseline_rows = context["baseline_rows"]
    transfer_rows = context["transfer_rows"]
    baseline_map = context["baseline_map"]
    candidate_map = context["candidate_map"]
    repair_surfaces = context["repair_surfaces"]
    transform = context["transform"]
    late_manifest = json.loads(LATE_INPUT.read_text())
    capture_row = late_manifest["capture_rows"][0]
    started = time.perf_counter()
    late_started_utc = datetime.now(UTC)
    entry = {
        "capture_kind": "N22 late transfer capture",
        "accepted_step": TRANSFER_STEP,
        "accepted_time_s": TRANSFER_TIME_S,
        "recorded_start_utc": late_started_utc.isoformat(),
        "stage": "load-separate-transfer-capture",
        "status": "RUNNING",
        "N27_continuation_claim": False,
        "source_compiler_invoked": False,
        "native_or_gpu_work_performed": False,
        "source_geometry_mutated": False,
    }
    report["late_transfer_capture"] = entry
    _write_json(PROGRESS_PATH, report)
    captured, identity, capture_meta = _load_capture(
        verifier, capture_row, expected_step=TRANSFER_STEP
    )
    _require(
        abs(capture_meta["accepted_time_s"] - TRANSFER_TIME_S) <= 1e-9,
        "late accepted capture is not the pinned 29.95-second transfer pose",
    )
    receipt = capture_meta["receipt"]
    late_source = late_manifest["source"]
    entry["capture_identity"] = {
        "pack_sha256": identity["pack_sha256"],
        "receipt_sha256": identity["capture_receipt_sha256"],
        "accepted_step": TRANSFER_STEP,
        "accepted_time_s": capture_meta["accepted_time_s"],
        "body20_pose": capture_meta["body20"],
        "source_anatomy_payload_sha256": late_source["sha256"],
    }
    entry["common_field_source"] = _verify_capture_common_field(
        receipt,
        expected_source_sha=late_source["sha256"],
        preparation=prep,
        label="N22 late transfer capture",
    )
    retention_owner = context["retention_owner"]
    owners = _capture_owner_map(
        retention_owner, Path(capture_row["pack_path"]), captured
    )
    binding, strict_targets = _audit_capture_row_bindings(
        surfaces=captured,
        captured_owners=owners,
        transfer_rows=transfer_rows,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        baseline_source_map=baseline_map,
        candidate_source_map=candidate_map,
        repair_surfaces=repair_surfaces,
        verifier=retention_owner,
        transfer_label="N22 late transfer capture",
        source_bound_keys=SOURCE_BOUND_TYPED_KEYS,
        arrangement_candidates=context.get("arrangement_candidates"),
    )
    entry["source_row_binding"] = binding
    _require(
        binding.get("passed") is True,
        "late per-target exact source row, source identity, or owner binding failed",
    )
    entry["compiled_body20_row_emission"] = (
        source007._body20_source_to_owner_f32_emission(
            candidate_rows,
            capture_meta["body20"],
            None,
            candidate_map,
            transform,
            comparison_surfaces=strict_targets,
            captured_owners=owners,
        )
    )
    entry["whole_source_payload_hash_difference"] = {
        "N22_source_payload_sha256": late_source["sha256"],
        "N27_capture_source_payload_sha256": N27_CAPTURE_SOURCE_SHA256,
        "whole_hashes_equal": late_source["sha256"] == N27_CAPTURE_SOURCE_SHA256,
        "disposition": (
            "not used as a blanket rejection; every captured target stable ID is "
            "compared by normalized ABI5 row bytes, source identity, and body owner"
        ),
        "row_binding_passed": True,
    }

    rigid = source007._bridge_surfaces(
        candidate_rows, capture_meta["body20"], transform
    )
    predicted, prediction = source007._predict_step0_bridges(
        candidate_rows, rigid, captured, capture_meta["body20"], prep, transform
    )
    entry["bridge_prediction"] = {
        "status": "OFFLINE_TRANSFER_CAPTURE_SVC_COMMON_FIELD_PREDICTION",
        "bridges": prediction,
        "native_common_field_error_bound": "not_available",
        "prediction_roundoff_bound_limit": "Interpolation-only F32 bound; no native map/polynomial/kernel error bound.",
        "native_confirmation_required": True,
    }
    pairs, _attribution, _stationarity = source007._read_port_pairs()
    entry["stage"] = "P6-all-decoded-target-collision-screen"
    _write_json(PROGRESS_PATH, report)
    collision = gate.audit_captured_pose_surface_intersections(
        verifier,
        strict_targets,
        {key: value[0] for key, value in predicted.items()},
        {key: value[1] for key, value in predicted.items()},
        pairs,
    )
    entry["all_target_collision"] = collision
    _require(
        collision.get("surface_intersection_stage_passed") is True,
        "central P6 all-target collision screen rejected late transfer capture",
    )
    entry["all_target_collision_passed"] = True

    entry["stage"] = "six-port-3D-continuity-and-calibre"
    _write_json(PROGRESS_PATH, report)
    candidate_surfaces = dict(strict_targets)
    candidate_surfaces.update(predicted)
    baseline_surfaces = dict(strict_targets)
    baseline_surfaces.update(rigid)
    continuity = _audit_pose_port_continuity(
        verifier, context["manifold"], candidate_surfaces, baseline_surfaces, pairs
    )
    entry["six_port_continuity"] = continuity
    _require(
        continuity.get("passed") is True
        and continuity.get("evaluated_port_count") == 6
        and continuity.get("requested_port_count") == 6,
        "late transfer six-port 3D continuity stopped or rejected",
    )
    entry["status"] = "N22_LATE_TRANSFER_SCREEN_PASSED_UNADMITTED"
    entry["stage"] = "complete"
    entry["elapsed_seconds"] = time.perf_counter() - started
    _write_json(PROGRESS_PATH, report)



def _initialize_capture_record(
    context: dict,
    capture_row: dict,
    *,
    expected_step: int,
    expected_source_sha: str,
    transfer_rows: dict,
    transfer_label: str,
) -> dict:
    """Decode and bind one accepted capture without invoking geometry predicates."""
    verifier = context["verifier"]
    retention_owner = context["retention_owner"]
    surfaces, identity, meta = _load_capture(
        verifier, capture_row, expected_step=expected_step
    )
    common = _verify_capture_common_field(
        meta["receipt"], expected_source_sha=expected_source_sha,
        preparation=context["preparation"], label=transfer_label,
    )
    owners = _capture_owner_map(
        retention_owner, Path(capture_row["pack_path"]), surfaces
    )
    binding, _projected = _audit_capture_row_bindings(
        surfaces=surfaces,
        captured_owners=owners,
        transfer_rows=transfer_rows,
        baseline_rows=context["baseline_rows"],
        candidate_rows=context["candidate_rows"],
        baseline_source_map=context["baseline_map"],
        candidate_source_map=context["candidate_map"],
        repair_surfaces=context["repair_surfaces"],
        verifier=retention_owner,
        transfer_label=transfer_label,
        source_bound_keys=SOURCE_BOUND_TYPED_KEYS,
        arrangement_candidates=context.get("arrangement_candidates"),
    )
    entry = {
        "capture_kind": transfer_label,
        "accepted_step": expected_step,
        "accepted_time_s": meta["accepted_time_s"],
        "pack_sha256": identity["pack_sha256"],
        "receipt_sha256": identity["capture_receipt_sha256"],
        "body20_pose": meta["body20"],
        "decoded_typed_surface_count": len(surfaces),
        "common_field_source": common,
        "source_row_binding": binding,
        "geometry_predicates_invoked": False,
        "boolean_or_collision_work_invoked": False,
        "native_or_gpu_work_performed": False,
        "source_geometry_mutated": False,
        "passed": binding.get("passed") is True,
    }
    if not entry["passed"]:
        entry["status"] = "CAPTURE_DECODED_BUT_SOURCE_BINDING_REJECTED"
        return entry
    rigid = source007._bridge_surfaces(
        context["candidate_rows"], meta["body20"], context["transform"]
    )
    _predicted, prediction = source007._predict_step0_bridges(
        context["candidate_rows"], rigid, surfaces, meta["body20"],
        context["preparation"], context["transform"],
    )
    entry["bridge_motion_prediction"] = {
        "status": "OFFLINE_CAPTURED_SVC_COMMON_FIELD_PREDICTION",
        "bridges": prediction,
        "native_common_field_error_bound": "not_available",
        "float32_roundoff_bound_scope": (
            "Interpolation-only F32 arithmetic; no native common-map, polynomial, "
            "or kernel error bound."
        ),
        "native_confirmation_required": True,
    }
    entry["status"] = "SOURCE_CAPTURE_INITIALIZED_UNADMITTED"
    return entry


def initialize_source_capture() -> dict:
    """Load real source, N27 step 0 and separate N22 capture, then stop before geometry."""
    _require(PREFLIGHT_PATH.is_file(), "write the pinned preflight first")
    _require(
        not INITIALIZATION_PATH.exists(),
        "retain existing source/capture initialization; do not overwrite",
    )
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    _assert_preflight(preflight)
    started = time.perf_counter()
    report = {
        "schema": "numi.human.brachiocephalic-source-capture-initialize.v1",
        "status": "RUNNING_SOURCE_AND_CAPTURE_INITIALIZATION",
        "recorded_start_utc": datetime.now(UTC).isoformat(),
        "preflight_path": str(PREFLIGHT_PATH),
        "preflight_sha256": sha_file(PREFLIGHT_PATH),
        "source_repository": _git_state(),
        "source_compiler_invoked": False,
        "native_or_gpu_work_performed": False,
        "source_geometry_mutated": False,
        "decoded_geometry_arrays_loaded": True,
        "geometry_predicates_invoked": False,
        "boolean_or_collision_work_invoked": False,
        "anatomical_admission": False,
        "capture_initialization": [],
    }
    _write_json(INITIALIZATION_PATH, report)
    try:
        context = _runtime_input_context(preflight)
        report["source_initialization"] = {
            "N27_payload_sha256": N27_CAPTURE_SOURCE_SHA256,
            "N27_row_count": len(context["baseline_rows"]),
            "N27_source_identity_count": len(context["baseline_map"]),
            "N27_registration_fingerprint": int(context["baseline_header"][5]),
            "candidate_payload_sha256": preflight["source_candidate"]["payload_sha256"],
            "candidate_row_count": len(context["candidate_rows"]),
            "candidate_source_identity_count": len(context["candidate_map"]),
            "late_N22_payload_sha256": preflight["late_transfer"]["source_sha256"],
            "late_N22_row_count": len(context["transfer_rows"]),
            "NHA_registration_fingerprint_matches": (
                int(context["candidate_header"][5])
                == int(context["baseline_header"][5])
                == int(context["transfer_header"][5])
            ),
            "common_field_map_sha256": context["preparation"]["input_common_map_sha256"],
            "common_field_polynomials_sha256": context["preparation"]["input_common_polynomials_sha256"],
            "common_field_domain_boxes_sha256": context["preparation"]["input_common_domain_boxes_sha256"],
            "appended_bridge_owner_inputs": context["append_owner"],
        }
        _write_json(INITIALIZATION_PATH, report)
        n27_rows = _capture_rows(source007.CAPTURE_MANIFEST)
        late_source, late_row = _late_capture_row()
        specs = (
            (n27_rows[0], 0, N27_CAPTURE_SOURCE_SHA256,
             context["baseline_rows"], "N27 accepted source-0 capture"),
            (late_row, TRANSFER_STEP, late_source["sha256"],
             context["transfer_rows"], "N22 separate late transfer capture"),
        )
        for row, step, source_sha, transfer_rows, label in specs:
            entry = {"capture_kind": label, "accepted_step": step,
                     "stage": "decode-and-source-bind", "status": "RUNNING"}
            report["capture_initialization"].append(entry)
            _write_json(INITIALIZATION_PATH, report)
            try:
                result = _initialize_capture_record(
                    context, row, expected_step=step, expected_source_sha=source_sha,
                    transfer_rows=transfer_rows, transfer_label=label,
                )
                entry.update(result)
                if not result.get("passed"):
                    raise ValueError(
                        f"{label} failed exact source-row, identity, or owner binding"
                    )
            except Exception as error:  # noqa: BLE001 - retain earliest failure
                entry["status"] = "SOURCE_CAPTURE_INITIALIZATION_REJECTED_OR_INCOMPLETE"
                entry["failure"] = {"type": type(error).__name__, "message": str(error)}
                report["status"] = "STOPPED_AT_FIRST_SOURCE_CAPTURE_FAILURE"
                report["first_failure"] = {
                    "capture_kind": label, "accepted_step": step,
                    "stage": entry.get("stage"), **entry["failure"],
                }
                _write_json(INITIALIZATION_PATH, report)
                break
            entry["status"] = "SOURCE_CAPTURE_INITIALIZED_UNADMITTED"
            entry["stage"] = "complete"
            _write_json(INITIALIZATION_PATH, report)
        else:
            report["status"] = "SOURCE_AND_TWO_CAPTURE_INITIALIZATION_PASSED_UNADMITTED"
            report["capture_scope"] = {
                "N27_accepted_source0_step": 0,
                "N27_eight_capture_geometry_not_run": True,
                "late_capture": "separate N22 transfer at accepted step 14975",
                "N27_continuation_claim_for_late_capture": False,
            }
    except Exception as error:  # noqa: BLE001 - retain initialization failure
        report["status"] = "SOURCE_CAPTURE_INITIALIZATION_REJECTED_OR_INCOMPLETE"
        report["first_failure"] = {
            "stage": "source/capture initialization",
            "type": type(error).__name__, "message": str(error),
        }
    report["recorded_finish_utc"] = datetime.now(UTC).isoformat()
    report["elapsed_seconds"] = time.perf_counter() - started
    _write_json(INITIALIZATION_PATH, report)
    return {
        "status": report["status"],
        "initialization_path": str(INITIALIZATION_PATH),
        "initialization_sha256": sha_file(INITIALIZATION_PATH),
        "elapsed_seconds": report["elapsed_seconds"],
    }


def run() -> dict:
    _require(
        PREFLIGHT_PATH.is_file(), "write the no-geometry multipose preflight first"
    )
    _require(
        not RUN_DIR.exists(), "retain existing multipose attempt; do not overwrite"
    )
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    _assert_preflight(preflight)
    manifest_rows = _capture_rows(source007.CAPTURE_MANIFEST)
    started = time.perf_counter()
    RUN_DIR.mkdir(parents=True)
    report = {
        "schema": "numi.human.brachiocephalic-multipose-transfer-gate.v1",
        "status": "RUNNING_N27_POSES_THEN_SEPARATE_N22_TRANSFER",
        "recorded_start_utc": datetime.now(UTC).isoformat(),
        "preflight_path": str(PREFLIGHT_PATH),
        "preflight_sha256": sha_file(PREFLIGHT_PATH),
        "source_candidate": preflight["source_candidate"],
        "source_compiler_invoked": False,
        "native_or_gpu_work_performed": False,
        "source_geometry_mutated": False,
        "source0_reuse_scope": preflight["source0_reuse"],
        "N27_capture_count_requested": 8,
        "N27_continuation_claim_for_late_transfer": False,
        "native_common_field_prediction": (
            "offline bridge coordinates from accepted captured-SVC10 minus same-pose "
            "rigid-only NHA SVC10 through compiled ancestry; native map/kernel error "
            "is not bounded and native confirmation is mandatory"
        ),
        "source0_calibre_limitation": (
            "reused source-0 ratios compare same-source geometry and are a geometry "
            "sanity check only, not physical or clinical calibre preservation"
        ),
        "calibre_baseline_for_nonzero_captures": (
            "same accepted target surfaces plus same-pose rigid-only bridge; this "
            "isolates predicted bridge displacement and is not a clinical baseline"
        ),
        "source006_rows_and_owner": None,
        "anatomical_admission": False,
        "all_pose_admission": False,
        "native_admission": False,
    }
    _write_json(PROGRESS_PATH, report)
    try:
        context = _runtime_input_context(preflight)
        report["appended_bridge_owner_inputs"] = context["append_owner"]
        report["source_rows_and_owner_verification"] = {
            "N27_capture_source_payload_sha256": N27_CAPTURE_SOURCE_SHA256,
            "late_transfer_source_payload_sha256": preflight["late_transfer"][
                "source_sha256"
            ],
            "late_transfer_payload_hash_mismatch_expected": True,
            "method": (
                "Per captured target stable ID: exact normalized ABI5 source-row "
                "bytes, N27 source ID/provenance/body/layer identity, accepted-pack "
                "primitive owner, and exact retained face ancestry when candidate "
                "source rows use the pinned 313/315/317 fragment repair."
            ),
            "candidate_output": {
                "payload_sha256": sha_file(
                    Path(preflight["source_candidate"]["payload_path"])
                ),
                "receipt_sha256": preflight["source_candidate"]["receipt_sha256"],
                "preparation_sha256": preflight["source_candidate"][
                    "preparation_sha256"
                ],
            },
        }
        _write_json(PROGRESS_PATH, report)
        _verify_n27_capture_rows(manifest_rows, context, report)
        report["status"] = "N27_EIGHT_CAPTURE_SCREENS_PASSED_UNADMITTED"
        report["N27_capture_count_passed"] = 8
        _write_json(PROGRESS_PATH, report)
        _verify_late_transfer(context, report)
        report["status"] = (
            "EIGHT_N27_CAPTURES_AND_SEPARATE_LATE_N22_TRANSFER_SCREEN_PASSED_UNADMITTED"
        )
        report["late_transfer_scope"] = {
            "accepted_step": TRANSFER_STEP,
            "accepted_time_s": TRANSFER_TIME_S,
            "source_capture": "N22",
            "N27_continuation_claim": False,
        }
        report["acceptance_gates_remaining"] = [
            "native common-map/emitter confirmation",
            "remaining route-wide artery, muscle, and all Human completion gates",
        ]
    except Exception as error:  # noqa: BLE001 - retain exact failed stage and partial evidence
        failure = {"type": type(error).__name__, "message": str(error)}
        if "first_failure" not in report:
            late = report.get("late_transfer_capture")
            if isinstance(late, dict):
                if late.get("status") == "RUNNING":
                    late["status"] = "N22_LATE_TRANSFER_REJECTED_OR_INCOMPLETE"
                    late["failure"] = failure
                    late["recorded_finish_utc"] = datetime.now(UTC).isoformat()
                    late["elapsed_seconds"] = max(
                        0.0,
                        (
                            datetime.fromisoformat(late["recorded_finish_utc"])
                            - datetime.fromisoformat(late["recorded_start_utc"])
                        ).total_seconds(),
                    )
                    _write_json(PROGRESS_PATH, report)
                report["first_failure"] = {
                    "capture_kind": "N22 late transfer capture",
                    "accepted_step": TRANSFER_STEP,
                    "stage": late.get("stage"),
                    **failure,
                }
            else:
                report["first_failure"] = {
                    "stage": "source/capture initialization",
                    **failure,
                }
        report["status"] = (
            "STOPPED_AT_FIRST_N27_CAPTURE_FAILURE"
            if report["first_failure"].get("capture_kind") == "N27 accepted capture"
            else "MULTIPOSE_TRANSFER_REJECTED_OR_INCOMPLETE"
        )
        report["failure"] = failure
        report["anatomical_admission"] = False
        report["all_pose_admission"] = False
        report["native_admission"] = False
    report["recorded_finish_utc"] = datetime.now(UTC).isoformat()
    report["elapsed_seconds"] = time.perf_counter() - started
    _write_json(REPORT_PATH, report)
    return {
        "status": report["status"],
        "report_path": str(REPORT_PATH),
        "report_sha256": sha_file(REPORT_PATH),
        "elapsed_seconds": report["elapsed_seconds"],
    }


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="write hash-pinned source/pack preflight without decoding geometry",
    )
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="decode pinned source and N27 step 0/N22 capture, then stop before geometry predicates",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="run the bounded eight-pose and separate late transfer gates",
    )
    args = parser.parse_args(argv)
    selected = [args.preflight, args.initialize_only, args.run]
    _require(sum(selected) == 1, "choose exactly one of --preflight, --initialize-only, or --run")
    result = (
        write_preflight()
        if args.preflight
        else initialize_source_capture()
        if args.initialize_only
        else run()
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if not str(result["status"]).endswith(("REJECTED", "INCOMPLETE")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
