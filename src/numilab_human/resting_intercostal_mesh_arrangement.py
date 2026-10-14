"""Local, source-bound ABI5 composition for retained intercostal arrangement patches.

This is a narrow source owner for stable IDs 313, 315, and 317. It keeps the
original NHA vertex slots and unchanged source faces, uses only exact positive
parent maps for appended vertices, and emits the existing ABI5 row layout.
It does not qualify the resulting anatomy or its motion.
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

N27_PAYLOAD_SHA256 = "cde8705a813fe56686776c9cc038cbf06f413e1de76099d2a019d5e1b748a8d4"
N27_RECEIPT_SHA256 = "944bc88d257137f2bedd5898404c45cefd2901d719707519771ae5f4cfce3752"
REPAIR_PARENT_RECEIPT_SHA256 = "92a2215222230e4f1ad02a06583240e5c569c57807c2b9558c289c495c62fa2e"
REPAIR_PARENT_PAYLOAD_SHA256 = "50ab8248d2e0f55dd6cb8b24afc34c8844061d5841ecdf89d7daf03288816bfd"
REPAIR_REPORT_SHA256 = "9327e129cc630c24d2abe1e316fe0077568729f2600dee40359da50e910f1bd6"
REPAIRED_IDS = frozenset((313, 315, 317))


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
) -> dict:
    """Load existing full_candidate/lineage/report files under explicit pins."""
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
    return {
        **loaded,
        "source_payload_sha256": source_payload_sha256,
        "source_payload_index": source_payload_index,
        "candidate": candidate,
        "lineage": lineage,
        "report": report,
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


def compose_pinned_arrangement_set(
    *,
    base_rows: dict,
    repair_context: dict,
    arrangement_descriptors: dict[int, dict],
) -> tuple[dict, dict]:
    """Rebuild the three local arrangement rows from only their pinned parents."""
    _require(set(map(int, base_rows)) >= REPAIRED_IDS,
             "pinned N27 base lacks one of the three intercostal rows")
    _require(set(map(int, arrangement_descriptors)) == REPAIRED_IDS,
             "arrangement descriptors must cover exactly IDs 313/315/317")
    composed = {}
    summaries = {}
    for stable_id in sorted(REPAIRED_IDS):
        descriptor = arrangement_descriptors[stable_id]
        _require(isinstance(descriptor, dict), f"arrangement descriptor {stable_id} is malformed")
        arrangement = load_pinned_arrangement(
            candidate_path=Path(descriptor["candidate_path"]),
            candidate_sha256=descriptor["candidate_sha256"],
            lineage_path=Path(descriptor["lineage_path"]),
            lineage_sha256=descriptor["lineage_sha256"],
            report_path=Path(descriptor["report_path"]),
            report_sha256=descriptor["report_sha256"],
            source_payload_sha256=descriptor["source_payload_sha256"],
            source_payload_index=descriptor["source_payload_index"],
        )
        row, summary = compose_intercostal_arrangement_row(
            stable_id=stable_id,
            source_row=base_rows[stable_id],
            parent_row=repair_context["repair_parent_rows"][stable_id],
            arrangement=arrangement,
            repair_report=repair_context["repair_report"],
        )
        composed[stable_id] = row
        summaries[str(stable_id)] = {
            key: value for key, value in summary.items()
            if key not in {"candidate_vertex_parent_maps", "candidate_faces"}
        }
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
    *,
    stable_id: int,
    source_vertices6: np.ndarray,
    points_exact,
    parent_maps,
    source_payload_index: int,
) -> tuple[np.ndarray, tuple[tuple[tuple[int, Fraction], ...], ...]]:
    source_xyz = np.ascontiguousarray(source_vertices6[:, :3], dtype="<f4")
    _require(
        source_vertices6.ndim == 2 and source_vertices6.shape[1] == 6
        and np.isfinite(source_vertices6).all(),
        "source ABI5 row vertices are invalid",
    )
    _require(
        len(points_exact) == len(parent_maps),
        "full_candidate point and parent-map vertex counts differ",
    )
    points = np.empty((len(points_exact), 3), dtype="<f4")
    parsed_maps = []
    for vertex_index, (point_row, map_row) in enumerate(zip(points_exact, parent_maps)):
        _require(
            isinstance(point_row, (list, tuple)) and len(point_row) == 3
            and isinstance(map_row, list) and bool(map_row),
            f"full_candidate vertex {vertex_index} lacks exact XYZ or parent terms",
        )
        exact_point = tuple(
            parse_fraction_pair(value, f"full_candidate point {vertex_index}")
            for value in point_row
        )
        weighted_xyz = [Fraction(0), Fraction(0), Fraction(0)]
        parsed_terms = []
        seen = set()
        weight_sum = Fraction(0)
        for term in map_row:
            _require(
                isinstance(term, (list, tuple)) and len(term) == 5,
                f"parent term {vertex_index} has the wrong schema",
            )
            payload_index, parent_id, parent_vertex = term[:3]
            _require(
                all(type(value) is int for value in (payload_index, parent_id, parent_vertex))
                and payload_index == source_payload_index
                and parent_id == stable_id
                and 0 <= parent_vertex < len(source_xyz),
                f"parent term {vertex_index} escapes its pinned N27 source row",
            )
            key = (payload_index, parent_id, parent_vertex)
            _require(key not in seen, "parent map repeats a source vertex")
            seen.add(key)
            weight = parse_fraction_pair(term[3:5], f"parent weight {vertex_index}")
            _require(weight > 0, "parent weights must be strictly positive")
            weight_sum += weight
            parsed_terms.append((parent_vertex, weight))
            for axis in range(3):
                weighted_xyz[axis] += Fraction.from_float(
                    float(source_xyz[parent_vertex, axis])
                ) * weight
        _require(weight_sum == 1, f"parent weights do not sum exactly to one at vertex {vertex_index}")
        _require(
            all(
                Fraction.from_float(float(np.float32(float(exact_point[axis]))))
                == Fraction.from_float(float(np.float32(float(weighted_xyz[axis]))))
                for axis in range(3)
            ),
            f"full_candidate point {vertex_index} is not canonical F32 of its exact positive map",
        )
        points[vertex_index] = [float(value) for value in exact_point]
        parsed_maps.append(tuple(parsed_terms))
    return points, tuple(parsed_maps)


def compose_intercostal_arrangement_row(
    *,
    stable_id: int,
    source_row: dict,
    parent_row: dict,
    arrangement: dict,
    repair_report: dict,
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
    )
    live_faces = []
    live_ancestry = []
    for slot, (face, origin) in enumerate(zip(raw_faces, ancestry)):
        _require(
            isinstance(face, (list, tuple))
            and len(face) == 3
            and all(type(index) is int for index in face)
            and min(face) >= 0
            and max(face) < len(points),
            f"full_candidate face {slot} has invalid local vertex indices",
        )
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
    qfaces = np.ascontiguousarray(np.asarray(live_faces, dtype="<u4"))
    original_slots = [
        slot for slot, origin in enumerate(live_ancestry)
        if origin["origin_kind"] == 0
    ]
    original_parent_faces = [
        live_ancestry[slot]["parent_face_index"] for slot in original_slots
    ]
    _require(
        original_slots == sorted(original_slots)
        and original_parent_faces == list(range(len(parent_faces))),
        "origin_kind=0 does not preserve every 028 source face exactly once in source order",
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
    _require(
        used_qvertices == set(range(len(points))),
        "full_candidate contains unreferenced quotient vertices",
    )
    patch_parent_faces = []
    patch_support = {}
    for slot in patch_slots:
        face_id = live_ancestry[slot]["parent_face_index"]
        parent_corner_ids = set(map(int, parent_faces[face_id]))
        for qvertex in qfaces[slot]:
            terms = parent_maps[int(qvertex)]
            support = {source_vertex for source_vertex, _weight in terms}
            _require(
                support.issubset(parent_corner_ids),
                f"patch face {slot} has positive parent support outside its declared parent triangle",
            )
            patch_support.setdefault(int(qvertex), set()).update(support)
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
    faces = np.ascontiguousarray(
        np.concatenate((parent_faces, mapped_patch_faces), axis=0), dtype="<u4"
    )
    _require(
        np.isfinite(vertices6).all()
        and int(faces.max(initial=0)) < len(vertices6)
        and len(vertices6) > len(source_vertices6),
        "arrangement did not append valid source-bound vertices",
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
        "retained_n27_face_map_sha256": sha_bytes(retained_n27_face_ids.astype("<u8").tobytes()),
        "retained_n27_source_face_count": len(retained_n27_face_ids),
        "full_candidate_sha256": arrangement["candidate_sha256"],
        "lineage_sha256": arrangement["lineage_sha256"],
        "report_sha256": arrangement["report_sha256"],
        "compiled_row_sha256": sha_bytes(_normalized_row_bytes(candidate_row)),
        "compiled_candidate_vertices6_prefix_matches_entire_N27_row": True,
        "all_028_faces_preserved_byte_exact_in_original_order": True,
        "full_candidate_origin_kind0_face_count": len(original_slots),
        "full_candidate_patch_face_count": len(patch_slots),
        "patch_parent_face_ids_in_N27_namespace": patch_parent_faces,
        "new_appended_vertex_count": len(appended_parent_maps),
        "positive_exact_parent_map_for_each_appended_vertex": True,
        "new_child_normals": "Float64 weighted interpolation of exact positive N27 parent normals, unit-normalized and emitted as Float32",
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
            "source_payload_sha256": descriptor.get("source_payload_sha256"),
            "source_payload_index": descriptor.get("source_payload_index"),
        }
    composed, summaries = compose_pinned_arrangement_set(
        base_rows=base_rows, repair_context=context, arrangement_descriptors=descriptors)
    for stable_id in base_rows:
        candidate = candidate_rows[stable_id]
        if stable_id in REPAIRED_IDS:
            row_record = next(record for record in records if int(record["stable_id"]) == stable_id)
            summary = summaries[str(stable_id)]
            _require(_normalized_row_bytes(composed[stable_id])
                     == _normalized_row_bytes(candidate),
                     f"compiled ABI5 row differs from exact source composition for {stable_id}")
            _require(row_record.get("compiled_row_sha256") == summary["compiled_row_sha256"]
                     and row_record.get("new_appended_vertex_count")
                     == summary["new_appended_vertex_count"]
                     and row_record.get("retained_n27_face_map_sha256")
                     == summary["retained_n27_face_map_sha256"],
                     f"receipt row repair proof differs from reconstructed row {stable_id}")
            actual_id = source_map.get(str(stable_id), {})
            expected_id = dict(base_map.get(str(stable_id), {}))
            actual_id = json.loads(json.dumps(actual_id))
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
        else:
            _require(_normalized_row_bytes(base_rows[stable_id])
                     == _normalized_row_bytes(candidate),
                     f"unrelated original N27 row changed: {stable_id}")
            _require(source_map.get(str(stable_id)) == base_map.get(str(stable_id)),
                     f"unrelated original source identity changed: {stable_id}")
    return {
        "base_row_count": len(base_rows),
        "unchanged_original_row_count": len(base_rows) - len(REPAIRED_IDS),
        "repaired_original_row_count": len(REPAIRED_IDS),
        "bridge_appended_row_count": 2,
        "all_520_unmodified_rows_byte_exact": True,
        "all_three_repaired_rows_reconstructed_from_pinned_positive_maps": True,
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
) -> dict:
    """Reconstruct the only permitted row composition and compare exact ABI5 bytes."""
    expected, summary = compose_intercostal_arrangement_row(
        stable_id=stable_id,
        source_row=source_row,
        parent_row=parent_row,
        arrangement=arrangement,
        repair_report=repair_report,
    )
    _require(
        _normalized_row_bytes(expected) == _normalized_row_bytes(candidate_row),
        f"compiled ABI5 row differs from exact source-prefix/face/parent-map composition for {stable_id}",
    )
    summary["compiled_row_sha256"] = sha_bytes(_normalized_row_bytes(candidate_row))
    return summary
