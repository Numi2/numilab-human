"""File-backed proof for the 1284 forearm capture carrier.

Canonical material GIDs are reconstructed only through the authenticated source-slot
map and exact ordered face-corner ownership. MRVPACK global vertex IDs are not
canonical material identities. Actual captured positions and predicted changed
positions are explicitly separate bases.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("forearm actual capture owner: " + message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _full21_rows_sha256(gids, values) -> str:
    ids = np.asarray(gids)
    array = np.asarray(values)
    _require(ids.ndim == 1 and ids.dtype.kind in "iu" and len(np.unique(ids)) == len(ids),
             "full21 proof GIDs must be unique integer keys")
    _require(array.ndim >= 1 and array.shape[0] == len(ids),
             "full21 proof rows do not align with canonical GIDs")
    order = np.argsort(ids, kind="stable")
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(ids[order].astype("<u8", copy=False)).tobytes())
    digest.update(np.ascontiguousarray(array[order]).tobytes())
    return digest.hexdigest()


def _canonical_file(path) -> Path:
    return Path(path).resolve(strict=True)


def _pinned_digest(path: Path, pins: dict[str, str]) -> str:
    resolved = str(path.resolve(strict=True))
    expected = pins.get(resolved)
    _require(isinstance(expected, str) and len(expected) == 64,
             f"input is not present in the prepared carrier pin map: {resolved}")
    actual = _sha256(path)
    _require(actual == expected, f"prepared carrier input SHA changed: {resolved}")
    return actual


def _source_slot_owner_map(source_slots, canonical_gids, weights, source_vertex_count):
    slots = np.asarray(source_slots)
    gids = np.asarray(canonical_gids)
    field = np.asarray(weights)
    _require(slots.dtype == np.dtype("<i8") and slots.ndim == 1,
             "006 source-slot map is not Int64")
    _require(gids.dtype == np.dtype("<u8") and gids.shape == slots.shape,
             "006 canonical GIDs do not align with source slots")
    _require(field.dtype == np.dtype("<f8") and field.shape == (len(slots), 21)
             and np.isfinite(field).all() and np.all(field > 0)
             and np.max(np.abs(field.sum(axis=1) - 1.0)) <= 1e-12,
             "006 full21 source weights are malformed")
    owners = {}
    for slot_value, gid_value, weight in zip(slots, gids, field, strict=True):
        slot, gid = int(slot_value), int(gid_value)
        _require(slot >= -1 and slot < source_vertex_count,
                 "006 source slot is outside the original ff8 row")
        if slot == -1:  # 006 inferred births are not vertices in accepted run002.
            continue
        weight = np.ascontiguousarray(weight, dtype="<f8")
        old = owners.get(slot)
        if old is None:
            owners[slot] = (gid, weight)
        else:
            _require(old[0] == gid and old[1].tobytes() == weight.tobytes(),
                     f"006 has conflicting material owners for source slot {slot}")
    return owners



def _validate_full21_parent_lane_routes(binding_indices, binding_weights, local_ids,
                                         core_body_indices, *, stable_id, source_slot):
    """Prove every extension term selects its exact one-hot ordered parent lane."""
    indices = np.asarray(binding_indices)
    weights = np.asarray(binding_weights)
    local_ids = np.asarray(local_ids)
    _require(indices.dtype == np.dtype("uint32") and indices.ndim == 2
             and indices.shape[1] == 4
             and weights.dtype == np.dtype("float32") and weights.shape == indices.shape
             and np.isfinite(weights).all(),
             f"row {stable_id} active parent carrier lacks exact four-lane route arrays")
    _require(local_ids.ndim == 1 and local_ids.dtype.kind in "iu"
             and len(local_ids) == 21 and np.all(local_ids >= 0)
             and np.all(local_ids < len(indices)),
             f"row {stable_id} source slot {source_slot} has invalid parent lane indices")
    _require(len(set(int(value) for value in local_ids)) == 21,
             f"row {stable_id} source slot {source_slot} repeats a parent lane vertex")
    _require(len(core_body_indices) == 21 and len(set(core_body_indices)) == 21,
             f"row {stable_id} active parent carrier does not have 21 ordered bodies")
    expected_route_weights = np.asarray([1, 0, 0, 0], dtype="<f4")
    for lane, local_id_value in enumerate(local_ids):
        local_id = int(local_id_value)
        expected_route_indices = np.zeros(4, dtype="<u4")
        expected_route_indices[0] = lane
        _require(indices[local_id].tobytes() == expected_route_indices.tobytes()
                 and weights[local_id].tobytes() == expected_route_weights.tobytes(),
                 f"row {stable_id} source slot {source_slot} term {lane} "
                 "does not reference its exact one-hot owner lane")

def authenticate_shared_actual_capture_gid_points(rows, *, first_row=132, second_row=134):
    """Compare canonical shared-GID points in two authenticated actual captures.

    This is deliberately a point-identity report, not an intersection waiver.
    Exact geometry predicates still decide whether pair intersections lie wholly
    on a shared point/edge; mismatched actual positions are retained as failures.
    """
    normalized = {}
    for row_id in (first_row, second_row):
        row = rows.get(str(row_id), rows.get(row_id))
        _require(isinstance(row, dict), f"missing authenticated row {row_id}")
        report = row.get("report")
        _require(isinstance(report, dict)
                 and report.get("ordered_face_corner_mapping_verified") is True,
                 f"row {row_id} lacks exact ordered source/capture face mapping")
        positions = np.ascontiguousarray(row.get("actual_capture_positions_f32"), dtype="<f4")
        gids = np.asarray(row.get("canonical_gids_u64"))
        slots = np.asarray(row.get("capture_to_source_slot_i64"))
        faces = np.asarray(row.get("actual_capture_faces_u32"))
        _require(positions.ndim == 2 and positions.shape[1] == 3
                 and np.isfinite(positions).all(), f"row {row_id} positions are malformed")
        _require(gids.dtype == np.dtype("<u8") and gids.shape == (len(positions),)
                 and len(np.unique(gids)) == len(gids),
                 f"row {row_id} canonical GIDs are malformed or duplicated")
        _require(slots.dtype == np.dtype("<i8") and slots.shape == gids.shape
                 and np.all(slots >= 0), f"row {row_id} source slots are malformed")
        _require(faces.dtype == np.dtype("<u4") and faces.ndim == 2
                 and faces.shape[1] == 3 and int(faces.max(initial=0)) < len(positions),
                 f"row {row_id} ordered capture faces are malformed")
        ordered_slots = np.ascontiguousarray(slots[faces], dtype="<i8")
        ordered_gids = np.ascontiguousarray(gids[faces], dtype="<u8")
        _require(report.get("ordered_face_corner_source_slots_sha256")
                 == hashlib.sha256(ordered_slots.tobytes()).hexdigest()
                 and report.get("ordered_face_corner_canonical_gids_sha256")
                 == hashlib.sha256(ordered_gids.tobytes()).hexdigest(),
                 f"row {row_id} ordered face-corner map changed after authentication")
        by_gid = {int(gid): index for index, gid in enumerate(gids)}
        normalized[row_id] = {
            "positions": positions, "gids": gids, "slots": slots, "faces": faces,
            "ordered_slots": ordered_slots, "ordered_gids": ordered_gids,
            "by_gid": by_gid,
        }

    a, b = normalized[first_row], normalized[second_row]
    common = sorted(set(a["by_gid"]) & set(b["by_gid"]))
    matching, mismatched = [], []
    for gid in common:
        ia, ib = a["by_gid"][gid], b["by_gid"][gid]
        pa, pb = a["positions"][ia], b["positions"][ib]
        record = {
            "canonical_gid": gid,
            str(first_row): {"capture_vertex_index": ia, "source_slot": int(a["slots"][ia]),
                             "xyz_f32": pa.tolist()},
            str(second_row): {"capture_vertex_index": ib, "source_slot": int(b["slots"][ib]),
                              "xyz_f32": pb.tolist()},
            "actual_capture_xyz_byte_equal": pa.tobytes() == pb.tobytes(),
        }
        (matching if record["actual_capture_xyz_byte_equal"] else mismatched).append(record)
    return {
        "schema": "numi.human.shared-actual-capture-gid-points.v1",
        "row_identities": [[51005, first_row], [51005, second_row]],
        "shared_canonical_gid_count": len(common),
        "exact_shared_actual_point_count": len(matching),
        "mismatched_shared_gid_position_count": len(mismatched),
        "all_shared_gid_actual_positions_equal": not mismatched,
        "ordered_face_corner_maps": {
            str(row_id): {
                "verified_against_exact_ordered_source_capture_crosswalk": True,
                "corner_count": int(normalized[row_id]["ordered_slots"].size),
                "source_slot_map_sha256": hashlib.sha256(
                    normalized[row_id]["ordered_slots"].tobytes()).hexdigest(),
                "canonical_gid_map_sha256": hashlib.sha256(
                    normalized[row_id]["ordered_gids"].tobytes()).hexdigest(),
            } for row_id in (first_row, second_row)
        },
        "exact_shared_actual_points": matching,
        "position_mismatches": mismatched,
        "geometry_intersection_waiver": False,
    }


def authenticate_actual_capture_row(
    *,
    source_faces_u32,
    source_vertex_count: int,
    capture_positions_f32,
    capture_faces_u32,
    source_map_slots_i64,
    source_map_gids_u64,
    source_map_weights_f64,
    prepared_positions_f32,
    prepared_faces_i64,
    prepared_source_slots_i64,
    prepared_gids_u64,
    prepared_weights_f64,
    index_candidate_positions_f32,
    index_candidate_gids_u64,
    index_candidate_faces_source_slots_u32,
    index_candidate_faces_capture_u32,
):
    """Bind one prepared row to actual captured Float32 positions by source ancestry.

    This uses the established exact ordered-face-corner source→capture owner. The
    1274 face patch is verified separately: its source-slot corners must map
    exactly into capture-local vertices, but this does not by itself prove the
    patch's source-face ancestry.
    """
    from . import resting_brachiocephalic_current_capture_network_preparation as prep

    source_faces = np.ascontiguousarray(source_faces_u32, dtype="<u4")
    capture_positions = np.ascontiguousarray(capture_positions_f32, dtype="<f4")
    capture_faces = np.ascontiguousarray(capture_faces_u32, dtype="<u4")
    _require(source_faces.ndim == 2 and source_faces.shape[1] == 3,
             "ff8 source faces must be an ordered triangle array")
    _require(capture_positions.ndim == 2 and capture_positions.shape[1] == 3
             and np.isfinite(capture_positions).all(),
             "accepted capture positions must be finite Float32 XYZ")
    _require(capture_faces.ndim == 2 and capture_faces.shape[1] == 3,
             "accepted capture faces must be an ordered triangle array")
    _require(type(source_vertex_count) is int and source_vertex_count > 0,
             "original source vertex count is invalid")

    source_to_capture, map_report = prep._source_to_capture_vertex_map(
        source_faces, capture_faces, source_vertex_count, len(capture_positions))
    _require(source_to_capture is not None
             and map_report.get("status") == "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED",
             "original ff8 faces do not establish an exact capture vertex crosswalk")
    source_to_capture = np.asarray(source_to_capture, dtype="<i8")
    ordered_source_face_capture_indices = np.ascontiguousarray(
        source_to_capture[source_faces], dtype="<u4")
    _require(ordered_source_face_capture_indices.tobytes() == capture_faces.tobytes(),
             "ordered source-face corners do not reproduce accepted capture face order")
    owners = _source_slot_owner_map(
        source_map_slots_i64, source_map_gids_u64, source_map_weights_f64,
        source_vertex_count,
    )
    capture_to_source = np.full(len(capture_positions), -1, dtype="<i8")
    capture_gids = np.zeros(len(capture_positions), dtype="<u8")
    capture_weights = np.zeros((len(capture_positions), 21), dtype="<f8")
    for source_slot, capture_index_value in enumerate(source_to_capture):
        capture_index = int(capture_index_value)
        if capture_index < 0:
            continue
        _require(source_slot in owners,
                 f"captured source slot {source_slot} has no canonical GID/full21 owner in 006")
        _require(capture_to_source[capture_index] == -1,
                 f"capture vertex {capture_index} maps to multiple source slots")
        gid, weight = owners[source_slot]
        capture_to_source[capture_index] = source_slot
        capture_gids[capture_index] = gid
        capture_weights[capture_index] = weight
    _require(np.all(capture_to_source >= 0),
             "accepted capture has a vertex without exact ff8 source-slot ancestry")
    _require(len(np.unique(capture_gids)) == len(capture_gids),
             "source-slot crosswalk does not yield unique canonical GIDs")

    prepared_positions = np.ascontiguousarray(prepared_positions_f32, dtype="<f4")
    prepared_faces = np.ascontiguousarray(prepared_faces_i64, dtype="<i8")
    prepared_slots = np.asarray(prepared_source_slots_i64)
    prepared_gids = np.asarray(prepared_gids_u64)
    prepared_weights = np.asarray(prepared_weights_f64)
    _require(prepared_positions.shape == capture_positions.shape
             and prepared_positions.tobytes() == capture_positions.tobytes(),
             "1284 prepared positions are not byte-exact accepted capture positions")
    _require(prepared_faces.shape == capture_faces.shape
             and np.ascontiguousarray(prepared_faces.astype("<u4")).tobytes()
             == capture_faces.tobytes(),
             "1284 prepared faces are not byte-exact accepted capture faces")
    _require(prepared_slots.dtype == np.dtype("<i8")
             and prepared_slots.shape == capture_to_source.shape
             and prepared_slots.tobytes() == capture_to_source.tobytes(),
             "1284 capture→source-slot map differs from exact ff8 face ownership")
    _require(prepared_gids.dtype == np.dtype("<u8")
             and prepared_gids.shape == capture_gids.shape
             and prepared_gids.tobytes() == capture_gids.tobytes(),
             "1284 canonical GIDs differ from the 006 slot map replay")
    _require(prepared_weights.dtype == np.dtype("<f8")
             and prepared_weights.shape == capture_weights.shape
             and prepared_weights.tobytes() == capture_weights.tobytes(),
             "1284 full21 weights differ from the 006 source owner")

    candidate_positions = np.ascontiguousarray(index_candidate_positions_f32, dtype="<f4")
    candidate_gids = np.asarray(index_candidate_gids_u64)
    _require(candidate_positions.shape == capture_positions.shape
             and candidate_positions.tobytes() == capture_positions.tobytes(),
             "1274 candidate changed positions from accepted run002")
    _require(candidate_gids.dtype == np.dtype("<u8")
             and candidate_gids.shape == capture_gids.shape
             and candidate_gids.tobytes() == capture_gids.tobytes(),
             "1274 GIDs differ from the source-slot/capture crosswalk")

    patch_source_faces = np.ascontiguousarray(
        index_candidate_faces_source_slots_u32, dtype="<u4")
    patch_capture_faces = np.ascontiguousarray(
        index_candidate_faces_capture_u32, dtype="<u4")
    _require(patch_source_faces.ndim == 2 and patch_source_faces.shape[1] == 3
             and patch_capture_faces.shape == patch_source_faces.shape,
             "1274 index-patch face arrays are malformed")
    _require(int(patch_source_faces.max(initial=0)) < len(source_to_capture),
             "1274 face patch references a source slot outside ff8")
    mapped_patch_faces = source_to_capture[patch_source_faces]
    _require(np.all(mapped_patch_faces >= 0)
             and np.ascontiguousarray(mapped_patch_faces.astype("<u4")).tobytes()
             == patch_capture_faces.tobytes(),
             "1274 source-slot faces do not map exactly through the ff8 capture owner")
    _require(int(patch_capture_faces.max(initial=0)) < len(capture_positions),
             "1274 face patch references a capture vertex outside the accepted row")

    ordered_face_source_slots = np.ascontiguousarray(capture_to_source[capture_faces], dtype="<i8")
    ordered_face_canonical_gids = np.ascontiguousarray(capture_gids[capture_faces], dtype="<u8")
    basis = {
        int(gid): {
            "basis": "accepted_native_capture",
            "source_slot": int(capture_to_source[index]),
            "capture_vertex_index": int(index),
            "position_f32": capture_positions[index].tolist(),
        }
        for index, gid in enumerate(capture_gids)
    }
    _require(len(basis) == len(capture_positions),
             "per-GID actual capture basis is incomplete")
    return {
        "schema": "numi.human.actual-capture-row-owner.v1",
        "position_basis": "accepted_native_capture",
        "actual_capture_positions_f32": capture_positions.copy(),
        "actual_capture_faces_u32": capture_faces.copy(),
        "canonical_gids_u64": capture_gids.copy(),
        "capture_to_source_slot_i64": capture_to_source.copy(),
        "ordered_face_corner_source_slots_i64": ordered_face_source_slots.copy(),
        "ordered_face_corner_canonical_gids_u64": ordered_face_canonical_gids.copy(),
        "full21_weights_f64": capture_weights.copy(),
        "index_patch_faces_capture_u32": patch_capture_faces.copy(),
        "index_patch_faces_source_slots_u32": patch_source_faces.copy(),
        "position_basis_by_gid": basis,
        "report": {
            "source_capture_crosswalk": map_report,
            "actual_capture_vertex_count": int(len(capture_positions)),
            "actual_capture_face_count": int(len(capture_faces)),
            "canonical_gids_replayed_from_006_source_slots": True,
            "prepared_actual_positions_byte_exact": True,
            "prepared_actual_faces_byte_exact": True,
            "prepared_full21_weights_byte_exact_to_006": True,
            "1274_positions_byte_exact_to_accepted_capture": True,
            "1274_faces_mapped_through_source_slots": True,
            "1274_source_face_ancestry_is_separate": True,
            "ordered_face_corner_mapping_verified": True,
            "ordered_face_corner_source_slots_sha256": hashlib.sha256(
                ordered_face_source_slots.tobytes()).hexdigest(),
            "ordered_face_corner_canonical_gids_sha256": hashlib.sha256(
                ordered_face_canonical_gids.tobytes()).hexdigest(),
            "mrvpack_global_vertex_ids_used_as_canonical_gids": False,
        },
    }


def _verify_producer_input_pins(pins, producer_input_recovery_paths=None):
    """Verify producer pins, optionally from explicit SHA-exact recovered blobs.

    The metadata's original lexical paths remain the provenance keys. Recovery is
    allowed only when a caller names that exact pinned path and the recovery file
    hashes to the metadata SHA; it never substitutes current, changed bytes.
    """
    _require(isinstance(pins, dict) and pins,
             "prepared-owner metadata lacks its producer input pins")
    requested = ({} if producer_input_recovery_paths is None
                 else producer_input_recovery_paths)
    _require(isinstance(requested, dict), "producer recovery mapping must be a mapping")
    recoveries = {}
    for original, recovered in requested.items():
        _require(isinstance(original, str) and original,
                 "producer recovery mapping keys must be original pinned paths")
        key = str(Path(original).expanduser().resolve(strict=False))
        _require(key not in recoveries,
                 f"duplicate producer recovery mapping for {original}")
        recoveries[key] = _canonical_file(recovered)

    verified_inputs = {}
    recovery_records = []
    used_recoveries = set()
    for pinned_path, expected in pins.items():
        _require(isinstance(pinned_path, str) and pinned_path,
                 "producer input pin path must be a nonempty string")
        _require(isinstance(expected, str) and len(expected) == 64,
                 f"malformed input digest in prepared-owner metadata: {pinned_path}")
        original = Path(pinned_path).expanduser().resolve(strict=False)
        key = str(original)
        actual_original = _sha256(original) if original.is_file() else None
        if actual_original == expected:
            _require(key not in recoveries,
                     f"unneeded recovery supplied for matching producer input: {pinned_path}")
            verified_inputs[key] = expected
            continue
        recovery_path = recoveries.get(key)
        _require(recovery_path is not None,
                 f"1284 producer input pin changed: {pinned_path}")
        actual_recovery = _sha256(recovery_path)
        _require(actual_recovery == expected,
                 f"recovered producer blob does not match pinned SHA: {pinned_path}")
        verified_inputs[key] = expected
        used_recoveries.add(key)
        recovery_records.append({
            "original_pinned_path": pinned_path,
            "original_path_current_sha256": actual_original,
            "expected_sha256": expected,
            "recovered_blob_path": str(recovery_path),
            "recovered_blob_sha256": actual_recovery,
            "recovery_basis": "exact-content-recovery-from-authenticated-producer-git-blob",
        })
    _require(used_recoveries == set(recoveries),
             "producer recovery mapping contains an unused or unpinned path")
    return verified_inputs, recovery_records


def build_actual_capture_owner_proof(
    *,
    prepared_owner_arrays_path,
    prepared_owner_metadata_path,
    accepted_pack_path,
    accepted_step_receipt_path,
    source_child_path,
    source_map_candidate_path,
    source_map_report_path,
    index_candidate_path,
    index_report_path,
    index_ancestry_path,
    witness_batch_report_path,
    witness_batch_checkpoint_path,
    producer_input_recovery_paths=None,
):
    """Authenticate the retained 1284 carrier against its pinned source/capture chain."""
    from . import passive_attachment_composition as pc
    from . import resting_anatomy_verify as verify

    paths = {
        "prepared_arrays": _canonical_file(prepared_owner_arrays_path),
        "prepared_metadata": _canonical_file(prepared_owner_metadata_path),
        "accepted_pack": _canonical_file(accepted_pack_path),
        "accepted_step_receipt": _canonical_file(accepted_step_receipt_path),
        "source_child": _canonical_file(source_child_path),
        "source_map_candidate": _canonical_file(source_map_candidate_path),
        "source_map_report": _canonical_file(source_map_report_path),
        "index_candidate": _canonical_file(index_candidate_path),
        "index_report": _canonical_file(index_report_path),
        "index_ancestry": _canonical_file(index_ancestry_path),
        "witness_batch_report": _canonical_file(witness_batch_report_path),
        "witness_batch_checkpoint": _canonical_file(witness_batch_checkpoint_path),
    }
    metadata = json.loads(paths["prepared_metadata"].read_text())
    _require(metadata.get("schema") == "numi.human.forearm.prepared-owner-arrays.v1"
             and metadata.get("not_admission") is True,
             "unsupported prepared-owner array metadata")
    with np.load(paths["prepared_arrays"], allow_pickle=False) as archive:
        _require(metadata.get("sha256") == _sha256(paths["prepared_arrays"]),
                 "prepared-owner NPZ does not match its metadata digest")
        arrays = {key: np.asarray(archive[key]).copy() for key in archive.files}
    pins = metadata.get("input_sha256")
    # Each input is hashed once, and any historical producer-source recovery must
    # be explicitly mapped to the original pinned path and match its exact SHA.
    verified_inputs, producer_input_recoveries = _verify_producer_input_pins(
        pins, producer_input_recovery_paths)
    def expect_pin(name):
        path = paths[name]
        actual = verified_inputs.get(str(path))
        _require(actual is not None, f"{name} is not in the exact 1284 input pin map")
        return actual

    _require(expect_pin("accepted_pack") == metadata.get("accepted_run002_pack_sha256"),
             "accepted pack does not match the 1284 metadata")
    _require(expect_pin("accepted_step_receipt") == metadata.get("accepted_run002_step_receipt_sha256"),
             "accepted step receipt does not match the 1284 metadata")
    _require(expect_pin("source_child") == metadata.get("source_child_sha256"),
             "ff8 source child does not match the 1284 metadata")
    _require(expect_pin("source_map_candidate") == metadata.get("source_map_candidate_sha256"),
             "006 candidate does not match the 1284 metadata")

    source_map_report = json.loads(paths["source_map_report"].read_text())
    _require(source_map_report.get("candidate_sha256") == expect_pin("source_map_candidate")
             and Path(source_map_report.get("candidate_path", "")).resolve() == paths["source_map_candidate"],
             "006 report does not authenticate its exact candidate path and bytes")
    _require(source_map_report.get("accepted_capture_step") == 0,
             "006 source identity map is not for accepted step zero")
    index_report = json.loads(paths["index_report"].read_text())
    _require(index_report.get("candidate_sha256") == expect_pin("index_candidate")
             and Path(index_report.get("candidate_path", "")).resolve() == paths["index_candidate"],
             "1274 report does not authenticate its exact candidate path and bytes")
    _require(index_report.get("accepted_run002_pack_sha256") == expect_pin("accepted_pack")
             and index_report.get("accepted_run002_receipt_sha256") == expect_pin("accepted_step_receipt"),
             "1274 candidate is not bound to accepted run002")
    _require(index_report.get("source_slots_or_gids_changed") is False
             and index_report.get("positions_changed") is False
             and index_report.get("geometry_admission") is False
             and index_report.get("native_admission") is False,
             "1274 index candidate does not retain its declared position-only contract")
    ancestry = json.loads(paths["index_ancestry"].read_text())
    _require(ancestry.get("candidate_npz_sha256") == expect_pin("index_candidate")
             and ancestry.get("parent_nhtiss_sha256") == expect_pin("source_child")
             and ancestry.get("accepted_run002_pack_sha256") == expect_pin("accepted_pack")
             and ancestry.get("all_source_positions_and_full21_owners_unchanged") is True,
             "1274 ancestry is not bound to exact candidate, ff8 child, and accepted capture")

    batch_report = json.loads(paths["witness_batch_report"].read_text())
    checkpoint = json.loads(paths["witness_batch_checkpoint"].read_text())
    _require(batch_report.get("prepared_owner_arrays", {}).get("sha256")
             == metadata.get("sha256")
             and checkpoint.get("full21_owner_arrays_sha256") == metadata.get("sha256"),
             "1284 reports do not bind the prepared owner arrays")
    _require(checkpoint.get("accepted_run002_pack_sha256")
             == expect_pin("accepted_pack")
             and checkpoint.get("accepted_run002_step_receipt_sha256")
             == expect_pin("accepted_step_receipt")
             and checkpoint.get("source_child_sha256") == expect_pin("source_child")
             and checkpoint.get("source_map_candidate_sha256")
             == expect_pin("source_map_candidate"),
             "1284 stop checkpoint does not bind the accepted/source chain")
    _require(batch_report.get("not_admission") is True
             and checkpoint.get("geometry_admission") is False
             and checkpoint.get("native_admission") is False,
             "1284 stopped candidate was incorrectly marked admitted")

    source_data = pc._read_nhtiss4(paths["source_child"])
    source_rows = {int(row[6]): row for row in source_data["records"]}
    actual_surfaces, capture_identity = verify.load_capture(
        paths["accepted_pack"], paths["accepted_step_receipt"])
    _require(capture_identity.get("pack_sha256") == expect_pin("accepted_pack")
             and capture_identity.get("capture_receipt_sha256")
             == expect_pin("accepted_step_receipt")
             and capture_identity.get("accepted_step") == 0,
             "existing capture owner returned different accepted pack/receipt identity")

    actual_receipt_candidates = [
        Path(path) for path in verified_inputs
        if Path(path).name == "resting-anatomy-receipt.json"
    ]
    _require(len(actual_receipt_candidates) == 1,
             "prepared input pins must identify one exact resting-anatomy receipt")
    actual_receipt_path = _canonical_file(actual_receipt_candidates[0])
    actual_rest_receipt = json.loads(actual_receipt_path.read_text())
    parent_extension = actual_rest_receipt.get("passive_tissue_parent_bindings")
    _require(isinstance(parent_extension, dict)
             and parent_extension.get("child_payload_sha256") == expect_pin("source_child"),
             "accepted capture parent extension is not bound to the exact source child")
    parent_extension_sha256 = _canonical_json_sha256(parent_extension)
    parent_surfaces = {int(item["stable_id"]): item
                       for item in parent_extension.get("surfaces", [])
                       if isinstance(item, dict) and type(item.get("stable_id")) is int}
    parent_sources = parent_extension.get("source_payloads")
    _require(isinstance(parent_sources, list) and parent_surfaces,
             "accepted capture parent extension lacks its typed full21 source table")
    parent_data_by_source_and_row = {}
    parent_sha_by_source_and_row = {}
    parent_core_bodies_by_source_and_row = {}

    with np.load(paths["source_map_candidate"], allow_pickle=False) as source_map:
        with np.load(paths["index_candidate"], allow_pickle=False) as index_candidate:
            rows = {}
            for stable_id in (132, 134):
                identity = (51005, stable_id)
                _require(identity in actual_surfaces,
                         f"accepted pack lacks typed forearm row {identity}")
                source_row = pc._nhtiss_row_arrays(source_data, source_rows[stable_id])
                prefix = f"row{stable_id}_"
                result = authenticate_actual_capture_row(
                    source_faces_u32=np.asarray(source_row["faces"], dtype="<u4"),
                    source_vertex_count=int(source_row["vertex_count"]),
                    capture_positions_f32=actual_surfaces[identity][0],
                    capture_faces_u32=actual_surfaces[identity][1],
                    source_map_slots_i64=source_map[prefix + "source_slots_i64"],
                    source_map_gids_u64=source_map[prefix + "canonical_material_gids_u64"],
                    source_map_weights_f64=source_map[prefix + "core_weights_float64"],
                    prepared_positions_f32=arrays[prefix + "accepted_positions_f32"],
                    prepared_faces_i64=arrays[prefix + "accepted_faces_i64"],
                    prepared_source_slots_i64=arrays[prefix + "capture_to_source_slot_i64"],
                    prepared_gids_u64=arrays[prefix + "canonical_gids_u64"],
                    prepared_weights_f64=arrays[prefix + "full21_weights_f64"],
                    index_candidate_positions_f32=index_candidate[prefix + "positions_actual_run002_f32"],
                    index_candidate_gids_u64=index_candidate[prefix + "canonical_material_gids_u64"],
                    index_candidate_faces_source_slots_u32=index_candidate[prefix + "faces_source_slots_u32"],
                    index_candidate_faces_capture_u32=index_candidate[prefix + "faces_capture_u32"],
                )
                source_points = np.ascontiguousarray(
                    arrays[prefix + "full21_source_points_f32"], dtype="<f4")
                parent_source_indices = np.asarray(
                    arrays[prefix + "full21_parent_source_indices_u32"], dtype="<u4")
                parent_vertex_locals = np.asarray(
                    arrays[prefix + "full21_parent_vertex_locals_i64"], dtype="<i8")
                capture_slots = np.asarray(result["capture_to_source_slot_i64"], dtype="<i8")
                owner_weights = np.asarray(result["full21_weights_f64"], dtype="<f8")
                _require(source_points.shape == result["actual_capture_positions_f32"].shape
                         and parent_source_indices.shape == capture_slots.shape
                         and parent_vertex_locals.shape == (len(capture_slots), 21),
                         f"row {stable_id} full21 source-owner arrays are malformed")
                source_ids = np.unique(parent_source_indices)
                _require(source_ids.shape == (1,),
                         f"row {stable_id} full21 parent source identity is not unique")
                source_index = int(source_ids[0])
                _require(0 <= source_index < len(parent_sources),
                         f"row {stable_id} full21 parent source index is outside the receipt")
                parent_meta = parent_sources[source_index]
                _require(isinstance(parent_meta, dict)
                         and type(parent_meta.get("path")) is str
                         and type(parent_meta.get("sha256")) is str,
                         f"row {stable_id} full21 parent payload metadata is malformed")
                parent_path = _canonical_file(parent_meta["path"])
                _require(verified_inputs.get(str(parent_path)) == parent_meta["sha256"],
                         f"row {stable_id} active full21 parent payload is not byte-pinned")
                parent_sha = _sha256(parent_path)
                _require(parent_sha == parent_meta["sha256"],
                         f"row {stable_id} active full21 parent payload SHA changed")
                cache_key = (source_index, stable_id)
                if cache_key not in parent_data_by_source_and_row:
                    parent_file = pc._read_nhtiss4(parent_path)
                    parent_record = next(
                        item for item in parent_file["records"]
                        if int(item[6]) == stable_id
                    )
                    parent_data_by_source_and_row[cache_key] = (
                        parent_path, parent_record,
                        pc._nhtiss_row_arrays(parent_file, parent_record)
                    )
                    parent_sha_by_source_and_row[cache_key] = parent_sha
                    parent_core_bodies_by_source_and_row[cache_key] = [
                        int(binding["core"])
                        for binding in parent_data_by_source_and_row[cache_key][2]["bindings"]
                    ]
                cached_path, parent_record, parent_data = parent_data_by_source_and_row[cache_key]
                _require(cached_path == parent_path
                         and parent_sha_by_source_and_row[cache_key] == parent_sha,
                         f"row {stable_id} full21 parent cache identity changed")
                parent_surface = parent_surfaces.get(stable_id)
                _require(isinstance(parent_surface, dict)
                         and isinstance(parent_surface.get("vertices"), list)
                         and len(parent_surface["vertices"]) == int(source_row["vertex_count"]),
                         f"row {stable_id} actual parent extension lacks source-slot terms")
                core_body_indices = parent_core_bodies_by_source_and_row[cache_key]
                _require(len(core_body_indices) == 21 and len(set(core_body_indices)) == 21,
                         f"row {stable_id} active parent carrier does not have 21 ordered bodies")
                parent_positions = np.asarray(parent_data["positions"], dtype="<f4")
                parent_binding_indices = np.asarray(parent_data["binding_indices"])
                parent_binding_weights = np.asarray(parent_data["weights"])
                _require(parent_binding_indices.shape == (len(parent_positions), 4)
                         and parent_binding_weights.shape == (len(parent_positions), 4),
                         f"row {stable_id} active parent carrier route arrays do not cover its vertices")
                parent_first_vertex = int(parent_record[2])
                for capture_index, source_slot_value in enumerate(capture_slots):
                    source_slot = int(source_slot_value)
                    _require(0 <= source_slot < len(parent_surface["vertices"]),
                             f"row {stable_id} capture vertex has no original source slot")
                    terms = parent_surface["vertices"][source_slot]
                    _require(isinstance(terms, list) and len(terms) == 21,
                             f"row {stable_id} source slot {source_slot} lacks 21 parent terms")
                    local_ids = np.asarray(
                        [int(term[2]) - parent_first_vertex for term in terms], dtype="<i8")
                    _require(np.array_equal(local_ids, parent_vertex_locals[capture_index]),
                             f"row {stable_id} source slot {source_slot} parent locals differ")
                    _require(all(int(term[0]) == source_index and int(term[1]) == stable_id
                                 for term in terms),
                             f"row {stable_id} source slot {source_slot} parent terms change typed owner")
                    term_weights = np.asarray([float(term[3]) for term in terms], dtype="<f8")
                    _require(term_weights.tobytes() == owner_weights[capture_index].tobytes(),
                             f"row {stable_id} source slot {source_slot} parent weights differ")
                    _require(np.all(local_ids >= 0)
                             and np.all(local_ids < len(parent_positions)),
                             f"row {stable_id} source slot {source_slot} parent vertex is out of range")
                    lane_points = parent_positions[local_ids, :3]
                    _validate_full21_parent_lane_routes(
                        parent_binding_indices, parent_binding_weights, local_ids,
                        core_body_indices, stable_id=stable_id, source_slot=source_slot)
                    _require(np.array_equal(lane_points, np.repeat(lane_points[:1], 21, axis=0))
                             and lane_points[0].tobytes() == source_points[capture_index].tobytes(),
                             f"row {stable_id} source slot {source_slot} source point differs from pinned parent")
                patch_faces = np.ascontiguousarray(
                    result["index_patch_faces_capture_u32"], dtype="<u4")
                patch_source_faces = np.ascontiguousarray(
                    result["index_patch_faces_source_slots_u32"], dtype="<u4")
                patch_gids = np.ascontiguousarray(
                    result["canonical_gids_u64"][patch_faces], dtype="<u8")
                patch_slots = np.ascontiguousarray(patch_source_faces, dtype="<i8")
                row_proof = {
                    "schema": "numi.human.full21-actual-capture-row-owner.v1",
                    "typed_identity": list(identity),
                    "accepted_step": int(capture_identity["accepted_step"]),
                    "accepted_pack_sha256": expect_pin("accepted_pack"),
                    "capture_receipt_sha256": expect_pin("accepted_step_receipt"),
                    "source_payload_sha256": expect_pin("source_child"),
                    "row_parent_payload_sha256": parent_sha,
                    "parent_extension_sha256": parent_extension_sha256,
                    "source_map_candidate_sha256": expect_pin("source_map_candidate"),
                    "prepared_owner_arrays_sha256": metadata["sha256"],
                    "index_candidate_sha256": expect_pin("index_candidate"),
                    "ordered_core_body_indices": core_body_indices,
                    "world_frame": "accepted_body_poses_from_capture_receipt",
                    "source_points_sha256": _full21_rows_sha256(
                        result["canonical_gids_u64"], source_points),
                    "weights_sha256": _full21_rows_sha256(
                        result["canonical_gids_u64"], owner_weights),
                    "displayed_world_points_sha256": _full21_rows_sha256(
                        result["canonical_gids_u64"], result["actual_capture_positions_f32"]),
                    "capture_source_slots_sha256": _full21_rows_sha256(
                        result["canonical_gids_u64"], capture_slots),
                    "index_patch_faces_sha256": hashlib.sha256(patch_faces.tobytes()).hexdigest(),
                    "index_patch_face_corner_source_slots_sha256": hashlib.sha256(
                        patch_slots.tobytes()).hexdigest(),
                    "index_patch_face_corner_canonical_gids_sha256": hashlib.sha256(
                        patch_gids.tobytes()).hexdigest(),
                    "displayed_world_owner_relation": "accepted_native_capture_byte_exact",
                    "source_face_ancestry_status": "separately_reported_not_proven_by_capture_owner",
                }
                rows[str(stable_id)] = {
                    "typed_identity": list(identity),
                    "canonical_gids_u64": result["canonical_gids_u64"],
                    "capture_to_source_slot_i64": result["capture_to_source_slot_i64"],
                    "ordered_face_corner_source_slots_i64": result["ordered_face_corner_source_slots_i64"],
                    "ordered_face_corner_canonical_gids_u64": result["ordered_face_corner_canonical_gids_u64"],
                    "actual_capture_positions_f32": result["actual_capture_positions_f32"],
                    "actual_capture_faces_u32": result["actual_capture_faces_u32"],
                    "full21_weights_f64": result["full21_weights_f64"],
                    "full21_source_points_f32": source_points,
                    "ordered_core_body_indices": core_body_indices,
                    "index_patch_faces_capture_u32": result["index_patch_faces_capture_u32"],
                    "index_patch_faces_source_slots_u32": result["index_patch_faces_source_slots_u32"],
                    "full21_actual_capture_owner_proof": row_proof,
                    "position_basis_by_gid": result["position_basis_by_gid"],
                    "report": result["report"],
                }

    cross_row_shared_points = authenticate_shared_actual_capture_gid_points(rows)
    ancestry_rows = ancestry.get("rows", {})
    face_lineage = {}
    for stable_id in (132, 134):
        components = ancestry_rows.get(str(stable_id), [])
        errors = [record for record in components
                  if isinstance(record, dict) and record.get("error")]
        face_lineage[str(stable_id)] = {
            "ancestry_component_count": len(components),
            "ancestry_error_component_count": len(errors),
            "complete_source_face_ancestry": False,
            "source_face_coverage_proven": False,
            "source_face_coverage_status": (
                "ANCESTRY_COMPONENTS_HAVE_ERRORS" if errors
                else "ANCESTRY_COVERAGE_NOT_PROVEN"),
            "candidate_face_array_sha256": _sha256(paths["index_candidate"]),
            "ancestry_file_sha256": verified_inputs.get(str(paths["index_ancestry"]))
                or _sha256(paths["index_ancestry"]),
            "separate_from_actual_position_proof": True,
            "error_examples": [
                {"component": item.get("component"), "error": item.get("error")}
                for item in errors[:4]
            ],
        }
    return {
        "schema": "numi.human.actual-capture-and-forearm-owner-chain.v1",
        "accepted_step": 0,
        "accepted_pack_sha256": expect_pin("accepted_pack"),
        "accepted_step_receipt_sha256": expect_pin("accepted_step_receipt"),
        "source_child_sha256": expect_pin("source_child"),
        "source_map_candidate_sha256": expect_pin("source_map_candidate"),
        "index_candidate_sha256": expect_pin("index_candidate"),
        "prepared_owner_arrays_sha256": metadata["sha256"],
        "prepared_owner_metadata_sha256": _sha256(paths["prepared_metadata"]),
        "index_ancestry_sha256": _sha256(paths["index_ancestry"]),
        "witness_batch_report_sha256": _sha256(paths["witness_batch_report"]),
        "witness_batch_checkpoint_sha256": _sha256(paths["witness_batch_checkpoint"]),
        "producer_input_pin_count_verified": len(verified_inputs),
        "producer_input_recoveries": producer_input_recoveries,
        "rows": rows,
        "shared_actual_capture_gid_points": cross_row_shared_points,
        "face_topology_provenance": face_lineage,
        "not_admission": True,
        "evidence_boundary": (
            "unchanged positions are byte-exact accepted native capture points; canonical "
            "GIDs are replayed through ff8 ordered-face source slots and pinned 006/1274 "
            "records; 1274 source-face ancestry is reported separately and incomplete "
            "ancestry does not become an anatomy or native admission"
        ),
    }



def build_mixed_full21_candidate_owner_record(
    *,
    actual_capture_row,
    candidate_positions_f32,
    candidate_faces_u32,
    candidate_gids_u64,
    candidate_source_slots_i64,
    candidate_weights_f64,
    candidate_source_points_f32,
    predicted_gids_u64,
    predicted_source_points_f32,
    predicted_lane_world_points_f32,
    candidate_parent_payload_sha256,
    candidate_parent_extension_sha256,
    candidate_child_payload_sha256,
    candidate_archive_sha256,
    source_face_lineage,
    source_vertex_lineage,
):
    """Bind candidate vertices to captured or independently recomputed full21 bases.

    The accepted capture proof is reused only for GIDs whose displayed positions
    remain byte-identical. Every changed GID must have a complete predicted lane
    record whose ordered Float32 blend exactly reproduces its candidate position.
    This is a proof-construction helper; it does not perform native capture.
    """
    from . import resting_anatomy_verify as verify

    _require(isinstance(actual_capture_row, dict),
             "mixed candidate needs one file-backed actual-capture row")
    actual_proof = actual_capture_row.get("full21_actual_capture_owner_proof")
    _require(isinstance(actual_proof, dict)
             and actual_proof.get("displayed_world_owner_relation")
                 == "accepted_native_capture_byte_exact",
             "mixed candidate lacks the exact accepted capture owner proof")
    positions = np.ascontiguousarray(candidate_positions_f32, dtype="<f4")
    faces = np.ascontiguousarray(candidate_faces_u32, dtype="<u4")
    gids = np.asarray(candidate_gids_u64)
    slots = np.asarray(candidate_source_slots_i64)
    weights = np.asarray(candidate_weights_f64)
    source_points = np.ascontiguousarray(candidate_source_points_f32, dtype="<f4")
    changed = np.asarray(predicted_gids_u64)
    changed_source = np.ascontiguousarray(predicted_source_points_f32, dtype="<f4")
    lanes = np.ascontiguousarray(predicted_lane_world_points_f32, dtype="<f4")
    _require(positions.ndim == 2 and positions.shape[1] == 3
             and np.isfinite(positions).all(),
             "candidate positions must be finite Float32 XYZ")
    _require(faces.ndim == 2 and faces.shape[1] == 3,
             "candidate faces must be ordered local triangles")
    _require(gids.dtype == np.dtype("<u8") and gids.shape == (len(positions),)
             and len(np.unique(gids)) == len(gids),
             "candidate canonical GIDs must uniquely cover displayed positions")
    _require(slots.dtype == np.dtype("<i8") and slots.shape == gids.shape
             and np.all(slots >= 0),
             "candidate source slots must retain the complete original slot map")
    _require(weights.dtype == np.dtype("<f8") and weights.shape == (len(gids), 21)
             and np.isfinite(weights).all() and np.all(weights > 0)
             and np.max(np.abs(weights.sum(axis=1)-1.0)) <= 1e-12,
             "candidate must retain all 21 positive normalized source weights")
    _require(source_points.shape == positions.shape and np.isfinite(source_points).all(),
             "candidate full21 source points must cover every GID")
    _require(changed.dtype.kind in "iu" and changed.ndim == 1
             and len(np.unique(changed)) == len(changed),
             "predicted GIDs must be a unique integer subset")
    changed = np.ascontiguousarray(changed.astype("<u8", copy=False))
    _require(changed_source.shape == (len(changed), 3)
             and lanes.shape == (len(changed), 21, 3)
             and np.isfinite(changed_source).all() and np.isfinite(lanes).all(),
             "each changed GID needs a source point and all 21 emitted lane points")
    gid_index = {int(g): i for i, g in enumerate(gids)}
    _require(set(map(int, changed)) <= set(gid_index),
             "predicted GID is absent from candidate")
    actual_gids = np.asarray(actual_capture_row["canonical_gids_u64"])
    actual_positions = np.ascontiguousarray(
        actual_capture_row["actual_capture_positions_f32"], dtype="<f4")
    actual_slots = np.asarray(actual_capture_row["capture_to_source_slot_i64"])
    actual_weights = np.asarray(actual_capture_row["full21_weights_f64"])
    actual_source_points = np.ascontiguousarray(
        actual_capture_row["full21_source_points_f32"], dtype="<f4")
    _require(actual_gids.dtype == np.dtype("<u8")
             and actual_positions.shape == positions.shape
             and actual_gids.shape == gids.shape
             and actual_slots.dtype == np.dtype("<i8")
             and actual_slots.shape == slots.shape
             and actual_weights.dtype == np.dtype("<f8")
             and actual_weights.shape == weights.shape
             and actual_source_points.shape == positions.shape,
             "actual capture row arrays are incomplete or mis-shaped")
    _require(actual_gids.tobytes() == gids.tobytes()
             and actual_slots.tobytes() == slots.tobytes(),
             "candidate changed canonical GID or source-slot membership/order")
    actual_index_faces = np.ascontiguousarray(
        actual_capture_row["index_patch_faces_capture_u32"], dtype="<u4")
    _require(faces.tobytes() == actual_index_faces.tobytes(),
             "candidate faces differ from the authenticated 1274 index candidate")
    _require(weights.tobytes() == actual_weights.tobytes(),
             "candidate changed a full21 owner coefficient or lane order")
    changed_set = set(map(int, changed))
    basis = {}
    lane_by_gid = {}
    predicted_source_by_gid = {}
    actual_position_by_gid = {}
    actual_source_by_gid = {}
    weight_by_gid = {}
    for i, raw_gid in enumerate(gids):
        gid = int(raw_gid)
        actual_position_by_gid[gid] = actual_positions[i].copy()
        actual_source_by_gid[gid] = actual_source_points[i].copy()
        weight_by_gid[gid] = actual_weights[i].copy()
        if gid not in changed_set:
            _require(positions[i].tobytes() == actual_positions[i].tobytes(),
                     f"unchanged GID {gid} differs from accepted native capture")
            _require(source_points[i].tobytes() == actual_source_points[i].tobytes(),
                     f"unchanged GID {gid} changed its captured parent source point")
            basis[str(gid)] = {
                "basis": "accepted_native_capture",
                "position_f32": positions[i].tolist(),
            }
    for j, raw_gid in enumerate(changed):
        gid = int(raw_gid)
        i = gid_index[gid]
        _require(source_points[i].tobytes() == changed_source[j].tobytes(),
                 f"predicted GID {gid} candidate source point differs")
        lane_weights = weights[i].astype("<f4")
        expected = np.zeros(3, dtype="<f4")
        for lane in range(21):
            contribution = np.asarray(lanes[j, lane] * lane_weights[lane], dtype="<f4")
            expected = np.asarray(expected + contribution, dtype="<f4")
        _require(expected.tobytes() == positions[i].tobytes(),
                 f"predicted GID {gid} does not match its exact ordered full21 Float32 replay")
        lane_by_gid[gid] = lanes[j].copy()
        predicted_source_by_gid[gid] = changed_source[j].copy()
        basis[str(gid)] = {
            "basis": "predicted_cpu_full21_float32_blend_at_accepted_poses",
            "position_f32": positions[i].tolist(),
            "source_point_f32": changed_source[j].tolist(),
            "lane_world_points_sha256": _full21_rows_sha256(
                np.asarray([gid], dtype="<u8"), lanes[j:j+1]),
            "candidate_parent_payload_sha256": candidate_parent_payload_sha256,
            "candidate_parent_extension_sha256": candidate_parent_extension_sha256,
            "candidate_child_payload_sha256": candidate_child_payload_sha256,
        }
    _require(set(basis) == set(map(str, map(int, gids))),
             "mixed per-GID basis does not exactly cover candidate membership")
    _require(all(isinstance(value, str) and len(value) == 64
                 and all(ch in "0123456789abcdef" for ch in value)
                 for value in (candidate_parent_payload_sha256,
                               candidate_parent_extension_sha256,
                               candidate_child_payload_sha256,
                               candidate_archive_sha256)),
             "candidate source/emission archive SHA pin is malformed")
    face_rows = tuple(source_face_lineage)
    _require(len(face_rows) == len(faces) and all(isinstance(x, dict) for x in face_rows),
             "candidate face ancestry must classify every displayed face")
    _require(isinstance(source_vertex_lineage, dict)
             and set(map(int, source_vertex_lineage)) == set(map(int, gids)),
             "candidate vertex ancestry must cover exactly all canonical GIDs")
    vertex_rows = [[int(gid), source_vertex_lineage[gid]
                    if gid in source_vertex_lineage else source_vertex_lineage[str(gid)]]
                   for gid in sorted(map(int, gids))]
    owner = actual_proof
    proof = {
        "schema": "numi.human.full21-mixed-candidate-row-owner.v1",
        "typed_identity": list(actual_capture_row["typed_identity"]),
        "accepted_step": int(owner["accepted_step"]),
        "accepted_pack_sha256": owner["accepted_pack_sha256"],
        "capture_receipt_sha256": owner["capture_receipt_sha256"],
        "source_payload_sha256": owner["source_payload_sha256"],
        "accepted_row_parent_payload_sha256": owner["row_parent_payload_sha256"],
        "accepted_parent_extension_sha256": owner["parent_extension_sha256"],
        "candidate_parent_payload_sha256": candidate_parent_payload_sha256,
        "candidate_parent_extension_sha256": candidate_parent_extension_sha256,
        "candidate_child_payload_sha256": candidate_child_payload_sha256,
        "candidate_archive_sha256": candidate_archive_sha256,
        "source_map_candidate_sha256": owner["source_map_candidate_sha256"],
        "prepared_owner_arrays_sha256": owner["prepared_owner_arrays_sha256"],
        "index_candidate_sha256": owner["index_candidate_sha256"],
        "ordered_core_body_indices": list(owner["ordered_core_body_indices"]),
        "world_frame": owner["world_frame"],
        "displayed_world_owner_relation": "per_GID_accepted_capture_or_predicted_full21",
        "candidate_positions_sha256": _full21_rows_sha256(gids, positions),
        "candidate_faces_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
        "candidate_gids_sha256": hashlib.sha256(gids.tobytes()).hexdigest(),
        "candidate_source_slots_sha256": _full21_rows_sha256(gids, slots),
        "candidate_source_points_sha256": _full21_rows_sha256(gids, source_points),
        "candidate_weights_sha256": _full21_rows_sha256(gids, weights),
        "source_face_lineage_sha256": _canonical_json_sha256(list(face_rows)),
        "source_vertex_lineage_sha256": _canonical_json_sha256(vertex_rows),
        "actual_capture_owner_proof": owner,
        "position_basis_by_gid": basis,
        "predicted_gids_u64": changed,
        "predicted_source_points_f32": changed_source.copy(),
        "predicted_lane_world_points_f32": lanes.copy(),
        "actual_capture_positions_f32": actual_positions.copy(),
        "actual_capture_source_points_f32": actual_source_points.copy(),
        "actual_capture_weights_f64": actual_weights.copy(),
        "actual_capture_source_slots_i64": actual_slots.copy(),
        "actual_capture_faces_u32": np.ascontiguousarray(
            actual_capture_row["actual_capture_faces_u32"], dtype="<u4"),
        "candidate_arrays": {
            "positions_f32": positions.copy(), "faces_u32": faces.copy(),
            "gids_u64": gids.copy(), "source_slots_i64": slots.copy(),
            "weights_f64": weights.copy(), "source_points_f32": source_points.copy(),
        },
    }
    return proof

def actual_capture_owner_json_summary(proof):
    """Return the complete compact, JSON-safe audit record for a builder result."""
    top_level = (
        "schema", "accepted_step", "accepted_pack_sha256",
        "accepted_step_receipt_sha256", "source_child_sha256",
        "source_map_candidate_sha256", "index_candidate_sha256",
        "prepared_owner_arrays_sha256", "prepared_owner_metadata_sha256",
        "index_ancestry_sha256", "witness_batch_report_sha256",
        "witness_batch_checkpoint_sha256", "producer_input_pin_count_verified",
        "producer_input_recoveries", "shared_actual_capture_gid_points",
        "face_topology_provenance", "not_admission", "evidence_boundary",
    )
    _require(isinstance(proof, dict) and all(key in proof for key in top_level),
             "actual capture proof is missing required summary fields")
    raw_rows = proof.get("rows")
    _require(isinstance(raw_rows, dict) and raw_rows,
             "actual capture proof has no authenticated rows")
    row_fields = (
        "typed_identity", "ordered_core_body_indices",
        "full21_actual_capture_owner_proof", "report",
    )
    rows = {}
    parent_extensions = {}
    for stable_id, row in raw_rows.items():
        _require(isinstance(row, dict) and all(key in row for key in row_fields),
                 f"actual capture row {stable_id} lacks summary provenance")
        owner = row["full21_actual_capture_owner_proof"]
        _require(isinstance(owner, dict) and owner.get("displayed_world_owner_relation")
                 == "accepted_native_capture_byte_exact",
                 f"actual capture row {stable_id} has an unsupported owner basis")
        parent_extensions[stable_id] = owner["parent_extension_sha256"]
        row_summary = {key: row[key] for key in row_fields}
        report = row["report"]
        row_summary["ordered_face_corner_source_slots_sha256"] = report[
            "ordered_face_corner_source_slots_sha256"]
        row_summary["ordered_face_corner_canonical_gids_sha256"] = report[
            "ordered_face_corner_canonical_gids_sha256"]
        rows[stable_id] = row_summary
    summary = {key: proof[key] for key in top_level}
    summary["rows"] = rows
    summary["parent_extension_sha256_by_row"] = parent_extensions
    try:
        json.dumps(summary, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("actual capture proof summary is not strict JSON") from exc
    return summary
