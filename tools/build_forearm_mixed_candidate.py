#!/usr/bin/env python3
"""Build an authenticated mixed actual-capture/predicted-full21 forearm candidate.

This tool only emits a reviewable offline candidate and checks three retained
exact witnesses. It never changes native inputs or admits anatomy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[1]
D = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
D1284 = D / "forearm-native007-actual972-full21-witness-batch-1284"
D1291 = D / "forearm-native007-actual972-capture-owner-proof-1291"
D1274 = D / "forearm-native007-actual972-witness-star-retriangulation-1274"
D006 = D / "forearm-native007-current972-joint-source-containing-006"
ACCEPTED = D / "pons323-348-current972-existing-owner-rebind-001/native-run-002"
OUT = D / "forearm-native007-actual972-mixed-owner-candidate-1297"
MAX_SECONDS = 900


def require(condition, message):
    if not condition:
        raise ValueError("mixed forearm candidate: " + message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def progress(stage, start, **extra):
    atomic_json(OUT / "progress.json", {
        "schema": "numi.human.forearm-mixed-owner-candidate-progress.v1",
        "stage": stage, "elapsed_seconds": time.monotonic() - start,
        **extra,
    })


def alarm_handler(signum, frame):
    raise TimeoutError(f"bounded mixed-owner candidate exceeded {MAX_SECONDS}s")


def read_json(path):
    with Path(path).open("r") as stream:
        return json.load(stream)


def cyclic_key(triangle):
    values = tuple(int(x) for x in triangle)
    require(len(values) == 3 and len(set(values)) == 3, "face has repeated corners")
    rotations = [values, values[1:] + values[:1], values[2:] + values[:2]]
    return min(rotations)


def direct_orientation(candidate_triangle, source_triangle):
    cand = tuple(int(x) for x in candidate_triangle)
    src = tuple(int(x) for x in source_triangle)
    for direction, ordered in (("same", src), ("reversed", (src[0], src[2], src[1]))):
        for shift in range(3):
            rot = ordered[shift:] + ordered[:shift]
            if cand == rot:
                return direction, [src.index(value) for value in cand]
    return None, None


def build_face_lineage(
    *, stable_id, source_path, source_sha, original_source_faces,
    candidate_face_source_slots, candidate_face_gids, generated_records,
    gids_by_source_slot,
):
    """Prove every output face direct or from explicit 1274 source-face support."""
    source_faces = np.asarray(original_source_faces, dtype="<u4")
    out_slots = np.asarray(candidate_face_source_slots, dtype="<u4")
    out_gids = np.asarray(candidate_face_gids, dtype="<u8")
    require(out_slots.shape == out_gids.shape and out_slots.ndim == 2 and out_slots.shape[1] == 3,
            f"row {stable_id} source-slot/GID faces do not align")
    source_face_map = {}
    for ordinal, face in enumerate(source_faces):
        triangle = tuple(int(x) for x in face)
        key = tuple(sorted(triangle))
        source_face_map.setdefault(key, []).append((ordinal, triangle))

    generated_map = {}
    support_hashes = {}
    for record in generated_records:
        require(record.get("typed_identity") == [51005, stable_id]
                and record.get("parent_nhtiss_sha256") == source_sha
                and record.get("inferred_local_retriangulation") is True,
                f"row {stable_id} generated-face record has wrong source owner")
        supports = record.get("support_source_face_ordinals")
        corners = record.get("new_face_corner_gids")
        require(isinstance(supports, list) and supports
                and all(type(x) is int and 0 <= x < len(source_faces) for x in supports),
                f"row {stable_id} generated face has invalid source support")
        expected_support_sha = hashlib.sha256(
            np.asarray(sorted(set(supports)), dtype="<u8").tobytes()).hexdigest()
        require(record.get("support_face_ordinal_sha256") == expected_support_sha,
                f"row {stable_id} generated support-face digest mismatch")
        support_gids = set()
        for face_id in supports:
            for slot in source_faces[face_id]:
                gid = gids_by_source_slot.get(int(slot))
                require(gid is not None, f"row {stable_id} support source slot has no canonical GID")
                support_gids.add(gid)
        require(isinstance(corners, list) and corners,
                f"row {stable_id} generated component lacks new face corner records")
        for corner_triplet in corners:
            tri = tuple(int(x) for x in corner_triplet)
            require(len(tri) == 3 and len(set(tri)) == 3
                    and set(tri) <= support_gids,
                    f"row {stable_id} generated face corners exceed authenticated source support")
            key = tuple(sorted(tri))
            generated_map.setdefault(key, []).append({
                "component": record["component"],
                "corner_gids": tri,
                "support_source_face_ordinals": sorted(set(supports)),
                "support_face_ordinal_sha256": expected_support_sha,
                "record_boundary_gids": record.get("boundary_gids", []),
            })
        support_hashes[str(record["component"])] = expected_support_sha

    lineage = []
    direct_count = inferred_count = duplicate_support_count = 0
    generated_direct_overlap_count = 0
    missing = []
    for face_ordinal, (slots, gids) in enumerate(zip(out_slots, out_gids, strict=True)):
        require(all(gids_by_source_slot.get(int(slot)) == int(gid)
                    for slot, gid in zip(slots, gids, strict=True)),
                f"row {stable_id} output face GIDs do not match authenticated source-slot crosswalk")
        slot_key = tuple(sorted(int(x) for x in slots))
        candidates = source_face_map.get(slot_key, [])
        if len(candidates) == 1:
            source_face, source_tri = candidates[0]
            direction, source_corner_order = direct_orientation(slots, source_tri)
            require(direction is not None, f"row {stable_id} direct source face orientation unresolved")
            bary = []
            for source_corner in source_corner_order:
                bary.append([["1", "1"] if i == source_corner else ["0", "1"]
                             for i in range(3)])
            lineage.append({
                "source_member": f"{source_path}#stable_id={stable_id}",
                "source_member_sha256": source_sha,
                "source_face": int(source_face),
                "source_row_face": int(source_face),
                "source_face_support_provenance": (
                    "exact original ff8 NHTISS source-face ordinal through authenticated "
                    "ordered source-slot crosswalk"),
                "source_face_orientation": direction,
                "corner_barycentric_exact": bary,
            })
            direct_count += 1
            if generated_map.get(tuple(sorted(int(x) for x in gids))):
                generated_direct_overlap_count += 1
            continue
        if len(candidates) > 1:
            # Keep all exact duplicate source ordinals as support. Do not select one.
            ordinals = sorted(int(item[0]) for item in candidates)
            lineage.append({
                "source_member": f"{source_path}#stable_id={stable_id}",
                "source_member_sha256": source_sha,
                "source_face_support": ordinals,
                "source_face_support_provenance": (
                    "exact duplicate original source faces with identical ordered corner support"),
                "candidate_face_ordinal": face_ordinal,
                "candidate_corner_gids": [int(x) for x in gids],
            })
            duplicate_support_count += 1
            continue
        generated = generated_map.get(tuple(sorted(int(x) for x in gids)), [])
        dedup = {}
        for row in generated:
            key = (row["component"], tuple(row["support_source_face_ordinals"]),
                   tuple(row["corner_gids"]))
            dedup[key] = row
        generated = list(dedup.values())
        if len(generated) != 1:
            missing.append({
                "candidate_face_ordinal": face_ordinal,
                "source_slots": [int(x) for x in slots],
                "canonical_gids": [int(x) for x in gids],
                "direct_source_match_count": len(candidates),
                "generated_support_match_count": len(generated),
            })
            continue
        proof = generated[0]
        direction, _corner_order = direct_orientation(gids, proof["corner_gids"])
        require(direction is not None,
                f"row {stable_id} generated face orientation is not an exact permutation")
        lineage.append({
            "source_member": f"{source_path}#stable_id={stable_id}",
            "source_member_sha256": source_sha,
            "source_face_support": proof["support_source_face_ordinals"],
            "source_face_support_provenance": (
                "1274 inferred local retriangulation; exact candidate face tuple and "
                "all canonical corner GIDs lie in the pinned source-face support"),
            "source_retriangulation_component": proof["component"],
            "source_support_face_ordinal_sha256": proof["support_face_ordinal_sha256"],
            "candidate_face_ordinal": face_ordinal,
            "candidate_corner_gids": [int(x) for x in gids],
            "output_orientation_relative_to_1274_face": direction,
            "source_support_boundary_gids": proof["record_boundary_gids"],
        })
        inferred_count += 1
    require(not missing,
            f"row {stable_id} has {len(missing)} candidate faces without unique direct/inferred ancestry")
    require(len(lineage) == len(out_slots),
            f"row {stable_id} face ancestry does not cover every output face")
    return lineage, {
        "candidate_face_count": int(len(out_slots)),
        "direct_exact_source_face_count": direct_count,
        "duplicate_source_face_support_count": duplicate_support_count,
        "inferred_retriangulated_face_count": inferred_count,
        "generated_records_overlapping_direct_faces": generated_direct_overlap_count,
        "uncovered_or_ambiguous_face_count": len(missing),
        "complete_candidate_face_coverage": True,
        "source_face_support_component_sha256": support_hashes,
        "not_admission": True,
    }


def exact_fraction_rows(points):
    return [[f"{x.numerator}/{x.denominator}" for x in point] for point in points]


def main():
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite retained candidate attempt: {OUT}")
    free_bytes = os.statvfs(D).f_bavail * os.statvfs(D).f_frsize
    require(free_bytes >= 10 * (1 << 30),
            f"CPU candidate preflight disk floor unmet: {free_bytes} bytes free")
    OUT.mkdir(parents=True)
    start = time.monotonic()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(MAX_SECONDS)
    atomic_json(OUT / "launch.json", {
        "schema": "numi.human.forearm-mixed-owner-candidate-launch.v1",
        "output": str(OUT), "max_seconds": MAX_SECONDS,
        "inputs": {
            "actual_capture_owner_proof": str(D1291 / "report.json"),
            "full21_witness_batch": str(D1284 / "report.json"),
            "index_candidate": str(D1274 / "candidate.npz"),
            "index_ancestry": str(D1274 / "candidate-face-ancestry.json"),
            "accepted_step_receipt": str(ACCEPTED / "accepted-geometry/step-0.receipt.json"),
        },
        "not_native_capture": True, "not_anatomy_admission": True,
    })
    try:
        progress("pin_and_schema_preflight", start)
        sys.path.insert(0, str(REPO / "src"))
        from numilab_human import passive_attachment_composition as pc
        from numilab_human.forearm_actual_capture_owner import (
            _canonical_json_sha256,
            build_mixed_full21_candidate_owner_record,
        )
        from numilab_human.forearm_parent_carrier_bridge import (
            build_authenticated_full21_lane_world_subset,
            validate_source_gid_slot_lineage,
        )
        from numilab_human import resting_anatomy_verify as verify

        actual_report_path = D1291 / "report.json"
        actual_checkpoint_path = D1291 / "builder-proof-checkpoint.json"
        producer_report = read_json(actual_report_path)
        checkpoint = read_json(actual_checkpoint_path)
        batch_report_path = D1284 / "report.json"
        batch_report = read_json(batch_report_path)
        batch_checkpoint_path = D1284 / "stop-checkpoint.json"
        require(sha(batch_report_path) == producer_report["witness_batch_report_sha256"],
                "1291 proof does not pin the exact 1284 witness-batch report")
        require(sha(batch_checkpoint_path) == producer_report["witness_batch_checkpoint_sha256"],
                "1291 proof does not pin the exact 1284 stop checkpoint")
        metadata_path = D1284 / "prepared-owner-arrays.json"
        metadata = read_json(metadata_path)
        index_report_path = D1274 / "report.json"
        index_report = read_json(index_report_path)
        ancestry_path = D1274 / "candidate-face-ancestry.json"
        ancestry = read_json(ancestry_path)
        index_npz_path = D1274 / "candidate.npz"
        actual_pack = ACCEPTED / "accepted-geometry/step-0.mrvpack"
        actual_receipt = ACCEPTED / "accepted-geometry/step-0.receipt.json"

        require(producer_report["accepted_pack_sha256"] == batch_report["accepted_run002_pack_sha256"]
                == index_report["accepted_run002_pack_sha256"],
                "producer reports do not identify one accepted run002 pack")
        require(producer_report["accepted_step_receipt_sha256"]
                == batch_report["accepted_run002_step_receipt_sha256"]
                == index_report["accepted_run002_receipt_sha256"],
                "producer reports do not identify one accepted step receipt")
        require(producer_report["accepted_step"] == 0,
                "accepted capture owner proof is not step zero")
        require(sha(actual_checkpoint_path) == producer_report["builder_proof_checkpoint_sha256"],
                "1291 proof checkpoint changed after producer report")
        require(sha(actual_receipt) == producer_report["accepted_step_receipt_sha256"],
                "accepted native step receipt changed")
        require(sha(actual_pack) == producer_report["accepted_pack_sha256"],
                "accepted native pack changed")
        require(sha(ancestry_path) == producer_report["index_ancestry_sha256"]
                and sha(index_npz_path) == producer_report["index_candidate_sha256"],
                "1274 candidate/ancestry changed after 1291 file-backed authentication")
        require(sha(D1284 / "prepared-owner-arrays.npz")
                == producer_report["prepared_owner_arrays_sha256"],
                "prepared 1284 owner arrays changed")
        require(sha(metadata_path) == producer_report["prepared_owner_metadata_sha256"],
                "1284 metadata file hash disagrees with actual capture proof")
        require(metadata["sha256"] == producer_report["prepared_owner_arrays_sha256"],
                "1284 metadata does not bind the prepared arrays hash")
        source_emission = batch_report["source_emission"]
        require(source_emission["all_21_authoritative_parent_terms_preserved"] is True
                and source_emission["full21_weights_order_and_bytes_preserved"] is True
                and source_emission["topology_and_face_ancestry_unchanged_on132134"] is True,
                "1284 source emission does not attest full21/source topology preservation")
        child_path = Path(source_emission["child_payload_path"]).resolve(strict=True)
        extension_path = Path(source_emission["parent_extension_path"]).resolve(strict=True)
        child_sha = source_emission["child_payload_sha256"]
        extension_sha = source_emission["parent_extension_sha256"]
        require(sha(child_path) == child_sha and sha(extension_path) == extension_sha,
                "1284 candidate child/parent extension changed")
        original_matches = [
            Path(path).resolve(strict=True) for path, digest in batch_report["input_sha256"].items()
            if digest == batch_report["source_child_sha256"]
            and Path(path).name == child_path.name
        ]
        require(len(original_matches) == 1, "pinned original ff8 source child path is absent or ambiguous")
        original_child_path = original_matches[0]
        source_child_sha = batch_report["source_child_sha256"]
        require(sha(original_child_path) == source_child_sha == producer_report["source_child_sha256"],
                "original source child differs from the 1291/1284 source pin")
        actual_rest_receipt_path = ACCEPTED / "resting-anatomy-receipt.json"
        actual_rest_receipt = read_json(actual_rest_receipt_path)
        actual_rest_receipt_sha = sha(actual_rest_receipt_path)
        actual_rest_receipt_resolved = actual_rest_receipt_path.resolve()
        receipt_pins = [
            digest for pinned_path, digest in batch_report["input_sha256"].items()
            if Path(pinned_path).resolve() == actual_rest_receipt_resolved
        ]
        require(len(receipt_pins) == 1 and receipt_pins[0] == actual_rest_receipt_sha,
                "1284 report does not uniquely pin the actual 1291 resting-anatomy receipt bytes")
        actual_parent_extension = actual_rest_receipt.get("passive_tissue_parent_bindings")
        require(isinstance(actual_parent_extension, dict)
                and actual_parent_extension.get("child_payload_sha256") == source_child_sha,
                "1291 actual parent extension is not bound to the exact source child")
        actual_parent_extension_sha = _canonical_json_sha256(actual_parent_extension)
        require(all(producer_report["parent_extension_sha256_by_row"][str(sid)]
                    == actual_parent_extension_sha for sid in (132, 134)),
                "1291 actual owner proof does not bind the active full21 parent extension")
        extension = read_json(extension_path)
        require(extension.get("child_payload_sha256") == child_sha
                and isinstance(extension.get("source_payloads"), list),
                "candidate extension lacks exact child/source payload owner")
        index_input_pins = batch_report["input_sha256"]
        for path, expected in (
            (actual_report_path, None),
            (batch_report_path, None),
            (metadata_path, None),
            (index_report_path, None),
            (D1291 / "row132-actual-capture-owner.npz",
             producer_report["output_npz_sha256"]["row132-actual-capture-owner.npz"]),
            (D1291 / "row134-actual-capture-owner.npz",
             producer_report["output_npz_sha256"]["row134-actual-capture-owner.npz"]),
            (index_npz_path, producer_report["index_candidate_sha256"]),
            (ancestry_path, producer_report["index_ancestry_sha256"]),
        ):
            if expected is not None:
                require(sha(path) == expected, f"pinned candidate input SHA changed: {path}")
        for stable_id, row in producer_report["rows"].items():
            require(row["full21_actual_capture_owner_proof"]["accepted_pack_sha256"]
                    == producer_report["accepted_pack_sha256"]
                    and row["full21_actual_capture_owner_proof"]["capture_receipt_sha256"]
                    == producer_report["accepted_step_receipt_sha256"],
                    f"row {stable_id} actual capture proof is not bound to exact run002 bytes")
            require(row["full21_actual_capture_owner_proof"]["displayed_world_owner_relation"]
                    == "accepted_native_capture_byte_exact",
                    f"row {stable_id} is not actual native capture basis")
        require(producer_report["shared_actual_capture_gid_points"]["all_shared_gid_actual_positions_equal"] is True,
                "accepted capture has mismatched shared canonical GID positions")

        actual_rows = {}
        for sid in (132, 134):
            npz_path = D1291 / f"row{sid}-actual-capture-owner.npz"
            npz_hash = producer_report["output_npz_sha256"][npz_path.name]
            require(sha(npz_path) == npz_hash, f"1291 row {sid} owner NPZ changed")
            with np.load(npz_path, allow_pickle=False) as z:
                row = {key: z[key].copy() for key in z.files}
            row_report = producer_report["rows"][str(sid)]
            row["typed_identity"] = row_report["typed_identity"]
            row["full21_actual_capture_owner_proof"] = row_report["full21_actual_capture_owner_proof"]
            row["ordered_core_body_indices"] = row_report["ordered_core_body_indices"]
            row["report"] = row_report["report"]
            actual_rows[sid] = row

        with np.load(D1284 / "prepared-owner-arrays.npz", allow_pickle=False) as prepared:
            prepared_arrays = {key: prepared[key].copy() for key in prepared.files}
        with np.load(index_npz_path, allow_pickle=False) as index:
            index_arrays = {key: index[key].copy() for key in index.files}
        with np.load(D006 / "candidate.npz", allow_pickle=False) as source_map:
            source_map_arrays = {
                sid: {
                    "source_slots": source_map[f"row{sid}_source_slots_i64"].copy(),
                    "gids": source_map[f"row{sid}_canonical_material_gids_u64"].copy(),
                    "vertex_lineage": json.loads(str(source_map[f"row{sid}_vertex_lineage_json"].item())),
                } for sid in (132, 134)
            }
        require(sha(D006 / "candidate.npz") == producer_report["source_map_candidate_sha256"],
                "006 material GID/source-slot candidate changed")
        require(ancestry["candidate_npz_sha256"] == sha(index_npz_path)
                and ancestry["parent_nhtiss_sha256"] == source_child_sha
                and ancestry["rows"]["132"] and ancestry["rows"]["134"],
                "1274 ancestry does not bind candidate and ff8 source child")

        original_data = pc._read_nhtiss4(original_child_path)
        candidate_data = pc._read_nhtiss4(child_path)
        row_data = {}
        original_row_data = {}
        for sid in (132, 134):
            original_record = next((r for r in original_data["records"] if int(r[6]) == sid), None)
            candidate_record = next((r for r in candidate_data["records"] if int(r[6]) == sid), None)
            require(original_record is not None and candidate_record is not None,
                    f"ff8/candidate child lacks stable row {sid}")
            original_row = pc._nhtiss_row_arrays(original_data, original_record)
            candidate_row = pc._nhtiss_row_arrays(candidate_data, candidate_record)
            require(original_row["faces"].shape == candidate_row["faces"].shape
                    and np.ascontiguousarray(original_row["faces"], dtype="<u4").tobytes()
                    == np.ascontiguousarray(candidate_row["faces"], dtype="<u4").tobytes(),
                    f"1284 changed authored source faces for row {sid}")
            row_data[sid] = candidate_row
            original_row_data[sid] = original_row
            actual = actual_rows[sid]

        progress("input_pins_and_row_schemas_verified", start,
                 accepted_pack_sha256=producer_report["accepted_pack_sha256"],
                 accepted_receipt_sha256=producer_report["accepted_step_receipt_sha256"],
                 rows={str(sid): {
                     "capture_vertices": len(actual_rows[sid]["canonical_gids_u64"]),
                     "capture_faces": len(actual_rows[sid]["actual_capture_faces_u32"]),
                 } for sid in (132, 134)})

        result_rows = {}
        candidate_numeric = {}
        face_lineages = {}
        vertex_lineages = {}
        witness_results = []
        for sid in (132, 134):
            actual = actual_rows[sid]
            ids = np.asarray(actual["canonical_gids_u64"], dtype="<u8")
            slots = np.asarray(actual["capture_to_source_slot_i64"], dtype="<i8")
            actual_positions = np.asarray(actual["actual_capture_positions_f32"], dtype="<f4")
            actual_faces = np.asarray(actual["index_patch_faces_capture_u32"], dtype="<u4")
            face_slots = np.asarray(actual["index_patch_faces_source_slots_u32"], dtype="<u4")
            weights = np.asarray(actual["full21_weights_f64"], dtype="<f8")
            captured_source_points = np.asarray(actual["full21_source_points_f32"], dtype="<f4")
            source_row = row_data[sid]
            source_points_all = np.asarray(source_row["positions"], dtype="<f4")
            # Full21 q points are the active parent-carrier source coordinates,
            # not the child's separately registered-rest XYZ coordinates.
            candidate_source_points = captured_source_points.copy()
            require(face_slots.shape == actual_faces.shape
                    and np.all(face_slots < len(source_points_all)),
                    f"row {sid} exact candidate source-slot faces are invalid")
            require(np.ascontiguousarray(index_arrays[f"row{sid}_faces_capture_u32"], dtype="<u4").tobytes()
                    == actual_faces.tobytes()
                    and np.ascontiguousarray(index_arrays[f"row{sid}_faces_source_slots_u32"], dtype="<u4").tobytes()
                    == face_slots.tobytes()
                    and np.ascontiguousarray(index_arrays[f"row{sid}_canonical_material_gids_u64"], dtype="<u8").tobytes()
                    == ids.tobytes()
                    and np.ascontiguousarray(index_arrays[f"row{sid}_positions_actual_run002_f32"], dtype="<f4").tobytes()
                    == actual_positions.tobytes(),
                    f"row {sid} authenticated capture owner differs from exact 1274 index candidate")
            source_map_slots = np.asarray(source_map_arrays[sid]["source_slots"], dtype="<i8")
            source_map_gids = np.asarray(source_map_arrays[sid]["gids"], dtype="<u8")
            require(source_map_slots.ndim == 1 and source_map_gids.shape == source_map_slots.shape,
                    f"row {sid} 006 source slot/GID arrays are malformed")
            map_by_slot = {}
            for map_slot, map_gid in zip(source_map_slots, source_map_gids, strict=True):
                map_slot = int(map_slot)
                if map_slot < 0:
                    continue
                require(map_slot not in map_by_slot,
                        f"row {sid} 006 duplicates original source slot {map_slot}")
                map_by_slot[map_slot] = int(map_gid)
            require(all(map_by_slot.get(int(slot)) == int(gid)
                        for slot, gid in zip(slots, ids, strict=True)),
                    f"row {sid} actual capture canonical GID/slot pairs differ from 006")
            require(candidate_source_points.dtype == np.dtype("<f4"),
                    f"row {sid} candidate source points have wrong type")
            changed_slots = source_emission["changed_source_slots_by_row"][str(sid)]
            ancestry_row_path = D1284 / f"row{sid}-actual972-full21-batch-ancestry.json"
            ancestry_row_sha = sha(ancestry_row_path)
            ancestry_row = read_json(ancestry_row_path)
            require(ancestry_row.get("schema")
                    == "numi.human.forearm.actual972.full21-position-batch-ancestry.v1"
                    and ancestry_row.get("status") == "SOURCE_PREPARATION_ONLY_NOT_ADMISSION"
                    and ancestry_row.get("parent_nhtiss_sha256") == source_child_sha
                    and ancestry_row.get("stable_id") == sid
                    and ancestry_row.get("semantic_id") == 51005
                    and ancestry_row.get("source_faces_and_order_preserved") is True,
                    f"row {sid} 1284 ancestry does not bind the typed source owner")
            changed_records = ancestry_row.get("changed_vertices")
            require(isinstance(changed_records, list)
                    and sorted(int(item["source_slot"]) for item in changed_records) == sorted(changed_slots),
                    f"row {sid} changed source slots disagree with ancestry records")
            slot_to_capture = {int(slot): i for i, slot in enumerate(slots)}
            require(len(slot_to_capture) == len(slots), f"row {sid} has duplicate source slots")
            changed_indices = np.asarray([slot_to_capture[int(slot)] for slot in changed_slots],
                                         dtype=np.int64)
            changed_gids = np.ascontiguousarray(ids[changed_indices], dtype="<u8")
            ancestry_by_slot = {int(item["source_slot"]): item for item in changed_records}
            require(len(ancestry_by_slot) == len(changed_records),
                    f"row {sid} full21 ancestry duplicates a changed source slot")
            for slot, index in zip(changed_slots, changed_indices, strict=True):
                item = ancestry_by_slot[int(slot)]
                require(int(item["canonical_gid"]) == int(ids[index])
                        and int(item["capture_index"]) == int(index),
                        f"row {sid} full21 ancestry GID/capture crosswalk differs at slot {slot}")
                before_world = np.asarray(item["actual_run002_world_before_f32"], dtype="<f4")
                source_after = np.asarray(item["source_point_after_f32"], dtype="<f4")
                recorded_weights = np.asarray(item["full21_weights_float64"], dtype="<f8")
                require(before_world.shape == (3,) and before_world.tobytes()
                        == actual_positions[index].tobytes(),
                        f"row {sid} ancestry baseline world differs from accepted capture at GID {int(ids[index])}")
                require(source_after.shape == (3,) and np.isfinite(source_after).all()
                        and recorded_weights.shape == (21,)
                        and recorded_weights.tobytes() == weights[index].tobytes()
                        and item.get("full21_weights_sha256")
                        == hashlib.sha256(recorded_weights.tobytes()).hexdigest(),
                        f"row {sid} ancestry parent source point/weights are malformed at GID {int(ids[index])}")
                candidate_source_points[index] = source_after
            changed_source = np.ascontiguousarray(candidate_source_points[changed_indices], dtype="<f4")
            require(np.any(np.any(changed_source != captured_source_points[changed_indices], axis=1)),
                    f"row {sid} changed source points unexpectedly remain byte-identical")
            unchanged = np.ones(len(ids), dtype=bool)
            unchanged[changed_indices] = False
            require(candidate_source_points[unchanged].tobytes()
                    == captured_source_points[unchanged].tobytes(),
                    f"row {sid} candidate changed an unselected source point")
            expected_core = tuple(int(x) for x in actual["ordered_core_body_indices"])
            source_table = extension["source_payloads"]
            row_surface = [x for x in extension["surfaces"]
                           if isinstance(x, dict) and x.get("stable_id") == sid]
            require(len(row_surface) == 1, f"row {sid} extension surface is absent/ambiguous")
            terms_by_slot = row_surface[0].get("vertices")
            require(isinstance(terms_by_slot, list)
                    and len(terms_by_slot) == source_row["vertex_count"],
                    f"row {sid} extension source slots do not cover child vertices")
            require(all(
                isinstance(terms_by_slot[int(slot)], list)
                and len(terms_by_slot[int(slot)]) == 21
                and isinstance(terms_by_slot[int(slot)][0], list)
                and len(terms_by_slot[int(slot)][0]) == 4
                for slot in changed_slots
            ), f"row {sid} changed source slot lacks a typed full21 term vector")
            source_indices = {
                int(terms_by_slot[int(slot)][0][0]) for slot in changed_slots
            }
            require(len(source_indices) == 1, f"row {sid} changed terms span multiple carrier owners")
            source_index = next(iter(source_indices))
            require(0 <= source_index < len(source_table), f"row {sid} parent source index is invalid")
            require(all(int(ancestry_by_slot[int(slot)]["source_parent_index"]) == source_index
                        for slot in changed_slots),
                    f"row {sid} explicit parent-carrier ancestry index differs from the candidate extension")
            parent_meta = source_table[source_index]
            parent_path = Path(parent_meta["path"]).resolve(strict=True)
            lane_worlds, lane_report = build_authenticated_full21_lane_world_subset(
                source_payload_path=child_path,
                parent_extension_path=extension_path,
                parent_payload_path=parent_path,
                accepted_step_receipt_path=actual_receipt,
                stable_id=sid,
                source_slots_i64=np.asarray(changed_slots, dtype="<i8"),
                canonical_gids_u64=np.ascontiguousarray(
                    ids[changed_indices], dtype="<u8"),
                source_points_float32=changed_source,
                core_weights_float64=np.ascontiguousarray(weights[changed_indices], dtype="<f8"),
                expected_ordered_core_body_indices=expected_core,
                source_point_basis="full21_parent_carrier",
            )
            lane_worlds = np.ascontiguousarray(lane_worlds, dtype="<f4")
            changed_world = np.zeros((len(changed_gids), 3), dtype="<f4")
            coeff = weights[changed_indices].astype("<f4")
            for lane in range(21):
                term = np.asarray(lane_worlds[:, lane, :] * coeff[:, lane, None], dtype="<f4")
                changed_world = np.asarray(changed_world + term, dtype="<f4")
            positions = actual_positions.copy()
            positions[changed_indices] = changed_world
            retained_targets = np.asarray(
                [ancestry_by_slot[int(slot)]["actual_full21_world_after_f32"]
                 for slot in changed_slots], dtype="<f4")
            replay_residual = np.max(
                np.abs(changed_world.astype(np.float64) - retained_targets.astype(np.float64)),
                axis=1,
            )
            require(float(replay_residual.max(initial=0.0)) <= 2e-7,
                    f"row {sid} selected full21 replay differs from 1284 ancestry by "
                    f"{float(replay_residual.max(initial=0.0)):.9g} m")
            require(sha(ancestry_row_path) == ancestry_row_sha,
                    f"row {sid} 1284 full21 ancestry changed during candidate replay")
            require(not np.array_equal(positions[changed_indices], actual_positions[changed_indices]),
                    f"row {sid} selected predicted positions did not change")
            source_gids_by_slot = {int(slot): int(gid) for slot, gid in zip(slots, ids, strict=True)}
            face_lineage, face_coverage = build_face_lineage(
                stable_id=sid, source_path=str(original_child_path), source_sha=source_child_sha,
                original_source_faces=original_row_data[sid]["faces"],
                candidate_face_source_slots=face_slots,
                candidate_face_gids=ids[actual_faces],
                generated_records=[
                    record for record in ancestry["rows"][str(sid)]
                    if record.get("inferred_local_retriangulation") is True
                ],
                gids_by_source_slot=source_gids_by_slot,
            )
            vertex_map = source_map_arrays[sid]["vertex_lineage"]
            lineage_check = validate_source_gid_slot_lineage(
                slots, ids, vertex_map)
            for i in changed_indices:
                gid = int(ids[i])
                slot = int(slots[i])
                vertex_map[str(gid)]["inferred_local_full21_repair"] = {
                    "kind": "predicted_cpu_full21_source_point_rebase",
                    "source_slot": slot,
                    "source_point_before_f32": captured_source_points[i].tolist(),
                    "source_point_after_f32": candidate_source_points[i].tolist(),
                    "predicted_world_after_f32": positions[i].tolist(),
                    "lane_world_points_sha256": hashlib.sha256(
                        np.ascontiguousarray(lane_worlds[
                            np.flatnonzero(changed_gids == gid)[0]], dtype="<f4").tobytes()).hexdigest(),
                    "candidate_parent_source_index": source_index,
                    "candidate_parent_payload_sha256": lane_report["input_sha256"]["parent"],
                    "candidate_parent_extension_sha256": lane_report["input_sha256"]["extension"],
                    "candidate_child_payload_sha256": lane_report["input_sha256"]["source"],
                    "full21_ancestry_path": str(ancestry_row_path),
                    "full21_ancestry_sha256": ancestry_row_sha,
                    "1291_actual_parent_extension_canonical_sha256": actual_parent_extension_sha,
                    "measured": False,
                }
            face_lineages[sid] = face_lineage
            vertex_lineages[sid] = {int(k): v for k, v in vertex_map.items()
                                    if int(k) in set(map(int, ids))}
            require(set(vertex_lineages[sid]) == set(map(int, ids)),
                    f"row {sid} vertex lineage does not cover all displayed canonical GIDs")
            candidate_numeric.update({
                f"row{sid}_positions_f32": positions,
                f"row{sid}_faces_u32": actual_faces,
                f"row{sid}_source_face_slots_u32": face_slots,
                f"row{sid}_canonical_gids_u64": ids,
                f"row{sid}_source_slots_i64": slots,
                f"row{sid}_weights_f64": weights,
                f"row{sid}_source_points_f32": candidate_source_points,
                f"row{sid}_child_registered_rest_points_f32": np.ascontiguousarray(
                    source_points_all[slots], dtype="<f4"),
                f"row{sid}_changed_gids_u64": changed_gids,
                f"row{sid}_changed_source_points_f32": changed_source,
                f"row{sid}_changed_lane_world_points_f32": lane_worlds,
                f"row{sid}_changed_world_points_predicted_f32": changed_world,
            })
            result_rows[sid] = {
                "typed_identity": [51005, sid],
                "captured_gids": int(len(ids) - len(changed_gids)),
                "predicted_gids": int(len(changed_gids)),
                "full21_parent_carrier_source_ancestry": {
                    "path": str(ancestry_row_path),
                    "sha256": ancestry_row_sha,
                    "1291_actual_parent_extension_canonical_sha256": actual_parent_extension_sha,
                    "1284_candidate_parent_extension_file_sha256": extension_sha,
                    "1291_actual_capture_owner_npz_sha256": npz_hash,
                    "1291_actual_capture_gid_slot_mapping_sha256": hashlib.sha256(
                        np.ascontiguousarray(np.stack((slots, ids.astype("<i8")), axis=1),
                                             dtype="<i8").tobytes()).hexdigest(),
                    "source_point_basis": "explicit full21 parent-carrier ancestry; not child registered-rest XYZ",
                    "all_21_parent_terms_replayed": True,
                },
                "face_coverage": face_coverage,
                "source_gid_slot_lineage": lineage_check,
                "predicted_owner_replay": lane_report,
                "candidate_positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
                "candidate_faces_sha256": hashlib.sha256(actual_faces.tobytes()).hexdigest(),
                "candidate_source_points_sha256": hashlib.sha256(candidate_source_points.tobytes()).hexdigest(),
            }

        progress("candidate_arrays_and_full_face_ancestry_ready", start,
                 row_counts={str(sid): result_rows[sid]["predicted_gids"] for sid in (132, 134)})
        numeric_path = OUT / "mixed-132-134-numeric-candidate.npz"
        np.savez_compressed(numeric_path, **candidate_numeric)
        numeric_sha = sha(numeric_path)
        ancestry_path_out = OUT / "mixed-source-lineage.json"
        ancestry_doc = {
            "schema": "numi.human.forearm-mixed-candidate-source-lineage.v1",
            "source_payload_path": str(original_child_path),
            "source_payload_sha256": source_child_sha,
            "1291_actual_parent_extension_canonical_sha256": actual_parent_extension_sha,
            "1291_actual_resting_anatomy_receipt_sha256": actual_rest_receipt_sha,
            "1291_witness_batch_report_sha256": sha(batch_report_path),
            "1291_witness_batch_checkpoint_sha256": sha(batch_checkpoint_path),
            "index_candidate_sha256": sha(index_npz_path),
            "index_ancestry_sha256": sha(ancestry_path),
            "rows": {str(sid): {
                "source_face_lineage": face_lineages[sid],
                "source_vertex_lineage": {str(g): v for g, v in vertex_lineages[sid].items()},
            } for sid in (132, 134)},
        }
        atomic_json(ancestry_path_out, ancestry_doc)
        ancestry_sha_out = sha(ancestry_path_out)
        progress("numeric_candidate_and_lineage_saved", start,
                 numeric_candidate_sha256=numeric_sha, source_lineage_sha256=ancestry_sha_out)

        mixed_surfaces = {}
        mixed_proof_summary = {}
        for sid in (132, 134):
            actual = actual_rows[sid]
            positions = candidate_numeric[f"row{sid}_positions_f32"]
            faces = candidate_numeric[f"row{sid}_faces_u32"]
            gids = candidate_numeric[f"row{sid}_canonical_gids_u64"]
            slots = candidate_numeric[f"row{sid}_source_slots_i64"]
            weights = candidate_numeric[f"row{sid}_weights_f64"]
            source_points = candidate_numeric[f"row{sid}_source_points_f32"]
            changed_gids = candidate_numeric[f"row{sid}_changed_gids_u64"]
            changed_source = candidate_numeric[f"row{sid}_changed_source_points_f32"]
            changed_lanes = candidate_numeric[f"row{sid}_changed_lane_world_points_f32"]
            parent_meta = result_rows[sid]["predicted_owner_replay"]
            parent_sha = parent_meta["input_sha256"]["parent"]
            extension_hash = parent_meta["input_sha256"]["extension"]
            proof = build_mixed_full21_candidate_owner_record(
                actual_capture_row=actual,
                candidate_positions_f32=positions,
                candidate_faces_u32=faces,
                candidate_gids_u64=gids,
                candidate_source_slots_i64=slots,
                candidate_weights_f64=weights,
                candidate_source_points_f32=source_points,
                predicted_gids_u64=changed_gids,
                predicted_source_points_f32=changed_source,
                predicted_lane_world_points_f32=changed_lanes,
                candidate_parent_payload_sha256=parent_sha,
                candidate_parent_extension_sha256=extension_hash,
                candidate_child_payload_sha256=child_sha,
                candidate_archive_sha256=numeric_sha,
                source_face_lineage=face_lineages[sid],
                source_vertex_lineage=vertex_lineages[sid],
            )
            surface = verify.prepare_surface(
                (51005, sid), positions, faces,
                source_vertex_ids=gids,
                source_face_lineage=face_lineages[sid],
                source_vertex_lineage=vertex_lineages[sid],
                ordered_core_body_indices=actual["ordered_core_body_indices"],
                core_weight_gids=gids,
                core_weights_float64=weights,
                core_source_point_gids=gids,
                core_source_points_float32=source_points,
                core_capture_source_slots_i64=slots,
                core_mixed_candidate_owner_proof=proof,
                expected_ordered_core_body_indices=actual["ordered_core_body_indices"],
            )
            mixed_surfaces[sid] = surface
            diagnostics = surface.diagnostics()
            mixed_proof_summary[str(sid)] = {
                "schema": proof["schema"],
                "typed_identity": proof["typed_identity"],
                "accepted_step": proof["accepted_step"],
                "accepted_pack_sha256": proof["accepted_pack_sha256"],
                "capture_receipt_sha256": proof["capture_receipt_sha256"],
                "candidate_archive_sha256": proof["candidate_archive_sha256"],
                "candidate_parent_payload_sha256": proof["candidate_parent_payload_sha256"],
                "candidate_parent_extension_sha256": proof["candidate_parent_extension_sha256"],
                "candidate_child_payload_sha256": proof["candidate_child_payload_sha256"],
                "candidate_positions_sha256": proof["candidate_positions_sha256"],
                "source_face_lineage_sha256": proof["source_face_lineage_sha256"],
                "source_vertex_lineage_sha256": proof["source_vertex_lineage_sha256"],
                "predicted_gid_count": len(changed_gids),
                "captured_gid_count": len(gids) - len(changed_gids),
                "displayed_world_owner_relation": proof["displayed_world_owner_relation"],
                "native_capture_performed": False,
                "anatomy_admission": False,
                "prepared_surface_mixed_owner_diagnostics": diagnostics.get(
                    "full21_mixed_candidate_owner_proof"),
            }
        progress("mixed_full21_surface_proofs_ready", start,
                 row_ids=[[51005, 132], [51005, 134]])

        # Remap the three retained pre-repair witness face ordinals through exact
        # source-face ancestry. If a face was locally retriangulated, use only
        # candidate faces whose authenticated support includes that exact ordinal.
        def descendants(sid, old_face):
            result = []
            for fi, record in enumerate(face_lineages[sid]):
                if record.get("source_face") == old_face:
                    result.append(fi)
                elif old_face in record.get("source_face_support", []):
                    result.append(fi)
            return result

        target_pairs = [
            {"kind": "self", "rows": [134, 134], "old_faces": [62099, 62101],
             "label": "row134_self"},
            {"kind": "mutual", "rows": [132, 134], "old_faces": [5030, 49329],
             "label": "row132_134_pair_5030_49329"},
            {"kind": "mutual", "rows": [132, 134], "old_faces": [5031, 49329],
             "label": "row132_134_pair_5031_49329"},
        ]
        for item in target_pairs:
            a_id, b_id = item["rows"]
            f0s = descendants(a_id, item["old_faces"][0])
            f1s = descendants(b_id, item["old_faces"][1])
            require(f0s and f1s,
                    f"retained witness {item['label']} could not be mapped through exact source face support")
            checked = []
            for f0 in f0s:
                for f1 in f1s:
                    if a_id == b_id and f0 >= f1:
                        continue
                    points = mixed_surfaces[a_id].intersection_points_meters(
                        mixed_surfaces[b_id], f0, f1)
                    checked.append({
                        "candidate_faces": [f0, f1],
                        "exact_intersection_point_count": len(points),
                        "exact_points_m": exact_fraction_rows(points),
                    })
            remaining = [row for row in checked if row["exact_intersection_point_count"]]
            witness_results.append({
                "label": item["label"], "typed_rows": [[51005, a_id], [51005, b_id]],
                "old_source_faces": item["old_faces"],
                "candidate_face_descendant_counts": [len(f0s), len(f1s)],
                "candidate_face_pair_count_checked": len(checked),
                "remaining_intersection_count": len(remaining),
                "remaining_exact_pairs": remaining,
                "passes_targeted_exact_geometry": not remaining,
            })
        atomic_json(OUT / "witness-results.json", {
            "schema": "numi.human.forearm-mixed-candidate-targeted-exact-witnesses.v1",
            "accepted_pack_sha256": producer_report["accepted_pack_sha256"],
            "accepted_step_receipt_sha256": producer_report["accepted_step_receipt_sha256"],
            "numeric_candidate_sha256": numeric_sha,
            "source_lineage_sha256": ancestry_sha_out,
            "basis": "actual accepted capture for unchanged GIDs; predicted CPU full21 owner replay for 44 changed GIDs",
            "witnesses": witness_results,
            "all_three_targeted_witnesses_clear": all(
                row["passes_targeted_exact_geometry"] for row in witness_results),
            "native_capture_performed": False,
            "anatomy_admission": False,
        })
        all_clear = all(row["passes_targeted_exact_geometry"] for row in witness_results)
        report = {
            "schema": "numi.human.forearm-mixed-full21-candidate.v1",
            "status": "TARGETED_WITNESSES_CLEAR_UNADMITTED" if all_clear else "TARGETED_WITNESS_FAILURE",
            "accepted_run002_pack_sha256": producer_report["accepted_pack_sha256"],
            "accepted_step_receipt_sha256": producer_report["accepted_step_receipt_sha256"],
            "source_child_sha256": source_child_sha,
            "actual_capture_owner_proof_sha256": sha(actual_report_path),
            "prepared_arrays_sha256": sha(D1284 / "prepared-owner-arrays.npz"),
            "source_map_candidate_sha256": producer_report["source_map_candidate_sha256"],
            "index_candidate_sha256": sha(index_npz_path),
            "index_ancestry_sha256": sha(ancestry_path),
            "numeric_candidate_path": str(numeric_path),
            "numeric_candidate_sha256": numeric_sha,
            "builder_script_path": str(Path(__file__).resolve()),
            "builder_script_sha256": sha(Path(__file__).resolve()),
            "loaded_owner_module_sha256": {
                "forearm_actual_capture_owner.py": sha(REPO / "src/numilab_human/forearm_actual_capture_owner.py"),
                "forearm_parent_carrier_bridge.py": sha(REPO / "src/numilab_human/forearm_parent_carrier_bridge.py"),
                "resting_anatomy_verify.py": sha(REPO / "src/numilab_human/resting_anatomy_verify.py"),
                "passive_attachment_composition.py": sha(REPO / "src/numilab_human/passive_attachment_composition.py"),
            },
            "source_lineage_path": str(ancestry_path_out),
            "source_lineage_sha256": ancestry_sha_out,
            "rows": result_rows,
            "mixed_owner_proof_rows": mixed_proof_summary,
            "targeted_witnesses_path": str(OUT / "witness-results.json"),
            "targeted_witnesses_sha256": sha(OUT / "witness-results.json"),
            "targeted_witness_results": witness_results,
            "all_three_targeted_witnesses_clear": all_clear,
            "native_capture_performed": False,
            "anatomy_admission": False,
            "evidence_boundary": (
                "unchanged per-GID positions remain exact accepted native capture bytes; "
                "44 changed points are exact CPU full21 Float32 re-emissions at accepted poses; "
                "candidate geometry and exact witnesses are offline predicted evidence only"
            ),
        }
        atomic_json(OUT / "report.json", report)
        progress("complete", start, status=report["status"],
                 numeric_candidate_sha256=numeric_sha)
    except BaseException as exc:
        atomic_json(OUT / "failure.json", {
            "schema": "numi.human.forearm-mixed-full21-candidate-failure.v1",
            "stage": read_json(OUT / "progress.json").get("stage"),
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - start,
            "not_native_capture": True, "not_anatomy_admission": True,
        })
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
