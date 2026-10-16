#!/usr/bin/env python3
"""Prepare current-972 full21 forearm source emission; no anatomy admission."""
from __future__ import annotations
import copy, hashlib, json, os, signal, sys, time, traceback, shutil
from pathlib import Path
import numpy as np

D = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378")
HP = Path("/Users/n/numi-human-final-integration-publish-1397")
BR = Path("/Users/n/numi-human-forearm-parent-bridge-20261017")
C = Path("/Users/n/numi-human-aortic-port-proof-consumer-20261016")
SMOKE_ONLY = "--smoke" in sys.argv
OUT = D / ("forearm-current972-full21-source-emission-1227-smoke" if SMOKE_ONLY else "forearm-current972-full21-source-emission-1228")
DISK_FREE_BEFORE_BYTES = shutil.disk_usage("/Users/n").free
if DISK_FREE_BEFORE_BYTES < 10 * 1024**3:
    raise RuntimeError(f"CPU source-emission guard requires 10 GiB free, has {DISK_FREE_BEFORE_BYTES}")
CHILD = D / "deltoid75-78-existing-shared-parent-source-824/conformed/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue"
CAND = D / "forearm-native007-current972-joint-source-containing-006/candidate.npz"
CAND_REPORT = D / "forearm-native007-current972-joint-source-containing-006/report.json"
CAP = D / "pons323-348-current972-existing-owner-rebind-001/native-run-002/accepted-geometry/step-0.mrvpack"
STEP = D / "pons323-348-current972-existing-owner-rebind-001/native-run-002/accepted-geometry/step-0.receipt.json"
REST = D / "pons323-348-current972-existing-owner-rebind-001/native-run-002/resting-anatomy-receipt.json"
MAP_REPORT = D / "shared-muscle-owner-replay-diagnosis-1197/report.json"
MAP_SCRIPT = D / "diagnose_shared_muscle_owner_replay_1197.py"
LINEAGE_REPORT = D / "forearm-current972-full21-source-emission-1209-lineage-contract/report.json"
R1_DIR = D / "forearm-native007-current972-baseline-local-bone-census-008r1"
R1_CANDIDATE = R1_DIR / "candidate-current972.npz"
R1_FRAME = R1_DIR / "frame-rebase.json"
R1_SHA = "7dd9b1427e5b4e9f1c0139de8afff8e68369a8b495aaa45be4e666a4d87fd88c"
OUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(HP / "src"))
sys.path.insert(0, str(C / "src"))
sys.path.insert(0, str(BR / "src"))
from numilab_human import passive_attachment_composition as pc
from numilab_human import resting_anatomy_verify as v
from numilab_human.forearm_parent_carrier_bridge import (
    accepted_poses_from_receipt, append_full21_carrier_points,
    derive_child_four_lane_birth, derive_full21_birth_source_point,
    full21_parent_world_positions, solve_full21_parent_source_points,
    remap_child_four_lane_route_by_binding_identity,
    emit_child_with_inferred_four_lane_route,
    remap_full21_terms_by_source_slot, replace_parent_surface_terms,
    validate_source_gid_slot_lineage,
    resolve_birth_source_parent_slots,
)

BASE_SHA = "ff8db8396bf49b307ca50ad85f22f19c1167a0f43a625c6439a05b2548bd9126"
CAND_SHA = "51fd5d77aaf1e1ac839cb3e9d83e947c6386df63d2cf9533db9f92da1dbe25c7"
PACK_SHA = "80fadf29f92ffff1ec7a3e22c70f645b003738720844689668a7081ed0232e7f"
STEP_SHA = "6b24f77959226729d54fddc8365c4d9e89fd932988b6279479df619f611192cd"
EXPECTED_MAP = {
    132: ("717268b841b9611b12234a603ca45cfe3c6c621a48a6592f9f4fe17a4df098c2", 14706),
    134: ("ff4fd410731e58f8ec3d3888c36b74ee1f1ad32bf5a2ed928f67e149eafcfa5b", 32351),
}
PATCHES = {
    86: (D / "muscle86-actual-native-source-index-repair-1180/row86-source-index-patch.npz",
         D / "muscle86-actual-native-source-index-repair-1180/local-repair-ancestry.json"),
    104: (D / "biceps104-actual-native-source-index-repair-1145/row104-source-index-patch.npz",
          D / "biceps104-actual-native-source-index-repair-1145/local-repair-ancestry.json"),
    105: (D / "muscle-precision-connected-batch-1192/105/source-index-patch.npz",
          D / "muscle-precision-connected-batch-1192/105/local-repair-ancestry.json"),
    106: (D / "muscle-precision-connected-batch-1192/106/source-index-patch.npz",
          D / "muscle-precision-connected-batch-1192/106/local-repair-ancestry.json"),
    153: (D / "muscle-precision-connected-batch-1192/153/source-index-patch.npz",
          D / "muscle-precision-connected-batch-1192/153/local-repair-ancestry.json"),
    158: (D / "muscle-precision-connected-batch-1192/158/source-index-patch.npz",
          D / "muscle-precision-connected-batch-1192/158/local-repair-ancestry.json"),
}
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def save_progress(report):
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")

def alarm_handler(_signum, _frame):
    raise TimeoutError("20-minute forearm source-emission bound reached")

signal.signal(signal.SIGALRM, alarm_handler)
signal.alarm(1200)
started = time.monotonic()
active_stage = "immutable_inputs_and_patch_ancestry"
report = {
    "schema": "numi.human.forearm-current972-full21-source-emission.v1",
    "status": "RUNNING",
    "output": str(OUT),
    "disk_free_bytes_before": DISK_FREE_BEFORE_BYTES,
    "source_candidate_path": str(CAND),
    "source_candidate_sha256_expected": CAND_SHA,
    "source_candidate_world_positions_used": False,
    "source_candidate_source_point_arrays_used": False,
    "source_candidate_full21_weight_arrays_used": False,
    "accepted_capture_path": str(CAP),
    "accepted_capture_step_receipt_path": str(STEP),
    "accepted_resting_receipt_path": str(REST),
    "changed_rows": [86, 104, 105, 106, 132, 134, 153, 158],
    "anatomical_admission": False,
    "native_admission": False,
    "stages": [],
}
save_progress(report)
try:
    # Pin and validate all files and row-patch ancestry before parsing geometry.
    active_stage = "immutable_inputs_and_patch_ancestry"
    child_manifest = CHILD.with_suffix(".manifest.json")
    base_inputs = [CHILD, child_manifest, CAND, CAND_REPORT, CAP, STEP, REST,
                   MAP_REPORT, MAP_SCRIPT, LINEAGE_REPORT, R1_CANDIDATE, R1_FRAME,
                   Path(pc.__file__), Path(v.__file__), Path(__file__),
                   BR / "src/numilab_human/forearm_parent_carrier_bridge.py"]
    for patch, proof in PATCHES.values():
        base_inputs += [patch, proof]
    input_pins = {str(p): sha(p) for p in base_inputs}
    if input_pins[str(CHILD)] != BASE_SHA:
        raise ValueError("current child payload SHA differs from pinned ff8 lineage")
    if input_pins[str(CAND)] != CAND_SHA:
        raise ValueError("wrong forearm candidate; only pinned 51fd ancestry is accepted")
    if input_pins[str(CAP)] != PACK_SHA or input_pins[str(STEP)] != STEP_SHA:
        raise ValueError("accepted002 capture or step receipt identity differs")
    if input_pins[str(R1_CANDIDATE)] != R1_SHA:
        raise ValueError("008r1 ancestral material table differs from its exact pinned candidate")
    r1_frame = json.loads(R1_FRAME.read_text())
    if (r1_frame.get("rebased_candidate_sha256") != R1_SHA
            or r1_frame.get("rebased_candidate") != str(R1_CANDIDATE)
            or r1_frame.get("status") != "PASS_FRAME_REBASE_PRECONDITION_ONLY_NOT_NATIVE_OR_ANATOMY_ADMISSION"
            or r1_frame.get("full21_source_terms_preserved") is not True
            or r1_frame.get("typed_rows") != [[51005, 132], [51005, 134]]):
        raise ValueError("008r1 frame-rebase receipt does not authenticate the typed source ancestry")
    cand_report = json.loads(CAND_REPORT.read_text())
    if cand_report.get("candidate_sha256") != CAND_SHA:
        raise ValueError("006 report does not bind the exact 51fd candidate")
    if cand_report.get("candidate_path") != str(CAND):
        raise ValueError("006 report candidate path differs")
    for sid, (patch, proof) in PATCHES.items():
        evidence = json.loads(proof.read_text())
        if evidence.get("source_payload_sha256") != BASE_SHA:
            raise ValueError(f"row {sid} source-index patch is for a different base payload")
        if evidence.get("all_source_position_and_binding_weight_bytes_unchanged") is not True:
            raise ValueError(f"row {sid} patch does not preserve source positions and bindings")
        if not patch.is_file():
            raise FileNotFoundError(patch)
    report["input_sha256"] = input_pins
    report["stages"].append({"stage": "immutable_inputs_and_patch_ancestry", "passed": True})
    save_progress(report)

    active_stage = "accepted002_and_full21_owner_load"
    source = pc._read_nhtiss4(CHILD)
    if len(source["records"]) != 164:
        raise ValueError(f"current source row count is {len(source['records'])}, expected 164")
    records = {int(r[6]): r for r in source["records"]}
    receipt = json.loads(REST.read_text())
    extension = copy.deepcopy(receipt["passive_tissue_parent_bindings"])
    if extension.get("child_payload_sha256") != BASE_SHA:
        raise ValueError("accepted002 extension is not bound to exact ff8 child payload")
    if len(extension.get("surfaces", [])) != 17 or len(extension.get("source_payloads", [])) != 8:
        raise ValueError("accepted002 parent extension inventory differs from pinned owner")
    carrier_index_by_sid = {132: 0, 134: 5}
    extension_surface_by_sid = {int(item["stable_id"]): item for item in extension["surfaces"]}
    for sid, source_index in carrier_index_by_sid.items():
        source_meta = extension["source_payloads"][source_index]
        source_path = Path(source_meta["path"]).resolve()
        source_digest = sha(source_path)
        if source_digest != source_meta.get("sha256"):
            raise ValueError(f"row {sid} active full21 carrier SHA differs from accepted002 extension")
        input_pins[str(source_path)] = source_digest
    poses, pose_report = accepted_poses_from_receipt(STEP)
    if pose_report["accepted_receipt_sha256"] != STEP_SHA:
        raise ValueError("accepted dynamic pose receipt SHA mismatch")
    captures, _capture_metadata = v.load_capture(CAP, STEP)
    report["pose_owner"] = pose_report
    source_row_cache = {}
    def load_source_row(stable_id):
        stable_id = int(stable_id)
        if stable_id not in source_row_cache:
            source_row_cache[stable_id] = pc._nhtiss_row_arrays(source, records[stable_id])
        return source_row_cache[stable_id]

    report["stages"].append({"stage": "accepted002_and_full21_owner_loaded",
                             "passed": True, "source_rows": len(records),
                             "accepted_capture_rows": len(captures)})
    save_progress(report)

    candidate_rows = {}
    all_birth_world_by_gid = {}
    parent_carrier_outputs = {}
    pending_terms = {}
    current_world_rows = {}
    map_report = json.loads(MAP_REPORT.read_text())
    lineage_contract_report = json.loads(LINEAGE_REPORT.read_text())
    if lineage_contract_report.get("candidate_sha256") != CAND_SHA:
        raise ValueError("1209 lineage contract report is not bound to exact 51fd candidate")
    if map_report.get("inputs", {}).get(str(CHILD)) != BASE_SHA:
        raise ValueError("1197 map report is not bound to the exact current source")
    if map_report.get("inputs", {}).get(str(CAP)) != PACK_SHA:
        raise ValueError("1197 map report is not bound to accepted002 pack")
    if map_report.get("inputs", {}).get(str(STEP)) != STEP_SHA:
        raise ValueError("1197 map report is not bound to accepted002 step receipt")

    with np.load(CAND, allow_pickle=False) as archive:
        arrays = {"new_shared_cut_gids_u64": archive["new_shared_cut_gids_u64"].copy()}
        for candidate_sid in (132, 134):
            prefix = f"row{candidate_sid}_"
            for field in ("source_slots_i64", "canonical_material_gids_u64",
                          "faces_local_u32", "vertex_lineage_json",
                          "source_points_f32", "core_weights_float64"):
                arrays[prefix + field] = archive[prefix + field].copy()

    # 008r1 is the pinned ancestral material namespace for candidate parent
    # GIDs retired from the final visible topology. It supplies typed row and
    # source-slot lineage, never absolute candidate world coordinates.
    r1_arrays = {}
    r1_gid_owners = {}
    with np.load(R1_CANDIDATE, allow_pickle=False) as archive:
        for owner_sid in (132, 134):
            prefix = f"row{owner_sid}_"
            owner_gids = np.asarray(archive[prefix + "canonical_material_gids_u64"], dtype="<u8").copy()
            owner_slots = np.asarray(archive[prefix + "source_slots_i64"], dtype=np.int64).copy()
            owner_lineage = json.loads(str(archive[prefix + "vertex_lineage_json"].item()))
            owner_points = np.asarray(archive[prefix + "source_points_f32"], dtype="<f4").copy()
            owner_weights = np.asarray(archive[prefix + "core_weights_float64"], dtype=np.float64).copy()
            owner_world = np.asarray(archive[prefix + "positions_world_f32"], dtype="<f4").copy()
            if (owner_gids.ndim != 1 or owner_slots.shape != owner_gids.shape
                    or owner_points.shape != (len(owner_gids), 3)
                    or owner_weights.shape != (len(owner_gids), 21)
                    or owner_world.shape != (len(owner_gids), 3)
                    or len(set(map(int, owner_gids))) != len(owner_gids)
                    or np.any(owner_slots < -1)
                    or not np.isfinite(owner_points).all()
                    or not np.isfinite(owner_weights).all()
                    or not np.isfinite(owner_world).all()):
                raise ValueError(f"008r1 row {owner_sid} ancestral material table is malformed")
            source_count = load_source_row(owner_sid)["vertex_count"]
            if np.any(owner_slots >= source_count):
                raise ValueError(f"008r1 row {owner_sid} source-slot crosswalk is out of range")
            r1_arrays[owner_sid] = {
                "gids": owner_gids, "slots": owner_slots, "lineage": owner_lineage,
                "source_points": owner_points, "core_weights": owner_weights,
                "world_positions": owner_world,
                "gid_to_vertex": {int(g): i for i, g in enumerate(owner_gids)},
            }
            for owner_vertex, owner_gid_u64 in enumerate(owner_gids):
                r1_gid_owners.setdefault(int(owner_gid_u64), []).append({
                    "owner_sid": owner_sid, "vertex": int(owner_vertex),
                    "source_slot": int(owner_slots[owner_vertex]),
                })

    # Authenticate both candidate rows' canonical GID/source-slot ancestry and
    # every birth edge before any full21 replay or inverse solve.
    active_stage = "candidate_lineage_and_008r1_parent_preflight"
    candidate_lineage_preflight = {}
    for pre_sid in (132, 134):
        pre_prefix = f"row{pre_sid}_"
        pre_slots = np.asarray(arrays[pre_prefix + "source_slots_i64"], dtype=np.int64)
        pre_gids = np.asarray(arrays[pre_prefix + "canonical_material_gids_u64"], dtype="<u8")
        pre_faces = np.asarray(arrays[pre_prefix + "faces_local_u32"], dtype="<u4")
        pre_lineage = json.loads(str(np.asarray(arrays[pre_prefix + "vertex_lineage_json"]).item()))
        if pre_slots.ndim != 1 or pre_gids.shape != pre_slots.shape:
            raise ValueError(f"row {pre_sid} candidate GID/source-slot arrays are not aligned")
        if pre_faces.ndim != 2 or pre_faces.shape[1] != 3 or int(pre_faces.max(initial=0)) >= len(pre_slots):
            raise ValueError(f"row {pre_sid} candidate faces have invalid shape/index")
        pre_row = pc._nhtiss_row_arrays(source, records[pre_sid])
        if np.any(pre_slots < -1) or np.any(pre_slots >= pre_row["vertex_count"]):
            raise ValueError(f"row {pre_sid} candidate source slot is outside current child row")
        if len(set(map(int, pre_gids))) != len(pre_gids):
            raise ValueError(f"row {pre_sid} candidate canonical GIDs are ambiguous")
        pre_gid_report = validate_source_gid_slot_lineage(pre_slots, pre_gids, pre_lineage)
        pre_gid_to_vertex = {int(g): i for i, g in enumerate(pre_gids)}
        pre_births = np.flatnonzero(pre_slots == -1)
        key_inventory = {}
        parent_lengths = {}
        source_point_count = 0
        full21_field_count = 0
        alias_count = 0
        for pre_vertex in pre_births:
            pre_gid = int(pre_gids[pre_vertex])
            pre_item = pre_lineage.get(str(pre_gid))
            if not isinstance(pre_item, dict):
                raise ValueError(f"row {pre_sid} birth GID {pre_gid} lacks lineage object")
            pre_kind = str(pre_item.get("source_face_support_provenance", ""))
            pre_keys = key_inventory.setdefault(pre_kind, {
                "count": 0, "sample_gid": pre_gid, "sample_keys": sorted(pre_item),
                "sample_types": {key: type(value).__name__ for key, value in sorted(pre_item.items())},
                "parent_reference_resolution": {
                    "same_typed_candidate_row": 0,
                    "pinned_008r1_typed_original_source_slot": 0,
                    "pinned_008r1_typed_declared_birth": 0,
                },
            })
            pre_keys["count"] += 1
            pre_parents = pre_item.get("source_parent_gids")
            if pre_parents is not None:
                if (not isinstance(pre_parents, list)
                        or not all(type(gid) is int for gid in pre_parents)
                        or len(set(pre_parents)) != len(pre_parents)):
                    raise ValueError(f"row {pre_sid} birth GID {pre_gid} has malformed source parent GIDs")
                parent_lengths[str(len(pre_parents))] = parent_lengths.get(str(len(pre_parents)), 0) + 1
            edge_fraction = pre_item.get("edge_fraction_from_lower_canonical_gid")
            is_edge_birth = (
                isinstance(pre_parents, list) and len(pre_parents) == 2
                and type(edge_fraction) in (int, float)
                and np.isfinite(float(edge_fraction))
                and 0.0 <= float(edge_fraction) <= 1.0
            )
            if edge_fraction is not None and not is_edge_birth:
                raise ValueError(f"row {pre_sid} birth GID {pre_gid} has a malformed edge fraction")
            for parent_gid in pre_parents or []:
                if parent_gid in pre_gid_to_vertex:
                    pre_keys["parent_reference_resolution"]["same_typed_candidate_row"] += 1
                    continue
                ancestors = r1_gid_owners.get(parent_gid, [])
                if len(ancestors) != 1:
                    raise ValueError(
                        f"row {pre_sid} birth GID {pre_gid} parent GID {parent_gid} "
                        f"has {len(ancestors)} pinned 008r1 typed source owners"
                    )
                ancestor = ancestors[0]
                if ancestor["source_slot"] >= 0:
                    pre_keys["parent_reference_resolution"]["pinned_008r1_typed_original_source_slot"] += 1
                else:
                    ancestor_item = r1_arrays[ancestor["owner_sid"]]["lineage"].get(str(parent_gid))
                    if not isinstance(ancestor_item, dict) or not (
                            ancestor_item.get("source_parent_gids")
                            or type(ancestor_item.get("source_vertex_id")) is int):
                        raise ValueError(
                            f"row {pre_sid} birth GID {pre_gid} parent GID {parent_gid} "
                            "has no declared 008r1 ancestry"
                        )
                    pre_keys["parent_reference_resolution"]["pinned_008r1_typed_declared_birth"] += 1
            if is_edge_birth:
                absent = [parent_gid for parent_gid in pre_parents if parent_gid not in pre_gid_to_vertex]
                if absent:
                    raise ValueError(
                        f"row {pre_sid} edge birth GID {pre_gid} endpoints are not in its exact candidate row: {absent}"
                    )
                if any(int(pre_slots[pre_gid_to_vertex[parent_gid]]) < 0 for parent_gid in pre_parents):
                    raise ValueError(f"row {pre_sid} edge birth GID {pre_gid} endpoint is not an original source slot")
            source_point = pre_item.get("source_point_f32")
            if source_point is not None:
                point = np.asarray(source_point, dtype="<f4")
                if point.shape != (3,) or not np.isfinite(point).all():
                    raise ValueError(f"row {pre_sid} birth GID {pre_gid} has malformed source_point_f32")
                source_point_count += 1
            core_weights = pre_item.get("core_weights")
            if core_weights is not None:
                weights = np.asarray(core_weights, dtype=np.float64)
                if (weights.shape != (21,) or not np.isfinite(weights).all()
                        or np.any(weights <= 0) or np.any(weights > 1)
                        or abs(float(weights.sum()) - 1.0) > 1e-12):
                    raise ValueError(f"row {pre_sid} birth GID {pre_gid} has invalid full21 core_weights")
                full21_field_count += 1
                body_ids = pre_item.get("core_body_indices")
                if (not isinstance(body_ids, list) or len(body_ids) != 21
                        or not all(type(body) is int for body in body_ids)):
                    raise ValueError(f"row {pre_sid} birth GID {pre_gid} lacks ordered 21 core_body_indices")
            if not is_edge_birth and (source_point is None or core_weights is None):
                is_identity_alias = (
                    pre_kind == "original-keyed-material-handoff-GID"
                    and type(pre_item.get("source_vertex_id")) is int
                    and pre_item["source_vertex_id"] >= 0
                )
                if not is_identity_alias:
                    raise ValueError(f"row {pre_sid} birth GID {pre_gid} has no full21 source point/field or exact material identity alias")
                alias_count += 1
        candidate_lineage_preflight[str(pre_sid)] = {
            **pre_gid_report,
            "candidate_vertex_count": int(len(pre_slots)),
            "candidate_face_count": int(len(pre_faces)),
            "birth_count": int(len(pre_births)),
            "births_with_explicit_full21_source_point": source_point_count,
            "births_with_explicit_full21_field": full21_field_count,
            "identity_alias_birth_count": alias_count,
            "source_parent_gid_list_lengths": parent_lengths,
            "lineage_kind_key_type_inventory": key_inventory,
            "birth_parent_gid_references_resolved_through_candidate_or_pinned_008r1": True,
            "edge_parent_gids_resolved_in_same_typed_candidate_row": True,
            "pinned_008r1_ancestral_candidate_sha256": R1_SHA,
        }
    report["candidate_lineage_preflight"] = candidate_lineage_preflight
    report["pinned_008r1_ancestral_candidate_sha256"] = R1_SHA
    report["stages"].append({
        "stage": "both_forearm_candidate_rows_gid_slot_and_birth_ancestry_preflight",
        "passed": True,
    })
    save_progress(report)

    carrier_cache = {}

    def load_carrier_owner(owner_sid):
        owner_sid = int(owner_sid)
        if owner_sid in carrier_cache:
            return carrier_cache[owner_sid]
        source_index = carrier_index_by_sid[owner_sid]
        source_meta = extension["source_payloads"][source_index]
        carrier_path = Path(source_meta["path"]).resolve()
        raw = carrier_path.read_bytes()
        raw_sha = hashlib.sha256(raw).hexdigest()
        if raw_sha != source_meta["sha256"] or raw_sha != input_pins[str(carrier_path)]:
            raise ValueError(f"row {owner_sid} active parent carrier changed after input pin")
        carrier_data = pc._read_nhtiss4(carrier_path)
        carrier_record = next((record for record in carrier_data["records"]
                               if int(record[6]) == owner_sid), None)
        if carrier_record is None:
            raise ValueError(f"row {owner_sid} active parent carrier lacks its typed source row")
        carrier = pc._nhtiss_row_arrays(carrier_data, carrier_record)
        if carrier["vertex_count"] % 21:
            raise ValueError(f"row {owner_sid} active parent has incomplete full21 source groups")
        grouped_points = carrier["positions"].reshape((-1, 21, 3))
        grouped_normals = carrier["normals"].reshape((-1, 21, 3))
        carrier_points = grouped_points[:, 0, :].copy()
        carrier_normals = grouped_normals[:, 0, :].copy()
        if not np.array_equal(grouped_points, np.repeat(carrier_points[:, None, :], 21, axis=1)):
            raise ValueError(f"row {owner_sid} active parent lanes differ in source position")
        if not np.array_equal(grouped_normals, np.repeat(carrier_normals[:, None, :], 21, axis=1)):
            raise ValueError(f"row {owner_sid} active parent lanes differ in source normal")
        terms = extension_surface_by_sid[owner_sid]["vertices"]
        owner_weights = np.asarray([[float(term[3]) for term in vertex_terms]
                                     for vertex_terms in terms], dtype=np.float64)
        if owner_weights.shape != (len(carrier_points), 21):
            raise ValueError(f"row {owner_sid} active full21 terms and carrier points do not align")
        if (not np.isfinite(owner_weights).all() or np.any(owner_weights <= 0)
                or np.max(np.abs(owner_weights.sum(axis=1) - 1.0)) > 1e-12):
            raise ValueError(f"row {owner_sid} active full21 source field is not positive normalized")
        core_ids = tuple(int(binding["core"]) for binding in carrier["bindings"])
        if len(core_ids) != 21 or len(set(core_ids)) != 21:
            raise ValueError(f"row {owner_sid} active carrier binding order is not full21")
        result = {
            "source_index": source_index, "carrier_path": carrier_path,
            "carrier_meta": source_meta, "carrier": carrier,
            "points": carrier_points, "normals": carrier_normals,
            "weights": owner_weights, "core_ids": core_ids,
        }
        carrier_cache[owner_sid] = result
        return result

    source_identity_owners = {}
    candidate_lineage_by_sid = {}
    candidate_slots_by_sid = {}
    candidate_gids_by_sid = {}
    for owner_sid in (132, 134):
        owner_prefix = f"row{owner_sid}_"
        owner_slots = np.asarray(arrays[owner_prefix + "source_slots_i64"], dtype=np.int64)
        owner_gids = np.asarray(arrays[owner_prefix + "canonical_material_gids_u64"], dtype="<u8")
        owner_lineage = json.loads(str(np.asarray(arrays[owner_prefix + "vertex_lineage_json"]).item()))
        candidate_lineage_by_sid[owner_sid] = owner_lineage
        candidate_slots_by_sid[owner_sid] = owner_slots
        candidate_gids_by_sid[owner_sid] = owner_gids
        for owner_vertex in np.flatnonzero(owner_slots >= 0):
            owner_gid = int(owner_gids[owner_vertex])
            owner_item = owner_lineage.get(str(owner_gid))
            if not isinstance(owner_item, dict):
                raise ValueError(f"row {owner_sid} original GID {owner_gid} lacks source identity")
            source_vertex_id = owner_item.get("source_vertex_id")
            if type(source_vertex_id) is not int or source_vertex_id < 0:
                raise ValueError(f"row {owner_sid} original GID {owner_gid} has invalid source_vertex_id")
            source_identity_owners.setdefault(source_vertex_id, []).append(
                (owner_sid, int(owner_slots[owner_vertex]), owner_gid)
            )

    def resolve_identity_alias(target_sid, source_vertex_id):
        records_for_identity = source_identity_owners.get(int(source_vertex_id), [])
        if not records_for_identity:
            raise ValueError(
                f"row {target_sid} identity alias source_vertex_id {source_vertex_id} "
                "has no exact original-keyed source owner"
            )
        target_parent = load_carrier_owner(target_sid)
        resolved = []
        for owner_sid, owner_slot, owner_gid in records_for_identity:
            owner_parent = load_carrier_owner(owner_sid)
            if owner_parent["core_ids"] != target_parent["core_ids"]:
                raise ValueError("identity alias parent full21 core order differs from target carrier")
            for lane, (owner_binding, target_binding) in enumerate(zip(
                    owner_parent["carrier"]["bindings"], target_parent["carrier"]["bindings"], strict=True)):
                for field in ("translation", "quaternion"):
                    if np.asarray(owner_binding[field], dtype="<f4").tobytes() != np.asarray(target_binding[field], dtype="<f4").tobytes():
                        raise ValueError(f"identity alias full21 {field} differs at lane {lane}")
                if np.float32(owner_binding["scale"]).tobytes() != np.float32(target_binding["scale"]).tobytes():
                    raise ValueError(f"identity alias full21 scale differs at lane {lane}")
            point = owner_parent["points"][owner_slot].copy()
            weights = owner_parent["weights"][owner_slot].copy()
            world = full21_parent_world_positions(
                target_parent["carrier"], poses, point[None, :], weights[None, :],
            )[0]
            resolved.append({
                "owner_sid": owner_sid, "owner_slot": owner_slot, "owner_gid": owner_gid,
                "point": point, "weights": weights, "world": world,
            })
        first = resolved[0]
        for other in resolved[1:]:
            if (other["point"].tobytes() != first["point"].tobytes()
                    or other["weights"].tobytes() != first["weights"].tobytes()
                    or other["world"].tobytes() != first["world"].tobytes()):
                raise ValueError(
                    f"row {target_sid} source identity {source_vertex_id} has conflicting exact owners"
                )
        return first

    def resolve_declared_route_ancestor(target_sid, parent_gids, source_vertex_id=None, force_r1_parent_gid=None):
        # Resolve through the current two-row candidate first, then the pinned
        # 008r1 typed ancestry table. Never infer an owner from geometry.
        candidate_locations = {}
        for owner_sid in (132, 134):
            owner_gids = candidate_gids_by_sid[owner_sid]
            for owner_vertex, owner_gid in enumerate(owner_gids):
                candidate_locations.setdefault(int(owner_gid), []).append({
                    "typed_row": owner_sid, "vertex": int(owner_vertex),
                    "source_slot": int(candidate_slots_by_sid[owner_sid][owner_vertex]),
                    "generation": "current-51fd",
                })

        def material_tuple(location):
            owner_sid = int(location["typed_row"])
            owner_slot = int(location["source_slot"])
            owner_gid = int(location["gid"] if "gid" in location else candidate_gids_by_sid[owner_sid][location["vertex"]])
            if owner_slot >= 0:
                parent = load_carrier_owner(owner_sid)
                point = parent["points"][owner_slot].copy()
                weights = parent["weights"][owner_slot].copy()
            else:
                if location.get("generation") == "pinned-008r1":
                    owner_item = r1_arrays[owner_sid]["lineage"].get(str(owner_gid), {})
                    owner_vertex = r1_arrays[owner_sid]["gid_to_vertex"].get(owner_gid)
                    if owner_vertex is not None:
                        point_value = r1_arrays[owner_sid]["source_points"][owner_vertex]
                        weights_value = r1_arrays[owner_sid]["core_weights"][owner_vertex]
                    else:
                        point_value, weights_value = None, None
                else:
                    owner_item = candidate_lineage_by_sid[owner_sid].get(str(owner_gid), {})
                    point_value, weights_value = owner_item.get("source_point_f32"), owner_item.get("core_weights")
                if point_value is None or weights_value is None:
                    alias_id = owner_item.get("source_vertex_id")
                    if type(alias_id) is not int:
                        raise ValueError(f"duplicate route ancestor GID {owner_gid} lacks exact material tuple")
                    alias = resolve_identity_alias(owner_sid, alias_id)
                    point, weights = alias["point"], alias["weights"]
                else:
                    point = np.asarray(point_value, dtype="<f4")
                    weights = np.asarray(weights_value, dtype=np.float64)
            parent = load_carrier_owner(owner_sid)
            world = full21_parent_world_positions(
                parent["carrier"], poses, point[None, :], weights[None, :],
            )[0]
            return point.tobytes(), weights.tobytes(), world.tobytes()

        def collapse_locations(locations, gid):
            if not locations:
                return []
            if len(locations) == 1:
                result = dict(locations[0])
                result["gid"] = int(gid)
                return [result]
            identities = [material_tuple({**location, "gid": int(gid)}) for location in locations]
            if any(identity != identities[0] for identity in identities[1:]):
                raise ValueError(f"canonical GID {gid} has conflicting typed source/full21 owners")
            # Distinct primitive allocations are expected; canonical material
            # identity and full21 replay are the equality contract.
            result = dict(sorted(locations, key=lambda item: (item["typed_row"], item["source_slot"]))[0])
            result["gid"] = int(gid)
            result["equivalent_typed_owners"] = len(locations)
            return [result]

        seen = set()
        def resolve_gid(gid, trail=(), force_r1=False):
            gid = int(gid)
            if gid in seen:
                raise ValueError(f"recursive declared source ancestry for route GID {gid}")
            seen.add(gid)
            try:
                current_locations = candidate_locations.get(gid, [])
                r1_locations = [
                    {"typed_row": owner["owner_sid"], "vertex": owner["vertex"],
                     "source_slot": owner["source_slot"], "generation": "pinned-008r1"}
                    for owner in r1_gid_owners.get(gid, [])
                ]
                generation = "current-51fd"
                locations = current_locations
                if force_r1 and r1_locations:
                    if current_locations:
                        current_identity = material_tuple({
                            **collapse_locations(current_locations, gid)[0],
                            "gid": gid, "generation": "current-51fd",
                        })
                        r1_identity = material_tuple({
                            **collapse_locations(r1_locations, gid)[0],
                            "gid": gid, "generation": "pinned-008r1",
                        })
                        if current_identity != r1_identity:
                            raise ValueError(
                                f"canonical GID {gid} current and pinned 008r1 source/full21 owners differ"
                            )
                    locations = r1_locations
                    generation = "pinned-008r1"
                elif not locations:
                    locations = r1_locations
                    generation = "pinned-008r1"
                locations = collapse_locations(locations, gid)
                if not locations:
                    raise ValueError(f"declared route ancestor GID {gid} is absent from current and 008r1 ancestry")
                location = locations[0]
                location["generation"] = generation
                owner_sid = int(location["typed_row"])
                slot = int(location["source_slot"])
                path = list(trail) + [{
                    "typed_row": owner_sid, "canonical_material_gid": gid,
                    "generation": generation, "source_slot": slot,
                }]
                if slot >= 0:
                    return {**location, "typed_row": owner_sid, "source_slot": slot,
                            "gid": gid, "lineage_path": path}
                if generation == "current-51fd":
                    lineage_item = candidate_lineage_by_sid[owner_sid].get(str(gid), {})
                else:
                    lineage_item = r1_arrays[owner_sid]["lineage"].get(str(gid), {})
                parents = lineage_item.get("source_parent_gids")
                if isinstance(parents, list) and parents:
                    failures = []
                    for parent_gid in parents:
                        try:
                            return resolve_gid(parent_gid, path)
                        except ValueError as exc:
                            failures.append(str(exc))
                    raise ValueError(
                        f"route birth GID {gid} has no resolvable declared ancestor: {failures[:3]}"
                    )
                alias_id = lineage_item.get("source_vertex_id")
                if type(alias_id) is int and alias_id >= 0:
                    owners = source_identity_owners.get(alias_id, [])
                    valid = [
                        {"typed_row": owner_sid, "vertex": None, "source_slot": owner_slot,
                         "generation": "current-source-identity", "gid": owner_gid}
                        for owner_sid, owner_slot, owner_gid in owners
                    ]
                    if not valid:
                        raise ValueError(f"route identity alias {alias_id} has no current typed source owner")
                    chosen = collapse_locations(valid, gid)[0] if len(valid) > 1 else valid[0]
                    chosen["gid"] = int(chosen.get("gid", gid))
                    chosen["lineage_path"] = path + [{
                        "typed_row": int(chosen["typed_row"]), "canonical_material_gid": int(chosen["gid"]),
                        "generation": "current-source-identity", "source_slot": int(chosen["source_slot"]),
                    }]
                    return chosen
                raise ValueError(f"route birth GID {gid} has neither declared parents nor source identity")
            finally:
                seen.remove(gid)

        ordered_parents = [int(value) for value in (parent_gids or [])]
        if not ordered_parents and type(source_vertex_id) is int and source_vertex_id >= 0:
            owners = source_identity_owners.get(int(source_vertex_id), [])
            if not owners:
                raise ValueError(f"row {target_sid} source identity {source_vertex_id} has no exact current owner")
            routes = [
                {"typed_row": owner_sid, "vertex": None, "source_slot": owner_slot,
                 "generation": "current-source-identity", "gid": owner_gid,
                 "lineage_path": [{"typed_row": owner_sid, "canonical_material_gid": owner_gid,
                                   "generation": "current-source-identity", "source_slot": owner_slot}]}
                for owner_sid, owner_slot, owner_gid in owners
            ]
            if len(routes) > 1:
                for route in routes:
                    material_tuple(route)
                route_keys = []
                for route in routes:
                    source_row = load_source_row(route["typed_row"])
                    idx, weights, _ = remap_child_four_lane_route_by_binding_identity(
                        source_row["bindings"], load_source_row(target_sid)["bindings"],
                        source_row["binding_indices"][route["source_slot"]],
                        source_row["weights"][route["source_slot"]],
                    )
                    route_keys.append((idx.tobytes(), weights.tobytes()))
                if any(key != route_keys[0] for key in route_keys[1:]):
                    raise ValueError(f"row {target_sid} source identity {source_vertex_id} has conflicting auxiliary routes")
            return routes[0]

        if not ordered_parents:
            raise ValueError(f"row {target_sid} birth has no declared source parent for auxiliary route")
        resolved = resolve_gid(ordered_parents[0], force_r1=(force_r1_parent_gid == ordered_parents[0]))
        resolved["selection_policy"] = "first resolvable ancestor in declared source_parent_gids order"
        return resolved

    def emit_auxiliary_route(target_sid, parent_gids, source_vertex_id, target_world, parent_point, force_r1_parent_gid=None):
        route_owner = resolve_declared_route_ancestor(
            target_sid, parent_gids, source_vertex_id, force_r1_parent_gid=force_r1_parent_gid,
        )
        route_source_row = load_source_row(route_owner["typed_row"])
        route_source_slot = int(route_owner["source_slot"])
        source_route_indices = np.asarray(
            route_source_row["binding_indices"][route_source_slot], dtype="<u4",
        )
        source_route_weights = np.asarray(
            route_source_row["weights"][route_source_slot], dtype="<f4",
        )
        child_bi, child_w, route_map_proof = remap_child_four_lane_route_by_binding_identity(
            route_source_row["bindings"], load_source_row(target_sid)["bindings"],
            source_route_indices, source_route_weights,
        )
        normal = np.asarray(route_source_row["normals"][route_source_slot], dtype="<f4")
        child_v6, child_bi, child_w, route_replay = emit_child_with_inferred_four_lane_route(
            load_source_row(target_sid), poses, child_bi, child_w, target_world,
            parent_point, normal,
        )
        return child_v6, child_bi, child_w, {
            "route_ancestor": route_owner,
            "active_binding_identity_remap": route_map_proof,
            "four_lane_replay": route_replay,
            "physical_attachment_owner": "complete ordered full21 extension; legacy ABI4 tuple is an auxiliary source-row carrier",
            "route_weights_source_bytes_hex": route_map_proof["source_route_weights_bytes"],
            "route_weights_output_bytes_hex": child_w.tobytes().hex(),
            "route_weights_byte_exact": child_w.tobytes() == source_route_weights.tobytes(),
            "four_lane_route_support_truncated": False,
        }

    for sid, source_index in ((132, 0), (134, 5)):
        active_stage = f"row_{sid}_crosswalk_full21_births_and_emission"
        prefix = f"row{sid}_"
        row = pc._nhtiss_row_arrays(source, records[sid])
        capture = captures.get((51005, sid))
        if capture is None:
            raise ValueError(f"accepted002 has no typed capture row [51005,{sid}]")
        actual_positions, actual_faces, _actual_normals, actual_gids = capture
        actual_positions = np.asarray(actual_positions, dtype="<f4")
        actual_faces = np.asarray(actual_faces, dtype="<i8")
        actual_gids = np.asarray(actual_gids, dtype="<u8")
        source_faces = np.asarray(row["faces"], dtype="<i8")
        if source_faces.shape != actual_faces.shape:
            raise ValueError(f"row {sid} face allocation differs from exact 1197 mapping contract")
        mapping = {}
        for source_slot, capture_slot in zip(source_faces.reshape(-1), actual_faces.reshape(-1)):
            s, c = int(source_slot), int(capture_slot)
            if s in mapping and mapping[s] != c:
                raise ValueError(f"row {sid} source slot has conflicting 1197 capture mapping")
            mapping[s] = c
        source_slots = np.asarray(sorted(mapping), dtype=np.int64)
        capture_slots = np.asarray([mapping[int(s)] for s in source_slots], dtype=np.int64)
        map_hash = hashlib.sha256(
            np.stack((source_slots, capture_slots), axis=1).astype("<i8").tobytes()
        ).hexdigest()
        expected_hash, expected_count = EXPECTED_MAP[sid]
        if map_hash != expected_hash or len(source_slots) != expected_count:
            raise ValueError(f"row {sid} exact source-slot crosswalk differs from 1197 pin")
        ext_row = next(x for x in extension["surfaces"] if int(x["stable_id"]) == sid)
        original_terms = ext_row["vertices"]
        if len(original_terms) != row["vertex_count"]:
            raise ValueError(f"row {sid} full21 term/source slot count mismatch")
        original_weights = np.asarray([[float(t[3]) for t in terms]
                                        for terms in original_terms], dtype=np.float64)
        if original_weights.shape != (row["vertex_count"], 21):
            raise ValueError(f"row {sid} lacks 21 full material terms")
        if not np.isfinite(original_weights).all() or np.any(original_weights <= 0):
            raise ValueError(f"row {sid} full21 terms are not positive and finite")
        if np.max(np.abs(original_weights.sum(axis=1) - 1.0)) > 1e-12:
            raise ValueError(f"row {sid} full21 terms do not partition to one")

        candidate_slots = np.asarray(arrays[prefix + "source_slots_i64"], dtype=np.int64)
        gids = np.asarray(arrays[prefix + "canonical_material_gids_u64"], dtype="<u8")
        faces = np.asarray(arrays[prefix + "faces_local_u32"], dtype="<u4")
        lineage_text = str(np.asarray(arrays[prefix + "vertex_lineage_json"]).item())
        lineage = json.loads(lineage_text)
        gid_slot_report = validate_source_gid_slot_lineage(candidate_slots, gids, lineage)
        n = len(candidate_slots)
        if gids.shape != (n,) or faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"row {sid} candidate vertex/face arrays are malformed")
        if int(faces.max(initial=0)) >= n or np.any(candidate_slots < -1):
            raise ValueError(f"row {sid} candidate face/source-slot index is invalid")
        if len(set(map(int, gids))) != n:
            raise ValueError(f"row {sid} canonical material GIDs are duplicated")
        existing = candidate_slots >= 0
        if np.any(candidate_slots[existing] >= row["vertex_count"]):
            raise ValueError(f"row {sid} candidate source slot is out of range")
        # The pinned 51fd candidate contributes topology/ancestry only. Existing
        # point coordinates and full21 fields come from current ff8/accepted002.
        full21_weights = np.empty((n, 21), dtype=np.float64)
        full21_weights[existing] = original_weights[candidate_slots[existing]]

        used_vertices = np.unique(faces.reshape(-1))
        used_existing = used_vertices[existing[used_vertices]]
        used_source_slots = np.unique(candidate_slots[used_existing])
        missing = [int(s) for s in used_source_slots if int(s) not in mapping]
        if missing:
            raise ValueError(f"row {sid} referenced source slots lack exact 1197 mapping: {missing[:12]}")
        parent_owner = load_carrier_owner(sid)
        carrier_meta = parent_owner["carrier_meta"]
        carrier_path = parent_owner["carrier_path"]
        carrier = parent_owner["carrier"]
        carrier_points = parent_owner["points"]
        carrier_normals = parent_owner["normals"]
        core_ids = parent_owner["core_ids"]
        if carrier["vertex_count"] != row["vertex_count"] * 21:
            raise ValueError(f"row {sid} full21 carrier source-point cardinality differs")
        if parent_owner["weights"].tobytes() != original_weights.tobytes():
            raise ValueError(f"row {sid} accepted extension weights differ from active carrier owner")
        if carrier["vertex_count"] != len(original_terms) * 21:
            raise ValueError(f"row {sid} full21 term table does not cover original carrier vertices")

        # Directly authenticate actual accepted002 positions through the exact
        # current full21 parent owner before constructing any births.
        actual_parent_points = carrier_points[used_source_slots]
        actual_parent_weights = original_weights[used_source_slots]
        parent_world = full21_parent_world_positions(
            carrier, poses, actual_parent_points, actual_parent_weights,
        )
        target_actual = np.asarray(actual_positions[[mapping[int(s)] for s in used_source_slots]],
                                   dtype="<f4")
        parent_residual = np.max(np.abs(
            parent_world.astype(np.float64) - target_actual.astype(np.float64)), axis=1)
        if float(parent_residual.max(initial=0.0)) > 2e-6:
            raise ValueError(
                f"row {sid} full21 source carrier misses accepted002 capture: "
                f"{float(parent_residual.max()):.9g} m"
            )

        source_index_actual = np.asarray(row["binding_indices"], dtype="<u4")
        child_weights_actual = np.asarray(row["weights"], dtype="<f4")
        nchild = len(candidate_slots)
        vertices6 = np.zeros((nchild, 6), dtype="<f4")
        binding_indices = np.zeros((nchild, 4), dtype="<u4")
        route_weights = np.zeros((nchild, 4), dtype="<f4")
        vertices6[existing, :3] = row["positions"][candidate_slots[existing]]
        vertices6[existing, 3:] = row["normals"][candidate_slots[existing]]
        binding_indices[existing] = source_index_actual[candidate_slots[existing]]
        route_weights[existing] = child_weights_actual[candidate_slots[existing]]
        parent_birth_points, parent_birth_normals, parent_birth_weights = [], [], []
        birth_reports = []
        births = np.flatnonzero(candidate_slots == -1)
        gid_to_vertex = {int(g): i for i, g in enumerate(gids)}
        actual_world_births = {}
        candidate_points = np.asarray(arrays[prefix + "source_points_f32"], dtype="<f4")
        candidate_full21_weights = np.asarray(arrays[prefix + "core_weights_float64"], dtype=np.float64)

        if sid == 132:
            smoke_examples = {}
            smoke_cross_parent_override = {}
            for smoke_vertex in births:
                smoke_gid = int(gids[smoke_vertex])
                smoke_item = lineage.get(str(smoke_gid), {})
                smoke_parents = smoke_item.get("source_parent_gids")
                smoke_fraction = smoke_item.get("edge_fraction_from_lower_canonical_gid")
                smoke_is_edge = (
                    isinstance(smoke_parents, list) and len(smoke_parents) == 2
                    and type(smoke_fraction) in (int, float)
                    and 0.0 <= float(smoke_fraction) <= 1.0
                )
                if smoke_is_edge and "edge" not in smoke_examples:
                    smoke_examples["edge"] = int(smoke_vertex)
                elif (smoke_item.get("source_point_f32") is not None
                      and smoke_item.get("core_weights") is not None):
                    if "explicit_sourcepoint" not in smoke_examples:
                        smoke_examples["explicit_sourcepoint"] = int(smoke_vertex)
                    if "cross_row_ancestry" not in smoke_examples:
                        for cross_parent_gid in (smoke_parents or []):
                            cross_parent_gid = int(cross_parent_gid)
                            if cross_parent_gid in gid_to_vertex:
                                continue
                            if not any(int(owner["owner_sid"]) != sid
                                       for owner in r1_gid_owners.get(cross_parent_gid, [])):
                                continue
                            cross_owner = resolve_declared_route_ancestor(
                                sid, [cross_parent_gid], smoke_item.get("source_vertex_id"),
                                force_r1_parent_gid=cross_parent_gid,
                            )
                            if (int(cross_owner["typed_row"]) != sid
                                    and cross_owner["generation"] == "pinned-008r1"):
                                smoke_examples["cross_row_ancestry"] = int(smoke_vertex)
                                smoke_cross_parent_override[int(smoke_vertex)] = cross_parent_gid
                                break
                elif (smoke_item.get("source_face_support_provenance") == "original-keyed-material-handoff-GID"
                      and type(smoke_item.get("source_vertex_id")) is int):
                    if "alias" not in smoke_examples:
                        smoke_examples["alias"] = int(smoke_vertex)
                if len(smoke_examples) == 4:
                    break
            if set(smoke_examples) != {"edge", "explicit_sourcepoint", "alias", "cross_row_ancestry"}:
                raise ValueError(f"real route smoke lacks required birth kinds: {sorted(smoke_examples)}")
            smoke_report = {}
            for smoke_kind, smoke_vertex in smoke_examples.items():
                smoke_gid = int(gids[smoke_vertex])
                smoke_item = lineage[str(smoke_gid)]
                smoke_parents = smoke_item.get("source_parent_gids")
                smoke_fraction = smoke_item.get("edge_fraction_from_lower_canonical_gid")
                if smoke_kind == "edge":
                    smoke_slots = resolve_birth_source_parent_slots(smoke_parents, gids, candidate_slots)
                    smoke_t = float(smoke_fraction)
                    smoke_endpoint_world = full21_parent_world_positions(
                        carrier, poses, carrier_points[smoke_slots],
                        original_weights[smoke_slots],
                    )
                    smoke_target = np.asarray(
                        smoke_endpoint_world[0].astype(np.float64) * (1.0-smoke_t)
                        + smoke_endpoint_world[1].astype(np.float64) * smoke_t,
                        dtype="<f4",
                    )
                    smoke_weights = np.asarray(smoke_item["core_weights"], dtype=np.float64)
                    smoke_point_batch, _ = solve_full21_parent_source_points(
                        carrier, poses, smoke_target[None, :], smoke_weights[None, :],
                    )
                    smoke_point = smoke_point_batch[0]
                elif (smoke_item.get("source_point_f32") is not None
                      and smoke_item.get("core_weights") is not None):
                    smoke_point = np.asarray(smoke_item["source_point_f32"], dtype="<f4")
                    smoke_weights = np.asarray(smoke_item["core_weights"], dtype=np.float64)
                    smoke_target = full21_parent_world_positions(
                        carrier, poses, smoke_point[None, :], smoke_weights[None, :],
                    )[0]
                else:
                    smoke_alias = resolve_identity_alias(sid, smoke_item["source_vertex_id"])
                    smoke_point, smoke_weights = smoke_alias["point"], smoke_alias["weights"]
                    smoke_target = smoke_alias["world"]
                smoke_route_parents = list(smoke_parents or [])
                if smoke_kind == "cross_row_ancestry":
                    first_cross = smoke_cross_parent_override[int(smoke_vertex)]
                    smoke_route_parents = [first_cross] + [
                        int(parent_gid) for parent_gid in smoke_route_parents
                        if int(parent_gid) != first_cross
                    ]
                _, _, _, smoke_route = emit_auxiliary_route(
                    sid, smoke_route_parents, smoke_item.get("source_vertex_id"),
                    smoke_target, smoke_point,
                    force_r1_parent_gid=(
                        smoke_cross_parent_override.get(int(smoke_vertex))
                        if smoke_kind == "cross_row_ancestry" else None
                    ),
                )
                smoke_report[smoke_kind] = {
                    "canonical_material_gid": smoke_gid,
                    "selected_declared_parent_override": (
                        smoke_cross_parent_override.get(int(smoke_vertex))
                        if smoke_kind == "cross_row_ancestry" else None
                    ),
                    "route_ancestor": smoke_route["route_ancestor"],
                    "legacy_route_weights_byte_exact": smoke_route["route_weights_byte_exact"],
                    "full21_terms_count": int(len(smoke_weights)),
                    "max_four_lane_replay_residual_m": smoke_route["four_lane_replay"]["max_four_lane_replay_residual_m"],
                    "candidate_absolute_world_coordinates_used": False,
                }
            report["real_birth_route_smoke"] = smoke_report
            report["stages"].append({"stage": "real_edge_explicit_alias_crossrow_route_smoke", "passed": True})
            save_progress(report)
            if SMOKE_ONLY:
                signal.alarm(0)
                report["status"] = "PASS_REAL_BIRTH_ROUTE_SMOKE_ONLY_NOT_ANATOMY_ADMISSION"
                save_progress(report)
                sys.exit(0)

        for vertex in births:
            gid = int(gids[vertex])
            item = lineage.get(str(gid))
            if not isinstance(item, dict):
                raise ValueError(f"row {sid} birth GID {gid} has no authenticated vertex lineage")
            kind = str(item.get("source_face_support_provenance", ""))
            parent_gids = item.get("source_parent_gids")
            edge_fraction = item.get("edge_fraction_from_lower_canonical_gid")
            is_edge_birth = (
                isinstance(parent_gids, list) and len(parent_gids) == 2
                and type(edge_fraction) in (int, float)
                and np.isfinite(float(edge_fraction))
                and 0.0 <= float(edge_fraction) <= 1.0
            )
            alias_resolution = None
            if is_edge_birth:
                if parent_gids[0] > parent_gids[1]:
                    raise ValueError(f"row {sid} edge birth GID {gid} parent order is not canonical")
                endpoint_slots = resolve_birth_source_parent_slots(parent_gids, gids, candidate_slots)
                s0, s1 = map(int, endpoint_slots)
                if s0 not in mapping or s1 not in mapping:
                    raise ValueError(f"row {sid} edge birth GID {gid} endpoint lacks exact accepted002 slot crosswalk")
                t = float(edge_fraction)
                endpoints = np.asarray([carrier_points[s0], carrier_points[s1]], dtype="<f4")
                endpoint_weights = original_weights[np.asarray([s0, s1], dtype=np.int64)]
                endpoint_world = full21_parent_world_positions(
                    carrier, poses, endpoints, endpoint_weights,
                )
                target_world = np.asarray(
                    endpoint_world[0].astype(np.float64) * (1.0 - t)
                    + endpoint_world[1].astype(np.float64) * t,
                    dtype="<f4",
                )
                lineage_weights = np.asarray(item.get("core_weights"), dtype=np.float64)
                if lineage_weights.shape != (21,):
                    raise ValueError(f"row {sid} edge birth GID {gid} lacks exact full21 edge field")
                expected_weights = endpoint_weights[0] * (1.0 - t) + endpoint_weights[1] * t
                expected_weights /= float(expected_weights.sum())
                weight_diff = float(np.max(np.abs(lineage_weights - expected_weights)))
                if weight_diff > 1e-12:
                    raise ValueError(
                        f"row {sid} edge birth GID {gid} full21 lineage differs from endpoint interpolation "
                        f"by {weight_diff:.9g}"
                    )
                body_ids = item.get("core_body_indices")
                if body_ids != list(core_ids):
                    raise ValueError(f"row {sid} edge birth GID {gid} full21 body order differs from active carrier")
                parent_point_batch, inverse_report = solve_full21_parent_source_points(
                    carrier, poses, target_world[None, :], lineage_weights[None, :],
                )
                parent_point = parent_point_batch[0]
                birth_weights = lineage_weights
                emitted_world = full21_parent_world_positions(
                    carrier, poses, parent_point[None, :], birth_weights[None, :],
                )[0]
                residual = float(np.max(np.abs(
                    emitted_world.astype(np.float64) - target_world.astype(np.float64),
                )))
                if residual > 2e-7:
                    raise ValueError(f"row {sid} edge birth GID {gid} full21 replay residual {residual:.9g} m")
                child_v6, child_bi, child_w, route_report = emit_auxiliary_route(
                    sid, parent_gids, item.get("source_vertex_id"), target_world, parent_point,
                )
                normal = child_v6[3:].copy()
                route_report["kind"] = "declared-inferred-legacy-ABI4-route-for-edge-birth"
                route_report["edge_parent_gids"] = [int(value) for value in parent_gids]
                route_report["edge_fraction"] = t
            else:
                source_point_value = item.get("source_point_f32")
                full21_value = item.get("core_weights")
                alias_resolution = None
                if source_point_value is not None and full21_value is not None:
                    parent_point = np.asarray(source_point_value, dtype="<f4")
                    birth_weights = np.asarray(full21_value, dtype=np.float64)
                    body_ids = item.get("core_body_indices")
                    if body_ids != list(core_ids):
                        raise ValueError(f"row {sid} birth GID {gid} full21 body order differs from active carrier")
                    if (parent_point.shape != (3,) or not np.isfinite(parent_point).all()
                            or birth_weights.shape != (21,) or not np.isfinite(birth_weights).all()
                            or np.any(birth_weights <= 0)
                            or abs(float(birth_weights.sum()) - 1.0) > 1e-12):
                        raise ValueError(f"row {sid} birth GID {gid} explicit full21 source record is malformed")
                elif (kind == "original-keyed-material-handoff-GID"
                      and type(item.get("source_vertex_id")) is int):
                    alias_resolution = resolve_identity_alias(sid, item["source_vertex_id"])
                    parent_point = alias_resolution["point"].copy()
                    birth_weights = alias_resolution["weights"].copy()
                else:
                    raise ValueError(f"row {sid} birth GID {gid} lacks explicit current full21 point/field ancestry")
                target_world = full21_parent_world_positions(
                    carrier, poses, parent_point[None, :], birth_weights[None, :],
                )[0]
                # The legacy ABI4 tuple is auxiliary only. Select one
                # authenticated source ancestor from declared typed lineage;
                # remap its route by exact Core+registered-transform identity
                # and solve only its child source point to the authoritative
                # full21 world point at accepted002.
                child_v6, child_bi, child_w, route_report = emit_auxiliary_route(
                    sid, parent_gids, item.get("source_vertex_id"), target_world, parent_point,
                )
                normal = child_v6[3:].copy()
                route_report["kind"] = "declared-inferred-legacy-ABI4-carrier-route"
            candidate_point_match = candidate_points[vertex].tobytes() == parent_point.tobytes()
            candidate_weight_match = candidate_full21_weights[vertex].tobytes() == birth_weights.tobytes()
            if not np.isfinite(target_world).all() or not np.isfinite(parent_point).all():
                raise ValueError(f"row {sid} birth GID {gid} emitted nonfinite accepted-frame source data")
            vertices6[vertex] = child_v6
            binding_indices[vertex] = child_bi
            route_weights[vertex] = child_w
            parent_birth_points.append(parent_point)
            parent_birth_normals.append(normal)
            parent_birth_weights.append(birth_weights)
            full21_weights[vertex] = birth_weights
            birth_reports.append({
                "candidate_vertex": int(vertex), "canonical_material_gid": gid,
                "source_face_support_provenance": kind,
                "source_parent_gids": parent_gids,
                "edge_fraction_from_lower_canonical_gid": float(edge_fraction) if is_edge_birth else None,
                "source_vertex_identity_alias": item.get("source_vertex_id") if alias_resolution is not None else None,
                "source_identity_owner": ({
                    "typed_row": alias_resolution["owner_sid"],
                    "source_slot": alias_resolution["owner_slot"],
                    "canonical_material_gid": alias_resolution["owner_gid"],
                } if alias_resolution is not None else None),
                "candidate_point_array_exact_crosscheck_only": candidate_point_match,
                "candidate_full21_array_exact_crosscheck_only": candidate_weight_match,
                "full21_source_point_f32": parent_point.astype(float).tolist(),
                "full21_source_weights_float64": birth_weights.tolist(),
                "actual_current972_full21_world_f32": target_world.astype(float).tolist(),
                "edge_full21_inverse": inverse_report if is_edge_birth else None,
                "legacy_route": route_report,
                "candidate_absolute_world_positions_used": False,
                "candidate_source_point_and_weight_arrays_used_for_output": False,
                "candidate_source_point_and_weight_arrays_used_only_for_exact_crosscheck": True,
                "lineage_source_point_and_full21_field_used": not is_edge_birth and alias_resolution is None,
            })
            actual_world_births[gid] = target_world.astype("<f4")
        # Shared canonical cut points must have the same full21 material field
        # and actual accepted-world coordinate from both muscles.
        for vertex in births:
            gid = int(gids[vertex])
            if gid in all_birth_world_by_gid:
                prior = all_birth_world_by_gid[gid]
                if prior["world"].tobytes() != actual_world_births[gid].tobytes():
                    raise ValueError(f"shared birth GID {gid} has different accepted-world coordinates")
                prior_weights = prior["weights"]
                if prior_weights.tobytes() != full21_weights[vertex].tobytes():
                    raise ValueError(f"shared birth GID {gid} has different full21 material fields")
                prior["rows"].add(sid)
            else:
                all_birth_world_by_gid[gid] = {
                    "world": actual_world_births[gid], "weights": full21_weights[vertex].copy(),
                    "first_row": sid, "rows": {sid},
                }

        # Validate exact parent carrier replay for both old source slots and new points.
        all_parent_points = np.empty((nchild, 3), dtype="<f4")
        all_parent_points[existing] = carrier_points[candidate_slots[existing]]
        all_parent_points[births] = np.asarray(parent_birth_points, dtype="<f4")
        full21_world = full21_parent_world_positions(
            carrier, poses, all_parent_points, full21_weights,
        )
        child_row = dict(row)
        child_row.update(
            vertex_count=nchild, binding_indices=binding_indices, weights=route_weights,
        )
        child_world = pc._native_blended_rest_positions(
            child_row, vertices6[:, :3], poses,
        )
        route_residual = np.max(
            np.abs(child_world.astype(np.float64) - full21_world.astype(np.float64)), axis=1,
        )
        route_max = float(route_residual.max(initial=0.0))
        if route_max > 2e-6:
            raise ValueError(f"row {sid} child four-lane/full21 pose mismatch {route_max:.9g} m")
        current_world_rows[sid] = {
            "positions_world_f32": full21_world,
            "faces_local_u32": faces,
            "canonical_material_gids_u64": gids,
            "source_slots_i64": candidate_slots,
            "core_weights_float64": full21_weights,
        }
        patch_path = OUT / "patches" / f"row{sid}-full21-source-emission.npz"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            patch_path, vertices6=vertices6, binding_indices=binding_indices,
            weights=route_weights, faces=faces,
        )
        proof_path = OUT / "patches" / f"row{sid}-source-ancestry.json"
        proof_path.write_text(json.dumps({
            "schema": "numi.human.forearm-actual-current972-source-birth-ancestry.v1",
            "typed_identity": [51005, sid],
            "source_payload_sha256": BASE_SHA,
            "accepted_capture_step_receipt_sha256": STEP_SHA,
            "source_to_capture_slot_map_sha256": map_hash,
            "source_to_capture_mapped_slots": len(source_slots),
            "candidate_sha256": CAND_SHA,
            "new_birth_count": len(birth_reports),
            "all_original_source_positions_and_21_terms_preserved": True,
            "all_births_have_edge_lineage_and_positive_ordered_21_terms": True,
            "new_parent_source_point_lane_count_per_birth": 21,
            "births": birth_reports,
            "source_ancestry": "Original vertices retain source slots and source faces remain as supplied by pinned 51fd candidate. New vertices use the candidate face-edge GID lineage, actual accepted002 parent replay, and full21 endpoint interpolation.",
            "anatomical_admission": False, "native_admission": False,
        }, indent=2, sort_keys=True) + "\n")
        terms, term_births, term_report = remap_full21_terms_by_source_slot(
            source_index=source_index, stable_id=sid,
            source_slots_i64=candidate_slots,
            core_weights_float64=full21_weights,
            original_terms=original_terms,
            first_new_parent_vertex=len(original_terms) * 21,
        )
        if term_births != births.astype(int).tolist():
            raise ValueError(f"row {sid} full21 remap birth order differs from geometry")
        terms_report = {
            "full21_term_mapping": term_report,
            "all_existing_source_slot_terms_byte_exact": True,
            "new_parent_vertex_first": len(original_terms) * 21,
            "new_parent_vertex_count": 21 * len(term_births),
        }
        pending_terms[sid] = {
            "source_index": source_index, "carrier_path": carrier_path,
            "carrier_meta": carrier_meta, "carrier": carrier, "core_ids": core_ids,
            "parent_birth_points": np.asarray(parent_birth_points, dtype="<f4").reshape(-1, 3),
            "parent_birth_normals": np.asarray(parent_birth_normals, dtype="<f4").reshape(-1, 3),
            "parent_birth_weights": np.asarray(parent_birth_weights, dtype=np.float64).reshape(-1, 21),
            "terms": terms, "original_terms": original_terms, "term_report": terms_report,
            "map_sha": map_hash, "birth_count": len(births),
        }
        candidate_rows[sid] = {
            "patch_path": patch_path, "proof_path": proof_path, "fields": {
                "vertices6": vertices6, "binding_indices": binding_indices,
                "weights": route_weights, "faces": faces,
            },
            "route_max_residual_m": route_max, "full21_world": full21_world,
            "faces": faces, "gids": gids, "source_slots": candidate_slots,
            "core_weights": full21_weights,
        }
        report.setdefault("rows", {})[str(sid)] = {
            "source_vertex_count": row["vertex_count"],
            "candidate_vertex_count": nchild, "candidate_face_count": len(faces),
            "source_to_capture_map_sha256": map_hash, "source_to_capture_map_count": len(source_slots),
            "canonical_gid_source_slot_lineage": gid_slot_report,
            "pack_global_vertex_ids_used_for_identity": False,
            "full21_existing_source_to_accepted_capture_max_abs_residual_m": float(parent_residual.max(initial=0.0)),
            "new_birth_count": int(len(births)),
            "child_four_lane_to_actual_full21_max_abs_residual_m": route_max,
            "full21_birth_parent_vertex_groups": 21 * int(len(births)),
            "births": birth_reports,
            "full21_term_mapping": terms_report,
            "static_candidate_world_positions_used": False,
            "candidate_absolute_source_and_full21_weight_arrays_used": False,
        }
        report["stages"].append({"stage": f"row_{sid}_crosswalk_full21_births_and_emission",
                                 "passed": True, "birth_count": int(len(births)),
                                 "map_sha256": map_hash, "route_residual_m": route_max})
        save_progress(report)

    active_stage = "shared_material_birth_identity"
    # Canonical full21 points shared by rows 132 and 134 must have identical owner fields.
    expected_shared_births = set(map(int, arrays.get("new_shared_cut_gids_u64", np.asarray([], dtype="<u8"))))
    if expected_shared_births and not expected_shared_births.issubset(all_birth_world_by_gid):
        raise ValueError("candidate has shared cut births absent from one or both typed row emissions")
    for gid in expected_shared_births:
        if all_birth_world_by_gid[gid]["rows"] != {132, 134}:
            raise ValueError(f"shared cut GID {gid} is not emitted by both typed rows")
    report["shared_birth_count"] = len(expected_shared_births)
    report["shared_births_material_identity"] = "exact actual-world Float32 and exact full21 Float64 field equality across rows"
    save_progress(report)

    active_stage = "append_full21_parent_carrier_points"
    # Append derived carrier groups, preserving each full21 parent input prefix.
    (OUT / "parents").mkdir()
    for sid in (132, 134):
        pending = pending_terms[sid]
        raw = pending["carrier_path"].read_bytes()
        original_hash = pending["carrier_meta"]["sha256"]
        if hashlib.sha256(raw).hexdigest() != original_hash:
            raise ValueError(f"row {sid} parent carrier changed during build")
        points6 = np.concatenate((
            pending["parent_birth_points"], pending["parent_birth_normals"],
        ), axis=1).astype("<f4")
        if len(points6) == 0:
            raise ValueError(f"row {sid} has no inferred source points to append")
        extended, first, append_report = append_full21_carrier_points(
            raw, stable_id=sid, core_body_indices=pending["core_ids"],
            source_points6=points6,
        )
        expected_first = len(pending["original_terms"]) * 21
        if first != expected_first:
            raise ValueError(f"row {sid} carrier append starts at {first}, expected {expected_first}")
        carrier_out = OUT / "parents" / f"parent-{pending['source_index']}.nhtissue"
        carrier_out.write_bytes(extended)
        pending["carrier_output_path"] = carrier_out
        pending["carrier_output_sha256"] = sha(carrier_out)
        pending["append_report"] = append_report

    active_stage = "compose_all_current_base_source_rows"
    # Compose the six current-base index repairs with the actual-frame forearm
    # source-emitted rows in one append-only NHTISS candidate.
    replacements = [(sid, path, proof) for sid, (path, proof) in PATCHES.items()]
    for sid in (132, 134):
        row_out = candidate_rows[sid]
        replacements.append((sid, row_out["patch_path"], row_out["proof_path"]))
    replacements.sort(key=lambda item: item[0])
    compose_out = OUT / "composed-child"
    compose_report = pc.compose(
        CHILD, compose_out, replacements,
        reference_surface_rows=tuple(item[0] for item in replacements),
    )
    composed_child = compose_out / CHILD.name
    composed_hash = sha(composed_child)
    if composed_hash != compose_report["payload_sha256"]:
        raise ValueError("composed child SHA differs from composer proof")

    active_stage = "rebind_full21_extension"
    # Rebind the exact current002 extension to the composed payload and append
    # only new source-carrier vertices for rows132/134.
    updated_extension = copy.deepcopy(extension)
    for sid in (132, 134):
        pending = pending_terms[sid]
        updated_extension = replace_parent_surface_terms(
            updated_extension, stable_id=sid, source_index=pending["source_index"],
            expected_child_payload_sha256=updated_extension["child_payload_sha256"], child_payload_sha256=composed_hash,
            expected_source_payload_sha256=pending["carrier_meta"]["sha256"],
            source_payload_path=str(pending["carrier_output_path"].resolve()),
            source_payload_sha256=pending["carrier_output_sha256"],
            expected_existing_term_count=len(pending["original_terms"]),
            replacement_terms=pending["terms"],
        )
    if updated_extension.get("child_payload_sha256") != composed_hash:
        raise ValueError("updated extension is not bound to the composed child")
    extension_path = OUT / "passive-tissue-parent-bindings.json"
    extension_path.write_text(json.dumps(updated_extension, indent=2, sort_keys=True) + "\n")

    active_stage = "reopen_full21_and_child_route_verification"
    # Reopen every emitted payload and verify all full21 and route ownership.
    check_source = pc._read_nhtiss4(composed_child)
    check_records = {int(r[6]): r for r in check_source["records"]}
    for sid in (132, 134):
        actual_row = pc._nhtiss_row_arrays(check_source, check_records[sid])
        expected_fields = candidate_rows[sid]["fields"]
        if actual_row["positions"].tobytes() != expected_fields["vertices6"][:, :3].tobytes():
            raise ValueError(f"row {sid} composed child points differ from emitted patch")
        if actual_row["binding_indices"].tobytes() != expected_fields["binding_indices"].tobytes():
            raise ValueError(f"row {sid} composed route bindings differ from emitted patch")
        if actual_row["weights"].tobytes() != expected_fields["weights"].tobytes():
            raise ValueError(f"row {sid} composed route weights differ from emitted patch")
        if not np.array_equal(actual_row["faces"], expected_fields["faces"]):
            raise ValueError(f"row {sid} composed faces differ from candidate ancestry")
        row_extension = next(x for x in updated_extension["surfaces"] if int(x["stable_id"]) == sid)
        if len(row_extension["vertices"]) != actual_row["vertex_count"]:
            raise ValueError(f"row {sid} updated extension does not cover every composed vertex")
        if not all(len(t) == 21 and all(float(x[3]) > 0 for x in t)
                   for t in row_extension["vertices"]):
            raise ValueError(f"row {sid} composed extension lost positive full21 lanes")
        parent_out = Path(updated_extension["source_payloads"][pending_terms[sid]["source_index"]]["path"])
        if sha(parent_out) != updated_extension["source_payloads"][pending_terms[sid]["source_index"]]["sha256"]:
            raise ValueError(f"row {sid} appended parent source hash mismatch")

    world_path = OUT / "forearm-current972-world-rows.npz"
    np.savez_compressed(
        world_path,
        row132_positions_world_f32=candidate_rows[132]["full21_world"],
        row132_faces_local_u32=candidate_rows[132]["faces"],
        row132_canonical_material_gids_u64=candidate_rows[132]["gids"],
        row132_source_slots_i64=candidate_rows[132]["source_slots"],
        row132_core_weights_float64=candidate_rows[132]["core_weights"],
        row134_positions_world_f32=candidate_rows[134]["full21_world"],
        row134_faces_local_u32=candidate_rows[134]["faces"],
        row134_canonical_material_gids_u64=candidate_rows[134]["gids"],
        row134_source_slots_i64=candidate_rows[134]["source_slots"],
        row134_core_weights_float64=candidate_rows[134]["core_weights"],
    )
    # The historical patches and all exact source inputs remain unchanged.
    if any(sha(path) != digest for path, digest in input_pins.items()):
        raise ValueError("one or more immutable source inputs changed during emission")
    report.update({
        "status": "SOURCE_EMISSION_READY_FOR_SHARED_COLLECTOR",
        "source_emission_only": True,
        "composed_child_path": str(composed_child),
        "composed_child_sha256": composed_hash,
        "composed_child_manifest_path": str(compose_out / CHILD.with_suffix(".manifest.json").name),
        "composer_report_path": str(compose_out / "report.json"),
        "composer_report_sha256": sha(compose_out / "report.json"),
        "parent_bindings_extension_path": str(extension_path),
        "parent_bindings_extension_sha256": sha(extension_path),
        "parent_carrier_outputs": {
            str(sid): {
                "path": str(pending_terms[sid]["carrier_output_path"]),
                "sha256": pending_terms[sid]["carrier_output_sha256"],
                "appended_source_points": pending_terms[sid]["append_report"]["appended_source_point_count"],
                "appended_parent_lane_count": pending_terms[sid]["append_report"]["appended_parent_lane_count"],
                "original_carrier_prefix_byte_exact": pending_terms[sid]["append_report"]["old_header_record_binding_vertex_prefix_and_indices_byte_exact"],
            } for sid in (132, 134)
        },
        "world_rows_path": str(world_path),
        "world_rows_sha256": sha(world_path),
        "immutable_inputs_unchanged": True,
        "integrated_index_patch_rows": [86, 104, 105, 106, 153, 158],
        "anatomical_admission": False,
        "native_admission": False,
    })
    report["stages"].append({"stage": "combined_child_parent_extension_and_reopen_verification",
                             "passed": True})
    report["elapsed_seconds"] = time.monotonic() - started
    save_progress(report)
    print(json.dumps({
        "status": report["status"], "output": str(OUT),
        "composed_child_sha256": composed_hash,
        "world_rows_sha256": report["world_rows_sha256"],
        "rows": {sid: report["rows"][str(sid)]["child_four_lane_to_actual_full21_max_abs_residual_m"]
                 for sid in (132, 134)},
    }, sort_keys=True), flush=True)
except SystemExit as exc:
    if SMOKE_ONLY and exc.code == 0 and report.get("stages", [])[-1].get("stage") == "real_edge_explicit_alias_crossrow_route_smoke":
        report["status"] = "PASS_REAL_BIRTH_ROUTE_SMOKE_ONLY_NOT_ANATOMY_ADMISSION"
        report["elapsed_seconds"] = time.monotonic() - started
        save_progress(report)
        print(json.dumps({
            "status": report["status"], "output": str(OUT),
            "real_birth_route_smoke": report.get("real_birth_route_smoke"),
        }, sort_keys=True), flush=True)
    else:
        raise
except BaseException as exc:
    report["status"] = "FAILED_SOURCE_EMISSION"
    report["failed_stage"] = active_stage
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
    report["elapsed_seconds"] = time.monotonic() - started
    save_progress(report)
    print(json.dumps({"status": report["status"], "failed_stage": report["failed_stage"],
                      "error": report["error"], "output": str(OUT)}, sort_keys=True),
          file=sys.stderr, flush=True)
    raise
finally:
    signal.alarm(0)
