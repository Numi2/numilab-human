"""Rebind authenticated aortic source rows to one exact current capture.

This module only joins source lineage to supplied capture arrays. It does not
compute contact, solid, calibre, motion, or anatomy-admission predicates.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np


_HANDOFF_SCHEMA = "numi.human.verified-aortic-arch-interface-handoff.v1"
_HANDOFF_STATUS = "VERIFIED_SOURCE_BOUND_AORTIC_ARCH_INTERFACE"
_ANCESTRY_STATUS = "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED"


def _sha256(value, label):
    if not isinstance(value, bytes):
        raise TypeError(label + " must be immutable bytes")
    return hashlib.sha256(value).hexdigest()


def _valid_sha256(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)):
        raise ValueError(label + " must be a lowercase SHA-256 digest")
    return value


def _array_sha(array, dtype):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def rebind_aortic_source_handoff_to_current_capture(
        pinned_handoff_bytes, *, expected_pinned_handoff_sha256,
        current_source_preflight_bytes, capture_input_bundle_sha256,
        capture_array_mapping, authenticated_source_owner_pins,
        parent_source_handoff_path=None):
    """Bind a pinned source handoff to exact rows from one current capture.

    capture_array_mapping maps each typed identity tuple to positions, faces,
    normals, and source_vertex_ids arrays. The source preflight is independently
    pinned to the same capture bundle and must describe their exact arrays.

    Authenticated source-owner pins must be resolved independently from the
    pinned source member and NHA row. This function returns a pending handoff:
    it never carries prior candidate geometry gates onto changed arrays.
    """
    parent_sha = _sha256(pinned_handoff_bytes, "pinned source handoff")
    if _valid_sha256(expected_pinned_handoff_sha256, "expected handoff SHA-256") != parent_sha:
        raise ValueError("pinned source handoff bytes differ from their SHA-256 pin")
    preflight_sha = _sha256(current_source_preflight_bytes, "current source preflight")
    capture_sha = _valid_sha256(capture_input_bundle_sha256, "capture input bundle SHA-256")
    if parent_source_handoff_path is not None and (
            not isinstance(parent_source_handoff_path, str) or not parent_source_handoff_path):
        raise ValueError("parent source handoff path must be nonempty")

    try:
        parent = json.loads(pinned_handoff_bytes)
        preflight = json.loads(current_source_preflight_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("handoff or preflight is not UTF-8 JSON") from exc
    if (not isinstance(parent, dict) or parent.get("schema") != _HANDOFF_SCHEMA
            or parent.get("status") != _HANDOFF_STATUS or parent.get("passed") is not True):
        raise ValueError("pinned source handoff is not an authenticated passing handoff")
    if (not isinstance(preflight, dict)
            or preflight.get("capture_input_bundle_sha256") != capture_sha
            or not isinstance(preflight.get("source_rows"), dict)):
        raise ValueError("current preflight is not pinned to the supplied capture bundle")

    link = parent.get("typed_link")
    if (not isinstance(link, list) or len(link) != 2
            or any(not isinstance(k, list) or len(k) != 2
                   or any(type(x) is not int or x <= 0 for x in k) for k in link)
            or any(k[0] != 51011 for k in link) or link[0][1] == link[1][1]):
        raise ValueError("pinned handoff must name two distinct typed aortic rows")
    identities = [tuple(k) for k in link]
    row_keys = [f"{k[0]}:{k[1]}" for k in identities]
    if (not isinstance(parent.get("source_rows"), dict)
            or set(parent["source_rows"]) != set(row_keys)
            or not isinstance(parent.get("candidate_surface_bindings"), dict)
            or set(parent["candidate_surface_bindings"]) != set(row_keys)):
        raise ValueError("pinned source rows do not exactly match its typed link")
    if (not isinstance(capture_array_mapping, Mapping)
            or set(capture_array_mapping) != set(identities)):
        raise ValueError("capture arrays must exactly cover the pinned typed link")
    if (not isinstance(authenticated_source_owner_pins, Mapping)
            or set(authenticated_source_owner_pins) != set(identities)):
        raise ValueError("authenticated source-owner pins must exactly cover the typed link")
    for gate in ("source_topology_passed", "interface_gate_passed", "solid_gate_passed",
                 "independent_source_collar_calibre_passed",
                 "candidate_pairs_equal_frozen_source_pairs",
                 "baseline_pairs_equal_frozen_source_pairs",
                 "candidate_baseline_exact_pairsets_equal"):
        if parent.get(gate) is not True:
            raise ValueError("pinned handoff lacks independent source gate: " + gate)

    pairs = parent.get("exact_face_pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pinned source handoff has no frozen face-pair ancestry")
    for pair in pairs:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(face) is not int or face < 0 for face in pair)):
            raise ValueError("pinned source handoff has malformed face-pair ancestry")
    if len({tuple(pair) for pair in pairs}) != len(pairs):
        raise ValueError("pinned source handoff repeats a face pair")
    pair_sha = hashlib.sha256(
        np.ascontiguousarray(np.asarray(pairs, dtype="<u4")).tobytes()).hexdigest()
    if parent.get("exact_face_pairset_sha256") != pair_sha:
        raise ValueError("pinned source handoff pairset hash is invalid")

    rows, bindings = {}, {}
    for key, row_key in zip(identities, row_keys):
        row = preflight["source_rows"].get(row_key)
        arrays = capture_array_mapping[key]
        source = parent["source_rows"][row_key]
        owner_pin = authenticated_source_owner_pins[key]
        if (not isinstance(row, dict) or not isinstance(arrays, Mapping)
                or not isinstance(owner_pin, Mapping)):

            raise ValueError("current source row or array mapping is malformed: " + row_key)
        required = ("positions", "faces", "normals", "source_vertex_ids")
        if any(name not in arrays for name in required):
            raise ValueError("current capture arrays are incomplete: " + row_key)
        owner = row.get("owner")
        ancestry = owner.get("capture_face_vertex_ancestry") if isinstance(owner, dict) else None
        pin_sha = {
            name: _valid_sha256(owner_pin.get(name), "authenticated " + name)
            for name in ("source_member_sha256", "source_nha_row_sha256")
        }
        if not (
                owner_pin.get("typed_identity") == list(key)
                and owner_pin.get("source_member") == source.get("source_member")
                and owner_pin.get("fma") == source.get("fma")
                and owner_pin.get("body_index") == source.get("body_index") == 20
                and owner_pin.get("layer") == source.get("layer") == 2
                and owner_pin.get("face_count") == source.get("face_count")
                and owner_pin.get("vertex_count") == source.get("vertex_count")
                and owner_pin.get("faces_sha256") == source.get("faces_sha256")
                and row.get("typed_identity") == list(key)
                and row.get("body_index") == source.get("body_index") == 20
                and row.get("layer") == source.get("layer") == 2
                and row.get("source_member") == source.get("source_member")
                and row.get("fma") == source.get("fma")
                and row.get("name") == source.get("name")
                and row.get("source_faces_count") == source.get("face_count")
                and row.get("source_vertices_count") == source.get("vertex_count")
                and row.get("source_faces_sha256") == source.get("faces_sha256")
                and row.get("capture_faces_u32_sha256") == row.get("source_faces_sha256")
                and isinstance(owner, dict)
                and owner.get("typed_identity") == list(key)
                and owner.get("source_member") == row.get("source_member")
                and owner.get("source_concept_id") == row.get("fma")
                and owner.get("source_member_sha256") == pin_sha["source_member_sha256"]
                and owner.get("source_nha_row_sha256") == pin_sha["source_nha_row_sha256"]
                and owner.get("capture_face_order_equals_nha") is True
                and isinstance(ancestry, dict)
                and ancestry.get("status") == _ANCESTRY_STATUS
                and ancestry.get("capture_face_count") == row.get("source_faces_count")
                and ancestry.get("source_face_count") == row.get("source_faces_count")
                and ancestry.get("capture_faces_u32_sha256") == row.get("capture_faces_u32_sha256")
                and ancestry.get("source_faces_u32_sha256") == row.get("source_faces_sha256")
                and owner.get("source_face_count") == row.get("source_faces_count")
                and owner.get("source_vertex_count") == row.get("source_vertices_count")):
            raise ValueError("current source row differs from owner or ordered-face ancestry: " + row_key)

        positions = np.asarray(arrays["positions"], dtype="<f4")
        faces = np.asarray(arrays["faces"], dtype="<u4")
        normals = np.asarray(arrays["normals"], dtype="<f4")
        gids = np.asarray(arrays["source_vertex_ids"], dtype="<u8")
        if (positions.ndim != 2 or positions.shape[1] != 3 or normals.shape != positions.shape
                or faces.ndim != 2 or faces.shape[1] != 3
                or gids.ndim != 1 or len(gids) != len(positions)
                or len(positions) != row.get("source_vertices_count")
                or len(faces) != row.get("source_faces_count")
                or not np.isfinite(positions).all() or not np.isfinite(normals).all()
                or (len(faces) and int(faces.max()) >= len(positions))):
            raise ValueError("current capture array shapes or finite/index bounds are invalid: " + row_key)

        psha, fsha = _array_sha(positions, "<f4"), _array_sha(faces, "<u4")
        nsha, gsha = _array_sha(normals, "<f4"), _array_sha(gids, "<u8")
        if (psha != row.get("capture_positions_f32_sha256")
                or fsha != row.get("capture_faces_u32_sha256")
                or nsha != row.get("capture_normals_f32_sha256")
                or gsha != row.get("capture_gids_u64_sha256")):
            raise ValueError("current arrays differ from their pinned preflight: " + row_key)

        first, offset = row.get("first_vertex"), row.get("row_local_gid_offset")
        if (type(first) is not int or first < 0 or type(offset) is not int or offset < 0):
            raise ValueError("current source row lacks valid NHA row and GID offset: " + row_key)
        gid_start = first + offset
        if (not len(gids) or int(gids[0]) != gid_start
                or not np.array_equal(gids, np.arange(gid_start, gid_start + len(gids), dtype="<u8"))):
            raise ValueError("current capture GIDs do not follow the NHA row allocation: " + row_key)
        if not (
                owner.get("captured_positions_world_f32_sha256") == psha
                and owner.get("captured_faces_u32_sha256") == fsha
                and owner.get("captured_global_vertex_ids_sha256") == gsha
                and owner.get("nha_faces_u32_sha256") == fsha):
            raise ValueError("current arrays differ from owner ancestry pins: " + row_key)

        rows[row_key] = {
            "typed_identity": list(key), "body_index": row["body_index"],
            "face_count": len(faces), "faces_sha256": fsha, "first_vertex": gid_start,
            "fma": row["fma"], "layer": row["layer"], "name": row["name"],
            "source_member": row["source_member"], "vertex_count": len(positions),
            "source_nha_first_vertex": first, "row_local_gid_offset": offset,
            "source_nha_row_sha256": pin_sha["source_nha_row_sha256"],
            "source_member_sha256": pin_sha["source_member_sha256"],
            "source_face_corner_ancestry": ancestry,
            "rigid_source_world_xyz_f32_sha256": owner["rigid_source_world_xyz_f32_sha256"],
        }
        bindings[row_key] = {
            "typed_identity": list(key), "vertex_count": len(positions), "face_count": len(faces),
            "positions_f32_sha256": psha, "faces_u32_sha256": fsha,
            "normals_f32_sha256": nsha, "source_vertex_ids_u64_sha256": gsha,
        }

    parent_gate_evidence = {
        "parent_source_handoff_sha256": parent_sha,
        "source_gate_results": {
            gate: parent[gate] for gate in (
                "source_topology_passed", "interface_gate_passed", "solid_gate_passed",
                "independent_source_collar_calibre_passed",
                "candidate_pairs_equal_frozen_source_pairs",
                "baseline_pairs_equal_frozen_source_pairs",
                "candidate_baseline_exact_pairsets_equal",
            )
        },
        "exact_face_pairset_sha256": pair_sha,
    }
    output = json.loads(json.dumps(parent))
    output.update({
        "status": "PENDING_CURRENT_AORTIC_PORT_GEOMETRY",
        "passed": False,
        "source_topology_passed": False,
        "interface_gate_passed": False,
        "solid_gate_passed": False,
        "independent_source_collar_calibre_passed": False,
        "candidate_pairs_equal_frozen_source_pairs": False,
        "baseline_pairs_equal_frozen_source_pairs": False,
        "candidate_baseline_exact_pairsets_equal": False,
        "parent_source_gate_evidence": parent_gate_evidence,
        "capture_input_bundle_sha256": capture_sha,
        "source_preflight_sha256": preflight_sha,
        "parent_source_handoff_sha256": parent_sha,
        "source_rows": rows,
        "candidate_surface_bindings": bindings,
        "current_port_geometry_evidence": {
            "status": "PENDING_FRESH_CURRENT_PORT_REPORT",
            "geometry_predicates_run": False,
            "anatomical_admission": False,
            "native_combined_scene_admission": False,
        },
        "current_capture_owner_rebind": {
            "status": "PASS_CURRENT_SOURCE_ROW_AND_CAPTURE_ARRAY_REBIND",
            "typed_rows": [list(k) for k in identities],
            "capture_input_bundle_sha256": capture_sha,
            "source_preflight_sha256": preflight_sha,
            "candidate_array_hashes_match_current_capture": True,
            "source_rows_match_ordered_face_corner_ancestry": True,
            "source_gid_allocation_matches_current_capture": True,
            "geometry_predicates_run": False,
            "anatomical_admission": False,
            "native_combined_scene_admission": False,
        },
    })
    output.pop("evidence_report_sha256", None)
    if parent_source_handoff_path is not None:
        output["parent_source_handoff_path"] = parent_source_handoff_path
    return output


def seal_current_aortic_source_handoff_from_port_report(
        pending_handoff_bytes, *, expected_pending_handoff_sha256,
        current_port_report_bytes, expected_current_port_report_sha256,
        current_source_preflight_bytes, expected_capture_input_bundle_sha256,
        capture_array_mapping, current_port_report_path=None):
    """Seal a pending handoff only against fresh geometry on those exact arrays.

    The source map is marked verified only after the current native port report
    binds to its arrays, exact ancestry pairset, closed-solid gates, and the
    original independent station-calibre limits. The normal proof consumer must
    still recheck the report against prepared surfaces before publication.
    """
    pending_sha = _sha256(pending_handoff_bytes, "pending source handoff")
    if _valid_sha256(expected_pending_handoff_sha256, "expected pending handoff SHA-256") != pending_sha:
        raise ValueError("pending source handoff differs from its SHA-256 pin")
    report_sha = _sha256(current_port_report_bytes, "current port report")
    if _valid_sha256(expected_current_port_report_sha256, "expected current port SHA-256") != report_sha:
        raise ValueError("current port report differs from its SHA-256 pin")
    preflight_sha = _sha256(current_source_preflight_bytes, "current source preflight")
    capture_sha = _valid_sha256(expected_capture_input_bundle_sha256, "capture input bundle SHA-256")
    if current_port_report_path is not None and (
            not isinstance(current_port_report_path, str) or not current_port_report_path):
        raise ValueError("current port report path must be nonempty")

    try:
        pending = json.loads(pending_handoff_bytes)
        report = json.loads(current_port_report_bytes)
        preflight = json.loads(current_source_preflight_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pending handoff, current report, or preflight is not UTF-8 JSON") from exc
    if (not isinstance(pending, dict)
            or pending.get("schema") != _HANDOFF_SCHEMA
            or pending.get("status") != "PENDING_CURRENT_AORTIC_PORT_GEOMETRY"
            or pending.get("passed") is not False
            or pending.get("source_topology_passed") is not False
            or pending.get("interface_gate_passed") is not False
            or pending.get("solid_gate_passed") is not False
            or pending.get("independent_source_collar_calibre_passed") is not False):
        raise ValueError("source handoff is not pending fresh current geometry")
    if (pending.get("capture_input_bundle_sha256") != capture_sha
            or pending.get("source_preflight_sha256") != preflight_sha
            or not isinstance(preflight, dict)
            or preflight.get("capture_input_bundle_sha256") != capture_sha):
        raise ValueError("pending handoff, source preflight, and capture input pins differ")

    link = pending.get("typed_link")
    identities = [tuple(k) for k in link] if isinstance(link, list) else []
    if (len(identities) != 2 or any(len(k) != 2 or k[0] != 51011 for k in identities)
            or identities[0][1] == identities[1][1]):
        raise ValueError("pending handoff has an invalid typed aortic link")
    identity_rows = [list(k) for k in identities]
    source_rows = pending.get("source_rows")
    bindings = pending.get("candidate_surface_bindings")
    pairs = pending.get("exact_face_pairs")
    pair_sha = pending.get("exact_face_pairset_sha256")
    if (not isinstance(source_rows, dict) or not isinstance(bindings, dict)
            or not isinstance(pairs, list) or not pairs):
        raise ValueError("pending handoff lacks current source rows, arrays, or ancestry pairs")
    if (not isinstance(report, dict)
            or report.get("schema") != "numi.human.current-native-aortic-port-result.v1"
            or report.get("status") != "PASS_CURRENT_NATIVE_PORT_DIAGNOSTIC"
            or report.get("typed_link") != identity_rows
            or report.get("anatomical_admission") is not False
            or report.get("geometry_predicates_run") is not True
            or report.get("current_port_passed") is not True
            or report.get("source_ancestry_handoff_sha256") != pending.get("parent_source_handoff_sha256")
            or report.get("source_ancestry_pairset_sha256") != pair_sha
            or report.get("source_ancestry_pair_count") != len(pairs)
            or report.get("current_candidate_and_current_source_intersection_recomputed") is not True
            or report.get("current_candidate_and_source_pairsets_equal") is not True
            or report.get("current_candidate_pairs_equal_source_ancestry_pairs") is not True
            or report.get("current_source_baseline_pairs_equal_source_ancestry_pairs") is not True
            or report.get("source_pairset_used_only_as_ancestry") is not True):
        raise ValueError("fresh current port report is missing its typed link or recomputed source gates")
    limitations = report.get("limitations")
    if (not isinstance(limitations, dict)
            or limitations.get("no_blood_mass_or_physical_state_modified") is not True
            or limitations.get("no_motion_or_duration_qualification") is not True
            or limitations.get("no_whole_anatomy_admission") is not True
            or limitations.get("source_capture_is_actual_native_one_step") is not True
            or limitations.get("source_station_calibre_is_not_clinical") is not True):
        raise ValueError("fresh current port report has missing or contradictory scope limits")

    if (not isinstance(capture_array_mapping, Mapping)
            or set(capture_array_mapping) != set(identities)):
        raise ValueError("current arrays do not exactly cover the typed link")
    for key in identities:
        row_key = f"{key[0]}:{key[1]}"
        row = source_rows.get(row_key)
        binding = bindings.get(row_key)
        arrays = capture_array_mapping[key]
        if not isinstance(row, dict) or not isinstance(binding, dict) or not isinstance(arrays, Mapping):
            raise ValueError("current source row or array binding is incomplete: " + row_key)
        positions = np.asarray(arrays.get("positions"), dtype="<f4")
        faces = np.asarray(arrays.get("faces"), dtype="<u4")
        normals = np.asarray(arrays.get("normals"), dtype="<f4")
        gids = np.asarray(arrays.get("source_vertex_ids"), dtype="<u8")
        actual = {
            "positions_f32_sha256": _array_sha(positions, "<f4"),
            "faces_u32_sha256": _array_sha(faces, "<u4"),
            "normals_f32_sha256": _array_sha(normals, "<f4"),
            "source_vertex_ids_u64_sha256": _array_sha(gids, "<u8"),
        }
        if (binding.get("typed_identity") != list(key)
                or binding.get("vertex_count") != len(positions)
                or binding.get("face_count") != len(faces)
                or any(binding.get(name) != value for name, value in actual.items())
                or row.get("first_vertex") != (int(gids[0]) if len(gids) else None)
                or len(gids) != len(positions)
                or not np.array_equal(gids, np.arange(
                    int(row["first_vertex"]), int(row["first_vertex"]) + len(gids), dtype="<u8"))
                or any(max(pair[i] for pair in pairs) >= len(faces) for i in range(2))):
            raise ValueError("pending source row arrays or GID allocation changed: " + row_key)

    topology = report.get("source_topology")
    if not isinstance(topology, dict):
        raise ValueError("fresh current port report lacks topology records")
    for key in identities:
        row_key = f"{key[0]}:{key[1]}"
        binding = bindings[row_key]
        candidate = topology.get("candidate:" + str(key[1]))
        baseline = topology.get("independent_source_rigid_baseline:" + str(key[1]))
        if (not isinstance(candidate, dict) or candidate.get("identity") != list(key)
                or candidate.get("source_vertex_ids_sha256") != binding["source_vertex_ids_u64_sha256"]
                or candidate.get("closed_self_free_single_component") is not True
                or candidate.get("exact_self_audit_count") != 0
                or candidate.get("referenced_face_component_count") != 1
                or not isinstance(candidate.get("prepared_surface"), dict)
                or candidate["prepared_surface"].get("identity") != list(key)
                or candidate["prepared_surface"].get("positions_sha256") != binding["positions_f32_sha256"]
                or candidate["prepared_surface"].get("faces_sha256") != binding["faces_u32_sha256"]
                or candidate["prepared_surface"].get("referenced_surface_closed") is not True
                or not isinstance(baseline, dict) or baseline.get("identity") != list(key)
                or baseline.get("closed_self_free_single_component") is not True
                or baseline.get("exact_self_audit_count") != 0
                or baseline.get("referenced_face_component_count") != 1):
            raise ValueError("fresh current report topology does not match exact candidate arrays")

    interfaces, solids = report.get("interface_gates"), report.get("solid_gates")
    if not isinstance(interfaces, dict) or not isinstance(solids, dict):
        raise ValueError("fresh current report lacks interface or solid gates")
    for role in ("candidate", "independent_source_rigid_baseline"):
        interface = interfaces.get(role)
        scope = interface.get("documented_vessel_junction_scope") if isinstance(interface, dict) else None
        partial = interface.get("exact_partial_overlap_witnesses") if isinstance(interface, dict) else None
        boundary = partial.get("boundary_predicate") if isinstance(partial, dict) else None
        transverse = partial.get("exact_transverse_overlap_side_witnesses") if isinstance(partial, dict) else None
        frozen_hash = hashlib.sha256(
            np.ascontiguousarray(np.asarray(pairs, dtype="<i8")).tobytes()).hexdigest()
        if (not isinstance(interface, dict) or interface.get("passed") is not True
                or interface.get("typed_port") != identity_rows
                or interface.get("observed_face_pairs") != pairs
                or interface.get("frozen_face_pairs") != pairs
                or interface.get("observed_pair_count") != len(pairs)
                or interface.get("frozen_face_pairs_sha256") != frozen_hash
                or interface.get("unexpected_face_pairs") != []
                or interface.get("observed_pairs_within_frozen_face_patch_cross_product") is not True
                or interface.get("observed_pairs_within_source_declared_region") is not True
                or not isinstance(scope, dict) or scope.get("passed") is not True
                or scope.get("typed_port") != identity_rows
                or scope.get("allowed_typed_port") != identity_rows
                or scope.get("scope_kind") != "declared_source_network_link"
                or scope.get("frozen_matched_face_patch_required") is not True
                or scope.get("downstream_closed_solid_confirmation_required") is not True
                or not isinstance(partial, dict) or partial.get("passed") is not True
                or partial.get("requires_separate_closed_solid_confirmation") is not True
                or not isinstance(boundary, dict) or boundary.get("passed") is not True
                or boundary.get("typed_port") != identity_rows
                or boundary.get("unexpected_pair_count") != 0
                or boundary.get("observed_pair_count") != len(pairs)
                or not isinstance(transverse, dict) or transverse.get("passed") is not True
                or transverse.get("requires_all_four_witnesses_from_same_face_pair") is not True
                or transverse.get("requires_strict_positive_exact_barycentric_coordinates") is not True
                or transverse.get("uses_exact_rational_point_location_in_opposite_closed_surface") is not True):
            raise ValueError("fresh current report interface pairset or witness failed: " + role)
        solid = solids.get(role)
        lineage = solid.get("union_float64_mesh64_lineage") if isinstance(solid, dict) else None
        if (not isinstance(solid, dict) or solid.get("passed") is not True
                or solid.get("manifold_inputs_passed") is not True
                or solid.get("positive_intersection_volume") is not True
                or solid.get("union_connected_numerical_solid") is not True
                or solid.get("union_component_count") != 1
                or solid.get("union_lineage_has_both_sources_without_generated_or_unmapped_faces") is not True
                or not isinstance(solid.get("exact_inside_both_witness"), dict)
                or solid["exact_inside_both_witness"].get("method") != "intersection_volume_centroid"
                or solid["exact_inside_both_witness"].get("first_location") != "inside"
                or solid["exact_inside_both_witness"].get("second_location") != "inside"
                or not isinstance(lineage, dict) or lineage.get("generated_face_count") != 0
                or lineage.get("unmapped_face_count") != 0
                or set(lineage.get("run_original_ids", [])) != {key[1] for key in identities}):
            raise ValueError("fresh current report closed-solid gate failed: " + role)

    calibre = report.get("independent_source_station_calibre")
    if not isinstance(calibre, dict) or len(calibre) != 2:
        raise ValueError("fresh current report lacks both independent source calibre stations")
    owners = set()
    for station in calibre.values():
        if (not isinstance(station, dict) or station.get("passed") is not True
                or type(station.get("owner_id")) is not int
                or station.get("owner_id") not in {key[1] for key in identities}
                or station.get("baseline_metric_error") is not None
                or station.get("candidate_metric_error") is not None
                or station.get("comparison_error") is not None
                or station.get("declared_area_ratio") != [0.8, 1.25]
                or station.get("declared_minimum_feret_ratio") != [0.9, 1.1]):
            raise ValueError("fresh current report source calibre gate or tolerance failed")
        owners.add(station["owner_id"])
        for name in ("baseline_section", "candidate_section"):
            if not isinstance(station.get(name), dict) or station[name].get("passed") is not True:
                raise ValueError("fresh current report calibre section failed")
    if owners != {key[1] for key in identities}:
        raise ValueError("fresh current report calibre does not cover both source rows")

    output = json.loads(json.dumps(pending))
    output.update({
        "status": _HANDOFF_STATUS,
        "passed": True,
        "source_topology_passed": True,
        "interface_gate_passed": True,
        "solid_gate_passed": True,
        "independent_source_collar_calibre_passed": True,
        "candidate_pairs_equal_frozen_source_pairs": True,
        "baseline_pairs_equal_frozen_source_pairs": True,
        "candidate_baseline_exact_pairsets_equal": True,
        "evidence_report_sha256": report_sha,
        "current_port_geometry_evidence": {
            "status": "PASS_FRESH_CURRENT_PORT_REPORT_BOUND_TO_CAPTURE_ARRAYS",
            "report_sha256": report_sha,
            "source_preflight_sha256": preflight_sha,
            "capture_input_bundle_sha256": capture_sha,
            "exact_pairset_sha256": pair_sha,
            "exact_pair_count": len(pairs),
            "candidate_array_hashes_rechecked": True,
            "candidate_and_source_pairsets_recomputed": True,
            "candidate_and_source_closed_solid_gates_passed": True,
            "independent_source_station_calibre_passed": True,
            "geometry_predicates_run": True,
            "anatomical_admission": False,
            "native_combined_scene_admission": False,
        },
    })
    if current_port_report_path is not None:
        output["evidence_report_path"] = current_port_report_path
    return output



def canonical_handoff_bytes(handoff):
    """Serialize a source handoff with deterministic UTF-8 JSON bytes."""
    if not isinstance(handoff, dict):
        raise TypeError("source handoff must be a mapping")
    return (json.dumps(handoff, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")

