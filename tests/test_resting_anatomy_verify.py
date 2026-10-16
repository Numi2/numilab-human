import copy
import hashlib
import json
import unittest
from fractions import Fraction

import numpy as np

from numilab_human import cardiac_cavity_intersections as ci
from numilab_human import resting_anatomy_verify as av
from numilab_human.resting_anatomy_verify import (
    audit_source_bound_connected_overlap,
    audit_surface_pair,
    prepare_surface,
    source_surface_key,
    typed_key,
    validate_capture_receipt,
)

FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)


def tetra(scale=1., offset=0.):
    return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32) * np.float32(scale) + np.float32(offset)


def feature_surface(key, positions, faces, gids, *, kind3_gids=(), weight_overrides=None):
    positions = np.asarray(positions, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.uint32)
    gids = np.asarray(gids, dtype=np.uint64)
    weights = np.full((len(gids), 21), 1.0 / 21.0, dtype=np.float64)
    for row, gid in enumerate(gids):
        if weight_overrides and int(gid) in weight_overrides:
            weights[row] = np.asarray(weight_overrides[int(gid)], dtype=np.float64)
    ordered = list(range(21))
    vertex_lineage = {}
    for gid in gids:
        record = {"source_gid": int(gid), "source_parent_gids": [int(gid)]}
        if int(gid) in kind3_gids:
            record.update({
                "material_mapping_kind": 3,
                "material_binding": "independent_existing_core_carrier",
                "core_body_indices": ordered,
                "core_weights": weights[np.flatnonzero(gids == gid)[0]].tolist(),
            })
        vertex_lineage[int(gid)] = record
    return prepare_surface(
        key, positions, faces, source_vertex_ids=gids,
        source_face_lineage=[{"source_parent_face": i} for i in range(len(faces))],
        source_vertex_lineage=vertex_lineage,
        ordered_core_body_indices=ordered,
        core_weight_gids=gids,
        core_weights_float64=weights,
    )


def feature_declaration(first, second, first_face, second_face, gids):
    return {
        "first_identity": list(first.key),
        "second_identity": list(second.key),
        "first_face_index": first_face,
        "second_face_index": second_face,
        "feature_gids": list(gids),
    }


def declarations_for_actual_shared_features(first, second):
    baseline = audit_surface_pair(first, second)
    declarations = []
    for first_face, second_face in baseline["unexpected_triangle_pairs"]:
        first_gids = {int(first.source_vertex_ids[int(i)]) for i in first.faces[first_face]}
        second_gids = {int(second.source_vertex_ids[int(i)]) for i in second.faces[second_face]}
        common = sorted(first_gids & second_gids)
        if 1 <= len(common) <= 2:
            declarations.append(feature_declaration(first, second, first_face, second_face, common))
    return declarations


def seal_source_point_crosswalk(value):
    payload = {key: item for key, item in value.items()
               if key != "source_capture_crosswalk_sha256"}
    value["source_capture_crosswalk_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True).encode("ascii")).hexdigest()
    return value


def crosswalk_point_case(*, second_owner=23, source_crosswalk=True,
                         second_endpoint=409, selected_endpoint=409,
                         transverse_segment=False, omit_second_seam_face=False):
    from numilab_human.resting_anatomy_verify import (
        _shared_source_crosswalk_point_contact,
    )

    member_sha, row_sha = "c" * 64, "b" * 64
    source_faces = [
        {"source_face": 825, "source_vertex_ids": [409, 416, 415]},
        {"source_face": 826, "source_vertex_ids": [409, 412, 416]},
    ]
    endpoints = [409, 416]
    first_rows = [
        {"face_index": 0, "source_face": 825, "source_row_face": 936,
         "source_row_payload_sha256": row_sha, "ordered_source_vertex_ids": [409, 416, 415],
         "ordered_capture_gids": [100, 101, 102]},
        {"face_index": 1, "source_face": 826, "source_row_face": 937,
         "source_row_payload_sha256": row_sha, "ordered_source_vertex_ids": [409, 412, 416],
         "ordered_capture_gids": [100, 103, 101]},
    ]
    second_rows = [
        {"face_index": 0, "source_face": 825, "source_row_face": 936,
         "source_row_payload_sha256": row_sha, "ordered_source_vertex_ids": [409, 415, 416],
         "ordered_capture_gids": [900, 901, 902]},
        {"face_index": 1, "source_face": 826, "source_row_face": 937,
         "source_row_payload_sha256": row_sha, "ordered_source_vertex_ids": [409, 416, 412],
         "ordered_capture_gids": [900, 902, 903]},
    ]
    if omit_second_seam_face:
        second_rows = second_rows[:1]
    crosswalk = seal_source_point_crosswalk({
        "schema": "numi.human.shared-source-seam-endpoint-crosswalk.v4",
        "seam": {
            "source_member": "FJ1769", "source_member_sha256": member_sha,
            "source_faces": source_faces, "endpoint_source_vertex_ids": endpoints,
        },
        "first": {
            "identity": [51010, 343], "source_row_payload_sha256": row_sha,
            "seam_faces": first_rows,
            "vertex_bindings": [
                {"capture_gid": 100, "source_vertex_id": 409},
                {"capture_gid": 101, "source_vertex_id": 416},
            ],
        },
        "second": {
            "identity": [51010, 374], "source_row_payload_sha256": row_sha,
            "seam_faces": second_rows,
            "vertex_bindings": [
                {"capture_gid": 900, "source_vertex_id": second_endpoint},
                {"capture_gid": 902, "source_vertex_id": 416 if second_endpoint == 409 else 409},
            ],
        },
    })
    owner = {
        "body_index": 23, "capture_receipt_sha256": "e" * 64,
        "position_m": [0.0, 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    points_a = np.asarray([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, -1, 0], [0, 0, 1],
    ], dtype=np.float32)
    faces_a = np.asarray([[0, 1, 2], [0, 3, 1], [0, 2, 4]], dtype=np.uint32)
    points_b = np.asarray([
        [0, 0, 0], [0, 1, 0], [1, 0, 0], [1, -1, 0],
        [0, -1, 0], [0, 0, -1],
    ], dtype=np.float32)
    if transverse_segment:
        points_b[4] = [1, -1, 0]
        points_b[5] = [-1, 2, 0.5]
    faces_b = np.asarray([[0, 1, 2], [0, 2, 3], [0, 4, 5]], dtype=np.uint32)

    def exact_unit_barycentric():
        return [
            [["1", "1"], ["0", "1"], ["0", "1"]],
            [["0", "1"], ["1", "1"], ["0", "1"]],
            [["0", "1"], ["0", "1"], ["1", "1"]],
        ]

    face_a = []
    for source_face, row_face, corners in ((825, 936, [409, 416, 415]),
                                           (826, 937, [409, 412, 416])):
        face_a.append({
            "operation": "retained_source_fragment", "owner_stable_id": 343,
            "source_member": "FJ1769", "source_member_sha256": member_sha,
            "source_face": source_face, "source_row_face": row_face,
            "source_row_payload_sha256": row_sha,
            "corner_barycentric_exact": exact_unit_barycentric(),
        })
    face_a.append({
        "operation": "retained_source_fragment", "owner_stable_id": 343,
        "source_member": "FJ1769", "source_member_sha256": member_sha,
        "source_face": 825, "source_row_face": 936, "source_row_payload_sha256": row_sha,
    })
    face_b = []
    for source_face, row_face in ((825, 936), (826, 937)):
        face_b.append({
            "operation": "reversed_medulla343_priority_interface", "owner_stable_id": 374,
            "source_member": "FJ1769", "source_member_sha256": member_sha,
            "source_face": source_face, "source_row_face": row_face,
            "source_row_payload_sha256": row_sha,
        })
    face_b.append({
        "operation": "retained_source_fragment", "owner_stable_id": 374,
        "source_member": "FJ1831", "source_member_sha256": "d" * 64,
        "source_face": 6057, "source_row_face": 6057, "source_row_payload_sha256": "f" * 64,
    })

    gids_a = np.asarray([100, 101, 102, 103, 104], dtype=np.uint64)
    gids_b = np.asarray([900, 901, 902, 903, 904, 905], dtype=np.uint64)
    crosswalk_sha = crosswalk["source_capture_crosswalk_sha256"]
    vertex_a = {int(g): {"source_gid": int(g)} for g in gids_a}
    vertex_b = {int(g): {"source_gid": int(g)} for g in gids_b}
    if source_crosswalk:
        for gid, endpoint in ((100, 409), (101, 416)):
            vertex_a[gid].update({
                "source_point_crosswalk_sha256": crosswalk_sha,
                "source_seam_endpoint_source_vertex_id": endpoint,
                "source_seam_member": "FJ1769",
                "source_seam_member_sha256": member_sha,
            })
        for gid, endpoint in ((900, second_endpoint),
                              (902, 416 if second_endpoint == 409 else 409)):
            vertex_b[gid].update({
                "source_point_crosswalk_sha256": crosswalk_sha,
                "source_seam_endpoint_source_vertex_id": endpoint,
                "source_seam_member": "FJ1769",
                "source_seam_member_sha256": member_sha,
            })
    first = prepare_surface(
        (51010, 343), points_a, faces_a, source_vertex_ids=gids_a,
        source_face_lineage=face_a, source_vertex_lineage=vertex_a,
        rigid_owner_proof=owner)
    second = prepare_surface(
        (51010, 374), points_b, faces_b, source_vertex_ids=gids_b,
        source_face_lineage=face_b, source_vertex_lineage=vertex_b,
        rigid_owner_proof=dict(owner, body_index=second_owner))
    gid_by_endpoint_a = {409: 100, 416: 101}
    gid_by_endpoint_b = {409: 900, 416: 902}
    declaration = {
        "first_identity": list(first.key), "second_identity": list(second.key),
        "first_face_index": 2, "second_face_index": 2,
        "feature_gid_pairs": [[gid_by_endpoint_a[selected_endpoint],
                               gid_by_endpoint_b[selected_endpoint]]],
        "source_point_crosswalk": crosswalk,
        "source_seam_face_pairs": [[0, 0], [1, 1]],
        "source_seam_endpoint_source_vertex_id": selected_endpoint,
    }
    return first, second, declaration, _shared_source_crosswalk_point_contact

def source_bound_overlap_case(*, altered_contact=False, open_first=False, disconnected_first=False):
    if open_first:
        first = feature_surface(
            (51010, 323), tetra(), FACES[:-1], [700, 701, 702, 703])
    elif disconnected_first:
        points = np.concatenate((tetra(), tetra(offset=3.0)))
        faces = np.concatenate((FACES, FACES + 4))
        first = feature_surface(
            (51010, 323), points, faces, list(range(700, 708)))
    else:
        first = feature_surface(
            (51010, 323), tetra(), FACES, [700, 701, 702, 703])
    second = feature_surface(
        (51010, 326), tetra(offset=.1), FACES, [800, 801, 802, 803])
    exact = ci._audit_pair_prepared_first(first.index, second.records, same_surface=False)
    pairs = sorted(tuple(int(face) for face in pair) for pair in exact["triangle_pairs"])
    contacts = []
    for first_face, second_face in pairs:
        points = first.intersection_points_meters(second, first_face, second_face)
        if altered_contact and not contacts:
            points = ((Fraction(99), Fraction(98), Fraction(97)),)
        contacts.append({"face_pair": [first_face, second_face],
                         "points_world_m": [[str(value) for value in point] for point in points]})
    inside = {"candidate_source": "synthetic exact binary64 point",
              "first_location": "inside", "method": "intersection_volume_centroid",
              "point_world_m": [.2, .2, .2], "predicate": "P6 exact rational point location",
              "second_location": "inside"}
    proof = {"schema": "numi.human.head-csf-pair-continuity.v1",
             "scope": ("only captured typed rows (51010,323) fourth ventricle and "
                       "(51010,326) cerebral aqueduct; no broad target scan and no native admission"),
             "status": "complete", "not_admission": True,
             "pins": {"verifier_sha256": "b" * 64},
             "surfaces": {}, "contact_witnesses": contacts,
             "exact_pair_gate": {"exact_face_pairs": [list(pair) for pair in pairs]},
             "closed_solid_connection": {"exact_inside_both_witness": inside,
                                         "union_components": 1}}
    for surface in (first, second):
        diag = surface.diagnostics()
        proof["surfaces"][str(surface.key)] = {
            "position_sha256": diag["positions_sha256"],
            "face_sha256": diag["faces_sha256"],
            "closed": surface.closed, "exact_self_count": surface.self_audit()["count"]}
    proof_bytes = json.dumps(proof, sort_keys=True, separators=(",", ":")).encode("utf-8")
    proof_sha = hashlib.sha256(proof_bytes).hexdigest()
    source_map_sha = "a" * 64
    binding = {"source_proof_sha256": proof_sha, "source_map_sha256": source_map_sha,
               "surface_identities": [list(first.key), list(second.key)],
               "surfaces": []}
    for surface in (first, second):
        binding["surfaces"].append({
            "identity": list(surface.key),
            **av._surface_source_hashes(surface),
            "source_face_lineage_sha256": av._canonical_json_sha256(surface.source_face_lineage),
            "source_vertex_lineage_sha256": av._canonical_json_sha256([
                [int(gid), surface.source_vertex_lineage[int(gid)]]
                for gid in sorted(surface.source_vertex_lineage)]),
        })
    return first, second, proof, proof_bytes, proof_sha, source_map_sha, binding


def source_bound_vessel_port_case():
    first = feature_surface((51011, 7), tetra(), FACES, [700, 701, 702, 703])
    second = feature_surface((51011, 6), tetra(offset=.1), FACES, [800, 801, 802, 803])
    for surface in (first, second):
        surface.normals = np.full_like(surface.positions, np.float32(1.0 / np.sqrt(3.0)))
    keys = [list(first.key), list(second.key)]
    exact = ci._audit_pair_prepared_first(first.index, second.records, same_surface=False)
    pairs = sorted(tuple(int(face) for face in pair) for pair in exact["triangle_pairs"])
    points = set()
    for first_face, second_face in pairs:
        first_triangle = [[Fraction.from_float(float(component)) for component in point]
                          for point in first.quotient_positions[
                              first.quotient_faces[first_face]].tolist()]
        second_triangle = [[Fraction.from_float(float(component)) for component in point]
                           for point in second.quotient_positions[
                               second.quotient_faces[second_face]].tolist()]
        points.update(tuple(float(component) for component in point)
                      for point in ci.triangle_intersection_points(
                          first_triangle, second_triangle))
    point_rows = [list(point) for point in sorted(points)]
    pair_sha = hashlib.sha256(
        np.ascontiguousarray(np.asarray(pairs, dtype="<i8")).tobytes()).hexdigest()
    source_pair_sha = hashlib.sha256(
        np.ascontiguousarray(np.asarray(pairs, dtype="<u4")).tobytes()).hexdigest()
    point_sha = hashlib.sha256(
        np.ascontiguousarray(np.asarray(point_rows, dtype="<f8")).tobytes()).hexdigest()
    all_first_faces = list(range(len(first.faces)))
    all_second_faces = list(range(len(second.faces)))

    def topology(surface, role):
        hashes = av._surface_source_hashes(surface)
        return {
            "role": role + ":" + str(surface.key[0]) + ":" + str(surface.key[1]),
            "identity": list(surface.key),
            "closed_self_free_single_component": True,
            "exact_self_audit_count": 0,
            "referenced_face_component_count": 1,
            "source_vertex_ids_sha256": hashes["gids_sha256"],
            "prepared_surface": {
                "identity": list(surface.key),
                "positions_sha256": hashes["positions_sha256"],
                "faces_sha256": hashes["faces_sha256"],
                "referenced_surface_closed": True,
            },
        }

    def interface_gate():
        return {
            "passed": True,
            "typed_port": keys,
            "p6_audit_passed": False,
            "observed_pair_count": len(pairs),
            "observed_face_pairs": [list(pair) for pair in pairs],
            "frozen_face_pairs": [list(pair) for pair in pairs],
            "frozen_face_pairs_sha256": pair_sha,
            "unexpected_face_pairs": [],
            "observed_pairs_within_frozen_face_patch_cross_product": True,
            "observed_pairs_within_source_declared_region": True,
            "bridge_frozen_collar_faces": all_first_faces,
            "bridge_frozen_collar_faces_sha256": hashlib.sha256(
                np.ascontiguousarray(np.asarray(all_first_faces, dtype="<i8")).tobytes()).hexdigest(),
            "target_frozen_collar_faces": all_second_faces,
            "target_frozen_collar_faces_sha256": hashlib.sha256(
                np.ascontiguousarray(np.asarray(all_second_faces, dtype="<i8")).tobytes()).hexdigest(),
            "documented_vessel_junction_scope": {
                "passed": True,
                "typed_port": keys,
                "allowed_typed_port": keys,
                "scope_kind": "declared_source_network_link",
                "frozen_matched_face_patch_required": True,
                "downstream_closed_solid_confirmation_required": True,
            },
            "exact_partial_overlap_witnesses": {
                "passed": True,
                "requires_separate_closed_solid_confirmation": True,
                "classification": "symmetric_exact_transverse_inside_outside_overlap",
                "boundary_predicate": {
                    "passed": True, "typed_port": keys,
                    "unexpected_pair_count": 0, "observed_pair_count": len(pairs),
                },
                "exact_transverse_overlap_side_witnesses": {
                    "passed": True,
                    "requires_all_four_witnesses_from_same_face_pair": True,
                    "requires_strict_positive_exact_barycentric_coordinates": True,
                    "uses_exact_rational_point_location_in_opposite_closed_surface": True,
                },
            },
            "exact_interface_point_count": len(point_rows),
            "exact_interface_points_world_m": point_rows,
            "exact_interface_points_sha256": point_sha,
        }

    def solid_gate():
        return {
            "passed": True,
            "manifold_inputs_passed": True,
            "positive_intersection_volume": True,
            "intersection_volume_m3": 1e-6,
            "union_connected_numerical_solid": True,
            "union_component_count": 1,
            "union_lineage_has_both_sources_without_generated_or_unmapped_faces": True,
            "union_float64_mesh64_lineage": {
                "generated_face_count": 0,
                "unmapped_face_count": 0,
                "run_original_ids": [6, 7],
            },
            "exact_inside_both_witness": {
                "method": "intersection_volume_centroid",
                "point_world_m": [.2, .2, .2],
                "first_location": "inside",
                "second_location": "inside",
            },
            "volume_accounting": {"passed": True},
        }

    calibre = {}
    for surface in (first, second):
        calibre[surface.key[1]] = {
            "owner_id": surface.key[1],
            "role": "synthetic",
            "passed": True,
            "baseline_metric_error": None,
            "candidate_metric_error": None,
            "comparison_error": None,
            "declared_area_ratio": [0.8, 1.25],
            "declared_minimum_feret_ratio": [0.9, 1.1],
            "baseline_section": {"passed": True},
            "candidate_section": {"passed": True},
        }
    def handoff_surface(surface):
        return {
            "typed_identity": list(surface.key),
            "vertex_count": len(surface.positions),
            "face_count": len(surface.faces),
            "positions_f32_sha256": hashlib.sha256(
                np.ascontiguousarray(surface.positions, dtype="<f4").tobytes()).hexdigest(),
            "faces_u32_sha256": hashlib.sha256(
                np.ascontiguousarray(surface.faces, dtype="<u4").tobytes()).hexdigest(),
            "normals_f32_sha256": hashlib.sha256(
                np.ascontiguousarray(surface.normals, dtype="<f4").tobytes()).hexdigest(),
            "source_vertex_ids_u64_sha256": hashlib.sha256(
                np.ascontiguousarray(surface.source_vertex_ids, dtype="<u8").tobytes()).hexdigest(),
        }
    parent_handoff_sha = "c" * 64
    source_map = {
        "schema": "numi.human.verified-aortic-arch-interface-handoff.v1",
        "status": "VERIFIED_SOURCE_BOUND_AORTIC_ARCH_INTERFACE",
        "passed": True,
        "typed_link": keys,
        "source_topology_passed": True,
        "interface_gate_passed": True,
        "solid_gate_passed": True,
        "independent_source_collar_calibre_passed": True,
        "candidate_pairs_equal_frozen_source_pairs": True,
        "baseline_pairs_equal_frozen_source_pairs": True,
        "candidate_baseline_exact_pairsets_equal": True,
        "exact_face_pairs": [list(pair) for pair in pairs],
        "exact_face_pairset_sha256": source_pair_sha,
        "source_rows": {
            "51011:7": {
                "body_index": 20, "layer": 2, "first_vertex": 700,
                "vertex_count": 4, "face_count": 4,
                "faces_sha256": hashlib.sha256(
                    np.ascontiguousarray(first.faces, dtype="<u4").tobytes()).hexdigest(),
                "source_member": "FJ-A", "fma": "FMA-A",
            },
            "51011:6": {
                "body_index": 20, "layer": 2, "first_vertex": 800,
                "vertex_count": 4, "face_count": 4,
                "faces_sha256": hashlib.sha256(
                    np.ascontiguousarray(second.faces, dtype="<u4").tobytes()).hexdigest(),
                "source_member": "FJ-B", "fma": "FMA-B",
            },
        },
        "candidate_surface_bindings": {
            "51011:7": handoff_surface(first),
            "51011:6": handoff_surface(second),
        },
        "evidence_report_sha256": "d" * 64,
        "source_preflight_sha256": "e" * 64,
    }
    source_map_bytes = json.dumps(source_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_map_sha = hashlib.sha256(source_map_bytes).hexdigest()
    proof = {
        "schema": "numi.human.current-native-aortic-port-result.v1",
        "status": "PASS_CURRENT_NATIVE_PORT_DIAGNOSTIC",
        "typed_link": keys,
        "anatomical_admission": False,
        "geometry_predicates_run": True,
        "current_port_passed": True,
        "current_candidate_and_current_source_intersection_recomputed": True,
        "current_candidate_and_source_pairsets_equal": True,
        "current_candidate_pairs_equal_source_ancestry_pairs": True,
        "current_source_baseline_pairs_equal_source_ancestry_pairs": True,
        "source_pairset_used_only_as_ancestry": True,
        "source_ancestry_handoff_path": "parent-handoff-7-6.json",
        "source_ancestry_handoff_sha256": parent_handoff_sha,
        "source_ancestry_pair_count": len(pairs),
        "source_ancestry_pairset_sha256": source_pair_sha,
        "source_topology": {
            "candidate:7": topology(first, "candidate"),
            "candidate:6": topology(second, "candidate"),
            "independent_source_rigid_baseline:7": topology(first, "independent_source_rigid_baseline"),
            "independent_source_rigid_baseline:6": topology(second, "independent_source_rigid_baseline"),
        },
        "interface_gates": {
            "candidate": interface_gate(),
            "independent_source_rigid_baseline": interface_gate(),
        },
        "solid_gates": {
            "candidate": solid_gate(),
            "independent_source_rigid_baseline": solid_gate(),
        },
        "independent_source_station_calibre": calibre,
        "limitations": {
            "no_blood_mass_or_physical_state_modified": True,
            "no_motion_or_duration_qualification": True,
            "no_whole_anatomy_admission": True,
            "source_capture_is_actual_native_one_step": True,
            "source_station_calibre_is_not_clinical": True,
        },
    }
    proof_bytes = json.dumps(proof, sort_keys=True, separators=(",", ":")).encode("utf-8")
    proof_sha = hashlib.sha256(proof_bytes).hexdigest()
    baseline_hashes = {}
    for surface in (first, second):
        hashes = av._surface_source_hashes(surface)
        row_key = f"{surface.key[0]}:{surface.key[1]}"
        baseline_hashes[row_key] = {
            "positions_sha256": hashes["positions_sha256"],
            "faces_sha256": hashes["faces_sha256"],
            "source_vertex_ids_sha256": hashes["gids_sha256"],
        }
        source_map["source_rows"][row_key][
            "rigid_source_world_xyz_f32_sha256"] = hashes["positions_sha256"]
    preflight_sha = "e" * 64
    capture_sha = "f" * 64
    source_map.update({
        "parent_source_handoff_sha256": parent_handoff_sha,
        "evidence_report_sha256": proof_sha,
        "source_preflight_sha256": preflight_sha,
        "capture_input_bundle_sha256": capture_sha,
        "current_port_geometry_evidence": {
            "status": "PASS_FRESH_CURRENT_PORT_REPORT_BOUND_TO_CAPTURE_ARRAYS",
            "report_sha256": proof_sha,
            "source_preflight_sha256": preflight_sha,
            "capture_input_bundle_sha256": capture_sha,
            "exact_pairset_sha256": source_pair_sha,
            "exact_pair_count": len(pairs),
            "candidate_array_hashes_rechecked": True,
            "independent_rigid_baseline_array_hashes_rechecked": True,
            "independent_rigid_baseline_array_hashes": baseline_hashes,
            "candidate_and_source_pairsets_recomputed": True,
            "candidate_and_source_closed_solid_gates_passed": True,
            "independent_source_station_calibre_passed": True,
        },
    })
    source_map_bytes = json.dumps(source_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
    source_map_sha = hashlib.sha256(source_map_bytes).hexdigest()
    binding = {
        "source_proof_sha256": proof_sha,
        "source_map_sha256": source_map_sha,
        "source_ancestry_handoff_sha256": proof["source_ancestry_handoff_sha256"],
        "source_ancestry_pairset_sha256": proof["source_ancestry_pairset_sha256"],
        "surface_identities": keys,
    }
    context = {
        "source_proof_bytes": proof_bytes,
        "expected_source_proof_sha256": proof_sha,
        "source_binding": binding,
        "expected_source_binding_sha256": av._canonical_json_sha256(binding),
        "expected_source_map_sha256": source_map_sha,
        "source_map_bytes": source_map_bytes,
    }
    return first, second, proof, context, pairs


def seal_source_patch_complex(value):
    payload = {key: item for key, item in value.items()
               if key != "source_patch_complex_sha256"}
    value["source_patch_complex_sha256"] = av._canonical_json_sha256(payload)
    return value


def source_patch_complex_case(*, off_edge=False, support_edge=False):
    source_faces = [
        {"source_face": 825, "source_vertex_ids": [10, 11, 12]},
        {"source_face": 826, "source_vertex_ids": [10, 12, 13]},
    ]
    patch_ids = [10, 11, 12, 13]
    member_sha = "a" * 64
    owner = {
        "body_index": 23, "capture_receipt_sha256": "e" * 64,
        "position_m": [0.0, 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    row_sha = ["b" * 64, "c" * 64]
    base_points = np.asarray(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    base_faces = [
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32),
        np.asarray([[0, 2, 1], [0, 3, 2]], dtype=np.uint32),
    ]
    gids = [[100, 101, 102, 103], [900, 901, 902, 903]]
    if not off_edge:
        base_points = [base_points.copy(), base_points.copy()]
    if off_edge:
        extra_a = np.asarray([[2, 0, 0], [2, 0, 1]], dtype=np.float32)
        extra_b = np.asarray([[2, 0, 0], [2, -1, 0]], dtype=np.float32)
        base_points = [np.concatenate((base_points, extra_a)),
                       np.concatenate((base_points, extra_b))]
        base_faces = [
            np.concatenate((base_faces[0], np.asarray([[0, 4, 5]], dtype=np.uint32))),
            np.concatenate((base_faces[1], np.asarray([[0, 4, 5]], dtype=np.uint32))),
        ]
        gids = [gids[0] + [104, 105], gids[1] + [904, 905]]
    elif support_edge:
        extra_a = np.asarray([[0, 0, -1]], dtype=np.float32)
        extra_b = np.asarray([[0, 0, 1]], dtype=np.float32)
        base_points = [np.concatenate((base_points[0].copy(), extra_a)),
                       np.concatenate((base_points[1].copy(), extra_b))]
        base_faces = [
            np.concatenate((base_faces[0], np.asarray([[1, 0, 4]], dtype=np.uint32))),
            np.concatenate((base_faces[1], np.asarray([[0, 1, 4]], dtype=np.uint32))),
        ]
        gids = [gids[0] + [104], gids[1] + [904]]

    surfaces = []
    side_payloads = []
    for side_index, (identity, points, faces, face_gids) in enumerate(zip(
            ((51010, 343), (51010, 374)), base_points, base_faces, gids)):
        ordered = ([ [10, 11, 12], [10, 12, 13] ] if side_index == 0 else
                   [ [10, 12, 11], [10, 13, 12] ])
        lineage = []
        support = []
        patch_bindings = []
        for face_index, face in enumerate(faces):
            if face_index < 2:
                source_face = source_faces[face_index]
                source_member = "FJ1769"
                source_member_sha = member_sha
                ordered_source_ids = ordered[face_index]
                face_id = source_face["source_face"]
                row_face = 930 + side_index * 10 + face_index
                bary = []
                for source_id in ordered_source_ids:
                    basis = [0, 0, 0]
                    basis[source_face["source_vertex_ids"].index(source_id)] = 1
                    bary.append([[str(value), "1"] for value in basis])
                record = {
                    "source_member": source_member,
                    "source_member_sha256": source_member_sha,
                    "source_face": face_id,
                    "source_row_face": row_face,
                    "source_row_payload_sha256": row_sha[side_index],
                    "owner_stable_id": identity[1],
                    "source_parent_face": face_id,
                }
                if side_index == 0:
                    record["corner_barycentric_exact"] = bary
                patch_bindings.append({
                    "face_index": face_index,
                    "source_face": face_id,
                    "source_row_face": row_face,
                    "source_row_payload_sha256": row_sha[side_index],
                    "ordered_source_vertex_ids": ordered_source_ids,
                    "ordered_capture_gids": [face_gids[i] for i in face],
                })
            else:
                source_member = "FJ9000"
                source_member_sha = "d" * 64
                face_id = 9000 + face_index
                row_face = face_id
                record = {
                    "source_member": source_member,
                    "source_member_sha256": source_member_sha,
                    "source_face": face_id,
                    "source_row_face": row_face,
                    "source_row_payload_sha256": row_sha[side_index],
                    "owner_stable_id": identity[1],
                    "source_parent_face": face_id,
                }
            lineage.append(record)
            support.append({
                "face_index": face_index,
                "source_member": source_member,
                "source_member_sha256": source_member_sha,
                "source_face": face_id,
                "source_row_face": row_face,
                "source_row_payload_sha256": row_sha[side_index],
                "ordered_capture_gids": [face_gids[i] for i in face],
            })
        vertex_lineage = {}
        for index, gid in enumerate(face_gids):
            vertex_lineage[gid] = {
                "source_gid": gid,
                "source_vertex_id": patch_ids[index] if index < 4 else 900 + index,
                "source_member": "FJ1769" if index < 4 else "FJ9000",
                "source_member_sha256": member_sha if index < 4 else "d" * 64,
            }
        surface = prepare_surface(
            identity, points, faces, source_vertex_ids=face_gids,
            source_face_lineage=lineage, source_vertex_lineage=vertex_lineage,
            rigid_owner_proof=owner)
        surfaces.append(surface)
        bindings = [
            {"source_vertex_id": patch_ids[i], "capture_gid": face_gids[i]}
            for i in range(4)
        ]
        side_payloads.append({
            "identity": list(identity),
            "source_row_payload_sha256": row_sha[side_index],
            "surface_hashes": av._source_patch_surface_hashes(surface),
            "vertex_bindings": bindings,
            "patch_faces": patch_bindings,
            "support_faces": support,
        })
    declaration = seal_source_patch_complex({
        "schema": "numi.human.reciprocal-source-patch-complex.v1",
        "identities": [list(surface.key) for surface in surfaces],
        "source_member": "FJ1769",
        "source_member_sha256": member_sha,
        "source_faces": source_faces,
        "patch_vertex_source_ids": patch_ids,
        "first": side_payloads[0],
        "second": side_payloads[1],
    })
    return surfaces[0], surfaces[1], declaration


def source_patch_carrier_case(*, offcarrier=False, malformed_ancestry=False, closed=False, rational_local=False):
    member, member_sha = "FJ1737", "a" * 64
    raw_triangles = [
        [[0, 0, 0], [3, 0, 0], [0, 1, 0]],
        [[10, 0, 0], [11, 0, 0], [10, 1, 0]],
        [[20, 0, 0], [21, 0, 0], [20, 1, 0]],
    ]
    coordinates = np.asarray(raw_triangles, dtype=np.float32).reshape((-1, 3))
    point_ids = [f"p{index}" for index in range(len(coordinates))]

    def pair(value):
        exact = Fraction.from_float(float(np.float32(value)))
        return [str(exact.numerator), str(exact.denominator)]

    def rational_pair(value):
        exact = Fraction(value)
        return [str(exact.numerator), str(exact.denominator)]

    def exact(point):
        return [pair(value) for value in point]

    local_exact = [exact(point) for point in coordinates]
    if rational_local:
        # Parent face 100 has vertices (0,0,0), (3,0,0), (0,1,0).
        # Barycentrics (1/2,1/6,1/3) give the exact non-Float32 point (1/2,1/3,0).
        local_exact[0] = [rational_pair(Fraction(1, 2)), rational_pair(Fraction(1, 3)),
                          rational_pair(Fraction(0))]
        coordinates[0] = np.asarray([0.5, np.float32(1 / 3), 0.0], dtype=np.float32)

    points = [{
        "point_id": point_id,
        "local_xyz_exact": local_exact[index],
        "capture_xyz_exact": exact(point),
        "local_float32_bits": av._patch_rational_float32_bits(
            [Fraction(int(value[0]), int(value[1])) for value in local_exact[index]]),
        "capture_float32_bits": av._patch_float32_bits(point),
    } for index, (point_id, point) in enumerate(zip(point_ids, coordinates))]
    source_faces, carrier_faces = [], []
    for index, triangle in enumerate(raw_triangles):
        source_face = 100 + index
        ids = [1000 + index * 3 + corner for corner in range(3)]
        source_faces.append({"source_face": source_face, "source_vertex_ids": ids,
                             "source_vertices_exact": [exact(point) for point in triangle]})
        face_point_ids = point_ids[index * 3:index * 3 + 3]
        barycentric = [[["1", "1"], ["0", "1"], ["0", "1"]],
                       [["0", "1"], ["1", "1"], ["0", "1"]],
                       [["0", "1"], ["0", "1"], ["1", "1"]]]
        if rational_local and index == 0:
            barycentric[0] = [["1", "2"], ["1", "6"], ["1", "3"]]
        record = {"carrier_face_id": f"c{index}", "source_member": member,
                  "source_member_sha256": member_sha, "source_face": source_face,
                  "point_ids": face_point_ids, "barycentric_exact": barycentric,
                  "local_xyz_exact": local_exact[index * 3:index * 3 + 3]}
        carrier_faces.append({"carrier_face_id": f"c{index}", "source_face": source_face,
                              "arrangement_record_index": index,
                              "arrangement_record_sha256": av._canonical_json_sha256(record),
                              "arrangement_record": record, "point_ids": face_point_ids,
                              "barycentric_exact": barycentric})

    owner = {"body_index": 23, "capture_receipt_sha256": "e" * 64,
             "position_m": [0.0, 0.0, 0.0],
             "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}
    surfaces, sides = [], []
    for side_index, key in enumerate(((51012, 12), (51010, 323))):
        gids = np.arange((100 if side_index == 0 else 900),
                         (100 if side_index == 0 else 900) + len(coordinates), dtype=np.uint64)
        carrier_local_faces = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        if closed:
            apexes = []
            for component in range(3):
                base = coordinates[component * 3:component * 3 + 3]
                apex = np.mean(base, axis=0)
                apex[2] += -2 if side_index == 0 else 2
                apexes.append(apex)
            all_positions = np.concatenate((coordinates.copy(), np.asarray(apexes, dtype=np.float32)), axis=0)
            gids = np.concatenate((gids, np.asarray([1200 + side_index * 10 + i
                                                    for i in range(3)], dtype=np.uint64)))
            faces = []
            carrier_face_indices = {}
            for component in range(3):
                a, b, c, d = component * 3, component * 3 + 1, component * 3 + 2, 9 + component
                carrier_face_indices[len(faces)] = component
                faces.extend([[a, c, b], [a, b, d], [a, d, c], [b, c, d]])
            faces = np.asarray(faces, dtype=np.uint32)
            if side_index == 0:
                faces[:, [1, 2]] = faces[:, [2, 1]].copy()
        else:
            extra_points = (np.asarray([[1, 0, 2 if side_index == 0 else -2]], dtype=np.float32)
                            if not offcarrier else
                            np.asarray([[1, 2, 0], [1, 2, 2 if side_index == 0 else -2]], dtype=np.float32))
            all_positions = np.concatenate((coordinates.copy(), extra_points), axis=0)
            if offcarrier:
                extra_face = [0, 9, 10] if side_index == 0 else [9, 0, 10]
                gids = np.concatenate((gids, np.asarray([1200 + side_index * 10,
                                                        1201 + side_index * 10], dtype=np.uint64)))
            else:
                # Attach the extra test triangle with opposite orientation along
                # the carrier boundary edge on each reciprocal side.
                extra_face = [1, 0, 9] if side_index == 0 else [0, 1, 9]
                gids = np.concatenate((gids, np.asarray([1200 + side_index * 10], dtype=np.uint64)))
            faces = np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8], extra_face], dtype=np.uint32)
            if side_index == 1:
                faces[:3, [1, 2]] = faces[:3, [2, 1]].copy()
            carrier_face_indices = {0: 0, 1: 1, 2: 2}
        row_sha = "b" * 64 if side_index == 0 else "c" * 64
        face_lineage, support_rows, patch_rows = [], [], []
        for face_index, face in enumerate(faces):
            face_gids = [int(gids[int(local)]) for local in face]
            carrier_index = carrier_face_indices.get(face_index)
            if carrier_index is not None:
                member_i, member_sha_i = member, member_sha
                source_face, row_face = 100 + carrier_index, 500 + side_index * 10 + carrier_index
                lineage = {"source_member": member_i, "source_member_sha256": member_sha_i,
                           "source_face": source_face, "source_row_face": row_face,
                           "source_row_payload_sha256": row_sha,
                           "owner_stable_id": key[1], "source_parent_face": source_face}
                if malformed_ancestry and side_index == 1 and face_index == 1:
                    lineage["source_face"] = 999
            else:
                member_i, member_sha_i = "FJ9000", "d" * 64
                source_face, row_face = 900 + side_index, 700 + side_index
                lineage = {"source_member": member_i, "source_member_sha256": member_sha_i,
                           "source_face": source_face, "source_row_face": row_face,
                           "source_row_payload_sha256": row_sha,
                           "owner_stable_id": key[1], "source_parent_face": source_face}
            face_lineage.append(lineage)
            if any(gid in set(int(value) for value in gids[:9]) for gid in face_gids):
                support_rows.append({"face_index": face_index, "source_member": member_i,
                                     "source_member_sha256": member_sha_i,
                                     "source_face": source_face, "source_row_face": row_face,
                                     "source_row_payload_sha256": row_sha,
                                     "ordered_capture_gids": face_gids})
            if carrier_index is not None:
                ids = point_ids[carrier_index * 3:carrier_index * 3 + 3]
                if side_index == 1:
                    ids = [ids[0], ids[2], ids[1]]
                patch_rows.append({"carrier_face_id": f"c{carrier_index}",
                                   "face_index": face_index, "source_member": member_i,
                                   "source_member_sha256": member_sha_i,
                                   "source_face": source_face, "source_row_face": row_face,
                                   "lineage_face_id": row_face, "ordered_point_ids": ids,
                                   "ordered_capture_gids": face_gids})
        vertex_lineage = {int(gid): {"source_gid": int(gid), "source_parent_gids": [int(gid)]}
                          for gid in gids}
        surface = prepare_surface(key, all_positions, faces, source_vertex_ids=gids,
                                  source_face_lineage=face_lineage,
                                  source_vertex_lineage=vertex_lineage,
                                  rigid_owner_proof=owner)
        surfaces.append(surface)
        sides.append({"identity": list(key), "source_row_payload_sha256": row_sha,
                      "surface_hashes": av._source_patch_surface_hashes(surface),
                      "point_bindings": [{"point_id": point_id,
                                          "capture_gids": [int(gids[index])],
                                          "world_float32_bits": av._patch_float32_bits(coordinates[index])}
                                         for index, point_id in enumerate(point_ids)],
                      "patch_faces": patch_rows, "support_faces": support_rows})
    pins = {"source_obj_sha256": "f" * 64, "arrangement_sha256": "1" * 64,
            "arrangement_certificate_sha256": "2" * 64,
            "first_payload_sha256": "b" * 64, "second_payload_sha256": "c" * 64,
            "capture_pack_sha256": "3" * 64, "capture_receipt_sha256": "4" * 64}
    declaration = seal_source_patch_complex({
        "schema": "numi.human.reciprocal-source-patch-complex.v1",
        "identities": [list(surface.key) for surface in surfaces],
        "source_member": member, "source_member_sha256": member_sha,
        "input_pins": pins, "expected_component_count": 3,
        "points": points, "source_faces": source_faces, "carrier_faces": carrier_faces,
        "first": sides[0], "second": sides[1],
    })
    return surfaces[0], surfaces[1], declaration




def seal_source_patch_fragment_complex(value):
    payload = {key: item for key, item in value.items()
               if key != "source_patch_complex_sha256"}
    value["source_patch_complex_sha256"] = av._canonical_json_sha256(payload)
    return value


def source_patch_fragment_case(*, corrupt_barycentric=False):
    def pair(value):
        value = Fraction(value)
        return [str(value.numerator), str(value.denominator)]

    def hex_pair(value):
        value = Fraction(value)
        return [hex(value.numerator), hex(value.denominator)]

    point_xyz = {
        "A": (Fraction(3, 4), Fraction(1, 4), Fraction(0)),
        "B": (Fraction(1, 4), Fraction(3, 4), Fraction(0)),
        "C": (Fraction(1, 4), Fraction(1, 4), Fraction(0)),
        "D": (Fraction(3, 4), Fraction(3, 4), Fraction(0)),
    }
    points = [{
        "point_id": point_id,
        "local_xyz_exact": [pair(value) for value in point_xyz[point_id]],
        "local_float32_bits": av._patch_float32_bits(
            [float(value) for value in point_xyz[point_id]]),
    } for point_id in ("A", "B", "C", "D")]
    member_sha, parent_sha, current_sha = "a" * 64, "b" * 64, "c" * 64
    input_pins = {
        "current_payload_sha256": current_sha,
        "parent_payload_sha256": parent_sha,
        "intermediate_payload_sha256": "d" * 64,
        "arrangement_sha256": "e" * 64,
        "arrangement_certificate_sha256": "f" * 64,
        "ancestry_first_sha256": "1" * 64,
        "ancestry_second_sha256": "2" * 64,
        "capture_pack_sha256": "3" * 64,
        "capture_receipt_sha256": "4" * 64,
        "source_obj_sha256": "5" * 64,
    }
    rows = [
        {
            "source_face": 825,
            "source_vertex_ids": [408, 411, 415],
            "source_row_face": 936,
            "parent_ordered_source_vertex_ids": [411, 415, 408],
            "parent_vertices_exact": [
                [[*pair(1)], [*pair(0)], [*pair(0)]],
                [[*pair(0)], [*pair(1)], [*pair(0)]],
                [[*pair(0)], [*pair(0)], [*pair(0)]],
            ],
            "arrangement_record_index": 940,
            "arrangement_point_ids": ["A", "B", "C"],
            "barycentric_exact": [
                [pair(Fraction(3, 4)), pair(Fraction(1, 4)), pair(0)],
                [pair(Fraction(1, 4)), pair(Fraction(3, 4)), pair(0)],
                [pair(Fraction(1, 4)), pair(Fraction(1, 4)), pair(Fraction(1, 2))],
            ],
        },
        {
            "source_face": 826,
            "source_vertex_ids": [411, 398, 415],
            "source_row_face": 937,
            "parent_ordered_source_vertex_ids": [398, 415, 411],
            "parent_vertices_exact": [
                [[*pair(1)], [*pair(1)], [*pair(0)]],
                [[*pair(0)], [*pair(1)], [*pair(0)]],
                [[*pair(1)], [*pair(0)], [*pair(0)]],
            ],
            "arrangement_record_index": 945,
            "arrangement_point_ids": ["D", "B", "A"],
            "barycentric_exact": [
                [pair(Fraction(1, 2)), pair(Fraction(1, 4)), pair(Fraction(1, 4))],
                [pair(0), pair(Fraction(3, 4)), pair(Fraction(1, 4))],
                [pair(0), pair(Fraction(1, 4)), pair(Fraction(3, 4))],
            ],
        },
    ]
    for row in rows:
        face_xyz = [point_xyz[point_id] for point_id in row["arrangement_point_ids"]]
        record = {
            "source": "medulla343",
            "source_face": row["source_row_face"],
            "other_location": "inside",
            "vertices": [[hex_pair(value) for value in point] for point in face_xyz],
        }
        row["source_row_payload_sha256"] = parent_sha
        row["arrangement_record"] = record
        row["arrangement_record_sha256"] = av._canonical_json_sha256(record)
    if corrupt_barycentric:
        rows[0]["barycentric_exact"][0][0] = pair(Fraction(1, 2))

    owner = {
        "body_index": 23, "capture_receipt_sha256": "9" * 64,
        "position_m": [0.0, 0.0, 0.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    xyz = np.asarray([[float(value) for value in point_xyz[key]]
                      for key in ("A", "B", "C", "D")], dtype=np.float32)
    face_sets = [
        np.asarray([[0, 1, 2], [3, 1, 0]], dtype=np.uint32),
        np.asarray([[0, 2, 1], [3, 0, 1]], dtype=np.uint32),
    ]
    keys = ((51010, 343), (51010, 374))
    gids_by_side = ([100, 101, 102, 103], [900, 901, 902, 903])
    surfaces, side_records = [], []
    identity_bary = [
        [["1", "1"], ["0", "1"], ["0", "1"]],
        [["0", "1"], ["1", "1"], ["0", "1"]],
        [["0", "1"], ["0", "1"], ["1", "1"]],
    ]
    for side_index, key in enumerate(keys):
        gids = gids_by_side[side_index]
        faces = face_sets[side_index]
        lineages = []
        for face_index, source in enumerate(rows):
            raw_face, row_face = source["source_face"], source["source_row_face"]
            if side_index == 0:
                lineage = {
                    "operation": "exact_arrangement_medulla_fragment_outside_csf",
                    "source_member": "FJ1769", "source_member_sha256": member_sha,
                    "source_face": 13398 + face_index, "owner_stable_id": key[1],
                    "corner_barycentric_exact": copy.deepcopy(identity_bary),
                    "parent_row_face_lineage": {
                        "source_member": "FJ1769", "source_member_sha256": member_sha,
                        "source_face": raw_face, "source_row_face": row_face,
                        "source_row_payload_sha256": parent_sha, "owner_stable_id": key[1],
                    },
                }
            else:
                lineage = {
                    "operation": "reversed_medulla343_priority_interface",
                    "source_member": "FJ1769", "source_member_sha256": member_sha,
                    "source_face": raw_face, "source_row_face": row_face,
                    "source_row_payload_sha256": parent_sha, "owner_stable_id": key[1],
                }
            lineages.append(lineage)
        surface = prepare_surface(
            key, xyz, faces, source_vertex_ids=np.asarray(gids, dtype=np.uint64),
            source_face_lineage=lineages, source_vertex_lineage={}, rigid_owner_proof=owner)
        surfaces.append(surface)
        point_bindings = [{
            "point_id": point_id, "capture_gids": [gids[point_index]],
            "world_float32_bits": av._patch_float32_bits(xyz[point_index]),
        } for point_index, point_id in enumerate(("A", "B", "C", "D"))]
        patch_rows = []
        support_rows = []
        for face_index, source in enumerate(rows):
            ordered_points = (
                [["A", "B", "C"], ["D", "B", "A"]][face_index] if side_index == 0 else
                [["A", "C", "B"], ["D", "A", "B"]][face_index])
            face_gids = [gids[index] for index in faces[face_index]]
            patch_rows.append({
                "face_index": face_index, "source_face": source["source_face"],
                "source_row_face": source["source_row_face"],
                "lineage_face_id": 13398 + face_index if side_index == 0 else source["source_face"],
                "ordered_point_ids": ordered_points, "ordered_capture_gids": face_gids,
            })
            support_rows.append({
                "face_index": face_index, "source_member": "FJ1769",
                "source_member_sha256": member_sha,
                "source_face": source["source_face"], "source_row_face": source["source_row_face"],
                "source_row_payload_sha256": parent_sha, "ordered_capture_gids": face_gids,
            })
        side_records.append({
            "identity": list(key), "source_row_payload_sha256": current_sha,
            "surface_hashes": av._source_patch_surface_hashes(surface),
            "point_bindings": point_bindings, "patch_faces": patch_rows,
            "support_faces": support_rows,
        })
    declaration = seal_source_patch_fragment_complex({
        "schema": "numi.human.reciprocal-source-patch-fragment-complex.v1",
        "identities": [list(key) for key in keys],
        "source_member": "FJ1769", "source_member_sha256": member_sha,
        "input_pins": input_pins, "points": points, "source_faces": rows,
        "first": side_records[0], "second": side_records[1],
    })
    return surfaces[0], surfaces[1], declaration


class RestingAnatomyVerifierTests(unittest.TestCase):
    def test_unused_source_slot_may_retain_zero_normal(self):
        positions = np.vstack((tetra(), np.asarray([[9, 9, 9]], dtype=np.float32)))
        normals = np.vstack((np.tile(np.asarray([[1, 0, 0]], dtype=np.float32), (4, 1)),
                             np.zeros((1, 3), dtype=np.float32)))
        surface = prepare_surface((51011, 701), positions, FACES, normals=normals)
        np.testing.assert_array_equal(surface.normals, normals)
        self.assertFalse(np.any(np.isfinite(surface.normals[4]) & (surface.normals[4] != 0)))

    def test_referenced_zero_normal_is_rejected(self):
        normals = np.tile(np.asarray([[1, 0, 0]], dtype=np.float32), (4, 1))
        normals[0] = 0
        with self.assertRaisesRegex(ValueError, "referenced emitted normals are not unit"):
            prepare_surface((51011, 702), tetra(), FACES, normals=normals)

    def test_source_patch_fragment_complex_accepts_nested_barycentric_reciprocal_patch(self):
        first, second, declaration = source_patch_fragment_case()
        result = audit_surface_pair(
            first, second, intended_source_patch_complexes=[declaration])
        self.assertTrue(result["passed"])
        self.assertEqual(result["unexpected_triangle_pairs"], [])
        self.assertGreater(len(result["declared_source_patch_complex_contacts"]), 0)

    def test_source_patch_fragment_complex_rejects_resealed_wrong_barycentric_map(self):
        first, second, declaration = source_patch_fragment_case(corrupt_barycentric=True)
        with self.assertRaisesRegex(ValueError, "barycentric"):
            av._prepare_source_patch_complex(first, second, declaration)



    def test_source_patch_fragment_complex_rejects_stale_arrangement_record(self):
        first, second, declaration = source_patch_fragment_case()
        declaration["source_faces"][0]["arrangement_record"]["vertices"][0][0][0] = "0x1"
        seal_source_patch_fragment_complex(declaration)
        with self.assertRaisesRegex(ValueError, "arrangement record is stale"):
            av._prepare_source_patch_complex(first, second, declaration)

    def test_source_patch_fragment_complex_rejects_missing_coordinate_alias(self):
        first, second, declaration = source_patch_fragment_case()
        positions = np.vstack((first.positions, first.positions[0]))
        faces = first.faces.copy()
        faces[0, 0] = len(first.positions)
        gids = np.append(first.source_vertex_ids, 104).astype(np.uint64)
        first = prepare_surface(
            first.key, positions, faces, source_vertex_ids=gids,
            source_face_lineage=first.source_face_lineage,
            source_vertex_lineage={100: {"source_gid": 100}, 104: {"source_gid": 104}},
            rigid_owner_proof={
                "body_index": 23, "capture_receipt_sha256": "9" * 64,
                "position_m": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            })
        declaration["first"]["surface_hashes"] = av._source_patch_surface_hashes(first)
        seal_source_patch_fragment_complex(declaration)
        with self.assertRaisesRegex(ValueError, "omits or invents exact-coordinate alias"):
            av._prepare_source_patch_complex(first, second, declaration)

    def test_bound_source_facet_projection_requires_full_namespace_correspondence(self):
        from numilab_human.resting_anatomy_verify import retain_bound_source_faces
        p=tetra();n=np.ones_like(p);g=np.arange(1000,1004,dtype=np.uint64)
        old=(p,FACES,n,g)
        projected,proof=retain_bound_source_faces((51010,313),old,FACES,200,[0,2,3])
        self.assertIs(projected[0],p);self.assertIs(projected[2],n);self.assertIs(projected[3],g)
        np.testing.assert_array_equal(projected[1],FACES[[0,2,3]])
        self.assertEqual(proof['source_capture_vertex_namespace_offset'],800)
        self.assertEqual(proof['source_face_ancestry'],[0,2,3])
        changed=FACES.copy();changed[-1]=changed[-1,[1,0,2]]
        with self.assertRaisesRegex(ValueError,'mapping disagrees'):
            retain_bound_source_faces((51010,313),(p,changed,n,g),FACES,200,[0,2,3])
        for ancestry in ([0,0],[2,1],[-1],[4]):
            with self.subTest(ancestry=ancestry),self.assertRaises(ValueError):
                retain_bound_source_faces((51010,313),old,FACES,200,ancestry)

    def test_metric_contacts_preserve_exact_float32_units_and_unique_points(self):
        a = prepare_surface((51010, 323), np.array([
            [.125, .25, .5], [.375, .25, .5], [.125, .5, .5],
        ], dtype=np.float32), np.array([[0, 1, 2]], dtype=np.uint32))
        b = prepare_surface((51010, 345), np.array([
            [.125, .25, .5], [.125, .125, .75], [0., .25, .75],
        ], dtype=np.float32), np.array([[0, 1, 2]], dtype=np.uint32))
        point = (Fraction(1, 8), Fraction(1, 4), Fraction(1, 2))
        self.assertEqual(a.triangle_meters(0)[0], point)
        self.assertEqual(a.intersection_points_meters(b, 0, 0), (point,))
        self.assertEqual(b.intersection_points_meters(a, 0, 0), (point,))

    def test_metric_transverse_segment_retains_nonbinary_rational_intersection(self):
        a = prepare_surface((51010, 323), np.array([
            [0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
        ], dtype=np.float32), np.array([[0, 1, 2]], dtype=np.uint32))
        b = prepare_surface((51010, 345), np.array([
            [0., 0., -1.], [1., 0., 2.], [0., 1., 2.],
        ], dtype=np.float32), np.array([[0, 1, 2]], dtype=np.uint32))
        self.assertEqual(a.intersection_points_meters(b, 0, 0), (
            (Fraction(0), Fraction(1, 3), Fraction(0)),
            (Fraction(1, 3), Fraction(0), Fraction(0)),
        ))

    def test_metric_face_ids_reject_coercion_and_out_of_range(self):
        surface = prepare_surface((51011, 7), tetra(), FACES)
        for face in (True, False, 0., "0", -1, len(FACES), np.int64(0)):
            with self.subTest(face=face), self.assertRaises(ValueError):
                surface.triangle_meters(face)
            with self.subTest(face=face), self.assertRaises(ValueError):
                surface.intersection_points_meters(surface, 0, face)
        with self.assertRaises(TypeError):
            surface.intersection_points_meters(None, 0, 0)

    def test_identity_requires_semantic_not_untyped_stable_id(self):
        with self.assertRaises(ValueError):
            typed_key(313)
        self.assertNotEqual(typed_key((51010, 313)), typed_key((51007, 1)))
        with self.assertRaises(ValueError):
            typed_key((51011, True))

    def test_no_implicit_float32_conversion(self):
        with self.assertRaisesRegex(ValueError, "emitted Float32"):
            prepare_surface((51011, 7), tetra().astype(np.float64), FACES)
        with self.assertRaises(ValueError):
            prepare_surface((51011, 7), tetra(), FACES.astype(float))

    def test_exact_source_seam_is_diagnostic_quotient_not_mesh_mutation(self):
        p = np.concatenate([tetra(), tetra()[0:1]])
        f = FACES.copy()
        f[0, 0] = 4
        surface = prepare_surface((51011, 7), p, f)
        self.assertFalse(surface.raw_topology["closed_oriented_manifold_candidate"])
        self.assertTrue(surface.closed)
        self.assertEqual(surface.diagnostics()["exact_duplicate_vertex_count"], 1)
        self.assertEqual(surface.self_audit()["count"], 0)
        np.testing.assert_array_equal(surface.positions, p)
        np.testing.assert_array_equal(surface.faces, f)

    def test_unused_ancestry_slot_is_not_a_solid_boundary(self):
        p = np.concatenate([tetra(), np.array([[8, 9, 10]], dtype=np.float32)])
        surface = prepare_surface((51006, 132), p, FACES)
        self.assertFalse(surface.quotient_topology["closed_oriented_manifold_candidate"])
        self.assertEqual(surface.quotient_topology["unused_vertex_ids"], [4])
        self.assertTrue(surface.closed)
        self.assertEqual(surface.self_audit()["count"], 0)
        self.assertEqual(surface.diagnostics()["exact_coordinate_quotient_topology"]["unused_vertex_ids_count"], 1)
        np.testing.assert_array_equal(surface.positions, p)
        inner = prepare_surface((51011, 525), tetra(.1, .1), FACES)
        result = audit_surface_pair(inner, surface)
        self.assertFalse(result["passed"])
        self.assertEqual(result["containment"][0]["status"], "checked_exact_rational_metres")

    def test_unused_slots_do_not_hide_an_open_boundary(self):
        p = np.concatenate([tetra(), np.array([[8, 9, 10]], dtype=np.float32)])
        surface = prepare_surface((51006, 132), p, FACES[:3])
        self.assertFalse(surface.closed)

    def test_seam_identification_does_not_hide_real_crossing(self):
        p = np.concatenate([tetra(), tetra(offset=.2)])
        f = np.concatenate([FACES, FACES + 4])
        surface = prepare_surface((51011, 7), p, f)
        self.assertGreater(surface.self_audit()["count"], 0)

    def test_nonembedded_target_retains_actual_self_witnesses(self):
        target = prepare_surface((51010, 313), np.concatenate([tetra(), tetra(offset=.2)]),
                                 np.concatenate([FACES, FACES + 4]))
        source = prepare_surface((51011, 7), tetra(.03, .05), FACES)
        result = audit_surface_pair(source, target)
        self.assertEqual(result["triangle_pair_count"], 0)
        self.assertFalse(result["passed"])
        row = result["containment"][0]
        self.assertEqual(row["status"], "not_checked_nonembedded_target")
        self.assertGreater(row["target_self_audit"]["count"], 0)
        self.assertEqual(row["target_self_audit"], target.self_audit())
        self.assertEqual(row["target_preparation"]["identity"], [51010, 313])

    def test_contained_mesh_fails_without_triangle_crossings(self):
        outer = prepare_surface((51004, 15), tetra(), FACES)
        inner = prepare_surface((51005, 132), tetra(.1, .1), FACES)
        result = audit_surface_pair(inner, outer)
        self.assertEqual(result["triangle_pair_count"], 0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["containment"][0]["status"], "checked_exact_rational_metres")
        self.assertEqual(len(result["containment"][0]["inside_component_witnesses"]), 1)

    def test_intended_skin_enclosure_preserves_inside_witness(self):
        outer = prepare_surface((51007, 1), tetra(), FACES)
        inner = prepare_surface((51011, 525), tetra(.1, .1), FACES)
        role = [(inner.key, outer.key)]
        result = audit_surface_pair(inner, outer, intended_enclosures=role)
        self.assertTrue(result["passed"])
        self.assertTrue(result["containment"][0]["declared_intended_enclosure"])
        self.assertEqual(len(result["containment"][0]["inside_component_witnesses"]), 1)
        self.assertTrue(audit_surface_pair(outer, inner, intended_enclosures=role)["passed"])

    def test_disjoint_mesh_cannot_satisfy_declared_anatomical_enclosure(self):
        outer = prepare_surface((51007, 1), tetra(), FACES)
        escaped = prepare_surface((51005, 121), tetra(.1, 2.), FACES)
        self.assertTrue(audit_surface_pair(escaped, outer)["passed"])
        for first, second in ((escaped, outer), (outer, escaped)):
            result = audit_surface_pair(first, second,
                                        intended_enclosures=[(escaped.key, outer.key)])
            self.assertFalse(result["passed"])
            self.assertTrue(result["aabb_disjoint"])
            self.assertTrue(result["declared_enclosure_outside_aabb"])
            self.assertEqual(result["triangle_pair_count"], 0)

    def test_changed_surface_dispatch_rejects_disjoint_declared_enclosure(self):
        outer = prepare_surface((51007, 1), tetra(), FACES)
        escaped = prepare_surface((51005, 121), tetra(.1, 2.), FACES)
        surfaces = {outer.key: outer, escaped.key: escaped}
        for changed, target in ((escaped, outer), (outer, escaped)):
            ordinary = av.audit_changed_surfaces(
                surfaces, [changed.key], targets=[target.key])
            self.assertTrue(ordinary["passed"])
            result = av.audit_changed_surfaces(
                surfaces, [changed.key], targets=[target.key],
                intended_enclosures=[(escaped.key, outer.key)])
            self.assertFalse(result["passed"])
            self.assertEqual(result["failed_stage"], "nearby_pair")
            self.assertTrue(result["rows"][-1]["declared_enclosure_outside_aabb"])
            self.assertEqual(result["rows"][-1]["intended_enclosures"],
                             [[list(escaped.key), list(outer.key)]])

    def test_intended_enclosure_direction_is_not_symmetric(self):
        outer = prepare_surface((51007, 1), tetra(), FACES)
        inner = prepare_surface((51011, 525), tetra(.1, .1), FACES)
        self.assertFalse(audit_surface_pair(
            inner, outer, intended_enclosures=[(outer.key, inner.key)])["passed"])

    def test_intended_enclosure_never_exempts_boundary_crossings(self):
        outer = prepare_surface((51007, 1), tetra(), FACES)
        crossing = prepare_surface((51011, 525), tetra(.5, .2), FACES)
        result = audit_surface_pair(
            crossing, outer, intended_enclosures=[(crossing.key, outer.key)])
        self.assertGreater(result["triangle_pair_count"], 0)
        self.assertFalse(result["passed"])

    def test_invalid_enclosure_identity_rejects(self):
        outer = prepare_surface((51007, 1), tetra(), FACES)
        inner = prepare_surface((51011, 525), tetra(.1, .1), FACES)
        with self.assertRaises(ValueError):
            audit_surface_pair(inner, outer, intended_enclosures=[(inner.key, (51007, 2))])
        with self.assertRaises(ValueError):
            audit_surface_pair(inner, outer, intended_enclosures=[(inner.key, inner.key)])

    def test_open_surface_is_not_a_closed_solid(self):
        outer = prepare_surface((51007, 1), tetra(), FACES[:3])
        inner = prepare_surface((51005, 132), tetra(.1, .1), FACES)
        result = audit_surface_pair(inner, outer)
        self.assertEqual(result["containment"][0]["status"], "not_checked_open_target")

    def test_source_patch_complex_accepts_only_reciprocal_faces_and_supported_patch_contacts(self):
        first, second, declaration = source_patch_complex_case()
        result = audit_surface_pair(
            first, second, intended_source_patch_complexes=[declaration])
        self.assertTrue(result["passed"], result["unexpected_triangle_pairs"])
        self.assertEqual(result["unexpected_triangle_pairs"], [])
        contacts = result["declared_source_patch_complex_contacts"]
        self.assertTrue(contacts)
        self.assertTrue(all(row["proof"] in {
            "exact_reciprocal_source_patch_face",
            "exact_contact_contained_in_supported_source_patch_corner_or_edge",
        } for row in contacts))
        changed_result = av.audit_changed_surfaces(
            {first.key: first, second.key: second}, [first.key], targets=[second.key],
            source_patch_complex_interfaces={(first.key, second.key): [declaration]})
        self.assertTrue(changed_result["passed"], changed_result["rows"])

    def test_multiface_source_patch_carrier_accepts_full_segment_and_requires_closed_solids(self):
        first, second, declaration = source_patch_carrier_case()
        context = av._prepare_source_patch_complex(first, second, declaration)
        points = first.intersection_points_meters(second, 3, 3)
        self.assertEqual(len(points), 2)
        accepted, reason = av._source_patch_complex_contact(first, second, 3, 3, context)
        self.assertTrue(accepted, reason)
        self.assertEqual(reason, "exact_segment_contact_fully_covered_by_authenticated_carrier")
        self.assertEqual(context["carrier_topology"]["component_count"], 3)
        self.assertEqual(context["carrier_topology"]["component_face_counts"], [1, 1, 1])
        result = audit_surface_pair(first, second, intended_source_patch_complexes=[declaration])
        self.assertFalse(result["passed"], "an open carrier fixture cannot pass the solid gate")
        self.assertFalse(result["source_patch_carrier_solid_interior_checked"])

    def test_multiface_carrier_closed_solids_pass_exact_interior_gate(self):
        first, second, declaration = source_patch_carrier_case(closed=True)
        result = audit_surface_pair(
            first, second, intended_source_patch_complexes=[declaration])
        self.assertTrue(result["passed"], result["unexpected_triangle_pairs"])
        self.assertTrue(result["source_patch_carrier_solid_interior_checked"])
        self.assertTrue(all(row["status"] == "checked_exact_rational_metres"
                            for row in result["containment"]))

    def test_carrier_topology_accepts_22_faces_in_three_declared_components(self):
        point_xyz, faces = {}, []
        circle = [(10, 0), (9, 5), (5, 9), (0, 10), (-5, 9), (-9, 5),
                  (-10, 0), (-9, -5), (-5, -9), (0, -10), (5, -9), (9, -5)]
        components = (list(range(12)), [0, 1, 3, 4, 6, 7, 9, 10],
                      [0, 2, 3, 5, 6, 8, 9, 11])
        for component_index, polygon_indices in enumerate(components):
            ids = []
            for polygon_index in polygon_indices:
                point_id = f"c{component_index}-p{polygon_index}"
                x, y = circle[polygon_index]
                point_xyz[point_id] = (Fraction(x + 100 * component_index),
                                       Fraction(y), Fraction(0))
                ids.append(point_id)
            for index in range(1, len(ids) - 1):
                faces.append((ids[0], ids[index], ids[index + 1]))
        topology = av._carrier_topology(point_xyz, faces, expected_components=3)
        self.assertEqual(len(faces), 22)
        self.assertEqual(topology["component_count"], 3)
        self.assertEqual(topology["component_face_counts"], [6, 6, 10])
        self.assertEqual(topology["boundary_edge_count"], 28)

    def test_carrier_query_cache_matches_uncached_exact_queries(self):
        first, second, declaration = source_patch_carrier_case()
        context = av._prepare_source_patch_complex(first, second, declaration)
        triangles = context["carrier_triangles"]

        points = (
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0), Fraction(1)),
            (Fraction(5), Fraction(0), Fraction(0)),
        )
        for point in points:
            self.assertEqual(
                av._cached_exact_point_on_carrier(point, context),
                av._exact_point_on_carrier(point, triangles))
        point_cache = context["_carrier_query_cache"]["point"]
        self.assertEqual(len(point_cache), len(points))
        for point in points:
            self.assertEqual(av._cached_exact_point_on_carrier(point, context),
                             av._exact_point_on_carrier(point, triangles))
        self.assertEqual(len(point_cache), len(points))

        segments = (
            ((Fraction(0), Fraction(0), Fraction(0)),
             (Fraction(3), Fraction(0), Fraction(0))),
            ((Fraction(0), Fraction(0), Fraction(0)),
             (Fraction(21), Fraction(0), Fraction(0))),
            ((Fraction(1), Fraction(0), Fraction(-1)),
             (Fraction(1), Fraction(0), Fraction(1))),
        )
        expected = (True, False, False)
        for (start, end), expected_result in zip(segments, expected):
            uncached = av._exact_segment_on_carrier(start, end, triangles)
            self.assertEqual(uncached, expected_result)
            self.assertEqual(av._cached_exact_segment_on_carrier(start, end, context), uncached)
            self.assertEqual(av._cached_exact_segment_on_carrier(end, start, context), uncached)
        segment_cache = context["_carrier_query_cache"]["segment"]
        self.assertEqual(len(segment_cache), len(segments))

        direct_lookup = {}
        for first_face, second_face, carrier_id, triangle in context["patch_face_pairs"]:
            direct_lookup.setdefault((first_face, second_face), (carrier_id, triangle))
        self.assertEqual(context["carrier_face_pair_lookup"], direct_lookup)

    def test_multiface_carrier_rejects_gap_even_when_endpoints_and_midpoint_lie_on_it(self):
        first, second, declaration = source_patch_carrier_case()
        context = av._prepare_source_patch_complex(first, second, declaration)
        start = (Fraction(0), Fraction(0), Fraction(0))
        middle = (Fraction(21, 2), Fraction(0), Fraction(0))
        end = (Fraction(21), Fraction(0), Fraction(0))
        triangles = context["carrier_triangles"]
        self.assertTrue(av._exact_point_on_carrier(start, triangles))
        self.assertTrue(av._exact_point_on_carrier(middle, triangles))
        self.assertTrue(av._exact_point_on_carrier(end, triangles))
        self.assertFalse(av._exact_segment_on_carrier(start, end, triangles))

    def test_multiface_carrier_rejects_off_carrier_crossing(self):
        first, second, declaration = source_patch_carrier_case(offcarrier=True)
        context = av._prepare_source_patch_complex(first, second, declaration)
        points = first.intersection_points_meters(second, 3, 3)
        self.assertEqual(len(points), 2)
        accepted, reason = av._source_patch_complex_contact(first, second, 3, 3, context)
        self.assertFalse(accepted)
        self.assertEqual(reason, "contact_is_not_fully_covered_by_authenticated_carrier")

    def test_multiface_carrier_accepts_exact_non_float32_local_barycentric_point(self):
        first, second, declaration = source_patch_carrier_case(rational_local=True)
        local = tuple(Fraction(int(pair[0]), int(pair[1]))
                      for pair in declaration["points"][0]["local_xyz_exact"])
        capture = tuple(Fraction(int(pair[0]), int(pair[1]))
                        for pair in declaration["points"][0]["capture_xyz_exact"])
        self.assertEqual(local, (Fraction(1, 2), Fraction(1, 3), Fraction(0)))
        self.assertNotEqual(local, capture)
        self.assertEqual(
            declaration["points"][0]["local_float32_bits"],
            [0x3f000000, 0x3eaaaaab, 0x00000000])
        context = av._prepare_source_patch_complex(first, second, declaration)
        self.assertEqual(context["carrier_topology"]["component_count"], 3)

    def test_multiface_carrier_rational_float32_rounding_is_exact_ties_to_even(self):
        self.assertEqual(av._patch_round_fraction_float32_bits(
            Fraction(1) + Fraction(1, 1 << 24)), 0x3f800000)
        self.assertEqual(av._patch_round_fraction_float32_bits(
            Fraction(1) + Fraction(3, 1 << 24)), 0x3f800002)
        self.assertEqual(av._patch_round_fraction_float32_bits(
            Fraction(-1) - Fraction(1, 1 << 24)), 0xbf800000)

    def test_multiface_carrier_rejects_changed_parent_or_barycentric_reconstruction(self):
        first, second, declaration = source_patch_carrier_case(rational_local=True)
        declaration["source_faces"][0]["source_vertices_exact"][1][0] = ["4", "1"]
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "ancestry does not reconstruct"):
            av._prepare_source_patch_complex(first, second, declaration)

        first, second, declaration = source_patch_carrier_case(rational_local=True)
        carrier = declaration["carrier_faces"][0]
        carrier["barycentric_exact"][0] = [["1", "2"], ["1", "3"], ["1", "6"]]
        carrier["arrangement_record"]["barycentric_exact"] = carrier["barycentric_exact"]
        carrier["arrangement_record_sha256"] = av._canonical_json_sha256(
            carrier["arrangement_record"])
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "ancestry does not reconstruct"):
            av._prepare_source_patch_complex(first, second, declaration)

    def test_multiface_carrier_rejects_changed_declared_and_current_float32_bits(self):
        first, second, declaration = source_patch_carrier_case(rational_local=True)
        declaration["points"][0]["local_float32_bits"][0] ^= 1
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "exact rational Float32 rounding"):
            av._prepare_source_patch_complex(first, second, declaration)

        first, second, declaration = source_patch_carrier_case(rational_local=True)
        declaration["points"][0]["capture_float32_bits"][0] ^= 1
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "exact rational Float32 rounding"):
            av._prepare_source_patch_complex(first, second, declaration)

        first, second, declaration = source_patch_carrier_case(rational_local=True)
        first.positions[0, 0] = np.nextafter(np.float32(0.5), np.float32(2.0))
        declaration["first"]["surface_hashes"] = av._source_patch_surface_hashes(first)
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "point differs from captured GID"):
            av._prepare_source_patch_complex(first, second, declaration)

    def test_multiface_carrier_capture_coordinate_must_remain_on_float32_lattice(self):
        first, second, declaration = source_patch_carrier_case(rational_local=True)
        declaration["points"][0]["capture_xyz_exact"][1] = ["1", "3"]
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "capture point is not its exact Float32 lattice"):
            av._prepare_source_patch_complex(first, second, declaration)

    def test_multiface_carrier_rejects_wrong_component_count_and_malformed_parent_ancestry(self):
        first, second, declaration = source_patch_carrier_case()
        declaration["expected_component_count"] = 2
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "component count"):
            av._prepare_source_patch_complex(first, second, declaration)

        first, second, declaration = source_patch_carrier_case(malformed_ancestry=True)
        with self.assertRaisesRegex(ValueError, "support ancestry"):
            av._prepare_source_patch_complex(first, second, declaration)

    def test_source_patch_complex_requires_ancestried_incident_edge_support(self):
        first, second, declaration = source_patch_complex_case(support_edge=True)
        context = av._prepare_source_patch_complex(first, second, declaration)
        points = first.intersection_points_meters(second, 2, 2)
        self.assertEqual(len(points), 2)
        accepted, reason = av._source_patch_complex_contact(first, second, 2, 2, context)
        self.assertTrue(accepted, reason)
        self.assertEqual(
            reason, "exact_contact_contained_in_supported_source_patch_corner_or_edge")

        declaration["first"]["support_faces"] = declaration["first"]["support_faces"][:2]
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "incident face support"):
            av._prepare_source_patch_complex(first, second, declaration)

    def test_source_patch_complex_rejects_off_edge_segment(self):
        first, second, declaration = source_patch_complex_case(off_edge=True)
        context = av._prepare_source_patch_complex(first, second, declaration)
        points = first.intersection_points_meters(second, 2, 2)
        self.assertEqual(len(points), 2)
        accepted, reason = av._source_patch_complex_contact(first, second, 2, 2, context)
        self.assertFalse(accepted)
        self.assertEqual(reason, "contact_point_is_not_on_one_authenticated_patch_edge")

    def test_source_patch_complex_rejects_wrong_face_ancestry_even_when_resealed(self):
        first, second, declaration = source_patch_complex_case()
        first.source_face_lineage[0]["source_member_sha256"] = "f" * 64
        declaration["first"]["surface_hashes"] = av._source_patch_surface_hashes(first)
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "ancestry does not match"):
            av._prepare_source_patch_complex(first, second, declaration)

    def test_source_patch_complex_rejects_mismatched_corner_vertex_lineage(self):
        first, second, declaration = source_patch_complex_case()
        first.source_vertex_lineage[100]["source_vertex_id"] = 99
        declaration["first"]["surface_hashes"] = av._source_patch_surface_hashes(first)
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "corner vertex ancestry"):
            av._prepare_source_patch_complex(first, second, declaration)

    def test_source_patch_complex_barycentric_frame_must_link_to_authenticated_face(self):
        first, second, declaration = source_patch_complex_case()
        lineage = first.source_face_lineage[0]
        lineage["source_face"] = 7777
        lineage["parent_row_face_lineage"] = {
            "output_face": 930, "source_member": "FJ1769",
            "source_face": 825, "source_row_face": 930,
            "source_row_payload_sha256": "b" * 64,
            "owner_stable_id": 343,
        }
        declaration["first"]["surface_hashes"] = av._source_patch_surface_hashes(first)
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "corner ancestry"):
            av._prepare_source_patch_complex(first, second, declaration)

    def test_source_patch_complex_rejects_stale_surface_and_malformed_patch(self):
        first, second, declaration = source_patch_complex_case()
        first.positions[0, 0] = np.float32(-0.125)
        with self.assertRaisesRegex(ValueError, "hashes are stale"):
            av._prepare_source_patch_complex(first, second, declaration)

        first, second, declaration = source_patch_complex_case()
        declaration["source_faces"][1]["source_vertex_ids"][2] = 99
        seal_source_patch_complex(declaration)
        with self.assertRaisesRegex(ValueError, "four-corner complex"):
            av._prepare_source_patch_complex(first, second, declaration)

    def test_declared_shared_edge_requires_exact_identity_lineage_and_owner_map(self):
        a = feature_surface(
            (51006, 132),
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [100, 101, 102],
            kind3_gids={100, 101})
        b = feature_surface(
            (51006, 134),
            [[0, 0, 0], [1, 0, 0], [0, -1, 0]], [[0, 1, 2]], [100, 101, 103],
            kind3_gids={100, 101})
        self.assertFalse(audit_surface_pair(a, b)["passed"])
        result = audit_surface_pair(
            a, b, intended_shared_feature_pairs=[
                feature_declaration(a, b, 0, 0, [100, 101])])
        self.assertTrue(result["passed"])
        self.assertEqual(result["triangle_pair_count"], 1)
        self.assertEqual(result["unexpected_triangle_pairs"], [])
        self.assertEqual(result["declared_interface_feature_contacts"][0]["feature_kind"], "edge")

        wrong_identity = feature_declaration(a, b, 0, 0, [100, 101])
        wrong_identity["second_identity"] = [51006, 135]
        with self.assertRaisesRegex(ValueError, "identities"):
            audit_surface_pair(a, b, intended_shared_feature_pairs=[wrong_identity])

        wrong_row = np.full(21, 1.0 / 21.0, dtype=np.float64)
        wrong_row[0] += 0.01
        wrong_row[1] -= 0.01
        b_wrong = feature_surface(
            (51006, 135),
            [[0, 0, 0], [1, 0, 0], [0, -1, 0]], [[0, 1, 2]], [100, 101, 104],
            kind3_gids={100, 101}, weight_overrides={100: wrong_row})
        rejected = audit_surface_pair(
            a, b_wrong, intended_shared_feature_pairs=[
                feature_declaration(a, b_wrong, 0, 0, [100, 101])])
        self.assertFalse(rejected["passed"])
        self.assertTrue(rejected["rejected_interface_feature_contacts"])
        self.assertIn([0, 0], rejected["unexpected_triangle_pairs"])

    def test_declared_shared_vertex_does_not_hide_a_transversal_segment(self):
        a = feature_surface(
            (51006, 132),
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [200, 201, 202])
        b = feature_surface(
            (51006, 134),
            [[0, 0, 0], [0.5, 0, 0], [0, 0, 1]], [[0, 1, 2]], [200, 203, 204])
        result = audit_surface_pair(
            a, b, intended_shared_feature_pairs=[
                feature_declaration(a, b, 0, 0, [200])])
        self.assertFalse(result["passed"])
        self.assertIn([0, 0], result["unexpected_triangle_pairs"])
        self.assertEqual(result["rejected_interface_feature_contacts"][0]["reason"],
                         "exact_intersection_exceeds_declared_shared_feature")

    def test_coordinate_coincidence_without_shared_gid_is_not_an_interface(self):
        a = feature_surface(
            (51006, 132),
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [300, 301, 302])
        b = feature_surface(
            (51006, 134),
            [[0, 0, 0], [-0.5, 0, 0], [0, 0, 1]], [[0, 1, 2]], [400, 401, 402])
        result = audit_surface_pair(
            a, b, intended_shared_feature_pairs=[
                feature_declaration(a, b, 0, 0, [300])])
        self.assertFalse(result["passed"])
        self.assertIn([0, 0], result["unexpected_triangle_pairs"])
        self.assertEqual(result["rejected_interface_feature_contacts"][0]["reason"],
                         "declared_feature_GIDs_not_shared_by_both_faces")

    def _run_crosswalk_contact(self, first, second, declaration, checker):
        return checker(
            first, second, declaration["first_face_index"], declaration["second_face_index"],
            declaration["feature_gid_pairs"], declaration["source_point_crosswalk"],
            declaration["source_seam_face_pairs"],
            declaration["source_seam_endpoint_source_vertex_id"])

    def test_authenticated_crosswalk_accepts_only_exact_shared_point_with_same_rigid_owner(self):
        first, second, declaration, checker = crosswalk_point_case()
        passed, reason = self._run_crosswalk_contact(first, second, declaration, checker)
        self.assertTrue(passed, reason)
        self.assertEqual(reason, "exact_singleton_at_authenticated_two_face_source_seam_endpoint")
        proof = declaration["source_point_crosswalk"]
        self.assertEqual(proof["seam"]["source_member"], "FJ1769")
        self.assertEqual([row["source_face"] for row in proof["seam"]["source_faces"]], [825, 826])
        self.assertEqual(len(declaration["source_seam_face_pairs"]), 2)
        result = audit_surface_pair(first, second, intended_shared_feature_pairs=[declaration])
        self.assertTrue(any(row["face_pair"] == [2, 2]
                            for row in result["declared_interface_feature_contacts"]))
        self.assertNotIn([2, 2], result["unexpected_triangle_pairs"])

        first, second, declaration, checker = crosswalk_point_case()
        # Both canonical endpoints are independently bound and incident to both
        # reversed source faces, even though this contact exercises endpoint 409.
        self.assertEqual(len(declaration["source_point_crosswalk"]["first"]["vertex_bindings"]), 2)
        self.assertEqual(len(declaration["source_point_crosswalk"]["second"]["vertex_bindings"]), 2)

    def test_authenticated_crosswalk_rejects_coordinate_only_or_wrong_source_ancestry(self):
        first, second, declaration, checker = crosswalk_point_case(source_crosswalk=False)
        declaration["source_point_crosswalk"] = None
        passed, reason = self._run_crosswalk_contact(first, second, declaration, checker)
        self.assertFalse(passed)
        self.assertEqual(reason, "source_point_crosswalk_is_missing_or_invalid")

        first, second, declaration, checker = crosswalk_point_case()
        second.source_face_lineage[2]["source_face"] = 937
        passed, reason = self._run_crosswalk_contact(first, second, declaration, checker)
        self.assertTrue(passed, reason)  # Incident-face ancestry is allowed to differ from the seam member.
        second.source_face_lineage[0]["source_face"] = 937
        passed, reason = self._run_crosswalk_contact(first, second, declaration, checker)
        self.assertFalse(passed)
        self.assertEqual(reason, "seam_face_ancestry_does_not_match_authenticated_source")

        first, second, declaration, checker = crosswalk_point_case(second_endpoint=416)
        passed, reason = self._run_crosswalk_contact(first, second, declaration, checker)
        self.assertFalse(passed)
        self.assertEqual(reason, "source_point_crosswalk_is_missing_or_invalid")

    def test_authenticated_crosswalk_uses_exact_endpoint_without_requiring_bitwise_other_corners(self):
        first, second, declaration, checker = crosswalk_point_case()
        # GPU/CPU owner emission can differ at other source-face corners while
        # the declared endpoint itself remains an exact shared Float32 point.
        second.positions[second.vertex_index(902), 1] += np.float32(2e-7)
        passed, reason = self._run_crosswalk_contact(first, second, declaration, checker)
        self.assertTrue(passed, reason)
        self.assertEqual(reason, "exact_singleton_at_authenticated_two_face_source_seam_endpoint")

    def test_authenticated_crosswalk_rejects_wrong_or_incomplete_two_face_seam(self):
        first, second, declaration, checker = crosswalk_point_case()
        declaration["source_point_crosswalk"]["seam"]["source_faces"][0]["source_vertex_ids"][0] = 412
        self.assertFalse(av._valid_source_point_crosswalk(declaration["source_point_crosswalk"]))

        first, second, declaration, checker = crosswalk_point_case()
        declaration["source_seam_face_pairs"] = [[0, 0]]
        passed, reason = self._run_crosswalk_contact(first, second, declaration, checker)
        self.assertFalse(passed)
        self.assertEqual(reason, "both_reciprocal_source_seam_face_pairs_are_required")

        first, second, declaration, checker = crosswalk_point_case(omit_second_seam_face=True)
        self.assertFalse(av._valid_source_point_crosswalk(declaration["source_point_crosswalk"]))

        first, second, declaration, checker = crosswalk_point_case()
        declaration["source_seam_face_pairs"] = [[0, 0], [0, 0]]
        passed, reason = self._run_crosswalk_contact(first, second, declaration, checker)
        self.assertFalse(passed)
        self.assertEqual(reason, "duplicate_source_seam_face_pair")

    def test_authenticated_crosswalk_rejects_different_owner_and_segment_contact(self):
        first, second, declaration, checker = crosswalk_point_case(second_owner=24)
        passed, reason = self._run_crosswalk_contact(first, second, declaration, checker)
        self.assertFalse(passed)
        self.assertEqual(reason, "accepted_rigid_owner_pose_proofs_differ_or_are_missing")

        first, second, declaration, checker = crosswalk_point_case(transverse_segment=True)
        passed, reason = self._run_crosswalk_contact(first, second, declaration, checker)
        self.assertFalse(passed)
        self.assertEqual(reason, "exact_intersection_exceeds_authenticated_source_seam_endpoint")

    def test_tangent_declared_interface_does_not_hide_solid_containment(self):
        outer_points = np.array([[0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]], dtype=np.float32)
        inner_points = np.array([[0, 0, 0], [.2, .1, .1], [.1, .2, .1], [.1, .1, .2]], dtype=np.float32)
        outer = feature_surface((51006, 132), outer_points, FACES, [500, 501, 502, 503])
        inner = feature_surface((51006, 134), inner_points, FACES, [500, 504, 505, 506])
        declarations = [
            feature_declaration(inner, outer, fi, fo, [500])
            for fi in range(3) for fo in range(3)
        ]
        result = audit_surface_pair(inner, outer, intended_shared_feature_pairs=declarations)
        self.assertEqual(result["unexpected_triangle_pairs"], [])
        self.assertEqual(len(result["declared_interface_feature_contacts"]), 9)
        self.assertFalse(result["passed"])
        self.assertEqual(result["containment"][0]["status"], "checked_exact_rational_metres")
        self.assertEqual(len(result["containment"][0]["inside_component_witnesses"]), 1)

    def test_shared_edge_tangent_solids_pass_but_nested_wedges_fail(self):
        outer_points = np.array(
            [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float32)
        nested_points = np.array(
            [[0, 0, 0], [2, 0, 0], [.3, .3, .3], [.4, .1, .2]], dtype=np.float32)
        outer = feature_surface((51006, 132), outer_points, FACES, [600, 601, 602, 603])
        nested = feature_surface((51006, 134), nested_points, FACES, [600, 601, 604, 605])
        nested_result = audit_surface_pair(
            nested, outer,
            intended_shared_feature_pairs=declarations_for_actual_shared_features(nested, outer))
        self.assertEqual(nested_result["unexpected_triangle_pairs"], [])
        self.assertFalse(nested_result["passed"])
        self.assertEqual(nested_result["containment"][0]["status"], "checked_exact_rational_metres")
        self.assertEqual(len(nested_result["containment"][0]["inside_component_witnesses"]), 1)

        separated_points = np.array(
            [[0, 0, 0], [2, 0, 0], [.3, -1, 0], [.5, -.2, -1]], dtype=np.float32)
        separated = feature_surface((51006, 134), separated_points, FACES, [600, 601, 606, 607])
        tangent_result = audit_surface_pair(
            outer, separated,
            intended_shared_feature_pairs=declarations_for_actual_shared_features(outer, separated))
        self.assertEqual(tangent_result["unexpected_triangle_pairs"], [])
        self.assertTrue(tangent_result["passed"])
        self.assertFalse(audit_surface_pair(
            outer, separated,
            intended_shared_feature_pairs=declarations_for_actual_shared_features(outer, separated),
            intended_enclosures=[(outer.key, separated.key)])["passed"])


    def test_current_native_vessel_port_overlap_is_consumed_only_when_exactly_bound(self):
        first, second, proof, context, pairs = source_bound_vessel_port_case()
        raw = context["source_proof_bytes"]
        result = audit_source_bound_connected_overlap(
            first, second, source_proof_bytes=raw,
            expected_source_proof_sha256=context["expected_source_proof_sha256"],
            source_binding=context["source_binding"],
            expected_source_binding_sha256=context["expected_source_binding_sha256"],
            expected_source_map_sha256=context["expected_source_map_sha256"],
            source_map_bytes=context["source_map_bytes"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["exact_face_pair_count"], len(pairs))
        self.assertTrue(result["matched_source_collar_and_region_passed"])
        self.assertTrue(result["both_material_station_calibers_passed"])

        strict = audit_surface_pair(first, second, intended_face_pairs=pairs)
        self.assertFalse(strict["passed"])
        scoped = audit_surface_pair(
            first, second, intended_face_pairs=pairs,
            source_bound_connected_overlaps=[context])
        self.assertTrue(scoped["passed"])
        self.assertTrue(scoped["source_bound_connected_overlap"]["passed"])
        self.assertEqual(scoped["triangle_pair_count"], len(pairs))
        self.assertEqual(scoped["unexpected_triangle_pairs"], [])
        self.assertEqual(scoped["source_bound_interface_triangle_pairs"], [list(pair) for pair in pairs])
        self.assertEqual(scoped["declared_interface_triangle_pairs"], [list(pair) for pair in pairs])
        self.assertTrue(any(row["inside_component_witnesses"]
                            for row in scoped["containment"]))
        self.assertTrue(any(row["outside_component_witnesses"]
                            for row in scoped["containment"]))

        combined = av.audit_changed_surfaces(
            {first.key: first, second.key: second},
            [first.key], targets=[second.key], interfaces={(first.key, second.key): pairs},
            required_closed=[first.key, second.key],
            source_bound_connected_overlap_interfaces={(first.key, second.key): [context]})
        self.assertTrue(combined["passed"])
        self.assertTrue(combined["rows"][-1]["source_bound_connected_overlap"]["passed"])

    def test_current_native_vessel_port_overlap_requires_sealed_current_handoff(self):
        first, second, _, context, _ = source_bound_vessel_port_case()
        source_map = json.loads(context["source_map_bytes"])
        source_map.pop("current_port_geometry_evidence")
        source_map.pop("parent_source_handoff_sha256")
        source_map_bytes = json.dumps(
            source_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
        source_map_sha = hashlib.sha256(source_map_bytes).hexdigest()
        binding = dict(context["source_binding"])
        binding["source_map_sha256"] = source_map_sha
        old_map_context = {
            **context,
            "source_binding": binding,
            "expected_source_binding_sha256": av._canonical_json_sha256(binding),
            "expected_source_map_sha256": source_map_sha,
            "source_map_bytes": source_map_bytes,
        }
        with self.assertRaisesRegex(ValueError, "not sealed to this exact current port report"):
            audit_source_bound_connected_overlap(
                first, second, source_proof_bytes=old_map_context["source_proof_bytes"],
                expected_source_proof_sha256=old_map_context["expected_source_proof_sha256"],
                source_binding=old_map_context["source_binding"],
                expected_source_binding_sha256=old_map_context["expected_source_binding_sha256"],
                expected_source_map_sha256=old_map_context["expected_source_map_sha256"],
                source_map_bytes=old_map_context["source_map_bytes"])

    def test_current_native_vessel_port_overlap_rechecks_baseline_hashes_in_sealed_report(self):
        first, second, _, context, _ = source_bound_vessel_port_case()
        proof = json.loads(context["source_proof_bytes"])
        proof["source_topology"]["independent_source_rigid_baseline:6"][
            "prepared_surface"]["positions_sha256"] = "a" * 64
        proof_bytes = json.dumps(proof, sort_keys=True, separators=(",", ":")).encode("utf-8")
        proof_sha = hashlib.sha256(proof_bytes).hexdigest()
        source_map = json.loads(context["source_map_bytes"])
        source_map["evidence_report_sha256"] = proof_sha
        source_map["current_port_geometry_evidence"]["report_sha256"] = proof_sha
        source_map_bytes = json.dumps(
            source_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
        source_map_sha = hashlib.sha256(source_map_bytes).hexdigest()
        binding = dict(context["source_binding"])
        binding["source_proof_sha256"] = proof_sha
        binding["source_map_sha256"] = source_map_sha
        corrupted = {
            **context,
            "source_proof_bytes": proof_bytes,
            "expected_source_proof_sha256": proof_sha,
            "source_binding": binding,
            "expected_source_binding_sha256": av._canonical_json_sha256(binding),
            "expected_source_map_sha256": source_map_sha,
            "source_map_bytes": source_map_bytes,
        }
        with self.assertRaisesRegex(ValueError, "baseline topology hashes or gates"):
            audit_source_bound_connected_overlap(
                first, second, source_proof_bytes=corrupted["source_proof_bytes"],
                expected_source_proof_sha256=corrupted["expected_source_proof_sha256"],
                source_binding=corrupted["source_binding"],
                expected_source_binding_sha256=corrupted["expected_source_binding_sha256"],
                expected_source_map_sha256=corrupted["expected_source_map_sha256"],
                source_map_bytes=corrupted["source_map_bytes"])

    def test_current_native_vessel_port_overlap_rejects_stale_and_off_port_reuse(self):
        first, second, _, context, pairs = source_bound_vessel_port_case()
        changed = feature_surface(
            second.key, tetra() + np.asarray([-.5, -.5, 0], dtype=np.float32),
            FACES, [800, 801, 802, 803])
        stale = audit_surface_pair(
            first, changed, intended_face_pairs=pairs,
            source_bound_connected_overlaps=[context])
        self.assertFalse(stale["passed"])
        self.assertFalse(stale["source_bound_connected_overlap"]["passed"])
        self.assertTrue(stale["unexpected_triangle_pairs"])
        self.assertTrue(set(map(tuple, stale["unexpected_triangle_pairs"])) - set(pairs))
        self.assertEqual(
            stale["source_bound_connected_overlap"]["reason"],
            "invalid_source_bound_connected_overlap_proof")

        off_port = feature_surface(
            (51011, 8), tetra(offset=.1), FACES, [800, 801, 802, 803])
        reused = audit_surface_pair(
            first, off_port, intended_face_pairs=pairs,
            source_bound_connected_overlaps=[context])
        self.assertFalse(reused["passed"])
        self.assertFalse(reused["source_bound_connected_overlap"]["passed"])

        generic = audit_surface_pair(first, off_port, intended_face_pairs=pairs)
        self.assertFalse(generic["passed"])

    def test_source_bound_connected_overlap_rechecks_exact_source_proof(self):
        first, second, proof, raw, proof_sha, map_sha, binding = source_bound_overlap_case()
        result = audit_source_bound_connected_overlap(
            first, second, source_proof_bytes=raw,
            expected_source_proof_sha256=proof_sha, source_binding=binding,
            expected_source_binding_sha256=av._canonical_json_sha256(binding),
            expected_source_map_sha256=map_sha)
        self.assertTrue(result["passed"])
        self.assertEqual(result["exact_face_pair_count"], len(proof["contact_witnesses"]))
        self.assertTrue(result["exact_inside_both_witness"]["first_location"] == "inside")
        self.assertIn("No CSF flow", result["claim_limit"])

    def test_source_bound_connected_overlap_rejects_wrong_proof_or_map_hash(self):
        first, second, _, raw, proof_sha, map_sha, binding = source_bound_overlap_case()
        with self.assertRaisesRegex(ValueError, "proof bytes"):
            audit_source_bound_connected_overlap(
                first, second, source_proof_bytes=raw,
                expected_source_proof_sha256="c" * 64, source_binding=binding,
                expected_source_binding_sha256=av._canonical_json_sha256(binding),
                expected_source_map_sha256=map_sha)
        with self.assertRaisesRegex(ValueError, "not pinned"):
            audit_source_bound_connected_overlap(
                first, second, source_proof_bytes=raw,
                expected_source_proof_sha256=proof_sha, source_binding=binding,
                expected_source_binding_sha256=av._canonical_json_sha256(binding),
                expected_source_map_sha256="d" * 64)
        with self.assertRaisesRegex(ValueError, "source binding bytes"):
            audit_source_bound_connected_overlap(
                first, second, source_proof_bytes=raw,
                expected_source_proof_sha256=proof_sha, source_binding=binding,
                expected_source_binding_sha256="e" * 64,
                expected_source_map_sha256=map_sha)

    def test_source_bound_connected_overlap_rejects_surface_gid_hash_mismatch(self):
        first, second, _, raw, proof_sha, map_sha, binding = source_bound_overlap_case()
        binding["surfaces"][0]["gids_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "gids_sha256"):
            audit_source_bound_connected_overlap(
                first, second, source_proof_bytes=raw,
                expected_source_proof_sha256=proof_sha, source_binding=binding,
                expected_source_binding_sha256=av._canonical_json_sha256(binding),
                expected_source_map_sha256=map_sha)

    def test_source_bound_connected_overlap_rejects_identity_mismatch(self):
        first, second, _, raw, proof_sha, map_sha, binding = source_bound_overlap_case()
        binding["surface_identities"][1] = [51010, 345]
        with self.assertRaisesRegex(ValueError, "identities or scope"):
            audit_source_bound_connected_overlap(
                first, second, source_proof_bytes=raw,
                expected_source_proof_sha256=proof_sha, source_binding=binding,
                expected_source_binding_sha256=av._canonical_json_sha256(binding),
                expected_source_map_sha256=map_sha)

    def test_source_bound_connected_overlap_rejects_pair_or_contact_changes(self):
        first, second, proof, _, _, map_sha, binding = source_bound_overlap_case()
        proof["exact_pair_gate"]["exact_face_pairs"].append([999, 999])
        proof["contact_witnesses"].append({
            "face_pair": [999, 999],
            "points_world_m": [["99", "98", "97"]],
        })
        raw = json.dumps(proof, sort_keys=True, separators=(",", ":")).encode("utf-8")
        proof_sha = hashlib.sha256(raw).hexdigest()
        binding["source_proof_sha256"] = proof_sha
        result = audit_source_bound_connected_overlap(
            first, second, source_proof_bytes=raw,
            expected_source_proof_sha256=proof_sha, source_binding=binding,
            expected_source_binding_sha256=av._canonical_json_sha256(binding),
            expected_source_map_sha256=map_sha)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "observed_exact_face_pairs_differ_from_source_proof")

        first, second, _, raw, proof_sha, map_sha, binding = source_bound_overlap_case(altered_contact=True)
        result = audit_source_bound_connected_overlap(
            first, second, source_proof_bytes=raw,
            expected_source_proof_sha256=proof_sha, source_binding=binding,
            expected_source_binding_sha256=av._canonical_json_sha256(binding),
            expected_source_map_sha256=map_sha)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "exact_contact_points_differ_from_source_proof")

    def test_source_bound_connected_overlap_rejects_open_or_disconnected_surfaces(self):
        first, second, _, raw, proof_sha, map_sha, binding = source_bound_overlap_case(open_first=True)
        result = audit_source_bound_connected_overlap(
            first, second, source_proof_bytes=raw,
            expected_source_proof_sha256=proof_sha, source_binding=binding,
            expected_source_binding_sha256=av._canonical_json_sha256(binding),
            expected_source_map_sha256=map_sha)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "surface_not_closed")

        first, second, _, raw, proof_sha, map_sha, binding = source_bound_overlap_case(
            disconnected_first=True)
        result = audit_source_bound_connected_overlap(
            first, second, source_proof_bytes=raw,
            expected_source_proof_sha256=proof_sha, source_binding=binding,
            expected_source_binding_sha256=av._canonical_json_sha256(binding),
            expected_source_map_sha256=map_sha)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "surface_not_single_component")

    def test_source_bound_connected_overlap_rechecks_inside_witness_and_topology(self):
        first, second, proof, _, _, map_sha, binding = source_bound_overlap_case()
        proof["closed_solid_connection"]["exact_inside_both_witness"]["point_world_m"] = [9., 9., 9.]
        raw = json.dumps(proof, sort_keys=True, separators=(",", ":")).encode("utf-8")
        proof_sha = hashlib.sha256(raw).hexdigest()
        binding["source_proof_sha256"] = proof_sha
        result = audit_source_bound_connected_overlap(
            first, second, source_proof_bytes=raw,
            expected_source_proof_sha256=proof_sha, source_binding=binding,
            expected_source_binding_sha256=av._canonical_json_sha256(binding),
            expected_source_map_sha256=map_sha)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "frozen_overlap_witness_not_inside_both")

        first, second, _, raw, proof_sha, map_sha, binding = source_bound_overlap_case()
        second.self_audit = lambda: {"count": 1, "pairs": [[0, 1]]}
        result = audit_source_bound_connected_overlap(
            first, second, source_proof_bytes=raw,
            expected_source_proof_sha256=proof_sha, source_binding=binding,
            expected_source_binding_sha256=av._canonical_json_sha256(binding),
            expected_source_map_sha256=map_sha)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reason"], "surface_self_intersection")

    def test_capture_rejects_hash_rejected_endpoint_and_duplicate_pose(self):
        receipt = {"schema": "numi.human.accepted-render-geometry.v1",
                       "physical_endpoint": "accepted", "pack_file_sha256": "a" * 64,
                       "accepted_step": 0, "accepted_time_s": 0., "accepted_body_pose_count": 2,
                       "accepted_body_poses": [
                           {"body_index": i, "position_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}
                           for i in range(2)],
                       "accepted_registered_body_poses": []}
        self.assertEqual(validate_capture_receipt(receipt, "a" * 64)["accepted_body_pose_count"], 2)
        for field, value in (("physical_endpoint", "rejected"), ("pack_file_sha256", "b" * 64)):
            bad = copy.deepcopy(receipt)
            bad[field] = value
            with self.assertRaises(ValueError):
                validate_capture_receipt(bad, "a" * 64)
        bad = copy.deepcopy(receipt)
        bad["accepted_body_poses"][1]["body_index"] = 0
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_capture_receipt(bad, "a" * 64)


class CaptureDecoderTests(unittest.TestCase):
    def test_same_typed_identity_primitive_chunks_are_all_retained(self):
        import hashlib
        import json
        import struct
        import tempfile
        from pathlib import Path

        from numilab_human.resting_anatomy_verify import load_capture
        vertices = np.zeros((4, 20), dtype="<f4")
        vertices[:, :3] = tetra()
        vertices[:, 4:7] = [0, 0, 1]
        indices = FACES.astype("<u4").tobytes()
        primitives = np.zeros((2, 16), dtype="<u4")
        primitives[:, :2] = [[0, 6], [6, 6]]
        primitives[:, 4:8] = [51011, 19, 0xffffffff, 7]
        sections = [(2, vertices.tobytes(), 4, 80), (3, indices, 12, 4),
                    (4, primitives.tobytes(), 2, 64)]
        directory_offset = 96
        offset = directory_offset + 72 * len(sections)
        directory, blobs = [], []
        for kind, raw, count, stride in sections:
            directory.append(struct.pack("<IIQQQII32s", kind, 0, offset, len(raw),
                                         count, stride, 0, hashlib.sha256(raw).digest()))
            blobs.append(raw)
            offset += len(raw)
        pack = struct.pack("<8sIIQQ32s24s", b"MRVPACK2", 2, 3, directory_offset,
                           offset, b"\0" * 32, b"\0" * 24) + b"\0" * 8 + b"".join(directory + blobs)
        receipt = {"schema": "numi.human.accepted-render-geometry.v1",
                   "physical_endpoint": "accepted",
                   "pack_file_sha256": hashlib.sha256(pack).hexdigest(),
                   "captured_vertex_buffer_sha256": hashlib.sha256(vertices.tobytes()).hexdigest(),
                   "accepted_step": 0, "accepted_time_s": 0.,
                   "vertex_count": 4, "index_count": 12, "primitive_count": 2,
                   "accepted_body_pose_count": 1,
                   "accepted_body_poses": [{"body_index": 0, "position_m": [0, 0, 0],
                                           "quaternion_xyzw": [0, 0, 0, 1]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "capture.mrvpack").write_bytes(pack)
            (path / "receipt.json").write_text(json.dumps(receipt))
            surfaces, _ = load_capture(path / "capture.mrvpack", path / "receipt.json")
        surface = prepare_surface((51011, 7), *surfaces[(51011, 7)][:2])
        np.testing.assert_array_equal(surface.faces, FACES)
        self.assertTrue(surface.closed)

    def test_rest_comparison_preserves_float32_and_rejects_disagreement(self):
        from numilab_human.resting_anatomy_verify import validate_rest_emission
        expected = tetra()
        self.assertTrue(validate_rest_emission(expected, expected)["passed"])
        displaced = expected.copy()
        displaced[1, 0] += np.float32(.001)
        report = validate_rest_emission(displaced, expected)
        self.assertFalse(report["passed"])
        self.assertEqual(report["outside_tolerance_vertex_ids"], [1])


    def test_cli_sequence_uses_existing_manifest_and_pinned_sources(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from numilab_human.resting_anatomy_verify import main
        with tempfile.TemporaryDirectory() as directory:
            manifest, digest, source, source_sha, *_ = self._sequence_fixture(directory)
            output = Path(directory) / "report.json"
            with patch("numilab_human.resting_anatomy_verify.audit_capture_sequence",
                       return_value={"passed": True}) as audit:
                code = main(["--capture-manifest", str(manifest),
                             "--manifest-sha256", digest,
                             "--source-pin", str(source) + "=" + source_sha,
                             "--changed", "51011:7", "--required-closed", "51011:7",
                             "--intended-enclosure", "51011:7,51007:1",
                             "--output", str(output)])
            self.assertEqual(code, 0)
            arguments = audit.call_args.kwargs
            self.assertEqual(arguments["expected_source_sha256"], {str(source): source_sha})
            self.assertEqual(arguments["required_closed"], [(51011, 7)])
            self.assertEqual(arguments["intended_enclosures_by_step"],
                             {0: [((51011, 7), (51007, 1))],
                              1: [((51011, 7), (51007, 1))]})
            self.assertTrue(json.loads(output.read_text())["passed"])

    def test_cli_sequence_rejects_changed_manifest_before_geometry(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from numilab_human.resting_anatomy_verify import main
        with tempfile.TemporaryDirectory() as directory:
            manifest, _digest, source, source_sha, *_ = self._sequence_fixture(directory)
            output = Path(directory) / "report.json"
            with patch("numilab_human.resting_anatomy_verify.load_capture") as load:
                code = main(["--capture-manifest", str(manifest),
                             "--manifest-sha256", "0" * 64,
                             "--source-pin", str(source) + "=" + source_sha,
                             "--changed", "51011:7", "--output", str(output)])
            load.assert_not_called()
            self.assertEqual(code, 1)
            result = json.loads(output.read_text())
            self.assertEqual(result["failed_stage"], "preparation")
            self.assertIn("manifest SHA", result["error"])

    def test_cli_pack_requires_receipt_and_rejects_sequence_pins(self):
        from unittest.mock import patch

        from numilab_human.resting_anatomy_verify import main
        for arguments in (["--pack", "a"],
                          ["--pack", "a", "--capture-receipt", "b", "--source-pin", "c=d"]):
            with self.subTest(arguments=arguments), patch("sys.stderr"), self.assertRaises(SystemExit):
                main(arguments + ["--changed", "51011:7", "--output", "unused.json"])

    def _sequence_fixture(self, directory):
        import hashlib
        import json
        from pathlib import Path
        root = Path(directory)
        source = root / "source.nhtissue"
        source.write_bytes(b"pinned-existing-source")
        source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        rows = []
        for index, step in enumerate((0, 1)):
            rows.append({
                "step": step, "accepted_time_s": float(index),
                "pack_path": str(root / (f"step-{step}.mrvpack")),
                "recorded_pack_sha256": (f"{index + 1:064x}"),
                "receipt_path": str(root / (f"step-{step}.receipt.json")),
                "receipt_sha256": (f"{index + 101:064x}"),
            })
        manifest_path = root / "accepted-capture-index.json"
        manifest_path.write_text(json.dumps({"source": "existing accepted capture index",
                                             "capture_rows": rows, "scope": "test"}))
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        positions = tetra()
        normals = np.asarray([[1, 0, 0]] * 4, dtype=np.float32)
        gids = np.arange(4, dtype=np.uint64)
        capture = {(51011, 7): (positions, FACES, normals, gids)}
        prepared_surface = prepare_surface(
            (51011, 7), positions.copy(), FACES.copy(), normals=normals.copy(),
            source_vertex_ids=gids.copy())
        prepared = {step: {(51011, 7): prepared_surface} for step in (0, 1)}
        identities = {
            step: {"pack_sha256": rows[index]["recorded_pack_sha256"],
                   "capture_receipt_sha256": rows[index]["receipt_sha256"],
                   "accepted_step": step, "accepted_time_s": float(index)}
            for index, step in enumerate((0, 1))
        }
        return manifest_path, manifest_sha, source, source_sha, rows, capture, prepared, identities

    def test_changed_surface_gate_preserves_material_feature_validation(self):
        from numilab_human.resting_anatomy_verify import audit_changed_surfaces
        first = feature_surface((51005, 132), tetra(), FACES, np.arange(4, dtype=np.uint64))
        points = tetra() * np.float32(-1)
        points[0] = first.positions[0]
        second = feature_surface((51005, 134), points, FACES[:, [0, 2, 1]],
                                 np.array([0, 11, 12, 13], dtype=np.uint64))
        declarations = declarations_for_actual_shared_features(first, second)
        self.assertTrue(declarations)
        surfaces = {first.key: first, second.key: second}
        result = audit_changed_surfaces(
            surfaces, [first.key], targets=[second.key], required_closed=[first.key],
            shared_feature_interfaces={(first.key, second.key): declarations})
        self.assertTrue(result["passed"], result)
        second.core_weights_by_gid[0] = second.core_weights_by_gid[0].copy()
        second.core_weights_by_gid[0][0] += 0.001
        failed = audit_changed_surfaces(
            surfaces, [first.key], targets=[second.key], required_closed=[first.key],
            shared_feature_interfaces={(first.key, second.key): declarations})
        self.assertFalse(failed["passed"], failed)
        self.assertTrue(failed["rows"][-1]["rejected_interface_feature_contacts"])

    def test_target_preparation_failure_reports_identity_and_exact_face_sources(self):
        from numilab_human.resting_anatomy_verify import audit_changed_surfaces

        vessel = prepare_surface((51011, 7), tetra(), FACES)
        triangle = np.asarray([[.05, .05, .05], [.2, .05, .05], [.05, .2, .05]],
                              dtype=np.float32)
        target_positions = np.concatenate((triangle, triangle), axis=0)
        target_faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
        target_gids = np.asarray([101, 102, 103, 201, 202, 203], dtype=np.uint64)
        surfaces = {
            vessel.key: vessel,
            (51010, 313): (target_positions, target_faces, None, target_gids),
        }

        result = audit_changed_surfaces(
            surfaces, changed=[vessel.key], targets=[(51010, 313)],
            required_closed=[vessel.key])

        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_stage"], "target_preparation")
        row = result["rows"][-1]
        self.assertEqual(row["identities"], [[51011, 7], [51010, 313]])
        self.assertEqual(row["failed_identity"], [51010, 313])
        self.assertIn("before pair predicates", row["reason"])
        witness = row["target_topology_witness"]
        self.assertEqual(witness["typed_identity"], [51010, 313])
        self.assertEqual(witness["topology_stage"], "exact_float32_coordinate_quotient")
        self.assertEqual(witness["duplicate_face_ids"], [1])
        self.assertEqual(witness["duplicate_face_groups"], [[0, 1]])
        self.assertEqual(
            [face["source_face_index"] for face in witness["witness_faces"]], [0, 1])
        self.assertEqual(
            [face["input_source_vertex_ids"] for face in witness["witness_faces"]],
            [[101, 102, 103], [201, 202, 203]])
        first_bits, second_bits = (
            face["input_float32_xyz_bits"] for face in witness["witness_faces"])
        self.assertEqual(first_bits, second_bits)

    def test_capture_sequence_passes_pose_specific_material_declarations(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from numilab_human.resting_anatomy_verify import audit_capture_sequence

        with TemporaryDirectory() as directory:
            manifest, manifest_sha, source, source_sha, _rows, capture, prepared, identities = self._sequence_fixture(directory)
            declaration_map = {0: {((51011, 7), (51011, 8)): [{"test_marker": 0}]}}
            def load(pack, receipt):
                step = int(Path(pack).stem.split("-")[1])
                return capture, identities[step]
            with patch("numilab_human.resting_anatomy_verify.load_capture", side_effect=load), \
                 patch("numilab_human.resting_anatomy_verify.audit_changed_surfaces",
                       return_value={"passed": True, "rows": []}) as audit:
                result = audit_capture_sequence(
                    manifest, expected_manifest_sha256=manifest_sha,
                    expected_source_sha256={str(source): source_sha}, changed=[(51011, 7)],
                    prepared_surfaces_by_step=prepared,
                    shared_feature_interfaces_by_step=declaration_map)
            self.assertTrue(result["passed"])
            self.assertIs(audit.call_args_list[0].kwargs["shared_feature_interfaces"], declaration_map[0])
            self.assertIsNone(audit.call_args_list[1].kwargs["shared_feature_interfaces"])

    def test_capture_sequence_stops_after_first_geometry_failure(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from numilab_human.resting_anatomy_verify import audit_capture_sequence

        with TemporaryDirectory() as directory:
            manifest, manifest_sha, source, source_sha, _rows, capture, prepared, identities = self._sequence_fixture(directory)
            def load(pack, receipt):
                step = int(Path(pack).stem.split("-")[1])
                return capture, identities[step]
            failed = {"passed": False, "failed_stage": "self", "rows": [{"stage": "self", "passed": False}]}
            with patch("numilab_human.resting_anatomy_verify.load_capture", side_effect=load) as loader, \
                 patch("numilab_human.resting_anatomy_verify.audit_changed_surfaces", return_value=failed) as audit:
                result = audit_capture_sequence(
                    manifest, expected_manifest_sha256=manifest_sha,
                    expected_source_sha256={str(source): source_sha}, changed=[(51011, 7)],
                    prepared_surfaces_by_step=prepared)
            self.assertFalse(result["passed"])
            self.assertEqual(result["capture_step"], 0)
            self.assertEqual(result["failed_stage"], "self")
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(audit.call_count, 1)

    def test_capture_sequence_binds_exact_ordered_faces_with_unused_source_slot(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from numilab_human.resting_anatomy_verify import audit_capture_sequence

        with TemporaryDirectory() as directory:
            manifest, manifest_sha, source, source_sha, _rows, _, _, identities = self._sequence_fixture(directory)
            captured_positions = tetra()
            captured_normals = np.asarray([[1, 0, 0]] * 4, dtype=np.float32)
            captured_gids = np.arange(1001, 1005, dtype=np.uint64)
            capture = {(51011, 7): (captured_positions, FACES, captured_normals, captured_gids)}
            source_positions = np.vstack((np.asarray([[9, 9, 9]], dtype=np.float32), captured_positions))
            source_normals = np.vstack((np.asarray([[0, 1, 0]], dtype=np.float32), captured_normals))
            source_faces = (FACES + np.uint32(1)).copy()
            source_gids = np.arange(900, 905, dtype=np.uint64)
            prepared_surface = prepare_surface(
                (51011, 7), source_positions, source_faces, normals=source_normals,
                source_vertex_ids=source_gids)
            prepared = {step: {(51011, 7): prepared_surface} for step in (0, 1)}
            def load(pack, receipt):
                step = int(Path(pack).stem.split("-")[1])
                return capture, identities[step]
            with patch("numilab_human.resting_anatomy_verify.load_capture", side_effect=load), \
                 patch("numilab_human.resting_anatomy_verify.audit_changed_surfaces",
                       return_value={"passed": True, "rows": []}):
                result = audit_capture_sequence(
                    manifest, expected_manifest_sha256=manifest_sha,
                    expected_source_sha256={str(source): source_sha}, changed=[(51011, 7)],
                    prepared_surfaces_by_step=prepared)
            self.assertTrue(result["passed"], result)
            self.assertEqual(result["completed_captures"][0]["surface_bindings"][0]
                             ["source_capture_vertex_namespace_offset"], 1000)
            self.assertEqual(result["completed_captures"][0]["surface_bindings"][0]
                             ["referenced_corner_count"], 12)

    def test_capture_sequence_rejects_misaligned_typed_lineage_before_predicate(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from numilab_human.resting_anatomy_verify import audit_capture_sequence

        with TemporaryDirectory() as directory:
            manifest, manifest_sha, source, source_sha, _rows, _capture, prepared, _identities = self._sequence_fixture(directory)
            for surface_map in prepared.values():
                surface = feature_surface((51011, 7), tetra(), FACES, np.arange(4, dtype=np.uint64))
                surface.source_face_lineage = surface.source_face_lineage[:-1]
                surface_map[(51011, 7)] = surface
            with (patch("numilab_human.resting_anatomy_verify.load_capture") as loader,
                  patch("numilab_human.resting_anatomy_verify.audit_changed_surfaces") as audit,
                  self.assertRaisesRegex(ValueError, "face lineage is not aligned")):
                audit_capture_sequence(
                    manifest, expected_manifest_sha256=manifest_sha,
                    expected_source_sha256={str(source): source_sha}, changed=[(51011, 7)],
                    prepared_surfaces_by_step=prepared)
            loader.assert_not_called()
            audit.assert_not_called()

    def test_capture_sequence_rejects_source_hash_identity_before_capture_load(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch

        from numilab_human.resting_anatomy_verify import audit_capture_sequence

        with TemporaryDirectory() as directory:
            manifest, manifest_sha, source, _source_sha, _rows, _capture, prepared, _identities = self._sequence_fixture(directory)
            with (patch("numilab_human.resting_anatomy_verify.load_capture") as loader,
                  patch("numilab_human.resting_anatomy_verify.audit_changed_surfaces") as audit,
                  self.assertRaisesRegex(ValueError, "source input SHA-256 mismatch")):
                audit_capture_sequence(
                    manifest, expected_manifest_sha256=manifest_sha,
                    expected_source_sha256={str(source): "0" * 64}, changed=[(51011, 7)],
                    prepared_surfaces_by_step=prepared)
            loader.assert_not_called()
            audit.assert_not_called()




class IncrementalSelfAuditTests(unittest.TestCase):
    def prepare(self, positions, faces, *, key=(51005, 134), gids=None):
        return prepare_surface(key, np.asarray(positions, dtype=np.float32),
                               np.asarray(faces, dtype=np.uint32),
                               source_vertex_ids=gids)

    def test_unchanged_geometry_reuses_zero_parent_without_narrow_phase(self):
        from unittest.mock import patch
        parent = self.prepare(tetra(), FACES)
        parent.self_audit()
        current = self.prepare(tetra(), FACES)
        with patch.object(ci, "triangle_intersection_points",
                          side_effect=AssertionError("unchanged pair retested")):
            result = current.self_audit_since(parent)
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["aabb_candidate_pairs"], 0)
        self.assertEqual(result["incremental_proof"]["changed_face_ids"], [])
        self.assertIs(current.self_audit(), result)

    def test_moved_vertex_checks_all_incident_faces_against_entire_mesh(self):
        positions = np.concatenate((tetra(), tetra(offset=3.0)))
        faces = np.concatenate((FACES, FACES + 4))
        parent = self.prepare(positions, faces)
        changed = positions.copy()
        changed[4] = [.2, .2, .2]
        current = self.prepare(changed, faces)
        result = current.self_audit_since(parent)
        exhaustive = self.prepare(changed, faces).self_audit()
        self.assertGreater(result["count"], 0)
        self.assertEqual(result["triangle_pairs"], exhaustive["triangle_pairs"])
        self.assertEqual(result["incremental_proof"]["changed_face_ids"], [4, 5, 6])

    def test_changed_connectivity_and_reversed_selected_id_direction(self):
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                              [3, 0, 0], [3, 1, 0], [4, 0, 0],
                              [.25, .25, -1], [.25, .25, 1], [.75, .25, 0]],
                             dtype=np.float32)
        parent = self.prepare(positions, [[3, 4, 5], [6, 7, 8]])
        current = self.prepare(positions, [[0, 1, 2], [6, 7, 8]])
        result = current.self_audit_since(parent)
        self.assertEqual(result["triangle_pairs"], [[0, 1]])
        self.assertEqual(result["incremental_proof"]["changed_face_ids"], [0])

    def test_exact_coordinate_quotient_renumbering_preserves_proof(self):
        # An unused earlier slot merges with a referenced point. Every later
        # quotient ID changes, but the ordered triangles remain exactly equal.
        positions = np.concatenate((np.array([[-3, -3, -3]], dtype=np.float32),
                                    tetra(), tetra(offset=3.0)))
        faces = np.concatenate((FACES + 1, FACES + 5))
        parent = self.prepare(positions, faces)
        changed = positions.copy()
        changed[0] = changed[1]
        current = self.prepare(changed, faces)
        self.assertFalse(np.array_equal(parent.quotient_faces, current.quotient_faces))
        result = current.self_audit_since(parent)
        exhaustive = self.prepare(changed, faces).self_audit()
        self.assertEqual(result["triangle_pairs"], exhaustive["triangle_pairs"])
        self.assertEqual(result["incremental_proof"]["changed_face_ids"], [])

    def test_nonzero_parent_and_namespace_drift_rejected(self):
        positions = np.concatenate((tetra(), tetra(offset=.1)))
        faces = np.concatenate((FACES, FACES + 4))
        bad = self.prepare(positions, faces)
        with self.assertRaisesRegex(ValueError, "zero-self"):
            self.prepare(positions, faces).self_audit_since(bad)
        parent = self.prepare(tetra(), FACES)
        with self.assertRaisesRegex(ValueError, "typed parent"):
            self.prepare(tetra(), FACES, key=(51005, 133)).self_audit_since(parent)
        with self.assertRaisesRegex(ValueError, "vertex identities"):
            self.prepare(tetra(), FACES, gids=np.arange(4, 8)).self_audit_since(parent)


if __name__ == "__main__":
    unittest.main()

class NativeSourceIdentityTests(unittest.TestCase):
    def test_source_layer_resolves_central_canal_row_as_native_nerve_layer(self):
        self.assertEqual(source_surface_key(12, 3, payload_kind="nhanatomy"),
                         (51012, 12))
        self.assertEqual(source_surface_key(323, 1, payload_kind="nhanatomy"),
                         (51010, 323))
        # Named muscle/tendon source surfaces use their emitted record layer.
        self.assertEqual(source_surface_key(132, 2, payload_kind="nhtissue"),
                         (51006, 132))
        self.assertEqual(source_surface_key(1, 15, payload_kind="nhanatomy"),
                         (51031, 1))

    def test_source_layer_rejects_untyped_or_unknown_inputs(self):
        for stable, layer, kind in ((True, 3, "nhanatomy"), (12, 3.0, "nhanatomy"),
                                   (12, 16, "nhanatomy"), (132, 4, "nhtissue"),
                                   (12, 3, "unknown")):
            with self.subTest(stable=stable, layer=layer, kind=kind), self.assertRaises(ValueError):
                source_surface_key(stable, layer, payload_kind=kind)


def test_geometry_progress_preserves_results_and_stops_with_first_failure():
    from numilab_human.resting_anatomy_verify import audit_changed_surfaces
    first = prepare_surface((51011, 7), tetra(), FACES)
    second = prepare_surface((51011, 8), tetra()+np.float32(5), FACES)
    bad_points = np.concatenate((tetra()[:3], tetra()[:3]), axis=0)
    bad_faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    surfaces = {first.key: first, second.key: second,
                (51010, 313): (bad_points, bad_faces, None, np.arange(6))}
    for targets in ([second.key], [(51010, 313), second.key]):
        expected = audit_changed_surfaces(surfaces, [first.key], targets=targets)
        rows = []
        observed = audit_changed_surfaces(surfaces, [first.key], targets=targets,
                                          progress=rows.append)
        assert observed == expected
        assert rows == expected["rows"]
    assert observed["failed_stage"] == "target_preparation"
    assert len(rows) == 2
    import pytest
    with pytest.raises(TypeError, match="progress observer"):
        audit_changed_surfaces(surfaces, [first.key], progress=True)
