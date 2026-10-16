import copy
import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np

from numilab_human import aortic_port_handoff_rebind as av
from numilab_human.aortic_port_handoff_rebind import (
    canonical_handoff_bytes,
    rebind_aortic_source_handoff_to_current_capture,
)

FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.uint32)
CAPTURE_SHA = "a" * 64
REPORT_SHA = "b" * 64


def digest(array, dtype):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=dtype).tobytes()).hexdigest()


def source_rebind_case(*, unequal_face_counts=False, edge=((51011, 7), (51011, 6))):
    keys = tuple(edge)
    metadata = {
        (51011, 7): ("FJ3411", "FMA3768", "arch of aorta", 700, 900),
        (51011, 6): ("FJ3413", "FMA3736", "ascending aorta", 800, 950),
        (51011, 8): ("FJ3427", "FMA3784", "descending aorta", 810, 970),
        (51011, 465): ("FJ3483", "FMA4058", "left common carotid artery", 820, 990),
    }
    parent_rows, parent_bindings, current_rows, arrays, owner_pins, baseline_arrays = {}, {}, {}, {}, {}, {}
    pairs = [[2, 0]] if unequal_face_counts else [[0, 0], [1, 1]]
    pair_sha = digest(pairs, "<u4")
    for key in keys:
        faces = FACES.copy()
        row_key = f"{key[0]}:{key[1]}"
        member, fma, name, old_gid, current_gid = metadata[key]
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                             dtype=np.float32)
        if unequal_face_counts and key[1] == 6:
            # Subdivide one tetrahedron face to keep the baseline closed while
            # giving the pair columns distinct, valid local face bounds.
            center = positions[[0, 2, 1]].mean(axis=0)
            positions = np.vstack((positions, center))
            faces[0] = [0, 2, 4]
            faces = np.vstack((faces, [[2, 1, 4], [1, 0, 4]])).astype(np.uint32)
        if key[1] == 6:
            positions += np.float32(.1)
        normals = np.full_like(positions, np.float32(1.0 / np.sqrt(3.0)))
        gids = np.arange(current_gid, current_gid + len(positions), dtype=np.uint64)
        face_sha = digest(faces, "<u4")
        position_sha = digest(positions, "<f4")
        normal_sha = digest(normals, "<f4")
        gid_sha = digest(gids, "<u8")
        rigid_baseline_positions = positions.copy()
        if key[1] == 6:
            rigid_baseline_positions -= np.float32(.1)
        rigid_baseline_sha = digest(rigid_baseline_positions, "<f4")
        body_index = 22 if key[1] == 465 else 20
        parent_face_field = {"source_faces_u32_sha256": face_sha} if key[1] == 465 else {
            "faces_sha256": face_sha}
        parent_rows[row_key] = {
            "typed_identity": list(key), "body_index": body_index, "layer": 2,
            "first_vertex": old_gid, "vertex_count": len(positions), "face_count": len(faces),
            **parent_face_field, "source_member": member, "fma": fma, "name": name,
        }
        parent_bindings[row_key] = {
            "typed_identity": list(key), "vertex_count": len(positions), "face_count": len(faces),
            "positions_f32_sha256": position_sha, "faces_u32_sha256": face_sha,
            "normals_f32_sha256": normal_sha, "source_vertex_ids_u64_sha256": digest(
                np.arange(old_gid, old_gid + len(positions), dtype=np.uint64), "<u8"),
        }
        ancestry = {
            "status": "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED",
            "capture_face_count": len(faces), "source_face_count": len(faces),
            "capture_faces_u32_sha256": face_sha, "source_faces_u32_sha256": face_sha,
        }
        owner = {
            "typed_identity": list(key), "source_member": member, "source_concept_id": fma,
            "capture_face_order_equals_nha": True, "capture_face_vertex_ancestry": ancestry,
            "source_face_count": len(faces), "source_vertex_count": len(positions),
            "captured_positions_world_f32_sha256": position_sha,
            "captured_faces_u32_sha256": face_sha,
            "captured_global_vertex_ids_sha256": gid_sha,
            "nha_faces_u32_sha256": face_sha,
            "source_nha_row_sha256": "c" * 64, "source_member_sha256": "d" * 64,
            "rigid_source_world_xyz_f32_sha256": rigid_baseline_sha,
        }
        owner_pins[key] = {
            "typed_identity": list(key), "source_member": member, "fma": fma,
            "body_index": body_index, "layer": 2, "face_count": len(faces),
            "vertex_count": len(positions), "faces_sha256": face_sha,
            "source_member_sha256": "d" * 64, "source_nha_row_sha256": "c" * 64,
            "rigid_source_world_xyz_f32_sha256": rigid_baseline_sha,
        }
        current_rows[row_key] = {
            "typed_identity": list(key), "body_index": body_index, "layer": 2,
            "source_member": member, "fma": fma, "name": name,
            "first_vertex": current_gid - 800, "row_local_gid_offset": 800,
            "source_faces_count": len(faces), "source_vertices_count": len(positions),
            "source_faces_sha256": face_sha,
            "capture_positions_f32_sha256": position_sha,
            "capture_faces_u32_sha256": face_sha,
            "capture_normals_f32_sha256": normal_sha,
            "capture_gids_u64_sha256": gid_sha, "owner": owner,
        }
        arrays[key] = {
            "positions": positions, "faces": faces.copy(), "normals": normals,
            "source_vertex_ids": gids,
        }
        baseline_arrays[key] = {
            "positions": rigid_baseline_positions, "faces": faces.copy(),
            "source_vertex_ids": gids.copy(),
        }

    is_arch_handoff = keys == ((51011, 7), (51011, 6))
    parent = {
        "schema": ("numi.human.verified-aortic-arch-interface-handoff.v1" if is_arch_handoff
                   else "numi.human.verified-aortic-source-interface-handoff.v1"),
        "status": ("VERIFIED_SOURCE_BOUND_AORTIC_ARCH_INTERFACE" if is_arch_handoff
                   else "VERIFIED_SOURCE_BOUND_AORTIC_INTERFACE"), "passed": True,
        "typed_link": [list(key) for key in keys], "source_rows": parent_rows,
        "candidate_surface_bindings": parent_bindings,
        "exact_face_pairs": pairs, "exact_face_pairset_sha256": pair_sha,
        "source_topology_passed": True, "interface_gate_passed": True,
        "solid_gate_passed": True, "independent_source_collar_calibre_passed": True,
        "candidate_pairs_equal_frozen_source_pairs": True,
        "baseline_pairs_equal_frozen_source_pairs": True,
        "candidate_baseline_exact_pairsets_equal": True,
    }
    preflight = {
        "schema": "test-current-source-preflight",
        "capture_input_bundle_sha256": CAPTURE_SHA,
        "source_rows": current_rows,
    }
    return (canonical_handoff_bytes(parent), canonical_handoff_bytes(preflight),
            arrays, owner_pins, baseline_arrays)


def exact_transverse_interface(link, pairs):
    link_rows = [list(map(int, key)) for key in link]
    pair_rows = [list(map(int, pair)) for pair in pairs]
    boundary = {
        "passed": False, "classification": "NOT_A_DECLARED_PARTIAL_VESSEL_PORT",
        "typed_port": link_rows, "unexpected_pair_count": 0,
        "observed_pair_count": len(pair_rows),
        "conditions": {"symmetric_exact_inside_and_outside_face_witnesses": False},
    }
    face_pair = pair_rows[0]
    barycentric = [["1", "3"], ["1", "3"], ["1", "3"]]
    directions = {
        role: {
            side: {"face_pair": face_pair, "location_in_opposite_surface": side,
                   "face_barycentric_rational": barycentric}
            for side in ("inside", "outside")
        }
        for role in ("first_surface_against_second", "second_surface_against_first")
    }
    transverse = {
        "passed": True,
        "requires_all_four_witnesses_from_same_face_pair": True,
        "requires_strict_positive_exact_barycentric_coordinates": True,
        "uses_exact_rational_point_location_in_opposite_closed_surface": True,
        "passing_face_pair": face_pair,
        "directional_surface_witnesses": directions,
        "pair_witness_summaries": [{
            "face_pair": face_pair, "witness_count": 4,
            "required_witness_count": 4,
            "locations_by_surface": {
                "first_surface_against_second": ["inside", "outside"],
                "second_surface_against_first": ["inside", "outside"],
            },
        }],
        "exact_point_location_call_count": 4,
    }
    points = [[0.0, 0.0, 0.0]]
    point_sha = hashlib.sha256(np.ascontiguousarray(np.asarray(points, dtype="<f8")).tobytes()).hexdigest()
    return {
        "passed": True, "typed_port": link_rows,
        "observed_pair_count": len(pair_rows),
        "observed_face_pairs": pair_rows, "frozen_face_pairs": pair_rows,
        "frozen_face_pairs_sha256": hashlib.sha256(
            np.ascontiguousarray(np.asarray(pair_rows, dtype="<i8")).tobytes()).hexdigest(),
        "unexpected_face_pairs": [],
        "observed_pairs_within_frozen_face_patch_cross_product": True,
        "observed_pairs_within_source_declared_region": True,
        "documented_vessel_junction_scope": {
            "passed": True, "typed_port": link_rows,
            "allowed_typed_port": link_rows,
            "scope_kind": "declared_source_network_link",
            "frozen_matched_face_patch_required": True,
            "downstream_closed_solid_confirmation_required": True,
        },
        "exact_interface_point_count": len(points),
        "exact_interface_points_world_m": points,
        "exact_interface_points_sha256": point_sha,
        "exact_partial_overlap_witnesses": {
            "passed": True,
            "classification": "symmetric_exact_transverse_inside_outside_overlap",
            "requires_separate_closed_solid_confirmation": True,
            "boundary_predicate": boundary,
            "exact_transverse_overlap_side_witnesses": transverse,
        },
    }


def passing_current_port_report(pending, baseline_arrays):
    link = pending["typed_link"]
    pairs = pending["exact_face_pairs"]
    rows, bindings = pending["source_rows"], pending["candidate_surface_bindings"]
    topology, interfaces, solids, calibre = {}, {}, {}, {}
    for key in link:
        row_key = f"{key[0]}:{key[1]}"
        binding = bindings[row_key]
        common = {
            "identity": key, "closed_self_free_single_component": True,
            "exact_self_audit_count": 0, "referenced_face_component_count": 1,
        }
        topology["candidate:" + str(key[1])] = {
            **common,
            "source_vertex_ids_sha256": binding["source_vertex_ids_u64_sha256"],
            "prepared_surface": {
                "identity": key,
                "positions_sha256": binding["positions_f32_sha256"],
                "faces_sha256": binding["faces_u32_sha256"],
                "referenced_surface_closed": True,
            },
        }
        baseline = baseline_arrays[tuple(key)]
        topology["independent_source_rigid_baseline:" + str(key[1])] = {
            **common,
            "source_vertex_ids_sha256": digest(baseline["source_vertex_ids"], "<u8"),
            "prepared_surface": {
                "identity": key,
                "positions_sha256": digest(baseline["positions"], "<f4"),
                "faces_sha256": digest(baseline["faces"], "<u4"),
                "referenced_surface_closed": True,
            },
        }
    for role in ("candidate", "independent_source_rigid_baseline"):
        interfaces[role] = {
            "passed": True, "typed_port": link, "observed_face_pairs": pairs,
            "frozen_face_pairs": pairs, "observed_pair_count": len(pairs),
            "unexpected_face_pairs": [],
            "observed_pairs_within_frozen_face_patch_cross_product": True,
            "observed_pairs_within_source_declared_region": True,
            "frozen_face_pairs_sha256": hashlib.sha256(
                np.ascontiguousarray(np.asarray(pairs, dtype="<i8")).tobytes()).hexdigest(),
            "documented_vessel_junction_scope": {
                "passed": True, "typed_port": link, "allowed_typed_port": link,
                "scope_kind": "declared_source_network_link",
                "frozen_matched_face_patch_required": True,
                "downstream_closed_solid_confirmation_required": True,
            },
            "exact_partial_overlap_witnesses": {
                "passed": True, "requires_separate_closed_solid_confirmation": True,
                "boundary_predicate": {
                    "passed": True, "typed_port": link,
                    "unexpected_pair_count": 0, "observed_pair_count": len(pairs),
                },
                "exact_transverse_overlap_side_witnesses": {
                    "passed": True,
                    "requires_all_four_witnesses_from_same_face_pair": True,
                    "requires_strict_positive_exact_barycentric_coordinates": True,
                    "uses_exact_rational_point_location_in_opposite_closed_surface": True,
                },
            },
        }
        solids[role] = {
            "passed": True, "manifold_inputs_passed": True,
            "positive_intersection_volume": True, "union_connected_numerical_solid": True,
            "union_component_count": 1,
            "union_lineage_has_both_sources_without_generated_or_unmapped_faces": True,
            "exact_inside_both_witness": {
                "method": "intersection_volume_centroid",
                "first_location": "inside", "second_location": "inside",
                "predicate": "P6 exact rational point location for the exact binary64 candidate on the original Float32 source surfaces",
                "candidate_source": "derived floating Manifold Boolean Mesh64; candidate construction is not an exact CSG witness",
                "point_world_m": [0.3, 0.2, 0.2],
            },
            "union_float64_mesh64_lineage": {
                "generated_face_count": 0, "unmapped_face_count": 0,
                "run_original_ids": [key[1] for key in link],
            },
        }
    for key in link:
        calibre[str(key[1])] = {
            "passed": True, "owner_id": key[1],
            "baseline_metric_error": None, "candidate_metric_error": None,
            "comparison_error": None, "declared_area_ratio": [0.8, 1.25],
            "declared_minimum_feret_ratio": [0.9, 1.1],
            "baseline_section": {"passed": True},
            "candidate_section": {"passed": True},
        }
    return {
        "schema": "numi.human.current-native-aortic-port-result.v1",
        "status": "PASS_CURRENT_NATIVE_PORT_DIAGNOSTIC",
        "typed_link": link, "anatomical_admission": False,
        "geometry_predicates_run": True, "current_port_passed": True,
        "source_ancestry_handoff_sha256": pending["parent_source_handoff_sha256"],
        "source_ancestry_pairset_sha256": pending["exact_face_pairset_sha256"],
        "source_ancestry_pair_count": len(pairs),
        "current_candidate_and_current_source_intersection_recomputed": True,
        "current_candidate_and_source_pairsets_equal": True,
        "current_candidate_pairs_equal_source_ancestry_pairs": True,
        "current_source_baseline_pairs_equal_source_ancestry_pairs": True,
        "source_pairset_used_only_as_ancestry": True,
        "source_topology": topology, "interface_gates": interfaces,
        "solid_gates": solids, "independent_source_station_calibre": calibre,
        "limitations": {
            "no_blood_mass_or_physical_state_modified": True,
            "no_motion_or_duration_qualification": True,
            "no_whole_anatomy_admission": True,
            "source_capture_is_actual_native_one_step": True,
            "source_station_calibre_is_not_clinical": True,
        },
    }


class AorticPortHandoffRebindTests(unittest.TestCase):
    def build(self, parent_bytes, preflight_bytes, arrays, owner_pins):
        return rebind_aortic_source_handoff_to_current_capture(
            parent_bytes,
            expected_pinned_handoff_sha256=hashlib.sha256(parent_bytes).hexdigest(),
            current_source_preflight_bytes=preflight_bytes,
            capture_input_bundle_sha256=CAPTURE_SHA,
            capture_array_mapping=arrays,
            authenticated_source_owner_pins=owner_pins,
            parent_source_handoff_path="/evidence/parent-handoff.json",
        )

    def seal(self, pending_bytes, report_bytes, preflight_bytes, arrays, baseline_arrays):
        return av.seal_current_aortic_source_handoff_from_port_report(
            pending_bytes,
            expected_pending_handoff_sha256=hashlib.sha256(pending_bytes).hexdigest(),
            current_port_report_bytes=report_bytes,
            expected_current_port_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
            current_source_preflight_bytes=preflight_bytes,
            expected_capture_input_bundle_sha256=CAPTURE_SHA,
            capture_array_mapping=arrays,
            independent_source_rigid_baseline_array_mapping=baseline_arrays)

    def test_rebinds_current_arrays_and_global_gid_allocation_with_ancestry(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case()
        result = self.build(parent_bytes, preflight_bytes, arrays, owner_pins)
        self.assertEqual(result["status"], "PENDING_CURRENT_AORTIC_PORT_GEOMETRY")
        self.assertFalse(result["passed"])
        self.assertFalse(result["interface_gate_passed"])
        self.assertTrue(result["parent_source_gate_evidence"]["source_gate_results"]["interface_gate_passed"])
        self.assertEqual(result["parent_source_handoff_sha256"],
                         hashlib.sha256(parent_bytes).hexdigest())
        self.assertEqual(result["source_rows"]["51011:7"]["first_vertex"], 900)
        self.assertEqual(result["source_rows"]["51011:6"]["first_vertex"], 950)
        self.assertEqual(result["source_rows"]["51011:7"]["source_nha_first_vertex"], 100)
        self.assertEqual(result["candidate_surface_bindings"]["51011:6"]["source_vertex_ids_u64_sha256"],
                         digest(arrays[(51011, 6)]["source_vertex_ids"], "<u8"))
        self.assertEqual(result["exact_face_pairset_sha256"],
                         json.loads(parent_bytes)["exact_face_pairset_sha256"])
        self.assertFalse(result["current_capture_owner_rebind"]["geometry_predicates_run"])
        self.assertFalse(result["current_capture_owner_rebind"]["anatomical_admission"])

    def test_rebinds_authenticated_source_interface_links_including_body_22_carotid(self):
        for edge in (((51011, 7), (51011, 8)), ((51011, 7), (51011, 465))):
            with self.subTest(edge=edge):
                parent_bytes, preflight_bytes, arrays, owner_pins, _ = source_rebind_case(edge=edge)
                result = self.build(parent_bytes, preflight_bytes, arrays, owner_pins)
                self.assertEqual(result["schema"], "numi.human.verified-aortic-arch-interface-handoff.v1")
                self.assertEqual(result["status"], "PENDING_CURRENT_AORTIC_PORT_GEOMETRY")
                self.assertEqual(result["typed_link"], [list(key) for key in edge])
                self.assertEqual(result["parent_source_gate_evidence"]["parent_source_handoff_schema"],
                                 "numi.human.verified-aortic-source-interface-handoff.v1")
                self.assertFalse(result["passed"])
                self.assertFalse(result["solid_gate_passed"])
                if edge[1][1] == 465:
                    self.assertEqual(result["source_rows"]["51011:465"]["body_index"], 22)

    def test_rejects_current_array_drift_from_source_preflight(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case()
        arrays[(51011, 7)]["positions"][0, 0] += np.float32(.001)
        with self.assertRaisesRegex(ValueError, "differ from their pinned preflight"):
            self.build(parent_bytes, preflight_bytes, arrays, owner_pins)

    def test_rejects_gid_offset_that_does_not_match_current_capture(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case()
        changed = json.loads(preflight_bytes)
        changed["source_rows"]["51011:7"]["row_local_gid_offset"] += 1
        with self.assertRaisesRegex(ValueError, "GIDs do not follow"):
            self.build(parent_bytes, canonical_handoff_bytes(changed), arrays, owner_pins)

    def test_changed_xyz_with_matching_preflight_stays_pending(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case()
        changed = json.loads(preflight_bytes)
        positions = arrays[(51011, 7)]["positions"]
        positions[0, 0] += np.float32(.001)
        row = changed["source_rows"]["51011:7"]
        position_sha = digest(positions, "<f4")
        row["capture_positions_f32_sha256"] = position_sha
        row["owner"]["captured_positions_world_f32_sha256"] = position_sha
        result = self.build(parent_bytes, canonical_handoff_bytes(changed), arrays, owner_pins)
        self.assertEqual(result["status"], "PENDING_CURRENT_AORTIC_PORT_GEOMETRY")
        self.assertFalse(result["passed"])
        self.assertFalse(result["interface_gate_passed"])
        self.assertFalse(result["solid_gate_passed"])

    def test_rejects_source_owner_hashes_without_independent_pins(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case()
        owner_pins[(51011, 7)]["source_nha_row_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "owner or ordered-face ancestry"):
            self.build(parent_bytes, preflight_bytes, arrays, owner_pins)

        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case()
        owner_pins[(51011, 7)]["rigid_source_world_xyz_f32_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "owner or ordered-face ancestry"):
            self.build(parent_bytes, preflight_bytes, arrays, owner_pins)

    def test_rejects_report_baseline_hash_not_bound_to_independent_rigid_source(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case()
        pending = self.build(parent_bytes, preflight_bytes, arrays, owner_pins)
        pending_bytes = canonical_handoff_bytes(pending)
        report = passing_current_port_report(pending, baseline_arrays)
        report["source_topology"]["independent_source_rigid_baseline:6"][
            "prepared_surface"]["positions_sha256"] = "f" * 64
        report_bytes = canonical_handoff_bytes(report)
        with self.assertRaisesRegex(ValueError, "topology does not match"):
            self.seal(pending_bytes, report_bytes, preflight_bytes, arrays, baseline_arrays)

    def test_rejects_independent_baseline_arrays_that_differ_from_owner_pin(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case()
        pending = self.build(parent_bytes, preflight_bytes, arrays, owner_pins)
        pending_bytes = canonical_handoff_bytes(pending)
        report_bytes = canonical_handoff_bytes(
            passing_current_port_report(pending, baseline_arrays))
        baseline_arrays[(51011, 6)]["positions"][0, 0] += np.float32(.01)
        with self.assertRaisesRegex(ValueError, "differ from pinned source"):
            self.seal(pending_bytes, report_bytes, preflight_bytes, arrays, baseline_arrays)

    def test_fresh_report_seals_only_its_exact_current_arrays(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case()
        pending = self.build(parent_bytes, preflight_bytes, arrays, owner_pins)
        pending_bytes = canonical_handoff_bytes(pending)
        report_bytes = canonical_handoff_bytes(
            passing_current_port_report(pending, baseline_arrays))
        sealed = av.seal_current_aortic_source_handoff_from_port_report(
            pending_bytes,
            expected_pending_handoff_sha256=hashlib.sha256(pending_bytes).hexdigest(),
            current_port_report_bytes=report_bytes,
            expected_current_port_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
            current_source_preflight_bytes=preflight_bytes,
            expected_capture_input_bundle_sha256=CAPTURE_SHA,
            capture_array_mapping=arrays,
            independent_source_rigid_baseline_array_mapping=baseline_arrays)
        self.assertTrue(sealed["passed"])
        self.assertEqual(sealed["status"], "VERIFIED_SOURCE_BOUND_AORTIC_ARCH_INTERFACE")
        self.assertEqual(sealed["evidence_report_sha256"], hashlib.sha256(report_bytes).hexdigest())
        self.assertTrue(sealed["current_port_geometry_evidence"]["candidate_array_hashes_rechecked"])
        self.assertTrue(sealed["current_port_geometry_evidence"][
            "independent_rigid_baseline_array_hashes_rechecked"])
        self.assertEqual(
            sealed["current_port_geometry_evidence"]["independent_rigid_baseline_array_hashes"][
                "51011:6"]["positions_sha256"],
            digest(baseline_arrays[(51011, 6)]["positions"], "<f4"))

    def test_sealer_checks_each_face_pair_column_against_its_own_row(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case(
            unequal_face_counts=True)
        pending = self.build(parent_bytes, preflight_bytes, arrays, owner_pins)
        pending_bytes = canonical_handoff_bytes(pending)
        report_bytes = canonical_handoff_bytes(
            passing_current_port_report(pending, baseline_arrays))
        sealed = self.seal(pending_bytes, report_bytes, preflight_bytes, arrays, baseline_arrays)
        self.assertTrue(sealed["passed"])
        self.assertEqual(sealed["typed_link"], [[51011, 7], [51011, 6]])

    def test_sealer_recomputes_strict_transverse_fallback_when_face_witnesses_are_absent(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case(
            edge=((51011, 7), (51011, 8)))
        pending = self.build(parent_bytes, preflight_bytes, arrays, owner_pins)
        pending_bytes = canonical_handoff_bytes(pending)
        report = passing_current_port_report(pending, baseline_arrays)
        for role in ("candidate", "independent_source_rigid_baseline"):
            report["interface_gates"][role] = exact_transverse_interface(
                pending["typed_link"], pending["exact_face_pairs"])
        report_bytes = canonical_handoff_bytes(report)
        with patch.object(av, "recompute_current_aortic_interface_gate",
                          side_effect=lambda link, surfaces, pairs:
                          exact_transverse_interface(link, pairs)) as recompute:
            sealed = self.seal(pending_bytes, report_bytes, preflight_bytes,
                               arrays, baseline_arrays)
        self.assertTrue(sealed["passed"])
        evidence = sealed["current_port_geometry_evidence"][
            "exact_transverse_fallback_recomputation"]
        self.assertEqual(evidence["status"],
                         "PASS_RECOMPUTED_ON_CURRENT_CANDIDATE_AND_SOURCE_BASELINE")
        self.assertEqual(set(evidence["roles"]),
                         {"candidate", "independent_source_rigid_baseline"})
        self.assertEqual(recompute.call_count, 2)
        self.assertFalse(report["interface_gates"]["candidate"][
            "exact_partial_overlap_witnesses"]["boundary_predicate"]["passed"])

    def test_fallback_evidence_digest_ignores_generic_audit_schema_fields(self):
        link = ((51011, 7), (51011, 8))
        pairs = [[0, 0]]
        report_interface = exact_transverse_interface(link, pairs)
        fresh_interface = copy.deepcopy(report_interface)
        fresh_interface["audit_result"] = {
            "source_bound_connected_overlap": None,
            "source_bound_interface_triangle_pairs": [],
        }
        baseline = av.validate_recomputed_exact_transverse_fallback(
            report_interface, report_interface, link, pairs)
        with_new_diagnostic_fields = av.validate_recomputed_exact_transverse_fallback(
            report_interface, fresh_interface, link, pairs)
        self.assertEqual(
            baseline["exact_interface_evidence_sha256"],
            with_new_diagnostic_fields["exact_interface_evidence_sha256"])

        fresh_interface["exact_partial_overlap_witnesses"]["exact_transverse_overlap_side_witnesses"][
            "passing_face_pair"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "exact transverse fallback"):
            av.validate_recomputed_exact_transverse_fallback(
                report_interface, fresh_interface, link, pairs)

    def test_sealer_rejects_drifted_fallback_witness_and_current_or_source_arrays(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case(
            edge=((51011, 7), (51011, 8)))
        pending = self.build(parent_bytes, preflight_bytes, arrays, owner_pins)
        pending_bytes = canonical_handoff_bytes(pending)
        report = passing_current_port_report(pending, baseline_arrays)
        for role in ("candidate", "independent_source_rigid_baseline"):
            report["interface_gates"][role] = exact_transverse_interface(
                pending["typed_link"], pending["exact_face_pairs"])
        report["interface_gates"]["candidate"]["exact_partial_overlap_witnesses"][
            "exact_transverse_overlap_side_witnesses"]["passing_face_pair"] = [1, 1]
        report_bytes = canonical_handoff_bytes(report)
        with patch.object(av, "recompute_current_aortic_interface_gate",
                          side_effect=lambda link, surfaces, pairs:
                          exact_transverse_interface(link, pairs)):
            with self.assertRaisesRegex(ValueError, "exact transverse fallback failed"):
                self.seal(pending_bytes, report_bytes, preflight_bytes,
                          arrays, baseline_arrays)

        report["interface_gates"]["candidate"] = exact_transverse_interface(
            pending["typed_link"], pending["exact_face_pairs"])
        report_bytes = canonical_handoff_bytes(report)
        arrays[(51011, 7)]["positions"][0, 0] += np.float32(.01)
        with patch.object(av, "recompute_current_aortic_interface_gate",
                          side_effect=lambda link, surfaces, pairs:
                          exact_transverse_interface(link, pairs)):
            with self.assertRaisesRegex(ValueError, "row arrays or GID allocation changed"):
                self.seal(pending_bytes, report_bytes, preflight_bytes,
                          arrays, baseline_arrays)

        arrays[(51011, 7)]["positions"][0, 0] -= np.float32(.01)
        baseline_arrays[(51011, 7)]["positions"][0, 0] += np.float32(.01)
        with patch.object(av, "recompute_current_aortic_interface_gate",
                          side_effect=lambda link, surfaces, pairs:
                          exact_transverse_interface(link, pairs)):
            with self.assertRaisesRegex(ValueError, "differ from pinned source"):
                self.seal(pending_bytes, report_bytes, preflight_bytes,
                          arrays, baseline_arrays)

    def test_sealer_requires_each_role_solid_and_lineage_gates_for_both_interface_paths(self):
        for use_fallback in (False, True):
            edge = ((51011, 7), (51011, 8)) if use_fallback else ((51011, 7), (51011, 6))
            for role in ("candidate", "independent_source_rigid_baseline"):
                for corrupted_field in ("passed", "lineage"):
                    with self.subTest(fallback=use_fallback, role=role,
                                      corrupted_field=corrupted_field):
                        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case(
                            edge=edge)
                        pending = self.build(parent_bytes, preflight_bytes, arrays, owner_pins)
                        pending_bytes = canonical_handoff_bytes(pending)
                        report = passing_current_port_report(pending, baseline_arrays)
                        if use_fallback:
                            for interface_role in ("candidate", "independent_source_rigid_baseline"):
                                report["interface_gates"][interface_role] = exact_transverse_interface(
                                    pending["typed_link"], pending["exact_face_pairs"])
                        solid = report["solid_gates"][role]
                        if corrupted_field == "passed":
                            solid["passed"] = False
                        else:
                            solid["union_float64_mesh64_lineage"]["generated_face_count"] = 1
                        report_bytes = canonical_handoff_bytes(report)
                        geometry_patch = patch.object(
                            av, "recompute_current_aortic_interface_gate",
                            side_effect=lambda link, surfaces, pairs:
                            exact_transverse_interface(link, pairs))
                        if use_fallback:
                            with geometry_patch:
                                with self.assertRaisesRegex(ValueError,
                                        "fresh current report closed-solid gate failed: " + role):
                                    self.seal(pending_bytes, report_bytes, preflight_bytes,
                                              arrays, baseline_arrays)
                        else:
                            with self.assertRaisesRegex(ValueError,
                                    "fresh current report closed-solid gate failed: " + role):
                                self.seal(pending_bytes, report_bytes, preflight_bytes,
                                          arrays, baseline_arrays)

    def test_sealer_requires_a_fresh_current_port_report(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case()
        pending = self.build(parent_bytes, preflight_bytes, arrays, owner_pins)
        pending_bytes = canonical_handoff_bytes(pending)
        bad_report = canonical_handoff_bytes({"schema": "numi.human.current-native-aortic-port-result.v1"})
        with self.assertRaisesRegex(ValueError, "missing its typed link or recomputed source gates"):
            av.seal_current_aortic_source_handoff_from_port_report(
                pending_bytes,
                expected_pending_handoff_sha256=hashlib.sha256(pending_bytes).hexdigest(),
                current_port_report_bytes=bad_report,
                expected_current_port_report_sha256=hashlib.sha256(bad_report).hexdigest(),
                current_source_preflight_bytes=preflight_bytes,
                expected_capture_input_bundle_sha256=CAPTURE_SHA,
                capture_array_mapping=arrays,
                independent_source_rigid_baseline_array_mapping=baseline_arrays)

    def test_rejects_changed_source_member_or_face_corner_ancestry(self):
        parent_bytes, preflight_bytes, arrays, owner_pins, baseline_arrays = source_rebind_case()
        changed = json.loads(preflight_bytes)
        changed["source_rows"]["51011:7"]["owner"]["source_member"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "owner or ordered-face ancestry"):
            self.build(parent_bytes, canonical_handoff_bytes(changed), arrays, owner_pins)


if __name__ == "__main__":
    unittest.main()

