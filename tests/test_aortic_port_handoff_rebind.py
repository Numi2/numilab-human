import hashlib
import json
import unittest

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


def source_rebind_case():
    keys = ((51011, 7), (51011, 6))
    metadata = {
        (51011, 7): ("FJ3411", "FMA3768", "arch of aorta", 700, 900),
        (51011, 6): ("FJ3413", "FMA3736", "ascending aorta", 800, 950),
    }
    parent_rows, parent_bindings, current_rows, arrays, owner_pins = {}, {}, {}, {}, {}
    pairs = [[0, 0], [1, 1]]
    pair_sha = digest(pairs, "<u4")
    for key in keys:
        row_key = f"{key[0]}:{key[1]}"
        member, fma, name, old_gid, current_gid = metadata[key]
        positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                             dtype=np.float32)
        if key[1] == 6:
            positions += np.float32(.1)
        normals = np.full_like(positions, np.float32(1.0 / np.sqrt(3.0)))
        gids = np.arange(current_gid, current_gid + len(positions), dtype=np.uint64)
        face_sha = digest(FACES, "<u4")
        position_sha = digest(positions, "<f4")
        normal_sha = digest(normals, "<f4")
        gid_sha = digest(gids, "<u8")
        parent_rows[row_key] = {
            "typed_identity": list(key), "body_index": 20, "layer": 2,
            "first_vertex": old_gid, "vertex_count": len(positions), "face_count": len(FACES),
            "faces_sha256": face_sha, "source_member": member, "fma": fma, "name": name,
        }
        parent_bindings[row_key] = {
            "typed_identity": list(key), "vertex_count": len(positions), "face_count": len(FACES),
            "positions_f32_sha256": position_sha, "faces_u32_sha256": face_sha,
            "normals_f32_sha256": normal_sha, "source_vertex_ids_u64_sha256": digest(
                np.arange(old_gid, old_gid + len(positions), dtype=np.uint64), "<u8"),
        }
        ancestry = {
            "status": "EXACT_ORDERED_FACE_CORNER_ANCESTRY_MAPPED",
            "capture_face_count": len(FACES), "source_face_count": len(FACES),
            "capture_faces_u32_sha256": face_sha, "source_faces_u32_sha256": face_sha,
        }
        owner = {
            "typed_identity": list(key), "source_member": member, "source_concept_id": fma,
            "capture_face_order_equals_nha": True, "capture_face_vertex_ancestry": ancestry,
            "source_face_count": len(FACES), "source_vertex_count": len(positions),
            "captured_positions_world_f32_sha256": position_sha,
            "captured_faces_u32_sha256": face_sha,
            "captured_global_vertex_ids_sha256": gid_sha,
            "nha_faces_u32_sha256": face_sha,
            "source_nha_row_sha256": "c" * 64, "source_member_sha256": "d" * 64,
            "rigid_source_world_xyz_f32_sha256": "e" * 64,
        }
        owner_pins[key] = {
            "typed_identity": list(key), "source_member": member, "fma": fma,
            "body_index": 20, "layer": 2, "face_count": len(FACES),
            "vertex_count": len(positions), "faces_sha256": face_sha,
            "source_member_sha256": "d" * 64, "source_nha_row_sha256": "c" * 64,
        }
        current_rows[row_key] = {
            "typed_identity": list(key), "body_index": 20, "layer": 2,
            "source_member": member, "fma": fma, "name": name,
            "first_vertex": current_gid - 800, "row_local_gid_offset": 800,
            "source_faces_count": len(FACES), "source_vertices_count": len(positions),
            "source_faces_sha256": face_sha,
            "capture_positions_f32_sha256": position_sha,
            "capture_faces_u32_sha256": face_sha,
            "capture_normals_f32_sha256": normal_sha,
            "capture_gids_u64_sha256": gid_sha, "owner": owner,
        }
        arrays[key] = {
            "positions": positions, "faces": FACES.copy(), "normals": normals,
            "source_vertex_ids": gids,
        }

    parent = {
        "schema": "numi.human.verified-aortic-arch-interface-handoff.v1",
        "status": "VERIFIED_SOURCE_BOUND_AORTIC_ARCH_INTERFACE", "passed": True,
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
    return canonical_handoff_bytes(parent), canonical_handoff_bytes(preflight), arrays, owner_pins


def passing_current_port_report(pending):
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
        topology["independent_source_rigid_baseline:" + str(key[1])] = common
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
                "point_world_m": [0.1, 0.1, 0.1],
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

    def test_rebinds_current_arrays_and_global_gid_allocation_with_ancestry(self):
        parent_bytes, preflight_bytes, arrays, owner_pins = source_rebind_case()
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

    def test_rejects_current_array_drift_from_source_preflight(self):
        parent_bytes, preflight_bytes, arrays, owner_pins = source_rebind_case()
        arrays[(51011, 7)]["positions"][0, 0] += np.float32(.001)
        with self.assertRaisesRegex(ValueError, "differ from their pinned preflight"):
            self.build(parent_bytes, preflight_bytes, arrays, owner_pins)

    def test_rejects_gid_offset_that_does_not_match_current_capture(self):
        parent_bytes, preflight_bytes, arrays, owner_pins = source_rebind_case()
        changed = json.loads(preflight_bytes)
        changed["source_rows"]["51011:7"]["row_local_gid_offset"] += 1
        with self.assertRaisesRegex(ValueError, "GIDs do not follow"):
            self.build(parent_bytes, canonical_handoff_bytes(changed), arrays, owner_pins)

    def test_changed_xyz_with_matching_preflight_stays_pending(self):
        parent_bytes, preflight_bytes, arrays, owner_pins = source_rebind_case()
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
        parent_bytes, preflight_bytes, arrays, owner_pins = source_rebind_case()
        owner_pins[(51011, 7)]["source_nha_row_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "owner or ordered-face ancestry"):
            self.build(parent_bytes, preflight_bytes, arrays, owner_pins)

    def test_fresh_report_seals_only_its_exact_current_arrays(self):
        parent_bytes, preflight_bytes, arrays, owner_pins = source_rebind_case()
        pending = self.build(parent_bytes, preflight_bytes, arrays, owner_pins)
        pending_bytes = canonical_handoff_bytes(pending)
        report_bytes = canonical_handoff_bytes(passing_current_port_report(pending))
        sealed = av.seal_current_aortic_source_handoff_from_port_report(
            pending_bytes,
            expected_pending_handoff_sha256=hashlib.sha256(pending_bytes).hexdigest(),
            current_port_report_bytes=report_bytes,
            expected_current_port_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
            current_source_preflight_bytes=preflight_bytes,
            expected_capture_input_bundle_sha256=CAPTURE_SHA,
            capture_array_mapping=arrays)
        self.assertTrue(sealed["passed"])
        self.assertEqual(sealed["status"], "VERIFIED_SOURCE_BOUND_AORTIC_ARCH_INTERFACE")
        self.assertEqual(sealed["evidence_report_sha256"], hashlib.sha256(report_bytes).hexdigest())
        self.assertTrue(sealed["current_port_geometry_evidence"]["candidate_array_hashes_rechecked"])

    def test_sealer_requires_a_fresh_current_port_report(self):
        parent_bytes, preflight_bytes, arrays, owner_pins = source_rebind_case()
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
                capture_array_mapping=arrays)

    def test_rejects_changed_source_member_or_face_corner_ancestry(self):
        parent_bytes, preflight_bytes, arrays, owner_pins = source_rebind_case()
        changed = json.loads(preflight_bytes)
        changed["source_rows"]["51011:7"]["owner"]["source_member"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "owner or ordered-face ancestry"):
            self.build(parent_bytes, canonical_handoff_bytes(changed), arrays, owner_pins)


if __name__ == "__main__":
    unittest.main()

