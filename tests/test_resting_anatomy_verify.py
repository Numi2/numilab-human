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


class RestingAnatomyVerifierTests(unittest.TestCase):

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
