import copy
import unittest

import numpy as np

from numilab_human.resting_anatomy_verify import (
    audit_surface_pair,
    prepare_surface,
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


if __name__ == "__main__":
    unittest.main()
