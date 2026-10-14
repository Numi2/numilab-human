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

if __name__ == "__main__":
    unittest.main()
