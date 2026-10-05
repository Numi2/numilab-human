"""Focused tests for the source-pinned downstream passive-neighbor compositor."""
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from tools import adjust_passive_neighbor_geometry as comp


SOURCE_SHA = bytes.fromhex("0123456789abcdef" * 4)
REG_FP = 0xB1B410AD


def encode_payload(order, vertices_by_id=None, faces_by_id=None, owner_overrides=None):
    vertices_by_id = vertices_by_id or {}
    faces_by_id = faces_by_id or {}
    owner_overrides = owner_overrides or {}
    records = []
    vertex_blob = bytearray()
    index_blob = bytearray()
    vertex_start = index_start = 0
    base_faces = [(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)]
    base_vertices = [
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
    ]
    for stable_id in order:
        vertices = vertices_by_id.get(stable_id, base_vertices)
        faces = faces_by_id.get(stable_id, base_faces)
        owner = owner_overrides.get(stable_id, 20 if stable_id in comp.STABLE_IDS else 7)
        packed_vertices = b"".join(struct.pack("<6f", *v) for v in vertices)
        packed_faces = tuple(i for tri in faces for i in tri)
        global_faces = tuple(i + vertex_start for i in packed_faces)
        vertex_blob.extend(packed_vertices)
        index_blob.extend(struct.pack("<" + "I" * len(global_faces), *global_faces))
        records.append(comp.RECORD.pack(owner, vertex_start, len(vertices), index_start,
                                        len(global_faces), stable_id, 1, 0))
        vertex_start += len(vertices)
        index_start += len(global_faces)
    header = comp.HEADER.pack(b"NHANAT1\0", 5, len(order), vertex_start, index_start,
                              REG_FP, SOURCE_SHA)
    return header + b"".join(records) + bytes(vertex_blob) + bytes(index_blob)


class PassiveNeighborCompositionTests(unittest.TestCase):
    def setUp(self):
        self.order = [4, 454, 461, 99]
        self.baseline = encode_payload(self.order)
        self.candidate_vertices = {}
        for stable_id, delta in ((4, .01), (454, .02), (461, .03)):
            verts = [
                (0.0 + delta, 0.0, 0.0, 1.0, 0.0, 0.0),
                (1.0 + delta, 0.0, 0.0, 1.0, 0.0, 0.0),
                (0.0 + delta, 1.0, 0.0, 1.0, 0.0, 0.0),
                (0.0 + delta, 0.0, 1.0, 1.0, 0.0, 0.0),
            ]
            self.candidate_vertices[stable_id] = verts
        self.candidate = encode_payload(self.order, self.candidate_vertices)
        # Downstream target has a different record order and an unrelated edited
        # surface; stable IDs and local topology, not absolute offsets, bind rows.
        target_vertices = {99: [(v[0] + .5, *v[1:]) for v in [
            (0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 1.0, 0.0, 0.0),
        ]]}
        self.target_order = [99, 4, 461, 454]
        self.target = encode_payload(self.target_order, target_vertices)

    def test_overlays_only_selected_rows_and_preserves_target_indices_and_other_rows(self):
        result, report = comp.compose_payload(self.baseline, self.candidate, self.target)
        out = comp.parse_payload(result, "test result")
        target = comp.parse_payload(self.target, "test target")
        candidate = comp.parse_payload(self.candidate, "test candidate")
        self.assertEqual(report["changed_stable_ids"], [4, 454, 461])
        self.assertTrue(report["preservation"]["target_indices_unchanged"])
        self.assertEqual(result[target["index_offset"]:], self.target[target["index_offset"]:])
        for stable_id in set(self.target_order) - set(comp.STABLE_IDS):
            self.assertEqual(comp.vertex_bytes(out, stable_id), comp.vertex_bytes(target, stable_id))
        for stable_id in comp.STABLE_IDS:
            self.assertEqual(comp.vertex_bytes(out, stable_id), comp.vertex_bytes(candidate, stable_id))
            self.assertEqual(comp.local_faces(out, stable_id), comp.local_faces(target, stable_id))
        self.assertEqual(out["header"], target["header"])

    def test_rejects_target_selected_rows_that_do_not_match_passive_baseline(self):
        altered = {4: [(0.001, 0., 0., 1., 0., 0.), (1., 0., 0., 1., 0., 0.),
                       (0., 1., 0., 1., 0., 0.), (0., 0., 1., 1., 0., 0.)]}
        wrong_target = encode_payload(self.target_order, altered)
        with self.assertRaisesRegex(comp.CompositionError, "target original vertices"):
            comp.compose_payload(self.baseline, self.candidate, wrong_target)

    def test_rejects_candidate_topology_change(self):
        altered_faces = {454: [(0, 2, 1), (0, 3, 2), (0, 1, 3), (1, 2, 3)]}
        wrong_candidate = encode_payload(self.order, self.candidate_vertices, altered_faces)
        with self.assertRaisesRegex(comp.CompositionError, "candidate local faces"):
            comp.compose_payload(self.baseline, wrong_candidate, self.target)

    def test_rejects_wrong_body_owner(self):
        wrong_owner = encode_payload(self.order, self.candidate_vertices, owner_overrides={461: 7})
        with self.assertRaisesRegex(comp.CompositionError, "body-20 association"):
            comp.compose_payload(self.baseline, wrong_owner, self.target)

    def test_rejects_nonfinite_candidate_vertices(self):
        bad = {**self.candidate_vertices, 4: [(float("nan"), 0., 0., 1., 0., 0.)] + self.candidate_vertices[4][1:]}
        wrong_candidate = encode_payload(self.order, bad)
        with self.assertRaisesRegex(comp.CompositionError, "non-finite"):
            comp.compose_payload(self.baseline, wrong_candidate, self.target)

    def test_rejects_candidate_that_splits_exact_source_duplicate_positions(self):
        base_vertices = [
            (0., 0., 0., 1., 0., 0.), (1., 0., 0., 1., 0., 0.),
            (0., 1., 0., 1., 0., 0.), (0., 0., 0., 1., 0., 0.),
        ]
        base = encode_payload(self.order, {4: base_vertices})
        candidate_vertices = {**self.candidate_vertices, 4: [
            (.01, 0., 0., 1., 0., 0.), (1.01, 0., 0., 1., 0., 0.),
            (.01, 1., 0., 1., 0., 0.), (.02, 0., 0., 1., 0., 0.),
        ]}
        candidate = encode_payload(self.order, candidate_vertices)
        target = encode_payload(self.target_order, {4: base_vertices})
        with self.assertRaisesRegex(comp.CompositionError, "split a baseline exact-position group"):
            comp.compose_payload(base, candidate, target)

    def test_composed_receipt_rebinds_output_hashes_and_recomputes_attachment_transition(self):
        target_vertices = {
            99: [(.0, -2.0, 0., 1., 0., 0.), (1., -2.0, 0., 1., 0., 0.),
                 (.0, -1.0, 0., 1., 0., 0.), (.0, -2.0, 1., 1., 0., 0.)]
        }
        target = encode_payload(self.target_order, target_vertices)
        output, _ = comp.compose_payload(self.baseline, self.candidate, target)
        source_id_map = {
            str(sid): {"source_member": member, "source_sha256": "source-" + str(sid),
                       "source_owner_metadata": {"concept_id": concept}}
            for sid, (member, concept) in comp.EXPECTED_OWNER.items()
        }
        receipt = {
            "payload": {"path": "old.nhanatomy", "sha256": "old", "input_payload_sha256": "history"},
            "functional_bindings": {
                "anatomy_payload_sha256": "old",
                "passive_viscera_geometry_binding": {
                    "upper_anchor_stable_ids": [4, 454, 461], "pelvic_anchor_stable_ids": [99],
                    "transition_superior_coordinates_m": [-9.0, -8.0],
                },
                "respiratory_geometry_binding": {"status": "preserved"},
            },
            "provenance": {
                "source_id_map": source_id_map,
                "cardiac_geometry_binding": {
                    "output_anatomy_payload_sha256": "old",
                    "ventricular_wall_binding": {"output_anatomy_payload_sha256": "old"},
                },
            },
        }
        digest = hashlib.sha256(output).hexdigest()
        updated = comp.compose_receipt(receipt, Path("/tmp/composed.nhanatomy"), digest,
                                       output, {"fixture": True}, "driver-hash")
        self.assertEqual(updated["payload"]["sha256"], digest)
        self.assertEqual(updated["payload"]["input_payload_sha256"], "history")
        self.assertEqual(updated["functional_bindings"]["anatomy_payload_sha256"], digest)
        self.assertEqual(updated["provenance"]["cardiac_geometry_binding"]["output_anatomy_payload_sha256"], digest)
        self.assertEqual(updated["provenance"]["cardiac_geometry_binding"]["ventricular_wall_binding"]["output_anatomy_payload_sha256"], digest)
        self.assertEqual(updated["functional_bindings"]["respiratory_geometry_binding"], {"status": "preserved"})
        transition = updated["functional_bindings"]["passive_viscera_geometry_binding"]["transition_superior_coordinates_m"]
        self.assertEqual(transition[0], -1.0)
        self.assertEqual(transition[1], 0.0)

    def test_receipt_binding_rejects_swapped_receipt(self):
        receipt = {"schema": "numi.human.resting-anatomy-receipt.v1",
                   "payload": {"sha256": "other-payload"}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            path.write_text(json.dumps(receipt))
            digest = comp.sha_file(path)
            with self.assertRaisesRegex(comp.CompositionError, "not bound to the pinned payload"):
                comp._validate_receipt_binding(path, receipt, digest, "expected-payload", "candidate")

    def test_path_guard_rejects_wrong_exact_predicate_module(self):
        with tempfile.TemporaryDirectory() as temp:
            wrong = Path(temp) / "predicate.py"
            wrong.write_text("# same name is not enough\n")
            with self.assertRaisesRegex(comp.CompositionError, "path does not match pinned input identity"):
                comp.assert_exact_path(wrong, comp.PREDICATE_PATH, "exact-predicate module")


if __name__ == "__main__":
    unittest.main()
