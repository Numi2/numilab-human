from __future__ import annotations

import hashlib
import unittest

import numpy as np

from numilab_human.forearm_actual_capture_owner import (
    _validate_full21_parent_lane_routes,
    _verify_producer_input_pins,
    authenticate_shared_actual_capture_gid_points,
)


def row(positions, gids, slots, faces):
    positions = np.ascontiguousarray(positions, dtype="<f4")
    gids = np.ascontiguousarray(gids, dtype="<u8")
    slots = np.ascontiguousarray(slots, dtype="<i8")
    faces = np.ascontiguousarray(faces, dtype="<u4")
    ordered_slots = np.ascontiguousarray(slots[faces], dtype="<i8")
    ordered_gids = np.ascontiguousarray(gids[faces], dtype="<u8")
    return {
        "actual_capture_positions_f32": positions,
        "canonical_gids_u64": gids,
        "capture_to_source_slot_i64": slots,
        "actual_capture_faces_u32": faces,
        "report": {
            "ordered_face_corner_mapping_verified": True,
            "ordered_face_corner_source_slots_sha256": hashlib.sha256(
                ordered_slots.tobytes()).hexdigest(),
            "ordered_face_corner_canonical_gids_sha256": hashlib.sha256(
                ordered_gids.tobytes()).hexdigest(),
        },
    }


class ProducerInputRecoveryTests(unittest.TestCase):
    def test_recovery_requires_explicit_original_path_and_exact_pinned_bytes(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            original = root / "producer.py"
            recovered = root / "recovery" / "producer.py"
            recovered.parent.mkdir()
            recovered.write_bytes(b"pinned producer bytes")
            original.write_bytes(b"current changed producer bytes")
            expected = hashlib.sha256(b"pinned producer bytes").hexdigest()
            pins = {str(original.resolve()): expected}
            mapping = {str(original.resolve()): recovered}

            verified, records = _verify_producer_input_pins(pins, mapping)
            self.assertEqual(verified[str(original.resolve())], expected)
            self.assertEqual(records[0]["original_pinned_path"], str(original.resolve()))
            self.assertEqual(records[0]["recovered_blob_sha256"], expected)
            self.assertEqual(records[0]["original_path_current_sha256"],
                             hashlib.sha256(original.read_bytes()).hexdigest())

            with self.assertRaisesRegex(ValueError, "producer input pin changed"):
                _verify_producer_input_pins(pins)
            recovered.write_bytes(b"wrong recovered bytes")
            with self.assertRaisesRegex(ValueError, "recovered producer blob"):
                _verify_producer_input_pins(pins, mapping)
            matching = root / "matching.py"
            matching.write_bytes(b"already pinned")
            matching_sha = hashlib.sha256(matching.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "unused or unpinned path"):
                _verify_producer_input_pins(
                    {str(matching.resolve()): matching_sha},
                    {str(root / "other.py"): recovered})


class SharedActualCapturePointTests(unittest.TestCase):
    def setUp(self):
        self.first = row(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [100, 101, 102, 103], [4, 5, 6, 7],
            [[0, 1, 2], [0, 2, 3]],
        )
        self.second = row(
            [[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            [100, 201, 202, 203], [14, 15, 16, 17],
            [[0, 2, 1], [0, 3, 2]],
        )

    def test_shared_gid_requires_exact_actual_xyz_and_reports_ordered_face_maps(self):
        result = authenticate_shared_actual_capture_gid_points(
            {"132": self.first, "134": self.second})
        self.assertEqual(result["shared_canonical_gid_count"], 1)
        self.assertEqual(result["exact_shared_actual_point_count"], 1)
        self.assertTrue(result["all_shared_gid_actual_positions_equal"])
        self.assertFalse(result["geometry_intersection_waiver"])
        self.assertTrue(result["ordered_face_corner_maps"]["132"][
            "verified_against_exact_ordered_source_capture_crosswalk"])
        self.assertEqual(
            result["exact_shared_actual_points"][0]["132"]["source_slot"], 4)
        self.assertEqual(
            result["exact_shared_actual_points"][0]["134"]["source_slot"], 14)

    def test_full21_identity_cannot_hide_different_captured_xyz(self):
        self.second["actual_capture_positions_f32"][0, 0] = np.nextafter(
            np.float32(0), np.float32(1), dtype=np.float32)
        result = authenticate_shared_actual_capture_gid_points(
            {"132": self.first, "134": self.second})
        self.assertFalse(result["all_shared_gid_actual_positions_equal"])
        self.assertEqual(result["exact_shared_actual_point_count"], 0)
        self.assertEqual(result["mismatched_shared_gid_position_count"], 1)
        self.assertEqual(
            result["position_mismatches"][0]["actual_capture_xyz_byte_equal"], False)

    def test_face_corner_mapping_digest_is_rechecked(self):
        self.second["report"]["ordered_face_corner_source_slots_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "ordered face-corner map changed"):
            authenticate_shared_actual_capture_gid_points(
                {"132": self.first, "134": self.second})


if __name__ == "__main__":
    unittest.main()


from numilab_human.forearm_actual_capture_owner import authenticate_actual_capture_row

class Full21ParentLaneRouteTests(unittest.TestCase):
    def fixture(self):
        indices = np.zeros((21, 4), dtype="<u4")
        weights = np.zeros((21, 4), dtype="<f4")
        indices[:, 0] = np.arange(21, dtype="<u4")
        weights[:, 0] = np.float32(1)
        return indices, weights, np.arange(21, dtype="<i8"), list(range(21))

    def test_every_extension_term_must_select_its_exact_one_hot_owner_lane(self):
        indices, weights, local_ids, bodies = self.fixture()
        _validate_full21_parent_lane_routes(
            indices, weights, local_ids, bodies, stable_id=132, source_slot=0)

        wrong_lane = indices.copy()
        wrong_lane[20, 0] = 19
        with self.assertRaisesRegex(ValueError, "term 20.*exact one-hot owner lane"):
            _validate_full21_parent_lane_routes(
                wrong_lane, weights, local_ids, bodies, stable_id=132, source_slot=0)

        duplicate = local_ids.copy()
        duplicate[20] = duplicate[19]
        with self.assertRaisesRegex(ValueError, "repeats a parent lane vertex"):
            _validate_full21_parent_lane_routes(
                indices, weights, duplicate, bodies, stable_id=132, source_slot=0)

        non_one_hot = weights.copy()
        non_one_hot[20] = np.asarray([0.5, 0.5, 0, 0], dtype="<f4")
        with self.assertRaisesRegex(ValueError, "term 20.*exact one-hot owner lane"):
            _validate_full21_parent_lane_routes(
                indices, non_one_hot, local_ids, bodies,
                stable_id=132, source_slot=0)


class ActualCaptureRowOwnerTests(unittest.TestCase):
    def test_row_owner_binds_ordered_source_face_corners_to_capture(self):
        source_faces = np.asarray([[2, 0, 1]], dtype="<u4")
        capture_faces = np.asarray([[0, 1, 2]], dtype="<u4")
        positions = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="<f4")
        slots = np.asarray([0, 1, 2], dtype="<i8")
        gids = np.asarray([900, 901, 902], dtype="<u8")
        weights = np.full((3, 21), 1.0 / 21.0, dtype="<f8")
        result = authenticate_actual_capture_row(
            source_faces_u32=source_faces,
            source_vertex_count=3,
            capture_positions_f32=positions,
            capture_faces_u32=capture_faces,
            source_map_slots_i64=slots,
            source_map_gids_u64=gids,
            source_map_weights_f64=weights,
            prepared_positions_f32=positions.copy(),
            prepared_faces_i64=capture_faces.astype("<i8"),
            prepared_source_slots_i64=np.asarray([2, 0, 1], dtype="<i8"),
            prepared_gids_u64=np.asarray([902, 900, 901], dtype="<u8"),
            prepared_weights_f64=weights[[2, 0, 1]].copy(),
            index_candidate_positions_f32=positions.copy(),
            index_candidate_gids_u64=np.asarray([902, 900, 901], dtype="<u8"),
            index_candidate_faces_source_slots_u32=source_faces,
            index_candidate_faces_capture_u32=capture_faces,
        )
        self.assertTrue(result["report"]["ordered_face_corner_mapping_verified"])
        self.assertEqual(result["ordered_face_corner_source_slots_i64"].tolist(),
                         [[2, 0, 1]])
        self.assertEqual(result["ordered_face_corner_canonical_gids_u64"].tolist(),
                         [[902, 900, 901]])



class ActualCaptureProofBuilderTests(unittest.TestCase):
    def test_public_file_backed_builder_checks_shared_capture_xyz(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from numilab_human import passive_attachment_composition as pc
        from numilab_human import resting_anatomy_verify as verify
        from numilab_human.forearm_actual_capture_owner import (
            build_actual_capture_owner_proof,
        )

        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = {
                "prepared": root / "prepared.npz",
                "metadata": root / "prepared.json",
                "pack": root / "accepted.mrvpack",
                "receipt": root / "step.receipt.json",
                "resting_receipt": root / "resting-anatomy-receipt.json",
                "child": root / "child.nhtissue",
                "parent": root / "parent.nhtissue",
                "source_map": root / "source-map.npz",
                "source_map_report": root / "source-map.json",
                "index": root / "index.npz",
                "index_report": root / "index.json",
                "ancestry": root / "ancestry.json",
                "batch_report": root / "batch-report.json",
                "checkpoint": root / "checkpoint.json",
                "producer_module": root / "producer.py",
                "producer_module_recovery": root / "recovery" / "producer.py",
            }
            paths["pack"].write_bytes(b"accepted pack fixture")
            paths["receipt"].write_bytes(b"accepted receipt fixture")
            paths["child"].write_bytes(b"source child fixture")
            paths["parent"].write_bytes(b"active parent fixture")
            paths["producer_module"].write_bytes(b"historical producer source bytes")
            paths["producer_module_recovery"].parent.mkdir()
            paths["producer_module_recovery"].write_bytes(
                b"historical producer source bytes")

            positions_by_row = {
                132: np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="<f4"),
                134: np.asarray([[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype="<f4"),
            }
            faces_by_row = {
                132: np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype="<u4"),
                134: np.asarray([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype="<u4"),
            }
            gids_by_row = {
                132: np.asarray([100, 101, 102, 103], dtype="<u8"),
                134: np.asarray([100, 201, 202, 203], dtype="<u8"),
            }
            weights_by_row = {
                stable_id: np.full((4, 21), 1.0 / 21.0, dtype="<f8")
                for stable_id in (132, 134)
            }

            prepared = {}
            source_map = {}
            index = {}
            surfaces = {}
            parent_surfaces = []
            parent_records = []
            parent_row_data = {}
            for row_number, stable_id in enumerate((132, 134)):
                prefix = f"row{stable_id}_"
                positions = positions_by_row[stable_id]
                gids = gids_by_row[stable_id]
                weights = weights_by_row[stable_id]
                capture_faces = faces_by_row[stable_id]
                source_slots = np.arange(4, dtype="<i8")
                first_parent_vertex = row_number * 84
                parent_vertex_locals = np.asarray(
                    [[slot * 21 + lane for lane in range(21)] for slot in range(4)],
                    dtype="<i8")
                prepared[prefix + "accepted_positions_f32"] = positions
                prepared[prefix + "accepted_faces_i64"] = capture_faces.astype("<i8")
                prepared[prefix + "capture_to_source_slot_i64"] = source_slots
                prepared[prefix + "canonical_gids_u64"] = gids
                prepared[prefix + "full21_weights_f64"] = weights
                prepared[prefix + "full21_source_points_f32"] = positions.copy()
                prepared[prefix + "full21_parent_source_indices_u32"] = np.zeros(4, dtype="<u4")
                prepared[prefix + "full21_parent_vertex_locals_i64"] = parent_vertex_locals
                source_map[prefix + "source_slots_i64"] = source_slots
                source_map[prefix + "canonical_material_gids_u64"] = gids
                source_map[prefix + "core_weights_float64"] = weights
                index[prefix + "positions_actual_run002_f32"] = positions
                index[prefix + "canonical_material_gids_u64"] = gids
                index[prefix + "faces_source_slots_u32"] = capture_faces
                index[prefix + "faces_capture_u32"] = capture_faces
                surfaces[(51005, stable_id)] = (
                    positions, capture_faces, np.zeros_like(positions), gids)

                vertices = []
                for slot in range(4):
                    vertices.append([
                        [0, stable_id, first_parent_vertex + slot * 21 + lane,
                         float(weights[slot, lane])]
                        for lane in range(21)
                    ])
                parent_surfaces.append({"stable_id": stable_id, "vertices": vertices})
                record = [None] * 8
                record[2] = first_parent_vertex
                record[6] = stable_id
                parent_records.append(record)

                parent_positions = np.repeat(positions, 21, axis=0)
                parent_normals = np.tile(
                    np.asarray([[0, 0, 1]], dtype="<f4"), (len(parent_positions), 1))
                binding_indices = np.zeros((len(parent_positions), 4), dtype="<u4")
                binding_weights = np.zeros((len(parent_positions), 4), dtype="<f4")
                for slot in range(4):
                    for lane in range(21):
                        local = slot * 21 + lane
                        binding_indices[local, 0] = lane
                        binding_weights[local, 0] = 1
                parent_row_data[stable_id] = {
                    "positions": np.concatenate((parent_positions, parent_normals), axis=1),
                    "binding_indices": binding_indices,
                    "weights": binding_weights,
                    "bindings": [{"core": lane} for lane in range(21)],
                }

            np.savez(paths["prepared"], **prepared)
            np.savez(paths["source_map"], **source_map)
            np.savez(paths["index"], **index)
            child_sha = digest(paths["child"])
            parent_sha = digest(paths["parent"])
            parent_extension = {
                "child_payload_sha256": child_sha,
                "source_payloads": [{
                    "path": str(paths["parent"].resolve()), "sha256": parent_sha,
                }],
                "surfaces": parent_surfaces,
            }
            paths["resting_receipt"].write_text(json.dumps({
                "passive_tissue_parent_bindings": parent_extension,
            }))

            pin_paths = [
                paths["prepared"], paths["pack"], paths["receipt"], paths["resting_receipt"],
                paths["child"], paths["parent"], paths["source_map"],
                paths["source_map_report"], paths["index"], paths["index_report"],
                paths["ancestry"], paths["batch_report"], paths["checkpoint"],
                paths["producer_module"],
            ]
            arrays_sha = digest(paths["prepared"])
            pack_sha = digest(paths["pack"])
            receipt_sha = digest(paths["receipt"])
            source_map_sha = digest(paths["source_map"])
            paths["source_map_report"].write_text(json.dumps({
                "candidate_sha256": source_map_sha,
                "candidate_path": str(paths["source_map"].resolve()),
                "accepted_capture_step": 0,
            }))
            paths["index_report"].write_text(json.dumps({
                "candidate_sha256": digest(paths["index"]),
                "candidate_path": str(paths["index"].resolve()),
                "accepted_run002_pack_sha256": pack_sha,
                "accepted_run002_receipt_sha256": receipt_sha,
                "source_slots_or_gids_changed": False,
                "positions_changed": False,
                "geometry_admission": False,
                "native_admission": False,
            }))
            paths["ancestry"].write_text(json.dumps({
                "candidate_npz_sha256": digest(paths["index"]),
                "parent_nhtiss_sha256": child_sha,
                "accepted_run002_pack_sha256": pack_sha,
                "all_source_positions_and_full21_owners_unchanged": True,
                "rows": {"132": [], "134": []},
            }))
            paths["batch_report"].write_text(json.dumps({
                "prepared_owner_arrays": {"sha256": arrays_sha},
                "not_admission": True,
            }))
            paths["checkpoint"].write_text(json.dumps({
                "full21_owner_arrays_sha256": arrays_sha,
                "accepted_run002_pack_sha256": pack_sha,
                "accepted_run002_step_receipt_sha256": receipt_sha,
                "source_child_sha256": child_sha,
                "source_map_candidate_sha256": source_map_sha,
                "geometry_admission": False,
                "native_admission": False,
            }))
            pins = {str(path.resolve()): digest(path) for path in pin_paths}
            paths["metadata"].write_text(json.dumps({
                "schema": "numi.human.forearm.prepared-owner-arrays.v1",
                "not_admission": True,
                "sha256": arrays_sha,
                "accepted_run002_pack_sha256": pack_sha,
                "accepted_run002_step_receipt_sha256": receipt_sha,
                "source_child_sha256": child_sha,
                "source_map_candidate_sha256": source_map_sha,
                "input_sha256": pins,
            }))
            paths["producer_module"].write_bytes(b"newer source at original path")

            source_records = []
            for stable_id in (132, 134):
                record = [None] * 8
                record[2] = 0
                record[6] = stable_id
                source_records.append(record)
            source_data = {"records": source_records, "fixture_kind": "child"}
            parent_data = {"records": parent_records, "fixture_kind": "parent"}
            identity = {
                "pack_sha256": pack_sha,
                "capture_receipt_sha256": receipt_sha,
                "accepted_step": 0,
            }

            def read_nhtiss(path):
                return parent_data if Path(path).resolve() == paths["parent"].resolve() else source_data

            def row_arrays(data, row_record):
                stable_id = int(row_record[6])
                if data is source_data:
                    return {"faces": faces_by_row[stable_id], "vertex_count": 4}
                return parent_row_data[stable_id]

            with patch.object(pc, "_read_nhtiss4", side_effect=read_nhtiss), \
                 patch.object(pc, "_nhtiss_row_arrays", side_effect=row_arrays), \
                 patch.object(verify, "load_capture", return_value=(surfaces, identity)):
                proof = build_actual_capture_owner_proof(
                    prepared_owner_arrays_path=paths["prepared"],
                    prepared_owner_metadata_path=paths["metadata"],
                    accepted_pack_path=paths["pack"],
                    accepted_step_receipt_path=paths["receipt"],
                    source_child_path=paths["child"],
                    source_map_candidate_path=paths["source_map"],
                    source_map_report_path=paths["source_map_report"],
                    index_candidate_path=paths["index"],
                    index_report_path=paths["index_report"],
                    index_ancestry_path=paths["ancestry"],
                    witness_batch_report_path=paths["batch_report"],
                    witness_batch_checkpoint_path=paths["checkpoint"],
                    producer_input_recovery_paths={
                        str(paths["producer_module"].resolve()):
                            paths["producer_module_recovery"]},
                )

            shared = proof["shared_actual_capture_gid_points"]
            self.assertEqual(shared["shared_canonical_gid_count"], 1)
            self.assertEqual(shared["exact_shared_actual_point_count"], 1)
            self.assertTrue(shared["all_shared_gid_actual_positions_equal"])
            self.assertFalse(shared["geometry_intersection_waiver"])
            self.assertFalse(proof["face_topology_provenance"]["132"][
                "complete_source_face_ancestry"])
            self.assertEqual(proof["rows"]["132"]["report"][
                "ordered_face_corner_mapping_verified"], True)
            self.assertEqual(len(proof["producer_input_recoveries"]), 1)
            self.assertEqual(
                proof["producer_input_recoveries"][0]["original_pinned_path"],
                str(paths["producer_module"].resolve()))
            self.assertEqual(
                proof["producer_input_recoveries"][0]["recovered_blob_sha256"],
                digest(paths["producer_module_recovery"]))

            prepared_surfaces = {}
            for stable_id in (132, 134):
                row = proof["rows"][str(stable_id)]
                gids = np.asarray(row["canonical_gids_u64"], dtype="<u8")
                face_lineage = [{
                    "source_member": "fixture.obj",
                    "source_member_sha256": "a" * 64,
                    "source_face": face_index,
                    "source_row_face": face_index,
                    "source_row_payload_sha256": child_sha,
                    "owner_stable_id": stable_id,
                } for face_index in range(len(row["index_patch_faces_capture_u32"]))]
                vertex_lineage = {
                    int(gid): {"source_gid": int(gid)} for gid in gids
                }
                prepared_surfaces[stable_id] = verify.prepare_surface(
                    (51005, stable_id),
                    np.asarray(row["actual_capture_positions_f32"], dtype="<f4"),
                    np.asarray(row["index_patch_faces_capture_u32"], dtype="<u4"),
                    source_vertex_ids=gids,
                    source_face_lineage=face_lineage,
                    source_vertex_lineage=vertex_lineage,
                    ordered_core_body_indices=row["ordered_core_body_indices"],
                    core_weight_gids=gids,
                    core_weights_float64=np.asarray(row["full21_weights_f64"], dtype="<f8"),
                    core_source_point_gids=gids,
                    core_source_points_float32=np.asarray(row["full21_source_points_f32"], dtype="<f4"),
                    core_capture_source_slots_i64=np.asarray(
                        row["capture_to_source_slot_i64"], dtype="<i8"),
                    core_actual_capture_proof=row["full21_actual_capture_owner_proof"],
                )
            pair = verify.audit_surface_pair(
                prepared_surfaces[132], prepared_surfaces[134],
                intended_shared_feature_pairs=[
                    {
                        "first_identity": [51005, 132],
                        "second_identity": [51005, 134],
                        "first_face_index": first_face,
                        "second_face_index": second_face,
                        "feature_gids": [100],
                    }
                    for first_face in range(3)
                    for second_face in range(3)
                ],
            )
            self.assertTrue(pair["passed"], pair)
            self.assertEqual(len(pair["declared_interface_feature_contacts"]), 9)
            self.assertEqual(
                pair["declared_interface_feature_contacts"][0]["proof"],
                "exact_shared_GID_feature_with_source_and_owner_proof",
            )
