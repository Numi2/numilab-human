"""Input-binding regressions for the shared resting geometry entrypoint."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

_SPEC = importlib.util.spec_from_file_location("verify_resting_entrypoint", Path(__file__).resolve().parents[1] / "tools/verify_resting_anatomy.py")
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)

KEY = (51005, 73)
POINTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="<f4")
FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype="<u4")
GIDS = np.arange(4, dtype="<u4") + 100
NORMALS = np.tile(np.array([[0, 0, 1]], dtype="<f4"), (4, 1))


class RestingVerificationEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def candidate(self, **changes):
        arrays = dict(capture_positions_m=POINTS, capture_faces=FACES, capture_global_vertex_ids=GIDS,
                      source_positions=POINTS, source_normals=NORMALS, source_faces=FACES)
        arrays.update(changes)
        path = self.root / "candidate.npz"
        np.savez(path, **arrays)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def surfaces(self):
        return {KEY: (POINTS.copy(), FACES.copy(), None, GIDS.copy())}

    def construction(self, **changes):
        arrays = dict(capture_positions_m=POINTS, capture_faces=FACES,
                      capture_global_vertex_ids=GIDS)
        arrays.update(changes)
        path = self.root / "construction.npz"
        np.savez(path, **arrays)
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    def test_explicit_construction_is_never_source_or_native_admission(self):
        path, digest = self.construction()
        surfaces = self.surfaces()
        report = tool.apply_construction_overlay(surfaces, KEY, path, digest)
        self.assertFalse(report["source_ancestry_admission"])
        self.assertFalse(report["actual_native_emission_verified"])
        self.assertFalse(report["material_fields_and_rest_owner_verified"])
        self.assertFalse(report["anatomical_admission"])
        self.assertEqual(report["coordinate_provenance"], "inferred_reference_construction_only")
        self.assertTrue(np.array_equal(surfaces[KEY][0], POINTS))
        self.assertTrue(np.allclose(np.linalg.norm(surfaces[KEY][2], axis=1), 1))

    def test_construction_float64_is_rejected_before_predicates(self):
        path, digest = self.construction(capture_positions_m=POINTS.astype("<f8"))
        with patch.object(tool.verify, "prepare_surface", side_effect=AssertionError("no predicate")):
            with self.assertRaisesRegex(ValueError, "Float32 metres"):
                tool.apply_construction_overlay(self.surfaces(), KEY, path, digest)

    def test_construction_cannot_replace_a_missing_anatomical_identity(self):
        path, digest = self.construction()
        with self.assertRaisesRegex(ValueError, "existing typed"):
            tool.apply_construction_overlay({}, KEY, path, digest)

    def test_construction_duplicate_global_ids_are_rejected(self):
        path, digest = self.construction(capture_global_vertex_ids=np.array([1, 1, 2, 3]))
        with self.assertRaises(ValueError):
            tool.apply_construction_overlay(self.surfaces(), KEY, path, digest)

    def test_construction_does_not_silently_ignore_source_fields(self):
        path, digest = self.construction(source_positions=POINTS)
        with self.assertRaisesRegex(ValueError, "exactly"):
            tool.apply_construction_overlay(self.surfaces(), KEY, path, digest)

    def test_degenerate_float32_construction_is_rejected(self):
        bad = POINTS.copy()
        bad[2] = (bad[0] + bad[1])/2
        path, digest = self.construction(capture_positions_m=bad)
        with self.assertRaisesRegex(ValueError, "degenerate emitted"):
            tool.apply_construction_overlay(self.surfaces(), KEY, path, digest)

    def test_wrong_hash_rejected_without_loading_archive(self):
        path, _ = self.candidate()
        with patch.object(tool.np, "load", side_effect=AssertionError("must not load")):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                tool.apply_overlay(self.surfaces(), KEY, path, "0" * 64)

    def test_reordered_source_ids_rejected(self):
        path, digest = self.candidate(capture_global_vertex_ids=GIDS[::-1])
        with self.assertRaisesRegex(ValueError, "ordered captured source GID"):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_float64_is_not_silently_rounded_for_predicates(self):
        path, digest = self.candidate(capture_positions_m=POINTS.astype("<f8"))
        with self.assertRaisesRegex(ValueError, "exact Float32"):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_face_index_bounds_use_shared_preparation(self):
        bad = FACES.copy()
        bad[0, 0] = 999
        path, digest = self.candidate(capture_faces=bad)
        with self.assertRaises(ValueError):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_invalid_normals_are_rejected_before_pair_predicates(self):
        path, digest = self.candidate(capture_normals=np.full_like(POINTS, np.nan))
        with self.assertRaises(ValueError):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_no_guessing_faces_key_and_no_native_admission(self):
        path = self.root / "candidate.npz"
        np.savez(path, capture_positions_m=POINTS, capture_faces_candidate=FACES, capture_global_vertex_ids=GIDS,
                 source_positions=POINTS, source_normals=NORMALS, source_faces=FACES)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(KeyError):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)
        surfaces = self.surfaces()
        report = tool.apply_overlay(surfaces, KEY, path, digest, "capture_faces_candidate")
        self.assertTrue(report["world_positions_are_counterfactual"])
        self.assertTrue(report["all_ordered_capture_gids_exact"])
        self.assertFalse(report["candidate_normals_present"])
        self.assertIsInstance(surfaces[KEY], tool.verify.PreparedSurface)

    def test_missing_normals_stop_before_exact_geometry(self):
        path = self.root / "candidate.npz"
        np.savez(path, capture_positions_m=POINTS, capture_faces=FACES, capture_global_vertex_ids=GIDS)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with patch.object(tool.verify, "prepare_surface", side_effect=AssertionError("no predicates")):
            with self.assertRaisesRegex(ValueError, "requires capture normals or validated source normals"):
                tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_stale_duplicate_source_fields_are_rejected(self):
        vertices = np.c_[POINTS, NORMALS].astype("<f4")
        vertices[0, 0] += 0.001
        path, digest = self.candidate(vertices6=vertices)
        with self.assertRaisesRegex(ValueError, "duplicate source positions disagree"):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_raw_vertex_records_must_match_declared_source_fields(self):
        vertices = np.c_[POINTS, NORMALS].astype("<f4")
        raw = np.zeros((4, 56), dtype=np.uint8)
        raw[:, :24] = vertices.view(np.uint8).reshape(4, 24)
        raw[0, 0] ^= 1
        path, digest = self.candidate(vertices6=vertices, vertex_records_raw=raw)
        with self.assertRaisesRegex(ValueError, "raw ABI5 vertex records disagree"):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_unused_source_normal_slots_do_not_invalidate_closed_surface(self):
        positions = np.vstack([POINTS, np.zeros((1, 3), dtype="<f4")])
        normals = np.vstack([NORMALS, np.zeros((1, 3), dtype="<f4")])
        path, digest = self.candidate(source_positions=positions, source_normals=normals)
        report = tool.apply_overlay(self.surfaces(), KEY, path, digest)
        self.assertEqual(report["source_field_preflight"]["referenced_source_normal_count"], 4)

    def test_invalid_rigid_rest_transform_is_rejected(self):
        raw = np.zeros((1, 36), dtype=np.uint8)
        rest = np.array([[0, 0, 0, 0, 0, 0, 2, 1]], dtype="<f4")
        raw[:, 4:] = rest.view(np.uint8).reshape(1, 32)
        path, digest = self.candidate(binding_records_raw=raw)
        with self.assertRaisesRegex(ValueError, "rigid rest transforms"):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_source_normals_are_not_implicitly_rounded(self):
        path, digest = self.candidate(source_normals=NORMALS.astype("<f8"))
        with self.assertRaisesRegex(ValueError, "exact Float32"):
            tool.apply_overlay(self.surfaces(), KEY, path, digest)

    def test_shared_counterfactual_uses_same_verifier_without_native_claim(self):
        from tests.test_resting_shared_partition_verify import scene
        from numilab_human import resting_shared_partition_verify as shared
        source, digest, captures = scene(self.root)
        pack, receipt = self.root/"pack", self.root/"receipt"
        pack.write_bytes(b"pack")
        receipt.write_bytes(b"receipt")
        args = ["--pack", str(pack), "--pack-sha256", hashlib.sha256(pack.read_bytes()).hexdigest(),
                "--capture-receipt", str(receipt), "--receipt-sha256", hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "--source-pin", str(source)+"="+digest,
                "--shared-partition", "51005:1", "51005:2", str(source), digest]
        for key, (x,f,n,g) in captures.items():
            candidate = self.root/f"candidate-{key[1]}.npz"
            np.savez(candidate, capture_positions_m=x, capture_faces=f,
                     capture_normals=n, capture_global_vertex_ids=g)
            args += ["--overlay", f"51005:{key[1]}={candidate}="+hashlib.sha256(candidate.read_bytes()).hexdigest(),
                     "--changed", f"51005:{key[1]}"]
        output = self.root/"shared.json"
        with patch.object(tool.verify, "load_capture", return_value=(dict(captures), {"accepted_step": 0})):
            result = tool.main(args+["--output", str(output)])
        report = json.loads(output.read_text())
        self.assertEqual(result, 0)
        self.assertTrue(report["counterfactual_candidates"])
        proof = report["shared_partition_certificates"][0]
        self.assertEqual(proof["coordinate_provenance"], "counterfactual_owner_replay")
        self.assertNotIn("exact_native_partition_certificate", proof)
        self.assertFalse(report["native_admission"])
        error = ValueError("retained exact witness")
        error.diagnostic = {"stage": "emitted_cross_domain_contact", "first_face": 7, "second_face": 9}
        failed = self.root/"shared-failed.json"
        # scene() arrays were not changed by the first invocation.
        with patch.object(tool.verify, "load_capture", return_value=(dict(captures), {"accepted_step": 0})):
            with patch.object(shared, "audit_with_shared_partitions", side_effect=error):
                self.assertEqual(tool.main(args+["--output", str(failed)]), 1)
        self.assertEqual(json.loads(failed.read_text())["exact_failure_witness"], error.diagnostic)

    def test_disjoint_typed_construction_keeps_exact_shared_partition(self):
        from tests.test_resting_shared_partition_verify import scene
        from numilab_human import resting_shared_partition_verify as shared
        source, digest, captures = scene(self.root)
        # Same numeric ID, different semantic identity: the skin construction
        # must not replace or excuse either participant of the muscle proof.
        skin = (51007, 1)
        captures[skin] = (POINTS + np.float32(10), FACES.copy(),
                          NORMALS.copy(), GIDS.copy() + 1000)
        candidate = self.root / "skin-construction.npz"
        np.savez(candidate, capture_positions_m=captures[skin][0],
                 capture_faces=FACES, capture_global_vertex_ids=captures[skin][3])
        pack, receipt = self.root / "pack", self.root / "receipt"
        pack.write_bytes(b"pack")
        receipt.write_bytes(b"receipt")
        output = self.root / "disjoint-shared.json"
        args = ["--pack", str(pack), "--pack-sha256", hashlib.sha256(pack.read_bytes()).hexdigest(),
                "--capture-receipt", str(receipt), "--receipt-sha256", hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "--source-pin", str(source) + "=" + digest,
                "--shared-partition", "51005:1", "51005:2", str(source), digest,
                "--construction-overlay", "51007:1=" + str(candidate) + "=" + hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "--changed", "51005:1", "--target", "51005:2", "--target", "51007:1",
                "--output", str(output)]
        with patch.object(tool.verify, "load_capture", return_value=(dict(captures), {"accepted_step": 0})):
            with patch.object(shared, "audit_with_shared_partitions",
                              wraps=shared.audit_with_shared_partitions) as exact:
                self.assertEqual(tool.main(args), 0)
                exact.assert_called_once()
        report = json.loads(output.read_text())
        self.assertTrue(report["passed"])
        self.assertEqual(report["shared_partition_certificates"][0]["shared_interface_face_count"], 1)
        self.assertTrue(report["counterfactual_candidates"])
        self.assertFalse(report["native_admission"])
        self.assertFalse(report["overlays"][0]["material_fields_and_rest_owner_verified"])

    def test_construction_cannot_replace_either_partition_participant(self):
        from tests.test_resting_shared_partition_verify import scene
        from numilab_human import resting_shared_partition_verify as shared
        source, digest, captures = scene(self.root)
        path, candidate_digest = self.construction()
        pack, receipt = self.root / "pack", self.root / "receipt"
        pack.write_bytes(b"pack")
        receipt.write_bytes(b"receipt")
        for sid in (1, 2):
            with self.subTest(participant=sid):
                output = self.root / f"participant-{sid}.json"
                args = ["--pack", str(pack), "--pack-sha256", hashlib.sha256(pack.read_bytes()).hexdigest(),
                        "--capture-receipt", str(receipt), "--receipt-sha256", hashlib.sha256(receipt.read_bytes()).hexdigest(),
                        "--source-pin", str(source) + "=" + digest,
                        "--shared-partition", "51005:1", "51005:2", str(source), digest,
                        "--construction-overlay", f"51005:{sid}={path}={candidate_digest}",
                        "--changed", "51005:1", "--output", str(output)]
                with patch.object(tool.verify, "load_capture", return_value=(dict(captures), {"accepted_step": 0})):
                    with patch.object(shared, "audit_with_shared_partitions") as exact:
                        self.assertEqual(tool.main(args), 1)
                        exact.assert_not_called()
                self.assertIn("source-interface participant", json.loads(output.read_text())["error"])

    def test_absent_shared_interface_stops_before_any_exact_certificate(self):
        from tests.test_resting_shared_partition_verify import scene
        from numilab_human import passive_attachment_composition as pac
        from numilab_human import resting_shared_partition_verify as shared
        source, _, captures = scene(self.root)
        data = pac._read_nhtiss4(source)
        raw = bytearray(data["raw"])
        row = data["records"][1]
        slots = np.ndarray((4, 4), "<u4", buffer=raw,
                           offset=data["vertex_start"]+int(row[2])*56+24,
                           strides=(56, 4))
        slots[:, 0] = 1
        source.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        pack, receipt = self.root/"pack", self.root/"receipt"
        pack.write_bytes(b"pack")
        receipt.write_bytes(b"receipt")
        args = ["--pack", str(pack), "--pack-sha256",
                hashlib.sha256(pack.read_bytes()).hexdigest(),
                "--capture-receipt", str(receipt), "--receipt-sha256",
                hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "--changed", "51005:1", "--shared-partition",
                "51005:1", "51005:2", str(source), digest]
        for only in (False, True):
            with self.subTest(preflight_only=only):
                output = self.root/f"absent-shared-{only}.json"
                with patch.object(tool.verify, "load_capture",
                                  return_value=(dict(captures), {"accepted_step": 0})):
                    with patch.object(shared, "audit_with_shared_partitions") as exact:
                        self.assertEqual(tool.main(args+["--output", str(output)]
                                                   +(["--preflight-only"] if only else [])), 1)
                        exact.assert_not_called()
                report = json.loads(output.read_text())
                self.assertEqual(report["failed_stage"], "preparation")
                self.assertIn("no reciprocal source/material shared interface", report["error"])

    def repacked_candidate(self):
        from tests.test_resting_shared_partition_verify import scene
        from numilab_human import passive_attachment_composition as pac
        source, digest, _ = scene(self.root)
        table = pac._read_nhtiss4(source)
        row = pac._nhtiss_row_arrays(table, table["records"][0])
        poses = {core: np.array([.125 if core == 41 else 0., 0, 0, 0, 0, 0, 1], "<f4")
                 for core in (41, 25, 29, 20)}
        world = pac._native_blended_rest_positions(row, row["positions"], poses)
        arrays = dict(capture_positions_m=world, capture_faces=row["faces"].astype("<u4"),
                      capture_global_vertex_ids=np.arange(row["vertex_count"], dtype="<u8")+9000,
                      capture_normals=row["normals"], source_positions=row["positions"],
                      source_normals=row["normals"], source_faces=row["faces"].astype("<u4"),
                      binding_indices=row["binding_indices"], weights=row["weights"])
        # Original native allocation includes an additional unused source slot.
        key = (51005, 1)
        native = {key: (np.vstack([POINTS, [[2, 2, 2]]]).astype("<f4"), FACES.copy(),
                        None, np.arange(5, dtype="<u4")+100)}
        return source, digest, arrays, poses, native, key

    def test_repacked_overlay_binds_compiled_row_and_exact_rest_owner(self):
        source, digest, arrays, poses, native, key = self.repacked_candidate()
        path = self.root/"repacked.npz"
        np.savez(path, **arrays)
        proof = tool.apply_overlay(native, key, path, hashlib.sha256(path.read_bytes()).hexdigest(),
                                   repacked_source=(source, digest), accepted_poses=poses)
        self.assertFalse(proof["all_ordered_capture_gids_exact"])
        self.assertTrue(proof["repacked_source_owner"]["all_compiled_source_fields_exact"])
        self.assertTrue(proof["repacked_source_owner"]["cpu_rest_owner_emission_byte_exact"])
        self.assertFalse(proof["repacked_source_owner"]["actual_native_emission_verified"])

    def test_repacked_source_rejects_world_material_and_rest_drift_before_geometry(self):
        source, digest, arrays, poses, native, key = self.repacked_candidate()
        for field, index in (("capture_positions_m", (0, 0)), ("weights", (0, 0)),
                             ("source_positions", (0, 0))):
            changed = {k: v.copy() for k, v in arrays.items()}
            changed[field][index] = np.nextafter(changed[field][index], np.float32(2.))
            path = self.root/f"drift-{field}.npz"
            np.savez(path, **changed)
            with patch.object(tool.verify, "prepare_surface", side_effect=AssertionError("no geometry")):
                with self.assertRaises(ValueError):
                    tool.apply_overlay(dict(native), key, path, hashlib.sha256(path.read_bytes()).hexdigest(),
                                       repacked_source=(source, digest), accepted_poses=poses)
        path = self.root/"rest-drift.npz"
        np.savez(path, **arrays)
        badposes = {k: v.copy() for k, v in poses.items()}
        badposes[41][3:] = [0, 0, 0, 2]
        with self.assertRaisesRegex(ValueError, "rigid rest pose"):
            tool.apply_overlay(native, key, path, hashlib.sha256(path.read_bytes()).hexdigest(),
                               repacked_source=(source, digest), accepted_poses=badposes)

    def test_repacked_overlay_rejects_reordered_gids_and_stale_source_hash(self):
        source, digest, arrays, poses, native, key = self.repacked_candidate()
        arrays["capture_global_vertex_ids"] = arrays["capture_global_vertex_ids"][::-1]
        path = self.root/"gid-drift.npz"
        np.savez(path, **arrays)
        for sha in (digest, "0"*64):
            with self.assertRaises((ValueError, tool.verify.HumanImportError)):
                tool.apply_overlay(dict(native), key, path, hashlib.sha256(path.read_bytes()).hexdigest(),
                                   repacked_source=(source, sha), accepted_poses=poses)

    def test_repacked_cli_preflight_uses_pinned_accepted_poses(self):
        source, digest, arrays, poses, native, key = self.repacked_candidate()
        candidate, pack, receipt = (self.root/n for n in ("repacked.npz", "pack", "receipt"))
        np.savez(candidate, **arrays)
        pack.write_bytes(b"native-pack")
        receipt.write_text(json.dumps({"accepted_body_poses": [
            {"body_index": k, "position_m": list(map(float, v[:3])),
             "quaternion_xyzw": list(map(float, v[3:]))} for k, v in poses.items()]}))
        output = self.root/"preflight.json"
        args = ["--pack", str(pack), "--pack-sha256", hashlib.sha256(pack.read_bytes()).hexdigest(),
                "--capture-receipt", str(receipt), "--receipt-sha256", hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "--source-pin", str(source)+"="+digest,
                "--overlay", f"51005:1={candidate}="+hashlib.sha256(candidate.read_bytes()).hexdigest(),
                "--overlay-source", f"51005:1={source}="+digest, "--changed", "51005:1",
                "--preflight-only", "--output", str(output)]
        with patch.object(tool.verify, "load_capture", return_value=(native, {"accepted_step": 0})):
            self.assertEqual(tool.main(args), 0)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "PASS_PREPARATION_ONLY_NO_PAIR_PREDICATES")
        self.assertTrue(report["overlays"][0]["repacked_source_owner"]["cpu_rest_owner_emission_byte_exact"])
        self.assertFalse(report["native_admission"])


    def repair_manifest(self, source, digest, key):
        candidate, report = self.root/"patch.npz", self.root/"source-report.json"
        candidate.write_bytes(b"frozen existing repair")
        report.write_text('{"status":"source preparation only"}')
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        manifest = source.with_suffix(".manifest.json")
        manifest.write_text(json.dumps({"payload": {"sha256": digest},
            "source": {"surfaces": [{"stable_id": key[1],
                "reference_surface_reconstruction": {
                    "stable_id": key[1], "candidate": str(candidate),
                    "candidate_sha256": sha(candidate), "report": str(report),
                    "report_sha256": sha(report)}}]}}))
        return candidate, report, manifest

    def test_nested_repair_inputs_are_pinned_before_geometry(self):
        source, digest, arrays, poses, native, key = self.repacked_candidate()
        _, report, _ = self.repair_manifest(source, digest, key)
        report.write_text('{"status":"mutated after compiler froze it"}')
        with patch.object(tool.verify, "prepare_surface", side_effect=AssertionError("no predicates")):
            with self.assertRaisesRegex(ValueError, "source SHA256 mismatch"):
                tool.apply_rest_source(native, key, source, digest, poses, {})

    def test_nested_candidate_and_manifest_payload_are_checked(self):
        source, digest, arrays, poses, native, key = self.repacked_candidate()
        candidate, _, manifest = self.repair_manifest(source, digest, key)
        candidate.write_bytes(b"stale patch bytes")
        with self.assertRaisesRegex(ValueError, "source SHA256 mismatch"):
            tool.apply_rest_source(native, key, source, digest, poses, {})
        value = json.loads(manifest.read_text())
        value["payload"]["sha256"] = "0"*64
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "manifest payload SHA256 differs"):
            tool.apply_rest_source(native, key, source, digest, poses, {})

    def test_valid_repair_references_share_source_cache_without_ancestry_claim(self):
        source, digest, arrays, poses, native, key = self.repacked_candidate()
        self.repair_manifest(source, digest, key)
        cache = {}
        proof = tool.apply_rest_source(native, key, source, digest, poses, cache)
        refs = proof["repacked_source_owner"]["source_reconstruction_references"]
        self.assertEqual(set(refs["declared_reconstruction_inputs"]), {"candidate", "report"})
        self.assertFalse(refs["source_ancestry_admission"])
        self.assertEqual(len(cache), 1)
        with patch.object(tool.json, "loads", side_effect=AssertionError("must reuse manifest proof")):
            tool.apply_rest_source(native, key, source, digest, poses, cache)

    def test_bad_capture_pin_stops_before_capture_loader(self):
        pack, receipt = self.root / "p", self.root / "r"
        pack.write_bytes(b"pack")
        receipt.write_bytes(b"receipt")
        output = self.root / "report.json"
        with patch.object(tool.verify, "load_capture", side_effect=AssertionError("must not load")):
            result = tool.main(["--pack", str(pack), "--pack-sha256", "0"*64, "--capture-receipt", str(receipt),
                                "--receipt-sha256", hashlib.sha256(receipt.read_bytes()).hexdigest(),
                                "--changed", "51005:73", "--output", str(output)])
        report = json.loads(output.read_text())
        self.assertEqual(result, 1)
        self.assertEqual(report["failed_stage"], "preparation")
        self.assertFalse(report["native_admission"])
        self.assertFalse(report["anatomical_admission"])


    def test_rest_world_normals_preserve_unused_slots_without_invalid_normals(self):
        points = np.vstack([POINTS, [[2, 2, 2]]]).astype("<f4")
        normals, count = tool.rest_world_normals(points, FACES)
        self.assertEqual(count, 1)
        np.testing.assert_array_equal(normals[4], [1, 0, 0])
        self.assertTrue(tool.verify.prepare_surface(KEY, points, FACES, normals=normals).closed)
        np.testing.assert_array_equal(points[:4], POINTS)

    def test_rest_world_normals_still_reject_zero_referenced_stars(self):
        points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype="<f4")
        with self.assertRaisesRegex(ValueError, "invalid referenced world normals"):
            tool.rest_world_normals(points, np.array([[0, 1, 2]], dtype="<u4"))

    def test_rest_source_replays_compiled_owner_without_an_npz(self):
        source, digest, arrays, poses, native, key = self.repacked_candidate()
        cache = {}
        proof = tool.apply_rest_source(native, key, source, digest, poses, cache)
        np.testing.assert_array_equal(native[key].positions, arrays["capture_positions_m"])
        self.assertEqual(len(cache), 1)
        self.assertTrue(proof["repacked_source_owner"]["cpu_rest_owner_emission_byte_exact"])
        self.assertFalse(proof["repacked_source_owner"]["actual_native_emission_verified"])
        self.assertTrue(proof["world_positions_are_counterfactual"])
        self.assertIn("not_parent_tissue_deformation", proof["preparation"])
        # The same pinned compiled source is loaded once across a pose replay.
        with patch("numilab_human.passive_attachment_composition._read_nhtiss4",
                   side_effect=AssertionError("source must be cached")):
            tool.apply_rest_source(native, key, source, digest, poses, cache)

    def test_rest_source_rejects_stale_hash_and_invalid_poses_before_geometry(self):
        source, digest, arrays, poses, native, key = self.repacked_candidate()
        badposes = {k: v.copy() for k, v in poses.items()}
        badposes[41][3:] = [0, 0, 0, 2]
        with patch.object(tool.verify, "prepare_surface", side_effect=AssertionError("no geometry")):
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                tool.apply_rest_source(native, key, source, "0"*64, poses, {})
            with self.assertRaisesRegex(ValueError, "rigid rest pose"):
                tool.apply_rest_source(native, key, source, digest, badposes, {})

    def test_rest_source_cli_preflight_preserves_counterfactual_boundary(self):
        source, digest, arrays, poses, native, key = self.repacked_candidate()
        pack, receipt = self.root/"pack", self.root/"receipt"
        pack.write_bytes(b"native-pack")
        receipt.write_text(json.dumps({"accepted_body_poses": [
            {"body_index": k, "position_m": list(map(float, v[:3])),
             "quaternion_xyzw": list(map(float, v[3:]))} for k, v in poses.items()]}))
        output = self.root/"rest-source.json"
        args = ["--pack", str(pack), "--pack-sha256", hashlib.sha256(pack.read_bytes()).hexdigest(),
                "--capture-receipt", str(receipt), "--receipt-sha256", hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "--source-pin", str(source)+"="+digest,
                "--rest-source", f"51005:1={source}="+digest,
                "--changed", "51005:1", "--preflight-only", "--output", str(output)]
        with patch.object(tool.verify, "load_capture", return_value=(native, {"accepted_step": 0})):
            self.assertEqual(tool.main(args), 0)
        report = json.loads(output.read_text())
        self.assertTrue(report["counterfactual_candidates"])
        self.assertFalse(report["native_admission"])
        self.assertEqual(report["status"], "PASS_PREPARATION_ONLY_NO_PAIR_PREDICATES")
        self.assertEqual(len(report["overlays"]), 1)
        self.assertFalse(list(self.root.glob("*.npz")))
        failed = self.root/"rest-source-duplicate.json"
        duplicate = args[:-2]+["--rest-source", f"51005:1={source}="+digest, "--output", str(failed)]
        with patch.object(tool.verify, "load_capture", return_value=(native, {"accepted_step": 0})):
            self.assertEqual(tool.main(duplicate), 1)
        self.assertIn("duplicate candidate typed identity", json.loads(failed.read_text())["error"])


    def test_rest_source_shared_certificate_remains_counterfactual(self):
        from tests.test_resting_shared_partition_verify import scene
        source, digest, captures = scene(self.root)
        pack, receipt = self.root/"pack", self.root/"receipt"
        pack.write_bytes(b"native-pack")
        receipt.write_text(json.dumps({"accepted_body_poses": [
            {"body_index": k, "position_m": [0,0,0], "quaternion_xyzw": [0,0,0,1]}
            for k in (41,25,29,20)]}))
        output = self.root/"rest-source-shared.json"
        args = ["--pack", str(pack), "--pack-sha256", hashlib.sha256(pack.read_bytes()).hexdigest(),
                "--capture-receipt", str(receipt), "--receipt-sha256", hashlib.sha256(receipt.read_bytes()).hexdigest(),
                "--source-pin", str(source)+"="+digest,
                "--shared-partition", "51005:1", "51005:2", str(source), digest]
        for sid in (1,2):
            args += ["--rest-source", f"51005:{sid}={source}="+digest, "--changed", f"51005:{sid}"]
        with patch.object(tool.verify, "load_capture", return_value=(captures, {"accepted_step": 0})):
            self.assertEqual(tool.main(args+["--output", str(output)]), 0)
        report = json.loads(output.read_text())
        self.assertFalse(report["native_admission"])
        proof = report["shared_partition_certificates"][0]
        self.assertTrue(proof["exact_partition_certificate"]["interiors_disjoint"])
        self.assertNotIn("exact_native_partition_certificate", proof)
        self.assertEqual(proof["coordinate_provenance"], "counterfactual_owner_replay")


    def test_all_declared_capture_identities_resolve_in_cold_preflight(self):
        pack, receipt = self.root/"pack", self.root/"receipt"
        pack.write_bytes(b"native-pack")
        receipt.write_bytes(b"native-receipt")
        base = ["--pack", str(pack), "--pack-sha256", hashlib.sha256(pack.read_bytes()).hexdigest(),
                "--capture-receipt", str(receipt), "--receipt-sha256",
                hashlib.sha256(receipt.read_bytes()).hexdigest(), "--changed", "51005:73"]
        declarations = [
            ("target", ["--target", "51005:999"]),
            ("required-closed", ["--required-closed", "51005:999"]),
            ("enclosure", ["--intended-enclosure", "51005:73,51005:999"]),
            ("shared-partition", ["--shared-partition", "51005:73", "51005:999",
                                  str(self.root/"must-not-be-read.nhtissue"), "0"*64]),
        ]
        for name, declaration in declarations:
            with self.subTest(declaration=name):
                output = self.root/(name+"-absent.json")
                with patch.object(tool.verify, "load_capture", return_value=(self.surfaces(), {})):
                    with patch.object(tool.verify, "audit_changed_surfaces",
                                      side_effect=AssertionError("must stop before predicates")):
                        result = tool.main(base+declaration+[
                            "--preflight-only", "--output", str(output)])
                report = json.loads(output.read_text())
                self.assertEqual(result, 1)
                self.assertEqual(report["failed_stage"], "preparation")
                self.assertIn(name+" identities absent from capture", report["error"])

    def test_missing_target_stops_before_expensive_shared_source_certificate(self):
        from numilab_human import resting_shared_partition_verify as shared
        pack, receipt = self.root/"pack", self.root/"receipt"
        pack.write_bytes(b"native-pack")
        receipt.write_bytes(b"native-receipt")
        output = self.root/"absent-before-cert.json"
        args = ["--pack", str(pack), "--pack-sha256", hashlib.sha256(pack.read_bytes()).hexdigest(),
                "--capture-receipt", str(receipt), "--receipt-sha256",
                hashlib.sha256(receipt.read_bytes()).hexdigest(), "--changed", "51005:73",
                "--target", "51005:999", "--shared-partition", "51005:73", "51005:74",
                str(self.root/"must-not-be-read.nhtissue"), "0"*64, "--output", str(output)]
        with patch.object(tool.verify, "load_capture", return_value=(self.surfaces(), {})):
            with patch.object(shared, "audit_with_shared_partitions") as audit:
                self.assertEqual(tool.main(args), 1)
        audit.assert_not_called()
        self.assertIn("target identities absent from capture",
                      json.loads(output.read_text())["error"])


if __name__ == "__main__":
    unittest.main()
