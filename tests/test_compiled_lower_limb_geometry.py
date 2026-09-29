"""Source/payload identity regressions; these do not qualify loaded mechanics."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from numilab_human import model as human
from numilab_human.lower_limb_pose_audit import (
    _compiled_bone_members, _compiled_member_geometry_check, audit_lower_limb_poses,
)
from numilab_human.lower_limb_source_registration import _source_frame_check
from numilab_human.upper_limb_pose_audit import (
    PoseAuditError, audit_upper_limb_poses, _compiled_bone_geometry_checks,
)


class CompiledLowerLimbGeometryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registration_path = self.root / "registration.json"
        self.payload_path = self.root / "bones.nhbones"
        self.manifest_path = self.root / "bodyparts3d-myosim-major-bones.manifest.json"
        self.archive = self.root / "source.zip"
        self.archive.write_bytes(b"source archive fixture")
        self.obj = b"source mesh fixture"
        self.vertices = [[11.1234567, -22.2345678, 33.3456789],
                         [50.4567891, 20.5678912, -10.6789123],
                         [-7.7891234, 15.8912345, 80.9123456]]
        self.triangles = [[0, 1, 2]]
        self.reference = {"source_archive_sha256": "11" * 32,
                          "rigid": {"sha256": "22" * 32},
                          "manifest": {"sha256": "33" * 32}}
        self.bodies = {"tibia_l": (150, {"source_record_index": 7})}
        self.write_fixture()

    def write_fixture(self, angle=.12, scale=.9695, abi=3):
        c, s = math.cos(angle), math.sin(angle)
        matrix = [[.001 * scale * c, -.001 * scale * s, 0., .02],
                  [.001 * scale * s, .001 * scale * c, 0., .03],
                  [0., 0., .001 * scale, .04], [0., 0., 0., 1.]]
        self.anchor = {"source": {"member_id": "FJ-tibia-fixture",
                       "hierarchy": "isa", "name": "left tibia", "archive_sha256": human.sha256(self.archive),
                       "member_sha256": hashlib.sha256(self.obj).hexdigest(),
                       "vertex_count": 3, "triangle_count": 1},
                       "target": {"name": "tibia_l", "core_body_index": 150},
                       "registration": {"source_obj_mm_to_core_inertial_body_m": matrix}}
        self.registration = {"anchors": [self.anchor]}
        self.registration_path.write_text(json.dumps(self.registration))
        registration_sha = human.sha256(self.registration_path)
        translation, quaternion, local_scale = human._bodyparts_visual_local_pose(matrix, "fixture")
        record = struct.pack("<6I8f", 150, 0, 3, 0, 3, 1,
                             *translation, *quaternion, local_scale)
        if abi == 3:
            record += struct.pack("<I", 7)
        self.raw = bytearray(b"".join([
            struct.pack("<8s5I32s", b"NHBONES1", abi, 1, 3, 3,
                        int(registration_sha[:8], 16), bytes.fromhex("11" * 32)),
            record,
            *(struct.pack("<6f", *(value * .001 for value in vertex), 0., 0., 1.)
              for vertex in self.vertices),
            struct.pack("<3I", 0, 1, 2),
        ]))
        self.manifest = {
            "schema": "numi.human.bodyparts3d-myosim-major-bone-visual-payload.v1",
            "payload": {"file": self.payload_path.name, "registration_fingerprint32": registration_sha[:8]},
            "source": {
                "myosim_source_archive_sha256": "11" * 32,
                "registration": {"sha256": registration_sha},
                "runtime_reference": copy.deepcopy(self.reference),
                "anchors": [{"member_id": "FJ-tibia-fixture", "core_body_index": 150,
                             "source_record_index": 7, "myosim_body": "tibia_l",
                             "member_sha256": hashlib.sha256(self.obj).hexdigest()}],
            },
        }
        self.persist()

    def persist(self):
        self.payload_path.write_bytes(self.raw)
        # Corruptions retain valid hashes, so a descriptor-only check cannot catch them.
        self.manifest["payload"]["sha256"] = human.sha256(self.payload_path)
        self.manifest_path.write_text(json.dumps(self.manifest))

    def members(self):
        return _compiled_bone_members(self.root, self.registration_path,
                                      self.registration, self.reference, self.bodies)[0]

    def check(self):
        surface = self.members()["FJ-tibia-fixture"]
        return _compiled_member_geometry_check(self.anchor, self.vertices, self.triangles, surface, np)

    def complete_check(self, registration=None, members=None):
        specification = {"member_id": "FJ-tibia-fixture", "hierarchy": "isa",
                         "bodyparts_name": "left tibia", "myosim_body": "tibia_l"}
        with patch.object(human, "_BODYPARTS_MYOSIM_BONE_ANCHORS", (specification,)), \
             patch.object(human, "_bodyparts_obj_member", return_value=(self.archive, "bone.obj", self.obj)), \
             patch.object(human, "_bodyparts_obj_triangles", return_value=(self.vertices, self.triangles)):
            return _compiled_bone_geometry_checks(
                self.root, self.registration if registration is None else registration,
                self.members() if members is None else members, np,
            )

    def test_valid_source_variation_preserves_exact_fp32_geometry(self):
        common = {"global_source_mm_to_myosim_world_m": np.diag([.001, .001, .001, 1.]).tolist()}
        for angle in (0., .12, .249):
            for scale in (.93, .9695, 1., 1.07):
                with self.subTest(angle=angle, scale=scale):
                    self.write_fixture(angle, scale)
                    self.assertTrue(_source_frame_check(self.anchor, common, np.eye(3).tolist())["passed"])
                    result = self.check()
                    self.assertTrue(result["passed"])
                    self.assertEqual(result["maximum_vertex_residual_m"], 0.)
                    self.assertTrue(self.complete_check()["tibia_l"]["passed"])

    def test_missing_complete_skeleton_member_is_retained_as_a_failed_measurement(self):
        groups = self.complete_check({"anchors": []}, {})
        check = groups["tibia_l"]["compiled_bone_geometry_checks"][0]
        self.assertFalse(groups["tibia_l"]["passed"])
        self.assertEqual(check["source_member_id"], "FJ-tibia-fixture")
        self.assertEqual(check["source_member_name"], "left tibia")
        self.assertEqual(check["coverage_status"], "missing_registration")
        self.assertEqual(check["compiled_vertex_count"], 0)
        self.assertEqual(check["source_vertex_count"], 3)

    def test_unknown_member_cannot_replace_the_expected_source_structure(self):
        altered = copy.deepcopy(self.registration)
        altered["anchors"][0]["source"]["member_id"] = "unexpected"
        groups = self.complete_check(altered, {"unexpected": self.members()["FJ-tibia-fixture"]})
        checks = groups["tibia_l"]["compiled_bone_geometry_checks"]
        self.assertEqual({c["coverage_status"] for c in checks}, {"missing_registration", "unexpected_source_member"})
        self.assertTrue(all(not c["passed"] for c in checks))

    def test_duplicate_registration_cannot_hide_behind_one_compiled_member(self):
        altered = copy.deepcopy(self.registration)
        altered["anchors"].append(copy.deepcopy(altered["anchors"][0]))
        check = self.complete_check(altered)["tibia_l"]["compiled_bone_geometry_checks"][0]
        self.assertFalse(check["passed"])
        self.assertEqual(check["coverage_status"], "duplicate_registration")
        self.assertEqual(check["registration_occurrence_count"], 2)
        self.assertEqual(check["maximum_vertex_residual_m"], 0.)

    def test_registered_mesh_identity_is_checked_against_the_current_pinned_mesh(self):
        altered = copy.deepcopy(self.registration)
        altered["anchors"][0]["source"]["member_sha256"] = "55" * 32
        check = self.complete_check(altered)["tibia_l"]["compiled_bone_geometry_checks"][0]
        self.assertFalse(check["registered_source_identity_matches"])
        self.assertEqual(check["expected_source_identity"]["member_sha256"], hashlib.sha256(self.obj).hexdigest())
        self.assertEqual(check["registered_source_identity"]["member_sha256"], "55" * 32)
        self.assertFalse(check["passed"])
        self.assertEqual(check["maximum_vertex_residual_m"], 0.)

    def test_displaced_payload_fails_with_unchanged_registration_and_valid_hash(self):
        header = struct.calcsize("<8s5I32s")
        translation = struct.unpack_from("<f", self.raw, header + 24)[0]
        struct.pack_into("<f", self.raw, header + 24, translation + .01)
        self.persist()
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["maximum_vertex_residual_m"], .01, places=7)
        self.assertEqual(result["maximum_allowed_vertex_residual_m"], 0.)

    def test_flipped_payload_fails_even_with_unchanged_centroid(self):
        offset = struct.calcsize("<8s5I32s") + struct.calcsize("<6I8fI")
        vertices = np.array([struct.unpack_from("<3f", self.raw, offset + 24 * i) for i in range(3)])
        flipped = (vertices - vertices.mean(axis=0)) @ np.diag([-1., 1., -1.]) + vertices.mean(axis=0)
        for i, vertex in enumerate(flipped):
            struct.pack_into("<3f", self.raw, offset + 24 * i, *vertex)
        self.persist()
        self.assertFalse(self.check()["passed"])

    def test_changed_topology_fails_even_when_vertices_match(self):
        struct.pack_into("<3I", self.raw, len(self.raw) - 12, 0, 2, 1)
        self.persist()
        result = self.check()
        self.assertEqual(result["maximum_vertex_residual_m"], 0.)
        self.assertFalse(result["source_topology_matches"])
        self.assertFalse(result["passed"])

    def test_nonfinite_vertex_is_rejected(self):
        offset = struct.calcsize("<8s5I32s") + struct.calcsize("<6I8fI")
        struct.pack_into("<f", self.raw, offset, float("nan"))
        self.persist()
        with self.assertRaisesRegex(human.ImportError, "FJ-tibia-fixture.*vertex is non-finite"):
            self.check()

    def test_nonfinite_instance_pose_is_rejected_before_surface_checks(self):
        struct.pack_into("<f", self.raw, struct.calcsize("<8s5I32s") + 24, float("inf"))
        self.persist()
        with self.assertRaisesRegex(human.ImportError, "FJ-tibia-fixture.*pose is non-finite"):
            self.members()

    def test_matching_binary_and_manifest_cannot_override_registered_core_owner(self):
        struct.pack_into("<I", self.raw, struct.calcsize("<8s5I32s"), 145)
        self.manifest["source"]["anchors"][0]["core_body_index"] = 145
        self.persist()
        with self.assertRaisesRegex(RuntimeError, "FJ-tibia-fixture.*Core=145.*requires Core=150"):
            self.members()

    def test_matching_binary_and_manifest_cannot_override_rigid_source_owner(self):
        offset = struct.calcsize("<8s5I32s") + struct.calcsize("<6I8f")
        struct.pack_into("<I", self.raw, offset, 6)
        self.manifest["source"]["anchors"][0]["source_record_index"] = 6
        self.persist()
        with self.assertRaisesRegex(RuntimeError, "source_record=6.*requires Core=150, source_record=7"):
            self.members()

    def test_registration_hash_suffix_is_checked_beyond_compact_fingerprint(self):
        sha = self.manifest["source"]["registration"]["sha256"]
        self.manifest["source"]["registration"]["sha256"] = sha[:8] + "0" * 56
        self.persist()
        with self.assertRaisesRegex(RuntimeError, "registration identity drifted"):
            self.members()

    def test_same_source_archive_does_not_allow_another_rigid_payload(self):
        self.manifest["source"]["runtime_reference"]["rigid"]["sha256"] = "55" * 32
        self.persist()
        with self.assertRaisesRegex(RuntimeError, "rigid identity drifted.*bone_sha256=.*rigid_sha256="):
            self.members()

    def test_legacy_abi_cannot_claim_source_owner_verification(self):
        self.write_fixture(abi=2)
        self.manifest["payload"]["payload_abi"] = 3
        self.persist()
        with self.assertRaisesRegex(RuntimeError, "requires NHBONES1 ABI 3 source owners"):
            self.members()


class SourceCompiledWholeBodyGeometryTests(unittest.TestCase):
    def test_all_non_limb_members_and_self_consistent_omission_are_checked(self):
        from numilab_human.lower_limb_pose_audit import LOWER_BODY_NAMES
        from numilab_human.upper_limb_registration import _upper_names
        paths = {key: os.environ.get("NUMILAB_HUMAN_MOTION_" + key)
                 for key in ("SOURCES", "ARTIFACT", "REPAIRED", "BONES")}
        if not all(paths.values()):
            self.skipTest("exact source/payload motion inputs were not supplied")
        registration_path = Path(paths["REPAIRED"])
        registration = json.loads(registration_path.read_text())
        reference, bodies = human._bodyparts_runtime_bindings(registration, Path(paths["ARTIFACT"]))

        def checks(bones, source_registration=registration, source_path=registration_path):
            members, _ = _compiled_bone_members(bones, source_path, source_registration, reference, bodies)
            groups = _compiled_bone_geometry_checks(Path(paths["SOURCES"]), source_registration, members, np)
            return [c for body in groups.values() for c in body["compiled_bone_geometry_checks"]]

        admitted = checks(Path(paths["BONES"]))
        self.assertEqual(len(admitted), 185)
        self.assertTrue(all(c["passed"] for c in admitted))
        limb_names = _upper_names("r") | _upper_names("l") | LOWER_BODY_NAMES
        formerly_unchecked = {a["source"]["member_id"] for a in registration["anchors"]
                              if a["target"]["name"] not in limb_names}
        self.assertEqual(len(formerly_unchecked), 61)
        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "bones"
            shutil.copytree(paths["BONES"], altered)
            manifest_path = altered / "bodyparts3d-myosim-major-bones.manifest.json"
            original_manifest = json.loads(manifest_path.read_text())
            payload_path = altered / original_manifest["payload"]["file"]
            original_raw = payload_path.read_bytes()
            header, stride = struct.calcsize("<8s5I32s"), struct.calcsize("<6I8fI")
            raw = bytearray(original_raw)
            for index, anchor in enumerate(original_manifest["source"]["anchors"]):
                if anchor["member_id"] in formerly_unchecked:
                    offset = header + stride * index + 24
                    struct.pack_into("<f", raw, offset, struct.unpack_from("<f", raw, offset)[0] + .04)
            payload_path.write_bytes(raw)
            manifest = copy.deepcopy(original_manifest)
            manifest["payload"]["sha256"] = human.sha256(payload_path)
            manifest_path.write_text(json.dumps(manifest))
            failed = [c for c in checks(altered) if not c["passed"]]
            self.assertEqual({c["source_member_id"] for c in failed}, formerly_unchecked)
            for check in failed:
                self.assertAlmostEqual(check["maximum_vertex_residual_m"], .04, places=7)
                self.assertEqual(check["maximum_allowed_vertex_residual_m"], 0.)

            # Remove the last cervical member from both registration and
            # payload, retaining valid owner records and updated identities.
            manifest = copy.deepcopy(original_manifest)
            omitted = manifest["source"]["anchors"].pop()["member_id"]
            partial = copy.deepcopy(registration)
            partial["anchors"] = [a for a in partial["anchors"] if a["source"]["member_id"] != omitted]
            partial_path = Path(temporary) / "partial.registration.json"
            partial_path.write_text(json.dumps(partial))
            fingerprint = human.sha256(partial_path)
            raw = bytearray(original_raw)
            del raw[header + 184 * stride:header + 185 * stride]
            struct.pack_into("<I", raw, 12, 184)
            struct.pack_into("<I", raw, 24, int(fingerprint[:8], 16))
            payload_path.write_bytes(raw)
            manifest["source"]["registration"]["sha256"] = fingerprint
            manifest["payload"].update(sha256=human.sha256(payload_path), bytes=len(raw),
                                       bone_count=184, registration_fingerprint32=fingerprint[:8])
            manifest_path.write_text(json.dumps(manifest))
            failed = [c for c in checks(altered, partial, partial_path) if not c["passed"]]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["source_member_id"], omitted)
            self.assertEqual(failed[0]["coverage_status"], "missing_registration")


class SourceCompiledLowerLimbGeometryTests(unittest.TestCase):
    def test_exact_source_pose_audit_uses_payload_and_rejects_postcompile_displacement(self):
        paths = {key: os.environ.get("NUMILAB_HUMAN_MOTION_" + key)
                 for key in ("SOURCES", "ARTIFACT", "REPAIRED", "BONES")}
        if not all(paths.values()):
            self.skipTest("exact source/payload motion inputs were not supplied")

        def measured(bones=None):
            try:
                return audit_lower_limb_poses(sources=Path(paths["SOURCES"]),
                    registration_path=Path(paths["REPAIRED"]), artifact=Path(paths["ARTIFACT"]),
                    bone_artifact=bones)
            except PoseAuditError as error:
                return error.result

        accepted = measured(Path(paths["BONES"]))
        checks = [c for body in accepted["source_geometry_checks"]
                  for c in body["compiled_bone_geometry_checks"]]
        self.assertEqual(len(checks), 185)
        self.assertTrue(all(c["passed"] for c in checks))
        self.assertEqual(accepted["pose_count"], 8)
        self.assertEqual(accepted["continuity_evaluation_count"], 320)
        self.assertEqual(accepted["default_frame_maximum_allowed_residual_m"], 1e-9)
        self.assertEqual(accepted["bilateral_gap_parity_maximum_m"], .004)
        # Preserve the current source-geometry result, including a failed
        # parity gate when present. A repaired registration must not inherit
        # a hard-coded expectation that the old gap can never be fixed.
        reference = measured()
        for native_pose, source_pose in zip(accepted["poses"], reference["poses"], strict=True):
            self.assertEqual(native_pose["name"], source_pose["name"])
            for field in ("continuity", "bilateral_gap_parity"):
                self.assertEqual([c["passed"] for c in native_pose[field]],
                                 [c["passed"] for c in source_pose[field]])
        self.assertFalse(any("compiled_vertex_residual" in f for f in accepted["failures"]))

        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "bones"
            shutil.copytree(paths["BONES"], altered)
            manifest_path = altered / "bodyparts3d-myosim-major-bones.manifest.json"
            manifest = json.loads(manifest_path.read_text())
            payload_path = altered / manifest["payload"]["file"]
            raw = bytearray(payload_path.read_bytes())
            index = next(i for i, a in enumerate(manifest["source"]["anchors"])
                         if a["member_id"] == "FJ3275")
            offset = struct.calcsize("<8s5I32s") + index * struct.calcsize("<6I8fI") + 24
            x = struct.unpack_from("<f", raw, offset)[0]
            struct.pack_into("<f", raw, offset, x + .02)
            payload_path.write_bytes(raw)
            manifest["payload"]["sha256"] = human.sha256(payload_path)
            manifest_path.write_text(json.dumps(manifest))
            corrupted = measured(altered)
            body = next(b for b in corrupted["source_geometry_checks"] if b["myosim_body"] == "patella_l")
            self.assertFalse(body["passed"])
            check = next(c for c in body["compiled_bone_geometry_checks"] if c["source_member_id"] == "FJ3275")
            self.assertAlmostEqual(check["maximum_vertex_residual_m"], .02, places=7)
            self.assertTrue(any("FJ3275:compiled_vertex_residual_m" in f for f in corrupted["failures"]))
            self.assertEqual(corrupted["inputs"]["registration"], accepted["inputs"]["registration"])
            self.assertNotEqual(corrupted["inputs"]["bone_payload"]["sha256"], accepted["inputs"]["bone_payload"]["sha256"])
            self.assertEqual(corrupted["default_frame_maximum_centroid_residual_m"],
                             accepted["default_frame_maximum_centroid_residual_m"])
            first = next(p for p in accepted["poses"] if p["name"] == "bilateral_deep_crouch")
            second = next(p for p in corrupted["poses"] if p["name"] == "bilateral_deep_crouch")
            gap = lambda p: next(c["minimum_vertex_gap_m"] for c in p["continuity"]
                                 if c["name"] == "left_femur_to_patella")
            self.assertNotEqual(gap(first), gap(second))


class SourceCompiledUpperLimbGeometryTests(unittest.TestCase):
    def test_actual_upper_limb_motion_uses_payload_and_rejects_displacement(self):
        paths = {key: os.environ.get("NUMILAB_HUMAN_MOTION_" + key)
                 for key in ("SOURCES", "ARTIFACT", "REPAIRED", "BONES")}
        if not all(paths.values()):
            self.skipTest("exact source/payload motion inputs were not supplied")

        def measured(bones=None):
            try:
                return audit_upper_limb_poses(sources=Path(paths["SOURCES"]),
                    registration_path=Path(paths["REPAIRED"]), artifact=Path(paths["ARTIFACT"]),
                    bone_artifact=bones)
            except PoseAuditError as error:
                return error.result

        registered = measured()
        accepted = measured(Path(paths["BONES"]))
        checks = [c for body in accepted["source_geometry_checks"]
                  for c in body["compiled_bone_geometry_checks"]]
        self.assertEqual(len(checks), 185)
        self.assertTrue(all(c["passed"] for c in checks))
        self.assertTrue(all(c["maximum_vertex_residual_m"] == 0. for c in checks))
        self.assertEqual(accepted["pose_count"], 7)
        self.assertEqual(accepted["continuity_evaluation_count"], 364)
        self.assertEqual(accepted["default_frame_maximum_allowed_residual_m"], 1e-9)
        self.assertEqual(accepted["bilateral_gap_parity_maximum_m"], .002)
        self.assertTrue(accepted["status"].startswith("passed_"), accepted.get("failures"))
        self.assertEqual(accepted["default_frame_maximum_centroid_residual_m"],
                         registered["default_frame_maximum_centroid_residual_m"])
        self.assertEqual(accepted["inputs"]["bone_payload"]["payload_abi"], 3)

        with tempfile.TemporaryDirectory() as temporary:
            altered = Path(temporary) / "bones"
            shutil.copytree(paths["BONES"], altered)
            manifest_path = altered / "bodyparts3d-myosim-major-bones.manifest.json"
            manifest = json.loads(manifest_path.read_text())
            payload_path = altered / manifest["payload"]["file"]
            raw = bytearray(payload_path.read_bytes())
            index = next(i for i, a in enumerate(manifest["source"]["anchors"])
                         if a["myosim_body"] == "humerus_r")
            member_id = manifest["source"]["anchors"][index]["member_id"]
            offset = struct.calcsize("<8s5I32s") + index * struct.calcsize("<6I8fI") + 24
            x = struct.unpack_from("<f", raw, offset)[0]
            struct.pack_into("<f", raw, offset, x + .02)
            payload_path.write_bytes(raw)
            manifest["payload"]["sha256"] = human.sha256(payload_path)
            manifest_path.write_text(json.dumps(manifest))
            corrupted = measured(altered)
            body = next(b for b in corrupted["source_geometry_checks"]
                        if b["myosim_body"] == "humerus_r")
            self.assertFalse(body["passed"])
            check = body["compiled_bone_geometry_checks"][0]
            self.assertAlmostEqual(check["maximum_vertex_residual_m"], .02, places=7)
            self.assertTrue(any(member_id + ":compiled_vertex_residual_m" in f
                                for f in corrupted["failures"]))
            self.assertEqual(corrupted["inputs"]["registration"], accepted["inputs"]["registration"])
            self.assertNotEqual(corrupted["inputs"]["bone_payload"]["sha256"],
                                accepted["inputs"]["bone_payload"]["sha256"])
            self.assertEqual(corrupted["default_frame_maximum_centroid_residual_m"],
                             accepted["default_frame_maximum_centroid_residual_m"])
            pose = lambda r: next(p for p in r["poses"] if p["name"] == "bilateral_coupled_reach")
            gap = lambda r: next(c["minimum_vertex_gap_m"] for c in pose(r)["continuity"]
                                 if c["name"] == "right_scapula_to_humerus")
            self.assertNotEqual(gap(corrupted), gap(accepted))


if __name__ == "__main__":
    unittest.main()
