"""Source/Core admission regressions; optional native runs are not loaded mechanics."""
import copy
import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from numilab_human.model import ImportError, _bodyparts_runtime_bindings
from numilab_human.upper_limb_pose_audit import PoseAuditError, _finish_pose_audit


class AnatomyRuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.artifact = Path(self.temporary.name)
        self.source = {"archive_sha256": "a" * 64, "revision": "fixture"}
        self.records = [
            {"source_body_id": sid, "core_body_index": index, "name": name,
             "default_com_position_world_m": [0., 0., 1. + index / 10],
             "default_inertial_quaternion_world_xyzw": [0., 0., 0., 1.]}
            for index, (sid, name) in enumerate([(1, "pelvis"), (2, "humerus_r"), (5, "tibia_r")])
        ]
        header = struct.pack("<8s10I32s", b"NHRIGID2", 1, 5, 3, 3, 2, 7, 6, 0, 0, 0, bytes.fromhex("a" * 64))
        bodies = bytearray(160 * 3)
        for index in range(3):
            struct.pack_into("<I", bodies, 160 * index + 8, 0xFFFFFFFF)
        self.raw = header + bytes(144) + bodies + bytes(144 * 2 + 64 * 6 + 4 * 13)
        self.raw += struct.pack("<3I", 0, 1, 2)
        self.raw += b"".join(struct.pack("<7f", *r["default_com_position_world_m"], *r["default_inertial_quaternion_world_xyzw"]) for r in self.records)
        self.manifest = {
            "schema": "numi.human.myosim-fullbody-reference.v1", "source": self.source,
            "core_tree": {"source_body_records": self.records, "body_order": [r["name"] for r in self.records],
                          "engine_body_count": 3, "joint_count": 2, "zero_inertia_serial_transform_carrier_count": 0},
            "payloads": {"rigid": {"file": "rigid.nhrigid", "sha256": hashlib.sha256(self.raw).hexdigest(), "bytes": len(self.raw)}},
        }
        self.registration = {
            "source": {"myosim": {"source": copy.deepcopy(self.source), "payloads": {"rigid": {"sha256": "b" * 64}}}},
            "anchors": [{
                "source": {"member_id": "FJ-humerus-fixture"},
                "target": copy.deepcopy(self.records[1]),
                "registration": {"source_obj_mm_to_core_inertial_body_m": [
                    [0.001, 0., 0., 0.02], [0., 0.001, 0., 0.03], [0., 0., 0.001, 0.04], [0., 0., 0., 1.]]},
            }],
        }
        self.write_artifact()

    def write_artifact(self):
        (self.artifact / "rigid.nhrigid").write_bytes(self.raw)
        (self.artifact / "myosim-fullbody-reference.manifest.json").write_text(json.dumps(self.manifest))

    def bind(self):
        return _bodyparts_runtime_bindings(self.registration, self.artifact)[0]

    def test_actual_rigid_identity_is_recorded_without_rewriting_historical_lineage(self):
        result = self.bind()
        self.assertEqual(result["bound_anchor_count"], 1)
        self.assertEqual(result["rigid"]["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertEqual(self.registration["source"]["myosim"]["payloads"]["rigid"]["sha256"], "b" * 64)

    def test_wrong_core_owner_rejected_even_when_source_id_and_geometry_are_unchanged(self):
        self.registration["anchors"][0]["target"]["core_body_index"] = 2
        with self.assertRaisesRegex(ImportError, r"FJ-humerus-fixture \(humerus_r\).*IDs 2/2.*requires 2/1"):
            self.bind()

    def test_wrong_source_owner_rejected(self):
        self.registration["anchors"][0]["target"]["source_body_id"] = 5
        with self.assertRaisesRegex(ImportError, "owner mismatch"):
            self.bind()

    def test_manifest_mapping_cannot_override_consumed_binary(self):
        self.records[1]["core_body_index"] = 2
        self.write_artifact()
        with self.assertRaisesRegex(ImportError, "NHRIGID2 source map index 1"):
            self.bind()

    def test_manifest_pose_cannot_override_consumed_binary(self):
        self.records[1]["default_com_position_world_m"][0] = 0.01
        self.write_artifact()
        with self.assertRaisesRegex(ImportError, "rest pose disagrees with NHRIGID2"):
            self.bind()

    def test_registration_frame_drift_is_actionable(self):
        self.registration["anchors"][0]["target"]["default_com_position_world_m"][0] = 0.01
        with self.assertRaisesRegex(ImportError, "COM error=0.01 m.*tolerance=1e-9.*rigid_sha256"):
            self.bind()

    def test_source_body_record_order_is_not_interchangeable(self):
        self.records[1]["source_body_id"] = 6
        self.write_artifact()
        with self.assertRaises(ImportError):
            self.bind()

    def test_rigid_bytes_drift_rejected(self):
        self.raw = self.raw[:-1] + b"\x01"
        (self.artifact / "rigid.nhrigid").write_bytes(self.raw)
        with self.assertRaisesRegex(ImportError, "missing or drifted"):
            self.bind()

    def test_source_model_drift_rejected(self):
        self.registration["source"]["myosim"]["source"]["archive_sha256"] = "c" * 64
        with self.assertRaisesRegex(ImportError, "different MyoSim sources"):
            self.bind()

    def test_duplicate_member_rejected_without_forbidding_shared_rigid_owner(self):
        duplicate = copy.deepcopy(self.registration["anchors"][0])
        duplicate["source"]["member_id"] = "FJ-second-same-owner"
        self.registration["anchors"].append(duplicate)
        self.assertEqual(self.bind()["bound_anchor_count"], 2)
        duplicate["source"]["member_id"] = "FJ-humerus-fixture"
        with self.assertRaisesRegex(ImportError, "duplicate"):
            self.bind()

    def test_positive_uniform_scale_and_equivalent_quaternion_remain_valid(self):
        anchor = self.registration["anchors"][0]
        anchor["target"]["default_inertial_quaternion_world_xyzw"] = [0., 0., 0., -1.]
        anchor["registration"]["source_obj_mm_to_core_inertial_body_m"] = [
            [0., -0.00112, 0., 0.03], [0.00112, 0., 0., -0.01],
            [0., 0., 0.00112, 0.02], [0., 0., 0., 1.]]
        self.assertEqual(self.bind()["bound_anchor_count"], 1)

    def test_reflection_and_shear_rejected(self):
        original = copy.deepcopy(self.registration)
        for change in ("reflection", "shear", "non_affine"):
            self.registration = copy.deepcopy(original)
            m = self.registration["anchors"][0]["registration"]["source_obj_mm_to_core_inertial_body_m"]
            if change == "reflection": m[0][0] *= -1
            elif change == "shear": m[0][1] = 0.0005
            else: m[3][0] = 0.001
            with self.subTest(change=change), self.assertRaises(ImportError):
                self.bind()

    def test_boolean_owner_ids_are_not_integer_identities(self):
        for key in ("source_body_id", "core_body_index"):
            with self.subTest(key=key):
                saved = self.registration["anchors"][0]["target"][key]
                self.registration["anchors"][0]["target"][key] = True
                with self.assertRaisesRegex(ImportError, "owner mismatch"):
                    self.bind()
                self.registration["anchors"][0]["target"][key] = saved

    def test_source_record_index_is_resolved_from_binary_map_order(self):
        _, bodies = _bodyparts_runtime_bindings(self.registration, self.artifact)
        self.assertEqual(bodies["tibia_r"][1]["source_body_id"], 5)
        self.assertEqual(bodies["tibia_r"][1]["source_record_index"], 2)


class FailedPoseDiagnosticTests(unittest.TestCase):
    def test_failed_measurements_and_reproduction_survive_both_cli_surfaces(self):
        from numilab_human import upper_limb_pose_audit as upper, lower_limb_pose_audit as lower
        result = {
            "status": "passed_source_owned_fixture", "poses": [{
                "name": "deep_crouch", "bilateral_gap_parity": [], "continuity": [{
                    "name": "left_femur_to_patella", "source_member_ids": ["FJ3259", "FJ3275"],
                    "passed": False, "minimum_vertex_gap_m": 0.00627,
                    "posed_maximum_allowed_gap_m": 0.00547,
                    "interface_patch": {"bidirectional_p90_m": 0.01077},
                    "posed_maximum_allowed_interface_patch_p90_m": 0.01129,
                    "minimum_gap_witness_world_m": [[0., 0., 0.], [0., 0., 0.00627]],
                }],
            }], "default_frame_maximum_centroid_residual_m": 0.,
            "default_frame_maximum_allowed_residual_m": 1e-9,
            "inputs": {"registration": {"sha256": "a" * 64},
                       "runtime_reference": {"rigid": {"sha256": "b" * 64}}},
        }
        with self.assertRaisesRegex(PoseAuditError, "FJ3259.*allowed_gap_m=0.00547") as caught:
            _finish_pose_audit(result, "lower-limb")
        self.assertIs(caught.exception.result, result)
        self.assertTrue(result["status"].startswith("failed_"))
        result["poses"][0]["projected_joint_range_checks"] = [{
            "passed": False, "source_joint_name": "knee_angle_rotation2_r", "q_index": 107,
            "projected_value": .10, "unit": "rad", "source_range": [-.00167821, .0335354],
            "source_range_violation": .0664646, "native_position_limit_enabled": True,
            "native_position_range": [-.00167821, .0335354], "native_range_violation": .0664646,
            "maximum_allowed_range_violation": 1e-9, "tolerance_basis": "native_pose_admission",
        }]
        with self.assertRaisesRegex(PoseAuditError, "knee_angle_rotation2_r q_index=107"):
            _finish_pose_audit(result, "lower-limb")
        for module, function in [(upper, "audit_upper_limb_poses"), (lower, "audit_lower_limb_poses")]:
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "failed.json"
                failure = PoseAuditError("measured gap exceeds source gate", copy.deepcopy(result))
                with patch.object(module, function, side_effect=failure) as audit:
                    code = module.main(["--sources", temporary, "--artifact", temporary,
                                        "--registration", temporary + "/source.json", "--output", str(output),
                                        "--bone-artifact", temporary])
                self.assertEqual(code, 2)
                self.assertEqual(audit.call_args.kwargs["bone_artifact"], Path(temporary).resolve())
                retained = json.loads(output.read_text())
                self.assertEqual(retained["poses"], result["poses"])
                self.assertIn(module.__name__, retained["inputs"]["reproduction_command"])
                self.assertIn("--bone-artifact", retained["inputs"]["reproduction_command"])


class NativeBoneOwnerBindingTests(unittest.TestCase):
    """Run on an explicitly selected native binary and paired source artifact."""

    def setUp(self):
        variables = ["NUMILAB_HUMAN_NATIVE_VISUAL_PROBE", "NUMILAB_HUMAN_NATIVE_BONE_PAYLOAD",
                     "NUMILAB_HUMAN_NATIVE_REFERENCE_ARTIFACT"]
        if any(not os.environ.get(name) for name in variables):
            self.skipTest("native ownership regression requires an explicit probe, bone payload and reference artifact")
        self.probe, self.bones, self.artifact = [Path(os.environ[name]).resolve() for name in variables]
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def run_probe(self, raw, arguments=(), dimension=512):
        bones = self.root / "candidate.nhbones"
        bones.write_bytes(raw)
        return subprocess.run([
            str(self.probe), str(self.artifact / "myosim-fullbody-core-reference.nhrigid"),
            str(self.artifact / "myosim-fullbody-muscle-reference.nhmyo"), str(bones),
            str(self.root / "views"), "--dimension", str(dimension), *arguments,
        ], capture_output=True, text=True, timeout=60)

    def test_valid_source_bound_payload_executes_and_reports_verified_owners(self):
        result = self.run_probe(self.bones.read_bytes())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bone_source_owner_bindings_verified=true", result.stdout)
        self.assertIn("bone_payload_abi=3", result.stdout)

    def test_filtered_left_knee_remains_visible_in_source_crouch(self):
        arguments = ["--joint-equality-payload", str(
            self.artifact / "myosim-fullbody-joint-equalities.nheq"
        )]
        for dof, value in [(101, .9), (106, 1.4), (109, -.25), (111, .2),
                           (115, .9), (120, 1.4), (123, -.25), (125, .2)]:
            arguments += ["--pose-q", str(dof), str(value)]
        for body in [145, 150, 156]:
            arguments += ["--visible-bone-body-index", str(body)]
        for focus in [["--focus-body-index", "156"],
                      ["--focus-joint-child-body-index", "156",
                       "--focus-distance-m", "0.32"]]:
            with self.subTest(focus=focus[0]):
                result = self.run_probe(self.bones.read_bytes(), arguments + focus, 1024)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("bone_source_owner_bindings_verified=true", result.stdout)
                coverage = re.findall(r"view=(\w+).*?bone_pixels=(\d+)", result.stdout)
                self.assertEqual({view for view, _ in coverage},
                                 {"front", "oblique", "side", "rear"})
                self.assertTrue(all(int(pixels) > 0 for _, pixels in coverage), coverage)

    def test_historical_records_remain_readable_with_owner_verification_unavailable(self):
        raw = self.bones.read_bytes()
        header_format = struct.Struct("<8s5I32s")
        header = list(header_format.unpack_from(raw))
        self.assertEqual(header[1], 3)
        header[1] = 2
        records = b"".join(raw[header_format.size + 60 * i:header_format.size + 60 * i + 56]
                           for i in range(header[2]))
        legacy = header_format.pack(*header) + records + raw[header_format.size + 60 * header[2]:]
        result = self.run_probe(legacy)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bone_payload_abi=2", result.stdout)
        self.assertIn("bone_source_owner_bindings_verified=false", result.stdout)

    def test_in_bounds_wrong_core_owner_is_rejected_before_rendering(self):
        raw = bytearray(self.bones.read_bytes())
        header = struct.unpack_from("<8s5I32s", raw)
        self.assertEqual(header[1], 3)
        offset = struct.calcsize("<8s5I32s")
        core = struct.unpack_from("<I", raw, offset)[0]
        rigid = (self.artifact / "myosim-fullbody-core-reference.nhrigid").read_bytes()
        body_count = struct.unpack_from("<8s10I32s", rigid)[4]
        self.assertGreater(body_count, 1)
        struct.pack_into("<I", raw, offset, (core + 1) % body_count)
        result = self.run_probe(raw)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("owner mismatch", result.stderr)
        self.assertIn("NHRIGID2 requires Core=", result.stderr)
        self.assertFalse((self.root / "views").exists())

    def test_invalid_source_ordinal_is_rejected_before_rendering(self):
        raw = bytearray(self.bones.read_bytes())
        struct.pack_into("<I", raw, struct.calcsize("<8s5I32s") + 56, 0xFFFFFFFF)
        result = self.run_probe(raw)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the NHRIGID2 source map", result.stderr)
        self.assertFalse((self.root / "views").exists())


if __name__ == "__main__":
    unittest.main()
