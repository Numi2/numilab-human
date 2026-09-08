import copy
import hashlib
import struct
import unittest

from numilab_human.joint_limits import compile_joint_limits
from numilab_human.model import ImportError


class JointLimitPayloadTests(unittest.TestCase):
    def setUp(self):
        source = {"archive_sha256": "ab" * 32}
        self.export = {"source": source, "model": {"joint_limit_solver": {
            "schema": "numi.human.mujoco-scalar-limit-solver.v1", "mujoco_version": "3.12.0",
            "integrator": "Euler", "refsafe": True, "diagexact": False, "enabled": True}},
            "joints": [{"id": 0, "limited": False}, {"id": 1, "name": "knee", "limited": True,
                "type": 3, "range": [-.25, .5], "limit_solref": [.02, 1.],
                "limit_solimp": [.7, .9, .1, .4, 2.], "limit_margin": .03125,
                "limit_dof_invweight0": 3.25}]}
        self.manifest = {"source": source, "core_tree": {"nq": 8, "nv": 7, "source_joint_map": [
            {"source_joint_id": 1, "core_q_index": 7, "core_v_index": 6, "source_name": "knee",
             "source_type": 3, "source_range": [-.25, .5], "source_limited": True,
             "core_limit_status": "enforced"}]}}

    def compile(self):
        return compile_joint_limits(self.export, self.manifest)

    def test_binary_contract_and_deterministic_identity(self):
        metadata, payload = self.compile()
        self.assertEqual(len(payload), 160)
        header = struct.unpack_from("<8s10I32s", payload)
        self.assertEqual(header[:11], (b"NHLIM1\0\0", 1, 8, 7, 1, 80, 1, 1, 1, 0, 0))
        self.assertEqual(header[11], bytes.fromhex("ab" * 32))
        self.assertEqual(struct.unpack_from("<4I4f", payload, 80), (7, 6, 1, 0, -.25, .5, .03125, 3.25))
        self.assertEqual(metadata["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(metadata["maximum_row_count"], 2)
        self.export["joints"].reverse()
        self.assertEqual(self.compile()[1], payload)

    def test_source_reset_outside_native_flag_is_retained(self):
        self.manifest["core_tree"]["source_joint_map"][0]["core_limit_status"] = "retained_in_manifest_not_enforced_at_source_default"
        metadata, payload = self.compile()
        self.assertEqual(len(payload), 160)
        self.assertEqual(metadata["joint_count"], 1)
        self.assertEqual(metadata["joints"][0]["legacy_core_limit_status"], "retained_in_manifest_not_enforced_at_source_default")

    def test_explicit_refsafe_off_and_direct_format(self):
        self.export["model"]["joint_limit_solver"]["refsafe"] = False
        self.export["joints"][1]["limit_solref"] = [0., -5.]
        _, payload = self.compile()
        self.assertEqual(struct.unpack_from("<I", payload, 36)[0], 0)
        self.assertEqual(struct.unpack_from("<2f", payload, 112), (0., -5.))

    def test_source_policy_fails_closed(self):
        initial = copy.deepcopy(self.export)
        for key, value in [("mujoco_version", "3.9.0"), ("integrator", "RK4"), ("diagexact", True),
                           ("enabled", False), ("refsafe", 1), ("diagexact", 0)]:
            with self.subTest(key=key, value=value):
                self.export = copy.deepcopy(initial)
                self.export["model"]["joint_limit_solver"][key] = value
                with self.assertRaises(ImportError): self.compile()

    def test_nonfinite_out_of_range_and_mixed_solref_rejected(self):
        initial = copy.deepcopy(self.export)
        changes = [("limit_margin", x) for x in [float("nan"), float("inf"), 1e40, 1e-45, True]]
        changes += [("limit_dof_invweight0", x) for x in [0, -1, 1e-40]]
        changes += [("limit_solref", [.01, -1]), ("limit_solimp", [1, 2]), ("range", [.5, -.25])]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                self.export = copy.deepcopy(initial)
                self.export["joints"][1][key] = value
                with self.assertRaises(ImportError): self.compile()

    def test_missing_limit_inventory_rejected(self):
        self.export["joints"] = self.export["joints"][:1]
        with self.assertRaises(ImportError): self.compile()

    def test_native_range_name_type_or_coverage_drift_rejected(self):
        initial = copy.deepcopy(self.manifest)
        for key, value in [("source_range", [-1, 1]), ("source_name", "other"), ("source_type", 2),
                           ("source_limited", False), ("source_limited", 1), ("core_q_index", 6),
                           ("core_limit_status", "source_unlimited")]:
            self.manifest = copy.deepcopy(initial)
            self.manifest["core_tree"]["source_joint_map"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ImportError): self.compile()

    def test_duplicate_bindings_rejected(self):
        self.manifest["core_tree"]["source_joint_map"] *= 2
        with self.assertRaises(ImportError): self.compile()

    def test_duplicate_or_noninteger_source_identity_rejected(self):
        initial = copy.deepcopy(self.export)
        for identity in [0, True, "1", -1, 2**32-1]:
            self.export = copy.deepcopy(initial)
            self.export["joints"][1]["id"] = identity
            with self.subTest(identity=identity), self.assertRaises(ImportError): self.compile()

    def test_source_digest_must_match_and_be_canonical(self):
        self.export["source"] = {"archive_sha256": "cd" * 32}
        with self.assertRaises(ImportError): self.compile()
        for digest in ["ab", "x" * 64, "AB" * 32]:
            self.export["source"] = self.manifest["source"] = {"archive_sha256": digest}
            with self.subTest(digest=digest), self.assertRaises(ImportError): self.compile()


if __name__ == "__main__":
    unittest.main()
