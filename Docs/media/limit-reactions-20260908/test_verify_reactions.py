#!/usr/bin/env python3
import copy
import json
import unittest

from verify_reactions import HERE, MANIFEST, audit, parse


class ReactionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads(MANIFEST.read_text())
        cls.log = (HERE / "qualified-240-stdout.log").read_text()
        cls.q, cls.reactions, _ = parse(cls.log)

    def test_recorded_case(self):
        self.assertEqual(audit(self.q, self.reactions, self.manifest)["status"], "offline_reactions_passed")

    def test_inward_loaded_stop(self):
        damaged = copy.deepcopy(self.reactions)
        v = next(i for i, f in enumerate(damaged["limit_force"]) if f != 0.0)
        damaged["acceleration"][v] = -1.0 if damaged["limit_force"][v] > 0.0 else 1.0
        self.assertIn("inward stop acceleration", " ".join(audit(self.q, damaged, self.manifest)["violations"]))

    def test_fictitious_root_reaction(self):
        damaged = copy.deepcopy(self.reactions)
        damaged["limit_force"][0] = 1.0
        self.assertIn("reaction on an unlimited coordinate", audit(self.q, damaged, self.manifest)["violations"])

    def test_source_range_violation(self):
        damaged = self.q.copy()
        joint = self.manifest["core_tree"]["source_joint_map"][0]
        damaged[joint["core_q_index"]] = joint["source_range"][0] - 0.01
        self.assertIn("source range", " ".join(audit(damaged, self.reactions, self.manifest)["violations"]))

    def test_force_omission(self):
        damaged = copy.deepcopy(self.reactions)
        damaged["force_residual"][0] += 1.0
        self.assertIn("physical force sum", audit(self.q, damaged, self.manifest)["violations"])

    def test_equality_acceleration_omission(self):
        damaged = copy.deepcopy(self.reactions)
        row = self.manifest["joint_equalities"]["records"][0]
        damaged["acceleration"][row["dependent_core_v"]] += 1.0
        self.assertIn("source equality tangent or virtual work", audit(self.q, damaged, self.manifest)["violations"])

    def test_nonfinite_and_short_vectors(self):
        for value in ([float("nan")] * 128, [0.0] * 127):
            damaged = copy.deepcopy(self.reactions)
            damaged["acceleration"] = value
            with self.assertRaises(ValueError):
                audit(self.q, damaged, self.manifest)

    def test_duplicate_certificate(self):
        with self.assertRaises(ValueError):
            parse(self.log + "\n" + self.log)


if __name__ == "__main__":
    unittest.main()
