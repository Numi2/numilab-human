#!/usr/bin/env python3
"""Negative checks of evidence gates, separate from native mechanics tests."""
import gzip
import json
import unittest
from pathlib import Path
import verify_receipt as receipt
import verify_source_mass as mass

HERE = Path(__file__).resolve().parent


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.log = gzip.decompress((HERE / 'clean-stdout.log.gz').read_bytes()).decode()
        cls.source = json.loads((HERE / 'source-mass.json').read_text())

    def mutate(self, prefix, change):
        old = receipt.unique(self.log, prefix)
        value = json.loads(old)
        change(value)
        return self.log.replace(prefix + old, prefix + json.dumps(value))

    def test_actual_certificate(self):
        self.assertLess(receipt.audit_state(self.log)['residual'], 0.05)

    def test_unbalanced_claim_rejected(self):
        with self.assertRaises(ValueError):
            receipt.audit_state(self.log.replace('internal_balanced=true', 'internal_balanced=false'))

    def test_replay_loss_rejected(self):
        with self.assertRaises(ValueError):
            receipt.audit_state(self.log.replace('replay=bitwise', 'replay=failed'))

    def test_trace_loss_and_objective_increase_rejected(self):
        for change in (lambda t: t.pop(1), lambda t: t[1].update(objective=t[0]['objective'] + 1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                receipt.audit_state(self.mutate('compiled_equilibrium_search_trace=', change))

    def test_muscle_transport_and_sign_rejected(self):
        for key, value in [('activation_fp32', 0.5), ('reference_fiber_length_m', -1),
                           ('actuator_force_n', 1), ('activation_fp64', float('nan'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                receipt.audit_state(self.mutate('compiled_equilibrium_muscles=',
                    lambda m: m[key].__setitem__(0, value)))

    def test_force_corruption_rejected(self):
        with self.assertRaises(ValueError):
            receipt.audit_state(self.mutate('compiled_equilibrium_reactions=',
                lambda m: m['force_residual'].__setitem__(0, 1)))

    def test_source_mass_action_and_gravity(self):
        names = ('native_force_residual', 'source_mass_times_native_acceleration', 'native_gravity', 'source_gravity')
        original = [self.source[k] for k in names]
        self.assertEqual(mass.audit_vectors(*original)['status'], 'offline_source_mass_action_passed')
        for index in range(4):
            values = [v.copy() for v in original]
            values[index][0] += 1
            with self.subTest(index=index), self.assertRaises(ValueError):
                mass.audit_vectors(*values)
        values = [v.copy() for v in original]
        values[0][0] = float('nan')
        with self.assertRaises(ValueError):
            mass.audit_vectors(*values)
        values[0].pop()
        with self.assertRaises(ValueError):
            mass.audit_vectors(*values)


if __name__ == '__main__':
    unittest.main()
