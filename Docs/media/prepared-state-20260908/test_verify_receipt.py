#!/usr/bin/env python3
import gzip
import json
import unittest
from pathlib import Path
import verify_receipt as audit

HERE = Path(__file__).resolve().parent


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = (HERE / 'fixture/prepared.nhinit').read_bytes()
        cls.fixture = json.loads((HERE / 'fixture/prepared.json').read_text())
        cls.certificate = gzip.decompress((HERE.parent / 'coupled-equilibrium-20260908/clean-stdout.log.gz').read_bytes()).decode()
        cls.log = gzip.decompress((HERE / 'prepared-clean.log.gz').read_bytes()).decode()

    def test_exact_transport(self):
        self.assertEqual(audit.audit_state(self.payload, self.certificate, self.fixture)['muscles'], 416)

    def test_state_mutations(self):
        for offset in [0, 8, 16, 32, 40, 48, 56, 64, 99, 1127, 1135]:
            data = bytearray(self.payload)
            data[offset] ^= 1
            fixture = dict(self.fixture, initial_state_fp=f'{audit.fnv(data):016x}')
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                audit.audit_state(data, self.certificate, fixture)

    def test_trailing_bytes(self):
        with self.assertRaises(ValueError):
            audit.audit_state(self.payload + b'\0', self.certificate, self.fixture)

    def test_trajectory(self):
        self.assertEqual(len(audit.audit_trajectory(self.log)[1]), 8)

    def test_missing_state_rejected(self):
        altered = self.log.replace('prepared_accepted_state=', 'omitted_state=', 1)
        with self.assertRaises(ValueError):
            audit.audit_trajectory(altered)

    def test_generation_drift_rejected(self):
        row = next(l for l in self.log.splitlines() if l.startswith('prepared_accepted_state='))
        state = json.loads(row.split('=', 1)[1])
        state['physics_generation'] = 9
        with self.assertRaises(ValueError):
            audit.audit_trajectory(self.log.replace(row, 'prepared_accepted_state=' + json.dumps(state)))

    def test_failed_run_rejected(self):
        with self.assertRaises(ValueError):
            audit.audit_trajectory(self.log.replace('Executed 2 tests, with 0 failures', 'Executed 2 tests, with 1 failure'))


if __name__ == '__main__':
    unittest.main()
