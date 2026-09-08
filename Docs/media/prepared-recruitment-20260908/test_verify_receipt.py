import base64
import gzip
import json
from pathlib import Path
import struct
import unittest

from verify_receipt import audit_trace


class TraceAdmissionTests(unittest.TestCase):
    def setUp(self):
        text = gzip.decompress((Path(__file__).parent / 'recruited-clean.log.gz').read_bytes()).decode()
        self.rows = [json.loads(l.split('=', 1)[1]) for l in text.splitlines()
                     if l.startswith('prepared_recruitment={')]

    def audit(self):
        return audit_trace('\n'.join('prepared_recruitment=' + json.dumps(r) for r in self.rows))

    def change_value(self, scenario, root, kind, index, value):
        row = next(r for r in self.rows if (r['scenario'], r['root'], r['kind']) == (scenario, root, kind))
        raw = bytearray(base64.b64decode(row['fp32_le_base64']))
        struct.pack_into('<f', raw, index * 4, value)
        row['fp32_le_base64'] = base64.b64encode(raw).decode()

    def test_complete_trace(self):
        self.assertEqual(self.audit()['accepted_roots_per_scenario'], 4)

    def test_missing_and_duplicate_reject(self):
        row = self.rows.pop()
        with self.assertRaises(ValueError): self.audit()
        self.rows.extend([row, row])
        with self.assertRaises(ValueError): self.audit()

    def test_clock_rejects(self):
        self.rows[0]['elapsed_microseconds'] += 1
        with self.assertRaises(ValueError): self.audit()

    def test_replay_drift_rejects(self):
        self.change_value('replay', 4, 'qv', 0, .1)
        with self.assertRaises(ValueError): self.audit()

    def test_nonfinite_rejects(self):
        self.change_value('recruited', 4, 'tendon_force', 0, float('nan'))
        with self.assertRaises(ValueError): self.audit()

    def test_bootstrap_nonzero_rejects(self):
        for scenario in ['recruited', 'replay']:
            self.change_value(scenario, 1, 'motor', 0, .1)
        with self.assertRaises(ValueError): self.audit()

    def test_missing_force_response_rejects(self):
        zero = next(r for r in self.rows if (r['scenario'], r['root'], r['kind']) == ('zero', 4, 'tendon_force'))
        for row in self.rows:
            if row['scenario'] in ['recruited', 'replay'] and row['root'] == 4 and row['kind'] == 'tendon_force':
                row['fp32_le_base64'] = zero['fp32_le_base64']
        with self.assertRaises(ValueError): self.audit()


if __name__ == '__main__':
    unittest.main()
