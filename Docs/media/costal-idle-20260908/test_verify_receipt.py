import copy
import gzip
import json
from pathlib import Path
import unittest
from verify_receipt import audit


class CostalReceiptTests(unittest.TestCase):
    def setUp(self):
        p = Path(__file__).parent
        self.log = gzip.decompress((p/'costal-idle-clean.log.gz').read_bytes()).decode()
        self.workload = json.loads((p/'costal-idle-clean-workload.json').read_text())

    def test_complete(self):
        self.assertEqual(audit(self.log, self.workload)['accepted_roots'], 8)

    def test_missing_physical_candidate(self):
        log = self.log.replace('GateB physical-token words=', 'removed=', 1)
        with self.assertRaises(ValueError): audit(log, self.workload)

    def test_failure(self):
        with self.assertRaises(ValueError): audit(self.log.replace('mrnx_matter_status code=0', 'mrnx_matter_status code=1', 1), self.workload)

    def test_competing_workload(self):
        w = copy.deepcopy(self.workload); w[5]['competing_gpu'] = ['md-run']
        with self.assertRaises(ValueError): audit(self.log, w)

    def test_truncated_monitor(self):
        with self.assertRaises(ValueError): audit(self.log, self.workload[:5])


if __name__ == '__main__':
    unittest.main()
