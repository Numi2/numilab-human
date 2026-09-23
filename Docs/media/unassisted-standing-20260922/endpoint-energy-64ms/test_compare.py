#!/usr/bin/env python3
"""Retained-log regression: incomplete energy evidence must fail closed."""
import gzip
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
VERIFIER = ROOT / 'compare.py'
ENERGY_LOG = ROOT / 'feedback-energy' / 'stdout.txt.gz'


class EndpointEnergyVerifierTests(unittest.TestCase):
    def verify(self, root, optimized=False):
        command = [sys.executable]
        if optimized:
            command.append('-O')
        command.extend((str(VERIFIER), str(root)))
        return subprocess.run(command,
                              capture_output=True, text=True)

    def test_retained_evidence_and_corruptions(self):
        self.assertEqual(self.verify(ROOT).returncode, 0)
        self.assertEqual(self.verify(ROOT, optimized=True).returncode, 0)
        original = gzip.decompress(ENERGY_LOG.read_bytes()).decode()
        corruptions = {
            'false_step_closure': ('closure=not_established', 'closure=established'),
            'missing_source_limit': ('source_limit_endpoint_work=unavailable',
                                     'source_limit_endpoint_work=0'),
            'false_zero_residual': ('unclosed_available_terms_residual_j=2.1163698436756279e-06',
                                    'unclosed_available_terms_residual_j=0'),
            'corrupt_muscle_total': ('muscle_midpoint_work_j=0.0015229869919870867',
                                     'muscle_midpoint_work_j=0.0025229869919870867'),
        }
        for name, (before, after) in corruptions.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                copy = Path(directory) / 'evidence'
                shutil.copytree(ROOT, copy)
                self.assertIn(before, original)
                (copy / 'feedback-energy' / 'stdout.txt').write_text(
                    original.replace(before, after, 1))
                for optimized in (False, True):
                    result = self.verify(copy, optimized=optimized)
                    self.assertNotEqual(result.returncode, 0, result.stdout)


if __name__ == '__main__':
    unittest.main()
