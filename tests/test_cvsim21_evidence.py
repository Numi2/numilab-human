"""Evidence admission and tamper tests; no simulator or physical stepping."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "Docs/media/cvsim21-circulation-20260912"
SPEC = importlib.util.spec_from_file_location("cvsim21_evidence_verifier", ROOT / "tools/verify_cvsim21_20260912.py")
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def load(name):
    return VERIFY.read_json(EVIDENCE / name)


class EvidenceAdmissionTests(unittest.TestCase):
    def test_duplicate_escaped_nested_fields_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.json"
            for text in ('{"a":1,"\\u0061":2}', '{"outer":{"a":1,"a":2}}',
                         '{"value":NaN}', '{"value":Infinity}', '{"value":1e999}', '[1,2]'):
                with self.subTest(text=text):
                    path.write_text(text)
                    with self.assertRaises(VERIFY.EvidenceError):
                        VERIFY.read_json(path)

    def test_unsafe_paths_and_symlink_components_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ('../outside', '/absolute', 'a/../b', 'a//b', './a', 'a\\b', 'a\nb', ''):
                with self.subTest(name=name), self.assertRaises(VERIFY.EvidenceError):
                    VERIFY.safe_path(root, name)
            (root / 'real').mkdir()
            (root / 'alias').symlink_to(root / 'real', target_is_directory=True)
            with self.assertRaisesRegex(VERIFY.EvidenceError, 'symlink'):
                VERIFY.safe_path(root, 'alias/record.json')
            self.assertEqual(VERIFY.safe_path(root, 'real/record.json'), root / 'real/record.json')

    def test_inventory_hash_drift_and_deleted_required_record_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'failure.log').write_text('retained failed admission\n')
            values = {'failure.log': VERIFY.digest(root / 'failure.log')}
            self.assertEqual(set(VERIFY.checked_inventory(root, values, {'failure.log'}, 'artifact')), {'failure.log'})
            with self.assertRaisesRegex(VERIFY.EvidenceError, 'missing required'):
                VERIFY.checked_inventory(root, values, {'failure.log', 'missing.csv.gz'}, 'artifact')
            (root / 'failure.log').write_text('pass\n')
            with self.assertRaisesRegex(VERIFY.EvidenceError, 'hash drift'):
                VERIFY.checked_inventory(root, values, {'failure.log'}, 'artifact')

    def test_resealed_json_inventory_still_rejects_duplicate_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / 'identity.json'
            path.write_text('{"abi":27,"abi":28}')
            with self.assertRaisesRegex(VERIFY.EvidenceError, 'duplicate'):
                VERIFY.checked_inventory(root, {'identity.json': VERIFY.digest(path)}, {'identity.json'}, 'artifact')

    def test_scope_promotion_and_unknown_top_level_claim_rejected(self):
        receipt = dict(schema=VERIFY.SCHEMA, native_commit=VERIFY.NATIVE_COMMIT,
                       source_lock_sha256=VERIFY.SOURCE_LOCK_SHA256, initial_sha256=VERIFY.INITIAL_SHA256,
                       scope=VERIFY.SCOPE.copy(), qualification='source_variant_numerical_comparison',
                       artifacts={}, owners={}, numerical_report_path='native-comparison.json')
        VERIFY.audit_scope(receipt)
        for key in VERIFY.SCOPE:
            altered = copy.deepcopy(receipt)
            altered['scope'][key] = True
            with self.subTest(key=key), self.assertRaisesRegex(VERIFY.EvidenceError, 'promotion'):
                VERIFY.audit_scope(altered)
        altered = copy.deepcopy(receipt)
        altered['full_human_complete'] = True
        with self.assertRaisesRegex(VERIFY.EvidenceError, 'unknown or missing'):
            VERIFY.audit_scope(altered)
        altered = copy.deepcopy(receipt)
        altered['scope']['tilt'] = 0
        with self.assertRaisesRegex(VERIFY.EvidenceError, 'promotion'):
            VERIFY.audit_scope(altered)

    def test_required_inventory_cannot_drop_failure_history_or_source_clock_drift(self):
        self.assertTrue(set(VERIFY.FAILURE_HASHES) <= VERIFY.REQUIRED_ARTIFACTS)
        self.assertTrue({'original-supine.csv.gz', 'continuous-original-times.csv.gz',
                         'legacy-cardiac-requalification.json'} <= VERIFY.REQUIRED_ARTIFACTS)
        for name in VERIFY.LEGACY_RUNS:
            self.assertIn('legacy-cardiac-requalification/' + name + '.execution.json', VERIFY.REQUIRED_ARTIFACTS)
        self.assertIn(VERIFY.HISTORICAL_RECEIPT, VERIFY.REQUIRED_OWNERS)
        self.assertEqual(len(VERIFY.SOURCE_FILES), 17)

    def test_ctest_actual_complete_cohort_and_missing_or_failed_case(self):
        text = (EVIDENCE / 'native-ctest.txt').read_text()
        VERIFY.audit_ctest(text)
        for altered in (text.replace('matter.compiler.cvsim_admission', 'unrelated.test'),
                        text.replace('Passed', '***Failed', 1),
                        text.replace('100% tests passed out of 15', '100% tests passed out of 14')):
            with self.assertRaises(VERIFY.EvidenceError):
                VERIFY.audit_ctest(altered)

    def test_native_identity_requires_final_publication_physical_host_and_all_owners(self):
        identity = load('native-identity.json')
        VERIFY.audit_native_identity(identity, VERIFY.NATIVE_COMMIT)
        mutations = [('abi', 27), ('worktree_status', ' M matter/src/runtime.mm'),
                     ('native_commit', VERIFY.NATIVE_BASE), ('snapshot_archive', 5)]
        for key, value in mutations:
            altered = copy.deepcopy(identity)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(VERIFY.EvidenceError):
                VERIFY.audit_native_identity(altered, VERIFY.NATIVE_COMMIT)
        altered = copy.deepcopy(identity)
        altered['device']['chip'] = 'Apple Paravirtual device'
        with self.assertRaises(VERIFY.EvidenceError):
            VERIFY.audit_native_identity(altered, VERIFY.NATIVE_COMMIT)
        altered = copy.deepcopy(identity)
        del altered['source_sha256']['matter/src/metal/vascular.metalinc']
        with self.assertRaisesRegex(VERIFY.EvidenceError, 'coverage'):
            VERIFY.audit_native_identity(altered, VERIFY.NATIVE_COMMIT)


class ExecutionBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        identity = load('native-identity.json')
        cls.sources, cls.binaries = VERIFY.audit_native_identity(identity, VERIFY.NATIVE_COMMIT)
        cls.artifacts = {name: EVIDENCE / name for name in VERIFY.REQUIRED_ARTIFACTS}
        cls.reference = '1423dbd3ad5c831780ec4462d8e3db8f151934211b36b100d4aea42bb0d43f7e'

    def check(self, identity, name='cycle-2ms'):
        VERIFY.audit_run_identity(identity, name, self.artifacts, self.sources, self.binaries, self.reference)

    def test_captured_main_and_table_aligned_commands_pass(self):
        for name in ('cycle-2ms', 'heldt-cycle-1ms'):
            self.check(load(name + '.identity.json'), name)

    def test_external_metallib_drift_even_with_unchanged_executable_rejected(self):
        identity = load('cycle-2ms.identity.json')
        identity['runtime_after_sha256']['matter/shaders/NumiMatter.metallib'] = '1' * 64
        with self.assertRaisesRegex(VERIFY.EvidenceError, 'changed or omitted'):
            self.check(identity)

    def test_omitted_runtime_owner_or_source_drift_rejected(self):
        for group, key in [('runtime', 'matter/shaders/NumiMatter.metallib'), ('source', 'matter/src/runtime.mm')]:
            identity = load('cycle-2ms.identity.json')
            for position in ('before', 'after'):
                del identity[f'{group}_{position}_sha256'][key]
            with self.subTest(group=group), self.assertRaisesRegex(VERIFY.EvidenceError, 'changed or omitted'):
                self.check(identity)
        identity = load('cycle-2ms.identity.json')
        for position in ('before', 'after'):
            identity[f'source_{position}_sha256']['matter/src/runtime.mm'] = '1' * 64
        with self.assertRaisesRegex(VERIFY.EvidenceError, 'final native identity'):
            self.check(identity)

    def test_payload_drift_and_failed_exit_rejected(self):
        for key, value in [('payload_after_sha256', '1' * 64), ('reference_after_sha256', '1' * 64), ('exit_code', 1)]:
            identity = load('cycle-2ms.identity.json')
            identity[key] = value
            with self.subTest(key=key), self.assertRaises(VERIFY.EvidenceError):
                self.check(identity)

    def test_renamed_old_trace_and_resealed_wrong_artifact_bindings_rejected(self):
        identity = load('cycle-2ms.identity.json')
        identity['command'][identity['command'].index('--trace') + 1] = '/old/renamed.csv'
        with self.assertRaisesRegex(VERIFY.EvidenceError, 'time or trace'):
            self.check(identity)
        for suffix in ('.log', '.csv', '.csv.gz'):
            identity = load('cycle-2ms.identity.json')
            identity['artifacts']['cycle-2ms' + suffix]['sha256'] = '1' * 64
            with self.subTest(suffix=suffix), self.assertRaisesRegex(VERIFY.EvidenceError, 'binding differs'):
                self.check(identity)

    def test_ten_cycle_cannot_use_short_run_or_wrong_volume_coordinates(self):
        identity = load('ten-cycles-2ms.identity.json')
        identity['command'][identity['command'].index('--steps') + 1] = '429'
        with self.assertRaisesRegex(VERIFY.EvidenceError, 'time or trace'):
            self.check(identity, 'ten-cycles-2ms')
        identity = load('heldt-cycle-1ms.identity.json')
        identity['command'][-1] = 'upstream_equation'
        with self.assertRaisesRegex(VERIFY.EvidenceError, 'variant'):
            self.check(identity, 'heldt-cycle-1ms')

    def test_legacy_bound_source_oracle_and_runtime_match_then_reject_drift(self):
        name = 'refinement_2ms'
        identity = load('legacy-cardiac-requalification/' + name + '.execution.json')
        args = (name, .002, 500, self.artifacts, self.sources, self.binaries)
        self.assertEqual(VERIFY.audit_legacy_identity(identity, *args), identity['before'])
        identity['trace_raw_sha256'] = '1' * 64
        with self.assertRaisesRegex(VERIFY.EvidenceError, 'raw trace'):
            VERIFY.audit_legacy_identity(identity, *args)
        identity = load('legacy-cardiac-requalification/' + name + '.execution.json')
        identity['after']['sources']['matter/tools/shi_hose_reference/shi_hose_reference_generated.hpp'] = '1' * 64
        with self.assertRaisesRegex(VERIFY.EvidenceError, 'changed or omitted'):
            VERIFY.audit_legacy_identity(identity, *args)


class CompleteReceiptTests(unittest.TestCase):
    def test_complete_retained_receipt_recomputes_source_native_and_legacy_evidence(self):
        result = VERIFY.verify(root=ROOT)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['ctest_passed'], 15)
        self.assertEqual(result['scope'], VERIFY.SCOPE)
        self.assertEqual(result['native_commit'], VERIFY.NATIVE_COMMIT)
        self.assertTrue(result['source_drift_retained'])
        self.assertTrue(result['historical_cardiac_receipt_unchanged'])
        self.assertFalse(result['python_physical_stepping'])


if __name__ == '__main__':
    unittest.main()
