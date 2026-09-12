"""Cavity receipt admission tests; never execute native physics or copy large evidence."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('cavity_evidence', ROOT/'tools/verify_cardiac_cavities_20260912.py')
assert SPEC is not None and SPEC.loader is not None
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)
EVIDENCE = ROOT/VERIFY.EVIDENCE


def receipt():
    return {'schema':'HumanPack.cardiac-cavities-evidence.v1','native_commit':VERIFY.NATIVE_COMMIT,
            'scope':VERIFY.SCOPE.copy(),'owners':{},'artifacts':{}}


class ReceiptAdmissionTests(unittest.TestCase):
    def test_scope_rejects_promotion_omission_numeric_booleans_and_unknown_fields(self):
        VERIFY.audit_scope(receipt())
        for key in VERIFY.SCOPE:
            altered=receipt();altered['scope'][key]=not altered['scope'][key]
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'scope'):
                VERIFY.audit_scope(altered)
        for value in (None,{},dict(VERIFY.SCOPE,physiological_calibration=0),dict(VERIFY.SCOPE,full_human_complete=True)):
            altered=receipt();altered['scope']=value
            with self.assertRaisesRegex(ValueError,'scope'):VERIFY.audit_scope(altered)
        altered=receipt();altered['production_complete']=True
        with self.assertRaisesRegex(ValueError,'fields'):VERIFY.audit_scope(altered)

    def test_missing_owners_and_artifacts_rejected_before_file_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaisesRegex(ValueError,'missing required owners'):
                VERIFY.checked_inventory(root,{},VERIFY.REQUIRED_OWNERS,'owners')
            with self.assertRaisesRegex(ValueError,'missing required artifacts'):
                VERIFY.checked_inventory(root,{},VERIFY.REQUIRED_ARTIFACTS,'artifacts')
            with self.assertRaisesRegex(ValueError,'outside cavity evidence'):
                VERIFY.checked_inventory(root,{'outside.txt':'1'*64},set(),'artifacts')
            extra=root/VERIFY.EVIDENCE/'extra.txt';extra.parent.mkdir(parents=True);extra.write_text('extra')
            path=extra.relative_to(root).as_posix()
            self.assertIn(path,VERIFY.checked_inventory(root,{path:VERIFY.sha(extra)},set(),'artifacts'))
            self.assertIn('src/numilab_human/cardiac_cavity_intersections.py',VERIFY.REQUIRED_OWNERS)
            self.assertIn('tests/test_cardiac_cavities_evidence.py',VERIFY.REQUIRED_OWNERS)
            self.assertIn((VERIFY.EVIDENCE/'native/physical-mutation.log').as_posix(),VERIFY.REQUIRED_ARTIFACTS)

    def test_hash_drift_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'record.txt';path.write_text('captured')
            hashes={'record.txt':VERIFY.sha(path)}
            VERIFY.checked_inventory(root,hashes,set(hashes),'artifact')
            path.write_text('altered')
            with self.assertRaisesRegex(ValueError,'hash drift'):
                VERIFY.checked_inventory(root,hashes,set(hashes),'artifact')

    def test_unsafe_paths_and_symlink_ancestors_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'real').mkdir();(root/'real/file').write_text('data')
            (root/'alias').symlink_to(root/'real',target_is_directory=True)
            for name in ('../file','/file','./real/file','real//file','real\\file','real/fi\nle','alias/file'):
                with self.subTest(name=name),self.assertRaises(ValueError):VERIFY.checked_path(root,name)
            self.assertEqual(VERIFY.checked_path(root,'real/file'),root/'real/file')

    def test_resealed_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'record.json'
            for text in ('{"scope":false,"\\u0073cope":true}','{"value":NaN}','{"value":1e999}'):
                path.write_text(text)
                with self.subTest(text=text),self.assertRaises(ValueError):
                    VERIFY.checked_inventory(root,{'record.json':VERIFY.sha(path)},{'record.json'},'artifact')

    def test_summary_rejects_substring_success_and_false_qualification(self):
        log=(EVIDENCE/'native/upstream_equation.native.log').read_text()
        VERIFY.audit_native_summary(log)
        for altered in (log.replace('accepted_steps=64','accepted_steps=640'),
                        log.replace('failed_steps=0','failed_steps=01'),
                        log.replace('physiological_calibration=unqualified','physiological_calibration=qualified'),
                        log.replace('finite_state=true','finite_state=true finite_state=false'),
                        log.replace('device=Apple M4 Pro','device=Apple Paravirtual'),
                        log+'cavity_native_check=failed reason=late failure\n'):
            with self.assertRaises(ValueError):VERIFY.audit_native_summary(altered)


class NativeBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.execution=VERIFY.cv.read_json(EVIDENCE/'native/execution.json')
        cls.manifests={variant:VERIFY.cv.read_json(EVIDENCE/(variant+'.manifest.json')) for variant in VERIFY.cv.VARIANTS}

    def check(self,execution):
        VERIFY.audit_execution(execution,self.manifests,lambda name:EVIDENCE/name,ROOT)

    def test_actual_record_commands_inputs_and_negative_control_pass(self):
        self.check(self.execution)
        for name,expected in VERIFY.CAPTURED_SHA256.items():
            self.assertEqual(VERIFY.sha(EVIDENCE/name),expected)

    def test_static_archive_drift_or_omission_cannot_be_resealed(self):
        for omit in (False,True):
            d=copy.deepcopy(self.execution)
            for phase in ('before','after'):
                values=d['build_inputs_'+phase]
                key=next(k for k in values if k.endswith('libnumi_matter_runtime.a'))
                if omit:del values[key]
                else:values[key]='1'*64
            with self.subTest(omit=omit),self.assertRaisesRegex(ValueError,'static build'):
                self.check(d)

    def test_build_command_cannot_link_an_unrecorded_library(self):
        d=copy.deepcopy(self.execution);d['build_command'][15]='/tmp/unbound-runtime.a'
        with self.assertRaisesRegex(ValueError,'build command'):self.check(d)
        d=copy.deepcopy(self.execution);d['dynamic_dependencies']+='\t/tmp/unbound.dylib (compatibility version 1.0.0)\n'
        with self.assertRaisesRegex(ValueError,'dynamic dependency'):self.check(d)

    def test_runtime_command_order_and_log_role_are_bound(self):
        d=copy.deepcopy(self.execution);d['runs'][0]['command'][1],d['runs'][0]['command'][2]=d['runs'][0]['command'][2],d['runs'][0]['command'][1]
        with self.assertRaisesRegex(ValueError,'native command'):self.check(d)
        d=copy.deepcopy(self.execution);d['runs'][0]['log']=d['runs'][1]['log']
        with self.assertRaisesRegex(ValueError,'log role'):self.check(d)

    def test_runtime_metallib_or_payload_change_rejected(self):
        for suffix in ('NumiMatter.metallib','.native.json'):
            d=copy.deepcopy(self.execution)
            for phase in ('before','after'):
                values=d['runs'][0]['inputs_'+phase];key=next(k for k in values if k.endswith(suffix));values[key]='1'*64
            with self.subTest(suffix=suffix),self.assertRaisesRegex(ValueError,'runtime input'):
                self.check(d)

    def test_negative_control_requires_same_executable_baseline_and_library(self):
        d=copy.deepcopy(self.execution)
        for phase in ('before','after'):
            values=d['negative_control']['inputs_'+phase]
            key=next(k for k in values if k.endswith('cardiac-cavity-native-check'));del values[key]
        with self.assertRaisesRegex(ValueError,'runtime input'):self.check(d)
        d=copy.deepcopy(self.execution);d['negative_control']['exit_code']=0
        with self.assertRaisesRegex(ValueError,'failed or drifted'):self.check(d)


class CompleteReceiptTests(unittest.TestCase):
    def test_complete_retained_receipt(self):
        result=VERIFY.verify(root=ROOT)
        self.assertEqual(result['status'],'pass')
        self.assertEqual(result['right_atrium_right_ventricle_intersections'],42)
        self.assertEqual(result['native_variant_pairs_passed'],2)
        self.assertFalse(result['disjoint_physical_volume_admission'])
        self.assertFalse(result['physiological_calibration'])
        self.assertFalse(result['blood_mass_partition'])


if __name__=='__main__':
    unittest.main()
