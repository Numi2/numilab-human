from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from numilab_human.capability_protocol import REQUIREMENTS, MEASUREMENTS, compile_protocol, Artifacts
from numilab_human.model import ImportError as HumanImportError
from numilab_human.target_coverage import digest


class CapabilityProtocolTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.sequence = 0
        self.raw = self.store('SYNTHETIC fixture demographic record; not a measured person')
        self.subject = dict(namespace='synthetic-test', participant_id='fixture-male', sex='male', age_years=30)
        demographics = dict(schema='numi.human.subject-demographics.v1', **self.subject,
                            raw_source=self.raw, source_locator='fixture row')
        self.subject['source'] = self.store(demographics)
        pack = self.store('SYNTHETIC HumanPack, no native mechanics')
        binding = self.store(dict(schema='numi.human.subject-model-binding.v1', namespace='synthetic-test',
                                 participant_id='fixture-male', age_years=30, human_pack_sha256=pack['sha256'],
                                 parameters=self.store('unqualified fixture parameters')))
        self.protocol = dict(schema='numi.human.capability-protocol.v1', subject=self.subject,
                             model=dict(human_pack=pack, subject_binding=binding), trials=[], cells=[],
                             freeze=dict(before_controller_selection=True, statistical_plan=self.store('fixture statistics'),
                                         confidence=.95, numerical_error_fraction=.1))
        self.add_trial('calibration', 'calibration', 'calibration/session', 'calibration', {'kinematics'})
        for family, scenarios in REQUIREMENTS.items():
            for scenario in scenarios:
                cid = family + '/' + scenario
                tid = 'heldout-' + cid
                record = self.add_trial(tid, 'validation', 'validation/session', cid, MEASUREMENTS[family])
                comparisons = []
                for measurement in record['measurements']:
                    margin = self.store(dict(schema='numi.human.empirical-margin.v1', measurement_id=measurement['id'],
                                             units=measurement['units'], metric='rmse', tolerance=.1,
                                             basis='measurement_repeatability', source=self.raw))
                    comparisons.append(dict(trial_id=tid, measurement_id=measurement['id'], metric='rmse',
                                            tolerance=.1, numerical_error_limit=.001, margin_source=margin))
                self.protocol['cells'].append(dict(id=cid, family=family, trial_ids=[tid], comparisons=comparisons))

    def store(self, value):
        self.sequence += 1
        data = (json.dumps(value, sort_keys=True) if isinstance(value, dict) else value).encode()
        path = self.root / f'{self.sequence}.data'
        path.write_bytes(data)
        return dict(path=path.name, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))

    def read(self, ref):
        return json.loads((self.root / ref['path']).read_bytes())

    def add_trial(self, tid, role, session, capability, kinds):
        measurements = [dict(id=k, kind=k, units='fixture-unit', origin='measured',
                             artifact=self.store(f'fixture-only {tid} {k}'), uncertainty=self.raw) for k in sorted(kinds)]
        record = dict(schema='numi.human.measured-trial.v1', namespace='synthetic-test', participant_id='fixture-male',
                      age_years=30, trial_id=tid, session_id=session, capability_id=capability,
                      conditions=self.store(f'fixture conditions {capability}'), measurements=measurements)
        self.protocol['trials'].append(dict(id=tid, role=role, session_id=session, source=self.store(record)))
        return record

    def run_compile(self, value=None):
        return compile_protocol(self.protocol if value is None else value, base=self.root)

    def change_record(self, mutate, index=1):
        trial = self.protocol['trials'][index]
        record = self.read(trial['source'])
        mutate(record)
        trial['source'] = self.store(record)

    def test_complete_contract_is_deterministic_and_never_qualification(self):
        result = self.run_compile()
        self.assertEqual(result, self.run_compile())
        self.assertEqual(result['counts'], dict(families=7, cells=25, trials=26))
        self.assertEqual(result['compiled_sha256'], digest({k:v for k,v in result.items() if k != 'compiled_sha256'}))
        for field in ('empirical_qualification', 'runtime_qualification', 'source_authenticity'):
            self.assertEqual(result[field], 'not_assessed')
        self.assertEqual(result['freeze_chronology'], 'not_attested')

    def test_age_sex_namespace_and_model_mismatch_rejected(self):
        for key, val in [('sex','female'), ('age_years',17), ('age_years',True), ('age_years',float('nan')),
                         ('namespace','another-study'), ('participant_id','another-person')]:
            with self.subTest(key=key, val=val):
                value = copy.deepcopy(self.protocol)
                value['subject'][key] = val
                with self.assertRaises(HumanImportError): self.run_compile(value)
        self.change_record(lambda r:r.update(participant_id='someone-else'))
        with self.assertRaisesRegex(HumanImportError, 'another participant'): self.run_compile()

    def test_other_age_and_model_binding_are_rejected(self):
        self.change_record(lambda r:r.update(age_years=31))
        with self.assertRaisesRegex(HumanImportError, 'another participant or age'): self.run_compile()
        binding = self.read(self.protocol['model']['subject_binding'])
        binding['human_pack_sha256'] = '0'*64
        self.protocol['model']['subject_binding'] = self.store(binding)
        with self.assertRaisesRegex(HumanImportError, 'model is not bound'): self.run_compile()

    def test_reused_sessions_and_data_do_not_create_heldout_evidence(self):
        self.protocol['trials'][1]['session_id'] = 'calibration/session'
        with self.assertRaisesRegex(HumanImportError, 'sessions overlap'): self.run_compile()
        self.protocol['trials'][1]['session_id'] = 'validation/session'
        first = self.read(self.protocol['trials'][0]['source'])['measurements'][0]['artifact']
        self.change_record(lambda r:r['measurements'][0].update(artifact=first))
        with self.assertRaisesRegex(HumanImportError, 'bytes reused'): self.run_compile()

    def test_source_values_cannot_be_relabelled_as_empirical(self):
        for origin in ('derived','simulated'):
            self.change_record(lambda r:r['measurements'][0].update(origin=origin))
            with self.assertRaisesRegex(HumanImportError, 'cannot satisfy empirical'): self.run_compile()

    def test_cell_coverage_and_trial_conditions_are_mandatory(self):
        cell = self.protocol['cells'].pop()
        with self.assertRaisesRegex(HumanImportError, 'cells are missing'): self.run_compile()
        self.protocol['cells'].append(cell)
        self.change_record(lambda r:r.update(capability_id='manipulation/reach'))
        with self.assertRaisesRegex(HumanImportError, 'another capability'): self.run_compile()

    def test_motion_alone_cannot_qualify_loading(self):
        cell = self.protocol['cells'][0]
        cell['comparisons'] = [c for c in cell['comparisons'] if c['measurement_id']=='kinematics']
        with self.assertRaisesRegex(HumanImportError, 'measurement coverage'): self.run_compile()

    def test_margin_units_and_numerical_budget_are_frozen(self):
        comparison = self.protocol['cells'][0]['comparisons'][0]
        comparison['numerical_error_limit'] = .1
        with self.assertRaisesRegex(HumanImportError, 'frozen fraction'): self.run_compile()
        comparison['numerical_error_limit'] = .001
        margin = self.read(comparison['margin_source'])
        margin['units'] = 'different-units'
        comparison['margin_source'] = self.store(margin)
        with self.assertRaisesRegex(HumanImportError, 'margin does not bind'): self.run_compile()

    def test_extra_promotion_fields_and_missing_freeze_rejected(self):
        value = copy.deepcopy(self.protocol)
        value['qualified'] = True
        with self.assertRaises(HumanImportError): self.run_compile(value)
        self.protocol['freeze']['before_controller_selection'] = False
        with self.assertRaisesRegex(HumanImportError, 'before controller'): self.run_compile()

    def test_artifact_tampering_and_mid_read_change_rejected(self):
        ref = self.protocol['model']['human_pack']
        path = self.root / ref['path']
        old = path.read_bytes()
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(HumanImportError, 'artifact bytes changed'): self.run_compile()
        path.write_bytes(old)
        original = Artifacts.recheck
        def change(reader):
            path.write_bytes(b'changed after admission')
            original(reader)
        with patch.object(Artifacts, 'recheck', change):
            with self.assertRaisesRegex(HumanImportError, 'during compilation'): self.run_compile()

    def test_cli_is_registered(self):
        from numilab_human.cli import parser
        args = parser().parse_args(['capability-protocol','--requirements'])
        self.assertTrue(args.requirements)

    def test_malformed_roles_report_validation_error(self):
        for role in ([], {}, None, True):
            with self.subTest(role=role):
                self.protocol['trials'][0]['role'] = role
                with self.assertRaisesRegex(HumanImportError, 'trial role'):
                    self.run_compile()

    def test_each_trial_requires_its_own_measurement_coverage(self):
        cell = self.protocol['cells'][0]
        tid = 'second-heldout'
        record = self.add_trial(tid, 'validation', 'validation/session', cell['id'], {'kinematics'})
        cell['trial_ids'].append(tid)
        comparison = copy.deepcopy(next(c for c in cell['comparisons'] if c['measurement_id'] == 'kinematics'))
        comparison['trial_id'] = tid
        cell['comparisons'].append(comparison)
        with self.assertRaisesRegex(HumanImportError, 'measurement coverage'):
            self.run_compile()

    def test_cli_compiles_and_preserves_immutable_output(self):
        from numilab_human.cli import parser
        source = self.root / 'protocol.json'
        source.write_text(json.dumps(self.protocol))
        output = self.root / 'compiled.json'
        args = parser().parse_args(['capability-protocol', '--protocol', str(source), '--output', str(output)])
        self.assertEqual(args.handler(args), 0)
        self.assertEqual(json.loads(output.read_bytes()), self.run_compile())
        self.assertEqual(args.handler(args), 0)
        output.write_bytes(b'previous evidence')
        with self.assertRaisesRegex(HumanImportError, 'immutable'):
            args.handler(args)
        self.assertEqual(output.read_bytes(), b'previous evidence')


if __name__ == '__main__':
    unittest.main()
