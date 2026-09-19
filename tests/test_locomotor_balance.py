import copy
import hashlib
import json
import struct
import unittest

from numilab_human.locomotor import compile_program, _fnv, _baseline_fingerprint


class BalanceAuthoringTests(unittest.TestCase):
    def fixture(self, muscles=3):
        records = [
            struct.pack('<4I37f', i, 0, 2, 0, *([0.] * 35 + [0.25 + i / 10000, 0.]))
            for i in range(muscles)
        ]
        payload = (
            struct.pack('<8s9I32s', b'NHMYO2\0\0', 2, 157, muscles, 0, 0, 0,
                        muscles, muscles, 32, bytes(32))
            + b''.join(records) + bytes(muscles * 32)
        )
        body = dict(
            format='numanx-locomotor-body-v1',
            muscle_payload_sha256=hashlib.sha256(payload).hexdigest(),
            model_source_fingerprint=1,
            sensory_profile_fingerprint=2,
            actuator_count=muscles,
        )
        return body, payload

    def prepared_fixture(self, muscles=3):
        body, payload = self.fixture(muscles)
        q = [0.] * 129
        q[6] = 1
        state = struct.pack(
            '<8s8I3Q32s', b'NHINIT1\0', 1, 96, 129, 128, muscles, 4, 0, 0,
            123, 456, 100, bytes(32),
        )
        state += struct.pack('<257f', *(q + [0.] * 128))
        state += b''.join(
            struct.pack('<4f', a, a, .2, 0) for a in [0., .25, 1.][:muscles]
        )
        fingerprint = _fnv(state)
        body = dict(
            body,
            model_source_fingerprint=_fnv(
                b'NHINIT1' + struct.pack('<Q', fingerprint), 123
            ),
            timestep_microseconds=100,
            prepared_initial_state=dict(
                sha256=hashlib.sha256(state).hexdigest(),
                fingerprint=fingerprint,
                world_fingerprint=456,
                q_count=129,
                dof_count=128,
            ),
        )
        return body, payload, state

    def balance(self, body):
        return {
            'format': 'numi-human-muscle-balance-map-v1',
            'modelSourceFingerprint': body['model_source_fingerprint'],
            'sensoryProfileFingerprint': body['sensory_profile_fingerprint'],
            'mode': 'supportAware',
            'updatePeriodMicroseconds': 4000,
            'initializationDurationMicroseconds': 100000,
            'sources': [
                {
                    'identifier': 1,
                    'bodyReceptorBindingIdentifier': 1003,
                    'referenceValue': 0,
                    'filterTimeConstantSeconds': 0,
                    'conductionDelayMicroseconds': 0,
                    'evidenceKind': 'kinematic',
                },
                {
                    'identifier': 2,
                    'bodyReceptorBindingIdentifier': 0x53500000,
                    'referenceValue': 350,
                    'filterTimeConstantSeconds': 0,
                    'conductionDelayMicroseconds': 0,
                    'evidenceKind': 'support',
                },
            ],
            'routes': [
                {
                    'sourceIdentifier': 1,
                    'muscleIdentifier': 0,
                    'gain': -0.2,
                    'maximumCorrection': 0.1,
                },
                {
                    'sourceIdentifier': 2,
                    'muscleIdentifier': 1,
                    'gain': 0.001,
                    'maximumCorrection': 0.2,
                },
            ],
        }

    def compile(self, balance=None, artifact=None):
        body, payload, state = self.prepared_fixture()
        return compile_program(
            body,
            payload,
            prepared_state=state,
            maximum=1,
            length_gain=0,
            velocity_gain=0,
            balance=balance,
            balance_artifact=artifact,
        )

    def test_prepared_baseline_is_preserved_and_bound_to_v2_feedback(self):
        body, _, _ = self.prepared_fixture()
        balance = self.balance(body)
        artifact = json.dumps(balance, sort_keys=True).encode()
        program = self.compile(balance, artifact)
        self.assertEqual(program['version'], 2)
        self.assertEqual(
            [channel['tonicExcitation'] for channel in program['channels']],
            [0., .25, 1.],
        )
        feedback = program['balanceFeedback']
        baseline = dict(program)
        baseline.pop('balanceFeedback')
        baseline['version'] = 1
        self.assertEqual(
            feedback['locomotorProgramFingerprint'],
            _baseline_fingerprint(baseline),
        )
        self.assertEqual(
            feedback['calibrationArtifactSHA256'],
            hashlib.sha256(artifact).hexdigest(),
        )
        self.assertEqual(feedback['mode'], 2)
        self.assertNotIn('evidenceKind', feedback['sources'][0])
        self.assertEqual(feedback['sources'][1]['referenceValue'], 350)

    def test_balance_bytes_are_provenance_not_a_replaceable_parse(self):
        body, _, _ = self.prepared_fixture()
        balance = self.balance(body)
        artifact = json.dumps(balance, sort_keys=True).encode()
        changed = json.dumps(balance, sort_keys=True, indent=2).encode()
        first = self.compile(balance, artifact)
        second = self.compile(balance, changed)
        self.assertNotEqual(
            first['balanceFeedback']['calibrationArtifactSHA256'],
            second['balanceFeedback']['calibrationArtifactSHA256'],
        )
        foreign = copy.deepcopy(balance)
        foreign['routes'][0]['gain'] = .3
        with self.assertRaisesRegex(ValueError, 'bytes'):
            self.compile(foreign, artifact)

    def test_balance_validation_fails_closed(self):
        body, _, _ = self.prepared_fixture()
        original = self.balance(body)

        def reject(mutator, match=None):
            balance = copy.deepcopy(original)
            mutator(balance)
            artifact = json.dumps(balance, sort_keys=True).encode()
            context = self.assertRaisesRegex(ValueError, match) if match else self.assertRaises(ValueError)
            with context:
                self.compile(balance, artifact)

        reject(lambda x: x.__setitem__('modelSourceFingerprint', x['modelSourceFingerprint'] + 1), 'another body')
        reject(lambda x: x.__setitem__('mode', 'unknown'), 'mode')
        reject(lambda x: x['sources'][0].__setitem__('identifier', 2), 'identity')
        reject(lambda x: x['sources'][0].__setitem__('bodyReceptorBindingIdentifier', x['sources'][1]['bodyReceptorBindingIdentifier']), 'identity')
        reject(lambda x: x['sources'][0].__setitem__('filterTimeConstantSeconds', .01), 'history')
        reject(lambda x: x['sources'][0].__setitem__('conductionDelayMicroseconds', 1), 'history')
        reject(lambda x: x['sources'][1].__setitem__('evidenceKind', 'kinematic'), 'support evidence')
        reject(lambda x: x['routes'][0].__setitem__('sourceIdentifier', 99), 'unbound')
        reject(lambda x: x['routes'][0].__setitem__('muscleIdentifier', 3), 'unbound')
        reject(lambda x: x['routes'][0].__setitem__('gain', 0), 'unbound')
        reject(lambda x: x['routes'][0].__setitem__('maximumCorrection', .6), 'bounds')
        reject(lambda x: x['routes'].append({
            'sourceIdentifier': 2,
            'muscleIdentifier': 0,
            'gain': .1,
            'maximumCorrection': .5,
        }), 'budget')

    def test_map_and_bytes_must_be_supplied_together(self):
        body, _, _ = self.prepared_fixture()
        balance = self.balance(body)
        with self.assertRaisesRegex(ValueError, 'together'):
            self.compile(balance, None)
        artifact = json.dumps(balance, sort_keys=True).encode()
        with self.assertRaisesRegex(ValueError, 'together'):
            self.compile(None, artifact)

    def test_v1_baseline_fingerprint_is_stable(self):
        body, payload = self.fixture(1)
        program = compile_program(
            body, payload, tonic=.03, length_gain=.4, velocity_gain=.02
        )
        self.assertEqual(_baseline_fingerprint(program), 6256353027231271110)


if __name__ == '__main__':
    unittest.main()
