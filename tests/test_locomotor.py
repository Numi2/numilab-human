import hashlib
import struct
import unittest

from numilab_human.locomotor import compile_program, _fnv


class LocomotorTests(unittest.TestCase):
    def fixture(self, muscles=416):
        records = [struct.pack('<4I37f', i, 0, 2, 0, *([0.] * 35 + [0.25 + i / 10000, 0.])) for i in range(muscles)]
        payload = struct.pack('<8s9I32s', b'NHMYO2\0\0', 2, 157, muscles, 0, 0, 0, muscles, muscles, 32, bytes(32)) + b''.join(records) + bytes(muscles * 32)
        body = dict(format='numanx-locomotor-body-v1', muscle_payload_sha256=hashlib.sha256(payload).hexdigest(),
                    model_source_fingerprint=1, sensory_profile_fingerprint=2, actuator_count=muscles)
        return body, payload

    def test_source_lengths_modes_and_explicit_phase_mapping(self):
        body, payload = self.fixture()
        standing = compile_program(body, payload, tonic=.03, length_gain=.4, velocity_gain=.02)
        self.assertEqual(len(standing['channels']), 416)
        self.assertEqual(standing['channels'][0]['referenceLengthMeters'], .25)
        self.assertEqual(standing['periodMicroseconds'], 0)
        self.assertTrue(all(x['gaitSine'] == x['gaitCosine'] == 0 for x in standing['channels']))
        phases = [dict(muscleIdentifier=3, gaitSine=.1, gaitCosine=0), dict(muscleIdentifier=9, gaitSine=-.1, gaitCosine=0)]
        gait = compile_program(body, payload, tonic=.03, length_gain=.4, velocity_gain=.02, period_us=1_000_000, gait=phases)
        self.assertEqual(gait['channels'][3]['gaitSine'], .1)
        self.assertEqual(gait['channels'][9]['gaitSine'], -.1)
        self.assertEqual(gait['channels'][8]['gaitSine'], 0)

    def test_foreign_truncated_nonfinite_and_ambiguous_inputs_reject(self):
        body, payload = self.fixture()
        def compile(b=body, p=payload, **kw):
            return compile_program(b, p, **(dict(tonic=.03, length_gain=.4, velocity_gain=.02) | kw))
        with self.assertRaises(ValueError): compile(p=payload[:-1])
        with self.assertRaises(ValueError): compile(b=dict(body, sensory_profile_fingerprint=True))
        for invalid in [float('nan'), True, None, '0.03']:
            with self.assertRaises(ValueError): compile(tonic=invalid)
        for invalid in [{}, 'mapping', [None], [3]]:
            with self.assertRaises(ValueError): compile(gait=invalid)
        with self.assertRaises(ValueError): compile(maximum=.01)
        with self.assertRaises(ValueError): compile(period_us=1)
        row = dict(muscleIdentifier=3, gaitSine=.1, gaitCosine=0)
        with self.assertRaises(ValueError): compile(gait=[row])
        with self.assertRaises(ValueError): compile(period_us=1_000_000, gait=[row, row])
        with self.assertRaises(ValueError): compile(period_us=1_000_000, gait=[dict(row, muscleIdentifier=416)])
        with self.assertRaises(ValueError): compile(period_us=1_000_000, gait=[dict(row, gaitSine=float('inf'))])
        bad = bytearray(payload); struct.pack_into('<f', bad, 76 + 16 + 35 * 4, float('nan'))
        with self.assertRaises(ValueError): compile(b=dict(body, muscle_payload_sha256=hashlib.sha256(bad).hexdigest()), p=bad)

    def prepared_fixture(self, muscles=3):
        body, payload = self.fixture(muscles)
        q = [0.] * 129; q[6] = 1
        state = struct.pack('<8s8I3Q32s', b'NHINIT1\0', 1, 96, 129, 128, muscles, 4, 0, 0,
                            123, 456, 100, bytes(32))
        state += struct.pack('<257f', *(q + [0.] * 128))
        state += b''.join(struct.pack('<4f', a, a, .2, 0) for a in [0., .25, 1.][:muscles])
        return self.bind_state(body, state), payload, state

    def bind_state(self, body, state):
        fingerprint = _fnv(state)
        body = dict(body, model_source_fingerprint=_fnv(b'NHINIT1' + struct.pack('<Q', fingerprint), 123),
                    timestep_microseconds=100, prepared_initial_state=dict(
                        sha256=hashlib.sha256(state).hexdigest(), fingerprint=fingerprint,
                        world_fingerprint=456, q_count=129, dof_count=128))
        return body

    def test_prepared_recruitment_preserves_each_inclusive_excitation(self):
        body, payload, state = self.prepared_fixture()
        program = compile_program(body, payload, prepared_state=state, maximum=1,
                                  length_gain=0, velocity_gain=0)
        self.assertEqual([c['tonicExcitation'] for c in program['channels']], [0., .25, 1.])
        self.assertEqual(program['modelSourceFingerprint'], body['model_source_fingerprint'])
        self.assertEqual(program['calibrationArtifactSHA256'], hashlib.sha256(state).hexdigest())
        self.assertEqual(len(program['channels']), 3)
        self.assertEqual(program, compile_program(body, payload, prepared_state=state, maximum=1,
                                                  length_gain=0, velocity_gain=0))

    def test_prepared_recruitment_rejects_rebinding_and_invalid_state(self):
        import copy
        body, payload, state = self.prepared_fixture()
        def compile(b=body, p=state, **kw):
            return compile_program(b, payload, **(dict(prepared_state=p, maximum=1, length_gain=0, velocity_gain=0) | kw))
        for kw in [dict(maximum=.999), dict(maximum=float('nan')), dict(tonic=0), dict(length_gain=.1),
                   dict(velocity_gain=.01), dict(period_us=100000), dict(gait=[])]:
            with self.assertRaises(ValueError): compile(**kw)
        for key in ['model_source_fingerprint', 'timestep_microseconds']:
            with self.assertRaises(ValueError): compile(b=dict(body, **{key: body[key] + 1}))
        for key in body['prepared_initial_state']:
            changed = copy.deepcopy(body); value = changed['prepared_initial_state'][key]
            changed['prepared_initial_state'][key] = 'f'*64 if isinstance(value, str) else value+1
            with self.assertRaises(ValueError): compile(b=changed)
        for p in [state[:-1], state + b'\0', bytes(96)]:
            with self.assertRaises(ValueError): compile(p=p)
        # Rehash every malformed artifact: rejection must validate semantics too.
        mutations = [(0, '<B', 0), (8, '<I', 2), (16, '<I', 128), (32, '<I', 1),
                     (40, '<Q', 124), (48, '<Q', 457), (56, '<Q', 101), (64, '<I', 1),
                     (96, '<f', float('nan')), (120, '<f', 0), (96+129*4, '<f', .1),
                     (1124, '<f', 1.1), (1128, '<f', .1), (1132, '<f', 0), (1136, '<f', .1)]
        for offset, fmt, value in mutations:
            p = bytearray(state); struct.pack_into(fmt, p, offset, value); p = bytes(p)
            with self.subTest(offset=offset), self.assertRaises(ValueError): compile(b=self.bind_state(body, p), p=p)
        with self.assertRaises(ValueError):
            compile_program(body, payload, tonic=.03, length_gain=0, velocity_gain=0)

if __name__ == '__main__': unittest.main()
