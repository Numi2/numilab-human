import hashlib
import struct
import unittest

from numilab_human.locomotor import compile_program


class LocomotorTests(unittest.TestCase):
    def fixture(self):
        records = [struct.pack('<4I37f', i, 0, 2, 0, *([0.] * 35 + [0.25 + i / 10000, 0.])) for i in range(416)]
        payload = struct.pack('<8s9I32s', b'NHMYO2\0\0', 2, 157, 416, 0, 0, 0, 416, 416, 32, bytes(32)) + b''.join(records) + bytes(416 * 32)
        body = dict(format='numanx-locomotor-body-v1', muscle_payload_sha256=hashlib.sha256(payload).hexdigest(),
                    model_source_fingerprint=1, sensory_profile_fingerprint=2, actuator_count=416)
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

if __name__ == '__main__': unittest.main()
